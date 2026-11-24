use std::ops::RangeInclusive;

use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::surface::{symmetry_facts, symmetry_facts_of_cell};
use papertilling_core::types::{Frame2D, Recipe, SurfaceCheckReport, SurfaceMesh};

#[cfg(test)]
use crate::backend::CentreModeUi;
use crate::backend::{
    AppBackend, CatalogueItemUi, SurfaceBuildParamsUi, SurfaceCellVerdictUi, SurfaceTargetUi,
    VerdictKindUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::export_obj::export_surface_obj;
use crate::live_ops;
use crate::mesh3d::GpuMeshData;
use crate::surfaces_elastic::{SurfaceElasticState, SurfaceShapeMode};
use crate::theme::ThemeTokens;
use crate::view3d::View3dState;

pub(crate) mod bc;
mod bc_ui;
pub(crate) mod build_async;
mod inspector;
mod left;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_bc;

/// Name the construction before its counts can be mistaken for the template's own faces.
fn surface_description(name: &str) -> String {
    for surface in ["sphere", "projective plane"] {
        if let Some(template) = name.strip_suffix(&format!(" ({surface})")) {
            return format!("{surface} · {template} gluing template");
        }
    }
    name.to_owned()
}

/// What the two size controls mean for a target, and the values offered (every value is used as shown).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeControls {
    pub h_label: &'static str,
    pub k_label: &'static str,
    pub h_range: RangeInclusive<i32>,
    pub k_range: RangeInclusive<i32>,
}

/// Size controls of `target` for a lattice whose highest rotation order is `max_rot` (2: the pillow needs h, k >= 1).
/// The largest values keep a rebuild well under a second on the heaviest catalogue lattices (see the timing test).
pub fn size_controls(target: SurfaceTargetUi, max_rot: u32) -> SizeControls {
    let c = |h_label, k_label, h_range, k_range| SizeControls { h_label, k_label, h_range, k_range };
    match target {
        SurfaceTargetUi::Torus => c("T1 = (h, 0)", "T2 = (0, k)", 1..=24, 1..=24),
        SurfaceTargetUi::Cylinder => c("rows h", "wrap (0, k)", 1..=24, 1..=24),
        SurfaceTargetUi::MobiusBand => c("width h", "along 2k+1", 1..=12, 1..=12),
        SurfaceTargetUi::KleinBottle => c("across h", "along 2k+1", 1..=12, 1..=12),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            c("size h", "size k", 1..=8, if max_rot == 2 { 1..=8 } else { 0..=8 })
        }
        SurfaceTargetUi::Genus2 | SurfaceTargetUi::Genus3 => c("torus (h, 0)", "torus (0, k)", 1..=12, 1..=12),
        // C1: the open patch and the hexagonal torus are sized by the boundary options (`bc`), not by (h, k)
        SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex => c("size h", "size k", 1..=12, 1..=12),
    }
}

/// Size controls of `target` for `item`'s realized cell (`symmetry_facts_of_cell`, falling back to `&item.group`
/// only when `item.recipe` cannot build a cell).
pub fn size_controls_for(target: SurfaceTargetUi, item: &CatalogueItemUi) -> SizeControls {
    let max_rot = match LatticeFamily::from_recipe(&item.recipe, Some(item.frame)) {
        Ok(fam) => symmetry_facts_of_cell(&item.group, &fam.cell()).max_rotation_order(),
        Err(_) => symmetry_facts(&item.group).max_rotation_order(),
    };
    size_controls(target, max_rot)
}

/// An empty net: shown when the surface is refused, so the canvas never keeps the previous surface's net.
pub fn empty_net(name: &str) -> SurfaceMesh {
    SurfaceMesh {
        name: name.to_string(),
        sites: Vec::new(),
        bonds: Vec::new(),
        faces: Vec::new(),
        plane_degree: Vec::new(),
        edge_sites: Vec::new(),
        chi: 0,
        orientable: true,
        closed: true,
        cones: Vec::new(),
        uv: Vec::new(),
        domain_polygons: Vec::new(),
        domain_edges: Vec::new(),
        slits: Vec::new(),
        xyz: Vec::new(),
        bond_curves_3d: Vec::new(),
        cover: None,
    }
}

pub(super) fn verdict_colour(v: VerdictKindUi, tokens: &ThemeTokens) -> egui::Color32 {
    match v {
        VerdictKindUi::Ok => tokens.status_ok,
        VerdictKindUi::Impossible => tokens.status_err,
        VerdictKindUi::NotFound => tokens.status_warn,
    }
}

pub struct SurfacesTabState {
    pub target: SurfaceTargetUi,
    pub params: SurfaceBuildParamsUi,
    /// The boundary options (layers, wrap, supercell, glide class...), used by the six boundary-condition targets.
    pub bc: bc::BcControls,
    /// Caption and read-out of the surface on screen.
    pub(crate) notes: bc::BcNotes,
    bc_summary: Option<(VerdictKey, Option<bc::BcSummary>)>,
    pub show_cover: bool,
    /// Sizes as in `surface_table` (the surface behind the verdict cell) instead of the (h, k) controls.
    pub auto_size: bool,
    pub elastic: SurfaceElasticState,
    pub(crate) active_disp: String,
    active_lattice_key: Option<(String, Recipe, Frame2D)>,
    pub(crate) current_mesh: Option<SurfaceMesh>,
    pub(super) current_report: Option<SurfaceCheckReport>,
    pub(crate) refusal_msg: Option<String>,
    /// Verdicts per lattice; `None` inside records a lattice whose verdicts cannot be built (not retried).
    lattice_verdicts: Option<(VerdictKey, Option<[SurfaceCellVerdictUi; 8]>)>,
    /// Synchronous builds run by `activate_and_sync` (lets callers and tests see whether a build already ran).
    pub(crate) syncs: u64,
    builder: Option<build_async::SurfaceWorker>,
    queued_build: Option<(std::time::Instant, build_async::SurfaceJob)>,
}

type VerdictKey = (String, Recipe, Frame2D);

impl SurfacesTabState {
    pub fn new() -> Self {
        Self {
            target: SurfaceTargetUi::Open,
            params: SurfaceBuildParamsUi::default(),
            bc: bc::BcControls::default(),
            notes: bc::BcNotes::default(),
            bc_summary: None,
            show_cover: false,
            auto_size: false,
            elastic: SurfaceElasticState::new(),
            active_disp: String::new(),
            active_lattice_key: None,
            current_mesh: None,
            current_report: None,
            refusal_msg: None,
            lattice_verdicts: None,
            syncs: 0,
            builder: None,
            queued_build: None,
        }
    }

    pub(crate) fn has_refusal(&self) -> bool {
        self.refusal_msg.is_some()
    }

    /// True if `Surfaces` has not yet built (or refused) for `item`'s current `(group, recipe, frame)`.
    pub(crate) fn needs_sync_for(&self, item: &CatalogueItemUi) -> bool {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        self.active_lattice_key.as_ref() != Some(&key)
            || (self.current_mesh.is_none() && self.refusal_msg.is_none())
    }

    /// Build the active surface mesh without touching `canvas2d` or `view3d` (for `Analysis` mode).
    pub(crate) fn build_quiet_for(&mut self, item: &CatalogueItemUi) {
        self.cancel_build();
        let job = self.prepare_build(item);
        let built = build_async::run_job(&job);
        self.elastic.clear_for_switch();
        match built {
            Ok((mesh, report, notes)) => {
                if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK {
                    self.elastic.request_relax(mesh.clone(), None);
                }
                self.notes = notes;
                self.current_mesh = Some(mesh);
                self.current_report = Some(report);
                self.refusal_msg = None;
            }
            Err(reason) => {
                self.notes = bc::BcNotes::default();
                self.current_mesh = None;
                self.current_report = None;
                self.refusal_msg = Some(reason);
            }
        }
    }

    /// The boundary options this build uses: always for the open patch and the hexagonal torus; for the other four
    /// boundary conditions unless the table sizes are asked for, from the options once the user has edited them and
    /// from the legacy `(h, k)` sizes before.
    fn bc_for_build(&self) -> Option<bc::BcControls> {
        match self.target {
            SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex => Some(self.bc.clone()),
            SurfaceTargetUi::Torus | SurfaceTargetUi::Cylinder | SurfaceTargetUi::MobiusBand | SurfaceTargetUi::KleinBottle
                if !self.auto_size =>
            {
                Some(if self.bc.explicit { self.bc.clone() } else { bc::BcControls::from_hk(self.params.h, self.params.k) })
            }
            _ => None,
        }
    }

    /// Select `t` from the Boundary selector, a deck card or "More": a boundary condition is sized by its options,
    /// the cone metrics and genus surfaces by the table sizes (custom `(h, k)` stays one click away under "More").
    pub(crate) fn choose_target(&mut self, t: SurfaceTargetUi) {
        self.target = t;
        if bc::BC_TARGETS.contains(&t) {
            self.auto_size = false;
            self.bc.explicit = true;
        } else {
            self.auto_size = true;
        }
    }

    /// The six boundary verdicts and glide classes of `item` (cached per lattice, failures included).
    pub(crate) fn bc_summary(&mut self, item: &CatalogueItemUi) -> Option<&bc::BcSummary> {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        if self.bc_summary.as_ref().is_none_or(|(k, _)| *k != key) {
            self.bc_summary = Some((key, bc::summarize(item)));
        }
        self.bc_summary.as_ref().and_then(|(_, s)| s.as_ref())
    }

    /// Clamp `(h, k)` into the values the controls offer for the current target and lattice.
    pub fn clamp_params(&mut self, item: &CatalogueItemUi) {
        let sc = size_controls_for(self.target, item);
        self.params.h = self.params.h.clamp(*sc.h_range.start(), *sc.h_range.end());
        self.params.k = self.params.k.clamp(*sc.k_range.start(), *sc.k_range.end());
        self.bc.clamp();
        match self.target {
            SurfaceTargetUi::Genus2 => self.params.genus_g = 2,
            SurfaceTargetUi::Genus3 => self.params.genus_g = 3,
            _ => {}
        }
    }

    pub fn activate_and_sync(
        &mut self,
        item: &CatalogueItemUi,
        _backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        self.syncs += 1;
        self.cancel_build();
        let job = self.prepare_build(item);
        let built = build_async::run_job(&job);
        self.apply_built(built, &item.group, canvas, view3d);
    }

    /// Everything before the build itself: clamp the sizes and remember the lattice. The per-surface panels are
    /// cleared when the new mesh is applied, so the mesh on screen keeps its own relaxation / spectrum meanwhile.
    fn prepare_build(&mut self, item: &CatalogueItemUi) -> build_async::SurfaceJob {
        self.clamp_params(item);
        self.active_lattice_key = Some((item.group.clone(), item.recipe.clone(), item.frame));
        self.active_disp = item.display_with_orig();
        build_async::SurfaceJob {
            item: item.clone(),
            target: self.target,
            params: self.params.clone(),
            auto_size: self.auto_size,
            bc: self.bc_for_build(),
        }
    }

    /// Show a built surface (net, 3D, elastic follow-up) or its refusal.
    fn apply_built(&mut self, built: build_async::Built, group: &str, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        // results (and running jobs) of the previous mesh must never be drawn or exported with the new one
        self.elastic.clear_for_switch();
        match built {
            Ok((mesh, report, notes)) => {
                self.notes = notes;
                let net_title = format!("{} ({group}) · {}", self.active_disp, surface_description(&mesh.name));
                canvas.set_scene_with_title(Scene2D::SurfaceNet { mesh: mesh.clone() }, Some(net_title), true);
                if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK {
                    self.elastic.request_relax(mesh.clone(), None);
                }
                self.current_mesh = Some(mesh);
                self.current_report = Some(report);
                self.refusal_msg = None;
                self.sync_view3d_and_colors(canvas, view3d);
            }
            Err(reason) => {
                view3d.set_mesh(GpuMeshData::default(), format!("{} on {}: {reason}", self.active_disp, self.target.label()));
                canvas.set_scene(Scene2D::SurfaceNet { mesh: empty_net(self.target.label()) }, true);
                self.notes = bc::BcNotes::default();
                self.current_mesh = None;
                self.current_report = None;
                self.refusal_msg = Some(reason);
            }
        }
    }

    pub fn sync_view3d_and_colors(&self, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        let Some(mesh) = &self.current_mesh else { return };
        let use_cov = self.show_cover && mesh.cover.is_some();
        let (gpu, fvk_note) = self.elastic.build_gpu_mesh(mesh, self.show_cover, None);
        let disp = if self.active_disp.is_empty() { mesh.name.as_str() } else { self.active_disp.as_str() };
        let caption = format!(
            "{disp}{} · V={} E={} F={} χ={} · {}{fvk_note}{}",
            if use_cov { " [S² double cover]" } else { "" },
            gpu.vertex_count,
            gpu.edge_count,
            gpu.face_count,
            if use_cov { 2 } else { mesh.chi },
            surface_description(&mesh.name),
            self.notes.picture_note.map(|n| format!(" | {n}")).unwrap_or_default(),
        );
        view3d.set_mesh(gpu, caption);
        view3d.set_legend(self.elastic.color_legend(None));
        canvas.set_gauge_overlay(None);
    }

    pub fn poll_background(&mut self, ctx: &egui::Context, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        let _ = self.pump_build(ctx, canvas, view3d);
        if self.elastic.poll(ctx) {
            self.sync_view3d_and_colors(canvas, view3d);
        }
    }

    pub fn export_active_obj(&self, path: &std::path::Path) -> Result<String, String> {
        let Some(mesh) = &self.current_mesh else {
            return Err(String::from("No valid 3D surface is built to export."));
        };
        if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK
            && let Some(res) = &self.elastic.last_result
        {
            let mut m = mesh.clone();
            m.xyz = res.xyz.clone();
            if let (Some(cov), Some(cxyz)) = (m.cover.as_deref_mut(), res.cover_xyz.as_ref()) {
                cov.xyz = cxyz.clone();
            }
            return export_surface_obj(&m, self.show_cover && m.cover.is_some(), path);
        }
        export_surface_obj(mesh, self.show_cover && mesh.cover.is_some(), path)
    }

    /// The 8 verdicts of `surface_table` on the active lattice itself (cached per lattice, failures included, so
    /// a lattice whose verdicts cannot be built is not rebuilt every frame).
    pub(crate) fn lattice_verdicts(&mut self, item: &CatalogueItemUi) -> Option<&[SurfaceCellVerdictUi; 8]> {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        if self.lattice_verdicts.as_ref().is_none_or(|(k, _)| *k != key) {
            let cells = live_ops::build_verdict_cells(item).ok();
            self.lattice_verdicts = Some((key, cells));
        }
        self.lattice_verdicts.as_ref().and_then(|(_, c)| c.as_ref())
    }

    /// Whether the verdicts of `item` (built or failed) are cached.
    #[cfg(test)]
    pub(crate) fn verdicts_cached_for(&self, item: &CatalogueItemUi) -> bool {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        self.lattice_verdicts.as_ref().is_some_and(|(k, _)| *k == key)
    }
}
