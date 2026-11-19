use std::ops::RangeInclusive;

use papertilling_core::surface::symmetry_facts;
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

pub(crate) mod build_async;
mod inspector;
mod left;
#[cfg(test)]
mod tests;

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
    }
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
            target: SurfaceTargetUi::Sphere,
            params: SurfaceBuildParamsUi::default(),
            show_cover: false,
            auto_size: true,
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
            Ok((mesh, report)) => {
                if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK {
                    self.elastic.request_relax(mesh.clone(), None);
                }
                self.current_mesh = Some(mesh);
                self.current_report = Some(report);
                self.refusal_msg = None;
            }
            Err(reason) => {
                self.current_mesh = None;
                self.current_report = None;
                self.refusal_msg = Some(reason);
            }
        }
    }

    /// Clamp `(h, k)` into the values the controls offer for the current target and lattice.
    pub fn clamp_params(&mut self, item: &CatalogueItemUi) {
        let sc = size_controls(self.target, symmetry_facts(&item.group).max_rotation_order());
        self.params.h = self.params.h.clamp(*sc.h_range.start(), *sc.h_range.end());
        self.params.k = self.params.k.clamp(*sc.k_range.start(), *sc.k_range.end());
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
        build_async::SurfaceJob { item: item.clone(), target: self.target, params: self.params.clone(), auto_size: self.auto_size }
    }

    /// Show a built surface (net, 3D, elastic follow-up) or its refusal.
    fn apply_built(&mut self, built: build_async::Built, group: &str, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        // results (and running jobs) of the previous mesh must never be drawn or exported with the new one
        self.elastic.clear_for_switch();
        match built {
            Ok((mesh, report)) => {
                let net_title = format!("{} ({}) — {} — 2D net", self.active_disp, group, mesh.name);
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
            "{} on {}{} | V={} E={} F={} χ={} (R={:.2}, c=[{:.1},{:.1},{:.1}]){fvk_note}",
            disp,
            mesh.name,
            if use_cov { " [S² double cover]" } else { "" },
            gpu.vertex_count,
            gpu.edge_count,
            gpu.face_count,
            if use_cov { 2 } else { mesh.chi },
            gpu.radius,
            gpu.center[0],
            gpu.center[1],
            gpu.center[2],
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
