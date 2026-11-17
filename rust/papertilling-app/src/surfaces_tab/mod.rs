use std::ops::RangeInclusive;

use papertilling_core::surface::symmetry_facts;
use papertilling_core::types::{Frame2D, Recipe, SurfaceCheckReport, SurfaceMesh};

#[cfg(test)]
use crate::backend::CentreModeUi;
use crate::backend::{
    AppBackend, CatalogueItemUi, SurfaceBuildParamsUi, SurfaceCellVerdictUi, SurfaceTargetUi,
    VerdictKindUi,
};
use crate::canvas2d::gauge_overlay::{CrystalOverlayData, color_sites_scalar};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::export_obj::export_surface_obj;
use crate::live_ops;
use crate::mesh3d::GpuMeshData;
use crate::spectrum_worker::SpectrumPanelState;
use crate::surfaces_elastic::{SurfaceElasticState, SurfaceShapeMode};
use crate::theme::ThemeTokens;
use crate::view3d::View3dState;

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
    pub spectrum: SpectrumPanelState,
    pub(crate) active_disp: String,
    active_lattice_key: Option<(String, Recipe, Frame2D)>,
    pub(super) current_mesh: Option<SurfaceMesh>,
    pub(super) current_report: Option<SurfaceCheckReport>,
    pub(super) refusal_msg: Option<String>,
    lattice_verdicts: Option<(VerdictKey, [SurfaceCellVerdictUi; 8])>,
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
            spectrum: SpectrumPanelState::new(),
            active_disp: String::new(),
            active_lattice_key: None,
            current_mesh: None,
            current_report: None,
            refusal_msg: None,
            lattice_verdicts: None,
        }
    }

    pub(crate) fn has_refusal(&self) -> bool {
        self.refusal_msg.is_some()
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
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        self.clamp_params(item);
        let lkey = (item.group.clone(), item.recipe.clone(), item.frame);
        let target_changed = self.active_lattice_key.as_ref() != Some(&lkey);
        self.active_lattice_key = Some(lkey);
        self.active_disp = item.display_with_orig();
        self.elastic.clear_for_switch();
        self.spectrum.clear_for_switch();
        let _ = target_changed;
        let built = if self.auto_size {
            live_ops::build_surface_mesh_auto(item, self.target, self.params.centre_mode, self.params.genus_g)
        } else {
            backend.build_surface(item, self.target, &self.params)
        };
        match built {
            Ok((mesh, report)) => {
                let net_title = format!("{} ({}) — {} — 2D net", self.active_disp, item.group, mesh.name);
                canvas.set_scene_with_title(Scene2D::SurfaceNet { mesh: mesh.clone() }, Some(net_title), true);
                if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK {
                    self.elastic.request_relax(mesh.clone(), None);
                }
                if self.spectrum.expanded {
                    self.spectrum.submit_surface(&mesh, None);
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
        let spec_pair = self.spectrum.active_scalar_overlay();
        let spec_ref = spec_pair.as_ref().map(|(v, d)| (v.as_slice(), *d));
        let (gpu, fvk_note) = self.elastic.build_gpu_mesh(mesh, self.show_cover, spec_ref);
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
        if let Some((vals, div)) = spec_ref {
            let ov = CrystalOverlayData {
                site_rgba: Some(color_sites_scalar(vals, div)),
                ..Default::default()
            };
            canvas.set_gauge_overlay(Some(ov));
        } else {
            canvas.set_gauge_overlay(None);
        }
    }

    pub fn poll_background(&mut self, ctx: &egui::Context, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        let e_hit = self.elastic.poll(ctx);
        let s_hit = self.spectrum.poll(ctx);
        if e_hit || s_hit {
            if let Some(el) = &self.elastic.last_result
                && self.spectrum.expanded
                && self.spectrum.opts.beta.abs() > 1e-12
                && e_hit
                && let Some(mesh) = &self.current_mesh
            {
                let mut m3 = mesh.clone();
                m3.xyz = el.xyz.clone();
                if let (Some(c), Some(cx)) = (m3.cover.as_deref_mut(), el.cover_xyz.as_ref()) {
                    c.xyz = cx.clone();
                }
                self.spectrum.submit_surface(&m3, Some(ctx));
            }
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

    /// The 8 verdicts of `surface_table` on the active lattice itself (cached per lattice).
    pub(super) fn lattice_verdicts(&mut self, item: &CatalogueItemUi) -> Option<&[SurfaceCellVerdictUi; 8]> {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        if self.lattice_verdicts.as_ref().is_none_or(|(k, _)| *k != key) {
            let cells = live_ops::build_verdict_cells(item).ok()?;
            self.lattice_verdicts = Some((key, cells));
        }
        self.lattice_verdicts.as_ref().map(|(_, c)| c)
    }
}
