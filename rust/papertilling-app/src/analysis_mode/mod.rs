//! Isolated Tight-Binding & Eigenmode `3. Analysis` mode (`AnalysisModeState`).
//!
//! Reads pristine or defect geometry from `Gallery`, `Surfaces`, `Defects`, or `Combine` without
//! leaking any spectral UI, workers, legends, or overlays back into `1. Lattices` or `2. Defects`.

use papertilling_core::defect::sheet3d::{Sheet3DOptions, Sheet3DResult, relax_sheet_3d};
use papertilling_core::surface::elastic3d::SurfaceElasticResult;
use papertilling_core::types::{BuiltCrystal, Frame2D, Recipe};

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::canvas2d::gauge_overlay::CrystalOverlayData;
use crate::canvas2d::overlay_legend::spectral_overlay_data;
use crate::combine::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::export_obj::export_sheet_obj;
use crate::live_ops::{build_pristine_disk, defect_context};
use crate::mesh3d::color_scales::{sheet_color_legend, surface_color_legend};
use crate::mesh3d::{
    DefectViewMode, GpuMeshData, Sheet3dColorMode, Sheet3dViewOpts, SurfaceColor3d,
    elastic_surface_gpu_mesh, mesh_from_sheet3d_with_spectral, sheet3d_caption,
};
use crate::scene2d::Scene2D;
use crate::surfaces_elastic::SurfaceShapeMode;
use crate::surfaces_tab::{SurfacesTabState, empty_net};
use crate::view3d::View3dState;

mod source_key;
pub(crate) mod spectrum_ui;
pub(crate) mod spectrum_worker;
#[cfg(test)]
pub(crate) mod tests;
mod ui;

use source_key::{SourceKey, build_source_key};
pub use spectrum_worker::SpectrumColorOverlay;
use spectrum_worker::SpectrumPanelState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnalysisTarget {
    PlaneLattice,
    Surface,
    #[default]
    SingleDefect,
    CombinedDefects,
}

impl AnalysisTarget {
    pub const ALL: [Self; 4] = [
        Self::PlaneLattice,
        Self::Surface,
        Self::SingleDefect,
        Self::CombinedDefects,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::PlaneLattice => "2D Lattice",
            Self::Surface => "Surface",
            Self::SingleDefect => "Single Defect",
            Self::CombinedDefects => "Combined Defects",
        }
    }
}

/// Borrowed workspace tabs, backend, and viewports needed while displaying `3. Analysis`.
pub(crate) struct AnalysisEnv<'a> {
    pub item: &'a CatalogueItemUi,
    pub backend: &'a AppBackend,
    pub surfaces: &'a mut SurfacesTabState,
    pub defects: &'a mut DefectsTabState,
    pub combine: &'a mut CombineTabState,
    pub canvas: &'a mut Canvas2D,
    pub view3d: &'a mut View3dState,
}

pub struct AnalysisModeState {
    pub target: AnalysisTarget,
    pub view_mode: DefectViewMode,
    pub plane_radius: f64,
    pub sheet_opts: Sheet3dViewOpts,
    pub(crate) spectrum: SpectrumPanelState,
    plane_key: Option<(String, Recipe, Frame2D, i64, i64)>,
    pub(crate) plane_crystal: BuiltCrystal,
    pub(crate) plane_sheet: Sheet3DResult,
    plane_title: String,
    plane_error: Option<String>,
    last_source: Option<SourceKey>,
    pub(crate) canvas_dirty: bool,
    pub(crate) view3d_dirty: bool,
    reset_camera_next: bool,
}

impl Default for AnalysisModeState {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisModeState {
    pub fn new() -> Self {
        Self {
            target: AnalysisTarget::SingleDefect,
            view_mode: DefectViewMode::Split2D3D,
            plane_radius: 4.2,
            sheet_opts: Sheet3dViewOpts::default(),
            spectrum: SpectrumPanelState::new(),
            plane_key: None,
            plane_crystal: BuiltCrystal {
                pos: Vec::new(),
                ref_pos: Vec::new(),
                basis_idx: Vec::new(),
                side: Vec::new(),
                cone_sites: Vec::new(),
                edges: Vec::new(),
                adj: Vec::new(),
                wall_sigma: None,
                is_non_orientable: false,
            },
            plane_sheet: crate::combine_worker::empty_sheet3d(),
            plane_title: String::new(),
            plane_error: None,
            last_source: None,
            canvas_dirty: true,
            view3d_dirty: true,
            reset_camera_next: true,
        }
    }

    pub fn on_enter(&mut self) {
        self.canvas_dirty = true;
        self.view3d_dirty = true;
        self.reset_camera_next = true;
    }

    pub fn select_target(&mut self, target: AnalysisTarget) {
        if self.target != target {
            self.target = target;
            self.spectrum.clear_for_switch();
            self.last_source = None;
            self.canvas_dirty = true;
            self.view3d_dirty = true;
            self.reset_camera_next = true;
        }
    }

    pub fn reset_transient_for_session(&mut self) {
        self.spectrum.clear_for_switch();
        self.last_source = None;
        self.plane_key = None;
        self.canvas_dirty = true;
        self.view3d_dirty = true;
        self.reset_camera_next = true;
    }

    pub fn is_busy(&self) -> bool {
        self.spectrum.is_busy()
    }

    pub fn supports_seam_sign(
        &self,
        defects: &DefectsTabState,
        combine: &CombineTabState,
    ) -> bool {
        match self.target {
            AnalysisTarget::PlaneLattice | AnalysisTarget::Surface => false,
            AnalysisTarget::SingleDefect => defects.last_crystal.wall_sigma.is_some(),
            AnalysisTarget::CombinedDefects => combine.last_build.crystal.wall_sigma.is_some(),
        }
    }

    pub fn allow_flux(&self) -> bool {
        !matches!(self.target, AnalysisTarget::Surface)
    }

    pub(crate) fn ensure_plane_source(&mut self, item: &CatalogueItemUi) {
        self.plane_radius = self.plane_radius.clamp(2.5, 8.0);
        let rq = (self.plane_radius * 100.0).round() as i64;
        let kq = (self.sheet_opts.kappa_b * 100_000.0).round() as i64;
        let key = (item.group.clone(), item.recipe.clone(), item.frame, rq, kq);
        if self.plane_key.as_ref() == Some(&key) {
            return;
        }
        self.plane_key = Some(key);
        self.plane_title = format!("{} ({}) — Pristine disk (R={:.1})", item.label, item.group, self.plane_radius);
        match defect_context(item) {
            Ok(ctx) => {
                let crystal = build_pristine_disk(&ctx.lat_info, self.plane_radius);
                let sheet = relax_sheet_3d(
                    &crystal,
                    &ctx.cell,
                    &Sheet3DOptions { kappa_b: self.sheet_opts.kappa_b, max_iters: 80, grad_tol: 1e-6 },
                    None,
                );
                self.plane_crystal = crystal;
                self.plane_sheet = sheet;
                self.plane_error = None;
            }
            Err(err) => {
                self.plane_crystal = crate::combine_worker::empty_crystal();
                self.plane_sheet = crate::combine_worker::empty_sheet3d();
                self.plane_error = Some(err);
            }
        }
        self.canvas_dirty = true;
        self.view3d_dirty = true;
    }

    fn current_source_key(
        &self,
        item: &CatalogueItemUi,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
    ) -> SourceKey {
        build_source_key(self, item, surfaces, defects, combine)
    }

    pub(crate) fn sync_and_poll(&mut self, ctx: &egui::Context, env: &mut AnalysisEnv<'_>) {
        match self.target {
            AnalysisTarget::PlaneLattice => {
                self.ensure_plane_source(env.item);
            }
            AnalysisTarget::Surface => {
                if env.surfaces.needs_sync_for(env.item) && !env.surfaces.is_building() {
                    env.surfaces.build_quiet_for(env.item);
                    self.canvas_dirty = true;
                    self.view3d_dirty = true;
                }
                if env.surfaces.elastic.poll(ctx) {
                    self.view3d_dirty = true;
                }
            }
            AnalysisTarget::SingleDefect => {
                let want = (env.item.group.as_str(), &env.item.recipe, env.item.frame);
                let have = (
                    env.defects.last_item.group.as_str(),
                    &env.defects.last_item.recipe,
                    env.defects.last_item.frame,
                );
                if want != have {
                    env.defects.reload_for_item(env.item, env.backend, env.canvas, false);
                    env.defects.submit_rebuild(env.item, true);
                }
                if env.defects.poll_worker_quiet(ctx) {
                    self.canvas_dirty = true;
                    self.view3d_dirty = true;
                }
            }
            AnalysisTarget::CombinedDefects => {
                if env.combine.poll(Some(ctx), env.item, None) {
                    self.canvas_dirty = true;
                    self.view3d_dirty = true;
                }
            }
        }

        if self.spectrum.poll(Some(ctx)) {
            self.canvas_dirty = true;
            self.view3d_dirty = true;
        }

        let key = self.current_source_key(env.item, env.surfaces, env.defects, env.combine);
        if self.last_source.as_ref() != Some(&key) {
            self.last_source = Some(key);
            self.spectrum.clear_for_switch();
            self.canvas_dirty = true;
            self.view3d_dirty = true;
        }

        if self.canvas_dirty {
            self.sync_canvas(env.surfaces, env.defects, env.combine, env.canvas);
        }
        if self.view3d_dirty && (self.target == AnalysisTarget::Surface || self.view_mode.uses_3d()) {
            self.sync_view3d(env.surfaces, env.defects, env.combine, env.view3d);
        }
    }

    pub fn submit_active_source(
        &mut self,
        item: &CatalogueItemUi,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
        ctx: Option<&egui::Context>,
    ) {
        if self.target == AnalysisTarget::PlaneLattice {
            self.ensure_plane_source(item);
        }
        self.last_source = Some(self.current_source_key(item, surfaces, defects, combine));
        if self.spectrum.color_overlay == SpectrumColorOverlay::None {
            self.spectrum.color_overlay = SpectrumColorOverlay::LdosMultiplet;
        }
        match self.target {
            AnalysisTarget::PlaneLattice => {
                let sheet = (!self.plane_sheet.xyz.is_empty()).then_some(&self.plane_sheet);
                self.spectrum.submit_crystal(&self.plane_crystal, sheet, ctx);
            }
            AnalysisTarget::Surface => {
                if let Some(mesh) = &surfaces.current_mesh {
                    let mut m = mesh.clone();
                    if surfaces.elastic.shape_mode == SurfaceShapeMode::ElasticFvK
                        && let Some(el) = &surfaces.elastic.last_result
                    {
                        m.xyz = el.xyz.clone();
                        if let (Some(cov), Some(cxyz)) = (m.cover.as_deref_mut(), el.cover_xyz.as_ref()) {
                            cov.xyz = cxyz.clone();
                        }
                    }
                    self.spectrum.submit_surface(&m, ctx);
                } else {
                    self.spectrum.clear_for_switch();
                    self.spectrum.last_error = surfaces
                        .refusal_msg
                        .clone()
                        .or_else(|| Some(String::from("No valid surface geometry is built yet.")));
                }
            }
            AnalysisTarget::SingleDefect => {
                let (c, s) = if defects.sheet_opts.show_flat_baseline {
                    (&defects.flat_crystal, &defects.flat_sheet)
                } else {
                    (&defects.last_crystal, &defects.last_sheet)
                };
                self.spectrum.submit_crystal(c, Some(s), ctx);
            }
            AnalysisTarget::CombinedDefects => {
                let (c, s) = if combine.sheet_opts.show_flat_baseline {
                    (&combine.last_sheet3d.flat_crystal, &combine.last_sheet3d.flat_sheet)
                } else {
                    (&combine.last_build.crystal, &combine.last_sheet3d.sheet)
                };
                self.spectrum.submit_crystal(c, Some(s), ctx);
            }
        }
        self.canvas_dirty = true;
        self.view3d_dirty = true;
    }

    pub fn sync_canvas(
        &mut self,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
        canvas: &mut Canvas2D,
    ) {
        let reset = self.reset_camera_next;
        self.reset_camera_next = false;
        self.canvas_dirty = false;
        canvas.set_circuit_overlay(None);

        let ov = spectral_overlay(&self.spectrum);
        match self.target {
            AnalysisTarget::PlaneLattice => {
                let note = self.plane_error.clone().unwrap_or_else(|| {
                    format!(
                        "{} sites, {} bonds (tight-binding analysis source)",
                        self.plane_crystal.pos.len(),
                        self.plane_crystal.edges.len()
                    )
                });
                let title = self.plane_title.clone();
                canvas.set_scene_with_title(
                    Scene2D::CrystalPatch {
                        crystal: self.plane_crystal.clone(),
                        title: title.clone(),
                        note,
                    },
                    Some(title),
                    reset,
                );
                canvas.set_gauge_overlay(ov);
            }
            AnalysisTarget::Surface => {
                if let Some(mesh) = &surfaces.current_mesh {
                    let net_title = format!("{} — {} — Tight-Binding Surface", surfaces.active_disp, mesh.name);
                    canvas.set_scene_with_title(
                        Scene2D::SurfaceNet { mesh: mesh.clone() },
                        Some(net_title),
                        reset,
                    );
                    canvas.set_gauge_overlay(ov);
                } else {
                    canvas.set_scene(Scene2D::SurfaceNet { mesh: empty_net(surfaces.target.label()) }, reset);
                    canvas.set_gauge_overlay(None);
                }
            }
            AnalysisTarget::SingleDefect => {
                let crystal = if defects.sheet_opts.show_flat_baseline {
                    defects.flat_crystal.clone()
                } else {
                    defects.last_crystal.clone()
                };
                let title = defects.readout.title.clone();
                canvas.set_scene_with_title(
                    Scene2D::CrystalPatch {
                        crystal,
                        title: title.clone(),
                        note: defects.readout.note.clone(),
                    },
                    Some(title),
                    reset,
                );
                canvas.set_gauge_overlay(ov);
            }
            AnalysisTarget::CombinedDefects => {
                let b = &combine.last_build;
                let title = if b.charge_label.is_empty() {
                    String::from("Combined defects (Analysis)")
                } else {
                    b.charge_label.clone()
                };
                canvas.set_scene_with_title(
                    Scene2D::CrystalPatch {
                        crystal: b.crystal.clone(),
                        title: title.clone(),
                        note: b.note.clone(),
                    },
                    Some(title),
                    reset,
                );
                canvas.set_gauge_overlay(ov);
            }
        }
    }

    pub fn sync_view3d(
        &mut self,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
        view3d: &mut View3dState,
    ) {
        self.view3d_dirty = false;
        let scalar_owned = self.spectrum.active_scalar_overlay();
        let scalar_ref = scalar_owned.as_ref().map(|(v, d)| (v.as_slice(), *d));

        match self.target {
            AnalysisTarget::Surface => {
                let Some(mesh) = &surfaces.current_mesh else {
                    let msg = surfaces
                        .refusal_msg
                        .as_deref()
                        .unwrap_or("Surface unavailable");
                    view3d.set_mesh(GpuMeshData::default(), format!("Analysis Surface: {msg}"));
                    view3d.set_legend(None);
                    return;
                };
                let color_mode = match self.spectrum.color_overlay {
                    SpectrumColorOverlay::LdosMultiplet => SurfaceColor3d::SpectralLdos,
                    SpectrumColorOverlay::SpectralCharge => SurfaceColor3d::SpectralCharge,
                    SpectrumColorOverlay::None => surfaces.elastic.color_mode,
                };
                let use_el: Option<&SurfaceElasticResult> = if surfaces.elastic.shape_mode == SurfaceShapeMode::ElasticFvK {
                    surfaces.elastic.last_result.as_ref()
                } else {
                    None
                };
                let cov = if surfaces.show_cover && mesh.cover.is_some() { " [S² cover]" } else { "" };
                let gpu = elastic_surface_gpu_mesh(mesh, use_el, color_mode, surfaces.show_cover, scalar_ref);
                let cap = format!("Analysis: {}{cov} | V={} F={}", mesh.name, gpu.vertex_count, gpu.face_count);
                view3d.set_mesh(gpu, cap);
                view3d.set_legend(surface_color_legend(use_el, color_mode, scalar_ref));
            }
            AnalysisTarget::PlaneLattice | AnalysisTarget::SingleDefect | AnalysisTarget::CombinedDefects => {
                let (c, s, base_opts, title): (&BuiltCrystal, &Sheet3DResult, Sheet3dViewOpts, &str) = match self.target {
                    AnalysisTarget::PlaneLattice => (
                        &self.plane_crystal,
                        &self.plane_sheet,
                        self.sheet_opts,
                        &self.plane_title,
                    ),
                    AnalysisTarget::SingleDefect => (
                        if defects.sheet_opts.show_flat_baseline { &defects.flat_crystal } else { &defects.last_crystal },
                        if defects.sheet_opts.show_flat_baseline { &defects.flat_sheet } else { &defects.last_sheet },
                        defects.sheet_opts,
                        &defects.readout.title,
                    ),
                    AnalysisTarget::CombinedDefects => (
                        if combine.sheet_opts.show_flat_baseline {
                            &combine.last_sheet3d.flat_crystal
                        } else {
                            &combine.last_build.crystal
                        },
                        if combine.sheet_opts.show_flat_baseline {
                            &combine.last_sheet3d.flat_sheet
                        } else {
                            &combine.last_sheet3d.sheet
                        },
                        combine.sheet_opts,
                        "Combined defects",
                    ),
                    AnalysisTarget::Surface => unreachable!(),
                };
                let mut opts = base_opts;
                opts.color_mode = match self.spectrum.color_overlay {
                    SpectrumColorOverlay::LdosMultiplet => Sheet3dColorMode::SpectralLdos,
                    SpectrumColorOverlay::SpectralCharge => Sheet3dColorMode::SpectralCharge,
                    SpectrumColorOverlay::None => Sheet3dColorMode::FaceRingSize,
                };
                let gpu = mesh_from_sheet3d_with_spectral(c, s, &opts, scalar_ref);
                let legend = sheet_color_legend(c, s, &opts, scalar_ref);
                let cap = sheet3d_caption(title, s, &opts);
                view3d.set_mesh(gpu, cap);
                view3d.set_legend(legend);
            }
        }
    }

    pub fn export_active_obj(
        &self,
        item: &CatalogueItemUi,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
        path: &std::path::Path,
    ) -> Result<String, String> {
        match self.target {
            AnalysisTarget::Surface => surfaces.export_active_obj(path),
            AnalysisTarget::PlaneLattice => export_sheet_obj(&self.plane_title, &self.plane_sheet, path),
            AnalysisTarget::SingleDefect => defects.export_active_obj(path),
            AnalysisTarget::CombinedDefects => combine.export_active_obj(item, path),
        }
    }
}

fn spectral_overlay(spectrum: &SpectrumPanelState) -> Option<CrystalOverlayData> {
    let (vals, diverging) = spectrum.active_scalar_overlay()?;
    let mut ov = spectral_overlay_data(&vals, diverging);
    ov.note_suffix = Some(String::from(if diverging {
        "Spectral charge δρ(μ)"
    } else {
        "LDOS |ψ|²"
    }));
    Some(ov)
}
