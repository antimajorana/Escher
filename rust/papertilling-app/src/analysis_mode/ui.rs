//! Left sidebar, right Tight-Binding Spectrum inspector, and central 2D/3D viewport for `3. Analysis`.

use papertilling_core::types::BuiltCrystal;

use super::spectrum_ui::show_spectrum_panel;
use super::{AnalysisEnv, AnalysisModeState, AnalysisTarget};
use crate::backend::SurfaceTargetUi;
use crate::combine::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::mesh3d::DefectViewMode;
use crate::surfaces_elastic::SurfaceShapeMode;
use crate::surfaces_tab::SurfacesTabState;
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::show_split_2d_3d;

impl AnalysisModeState {
    pub(crate) fn show_left_panel(&mut self, ui: &mut egui::Ui, env: &mut AnalysisEnv<'_>) {
        self.sync_and_poll(ui.ctx(), env);
        ui.strong("Tight-Binding Analysis Source");
        ui.add_space(3.0);
        ui.small(format!("Lattice: {} ({})", env.item.display_with_orig(), env.item.group));
        ui.add_space(4.0);

        for target in AnalysisTarget::ALL {
            if ui.selectable_label(self.target == target, target.label()).clicked() {
                self.select_target(target);
            }
        }

        ui.separator();
        match self.target {
            AnalysisTarget::PlaneLattice => {
                ui.strong("2D Pristine Patch");
                ui.horizontal(|ui| {
                    ui.label("Radius R:");
                    if ui.add(egui::Slider::new(&mut self.plane_radius, 2.5..=8.0).step_by(0.2)).changed() {
                        self.plane_key = None;
                        self.ensure_plane_source(env.item);
                    }
                });
                ui.small(format!(
                    "Sites: N = {} | Bonds: {}",
                    self.plane_crystal.pos.len(),
                    self.plane_crystal.edges.len()
                ));
                if let Some(err) = &self.plane_error {
                    ui.small(err);
                }
            }
            AnalysisTarget::Surface => {
                ui.strong("Closed / Open Surface");
                ui.horizontal_wrapped(|ui| {
                    for t in SurfaceTargetUi::ALL {
                        if ui.selectable_label(env.surfaces.target == t, t.short_header()).clicked()
                            && env.surfaces.target != t
                        {
                            env.surfaces.target = t;
                            env.surfaces.clamp_params(env.item);
                            env.surfaces.build_quiet_for(env.item);
                            self.canvas_dirty = true;
                            self.view3d_dirty = true;
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("3D Shape:");
                    for sm in [SurfaceShapeMode::AnalyticRadial, SurfaceShapeMode::ElasticFvK] {
                        if ui.selectable_label(env.surfaces.elastic.shape_mode == sm, sm.label()).clicked()
                            && env.surfaces.elastic.shape_mode != sm
                        {
                            env.surfaces.elastic.shape_mode = sm;
                            env.surfaces.build_quiet_for(env.item);
                            self.canvas_dirty = true;
                            self.view3d_dirty = true;
                        }
                    }
                });
                if let Some(mesh) = &env.surfaces.current_mesh {
                    ui.small(format!("Active: {} | V = {} | χ = {}", mesh.name, mesh.sites.len(), mesh.chi));
                } else if let Some(msg) = &env.surfaces.refusal_msg {
                    ui.small(format!("Unavailable: {msg}"));
                }
            }
            AnalysisTarget::SingleDefect => {
                ui.strong("Active Single Defect");
                ui.small(&env.defects.readout.title);
                if !env.defects.readout.charge_summary.is_empty() {
                    ui.small(&env.defects.readout.charge_summary);
                }
                ui.small(format!(
                    "N = {} sites (cap {})",
                    env.defects.last_crystal.pos.len(),
                    super::spectrum_worker::SPECTRUM_MAX_SITES
                ));
                if env.defects.is_busy() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.small("Rebuilding defect source…");
                    });
                }
            }
            AnalysisTarget::CombinedDefects => {
                ui.strong("Active Combined Defects");
                ui.small(format!(
                    "{} pins | N = {} sites (cap {})",
                    env.combine.specs.len(),
                    env.combine.last_build.crystal.pos.len(),
                    super::spectrum_worker::SPECTRUM_MAX_SITES
                ));
                if !env.combine.last_build.charge_label.is_empty() {
                    ui.small(&env.combine.last_build.charge_label);
                }
                if env.combine.is_building() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.small("Rebuilding combined source…");
                    });
                }
            }
        }
    }

    pub(crate) fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        env: &mut AnalysisEnv<'_>,
        tokens: &ThemeTokens,
    ) {
        self.sync_and_poll(ui.ctx(), env);
        let allow_flux = self.allow_flux();
        let allow_seam_sign = self.supports_seam_sign(env.defects, env.combine);
        let ev = show_spectrum_panel(ui, &mut self.spectrum, allow_flux, allow_seam_sign, tokens);
        if ev.recompute {
            let ctx = ui.ctx().clone();
            self.submit_active_source(env.item, env.surfaces, env.defects, env.combine, Some(&ctx));
        } else if ev.redraw_colors {
            self.canvas_dirty = true;
            self.view3d_dirty = true;
        }
    }

    pub(crate) fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        env: &mut AnalysisEnv<'_>,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        self.sync_and_poll(ui.ctx(), env);

        ui.horizontal_wrapped(|ui| {
            ui.strong("View:");
            for vm in DefectViewMode::ALL {
                if ui.selectable_label(self.view_mode == vm, vm.label()).clicked() && self.view_mode != vm {
                    self.view_mode = vm;
                    self.view3d_dirty = true;
                }
            }
            ui.separator();
            ui.small(format!("Target: {}", self.target.label()));
            if let Some(c) = self.active_crystal_len(env.surfaces, env.defects, env.combine) {
                ui.small(format!("N = {c}"));
            }
            if self.is_busy() {
                ui.spinner();
            }
        });
        ui.add_space(2.0);

        match self.view_mode {
            DefectViewMode::Planar2D => {
                ui.strong("2D Spectral View (LDOS / excess charge overlay)");
                ui.add_space(2.0);
                env.canvas.show(ui, tokens);
            }
            DefectViewMode::Sheet3D => {
                ui.strong("3D Spectral View (orbit & zoom)");
                ui.add_space(2.0);
                env.view3d.show(ui, tokens);
            }
            DefectViewMode::Split2D3D => {
                show_split_2d_3d(
                    ui,
                    split_ratio,
                    "analysis_split",
                    tokens,
                    |ui| {
                        ui.strong("2D Spectral View");
                        ui.add_space(2.0);
                        env.canvas.show(ui, tokens);
                    },
                    |ui| {
                        ui.strong("3D Spectral View");
                        ui.add_space(2.0);
                        env.view3d.show(ui, tokens);
                    },
                );
            }
        }
    }

    fn active_crystal_len(
        &self,
        surfaces: &SurfacesTabState,
        defects: &DefectsTabState,
        combine: &CombineTabState,
    ) -> Option<usize> {
        match self.target {
            AnalysisTarget::PlaneLattice => Some(self.plane_crystal.pos.len()),
            AnalysisTarget::Surface => surfaces.current_mesh.as_ref().map(|m| m.sites.len()),
            AnalysisTarget::SingleDefect => {
                let c: &BuiltCrystal = &defects.last_crystal;
                Some(c.pos.len())
            }
            AnalysisTarget::CombinedDefects => Some(combine.last_build.crystal.pos.len()),
        }
    }
}
