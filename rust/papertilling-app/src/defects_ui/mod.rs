//! Side panel, inspector, and central 2D / 3D / Split viewports of the `Defects` tab.

#[cfg(test)]
use crate::backend::AppBackend;
use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::defects_tab::DefectsTabState;
use crate::mesh3d::DefectViewMode;
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::{show_defect_viewport_header, show_split_2d_3d};
use crate::view3d::View3dState;

pub(crate) mod growth;
mod inspector;
mod left;

impl DefectsTabState {
    #[cfg(test)]
    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
        let tokens = ThemeTokens::paper();
        if self.show_left_panel(ui, active_item, backend, canvas, &tokens) {
            ui.separator();
            self.show_inspector_panel(ui, active_item, canvas, &tokens, true);
        }
    }

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        self.poll_worker(ui.ctx(), canvas);
        let busy = self.is_busy();
        show_defect_viewport_header(ui, busy);
        if self.view3d_dirty {
            if self.view_mode.uses_3d() {
                self.sync_view3d(view3d);
            } else {
                self.view3d_dirty = false;
            }
        }
        ui.add_space(2.0);

        match self.view_mode {
            DefectViewMode::Planar2D => {
                ui.strong("2D Surgical Crystal Patch (irregular core rings, cones, A|B domains)");
                ui.add_space(2.0);
                self.show_2d_patch(ui, active_item, canvas, tokens);
            }
            DefectViewMode::Sheet3D => {
                ui.strong("3D Elastic Sheet Geometry (buckling relieves in-plane defect strain)");
                ui.add_space(2.0);
                view3d.show(ui, tokens);
            }
            DefectViewMode::Split2D3D => {
                show_split_2d_3d(
                    ui,
                    split_ratio,
                    "defects_split",
                    tokens,
                    |ui| {
                        ui.strong("2D Surgical Crystal Patch");
                        ui.add_space(2.0);
                        self.show_2d_patch(ui, active_item, canvas, tokens);
                    },
                    |ui| {
                        ui.strong("3D Elastic Sheet Geometry (orbit & zoom)");
                        ui.add_space(2.0);
                        view3d.show(ui, tokens);
                    },
                );
            }
        }
    }

    fn show_2d_patch(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) {
        if self.inspectors.circuit.enabled {
            let inter = canvas.show_interactive(ui, &[], false, None, tokens);
            if let Some(v0) = inter.clicked_circuit_vertex {
                self.inspectors.circuit.base_v0 = Some(v0);
                self.refresh_and_sync_inspectors(active_item, canvas);
            }
        } else {
            canvas.show(ui, tokens);
        }
    }
}
