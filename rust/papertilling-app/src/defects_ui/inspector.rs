use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::defects_tab::DefectsTabState;
use crate::theme::ThemeTokens;

impl DefectsTabState {
    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
        advanced: bool,
    ) {
        if self.initializing {
            return;
        }
        ui.strong("Topological Charge & Core");
        if self.is_busy() || self.error.is_some() {
            ui.horizontal(|ui| {
                if self.is_busy() {
                    ui.spinner();
                }
                ui.label("Previous readout — stale");
            });
        }
        ui.label(&self.readout.charge_summary);
        ui.small(&self.readout.holonomy_detail);
        ui.small(format!("Core structure: {}", self.readout.core_rings));
        if !self.readout.note.is_empty() {
            ui.colored_label(
                tokens.status_warn,
                format!("Note: {}", self.readout.note),
            );
        }

        let (circ_dirty, ov_dirty) = self.inspectors.show_controls_ui(ui, "def_insp", 0, tokens);
        if circ_dirty || ov_dirty {
            self.refresh_and_sync_inspectors(active_item, canvas);
        }
        if advanced {
            crate::bieberbach_ui::show_bieberbach_inspector_section(
                ui,
                &active_item.group,
                &active_item.recipe,
                &self.inspectors,
                "def_bieberbach",
                tokens,
            );
        }
    }
}
