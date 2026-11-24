pub mod lattice_switcher;
pub mod left_rail;
pub mod presets_menu;
pub mod right_inspector;
pub mod shortcuts;
pub mod status_bar;
pub mod top_bar;
pub mod viewport_pane;

use crate::app::PapertillingApp;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CompactPanel {
    #[default]
    Controls,
    Inspector,
    Canvas,
}

pub(crate) fn compact_layout(ctx: &egui::Context) -> bool {
    ctx.content_rect().width() < 1080.0
}

impl PapertillingApp {
    pub(crate) fn show_compact_panel_switcher(&mut self, ui: &mut egui::Ui) {
        if compact_layout(ui.ctx()) {
            egui::Panel::top("compact_panel_switcher").show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.compact_panel, CompactPanel::Controls, "Controls");
                    ui.selectable_value(&mut self.compact_panel, CompactPanel::Inspector, "Inspector");
                    ui.selectable_value(&mut self.compact_panel, CompactPanel::Canvas, "Canvas only");
                });
            });
        }
    }

    pub(crate) fn toggle_side_panel(&mut self, panel: CompactPanel, ctx: &egui::Context) {
        if compact_layout(ctx) {
            self.compact_panel = if self.compact_panel == panel { CompactPanel::Canvas } else { panel };
        } else {
            match panel {
                CompactPanel::Controls => self.left_rail_open = !self.left_rail_open,
                CompactPanel::Inspector => self.right_inspector_open = !self.right_inspector_open,
                CompactPanel::Canvas => {}
            }
        }
    }
}

#[cfg(test)]
mod presets_menu_tests;
#[cfg(test)]
mod shortcuts_tests;
#[cfg(test)]
mod switcher_tests;
#[cfg(test)]
mod tests;
