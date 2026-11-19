//! The one-row HUD over the viewport (`Panel::top("top_bar")`). WP0 version: Codex button, mode segment and the
//! rename bar on the left, the full tool cluster on the right. WP-A adds the breadcrumb segments, the lattice
//! thumbnail and the narrow-width truncation (`egui::Sides`).

use crate::app::{AppMode, NavTab, PapertillingApp};
use crate::mesh3d::DefectViewMode;

const UNDO_SCOPE: &str = "Undo and Redo apply to Combine, Fuse & Move";

impl PapertillingApp {
    pub(crate) fn show_hud_bar(&mut self, ui: &mut egui::Ui) {
        let resp = egui::Panel::top("top_bar").exact_size(34.0).show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                self.hud_left(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| self.hud_right(ui));
            });
        });
        let rect = resp.response.rect;
        let content = ui.ctx().content_rect();
        self.deck.host_rect.min = egui::pos2(content.left(), rect.bottom());
        self.deck.host_rect.max.x = content.right();
        #[cfg(test)]
        {
            self.test_rects.top = rect;
        }
    }

    fn hud_left(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("❖ Codex (Tab)")
            .on_hover_text("Command Deck: every geometry, group, tiling, manifold, defect and preset")
            .clicked()
        {
            if self.deck.open {
                self.deck_close();
            } else {
                self.deck_open(self.current_realm(), ui.ctx());
            }
        }
        ui.weak("▸");
        ui.label(self.mode_segment_text());
        let rev0 = self.gallery.revision();
        self.gallery.show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "top_rename");
        if self.gallery.revision() != rev0 {
            self.sync_active_tab_after_rename();
        }
    }

    /// Right cluster, drawn right to left: Theme, Search, View, Export, Session, Presets, Redo, Undo, view toggles.
    fn hud_right(&mut self, ui: &mut egui::Ui) {
        if ui
            .button(format!("Theme: {}", self.theme.label()))
            .on_hover_text("Toggle between Light (Paper) and Dark (Studio) theme")
            .clicked()
        {
            self.theme = self.theme.toggle();
            self.theme.apply(ui.ctx());
        }
        if ui.button("Search (Ctrl+K)").on_hover_text("Search all stored lattices").clicked() {
            if self.deck.open {
                self.deck.focus_filter = true;
            } else {
                self.toggle_switcher();
            }
        }
        ui.menu_button("View", |ui| {
            if ui.button("Reset views").clicked() {
                self.canvas2d.reset();
                self.view3d.reset();
            }
            ui.checkbox(&mut self.left_rail_open, "Sidebar (Ctrl+B)");
            ui.checkbox(&mut self.right_inspector_open, "Inspector (Ctrl+I)");
        });
        self.show_export_menu(ui);
        self.session_menu(ui);
        self.show_presets_menu(ui);
        let in_combine = self.active_mode == AppMode::Defects && self.active_tab == NavTab::Combine;
        let free = in_combine && !self.combine.dragging && !self.combine.cut_dragging;
        let can_redo = free && !self.combine.redo_stack.is_empty();
        if ui.add_enabled(can_redo, egui::Button::new("Redo")).on_disabled_hover_text(UNDO_SCOPE).clicked() {
            self.combine_redo();
        }
        let can_undo = free && !self.combine.undo_stack.is_empty();
        if ui.add_enabled(can_undo, egui::Button::new("Undo")).on_disabled_hover_text(UNDO_SCOPE).clicked() {
            self.combine_undo();
        }
        self.hud_view_toggles(ui);
    }

    /// `2D` / `3D` / `Split` for the tabs with a view mode (Defects, Combine, Analysis); disabled elsewhere.
    fn hud_view_toggles(&mut self, ui: &mut egui::Ui) {
        let target = match self.active_tab {
            NavTab::Defects => Some((&mut self.defects.view_mode, &mut self.defects.view3d_dirty)),
            NavTab::Combine => Some((&mut self.combine.view_mode, &mut self.combine.view3d_dirty)),
            NavTab::Analysis => Some((&mut self.analysis.view_mode, &mut self.analysis.view3d_dirty)),
            _ => None,
        };
        // right to left: Split, 3D, 2D
        let modes = [
            (DefectViewMode::Split2D3D, "Split"),
            (DefectViewMode::Sheet3D, "3D"),
            (DefectViewMode::Planar2D, "2D"),
        ];
        match target {
            Some((mode, dirty)) => {
                for (vm, label) in modes {
                    if ui.selectable_label(*mode == vm, label).clicked() && *mode != vm {
                        *mode = vm;
                        *dirty = true;
                    }
                }
            }
            None => {
                for (_, label) in modes {
                    ui.add_enabled(false, egui::Button::selectable(false, label))
                        .on_disabled_hover_text("This view has a fixed layout");
                }
            }
        }
    }
}
