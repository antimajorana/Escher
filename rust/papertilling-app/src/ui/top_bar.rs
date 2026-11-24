//! The HUD "Save figure" menu (`crate::game_menu::hud_bar`) and its actions. Every action skips its file dialog in
//! `--smoke-test` mode.

use crate::app::{NavTab, PapertillingApp};

impl PapertillingApp {
    pub(crate) fn show_export_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Save figure", |ui| {
            if ui.button("Save 2D figure (SVG / PNG)...").clicked() && !self.smoke_test {
                ui.close();
                self.export_svg_png_action();
            }
            ui.horizontal(|ui| {
                ui.label("PNG scale:");
                for (s, tip) in [(1_u8, "1024 x 1024 px"), (2, "2048 x 2048 px"), (4, "4096 x 4096 px")] {
                    ui.selectable_value(&mut self.png_scale, s, format!("{s}x"))
                        .on_hover_text(format!("PNG export at {s}x pixel density (2D: {tip}; 3D: the viewport's on-screen pixels x {s}), same composition"));
                }
            });
            let showing_3d = match self.active_tab {
                NavTab::Surfaces => true,
                NavTab::Defects => self.defects.sub_tab != crate::defects_tab::DefectSubTab::Combinations
                    && self.defects.view_mode.uses_3d(),
                NavTab::Combine => self.combine.view_mode.uses_3d(),
                _ => false,
            };
            if (self.advanced || showing_3d)
                && ui
                    .button("Save 3D image (PNG)...")
                    .on_hover_text("The 3D view as on screen (caption, legend), at the PNG scale")
                    .clicked()
                && !self.smoke_test
            {
                ui.close();
                self.export_3d_png_action(&ui.ctx().clone());
            }
            if self.advanced && ui.button("Export 3D mesh (.obj)...").clicked() && !self.smoke_test {
                ui.close();
                self.export_obj_action();
            }
            if self.advanced || self.active_tab == NavTab::Combine {
                ui.separator();
                if ui
                    .button("Save defect config (.json)...")
                    .on_hover_text("Save current defect configuration (.json)")
                    .clicked()
                    && !self.smoke_test
                {
                    ui.close();
                    self.export_json_action();
                }
                if ui
                    .button("Load defect config (.json)...")
                    .on_hover_text("Load a saved defect configuration (.json)")
                    .clicked()
                    && !self.smoke_test
                {
                    ui.close();
                    self.import_json_action();
                }
            }
        });
    }

    pub(crate) fn export_svg_png_action(&mut self) {
        if !self.smoke_test {
            self.export_dialog();
        }
    }

    pub(crate) fn export_3d_png_action(&mut self, ctx: &egui::Context) {
        if !self.smoke_test {
            self.export_3d_png_dialog(ctx);
        }
    }

    pub(crate) fn export_obj_action(&mut self) {
        if !self.smoke_test {
            self.export_obj_dialog();
        }
    }

    pub(crate) fn export_json_action(&mut self) {
        if self.smoke_test {
            return;
        }
        self.finish_gallery_moves();
        let item = self.gallery.active_item().clone();
        self.combine.export_json_dialog(&item);
        self.export_status = self.combine.json_status.clone();
    }

    pub(crate) fn import_json_action(&mut self) {
        if self.smoke_test {
            return;
        }
        self.switch_tab(NavTab::Combine);
        self.combine.import_json_dialog(&self.backend, &mut self.canvas2d);
        if !self.combine.json_status.is_empty() {
            self.export_status = self.combine.json_status.clone();
        }
    }
}
