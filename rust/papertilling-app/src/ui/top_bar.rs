use crate::app::{NavTab, PapertillingApp};

impl PapertillingApp {
    pub(crate) fn show_top_bar(&mut self, ui: &mut egui::Ui) {
        let _resp = egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!("papertilling {}", papertilling_core::VERSION));
                ui.separator();
                for tab in NavTab::ALL {
                    if ui
                        .selectable_label(self.active_tab == tab, tab.label())
                        .clicked()
                    {
                        self.switch_tab(tab);
                    }
                }
                ui.separator();
                self.show_switcher_chip(ui);
                let rev0 = self.gallery.revision();
                self.gallery
                    .show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "top_rename");
                if self.gallery.revision() != rev0 {
                    self.sync_active_tab_after_rename();
                }
                ui.separator();
                if ui
                    .button(format!("Theme: {}", self.theme.label()))
                    .on_hover_text("Toggle between Light (Paper) and Dark (Studio) theme")
                    .clicked()
                {
                    self.theme = self.theme.toggle();
                    self.theme.apply(ui.ctx());
                }
                if ui.button("Reset views").clicked() {
                    self.canvas2d.reset();
                    self.view3d.reset();
                }
                ui.menu_button("Export", |ui| {
                    if ui.button("Export SVG / PNG...").clicked() && !self.smoke_test {
                        ui.close();
                        self.export_dialog();
                    }
                    ui.horizontal(|ui| {
                        ui.label("PNG scale:");
                        for (s, tip) in [(1_u8, "1024 x 1024 px"), (2, "2048 x 2048 px"), (4, "4096 x 4096 px")] {
                            ui.selectable_value(&mut self.png_scale, s, format!("{s}x"))
                                .on_hover_text(format!("2D PNG export at {s}x pixel density: {tip}, same composition"));
                        }
                    });
                    if ui.button("Export .obj (3D)...").clicked() && !self.smoke_test {

                        ui.close();
                        self.export_obj_dialog();
                    }
                    ui.separator();
                    if ui
                        .button("Export .json (Notebook 3)...")
                        .on_hover_text("Export current Combine defect configuration (.json)")
                        .clicked()
                        && !self.smoke_test
                    {
                        ui.close();
                        self.finish_gallery_moves();
                        let item = self.gallery.active_item().clone();
                        self.combine.export_json_dialog(&item);
                        self.export_status = self.combine.json_status.clone();
                    }
                    if ui
                        .button("Import .json...")
                        .on_hover_text("Import a Notebook 3 defect configuration (.json)")
                        .clicked()
                        && !self.smoke_test
                    {
                        ui.close();
                        self.switch_tab(NavTab::Combine);
                        self.combine
                            .import_json_dialog(&self.backend, &mut self.canvas2d);
                        if !self.combine.json_status.is_empty() {
                            self.export_status = self.combine.json_status.clone();
                        }
                    }
                });
                if ui
                    .selectable_label(self.right_inspector_open, "Inspector")
                    .on_hover_text("Toggle Right Inspector panel")
                    .clicked()
                {
                    self.right_inspector_open = !self.right_inspector_open;
                }
            });
        });
        #[cfg(test)]
        {
            self.test_rects.top = _resp.response.rect;
        }
    }
}
