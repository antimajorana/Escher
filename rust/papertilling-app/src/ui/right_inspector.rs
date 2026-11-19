use crate::app::{NavTab, PapertillingApp};

impl PapertillingApp {
    pub(crate) fn show_right_inspector(&mut self, ui: &mut egui::Ui) {
        if !self.right_inspector_open {
            #[cfg(test)]
            {
                self.test_rects.right = egui::Rect::NOTHING;
            }
            return;
        }
        let tokens = self.theme.tokens();
        let resp = egui::Panel::right("right_inspector")
            .resizable(true)
            .default_size(self.right_inspector_width)
            .size_range(260.0..=440.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Inspector");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("Collapse inspector").clicked() {
                            self.right_inspector_open = false;
                        }
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.active_tab {
                    NavTab::Gallery => {
                        self.gallery.show_inspector_panel(
                            ui,
                            &self.backend,
                            &mut self.canvas2d,
                            &tokens,
                        );
                    }
                    NavTab::Surfaces => {
                        self.surfaces.show_inspector_panel(
                            ui,
                            &mut self.canvas2d,
                            &mut self.view3d,
                            &tokens,
                        );
                    }
                    NavTab::Defects => {
                        let item = self.gallery.active_item().clone();
                        self.defects.show_inspector_panel(
                            ui,
                            &item,
                            &mut self.canvas2d,
                            &tokens,
                        );
                    }
                    NavTab::Combine => {
                        let item = self.gallery.active_item().clone();
                        self.combine.show_inspector_panel(
                            ui,
                            &item,
                            &mut self.canvas2d,
                            &tokens,
                        );
                    }
                    NavTab::Atlas => {
                        self.atlas.show_inspector_panel(ui, &tokens);
                    }
                    NavTab::Analysis => {
                        let item = self.gallery.active_item().clone();
                        let mut env = crate::analysis_mode::AnalysisEnv {
                            item: &item,
                            backend: &self.backend,
                            surfaces: &mut self.surfaces,
                            defects: &mut self.defects,
                            combine: &mut self.combine,
                            canvas: &mut self.canvas2d,
                            view3d: &mut self.view3d,
                        };
                        self.analysis.show_inspector_panel(ui, &mut env, &tokens);
                    }
                    NavTab::Workbench => {
                        self.workbench_call(|wb, cx, _, _| wb.show_inspector_panel(ui, cx, &tokens));
                    }
                });
            });
        self.right_inspector_width = resp.response.rect.width().clamp(260.0, 440.0);
        #[cfg(test)]
        {
            self.test_rects.right = resp.response.rect;
        }
    }
}
