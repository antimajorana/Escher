use crate::app::{NavTab, PapertillingApp};

use super::{CompactPanel, compact_layout};

impl PapertillingApp {
    pub(crate) fn show_right_inspector(&mut self, ui: &mut egui::Ui) {
        let compact = compact_layout(ui.ctx());
        let open = if compact { self.compact_panel == CompactPanel::Inspector } else { self.right_inspector_open };
        if !open {
            #[cfg(test)]
            {
                self.test_rects.right = egui::Rect::NOTHING;
            }
            return;
        }
        let tokens = self.theme.tokens();
        let max_width = if compact { (0.40 * ui.available_width()).clamp(260.0, 440.0) } else { 440.0 };
        let resp = egui::Panel::right(if compact { "compact_right_inspector" } else { "right_inspector" })
            .resizable(true)
            .default_size(self.right_inspector_width.min(max_width))
            .size_range(260.0..=max_width)
            .show(ui, |ui| {
                ui.set_clip_rect(ui.max_rect());
                ui.horizontal(|ui| {
                    ui.strong("Inspector");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("Collapse inspector").clicked() {
                            self.toggle_side_panel(CompactPanel::Inspector, ui.ctx());
                        }
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width());
                        match self.active_tab {
                    NavTab::Gallery => {
                        self.gallery.show_inspector_panel(
                            ui,
                            &self.backend,
                            &mut self.canvas2d,
                            &tokens,
                            self.advanced,
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
                            self.advanced,
                        );
                        if let Some(specs) = self.defects.take_open_in_combine() {
                            self.open_specs_in_combine(specs);
                        }
                    }
                    NavTab::Combine => {
                        let item = self.gallery.active_item().clone();
                        self.combine.show_inspector_panel(
                            ui,
                            &item,
                            &mut self.canvas2d,
                            &tokens,
                            self.advanced,
                        );
                        self.drain_combine_fusion_cavity();
                    }
                    NavTab::Fuse => {
                        self.fuse.show_inspector_panel(ui, &tokens);
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
                }
                    });
            });
        if !compact {
            self.right_inspector_width = resp.response.rect.width().clamp(260.0, 440.0);
        }
        #[cfg(test)]
        {
            self.test_rects.right = resp.response.rect;
        }
    }
}
