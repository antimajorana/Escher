use crate::app::{NavTab, PapertillingApp};
use crate::backend::CatalogueRuleUi;

impl PapertillingApp {
    pub(crate) fn show_left_rail(&mut self, ui: &mut egui::Ui) {
        if !self.left_rail_open {
            #[cfg(test)]
            {
                self.test_rects.left = egui::Rect::NOTHING;
            }
            return;
        }
        let tokens = self.theme.tokens();
        let resp = egui::Panel::left("left_rail")
            .resizable(true)
            .default_size(self.left_rail_width)
            .size_range(260.0..=420.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| match self.active_tab {
                    NavTab::Gallery => {
                        let _ = self.gallery.show_left_panel(
                            ui,
                            &self.backend,
                            &mut self.canvas2d,
                            self.advanced,
                        );
                    }
                    NavTab::Surfaces => {
                        let item = self.gallery.active_item().clone();
                        if let Some(new_grp) = self.surfaces.show_left_panel(
                            ui,
                            &item,
                            &self.backend,
                            &mut self.canvas2d,
                            &mut self.view3d,
                            &tokens,
                        ) {
                            self.gallery.rule = CatalogueRuleUi::Points;
                            self.gallery.select_group_by_symbol(
                                &new_grp,
                                &self.backend,
                                &mut self.canvas2d,
                            );
                            let active = self.gallery.active_item().clone();
                            self.surfaces.activate_and_sync(
                                &active,
                                &self.backend,
                                &mut self.canvas2d,
                                &mut self.view3d,
                            );
                        }
                    }
                    NavTab::Defects => {
                        let item = self.gallery.active_item().clone();
                        let _ = self.defects.show_left_panel(
                            ui,
                            &item,
                            &self.backend,
                            &mut self.canvas2d,
                            &tokens,
                        );
                    }
                    NavTab::Combine => {
                        let item = self.gallery.active_item().clone();
                        let _ = self.combine.show_left_panel(
                            ui,
                            &item,
                            &self.backend,
                            &mut self.canvas2d,
                            &tokens,
                            self.advanced,
                        );
                    }
                    NavTab::Atlas => {
                        let offered = self.atlas_offered_tabs();
                        self.atlas.show_left_panel(
                            ui,
                            &mut self.canvas2d,
                            &mut self.view3d,
                            offered,
                        );
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
                        self.analysis.show_left_panel(ui, &mut env);
                    }
                    NavTab::Workbench => {
                        self.workbench_call(|wb, cx, _, _| wb.show_left_panel(ui, cx));
                    }
                });
            });
        self.left_rail_width = resp.response.rect.width().clamp(260.0, 420.0);
        #[cfg(test)]
        {
            self.test_rects.left = resp.response.rect;
        }
    }
}
