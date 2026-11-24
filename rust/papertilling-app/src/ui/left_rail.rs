use crate::app::{NavTab, PapertillingApp};
use crate::backend::CatalogueRuleUi;

use super::{CompactPanel, compact_layout};

impl PapertillingApp {
    pub(crate) fn show_left_rail(&mut self, ui: &mut egui::Ui) {
        let compact = compact_layout(ui.ctx());
        let open = if compact { self.compact_panel == CompactPanel::Controls } else { self.left_rail_open };
        if !open {
            #[cfg(test)]
            {
                self.test_rects.left = egui::Rect::NOTHING;
            }
            return;
        }
        let tokens = self.theme.tokens();
        let max_width = if compact { (0.40 * ui.available_width()).clamp(260.0, 420.0) } else { 420.0 };
        let resp = egui::Panel::left(if compact { "compact_left_rail" } else { "left_rail" })
            .resizable(true)
            .default_size(self.left_rail_width.min(max_width))
            .size_range(260.0..=max_width)
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
                        self.drain_combine_fusion_cavity();
                    }
                    NavTab::Fuse => {
                        let item = self.gallery.active_item().clone();
                        self.fuse.show_left_panel(ui, &item, &mut self.canvas2d, &tokens);
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
        if !compact {
            self.left_rail_width = resp.response.rect.width().clamp(260.0, 420.0);
        }
        #[cfg(test)]
        {
            self.test_rects.left = resp.response.rect;
        }
    }
}
