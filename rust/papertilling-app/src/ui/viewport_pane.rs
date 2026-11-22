use crate::app::{NavTab, PapertillingApp};
use crate::gallery_tab::filter::group_with_orbifold;
use crate::theme::ThemeTokens;

impl PapertillingApp {
    pub(crate) fn show_viewport_pane(&mut self, ui: &mut egui::Ui) {
        let tokens = self.theme.tokens();
        egui::CentralPanel::default().show(ui, |ui| {
            #[cfg(test)]
            {
                self.test_rects.center = ui.max_rect();
            }
            match self.active_tab {
            NavTab::Gallery => {
                ui.strong(format!(
                    "{} · {}  [drag: pan, scroll/+/-: zoom, 0/double-click: fit]",
                    self.gallery.active_item().display_with_orig(),
                    group_with_orbifold(&self.gallery.active_item().group)
                ));
                ui.add_space(2.0);
                self.canvas2d.show(ui, &tokens);
            }
            NavTab::Surfaces => {
                self.surfaces
                    .poll_background(ui.ctx(), &mut self.canvas2d, &mut self.view3d);
                let mut ratio = self.viewport_split_ratio;
                let exact_2d = self.surfaces.notes.exact_view;
                let not_iso_3d = self.surfaces.notes.picture_note.is_some();
                show_split_2d_3d(
                    ui,
                    &mut ratio,
                    "surfaces_split",
                    &tokens,
                    |ui| {
                        let exact = if exact_2d { " (exact isometric domain)" } else { "" };
                        ui.strong(format!("2D Fundamental Domain / Gluing Net (drag: pan, scroll: zoom){exact}"));
                        ui.add_space(2.0);
                        self.canvas2d.show(ui, &tokens);
                    },
                    |ui| {
                        let not_iso = if not_iso_3d { " — topological picture, not isometric" } else { "" };
                        ui.strong(format!("3D Surface Preview{not_iso}"));
                        ui.add_space(2.0);
                        self.view3d.show(ui, &tokens);
                    },
                );
                self.viewport_split_ratio = ratio;
            }
            NavTab::Defects => {
                let item = self.gallery.active_item().clone();
                self.defects.show_central(
                    ui,
                    &item,
                    &mut self.canvas2d,
                    &mut self.view3d,
                    &mut self.viewport_split_ratio,
                    &tokens,
                );
                if let Some(specs) = self.defects.take_open_in_combine() {
                    self.open_specs_in_combine(specs);
                }
            }
            NavTab::Combine => {
                let item = self.gallery.active_item().clone();
                self.combine.show_central(
                    ui,
                    &item,
                    &self.backend,
                    &mut self.canvas2d,
                    &mut self.view3d,
                    &mut self.viewport_split_ratio,
                    &tokens,
                );
                self.drain_combine_fusion_cavity();
            }
            NavTab::Atlas => {
                self.atlas.show_central(
                    ui,
                    &mut self.canvas2d,
                    &mut self.view3d,
                    &mut self.viewport_split_ratio,
                    &tokens,
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
                self.analysis
                    .show_central(ui, &mut env, &mut self.viewport_split_ratio, &tokens);
            }
            NavTab::Workbench => {
                self.workbench_pump_sources(ui.ctx());
                let mut ratio = self.viewport_split_ratio;
                self.workbench_call(|wb, cx, canvas, view3d| {
                    wb.show_central(ui, cx, canvas, view3d, &mut ratio, &tokens);
                });
                self.viewport_split_ratio = ratio;
            }
        }
        });
    }
}

/// Shared busy indicator for `Defects` and `Combine` while a background build is running.
pub fn show_defect_viewport_header(ui: &mut egui::Ui, is_busy: bool) {
    if is_busy {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.small("Building crystal…");
        });
    }
}

pub fn show_split_2d_3d(
    ui: &mut egui::Ui,
    split_ratio: &mut f32,
    id_salt: &str,
    tokens: &ThemeTokens,
    left_fn: impl FnOnce(&mut egui::Ui),
    right_fn: impl FnOnce(&mut egui::Ui),
) {
    *split_ratio = split_ratio.clamp(0.2, 0.8);
    let total = ui.available_rect_before_wrap();
    let sep_w = 6.0_f32;
    let usable_w = (total.width() - sep_w).max(0.0);
    let min_pane = (usable_w * 0.5).min(80.0);
    let left_w = (usable_w * *split_ratio).clamp(min_pane, (usable_w - min_pane).max(min_pane));
    let left_rect = egui::Rect::from_min_size(total.min, egui::vec2(left_w, total.height()));
    let sep_rect = egui::Rect::from_min_size(
        egui::pos2(left_rect.right(), total.top()),
        egui::vec2(sep_w.min((total.right() - left_rect.right()).max(0.0)), total.height()),
    );
    let right_rect = egui::Rect::from_min_size(
        egui::pos2(sep_rect.right(), total.top()),
        egui::vec2((total.right() - sep_rect.right()).max(0.0), total.height()),
    );

    let sep_resp = ui.interact(sep_rect, ui.make_persistent_id(id_salt), egui::Sense::click_and_drag());
    if sep_resp.hovered() || sep_resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    if sep_resp.double_clicked() {
        *split_ratio = 0.5;
    } else if sep_resp.dragged() && usable_w > 1.0 {
        *split_ratio = (*split_ratio + sep_resp.drag_delta().x / usable_w).clamp(0.2, 0.8);
    }

    ui.scope_builder(egui::UiBuilder::new().max_rect(left_rect), left_fn);
    let sep_col = if sep_resp.hovered() || sep_resp.dragged() {
        tokens.border_focus
    } else {
        tokens.border_strong
    };
    ui.painter().line_segment(
        [sep_rect.center_top(), sep_rect.center_bottom()],
        egui::Stroke::new(if sep_resp.dragged() { 2.0 } else { 1.0 }, sep_col),
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(right_rect), right_fn);
    ui.advance_cursor_after_rect(total);
}
