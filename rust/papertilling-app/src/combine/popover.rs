//! Compact floating pin inspector popover rendered next to the selected pin on `Canvas2D`.

use super::CombineTabState;
use super::history::{CutDragResponse, show_cut_angle_drag_value};
use crate::backend::CatalogueItemUi;
use crate::canvas2d::{Canvas2D, cut_ray_handle_screen_pos};
use crate::live_combine::cuts::{default_cut_angle_deg, pin_explicit_cut_vec};
use crate::live_combine::{spec_short_label, spec_world_pos};
use crate::theme::ThemeTokens;

pub(crate) const POPOVER_SIZE: egui::Vec2 = egui::vec2(236.0, 112.0);

#[derive(Clone, Copy)]
enum PopoverAction {
    Close,
    Flip,
    AutoAlign,
    Delete,
}

/// Place the popover box near `sp` without covering `sp` or `avoid_handle`, clamped inside `canvas_rect`.
pub(crate) fn compute_popover_pos(
    sp: egui::Pos2,
    avoid_handle: Option<egui::Pos2>,
    canvas_rect: egui::Rect,
) -> egui::Pos2 {
    let usable = canvas_rect.shrink(6.0);
    let max_x = (usable.right() - POPOVER_SIZE.x).max(usable.left());
    let max_y = (usable.bottom() - POPOVER_SIZE.y).max(usable.top());
    let raw_candidates = [
        sp + egui::vec2(18.0, -12.0 - POPOVER_SIZE.y),
        sp + egui::vec2(18.0, 20.0),
        sp + egui::vec2(-18.0 - POPOVER_SIZE.x, -12.0 - POPOVER_SIZE.y),
        sp + egui::vec2(-18.0 - POPOVER_SIZE.x, 20.0),
    ];
    let clamp_pt = |p: egui::Pos2| egui::pos2(p.x.clamp(usable.left(), max_x), p.y.clamp(usable.top(), max_y));
    let clear = |r: egui::Rect| {
        let pad = r.expand(14.0);
        !pad.contains(sp) && avoid_handle.is_none_or(|hp| !pad.contains(hp))
    };
    for raw in raw_candidates {
        let c = clamp_pt(raw);
        if (c - raw).length_sq() < 1.0 && clear(egui::Rect::from_min_size(c, POPOVER_SIZE)) {
            return c;
        }
    }
    for raw in raw_candidates {
        let c = clamp_pt(raw);
        if clear(egui::Rect::from_min_size(c, POPOVER_SIZE)) {
            return c;
        }
    }
    raw_candidates
        .into_iter()
        .map(clamp_pt)
        .max_by(|a, b| {
            let score = |p: egui::Pos2| {
                let r = egui::Rect::from_min_size(p, POPOVER_SIZE);
                let ds = r.distance_to_pos(sp);
                avoid_handle.map_or(ds, |hp| ds.min(r.distance_to_pos(hp)))
            };
            score(*a).total_cmp(&score(*b))
        })
        .unwrap_or_else(|| clamp_pt(raw_candidates[0]))
}

impl CombineTabState {
    pub(super) fn show_pin_popover(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) {
        #[cfg(test)]
        {
            self.last_popover_rect = None;
        }
        let Some(idx) = self.selected_pin else {
            return;
        };
        if !self.popover_open
            || idx >= self.specs.len()
            || self.dragging
            || self.cut_dragging
            || self.lattice_unavailable()
        {
            return;
        }
        let canvas_rect = canvas.last_rect;
        if canvas_rect == egui::Rect::NOTHING || canvas_rect.width() < 64.0 || canvas_rect.height() < 64.0 {
            return;
        }

        let spec = &self.specs[idx];
        let pos_w = spec_world_pos(&self.info, idx, spec);
        let sp = canvas.world_to_screen_at(canvas_rect, [pos_w[0] as f32, pos_w[1] as f32]);
        let avoid_handle = canvas
            .cut_rays()
            .iter()
            .find(|r| r.spec_index == Some(idx))
            .and_then(|r| cut_ray_handle_screen_pos(canvas, canvas_rect, r));
        let pop_pos = compute_popover_pos(sp, avoid_handle, canvas_rect);
        let label = spec_short_label(&self.info, spec);
        let explicit_cut = pin_explicit_cut_vec(spec);
        let deg = explicit_cut.map_or_else(
            || default_cut_angle_deg(&self.info, &self.specs, idx, &self.last_build.applied_cuts),
            |c| c[1].atan2(c[0]).to_degrees(),
        );
        let has_explicit = explicit_cut.is_some();

        let mut action: Option<PopoverAction> = None;
        let mut cut_ev: Option<CutDragResponse> = None;

        let area_resp = egui::Area::new(egui::Id::unique("combine_pin_popover"))
            .fixed_pos(pop_pos)
            .default_size(POPOVER_SIZE)
            .constrain_to(canvas_rect.shrink(6.0))
            .order(egui::Order::Middle)
            .fade_in(false)
            .show(ui.ctx(), |ui| {
                ui.set_min_width(POPOVER_SIZE.x - 16.0);
                ui.set_max_width(POPOVER_SIZE.x);
                egui::Frame::NONE
                    .fill(tokens.bg_popover)
                    .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                    .corner_radius(6.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.push_id(("pin_popover", idx), |ui| {
                            ui.horizontal(|ui| {
                                ui.strong(format!("#{}", idx + 1));
                                ui.label(&label);
                                ui.small(format!("({:+.2}, {:+.2})", pos_w[0], pos_w[1]));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui
                                        .small_button("×")
                                        .on_hover_text("Close popover (Esc)")
                                        .clicked()
                                    {
                                        action = Some(PopoverAction::Close);
                                    }
                                });
                            });
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.small("Cut Angle:");
                                let ev = show_cut_angle_drag_value(ui, deg, 1.5);
                                if ev.started || ev.changed || ev.finished {
                                    cut_ev = Some(ev);
                                }
                                if !has_explicit {
                                    ui.colored_label(tokens.text_muted, "(auto)");
                                }
                            });
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .small_button("Flip 180°")
                                    .on_hover_text("Rotate branch-cut / half-wall ray by 180°")
                                    .clicked()
                                {
                                    action = Some(PopoverAction::Flip);
                                }
                                if ui
                                    .add_enabled(has_explicit, egui::Button::new("Auto-Align").small())
                                    .clicked()
                                {
                                    action = Some(PopoverAction::AutoAlign);
                                }
                                if ui.small_button("Delete").clicked() {
                                    action = Some(PopoverAction::Delete);
                                }
                            });
                        });
                    });
            });

        #[cfg(test)]
        {
            self.last_popover_rect = Some(area_resp.response.rect);
        }
        let _ = area_resp;

        if let Some(ev) = cut_ev
            && let Some(preview_only) = self.apply_cut_drag_response(idx, ev)
        {
            self.sync_selection_to_canvas(canvas);
            self.request_build(gallery_item, preview_only, !preview_only && self.auto_compare, false);
        }
        match action {
            Some(PopoverAction::Close) => {
                if self.commit_cut_edit_anchor() {
                    self.request_build(gallery_item, false, self.auto_compare, false);
                }
                self.popover_open = false;
            }
            Some(PopoverAction::Flip) => {
                self.flip_pin_cut_180(idx, gallery_item, Some(canvas));
            }
            Some(PopoverAction::AutoAlign) => {
                self.auto_align_pin_cut(idx, gallery_item, Some(canvas));
            }
            Some(PopoverAction::Delete) => {
                self.delete_pin(idx, gallery_item, Some(canvas));
            }
            None => {}
        }
    }
}
