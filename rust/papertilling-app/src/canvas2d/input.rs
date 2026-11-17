use super::circuit;
use super::hit::{
    PIN_PICK_RADIUS_PX, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_FACTOR, ZoomKeyAction, input_zoom_factor,
    pick_marker, take_keyboard_zoom_action,
};
use super::{Canvas2D, CanvasInteraction, CanvasMarker};

pub(crate) fn zoom_hud_rect(rect: egui::Rect) -> Option<egui::Rect> {
    if rect.width() >= 190.0 && rect.height() >= 80.0 {
        Some(egui::Rect::from_min_max(
            rect.right_bottom() - egui::vec2(164.0, 30.0),
            rect.right_bottom() - egui::vec2(6.0, 6.0),
        ))
    } else {
        None
    }
}

impl Canvas2D {
    pub(super) fn handle_canvas_input(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        interactive: bool,
    ) -> (egui::Rect, bool, egui::Rect, CanvasInteraction) {
        let mut out = CanvasInteraction {
            shift_held: ui.input(|i| i.modifiers.shift),
            ..Default::default()
        };
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

        if self.auto_fit_pending && rect.width() > 32.0 && rect.height() > 32.0 {
            let (bmin, bmax) = self.compiled.bbox;
            self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            let span_x = (bmax[0] - bmin[0]).max(0.5);
            let span_y = (bmax[1] - bmin[1]).max(0.5);
            let pad_h = if self.compiled.note.is_empty() { 40.0 } else { 72.0 };
            let zx = (rect.width() - 40.0).max(24.0) / span_x;
            let zy = (rect.height() - pad_h).max(24.0) / span_y;
            let fit = zx.min(zy).clamp(ZOOM_MIN, ZOOM_MAX);
            self.fit_zoom = fit;
            self.zoom = fit;
            self.pan = egui::Vec2::ZERO;
            self.auto_fit_pending = false;
        }

        let hud_rect_opt = zoom_hud_rect(rect);
        let has_hud = hud_rect_opt.is_some();
        let hud_rect = hud_rect_opt.unwrap_or(egui::Rect::NOTHING);
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let over_hud = has_hud && ptr.is_some_and(|p| hud_rect.contains(p));
        let drag_from_hud = has_hud && press_orig.is_some_and(|p| hud_rect.contains(p));

        let hit_pin_at = |this: &Self, pt: egui::Pos2| {
            pick_marker(markers, &this.view_at(rect), pt, PIN_PICK_RADIUS_PX)
        };

        // Smooth wheel + pinch / Ctrl+wheel zoom anchored at the cursor position inside `rect`.
        let anchor = response.hover_pos().or(ptr).filter(|p| rect.contains(*p));
        if ui.rect_contains_pointer(rect) {
            let factor = ui.input(input_zoom_factor);
            if (factor - 1.0).abs() > 1e-5 {
                self.zoom_at(rect, factor, anchor);
            }
            if let Some(act) = take_keyboard_zoom_action(ui, &response) {
                match act {
                    ZoomKeyAction::In => self.zoom_at(rect, ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Out => self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Fit => self.reset(),
                }
            }
        }

        if interactive && !drag_from_hud && response.drag_started() {
            let origin = press_orig.or_else(|| response.interact_pointer_pos());
            self.active_drag_pin = origin.and_then(|pt| hit_pin_at(self, pt));
        }

        if !drag_from_hud && response.dragged() {
            if let Some(pin_idx) = self.active_drag_pin {
                if markers.iter().any(|m| m.index == pin_idx) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        out.dragged_marker = Some((pin_idx, self.screen_to_world_at(rect, pos)));
                    }
                } else {
                    self.active_drag_pin = None;
                }
            } else {
                self.pan += response.drag_delta();
            }
        }

        if interactive && response.drag_stopped() && self.active_drag_pin.take().is_some() {
            out.drag_released = true;
        }

        if !over_hud && !drag_from_hud && response.double_clicked() {
            self.reset();
        } else if interactive
            && !over_hud
            && !drag_from_hud
            && response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            match hit_pin_at(self, pos) {
                Some(idx) => out.clicked_marker = Some(idx),
                None => {
                    if let Some(circ) = &self.circuit_overlay
                        && let Some(v) = circuit::pick_circuit_vertex(self, rect, circ, pos, 11.0)
                    {
                        out.clicked_circuit_vertex = Some(v);
                    } else if click_to_place {
                        out.clicked_world = Some(self.screen_to_world_at(rect, pos));
                    }
                }
            }
        }

        if !over_hud
            && response.hovered()
            && interactive
            && let Some(hpos) = response.hover_pos()
        {
            out.hovered_world = Some(self.screen_to_world_at(rect, hpos));
            if self.active_drag_pin.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if let Some(pidx) = hit_pin_at(self, hpos) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                if let Some(m) = markers.iter().find(|m| m.index == pidx) {
                    let hint = if self.kinematic_overlay.locked_pins.contains(&m.index) {
                        "immobile fracton (ΔN=0) — Shift+drag to step & emit b"
                    } else {
                        "drag to move"
                    };
                    response.clone().on_hover_text(format!(
                        "#{} {} at ({:+.3}, {:+.3}) — {hint}",
                        m.index + 1,
                        m.label,
                        m.pos[0],
                        m.pos[1]
                    ));
                }
            } else if click_to_place {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            }
        }

        (rect, has_hud, hud_rect, out)
    }
}
