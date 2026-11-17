use crate::canvas2d::{ZOOM_STEP_FACTOR, ZoomKeyAction, input_zoom_factor, take_keyboard_zoom_action};
use crate::mesh3d::{build_mvp, orbit_rotation};

use super::View3dState;

pub const DIST_MIN: f32 = 0.45;
pub const DIST_MAX: f32 = 24.0;
pub const DIST_DEFAULT: f32 = 3.1;

impl View3dState {
    pub fn reset(&mut self) {
        (self.yaw, self.pitch, self.distance) = (0.55, 0.35, DIST_DEFAULT);
        self.camera_offset = egui::Vec2::ZERO;
    }

    pub fn zoom_by(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance / factor).clamp(DIST_MIN, DIST_MAX);
        }
    }

    /// Dolly along the cursor ray, preserving every visible point on that ray (at any depth).
    /// No picking, geometry changes, or camera FOV convention changes are needed.
    pub fn zoom_at(&mut self, rect: egui::Rect, factor: f32, anchor: Option<egui::Pos2>) {
        let old_distance = self.distance;
        self.zoom_by(factor);
        if let Some(anchor) = anchor.filter(|p| rect.contains(*p))
            && rect.width() > 0.0
            && rect.height() > 0.0
        {
            let aspect = (rect.width() / rect.height()).max(0.1);
            let projection = build_mvp(aspect, old_distance, &orbit_rotation(0.0, 0.0));
            let ndc = egui::vec2(
                2.0 * (anchor.x - rect.center().x) / rect.width(),
                2.0 * (rect.center().y - anchor.y) / rect.height(),
            );
            let delta = self.distance - old_distance; // Use the actual clamped dolly distance.
            self.camera_offset += egui::vec2(ndc.x / projection[0][0], ndc.y / projection[1][1]) * delta;
        }
    }

    pub(super) fn view_mvp(&self, aspect: f32, rotation: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut translated = *rotation;
        translated[3][0] = self.camera_offset.x;
        translated[3][1] = self.camera_offset.y;
        build_mvp(aspect.max(0.1), self.distance, &translated)
    }

    /// Orbit by `(dyaw, dpitch)` radians: yaw wraps to (-pi, pi], pitch stops short of the poles.
    pub fn orbit_by(&mut self, dyaw: f32, dpitch: f32) {
        let yaw = (self.yaw + dyaw).rem_euclid(core::f32::consts::TAU);
        self.yaw = if yaw > core::f32::consts::PI { yaw - core::f32::consts::TAU } else { yaw };
        self.pitch = (self.pitch + dpitch).clamp(-1.50, 1.50);
    }

    pub(super) fn handle_camera_input(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        has_hud: bool,
        hud_rect: egui::Rect,
    ) {
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let over_hud = has_hud && ptr.is_some_and(|p| hud_rect.contains(p));
        let drag_from_hud = has_hud && press_orig.is_some_and(|p| hud_rect.contains(p));

        if !drag_from_hud && response.dragged() {
            let delta = response.drag_delta();
            self.orbit_by(delta.x * 0.01, delta.y * 0.01);
        }
        if !over_hud && !drag_from_hud && response.double_clicked() {
            self.reset();
        }
        if ui.rect_contains_pointer(rect) {
            let anchor = ptr.filter(|p| rect.contains(*p));
            let factor = ui.input(input_zoom_factor);
            if (factor - 1.0).abs() > 1e-5 {
                self.zoom_at(rect, factor, anchor);
            }
            if let Some(act) = take_keyboard_zoom_action(ui, response) {
                match act {
                    ZoomKeyAction::In => self.zoom_at(rect, ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Out => self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Fit => self.reset(),
                }
            }
        }
    }
}
