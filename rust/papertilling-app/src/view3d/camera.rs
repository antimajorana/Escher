//! Camera state of the 3D viewport: a turntable orbit (yaw, pitch) about world y or world z, dolly distance and a
//! camera-plane pan offset.
//!
//! The view rotation is `R = orbit_rotation(yaw, pitch) * U` with `orbit_rotation = Rx(pitch) * Ry(yaw)` (mesh3d) and
//! `U` the turntable frame: `U = I` (world y is the turntable axis: default, Iso, Top) or `U = Rx(-pi/2)`, which maps
//! world z to the frame's y axis (world z is the turntable axis: Front, Side). Yaw turns about the turntable axis and
//! pitch tilts about the screen horizontal, so at pitch 0 the turntable axis is exactly screen-vertical. With `U = I`
//! the screen image of world y is always vertical (`R e_y = (0, cos p, sin p)`), so the z-up views Front (XZ) and
//! Side (YZ) need the z frame; there `R e_z = (0, cos p, sin p)` and both sit at pitch 0, away from the pitch poles,
//! so horizontal drags spin freely about world z and vertical drags tilt both ways up to the drag limit.

use crate::canvas2d::{ZOOM_STEP_FACTOR, ZoomKeyAction, input_zoom_factor, take_keyboard_zoom_action};
use crate::mesh3d::{build_mvp, orbit_rotation};

use super::View3dState;

pub const DIST_MIN: f32 = 0.45;
pub const DIST_MAX: f32 = 24.0;
pub const DIST_DEFAULT: f32 = 3.1;
/// Bound on `|camera_offset|` (pan and cursor-anchored dolly), far outside any mesh (radius ~1).
pub const OFFSET_MAX: f32 = 4.0 * DIST_MAX;
/// Soft pitch limit of orbit drags; presets may sit exactly on a pole (+-pi/2).
pub const PITCH_DRAG_LIMIT: f32 = 1.50;

const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// Axis-aligned camera presets. Each sets the turntable frame, yaw and pitch, clears the pan offset and keeps the
/// distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraPreset {
    /// The default orientation (yaw 0.55, pitch 0.35).
    Iso,
    /// Looks down -z (camera on +z): x right, y up.
    Top,
    /// Looks along +y (camera on -y): x right, z up. z-up turntable frame.
    Front,
    /// Looks along -x (camera on +x): y right, z up. z-up turntable frame.
    Side,
}

impl CameraPreset {
    /// `(yaw, pitch, z_up)`: angles in radians and whether the turntable axis is world z.
    ///
    /// Derivation (column i of `R` is the view-space image of world axis i; screen x = view x, screen up = view y,
    /// view +z points at the camera; `Ry(t)` sends x -> (cos t, 0, -sin t), z -> (sin t, 0, cos t)):
    /// Top: `R = I`. Front: `R = U = Rx(-pi/2)` sends x -> view x, y -> view -z, z -> view y.
    /// Side: `R = Ry(-pi/2) U` sends x -> Ry(-pi/2) e_x = view +z, y -> Ry(-pi/2)(-e_z) = view +x, z -> view y.
    pub fn orientation(self) -> (f32, f32, bool) {
        match self {
            CameraPreset::Iso => (0.55, 0.35, false),
            CameraPreset::Top => (0.0, 0.0, false),
            CameraPreset::Front => (0.0, 0.0, true),
            CameraPreset::Side => (-HALF_PI, 0.0, true),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CameraPreset::Iso => "Iso",
            CameraPreset::Top => "Top (XY)",
            CameraPreset::Front => "Front (XZ)",
            CameraPreset::Side => "Side (YZ)",
        }
    }
}

/// Scale `offset` back onto the disc `|offset| <= OFFSET_MAX`.
fn clamp_offset(offset: egui::Vec2) -> egui::Vec2 {
    let len = f64::from(offset.x).hypot(f64::from(offset.y));
    if len > f64::from(OFFSET_MAX) {
        let s = f64::from(OFFSET_MAX) / len;
        egui::vec2((f64::from(offset.x) * s) as f32, (f64::from(offset.y) * s) as f32)
    } else {
        offset
    }
}

impl View3dState {
    pub fn reset(&mut self) {
        (self.yaw, self.pitch, self.distance) = (0.55, 0.35, DIST_DEFAULT);
        self.z_up = false;
        self.camera_offset = egui::Vec2::ZERO;
    }

    /// Set the camera from stored values (session restore). `None` or a non-finite value keeps the current one; a
    /// yaw outside (-pi, pi] is wrapped into it, the pitch is clamped to [-pi/2, pi/2], the distance to
    /// [`DIST_MIN`, `DIST_MAX`] and the pan offset to length [`OFFSET_MAX`].
    pub(crate) fn restore_camera(&mut self, yaw: Option<f32>, pitch: Option<f32>, distance: Option<f32>, offset: Option<[f32; 2]>, z_up: Option<bool>) {
        use core::f32::consts::{PI, TAU};
        if let Some(y) = yaw.filter(|v| v.is_finite()) {
            self.yaw = if y > -PI && y <= PI { y } else {
                let w = y.rem_euclid(TAU);
                if w > PI { w - TAU } else { w }
            };
        }
        if let Some(p) = pitch.filter(|v| v.is_finite()) {
            self.pitch = p.clamp(-HALF_PI, HALF_PI);
        }
        if let Some(d) = distance.filter(|v| v.is_finite()) {
            self.distance = d.clamp(DIST_MIN, DIST_MAX);
        }
        if let Some([x, y]) = offset.filter(|o| o.iter().all(|v| v.is_finite())) {
            self.camera_offset = clamp_offset(egui::vec2(x, y));
        }
        if let Some(z) = z_up {
            self.z_up = z;
        }
    }

    /// Apply an axis preset: frame and angles from [`CameraPreset::orientation`], pan cleared, distance kept.
    pub fn apply_preset(&mut self, preset: CameraPreset) {
        (self.yaw, self.pitch, self.z_up) = preset.orientation();
        self.camera_offset = egui::Vec2::ZERO;
    }

    /// View rotation `orbit_rotation(yaw, pitch) * U` (column-major; module docs), used for the MVP, lighting,
    /// depth sorting and the axis gizmo. For `U = Rx(-pi/2)` (e_x -> e_x, e_y -> -e_z, e_z -> e_y) the columns of
    /// `O U` are `(O e_x, -O e_z, O e_y)`.
    pub fn rotation(&self) -> [[f32; 4]; 4] {
        let o = orbit_rotation(self.yaw, self.pitch);
        if !self.z_up {
            return o;
        }
        let neg = |c: [f32; 4]| [-c[0], -c[1], -c[2], c[3]];
        [o[0], neg(o[2]), o[1], o[3]]
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
            let offset = self.camera_offset + egui::vec2(ndc.x / projection[0][0], ndc.y / projection[1][1]) * delta;
            self.camera_offset = clamp_offset(offset);
        }
    }

    /// Translate the camera so points at the orbit-centre depth follow a screen drag of `delta_px` pixels.
    ///
    /// Derivation from `build_mvp`: a point at view position `(x, y, -distance)` (the orbit centre plus the pan
    /// offset) has clip `w = distance` and NDC `x_ndc = P00 x / distance`, `y_ndc = P11 y / distance`; screen pixels
    /// are `px = w_px / 2 * x_ndc` and `py = -h_px / 2 * y_ndc` (screen y points down). Hence one pixel is
    /// `2 distance / (w_px P00)` view units horizontally and `2 distance / (h_px P11)` vertically. Both equal
    /// `2 distance / (h_px P11)` unless the renderer's aspect clamp (>= 0.1) is active, so they are kept per axis.
    /// Points nearer or farther than the orbit centre move by `distance / depth` times as much (parallax).
    pub fn pan_by_pixels(&mut self, rect: egui::Rect, delta_px: egui::Vec2) {
        if !(delta_px.x.is_finite() && delta_px.y.is_finite()) || rect.width() <= 0.0 || rect.height() <= 0.0 {
            return;
        }
        let aspect = (rect.width() / rect.height()).max(0.1);
        let proj = build_mvp(aspect, self.distance, &orbit_rotation(0.0, 0.0));
        let per_px = egui::vec2(
            2.0 * self.distance / (rect.width() * proj[0][0]),
            2.0 * self.distance / (rect.height() * proj[1][1]),
        );
        let offset = self.camera_offset + egui::vec2(delta_px.x * per_px.x, -delta_px.y * per_px.y);
        if offset.x.is_finite() && offset.y.is_finite() {
            self.camera_offset = clamp_offset(offset);
        }
    }

    pub(super) fn view_mvp(&self, aspect: f32, rotation: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut translated = *rotation;
        translated[3][0] = self.camera_offset.x;
        translated[3][1] = self.camera_offset.y;
        build_mvp(aspect.max(0.1), self.distance, &translated)
    }

    /// Orbit by `(dyaw, dpitch)` radians: yaw wraps to (-pi, pi], pitch stops at +-[`PITCH_DRAG_LIMIT`].
    ///
    /// A preset may leave the pitch beyond the soft limit (exactly on a pole); orbiting from there moves only
    /// towards the interior and never snaps to the limit, and the pitch never leaves [-pi/2, pi/2].
    pub fn orbit_by(&mut self, dyaw: f32, dpitch: f32) {
        if !(dyaw.is_finite() && dpitch.is_finite()) {
            return;
        }
        let yaw = (self.yaw + dyaw).rem_euclid(core::f32::consts::TAU);
        self.yaw = if yaw > core::f32::consts::PI { yaw - core::f32::consts::TAU } else { yaw };
        let lo = (-PITCH_DRAG_LIMIT).min(self.pitch).max(-HALF_PI);
        let hi = PITCH_DRAG_LIMIT.max(self.pitch).min(HALF_PI);
        self.pitch = (self.pitch + dpitch).clamp(lo, hi);
    }

    /// Orbit from a screen drag of `delta_px` pixels (screen y down), 0.01 rad per pixel: horizontal = yaw about
    /// the turntable axis (world y or z), vertical = pitch about the screen horizontal.
    pub fn orbit_by_screen_drag(&mut self, delta_px: egui::Vec2) {
        self.orbit_by(delta_px.x * 0.01, delta_px.y * 0.01);
    }

    /// Mouse/keyboard camera input of the viewport. Drags that start inside any of `blockers` (zoom HUD, axis
    /// gizmo, preset buttons) neither orbit nor pan. Primary drag orbits; Shift+primary, secondary and middle drags
    /// pan. Pan or orbit is decided when the drag starts and kept until release, whatever Shift does meanwhile.
    pub(super) fn handle_camera_input(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        blockers: &[egui::Rect],
    ) {
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let in_blocker = |p: egui::Pos2| blockers.iter().any(|b| b.contains(p));
        let over_overlay = ptr.is_some_and(in_blocker);
        let drag_from_overlay = press_orig.is_some_and(in_blocker);

        if !response.dragged() {
            self.drag_pans = None;
        }
        if !drag_from_overlay && response.dragged() {
            let shift = ui.input(|i| i.modifiers.shift);
            let starts_pan = response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle)
                || (shift && response.dragged_by(egui::PointerButton::Primary));
            if *self.drag_pans.get_or_insert(starts_pan) {
                // egui reports a drag only once the pointer has left the click radius; on that first frame use the
                // whole motion since the press so the grabbed point stays under the cursor.
                let delta = match (response.drag_started(), press_orig, ptr) {
                    (true, Some(o), Some(p)) => p - o,
                    _ => response.drag_delta(),
                };
                self.pan_by_pixels(rect, delta);
            } else if response.dragged_by(egui::PointerButton::Primary) {
                self.orbit_by_screen_drag(response.drag_delta());
            }
        }
        if !over_overlay && !drag_from_overlay && response.double_clicked() {
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
