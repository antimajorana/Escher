//! Pure view transform, pin picking, cursor-anchored zoom, and input helpers of `Canvas2D`
//! (unit-testable without a window).

use super::CanvasMarker;

/// Pin pick radius in screen points (the drawn halo radius), independent of the zoom level.
pub const PIN_PICK_RADIUS_PX: f32 = 14.0;
/// Zoom limits in screen points per world unit (widened for deep core inspection and large patches).
pub const ZOOM_MIN: f32 = 2.0;
pub const ZOOM_MAX: f32 = 4000.0;
/// Zoom factor per point of vertical scroll.
pub const ZOOM_PER_SCROLL: f32 = 0.002;
/// Multiplicative step for `+` / `-` HUD buttons and keyboard shortcuts.
pub const ZOOM_STEP_FACTOR: f32 = 1.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoomKeyAction {
    In,
    Out,
    Fit,
}

/// World (y up) <-> screen (y down) map: `screen = origin + (p - world_center) * zoom`, y flipped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    /// Screen position of `world_center` (canvas rect centre plus pan).
    pub origin: egui::Pos2,
    pub zoom: f32,
    pub world_center: [f32; 2],
}

impl ViewTransform {
    pub fn world_to_screen(&self, p: [f32; 2]) -> egui::Pos2 {
        egui::pos2(
            self.origin.x + (p[0] - self.world_center[0]) * self.zoom,
            self.origin.y - (p[1] - self.world_center[1]) * self.zoom,
        )
    }

    pub fn screen_to_world(&self, pos: egui::Pos2) -> [f64; 2] {
        let z = self.zoom.max(1e-3);
        [
            (self.world_center[0] + (pos.x - self.origin.x) / z) as f64,
            (self.world_center[1] - (pos.y - self.origin.y) / z) as f64,
        ]
    }
}

/// The marker (its `index`) whose drawn centre is nearest to `pt`, within `radius_px` screen points.
/// On equal distance the marker drawn last (on top) wins.
pub fn pick_marker(markers: &[CanvasMarker], view: &ViewTransform, pt: egui::Pos2, radius_px: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for m in markers {
        let d = view.world_to_screen([m.pos[0] as f32, m.pos[1] as f32]).distance(pt);
        if d <= radius_px && best.is_none_or(|(_, bd)| d <= bd) {
            best = Some((m.index, d));
        }
    }
    best.map(|(idx, _)| idx)
}

/// Multiplicative zoom factor read from both trackpad pinch / `Ctrl+wheel` (`InputState::zoom_delta`)
/// and the processed vertical mouse wheel delta. Do not reread raw wheel events: egui has already
/// smoothed, unit-scaled, modifier-remapped, or consumed them (e.g. Shift-wheel is horizontal).
pub fn input_zoom_factor(i: &egui::InputState) -> f32 {
    let factor = i.zoom_delta() * (i.smooth_scroll_delta.y * ZOOM_PER_SCROLL).exp();
    if factor.is_finite() && factor > 0.0 { factor } else { 1.0 }
}

fn scene_modifiers(m: egui::Modifiers) -> bool {
    // Shift is needed for `+` on many keyboard layouts. Reserve Ctrl/Cmd shortcuts for egui's GUI
    // scale controls, and don't intercept Alt/AltGr text entry or application shortcuts.
    !m.alt && !m.ctrl && !m.mac_cmd && !m.command
}

fn zoom_key(key: egui::Key) -> Option<ZoomKeyAction> {
    match key {
        egui::Key::Plus | egui::Key::Equals => Some(ZoomKeyAction::In),
        egui::Key::Minus => Some(ZoomKeyAction::Out),
        egui::Key::Num0 => Some(ZoomKeyAction::Fit),
        _ => None,
    }
}

fn zoom_text(text: &str) -> Option<ZoomKeyAction> {
    match text {
        "+" | "=" => Some(ZoomKeyAction::In),
        "-" => Some(ZoomKeyAction::Out),
        "0" => Some(ZoomKeyAction::Fit),
        _ => None,
    }
}

/// Unmodified keyboard zoom (`=` / `+`, `-`, `0`); callers must establish ownership and consume it.
pub fn keyboard_zoom_action(i: &egui::InputState) -> Option<ZoomKeyAction> {
    if !scene_modifiers(i.modifiers) {
        return None;
    }
    let mut saw_zoom_key = false;
    for ev in &i.events {
        if let egui::Event::Key { key, pressed: true, modifiers, .. } = ev
            && let Some(action) = zoom_key(*key)
        {
            saw_zoom_key = true;
            if scene_modifiers(*modifiers) {
                return Some(action);
            }
        }
    }
    // A modified key must not sneak back in via an accompanying Text event.
    if saw_zoom_key { return None; }
    i.events.iter().find_map(|ev| match ev {
        egui::Event::Text(text) => zoom_text(text),
        _ => None,
    })
}

/// Pointer-targeted shortcuts: ignore covered/outside viewports and text edits, and deliver an action
/// once even if another viewport is drawn later in this pass. Keep Cmd/Ctrl events for global GUI zoom.
pub(crate) fn take_keyboard_zoom_action(ui: &egui::Ui, response: &egui::Response) -> Option<ZoomKeyAction> {
    if !ui.rect_contains_pointer(response.interact_rect) || ui.ctx().text_edit_focused() {
        return None;
    }
    ui.input_mut(|i| {
        let action = super::keyboard_zoom_action(i)?;
        i.events.retain(|ev| match ev {
            egui::Event::Key { key, pressed: true, modifiers, .. } =>
                !(scene_modifiers(*modifiers) && zoom_key(*key).is_some()),
            egui::Event::Text(text) => zoom_text(text).is_none(),
            _ => true,
        });
        Some(action)
    })
}

/// New `(pan, zoom)` after multiplying zoom by `factor`, clamped to `[ZOOM_MIN, ZOOM_MAX]`.
/// With `anchor = Some(cursor)`, the world coordinate at `cursor` stays stationary.
pub fn zoom_by_factor_about(
    pan: egui::Vec2,
    zoom: f32,
    factor: f32,
    rect_center: egui::Pos2,
    anchor: Option<egui::Pos2>,
) -> (egui::Vec2, f32) {
    let f = if factor.is_finite() && factor > 0.0 { factor } else { 1.0 };
    let new_zoom = (zoom * f).clamp(ZOOM_MIN, ZOOM_MAX);
    let Some(a) = anchor else {
        return (pan, new_zoom);
    };
    let rel = a - rect_center - pan;
    let k = new_zoom / zoom.max(1e-3);
    (a - rect_center - rel * k, new_zoom)
}

/// New `(pan, zoom)` after a vertical scroll of `scroll` points.
#[cfg(test)]
pub fn zoom_about(
    pan: egui::Vec2,
    zoom: f32,
    scroll: f32,
    rect_center: egui::Pos2,
    anchor: Option<egui::Pos2>,
) -> (egui::Vec2, f32) {
    zoom_by_factor_about(pan, zoom, (scroll * ZOOM_PER_SCROLL).exp(), rect_center, anchor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas2d::MarkerKind;

    fn marker(index: usize, x: f64, y: f64) -> CanvasMarker {
        CanvasMarker { index, pos: [x, y], kind: MarkerKind::DisclinationPlus, label: String::new() }
    }

    fn view(zoom: f32) -> ViewTransform {
        ViewTransform { origin: egui::pos2(200.0, 150.0), zoom, world_center: [1.0, -2.0] }
    }

    #[test]
    fn transform_round_trip_and_y_flip() {
        let v = view(37.5);
        for p in [[1.0_f32, -2.0], [3.25, 0.5], [-4.0, 7.0]] {
            let s = v.world_to_screen(p);
            let w = v.screen_to_world(s);
            assert!((w[0] - p[0] as f64).abs() < 1e-4 && (w[1] - p[1] as f64).abs() < 1e-4);
        }
        assert_eq!(v.world_to_screen([1.0, -1.0]), egui::pos2(200.0, 112.5));
        assert_eq!(v.world_to_screen([2.0, -2.0]), egui::pos2(237.5, 150.0));
    }

    #[test]
    fn pick_radius_is_in_screen_points_at_every_zoom() {
        let ms = [marker(0, 1.5, -2.0)];
        let pt = egui::pos2(200.0, 150.0);
        assert_eq!(pick_marker(&ms, &view(20.0), pt, PIN_PICK_RADIUS_PX), Some(0));
        assert_eq!(pick_marker(&ms, &view(28.0), pt, PIN_PICK_RADIUS_PX), Some(0));
        assert_eq!(pick_marker(&ms, &view(40.0), pt, PIN_PICK_RADIUS_PX), None);
        assert_eq!(pick_marker(&[], &view(40.0), pt, PIN_PICK_RADIUS_PX), None);
    }

    #[test]
    fn overlapping_pins_pick_nearest_then_topmost() {
        let v = view(10.0);
        let ms = [marker(0, 1.0, -2.0), marker(1, 1.6, -2.0), marker(2, 1.0, -2.0)];
        assert_eq!(pick_marker(&ms, &v, egui::pos2(204.0, 150.0), PIN_PICK_RADIUS_PX), Some(1));
        assert_eq!(pick_marker(&ms, &v, egui::pos2(199.0, 150.0), PIN_PICK_RADIUS_PX), Some(2));
        let ms2 = [marker(7, 1.0, -2.0)];
        assert_eq!(pick_marker(&ms2, &v, egui::pos2(200.0, 150.0), PIN_PICK_RADIUS_PX), Some(7));
    }

    #[test]
    fn scroll_and_factor_zoom_keep_world_point_under_pointer_across_wide_range() {
        let rect_c = egui::pos2(300.0, 200.0);
        let (pan, zoom) = (egui::vec2(13.0, -7.0), 50.0);
        let anchor = egui::pos2(420.0, 90.0);
        let before = ViewTransform { origin: rect_c + pan, zoom, world_center: [0.5, 0.5] }.screen_to_world(anchor);
        for scroll in [120.0_f32, -80.0, 3000.0, -3000.0] {
            let (p2, z2) = zoom_about(pan, zoom, scroll, rect_c, Some(anchor));
            assert!((ZOOM_MIN..=ZOOM_MAX).contains(&z2));
            let after = ViewTransform { origin: rect_c + p2, zoom: z2, world_center: [0.5, 0.5] }.screen_to_world(anchor);
            assert!((after[0] - before[0]).abs() < 1e-4 && (after[1] - before[1]).abs() < 1e-4, "scroll {scroll}");
        }
        for factor in [0.1_f32, 0.4, 1.25, 8.0, 60.0] {
            let (p2, z2) = zoom_by_factor_about(pan, zoom, factor, rect_c, Some(anchor));
            assert!((ZOOM_MIN..=ZOOM_MAX).contains(&z2));
            let after = ViewTransform { origin: rect_c + p2, zoom: z2, world_center: [0.5, 0.5] }.screen_to_world(anchor);
            assert!((after[0] - before[0]).abs() < 1e-4 && (after[1] - before[1]).abs() < 1e-4, "factor {factor}");
        }
        let (p3, z3) = zoom_about(pan, zoom, 100.0, rect_c, None);
        assert_eq!(p3, pan);
        assert!((z3 - 50.0 * (0.2_f32).exp()).abs() < 1e-3);
    }
}

#[cfg(test)]
mod input_tests;
