//! Pure view transform, pin picking and scroll-zoom helpers of `Canvas2D` (unit-testable without a window).

use super::CanvasMarker;

/// Pin pick radius in screen points (the drawn halo radius), independent of the zoom level.
pub const PIN_PICK_RADIUS_PX: f32 = 14.0;
/// Zoom limits in screen points per world unit.
pub const ZOOM_MIN: f32 = 6.0;
pub const ZOOM_MAX: f32 = 420.0;
/// Zoom factor per point of vertical scroll.
pub const ZOOM_PER_SCROLL: f32 = 0.002;

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

/// New `(pan, zoom)` after a vertical scroll of `scroll` points. With an `anchor` (the pointer), the world point
/// under it stays under it; without one the zoom is about the canvas centre (`rect_center`).
pub fn zoom_about(pan: egui::Vec2, zoom: f32, scroll: f32, rect_center: egui::Pos2, anchor: Option<egui::Pos2>) -> (egui::Vec2, f32) {
    let new_zoom = (zoom * (scroll * ZOOM_PER_SCROLL).exp()).clamp(ZOOM_MIN, ZOOM_MAX);
    let Some(a) = anchor else {
        return (pan, new_zoom);
    };
    // a = c + pan + d * zoom (per axis, d the scaled world offset) must hold before and after.
    let rel = a - rect_center - pan;
    let k = new_zoom / zoom.max(1e-3);
    (a - rect_center - rel * k, new_zoom)
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
        // world +y is screen up (smaller y); one world unit is `zoom` points
        assert_eq!(v.world_to_screen([1.0, -1.0]), egui::pos2(200.0, 112.5));
        assert_eq!(v.world_to_screen([2.0, -2.0]), egui::pos2(237.5, 150.0));
    }

    #[test]
    fn pick_radius_is_in_screen_points_at_every_zoom() {
        // a pin 0.5 world units away from the pointer: 0.5 * zoom points on screen
        let ms = [marker(0, 1.5, -2.0)];
        let pt = egui::pos2(200.0, 150.0);
        assert_eq!(pick_marker(&ms, &view(20.0), pt, PIN_PICK_RADIUS_PX), Some(0)); // 10 pt
        assert_eq!(pick_marker(&ms, &view(28.0), pt, PIN_PICK_RADIUS_PX), Some(0)); // 14 pt: boundary included
        assert_eq!(pick_marker(&ms, &view(40.0), pt, PIN_PICK_RADIUS_PX), None); // 20 pt
        assert_eq!(pick_marker(&[], &view(40.0), pt, PIN_PICK_RADIUS_PX), None);
    }

    #[test]
    fn overlapping_pins_pick_nearest_then_topmost() {
        let v = view(10.0);
        // pins 0.6 apart (6 pt): both inside the radius of a pointer between them
        let ms = [marker(0, 1.0, -2.0), marker(1, 1.6, -2.0), marker(2, 1.0, -2.0)];
        // nearer to pin 1
        assert_eq!(pick_marker(&ms, &v, egui::pos2(204.0, 150.0), PIN_PICK_RADIUS_PX), Some(1));
        // pins 0 and 2 coincide: the one drawn last (2) wins
        assert_eq!(pick_marker(&ms, &v, egui::pos2(199.0, 150.0), PIN_PICK_RADIUS_PX), Some(2));
        // markers report their own index, not their slot
        let ms2 = [marker(7, 1.0, -2.0)];
        assert_eq!(pick_marker(&ms2, &v, egui::pos2(200.0, 150.0), PIN_PICK_RADIUS_PX), Some(7));
    }

    #[test]
    fn scroll_zoom_keeps_world_point_under_pointer() {
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
        let (p3, z3) = zoom_about(pan, zoom, 100.0, rect_c, None);
        assert_eq!(p3, pan);
        assert!((z3 - 50.0 * (0.2_f32).exp()).abs() < 1e-3);
    }
}
