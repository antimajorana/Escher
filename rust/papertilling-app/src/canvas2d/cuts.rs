//! Overlay rendering for active Volterra branch-cut rays and dislocation strips on `Canvas2D`.

use crate::canvas2d::{Canvas2D, draw_dashed_egui};

/// World-space Volterra branch-cut ray (from defect core out to patch rim) and optional strip segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasCutRay {
    pub origin: [f64; 2],
    pub end: [f64; 2],
    pub strip: Option<([f64; 2], [f64; 2])>,
    pub is_insertion: bool,
}

/// Render Volterra branch-cut rays (`purple` for removal/disclination, `teal` for dislocation insertion).
pub(crate) fn draw_cut_rays(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    rays: &[CanvasCutRay],
) {
    for ray in rays {
        let col = if ray.is_insertion {
            egui::Color32::from_rgba_unmultiplied(0, 121, 107, 215)
        } else {
            egui::Color32::from_rgba_unmultiplied(123, 31, 162, 215)
        };
        let p0 = canvas.world_to_screen_at(rect, [ray.origin[0] as f32, ray.origin[1] as f32]);
        let p1 = canvas.world_to_screen_at(rect, [ray.end[0] as f32, ray.end[1] as f32]);
        draw_dashed_egui(painter, p0, p1, egui::Stroke::new(2.0, col), 7.0, 4.5);

        // Small directional barb halfway out along the branch-cut ray.
        let dv = p1 - p0;
        let len = dv.length();
        if len >= 28.0 {
            let u = dv / len;
            let n = egui::vec2(-u.y, u.x);
            let mid = p0 + u * (0.45 * len);
            let barb = 5.5;
            let st = egui::Stroke::new(1.8, col);
            painter.line_segment([mid + u * 3.0, mid - u * barb + n * (0.65 * barb)], st);
            painter.line_segment([mid + u * 3.0, mid - u * barb - n * (0.65 * barb)], st);
        }

        if let Some((sa, sb)) = ray.strip {
            let pa = canvas.world_to_screen_at(rect, [sa[0] as f32, sa[1] as f32]);
            let pb = canvas.world_to_screen_at(rect, [sb[0] as f32, sb[1] as f32]);
            painter.line_segment([pa, pb], egui::Stroke::new(3.0, col));
            painter.circle_filled(pa, 2.8, col);
            painter.circle_filled(pb, 2.8, col);
        }
    }
}
