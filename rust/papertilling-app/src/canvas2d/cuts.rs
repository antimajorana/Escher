//! Overlay rendering for active Volterra branch-cut rays, glide rails, immobile fracton rings,
//! and live reaction arrows on `Canvas2D`.

use crate::canvas2d::{Canvas2D, CanvasMarker, draw_dashed_egui};
use crate::theme::ThemeTokens;

/// World-space Volterra branch-cut ray (from defect core out to patch rim) and optional strip segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasCutRay {
    pub origin: [f64; 2],
    pub end: [f64; 2],
    pub strip: Option<([f64; 2], [f64; 2])>,
    pub is_insertion: bool,
    pub is_half_wall: bool,
}

/// Live preview arrow showing a disclination step + emitted Burgers vector or dislocation absorption.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasReactionArrow {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub label: String,
    pub valid: bool,
}

/// Kinematic guides drawn over the crystal patch (`Combine` tab).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CanvasKinematicOverlay {
    pub glide_lines: Vec<([f64; 2], [f64; 2], bool)>,
    pub locked_pins: Vec<usize>,
    pub reaction: Option<CanvasReactionArrow>,
}

/// Render Volterra branch-cut rays (`purple` for removal/disclination, `teal` for dislocation insertion).
pub(crate) fn draw_cut_rays(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    rays: &[CanvasCutRay],
    tokens: &ThemeTokens,
) {
    for ray in rays {
        let col = if ray.is_half_wall {
            tokens.cut_half_wall
        } else if ray.is_insertion {
            tokens.cut_insertion
        } else {
            tokens.cut_ray_line
        };
        let p0 = canvas.world_to_screen_at(rect, [ray.origin[0] as f32, ray.origin[1] as f32]);
        let p1 = canvas.world_to_screen_at(rect, [ray.end[0] as f32, ray.end[1] as f32]);
        if ray.is_half_wall {
            painter.line_segment([p0, p1], egui::Stroke::new(2.8, col));
            painter.circle_filled(p0, 4.5, col);
        } else {
            draw_dashed_egui(painter, p0, p1, egui::Stroke::new(2.0, col), 7.0, 4.5);
        }

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

/// Render 1D glide rails, immobile fracton lock rings, and live step/absorption reaction previews.
pub(crate) fn draw_kinematic_overlay(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    markers: &[CanvasMarker],
    ov: &CanvasKinematicOverlay,
    tokens: &ThemeTokens,
) {
    for &(a, b, active) in &ov.glide_lines {
        let p0 = canvas.world_to_screen_at(rect, [a[0] as f32, a[1] as f32]);
        let p1 = canvas.world_to_screen_at(rect, [b[0] as f32, b[1] as f32]);
        let alpha = if active { tokens.glide_rail.a() } else { 115 };
        let width = if active { 2.4 } else { 1.3 };
        let col = egui::Color32::from_rgba_unmultiplied(
            tokens.glide_rail.r(),
            tokens.glide_rail.g(),
            tokens.glide_rail.b(),
            alpha,
        );
        draw_dashed_egui(painter, p0, p1, egui::Stroke::new(width, col), 8.0, 5.0);
    }

    if !ov.locked_pins.is_empty() {
        let ring_col = tokens.locked_pin_ring;
        for m in markers {
            if ov.locked_pins.contains(&m.index) {
                let sp = canvas.world_to_screen_at(rect, [m.pos[0] as f32, m.pos[1] as f32]);
                painter.circle_stroke(sp, 12.5, egui::Stroke::new(1.6, ring_col));
            }
        }
    }

    if let Some(rx) = &ov.reaction {
        let col = if rx.valid { tokens.status_ok } else { tokens.status_err };
        let p0 = canvas.world_to_screen_at(rect, [rx.from[0] as f32, rx.from[1] as f32]);
        let p1 = canvas.world_to_screen_at(rect, [rx.to[0] as f32, rx.to[1] as f32]);
        let st = egui::Stroke::new(2.6, col);
        painter.line_segment([p0, p1], st);
        painter.circle_stroke(p1, 10.5, st);
        let d = p1 - p0;
        let len = d.length();
        if len >= 14.0 {
            let u = d / len;
            let n = egui::vec2(-u.y, u.x);
            painter.line_segment([p1, p1 - u * 9.0 + n * 5.0], st);
            painter.line_segment([p1, p1 - u * 9.0 - n * 5.0], st);
        }
        if !rx.label.is_empty() {
            let mid = p0 + d * 0.5 + egui::vec2(0.0, -14.0);
            let font = egui::FontId::proportional(11.0);
            let galley = painter.layout_no_wrap(rx.label.clone(), font.clone(), col);
            let box_rect = egui::Rect::from_center_size(mid, galley.size() + egui::vec2(10.0, 5.0));
            painter.rect_filled(box_rect, 4.0, tokens.bg_hud);
            painter.rect_stroke(box_rect, 4.0, egui::Stroke::new(1.0, col), egui::StrokeKind::Inside);
            painter.text(mid, egui::Align2::CENTER_CENTER, &rx.label, font, col);
        }
    }
}
