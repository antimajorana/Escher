//! Overlay rendering for active Volterra branch-cut rays, glide rails, immobile fracton rings,
//! and live reaction arrows on `Canvas2D`.

use crate::canvas2d::{Canvas2D, CanvasMarker, draw_dashed_egui};
use crate::scene2d::{CirclePrim, LinePrim, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;

/// Pick radius in screen points for a branch-cut drag handle.
pub(super) const CUT_HANDLE_PICK_RADIUS_PX: f32 = 12.0;

/// World-space Volterra branch-cut ray (from defect core out to patch rim) and optional strip segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasCutRay {
    pub origin: [f64; 2],
    pub end: [f64; 2],
    pub strip: Option<([f64; 2], [f64; 2])>,
    pub is_insertion: bool,
    pub is_half_wall: bool,
    pub spec_index: Option<usize>,
}

/// Screen position of the draggable handle along a ray associated with a specific pin.
pub fn cut_ray_handle_screen_pos(
    canvas: &Canvas2D,
    rect: egui::Rect,
    ray: &CanvasCutRay,
) -> Option<egui::Pos2> {
    ray.spec_index?;
    let p0 = canvas.world_to_screen_at(rect, [ray.origin[0] as f32, ray.origin[1] as f32]);
    let p1 = canvas.world_to_screen_at(rect, [ray.end[0] as f32, ray.end[1] as f32]);
    let dv = p1 - p0;
    let len = dv.length();
    if len < 26.0 {
        return None;
    }
    let d = (0.58 * len).clamp(24.0, 96.0).min(len - 4.0);
    Some(p0 + (dv / len) * d)
}

/// Nearest cut-ray handle within `radius_px`, returning `(spec_index, ray.origin)`.
pub(crate) fn pick_cut_handle(
    rays: &[CanvasCutRay],
    canvas: &Canvas2D,
    rect: egui::Rect,
    pt: egui::Pos2,
    radius_px: f32,
) -> Option<(usize, [f64; 2])> {
    let mut best: Option<(f32, usize, [f64; 2])> = None;
    for ray in rays {
        let Some(spec_idx) = ray.spec_index else {
            continue;
        };
        let Some(hp) = cut_ray_handle_screen_pos(canvas, rect, ray) else {
            continue;
        };
        let d = hp.distance(pt);
        if d <= radius_px && best.is_none_or(|(bd, _, _)| d <= bd) {
            best = Some((d, spec_idx, ray.origin));
        }
    }
    best.map(|(_, idx, origin)| (idx, origin))
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

fn ray_color(ray: &CanvasCutRay, tokens: &ThemeTokens) -> egui::Color32 {
    if ray.is_half_wall {
        tokens.cut_half_wall
    } else if ray.is_insertion {
        tokens.cut_insertion
    } else {
        tokens.cut_ray_line
    }
}

/// Vector layer for the visible cut geometry. The drag handles belong to the editor.
pub(super) fn cut_export_layer(rays: &[CanvasCutRay]) -> ScenePrimitives {
    let mut layer = ScenePrimitives::default();
    let tokens = ThemeTokens::paper();
    for ray in rays {
        let color = ray_color(ray, &tokens).to_srgba_unmultiplied();
        let line = |a: [f64; 2], b: [f64; 2], width, dashed, arrow_mid| LinePrim {
            a: a.map(|x| x as f32),
            b: b.map(|x| x as f32),
            color,
            width,
            dashed,
            arrow_mid,
        };
        let dot = |center: [f64; 2], radius| CirclePrim {
            center: center.map(|x| x as f32),
            radius_world: 0.0,
            min_px: radius,
            max_px: radius,
            fill: color,
            stroke: ([0; 4], 0.0),
        };
        let width = if ray.is_half_wall { 2.8 } else { 2.0 };
        layer.lines.push(line(ray.origin, ray.end, width, !ray.is_half_wall, true));
        if ray.is_half_wall {
            layer.circles.push(dot(ray.origin, 4.5));
        }
        if let Some((a, b)) = ray.strip {
            layer.lines.push(line(a, b, 3.0, false, false));
            layer.circles.extend([dot(a, 2.8), dot(b, 2.8)]);
        }
    }
    layer.bbox = compute_bbox(layer.lines.iter().flat_map(|line| [line.a, line.b]));
    layer
}

/// Render Volterra branch cuts with purple removal rays, teal insertion rays, and amber half-walls.
pub(crate) fn draw_cut_rays(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    rays: &[CanvasCutRay],
    tokens: &ThemeTokens,
) {
    for ray in rays {
        let col = ray_color(ray, tokens);
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

        if let Some(hp) = cut_ray_handle_screen_pos(canvas, rect, ray) {
            painter.circle_filled(hp, 5.5, tokens.cut_ray_handle);
            painter.circle_stroke(hp, 5.5, egui::Stroke::new(1.6, tokens.bg_canvas_2d));
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
