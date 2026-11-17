//! Interactive Burgers circuit parallel-transport overlay & Reference-Chart Development Inset.

use papertilling_core::defect::circuit::CircuitResult;

use crate::canvas2d::{Canvas2D, draw_dashed_egui};
use crate::scene2d::{CirclePrim, LinePrim, ScenePrimitives, TextPrim};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CircuitTargetMode {
    #[default]
    AllDefects,
    SelectedPin,
}

#[derive(Clone, Debug)]
pub struct CanvasCircuitOverlay {
    pub path: Vec<usize>,
    pub world_pts: Vec<[f64; 2]>,
    pub across_steps: Vec<bool>,
    pub result: CircuitResult,
    pub step_k: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CircuitUiState {
    pub enabled: bool,
    pub target_mode: CircuitTargetMode,
    pub pin_idx: usize,
    pub base_v0: Option<usize>,
    pub step_k: usize,
    pub overlay: Option<CanvasCircuitOverlay>,
    pub error: Option<String>,
}

/// Find the closest vertex index in `ov.path` within `max_px` screen distance of `screen_pos`.
pub fn pick_circuit_vertex(
    canvas: &Canvas2D,
    rect: egui::Rect,
    ov: &CanvasCircuitOverlay,
    screen_pos: egui::Pos2,
    max_px: f32,
) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    let m = ov.world_pts.len().min(ov.path.len());
    for i in 0..m {
        let wp = ov.world_pts[i];
        let sp = canvas.world_to_screen_at(rect, [wp[0] as f32, wp[1] as f32]);
        let d = sp.distance(screen_pos);
        if d <= max_px && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, ov.path[i]));
        }
    }
    best.map(|(_, v)| v)
}

/// Render circuit loop on `Canvas2D`, active step parallel-transport triad `(e1, e2)`,
/// and bottom-right Reference-Chart Development Inset showing open ref chain + closure vector `b`.
pub(crate) fn draw_circuit_overlay(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    ov: &CanvasCircuitOverlay,
) {
    let m = ov.world_pts.len();
    if m < 2 {
        return;
    }
    let to_sc = |p: [f64; 2]| canvas.world_to_screen_at(rect, [p[0] as f32, p[1] as f32]);
    let k_max = ov.result.ref_path.len().saturating_sub(1);
    let step_k = ov.step_k.min(k_max);

    for i in 0..m - 1 {
        let p0 = to_sc(ov.world_pts[i]);
        let p1 = to_sc(ov.world_pts[i + 1]);
        let active = i < step_k;
        let across = ov.across_steps.get(i).copied().unwrap_or(false);
        let col = if active {
            if across { egui::Color32::from_rgb(216, 67, 21) } else { egui::Color32::from_rgb(21, 101, 192) }
        } else {
            egui::Color32::from_rgba_unmultiplied(21, 101, 192, 135)
        };
        let width = if active { 3.2 } else { 2.0 };
        if across {
            draw_dashed_egui(painter, p0, p1, egui::Stroke::new(width, col), 5.0, 3.0);
        } else {
            painter.line_segment([p0, p1], egui::Stroke::new(width, col));
        }
        draw_arrow_head(painter, p0, p1, col);
    }

    for (i, &wp) in ov.world_pts.iter().take(m - 1).enumerate() {
        let sp = to_sc(wp);
        let (r, fill) = if i == 0 {
            (6.0, egui::Color32::from_rgb(46, 125, 50))
        } else if i == step_k {
            (6.5, egui::Color32::from_rgb(230, 81, 0))
        } else {
            (3.4, egui::Color32::from_rgb(25, 118, 210))
        };
        painter.circle_filled(sp, r, fill);
        painter.circle_stroke(sp, r, egui::Stroke::new(1.3, egui::Color32::WHITE));
    }

    // Parallel-transported local triad (e1, e2) at step_k.
    if let (Some(&wp), Some(&frame)) = (ov.world_pts.get(step_k), ov.result.frames.get(step_k)) {
        let sc = to_sc(wp);
        let len = (canvas.zoom * 0.48).clamp(16.0, 36.0);
        let e1 = egui::vec2(frame[0][0] as f32, -frame[1][0] as f32) * len;
        let e2 = egui::vec2(frame[0][1] as f32, -frame[1][1] as f32) * len;
        draw_triad_vec(painter, sc, sc + e1, egui::Color32::from_rgb(211, 47, 47), "e1");
        draw_triad_vec(painter, sc, sc + e2, egui::Color32::from_rgb(0, 121, 107), "e2");
    }

    draw_reference_inset(painter, rect, ov, step_k);
}

fn draw_triad_vec(painter: &egui::Painter, a: egui::Pos2, b: egui::Pos2, col: egui::Color32, label: &str) {
    painter.line_segment([a, b], egui::Stroke::new(2.3, col));
    draw_arrow_head(painter, a, b, col);
    painter.text(b + (b - a).normalized() * 7.0, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(9.5), col);
}

fn draw_arrow_head(painter: &egui::Painter, p0: egui::Pos2, p1: egui::Pos2, col: egui::Color32) {
    let d = p1 - p0;
    let len = d.length();
    if len < 9.0 {
        return;
    }
    let u = d / len;
    let n = egui::vec2(-u.y, u.x);
    let mid = p0 + d * 0.68;
    let w = 4.6;
    let st = egui::Stroke::new(1.8, col);
    painter.line_segment([mid + u * 2.5, mid - u * w + n * (0.65 * w)], st);
    painter.line_segment([mid + u * 2.5, mid - u * w - n * (0.65 * w)], st);
}

fn draw_reference_inset(painter: &egui::Painter, rect: egui::Rect, ov: &CanvasCircuitOverlay, step_k: usize) {
    let size = rect.width().min(rect.height()).clamp(120.0, 210.0).min(196.0);
    if size < 96.0 || ov.result.ref_path.len() < 2 {
        return;
    }
    let inset = egui::Rect::from_min_max(
        rect.right_bottom() - egui::vec2(size + 10.0, size + 10.0),
        rect.right_bottom() - egui::vec2(10.0, 10.0),
    );
    painter.rect_filled(inset, 6.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 238));
    painter.rect_stroke(inset, 6.0, egui::Stroke::new(1.2, egui::Color32::from_gray(145)), egui::StrokeKind::Inside);

    let title = match ov.result.frank_angle_deg {
        Some(a) => format!("Ref Chart (α={a:+.0}°, |b|={:.2})", ov.result.burgers_norm),
        None => format!("Ref Chart (non-orientable, |b|={:.2})", ov.result.burgers_norm),
    };
    painter.text(inset.left_top() + egui::vec2(7.0, 5.0), egui::Align2::LEFT_TOP, title, egui::FontId::proportional(10.0), egui::Color32::from_gray(40));

    let rp = &ov.result.ref_path;
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in rp {
        minx = minx.min(p[0]); maxx = maxx.max(p[0]);
        miny = miny.min(p[1]); maxy = maxy.max(p[1]);
    }
    let cx = 0.5 * (minx + maxx);
    let cy = 0.5 * (miny + maxy);
    let span = (maxx - minx).max(maxy - miny).max(0.8);
    let plot = inset.shrink2(egui::vec2(14.0, 22.0)).translate(egui::vec2(0.0, 6.0));
    let sc = (plot.width().min(plot.height()) as f64) / span;
    let map = |p: [f64; 2]| -> egui::Pos2 {
        egui::pos2(plot.center().x + ((p[0] - cx) * sc) as f32, plot.center().y - ((p[1] - cy) * sc) as f32)
    };

    for i in 0..rp.len() - 1 {
        let col = if i < step_k { egui::Color32::from_rgb(21, 101, 192) } else { egui::Color32::from_gray(175) };
        painter.line_segment([map(rp[i]), map(rp[i + 1])], egui::Stroke::new(1.8, col));
    }
    let p_start = map(rp[0]);
    let p_end = map(*rp.last().unwrap());
    if ov.result.burgers_norm > 1e-4 {
        let bcol = egui::Color32::from_rgb(230, 81, 0);
        painter.line_segment([p_start, p_end], egui::Stroke::new(2.2, bcol));
        draw_arrow_head(painter, p_start, p_end, bcol);
    }
    painter.circle_filled(p_start, 3.6, egui::Color32::from_rgb(46, 125, 50));
    painter.circle_filled(map(rp[step_k.min(rp.len() - 1)]), 4.0, egui::Color32::from_rgb(230, 81, 0));
}

/// Append the active Burgers circuit loop and basepoint marker to `ScenePrimitives` (for SVG/PNG export).
pub(crate) fn append_circuit_export_prims(prims: &mut ScenePrimitives, ov: &CanvasCircuitOverlay) {
    let m = ov.world_pts.len();
    for i in 0..m.saturating_sub(1) {
        let across = ov.across_steps.get(i).copied().unwrap_or(false);
        let col = if across { [216, 67, 21, 240] } else { [21, 101, 192, 240] };
        prims.lines.push(LinePrim {
            a: [ov.world_pts[i][0] as f32, ov.world_pts[i][1] as f32],
            b: [ov.world_pts[i + 1][0] as f32, ov.world_pts[i + 1][1] as f32],
            color: col,
            width: 2.6,
            dashed: across,
            arrow_mid: true,
        });
    }
    if let Some(&v0) = ov.world_pts.first() {
        prims.circles.push(CirclePrim {
            center: [v0[0] as f32, v0[1] as f32],
            radius_world: 0.14,
            min_px: 4.5,
            max_px: 7.5,
            fill: [46, 125, 50, 255],
            stroke: ([255, 255, 255, 255], 1.5),
        });
        if ov.result.burgers_norm > 1e-4 {
            let b = ov.result.closure_vector;
            let p1 = [v0[0] as f32 + b[0] as f32, v0[1] as f32 + b[1] as f32];
            prims.lines.push(LinePrim {
                a: [v0[0] as f32, v0[1] as f32],
                b: p1,
                color: [230, 81, 0, 245],
                width: 2.8,
                dashed: false,
                arrow_mid: true,
            });
            prims.labels.push(TextPrim {
                pos: p1,
                text: format!("b (|b|={:.2})", ov.result.burgers_norm),
                color: [230, 81, 0, 255],
                size_px: 10.5,
            });
        }
    }
}
