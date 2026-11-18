//! Camera overlays of the 3D viewport: the clickable X/Y/Z axis gizmo (top-right, below the caption band) and the
//! preset buttons (top-left, below the caption, wrapped onto up to three rows in narrow panes). Layout keeps them clear of the caption (top 24 px), the input hint
//! (bottom-left), the colour legend and the zoom HUD (bottom-right).
//!
//! The gizmo draws each world axis `e_i` as the orthographic screen image of the view rotation's column `i`:
//! `end = centre + AXIS_LEN_PX * (R[i][0], -R[i][1])` (screen y points down), with depth `R[i][2]` (> 0 = towards the
//! camera) used for painter's order and for picking the front-most end when two ends overlap.

use super::camera::CameraPreset;
use crate::theme::ThemeTokens;

/// Side of the square gizmo box (px).
pub(crate) const GIZMO_SIZE: f32 = 76.0;
/// Length of a positive axis on screen when it lies in the screen plane (px).
pub(crate) const AXIS_LEN_PX: f32 = 26.0;
/// Pick radius around an axis end (px).
pub(crate) const AXIS_HIT_RADIUS: f32 = 10.0;
/// Maximum width of the preset button rows (px); the five buttons need about 296 px on one row (Inter 11 pt).
pub(crate) const PRESET_ROW_W: f32 = 300.0;
/// Height of a preset button and the gap between buttons and rows (px).
pub(crate) const PRESET_BTN_H: f32 = 18.0;
pub(crate) const PRESET_GAP: f32 = 3.0;
/// At most three rows: they end 28 + 3 * 18 + 2 * 3 = 88 px below the top, clear of the legend (which starts 84 px
/// above the bottom, i.e. >= 116 px below the top in the smallest 200 px tall viewport) and of the hint.
const PRESET_MAX_ROWS: usize = 3;
/// Top of both overlays: below the 24 px caption band.
const OVERLAY_TOP: f32 = 28.0;

/// Gizmo box in the viewport's top-right corner, below the caption band; `None` below 200 x 200 px (so it never
/// reaches the legend + zoom HUD stack, which needs the bottom 84 px).
pub(crate) fn gizmo_rect_3d(rect: egui::Rect) -> Option<egui::Rect> {
    (rect.width() >= 200.0 && rect.height() >= 200.0).then(|| {
        let min = egui::pos2(rect.right() - 6.0 - GIZMO_SIZE, rect.top() + OVERLAY_TOP);
        egui::Rect::from_min_size(min, egui::vec2(GIZMO_SIZE, GIZMO_SIZE))
    })
}

/// Top-left corner and maximum row width of the preset buttons: below the caption, ending 12 px left of the gizmo
/// column; `None` below 200 px tall.
pub(crate) fn preset_area_3d(rect: egui::Rect) -> Option<(egui::Pos2, f32)> {
    let max_w = (rect.width() - (6.0 + 12.0 + GIZMO_SIZE + 6.0)).min(PRESET_ROW_W);
    (rect.height() >= 200.0 && max_w > 0.0).then(|| (rect.left_top() + egui::vec2(6.0, OVERLAY_TOP), max_w))
}

/// Greedy line wrapping: buttons of the given widths placed left to right from `origin`, starting a new row when the
/// next one would end past `max_w`. `None` when a button is wider than `max_w` or more than three rows are needed.
pub(crate) fn wrap_buttons(origin: egui::Pos2, max_w: f32, widths: &[f32]) -> Option<Vec<egui::Rect>> {
    let (mut x, mut row) = (0.0_f32, 0usize);
    let mut out = Vec::with_capacity(widths.len());
    for &w in widths {
        if !(w > 0.0 && w <= max_w) {
            return None;
        }
        if x > 0.0 && x + w > max_w {
            (x, row) = (0.0, row + 1);
        }
        if row >= PRESET_MAX_ROWS {
            return None;
        }
        let min = origin + egui::vec2(x, row as f32 * (PRESET_BTN_H + PRESET_GAP));
        out.push(egui::Rect::from_min_size(min, egui::vec2(w, PRESET_BTN_H)));
        x += w + PRESET_GAP;
    }
    Some(out)
}

/// The preset buttons in order; `None` is Reset.
pub(crate) const PRESET_BUTTONS: [Option<CameraPreset>; 5] =
    [Some(CameraPreset::Iso), Some(CameraPreset::Top), Some(CameraPreset::Front), Some(CameraPreset::Side), None];

pub(crate) fn button_label(button: Option<CameraPreset>) -> &'static str {
    button.map_or("Reset", CameraPreset::label)
}

/// Rects of the preset buttons in viewport `rect` (empty when they do not fit): each is its 11 pt label's measured
/// width plus the button padding, so the input blockers are exactly the drawn buttons.
pub(crate) fn preset_button_rects(ui: &egui::Ui, rect: egui::Rect) -> Vec<(Option<CameraPreset>, egui::Rect)> {
    let Some((origin, max_w)) = preset_area_3d(rect) else { return Vec::new() };
    let pad = 2.0 * ui.spacing().button_padding.x + 2.0;
    let widths: Vec<f32> = PRESET_BUTTONS
        .iter()
        .map(|b| {
            let galley = ui.painter().layout_no_wrap(button_label(*b).to_owned(), egui::FontId::proportional(11.0), egui::Color32::PLACEHOLDER);
            galley.size().x.ceil() + pad
        })
        .collect();
    wrap_buttons(origin, max_w, &widths).map_or_else(Vec::new, |rs| PRESET_BUTTONS.into_iter().zip(rs).collect())
}

/// One end of a gizmo axis: `axis` 0/1/2 = x/y/z, `positive` = the +axis end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AxisEnd {
    pub axis: usize,
    pub positive: bool,
    pub pos: egui::Pos2,
    pub depth: f32,
}

/// The six axis ends of the gizmo centred at `centre` under view rotation `rot` (column-major).
pub(crate) fn axis_ends(rot: &[[f32; 4]; 4], centre: egui::Pos2) -> [AxisEnd; 6] {
    std::array::from_fn(|k| {
        let (axis, positive) = (k % 3, k < 3);
        let sign = if positive { 1.0 } else { -1.0 };
        let col = rot[axis];
        AxisEnd {
            axis,
            positive,
            pos: centre + egui::vec2(col[0], -col[1]) * (sign * AXIS_LEN_PX),
            depth: sign * col[2],
        }
    })
}

/// Preset that looks along world axis `axis` (0 = x: Side (YZ), 1 = y: Front (XZ), 2 = z: Top (XY)).
pub(crate) fn preset_for_axis(axis: usize) -> CameraPreset {
    match axis {
        0 => CameraPreset::Side,
        1 => CameraPreset::Front,
        _ => CameraPreset::Top,
    }
}

/// Front-most axis end within [`AXIS_HIT_RADIUS`] of `p`.
pub(crate) fn gizmo_hit(rot: &[[f32; 4]; 4], gizmo: egui::Rect, p: egui::Pos2) -> Option<AxisEnd> {
    axis_ends(rot, gizmo.center())
        .into_iter()
        .filter(|e| e.pos.distance(p) <= AXIS_HIT_RADIUS)
        .max_by(|a, b| a.depth.total_cmp(&b.depth))
}

fn axis_color(axis: usize, is_light: bool) -> egui::Color32 {
    match (axis, is_light) {
        (0, true) => egui::Color32::from_rgb(200, 40, 40),
        (0, false) => egui::Color32::from_rgb(240, 90, 90),
        (1, true) => egui::Color32::from_rgb(30, 140, 50),
        (1, false) => egui::Color32::from_rgb(100, 210, 110),
        (_, true) => egui::Color32::from_rgb(40, 90, 220),
        (_, false) => egui::Color32::from_rgb(110, 150, 250),
    }
}

/// Paint the gizmo; `hovered` is the end under the pointer (outlined).
pub(crate) fn draw_gizmo(
    painter: &egui::Painter,
    gizmo: egui::Rect,
    rot: &[[f32; 4]; 4],
    hovered: Option<AxisEnd>,
    tokens: &ThemeTokens,
) {
    painter.rect_filled(gizmo, 6.0, tokens.bg_hud);
    painter.rect_stroke(gizmo, 6.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
    let centre = gizmo.center();
    let mut ends = axis_ends(rot, centre);
    ends.sort_by(|a, b| a.depth.total_cmp(&b.depth)); // back to front
    for end in ends {
        let full = axis_color(end.axis, tokens.is_light);
        let color = if end.positive { full } else { full.lerp_to_gamma(tokens.bg_hud, 0.6) };
        painter.line_segment([centre, end.pos], egui::Stroke::new(if end.positive { 2.0 } else { 1.5 }, color));
        let radius = if end.positive { 7.0 } else { 4.5 };
        painter.circle_filled(end.pos, radius, color);
        if end.positive {
            painter.text(
                end.pos,
                egui::Align2::CENTER_CENTER,
                ["X", "Y", "Z"][end.axis],
                egui::FontId::proportional(9.5),
                egui::Color32::WHITE,
            );
        }
        if hovered.is_some_and(|h| h.axis == end.axis && h.positive == end.positive) {
            painter.circle_stroke(end.pos, radius + 2.0, egui::Stroke::new(1.5, tokens.border_focus));
        }
    }
}

/// Hover text of a preset button (`None` = Reset).
pub(crate) fn preset_tooltip(button: Option<CameraPreset>) -> &'static str {
    match button {
        Some(CameraPreset::Iso) => "Iso: default oblique view",
        Some(CameraPreset::Top) => "Top (XY): look down -z, x right, y up",
        Some(CameraPreset::Front) => "Front (XZ): look along +y, x right, z up; drag turns about z",
        Some(CameraPreset::Side) => "Side (YZ): look along -x, y right, z up; drag turns about z",
        None => "Reset orientation, zoom and pan (0)",
    }
}
