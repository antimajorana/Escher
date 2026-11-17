//! Shared drawing primitives for `Canvas2D` (`to_color32`, markers, overlays, dashed/arrow lines).

use crate::canvas2d::MarkerKind;

pub(crate) fn marker_colors(kind: MarkerKind) -> (egui::Color32, egui::Color32) {
    match kind {
        MarkerKind::DisclinationPlus => (
            egui::Color32::from_rgb(211, 47, 47),
            egui::Color32::from_rgba_unmultiplied(211, 47, 47, 68),
        ),
        MarkerKind::DisclinationMinus => (
            egui::Color32::from_rgb(25, 118, 210),
            egui::Color32::from_rgba_unmultiplied(25, 118, 210, 68),
        ),
        MarkerKind::Dislocation => (
            egui::Color32::from_rgb(237, 108, 2),
            egui::Color32::from_rgba_unmultiplied(237, 108, 2, 68),
        ),
    }
}

pub(crate) fn draw_overlays(painter: &egui::Painter, rect: egui::Rect, title: &str, note: &str) {
    if !title.is_empty() {
        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            title,
            egui::FontId::proportional(12.5),
            egui::Color32::from_rgb(38, 48, 64),
        );
    }
    if !note.is_empty() {
        let banner_h = 24.0;
        let top_y = if title.is_empty() { 6.0 } else { 24.0 };
        let banner = egui::Rect::from_min_max(
            rect.left_top() + egui::vec2(8.0, top_y),
            egui::pos2(rect.right() - 8.0, (rect.top() + top_y + banner_h).min(rect.bottom() - 4.0)),
        );
        painter.rect_filled(banner, 4.0, egui::Color32::from_rgba_unmultiplied(255, 244, 214, 238));
        painter.rect_stroke(banner, 4.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(218, 160, 48)), egui::StrokeKind::Inside);
        painter.text(banner.left_center() + egui::vec2(8.0, 0.0), egui::Align2::LEFT_CENTER, note, egui::FontId::proportional(11.5), egui::Color32::from_rgb(124, 68, 10));
    }
}

pub(crate) fn to_color32(c: [u8; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3])
}

pub(crate) fn draw_dashed_egui(
    painter: &egui::Painter,
    p0: egui::Pos2,
    p1: egui::Pos2,
    stroke: egui::Stroke,
    dash: f32,
    gap: f32,
) {
    let delta = p1 - p0;
    let total = delta.length();
    if total < 1e-3 {
        return;
    }
    let dir = delta / total;
    let mut s = 0.0;
    while s < total {
        let e = (s + dash).min(total);
        painter.line_segment([p0 + dir * s, p0 + dir * e], stroke);
        s += dash + gap;
    }
}

pub(crate) fn draw_arrow_mid_egui(
    painter: &egui::Painter,
    p0: egui::Pos2,
    p1: egui::Pos2,
    stroke: egui::Stroke,
) {
    let d = p1 - p0;
    let len = d.length();
    if len < 10.0 {
        return;
    }
    let u = d / len;
    let n = egui::vec2(-u.y, u.x);
    let mid = p0 + d * 0.54;
    let wing = 6.0;
    let tip = mid + u * 3.5;
    painter.line_segment([tip, tip - u * wing + n * (wing * 0.62)], stroke);
    painter.line_segment([tip, tip - u * wing - n * (wing * 0.62)], stroke);
}
