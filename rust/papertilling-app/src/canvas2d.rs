use std::path::Path;

use crate::export::export_scene_to_path;
pub use crate::scene2d::Scene2D;
use crate::scene2d::{ScenePrimitives, compile_scene};

pub struct Canvas2D {
    pub pan: egui::Vec2,
    pub zoom: f32,
    auto_fit_pending: bool,
    scene: Scene2D,
    compiled: ScenePrimitives,
}

impl Canvas2D {
    pub fn new(scene: Scene2D) -> Self {
        let compiled = compile_scene(&scene);
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 56.0,
            auto_fit_pending: true,
            scene,
            compiled,
        }
    }

    pub fn set_scene(&mut self, scene: Scene2D, reset_view: bool) {
        self.compiled = compile_scene(&scene);
        self.scene = scene;
        if reset_view {
            self.auto_fit_pending = true;
        }
    }

    pub fn reset(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.auto_fit_pending = true;
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.compiled.site_count, self.compiled.bond_count)
    }

    pub fn export_to_path(&self, path: &Path) -> Result<String, String> {
        export_scene_to_path(&self.scene, path)
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());

        let (bmin, bmax) = self.compiled.bbox;
        let world_c = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        if self.auto_fit_pending && rect.width() > 32.0 && rect.height() > 32.0 {
            let span_x = (bmax[0] - bmin[0]).max(0.5);
            let span_y = (bmax[1] - bmin[1]).max(0.5);
            let pad_h = if self.compiled.note.is_empty() { 40.0 } else { 72.0 };
            let zx = (rect.width() - 40.0).max(24.0) / span_x;
            let zy = (rect.height() - pad_h).max(24.0) / span_y;
            self.zoom = zx.min(zy).clamp(8.0, 260.0);
            self.pan = egui::Vec2::ZERO;
            self.auto_fit_pending = false;
        }

        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.double_clicked() {
            self.reset();
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(6.0, 420.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(248, 249, 252));

        let center = rect.center() + self.pan;
        let zoom = self.zoom;
        let to_screen = |p: [f32; 2]| -> egui::Pos2 {
            egui::pos2(
                center.x + (p[0] - world_c[0]) * zoom,
                center.y - (p[1] - world_c[1]) * zoom,
            )
        };

        for poly in &self.compiled.polygons {
            if poly.points.len() >= 3 {
                let pts: Vec<egui::Pos2> = poly.points.iter().copied().map(to_screen).collect();
                let fill = to_color32(poly.fill);
                let stroke = poly
                    .stroke
                    .map_or(egui::Stroke::NONE, |(c, w)| egui::Stroke::new(w, to_color32(c)));
                // Fan triangulate so even slightly concave rings paint cleanly in egui
                let mut c_pos = egui::Pos2::ZERO;
                for &pt in &pts {
                    c_pos.x += pt.x / (pts.len() as f32);
                    c_pos.y += pt.y / (pts.len() as f32);
                }
                for m in 0..pts.len() {
                    painter.add(egui::Shape::convex_polygon(
                        vec![c_pos, pts[m], pts[(m + 1) % pts.len()]],
                        fill,
                        egui::Stroke::NONE,
                    ));
                }
                if stroke != egui::Stroke::NONE {
                    painter.add(egui::Shape::closed_line(pts, stroke));
                }
            }
        }

        for ln in &self.compiled.lines {
            let p0 = to_screen(ln.a);
            let p1 = to_screen(ln.b);
            let stroke = egui::Stroke::new(ln.width, to_color32(ln.color));
            if ln.dashed {
                draw_dashed_egui(&painter, p0, p1, stroke, 6.5, 4.5);
            } else {
                painter.line_segment([p0, p1], stroke);
            }
            if ln.arrow_mid {
                draw_arrow_mid_egui(&painter, p0, p1, stroke);
            }
        }

        for circ in &self.compiled.circles {
            let pos = to_screen(circ.center);
            let r = (circ.radius_world * zoom).clamp(circ.min_px, circ.max_px);
            if circ.fill[3] > 0 {
                painter.circle_filled(pos, r, to_color32(circ.fill));
            }
            if circ.stroke.1 > 0.0 && circ.stroke.0[3] > 0 {
                painter.circle_stroke(pos, r, egui::Stroke::new(circ.stroke.1, to_color32(circ.stroke.0)));
            }
        }

        for lbl in &self.compiled.labels {
            painter.text(
                to_screen(lbl.pos),
                egui::Align2::CENTER_CENTER,
                &lbl.text,
                egui::FontId::proportional(lbl.size_px),
                to_color32(lbl.color),
            );
        }

        if !self.compiled.title.is_empty() {
            painter.text(
                rect.left_top() + egui::vec2(8.0, 6.0),
                egui::Align2::LEFT_TOP,
                &self.compiled.title,
                egui::FontId::proportional(12.5),
                egui::Color32::from_rgb(38, 48, 64),
            );
        }

        if !self.compiled.note.is_empty() {
            let banner_h = 24.0;
            let top_y = if self.compiled.title.is_empty() { 6.0 } else { 24.0 };
            let banner = egui::Rect::from_min_max(
                rect.left_top() + egui::vec2(8.0, top_y),
                egui::pos2(rect.right() - 8.0, (rect.top() + top_y + banner_h).min(rect.bottom() - 4.0)),
            );
            painter.rect_filled(banner, 4.0, egui::Color32::from_rgba_unmultiplied(255, 244, 214, 238));
            painter.rect_stroke(
                banner,
                4.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(218, 160, 48)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                banner.left_center() + egui::vec2(8.0, 0.0),
                egui::Align2::LEFT_CENTER,
                &self.compiled.note,
                egui::FontId::proportional(11.5),
                egui::Color32::from_rgb(124, 68, 10),
            );
        }

        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(180)),
            egui::StrokeKind::Inside,
        );
    }
}

fn to_color32(c: [u8; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3])
}

fn draw_dashed_egui(
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

fn draw_arrow_mid_egui(painter: &egui::Painter, p0: egui::Pos2, p1: egui::Pos2, stroke: egui::Stroke) {
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
