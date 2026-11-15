use std::path::Path;

use crate::export::export_scene_to_path;
pub use crate::scene2d::Scene2D;
use crate::scene2d::{ScenePrimitives, compile_scene};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    DisclinationPlus,
    DisclinationMinus,
    Dislocation,
}

#[derive(Clone, Debug)]
pub struct CanvasMarker {
    pub index: usize,
    pub pos: [f64; 2],
    pub kind: MarkerKind,
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct CanvasInteraction {
    pub clicked_world: Option<[f64; 2]>,
    pub dragged_marker: Option<(usize, [f64; 2])>,
    pub drag_released: bool,
    pub hovered_world: Option<[f64; 2]>,
}

pub struct Canvas2D {
    pub pan: egui::Vec2,
    pub zoom: f32,
    world_center: [f32; 2],
    auto_fit_pending: bool,
    active_drag_pin: Option<usize>,
    scene: Scene2D,
    compiled: ScenePrimitives,
}

impl Canvas2D {
    pub fn new(scene: Scene2D) -> Self {
        let compiled = compile_scene(&scene);
        let (bmin, bmax) = compiled.bbox;
        let world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 56.0,
            world_center,
            auto_fit_pending: true,
            active_drag_pin: None,
            scene,
            compiled,
        }
    }

    pub fn set_scene(&mut self, scene: Scene2D, reset_view: bool) {
        self.compiled = compile_scene(&scene);
        self.scene = scene;
        if reset_view {
            let (bmin, bmax) = self.compiled.bbox;
            self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            self.auto_fit_pending = true;
            self.active_drag_pin = None;
        }
    }

    pub fn reset(&mut self) {
        let (bmin, bmax) = self.compiled.bbox;
        self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        self.pan = egui::Vec2::ZERO;
        self.auto_fit_pending = true;
        self.active_drag_pin = None;
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.compiled.site_count, self.compiled.bond_count)
    }

    pub fn export_to_path(&self, path: &Path) -> Result<String, String> {
        export_scene_to_path(&self.scene, path)
    }

    pub fn world_to_screen_at(&self, rect: egui::Rect, p: [f32; 2]) -> egui::Pos2 {
        let center = rect.center() + self.pan;
        egui::pos2(
            center.x + (p[0] - self.world_center[0]) * self.zoom,
            center.y - (p[1] - self.world_center[1]) * self.zoom,
        )
    }

    pub fn screen_to_world_at(&self, rect: egui::Rect, pos: egui::Pos2) -> [f64; 2] {
        let center = rect.center() + self.pan;
        let z = self.zoom.max(1e-3);
        [
            (self.world_center[0] + (pos.x - center.x) / z) as f64,
            (self.world_center[1] - (pos.y - center.y) / z) as f64,
        ]
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let _ = self.show_inner(ui, &[], false, None, false);
    }

    pub fn show_interactive(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        preview_world: Option<[f64; 2]>,
    ) -> CanvasInteraction {
        self.show_inner(ui, markers, click_to_place, preview_world, true)
    }

    fn show_inner(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        preview_world: Option<[f64; 2]>,
        interactive: bool,
    ) -> CanvasInteraction {
        let mut out = CanvasInteraction::default();
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let sense = if interactive {
            egui::Sense::click_and_drag()
        } else {
            egui::Sense::drag()
        };
        let (rect, response) = ui.allocate_exact_size(available, sense);

        if self.auto_fit_pending && rect.width() > 32.0 && rect.height() > 32.0 {
            let (bmin, bmax) = self.compiled.bbox;
            self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            let span_x = (bmax[0] - bmin[0]).max(0.5);
            let span_y = (bmax[1] - bmin[1]).max(0.5);
            let pad_h = if self.compiled.note.is_empty() { 40.0 } else { 72.0 };
            let zx = (rect.width() - 40.0).max(24.0) / span_x;
            let zy = (rect.height() - pad_h).max(24.0) / span_y;
            self.zoom = zx.min(zy).clamp(8.0, 260.0);
            self.pan = egui::Vec2::ZERO;
            self.auto_fit_pending = false;
        }

        let hit_r = 14.0_f32;
        let hit_pin_at = |this: &Self, pt: egui::Pos2| -> Option<usize> {
            markers
                .iter()
                .rev()
                .filter_map(|m| {
                    let sp = this.world_to_screen_at(rect, [m.pos[0] as f32, m.pos[1] as f32]);
                    let d = sp.distance(pt);
                    (d <= hit_r).then_some((m.index, d))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(idx, _)| idx)
        };

        if interactive && response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos());
            self.active_drag_pin = origin.and_then(|pt| hit_pin_at(self, pt));
        }

        if response.dragged() {
            if let Some(pin_idx) = self.active_drag_pin {
                if pin_idx < markers.len() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        out.dragged_marker = Some((pin_idx, self.screen_to_world_at(rect, pos)));
                    }
                } else {
                    self.active_drag_pin = None;
                }
            } else {
                self.pan += response.drag_delta();
            }
        }

        if interactive && response.drag_stopped() && self.active_drag_pin.take().is_some() {
            out.drag_released = true;
        }

        if response.double_clicked() {
            self.reset();
        } else if interactive
            && click_to_place
            && response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            out.clicked_world = Some(self.screen_to_world_at(rect, pos));
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(6.0, 420.0);
            }
            if interactive && let Some(hpos) = response.hover_pos() {
                out.hovered_world = Some(self.screen_to_world_at(rect, hpos));
                if self.active_drag_pin.is_some() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if let Some(pidx) = hit_pin_at(self, hpos) {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    if let Some(m) = markers.iter().find(|m| m.index == pidx) {
                        response.clone().on_hover_text(format!(
                            "#{} {} at ({:+.3}, {:+.3}) — drag to move",
                            m.index + 1,
                            m.label,
                            m.pos[0],
                            m.pos[1]
                        ));
                    }
                } else if click_to_place {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                }
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(248, 249, 252));

        let center = rect.center() + self.pan;
        let zoom = self.zoom;
        let world_c = self.world_center;
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
                painter.circle_stroke(
                    pos,
                    r,
                    egui::Stroke::new(circ.stroke.1, to_color32(circ.stroke.0)),
                );
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

        if let Some(pw) = preview_world {
            let sp = to_screen([pw[0] as f32, pw[1] as f32]);
            painter.circle_filled(
                sp,
                8.5,
                egui::Color32::from_rgba_unmultiplied(46, 125, 50, 55),
            );
            painter.circle_stroke(
                sp,
                9.5,
                egui::Stroke::new(1.8, egui::Color32::from_rgb(46, 125, 50)),
            );
        }

        for m in markers {
            let sp = to_screen([m.pos[0] as f32, m.pos[1] as f32]);
            let (fill, halo) = marker_colors(m.kind);
            let is_dragged = self.active_drag_pin == Some(m.index);
            painter.circle_filled(sp, if is_dragged { 16.5 } else { 13.5 }, halo);
            painter.circle_filled(sp, 9.5, fill);
            painter.circle_stroke(sp, 9.5, egui::Stroke::new(1.8, egui::Color32::WHITE));
            painter.text(
                sp,
                egui::Align2::CENTER_CENTER,
                format!("{}", m.index + 1),
                egui::FontId::proportional(10.5),
                egui::Color32::WHITE,
            );
            if !m.label.is_empty() {
                painter.text(
                    sp + egui::vec2(0.0, -15.5),
                    egui::Align2::CENTER_CENTER,
                    &m.label,
                    egui::FontId::proportional(10.5),
                    fill,
                );
            }
        }

        draw_overlays(&painter, rect, &self.compiled.title, &self.compiled.note);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(180)),
            egui::StrokeKind::Inside,
        );
        out
    }
}

fn marker_colors(kind: MarkerKind) -> (egui::Color32, egui::Color32) {
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

fn draw_overlays(painter: &egui::Painter, rect: egui::Rect, title: &str, note: &str) {
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
            egui::pos2(
                rect.right() - 8.0,
                (rect.top() + top_y + banner_h).min(rect.bottom() - 4.0),
            ),
        );
        painter.rect_filled(
            banner,
            4.0,
            egui::Color32::from_rgba_unmultiplied(255, 244, 214, 238),
        );
        painter.rect_stroke(
            banner,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(218, 160, 48)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            banner.left_center() + egui::vec2(8.0, 0.0),
            egui::Align2::LEFT_CENTER,
            note,
            egui::FontId::proportional(11.5),
            egui::Color32::from_rgb(124, 68, 10),
        );
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

fn draw_arrow_mid_egui(
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
