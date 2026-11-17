//! Click-to-enlarge interactive zoomable inspector modal for Core Gallery cards (`core_panel.rs`).

use papertilling_core::defect::combine_types::CoreGalleryEntry;

use super::card::compile_core_card;
use crate::canvas2d::{
    ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_FACTOR, draw_dashed_egui, input_zoom_factor,
    take_keyboard_zoom_action, to_color32, zoom_by_factor_about, ZoomKeyAction,
};
use crate::scene2d::ScenePrimitives;
use crate::theme::ThemeTokens;

pub struct CoreInspectorModal {
    pub title: String,
    pub group_tag: String,
    pub is_match: bool,
    pub description: String,
    pub rings_text: String,
    pub also_text: String,
    prims: ScenePrimitives,
    pan: egui::Vec2,
    zoom: f32,
    fit_zoom: f32,
    world_center: [f32; 2],
    auto_fit: bool,
}

impl CoreInspectorModal {
    pub fn from_entry(entry: &CoreGalleryEntry, group_tag: &str, is_match: bool) -> Self {
        let prims = compile_core_card(entry);
        let (bmin, bmax) = prims.bbox;
        let world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        let rings_text = if entry.rings.is_empty() {
            String::from("-")
        } else {
            entry.rings.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
        };
        let also_text = if entry.also.is_empty() {
            String::new()
        } else {
            format!("Also reached by: {}", entry.also.join(", "))
        };
        Self {
            title: entry.name.clone(),
            group_tag: group_tag.to_string(),
            is_match,
            description: entry.description.clone(),
            rings_text,
            also_text,
            prims,
            pan: egui::Vec2::ZERO,
            zoom: 80.0,
            fit_zoom: 80.0,
            world_center,
            auto_fit: true,
        }
    }

    pub fn reset_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.auto_fit = true;
    }

    /// Show the floating modal window; returns `false` when the user closes it.
    pub fn show(&mut self, ctx: &egui::Context, tokens: &ThemeTokens) -> bool {
        let mut open = true;
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            return false;
        }
        egui::Window::new(format!("Core Inspector — {} ({})", self.title, self.group_tag))
            .open(&mut open)
            .default_size([540.0, 500.0])
            .min_size([340.0, 320.0])
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if self.is_match {
                        ui.colored_label(tokens.status_ok, egui::RichText::new("MATCH").strong());
                        ui.separator();
                    }
                    ui.label(format!("Rings: [{}]", self.rings_text));
                    if !self.description.is_empty() {
                        ui.separator();
                        ui.small(&self.description);
                    }
                });
                if !self.also_text.is_empty() {
                    ui.small(&self.also_text);
                }
                ui.add_space(4.0);

                let avail = ui.available_size().max(egui::vec2(160.0, 160.0));
                let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

                if self.auto_fit && rect.width() > 24.0 && rect.height() > 24.0 {
                    let (bmin, bmax) = self.prims.bbox;
                    self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
                    let sx = (bmax[0] - bmin[0]).max(0.5);
                    let sy = (bmax[1] - bmin[1]).max(0.5);
                    let fit = ((rect.width() - 28.0) / sx)
                        .min((rect.height() - 28.0) / sy)
                        .clamp(ZOOM_MIN, ZOOM_MAX);
                    self.fit_zoom = fit;
                    self.zoom = fit;
                    self.pan = egui::Vec2::ZERO;
                    self.auto_fit = false;
                }

                let hud_rect = egui::Rect::from_min_max(
                    rect.right_bottom() - egui::vec2(164.0, 30.0),
                    rect.right_bottom() - egui::vec2(6.0, 6.0),
                );
                self.handle_input(ui, rect, &resp, hud_rect);

                paint_zoomed_prims(
                    &ui.painter_at(rect),
                    rect,
                    &self.prims,
                    self.world_center,
                    self.pan,
                    self.zoom,
                    tokens,
                );

                ui.scope_builder(egui::UiBuilder::new().max_rect(hud_rect), |ui| {
                    egui::Frame::NONE
                        .fill(tokens.bg_hud)
                        .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(5, 3))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                if ui.small_button("−").on_hover_text("Zoom out (-)" ).clicked() {
                                    (self.pan, self.zoom) = zoom_by_factor_about(
                                        self.pan, self.zoom, 1.0 / ZOOM_STEP_FACTOR, rect.center(), None,
                                    );
                                }
                                let pct = ((self.zoom / self.fit_zoom.max(1e-3)) * 100.0).round() as i32;
                                ui.small(format!("{pct}%"));
                                if ui.small_button("+").on_hover_text("Zoom in (+)").clicked() {
                                    (self.pan, self.zoom) = zoom_by_factor_about(
                                        self.pan, self.zoom, ZOOM_STEP_FACTOR, rect.center(), None,
                                    );
                                }
                                if ui.small_button("Fit").on_hover_text("Reset zoom (0 / double-click)").clicked() {
                                    self.reset_view();
                                }
                            });
                        });
                });
            });
        open
    }

    fn handle_input(&mut self, ui: &egui::Ui, rect: egui::Rect, resp: &egui::Response, hud_rect: egui::Rect) {
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let over_hud = ptr.is_some_and(|p| hud_rect.contains(p));
        let drag_from_hud = press_orig.is_some_and(|p| hud_rect.contains(p));
        if !drag_from_hud && resp.dragged() {
            self.pan += resp.drag_delta();
        }
        if !over_hud && !drag_from_hud && resp.double_clicked() {
            self.reset_view();
        }
        // Geometric containment alone also matches covered/background windows.
        if ui.rect_contains_pointer(resp.interact_rect) {
            let anchor = ptr.filter(|p| rect.contains(*p));
            let factor = ui.input(input_zoom_factor);
            if (factor - 1.0).abs() > 1e-5 {
                (self.pan, self.zoom) = zoom_by_factor_about(self.pan, self.zoom, factor, rect.center(), anchor);
            }
            if let Some(act) = take_keyboard_zoom_action(ui, resp) {
                match act {
                    ZoomKeyAction::In | ZoomKeyAction::Out => {
                        let factor = if act == ZoomKeyAction::In { ZOOM_STEP_FACTOR } else { 1.0 / ZOOM_STEP_FACTOR };
                        (self.pan, self.zoom) = zoom_by_factor_about(self.pan, self.zoom, factor, rect.center(), anchor);
                    }
                    ZoomKeyAction::Fit => self.reset_view(),
                }
            }
        }
    }
}

fn paint_zoomed_prims(
    painter: &egui::Painter,
    rect: egui::Rect,
    prims: &ScenePrimitives,
    wc: [f32; 2],
    pan: egui::Vec2,
    zoom: f32,
    tokens: &ThemeTokens,
) {
    painter.rect_filled(rect, 4.0, tokens.bg_canvas_2d);
    let origin = rect.center() + pan;
    let to_screen = |p: [f32; 2]| -> egui::Pos2 {
        egui::pos2(origin.x + (p[0] - wc[0]) * zoom, origin.y - (p[1] - wc[1]) * zoom)
    };
    for poly in &prims.polygons {
        if poly.points.len() >= 3 {
            let pts: Vec<egui::Pos2> = poly.points.iter().copied().map(to_screen).collect();
            let fill = to_color32(poly.fill);
            let mut cp = egui::Pos2::ZERO;
            for &pt in &pts {
                cp.x += pt.x / (pts.len() as f32);
                cp.y += pt.y / (pts.len() as f32);
            }
            for m in 0..pts.len() {
                painter.add(egui::Shape::convex_polygon(
                    vec![cp, pts[m], pts[(m + 1) % pts.len()]],
                    fill,
                    egui::Stroke::NONE,
                ));
            }
            if let Some((sc, sw)) = poly.stroke {
                painter.add(egui::Shape::closed_line(pts, egui::Stroke::new(sw, tokens.remap_scene_ink(sc))));
            }
        }
    }
    for ln in &prims.lines {
        let (p0, p1) = (to_screen(ln.a), to_screen(ln.b));
        let stroke = egui::Stroke::new(ln.width * 1.15, tokens.remap_scene_ink(ln.color));
        if ln.dashed {
            draw_dashed_egui(painter, p0, p1, stroke, 6.0, 4.0);
        } else {
            painter.line_segment([p0, p1], stroke);
        }
    }
    for circ in &prims.circles {
        let pos = to_screen(circ.center);
        let r = (circ.radius_world * zoom).clamp(circ.min_px, circ.max_px * 1.8);
        if circ.fill[3] > 0 {
            painter.circle_filled(pos, r, tokens.remap_scene_ink(circ.fill));
        }
        if circ.stroke.1 > 0.0 && circ.stroke.0[3] > 0 {
            painter.circle_stroke(pos, r, egui::Stroke::new(circ.stroke.1, tokens.remap_scene_ink(circ.stroke.0)));
        }
    }
    let has_faces = !prims.polygons.is_empty();
    for lbl in &prims.labels {
        let col = if has_faces { to_color32(lbl.color) } else { tokens.remap_scene_ink(lbl.color) };
        painter.text(
            to_screen(lbl.pos),
            egui::Align2::CENTER_CENTER,
            &lbl.text,
            egui::FontId::proportional((lbl.size_px * 1.2).clamp(10.0, 16.0)),
            col,
        );
    }
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, tokens.border_strong),
        egui::StrokeKind::Inside,
    );
}

#[cfg(test)]
mod tests;
