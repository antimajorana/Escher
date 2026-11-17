use std::path::Path;

use crate::export::{export_scene_full, export_scene_with_overlays};
pub use crate::scene2d::Scene2D;
use crate::scene2d::{ScenePrimitives, compile_scene};

pub mod circuit;
mod cuts;
mod draw_helpers;
pub mod gauge_overlay;
mod hit;
#[cfg(test)]
mod tests;

pub use circuit::CanvasCircuitOverlay;
pub use cuts::{CanvasCutRay, CanvasKinematicOverlay, CanvasReactionArrow};
pub(crate) use draw_helpers::{draw_arrow_mid_egui, draw_dashed_egui, draw_overlays, marker_colors, to_color32};
pub use gauge_overlay::CrystalOverlayData;
pub use hit::{
    PIN_PICK_RADIUS_PX, ViewTransform, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_FACTOR, ZoomKeyAction,
    input_zoom_factor, keyboard_zoom_action, pick_marker, zoom_by_factor_about,
};
pub(crate) use hit::take_keyboard_zoom_action;

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
    /// A click (press and release without dragging) on empty canvas, in world coordinates.
    pub clicked_world: Option<[f64; 2]>,
    pub dragged_marker: Option<(usize, [f64; 2])>,
    pub drag_released: bool,
    pub hovered_world: Option<[f64; 2]>,
    pub shift_held: bool,
    /// A click that landed on a pin (its `CanvasMarker::index`); `clicked_world` is then `None`, so a click
    /// on an existing pin never places a second defect on top of it.
    pub clicked_marker: Option<usize>,
    /// A click that landed on a vertex of the active Burgers circuit loop.
    pub clicked_circuit_vertex: Option<usize>,
}

pub struct Canvas2D {
    pub pan: egui::Vec2,
    pub zoom: f32,
    fit_zoom: f32,
    world_center: [f32; 2],
    auto_fit_pending: bool,
    active_drag_pin: Option<usize>,
    scene: Scene2D,
    compiled: ScenePrimitives,
    export_markers: Vec<CanvasMarker>,
    cut_rays: Vec<CanvasCutRay>,
    kinematic_overlay: CanvasKinematicOverlay,
    circuit_overlay: Option<CanvasCircuitOverlay>,
    gauge_overlay: Option<CrystalOverlayData>,
    export_title: Option<String>,
}

impl Canvas2D {
    pub fn new(scene: Scene2D) -> Self {
        let compiled = compile_scene(&scene);
        let (bmin, bmax) = compiled.bbox;
        let world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 56.0,
            fit_zoom: 56.0,
            world_center,
            auto_fit_pending: true,
            active_drag_pin: None,
            scene,
            compiled,
            export_markers: Vec::new(),
            cut_rays: Vec::new(),
            kinematic_overlay: CanvasKinematicOverlay::default(),
            circuit_overlay: None,
            gauge_overlay: None,
            export_title: None,
        }
    }

    pub fn set_scene(&mut self, scene: Scene2D, reset_view: bool) {
        self.set_scene_with_title(scene, None, reset_view);
    }

    pub fn set_scene_with_title(&mut self, scene: Scene2D, export_title: Option<String>, reset_view: bool) {
        self.export_markers.clear();
        self.cut_rays.clear();
        self.kinematic_overlay = CanvasKinematicOverlay::default();
        self.circuit_overlay = None;
        self.gauge_overlay = None;
        self.export_title = export_title;
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

    pub fn zoom_at(&mut self, rect: egui::Rect, factor: f32, anchor: Option<egui::Pos2>) {
        (self.pan, self.zoom) = zoom_by_factor_about(self.pan, self.zoom, factor, rect.center(), anchor);
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.compiled.site_count, self.compiled.bond_count)
    }

    #[cfg(test)]
    pub(crate) fn scene(&self) -> &Scene2D { &self.scene }

    pub fn export_to_path(&self, path: &Path) -> Result<String, String> {
        if self.circuit_overlay.is_some() || self.gauge_overlay.is_some() {
            let mut prims = self.compiled.clone();
            if let Some(ov) = &self.gauge_overlay {
                gauge_overlay::apply_overlay_to_prims(&mut prims, ov);
            }
            if let Some(circ) = &self.circuit_overlay {
                circuit::append_circuit_export_prims(&mut prims, circ);
            }
            export_scene_with_overlays(&prims, &self.export_markers, self.export_title.as_deref(), path)
        } else {
            export_scene_full(&self.scene, &self.export_markers, self.export_title.as_deref(), path)
        }
    }

    pub(crate) fn set_circuit_overlay(&mut self, circuit: Option<CanvasCircuitOverlay>) {
        self.circuit_overlay = circuit;
    }

    pub(crate) fn set_gauge_overlay(&mut self, ov: Option<CrystalOverlayData>) {
        self.gauge_overlay = ov;
        self.compiled = compile_scene(&self.scene);
        if let Some(data) = &self.gauge_overlay {
            gauge_overlay::apply_overlay_to_prims(&mut self.compiled, data);
        }
    }

    pub(crate) fn set_export_markers(&mut self, markers: &[CanvasMarker]) {
        self.export_markers.clear();
        self.export_markers.extend_from_slice(markers);
    }

    pub(crate) fn set_cut_rays(&mut self, rays: &[CanvasCutRay]) {
        self.cut_rays.clear();
        self.cut_rays.extend_from_slice(rays);
    }

    pub(crate) fn set_kinematic_overlay(&mut self, overlay: CanvasKinematicOverlay) {
        self.kinematic_overlay = overlay;
    }

    #[cfg(test)]
    pub(crate) fn cut_rays(&self) -> &[CanvasCutRay] {
        &self.cut_rays
    }

    #[cfg(test)]
    pub(crate) fn kinematic_overlay(&self) -> &CanvasKinematicOverlay {
        &self.kinematic_overlay
    }

    /// The world <-> screen map of the current view inside `rect`.
    pub fn view_at(&self, rect: egui::Rect) -> ViewTransform {
        ViewTransform { origin: rect.center() + self.pan, zoom: self.zoom, world_center: self.world_center }
    }

    pub fn world_to_screen_at(&self, rect: egui::Rect, p: [f32; 2]) -> egui::Pos2 {
        self.view_at(rect).world_to_screen(p)
    }

    pub fn screen_to_world_at(&self, rect: egui::Rect, pos: egui::Pos2) -> [f64; 2] {
        self.view_at(rect).screen_to_world(pos)
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        self.export_markers.clear();
        let _ = self.show_inner(ui, &[], false, None, false);
    }

    pub fn show_interactive(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        preview_world: Option<[f64; 2]>,
    ) -> CanvasInteraction {
        self.set_export_markers(markers);
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
        let mut out = CanvasInteraction {
            shift_held: ui.input(|i| i.modifiers.shift),
            ..Default::default()
        };
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

        if self.auto_fit_pending && rect.width() > 32.0 && rect.height() > 32.0 {
            let (bmin, bmax) = self.compiled.bbox;
            self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            let span_x = (bmax[0] - bmin[0]).max(0.5);
            let span_y = (bmax[1] - bmin[1]).max(0.5);
            let pad_h = if self.compiled.note.is_empty() { 40.0 } else { 72.0 };
            let zx = (rect.width() - 40.0).max(24.0) / span_x;
            let zy = (rect.height() - pad_h).max(24.0) / span_y;
            let fit = zx.min(zy).clamp(ZOOM_MIN, ZOOM_MAX);
            self.fit_zoom = fit;
            self.zoom = fit;
            self.pan = egui::Vec2::ZERO;
            self.auto_fit_pending = false;
        }

        let has_hud = rect.width() >= 190.0 && rect.height() >= 80.0;
        let hud_rect = egui::Rect::from_min_max(
            rect.right_bottom() - egui::vec2(164.0, 30.0),
            rect.right_bottom() - egui::vec2(6.0, 6.0),
        );
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let over_hud = has_hud && ptr.is_some_and(|p| hud_rect.contains(p));
        let drag_from_hud = has_hud && press_orig.is_some_and(|p| hud_rect.contains(p));

        let hit_pin_at = |this: &Self, pt: egui::Pos2| pick_marker(markers, &this.view_at(rect), pt, PIN_PICK_RADIUS_PX);

        // Smooth wheel + pinch / Ctrl+wheel zoom anchored at the cursor position inside `rect`.
        let anchor = response.hover_pos().or(ptr).filter(|p| rect.contains(*p));
        if ui.rect_contains_pointer(rect) {
            let factor = ui.input(input_zoom_factor);
            if (factor - 1.0).abs() > 1e-5 {
                self.zoom_at(rect, factor, anchor);
            }
            if let Some(act) = take_keyboard_zoom_action(ui, &response) {
                match act {
                    ZoomKeyAction::In => self.zoom_at(rect, ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Out => self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Fit => self.reset(),
                }
            }
        }

        if interactive && !drag_from_hud && response.drag_started() {
            let origin = press_orig.or_else(|| response.interact_pointer_pos());
            self.active_drag_pin = origin.and_then(|pt| hit_pin_at(self, pt));
        }

        if !drag_from_hud && response.dragged() {
            if let Some(pin_idx) = self.active_drag_pin {
                if markers.iter().any(|m| m.index == pin_idx) {
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

        if !over_hud && !drag_from_hud && response.double_clicked() {
            self.reset();
        } else if interactive
            && !over_hud
            && !drag_from_hud
            && response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            match hit_pin_at(self, pos) {
                Some(idx) => out.clicked_marker = Some(idx),
                None => {
                    if let Some(circ) = &self.circuit_overlay
                        && let Some(v) = circuit::pick_circuit_vertex(self, rect, circ, pos, 11.0)
                    {
                        out.clicked_circuit_vertex = Some(v);
                    } else if click_to_place {
                        out.clicked_world = Some(self.screen_to_world_at(rect, pos));
                    }
                }
            }
        }

        if !over_hud
            && response.hovered()
            && interactive
            && let Some(hpos) = response.hover_pos()
        {
            out.hovered_world = Some(self.screen_to_world_at(rect, hpos));
            if self.active_drag_pin.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if let Some(pidx) = hit_pin_at(self, hpos) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                if let Some(m) = markers.iter().find(|m| m.index == pidx) {
                    let hint = if self.kinematic_overlay.locked_pins.contains(&m.index) {
                        "immobile fracton (ΔN=0) — Shift+drag to step & emit b"
                    } else {
                        "drag to move"
                    };
                    response.clone().on_hover_text(format!(
                        "#{} {} at ({:+.3}, {:+.3}) — {hint}",
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

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(248, 249, 252));

        let zoom = self.zoom;
        let to_screen = |p: [f32; 2]| -> egui::Pos2 { self.world_to_screen_at(rect, p) };

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

        if !self.cut_rays.is_empty() {
            cuts::draw_cut_rays(&painter, self, rect, &self.cut_rays);
        }
        if let Some(ov) = &self.gauge_overlay {
            gauge_overlay::draw_gauge_overlays(&painter, self, rect, ov);
        }
        if !self.kinematic_overlay.glide_lines.is_empty()
            || !self.kinematic_overlay.locked_pins.is_empty()
            || self.kinematic_overlay.reaction.is_some()
        {
            cuts::draw_kinematic_overlay(&painter, self, rect, markers, &self.kinematic_overlay);
        }
        if let Some(circ) = &self.circuit_overlay {
            circuit::draw_circuit_overlay(&painter, self, rect, circ);
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

        let overlay_title = if matches!(self.scene, Scene2D::SurfaceNet { .. }) {
            self.export_title.as_deref().unwrap_or(&self.compiled.title)
        } else {
            &self.compiled.title
        };
        draw_overlays(&painter, rect, overlay_title, &self.compiled.note);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(180)),
            egui::StrokeKind::Inside,
        );
        if has_hud {
            ui.scope_builder(egui::UiBuilder::new().max_rect(hud_rect), |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(245, 247, 251, 234))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(182)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if ui.small_button("−").on_hover_text("Zoom out (-)").clicked() {
                                self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, None);
                            }
                            let pct = ((self.zoom / self.fit_zoom.max(1e-3)) * 100.0).round() as i32;
                            ui.small(format!("{pct}%"));
                            if ui.small_button("+").on_hover_text("Zoom in (+)").clicked() {
                                self.zoom_at(rect, ZOOM_STEP_FACTOR, None);
                            }
                            if ui.small_button("Fit").on_hover_text("Fit scene (0 / double-click)").clicked() {
                                self.reset();
                            }
                        });
                    });
            });
        }
        out
    }
}
