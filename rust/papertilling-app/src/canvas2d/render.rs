use crate::scene2d::{PolyPrim, Scene2D, TextPrim};
use crate::theme::{ThemeTokens, contrast};

use super::hit::ZOOM_STEP_FACTOR;
use super::{
    Canvas2D, CanvasMarker, circuit, cuts, draw_arrow_mid_egui, draw_dashed_egui, draw_overlays,
    gauge_overlay, marker_colors, to_color32,
};

pub(crate) fn point_in_polygon(pt: [f32; 2], poly: &[[f32; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        let crosses = (yi > pt[1]) != (yj > pt[1]);
        if crosses {
            let x_int = (xj - xi) * (pt[1] - yi) / (yj - yi) + xi;
            if pt[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Label ink on the 2D canvas. In `Studio`, the ring numbers of a `CrystalPatch` take near-black or white ink,
/// whichever has the higher WCAG contrast against every face under the label composited on the canvas in draw
/// order (the compiled dark ink reached only about 2:1 there). Every other label keeps its compiled colour over a
/// filled polygon (fill alpha >= 80; e.g. the colour-coded gluing labels of a surface net) and is lifted by
/// `remap_scene_ink` on the bare canvas; `Paper` draws the compiled colour, as the export does.
pub(crate) fn scene_label_color(
    scene: &Scene2D,
    polygons: &[PolyPrim],
    lbl: &TextPrim,
    tokens: &ThemeTokens,
) -> egui::Color32 {
    let crystal = matches!(scene, Scene2D::CrystalPatch { .. });
    if crystal && !tokens.is_light {
        let c = tokens.bg_canvas_2d;
        let mut under = [c.r(), c.g(), c.b()];
        for p in polygons.iter().filter(|p| point_in_polygon(lbl.pos, &p.points)) {
            under = contrast::over(p.fill, egui::Color32::from_rgb(under[0], under[1], under[2]));
        }
        return contrast::best_ink(under);
    }
    let over_filled_poly = (!polygons.is_empty() && crystal)
        || polygons
            .iter()
            .any(|p| p.fill[3] >= 80 && point_in_polygon(lbl.pos, &p.points));
    if over_filled_poly {
        to_color32(lbl.color)
    } else {
        tokens.remap_scene_ink(lbl.color)
    }
}

impl Canvas2D {
    pub(super) fn render_canvas(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        hud_rect: Option<egui::Rect>,
        markers: &[CanvasMarker],
        preview_world: Option<[f64; 2]>,
        tokens: &ThemeTokens,
    ) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, tokens.bg_canvas_2d);

        let zoom = self.zoom;
        let to_screen = |p: [f32; 2]| -> egui::Pos2 { self.world_to_screen_at(rect, p) };

        for poly in &self.compiled.polygons {
            if poly.points.len() >= 3 {
                let pts: Vec<egui::Pos2> = poly.points.iter().copied().map(to_screen).collect();
                let fill = tokens.scene_fill(poly.fill);
                let stroke = poly
                    .stroke
                    .map_or(egui::Stroke::NONE, |(c, w)| egui::Stroke::new(w, tokens.remap_scene_ink(c)));
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
            let stroke = egui::Stroke::new(ln.width, tokens.remap_scene_ink(ln.color));
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
                painter.circle_filled(pos, r, tokens.remap_scene_ink(circ.fill));
            }
            if circ.stroke.1 > 0.0 && circ.stroke.0[3] > 0 {
                painter.circle_stroke(
                    pos,
                    r,
                    egui::Stroke::new(circ.stroke.1, tokens.remap_scene_ink(circ.stroke.0)),
                );
            }
        }

        for lbl in &self.compiled.labels {
            let col = scene_label_color(&self.scene, &self.compiled.polygons, lbl, tokens);
            painter.text(
                to_screen(lbl.pos),
                egui::Align2::CENTER_CENTER,
                &lbl.text,
                egui::FontId::proportional(lbl.size_px),
                col,
            );
        }

        if let Some(g) = &self.gallery_overlay {
            self.paint_gallery_over(&painter, rect, g, tokens);
        }

        if let Some(pw) = preview_world {
            let sp = to_screen([pw[0] as f32, pw[1] as f32]);
            painter.circle_filled(
                sp,
                8.5,
                egui::Color32::from_rgba_unmultiplied(
                    tokens.status_ok.r(),
                    tokens.status_ok.g(),
                    tokens.status_ok.b(),
                    55,
                ),
            );
            painter.circle_stroke(
                sp,
                9.5,
                egui::Stroke::new(1.8, tokens.status_ok),
            );
        }

        if !self.cut_rays.is_empty() {
            cuts::draw_cut_rays(&painter, self, rect, &self.cut_rays, tokens);
        }
        if let Some(ov) = &self.gauge_overlay {
            gauge_overlay::draw_gauge_overlays(&painter, self, rect, ov, tokens);
        }
        if !self.kinematic_overlay.glide_lines.is_empty()
            || !self.kinematic_overlay.locked_pins.is_empty()
            || self.kinematic_overlay.reaction.is_some()
        {
            cuts::draw_kinematic_overlay(&painter, self, rect, markers, &self.kinematic_overlay, tokens);
        }
        if let Some(circ) = &self.circuit_overlay {
            circuit::draw_circuit_overlay(&painter, self, rect, circ, tokens);
        }

        for m in markers {
            let sp = to_screen([m.pos[0] as f32, m.pos[1] as f32]);
            let (fill, halo) = marker_colors(m.kind, tokens);
            let is_dragged = self.active_drag_pin == Some(m.index);
            painter.circle_filled(sp, if is_dragged { 16.5 } else { 13.5 }, halo);
            if self.selected_marker == Some(m.index) {
                painter.circle_stroke(sp, 15.5, egui::Stroke::new(2.4, tokens.selection_ring));
            }
            painter.circle_filled(sp, 9.5, fill);
            painter.circle_stroke(sp, 9.5, egui::Stroke::new(1.8, tokens.bg_canvas_2d));
            painter.text(
                sp,
                egui::Align2::CENTER_CENTER,
                format!("{}", m.index + 1),
                egui::FontId::proportional(10.5),
                tokens.text_on_accent,
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
        draw_overlays(&painter, rect, overlay_title, &self.compiled.note, tokens);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, tokens.border_subtle),
            egui::StrokeKind::Inside,
        );
        if let Some(hud_rect) = hud_rect {
            ui.scope_builder(egui::UiBuilder::new().max_rect(hud_rect), |ui| {
                egui::Frame::NONE
                    .fill(tokens.bg_hud)
                    .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if ui.small_button("−").on_hover_text("Zoom out (-)").clicked() {
                                self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, None);
                            }
                            let pct = ((self.zoom / self.fit_zoom.max(1e-3)) * 100.0).round() as i32;
                            ui.colored_label(
                                tokens.text_primary,
                                egui::RichText::new(format!("{pct}%")).small(),
                            );
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
    }
}
