//! Live painting of the Gallery overlay's top layer (mirrors, glides, rotation symbols) and the Wyckoff read-out of
//! the rotation centre under the pointer.

use crate::scene2d::overlays::{GalleryOverlay, centre_near};
use crate::theme::ThemeTokens;

use super::{Canvas2D, draw_arrow_mid_egui, draw_dashed_egui};

/// Pointer distance (px) within which a rotation centre is read out.
const HOVER_PX: f32 = 10.0;

impl Canvas2D {
    pub(super) fn paint_gallery_over(&self, painter: &egui::Painter, rect: egui::Rect, g: &GalleryOverlay, tokens: &ThemeTokens) {
        let to_screen = |p: [f32; 2]| self.world_to_screen_at(rect, p);
        for ln in &g.over.lines {
            let (p0, p1) = (to_screen(ln.a), to_screen(ln.b));
            let stroke = egui::Stroke::new(ln.width, tokens.remap_scene_ink(ln.color));
            if ln.dashed {
                draw_dashed_egui(painter, p0, p1, stroke, 6.5, 4.5);
            } else {
                painter.line_segment([p0, p1], stroke);
            }
            if ln.arrow_mid {
                draw_arrow_mid_egui(painter, p0, p1, stroke);
            }
        }
        for poly in &g.over.polygons {
            let pts: Vec<egui::Pos2> = poly.points.iter().copied().map(to_screen).collect();
            let stroke = poly.stroke.map_or(egui::Stroke::NONE, |(c, w)| egui::Stroke::new(w, tokens.remap_scene_ink(c)));
            painter.add(egui::Shape::convex_polygon(pts, tokens.scene_fill(poly.fill), stroke));
        }
        let Some(hp) = painter.ctx().pointer_hover_pos().filter(|p| rect.contains(*p)) else { return };
        let w = self.screen_to_world_at(rect, hp);
        let radius = f64::from(HOVER_PX / self.zoom.max(1e-3));
        if let Some(c) = centre_near(&g.centres, w, radius) {
            let text = format!("{}-fold rotation centre · Wyckoff {}\n({:.3}, {:.3})", c.order, c.wyckoff, c.pos[0], c.pos[1]);
            let galley = painter.layout_no_wrap(text, egui::FontId::proportional(11.5), tokens.text_primary);
            let at = hp + egui::vec2(14.0, 10.0);
            let bg = egui::Rect::from_min_size(at, galley.size() + egui::vec2(12.0, 8.0));
            painter.rect_filled(bg, 4.0, tokens.bg_popover);
            painter.rect_stroke(bg, 4.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
            painter.galley(at + egui::vec2(6.0, 4.0), galley, tokens.text_primary);
        }
    }
}
