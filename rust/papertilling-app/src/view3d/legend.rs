//! Colorbar HUD of the 3D viewport: the scalar colour mode's title, range and colour map (the same function and
//! normalisation the mesh builders colour with), painted bottom-right directly above the zoom HUD, clear of the
//! caption (top-left) and the input hint (bottom-left). Paint only: it takes no input, so camera gating is unchanged.

use crate::mesh3d::sheet_colors::{diverging_rgba, height_rgba, rgba_u8, sequential_heat_rgba, strain_rgba};
use crate::theme::ThemeTokens;

/// Colour map of a 3D legend, one per map function the mesh builders use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Legend3dMap {
    /// `sequential_heat_rgba(v, max)` over `[0, max]`.
    Sequential,
    /// `sequential_heat_rgba(sqrt(v / max), 1)` over `[0, max]` (perceptual energy scale).
    SequentialSqrt,
    /// `diverging_rgba(v, max)` over `[-max, max]`.
    Diverging,
    /// `strain_rgba(v)` (saturates at |v| = 0.08) over `[-0.08, 0.08]`.
    Strain,
    /// `height_rgba(v, max)` over `[-max, max]`.
    Height,
}

/// Scalar colour legend of the 3D viewport.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorLegend3d {
    pub title: String,
    pub min_val: f64,
    pub max_val: f64,
    pub map: Legend3dMap,
}

impl ColorLegend3d {
    /// Colour the mesh gives a site of value `v` (opaque).
    pub fn rgba_of_value(&self, v: f64) -> [u8; 4] {
        let c = match self.map {
            Legend3dMap::Sequential => sequential_heat_rgba(v.max(0.0), self.max_val, 1.0),
            Legend3dMap::SequentialSqrt => {
                sequential_heat_rgba((v.max(0.0) / self.max_val.max(1e-300)).clamp(0.0, 1.0).sqrt(), 1.0, 1.0)
            }
            Legend3dMap::Diverging => diverging_rgba(v, self.max_val.abs().max(self.min_val.abs()), 1.0),
            Legend3dMap::Strain => strain_rgba(v, 1.0),
            Legend3dMap::Height => height_rgba(v, self.max_val.abs().max(self.min_val.abs()), 1.0),
        };
        rgba_u8(c)
    }

    /// Value at bar position `t` in [0, 1] (quadratic along a √-scaled bar, so its colour is linear in `t`).
    pub fn value_at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        let s = if self.map == Legend3dMap::SequentialSqrt { t * t } else { t };
        self.min_val + s * (self.max_val - self.min_val)
    }

    /// Bar colour at `t` in [0, 1]: `t = 0` is the colour at `min_val`, `t = 1` the colour at `max_val`.
    pub fn sample(&self, t: f64) -> [u8; 4] {
        self.rgba_of_value(self.value_at(t))
    }
}

/// Zoom HUD of the 3D viewport (`None` when the viewport is too small for it), as `View3dState::show` lays it out.
pub(crate) fn zoom_hud_rect_3d(rect: egui::Rect) -> Option<egui::Rect> {
    (rect.width() >= 190.0 && rect.height() >= 80.0).then(|| {
        egui::Rect::from_min_max(rect.right_bottom() - egui::vec2(164.0, 30.0), rect.right_bottom() - egui::vec2(6.0, 6.0))
    })
}

/// Bottom-right legend box stacked 6 px above the zoom HUD; `None` below 200 x 130.
pub(crate) fn legend_rect_3d(rect: egui::Rect) -> Option<egui::Rect> {
    if rect.width() < 200.0 || rect.height() < 130.0 {
        return None;
    }
    let w = 174.0_f32.min(rect.width() - 16.0);
    Some(egui::Rect::from_min_max(
        rect.right_bottom() - egui::vec2(6.0 + w, 84.0),
        rect.right_bottom() - egui::vec2(6.0, 36.0),
    ))
}

pub(super) fn fmt_v(v: f64) -> String {
    if v.abs() >= 0.01 || v == 0.0 { format!("{v:.3}") } else { format!("{v:.1e}") }
}

/// Title of the legend box (a √-scaled bar says so).
pub(super) fn legend_title(leg: &ColorLegend3d) -> String {
    if leg.map == Legend3dMap::SequentialSqrt { format!("{} (√-scale)", leg.title) } else { leg.title.clone() }
}

pub(crate) fn draw_legend_3d(painter: &egui::Painter, box_r: egui::Rect, leg: &ColorLegend3d, tokens: &ThemeTokens) {
    painter.rect_filled(box_r, 5.0, tokens.bg_hud);
    painter.rect_stroke(box_r, 5.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
    let title = legend_title(leg);
    painter.text(
        box_r.left_top() + egui::vec2(7.0, 4.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(9.5),
        tokens.text_primary,
    );
    let bar = egui::Rect::from_min_max(box_r.left_top() + egui::vec2(8.0, 19.0), box_r.right_top() + egui::vec2(-8.0, 27.0));
    let steps = 24usize;
    let dx = bar.width() / steps as f32;
    for i in 0..steps {
        let c = leg.sample((i as f64 + 0.5) / steps as f64);
        let seg = egui::Rect::from_min_max(
            egui::pos2(bar.left() + i as f32 * dx, bar.top()),
            egui::pos2(bar.left() + (i + 1) as f32 * dx + 0.5, bar.bottom()),
        );
        painter.rect_filled(seg, 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
    }
    painter.rect_stroke(bar, 1.0, egui::Stroke::new(0.8, tokens.border_subtle), egui::StrokeKind::Inside);
    let font = egui::FontId::proportional(9.0);
    painter.text(box_r.left_bottom() + egui::vec2(7.0, -3.0), egui::Align2::LEFT_BOTTOM, fmt_v(leg.min_val), font.clone(), tokens.text_secondary);
    painter.text(box_r.right_bottom() + egui::vec2(-7.0, -3.0), egui::Align2::RIGHT_BOTTOM, fmt_v(leg.max_val), font, tokens.text_secondary);
}
