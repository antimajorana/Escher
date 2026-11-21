//! Legends of the 2D crystal overlays: spectral colorbars (LDOS sequential, δρ diverging) built from the same
//! `vmax` that paints the sites, and categorical chip legends for plaquette-flux badges and Volterra seams whose
//! colours come from the one lookup the badges and seams are painted with (live: theme tokens; export: RGBA).

use crate::mesh3d::sheet_colors::{diverging_rgba, rgba_u8, sequential_heat_rgba, strain_rgba};
use crate::theme::ThemeTokens;

use super::gauge_overlay::{CrystalOverlayData, OverlayColorbar, PlaquetteBadge, SeamConnectionMark, color_sites_scalar};
use super::overlay_scales::STRAIN_SATURATION;

/// A painted category of a categorical overlay (badge or seam class).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LegendCategory {
    FluxImproper,
    FluxRotation,
    FluxTranslation,
    SeamImproper,
    SeamProper,
    SeamWall,
}

impl LegendCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::FluxImproper => "Improper flux (det=-1)",
            Self::FluxRotation => "Rotation flux Ω (curvature)",
            Self::FluxTranslation => "Torsion α / τ-shift",
            Self::SeamImproper => "Seam g_e with det=-1",
            Self::SeamProper => "Seam g_e with det=+1",
            Self::SeamWall => "Wall bond (no g_e)",
        }
    }

    /// Colour of this category in SVG / PNG exports (the constants the export primitives use).
    pub fn export_rgba(self) -> [u8; 4] {
        match self {
            Self::FluxImproper => [142, 36, 170, 235],
            Self::FluxRotation => [198, 40, 40, 235],
            Self::FluxTranslation => [230, 81, 0, 235],
            Self::SeamImproper => [186, 45, 118, 240],
            Self::SeamProper => [0, 131, 143, 235],
            Self::SeamWall => [96, 110, 128, 235],
        }
    }

    /// Colour of this category on the live, themed canvas.
    pub fn live_color(self, tokens: &ThemeTokens) -> egui::Color32 {
        match self {
            Self::FluxImproper => tokens.cut_ray_line,
            Self::FluxRotation => tokens.status_err,
            Self::FluxTranslation => tokens.status_warn,
            Self::SeamImproper => {
                if tokens.is_light { egui::Color32::from_rgb(186, 45, 118) } else { egui::Color32::from_rgb(244, 114, 182) }
            }
            Self::SeamProper => tokens.cut_insertion,
            Self::SeamWall => tokens.text_muted,
        }
    }
}

/// One entry of a categorical legend: label, export colour, and the category that resolves its live colour.
#[derive(Clone, Debug, PartialEq)]
pub struct LegendChip {
    pub label: String,
    pub rgba: [u8; 4],
    pub category: LegendCategory,
}

impl LegendChip {
    pub fn new(category: LegendCategory) -> Self {
        Self { label: category.label().to_owned(), rgba: category.export_rgba(), category }
    }
}

/// Category a plaquette badge is painted with (precedence: improper, rotation, other).
pub fn badge_category(b: &PlaquetteBadge) -> LegendCategory {
    if b.improper {
        LegendCategory::FluxImproper
    } else if b.is_rotation {
        LegendCategory::FluxRotation
    } else {
        LegendCategory::FluxTranslation
    }
}

pub fn seam_category(s: &SeamConnectionMark) -> LegendCategory {
    if !s.has_ge {
        LegendCategory::SeamWall
    } else if s.det_neg {
        LegendCategory::SeamImproper
    } else {
        LegendCategory::SeamProper
    }
}

fn chips_of(mut cats: Vec<LegendCategory>) -> Vec<LegendChip> {
    cats.sort();
    cats.dedup();
    cats.into_iter().map(LegendChip::new).collect()
}

/// One chip per distinct badge category actually present.
pub fn flux_legend_chips(badges: &[PlaquetteBadge]) -> Vec<LegendChip> {
    chips_of(badges.iter().map(badge_category).collect())
}

/// One chip per distinct seam category actually present.
pub fn seam_legend_chips(seams: &[SeamConnectionMark]) -> Vec<LegendChip> {
    chips_of(seams.iter().map(seam_category).collect())
}

/// Colour of the colorbar at `t` in [0, 1] (`t = 0` at `min_val`, `t = 1` at `max_val`): the same map and
/// normalisation as `color_sites_scalar`, so the bar ends equal the colours of the extreme sites. A `strain_map`
/// bar runs `strain_rgba` over its saturation range `[-0.08, 0.08]` (see `overlay_scales::bond_strain_overlay`).
pub fn colorbar_rgba(leg: &OverlayColorbar, t: f64) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    if leg.strain_map {
        rgba_u8(strain_rgba((2.0 * t - 1.0) * STRAIN_SATURATION, 1.0))
    } else if leg.diverging {
        rgba_u8(diverging_rgba(2.0 * t - 1.0, 1.0, 1.0))
    } else {
        rgba_u8(sequential_heat_rgba(t, 1.0, 1.0))
    }
}

/// Spectral overlay (`LDOS` sequential over `[0, vmax]`, `δρ` diverging over `[-vmax, vmax]`) with the colorbar
/// taken from the same `vmax = max|v|` (floored at 1e-6) that `color_sites_scalar` paints the sites with.
pub fn spectral_overlay_data(vals: &[f64], diverging: bool) -> CrystalOverlayData {
    let vmax = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-6);
    let legend = OverlayColorbar {
        title: String::from(if diverging { "Spectral charge δρ(μ)" } else { "LDOS |ψ|²" }),
        min_val: if diverging { -vmax } else { 0.0 },
        max_val: vmax,
        diverging,
        sqrt_scale: false,
        strain_map: false,
    };
    CrystalOverlayData { site_rgba: Some(color_sites_scalar(vals, diverging)), legend: Some(legend), ..Default::default() }
}

const CHIP_ROW_H: f32 = 14.0;

/// Bottom-left box of a categorical legend with `n` chips, under the same anchoring rule as the colorbar
/// (`colorbar_legend_rect`): stacked above the zoom HUD's row when the canvas is narrow; grows upward.
pub(crate) fn categorical_legend_rect(rect: egui::Rect, n: usize) -> Option<egui::Rect> {
    if n == 0 {
        return None;
    }
    let stack_above_hud = rect.width() < 360.0;
    let bottom_gap = if stack_above_hud { 40.0 } else { 8.0 };
    let h = 20.0 + CHIP_ROW_H * n as f32;
    if rect.width() < 220.0 || rect.height() < h + bottom_gap + 60.0 {
        return None;
    }
    Some(egui::Rect::from_min_max(
        rect.left_bottom() + egui::vec2(8.0, -bottom_gap - h),
        rect.left_bottom() + egui::vec2(182.0, -bottom_gap),
    ))
}

pub(crate) fn draw_categorical_legend(painter: &egui::Painter, box_r: egui::Rect, chips: &[LegendChip], tokens: &ThemeTokens) {
    painter.rect_filled(box_r, 5.0, tokens.bg_hud);
    painter.rect_stroke(box_r, 5.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
    painter.text(
        box_r.left_top() + egui::vec2(7.0, 4.0),
        egui::Align2::LEFT_TOP,
        "Legend",
        egui::FontId::proportional(9.5),
        tokens.text_primary,
    );
    for (k, chip) in chips.iter().enumerate() {
        let y = box_r.top() + 18.0 + CHIP_ROW_H * k as f32 + 0.5 * CHIP_ROW_H;
        let col = chip.category.live_color(tokens);
        let sw = egui::Rect::from_center_size(egui::pos2(box_r.left() + 14.0, y), egui::vec2(10.0, 10.0));
        painter.rect_filled(sw, 2.0, col);
        painter.text(
            egui::pos2(box_r.left() + 24.0, y),
            egui::Align2::LEFT_CENTER,
            &chip.label,
            egui::FontId::proportional(9.5),
            tokens.text_secondary,
        );
    }
}

/// The live legend box of `ov` in canvas `rect` (colorbar first, else chips), as drawn by `draw_gauge_overlays`.
pub(crate) fn overlay_legend_rect(rect: egui::Rect, ov: &CrystalOverlayData) -> Option<egui::Rect> {
    if ov.legend.is_some() {
        super::gauge_overlay::colorbar_legend_rect(rect)
    } else {
        categorical_legend_rect(rect, ov.chips.len())
    }
}

#[cfg(test)]
#[path = "overlay_legend_tests.rs"]
mod tests;
