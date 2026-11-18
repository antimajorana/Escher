//! Legend of an exported 2D picture: the active overlay's colorbar (title, `LEGEND_STEPS` stepped colour rects,
//! min / max) or its categorical chips, in a bottom-left inset box in picture space. Both SVG and PNG keep the
//! Unicode labels (the PNG draws them with the bundled TTFs, `super::text_ttf`); a PNG at pixel scale `s`
//! multiplies every length of the box (inset, size, padding, bar, chips, font sizes, border) by `s`.

use std::fmt::Write as _;

use super::raster::Raster;
use super::{hex_rgb, opacity, xml_escape};
use crate::canvas2d::gauge_overlay::{CrystalOverlayData, OverlayColorbar};
use crate::canvas2d::overlay_legend::{LegendChip, colorbar_rgba};

/// Number of stepped rects of an exported colorbar gradient.
pub const LEGEND_STEPS: usize = 32;

const BOX_W: i32 = 252;
const PAD: i32 = 10;
const CHIP_ROW: i32 = 18;
const BOX_FILL: [u8; 4] = [255, 255, 255, 235];
const BOX_BORDER: [u8; 4] = [120, 130, 145, 255];
const INK: [u8; 4] = [38, 48, 64, 255];
const INK_SOFT: [u8; 4] = [70, 80, 96, 255];

/// The legend an export carries.
#[derive(Clone, Copy, Debug)]
pub enum ExportLegend<'a> {
    Colorbar(&'a OverlayColorbar),
    Categorical(&'a [LegendChip]),
}

impl<'a> ExportLegend<'a> {
    /// The legend drawn live for `ov`: its colorbar if any, else its chips if any.
    pub fn from_overlay(ov: &'a CrystalOverlayData) -> Option<Self> {
        match &ov.legend {
            Some(cb) => Some(Self::Colorbar(cb)),
            None if !ov.chips.is_empty() => Some(Self::Categorical(&ov.chips)),
            None => None,
        }
    }
}

/// `(x0, y0, x1, y1)` of the legend box, inset 16 px from the left and 12 px from the bottom of the picture.
pub(crate) fn legend_box(legend: &ExportLegend, height: u32) -> (i32, i32, i32, i32) {
    legend_box_scaled(legend, height, 1)
}

/// [`legend_box`] with every length multiplied by the pixel scale `s`.
pub(crate) fn legend_box_scaled(legend: &ExportLegend, height: u32, s: i32) -> (i32, i32, i32, i32) {
    let h = match legend {
        ExportLegend::Colorbar(_) => 62,
        ExportLegend::Categorical(chips) => 26 + CHIP_ROW * chips.len() as i32,
    };
    let y1 = height as i32 - 12 * s;
    (16 * s, y1 - h * s, (16 + BOX_W) * s, y1)
}

/// Bottom margin of the fitted picture: 36 px without a legend (the pre-legend layout), else the legend band
/// (box height + its 12 px inset + an 8 px gap) so the box sits below the picture rather than over it.
pub(crate) fn bottom_pad(legend: Option<&ExportLegend>, height: u32) -> f32 {
    bottom_pad_scaled(legend, height, 1)
}

/// [`bottom_pad`] with every length multiplied by the pixel scale `s`.
pub(crate) fn bottom_pad_scaled(legend: Option<&ExportLegend>, height: u32, s: i32) -> f32 {
    match legend {
        None => (36 * s) as f32,
        Some(l) => {
            let (_, y0, _, _) = legend_box_scaled(l, height, s);
            ((height as i32 - y0 + 8 * s) as f32).max((36 * s) as f32)
        }
    }
}

fn fmt_v(v: f64) -> String {
    if v.abs() >= 0.01 || v == 0.0 { format!("{v:.3}") } else { format!("{v:.1e}") }
}

fn title_of(cb: &OverlayColorbar) -> String {
    if cb.sqrt_scale && !cb.title.contains('√') { format!("{} (√-scale)", cb.title) } else { cb.title.clone() }
}

/// Bar geometry `(x0, y0, x1, y1)` inside a colorbar box drawn at pixel scale `s`.
fn bar_of((x0, y0, x1, _): (i32, i32, i32, i32), s: i32) -> (i32, i32, i32, i32) {
    (x0 + PAD * s, y0 + 24 * s, x1 - PAD * s, y0 + 38 * s)
}

/// Step `i` of the gradient: `[a, b)` in x and its colour (sampled at the step centre).
fn step(cb: &OverlayColorbar, bar: (i32, i32, i32, i32), i: usize) -> (i32, i32, [u8; 4]) {
    let w = bar.2 - bar.0;
    let a = bar.0 + (w * i as i32) / LEGEND_STEPS as i32;
    let b = bar.0 + (w * (i as i32 + 1)) / LEGEND_STEPS as i32;
    (a, b, colorbar_rgba(cb, (i as f64 + 0.5) / LEGEND_STEPS as f64))
}

/// Append the legend as one `<g id="legend">` group (to be placed just before `</svg>`).
pub(crate) fn svg_legend(out: &mut String, legend: &ExportLegend, height: u32) {
    let bx = legend_box(legend, height);
    let (x0, y0, x1, y1) = bx;
    let _ = writeln!(out, "<g id=\"legend\">");
    let _ = writeln!(
        out,
        "<rect x=\"{x0}\" y=\"{y0}\" width=\"{}\" height=\"{}\" rx=\"4\" fill=\"{}\" fill-opacity=\"{:.2}\" stroke=\"{}\" stroke-width=\"1\"/>",
        x1 - x0,
        y1 - y0,
        hex_rgb(BOX_FILL),
        opacity(BOX_FILL[3]),
        hex_rgb(BOX_BORDER)
    );
    let text = |out: &mut String, x: i32, y: i32, anchor: &str, size: f32, col: [u8; 4], s: &str| {
        let _ = writeln!(
            out,
            "<text x=\"{x}\" y=\"{y}\" text-anchor=\"{anchor}\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{size:.1}\" fill=\"{}\">{}</text>",
            hex_rgb(col),
            xml_escape(s)
        );
    };
    match legend {
        ExportLegend::Colorbar(cb) => {
            text(out, x0 + PAD, y0 + 12, "start", 12.0, INK, &title_of(cb));
            let bar = bar_of(bx, 1);
            for i in 0..LEGEND_STEPS {
                let (a, b, c) = step(cb, bar, i);
                let _ = writeln!(

                    out,
                    "<rect class=\"legend-step\" x=\"{a}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                    bar.1,
                    b - a,
                    bar.3 - bar.1,
                    hex_rgb(c)
                );
            }
            let _ = writeln!(
                out,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"0.8\"/>",
                bar.0,
                bar.1,
                bar.2 - bar.0,
                bar.3 - bar.1,
                hex_rgb(BOX_BORDER)
            );
            text(out, x0 + PAD, y1 - 11, "start", 11.0, INK_SOFT, &fmt_v(cb.min_val));
            text(out, x1 - PAD, y1 - 11, "end", 11.0, INK_SOFT, &fmt_v(cb.max_val));
        }
        ExportLegend::Categorical(chips) => {
            text(out, x0 + PAD, y0 + 12, "start", 12.0, INK, "Legend");
            for (k, chip) in chips.iter().enumerate() {
                let yc = y0 + 24 + CHIP_ROW * k as i32 + CHIP_ROW / 2;
                let _ = writeln!(
                    out,
                    "<rect class=\"legend-chip\" x=\"{}\" y=\"{}\" width=\"12\" height=\"12\" rx=\"2\" fill=\"{}\" fill-opacity=\"{:.2}\"/>",
                    x0 + PAD,
                    yc - 6,
                    hex_rgb(chip.rgba),
                    opacity(chip.rgba[3])
                );
                text(out, x0 + PAD + 20, yc, "start", 11.0, INK_SOFT, &chip.label);
            }
        }
    }
    let _ = writeln!(out, "</g>");
}

/// Draw the legend into the PNG raster at pixel scale `s`: the SVG legend's layout, text sizes and positions
/// (text vertically centred where the SVG uses `dominant-baseline="central"`).
pub(crate) fn raster_legend(img: &mut Raster, legend: &ExportLegend, height: u32, s: i32) {
    let bx = legend_box_scaled(legend, height, s);
    let (x0, y0, x1, y1) = bx;
    let k = s as f32;
    let at = |x: i32, y: i32| [x as f32, y as f32];
    img.rect_bw(bx, BOX_FILL, BOX_BORDER, s);
    match legend {
        ExportLegend::Colorbar(cb) => {
            img.text(at(x0 + PAD * s, y0 + 12 * s), &title_of(cb), 12.0 * k, INK, (0.0, 0.5));
            let bar = bar_of(bx, s);
            for i in 0..LEGEND_STEPS {
                let (a, b, c) = step(cb, bar, i);
                img.rect((a, bar.1, b, bar.3), c, c);
            }
            img.text(at(x0 + PAD * s, y1 - 11 * s), &fmt_v(cb.min_val), 11.0 * k, INK_SOFT, (0.0, 0.5));
            img.text(at(x1 - PAD * s, y1 - 11 * s), &fmt_v(cb.max_val), 11.0 * k, INK_SOFT, (1.0, 0.5));
        }
        ExportLegend::Categorical(chips) => {
            img.text(at(x0 + PAD * s, y0 + 12 * s), "Legend", 12.0 * k, INK, (0.0, 0.5));
            for (n, chip) in chips.iter().enumerate() {
                let yc = y0 + (24 + CHIP_ROW * n as i32 + CHIP_ROW / 2) * s;
                let c = [chip.rgba[0], chip.rgba[1], chip.rgba[2], 255];
                img.rect((x0 + PAD * s, yc - 6 * s, x0 + (PAD + 12) * s, yc + 6 * s), c, c);
                img.text(at(x0 + (PAD + 20) * s, yc), &chip.label, 11.0 * k, INK_SOFT, (0.0, 0.5));
            }
        }
    }
}

#[cfg(test)]
#[path = "legend_tests.rs"]
mod tests;
