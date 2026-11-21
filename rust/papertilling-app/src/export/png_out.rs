//! PNG side of the 2D export: the composition of the SVG drawn into a [`Raster`] at an integer pixel scale `s`
//! (picture side `PNG_SIZE * s`; margins, stroke widths, radii, marker sizes, font sizes and the legend box all
//! multiplied by `s`), so a 2x PNG is the 1x picture at twice the resolution, not a larger view.

use super::raster::Raster;
use super::text_ttf::{Weight, draw_text_weight};
use super::{BACKGROUND, ExportLegend, NOTE_BORDER, NOTE_FILL, NOTE_TEXT, PNG_SIZE, TITLE, Transform, legend, marker_colors};
use crate::canvas2d::CanvasMarker;
#[cfg(test)]
use crate::canvas2d::Scene2D;
use crate::scene2d::ScenePrimitives;
#[cfg(test)]
use crate::scene2d::compile_scene;

/// Pixel size of an exported PNG: `w` x `h` pixels with every composition length multiplied by `scale`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PngSize {
    pub w: u32,
    pub h: u32,
    pub scale: u32,
}

impl PngSize {
    /// The square export picture at `scale` (1, 2, 4, ...): `PNG_SIZE * scale` a side; `Err` for scale 0 or an
    /// overflowing side (sides above `Raster` limits are refused later, by `Raster::new`).
    pub(crate) fn export(scale: u32) -> Result<Self, String> {
        let side = PNG_SIZE.checked_mul(scale).filter(|_| scale > 0).ok_or_else(|| format!("PNG scale {scale}x out of range"))?;
        Ok(Self { w: side, h: side, scale })
    }

    /// A `w` x `h` picture at scale 1 (the composition of the pre-scale exports).
    #[cfg(test)]
    pub(crate) const fn plain
(w: u32, h: u32) -> Self {
        Self { w, h, scale: 1 }
    }
}

#[cfg(test)]
pub fn encode_scene_png(scene: &Scene2D, width: u32, height: u32) -> Result<Vec<u8>, String> {
    encode_scene_png_with_markers(scene, &[], width, height)
}

/// PNG of the scene (`width` x `height`, RGBA 8-bit); `Err` for a size out of range (no panic, no huge buffer).
#[cfg(test)]
pub fn encode_scene_png_with_markers(scene: &Scene2D, markers: &[CanvasMarker], width: u32, height: u32) -> Result<Vec<u8>, String> {
    encode_scene_png_with_title(scene, markers, None, width, height)
}

#[cfg(test)]
pub fn encode_scene_png_with_title(
    scene: &Scene2D,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    encode_prims_png(&compile_scene(scene), markers, title_override, width, height)
}

#[cfg(test)]
pub(crate) fn encode_prims_png(
    base_prims: &ScenePrimitives,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    encode_prims_png_with_legend(base_prims, markers, title_override, None, width, height)
}

#[cfg(test)]
/// [`encode_prims_png`] with an optional legend box drawn last (none: the same pixels as without).
pub(crate) fn encode_prims_png_with_legend(
    base_prims: &ScenePrimitives,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: Option<&ExportLegend>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    encode_layers_png(base_prims, &[], markers, title_override, legend, PngSize::plain(width, height))
}

/// PNG of `base_prims` with the `tops` layers drawn over it in order (e.g. Gallery symmetry elements over the sites).
pub(crate) fn encode_layers_png(
    base_prims: &ScenePrimitives,
    tops: &[ScenePrimitives],
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: Option<&ExportLegend>,
    size: PngSize,
) -> Result<Vec<u8>, String> {
    let PngSize { w: width, h: height, scale } = size;
    let mut img = Raster::new(width, height, BACKGROUND)?;
    let (s, k) = (i32::try_from(scale).map_err(|e| e.to_string())?, scale as f32);
    let mut prims = base_prims.clone();
    if let Some(t) = title_override.filter(|t| !t.is_empty()) {
        prims.title = t.to_string();
    }
    let tf = Transform::new(&prims, width, height, legend::bottom_pad_scaled(legend, height, s), k);

    raster_layer(&mut img, &prims, &tf, k);
    for t in tops {
        raster_layer(&mut img, t, &tf, k);
    }

    for m in markers {
        let Some(p) = tf.map64(m.pos) else { continue };
        let c = marker_colors(m.kind);
        img.circle(p, 13.5 * k, [c[0], c[1], c[2], 68], ([0, 0, 0, 0], 0.0));
        img.circle(p, 9.5 * k, [c[0], c[1], c[2], 255], ([255, 255, 255, 255], 1.8 * k));
        img.text(p, &(m.index + 1).to_string(), 10.5 * k, [255, 255, 255, 255], (0.5, 0.5));
        img.text([p[0], p[1] - 15.5 * k], &m.label, 10.5 * k, [c[0], c[1], c[2], 255], (0.5, 0.5));
    }

    if !prims.title.is_empty() {
        // semibold, as the SVG title's font-weight="bold"
        draw_text_weight(&mut img, [16.0 * k, 10.0 * k], &prims.title, 14.0 * k, TITLE, (0.0, 0.0), Weight::SemiBold);
    }
    if !prims.note.is_empty() {
        let y0 = if prims.title.is_empty() { 10 } else { 30 };
        img.rect_bw((16 * s, y0 * s, width as i32 - 16 * s, (y0 + 24) * s), NOTE_FILL, NOTE_BORDER, s);
        img.text([26.0 * k, (y0 as f32 + 12.0) * k], &prims.note, 12.0 * k, NOTE_TEXT, (0.0, 0.5));
    }
    if let Some(l) = legend {
        legend::raster_legend(&mut img, l, height, s);
    }

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&img.rgba).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

/// Pixel width of a stroke of nominal width `w` at scale `k`: `max(w, 1.1) * k`. `Raster::segment` draws no
/// stroke thinner than 1.1 px (half-width floor 0.55); applying that floor before scaling keeps strokes under
/// 1.1 (cell outlines 0.8, Voronoi edges 0.9, ...) at the same width relative to the picture at every scale.
/// At k = 1 this is exactly the raster's own floor, so 1x pictures are unchanged.
fn stroke_px(w: f32, k: f32) -> f32 {
    w.max(1.1) * k
}

/// Polygons, lines, circles and labels of one layer, in that order; every pixel length multiplied by `k`.
fn raster_layer(img: &mut Raster, prims: &ScenePrimitives, tf: &Transform, k: f32) {
    for poly in &prims.polygons {
        let pts: Option<Vec<[f32; 2]>> = poly.points.iter().map(|&p| tf.map(p)).collect();
        let Some(pts) = pts.filter(|v| v.len() >= 3) else { continue };
        img.fill_polygon(&pts, poly.fill);
        if let Some((scol, sw)) = poly.stroke {
            for m in 0..pts.len() {
                img.segment(pts[m], pts[(m + 1) % pts.len()], scol, stroke_px(sw, k));
            }
        }
    }

    for ln in &prims.lines {
        let (Some(p0), Some(p1)) = (tf.map(ln.a), tf.map(ln.b)) else { continue };
        if ln.dashed {
            img.dashed(p0, p1, ln.color, stroke_px(ln.width, k), (8.0 * k, 5.5 * k));
        } else {
            img.segment(p0, p1, ln.color, stroke_px(ln.width * 1.15, k));
        }
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = dx.hypot(dy);
        if ln.arrow_mid && len >= 10.0 * k {
            let (ux, uy) = (dx / len, dy / len);
            let tip = [p0[0] + dx * 0.56, p0[1] + dy * 0.56];
            let w = 7.0 * k;
            let sw = stroke_px(ln.width * 1.15, k);
            img.segment(tip, [tip[0] - ux * w - uy * w * 0.62, tip[1] - uy * w + ux * w * 0.62], ln.color, sw);
            img.segment(tip, [tip[0] - ux * w + uy * w * 0.62, tip[1] - uy * w - ux * w * 0.62], ln.color, sw);
        }
    }

    for circ in &prims.circles {
        let Some(c) = tf.map(circ.center) else { continue };
        let r = (circ.radius_world * tf.scale).clamp(circ.min_px * k, circ.max_px * 1.4 * k);
        img.circle(c, r, circ.fill, (circ.stroke.0, circ.stroke.1 * k));
    }

    for lbl in &prims.labels {
        if let Some(p) = tf.map(lbl.pos) {
            img.text(p, &lbl.text, lbl.size_px * k, lbl.color, (0.5, 0.5));
        }
    }
}

#[cfg(test)]
#[path = "scale_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "element_scale_tests.rs"]
mod element_tests;
