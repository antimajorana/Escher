use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::canvas2d::{CanvasMarker, MarkerKind, Scene2D};
use crate::scene2d::{ScenePrimitives, compile_scene};

mod kern;
mod legend;
mod png_out;
#[path = "../export_raster.rs"]
pub(crate) mod raster;
pub(crate) mod text_ttf;

pub use legend::ExportLegend;
#[cfg(test)]
use png_out::{encode_prims_png, encode_prims_png_with_legend, encode_scene_png};
use png_out::{PngSize, encode_layers_png};
#[cfg(test)]
use raster::Raster;

const SVG_SIZE: u32 = 960;
const PNG_SIZE: u32 = 1024;
const BACKGROUND: [u8; 4] = [248, 249, 252, 255];
const TITLE: [u8; 4] = [38, 48, 64, 255];
const NOTE_FILL: [u8; 4] = [255, 244, 214, 238];
const NOTE_BORDER: [u8; 4] = [218, 160, 48, 255];
const NOTE_TEXT: [u8; 4] = [124, 68, 10, 255];

/// Export `scene` to `path`: `.svg` (any case) writes SVG, `.png` (any case) writes PNG; no extension writes
/// SVG to `path` + `.svg` (the dialog's default type); any other extension is refused with a message.
#[cfg(test)]
pub fn export_scene_to_path(scene: &Scene2D, path: &Path) -> Result<String, String> {
    export_scene_with_markers(scene, &[], path)
}

/// As [`export_scene_to_path`], with the Combine tab's defect pins drawn as on screen.
#[cfg(test)]
pub fn export_scene_with_markers(scene: &Scene2D, markers: &[CanvasMarker], path: &Path) -> Result<String, String> {
    export_scene_full(scene, markers, None, path, 1)
}

/// Export `scene` with optional markers and an optional custom title override (used by Gallery and Surfaces
/// so custom lattice names appear in exported SVG and PNG headers). A PNG is drawn at `png_scale` (1, 2, 4):
/// `PNG_SIZE * png_scale` pixels a side, the same composition as at 1x; the SVG does not depend on it.
pub fn export_scene_full(
    scene: &Scene2D,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    path: &Path,
    png_scale: u32,
) -> Result<String, String> {
    let prims = compile_scene(scene);
    export_prims(&prims, &[], markers, title_override, None, (path, png_scale))
}

/// As [`export_scene_full`] for compiled primitives, with an optional legend box and `top` drawn over the base layer.
pub(crate) fn export_scene_layers(
    prims: &ScenePrimitives,
    tops: &[ScenePrimitives],
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: Option<&ExportLegend>,
    (path, png_scale): (&Path, u32),
) -> Result<String, String> {
    export_prims(prims, tops, markers, title_override, legend, (path, png_scale))
}

fn export_prims(
    prims: &ScenePrimitives,
    tops: &[ScenePrimitives],
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: Option<&ExportLegend>,
    (path, png_scale): (&Path, u32),
) -> Result<String, String> {
    let ext = path.extension().map(|s| s.to_string_lossy().to_ascii_lowercase());
    let (target, is_svg): (PathBuf, bool) = match ext.as_deref() {
        Some("svg") => (path.to_path_buf(), true),
        Some("png") => (path.to_path_buf(), false),
        None => (path.with_extension("svg"), true),
        Some(other) => return Err(format!("cannot export a .{other} file: choose a .svg or .png file name")),
    };
    let name = target.file_name().map_or_else(|| target.display().to_string(), |s| s.to_string_lossy().into_owned());
    if is_svg {
        let svg = svg_with_legend(prims, tops, markers, title_override, legend, SVG_SIZE, SVG_SIZE);
        std::fs::write(&target, svg).map_err(|e| format!("{}: {e}", target.display()))?;
        Ok(format!("Exported SVG ({name})"))
    } else {
        let size = PngSize::export(png_scale)?;
        let png_bytes = encode_layers_png(prims, tops, markers, title_override, legend, size)?;
        std::fs::write(&target, png_bytes).map_err(|e| format!("{}: {e}", target.display()))?;
        Ok(format!("Exported {}x{} PNG at {png_scale}x ({name})", size.w, size.h))
    }
}

/// World -> picture map, the same fit as the canvas (title / note band on top, `bottom_pad` px below: 36, or
/// the legend band when a legend is exported so the legend box never covers the picture). The fixed margins
/// are multiplied by the pixel scale `k` (1 for SVG; 2 / 4 for a 2x / 4x PNG).
struct Transform {
    wc: [f32; 2],
    sc: [f32; 2],
    scale: f32,
}

impl Transform {
    fn new(prims: &ScenePrimitives, width: u32, height: u32, bottom_pad: f32, k: f32) -> Self {
        let (bmin, bmax) = prims.bbox;
        let wc = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        let span_x = (bmax[0] - bmin[0]).max(0.5);
        let span_y = (bmax[1] - bmin[1]).max(0.5);
        let top_pad = k * if !prims.note.is_empty() { 76.0 } else if !prims.title.is_empty() { 48.0 } else { 36.0 };
        let avail_w = (width as f32 - 64.0 * k).max(32.0 * k);
        let avail_h = (height as f32 - top_pad - bottom_pad).max(32.0 * k);
        let scale = (avail_w / span_x).min(avail_h / span_y);
        Self { wc, sc: [width as f32 * 0.5, top_pad + avail_h * 0.5], scale }
    }

    /// Picture coordinates; None for a non-finite point (nothing is drawn for it).
    fn map(&self, p: [f32; 2]) -> Option<[f32; 2]> {
        let q = [self.sc[0] + (p[0] - self.wc[0]) * self.scale, self.sc[1] - (p[1] - self.wc[1]) * self.scale];
        (q[0].is_finite() && q[1].is_finite()).then_some(q)
    }

    fn map64(&self, p: [f64; 2]) -> Option<[f32; 2]> {
        self.map([p[0] as f32, p[1] as f32])
    }
}

pub(crate) fn marker_colors(kind: MarkerKind) -> [u8; 3] {
    match kind {
        MarkerKind::DisclinationPlus => [211, 47, 47],
        MarkerKind::DisclinationMinus => [25, 118, 210],
        MarkerKind::Dislocation => [237, 108, 2],
    }
}

fn hex_rgb(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn opacity(a: u8) -> f32 {
    f32::from(a) / 255.0
}

fn xml_escape(s: &str) -> String {
    s.chars()
        .filter(|&c| c == '\t' || c == '\n' || c >= ' ')
        .collect::<String>()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
pub fn render_scene_svg(scene: &Scene2D, width: u32, height: u32) -> String {
    render_scene_svg_with_markers(scene, &[], width, height)
}

#[cfg(test)]
pub fn render_scene_svg_with_markers(scene: &Scene2D, markers: &[CanvasMarker], width: u32, height: u32) -> String {
    render_scene_svg_with_title(scene, markers, None, width, height)
}

#[cfg(test)]
pub fn render_scene_svg_with_title(
    scene: &Scene2D,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    width: u32,
    height: u32,
) -> String {
    render_prims_svg(&compile_scene(scene), markers, title_override, width, height)
}

#[cfg(test)]
pub(crate) fn render_prims_svg(
    base_prims: &ScenePrimitives,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    width: u32,
    height: u32,
) -> String {
    svg_with_legend(base_prims, &[], markers, title_override, None, width, height)
}

#[cfg(test)]
/// [`render_prims_svg`] with `legend` as a final `<g id="legend">` group, the picture fitted above it.
pub(crate) fn render_prims_svg_with_legend(
    base_prims: &ScenePrimitives,
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: &ExportLegend,
    width: u32,
    height: u32,
) -> String {
    svg_with_legend(base_prims, &[], markers, title_override, Some(legend), width, height)
}

fn svg_with_legend(
    base_prims: &ScenePrimitives,
    tops: &[ScenePrimitives],
    markers: &[CanvasMarker],
    title_override: Option<&str>,
    legend: Option<&ExportLegend>,
    width: u32,
    height: u32,
) -> String {
    let mut prims = base_prims.clone();
    if let Some(t) = title_override.filter(|t| !t.is_empty()) {
        prims.title = t.to_string();
    }
    let tf = Transform::new(&prims, width, height, legend::bottom_pad(legend, height), 1.0);

    let mut out = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n<rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\n",
        hex_rgb(BACKGROUND)
    );

    svg_layer(&mut out, &prims, &tf);
    for t in tops {
        svg_layer(&mut out, t, &tf);
    }

    for m in markers {
        let Some([x, y]) = tf.map64(m.pos) else { continue };
        let c = marker_colors(m.kind);
        let col = hex_rgb([c[0], c[1], c[2], 255]);
        let _ = writeln!(out, "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"13.5\" fill=\"{col}\" fill-opacity=\"0.27\"/>");
        let _ = writeln!(out, "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"9.5\" fill=\"{col}\" stroke=\"#ffffff\" stroke-width=\"1.8\"/>");
        let _ = writeln!(
            out,
            "<text x=\"{x:.2}\" y=\"{y:.2}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"10.5\" fill=\"#ffffff\">{}</text>",
            m.index + 1
        );
        if !m.label.is_empty() {
            let _ = writeln!(
                out,
                "<text x=\"{x:.2}\" y=\"{:.2}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"10.5\" fill=\"{col}\">{}</text>",
                y - 15.5,
                xml_escape(&m.label)
            );
        }
    }

    if !prims.title.is_empty() {
        let _ = writeln!(
            out,
            "<text x=\"16\" y=\"22\" font-family=\"sans-serif\" font-size=\"14\" font-weight=\"bold\" fill=\"{}\">{}</text>",
            hex_rgb(TITLE),
            xml_escape(&prims.title)
        );
    }
    if !prims.note.is_empty() {
        let y_box = if prims.title.is_empty() { 10 } else { 32 };
        let _ = writeln!(
            out,
            "<rect x=\"16\" y=\"{y_box}\" width=\"{}\" height=\"26\" rx=\"4\" fill=\"{}\" fill-opacity=\"{:.2}\" stroke=\"{}\" stroke-width=\"1\"/>\n<text x=\"26\" y=\"{}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"{}\">{}</text>",
            width.saturating_sub(32),
            hex_rgb(NOTE_FILL),
            opacity(NOTE_FILL[3]),
            hex_rgb(NOTE_BORDER),
            y_box + 17,
            hex_rgb(NOTE_TEXT),
            xml_escape(&prims.note)
        );
    }

    if let Some(l) = legend {
        legend::svg_legend(&mut out, l, height);
    }
    out.push_str("</svg>\n");
    out
}

/// Polygons, lines, circles and labels of one layer, in that order.
fn svg_layer(out: &mut String, prims: &ScenePrimitives, tf: &Transform) {
    for poly in &prims.polygons {
        let pts: Option<Vec<[f32; 2]>> = poly.points.iter().map(|&p| tf.map(p)).collect();
        let Some(pts) = pts.filter(|v| v.len() >= 3) else { continue };
        let pts_str = pts.iter().map(|p| format!("{:.2},{:.2}", p[0], p[1])).collect::<Vec<_>>().join(" ");
        let stroke_attr = poly.stroke.map_or_else(
            || String::from(" stroke=\"none\""),
            |(scol, sw)| format!(" stroke=\"{}\" stroke-opacity=\"{:.2}\" stroke-width=\"{sw:.2}\"", hex_rgb(scol), opacity(scol[3])),
        );
        let _ = writeln!(out, "<polygon points=\"{pts_str}\" fill=\"{}\" fill-opacity=\"{:.2}\"{stroke_attr}/>", hex_rgb(poly.fill), opacity(poly.fill[3]));
    }

    for ln in &prims.lines {
        let (Some([x1, y1]), Some([x2, y2])) = (tf.map(ln.a), tf.map(ln.b)) else { continue };
        let dash = if ln.dashed { " stroke-dasharray=\"7,5\"" } else { "" };
        let col = hex_rgb(ln.color);
        let _ = writeln!(
            out,
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"{col}\" stroke-opacity=\"{:.2}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\"{dash}/>",
            opacity(ln.color[3]),
            ln.width
        );
        let len = (x2 - x1).hypot(y2 - y1);
        if ln.arrow_mid && len >= 10.0 {
            let (ux, uy) = ((x2 - x1) / len, (y2 - y1) / len);
            let (tx, ty) = (x1 + (x2 - x1) * 0.56, y1 + (y2 - y1) * 0.56);
            let w = 6.5_f32;
            let (ax, ay) = (tx - ux * w - uy * w * 0.62, ty - uy * w + ux * w * 0.62);
            let (bx, by) = (tx - ux * w + uy * w * 0.62, ty - uy * w - ux * w * 0.62);
            let _ = writeln!(
                out,
                "<polyline points=\"{ax:.2},{ay:.2} {tx:.2},{ty:.2} {bx:.2},{by:.2}\" fill=\"none\" stroke=\"{col}\" stroke-opacity=\"{:.2}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\"/>",
                opacity(ln.color[3]),
                ln.width
            );
        }
    }

    for circ in &prims.circles {
        let Some([cx, cy]) = tf.map(circ.center) else { continue };
        let r = (circ.radius_world * tf.scale).clamp(circ.min_px, circ.max_px * 1.35);
        let (scol, sw) = circ.stroke;
        let _ = writeln!(
            out,
            "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r:.2}\" fill=\"{}\" fill-opacity=\"{:.2}\" stroke=\"{}\" stroke-opacity=\"{:.2}\" stroke-width=\"{sw:.2}\"/>",
            hex_rgb(circ.fill),
            opacity(circ.fill[3]),
            hex_rgb(scol),
            opacity(scol[3])
        );
    }

    for lbl in &prims.labels {
        let Some([x, y]) = tf.map(lbl.pos) else { continue };
        let _ = writeln!(
            out,
            "<text x=\"{x:.2}\" y=\"{y:.2}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\" fill-opacity=\"{:.2}\">{}</text>",
            lbl.size_px,
            hex_rgb(lbl.color),
            opacity(lbl.color[3]),
            xml_escape(&lbl.text)
        );
    }
}

#[cfg(test)]
#[path = "../export_tests.rs"]
mod tests;
