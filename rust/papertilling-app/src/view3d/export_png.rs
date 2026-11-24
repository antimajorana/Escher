//! 3D PNG export: the live view re-rendered offscreen (`super::offscreen`) at the viewport's aspect and size x the
//! PNG scale, with the caption and the colour legend composited on the CPU (`Raster` + the TTF text of the 2D
//! export), so the composition is testable without a GPU.
//!
//! Size: the viewport's on-screen pixels (points x pixels per point) times the scale, so 1x is the screen's pixel
//! density; overlay lengths in points are multiplied by `k = width / viewport width in points`, and the wires are
//! `width / on-screen width` pixels wide (`wire_quads`). When a side would exceed the GPU texture limit or `MAX_SIDE`
//! (8192), or the staging buffer the device's `max_buffer_size`, the picture is scaled down uniformly (aspect within
//! 1 px). The file is written through a temp file renamed over the target, so it is never left half written.

use std::path::Path;

use super::View3dState;
use super::legend::{ColorLegend3d, fmt_v, legend_rect_3d, legend_title};
use super::offscreen::{OffscreenGpuContext, OffscreenScene, render_rgba};
use super::wire_quads::export_wire_width;
use crate::export::raster::{MAX_SIDE, Raster};
use crate::export::text_ttf::{draw_text, measure_text};
use crate::theme::ThemeTokens;

/// Pixel size of a 3D PNG export and whether it was scaled down to fit the limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Export3dInfo {
    pub w: u32,
    pub h: u32,
    pub clamped: bool,
}

/// `view` (on-screen pixels) x `scale`, rounded, scaled down uniformly so that both sides are at most `max_side` and the
/// pixel count at most `max_pixels`; `Err` for an empty or non-finite view or a zero scale.
pub(crate) fn clamp_export_size(view: [f32; 2], scale: u32, max_side: u32, max_pixels: u64) -> Result<Export3dInfo, String> {
    let (vw, vh) = (f64::from(view[0]) * f64::from(scale), f64::from(view[1]) * f64::from(scale));
    if !(vw.is_finite() && vh.is_finite() && vw >= 1.0 && vh >= 1.0) || max_side == 0 || max_pixels == 0 {
        return Err(format!("3D view size {}x{} at {scale}x cannot be exported", view[0], view[1]));
    }
    let (w0, h0) = (vw.round(), vh.round());
    let side = f64::from(max_side);
    let f = (side / w0).min(side / h0).min((max_pixels as f64 / (w0 * h0)).sqrt());
    if f >= 1.0 {
        return Ok(Export3dInfo { w: w0 as u32, h: h0 as u32, clamped: false });
    }
    let (mut w, mut h) = (((w0 * f).round()).clamp(1.0, side) as u32, ((h0 * f).round()).clamp(1.0, side) as u32);
    while u64::from(w) * u64::from(h) > max_pixels {
        // rounding up both sides can overshoot the pixel budget by a row or column
        if w >= h { w -= 1 } else { h -= 1 }
    }
    Ok(Export3dInfo { w, h, clamped: true })
}

/// Size of an export of a `view` point viewport shown at `ppp` pixels per point, at `scale` times its on-screen
/// pixels, within `max_side` a side and a staging buffer of `max_buffer` bytes (4 bytes a pixel), and its wire width
/// in pixels ([`export_wire_width`]: `None` for the 1 px lines of the screen).
pub(crate) fn export_plan(view: egui::Vec2, ppp: f32, scale: u32, max_side: u32, max_buffer: u64) -> Result<(Export3dInfo, Option<f32>), String> {
    let screen = view * ppp;
    let info = clamp_export_size([screen.x, screen.y], scale, max_side, max_buffer / 4)?;
    Ok((info, export_wire_width(info.w, screen.x)))
}

/// Write `bytes` to `path` through the sibling `<name>.part`, renamed over `path` once complete (removed on error),
/// so an interrupted export never leaves a truncated file under the chosen name.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path.file_name().ok_or_else(|| format!("{} is not a file path", path.display()))?;
    let mut part = name.to_os_string();
    part.push(".part");
    let tmp = path.with_file_name(part);
    std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot write {}: {e}", path.display())
    })
}

fn rgba(c: egui::Color32) -> [u8; 4] {
    c.to_srgba_unmultiplied()
}

/// Caption (top-left) and, when `legend` is given and the viewport is large enough for the live HUD, the colorbar
/// box, drawn over `img` as `View3dState::show` lays them out in a viewport of `view` points, every length x `k`.
/// The bar is a per-column gradient from the colour at `min_val` (first column) to the colour at `max_val` (last).
pub(crate) fn compose_overlays(
    img: &mut Raster,
    view: egui::Vec2,
    k: f32,
    caption: &str,
    legend: Option<&ColorLegend3d>,
    tokens: &ThemeTokens,
) {
    if !caption.is_empty() {
        // The export has no camera controls below the title, so all counts can wrap in full.
        let width = (view.x - 16.0).max(1.0);
        let mut line = String::new();
        let mut y = 6.0;
        let mut draw_line = |text: &str| {
            let [w, height] = measure_text(text, 12.0);
            let size = 12.0 * (width / w.max(1.0)).min(1.0);
            draw_text(img, [8.0 * k, y * k], text, size * k, rgba(tokens.text_primary), (0.0, 0.0));
            y += height + 2.0;
        };
        for word in caption.split_whitespace() {
            let next = if line.is_empty() { word.to_owned() } else { format!("{line} {word}") };
            if !line.is_empty() && measure_text(&next, 12.0)[0] > width {
                draw_line(&line);
                line = word.to_owned();
            } else {
                line = next;
            }
        }
        if !line.is_empty() {
            draw_line(&line);
        }
    }
    let Some((leg, r)) = legend.zip(legend_rect_3d(egui::Rect::from_min_size(egui::Pos2::ZERO, view))) else {
        return;
    };
    let px = |v: f32| (v * k).round() as i32;
    let bw = k.round().max(1.0) as i32;
    let (bx0, by0, bx1, by1) = (px(r.left()), px(r.top()), px(r.right()), px(r.bottom()));
    img.rect_bw((bx0, by0, bx1, by1), rgba(tokens.bg_hud), rgba(tokens.border_strong), bw);
    let title = legend_title(leg);
    draw_text(img, [(r.left() + 7.0) * k, (r.top() + 4.0) * k], &title, 9.5 * k, rgba(tokens.text_primary), (0.0, 0.0));
    let (x0, y0, x1, y1) = (px(r.left() + 8.0), px(r.top() + 19.0), px(r.right() - 8.0), px(r.top() + 27.0));
    // border just outside the bar, so every gradient column keeps its colour
    let edge = rgba(tokens.border_subtle);
    img.rect_bw((x0 - bw, y0 - bw, x1 + bw, y1 + bw), edge, edge, bw);
    let n = (x1 - x0).max(1);
    for i in 0..n {
        let t = if n > 1 { f64::from(i) / f64::from(n - 1) } else { 0.0 };
        let c = leg.sample(t);
        img.rect_bw((x0 + i, y0, x0 + i + 1, y1), [c[0], c[1], c[2], 255], [c[0], c[1], c[2], 255], 1);
    }
    let label = rgba(tokens.text_secondary);
    let by = (r.bottom() - 3.0) * k;
    draw_text(img, [(r.left() + 7.0) * k, by], &fmt_v(leg.min_val), 9.0 * k, label, (0.0, 1.0));
    draw_text(img, [(r.right() - 7.0) * k, by], &fmt_v(leg.max_val), 9.0 * k, label, (1.0, 1.0));
}

/// PNG bytes (RGBA 8-bit) of `img`.
pub(crate) fn encode_png(img: &Raster) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, img.w, img.h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&img.rgba).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

impl View3dState {
    /// Viewport size (points) of the last `show`, if it ran in egui pass `pass_nr` or the one before (the view is
    /// on screen now); `None` when the 3D view is not being shown.
    pub(crate) fn shown_view_size(&self, pass_nr: u64) -> Option<egui::Vec2> {
        self.last_view.filter(|&(_, _, shown)| pass_nr <= shown.saturating_add(1)).map(|(size, _, _)| size)
    }

    /// [`Self::shown_view_size`] in physical pixels (points x pixels per point).
    pub(crate) fn shown_view_pixels(&self, pass_nr: u64) -> Option<egui::Vec2> {
        self.shown_view_size(pass_nr).zip(self.last_view).map(|(size, (_, ppp, _))| size * ppp)
    }

    /// Render the current view (same camera, mesh and theme as on screen) at the last viewport's on-screen pixels x
    /// `scale`, clamped to the GPU limits: the GPU part of the export, on the calling (UI) thread. The returned job
    /// composites the caption / legend and writes the PNG, on any thread ([`RenderedExport3d::finish`]).
    pub(crate) fn render_export(&self, gpu: &OffscreenGpuContext, tokens: &ThemeTokens, scale: u32) -> Result<RenderedExport3d, String> {
        let (view, ppp, _) = self.last_view.ok_or_else(|| String::from("Show the 3D view before exporting a 3D PNG."))?;
        let (info, wire_width) = export_plan(view, ppp, scale, gpu.max_texture_side.min(MAX_SIDE), gpu.max_buffer_size)?;
        let aspect = (view.x / view.y.max(1.0)).max(0.1);
        let rot = self.rotation();
        let scene = OffscreenScene {
            mesh: &self.mesh_data,
            mvp: self.view_mvp(aspect, &rot),
            rot,
            is_light: tokens.is_light,
            background: rgba(tokens.bg_viewport_3d),
            wire_width,
        };
        let pixels = render_rgba(gpu, &scene, info.w, info.h)?;
        Ok(RenderedExport3d {
            img: Raster { w: info.w, h: info.h, rgba: pixels },
            info,
            view,
            caption: self.overlay_caption.clone(),
            legend: self.legend().filter(|_| self.has_geometry()).cloned(),
            tokens: *tokens,
        })
    }

    /// [`Self::render_export`] and [`RenderedExport3d::finish`] in one call (smoke run, tests).
    pub(crate) fn export_png(&self, gpu: &OffscreenGpuContext, tokens: &ThemeTokens, scale: u32, path: &Path) -> Result<Export3dInfo, String> {
        self.render_export(gpu, tokens, scale)?.finish(path)
    }
}

/// The rendered 3D picture with what its overlays need: no GPU handles, so it can be finished on a worker thread.
pub(crate) struct RenderedExport3d {
    img: Raster,
    info: Export3dInfo,
    view: egui::Vec2,
    caption: String,
    legend: Option<ColorLegend3d>,
    tokens: ThemeTokens,
}

impl RenderedExport3d {
    /// Composite the caption and legend (sizes x the effective scale), encode the PNG and write it to `path`.
    pub(crate) fn finish(mut self, path: &Path) -> Result<Export3dInfo, String> {
        let k = self.info.w as f32 / self.view.x.max(1.0);
        compose_overlays(&mut self.img, self.view, k, &self.caption, self.legend.as_ref(), &self.tokens);
        let bytes = encode_png(&self.img)?;
        write_atomic(path, &bytes)?;
        Ok(self.info)
    }
}
