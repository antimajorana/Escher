//! Text of the PNG export, rasterized from the bundled TTFs (the bytes egui registers, see `theme::typography`).
//! Each character takes the first face of the chain Inter -> DejaVu Sans that maps it (DejaVu covers Greek, math
//! and sub / superscripts Inter lacks); advances and same-face pair kerning come from the fonts, positions are
//! sub-pixel, and coverage is accumulated over the whole string, clamped to 1, then alpha-blended once per pixel.
//! A character no face maps draws as an outlined box (tofu); nothing here panics on any input.
//!
//! `size_px` is the em size in pixels (as SVG `font-size` and egui `FontId::size`). The text box is
//! `measure_text` wide (sum of advances and kerning) and `ascent - descent` of the first face high, baseline
//! `ascent` below its top; `anchor` (ax, ay) puts the point (ax, ay) of the box (fractions of its width and
//! height, (0, 0) = left-top, (0.5, 0.5) = centre) on `pos`, as the former 5x7 `Raster::text` did.

use std::sync::LazyLock;

use ab_glyph::{Font, FontRef, GlyphId, OutlinedGlyph, PxScale, point};

use super::kern::Kerning;
use super::raster::Raster;
use crate::theme::typography::{DEJAVU_SANS, INTER_REGULAR, INTER_SEMIBOLD};

/// Weight of the first face of the chain (DejaVu Sans Book is the fallback for both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Regular,
    SemiBold,
}

/// Tofu box: advance 0.6 em, box from 0.1 to 0.5 em across and 0.7 em up from the baseline, stroke 0.06 em.
const TOFU_ADVANCE: f32 = 0.6;
const TOFU_BOX: [f32; 3] = [0.1, 0.5, 0.7];
const TOFU_STROKE: f32 = 0.06;
/// A glyph whose pixel box exceeds this many pixels is skipped (the rasterizer allocates its whole box).
const MAX_GLYPH_PIXELS: f32 = 16_777_216.0;

struct Face {
    font: FontRef<'static>,
    kern: Option<Kerning>,
    upem: f32,
}

fn load(data: &'static [u8]) -> Option<Face> {
    let font = FontRef::try_from_slice(data).ok()?;
    let upem = font.units_per_em().filter(|u| *u > 0.0)?;
    Some(Face { font, kern: Kerning::new(data), upem })
}

static INTER: LazyLock<Option<Face>> = LazyLock::new(|| load(INTER_REGULAR));
static INTER_SB: LazyLock<Option<Face>> = LazyLock::new(|| load(INTER_SEMIBOLD));
static DEJAVU: LazyLock<Option<Face>> = LazyLock::new(|| load(DEJAVU_SANS));

fn chain(weight: Weight) -> [Option<&'static Face>; 2] {
    let first = match weight {
        Weight::Regular => INTER.as_ref(),
        Weight::SemiBold => INTER_SB.as_ref(),
    };
    [first, DEJAVU.as_ref()]
}

/// Face and glyph of `c` in the chain; None when no face maps it (tofu).
fn lookup(faces: &[Option<&'static Face>; 2], c: char) -> Option<(&'static Face, GlyphId)> {
    faces.iter().flatten().find_map(|f| {
        let id = f.font.glyph_id(c);
        (id.0 != 0).then_some((*f, id))
    })
}

/// True when some face of the regular chain maps `c` (so it does not draw as tofu).
#[cfg(test)]
pub fn has_glyph(c: char) -> bool {
    lookup(&chain(Weight::Regular), c).is_some()
}

struct Placed {
    glyph: Option<(&'static Face, GlyphId)>,
    x: f32,
}

struct Layout {
    glyphs: Vec<Placed>,
    width: f32,
    ascent: f32,
    height: f32,
}

fn layout(s: &str, size: f32, weight: Weight) -> Layout {
    let faces = chain(weight);
    let (asc, desc) = faces
        .iter()
        .flatten()
        .next()
        .map_or((0.97, -0.24), |f| (f.font.ascent_unscaled() / f.upem, f.font.descent_unscaled() / f.upem));
    let (mut glyphs, mut x, mut prev) = (Vec::new(), 0.0_f32, None::<(&'static Face, GlyphId)>);
    for c in s.chars().filter(|c| !c.is_control()) {
        let found = lookup(&faces, c);
        if let Some((f, id)) = found {
            let k = size / f.upem;
            if let (Some((pf, pid)), Some(kern)) = (prev, &f.kern)
                && std::ptr::eq(pf, f)
            {
                x += kern.pair(pid.0, id.0) * k;
            }
            glyphs.push(Placed { glyph: found, x });
            x += f.font.h_advance_unscaled(id) * k;
        } else {
            glyphs.push(Placed { glyph: None, x });
            x += TOFU_ADVANCE * size;
        }
        prev = found;
    }
    Layout { glyphs, width: x, ascent: asc * size, height: (asc - desc) * size }
}

/// `[width, height]` of the text box of `s` at em size `size_px` (regular weight); `[0, 0]` for a bad size.
/// Layout API for text placement by other exports (the 3D PNG export); the 2D export only anchors.
#[allow(dead_code)]
pub fn measure_text(s: &str, size_px: f32) -> [f32; 2] {
    measure_text_weight(s, size_px, Weight::Regular)
}

#[allow(dead_code)]
pub fn measure_text_weight(
s: &str, size_px: f32, weight: Weight) -> [f32; 2] {
    if !(size_px.is_finite() && size_px > 0.0) {
        return [0.0, 0.0];
    }
    let l = layout(s, size_px, weight);
    [l.width, l.height]
}

/// Draw `s` (regular weight) with the point `anchor` of its text box on `pos`; see the module notes.
pub fn draw_text(img: &mut Raster, pos: [f32; 2], s: &str, size_px: f32, color: [u8; 4], anchor: (f32, f32)) {
    draw_text_weight(img, pos, s, size_px, color, anchor, Weight::Regular);
}

/// Coverage accumulator over a pixel window `[x0, x0 + w) x [y0, y0 + h)` of the image.
struct Coverage {
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    cov: Vec<f32>,
}

impl Coverage {
    fn add(&mut self, x: i32, y: i32, v: f32) {
        let (dx, dy) = (x - self.x0, y - self.y0);
        if dx >= 0 && dy >= 0 && (dx as usize) < self.w && (dy as usize) < self.h {
            self.cov[dy as usize * self.w + dx as usize] += v;
        }
    }

    /// Exact area coverage of the axis-aligned rectangle `[l, r] x [t, b]`.
    fn add_rect(&mut self, [l, t, r, b]: [f32; 4]) {
        let (wx1, wy1) = (self.x0 + self.w as i32, self.y0 + self.h as i32);
        for y in (t.floor() as i32).max(self.y0)..(b.ceil() as i32).min(wy1) {
            let oy = (b.min(y as f32 + 1.0) - t.max(y as f32)).max(0.0);
            for x in (l.floor() as i32).max(self.x0)..(r.ceil() as i32).min(wx1) {
                let ox = (r.min(x as f32 + 1.0) - l.max(x as f32)).max(0.0);
                self.add(x, y, ox * oy);
            }
        }
    }
}

enum Ink {
    Outline(OutlinedGlyph),
    Rect([f32; 4]),
}

impl Ink {
    fn bounds(&self) -> [f32; 4] {
        match self {
            Self::Outline(g) => {
                let b = g.px_bounds();
                [b.min.x, b.min.y, b.max.x, b.max.y]
            }
            Self::Rect(r) => *r,
        }
    }
}

pub fn draw_text_weight(
    img: &mut Raster,
    pos: [f32; 2],
    s: &str,
    size_px: f32,
    color: [u8; 4],
    anchor: (f32, f32),
    weight: Weight,
) {
    let finite = pos[0].is_finite() && pos[1].is_finite() && anchor.0.is_finite() && anchor.1.is_finite();
    if !finite || s.is_empty() || !(size_px.is_finite() && size_px > 0.0) || color[3] == 0 {
        return;
    }
    let lay = layout(s, size_px, weight);
    let x0 = pos[0] - anchor.0 * lay.width;
    let base = pos[1] - anchor.1 * lay.height + lay.ascent;
    // whole box (plus one em for overhangs) off the image: nothing to draw
    let (iw, ih) = (img.w as f32, img.h as f32);
    if x0 > iw + size_px || x0 + lay.width < -size_px || base - lay.ascent > ih + size_px || base - lay.ascent + lay.height < -size_px {
        return;
    }

    let mut inks = Vec::with_capacity(lay.glyphs.len() * 4);
    for p in &lay.glyphs {
        let gx = x0 + p.x;
        if let Some((f, id)) = p.glyph {
            let scale = PxScale::from(size_px * f.font.height_unscaled() / f.upem);
            if let Some(g) = f.font.outline_glyph(id.with_scale_and_position(scale, point(gx, base)))
                && g.px_bounds().width() * g.px_bounds().height() <= MAX_GLYPH_PIXELS
            {
                inks.push(Ink::Outline(g));
            }

        } else {
            let (l, r, top, t) = (gx + TOFU_BOX[0] * size_px, gx + TOFU_BOX[1] * size_px, base - TOFU_BOX[2] * size_px, (TOFU_STROKE * size_px).max(1.0));
            inks.extend([[l, top, r, top + t], [l, base - t, r, base], [l, top + t, l + t, base - t], [r - t, top + t, r, base - t]].map(Ink::Rect));
        }
    }
    let Some(bb) = inks.iter().map(Ink::bounds).reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]) else {
        return;
    };
    let (cx0, cy0) = (bb[0].floor().max(0.0), bb[1].floor().max(0.0));
    let (cx1, cy1) = (bb[2].ceil().min(iw), bb[3].ceil().min(ih));
    if !(cx0 < cx1 && cy0 < cy1) {
        return;
    }
    let (w, h) = ((cx1 - cx0) as usize, (cy1 - cy0) as usize);
    let mut cov = Coverage { x0: cx0 as i32, y0: cy0 as i32, w, h, cov: vec![0.0; w * h] };
    for ink in &inks {
        match ink {
            Ink::Outline(g) => {
                let b = g.px_bounds();
                let (bx, by) = (b.min.x as i32, b.min.y as i32);
                g.draw(|x, y, c| cov.add(bx + x as i32, by + y as i32, c));
            }
            Ink::Rect(r) => cov.add_rect(*r),
        }
    }
    for (k, c) in cov.cov.iter().enumerate() {
        let a = (f32::from(color[3]) * c.clamp(0.0, 1.0)).round() as u8;
        if a > 0 {
            img.blend(cov.x0 + (k % w) as i32, cov.y0 + (k / w) as i32, [color[0], color[1], color[2], a]);
        }
    }
}

#[cfg(test)]
#[path = "text_ttf_tests.rs"]
mod tests;
