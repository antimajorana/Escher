//! RGBA raster of the PNG export: alpha blending, antialiased segments and circles, polygon fill (scanline,
//! even-odd: each pixel blended once), bordered rects, and TTF text (`super::text_ttf`). Every coordinate is
//! clamped to the image, and non-finite input draws nothing.

pub(crate) const MAX_SIDE: u32 = 8192;
pub(crate) const MAX_PIXELS: u64 = 40_000_000;

pub(crate) struct Raster {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

fn fin(p: [f32; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

impl Raster {
    pub fn new(w: u32, h: u32, bg: [u8; 4]) -> Result<Self, String> {
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || u64::from(w) * u64::from(h) > MAX_PIXELS {
            return Err(format!("PNG size {w}x{h} out of range (1..={MAX_SIDE} pixels a side, at most {MAX_PIXELS} pixels)"));
        }
        Ok(Self { w, h, rgba: bg.repeat(w as usize * h as usize) })
    }

    /// Pixel index range [lo, hi] of the box [a - pad, b + pad] clipped to the image; None when outside.
    fn span(&self, a: f32, b: f32, pad: f32, n: u32) -> Option<(i32, i32)> {
        let lo = (a.min(b) - pad).floor().max(0.0);
        let hi = (a.max(b) + pad).ceil().min(n as f32 - 1.0);
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some((lo as i32, hi as i32))
    }

    pub fn blend(&mut self, x: i32, y: i32, src: [u8; 4]) {
        if x < 0 || y < 0 || x as u32 >= self.w || y as u32 >= self.h || src[3] == 0 {
            return;
        }
        let idx = (y as usize * self.w as usize + x as usize) * 4;
        let sa = f32::from(src[3]) / 255.0;
        for k in 0..3 {
            self.rgba[idx + k] = (f32::from(src[k]) * sa + f32::from(self.rgba[idx + k]) * (1.0 - sa)).round() as u8;
        }
        self.rgba[idx + 3] = 255;
    }

    pub fn fill_polygon(&mut self, pts: &[[f32; 2]], color: [u8; 4]) {
        if pts.len() < 3 || !pts.iter().all(|&p| fin(p)) {
            return;
        }
        let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
        let Some((ylo, yhi)) = self.span(y0, y1, 0.0, self.h) else { return };
        let mut xs = Vec::new();
        for y in ylo..=yhi {
            let py = y as f32 + 0.5;
            xs.clear();
            for k in 0..pts.len() {
                let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
                if (a[1] <= py && py < b[1]) || (b[1] <= py && py < a[1]) {
                    xs.push(a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                }
            }
            xs.sort_by(f32::total_cmp);
            for pair in xs.as_chunks::<2>().0 {
                let x_from = (pair[0] - 0.5).ceil().max(0.0) as i32;
                let x_to = ((pair[1] - 0.5).ceil() - 1.0).min(self.w as f32 - 1.0) as i32;
                for x in x_from..=x_to {
                    self.blend(x, y, color);
                }
            }
        }
    }

    pub fn segment(&mut self, p0: [f32; 2], p1: [f32; 2], color: [u8; 4], stroke_w: f32) {
        if !fin(p0) || !fin(p1) {
            return;
        }
        let half = (stroke_w * 0.5).max(0.55);
        let pad = half + 1.2;
        let (Some((xlo, xhi)), Some((ylo, yhi))) = (self.span(p0[0], p1[0], pad, self.w), self.span(p0[1], p1[1], pad, self.h)) else {
            return;
        };
        let (vx, vy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len2 = vx * vx + vy * vy;
        let len = len2.sqrt();
        for y in ylo..=yhi {
            let py = y as f32 + 0.5;
            // pixels within `pad` of the infinite line on this row: |x - xc| <= pad * len / |vy| (a superset of
            // the stroke), so a long diagonal costs O(length) instead of its whole bounding box
            let (mut xa, mut xb) = (xlo, xhi);
            if vy.abs() > 1e-3 {
                let xc = p0[0] + (py - p0[1]) * vx / vy;
                let half_w = pad * len / vy.abs();
                xa = xa.max((xc - half_w - 0.5).floor().max(-1.0e9) as i32);
                xb = xb.min((xc + half_w - 0.5).ceil().min(1.0e9) as i32);
            }
            for x in xa..=xb {
                let px = x as f32 + 0.5;
                let t = if len2 > 1e-6 { (((px - p0[0]) * vx + (py - p0[1]) * vy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let dist = (px - (p0[0] + t * vx)).hypot(py - (p0[1] + t * vy));
                let cov = (half + 0.5 - dist).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(x, y, [color[0], color[1], color[2], (f32::from(color[3]) * cov).round() as u8]);
                }
            }
        }
    }

    pub fn dashed(&mut self, p0: [f32; 2], p1: [f32; 2], color: [u8; 4], stroke_w: f32, (dash, gap): (f32, f32)) {
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let total = dx.hypot(dy);
        if !total.is_finite() || total < 1e-3 {
            return;
        }
        let (ux, uy) = (dx / total, dy / total);
        let mut s = 0.0;
        while s < total {
            let e = (s + dash).min(total);
            self.segment([p0[0] + ux * s, p0[1] + uy * s], [p0[0] + ux * e, p0[1] + uy * e], color, stroke_w);
            s += dash + gap;
        }
    }

    pub fn circle(&mut self, c: [f32; 2], r: f32, fill: [u8; 4], (scol, sw): ([u8; 4], f32)) {
        if !fin(c) || !r.is_finite() {
            return;
        }
        let outer = r + sw * 0.5 + 1.2;
        let (Some((xlo, xhi)), Some((ylo, yhi))) = (self.span(c[0], c[0], outer, self.w), self.span(c[1], c[1], outer, self.h)) else {
            return;
        };
        for y in ylo..=yhi {
            let py = y as f32 + 0.5;
            for x in xlo..=xhi {
                let d = (x as f32 + 0.5 - c[0]).hypot(py - c[1]);
                let fill_cov = (r + 0.5 - d).clamp(0.0, 1.0);
                if fill_cov > 0.0 && fill[3] > 0 {
                    self.blend(x, y, [fill[0], fill[1], fill[2], (f32::from(fill[3]) * fill_cov).round() as u8]);
                }
                if sw > 0.0 && scol[3] > 0 {
                    let ring_cov = (sw * 0.5 + 0.5 - (d - r).abs()).clamp(0.0, 1.0);
                    if ring_cov > 0.0 {
                        self.blend(x, y, [scol[0], scol[1], scol[2], (f32::from(scol[3]) * ring_cov).round() as u8]);
                    }
                }
            }
        }
    }

    pub fn rect(&mut self, bx: (i32, i32, i32, i32), fill: [u8; 4], border: [u8; 4]) {
        self.rect_bw(bx, fill, border, 1);
    }

    /// [`Self::rect`] with a border `bw` pixels wide (the 1 px border of a picture drawn at `bw`x scale).
    pub fn rect_bw(&mut self, (x0, y0, x1, y1): (i32, i32, i32, i32), fill: [u8; 4], border: [u8; 4], bw: i32) {
        for y in y0.max(0)..y1.min(self.h as i32) {
            for x in x0.max(0)..x1.min(self.w as i32) {
                let edge = x < x0 + bw || x >= x1 - bw || y < y0 + bw || y >= y1 - bw;
                self.blend(x, y, if edge { border } else { fill });
            }
        }
    }

    /// Text of em size `size_px` at `pos`, `anchor` = (0, 0) left-top .. (0.5, 0.5) centre of the text box
    /// (TTF glyphs with the Inter -> DejaVu Sans fallback chain, see [`super::text_ttf`]).
    pub fn text(&mut self, pos: [f32; 2], s: &str, size_px: f32, color: [u8; 4], anchor: (f32, f32)) {
        super::text_ttf::draw_text(self, pos, s, size_px, color, anchor);
    }
}
