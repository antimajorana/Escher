//! Snapshot interpolation, relaxation / motion playback frames, and a pure-Rust GIF89a encoder (`#056`, `#094`, `#172`).
//!
//! GIF89a layout: header, logical screen descriptor with a global colour table (padded to a power of two), NETSCAPE2.0
//! loop extension (loop forever), then per frame a graphic control extension (delay in 1/100 s) and an image whose
//! pixels are LZW-coded (variable code width 3..12 bits, clear code first, clear again when the table is full, end code
//! last, codes packed LSB first, 255-byte sub-blocks), and the trailer `0x3B`.

use std::collections::HashMap;

/// `n_frames` positions linearly interpolated from `start` to `end` (both included).
pub fn interpolate_crystal_frames(start: &[[f64; 2]], end: &[[f64; 2]], n_frames: usize) -> Result<Vec<Vec<[f64; 2]>>, String> {
    if start.len() != end.len() {
        return Err(format!("start has {} sites, end has {}", start.len(), end.len()));
    }
    if n_frames < 2 {
        return Err("need at least 2 frames".to_string());
    }
    Ok((0..n_frames).map(|k| {
        let t = k as f64 / (n_frames - 1) as f64;
        start.iter().zip(end).map(|(a, b)| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]).collect()
    }).collect())
}

/// Palette used by [`render_frames`]: background, bonds, sites.
pub const PALETTE: [[u8; 3]; 3] = [[255, 255, 255], [40, 40, 40], [200, 30, 30]];

/// Rasterise each frame (bonds index 1, sites index 2 on background 0) with one common scale so motion is visible.
pub fn render_frames(frames: &[Vec<[f64; 2]>], edges: &[(usize, usize)], width: u16, height: u16) -> Result<Vec<Vec<u8>>, String> {
    let (w, h) = (usize::from(width), usize::from(height));
    if w < 4 || h < 4 {
        return Err("image must be at least 4 x 4".to_string());
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in frames.iter().flatten() {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if !lo[0].is_finite() {
        return Err("no positions to draw".to_string());
    }
    let margin = 3.0;
    let sx = (w as f64 - 2.0 * margin) / (hi[0] - lo[0]).max(1e-9);
    let sy = (h as f64 - 2.0 * margin) / (hi[1] - lo[1]).max(1e-9);
    let s = sx.min(sy);
    let px = |p: [f64; 2]| -> (i64, i64) {
        ((margin + (p[0] - lo[0]) * s).round() as i64, (h as f64 - 1.0 - margin - (p[1] - lo[1]) * s).round() as i64)
    };
    Ok(frames.iter().map(|fr| {
        let mut img = vec![0u8; w * h];
        let mut put = |x: i64, y: i64, c: u8| {
            if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                img[y as usize * w + x as usize] = c;
            }
        };
        for &(i, j) in edges {
            if i >= fr.len() || j >= fr.len() {
                continue;
            }
            let ((mut x0, mut y0), (x1, y1)) = (px(fr[i]), px(fr[j]));
            let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
            let (stx, sty) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
            let mut err = dx + dy;
            loop {
                put(x0, y0, 1);
                if x0 == x1 && y0 == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy { err += dy; x0 += stx; }
                if e2 <= dx { err += dx; y0 += sty; }
            }
        }
        for &p in fr {
            let (x, y) = px(p);
            for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)] {
                put(x + ox, y + oy, 2);
            }
        }
        img
    }).collect())
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn put(&mut self, code: u16, width: u32) {
        self.acc |= u32::from(code) << self.nbits;
        self.nbits += width;
        while self.nbits >= 8 {
            self.out.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.acc & 0xFF) as u8);
        }
        self.out
    }
}

/// LZW code stream (before sub-block packing) of `pixels` with minimum code size `min`.
pub fn lzw_encode(pixels: &[u8], min: u32) -> Vec<u8> {
    let clear = 1u16 << min;
    let eoi = clear + 1;
    let mut w = BitWriter { out: Vec::new(), acc: 0, nbits: 0 };
    let (mut next, mut width) = (eoi + 1, min + 1);
    let mut dict: HashMap<(u16, u8), u16> = HashMap::new();
    w.put(clear, width);
    let mut prefix: Option<u16> = None;
    for &k in pixels {
        let Some(p) = prefix else {
            prefix = Some(u16::from(k));
            continue;
        };
        if let Some(&c) = dict.get(&(p, k)) {
            prefix = Some(c);
            continue;
        }
        w.put(p, width);
        if next < 4096 {
            dict.insert((p, k), next);
            next += 1;
            // decoders widen one entry late, hence `>` (the GIF "early change" convention)
            if u32::from(next) > (1 << width) && width < 12 {
                width += 1;
            }
        } else {
            w.put(clear, width);
            dict.clear();
            next = eoi + 1;
            width = min + 1;
        }
        prefix = Some(u16::from(k));
    }
    if let Some(p) = prefix {
        w.put(p, width);
        if next < 4096 && u32::from(next + 1) > (1 << width) && width < 12 {
            width += 1;
        }
    }
    w.put(eoi, width);
    w.finish()
}

/// Encode indexed frames as a looping GIF89a.
pub fn encode_gif89a(width: u16, height: u16, palette_rgb: &[[u8; 3]], indexed_frames: &[Vec<u8>], delay_cs: u16) -> Result<Vec<u8>, String> {
    if palette_rgb.is_empty() || palette_rgb.len() > 256 {
        return Err("palette must have 1..=256 colours".to_string());
    }
    if indexed_frames.is_empty() {
        return Err("no frames".to_string());
    }
    let npix = usize::from(width) * usize::from(height);
    let mut k = 1u32;
    while (1usize << k) < palette_rgb.len() {
        k += 1;
    }
    for (fi, fr) in indexed_frames.iter().enumerate() {
        if fr.len() != npix {
            return Err(format!("frame {fi} has {} pixels, expected {npix}", fr.len()));
        }
        if fr.iter().any(|&c| usize::from(c) >= palette_rgb.len()) {
            return Err(format!("frame {fi} uses a colour index outside the palette"));
        }
    }
    let mut out = b"GIF89a".to_vec();
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&[0x80 | (((k - 1) as u8) << 4) | (k - 1) as u8, 0, 0]);
    for c in 0..(1usize << k) {
        out.extend_from_slice(&palette_rgb.get(c).copied().unwrap_or([0, 0, 0]));
    }
    out.extend_from_slice(&[0x21, 0xFF, 0x0B]);
    out.extend_from_slice(b"NETSCAPE2.0");
    out.extend_from_slice(&[0x03, 0x01, 0x00, 0x00, 0x00]);
    let min = k.max(2);
    for fr in indexed_frames {
        out.extend_from_slice(&[0x21, 0xF9, 0x04, 0x04]);
        out.extend_from_slice(&delay_cs.to_le_bytes());
        out.extend_from_slice(&[0x00, 0x00, 0x2C, 0, 0, 0, 0]);
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.push(0x00);
        out.push(min as u8);
        for chunk in lzw_encode(fr, min).chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0x00);
    }
    out.push(0x3B);
    Ok(out)
}

/// Render `frames` (e.g. a relaxation trajectory or [`interpolate_crystal_frames`]) and encode them as a GIF.
pub fn export_motion_gif(frames: &[Vec<[f64; 2]>], edges: &[(usize, usize)], width: u16, height: u16, delay_cs: u16) -> Result<Vec<u8>, String> {
    let imgs = render_frames(frames, edges, width, height)?;
    encode_gif89a(width, height, &PALETTE, &imgs, delay_cs)
}
