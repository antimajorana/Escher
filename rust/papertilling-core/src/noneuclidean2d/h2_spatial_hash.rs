//! `SU(1,1)` Poincare-disk arithmetic and `O(1)`-bucket polar hyperbolic spatial hash.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum H2Kernel {
    GraphCombinatorial,
    DunhamHyperboloid,
    SpatialHashPoincare,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RegionFilter {
    PoincareRadiusMax(f64),
    HyperbolicRadiusMax(f64),
    ShellRange(usize, usize),
    DiskBall { center: [f64; 2], radius: f64 },
    HalfPlane { nx: f64, ny: f64, offset: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H2LargeSummary {
    pub p: usize,
    pub q: usize,
    pub kernel: H2Kernel,
    pub total_faces: usize,
    pub n_active_faces: usize,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub max_shell: usize,
}

#[derive(Default)]
pub(crate) struct FastHasher(u64);

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0_u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// Normalized `SU(1,1)` element `(a, b)` with `|a|^2 - |b|^2 = 1`: `[a_re, a_im, b_re, b_im]`.
pub(crate) type Su11 = [f64; 4];

pub(crate) const SU_ID: Su11 = [1.0, 0.0, 0.0, 0.0];

#[inline]
pub(crate) fn su_mul(m1: Su11, m2: Su11) -> Su11 {
    let (a1r, a1i, b1r, b1i) = (m1[0], m1[1], m1[2], m1[3]);
    let (a2r, a2i, b2r, b2i) = (m2[0], m2[1], m2[2], m2[3]);
    // a = a1*a2 + b1*conj(b2)
    let ar = (a1r * a2r - a1i * a2i) + (b1r * b2r + b1i * b2i);
    let ai = (a1r * a2i + a1i * a2r) + (b1i * b2r - b1r * b2i);
    // b = a1*b2 + b1*conj(a2)
    let br = (a1r * b2r - a1i * b2i) + (b1r * a2r + b1i * a2i);
    let bi = (a1r * b2i + a1i * b2r) + (b1i * a2r - b1r * a2i);
    let n2 = (ar * ar + ai * ai - (br * br + bi * bi)).max(1e-18);
    let inv = 1.0 / n2.sqrt();
    [ar * inv, ai * inv, br * inv, bi * inv]
}

#[inline]
pub(crate) fn su_origin(m: Su11) -> [f64; 2] {
    // z = b / conj(a)
    let den = (m[0] * m[0] + m[1] * m[1]).max(1e-18);
    [(m[2] * m[0] - m[3] * m[1]) / den, (m[3] * m[0] + m[2] * m[1]) / den]
}

#[inline]
pub(crate) fn poincare_to_su_trans(z: [f64; 2]) -> Option<Su11> {
    let r2 = z[0] * z[0] + z[1] * z[1];
    if !z[0].is_finite() || !z[1].is_finite() || r2 >= 1.0 {
        return None;
    }
    let inv_s = 1.0 / (1.0 - r2).sqrt();
    Some([inv_s, 0.0, z[0] * inv_s, z[1] * inv_s])
}

/// Polar hyperbolic integer-grid spatial hash in the Poincare disk (`O(1)` points per cell).
#[derive(Clone, Debug)]
pub(crate) struct PoincarePolarHash {
    delta_rho: f64,
    tol: f64,
    tol_sh: f64,
    tol_sh2: f64,
    max_br: i64,
    pub(crate) items: Vec<Su11>,
    pub(crate) z_pts: Vec<[f64; 2]>,
    next: Vec<u32>,
    shells: Vec<(i64, f64)>,
    buckets: FastMap<u64, u32>,
}

#[inline]
fn wrap_sector(s: i64, nsec: i64) -> i64 {
    if s < 0 {
        s + nsec
    } else if s >= nsec {
        s - nsec
    } else {
        s
    }
}

#[inline]
fn pack_key(br: i64, bt: i64) -> u64 {
    ((br as u64) << 32) | ((bt as u64) & 0xffff_ffff)
}

/// `sinh^2(d / 2)` for `d` the hyperbolic distance between `m1(0)` and `m2(0)`: it is
/// `|(m1^{-1} m2).b|^2`, with `(m1^{-1} m2).b = conj(a1) b2 - b1 conj(a2)` (for `|a|^2 - |b|^2 = 1`,
/// `m(0) = b / conj(a)` lies at distance `2 asinh |b|` from the origin).
#[inline]
fn su_half_sinh2(m1: Su11, m2: Su11) -> f64 {
    let (a1r, a1i, b1r, b1i) = (m1[0], m1[1], m1[2], m1[3]);
    let (a2r, a2i, b2r, b2i) = (m2[0], m2[1], m2[2], m2[3]);
    let dr = (a1r * b2r + a1i * b2i) - (b1r * a2r + b1i * a2i);
    let di = (a1r * b2i - a1i * b2r) - (b1i * a2r - b1r * a2i);
    dr * dr + di * di
}

impl PoincarePolarHash {
    pub(crate) fn new(min_sep: f64) -> Self {
        Self::with_capacity(min_sep, 0)
    }

    pub(crate) fn with_capacity(min_sep: f64, cap: usize) -> Self {
        let sep = min_sep.max(1e-4);
        let tol = 0.5 * sep;
        let tol_sh = (0.5 * tol).sinh();
        let mut h = Self {
            delta_rho: sep,
            tol,
            tol_sh,
            tol_sh2: tol_sh * tol_sh,
            max_br: -1,
            items: Vec::with_capacity(cap),
            z_pts: Vec::with_capacity(cap),
            next: Vec::with_capacity(cap),
            shells: Vec::new(),
            buckets: FastMap::default(),
        };
        if cap > 0 {
            h.buckets.reserve(cap);
        }
        h.ensure_shell(8);
        h
    }

    #[inline]
    fn shell_params(&self, br: i64) -> (i64, f64) {
        if let Some(&cached) = self.shells.get(br as usize) {
            return cached;
        }
        if br <= 0 {
            return (1, 0.0);
        }
        let rho_mid = (br as f64 + 0.5) * self.delta_rho;
        let circ = TAU * rho_mid.sinh();
        let nsec = (circ / self.delta_rho).floor().clamp(1.0, 1_000_000_000.0) as i64;
        (nsec, ((br as f64) * self.delta_rho).sinh())
    }

    fn ensure_shell(&mut self, br: usize) {
        while self.shells.len() <= br {
            let b = self.shells.len() as i64;
            let val = if b <= 0 {
                (1, 0.0)
            } else {
                let rho_mid = (b as f64 + 0.5) * self.delta_rho;
                let circ = TAU * rho_mid.sinh();
                let nsec = (circ / self.delta_rho).floor().clamp(1.0, 1_000_000_000.0) as i64;
                (nsec, ((b as f64) * self.delta_rho).sinh())
            };
            self.shells.push(val);
        }
    }

    #[inline]
    fn polar_of(m: Su11) -> (f64, f64, f64, [f64; 2]) {
        let b_norm = (m[2] * m[2] + m[3] * m[3]).sqrt();
        let sh_rho = 2.0 * b_norm * (1.0 + b_norm * b_norm).sqrt();
        let rho = 2.0 * b_norm.asinh();
        let z = su_origin(m);
        let th = z[1].atan2(z[0]).rem_euclid(TAU);
        (rho, sh_rho, th, z)
    }

    #[inline]
    fn bucket_of(&mut self, rho: f64, th: f64) -> (i64, i64) {
        let br = (rho / self.delta_rho).floor().max(0.0) as i64;
        self.ensure_shell(br as usize);
        let nsec = self.shells[br as usize].0;
        let bt = ((th / TAU) * (nsec as f64)).floor().rem_euclid(nsec as f64) as i64;
        (br, bt)
    }

    #[inline]
    fn push_bucket(&mut self, br: i64, bt: i64, m: Su11, z: [f64; 2]) -> usize {
        let idx = self.items.len();
        self.items.push(m);
        self.z_pts.push(z);
        self.max_br = self.max_br.max(br);
        let key = pack_key(br, bt);
        let prev = self.buckets.insert(key, idx as u32).unwrap_or(u32::MAX);
        self.next.push(prev);
        idx
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn sector_window(&self, th: f64, rho: f64, sh_rho: f64, r_lo_sh: f64, rad: f64, rad_sh: f64, b: i64) -> (i64, i64, i64) {
        let (nsec, b_inner_sh) = self.shell_params(b);
        let bt_center = ((th / TAU) * (nsec as f64)).floor().rem_euclid(nsec as f64) as i64;
        let span = if rho <= rad + 2.0 * self.tol || b <= 1 {
            nsec
        } else {
            let s_val = rad_sh / (sh_rho * b_inner_sh.max(r_lo_sh)).sqrt().max(1e-6);
            let dth = 2.0 * s_val.min(1.0).asin();
            let d_sec = ((dth / TAU) * (nsec as f64)).ceil() as i64 + 1;
            (2 * d_sec + 1).min(nsec)
        };
        (nsec, bt_center - span / 2, span)
    }

    pub(crate) fn insert(&mut self, m: Su11) -> (usize, bool) {
        let (rho, sh_rho, th, z) = Self::polar_of(m);
        let (br, bt) = self.bucket_of(rho, th);
        let br_min = ((rho - self.tol - 1e-12).max(0.0) / self.delta_rho).floor() as i64;
        let br_max = (((rho + self.tol + 1e-12) / self.delta_rho).floor() as i64).min(self.max_br);
        let mut best: Option<(usize, f64)> = None;
        if br_min <= br_max {
            let r_lo_sh = (rho - self.tol).max(0.0).sinh();
            for b in br_min..=br_max {
                let (nsec, start, span) = self.sector_window(th, rho, sh_rho, r_lo_sh, self.tol, self.tol_sh, b);
                for off in 0..span {
                    let s = wrap_sector(start + off, nsec);
                    let mut cur = self.buckets.get(&pack_key(b, s)).copied().unwrap_or(u32::MAX);
                    while cur != u32::MAX {
                        let d2 = su_half_sinh2(self.items[cur as usize], m);
                        if d2 < self.tol_sh2 && best.is_none_or(|(_, bd)| d2 < bd) {
                            best = Some((cur as usize, d2));
                        }
                        cur = self.next[cur as usize];
                    }
                }
            }
        }
        if let Some((idx, _)) = best {
            return (idx, false);
        }
        (self.push_bucket(br, bt, m, z), true)
    }

    /// Push a known-distinct item (used by `GraphCombinatorial` / `DunhamHyperboloid`) into the spatial index.
    pub(crate) fn push_unchecked(&mut self, m: Su11) -> usize {
        let (rho, _sh_rho, th, z) = Self::polar_of(m);
        let (br, bt) = self.bucket_of(rho, th);
        self.push_bucket(br, bt, m, z)
    }

    /// Return indices of all items within hyperbolic distance `<= radius` of `center_su`.
    pub(crate) fn ball_query(&self, center_su: Su11, radius: f64) -> Vec<usize> {
        let (rho0, sh_rho0, th0, _) = Self::polar_of(center_su);
        let br_min = ((rho0 - radius - 1e-12).max(0.0) / self.delta_rho).floor() as i64;
        let br_max = (((rho0 + radius + 1e-12) / self.delta_rho).floor() as i64).min(self.max_br);
        let mut out = Vec::new();
        if br_min > br_max {
            return out;
        }
        let rad_sh = (0.5 * radius).sinh();
        let rad_sh2 = rad_sh * rad_sh + 1e-15;
        let r_lo_sh = (rho0 - radius).max(0.0).sinh();
        for b in br_min..=br_max {
            let (nsec, start, span) = self.sector_window(th0, rho0, sh_rho0, r_lo_sh, radius, rad_sh, b);
            for off in 0..span {
                let s = wrap_sector(start + off, nsec);
                let mut cur = self.buckets.get(&pack_key(b, s)).copied().unwrap_or(u32::MAX);
                while cur != u32::MAX {
                    if su_half_sinh2(center_su, self.items[cur as usize]) <= rad_sh2 {
                        out.push(cur as usize);
                    }
                    cur = self.next[cur as usize];
                }
            }
        }
        out
    }
}
