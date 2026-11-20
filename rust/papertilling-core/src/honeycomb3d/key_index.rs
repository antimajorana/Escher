//! Duplicate merging with a measured margin (plan §2.3), for the coset adapter (`growth::h3_complex`).
//!
//! Points are merged when they lie within `tol` of a stored point. Each insert searches every
//! bucket (side `1000 · tol`) that meets the ball of radius `probe = 100 · tol` around the new
//! point, so it also sees every stored point closer than `probe`. Bucket coordinates are
//! hashed to one `u64`; two buckets sharing a hash only share a chain, and every candidate is
//! checked by its distance, so a hash collision cannot cause a wrong merge.
//! Two numbers are kept:
//! - `max_merge`: the largest distance between two representations that were merged (the
//!   round-off actually met);
//! - `min_gap`: the smallest distance between two *distinct* stored points that is below
//!   `probe` (`f64::INFINITY` when no two distinct points came that close).
//!
//! The §2.3 rule "smallest key gap ≥ 100 × tol" holds exactly when `min_gap` is infinite.
//! The adapter turns a finite `min_gap` into `GrowthError::Precision`.
//!
//! Why hyperboloid keys (not Klein keys) in H³: for x, y on the hyperboloid ⟨x,x⟩ = −1 with
//! spatial parts X, Y, the metric is ds² = |dX|² − dx₀² ≤ |dX|², so the straight segment from
//! X to Y lifts to a path of hyperbolic length ≤ |X − Y|, hence |X − Y| ≥ d_H(x, y) [PROVED].
//! Distinct cell centres, vertices, edge midpoints and face centres of a regular honeycomb are
//! a fixed hyperbolic distance apart, so their keys never crowd together at depth, unlike
//! Klein coordinates (1 − r shrinks about 25× per face shell).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A small multiplicative hasher for integer keys (the default SipHash dominated the build
/// time of deep layers). Not DoS-resistant; the keys are our own bucket coordinates.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// `HashMap` with [`FastHasher`].
pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// A tolerance index over points of dimension `dims` (3 or 4), stored as `[f64; 4]`.
pub struct KeyIndex {
    dims: usize,
    tol: f64,
    probe: f64,
    side: f64,
    heads: FastMap<u64, u32>,
    next: Vec<u32>,
    pts: Vec<[f64; 4]>,
    pub max_merge: f64,
    pub min_gap: f64,
}

const NONE: u32 = u32::MAX;

impl KeyIndex {
    pub fn new(dims: usize, tol: f64) -> Self {
        Self {
            dims,
            tol,
            probe: 100.0 * tol,
            side: 1000.0 * tol,
            heads: FastMap::default(),
            next: Vec::new(),
            pts: Vec::new(),
            max_merge: 0.0,
            min_gap: f64::INFINITY,
        }
    }

    pub fn tol(&self) -> f64 {
        self.tol
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    pub fn point(&self, i: u32) -> [f64; 4] {
        self.pts[i as usize]
    }

    fn bucket(&self, x: &[f64; 4]) -> [i64; 4] {
        let mut k = [0i64; 4];
        for d in 0..self.dims {
            k[d] = (x[d] / self.side).floor() as i64;
        }
        k
    }

    fn hash(k: &[i64; 4]) -> u64 {
        let mut h = FastHasher::default();
        for &c in k {
            h.write_u64(c as u64);
        }
        h.finish()
    }

    /// Nearest stored point within `probe` of `x`: `(index, distance)`.
    fn nearest(&self, x: &[f64; 4]) -> Option<(u32, f64)> {
        let mut lo = [0i64; 4];
        let mut hi = [0i64; 4];
        for d in 0..self.dims {
            lo[d] = ((x[d] - self.probe) / self.side).floor() as i64;
            hi[d] = ((x[d] + self.probe) / self.side).floor() as i64;
        }
        let mut best: Option<(u32, f64)> = None;
        let mut k = lo;
        loop {
            let mut i = self.heads.get(&Self::hash(&k)).copied().unwrap_or(NONE);
            while i != NONE {
                let p = &self.pts[i as usize];
                let d2: f64 = (0..self.dims).map(|d| (p[d] - x[d]) * (p[d] - x[d])).sum();
                let dist = d2.sqrt();
                if dist < self.probe && best.is_none_or(|(_, b)| dist < b) {
                    best = Some((i, dist));
                }
                i = self.next[i as usize];
            }
            // odometer over the box lo..=hi
            let mut d = 0;
            loop {
                if d == self.dims {
                    return best;
                }
                if k[d] < hi[d] {
                    k[d] += 1;
                    break;
                }
                k[d] = lo[d];
                d += 1;
            }
        }
    }

    /// Insert `x`, merging with a stored point within `tol`. Returns `(index, inserted)`.
    pub fn insert(&mut self, x: [f64; 4]) -> (u32, bool) {
        if let Some((i, dist)) = self.nearest(&x) {
            if dist <= self.tol {
                self.max_merge = self.max_merge.max(dist);
                return (i, false);
            }
            self.min_gap = self.min_gap.min(dist);
        }
        let i = self.pts.len() as u32;
        let k = self.bucket(&x);
        let head = self.heads.insert(Self::hash(&k), i).unwrap_or(NONE);
        self.next.push(head);
        self.pts.push(x);
        (i, true)
    }

    /// Like [`Self::insert`] without inserting: the stored point within `tol` of `x`, with the
    /// same statistics (a distinct point closer than `probe` lowers `min_gap`).
    pub fn lookup(&mut self, x: &[f64; 4]) -> Option<u32> {
        match self.nearest(x) {
            Some((i, dist)) if dist <= self.tol => {
                self.max_merge = self.max_merge.max(dist);
                Some(i)
            }
            Some((_, dist)) => {
                self.min_gap = self.min_gap.min(dist);
                None
            }
            None => None,
        }
    }

    /// The stored point within `tol` of `x`, if any (no insert, no statistics).
    pub fn find(&self, x: &[f64; 4]) -> Option<u32> {
        self.nearest(x).filter(|&(_, d)| d <= self.tol).map(|(i, _)| i)
    }
}
