//! Disk-model maths of the interactive H² view: orientation-preserving disk isometries (SU(1,1) Mobius maps) for
//! the hyperbolic pan, Klein <-> Poincare maps, the distance from the origin, face / vertex generations of the
//! patch, the Lorentz centroid of a polygon, and Klein-model point-in-polygon picking.
//!
//! Conventions: complex numbers are `[re, im]`; the Poincare point `z` has Klein image `k = 2 z / (1 + |z|^2)` and
//! hyperboloid image `(2 z, 1 + |z|^2) / (1 - |z|^2)`; `d(0, z) = 2 artanh |z|` (curvature -1).

use std::collections::{HashMap, VecDeque};

use papertilling_core::hyperbolic::{HyperbolicTiling, to_hyperboloid, to_klein};

pub type C = [f64; 2];

fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn conj(a: C) -> C {
    [a[0], -a[1]]
}
fn cdiv(a: C, b: C) -> C {
    let d = (b[0] * b[0] + b[1] * b[1]).max(1e-300);
    [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}
fn cadd(a: C, b: C) -> C {
    [a[0] + b[0], a[1] + b[1]]
}
pub fn cabs(a: C) -> f64 {
    a[0].hypot(a[1])
}

/// The disk isometry `z -> (a z + b) / (conj(b) z + conj(a))` with `|a|^2 - |b|^2 = 1` (an element of SU(1,1)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskIsometry {
    pub a: C,
    pub b: C,
}

impl Default for DiskIsometry {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DiskIsometry {
    pub const IDENTITY: Self = Self { a: [1.0, 0.0], b: [0.0, 0.0] };

    /// The hyperbolic translation `z -> (z - c) / (1 - conj(c) z)`, which sends `c` to the origin (`|c| < 1`).
    pub fn to_origin(c: C) -> Self {
        let s = (1.0 - (c[0] * c[0] + c[1] * c[1])).max(1e-12).sqrt();
        Self { a: [1.0 / s, 0.0], b: [-c[0] / s, -c[1] / s] }
    }

    /// The hyperbolic translation `z -> (z + c) / (1 + conj(c) z)`, which sends the origin to `c`.
    pub fn from_origin(c: C) -> Self {
        Self::to_origin([-c[0], -c[1]])
    }

    pub fn apply(&self, z: C) -> C {
        cdiv(cadd(cmul(self.a, z), self.b), cadd(cmul(conj(self.b), z), conj(self.a)))
    }

    /// `next` after `self` (matrix product `M_next M_self`), renormalised to `|a|^2 - |b|^2 = 1`.
    pub fn then(self, next: Self) -> Self {
        let a = cadd(cmul(next.a, self.a), cmul(next.b, conj(self.b)));
        let b = cadd(cmul(next.a, self.b), cmul(next.b, conj(self.a)));
        let det = (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]);
        if !(det.is_finite() && det > 1e-12) {
            return self;
        }
        let s = det.sqrt();
        Self { a: [a[0] / s, a[1] / s], b: [b[0] / s, b[1] / s] }
    }

    #[cfg(test)]
    pub fn inverse(&self) -> Self {
        Self { a: conj(self.a), b: [-self.b[0], -self.b[1]] }
    }
}

pub fn poincare_to_klein(z: C) -> C {
    to_klein(z)
}

/// Inverse of [`poincare_to_klein`]: `z = k / (1 + sqrt(1 - |k|^2))`.
pub fn klein_to_poincare(k: C) -> C {
    let s = 1.0 + (1.0 - (k[0] * k[0] + k[1] * k[1])).max(0.0).sqrt();
    [k[0] / s, k[1] / s]
}

/// Hyperbolic distance from the origin of the Poincare point `z`: `2 artanh |z|`.
pub fn dist_from_origin(z: C) -> f64 {
    2.0 * cabs(z).min(1.0 - 1e-15).atanh()
}

/// Hyperbolic centre of a polygon (Poincare vertices): the normalised sum of the hyperboloid images, mapped back.
/// Exact for a regular polygon (the sum is fixed by its rotation group).
pub fn lorentz_centroid(pts: &[C]) -> C {
    let mut s = [0.0; 3];
    for &z in pts {
        let r2 = (z[0] * z[0] + z[1] * z[1]).min(1.0 - 1e-12);
        let d = 1.0 - r2;
        s[0] += 2.0 * z[0] / d;
        s[1] += 2.0 * z[1] / d;
        s[2] += (1.0 + r2) / d;
    }
    let n = (s[2] * s[2] - s[0] * s[0] - s[1] * s[1]).max(1e-300).sqrt();
    let (x, y, t) = (s[0] / n, s[1] / n, s[2] / n);
    [x / (1.0 + t), y / (1.0 + t)]
}

/// Whether `p` lies inside (or on) the convex polygon `poly` (either orientation).
pub fn point_in_convex(poly: &[C], p: C) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let (mut pos, mut neg) = (false, false);
    for (i, a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        let cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        pos |= cr > 1e-12;
        neg |= cr < -1e-12;
    }
    !(pos && neg)
}

fn for_each_face_key(f: &[usize], share_vertex: bool, mut visit: impl FnMut((usize, usize))) {
    if share_vertex {
        for &u in f {
            visit((u, u));
        }
    } else if f.len() >= 2 {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            visit((a.min(b), a.max(b)));
        }
    }
}

/// Generation (layer) of each face: graph distance from face 0 in the patch's face adjacency (faces sharing an edge;
/// for the kagome style, whose faces are the midpoint polygons of the `{p,q}` faces, sharing a vertex). This is the
/// reflection depth at which the core's breadth-first construction reached the face.
pub fn face_generations(faces: &[Vec<usize>], share_vertex: bool) -> Vec<usize> {
    let mut gens = vec![usize::MAX; faces.len()];
    if faces.is_empty() {
        return gens;
    }
    let mut by_key: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for_each_face_key(f, share_vertex, |key| by_key.entry(key).or_default().push(fi));
    }
    gens[0] = 0;
    let mut queue = VecDeque::from([0]);
    while let Some(i) = queue.pop_front() {
        let next_gen = gens[i] + 1;
        for_each_face_key(&faces[i], share_vertex, |key| {
            if let Some(nbrs) = by_key.get(&key) {
                for &j in nbrs {
                    if gens[j] == usize::MAX {
                        gens[j] = next_gen;
                        queue.push_back(j);
                    }
                }
            }
        });
    }
    gens
}

/// Generation of each vertex: the smallest generation of a face containing it.
pub fn vertex_generations(n_vertices: usize, faces: &[Vec<usize>], face_gen: &[usize]) -> Vec<usize> {
    let mut g = vec![usize::MAX; n_vertices];
    for (f, face) in faces.iter().enumerate() {
        for &u in face {
            if u < n_vertices {
                g[u] = g[u].min(face_gen[f]);
            }
        }
    }
    g
}

/// The tiling moved by `iso`: Poincare points and sampled geodesic arcs mapped (Mobius maps send geodesics to
/// geodesics, so the image is again a valid `{p,q}` patch), Klein and hyperboloid points recomputed from them.
pub fn view_tiling(base: &HyperbolicTiling, iso: DiskIsometry) -> HyperbolicTiling {
    let mut t = base.clone();
    if iso == DiskIsometry::IDENTITY {
        return t;
    }
    t.poincare_pts = base.poincare_pts.iter().map(|&z| iso.apply(z)).collect();
    t.klein_pts = t.poincare_pts.iter().copied().map(to_klein).collect();
    t.hyperboloid_pts = t.poincare_pts.iter().copied().map(to_hyperboloid).collect();
    t.edge_arcs = base.edge_arcs.iter().map(|arc| arc.iter().map(|&z| iso.apply(z)).collect()).collect();
    t
}

#[cfg(test)]
mod tests {
    use papertilling_core::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling};

    use super::*;

    /// Hyperbolic distance in the Poincare disk, written independently of the code under test:
    /// `d(z, w) = arccosh(1 + 2 |z - w|^2 / ((1 - |z|^2)(1 - |w|^2)))`.
    fn dist(z: C, w: C) -> f64 {
        let dz = (z[0] - w[0]).powi(2) + (z[1] - w[1]).powi(2);
        let nz = 1.0 - (z[0] * z[0] + z[1] * z[1]);
        let nw = 1.0 - (w[0] * w[0] + w[1] * w[1]);
        (1.0 + 2.0 * dz / (nz * nw)).acosh()
    }

    #[test]
    fn poincare_distance_of_tanh_half_is_one() {
        let r = 0.5_f64.tanh();
        assert!((dist_from_origin([r, 0.0]) - 1.0).abs() < 1e-12);
        assert!((dist_from_origin([0.0, -r]) - 1.0).abs() < 1e-12);
        // agrees with the arccosh formula
        assert!((dist([r, 0.0], [0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn klein_poincare_maps_round_trip() {
        for z in [[0.0, 0.0], [0.3, -0.4], [-0.9, 0.1], [0.05, 0.97]] {
            let k = poincare_to_klein(z);
            // k = 2 z / (1 + |z|^2)
            let n = 1.0 + z[0] * z[0] + z[1] * z[1];
            assert!((k[0] - 2.0 * z[0] / n).abs() < 1e-15 && (k[1] - 2.0 * z[1] / n).abs() < 1e-15);
            let back = klein_to_poincare(k);
            assert!((back[0] - z[0]).abs() < 1e-12 && (back[1] - z[1]).abs() < 1e-12, "{z:?} -> {back:?}");
        }
    }

    #[test]
    fn mobius_pan_preserves_the_disk_and_hyperbolic_distance() {
        let iso = DiskIsometry::to_origin([0.4, 0.3]).then(DiskIsometry::from_origin([-0.2, 0.5]));
        // the moved centre: to_origin sends (0.4, 0.3) to 0, from_origin sends 0 to (-0.2, 0.5)
        let m = iso.apply([0.4, 0.3]);
        assert!((m[0] + 0.2).abs() < 1e-12 && (m[1] - 0.5).abs() < 1e-12);
        // the unit circle maps to itself, the open disk into itself
        for k in 0..16 {
            let t = k as f64 * std::f64::consts::TAU / 16.0;
            assert!((cabs(iso.apply([t.cos(), t.sin()])) - 1.0).abs() < 1e-12);
            assert!(cabs(iso.apply([0.9 * t.cos(), 0.9 * t.sin()])) < 1.0);
        }
        let (z, w) = ([0.1, -0.6], [-0.7, 0.2]);
        let d0 = dist(z, w);
        let d1 = dist(iso.apply(z), iso.apply(w));
        assert!((d0 - d1).abs() < 1e-9, "{d0} vs {d1}");
        let back = iso.inverse().apply(iso.apply(z));
        assert!((back[0] - z[0]).abs() < 1e-12 && (back[1] - z[1]).abs() < 1e-12);
    }

    #[test]
    fn hyperbolic_schlaefli_condition() {
        // (p - 2)(q - 2) > 4: {7,3}: 5 * 1 = 5; {4,5}: 2 * 3 = 6; {4,4} = 4 and {6,3} = 4 are Euclidean.
        let hyp = |p: usize, q: usize| (p - 2) * (q - 2) > 4;
        assert!(hyp(7, 3) && hyp(4, 5));
        assert!(!hyp(4, 4) && !hyp(6, 3));
        assert!(build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 1, 0).is_ok());
        assert!(build_hyperbolic_tiling(4, 5, HyperbolicStyle::RegularPQ, 1, 0).is_ok());
        assert!(build_hyperbolic_tiling(4, 4, HyperbolicStyle::RegularPQ, 1, 0).is_err());
    }

    #[test]
    fn generations_of_a_one_layer_heptagon_patch() {
        // {7,3}, one layer: the central heptagon and its 7 edge neighbours.
        let t = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 1, 0).unwrap();
        assert_eq!(t.faces.len(), 8);
        let g = face_generations(&t.faces, false);
        assert_eq!(g[0], 0);
        assert_eq!(g.iter().filter(|&&x| x == 1).count(), 7);
        let vg = vertex_generations(t.poincare_pts.len(), &t.faces, &g);
        assert_eq!(vg.iter().filter(|&&x| x == 0).count(), 7); // the 7 central vertices
        // the centre of the central regular heptagon is the origin
        let c = lorentz_centroid(&t.faces[0].iter().map(|&u| t.poincare_pts[u]).collect::<Vec<_>>());
        assert!(cabs(c) < 1e-9, "{c:?}");
    }

    #[test]
    fn klein_polygon_test_finds_the_central_face() {
        let t = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 1, 0).unwrap();
        let k: Vec<C> = t.faces[0].iter().map(|&u| t.klein_pts[u]).collect();
        assert!(point_in_convex(&k, [0.0, 0.0]));
        assert!(!point_in_convex(&k, [0.95, 0.0]));
    }
}
