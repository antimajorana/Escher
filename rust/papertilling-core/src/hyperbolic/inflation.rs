//! Boyle-Dickens-Flicker vertex completion on a regular hyperbolic `{p, q}` tiling.
//!
//! Reference: L. Boyle, M. Dickens, F. Flicker, "Conformal Quasicrystals and Holography",
//! Phys. Rev. X 10, 011009 (2020), arXiv:1805.02665 (v2), Sec. IV-V. With `S_0` one `p`-gon,
//! `S_{n+1} = S_n + U(S_n)`, `U(S)` = every tile not in `S` sharing a vertex with `S` ("vertex
//! completion"). The paper states it preserves convexity (Sec. IV); no hole filling is done here, and
//! the tests check chi = 1 and no pinch vertex for each `S_n` on the tested tilings only.
//!
//! Geometry is done in the hyperboloid model (`t^2 - x^2 - y^2 = 1`) with faces stored as Lorentz
//! matrices `g` (face = `g` applied to the base face centred at `(0, 0, 1)`). The neighbour of face
//! `g` across its own edge `k` is `g * s_k`, where `s_k` reflects the base face across its edge `k`.
//! Face centres are deduplicated by hyperbolic distance below half the minimum centre separation
//! (`2 r`, `r` the inradius); vertices likewise against half the edge length.
//!
//! Precision (what is and is not checked). The f64 guards ([`BaseFace::new`] closure screen,
//! [`HypPointSet`] band guard, `T_PRECISION_LIMIT`) all measure distances with [`hyp_dist`], which
//! sees only the displacement along the hyperboloid: drift off it (`<x, x> != -1`) is invisible
//! (a timelike difference clamps to distance 0). No point is checked to lie on the hyperboloid, so
//! "every computed point is within `l/8` of its exact position" is not verified. Measured 2026-11-18:
//! in the UI range (`p, q <= 12`, `n <= 6`, budget 20000) the last built shell's vertices reach
//! `t ~ 8e6` with `|<x, x> + 1|` up to ~2e2 (some spacelike, in 11 pairs, e.g. {11,9}, {12,12}),
//! and face centres up to 0.3 ({12,12} `S_2`); the drawn patches' points stay below 1e-4. For large `q`
//! (e.g. {6,1082}, {4,2221}) `S_1` has been accepted with face vertices outside the Poincare disk.
//! The counts rest on cross-checks, not on these guards: hand counts, GT and the combinatorial
//! counter (equal to the geometric one for all 92 UI pairs through the table's shells, measured
//! 2026-11-18; tested through `S_2`). Counts beyond the UI range are empirical.
//!
//! An exact integer counter that tracks only the boundary (number of patch faces at each boundary
//! vertex) is included as an independent cross-check; it is exact while the patch stays convex
//! in the paper's sense and returns `None` otherwise.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// Stop growing once a face centre has time coordinate above this: past it the f64 dedup test
/// (cancellation of order `eps * t^2`) is no longer trustworthy. Distance ~ `ln(2e6)` ~ 14.5.
/// It gates face centres only: vertices of the last built shell lie up to ~`2 cosh R` times
/// further out in `t` and are not limited (see the module doc, "Precision").
const T_PRECISION_LIMIT: f64 = 1.0e6;

fn mink(a: Vec3, b: Vec3) -> f64 { a[0] * b[0] + a[1] * b[1] - a[2] * b[2] }
fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 { core::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]) }
fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    core::array::from_fn(|i| core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}
/// Hyperbolic distance from the Minkowski square norm of `a - b` (= `2 (cosh d - 1)`), which avoids
/// the cancellation in `acosh(-<a, b>)` for close points. Non-finite input gives NaN (`f64::max` is
/// not used for the clamp since it would turn NaN into 0). Every finite negative square is clamped to
/// 0, not only tiny rounding: the inputs are not checked to lie on the hyperboloid, and a difference
/// with a timelike part (points drifted off it) can read as a smaller distance, or 0.
pub fn hyp_dist(a: Vec3, b: Vec3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let m = mink(d, d);
    let m = if m < 0.0 && m.is_finite() { 0.0 } else { m };
    2.0 * (0.5 * m.sqrt()).asinh()
}

/// `max` that keeps NaN (`f64::max` returns the non-NaN operand).
fn max_or_nan(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) } }

/// Hyperboloid point to the Poincare disk.
pub fn hyperboloid_to_poincare(x: Vec3) -> [f64; 2] { [x[0] / (1.0 + x[2]), x[1] / (1.0 + x[2])] }

/// `true` when `{p, q}` tiles the hyperbolic plane, i.e. `(p - 2)(q - 2) > 4`.
pub fn is_hyperbolic(p: usize, q: usize) -> bool { p >= 3 && q >= 3 && (p - 2) * (q - 2) > 4 }

/// `gamma_{p,q} = ((p - 2)(q - 2) - 2) / 2`, BDF Eq. (11).
pub fn gamma_pq(p: usize, q: usize) -> f64 {
    ((p as f64 - 2.0) * (q as f64 - 2.0) - 2.0) / 2.0
}

/// `lambda_+ = gamma + sqrt(gamma^2 - 1)`: BDF Eq. (10), the largest eigenvalue of the boundary-letter
/// substitution matrix, i.e. the asymptotic growth ratio of the boundary per vertex completion.
/// It is not an exact tile-count formula. Returns `1.0` for the Euclidean `gamma = 1` cases
/// (`{4,4}`, `{3,6}`, `{6,3}`) and `f64::NAN` for spherical pairs (`gamma < 1`), where the formula
/// has no meaning.
pub fn lambda_plus(p: usize, q: usize) -> f64 {
    let g = gamma_pq(p, q);
    if g < 1.0 {
        return f64::NAN;
    }
    g + (g * g - 1.0).sqrt()
}

/// The base `p`-gon centred at `(0, 0, 1)`, its edge reflections and its vertex stars.
#[derive(Clone, Debug)]
pub struct BaseFace {
    pub p: usize,
    pub q: usize,
    /// Inradius `r`: `cosh r = cos(pi/q) / sin(pi/p)`.
    pub inradius: f64,
    /// Circumradius `R`: `cosh R = cot(pi/p) cot(pi/q)`.
    pub circumradius: f64,
    /// Vertex `j` at polar angle `2 pi j / p`; edge `k` joins vertices `k` and `k + 1`.
    pub vertices: Vec<Vec3>,
    /// Geodesic midpoints of the edges (kagome vertices).
    pub edge_midpoints: Vec<Vec3>,
    /// `s_k`: Lorentz reflection across edge `k`.
    pub reflections: Vec<Mat3>,
    /// `star[j][m]`, `m = 1..q-1`: the faces around vertex `j` are `g * star[j][m]`.
    star: Vec<Vec<Mat3>>,
}

impl BaseFace {
    pub fn new(p: usize, q: usize) -> Result<Self, String> {
        if !is_hyperbolic(p, q) {
            return Err(format!("{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)"));
        }
        let (ap, aq) = (PI / p as f64, PI / q as f64);
        let inradius = (aq.cos() / ap.sin()).acosh();
        let circumradius = (1.0 / (ap.tan() * aq.tan())).acosh();
        let (sr, cr) = (circumradius.sinh(), circumradius.cosh());
        let vertices: Vec<Vec3> = (0..p)
            .map(|j| {
                let th = 2.0 * PI * j as f64 / p as f64;
                [sr * th.cos(), sr * th.sin(), cr]
            })
            .collect();
        let edge_midpoints: Vec<Vec3> = (0..p)
            .map(|k| {
                let s: Vec3 = core::array::from_fn(|i| vertices[k][i] + vertices[(k + 1) % p][i]);
                let nrm = (-mink(s, s)).sqrt();
                s.map(|c| c / nrm)
            })
            .collect();
        let (shr, chr) = (inradius.sinh(), inradius.cosh());
        let reflections: Vec<Mat3> = (0..p)
            .map(|k| {
                // Unit spacelike normal of the edge geodesic: X -> X - 2 <X, n> n.
                let th = 2.0 * PI * (k as f64 + 0.5) / p as f64;
                let n = [chr * th.cos(), chr * th.sin(), shr];
                let jn = [n[0], n[1], -n[2]];
                core::array::from_fn(|i| core::array::from_fn(|j| IDENTITY[i][j] - 2.0 * n[i] * jn[j]))
            })
            .collect();
        let origin = [0.0, 0.0, 1.0];
        let edge = hyp_dist(vertices[0], vertices[1 % p]);
        // Error budget (a design target, not a verified bound). Distinct face centres are >= 2r apart
        // and distinct vertices >= l apart (l, the edge length, is the least distance between distinct
        // vertices of a regular tiling); the dedup merges below half of that. IF every computed point
        // were within e of its exact position, copies would be < 2e apart and distinct points
        // > sep - 2e, so e < sep / 8 would keep every pair out of the band [sep / 4, 3 sep / 4] (the
        // band [`HypPointSet`] reports as suspect). Hence the tolerances r / 4 (centres), l / 8
        // (vertices). What is checked: the closing face only (assuming rounding grows along the walk),
        // with `hyp_dist`, which sees the error along the hyperboloid but not drift off it (see the
        // module doc, "Precision"). So the premise is screened, not established, for the star faces.
        let (tol_c, tol_v) = (0.25 * inradius, 0.125 * edge);
        let mut star = Vec::with_capacity(p);
        for j in 0..p {
            // Vertex j lies on edges j and j-1; walk around it by alternating the two reflections.
            let (a, b) = (j, (j + p - 1) % p);
            let mut w = IDENTITY;
            let mut ring = Vec::with_capacity(q);
            // Every face of the walk contains vertex j (both reflections fix it): its drift measures
            // an error in the rotation centre, which the closure test below cannot see.
            let mut drift = 0.0_f64;
            for m in 0..q {
                w = mat_mul(&w, &reflections[if m % 2 == 0 { a } else { b }]);
                drift = max_or_nan(drift, hyp_dist(mat_vec(&w, vertices[j]), vertices[j]));
                if m + 1 < q {
                    ring.push(w);
                }
            }
            // Star closure: after exactly q faces the walk is back on the base face, as the identity
            // (q even) or as the reflection fixing vertex j (q odd: vertex k -> vertex 2j - k). The
            // vertex error of this last face is ~100x its centre error for large q (an angle error
            // about the centre moves points at distance R by ~sinh R times more), so both are tested.
            // Rounding grows with q and R; a miss (or NaN) means f64 cannot resolve the vertex star.
            // A pass does not mean the walk matrices are Lorentz: for large q (e.g. {6,1082},
            // {4,2221}) it has passed with star-face vertices off the hyperboloid, outside the disk.
            let back = hyp_dist(mat_vec(&w, origin), origin);
            let vert_err = (0..p)
                .map(|k| {
                    let k_img = if q.is_multiple_of(2) { k } else { (2 * j + p - k) % p };
                    hyp_dist(mat_vec(&w, vertices[k]), vertices[k_img])
                })
                .fold(drift, max_or_nan);
            if back.is_nan() || back >= tol_c || vert_err.is_nan() || vert_err >= tol_v {
                return Err(format!(
                    "{{{p},{q}}}: vertex star does not close within f64 precision (centre off by \
                     {back:.3e}, tolerance r/4 = {tol_c:.3e}; vertices off by {vert_err:.3e}, \
                     tolerance l/8 = {tol_v:.3e}); q is too large for this construction"
                ));
            }
            star.push(ring);
        }
        Ok(Self { p, q, inradius, circumradius, vertices, edge_midpoints, reflections, star })
    }

    /// Hyperbolic edge length `l`: `cosh(l/2) = cos(pi/p) / sin(pi/q)`.
    pub fn edge_length(&self) -> f64 {
        hyp_dist(self.vertices[0], self.vertices[1 % self.p])
    }
}

/// Multiply-rotate hash for the integer grid keys (27 lookups per insert; the default SipHash was
/// the main cost). Not DoS-resistant, which does not matter for computed keys.
#[derive(Default)]
struct CellHasher(u64);

impl Hasher for CellHasher {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0_u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u64(&mut self, x: u64) { self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95); }
    fn write_usize(&mut self, x: usize) { self.write_u64(x as u64) }
    fn finish(&self) -> u64 { self.0 }
}

/// Points on the hyperboloid deduplicated by hyperbolic distance (grid hash on ambient `R^3`).
///
/// Distinct points at hyperbolic distance `>= min_sep` have ambient chord `>= 2 sinh(min_sep/2)`
/// (the Euclidean norm dominates the Minkowski one), which is the grid cell size.
///
/// Suspect inserts: if every point were within `e` of its exact position, copies of one point would
/// be `< 2e` apart and distinct points `> min_sep - 2e`, so the merge threshold `min_sep / 2` is safe
/// while no nearest-neighbour distance falls in the band `[min_sep / 4, 3 min_sep / 4]` (it never
/// does for `e < min_sep / 8`). An insert whose nearest grid neighbour lies in that band, or whose
/// point is not finite, is counted in `suspect`; callers treat `suspect > 0` as a precision failure.
/// Only grid neighbours are seen, so this is a guard on the merges actually made, not a proof. It
/// measures distances with [`hyp_dist`] and does not check that inserted points lie on the
/// hyperboloid, so it cannot see points that drifted off it (they may even be spacelike; see the
/// module doc, "Precision"): a guard on the observed distances, not on the `e < min_sep / 8` premise.
pub struct HypPointSet {
    cell: f64,
    tol: f64,
    pub pts: Vec<Vec3>,
    grid: HashMap<[i64; 3], Vec<u32>, BuildHasherDefault<CellHasher>>,
    /// Largest distance between two points that were merged (numerical duplicates).
    pub max_merged: f64,
    /// Smallest distance between two points kept distinct (among grid neighbours).
    pub min_distinct: f64,
    /// Inserts with a nearest-neighbour distance in `[min_sep / 4, 3 min_sep / 4]` or a non-finite point.
    pub suspect: usize,
}

impl HypPointSet {
    /// Merge points closer than `min_sep / 2`.
    pub fn new(min_sep: f64) -> Self {
        Self {
            cell: 2.0 * (0.5 * min_sep).sinh(),
            tol: 0.5 * min_sep,
            pts: Vec::new(),
            grid: HashMap::default(),
            max_merged: 0.0,
            min_distinct: f64::INFINITY,
            suspect: 0,
        }
    }

    fn key(&self, x: Vec3) -> [i64; 3] {
        x.map(|c| (c / self.cell).floor() as i64)
    }

    /// Index of the point within tolerance of `x`, or insert it. Returns `(index, inserted)`.
    /// A non-finite `x` is stored (so the index is valid) but kept out of the grid and counted in
    /// `suspect`.
    pub fn insert(&mut self, x: Vec3) -> (usize, bool) {
        if !x.iter().all(|c| c.is_finite()) {
            self.suspect += 1;
            self.pts.push(x);
            return (self.pts.len() - 1, true);
        }
        let k = self.key(x);
        let mut best: Option<(usize, f64)> = None;
        for o in 0..27_i64 {
            let cell = [k[0] + o % 3 - 1, k[1] + (o / 3) % 3 - 1, k[2] + o / 9 - 1];
            for &i in self.grid.get(&cell).into_iter().flatten() {
                let d = hyp_dist(self.pts[i as usize], x);
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((i as usize, d));
                }
            }
        }
        if let Some((i, d)) = best {
            if d.is_nan() || (d >= 0.5 * self.tol && d <= 1.5 * self.tol) {
                self.suspect += 1;
            }
            if d < self.tol {
                self.max_merged = self.max_merged.max(d);
                return (i, false);
            }
            self.min_distinct = self.min_distinct.min(d);
        }
        let idx = self.pts.len();
        self.pts.push(x);
        self.grid.entry(k).or_default().push(idx as u32);
        (idx, true)
    }
}

/// A vertex-completion patch: faces as Lorentz matrices, in shell order.
pub struct VcPatch {
    pub base: BaseFace,
    pub faces: Vec<Mat3>,
    /// Shell index of each face (0 = seed).
    pub shell: Vec<usize>,
    /// `a_n = |S_n| - |S_{n-1}|`, `a_0 = 1`; only fully built shells.
    pub shell_sizes: Vec<usize>,
    /// Measured dedup gap for face centres: largest merged / smallest kept-distinct distance.
    pub max_merged_centre: f64,
    pub min_distinct_centre: f64,
    /// Why growth stopped before `n_max`, if it did.
    pub stopped: Option<String>,
}

impl VcPatch {
    /// Face centre of face `i` on the hyperboloid.
    pub fn centre(&self, i: usize) -> Vec3 {
        mat_vec(&self.faces[i], [0.0, 0.0, 1.0])
    }
    /// Vertices of face `i` on the hyperboloid, in cyclic order.
    pub fn face_vertices(&self, i: usize) -> Vec<Vec3> {
        self.base.vertices.iter().map(|&v| mat_vec(&self.faces[i], v)).collect()
    }
    /// Edge midpoints of face `i` on the hyperboloid, in cyclic order.
    pub fn face_edge_midpoints(&self, i: usize) -> Vec<Vec3> {
        self.base.edge_midpoints.iter().map(|&v| mat_vec(&self.faces[i], v)).collect()
    }
}

/// Grow `S_0 .. S_{n_max}` from one `p`-gon. Before each shell, growth stops if the cumulative count
/// plus an estimate of the next shell (`p (q - 2)` for shell 1, else `2 gamma a_n`) exceeds
/// `face_budget`, or if a face centre passes the f64 precision limit, or if the centre or vertex
/// dedup becomes ambiguous ([`HypPointSet::suspect`]; the vertices of every built face, the last
/// shell included, are deduplicated); `stopped` says which, and the partial shell is dropped.
/// No stop means none of these guards fired, not that every point is accurate: the guards are blind
/// to drift off the hyperboloid, which in the UI range already makes some last-shell vertices
/// spacelike (module doc, "Precision"). The last shell's vertices are never expanded, so they enter
/// the counts only by possibly triggering a stop; the counts are backed by the cross-checks there.
pub fn grow_vertex_completion(p: usize, q: usize, n_max: usize, face_budget: usize) -> Result<VcPatch, String> {
    let base = BaseFace::new(p, q)?;
    let mut centres = HypPointSet::new(2.0 * base.inradius);
    let mut verts = HypPointSet::new(base.edge_length());
    // Vertex indices of every face (`p` per face, in `verts`), and `completed[v]`: vertex v's star
    // has been added (v is a vertex of an expanded face).
    let (mut face_verts, mut completed): (Vec<u32>, Vec<bool>) = (Vec::new(), Vec::new());
    let add_vertices = |verts: &mut HypPointSet, face_verts: &mut Vec<u32>, completed: &mut Vec<bool>, g: &Mat3| {
        for &v in &base.vertices {
            let (vi, new) = verts.insert(mat_vec(g, v));
            if new {
                completed.push(false);
            }
            face_verts.push(vi as u32);
        }
    };
    centres.insert([0.0, 0.0, 1.0]);
    add_vertices(&mut verts, &mut face_verts, &mut completed, &IDENTITY);
    let (mut faces, mut shell, mut shell_sizes, mut stopped) = (vec![IDENTITY], vec![0_usize], vec![1_usize], None);
    let two_gamma = 2.0 * gamma_pq(p, q);
    let star_centres: Vec<Vec<Vec3>> =
        base.star.iter().map(|ring| ring.iter().map(|w| [w[0][2], w[1][2], w[2][2]]).collect()).collect();
    for n in 0..n_max {
        let a_n = shell_sizes[n];
        let estimate = if n == 0 { p * (q - 2) } else { (two_gamma * a_n as f64).ceil() as usize };
        if faces.len() + estimate > face_budget {
            stopped = Some(format!("face budget {face_budget}: shell {} (~{estimate} faces) not built", n + 1));
            break;
        }
        let (start, end) = (faces.len() - a_n, faces.len());
        let mut precision_hit = None;
        'shell: for fi in start..end {
            let g = faces[fi];
            for j in 0..p {
                let vi = face_verts[fi * p + j] as usize;
                if completed[vi] {
                    continue; // vertex already completed (it belongs to an earlier expanded face)
                }
                completed[vi] = true;
                for (m, &c0) in star_centres[j].iter().enumerate() {
                    let c = mat_vec(&g, c0);
                    // Written so that a NaN coordinate also stops.
                    if !c.iter().all(|x| x.is_finite()) || c[2] > T_PRECISION_LIMIT {
                        precision_hit = Some("f64 precision limit");
                        break 'shell;
                    }
                    if centres.insert(c).1 {
                        let h = mat_mul(&g, &base.star[j][m]);
                        add_vertices(&mut verts, &mut face_verts, &mut completed, &h);
                        faces.push(h);
                        shell.push(n + 1);
                    }
                }
            }
        }
        if precision_hit.is_none() && centres.suspect + verts.suspect > 0 {
            precision_hit = Some("f64 precision: ambiguous dedup (a distance within [sep/4, 3 sep/4])");
        }
        if let Some(why) = precision_hit {
            faces.truncate(end);
            shell.truncate(end);
            stopped = Some(format!("{why}: shell {} not built", n + 1));
            break;
        }
        shell_sizes.push(faces.len() - end);
    }
    let (max_merged_centre, min_distinct_centre) = (centres.max_merged, centres.min_distinct);
    Ok(VcPatch { base, faces, shell, shell_sizes, max_merged_centre, min_distinct_centre, stopped })
}

/// One row of the shell table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellStats {
    pub n: usize,
    /// `|S_n|`, `a_n = |S_n| - |S_{n-1}|` (`a_0 = 1`) and `a_n / a_{n-1}` (`n >= 1`).
    pub cumulative: usize,
    pub shell: usize,
    pub ratio: Option<f64>,
}

/// Exact shell counts of vertex completion (no radius or drawing cap).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellCounts {
    pub p: usize,
    pub q: usize,
    pub shells: Vec<ShellStats>,
    pub lambda_plus: f64,
    pub max_merged_centre: f64,
    pub min_distinct_centre: f64,
    pub stopped: Option<String>,
}

/// Default compute budget of [`vertex_completion_counts`] (faces kept in memory).
pub const COUNT_FACE_BUDGET: usize = 2_000_000;

/// Per-shell statistics of `S_0 .. S_{n_max}` for one `p`-gon seed (stops early only on the
/// [`COUNT_FACE_BUDGET`] or the precision limit; see `stopped`).
pub fn vertex_completion_counts(p: usize, q: usize, n_max: usize) -> Result<ShellCounts, String> {
    vertex_completion_counts_budget(p, q, n_max, COUNT_FACE_BUDGET)
}

/// [`vertex_completion_counts`] with an explicit face budget (for interactive use).
pub fn vertex_completion_counts_budget(p: usize, q: usize, n_max: usize, budget: usize) -> Result<ShellCounts, String> {
    let patch = grow_vertex_completion(p, q, n_max, budget)?;
    let mut cumulative = 0;
    let shells = patch
        .shell_sizes
        .iter()
        .enumerate()
        .map(|(n, &a)| {
            cumulative += a;
            let ratio = (n >= 1).then(|| a as f64 / patch.shell_sizes[n - 1] as f64);
            ShellStats { n, cumulative, shell: a, ratio }
        })
        .collect();
    Ok(ShellCounts {
        p,
        q,
        shells,
        lambda_plus: lambda_plus(p, q),
        max_merged_centre: patch.max_merged_centre,
        min_distinct_centre: patch.min_distinct_centre,
        stopped: patch.stopped,
    })
}

/// Exact integer shell counts `[a_0, .., a_{n_max}]` from the boundary alone.
///
/// The boundary is the cyclic list of `k_i` = number of patch faces at boundary vertex `i`. Around
/// vertex `i` there are `o_i = q - k_i` outside faces: the faces across its two boundary edges and
/// `o_i - 2` corner faces; if `o_i = 1` the faces across both edges coincide. A new face touching
/// `m` old boundary edges gets `p - m - 1` new vertices; consecutive new faces share one new vertex.
/// Valid while the patch is convex (BDF Sec. III): `o_i >= 2` for `q > 3`, and no two adjacent
/// `o_i = 1` for `q = 3`. Returns `None` if that fails.
pub fn vertex_completion_counts_combinatorial(p: usize, q: usize, n_max: usize) -> Option<Vec<u64>> {
    if !is_hyperbolic(p, q) {
        return None;
    }
    let mut boundary: Vec<usize> = vec![1; p];
    let mut out = vec![1_u64];
    for _ in 0..n_max {
        let l = boundary.len();
        let o: Vec<usize> = boundary.iter().map(|&k| q.checked_sub(k).filter(|&x| x >= 1)).collect::<Option<_>>()?;
        let convex = if q > 3 { o.iter().all(|&x| x >= 2) } else { (0..l).all(|i| o[i] >= 2 || o[(i + 1) % l] >= 2) };
        if !convex {
            return None;
        }
        // New faces in cyclic order, as their new-vertex counts c = p - m - 1.
        let i0 = (0..l).find(|&i| o[i] >= 2)?;
        let mut c_list: Vec<usize> = Vec::new();
        let mut open_m = 0_usize; // edges touched by the currently open edge face
        for t in 0..l {
            let i = (i0 + t) % l;
            if o[i] == 1 {
                open_m += 1;
                continue;
            }
            if open_m > 0 {
                c_list.push(p.checked_sub(open_m + 1).filter(|&c| c >= 1)?);
            }
            c_list.extend(std::iter::repeat_n(p - 1, o[i] - 2));
            open_m = 1;
        }
        c_list.push(p.checked_sub(open_m + 1).filter(|&c| c >= 1)?);
        out.push(c_list.len() as u64);
        // New boundary: private vertices (k = 1) and shared vertices (k = faces containing them).
        let f0 = (0..c_list.len()).find(|&j| c_list[j] >= 2)?;
        let mut next = Vec::with_capacity(c_list.len() * p);
        let mut pending = 0_usize;
        for t in 0..c_list.len() {
            let c = c_list[(f0 + t) % c_list.len()];
            if c == 1 {
                pending += 1; // its single new vertex is also in the following face
                continue;
            }
            if t > 0 {
                next.push(pending);
            }
            next.extend(std::iter::repeat_n(1, c - 2));
            pending = 2;
        }
        next.push(pending);
        boundary = next;
    }
    Some(out)
}
