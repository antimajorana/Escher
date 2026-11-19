//! Wedge-disclination patch (`wedge_delta != 0`): a regular `p_core`-gon at the centre and its copies
//! under reflections across their edges, grown by vertex completion with hyperbolic-distance dedup.
//!
//! Every face is a copy of the core polygon whose edge length matches `{p, q}` (`cosh(l/2) = cos(pi/p)/sin(pi/q)`).
//! When `p_core != p`, its interior angle differs from `2 pi / q`, so the copies do not close up around a
//! vertex (except `p_core == p` when clamped at `p = 3, delta = -1` or `p = 12, delta = +1`); this is a
//! picture of a wedge defect, not a tiling.
//!
//! Growth rule. `S_0` = the core; `S_{n+1}` = `S_n` plus, at every vertex `j` of every face `F` of
//! `S_n \ S_{n-1}`, two star walks of `walk = (q - 1) / 2` alternating edge reflections starting on each side
//! (`1 step` each side for `q = 3, 4`; for `q = 3` this is edge-BFS), plus for even `q` the centred opposite
//! tile obtained by the `180`-degree hyperboloid half-turn `H_{v_j}` about vertex `j`. When the interior angle
//! equals `2 pi / q` this equals BDF vertex completion; for non-closing even-`q` wedges the single opposite
//! copy stays centred and keeps reflection symmetry, leaving two equal gaps on either side.

use std::f64::consts::PI;

use super::inflation::{hyperboloid_to_poincare, mat_mul, mat_vec, HypPointSet, Mat3, Vec3, IDENTITY};

/// A regular `n`-gon centred at `(0, 0, 1)` on the hyperboloid, with its edge reflections and vertex half-turns.
pub(crate) struct Polygon {
    pub n: usize,
    /// Vertex `j` at polar angle `2 pi j / n`; edge `k` joins vertices `k` and `k + 1`.
    pub vertices: Vec<Vec3>,
    pub reflections: Vec<Mat3>,
    /// Orientation-preserving `180`-degree rotation about each vertex `v_j`: `H_v(X) = -X - 2 <X, v> v`.
    pub half_turns: Vec<Mat3>,
}

impl Polygon {
    /// Regular `n`-gon of circumradius `circumradius` (hyperbolic distance centre-vertex).
    pub fn regular(n: usize, circumradius: f64) -> Self {
        // Right triangle centre / edge midpoint / vertex: tanh r = tanh R cos(pi / n).
        let inradius = (circumradius.tanh() * (PI / n as f64).cos()).atanh();
        let (sr, cr) = (circumradius.sinh(), circumradius.cosh());
        let vertices: Vec<Vec3> = (0..n)
            .map(|j| {
                let th = 2.0 * PI * j as f64 / n as f64;
                [sr * th.cos(), sr * th.sin(), cr]
            })
            .collect();
        let (shr, chr) = (inradius.sinh(), inradius.cosh());
        let reflections = (0..n)
            .map(|k| {
                // Unit spacelike normal of edge k's geodesic: X -> X - 2 <X, nv> nv.
                let th = 2.0 * PI * (k as f64 + 0.5) / n as f64;
                let nv = [chr * th.cos(), chr * th.sin(), shr];
                let jn = [nv[0], nv[1], -nv[2]];
                core::array::from_fn(|i| core::array::from_fn(|j| IDENTITY[i][j] - 2.0 * nv[i] * jn[j]))
            })
            .collect();
        let half_turns = vertices
            .iter()
            .map(|&v| {
                let jv = [v[0], v[1], -v[2]];
                core::array::from_fn(|i| core::array::from_fn(|j| -IDENTITY[i][j] - 2.0 * v[i] * jv[j]))
            })
            .collect();
        Self { n, vertices, reflections, half_turns }
    }
}

/// Faces (as Lorentz matrices applied to the core) of a polygon vertex-completion patch, in shell order.
pub(crate) struct PolyGrowth {
    pub faces: Vec<Mat3>,
    pub shell: Vec<usize>,
    /// Largest `n` with `S_n` complete (no face dropped by the radius cap).
    pub complete_through: usize,
    /// Why the last requested shell is missing or partial, if it is.
    pub notes: Vec<String>,
    /// Centre dedup: largest merged and smallest kept-distinct distance, plus split suspect band hits
    /// ([`HypPointSet::suspect`]). The distances are read by the measurement tests only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub max_merged: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    pub min_distinct: f64,
    pub near_coincident_merged: usize,
    pub near_coincident_kept: usize,
}

struct WalkState {
    centres: HypPointSet,
    faces: Vec<Mat3>,
    shell: Vec<usize>,
    max_faces: usize,
    r2_max: f64,
    near_coincident_merged: usize,
    near_coincident_kept: usize,
}

impl WalkState {
    fn try_push(&mut self, poly: &Polygon, w: Mat3, s: usize, shell_truncated: &mut bool) -> bool {
        let origin = [0.0, 0.0, 1.0];
        let c = mat_vec(&w, origin);
        let n = poly.n as f64;
        // Radius cap on the Euclidean centroid of the Poincare vertices.
        let z = poly.vertices.iter().fold([0.0, 0.0], |a, &v| {
            let x = hyperboloid_to_poincare(mat_vec(&w, v));
            [a[0] + x[0] / n, a[1] + x[1] / n]
        });
        let z2 = z[0] * z[0] + z[1] * z[1];
        if z2.is_nan() || z2 > self.r2_max {
            *shell_truncated = true;
            return false;
        }
        let prev_suspect = self.centres.suspect;
        let (_, inserted) = self.centres.insert(c);
        if self.centres.suspect > prev_suspect {
            if inserted {
                self.near_coincident_kept += 1;
            } else {
                self.near_coincident_merged += 1;
            }
        }
        if inserted {
            self.faces.push(w);
            self.shell.push(s + 1);
            if self.faces.len() > self.max_faces {
                return true;
            }
        }
        false
    }
}

/// Grow `S_0 .. S_{n_max}` of copies of `poly` (rule in the module doc). Faces whose Poincare vertex
/// centroid lies beyond radius `max_centre_radius` are dropped and not expanded (this truncates their
/// shell). Only whole shells are kept: if a shell would bring the total above `max_faces`, it is dropped.
/// Within [`HypPointSet`]'s 27-cell ambient-`R^3` neighbourhood (cell side `2 * merge_dist`), centres
/// closer than `merge_dist` (hyperbolic) are merged into one face (near the rim, radial separations can
/// span non-adjacent ambient cells, so only near-exact or tangential duplicates are merged there);
/// near-coincident copies checked in `[merge_dist/2, 3 merge_dist/2]` are split into
/// `near_coincident_merged` (merged) vs `near_coincident_kept` (inserted).
pub(crate) fn grow_polygon_vertex_completion(
    poly: &Polygon,
    q: usize,
    n_max: usize,
    max_faces: usize,
    max_centre_radius: f64,
    merge_dist: f64,
) -> PolyGrowth {
    let n = poly.n;
    let walk = (q - 1) / 2;
    let even_q = q.is_multiple_of(2);
    let mut centres = HypPointSet::new(2.0 * merge_dist);
    centres.insert([0.0, 0.0, 1.0]);
    let mut st = WalkState {
        centres,
        faces: vec![IDENTITY],
        shell: vec![0_usize],
        max_faces,
        r2_max: max_centre_radius * max_centre_radius,
        near_coincident_merged: 0,
        near_coincident_kept: 0,
    };
    let mut notes = Vec::new();
    let mut complete_through = 0;
    let mut truncated = false;
    for s in 0..n_max {
        let start = st.shell.iter().position(|&x| x == s).unwrap_or(st.faces.len());
        let end = st.faces.len();
        let mut over_cap = false;
        let mut shell_truncated = false;
        'shell: for fi in start..end {
            let g = st.faces[fi];
            for j in 0..n {
                for first in [j, (j + n - 1) % n] {
                    let second = if first == j { (j + n - 1) % n } else { j };
                    let mut w = g;
                    for step in 0..walk {
                        w = mat_mul(&w, &poly.reflections[if step.is_multiple_of(2) { first } else { second }]);
                        if st.try_push(poly, w, s, &mut shell_truncated) {
                            over_cap = true;
                            break 'shell;
                        }
                    }
                }
                if even_q && st.try_push(poly, mat_mul(&g, &poly.half_turns[j]), s, &mut shell_truncated) {
                    over_cap = true;
                    break 'shell;
                }
            }
        }
        if over_cap {
            st.faces.truncate(end);
            st.shell.truncate(end);
            notes.push(format!("shell {} omitted by the {max_faces}-face cap", s + 1));
            break;
        }
        truncated |= shell_truncated;
        if st.faces.len() == end {
            break; // every candidate was beyond the radius cap
        }
        if !truncated {
            complete_through = s + 1;
        }
    }
    if truncated {
        notes.push(format!("shell {} truncated by the centre radius cap {max_centre_radius}", complete_through + 1));
    }
    PolyGrowth {
        faces: st.faces,
        shell: st.shell,
        complete_through,
        notes,
        max_merged: st.centres.max_merged,
        min_distinct: st.centres.min_distinct,
        near_coincident_merged: st.near_coincident_merged,
        near_coincident_kept: st.near_coincident_kept,
    }
}

/// Vertices (or edge midpoints, for kagome) of each face, deduplicated by hyperbolic distance below
/// `merge_dist`, as Poincare points and index lists.
pub(crate) fn polygon_faces_to_poincare(
    poly: &Polygon,
    faces: &[Mat3],
    kagome: bool,
    merge_dist: f64,
) -> (Vec<[f64; 2]>, Vec<Vec<usize>>) {
    let n = poly.n;
    let base: Vec<Vec3> = if kagome {
        (0..n)
            .map(|k| {
                let (a, b) = (poly.vertices[k], poly.vertices[(k + 1) % n]);
                let s: Vec3 = core::array::from_fn(|i| a[i] + b[i]);
                let nrm = (s[2] * s[2] - s[0] * s[0] - s[1] * s[1]).sqrt();
                s.map(|c| c / nrm)
            })
            .collect()
    } else {
        poly.vertices.clone()
    };
    let mut set = HypPointSet::new(2.0 * merge_dist);
    let idx = faces.iter().map(|g| base.iter().map(|&v| set.insert(mat_vec(g, v)).0).collect()).collect();
    (set.pts.iter().map(|&x| hyperboloid_to_poincare(x)).collect(), idx)
}
