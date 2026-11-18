//! Wedge-disclination patch (`wedge_delta != 0`): a regular `p_core`-gon at the centre and its copies
//! under reflections across their edges, grown by vertex completion with hyperbolic-distance dedup.
//!
//! Every face is a copy of the core polygon (as in the original edge-BFS construction of e09ee15,
//! whose geometry is kept unchanged). Its interior angle is not `2 pi / q` in general, so the copies
//! do not close up around a vertex: this is a picture of a wedge defect, not a tiling.
//!
//! Growth rule. `S_0` = the core; `S_{n+1}` = `S_n` plus, at every vertex `j` of every face `F` of
//! `S_n \ S_{n-1}`, the faces reached from `F` by alternately reflecting across the two edges of
//! the current face at `j` (the "star walk"), `floor(q/2)` steps starting on each side. When the angle
//! is `2 pi / q` the two walks cover the `q - 1` other faces at `j` (for even `q` both reach the
//! opposite face, deduplicated), so this is BDF vertex completion; for `q = 3` each walk is one step
//! and the rule is exactly edge-BFS (the faces across the two edges at `j`).

use std::f64::consts::PI;

use super::inflation::{hyperboloid_to_poincare, mat_mul, mat_vec, HypPointSet, Mat3, Vec3, IDENTITY};

/// A regular `n`-gon centred at `(0, 0, 1)` on the hyperboloid, with its edge reflections.
pub(crate) struct Polygon {
    pub n: usize,
    /// Vertex `j` at polar angle `2 pi j / n`; edge `k` joins vertices `k` and `k + 1`.
    pub vertices: Vec<Vec3>,
    pub reflections: Vec<Mat3>,
}

impl Polygon {
    /// Regular `n`-gon of circumradius `circumradius` (hyperbolic distance centre-vertex).
    pub fn regular(n: usize, circumradius: f64) -> Self {
        // Right triangle centre / edge midpoint / vertex: tanh r = tanh R cos(pi / n).
        let inradius = (circumradius.tanh() * (PI / n as f64).cos()).atanh();
        let (sr, cr) = (circumradius.sinh(), circumradius.cosh());
        let vertices = (0..n)
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
        Self { n, vertices, reflections }
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
    /// Centre dedup: largest merged and smallest kept-distinct distance, and band-guard hits
    /// ([`HypPointSet::suspect`]). The distances are read by the measurement tests only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub max_merged: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    pub min_distinct: f64,
    pub suspect: usize,
}

/// Grow `S_0 .. S_{n_max}` of copies of `poly` (rule in the module doc). Faces whose centre lies beyond
/// Poincare radius `max_centre_radius` are dropped and not expanded (this truncates their shell). Only
/// whole shells are kept: if a shell would bring the total above `max_faces`, it is dropped.
/// Centres closer than `merge_dist` (hyperbolic) are one face; the grid of [`HypPointSet`] only
/// looks one cell (`~merge_dist` in ambient coordinates) around a point, so far from the origin only
/// near-exact duplicates are found: `max_merged` and `min_distinct` report what actually happened.
pub(crate) fn grow_polygon_vertex_completion(
    poly: &Polygon,
    q: usize,
    n_max: usize,
    max_faces: usize,
    max_centre_radius: f64,
    merge_dist: f64,
) -> PolyGrowth {
    let n = poly.n;
    let walk = q / 2;
    let origin = [0.0, 0.0, 1.0];
    let mut centres = HypPointSet::new(2.0 * merge_dist);
    centres.insert(origin);
    let (mut faces, mut shell, mut notes) = (vec![IDENTITY], vec![0_usize], Vec::new());
    let mut complete_through = 0;
    let mut truncated = false;
    let r2_max = max_centre_radius * max_centre_radius;
    for s in 0..n_max {
        let start = shell.iter().position(|&x| x == s).unwrap_or(faces.len());
        let end = faces.len();
        let mut over_cap = false;
        'shell: for fi in start..end {
            let g = faces[fi];
            for j in 0..n {
                for first in [j, (j + n - 1) % n] {
                    let second = if first == j { (j + n - 1) % n } else { j };
                    let mut w = g;
                    for step in 0..walk {
                        w = mat_mul(&w, &poly.reflections[if step % 2 == 0 { first } else { second }]);
                        let c = mat_vec(&w, origin);
                        // Radius cap on the Euclidean centroid of the Poincare vertices (the
                        // pre-upgrade criterion, kept so that q = 3 reproduces its face set).
                        let z = poly.vertices.iter().fold([0.0, 0.0], |a, &v| {
                            let x = hyperboloid_to_poincare(mat_vec(&w, v));
                            [a[0] + x[0] / n as f64, a[1] + x[1] / n as f64]
                        });
                        let z2 = z[0] * z[0] + z[1] * z[1];
                        if z2.is_nan() || z2 > r2_max {
                            truncated = true;
                            continue;
                        }
                        if centres.insert(c).1 {
                            faces.push(w);
                            shell.push(s + 1);
                            if faces.len() > max_faces {
                                over_cap = true;
                                break 'shell;
                            }
                        }
                    }
                }
            }
        }
        if over_cap {
            faces.truncate(end);
            shell.truncate(end);
            notes.push(format!("shell {} omitted by the {max_faces}-face cap", s + 1));
            break;
        }
        if faces.len() == end {
            break; // every candidate was beyond the radius cap
        }
        if !truncated {
            complete_through = s + 1;
        }
    }
    if truncated {
        notes.push(format!("shell {} truncated by the centre radius cap {max_centre_radius}", complete_through + 1));
    }
    let (max_merged, min_distinct, suspect) = (centres.max_merged, centres.min_distinct, centres.suspect);
    PolyGrowth { faces, shell, complete_through, notes, max_merged, min_distinct, suspect }
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
