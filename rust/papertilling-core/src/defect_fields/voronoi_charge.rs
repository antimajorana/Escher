//! Voronoi coordination `z_i` of a bare point pattern and the coordination (disclination) charge `q_i = z_ideal - z_i`
//! (`#036`): `6 - z` in units of 60 degrees for a triangular crystal, so a 5-fold site is `+1`, a 7-fold site `-1` and
//! a 5-7 dislocation core is neutral (Nelson, *Defects and Geometry in Condensed Matter Physics*, CUP 2002, ch. 1;
//! Bowick & Giomi, Adv. Phys. 58 (2009) 449).
//!
//! Each Voronoi cell is built by clipping a frame (the bounding box grown by `FRAME` times its size) with the
//! perpendicular bisectors of nearby sites, nearest first, until the next site is farther than twice the cell's
//! radius (no farther bisector can cut it). Voronoi edges shorter than `tol` times the site's nearest-neighbour
//! distance are ignored, so cocircular quadruples (the square lattice) do not create spurious neighbours. A cell
//! that still touches the frame is unbounded (its site is on the convex hull) or unreliable: that site is on the
//! rim and gets no charge.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::{norm, sub};

const FRAME: f64 = 1.0e3;
const FRAME_LABEL: usize = usize::MAX;

/// Voronoi neighbours, coordination and charge per site.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoronoiCoordination {
    /// Voronoi neighbours (sites sharing an edge longer than the tolerance), ascending.
    pub neighbours: Vec<Vec<usize>>,
    /// `z_i = neighbours[i].len()`.
    pub z: Vec<usize>,
    /// Whether the cell touches the frame (convex-hull site): no charge.
    pub rim: Vec<bool>,
    /// `z_ideal - z_i` for non-rim sites.
    pub charge: Vec<Option<i64>>,
    /// Sum of the non-rim charges.
    pub total_charge: i64,
}

type Poly = Vec<([f64; 2], usize)>;

/// Keep the part of the convex polygon on the `p_i` side of the bisector of `p_i p_j` (Sutherland-Hodgman); each
/// vertex carries the label of the edge leaving it.
fn clip(poly: &Poly, pi: [f64; 2], pj: [f64; 2], j: usize) -> Poly {
    let d = sub(pj, pi);
    let m = [0.5 * (pi[0] + pj[0]), 0.5 * (pi[1] + pj[1])];
    let eps = 1e-14 * (d[0] * d[0] + d[1] * d[1]);
    let f = |x: [f64; 2]| (x[0] - m[0]) * d[0] + (x[1] - m[1]) * d[1];
    let n = poly.len();
    let mut out = Vec::with_capacity(n + 1);
    for (k, &(a, la)) in poly.iter().enumerate() {
        let (b, _) = poly[(k + 1) % n];
        let (fa, fb) = (f(a), f(b));
        let (ina, inb) = (fa <= eps, fb <= eps);
        if ina {
            out.push((a, la));
        }
        if ina != inb {
            let t = fa / (fa - fb);
            let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            out.push((p, if ina { j } else { la }));
        }
    }
    out
}

/// Voronoi coordination of every site; `z_ideal` is the coordination of the perfect crystal (6 for triangular).
pub fn voronoi_coordination(pos: &[[f64; 2]], z_ideal: usize, tol: f64) -> Result<VoronoiCoordination, String> {
    let n = pos.len();
    if n < 3 {
        return Err("REFUSED: need at least 3 sites".to_string());
    }
    if pos.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return Err("REFUSED: non-finite coordinates".to_string());
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in pos {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if ext <= 0.0 {
        return Err("REFUSED: all sites coincide".to_string());
    }
    let (flo, fhi) = ([lo[0] - FRAME * ext, lo[1] - FRAME * ext], [hi[0] + FRAME * ext, hi[1] + FRAME * ext]);
    let frame: Poly = vec![([flo[0], flo[1]], FRAME_LABEL), ([fhi[0], flo[1]], FRAME_LABEL), ([fhi[0], fhi[1]], FRAME_LABEL), ([flo[0], fhi[1]], FRAME_LABEL)];
    let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    let h = (area / n as f64).sqrt().max(ext / n as f64);
    let key = |p: [f64; 2]| (((p[0] - lo[0]) / h).floor() as i64, ((p[1] - lo[1]) / h).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in pos.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let max_ring = (ext / h).ceil() as i64 + 1;
    let mut neighbours = Vec::with_capacity(n);
    let mut rim = Vec::with_capacity(n);
    for (i, &pi) in pos.iter().enumerate() {
        let (kx, ky) = key(pi);
        let mut ring = 1i64;
        let (poly, dnn) = loop {
            let mut cand: Vec<(f64, usize)> = Vec::new();
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for &j in grid.get(&(kx + dx, ky + dy)).map_or(&[][..], Vec::as_slice) {
                        let d = norm(sub(pos[j], pi));
                        if j != i && d > 0.0 {
                            cand.push((d, j));
                        }
                    }
                }
            }
            cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut poly = frame.clone();
            for &(d, j) in &cand {
                // a site farther than twice the cell radius cannot cut the cell (the radius only shrinks)
                if d > 2.0 * poly.iter().map(|&(v, _)| norm(sub(v, pi))).fold(0.0, f64::max) {
                    break;
                }
                poly = clip(&poly, pi, pos[j], j);
            }
            // every site outside the scanned square is at least `ring * h` away, so the cell is final once
            // `2 * radius <= ring * h`, or once the square covers every site
            let rmax = poly.iter().map(|&(v, _)| norm(sub(v, pi))).fold(0.0, f64::max);
            if 2.0 * rmax <= ring as f64 * h || ring > max_ring {
                break (poly, cand.first().map_or(0.0, |c| c.0));
            }
            ring *= 2;
        };
        let m = poly.len();
        let mut nb: BTreeSet<usize> = BTreeSet::new();
        let mut on_frame = false;
        for (k, &(a, la)) in poly.iter().enumerate() {
            let (b, _) = poly[(k + 1) % m];
            if la == FRAME_LABEL {
                on_frame = true;
            } else if norm(sub(b, a)) > tol * dnn {
                nb.insert(la);
            }
        }
        neighbours.push(nb.into_iter().collect::<Vec<usize>>());
        rim.push(on_frame);
    }
    let z: Vec<usize> = neighbours.iter().map(Vec::len).collect();
    let charge: Vec<Option<i64>> = z.iter().zip(&rim).map(|(&zi, &r)| if r { None } else { Some(z_ideal as i64 - zi as i64) }).collect();
    let total_charge = charge.iter().flatten().sum();
    Ok(VoronoiCoordination { neighbours, z, rim, charge, total_charge })
}
