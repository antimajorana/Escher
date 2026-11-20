//! Checks run on the bond graph before the detector and the CNA search use it.
//!
//! - `check_plane`: the face walk, Frank charges and surrounding loops need a plane straight-line drawing, so a graph
//!   in which two bonds cross, a site lies on another bond, or two bonds overlap along a line is refused with the pair.
//!   A distance cutoff above sqrt 2 (square) or sqrt 3 (triangular, honeycomb) longest bonds produces such a graph.
//! - `cna_histogram`: the core's longest-chain search enumerates every edge-simple trail of the common-neighbour
//!   subgraph of a bond, which is exponential in general. It is run only when every such subgraph has maximum degree
//!   at most 2 (paths and cycles: at most two trails per start, so the search is polynomial). In a plane graph this
//!   always holds: if the common neighbours of bond (i, j) contained a site c bonded to three others x, y, z, then
//!   {i, j, c} and {x, y, z} would span K_{3,3}; a cycle among them plus i and j contracts to K_5.

use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::HashSet;

#[cfg(test)]
use papertilling_core::defect_fields::{BondGraph, CnaSignature, cna_2d};

/// Relative tolerance of the crossing test (orientations below `REL_TOL * scale^2` count as collinear).
pub(crate) const REL_TOL: f64 = 1e-9;

/// CNA signature counts, most frequent first.
#[cfg(test)]
pub(crate) type CnaCounts = Vec<(CnaSignature, usize)>;

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn len(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

/// Sign of `cross(q - p, r - p)`, with 0 inside the tolerance.
fn orient(p: [f64; 2], q: [f64; 2], r: [f64; 2], tol: f64) -> i8 {
    let o = cross(sub(q, p), sub(r, p));
    if o > tol {
        1
    } else if o < -tol {
        -1
    } else {
        0
    }
}

/// Whether `r` (collinear with `pq`) lies in the closed bounding box of `pq`, widened by `tl`.
fn within(p: [f64; 2], q: [f64; 2], r: [f64; 2], tl: f64) -> bool {
    (0..2).all(|k| r[k] >= p[k].min(q[k]) - tl && r[k] <= p[k].max(q[k]) + tl)
}

/// Whether segments `ab` and `cd` (four distinct sites) cross, touch or overlap.
pub(crate) fn segments_meet(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let scale = len(sub(b, a)).max(len(sub(d, c)));
    let tol = REL_TOL * scale * scale;
    let (s1, s2, s3, s4) = (orient(a, b, c, tol), orient(a, b, d, tol), orient(c, d, a, tol), orient(c, d, b, tol));
    if s1 * s2 < 0 && s3 * s4 < 0 {
        return true;
    }
    let tl = REL_TOL * scale;
    (s1 == 0 && within(a, b, c, tl)) || (s2 == 0 && within(a, b, d, tl)) || (s3 == 0 && within(c, d, a, tl)) || (s4 == 0 && within(c, d, b, tl))
}

/// Whether bonds `ab` and `ac` (sharing site `a`) overlap: collinear and pointing the same way.
fn overlap_at(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let (u, v) = (sub(b, a), sub(c, a));
    let s = len(u).max(len(v));
    cross(u, v).abs() <= REL_TOL * s * s && u[0] * v[0] + u[1] * v[1] > 0.0
}

fn bonds_meet(pos: &[[f64; 2]], (a, b): (usize, usize), (c, d): (usize, usize)) -> bool {
    if (a == c && b == d) || (a == d && b == c) {
        false
    } else if a == c {
        overlap_at(pos[a], pos[b], pos[d])
    } else if a == d {
        overlap_at(pos[a], pos[b], pos[c])
    } else if b == c {
        overlap_at(pos[b], pos[a], pos[d])
    } else if b == d {
        overlap_at(pos[b], pos[a], pos[c])
    } else {
        segments_meet(pos[a], pos[b], pos[c], pos[d])
    }
}

/// Refuse (with the first offending pair) a graph whose straight-line drawing is not plane. Edges must index `pos`.
/// Bonds are bucketed by bounding box on a grid of the longest bond's size, so each bond touches at most 2 x 2 cells
/// and two bonds that meet share a cell.
pub(crate) fn check_plane(pos: &[[f64; 2]], edges: &[(usize, usize)]) -> Result<(), String> {
    let mut h = 0.0_f64;
    for &(u, v) in edges {
        let l = len(sub(pos[v], pos[u]));
        if !(l.is_finite() && l > 0.0) {
            return Err(format!("REFUSED: bond {u}-{v} has zero or non-finite length (coincident sites)"));
        }
        h = h.max(l);
    }
    let cell = |x: f64| (x / h).floor() as i64;
    let mut grid: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (k, &(u, v)) in edges.iter().enumerate() {
        let (p, q) = (pos[u], pos[v]);
        for cx in cell(p[0].min(q[0]))..=cell(p[0].max(q[0])) {
            for cy in cell(p[1].min(q[1]))..=cell(p[1].max(q[1])) {
                grid.entry((cx, cy)).or_default().push(k);
            }
        }
    }
    for list in grid.values() {
        for (x, &e) in list.iter().enumerate() {
            for &f in &list[x + 1..] {
                if bonds_meet(pos, edges[e], edges[f]) {
                    let ((a, b), (c, d)) = (edges[e], edges[f]);
                    return Err(format!(
                        "REFUSED: the bond graph is not plane: bonds {a}-{b} and {c}-{d} cross, touch or overlap \
                         (faces, Frank charges and loops need a plane drawing)"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// CNA signature counts (most frequent first), or the reason the longest-chain search was not run.
#[cfg(test)]
pub(crate) fn cna_histogram(g: &BondGraph) -> Result<CnaCounts, String> {
    let sets: Vec<HashSet<usize>> = g.nbrs.iter().map(|l| l.iter().copied().collect()).collect();
    for &(i, j) in &g.edges {
        let common: Vec<usize> = sets[i].intersection(&sets[j]).copied().collect();
        let mut deg = vec![0_usize; common.len()];
        for (a, &x) in common.iter().enumerate() {
            for (b, &y) in common.iter().enumerate().skip(a + 1) {
                if sets[x].contains(&y) {
                    deg[a] += 1;
                    deg[b] += 1;
                }
            }
        }
        if deg.iter().any(|&d| d > 2) {
            return Err(format!(
                "not computed: a common neighbour of bond {i}-{j} is bonded to three or more of the others \
                 (the longest-chain search is exponential there)"
            ));
        }
    }
    let mut hist: BTreeMap<CnaSignature, usize> = BTreeMap::new();
    for s in cna_2d(g).0 {
        *hist.entry(s).or_default() += 1;
    }
    let mut cna: CnaCounts = hist.into_iter().collect();
    cna.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(cna)
}
