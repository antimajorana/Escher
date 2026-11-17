//! Exact periodic 2D Voronoi / Delaunay dual bond extraction via tagged
//! convex-polygon half-plane clipping (matching `scipy.spatial.Voronoi` ridge lengths).

use std::collections::BTreeSet;

use crate::plane::cell::{mat2_inv, mat2_vec};
use crate::types::Cell2D;

#[derive(Clone, Copy)]
struct ClipVert {
    pos: [f64; 2],
    owner: Option<(usize, i32, i32)>,
}

fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn clip_half_plane(
    poly: &[ClipVert],
    mid: [f64; 2],
    normal: [f64; 2],
    new_owner: (usize, i32, i32),
) -> Vec<ClipVert> {
    if poly.is_empty() {
        return Vec::new();
    }
    let val = |p: [f64; 2]| (p[0] - mid[0]) * normal[0] + (p[1] - mid[1]) * normal[1];
    let eps = 1e-11;
    let m = poly.len();
    let mut out = Vec::with_capacity(m + 2);
    for idx in 0..m {
        let cur = poly[idx];
        let nxt = poly[(idx + 1) % m];
        let fa = val(cur.pos);
        let fb = val(nxt.pos);
        let in_a = fa <= eps;
        let in_b = fb <= eps;
        match (in_a, in_b) {
            (true, true) => out.push(cur),
            (true, false) => {
                out.push(cur);
                let denom = fa - fb;
                if denom.abs() > 1e-18 {
                    let t = (fa / denom).clamp(0.0, 1.0);
                    out.push(ClipVert { pos: lerp2(cur.pos, nxt.pos, t), owner: Some(new_owner) });
                }
            }
            (false, true) => {
                let denom = fa - fb;
                if denom.abs() > 1e-18 {
                    let t = (fa / denom).clamp(0.0, 1.0);
                    out.push(ClipVert { pos: lerp2(cur.pos, nxt.pos, t), owner: cur.owner });
                }
            }
            (false, false) => {}
        }
    }
    out
}

/// Extract canonical undirected periodic Voronoi bonds `(i, j, [n1, n2])` between
/// fractional coordinates `frac_pts` in `[0, 1)^2` under periodic basis `(a1, a2)`.
/// Shares `scipy.spatial.Voronoi` ridge length filter `ridge_len > tol`.
pub fn voronoi_bonds(
    frac_pts: &[[f64; 2]],
    a1: [f64; 2],
    a2: [f64; 2],
    tol: f64,
) -> Vec<(usize, usize, [i32; 2])> {
    let n = frac_pts.len();
    if n == 0 {
        return Vec::new();
    }
    let cart_pts: Vec<[f64; 2]> = frac_pts
        .iter()
        .map(|f| [f[0] * a1[0] + f[1] * a2[0], f[0] * a1[1] + f[1] * a2[1]])
        .collect();
    let l1 = a1[0].hypot(a1[1]);
    let l2 = a2[0].hypot(a2[1]);
    let r_box = 4.0 * l1.max(l2).max(1.0);
    let m_range = 4_i32;

    type VoronoiNbr = (f64, [f64; 2], (usize, i32, i32));
    let mut bonds: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
    let mut nbrs: Vec<VoronoiNbr> = Vec::new();

    for (i, &pi) in cart_pts.iter().enumerate().take(n) {
        nbrs.clear();
        for (j, &pj) in cart_pts.iter().enumerate().take(n) {
            for m1 in -m_range..=m_range {
                for m2 in -m_range..=m_range {
                    if j == i && m1 == 0 && m2 == 0 {
                        continue;
                    }
                    let q = [
                        pj[0] + f64::from(m1) * a1[0] + f64::from(m2) * a2[0],
                        pj[1] + f64::from(m1) * a1[1] + f64::from(m2) * a2[1],
                    ];
                    let dx = q[0] - pi[0];
                    let dy = q[1] - pi[1];
                    let d2 = dx * dx + dy * dy;
                    if d2 > 1e-12 {
                        nbrs.push((d2, q, (j, m1, m2)));
                    }
                }
            }
        }
        nbrs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut poly = vec![
            ClipVert { pos: [pi[0] - r_box, pi[1] - r_box], owner: None },
            ClipVert { pos: [pi[0] + r_box, pi[1] - r_box], owner: None },
            ClipVert { pos: [pi[0] + r_box, pi[1] + r_box], owner: None },
            ClipVert { pos: [pi[0] - r_box, pi[1] + r_box], owner: None },
        ];

        for &(d2, q, owner) in &nbrs {
            let r_max = poly
                .iter()
                .map(|v| (v.pos[0] - pi[0]).hypot(v.pos[1] - pi[1]))
                .fold(0.0_f64, f64::max);
            let cut_limit = 2.0 * r_max + 1e-5;
            if d2 > cut_limit * cut_limit {
                break;
            }
            let mid = [0.5 * (pi[0] + q[0]), 0.5 * (pi[1] + q[1])];
            let normal = [q[0] - pi[0], q[1] - pi[1]];
            poly = clip_half_plane(&poly, mid, normal, owner);
            if poly.len() < 3 {
                break;
            }
        }

        let plen = poly.len();
        for k in 0..plen {
            let Some((j, m1, m2)) = poly[k].owner else { continue };
            let next_pos = poly[(k + 1) % plen].pos;
            let len = (next_pos[0] - poly[k].pos[0]).hypot(next_pos[1] - poly[k].pos[1]);
            if len > tol {
                let canon = if (i, j, m1, m2) <= (j, i, -m1, -m2) {
                    (i, j, m1, m2)
                } else {
                    (j, i, -m1, -m2)
                };
                bonds.insert(canon);
            }
        }
    }

    bonds.into_iter().map(|(i, j, n1, n2)| (i, j, [n1, n2])).collect()
}

/// Extract periodic Voronoi bonds directly from fractionalized `Cell2D` basis positions.
pub fn voronoi_bonds_of_cell(cell: &Cell2D, tol: f64) -> Result<Vec<(usize, usize, [i32; 2])>, String> {
    let a_inv = mat2_inv(cell.mat_a())?;
    let frac: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    Ok(voronoi_bonds(&frac, cell.a1, cell.a2, tol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::catalogue::parse_named_str;

    #[test]
    fn voronoi_bonds_match_standard_planar_tilings() {
        let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
        for key in ["square", "triangular", "honeycomb", "kagome", "elongated_triangular"] {
            let nl = named.named.iter().find(|n| n.key == key).unwrap();
            let got = voronoi_bonds_of_cell(&nl.cell, 1e-7).unwrap();
            let expected_edges: BTreeSet<(usize, usize, i32, i32)> = nl
                .cell
                .bonds
                .iter()
                .map(|b| {
                    if (b.i, b.j, b.n1, b.n2) <= (b.j, b.i, -b.n1, -b.n2) {
                        (b.i, b.j, b.n1, b.n2)
                    } else {
                        (b.j, b.i, -b.n1, -b.n2)
                    }
                })
                .collect();
            let got_set: BTreeSet<(usize, usize, i32, i32)> =
                got.iter().map(|&(i, j, n)| (i, j, n[0], n[1])).collect();
            assert_eq!(got_set, expected_edges, "Voronoi mismatch on {key}");
        }
        let rect = named.named.iter().find(|n| n.key == "rectangular").unwrap();
        assert_eq!(voronoi_bonds_of_cell(&rect.cell, 1e-7).unwrap().len(), rect.cell.bonds.len());
    }
}
