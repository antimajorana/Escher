//! Exact periodic 2D Voronoi / Delaunay dual bond extraction via unimodular
//! Lagrange-Gauss basis reduction, floor-wrapped sites, certified covering-radius
//! shift windows, and scale-invariant half-plane clipping.

use std::collections::BTreeSet;

use crate::plane::basis_reduce::ReducedBasis2D;
use crate::plane::cell::{canon_bond, mat2_inv, mat2_vec};
use crate::types::Cell2D;

#[derive(Clone, Copy)]
pub(crate) struct ClipVert {
    pub(crate) pos: [f64; 2],
    pub(crate) owner: Option<(usize, i32, i32)>,
}

fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn clip_half_plane(
    poly: &[ClipVert],
    mid: [f64; 2],
    normal: [f64; 2],
    new_owner: (usize, i32, i32),
    eps: f64,
) -> Vec<ClipVert> {
    if poly.is_empty() {
        return Vec::new();
    }
    let val = |p: [f64; 2]| (p[0] - mid[0]) * normal[0] + (p[1] - mid[1]) * normal[1];
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
                if denom.abs() > 1e-30 {
                    let t = (fa / denom).clamp(0.0, 1.0);
                    out.push(ClipVert { pos: lerp2(cur.pos, nxt.pos, t), owner: Some(new_owner) });
                }
            }
            (false, true) => {
                let denom = fa - fb;
                if denom.abs() > 1e-30 {
                    let t = (fa / denom).clamp(0.0, 1.0);
                    out.push(ClipVert { pos: lerp2(cur.pos, nxt.pos, t), owner: cur.owner });
                }
            }
            (false, false) => {}
        }
    }
    out
}

/// Extract canonical undirected periodic Voronoi/Delaunay bonds `(i, j, [n1, n2])` between
/// fractional coordinates `frac_pts` under periodic basis `(a1, a2)`.
/// Reduces `(a1, a2)` via Lagrange-Gauss, wraps sites into `[0, 1)^2`, uses a certified
/// covering-radius shift window, scales thresholds by characteristic spacing `l0`, and maps
/// shifts back to `(a1, a2)`.
pub fn voronoi_bonds(
    frac_pts: &[[f64; 2]],
    a1: [f64; 2],
    a2: [f64; 2],
    tol: f64,
) -> Vec<(usize, usize, [i32; 2])> {
    let Some(pc) = PeriodicVoronoi::build(frac_pts, a1, a2) else {
        return Vec::new();
    };
    let ridge_tol = tol.abs() * pc.l0;
    let mut bonds: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
    for (i, poly) in pc.polys.iter().enumerate() {
        let plen = poly.len();
        for k in 0..plen {
            let Some((j, m1, m2)) = poly[k].owner else { continue };
            let next_pos = poly[(k + 1) % plen].pos;
            let len = (next_pos[0] - poly[k].pos[0]).hypot(next_pos[1] - poly[k].pos[1]);
            if len > ridge_tol {
                let n_orig = pc.rb.map_shift_to_orig(pc.floors[i], pc.floors[j], [m1, m2]);
                let cb = canon_bond(i, j, n_orig[0], n_orig[1], 0);
                bonds.insert((cb.i, cb.j, cb.n1, cb.n2));
            }
        }
    }
    bonds.into_iter().map(|(i, j, n1, n2)| (i, j, [n1, n2])).collect()
}

/// Per-site clipped Voronoi polygons in the reduced, floor-wrapped Cartesian frame (shared by
/// `voronoi_bonds` and `voronoi_cells`).
pub(crate) struct PeriodicVoronoi {
    pub(crate) rb: ReducedBasis2D,
    pub(crate) floors: Vec<[i32; 2]>,
    /// Wrapped Cartesian site positions (`polys[i]` surrounds `cart_pts[i]`).
    pub(crate) cart_pts: Vec<[f64; 2]>,
    /// Characteristic spacing `sqrt(area / n)`.
    pub(crate) l0: f64,
    /// Counter-clockwise clipped polygon of each site; `owner` names the neighbour across the edge
    /// starting at that vertex.
    pub(crate) polys: Vec<Vec<ClipVert>>,
}

impl PeriodicVoronoi {
    pub(crate) fn build(frac_pts: &[[f64; 2]], a1: [f64; 2], a2: [f64; 2]) -> Option<Self> {
        let n = frac_pts.len();
        if n == 0 {
            return None;
        }
        let rb = ReducedBasis2D::reduce(a1, a2).ok()?;
        let (wrapped, floors) = rb.wrap_sites(frac_pts);
        let cart_pts: Vec<[f64; 2]> = wrapped
            .iter()
            .map(|f| [f[0] * rb.b1[0] + f[1] * rb.b2[0], f[0] * rb.b1[1] + f[1] * rb.b2[1]])
            .collect();
        let (m1_max, m2_max, r_box) = rb.covering_shift_bounds();
        let l0 = (rb.area() / (n as f64)).sqrt().max(1e-30);
        let min_d2 = (1e-6 * l0).powi(2);
        let clip_eps = 1e-11 * l0 * l0;

        type VoronoiNbr = (f64, [f64; 2], (usize, i32, i32));
        let mut polys: Vec<Vec<ClipVert>> = Vec::with_capacity(n);
        let mut nbrs: Vec<VoronoiNbr> = Vec::new();

        for (i, &pi) in cart_pts.iter().enumerate().take(n) {
            nbrs.clear();
            for (j, &pj) in cart_pts.iter().enumerate().take(n) {
                for m1 in -m1_max..=m1_max {
                    for m2 in -m2_max..=m2_max {
                        if j == i && m1 == 0 && m2 == 0 {
                            continue;
                        }
                        let q = [
                            pj[0] + f64::from(m1) * rb.b1[0] + f64::from(m2) * rb.b2[0],
                            pj[1] + f64::from(m1) * rb.b1[1] + f64::from(m2) * rb.b2[1],
                        ];
                        let dx = q[0] - pi[0];
                        let dy = q[1] - pi[1];
                        let d2 = dx * dx + dy * dy;
                        if d2 > min_d2 {
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
                let cut_limit = 2.0 * r_max + 1e-5 * l0;
                if d2 > cut_limit * cut_limit {
                    break;
                }
                let mid = [0.5 * (pi[0] + q[0]), 0.5 * (pi[1] + q[1])];
                let normal = [q[0] - pi[0], q[1] - pi[1]];
                poly = clip_half_plane(&poly, mid, normal, owner, clip_eps);
                if poly.len() < 3 {
                    break;
                }
            }
            polys.push(poly);
        }
        Some(Self { rb, floors, cart_pts, l0, polys })
    }
}

/// Extract periodic Voronoi/Delaunay bonds directly from fractionalized `Cell2D` basis positions.
pub fn voronoi_bonds_of_cell(cell: &Cell2D, tol: f64) -> Result<Vec<(usize, usize, [i32; 2])>, String> {
    let a_inv = mat2_inv(cell.mat_a())?;
    let frac: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    Ok(voronoi_bonds(&frac, cell.a1, cell.a2, tol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::catalogue::{parse_catalogue_str, parse_named_str};

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
                    let cb = canon_bond(b.i, b.j, b.n1, b.n2, 0);
                    (cb.i, cb.j, cb.n1, cb.n2)
                })
                .collect();
            let got_set: BTreeSet<(usize, usize, i32, i32)> =
                got.iter().map(|&(i, j, n)| (i, j, n[0], n[1])).collect();
            assert_eq!(got_set, expected_edges, "Voronoi mismatch on {key}");
        }
        let rect = named.named.iter().find(|n| n.key == "rectangular").unwrap();
        assert_eq!(voronoi_bonds_of_cell(&rect.cell, 1e-7).unwrap().len(), rect.cell.bonds.len());
    }

    #[test]
    fn points_catalogue_181_drawings_keep_exact_voronoi_bonds() {
        let pts = parse_catalogue_str(include_str!("../../../../papertilling/data/points.json")).unwrap();
        let mut total = 0usize;
        for (grp, cat) in &pts {
            for (idx, ent) in cat.drawings.iter().enumerate() {
                let cell = ent.family.cell();
                let got: BTreeSet<(usize, usize, i32, i32)> = voronoi_bonds_of_cell(&cell, 1e-7)
                    .unwrap()
                    .into_iter()
                    .map(|(i, j, n)| (i, j, n[0], n[1]))
                    .collect();
                let stored: BTreeSet<(usize, usize, i32, i32)> = cell
                    .bonds
                    .iter()
                    .map(|b| {
                        let cb = canon_bond(b.i, b.j, b.n1, b.n2, 0);
                        (cb.i, cb.j, cb.n1, cb.n2)
                    })
                    .collect();
                assert_eq!(got, stored, "points.json regression on {grp} drawing #{idx}");
                total += 1;
            }
        }
        assert_eq!(total, 181);
    }

    #[test]
    fn sheared_skew_and_scale_invariance_controls() {
        // Sheared square: a1=[1,0], a2=[6,1] has true shortest neighbours at [1,0] and [-6,1] (deg 4).
        let sq_shear = voronoi_bonds(&[[0.0, 0.0]], [1.0, 0.0], [6.0, 1.0], 1e-7);
        assert_eq!(sq_shear, vec![(0, 0, [-6, 1]), (0, 0, [-1, 0])]);

        // Skewed oblique (a=1, b=6, gamma=15 deg): degree 6 including (-6,1) and (-5,1).
        let g15 = 15.0_f64.to_radians();
        let obl_6_15 = voronoi_bonds(&[[0.0, 0.0]], [1.0, 0.0], [6.0 * g15.cos(), 6.0 * g15.sin()], 1e-7);
        assert_eq!(obl_6_15.len(), 3, "expected degree 6 (3 canonical bonds), got {obl_6_15:?}");
        assert!(obl_6_15.contains(&(0, 0, [-6, 1])), "{obl_6_15:?}");
        assert!(obl_6_15.contains(&(0, 0, [-5, 1])), "{obl_6_15:?}");

        // Skewed oblique (a=1, b=8, gamma=10 deg): degree 6.
        let g10 = 10.0_f64.to_radians();
        let obl_8_10 = voronoi_bonds(&[[0.0, 0.0]], [1.0, 0.0], [8.0 * g10.cos(), 8.0 * g10.sin()], 1e-7);
        assert_eq!(obl_8_10.len(), 3, "expected degree 6 (3 canonical bonds), got {obl_8_10:?}");

        // Scale invariance (1e-8 and 1e6) + unwrapped fractional coords [4.3, -2.7] vs [0.3, 0.3].
        let base = voronoi_bonds(&[[0.0, 0.0], [0.3, 0.3]], [1.0, 0.0], [0.0, 1.0], 1e-7);
        let unwrapped = voronoi_bonds(&[[0.0, 0.0], [4.3, -2.7]], [1.0, 0.0], [0.0, 1.0], 1e-7);
        let unwrapped_base: Vec<_> = unwrapped
            .into_iter()
            .map(|(i, j, n)| {
                if i == 0 && j == 1 { (i, j, [n[0] + 4, n[1] - 3]) } else { (i, j, n) }
            })
            .collect();
        let b_set: BTreeSet<_> = base.iter().copied().collect();
        let u_set: BTreeSet<_> = unwrapped_base.into_iter().collect();
        assert_eq!(u_set, b_set);

        for scale in [1e-8_f64, 1e6_f64] {
            let scaled = voronoi_bonds(&[[0.0, 0.0], [0.3, 0.3]], [scale, 0.0], [0.0, scale], 1e-7);
            assert_eq!(scaled, base, "scale invariance failed at scale={scale}");
        }
    }
}
