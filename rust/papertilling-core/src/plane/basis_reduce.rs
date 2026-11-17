//! Unimodular Lagrange-Gauss 2D basis reduction, site floor wrapping,
//! certified periodic shift bounds, and geometric chirality detection.

use std::collections::BTreeSet;

use crate::plane::cell::canon_bond;
use crate::types::Cell2D;

/// Lagrange-Gauss reduced right-handed 2D lattice basis `[b1 | b2] = [a1 | a2] * u`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedBasis2D {
    pub b1: [f64; 2],
    pub b2: [f64; 2],
    pub u: [[i32; 2]; 2],
    pub u_inv: [[i32; 2]; 2],
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn mat2_i32_vec_f(m: [[i32; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [
        f64::from(m[0][0]) * v[0] + f64::from(m[0][1]) * v[1],
        f64::from(m[1][0]) * v[0] + f64::from(m[1][1]) * v[1],
    ]
}

fn mat2_i32_vec_i(m: [[i32; 2]; 2], v: [i32; 2]) -> [i32; 2] {
    [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
}

impl ReducedBasis2D {
    /// Reduce `(a1, a2)` via unimodular Gauss-Lagrange steps, tracking exact `u` and `u_inv`.
    pub fn reduce(a1: [f64; 2], a2: [f64; 2]) -> Result<Self, String> {
        let det = a1[0] * a2[1] - a1[1] * a2[0];
        if !det.is_finite() || det.abs() < 1e-30 {
            return Err("singular or non-finite 2D lattice basis".to_string());
        }
        let mut b1 = a1;
        let mut b2 = a2;
        let mut u = [[1_i32, 0], [0, 1]];

        for _ in 0..200 {
            let n1 = dot2(b1, b1);
            let n2 = dot2(b2, b2);
            if n2 < n1 - 1e-12 * n1 {
                std::mem::swap(&mut b1, &mut b2);
                u[0].swap(0, 1);
                u[1].swap(0, 1);
            }
            let denom = dot2(b1, b1);
            if denom < 1e-30 {
                break;
            }
            let k_f = (dot2(b1, b2) / denom).round_ties_even();
            if !k_f.is_finite() {
                return Err("non-finite Lagrange-Gauss step".to_string());
            }
            if k_f == 0.0 {
                break;
            }
            if k_f.abs() > f64::from(i32::MAX / 4) {
                return Err("extreme basis shear exceeds integer reduction range".to_string());
            }
            let k = k_f as i32;
            b2 = [b2[0] - k_f * b1[0], b2[1] - k_f * b1[1]];
            u[0][1] = u[0][1].checked_sub(k.checked_mul(u[0][0]).ok_or("reduction overflow")?)
                .ok_or("reduction overflow")?;
            u[1][1] = u[1][1].checked_sub(k.checked_mul(u[1][0]).ok_or("reduction overflow")?)
                .ok_or("reduction overflow")?;
        }

        if b1[0] * b2[1] - b1[1] * b2[0] < 0.0 {
            b2 = [-b2[0], -b2[1]];
            u[0][1] = -u[0][1];
            u[1][1] = -u[1][1];
        }
        let det_u = u[0][0] * u[1][1] - u[0][1] * u[1][0];
        if det_u.abs() != 1 {
            return Err("basis reduction matrix is not unimodular".to_string());
        }
        let u_inv = [
            [det_u * u[1][1], -det_u * u[0][1]],
            [-det_u * u[1][0], det_u * u[0][0]],
        ];
        Ok(Self { b1, b2, u, u_inv })
    }

    /// Positive unit-cell area `|det([b1, b2])|`.
    pub fn area(&self) -> f64 {
        (self.b1[0] * self.b2[1] - self.b1[1] * self.b2[0]).abs()
    }

    /// Wrap fractional coords `frac_pts` (in `(a1, a2)`) into `[0, 1)^2` of the reduced basis `(b1, b2)`,
    /// returning `(wrapped_frac_red, floor_shifts)` where `u_inv * frac_pts[i] == wrapped[i] + floor_shifts[i]`.
    pub fn wrap_sites(&self, frac_pts: &[[f64; 2]]) -> (Vec<[f64; 2]>, Vec<[i32; 2]>) {
        let mut wrapped = Vec::with_capacity(frac_pts.len());
        let mut floors = Vec::with_capacity(frac_pts.len());
        let hi = 1.0 - f64::EPSILON;
        for &f in frac_pts {
            let g = mat2_i32_vec_f(self.u_inv, f);
            let fl = [
                (g[0] + 1e-12).floor() as i32,
                (g[1] + 1e-12).floor() as i32,
            ];
            let w = [
                (g[0] - f64::from(fl[0])).clamp(0.0, hi),
                (g[1] - f64::from(fl[1])).clamp(0.0, hi),
            ];
            wrapped.push(w);
            floors.push(fl);
        }
        (wrapped, floors)
    }

    /// Map reduced-cell shift `m_red` between wrapped sites `(i, j)` back to shift in `(a1, a2)`.
    pub fn map_shift_to_orig(&self, fl_i: [i32; 2], fl_j: [i32; 2], m_red: [i32; 2]) -> [i32; 2] {
        let diff = [
            m_red[0] - fl_j[0] + fl_i[0],
            m_red[1] - fl_j[1] + fl_i[1],
        ];
        mat2_i32_vec_i(self.u, diff)
    }

    /// Certified shift window `(m1_max, m2_max, r_box)` using covering radius `<= 0.5*(|b1| + |b2|)`.
    pub fn covering_shift_bounds(&self) -> (i32, i32, f64) {
        let l1 = self.b1[0].hypot(self.b1[1]);
        let l2 = self.b2[0].hypot(self.b2[1]);
        let area = self.area();
        let reach_dist = 2.5 * (l1 + l2);
        let r_box = 2.0 * (l1 + l2);
        let h1 = (area / l2.max(1e-30)).max(1e-30);
        let h2 = (area / l1.max(1e-30)).max(1e-30);
        let m1_max = ((reach_dist / h1).ceil() as i32 + 1).clamp(3, 64);
        let m2_max = ((reach_dist / h2).ceil() as i32 + 1).clamp(3, 64);
        (m1_max, m2_max, r_box)
    }
}

/// Canonical `(i, j, [n1, n2])` bond candidates within `[-reach..=reach]^2` in the reduced wrapped basis.
pub fn reduced_reach_candidates(
    frac_pts: &[[f64; 2]],
    a1: [f64; 2],
    a2: [f64; 2],
    reach: i32,
) -> Result<Vec<(usize, usize, [i32; 2])>, String> {
    let rb = ReducedBasis2D::reduce(a1, a2)?;
    let (_, floors) = rb.wrap_sites(frac_pts);
    let n = frac_pts.len();
    let mut set: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
    for (i, &fl_i) in floors.iter().enumerate().take(n) {
        for (j, &fl_j) in floors.iter().enumerate().take(n) {
            for m1 in -reach..=reach {
                for m2 in -reach..=reach {
                    let sh = rb.map_shift_to_orig(fl_i, fl_j, [m1, m2]);
                    if i == j && sh == [0, 0] {
                        continue;
                    }
                    let cb = canon_bond(i, j, sh[0], sh[1], 0);
                    set.insert((cb.i, cb.j, cb.n1, cb.n2));
                }
            }
        }
    }
    Ok(set.into_iter().map(|(i, j, n1, n2)| (i, j, [n1, n2])).collect())
}

/// True iff the realized Euclidean cell has no orientation-reversing (`det < 0`) symmetries in `E(2)`.
pub fn is_geometrically_chiral(cell: &Cell2D) -> Result<bool, String> {
    if cell.basis.is_empty() {
        return Err("empty cell has no symmetry group".to_string());
    }
    let prim = cell.try_primitive_cell(false)?;
    let ops = prim.symmetry_frac_ops(false);
    if ops.is_empty() {
        return Err("could not determine Euclidean symmetries of primitive cell".to_string());
    }
    Ok(!ops
        .iter()
        .any(|op| op.mf[0][0] * op.mf[1][1] - op.mf[0][1] * op.mf[1][0] < 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::catalogue::parse_named_str;

    #[test]
    fn reduction_wraps_and_maps_shifts_with_exact_cartesian_vectors() {
        let a1 = [1.0, 0.0];
        let a2 = [6.0, 1.0];
        let rb = ReducedBasis2D::reduce(a1, a2).unwrap();
        assert!((rb.b1[0].hypot(rb.b1[1]) - 1.0).abs() < 1e-12);
        assert!((rb.b2[0].hypot(rb.b2[1]) - 1.0).abs() < 1e-12);

        let frac_pts = vec![[3.2, -1.4], [-2.7, 4.1]];
        let (wrapped, floors) = rb.wrap_sites(&frac_pts);
        for &(i, j, m_red) in &[(0_usize, 1_usize, [1, -1]), (1, 0, [0, 1]), (0, 0, [1, 0])] {
            let n_orig = rb.map_shift_to_orig(floors[i], floors[j], m_red);
            let disp_orig = [
                (frac_pts[j][0] + f64::from(n_orig[0]) - frac_pts[i][0]) * a1[0]
                    + (frac_pts[j][1] + f64::from(n_orig[1]) - frac_pts[i][1]) * a2[0],
                (frac_pts[j][0] + f64::from(n_orig[0]) - frac_pts[i][0]) * a1[1]
                    + (frac_pts[j][1] + f64::from(n_orig[1]) - frac_pts[i][1]) * a2[1],
            ];
            let disp_red = [
                (wrapped[j][0] + f64::from(m_red[0]) - wrapped[i][0]) * rb.b1[0]
                    + (wrapped[j][1] + f64::from(m_red[1]) - wrapped[i][1]) * rb.b2[0],
                (wrapped[j][0] + f64::from(m_red[0]) - wrapped[i][0]) * rb.b1[1]
                    + (wrapped[j][1] + f64::from(m_red[1]) - wrapped[i][1]) * rb.b2[1],
            ];
            assert!((disp_orig[0] - disp_red[0]).abs() < 1e-11);
            assert!((disp_orig[1] - disp_red[1]).abs() < 1e-11);
        }
    }

    #[test]
    fn geometric_chirality_matches_known_named_lattices() {
        let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
        for key in ["chiral_square", "chiral_kagome", "chiral_triangular", "straight_p1", "oblique"] {
            let nl = named.named.iter().find(|n| n.key == key).unwrap();
            assert!(is_geometrically_chiral(&nl.cell).unwrap(), "{key} should be geometrically chiral");
        }
        for key in ["square", "kagome", "honeycomb", "shastry_sutherland", "rectangular", "brick_pg"] {
            let nl = named.named.iter().find(|n| n.key == key).unwrap();
            assert!(!is_geometrically_chiral(&nl.cell).unwrap(), "{key} should be achiral");
        }
    }
}
