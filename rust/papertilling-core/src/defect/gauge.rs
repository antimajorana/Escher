//! Layer 1: Exact nonsymmorphic space-group extensions, Wyckoff rotocentre cosets,
//! conjugate defect fusion channels, signed ribbon-dart face walks, and Cartan holonomies.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::defect::catalogue::DefectCatalogue;
use crate::defect::cone_geometry::{mat2_inv, mat2_vec, rotation_about};
use crate::defect::sym_helper::{
    LatticeGroup, imat2_det, imat2_inv, imat2_mul, imat2_to_f64, imat2_vec, mat2_mul,
    np_allclose_mat, round_even,
};
use crate::plane::groups::{canonical_symbol, conventional_cell, prim_ops, prim_u};
use crate::types::{Cell2D, FracOp2D, Frame2D, Isometry2};

const ID2: [[i32; 2]; 2] = [[1, 0], [0, 1]];

fn frac_unit(x: f64) -> f64 {
    let r = x - (x + 1e-6).floor();
    if (r - 1.0).abs() < 1e-6 || r.abs() < 1e-6 { 0.0 } else { r }
}

fn is_int2(v: [f64; 2], tol: f64) -> bool {
    (v[0] - round_even(v[0])).abs() < tol && (v[1] - round_even(v[1])).abs() < tol
}

fn round_i32_2(v: [f64; 2]) -> [i32; 2] {
    [round_even(v[0]) as i32, round_even(v[1]) as i32]
}

pub(crate) fn translation_coset(m: [[i32; 2]; 2], offset: [i32; 2]) -> super::algebraic_fusion_snf::LatticeCoset {
    let cols = [0, 1].map(|j| [i64::from(m[0][j]), i64::from(m[1][j])]);
    super::algebraic_fusion_snf::LatticeCoset::new(offset.map(i64::from), &cols)
}

pub(crate) fn in_image_of_int_mat(m: [[i32; 2]; 2], v: [i32; 2]) -> bool {
    translation_coset(m, [0, 0]).contains(v.map(i64::from))
}

fn canonical_coset_rep(m: [[i32; 2]; 2], v: [i32; 2]) -> [i32; 2] {
    let coset = translation_coset(m, v);
    if coset.contains([0, 0]) {
        return [0, 0];
    }
    super::algebraic_fusion_snf::shortest_coset_vectors(&coset, [[1.0, 0.0], [0.0, 1.0]])
        .into_iter()
        .min_by_key(|w| (w[0].abs() + w[1].abs(), -w[0], -w[1]))
        .expect("nonempty coset")
        .map(|x| x as i32)
}

/// Element `(t_frac | R)` of a 2D space group `G` in primitive fractional coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpaceGroupElement {
    pub l_vec: [i32; 2],
    pub r_idx: usize,
    pub r: [[i32; 2]; 2],
    pub t_frac: [f64; 2],
}

/// Exact group extension `1 -> Lambda -> G -> P -> 1` in primitive fractional coordinates (`Lambda = Z^2`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpaceGroupExtension {
    pub group: String,
    pub a: [[f64; 2]; 2],
    pub a_inv: [[f64; 2]; 2],
    pub point_group: Vec<[[i32; 2]; 2]>,
    pub point_group_cart: Vec<[[f64; 2]; 2]>,
    pub tau: Vec<[f64; 2]>,
    pub nu: Vec<Vec<[i32; 2]>>,
    pub mult_table: Vec<Vec<usize>>,
    pub is_nonsymmorphic: bool,
}

impl SpaceGroupExtension {
    pub fn from_group(group: &str) -> Result<Self, String> {
        let sym = canonical_symbol(group)?;
        let (a1_c, a2_c) = conventional_cell(sym, 1.0, None, None)?;
        let u = prim_u(sym)?;
        let a_conv = [[a1_c[0], a2_c[0]], [a1_c[1], a2_c[1]]];
        let a = mat2_mul(a_conv, u);
        let opsf = prim_ops(sym)?;
        Ok(Self::build_from_ops(sym, a, opsf))
    }

    pub fn from_cell(group: &str, cell: &Cell2D) -> Result<Self, String> {
        let grp = LatticeGroup::new(cell);
        let sym = if group.is_empty() {
            grp.name
        } else {
            canonical_symbol(group).unwrap_or(grp.name)
        };
        Ok(Self::build_from_ops(sym, grp.a, grp.opsf))
    }

    fn build_from_ops(group: &str, a: [[f64; 2]; 2], mut opsf: Vec<FracOp2D>) -> Self {
        opsf.sort_by_key(|op| {
            let is_id = op.mf == ID2;
            let det_rank = if imat2_det(op.mf) > 0 { 0i32 } else { 1 };
            (!is_id, det_rank, op.mf[0][0], op.mf[0][1], op.mf[1][0], op.mf[1][1])
        });
        let a_inv = mat2_inv(a);
        let m = opsf.len();
        let point_group: Vec<[[i32; 2]; 2]> = opsf.iter().map(|op| op.mf).collect();
        let point_group_cart: Vec<[[f64; 2]; 2]> = point_group
            .iter()
            .map(|&mf| mat2_mul(mat2_mul(a, imat2_to_f64(mf)), a_inv))
            .collect();
        let tau: Vec<[f64; 2]> = opsf.iter().map(|op| [frac_unit(op.t[0]), frac_unit(op.t[1])]).collect();

        let mut mult_table = vec![vec![0usize; m]; m];
        let mut nu = vec![vec![[0i32; 2]; m]; m];
        for i in 0..m {
            for j in 0..m {
                let rp = imat2_mul(point_group[i], point_group[j]);
                let k = point_group.iter().position(|&r| r == rp).expect("point group closed");
                mult_table[i][j] = k;
                let r_tau = mat2_vec(imat2_to_f64(point_group[i]), tau[j]);
                let raw = [tau[i][0] + r_tau[0] - tau[k][0], tau[i][1] + r_tau[1] - tau[k][1]];
                debug_assert!(is_int2(raw, 1e-4), "2-cocycle must land in Lambda");
                nu[i][j] = round_i32_2(raw);
            }
        }
        let is_nonsymmorphic = Self::detect_nonsymmorphic(&point_group, &tau);
        Self {
            group: group.to_string(),
            a,
            a_inv,
            point_group,
            point_group_cart,
            tau,
            nu,
            mult_table,
            is_nonsymmorphic,
        }
    }

    fn detect_nonsymmorphic(pg: &[[[i32; 2]; 2]], tau: &[[f64; 2]]) -> bool {
        if let Some(rot_idx) = pg.iter().position(|&r| imat2_det(r) == 1 && r != ID2) {
            let r0 = pg[rot_idx];
            let im = [[(1 - r0[0][0]) as f64, -r0[0][1] as f64], [-r0[1][0] as f64, (1 - r0[1][1]) as f64]];
            let im_inv = mat2_inv(im);
            for l0 in -2..=2 {
                for l1 in -2..=2 {
                    let s = mat2_vec(im_inv, [l0 as f64 - tau[rot_idx][0], l1 as f64 - tau[rot_idx][1]]);
                    let ok = pg.iter().zip(tau).all(|(&r, &t)| {
                        let sh = [
                            t[0] + (1 - r[0][0]) as f64 * s[0] - r[0][1] as f64 * s[1],
                            t[1] - r[1][0] as f64 * s[0] + (1 - r[1][1]) as f64 * s[1],
                        ];
                        is_int2(sh, 1e-5)
                    });
                    if ok {
                        return false;
                    }
                }
            }
            return true;
        }
        if let Some(m_idx) = pg.iter().position(|&r| imat2_det(r) == -1) {
            let rm = pg[m_idx];
            let t = tau[m_idx];
            let sq = [
                t[0] + rm[0][0] as f64 * t[0] + rm[0][1] as f64 * t[1],
                t[1] + rm[1][0] as f64 * t[0] + rm[1][1] as f64 * t[1],
            ];
            let i_plus_r = [[1 + rm[0][0], rm[0][1]], [rm[1][0], 1 + rm[1][1]]];
            if !is_int2(sq, 1e-5) {
                return true;
            }
            let sq_i = round_i32_2(sq);
            return !(-2..=2).any(|a| (-2..=2).any(|b| imat2_vec(i_plus_r, [a, b]) == sq_i));
        }
        false
    }

    pub fn identity(&self) -> SpaceGroupElement {
        SpaceGroupElement { l_vec: [0, 0], r_idx: 0, r: ID2, t_frac: self.tau[0] }
    }

    pub fn coset_rep(&self, r_idx: usize) -> SpaceGroupElement {
        SpaceGroupElement { l_vec: [0, 0], r_idx, r: self.point_group[r_idx], t_frac: self.tau[r_idx] }
    }

    pub fn element(&self, l_vec: [i32; 2], r: [[i32; 2]; 2]) -> Option<SpaceGroupElement> {
        let r_idx = self.point_group.iter().position(|&m| m == r)?;
        Some(SpaceGroupElement {
            l_vec,
            r_idx,
            r,
            t_frac: [l_vec[0] as f64 + self.tau[r_idx][0], l_vec[1] as f64 + self.tau[r_idx][1]],
        })
    }

    pub fn from_isometry(&self, iso: &Isometry2) -> Option<SpaceGroupElement> {
        let r_idx = self
            .point_group_cart
            .iter()
            .position(|&mc| np_allclose_mat(mc, iso.m, 1e-4, 1e-5))?;
        let t_f = mat2_vec(self.a_inv, iso.t);
        let diff = [t_f[0] - self.tau[r_idx][0], t_f[1] - self.tau[r_idx][1]];
        is_int2(diff, 1e-4).then(|| {
            let l_vec = round_i32_2(diff);
            self.element(l_vec, self.point_group[r_idx]).unwrap()
        })
    }

    pub fn to_isometry(&self, elem: &SpaceGroupElement) -> Isometry2 {
        Isometry2 { m: self.point_group_cart[elem.r_idx], t: mat2_vec(self.a, elem.t_frac) }
    }

    pub fn compose(&self, a: &SpaceGroupElement, b: &SpaceGroupElement) -> SpaceGroupElement {
        let r_idx = self.mult_table[a.r_idx][b.r_idx];
        let rb = imat2_vec(a.r, b.l_vec);
        let c = self.nu[a.r_idx][b.r_idx];
        let l_vec = [a.l_vec[0] + rb[0] + c[0], a.l_vec[1] + rb[1] + c[1]];
        SpaceGroupElement {
            l_vec,
            r_idx,
            r: self.point_group[r_idx],
            t_frac: [l_vec[0] as f64 + self.tau[r_idx][0], l_vec[1] as f64 + self.tau[r_idx][1]],
        }
    }

    pub fn inverse(&self, a: &SpaceGroupElement) -> SpaceGroupElement {
        let r_inv = imat2_inv(a.r);
        let r_idx = self.point_group.iter().position(|&m| m == r_inv).expect("inverse in P");
        let neg_r_l = imat2_vec(r_inv, [-a.l_vec[0], -a.l_vec[1]]);
        let c = self.nu[r_idx][a.r_idx];
        let l_vec = [neg_r_l[0] - c[0], neg_r_l[1] - c[1]];
        SpaceGroupElement {
            l_vec,
            r_idx,
            r: r_inv,
            t_frac: [l_vec[0] as f64 + self.tau[r_idx][0], l_vec[1] as f64 + self.tau[r_idx][1]],
        }
    }

    pub fn conjugate(&self, a: &SpaceGroupElement, by: &SpaceGroupElement) -> SpaceGroupElement {
        let by_inv = self.inverse(by);
        self.compose(&self.compose(by, a), &by_inv)
    }

    pub fn find_op(&self, r: [[i32; 2]; 2]) -> Option<SpaceGroupElement> {
        self.element([0, 0], r)
    }
}

/// Proper Wyckoff rotation centre with `Lambda / (I - R)Lambda` coset classification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WyckoffRotocentre {
    pub label: String,
    pub gclass: String,
    pub order: usize,
    pub point: [f64; 2],
    pub frac: [f64; 2],
    pub location: String,
    pub split: bool,
    pub mirror_partner: Option<String>,
    pub quotient_order: i32,
    pub coset_rep: [i32; 2],
}

pub fn wyckoff_rotocentres_with_frame(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
) -> Result<Vec<WyckoffRotocentre>, String> {
    let ext = SpaceGroupExtension::from_cell(group, cell)?;
    let mut cat = DefectCatalogue::with_bounds(group, cell, frame, 0.0, 0.0);
    if frame == Frame2D::default()
        && let Some(first) = cat.centre_classes.first()
        && cat.centre_classes.iter().any(|c| c.label.starts_with("order-"))
    {
        let shifted = Frame2D { q: frame.q, origin: first.point };
        let retry = DefectCatalogue::with_bounds(cat.group.as_str(), cell, shifted, 0.0, 0.0);
        if retry.centre_classes.iter().all(|c| !c.label.starts_with("order-")) {
            cat = retry;
        }
    }
    let id_to_label: HashMap<String, String> = cat
        .centre_classes
        .iter()
        .map(|c| (c.id.clone(), c.label.clone()))
        .collect();
    let mut out = Vec::with_capacity(cat.centre_classes.len());
    for c in &cat.centre_classes {
        let iso = rotation_about(c.point, (360.0 / c.order as f64).to_radians());
        let elem = ext
            .from_isometry(&iso)
            .ok_or_else(|| format!("rotation at {} not found in space group", c.label))?;
        let i_minus_r = [[1 - elem.r[0][0], -elem.r[0][1]], [-elem.r[1][0], 1 - elem.r[1][1]]];
        let quotient_order = imat2_det(i_minus_r).abs();
        let coset_rep = canonical_coset_rep(i_minus_r, elem.l_vec);
        out.push(WyckoffRotocentre {
            label: c.label.clone(),
            gclass: c.gclass.clone(),
            order: c.order,
            point: c.point,
            frac: c.frac,
            location: c.location.clone(),
            split: c.split,
            mirror_partner: c.mirror_of.as_ref().and_then(|id| id_to_label.get(id).cloned()),
            quotient_order,
            coset_rep,
        });
    }
    Ok(out)
}

pub fn wyckoff_rotocentres(group: &str, cell: &Cell2D) -> Result<Vec<WyckoffRotocentre>, String> {
    wyckoff_rotocentres_with_frame(group, cell, Frame2D::default())
}

pub use crate::defect::gauge_holonomy::{
    FusionChannel, PlaquetteHolonomy, apply_gauge_transform, fusion_channels,
    fusion_channels_for_elements, plaquette_holonomies, ribbon_faces, twisted_coboundary,
    verify_d_r_squared,
};

#[cfg(test)]
mod translation_tests {
    use super::*;
    use crate::defect::algebraic_fusion_snf::{LatticeCoset, shortest_coset_vectors};

    #[test]
    fn integer_images_include_divisibility_and_only_the_column_direction() {
        assert!(!in_image_of_int_mat([[0, 0], [0, 0]], [1, 0]));
        let mirror = [[0, 0], [2, 2]];
        assert!(in_image_of_int_mat(mirror, [0, 12]));
        assert!(!in_image_of_int_mat(mirror, [0, 1]));
        assert!(!in_image_of_int_mat(mirror, [2, 2]));
    }

    #[test]
    fn shortest_burgers_vectors_use_the_metric_and_arbitrary_periods() {
        let full = LatticeCoset::new([0, 0], &[[1, 0], [0, 1]]);
        // A maps (-30,1) to (0,0.25), whereas (1,0) has length 1.
        assert_eq!(shortest_coset_vectors(&full, [[1.0, 30.0], [0.0, 0.25]]), [[30, -1], [-30, 1]]);
        let line = LatticeCoset::new([700, 3], &[[2, 0]]);
        assert_eq!(shortest_coset_vectors(&line, [[1.0, 0.0], [0.0, 1.0]]), [[0, 3]]);
        assert_eq!(canonical_coset_rep([[0, 0], [0, 0]], [200, 1]), [200, 1]);
    }
}
