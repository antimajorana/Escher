//! Phase 3A: Exact algebraic fusion ring in the lifted proper group `G~+`,
//! universal abelianizations `(G+)^ab` and `G^ab`, and improper reflection/glide
//! holonomy classifier via Theorem 3.1 (`Q_R // Z_P(R)`).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::defect::algebraic_fusion_snf::{
    AbelianElement, AbelianMap, Frac2, LatticeCoset, LatticeCosetUnion, det_i64,
    format_abelian_group, i_minus, i_plus, inv_i64, invariant_factors_from_snf, mat_mul_i64,
    mat_vec_i64, reflection_axis, rot_order, shortest_nonzero_in_coset, shortest_nonzero_in_set,
    shortest_nonzero_in_union, smith_normal_form, solve_int_system_exists,
};
use crate::defect::catalogue::DefectCatalogue;
use crate::defect::surgery::DefectSpec;
use crate::defect::sym_helper::{
    imat2_det, imat2_to_f64, mat2_mul, mat2_vec, np_allclose_mat, rotation_about, round_even,
    solve2,
};
use crate::types::{Cell2D, FracOp2D, Frame2D, Isometry2};

const PROVENANCE: &str = "[PROVED] Exact integer Smith normal form and lifted proper-group (G~+) affine coset algebra; holds at all defect separations.";

/// Exact invariant-factor summary of a finitely generated abelian group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbelianGroupSummary {
    pub free_rank: usize,
    pub torsion: Vec<i64>,
    pub label: String,
}

/// Classification of one improper (`det = -1`) point-group conjugacy sector via `Q_R // Z_P(R)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImproperSectorInfo {
    pub rep_matrix: [[i64; 2]; 2],
    pub rep_tau: [f64; 2],
    pub label: String,
    pub parallel_axis: [i64; 2],
    pub quotient_label: String,
    pub has_pure_mirror: bool,
    pub fixed_point_free_centralizer: bool,
    pub centralizer_summary: String,
    pub square_proper_channel: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BurgersDomain {
    CosetUnion(LatticeCosetUnion),
    FiniteSet(BTreeSet<[i64; 2]>),
}

/// One algebraic fusion outcome channel in the lifted proper group `G~+`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlgebraicFusionOutcome {
    pub kind: String,
    pub frank_angle_deg: f64,
    pub allowed_single_defect: bool,
    pub summary: String,
    pub allowed_centre_classes: Vec<String>,
    pub burgers_sublattice_index: Option<i64>,
    pub burgers_coset_contains_zero: Option<bool>,
    pub shortest_burgers_frac: Vec<[i64; 2]>,
    #[serde(skip)]
    pub(crate) burgers_domain: Option<BurgersDomain>,
}

impl AlgebraicFusionOutcome {
    /// Exact membership test for an integer primitive-cell Burgers vector `b` in this outcome's domain.
    pub fn contains_burgers(&self, b: [i64; 2]) -> bool {
        match &self.burgers_domain {
            Some(BurgersDomain::CosetUnion(u)) => u.contains(b),
            Some(BurgersDomain::FiniteSet(s)) => s.contains(&b),
            None => false,
        }
    }
}

/// Unordered elementary defect pair `(A, B)` and its exact lifted `G~+` fusion channels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlgebraicFusionRow {
    pub defect_a: String,
    pub defect_b: String,
    pub abelian_sum_proper: String,
    pub outcomes: Vec<AlgebraicFusionOutcome>,
}

/// Complete exact space-group algebra and lifted `G~+` fusion report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpaceGroupAlgebraReport {
    pub group: String,
    pub point_group_order: usize,
    pub proper_order: usize,
    pub symmorphic: bool,
    pub ab_full: AbelianGroupSummary,
    pub ab_proper: AbelianGroupSummary,
    pub improper_sectors: Vec<ImproperSectorInfo>,
    pub fusion_rows: Vec<AlgebraicFusionRow>,
    pub provenance: String,
}

/// Exact integer/rational space-group extension `1 -> Z^2 -> G -> P -> 1` built from `DefectCatalogue`.
#[derive(Clone, Debug)]
pub struct SpaceGroupAlgebra {
    pub group: String,
    pub proper_order: usize,
    pub matrices: Vec<[[i64; 2]; 2]>,
    pub cart_matrices: Vec<[[f64; 2]; 2]>,
    pub(crate) tau: Vec<Frac2>,
    pub nu: Vec<Vec<[i64; 2]>>,
    pub mult: Vec<Vec<usize>>,
    pub origin_shift_frac: [f64; 2],
    a: [[f64; 2]; 2],
    a_inv: [[f64; 2]; 2],
    ab_full: AbelianMap,
    ab_proper: AbelianMap,
}

impl SpaceGroupAlgebra {
    /// Extract exact integer ops, rational translations `tau(R)`, 2-cocycle `nu(R, S)`, and abelianizations.
    pub fn from_catalogue(cat: &DefectCatalogue) -> Result<Self, String> {
        let mut raw_ops = cat.grp.opsf.clone();
        raw_ops.sort_by_key(|op| {
            let det_rank = if imat2_det(op.mf) > 0 { 0i32 } else { 1 };
            (op.mf != [[1, 0], [0, 1]], det_rank, op.mf[0][0], op.mf[0][1], op.mf[1][0], op.mf[1][1])
        });
        let m = raw_ops.len();
        if m == 0 || raw_ops[0].mf != [[1, 0], [0, 1]] {
            return Err("space group operations missing identity".to_string());
        }
        let matrices: Vec<[[i64; 2]; 2]> = raw_ops.iter()
            .map(|op| [[i64::from(op.mf[0][0]), i64::from(op.mf[0][1])], [i64::from(op.mf[1][0]), i64::from(op.mf[1][1])]])
            .collect();
        let proper_order = matrices.iter().take_while(|&&r| det_i64(r) == 1).count();
        let cart_matrices = raw_ops.iter()
            .map(|op| mat2_mul(mat2_mul(cat.grp.a, imat2_to_f64(op.mf)), cat.grp.ainv))
            .collect();
        let (origin_shift_frac, tau) = Self::quantize_taus(&matrices, proper_order, &raw_ops)?;
        let (mut mult, mut nu) = (vec![vec![0usize; m]; m], vec![vec![[0i64; 2]; m]; m]);
        for i in 0..m {
            for j in 0..m {
                let prod = mat_mul_i64(matrices[i], matrices[j]);
                let k = matrices.iter().position(|&r| r == prod)
                    .ok_or_else(|| "point group is not closed under multiplication".to_string())?;
                mult[i][j] = k;
                let diff = tau[i].add(tau[j].mat_mul(matrices[i])).sub(tau[k]);
                nu[i][j] = diff.to_int_exact().ok_or_else(|| {
                    format!("2-cocycle nu({i}, {j}) = {:?}/{} is not an integer vector", diff.num, diff.den)
                })?;
            }
        }
        let ab_full = AbelianMap::from_space_group(m, &matrices, &mult, &nu);
        let ab_proper = AbelianMap::from_space_group(proper_order, &matrices[..proper_order], &mult, &nu);
        Ok(Self {
            group: cat.group.clone(),
            proper_order,
            matrices,
            cart_matrices,
            tau,
            nu,
            mult,
            origin_shift_frac,
            a: cat.grp.a,
            a_inv: cat.grp.ainv,
            ab_full,
            ab_proper,
        })
    }

    fn quantize_taus(
        matrices: &[[[i64; 2]; 2]],
        proper_order: usize,
        raw_ops: &[FracOp2D],
    ) -> Result<([f64; 2], Vec<Frac2>), String> {
        let try_shift = |shift: [f64; 2], tol: f64| -> Option<Vec<Frac2>> {
            matrices.iter().zip(raw_ops).map(|(&r, op)| {
                let im = i_minus(r);
                let s0 = op.t[0] - (im[0][0] as f64 * shift[0] + im[0][1] as f64 * shift[1]);
                let s1 = op.t[1] - (im[1][0] as f64 * shift[0] + im[1][1] as f64 * shift[1]);
                Frac2::from_f64_quantized([s0.rem_euclid(1.0), s1.rem_euclid(1.0)], tol).map(Frac2::mod_int)
            }).collect()
        };
        if let Some(taus) = try_shift([0.0, 0.0], 1e-6) { return Ok(([0.0, 0.0], taus)); }
        let shift = if proper_order > 1 {
            let im = i_minus(matrices[1]);
            solve2([[im[0][0] as f64, im[0][1] as f64], [im[1][0] as f64, im[1][1] as f64]], raw_ops[1].t)
        } else if matrices.len() > proper_order {
            [0.5 * raw_ops[proper_order].t[0], 0.5 * raw_ops[proper_order].t[1]]
        } else { [0.0, 0.0] };
        let taus = try_shift(shift, 1e-5)
            .ok_or_else(|| "failed to quantize fractional translations tau(R) to exact rationals".to_string())?;
        Ok((shift, taus))
    }

    /// Exact symmorphic test using only `Frac2` and `SNF` (zero floating-point heuristics).
    pub fn is_symmorphic(&self) -> bool {
        if self.matrices.len() == 1 { return true; }
        if self.proper_order == 1 { return self.has_pure_mirror_for_op(1); }
        let im0 = i_minus(self.matrices[1]);
        let (det, adj) = (det_i64(im0), [[im0[1][1], -im0[0][1]], [-im0[1][0], im0[0][0]]]);
        let sub = LatticeCoset::new([0, 0], &[[im0[0][0], im0[1][0]], [im0[0][1], im0[1][1]]]);
        for l0 in 0..sub.basis_cols[0][0] {
            for l1 in 0..sub.basis_cols[1][1] {
                let s = self.tau[1].add(Frac2::from_int([l0, l1])).mat_mul(adj).div_scalar(det);
                if self.matrices.iter().zip(&self.tau).all(|(&r, &tr)| s.mat_mul(i_minus(r)).sub(tr).to_int_exact().is_some()) {
                    return true;
                }
            }
        }
        false
    }

    fn has_pure_mirror_for_op(&self, op_idx: usize) -> bool {
        let (ip, sq) = (i_plus(self.matrices[op_idx]), self.nu[op_idx][op_idx]);
        solve_int_system_exists(&[vec![ip[0][0], ip[0][1]], vec![ip[1][0], ip[1][1]]], &[-sq[0], -sq[1]])
    }

    /// Exact order of coset element `(l, op_idx)` in full abelianization `G^ab` (`None` if infinite).
    pub fn full_abelian_order(&self, l: [i64; 2], op_idx: usize) -> Option<i64> {
        self.ab_full.order_of(&self.ab_full.map_op(l, op_idx))
    }

    /// Convert a Cartesian isometry `iso in G` into exact `(l_vec, op_idx)` in the shifted algebra frame.
    pub fn isometry_to_exact(&self, iso: &Isometry2) -> Result<([i64; 2], usize), String> {
        let op_idx = self.cart_matrices.iter()
            .position(|&mc| np_allclose_mat(mc, iso.m, 1e-4, 1e-5))
            .ok_or_else(|| "isometry linear part not in space group point group".to_string())?;
        let (t_frac, im, tau_f) = (mat2_vec(self.a_inv, iso.t), i_minus(self.matrices[op_idx]), self.tau[op_idx].to_f64());
        let diff = [
            t_frac[0] - (im[0][0] as f64 * self.origin_shift_frac[0] + im[0][1] as f64 * self.origin_shift_frac[1]) - tau_f[0],
            t_frac[1] - (im[1][0] as f64 * self.origin_shift_frac[0] + im[1][1] as f64 * self.origin_shift_frac[1]) - tau_f[1],
        ];
        let l = [round_even(diff[0]) as i64, round_even(diff[1]) as i64];
        if (diff[0] - l[0] as f64).abs() > 1e-4 || (diff[1] - l[1] as f64).abs() > 1e-4 {
            return Err(format!("isometry translation {:?} is not in space group coset", diff));
        }
        Ok((l, op_idx))
    }

    fn delta(&self, r_idx: usize, s_idx: usize) -> [i64; 2] {
        [self.nu[s_idx][r_idx][0] - self.nu[r_idx][s_idx][0], self.nu[s_idx][r_idx][1] - self.nu[r_idx][s_idx][1]]
    }

    fn improper_sectors(&self) -> Vec<ImproperSectorInfo> {
        let m = self.matrices.len();
        if self.proper_order == m { return Vec::new(); }
        let (mut visited, mut classes) = (vec![false; m], Vec::new());
        for i in self.proper_order..m {
            if visited[i] { continue; }
            let mut cls = BTreeSet::new();
            for s in 0..m {
                let s_inv = self.matrices.iter().position(|&r| r == inv_i64(self.matrices[s])).expect("inverse in P");
                let conj = self.mult[self.mult[s][i]][s_inv];
                cls.insert(conj);
                visited[conj] = true;
            }
            classes.push(cls.into_iter().collect::<Vec<_>>());
        }
        let mut k_plus_cols = Vec::new();
        for &r in &self.matrices[..self.proper_order] {
            let im = i_minus(r);
            k_plus_cols.extend([[im[0][0], im[1][0]], [im[0][1], im[1][1]]]);
        }
        let k_plus = LatticeCoset::new([0, 0], &k_plus_cols);

        let mut sectors: Vec<ImproperSectorInfo> = classes.into_iter().map(|cls| {
            let rep = *cls.iter().min_by_key(|&&idx| {
                let u = reflection_axis(self.matrices[idx]);
                (u[0].abs() + u[1].abs(), -u[0], -u[1])
            }).expect("non-empty improper class");
            let (r, axis, im_r) = (self.matrices[rep], reflection_axis(self.matrices[rep]), i_minus(self.matrices[rep]));
            let snf_qr = smith_normal_form(&[vec![im_r[0][0], im_r[0][1]], vec![im_r[1][0], im_r[1][1]]]);
            let (qr_free, qr_tor) = invariant_factors_from_snf(2, &snf_qr.diag);
            let quotient_label = format_abelian_group(qr_free, &qr_tor);
            let has_pure_mirror = self.has_pure_mirror_for_op(rep);
            let centralizer: Vec<usize> = (0..m).filter(|&s| self.mult[s][rep] == self.mult[rep][s]).collect();
            let fixed_point_free_centralizer = centralizer.iter().any(|&s| {
                let im_s = i_minus(self.matrices[s]);
                let mat = vec![
                    vec![im_s[0][0], im_s[0][1], im_r[0][0], im_r[0][1]],
                    vec![im_s[1][0], im_s[1][1], im_r[1][0], im_r[1][1]],
                ];
                !solve_int_system_exists(&mat, &self.delta(rep, s))
            });
            let square_proper_channel = if has_pure_mirror {
                "trivial (0)".to_string()
            } else {
                let (ip_r, sq) = (i_plus(r), self.nu[rep][rep]);
                let sq_coset = LatticeCoset::new([-sq[0], -sq[1]], &[[ip_r[0][0], ip_r[1][0]], [ip_r[0][1], ip_r[1][1]]]);
                let b_sq = shortest_nonzero_in_coset(&sq_coset, self.a).unwrap_or([-sq[0], -sq[1]]);
                if k_plus.index() == Some(2) {
                    let desc = LatticeCoset::new(b_sq, &k_plus.basis_cols).describe();
                    if desc.contains("b1+b2 = 1 mod 2") {
                        "odd-parity dislocation (b1+b2 = 1 mod 2)".to_string()
                    } else { format!("non-trivial glide-square dislocation ({desc})") }
                } else { format!("glide-square dislocation (shortest b = ({}, {}))", b_sq[0], b_sq[1]) }
            };
            let dir = match axis { [1, 0] | [0, 1] => "axial", [1, 1] | [1, -1] => "diagonal", _ => "oblique" };
            let kind = if !has_pure_mirror { "pure glide" } else if qr_tor.is_empty() { "mirror/glide" } else { "mirror" };
            let fp_note = if fixed_point_free_centralizer { "fixed-point-free: no Z_P(R)-invariant mirror" } else { "has fixed points in Q_R" };
            ImproperSectorInfo {
                rep_matrix: r,
                rep_tau: self.tau[rep].to_f64(),
                label: format!("{dir} {kind}"),
                parallel_axis: axis,
                quotient_label: quotient_label.clone(),
                has_pure_mirror,
                fixed_point_free_centralizer,
                centralizer_summary: format!("|Z_P(R)| = {} on Q_R = {quotient_label} ({fp_note}); square: {square_proper_channel}", centralizer.len()),
                square_proper_channel,
            }
        }).collect();
        sectors.sort_by_key(|s| (s.parallel_axis[0].abs() + s.parallel_axis[1].abs(), -s.parallel_axis[0], -s.parallel_axis[1]));
        sectors
    }

    fn build_centre_lookup(&self, cat: &DefectCatalogue) -> BTreeMap<usize, Vec<(LatticeCoset, String)>> {
        let mut table = BTreeMap::new();
        for op_idx in 1..self.proper_order {
            let (r, mc) = (self.matrices[op_idx], self.cart_matrices[op_idx]);
            let (ord, im) = (rot_order(r), i_minus(r));
            let cols = [[im[0][0], im[1][0]], [im[0][1], im[1][1]]];
            let mut entries = Vec::new();
            for cc in &cat.centre_classes {
                if cc.order % ord != 0 { continue; }
                let mc_pt = mat2_vec(mc, cc.point);
                let iso = Isometry2 { m: mc, t: [cc.point[0] - mc_pt[0], cc.point[1] - mc_pt[1]] };
                let (l0, idx) = self.isometry_to_exact(&iso).expect("centre rotation in G");
                debug_assert_eq!(idx, op_idx);
                let mut seen = BTreeSet::new();
                for s in 0..self.proper_order {
                    let (ls, del) = (mat_vec_i64(self.matrices[s], l0), self.delta(op_idx, s));
                    let coset = LatticeCoset::new([ls[0] + del[0], ls[1] + del[1]], &cols);
                    if seen.insert(coset.offset) { entries.push((coset, cc.label.clone())); }
                }
            }
            table.insert(op_idx, entries);
        }
        table
    }

    fn compute_fusion_rows(&self, cat: &DefectCatalogue) -> Result<Vec<AlgebraicFusionRow>, String> {
        enum Elem {
            Discl { id: String, op_idx: usize, l_rep: [i64; 2], angle_deg: i64 },
            Disloc { id: String, members: Vec<[i64; 2]>, b_rep: [i64; 2] },
        }
        impl Elem { fn id(&self) -> &str { match self { Self::Discl { id, .. } | Self::Disloc { id, .. } => id } } }
        let mut elems = Vec::new();
        for d in &cat.disclinations {
            let (l_rep, op_idx) = self.isometry_to_exact(&d.holonomy)?;
            elems.push(Elem::Discl { id: d.id.clone(), op_idx, l_rep, angle_deg: round_even(d.angle) as i64 });
        }
        for t in &cat.dislocations {
            let members = t.members.iter().map(|&m| [i64::from(m[0]), i64::from(m[1])]).collect();
            elems.push(Elem::Disloc { id: t.id.clone(), members, b_rep: [i64::from(t.b_frac[0]), i64::from(t.b_frac[1])] });
        }
        let lookup = self.build_centre_lookup(cat);
        let order: BTreeMap<String, usize> = cat.centre_classes.iter().enumerate().map(|(i, c)| (c.label.clone(), i)).collect();
        let charge_of = |e: &Elem| -> AbelianElement {
            match e {
                Elem::Discl { op_idx, l_rep, .. } => self.ab_proper.map_op(*l_rep, *op_idx),
                Elem::Disloc { b_rep, .. } => self.ab_proper.map_op([-b_rep[0], -b_rep[1]], 0),
            }
        };
        let mut rows = Vec::new();
        for i in 0..elems.len() {
            for j in i..elems.len() {
                let (ea, eb) = (&elems[i], &elems[j]);
                let abelian_sum_proper = self.ab_proper.format_elem(&self.ab_proper.add(&charge_of(ea), &charge_of(eb)));
                let outcome = match (ea, eb) {
                    (Elem::Discl { op_idx: ra, l_rep: la, angle_deg: aa, .. }, Elem::Discl { op_idx: rb, l_rep: lb, angle_deg: ab, .. }) => {
                        self.fuse_discl_discl(*ra, *la, *aa, *rb, *lb, *ab, &lookup, &order)
                    }
                    (Elem::Discl { op_idx: rd, l_rep: ld, angle_deg: ad, .. }, Elem::Disloc { members, .. })
                    | (Elem::Disloc { members, .. }, Elem::Discl { op_idx: rd, l_rep: ld, angle_deg: ad, .. }) => {
                        self.fuse_discl_disloc(*rd, *ld, *ad, members, &lookup, &order)
                    }
                    (Elem::Disloc { members: ma, .. }, Elem::Disloc { members: mb, .. }) => self.fuse_disloc_disloc(ma, mb),
                };
                rows.push(AlgebraicFusionRow { defect_a: ea.id().to_string(), defect_b: eb.id().to_string(), abelian_sum_proper, outcomes: vec![outcome] });
            }
        }
        Ok(rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn fuse_discl_discl(
        &self, ra: usize, la: [i64; 2], angle_a: i64, rb: usize, lb: [i64; 2], angle_b: i64,
        lookup: &BTreeMap<usize, Vec<(LatticeCoset, String)>>, order: &BTreeMap<String, usize>,
    ) -> AlgebraicFusionOutcome {
        let (r_out, angle_out, ma, mb) = (self.mult[ra][rb], angle_a + angle_b, self.matrices[ra], self.matrices[rb]);
        let (ima, r_imb) = (i_minus(ma), mat_mul_i64(ma, i_minus(mb)));
        let k_cols = [[ima[0][0], ima[1][0]], [ima[0][1], ima[1][1]], [r_imb[0][0], r_imb[1][0]], [r_imb[0][1], r_imb[1][1]]];
        let mut raw_t = Vec::new();
        for sa in 0..self.proper_order {
            let (las, da) = (mat_vec_i64(self.matrices[sa], la), self.delta(ra, sa));
            let la_img = [las[0] + da[0], las[1] + da[1]];
            for sb in 0..self.proper_order {
                let (lbs, db) = (mat_vec_i64(self.matrices[sb], lb), self.delta(rb, sb));
                let ra_lb = mat_vec_i64(ma, [lbs[0] + db[0], lbs[1] + db[1]]);
                let nu_ab = self.nu[ra][rb];
                raw_t.push([la_img[0] + ra_lb[0] + nu_ab[0], la_img[1] + ra_lb[1] + nu_ab[1]]);
            }
        }
        if r_out == 0 {
            let raw_b: Vec<[i64; 2]> = raw_t.iter().map(|&t| [-t[0], -t[1]]).collect();
            let union = LatticeCosetUnion::from_sublattice_and_offsets(&k_cols, &raw_b);
            let (idx, has_zero, shortest) = (union.index(), union.contains_zero(), shortest_nonzero_in_union(&union, self.a));
            let (kind, allowed, prefix) = if angle_out == 0 {
                ("dislocation", true, "dislocation".to_string())
            } else { ("whole_turn", false, format!("whole turn {:+}°", angle_out)) };
            return AlgebraicFusionOutcome {
                kind: kind.to_string(),
                frank_angle_deg: angle_out as f64,
                allowed_single_defect: allowed,
                summary: format!("{prefix} in {} ({})", union.describe(), if has_zero { "contains 0" } else { "excludes 0" }),
                allowed_centre_classes: Vec::new(),
                burgers_sublattice_index: idx,
                burgers_coset_contains_zero: Some(has_zero),
                shortest_burgers_frac: shortest,
                burgers_domain: Some(BurgersDomain::CosetUnion(union)),
            };
        }
        let im_out = i_minus(self.matrices[r_out]);
        let mut combined = vec![[im_out[0][0], im_out[1][0]], [im_out[0][1], im_out[1][1]]];
        combined.extend_from_slice(&k_cols);
        let q_union = LatticeCosetUnion::from_sublattice_and_offsets(&combined, &raw_t);
        disclination_outcome(angle_out, match_centre_classes(&q_union, &lookup[&r_out], order))
    }

    fn fuse_discl_disloc(
        &self, rd: usize, ld: [i64; 2], angle_d: i64, members: &[[i64; 2]],
        lookup: &BTreeMap<usize, Vec<(LatticeCoset, String)>>, order: &BTreeMap<String, usize>,
    ) -> AlgebraicFusionOutcome {
        let im_d = i_minus(self.matrices[rd]);
        let cols = [[im_d[0][0], im_d[1][0]], [im_d[0][1], im_d[1][1]]];
        let mut raw_t = Vec::new();
        for sd in 0..self.proper_order {
            let (lds, dd) = (mat_vec_i64(self.matrices[sd], ld), self.delta(rd, sd));
            for &b in members { raw_t.push([lds[0] + dd[0] - b[0], lds[1] + dd[1] - b[1]]); }
        }
        let q_union = LatticeCosetUnion::from_sublattice_and_offsets(&cols, &raw_t);
        disclination_outcome(angle_d, match_centre_classes(&q_union, &lookup[&rd], order))
    }

    fn fuse_disloc_disloc(&self, ma: &[[i64; 2]], mb: &[[i64; 2]]) -> AlgebraicFusionOutcome {
        let mut set = BTreeSet::new();
        for &a in ma { for &b in mb { set.insert([a[0] + b[0], a[1] + b[1]]); } }
        let (has_zero, shortest) = (set.contains(&[0, 0]), shortest_nonzero_in_set(&set, self.a));
        let summary = if shortest.is_empty() && has_zero {
            "dislocation annihilation (0)".to_string()
        } else { format!("dislocation finite sum ({}; {} reachable vectors)", if has_zero { "contains 0" } else { "excludes 0" }, set.len()) };
        AlgebraicFusionOutcome {
            kind: "dislocation".to_string(),
            frank_angle_deg: 0.0,
            allowed_single_defect: true,
            summary,
            allowed_centre_classes: Vec::new(),
            burgers_sublattice_index: None,
            burgers_coset_contains_zero: Some(has_zero),
            shortest_burgers_frac: shortest,
            burgers_domain: Some(BurgersDomain::FiniteSet(set)),
        }
    }
}

fn match_centre_classes(
    q_union: &LatticeCosetUnion, lookup: &[(LatticeCoset, String)], order: &BTreeMap<String, usize>,
) -> Vec<String> {
    let mut labels: Vec<String> = lookup.iter().filter(|(c, _)| q_union.contains(c.offset))
        .map(|(_, l)| l.clone()).collect::<BTreeSet<_>>().into_iter().collect();
    labels.sort_by_key(|l| order.get(l).copied().unwrap_or(usize::MAX));
    labels
}

fn disclination_outcome(angle_deg: i64, allowed_centre_classes: Vec<String>) -> AlgebraicFusionOutcome {
    let allowed = angle_deg.abs() <= 180 && !allowed_centre_classes.is_empty();
    let where_str = match allowed_centre_classes.as_slice() {
        [] => "none".to_string(),
        [one] => one.clone(),
        many => format!("{{{}}}", many.join(", ")),
    };
    let summary = if allowed { format!("disclination {:+}° at {where_str}", angle_deg) } else {
        format!("disclination {:+}° at {where_str} (|angle| > 180°, not a single catalogue defect)", angle_deg)
    };
    AlgebraicFusionOutcome {
        kind: "disclination".to_string(),
        frank_angle_deg: angle_deg as f64,
        allowed_single_defect: allowed,
        summary,
        allowed_centre_classes,
        burgers_sublattice_index: None,
        burgers_coset_contains_zero: None,
        shortest_burgers_frac: Vec::new(),
        burgers_domain: None,
    }
}

/// Analyze a 2D wallpaper lattice's exact space-group algebra, abelianizations `G^ab` & `(G+)^ab`,
/// improper reflection/glide sectors (`Q_R // Z_P(R)`), and lifted `G~+` algebraic fusion rules.
pub fn analyze_space_group_fusion(
    group: &str, cell: &Cell2D, frame: Frame2D, bmax: f64,
) -> Result<SpaceGroupAlgebraReport, String> {
    let cat = DefectCatalogue::with_bounds(group, cell, frame, bmax, 0.0);
    let alg = SpaceGroupAlgebra::from_catalogue(&cat)?;
    Ok(SpaceGroupAlgebraReport {
        group: alg.group.clone(),
        point_group_order: alg.matrices.len(),
        proper_order: alg.proper_order,
        symmorphic: alg.is_symmorphic(),
        ab_full: AbelianGroupSummary { free_rank: alg.ab_full.free_rank, torsion: alg.ab_full.torsion.clone(), label: alg.ab_full.label.clone() },
        ab_proper: AbelianGroupSummary { free_rank: alg.ab_proper.free_rank, torsion: alg.ab_proper.torsion.clone(), label: alg.ab_proper.label.clone() },
        improper_sectors: alg.improper_sectors(),
        fusion_rows: alg.compute_fusion_rows(&cat)?,
        provenance: PROVENANCE.to_string(),
    })
}

/// Compute exact order-independent abelian charges `(proper_ab_charge, full_ab_charge)` in
/// `(G+)^ab` and `G^ab` for any sequence of `DefectSpec`s.
pub fn specs_abelian_charge(
    group: &str, cell: &Cell2D, frame: Frame2D, specs: &[DefectSpec],
) -> Result<(String, String), String> {
    let cat = DefectCatalogue::with_bounds(group, cell, frame, 0.0, 0.0);
    let alg = SpaceGroupAlgebra::from_catalogue(&cat)?;
    let (mut sum_full, mut sum_proper, mut all_proper) = (alg.ab_full.zero(), alg.ab_proper.zero(), true);
    for spec in specs {
        let (l, op_idx) = match spec {
            DefectSpec::Dislocation { b, .. } => {
                let bf = mat2_vec(cat.grp.ainv, *b);
                let bi = [round_even(bf[0]) as i64, round_even(bf[1]) as i64];
                if (bf[0] - bi[0] as f64).abs() > 1e-4 || (bf[1] - bi[1] as f64).abs() > 1e-4 {
                    return Err(format!("dislocation Burgers vector {:?} is not a lattice vector", b));
                }
                ([-bi[0], -bi[1]], 0)
            }
            DefectSpec::Disclination { angle_deg, at, .. } => alg.isometry_to_exact(&rotation_about(*at, angle_deg.to_radians()))?,
            DefectSpec::HalfWall { wall_id, sigma, .. } => {
                let iso = if let Some((m, t)) = sigma { Isometry2 { m: *m, t: *t } }
                else if let Some(wid) = wall_id {
                    cat.mobius_seams.iter().find(|w| w.id == *wid).map(|w| w.sigma)
                        .or_else(|| cat.domain_walls.iter().find(|w| w.id == *wid).map(|w| w.sigma))
                        .ok_or_else(|| format!("half-wall '{wid}' has no space-group holonomy sigma"))?
                } else { return Err("half-wall spec requires sigma or wall_id".to_string()); };
                alg.isometry_to_exact(&iso)?
            }
        };
        sum_full = alg.ab_full.add(&sum_full, &alg.ab_full.map_op(l, op_idx));
        if op_idx < alg.proper_order { sum_proper = alg.ab_proper.add(&sum_proper, &alg.ab_proper.map_op(l, op_idx)); }
        else { all_proper = false; }
    }
    let proper_str = if all_proper { alg.ab_proper.format_elem(&sum_proper) } else { "improper (not in G+)".to_string() };
    Ok((proper_str, alg.ab_full.format_elem(&sum_full)))
}

#[cfg(test)]
#[path = "algebraic_fusion_tests.rs"]
mod tests;
