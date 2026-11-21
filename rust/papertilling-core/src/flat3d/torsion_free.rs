//! Exact rational fixed-point-free (torsion-free) space-group classifier
//! over arbitrary 2D and 3D `SpaceGroupAnalyzer` extensions.

use serde::{Deserialize, Serialize};

use crate::bieberbach::analyzer::imat_to_rmat;
use crate::bieberbach::rat::rmat_vec;
use crate::bieberbach::{
    ConvOp, IntMat, Rat, SpaceGroupAnalyzer, from_wallpaper_group, solve_int_linear,
    summarize_analyzer,
};
use crate::plane::groups::WALLPAPER_GROUPS;
use crate::space3d::{build_analyzer_3d, list_space_groups_3d};

/// Witness of a non-trivial finite-order / fixed-point-bearing element in a space group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedPointObstruction {
    pub r_idx: usize,
    pub order: usize,
    pub power_k: usize,
    pub r_conv: Vec<Vec<i64>>,
    pub tau_conv: Vec<String>,
    pub fixed_point_conv: Option<Vec<String>>,
    pub description: String,
}

/// Complete exact torsion-free classification report for a crystallographic space group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TorsionFreeReport {
    pub dim: usize,
    pub number: i32,
    pub symbol: String,
    pub is_torsion_free: bool,
    pub point_order: usize,
    pub is_split: bool,
    pub h2_label: String,
    pub ab_full_label: String,
    pub obstructions: Vec<FixedPointObstruction>,
}

fn sum_powers_int_mat(sg: &SpaceGroupAnalyzer, r_idx: usize, ord: usize) -> IntMat {
    let n = sg.dim;
    let mut s_mat = IntMat::zeros(n, n);
    let mut cur = 0usize;
    for _ in 0..ord {
        let rm = &sg.p_ops[cur];
        for (i, row) in rm.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                s_mat.add_assign(i, j, i128::from(val));
            }
        }
        cur = sg.mul_idx(cur, r_idx);
    }
    s_mat
}

fn rmat_to_i64_checked(m: &[Vec<Rat>]) -> Vec<Vec<i64>> {
    m.iter()
        .map(|row| row.iter().map(|r| r.as_i64().unwrap_or(0)).collect())
        .collect()
}

/// Check if coset `(R | tau[r_idx] + Z^n)` contains a rational fixed point `x in Q^n`.
/// Returns `Some((lattice_shift_l, fixed_point_prim))` iff `S_R * tau in S_R(Z^n)`.
fn coset_fixed_point_prim(
    sg: &SpaceGroupAnalyzer,
    r_idx: usize,
) -> Option<(Vec<i64>, Vec<Rat>)> {
    if r_idx == 0 {
        return None;
    }
    let ord = sg.mat_order(r_idx);
    let s_mat = sum_powers_int_mat(sg, r_idx, ord);
    let n = sg.dim;
    let tau = &sg.tau[r_idx];
    let mut rhs = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = Rat::ZERO;
        for (j, &tj) in tau.iter().enumerate() {
            let sij = i64::try_from(s_mat.get(i, j)).ok()?;
            acc = acc + Rat::from_int(sij) * tj;
        }
        let int_val = acc.as_i64()?;
        rhs.push(-i128::from(int_val));
    }
    let shift = solve_int_linear(&s_mat, &rhs)?;
    let l_i64: Vec<i64> = shift.iter().map(|&v| v as i64).collect();
    let w: Vec<Rat> = tau
        .iter()
        .zip(&l_i64)
        .map(|(&t, &l)| t + Rat::from_int(l))
        .collect();
    // Exact rational fixed point: x = (1 / ord) * sum_{k=1}^{ord-1} (ord - k) * R^{k-1} * w.
    let mut x_sum = vec![Rat::ZERO; n];
    let mut cur_r = 0usize;
    for k in 1..ord {
        let coeff = Rat::from_int((ord - k) as i64);
        let r_mat = imat_to_rmat(&sg.p_ops[cur_r]);
        let r_w = rmat_vec(&r_mat, &w);
        for (xi, rw) in x_sum.iter_mut().zip(r_w) {
            *xi = *xi + coeff * rw;
        }
        cur_r = sg.mul_idx(cur_r, r_idx);
    }
    let inv_ord = Rat::new(1, ord as i64);
    let x_prim: Vec<Rat> = x_sum.into_iter().map(|v| v * inv_ord).collect();
    Some((l_i64, x_prim))
}

/// Exact rational torsion-free check across all non-identity cosets `(R | tau_R + Z^n)`
/// (including every nontrivial power `(R | tau_R)^k` via closure of `sg.p_ops`).
#[must_use]
pub fn is_torsion_free_exact(sg: &SpaceGroupAnalyzer) -> bool {
    (1..sg.p_ops.len()).all(|r_idx| coset_fixed_point_prim(sg, r_idx).is_none())
}

/// Detailed torsion-free classification report, identifying all fixed-point-bearing
/// cosets and whether an element `(R|t)` fails directly or via a power `(R|t)^k`.
#[must_use]
pub fn analyze_torsion_free(sg: &SpaceGroupAnalyzer) -> TorsionFreeReport {
    let summary = summarize_analyzer(sg, &[2]);
    let mut obstructions = Vec::new();
    for r_idx in 1..sg.p_ops.len() {
        let ord = sg.mat_order(r_idx);
        // Check direct fixed point first, else check proper powers 2..ord (e.g. 4_2^2 = 2).
        let mut hit = coset_fixed_point_prim(sg, r_idx).map(|(l, x)| (1usize, r_idx, l, x));
        if hit.is_none() {
            let mut pow_r = r_idx;
            for k in 2..ord {
                pow_r = sg.mul_idx(pow_r, r_idx);
                if pow_r != 0
                    && let Some((l, x)) = coset_fixed_point_prim(sg, pow_r)
                {
                    hit = Some((k, pow_r, l, x));
                    break;
                }
            }
        }
        if let Some((power_k, hit_r, _l, x_prim)) = hit {
            let r_conv = rmat_to_i64_checked(&sg.r_conv_rat(r_idx));
            let tau_conv: Vec<String> = sg
                .to_conv_frac(&sg.tau[r_idx])
                .into_iter()
                .map(|q| q.to_string())
                .collect();
            let fp_conv: Vec<String> = sg
                .to_conv_frac(&x_prim)
                .into_iter()
                .map(|q| q.to_string())
                .collect();
            let desc = if power_k == 1 {
                format!(
                    "[PROVED] coset #{r_idx} (order {ord}, tau={}) fixes [{}]",
                    tau_conv.join(","),
                    fp_conv.join(", ")
                )
            } else {
                format!(
                    "[PROVED] coset #{r_idx} (order {ord}) power ^{power_k} -> coset #{hit_r} fixes [{}]",
                    fp_conv.join(", ")
                )
            };
            obstructions.push(FixedPointObstruction {
                r_idx,
                order: ord,
                power_k,
                r_conv,
                tau_conv,
                fixed_point_conv: Some(fp_conv),
                description: desc,
            });
        }
    }
    TorsionFreeReport {
        dim: sg.dim,
        number: sg.number,
        symbol: sg.symbol.clone(),
        is_torsion_free: obstructions.is_empty(),
        point_order: sg.p_ops.len(),
        is_split: summary.is_split,
        h2_label: summary.h2_label,
        ab_full_label: summary.ab_full_label,
        obstructions,
    }
}

/// Analyze a custom space group specification `(dim, number, symbol, ops, tb_rows)`
/// for exact fixed-point-freeness.
///
/// # Errors
/// Returns an error if the generators or translation basis cannot construct a valid
/// `SpaceGroupAnalyzer`.
pub fn is_torsion_free_conv_ops(
    dim: usize,
    number: i32,
    symbol: &str,
    ops: &[ConvOp],
    tb_rows: &[Vec<Rat>],
) -> Result<TorsionFreeReport, String> {
    let sg = SpaceGroupAnalyzer::new(dim, number, symbol, ops, tb_rows)?;
    Ok(analyze_torsion_free(&sg))
}

/// Classify all 17 2D wallpaper groups; only `p1` and `pg` are torsion-free.
///
/// # Errors
/// Returns an error if any built-in wallpaper group definition fails to initialize.
pub fn classify_all_2d_torsion_free() -> Result<Vec<TorsionFreeReport>, String> {
    let mut out = Vec::with_capacity(WALLPAPER_GROUPS.len());
    for sym in WALLPAPER_GROUPS {
        let sg = from_wallpaper_group(sym)?;
        out.push(analyze_torsion_free(&sg));
    }
    Ok(out)
}

/// Classify all 230 3D ITA space groups; only the 13 Bieberbach groups pass
/// (ITA #1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170; cf. Conway-Rossetti 2003).
///
/// # Errors
/// Returns an error if any ITA entry fails to construct a `SpaceGroupAnalyzer`.
pub fn classify_all_3d_torsion_free() -> Result<Vec<TorsionFreeReport>, String> {
    let groups = list_space_groups_3d();
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let sg = build_analyzer_3d(g.number)?;
        out.push(analyze_torsion_free(&sg));
    }
    Ok(out)
}
