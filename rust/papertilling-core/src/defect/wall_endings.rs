//! Algebraic classification of wall endings via `(sigma * G) ∩ E+(2)`,
//! matching `papertilling/defects/_wall_endings.py`.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::defect::core_graph::TWO_PI;
use crate::defect::equivalence::fmt_vec2_round4;
use crate::defect::sym_helper::{
    LatticeGroup, mat2_det, mat2_inv, mat2_mul, mat2_transpose, mat2_vec, norm, np_allclose_mat,
    py_mod, round_dec, round_even, solve2,
};
use crate::types::{Cell2D, Isometry2};

/// Algebraic classification of how a straight wall `sigma` can terminate in an orientable sheet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WallEndingInfo {
    pub can_end_orientable: bool,
    pub ending_kind: String,
    pub is_full_lattice_burgers: bool,
    pub burgers: Option<[f64; 2]>,
    pub burgers_norm: Option<f64>,
    pub frank_angle_deg: Option<f64>,
    pub centre: Option<[f64; 2]>,
    pub holonomy_linear: Option<[[f64; 2]; 2]>,
    pub holonomy_trans: Option<[f64; 2]>,
    pub summary: String,
}

#[inline]
fn clean_val(x: f64) -> f64 { if x.abs() < 1e-11 { 0.0 } else { x } }

#[inline]
fn clean_vec(v: [f64; 2]) -> [f64; 2] { [clean_val(v[0]), clean_val(v[1])] }

#[inline]
fn clean_mat(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] { [clean_vec(m[0]), clean_vec(m[1])] }

fn norm_angle_deg(r: [[f64; 2]; 2]) -> f64 {
    let ang = r[1][0].atan2(r[0][0]).to_degrees();
    if ang <= -180.0 + 1e-6 { 180.0 } else if ang > 180.0 { ang - 360.0 } else { ang }
}

fn strip_trailing_zeros(mut s: String) -> String {
    if s.contains('.') {
        while s.ends_with('0') { s.pop(); }
        if s.ends_with('.') { s.pop(); }
    }
    s
}

fn fmt_signed_g4(val: f64) -> String {
    if val == 0.0 {
        return "+0".to_string();
    }
    let sign = if val < 0.0 { '-' } else { '+' };
    let a = val.abs();
    let scale = 10.0_f64.powi(3 - a.log10().floor() as i32);
    let rounded = (a * scale).round_ties_even() / scale;
    let rexp = rounded.log10().floor() as i32;
    let body = if !(-4..4).contains(&rexp) {
        let mant = rounded / 10.0_f64.powi(rexp);
        format!("{}e{rexp:+03}", strip_trailing_zeros(format!("{mant:.3}")))
    } else {
        let dec = (3 - rexp).max(0) as usize;
        strip_trailing_zeros(format!("{rounded:.dec$}"))
    };
    format!("{sign}{body}")
}

fn obstructed(ending_kind: &str, summary: &str) -> WallEndingInfo {
    WallEndingInfo {
        can_end_orientable: false, ending_kind: ending_kind.to_string(), is_full_lattice_burgers: false,
        burgers: None, burgers_norm: None, frank_angle_deg: None, centre: None,
        holonomy_linear: None, holonomy_trans: None, summary: summary.to_string(),
    }
}

struct BestDisloc {
    score: (f64, f64),
    b: [f64; 2],
    t: [f64; 2],
    norm: f64,
}

fn best_partial_dislocation(grp: &LatticeGroup, m_sig: [[f64; 2]; 2], proper_trans: &[[f64; 2]]) -> WallEndingInfo {
    let mt_sig = mat2_transpose(m_sig);
    let mut best: Option<BestDisloc> = None;
    for &t_h in proper_trans {
        let mt_t = mat2_vec(mt_sig, t_h);
        let n0_f = mat2_vec(grp.ainv, [-mt_t[0], -mt_t[1]]);
        let n0 = [round_even(n0_f[0]) as i32, round_even(n0_f[1]) as i32];
        for dm in -4..=4 {
            for dn in -4..=4 {
                let lam = mat2_vec(grp.a, [(n0[0] + dm) as f64, (n0[1] + dn) as f64]);
                let ms_lam = mat2_vec(m_sig, lam);
                let mut t_cand = clean_vec([t_h[0] + ms_lam[0], t_h[1] + ms_lam[1]]);
                let mut b_cand = clean_vec([-t_cand[0], -t_cand[1]]);
                let mut bn = norm(b_cand);
                let ang = if bn < 1e-9 { b_cand = [0.0, 0.0]; t_cand = [0.0, 0.0]; bn = 0.0; 0.0 }
                          else { py_mod(b_cand[1].atan2(b_cand[0]), TWO_PI) };
                let score = (round_dec(bn, 9), round_dec(ang, 9));
                if best.as_ref().is_none_or(|b| score.0.total_cmp(&b.score.0).then_with(|| score.1.total_cmp(&b.score.1)) == Ordering::Less) {
                    best = Some(BestDisloc { score, b: b_cand, t: t_cand, norm: bn });
                }
            }
        }
    }
    let best = best.expect("proper_trans is non-empty");
    let (b_best, t_best, bn) = (best.b, best.t, best.norm);
    let f = mat2_vec(grp.ainv, b_best);
    let is_full = (f[0] - round_even(f[0])).abs() < 1e-5 && (f[1] - round_even(f[1])).abs() < 1e-5;
    let summary = if bn < 1e-7 {
        "full-lattice trivial ending (b = [0.0, 0.0], sigma in G)".to_string()
    } else if is_full {
        format!("full-lattice dislocation ending b = {} (|b| = {:.4})", fmt_vec2_round4(b_best), bn)
    } else {
        format!("partial dislocation ending b = {} (|b| = {:.4}, fractional)", fmt_vec2_round4(b_best), bn)
    };
    WallEndingInfo {
        can_end_orientable: true, ending_kind: "partial_dislocation".to_string(), is_full_lattice_burgers: is_full,
        burgers: Some(b_best), burgers_norm: Some(bn), frank_angle_deg: None, centre: None,
        holonomy_linear: Some([[1.0, 0.0], [0.0, 1.0]]), holonomy_trans: Some(t_best), summary,
    }
}

struct BestDiscl {
    score: (f64, u8, f64, f64),
    alpha: f64,
    r: [[f64; 2]; 2],
    t: [f64; 2],
    c: [f64; 2],
}

fn best_partial_disclination(grp: &LatticeGroup, m_sig: [[f64; 2]; 2], proper_rots: &[([[f64; 2]; 2], [f64; 2])]) -> WallEndingInfo {
    let mut best: Option<BestDiscl> = None;
    for &(r_h, t_h) in proper_rots {
        let alpha = norm_angle_deg(r_h);
        let i_r_inv = mat2_inv([[1.0 - r_h[0][0], -r_h[0][1]], [-r_h[1][0], 1.0 - r_h[1][1]]]);
        let c0 = mat2_vec(i_r_inv, t_h);
        let n0_f = solve2(mat2_mul(i_r_inv, mat2_mul(m_sig, grp.a)), [-c0[0], -c0[1]]);
        let n0 = [round_even(n0_f[0]) as i32, round_even(n0_f[1]) as i32];
        for dm in -4..=4 {
            for dn in -4..=4 {
                let lam = mat2_vec(grp.a, [(n0[0] + dm) as f64, (n0[1] + dn) as f64]);
                let ms_lam = mat2_vec(m_sig, lam);
                let t_cand = clean_vec([t_h[0] + ms_lam[0], t_h[1] + ms_lam[1]]);
                let c_cand = clean_vec(mat2_vec(i_r_inv, t_cand));
                let cn = norm(c_cand);
                let ang_c = if cn < 1e-9 { 0.0 } else { py_mod(c_cand[1].atan2(c_cand[0]), TWO_PI) };
                let score = (round_dec(alpha.abs(), 7), u8::from(alpha <= 0.0), round_dec(cn, 7), round_dec(ang_c, 7));
                let better = best.as_ref().is_none_or(|b| {
                    score.0.total_cmp(&b.score.0).then_with(|| score.1.cmp(&b.score.1))
                        .then_with(|| score.2.total_cmp(&b.score.2)).then_with(|| score.3.total_cmp(&b.score.3)) == Ordering::Less
                });
                if better {
                    best = Some(BestDiscl { score, alpha, r: clean_mat(r_h), t: t_cand, c: c_cand });
                }
            }
        }
    }
    let best = best.expect("proper_rots is non-empty");
    let (alpha_val, r_b, t_b, c_b) = (round_dec(best.alpha, 9), best.r, best.t, best.c);
    let summary = format!("partial disclination ending alpha = {} deg about {}", fmt_signed_g4(alpha_val), fmt_vec2_round4(c_b));
    WallEndingInfo {
        can_end_orientable: true, ending_kind: "partial_disclination".to_string(), is_full_lattice_burgers: false,
        burgers: None, burgers_norm: None, frank_angle_deg: Some(alpha_val), centre: Some(c_b),
        holonomy_linear: Some(r_b), holonomy_trans: Some(t_b), summary,
    }
}

pub(crate) fn classify_wall_ending_in_grp(
    grp: &LatticeGroup,
    sigma: Isometry2,
    wall_kind: &str,
) -> WallEndingInfo {
    if matches!(wall_kind, "mobius_seam" | "c" | "Wc") {
        return obstructed(
            "non_orientable_twist",
            "non-orientable twist (Mobius seam cannot terminate inside an orientable sheet)",
        );
    }
    if mat2_det(sigma.m) < 0.0 && grp.chiral {
        return obstructed(
            "chirality_obstruction",
            "chirality obstruction (orientation-reversing wall cannot end in an orientable chiral sheet)",
        );
    }

    let ident = [[1.0, 0.0], [0.0, 1.0]];
    let mut proper_trans = Vec::new();
    let mut proper_rots = Vec::new();
    for g in &grp.ops {
        let r_h = mat2_mul(sigma.m, g.m);
        if mat2_det(r_h) <= 0.0 {
            continue;
        }
        let mt = mat2_vec(sigma.m, g.t);
        let t_h = [mt[0] + sigma.t[0], mt[1] + sigma.t[1]];
        if np_allclose_mat(r_h, ident, 1e-5, 1e-6) {
            proper_trans.push(t_h);
        } else {
            proper_rots.push((r_h, t_h));
        }
    }
    if !proper_trans.is_empty() {
        return best_partial_dislocation(grp, sigma.m, &proper_trans);
    }
    if !proper_rots.is_empty() {
        return best_partial_disclination(grp, sigma.m, &proper_rots);
    }
    obstructed(
        "chirality_obstruction",
        "no proper coset element in (sigma * G) ∩ E+(2)",
    )
}

/// Classify how a wall `sigma` can terminate in an orientable sheet (`(sigma * G) ∩ E+(2)`).
pub fn classify_wall_ending(cell: &Cell2D, sigma: Isometry2, wall_kind: &str) -> WallEndingInfo {
    let grp = LatticeGroup::new(cell);
    classify_wall_ending_in_grp(&grp, sigma, wall_kind)
}

pub(crate) fn nearest_ending_holonomy_in_grp(
    grp: &LatticeGroup,
    sigma: Isometry2,
    at: [f64; 2],
    wall_kind: &str,
) -> Result<(WallEndingInfo, Isometry2), String> {
    let ending = classify_wall_ending_in_grp(grp, sigma, wall_kind);
    if !ending.can_end_orientable {
        return Err(ending.summary);
    }
    let r_h = ending.holonomy_linear.expect("orientable ending has linear part");
    let t_h = ending.holonomy_trans.expect("orientable ending has translation");
    if ending.ending_kind == "partial_dislocation" {
        return Ok((ending, Isometry2 { m: r_h, t: t_h }));
    }
    let i_r = [[1.0 - r_h[0][0], -r_h[0][1]], [-r_h[1][0], 1.0 - r_h[1][1]]];
    let i_r_inv = mat2_inv(i_r);
    let c0 = mat2_vec(i_r_inv, t_h);
    let l_mat = mat2_mul(i_r_inv, mat2_mul(sigma.m, grp.a));
    let n0_f = solve2(l_mat, [at[0] - c0[0], at[1] - c0[1]]);
    let n0 = [round_even(n0_f[0]) as i32, round_even(n0_f[1]) as i32];
    let mut best: Option<(f64, [f64; 2])> = None;
    for dm in -3..=3 {
        for dn in -3..=3 {
            let lam = mat2_vec(grp.a, [(n0[0] + dm) as f64, (n0[1] + dn) as f64]);
            let ms_lam = mat2_vec(sigma.m, lam);
            let t_cand = clean_vec([t_h[0] + ms_lam[0], t_h[1] + ms_lam[1]]);
            let c_cand = clean_vec(mat2_vec(i_r_inv, t_cand));
            let dist = (c_cand[0] - at[0]).hypot(c_cand[1] - at[1]);
            if best.as_ref().is_none_or(|(bd, _)| dist < *bd - 1e-9) {
                best = Some((dist, t_cand));
            }
        }
    }
    let (_, best_t) = best.expect("7x7 search is non-empty");
    Ok((ending, Isometry2 { m: r_h, t: best_t }))
}

/// Return `(ending_info, h)` with `h in (sigma * G) ∩ E+(2)` centred nearest `at`.
pub fn nearest_ending_holonomy(
    cell: &Cell2D,
    sigma: Isometry2,
    at: [f64; 2],
    wall_kind: &str,
) -> Result<(WallEndingInfo, Isometry2), String> {
    let grp = LatticeGroup::new(cell);
    nearest_ending_holonomy_in_grp(&grp, sigma, at, wall_kind)
}
