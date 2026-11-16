//! Crossing-free parameter intervals, enhanced-symmetry special points, and constrained sliders.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::loci::{Locus, supergroup_loci_for_family};
use super::tuning_eval::{BareBond, EvalFamily, cell_mat, mat2_inv, mat2_vec};
use crate::types::{Cell2D, Recipe};

mod sliders;
pub use sliders::{
    SliderSpec, SliderState, inward_limits, nominal_range, slider_value, snap_to_special,
};

const REFINE: usize = 33;
const ROUNDS: usize = 2;
const TOL_LOCUS: f64 = 1e-7;
const TOL_CROSS: f64 = 1e-9;
const TOL_ON: f64 = 1e-7;

type Seg2 = ([f64; 2], [f64; 2]);

/// Legacy stub kept for backwards compatibility with Stage 0 module exports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterBounds {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub special_values: Vec<f64>,
}

/// One forbidden sub-interval of `[lo, hi]` with human-readable failure reason(s).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForbiddenSegment {
    pub lo: f64,
    pub hi: f64,
    pub reason: String,
}

/// Connected crossing-free interval around the current parameter value (`tuning.allowed_interval`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AllowedInterval {
    pub interval: (f64, f64),
    pub edges: (Option<f64>, Option<f64>),
    pub segments: Vec<ForbiddenSegment>,
    pub current_ok: bool,
}

/// Parameter value where the lattice crosses a supergroup locus (`tuning.special_points`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecialPoint {
    pub value: f64,
    pub group: String,
    pub equations: Vec<String>,
    pub dim: usize,
}

fn reason_label(code: u8) -> &'static str {
    match code {
        1 => "two sites coincide",
        2 => "bonds cross / a bond passes through a site",
        3 => "other tiling (rotation system)",
        _ => "invalid",
    }
}

fn spacing(cell: &Cell2D) -> f64 {
    let a = cell_mat(cell);
    ((a[0][0] * a[1][1] - a[0][1] * a[1][0]).abs() / (cell.basis.len().max(1) as f64)).sqrt()
}

/// Smallest distance between distinct lattice sites, normalised by average site spacing.
pub fn min_separation(cell: &Cell2D) -> f64 {
    if cell.basis.is_empty() {
        return f64::INFINITY;
    }
    let a = cell_mat(cell);
    let mut best = f64::INFINITY;
    for (i, &pi) in cell.basis.iter().enumerate() {
        for (j, &pj) in cell.basis.iter().enumerate() {
            for s1 in -3..=3 {
                for s2 in -3..=3 {
                    if i != j || s1 != 0 || s2 != 0 {
                        let sh = mat2_vec(a, [s1 as f64, s2 as f64]);
                        best = best.min((pi[0] - pj[0] - sh[0]).hypot(pi[1] - pj[1] - sh[1]));
                    }
                }
            }
        }
    }
    best / spacing(cell)
}

/// Smallest distance from a lattice site to any bond not incident on it (average spacing = 1).
pub fn clearance(cell: &Cell2D) -> f64 {
    let a = cell_mat(cell);
    let mut best = f64::INFINITY;
    for bd in &cell.bonds {
        let p = cell.basis[bd.i];
        let sh_b = mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]);
        let v = [cell.basis[bd.j][0] + sh_b[0] - p[0], cell.basis[bd.j][1] + sh_b[1] - p[1]];
        let l2 = v[0] * v[0] + v[1] * v[1];
        for (k, &xk) in cell.basis.iter().enumerate() {
            for s1 in (bd.n1.min(0) - 2)..=(bd.n1.max(0) + 2) {
                for s2 in (bd.n2.min(0) - 2)..=(bd.n2.max(0) + 2) {
                    if (k == bd.i && s1 == 0 && s2 == 0) || (k == bd.j && s1 == bd.n1 && s2 == bd.n2) {
                        continue;
                    }
                    let sh = mat2_vec(a, [s1 as f64, s2 as f64]);
                    let pos = [xk[0] + sh[0], xk[1] + sh[1]];
                    let s = (((pos[0] - p[0]) * v[0] + (pos[1] - p[1]) * v[1]) / l2).clamp(0.0, 1.0);
                    best = best.min((p[0] + s * v[0] - pos[0]).hypot(p[1] + s * v[1] - pos[1]));
                }
            }
        }
    }
    best / spacing(cell)
}

fn bonds_cross_frac(p: [f64; 2], q: [f64; 2], r: [f64; 2], s: [f64; 2]) -> bool {
    let (ax, ay, bx, by) = (q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]);
    let den = ax * by - ay * bx;
    if den.abs() < TOL_CROSS {
        return false;
    }
    let lo0 = (p[0].min(q[0]) - r[0].max(s[0]) - 1e-12).ceil() as i32;
    let hi0 = (p[0].max(q[0]) - r[0].min(s[0]) + 1e-12).floor() as i32;
    let lo1 = (p[1].min(q[1]) - r[1].max(s[1]) - 1e-12).ceil() as i32;
    let hi1 = (p[1].max(q[1]) - r[1].min(s[1]) + 1e-12).floor() as i32;
    for n1 in lo0..=hi0 {
        for n2 in lo1..=hi1 {
            let (wx, wy) = (r[0] + n1 as f64 - p[0], r[1] + n2 as f64 - p[1]);
            let (u, v) = ((wx * by - wy * bx) / den, (wx * ay - wy * ax) / den);
            if u > TOL_CROSS && u < 1.0 - TOL_CROSS && v > TOL_CROSS && v < 1.0 - TOL_CROSS {
                return true;
            }
        }
    }
    false
}

fn site_on_bond_frac(f: &[[f64; 2]], p: [f64; 2], q: [f64; 2]) -> bool {
    let (vx, vy) = (q[0] - p[0], q[1] - p[1]);
    let l2 = vx * vx + vy * vy;
    for &fr in f {
        let lo0 = (p[0].min(q[0]) - fr[0] - 1e-9).ceil() as i32;
        let hi0 = (p[0].max(q[0]) - fr[0] + 1e-9).floor() as i32;
        let lo1 = (p[1].min(q[1]) - fr[1] - 1e-9).ceil() as i32;
        let hi1 = (p[1].max(q[1]) - fr[1] + 1e-9).floor() as i32;
        for n1 in lo0..=hi0 {
            for n2 in lo1..=hi1 {
                let (wx, wy) = (fr[0] + n1 as f64 - p[0], fr[1] + n2 as f64 - p[1]);
                let s = (wx * vx + wy * vy) / l2;
                if s > TOL_ON && s < 1.0 - TOL_ON && (wx * vy - wy * vx).abs() <= TOL_ON * l2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Whether any pair of bonds (or their periodic images) cross properly (`has_crossings`).
pub fn has_crossings(cell: &Cell2D) -> bool {
    let Some(ainv) = mat2_inv(cell_mat(cell)) else { return true };
    let f: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(ainv, p)).collect();
    let segs: Vec<Seg2> = cell.bonds.iter().map(|bd| (f[bd.i], [f[bd.j][0] + bd.n1 as f64, f[bd.j][1] + bd.n2 as f64])).collect();
    (0..segs.len()).any(|e| (e..segs.len()).any(|g| bonds_cross_frac(segs[e].0, segs[e].1, segs[g].0, segs[g].1)))
}

fn rotation_signature(f: &[[f64; 2]], m: usize, bonds: &[BareBond]) -> Vec<usize> {
    let e_len = bonds.len();
    let mut by_tail: Vec<Vec<(f64, usize)>> = vec![Vec::new(); m];
    for (idx, &(i, j, n1, n2)) in bonds.iter().enumerate() {
        let vf = [f[j][0] + n1 as f64 - f[i][0], f[j][1] + n2 as f64 - f[i][1]];
        by_tail[i].push((vf[1].atan2(vf[0]) + PI, idx));
    }
    for (idx, &(i, j, n1, n2)) in bonds.iter().enumerate() {
        let vr = [f[i][0] - n1 as f64 - f[j][0], f[i][1] - n2 as f64 - f[j][1]];
        by_tail[j].push((vr[1].atan2(vr[0]) + PI, e_len + idx));
    }
    let mut nxt = vec![0usize; 2 * e_len];
    for lst in &mut by_tail {
        lst.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for p in 0..lst.len() {
            nxt[lst[p].1] = lst[(p + 1) % lst.len()].1;
        }
    }
    nxt
}

pub(crate) fn classify_grid_family(fam: &EvalFamily, u_rows: &[Vec<f64>], ref_sig: Option<&[usize]>) -> (Vec<u8>, Vec<Vec<usize>>) {
    let (p_mat, q_vec) = fam.affine_sites();
    let (m, d, k_orbs, sorted_bonds) = (fam.m(), fam.site_names.len(), fam.orbits.len(), fam.bonds());
    let (mut codes, mut sigs, mut f) = (Vec::with_capacity(u_rows.len()), Vec::with_capacity(u_rows.len()), vec![[0.0; 2]; m]);

    for u in u_rows {
        for site in 0..m {
            let (mut x, mut y) = (q_vec[site][0], q_vec[site][1]);
            for col in 0..d {
                x += p_mat[site][0][col] * u[col];
                y += p_mat[site][1][col] * u[col];
            }
            f[site] = [x, y];
        }
        let sep_bad = (0..m).any(|i| (0..i).any(|j| {
            let (dx, dy) = (f[i][0] - f[j][0], f[i][1] - f[j][1]);
            (dx - dx.round()).abs().max((dy - dy.round()).abs()) <= 1e-5
        }));
        let mut code = if sep_bad { 1u8 } else { 0u8 };
        if code == 0 {
            let orb_segs: Vec<Vec<Seg2>> = fam.orbits.iter().map(|orb| {
                orb.iter().map(|&(bi, bj, n1, n2)| (f[bi], [f[bj][0] + n1 as f64, f[bj][1] + n2 as f64])).collect()
            }).collect();
            let on_bond = orb_segs.iter().any(|segs| site_on_bond_frac(&f, segs[0].0, segs[0].1));
            let crosses = !on_bond && (0..k_orbs).any(|a| (a..k_orbs).any(|b| {
                let (r_orb, o_orb) = if orb_segs[a].len() >= orb_segs[b].len() { (a, b) } else { (b, a) };
                let (pe, qe) = orb_segs[r_orb][0];
                orb_segs[o_orb].iter().any(|&(pf, qf)| bonds_cross_frac(pe, qe, pf, qf))
            }));
            if on_bond || crosses {
                code = 2;
            }
        }
        let sig = rotation_signature(&f, m, &sorted_bonds);
        if code == 0 && let Some(rs) = ref_sig && sig.as_slice() != rs {
            code = 3;
        }
        codes.push(code);
        sigs.push(sig);
    }
    (codes, sigs)
}

/// Classify rows of site-parameter vectors (`0` valid, `1` coincidence, `2` crossing/on-bond, `3` rotation change).
pub fn classify_grid(recipe: &Recipe, u_rows: &[Vec<f64>], ref_sig: Option<&[usize]>) -> Result<(Vec<u8>, Vec<Vec<usize>>), String> {
    Ok(classify_grid_family(&EvalFamily::from_recipe(recipe)?, u_rows, ref_sig))
}

fn build_rows(base_u: &[f64], col: usize, xs: &[f64]) -> Vec<Vec<f64>> {
    xs.iter().map(|&x| { let mut r = base_u.to_vec(); r[col] = x; r }).collect()
}

pub(crate) fn allowed_interval_family(
    fam: &EvalFamily,
    param_name: &str,
    lo_in: f64,
    hi_in: f64,
    values_override: Option<&BTreeMap<String, f64>>,
    n_scan: Option<usize>,
) -> Result<AllowedInterval, String> {
    let vals = fam.merged_values(values_override);
    let x0 = *vals.get(param_name).ok_or_else(|| format!("{param_name:?} is not a free parameter; free: {:?}", fam.names))?;
    let (lo, hi) = (lo_in.min(x0), hi_in.max(x0));
    let Some(col) = fam.site_names.iter().position(|nm| nm == param_name) else {
        // cell parameter: never forbidden (proved; tuning.py), current_ok = True whatever the sites do, as in Python
        return Ok(AllowedInterval { interval: (lo, hi), edges: (None, None), segments: Vec::new(), current_ok: true });
    };
    let u0 = fam.site_vector(&vals);
    let (code0, sig0) = classify_grid_family(fam, std::slice::from_ref(&u0), None);
    let current_ok = code0[0] == 0;

    let n = n_scan.unwrap_or(401).max(2);
    let step = (hi - lo) / ((n - 1) as f64);
    let mut xs: Vec<f64> = (0..n).map(|i| if i + 1 == n { hi } else { lo + (i as f64) * step }).collect();
    xs.push(x0);
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| *a == *b);
    let i0 = xs.binary_search_by(|v| v.total_cmp(&x0)).unwrap();

    let ref_sig = current_ok.then_some(sig0[0].as_slice());
    let (code, _) = classify_grid_family(fam, &build_rows(&u0, col, &xs), ref_sig);
    let valid: Vec<bool> = code.iter().map(|&c| c == 0).collect();

    let trans: Vec<usize> = (0..xs.len() - 1).filter(|&k| valid[k] != valid[k + 1]).collect();
    let mut brackets: Vec<(f64, f64)> = trans.iter().map(|&k| if valid[k] { (xs[k], xs[k + 1]) } else { (xs[k + 1], xs[k]) }).collect();

    if !brackets.is_empty() {
        let t_grid: Vec<f64> = (0..REFINE).map(|i| if i + 1 == REFINE { 1.0 } else { (i as f64) / ((REFINE - 1) as f64) }).collect();
        for _ in 0..ROUNDS {
            let sub_xs: Vec<f64> = brackets.iter().flat_map(|&(g, b)| t_grid.iter().map(move |&t| g + (b - g) * t)).collect();
            let (sub_code, _) = classify_grid_family(fam, &build_rows(&u0, col, &sub_xs), ref_sig);
            for (idx, br) in brackets.iter_mut().enumerate() {
                let c_slice = &sub_code[idx * REFINE..(idx + 1) * REFINE];
                let x_slice = &sub_xs[idx * REFINE..(idx + 1) * REFINE];
                let j = c_slice.iter().position(|&c| c != 0).unwrap_or(REFINE - 1).max(1);
                *br = (x_slice[j - 1], x_slice[j]);
            }
        }
    }

    let refined: HashMap<usize, (f64, f64)> = trans.iter().copied().zip(brackets).collect();
    let edge: HashMap<usize, f64> = refined.iter().map(|(&k, &(g, b))| (k, 0.5 * (g + b))).collect();

    let (mut segments, mut k) = (Vec::new(), 0usize);
    while k < xs.len() {
        if valid[k] {
            k += 1;
            continue;
        }
        let mut j = k;
        while j + 1 < xs.len() && !valid[j + 1] {
            j += 1;
        }
        let reasons: BTreeSet<u8> = code[k..=j].iter().copied().filter(|&c| c != 0).collect();
        let reason = reasons.into_iter().map(reason_label).collect::<Vec<_>>().join("; ");
        let seg_lo = if k > 0 { edge.get(&(k - 1)).copied().unwrap_or(lo) } else { lo };
        let seg_hi = if j + 1 < xs.len() { edge.get(&j).copied().unwrap_or(hi) } else { hi };
        segments.push(ForbiddenSegment { lo: seg_lo, hi: seg_hi, reason });
        k = j + 1;
    }

    if !current_ok {
        return Ok(AllowedInterval { interval: (lo, hi), edges: (None, None), segments, current_ok: false });
    }
    let mut kl = i0;
    while kl > 0 && valid[kl - 1] {
        kl -= 1;
    }
    let mut kr = i0;
    while kr + 1 < xs.len() && valid[kr + 1] {
        kr += 1;
    }
    let (mut a, ea) = if kl == 0 { (lo, None) } else { (refined[&(kl - 1)].0, Some(edge[&(kl - 1)])) };
    let (mut b, eb) = if kr + 1 == xs.len() { (hi, None) } else { (refined[&kr].0, Some(edge[&kr])) };
    let margin = 1e-4 * (hi - lo);
    if ea.is_some() {
        a = (a + margin).min(x0);
    }
    if eb.is_some() {
        b = (b - margin).max(x0);
    }
    Ok(AllowedInterval { interval: (a, b), edges: (ea, eb), segments, current_ok: true })
}

/// Compute the allowed crossing-free interval `[a, b]` around the current value of `param_name`.
pub fn allowed_interval(
    recipe: &Recipe,
    param_name: &str,
    lo: f64,
    hi: f64,
    values_override: Option<&BTreeMap<String, f64>>,
    n_scan: Option<usize>,
) -> Result<AllowedInterval, String> {
    allowed_interval_family(&EvalFamily::from_recipe(recipe)?, param_name, lo, hi, values_override, n_scan)
}

fn v_of_t(fam: &EvalFamily, vals: &BTreeMap<String, f64>, name: &str, t: f64, base: &[f64]) -> Vec<f64> {
    let mut v = base.to_vec();
    let ny = fam.metric_vars().len();
    if let Some(pos) = fam.site_names.iter().position(|nm| nm == name) {
        v[ny + pos] = t;
    } else if name == "gamma" {
        v[0] = (vals["b"] / fam.a) * t;
    } else {
        if fam.cellpar.contains_key("gamma") {
            v[0] = t * vals["gamma"].to_radians().cos();
        }
        v[ny - 1] = t * t;
    }
    v
}

fn t_to_x(fam: &EvalFamily, name: &str, t: f64) -> Option<f64> {
    if fam.site_names.iter().any(|nm| nm == name) {
        Some(t)
    } else if name == "gamma" {
        (-1.0 < t && t < 1.0).then(|| t.acos().to_degrees())
    } else {
        (t > 0.0).then_some(t * fam.a)
    }
}

fn poly_real_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() <= 1e-12 {
        return if b.abs() > 1e-12 { vec![-c / b] } else { Vec::new() };
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let s = disc.sqrt();
        vec![(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
    } else if (-disc).sqrt() / (2.0 * a).abs() <= 1e-6 {
        vec![-b / (2.0 * a)]
    } else {
        Vec::new()
    }
}

pub(crate) fn special_points_with_loci(
    fam: &EvalFamily,
    loci: &[Locus],
    param_name: &str,
    lo: f64,
    hi: f64,
    values_override: Option<&BTreeMap<String, f64>>,
) -> Result<Vec<SpecialPoint>, String> {
    if !fam.names.iter().any(|nm| nm == param_name) {
        return Err(format!("{param_name:?} is not a free parameter; free: {:?}", fam.names));
    }
    let vals = fam.merged_values(values_override);
    let base = fam.v_vec(&vals);
    let n_vars = base.len();
    let mut found: Vec<SpecialPoint> = Vec::new();

    for loc in loci {
        let Some(ref grp) = loc.group else { continue };
        let eval_res = |t: f64| -> Vec<f64> {
            let vt = v_of_t(fam, &vals, param_name, t, &base);
            loc.aug.iter().map(|row| row[..n_vars].iter().zip(&vt).map(|(&ci, &vi)| ci * vi).sum::<f64>() - row[n_vars]).collect()
        };
        let (rm, r0, rp) = (eval_res(-1.0), eval_res(0.0), eval_res(1.0));
        let coeffs: Vec<(f64, f64, bool)> = (0..loc.codim).map(|r| {
            let (a, b) = (0.5 * (rp[r] + rm[r]) - r0[r], 0.5 * (rp[r] - rm[r]));
            (a, b, a.abs() > 1e-12 || b.abs() > 1e-12)
        }).collect();
        if !coeffs.iter().any(|c| c.2) || (0..loc.codim).any(|r| !coeffs[r].2 && r0[r].abs() >= TOL_LOCUS) {
            continue;
        }
        let k = (0..loc.codim).max_by(|&i, &j| (coeffs[i].0.abs() + coeffs[i].1.abs()).total_cmp(&(coeffs[j].0.abs() + coeffs[j].1.abs()))).unwrap();
        for t in poly_real_roots(coeffs[k].0, coeffs[k].1, r0[k]) {
            if eval_res(t).into_iter().map(f64::abs).fold(0.0, f64::max) < TOL_LOCUS
                && let Some(x) = t_to_x(fam, param_name, t)
                && lo - 1e-12 <= x
                && x <= hi + 1e-12
            {
                found.push(SpecialPoint { value: x, group: grp.clone(), equations: loc.equations.clone(), dim: loc.dim });
            }
        }
    }
    found.sort_by(|a, b| a.value.total_cmp(&b.value).then_with(|| a.dim.cmp(&b.dim)));
    let mut out: Vec<SpecialPoint> = Vec::new();
    for sp in found {
        if let Some(last) = out.last_mut() && (sp.value - last.value).abs() < 1e-6 {
            if sp.dim < last.dim {
                *last = sp;
            }
        } else {
            out.push(sp);
        }
    }
    Ok(out)
}

/// Solve exact 1D intersections of `param_name` in `[lo, hi]` with all supergroup loci of `recipe`.
pub fn special_points(
    recipe: &Recipe,
    param_name: &str,
    lo: f64,
    hi: f64,
    values_override: Option<&BTreeMap<String, f64>>,
) -> Result<Vec<SpecialPoint>, String> {
    let fam = EvalFamily::from_recipe(recipe)?;
    let loci = supergroup_loci_for_family(&fam).unwrap_or_default();
    special_points_with_loci(&fam, &loci, param_name, lo, hi, values_override)
}
