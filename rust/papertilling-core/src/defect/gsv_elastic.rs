//! Layer 2: Exact G-gauge-invariant GSV metric elasticity (2D damped Gauss-Newton + Newton
//! polish, exact analytic Hessian, and canonical E(2) alignment).

use faer::prelude::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer::{Mat, Side};

pub use crate::defect::gsv_model::{Elastic2DOptions, Elastic2DResult};
pub(crate) use crate::defect::gsv_model::{
    GsvModel, compute_gsv_diagnostics, eval_gsv_fg,
};
use crate::types::BuiltCrystal;

fn assemble_2d_system(
    model: &GsvModel,
    x: &[f64],
    include_prestress: bool,
    grad: &mut [f64],
    diag: &mut [f64],
    mut vals: Option<&mut Vec<f64>>,
    mut idx: Option<&mut Vec<Pair<usize, usize>>>,
) -> f64 {
    grad.fill(0.0);
    diag.fill(0.0);
    if let Some(v) = vals.as_deref_mut() {
        v.clear();
    }
    let mut push = |r: usize, c: usize, val: f64, diag: &mut [f64]| {
        if r == c {
            diag[r] += val;
        }
        if let Some(v) = vals.as_deref_mut() {
            v.push(val);
        }
        if let Some(ix) = idx.as_deref_mut() {
            ix.push(Pair::new(r, c));
        }
    };

    let mut total_e = 0.0;
    for s in &model.stretches {
        let (dx, dy) = (x[2 * s.v] - x[2 * s.u], x[2 * s.v + 1] - x[2 * s.u + 1]);
        let r = (dx * dx + dy * dy - s.l0 * s.l0) / (2.0 * s.l0);
        total_e += 0.5 * r * r;
        let inv = 1.0 / s.l0;
        let cols = [2 * s.u, 2 * s.u + 1, 2 * s.v, 2 * s.v + 1];
        let j = [-dx * inv, -dy * inv, dx * inv, dy * inv];
        for p in 0..4 {
            grad[cols[p]] += j[p] * r;
            push(cols[p], cols[p], j[p] * j[p], diag);
            for q in (p + 1)..4 {
                let vpq = j[p] * j[q];
                push(cols[p], cols[q], vpq, diag);
                push(cols[q], cols[p], vpq, diag);
            }
        }
        let c = if include_prestress { r * inv } else { 0.0 };
        for ax in 0..2 {
            push(2 * s.u + ax, 2 * s.u + ax, c, diag);
            push(2 * s.v + ax, 2 * s.v + ax, c, diag);
            push(2 * s.u + ax, 2 * s.v + ax, -c, diag);
            push(2 * s.v + ax, 2 * s.u + ax, -c, diag);
        }
    }

    for sh in &model.shears_2d {
        let (d1x, d1y) = (x[2 * sh.v] - x[2 * sh.u], x[2 * sh.v + 1] - x[2 * sh.u + 1]);
        let (d2x, d2y) = (x[2 * sh.w] - x[2 * sh.u], x[2 * sh.w + 1] - x[2 * sh.u + 1]);
        let r = sh.alpha * (d1x * d2x + d1y * d2y - sh.c_bar);
        total_e += 0.5 * r * r;
        let a = sh.alpha;
        let cols = [2 * sh.u, 2 * sh.u + 1, 2 * sh.v, 2 * sh.v + 1, 2 * sh.w, 2 * sh.w + 1];
        let j = [-a * (d1x + d2x), -a * (d1y + d2y), a * d2x, a * d2y, a * d1x, a * d1y];
        for p in 0..6 {
            grad[cols[p]] += j[p] * r;
            push(cols[p], cols[p], j[p] * j[p], diag);
            for q in (p + 1)..6 {
                let vpq = j[p] * j[q];
                push(cols[p], cols[q], vpq, diag);
                push(cols[q], cols[p], vpq, diag);
            }
        }
        let c = if include_prestress { a * r } else { 0.0 };
        for ax in 0..2 {
            push(2 * sh.u + ax, 2 * sh.u + ax, 2.0 * c, diag);
            push(2 * sh.u + ax, 2 * sh.v + ax, -c, diag);
            push(2 * sh.v + ax, 2 * sh.u + ax, -c, diag);
            push(2 * sh.u + ax, 2 * sh.w + ax, -c, diag);
            push(2 * sh.w + ax, 2 * sh.u + ax, -c, diag);
            push(2 * sh.v + ax, 2 * sh.w + ax, c, diag);
            push(2 * sh.w + ax, 2 * sh.v + ax, c, diag);
        }
    }
    total_e
}

struct Cached2DFactor {
    ndof: usize,
    sym: SymbolicSparseColMat<usize>,
    order: Argsort<usize>,
    llt: SymbolicLlt<usize>,
    n_base: usize,
}

impl Cached2DFactor {
    fn new(model: &GsvModel, x: &[f64]) -> Option<Self> {
        let ndof = 2 * model.n;
        let mut grad = vec![0.0; ndof];
        let mut diag = vec![0.0; ndof];
        let mut idx = Vec::new();
        let _ = assemble_2d_system(model, x, false, &mut grad, &mut diag, None, Some(&mut idx));
        let n_base = idx.len();
        idx.extend((0..ndof).map(|k| Pair::new(k, k)));
        let (sym, order) = SymbolicSparseColMat::try_new_from_indices(ndof, ndof, &idx).ok()?;
        let dummy = vec![1.0; idx.len()];
        let mat = SparseColMat::new_from_argsort(sym.clone(), &order, &dummy).ok()?;
        let llt = SymbolicLlt::try_new(mat.symbolic(), Side::Lower).ok()?;
        Some(Self { ndof, sym, order, llt, n_base })
    }

    fn solve(&self, vals: &[f64], diag: &[f64], lambda: f64, grad: &[f64]) -> Option<Vec<f64>> {
        let mut full = Vec::with_capacity(self.n_base + self.ndof);
        full.extend_from_slice(vals);
        full.extend(diag.iter().map(|&d| lambda * d.abs().max(1.0)));
        let mat = SparseColMat::new_from_argsort(self.sym.clone(), &self.order, &full).ok()?;
        let fac = Llt::try_new_with_symbolic(self.llt.clone(), mat.as_ref(), Side::Lower).ok()?;
        let rhs = Mat::from_fn(self.ndof, 1, |r, _| -grad[r]);
        let sol = fac.solve(&rhs);
        Some((0..self.ndof).map(|k| sol[(k, 0)]).collect())
    }
}

/// Deterministic canonical centroid + multipole `SO(2)` alignment (vertex-order independent).
pub(crate) fn canonical_align_2d(pos: &mut [[f64; 2]]) {
    let n = pos.len();
    if n == 0 {
        return;
    }
    let inv = 1.0 / (n as f64);
    let cx = pos.iter().map(|p| p[0]).sum::<f64>() * inv;
    let cy = pos.iter().map(|p| p[1]).sum::<f64>() * inv;
    for p in pos.iter_mut() {
        p[0] -= cx;
        p[1] -= cy;
    }
    if n < 2 {
        return;
    }
    let scale2 = pos.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum::<f64>().max(1e-24);
    let (mut m2r, mut m2i) = (0.0, 0.0);
    for p in pos.iter() {
        m2r += p[0] * p[0] - p[1] * p[1];
        m2i += 2.0 * p[0] * p[1];
    }
    let mut ang = if m2r.hypot(m2i) > 1e-8 * scale2 {
        -0.5 * m2i.atan2(m2r)
    } else {
        let mut best_ang = 0.0;
        for k in [3_i32, 4, 6] {
            let (mut kr, mut ki) = (0.0, 0.0);
            for p in pos.iter() {
                let r = p[0].hypot(p[1]);
                let th = p[1].atan2(p[0]) * (k as f64);
                let rk = r.powi(k);
                kr += rk * th.cos();
                ki += rk * th.sin();
            }
            if kr.hypot(ki) > 1e-8 * scale2.powf(0.5 * k as f64) {
                best_ang = -ki.atan2(kr) / (k as f64);
                break;
            }
        }
        best_ang
    };
    let (s, c) = ang.sin_cos();
    let skew_x: f64 = pos
        .iter()
        .map(|p| {
            let x = c * p[0] - s * p[1];
            let r2 = p[0] * p[0] + p[1] * p[1];
            x * r2
        })
        .sum();
    if skew_x < -1e-9 * scale2.powf(1.5) {
        ang += std::f64::consts::PI;
    }
    let (s, c) = ang.sin_cos();
    for p in pos.iter_mut() {
        let (x, y) = (p[0], p[1]);
        p[0] = c * x - s * y;
        p[1] = s * x + c * y;
    }
}

/// Evaluate exact 2D GSV energy and gradient at `pos`.
pub fn gsv_eval_2d(crystal: &BuiltCrystal, pos: &[[f64; 2]]) -> Result<(f64, Vec<[f64; 2]>), String> {
    let model = GsvModel::from_crystal(crystal, None)?;
    let x: Vec<f64> = pos.iter().flat_map(|p| *p).collect();
    let mut g = vec![0.0; 2 * model.n];
    let (es, eh) = eval_gsv_fg::<2>(&x, &mut g, &model.stretches, &model.shears_2d);
    let grad_pts = g.as_chunks::<2>().0.to_vec();
    Ok((es + eh, grad_pts))
}

/// Evaluate $O(|E|)$ per-bond and per-site GSV diagnostics at given 2D coordinates without running the solver.
pub fn gsv_diagnostics_2d(
    crystal: &BuiltCrystal,
    pos: &[[f64; 2]],
) -> Result<Elastic2DResult, String> {
    let model = GsvModel::from_crystal(crystal, None)?;
    if pos.len() != model.n {
        return Err(format!("pos len {} != crystal len {}", pos.len(), model.n));
    }
    let x: Vec<f64> = pos.iter().flat_map(|p| *p).collect();
    let diag = compute_gsv_diagnostics::<2>(&x, &model, &model.shears_2d);
    Ok(Elastic2DResult {
        pos: pos.to_vec(),
        total_energy: diag.stretch_energy + diag.shear_energy,
        stretch_energy: diag.stretch_energy,
        shear_energy: diag.shear_energy,
        bond_strain: diag.bond_strain,
        bond_shear_strain: diag.bond_shear_strain,
        bond_gsv_energy: diag.bond_gsv_energy,
        site_gsv_energy: diag.site_gsv_energy,
        site_shear: diag.site_shear,
        converged: true,
        iterations: 0,
    })
}

/// Compute the exact `2V x 2V` E(2)-invariant GSV Hessian at `pos`.
pub fn gsv_hessian_2d(crystal: &BuiltCrystal, pos: &[[f64; 2]]) -> Result<Mat<f64>, String> {
    let model = GsvModel::from_crystal(crystal, None)?;
    let ndof = 2 * model.n;
    let x: Vec<f64> = pos.iter().flat_map(|p| *p).collect();
    let mut grad = vec![0.0; ndof];
    let mut diag = vec![0.0; ndof];
    let mut vals = Vec::new();
    let mut idx = Vec::new();
    let _ = assemble_2d_system(&model, &x, true, &mut grad, &mut diag, Some(&mut vals), Some(&mut idx));
    let mut hess = Mat::<f64>::zeros(ndof, ndof);
    for (p, &v) in idx.iter().zip(&vals) {
        hess[(p.row, p.col)] += v;
    }
    Ok(hess)
}

/// Relax `crystal` in 2D (`D=2`) under exact GSV metric elasticity with NO `anchor * (y - X0)` term.
pub fn relax_elastic_2d(crystal: &BuiltCrystal) -> Result<Elastic2DResult, String> {
    relax_elastic_2d_with_opts(crystal, &Elastic2DOptions::default())
}

pub fn relax_elastic_2d_with_opts(
    crystal: &BuiltCrystal,
    opts: &Elastic2DOptions,
) -> Result<Elastic2DResult, String> {
    let model = GsvModel::from_crystal(crystal, None)?;
    relax_elastic_2d_with_model(&model, opts)
}

pub(crate) fn relax_elastic_2d_with_model(
    model: &GsvModel,
    opts: &Elastic2DOptions,
) -> Result<Elastic2DResult, String> {
    if model.n == 0 {
        return Ok(Elastic2DResult {
            pos: Vec::new(),
            total_energy: 0.0,
            stretch_energy: 0.0,
            shear_energy: 0.0,
            bond_strain: Vec::new(),
            bond_shear_strain: Vec::new(),
            bond_gsv_energy: Vec::new(),
            site_gsv_energy: Vec::new(),
            site_shear: Vec::new(),
            converged: true,
            iterations: 0,
        });
    }
    let ndof = 2 * model.n;
    let mut x: Vec<f64> = model.chart_2d.iter().flat_map(|p| *p).collect();
    let factor = Cached2DFactor::new(model, &x).ok_or("failed to build sparse factor")?;
    let mut grad = vec![0.0; ndof];
    let mut diag = vec![0.0; ndof];
    let mut vals = Vec::with_capacity(factor.n_base);
    let mut x_try = vec![0.0; ndof];
    let mut g_try = vec![0.0; ndof];

    let mut iters = 0usize;
    let mut lambda = 1e-8_f64;
    let mut f = assemble_2d_system(model, &x, false, &mut grad, &mut diag, Some(&mut vals), None);
    let max_abs = |v: &[f64]| v.iter().fold(0.0_f64, |m, &a| m.max(a.abs()));

    for _ in 0..opts.max_gn_iters {
        if max_abs(&grad) <= 1e-6 {
            break;
        }
        iters += 1;
        let mut accepted = false;
        for _ in 0..16 {
            if let Some(dx) = factor.solve(&vals, &diag, lambda, &grad) {
                let dg: f64 = grad.iter().zip(&dx).map(|(g, d)| g * d).sum();
                if dg < 0.0 && dg.is_finite() {
                    let mut step = 1.0_f64;
                    for _ in 0..16 {
                        for i in 0..ndof {
                            x_try[i] = x[i] + step * dx[i];
                        }
                        let (es, eh) = eval_gsv_fg::<2>(&x_try, &mut g_try, &model.stretches, &model.shears_2d);
                        let ft = es + eh;
                        if ft.is_finite() && ft <= f + 1e-4 * step * dg {
                            x.copy_from_slice(&x_try);
                            f = assemble_2d_system(model, &x, false, &mut grad, &mut diag, Some(&mut vals), None);
                            lambda = (lambda * 0.25).max(1e-13);
                            accepted = true;
                            break;
                        }
                        step *= 0.5;
                    }
                    if accepted {
                        break;
                    }
                }
            }
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    let mut lambda_n = 1e-13_f64;
    for _ in 0..opts.max_newton_iters {
        if max_abs(&grad) <= opts.grad_tol {
            break;
        }
        iters += 1;
        let _ = assemble_2d_system(model, &x, true, &mut grad, &mut diag, Some(&mut vals), None);
        let mut dx_opt = factor.solve(&vals, &diag, lambda_n, &grad).filter(|dx| {
            let dg: f64 = grad.iter().zip(dx).map(|(g, d)| g * d).sum();
            dg < 0.0 && dg.is_finite()
        });
        if dx_opt.is_none() {
            let _ = assemble_2d_system(model, &x, false, &mut grad, &mut diag, Some(&mut vals), None);
            dx_opt = factor.solve(&vals, &diag, 1e-8, &grad);
        }
        let Some(dx) = dx_opt else { break };
        let dg: f64 = grad.iter().zip(&dx).map(|(g, d)| g * d).sum();
        if dg >= 0.0 || !dg.is_finite() {
            break;
        }
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..22 {
            for i in 0..ndof {
                x_try[i] = x[i] + step * dx[i];
            }
            let (es, eh) = eval_gsv_fg::<2>(&x_try, &mut g_try, &model.stretches, &model.shears_2d);
            let ft = es + eh;
            let near_floor = (-dg) <= 1e-13 * (1.0 + f.abs())
                && ft <= f + 1e-14 * (1.0 + f.abs())
                && max_abs(&g_try) < max_abs(&grad);
            if ft.is_finite() && (ft <= f + 1e-4 * step * dg || near_floor) {
                x.copy_from_slice(&x_try);
                f = ft;
                grad.copy_from_slice(&g_try);
                lambda_n = (lambda_n * 0.5).max(1e-14);
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            lambda_n *= 10.0;
            if lambda_n > 1e-3 {
                break;
            }
        }
    }

    let converged = max_abs(&grad) <= opts.grad_tol;
    let mut pos: Vec<[f64; 2]> = x.as_chunks::<2>().0.to_vec();
    canonical_align_2d(&mut pos);
    let x_aligned: Vec<f64> = pos.iter().flat_map(|p| *p).collect();
    let diag = compute_gsv_diagnostics::<2>(&x_aligned, model, &model.shears_2d);
    Ok(Elastic2DResult {
        pos,
        total_energy: diag.stretch_energy + diag.shear_energy,
        stretch_energy: diag.stretch_energy,
        shear_energy: diag.shear_energy,
        bond_strain: diag.bond_strain,
        bond_shear_strain: diag.bond_shear_strain,
        bond_gsv_energy: diag.bond_gsv_energy,
        site_gsv_energy: diag.site_gsv_energy,
        site_shear: diag.site_shear,
        converged,
        iterations: iters,
    })
}
