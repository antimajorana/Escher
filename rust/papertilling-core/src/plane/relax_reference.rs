//! Test-only copy of the original Gauss-Newton solvers (triplets + full `sp_cholesky` per factorisation),
//! kept as the exact reference for the cached-pattern solvers in `relax.rs`.

use faer::Mat;
use faer::Side;
use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};

use crate::plane::relax::RelaxSetup;

type NormalEqs = (Vec<Triplet<usize, usize, f64>>, Vec<f64>, Vec<f64>, f64);

fn assemble(setup: &RelaxSetup, z: &[f64], free_map: Option<&[Option<usize>]>, ndof: usize, build_jtj: bool) -> NormalEqs {
    let cap = if build_jtj { setup.springs.len() * 16 + setup.tris.len() * 36 + ndof } else { 0 };
    let mut triplets = Vec::with_capacity(cap);
    let mut diag = if build_jtj { vec![0.0; ndof] } else { Vec::new() };
    let mut g = vec![0.0; ndof];
    let mut sum_sq = 0.0;

    let mut add_row = |cols: &[usize], vals: &[f64], r: f64| {
        sum_sq += r * r;
        for p in 0..cols.len() {
            let Some(cp) = (match free_map {
                Some(m) => m[cols[p]],
                None => Some(cols[p]),
            }) else {
                continue;
            };
            let vp = vals[p];
            g[cp] += vp * r;
            if !build_jtj {
                continue;
            }
            diag[cp] += vp * vp;
            triplets.push(Triplet::new(cp, cp, vp * vp));
            for q in (p + 1)..cols.len() {
                let Some(cq) = (match free_map {
                    Some(m) => m[cols[q]],
                    None => Some(cols[q]),
                }) else {
                    continue;
                };
                let vpq = vp * vals[q];
                if cp == cq {
                    diag[cp] += 2.0 * vpq;
                    triplets.push(Triplet::new(cp, cp, 2.0 * vpq));
                } else {
                    triplets.push(Triplet::new(cp, cq, vpq));
                    triplets.push(Triplet::new(cq, cp, vpq));
                }
            }
        }
    };

    for &(u, v, rest, wt) in &setup.springs {
        let (dx, dy) = (z[2 * u] - z[2 * v], z[2 * u + 1] - z[2 * v + 1]);
        let ln_raw = dx.hypot(dy);
        let inv_ln = wt / ln_raw.max(1e-12);
        let (gx, gy) = (inv_ln * dx, inv_ln * dy);
        add_row(&[2 * u, 2 * u + 1, 2 * v, 2 * v + 1], &[gx, gy, -gx, -gy], wt * (ln_raw - rest));
    }
    for &(u, v, w, ta, tsc) in &setup.tris {
        let (e1x, e1y) = (z[2 * v] - z[2 * u], z[2 * v + 1] - z[2 * u + 1]);
        let (e2x, e2y) = (z[2 * w] - z[2 * u], z[2 * w + 1] - z[2 * u + 1]);
        let (dvx, dvy, dwx, dwy) = (tsc * e2y, -tsc * e2x, -tsc * e1y, tsc * e1x);
        add_row(
            &[2 * v, 2 * v + 1, 2 * w, 2 * w + 1, 2 * u, 2 * u + 1],
            &[dvx, dvy, dwx, dwy, -dvx - dwx, -dvy - dwy],
            tsc * (e1x * e2y - e1y * e2x - ta),
        );
    }
    let a2 = setup.anchor * setup.anchor;
    for c in 0..(2 * setup.n) {
        let rc = setup.anchor * (z[c] - setup.x0[c]);
        sum_sq += rc * rc;
        if let Some(k) = free_map.map_or(Some(c), |m| m[c]) {
            g[k] += a2 * (z[c] - setup.x0[c]);
            if build_jtj {
                diag[k] += a2;
                triplets.push(Triplet::new(k, k, a2));
            }
        }
    }
    (triplets, diag, g, 0.5 * sum_sq)
}

fn build_spd_cholesky(
    ndof: usize,
    base_triplets: &[Triplet<usize, usize, f64>],
    diag: &[f64],
    mu: f64,
) -> Option<faer::sparse::linalg::solvers::Llt<usize, f64>> {
    let mat = if mu == 0.0 {
        SparseColMat::<usize, f64>::try_new_from_triplets(ndof, ndof, base_triplets).ok()?
    } else {
        let mut aug = Vec::with_capacity(base_triplets.len() + ndof);
        aug.extend_from_slice(base_triplets);
        for (k, &dk) in diag.iter().enumerate() {
            aug.push(Triplet::new(k, k, mu * dk.max(1e-12)));
        }
        SparseColMat::<usize, f64>::try_new_from_triplets(ndof, ndof, &aug).ok()?
    };
    mat.sp_cholesky(Side::Lower).ok()
}

pub(crate) fn factor_for_test(setup: &RelaxSetup, z: &[f64], mu: f64) -> Option<faer::sparse::linalg::solvers::Llt<usize, f64>> {
    let ndof = 2 * setup.n;
    let (t, d, _, _) = assemble(setup, z, None, ndof, true);
    build_spd_cholesky(ndof, &t, &d, mu)
}

pub(crate) fn gauss_newton(
    setup: &RelaxSetup,
    z0: &[f64],
    max_iter: usize,
    rtol: f64,
    freeze: f64,
) -> (Vec<f64>, f64, usize, usize) {
    let ndof = 2 * setup.n;
    if ndof == 0 || setup.springs.is_empty() {
        return (z0.to_vec(), setup.cost(z0), 0, 0);
    }
    let mut z = z0.to_vec();
    let mut f = setup.cost(&z);
    let mut mu = 0.0_f64;
    let mut llt: Option<faer::sparse::linalg::solvers::Llt<usize, f64>> = None;
    let (mut frozen, mut nfact, mut last_it) = (false, 0usize, 0usize);

    for it in 1..=max_iter {
        last_it = it;
        let (mut base_trips, mut diag, g, _) = assemble(setup, &z, None, ndof, llt.is_none() || !frozen);
        let rhs = Mat::from_fn(ndof, 1, |row, _| -g[row]);
        let (z2, f2) = loop {
            if llt.is_none() || !frozen {
                let Some(fac) = build_spd_cholesky(ndof, &base_trips, &diag, mu) else {
                    mu = if mu == 0.0 { 1e-4 } else { 10.0 * mu };
                    if mu > 1e12 {
                        return (z, f, it, nfact);
                    }
                    continue;
                };
                llt = Some(fac);
                nfact += 1;
            }
            let dz = llt.as_ref().expect("cholesky factor ready").solve(&rhs);
            let cand: Vec<f64> = (0..ndof).map(|k| z[k] + dz[(k, 0)]).collect();
            let f_cand = setup.cost(&cand);
            if f_cand <= f {
                break (cand, f_cand);
            }
            if frozen {
                frozen = false;
                let (t, d, _, _) = assemble(setup, &z, None, ndof, true);
                base_trips = t;
                diag = d;
                continue;
            }
            mu = if mu == 0.0 { 1e-4 } else { 10.0 * mu };
            if mu > 1e12 {
                return (z, f, it, nfact);
            }
        };
        let drop = f - f2;
        z = z2;
        f = f2;
        mu = if mu < 1e-10 { 0.0 } else { 0.1 * mu };
        if drop <= rtol * f {
            break;
        }
        if drop <= freeze * f && mu == 0.0 {
            frozen = true;
        }
    }
    (z, f, last_it, nfact)
}

fn gn_step(setup: &RelaxSetup, z: &[f64], free: Option<&[usize]>) -> Option<(Vec<f64>, f64)> {
    let total = 2 * setup.n;
    let (free_map, ndof) = match free {
        Some(f) => {
            let mut m = vec![None; total];
            for (k, &c) in f.iter().enumerate() {
                m[c] = Some(k);
            }
            (Some(m), f.len())
        }
        None => (None, total),
    };
    if ndof == 0 {
        return Some((Vec::new(), setup.cost(z)));
    }
    let (trips, diag, g, f) = assemble(setup, z, free_map.as_deref(), ndof, true);
    let rhs = Mat::from_fn(ndof, 1, |row, _| -g[row]);
    for mu in [0.0, 1e-8, 1e-5, 1e-2] {
        if let Some(llt) = build_spd_cholesky(ndof, &trips, &diag, mu) {
            let sol = llt.solve(&rhs);
            return Some(((0..ndof).map(|k| sol[(k, 0)]).collect(), f));
        }
    }
    None
}

pub(crate) fn relax_pinned(setup: &RelaxSetup, z0: &[f64], free: &[usize], max_iter: usize, rtol: f64) -> (Vec<f64>, usize) {
    if free.is_empty() || setup.springs.is_empty() {
        return (z0.to_vec(), 0);
    }
    let mut z = z0.to_vec();
    let mut last_it = 0usize;
    for it in 1..=max_iter {
        last_it = it;
        let Some((dz, f)) = gn_step(setup, &z, Some(free)) else { break };
        let mut a = 1.0_f64;
        let mut accepted = None;
        while a > 1e-3 {
            let mut z2 = z.clone();
            for (k, &c) in free.iter().enumerate() {
                z2[c] += a * dz[k];
            }
            let f2 = setup.cost(&z2);
            if f2 < f {
                accepted = Some((z2, f2));
                break;
            }
            a *= 0.5;
        }
        let Some((z2, f2)) = accepted else { return (z, it) };
        z = z2;
        if f - f2 <= rtol * f2 {
            break;
        }
    }
    (z, last_it)
}

pub(crate) fn relax_guarded<F: FnMut(&[f64]) -> usize>(
    setup: &RelaxSetup,
    z0: &[f64],
    mut badness: F,
    max_iter: usize,
    rtol: f64,
    halvings: usize,
    max_checks: usize,
) -> (Vec<f64>, usize) {
    if setup.springs.is_empty() {
        return (z0.to_vec(), 0);
    }
    let mut z = z0.to_vec();
    let mut b = badness(&z);
    let (mut steps, mut checks) = (0usize, 0usize);
    for _ in 0..max_iter {
        let Some((dz, f)) = gn_step(setup, &z, None) else { break };
        let mut a = 1.0_f64;
        let mut accepted = None;
        for _ in 0..=halvings {
            let z2: Vec<f64> = z.iter().zip(&dz).map(|(&zi, &di)| zi + a * di).collect();
            let f2 = setup.cost(&z2);
            if f2 < f && checks < max_checks {
                checks += 1;
                let b2 = badness(&z2);
                if b2 <= b {
                    accepted = Some((z2, f2, b2));
                    break;
                }
            }
            a *= 0.5;
        }
        let Some((z2, f2, b2)) = accepted else { break };
        (z, b) = (z2, b2);
        steps += 1;
        if f - f2 <= rtol * f2 {
            break;
        }
    }
    (z, steps)
}
