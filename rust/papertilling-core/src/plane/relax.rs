//! Fast Gauss-Newton spring + corner-area relaxation using `faer` sparse Cholesky.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use faer::Mat;
use faer::Side;
use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};

use crate::defect::core_graph::{TWO_PI, corners_order, cross2, norm2, sub2};
use crate::defect::surgery::mark_long_bonds;
use crate::plane::planarity::{has_frames, median_slice};
use crate::types::BuiltCrystal;

/// Precomputed spring, corner-triangle, and weak-anchor data matching `fast_relax.py`.
#[derive(Clone, Debug)]
pub struct RelaxSetup {
    pub n: usize,
    pub x0: Vec<f64>,
    pub anchor: f64,
    pub springs: Vec<(usize, usize, f64, f64)>,
    pub tris: Vec<(usize, usize, usize, f64, f64)>,
}

/// Build a working copy of `crystal` whose bonds all carry valid rest-frame vectors `vec`
/// (matching `draw._work_copy`: frameless/wall `across` bonds get `max(drawn, type_median)`).
pub(crate) fn work_copy_crystal(crystal: &BuiltCrystal, x0: &[[f64; 2]]) -> BuiltCrystal {
    let mut w = crystal.clone();
    w.pos = x0.to_vec();
    if has_frames(crystal) {
        return w;
    }
    let mut by_kind: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (u, edges) in w.adj.iter().enumerate() {
        for e in edges {
            if !e.across {
                by_kind.entry(e.kind).or_default().push(norm2(sub2(x0[e.to], x0[u])));
            }
        }
    }
    let mut all_lens: Vec<f64> = by_kind.values().flatten().copied().collect();
    let all_med = if all_lens.is_empty() { 1.0 } else { median_slice(&mut all_lens) };
    let kind_med: HashMap<usize, f64> = by_kind
        .into_iter()
        .map(|(k, mut v)| (k, if v.is_empty() { all_med } else { median_slice(&mut v) }))
        .collect();
    for u in 0..w.adj.len() {
        for e in &mut w.adj[u] {
            let mut vec = sub2(x0[e.to], x0[u]);
            let ell = norm2(vec);
            if e.across || ell < 1e-9 {
                let want = kind_med.get(&e.kind).copied().unwrap_or(all_med);
                let want = if e.across { want.max(ell) } else { want };
                let dir = if ell > 1e-9 { [vec[0] / ell, vec[1] / ell] } else { [1.0, 0.0] };
                vec = [dir[0] * want, dir[1] * want];
            }
            e.vec = vec;
        }
    }
    w
}

impl RelaxSetup {
    pub fn from_crystal(crystal: &BuiltCrystal, anchor_pos: &[[f64; 2]]) -> Self {
        let n = crystal.pos.len();
        let non_orient = crystal.is_non_orientable;
        let is_wrap = |u: usize, v: usize| -> bool {
            if !non_orient {
                return false;
            }
            let Some(e) = crystal.adj[u].iter().find(|b| b.to == v) else { return false };
            e.seam.is_some_and(|g| g.det() < 0.0)
                || norm2(sub2(anchor_pos[v], anchor_pos[u])) > 3.2 * norm2(e.vec).max(1e-6)
        };
        let mut spring_idx: HashMap<(usize, usize), usize> = HashMap::new();
        let mut springs: Vec<(usize, usize, f64, f64)> = Vec::new();
        for &(u, v) in &crystal.edges {
            if !is_wrap(u, v)
                && let Some(e) = crystal.adj[u].iter().find(|b| b.to == v)
            {
                spring_idx.insert((u, v), springs.len());
                springs.push((u, v, norm2(e.vec), 1.0));
            }
        }
        let corners = corners_order(crystal);
        let mut tris: Vec<(usize, usize, usize, f64, f64)> = Vec::new();
        for (u, lst) in corners.iter().enumerate() {
            let unrolled = crystal.adj[u].iter().any(|e| e.ang.is_some());
            let mut pairs: Vec<((f64, usize), (f64, usize))> =
                lst.windows(2).map(|w| (w[0], w[1])).collect();
            if !unrolled && lst.len() >= 3 {
                pairs.push((lst[lst.len() - 1], lst[0]));
            }
            for ((a1, v), (a2, w)) in pairs {
                if is_wrap(u, v) || is_wrap(u, w) {
                    continue;
                }
                let gap = if unrolled { a2 - a1 } else { (a2 - a1).rem_euclid(TWO_PI) };
                if v == w || !(1e-6 < gap && gap < PI - 1e-6) {
                    continue;
                }
                let Some(ev) = crystal.adj[u].iter().find(|b| b.to == v) else { continue };
                let Some(ew) = crystal.adj[u].iter().find(|b| b.to == w) else { continue };
                let key = (v.min(w), v.max(w));
                if let std::collections::hash_map::Entry::Vacant(ent) = spring_idx.entry(key) {
                    ent.insert(springs.len());
                    springs.push((key.0, key.1, norm2(sub2(ew.vec, ev.vec)), 0.5));
                }
                let a0 = cross2(ev.vec, ew.vec);
                if a0 > 1e-9 {
                    tris.push((u, v, w, a0, 0.7 / a0.sqrt()));
                }
            }
        }
        let x0: Vec<f64> = anchor_pos.iter().flat_map(|p| *p).collect();
        Self { n, x0, anchor: if non_orient { 0.05 } else { 1e-3 }, springs, tris }
    }

    pub fn cost(&self, z: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for &(u, v, rest, wt) in &self.springs {
            let r = wt * ((z[2 * u] - z[2 * v]).hypot(z[2 * u + 1] - z[2 * v + 1]) - rest);
            sum_sq += r * r;
        }
        for &(u, v, w, ta, tsc) in &self.tris {
            let (e1x, e1y) = (z[2 * v] - z[2 * u], z[2 * v + 1] - z[2 * u + 1]);
            let (e2x, e2y) = (z[2 * w] - z[2 * u], z[2 * w + 1] - z[2 * u + 1]);
            let r = tsc * (e1x * e2y - e1y * e2x - ta);
            sum_sq += r * r;
        }
        for (zc, &x0c) in z.iter().zip(&self.x0) {
            let r = self.anchor * (zc - x0c);
            sum_sq += r * r;
        }
        0.5 * sum_sq
    }
}

type NormalEqs = (Vec<Triplet<usize, usize, f64>>, Vec<f64>, Vec<f64>, f64);

impl RelaxSetup {

    fn assemble(
        &self,
        z: &[f64],
        free_map: Option<&[Option<usize>]>,
        ndof: usize,
        build_jtj: bool,
    ) -> NormalEqs {
        let cap = if build_jtj { self.springs.len() * 16 + self.tris.len() * 36 + ndof } else { 0 };
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

        for &(u, v, rest, wt) in &self.springs {
            let (dx, dy) = (z[2 * u] - z[2 * v], z[2 * u + 1] - z[2 * v + 1]);
            let ln_raw = dx.hypot(dy);
            let inv_ln = wt / ln_raw.max(1e-12);
            let (gx, gy) = (inv_ln * dx, inv_ln * dy);
            add_row(&[2 * u, 2 * u + 1, 2 * v, 2 * v + 1], &[gx, gy, -gx, -gy], wt * (ln_raw - rest));
        }
        for &(u, v, w, ta, tsc) in &self.tris {
            let (e1x, e1y) = (z[2 * v] - z[2 * u], z[2 * v + 1] - z[2 * u + 1]);
            let (e2x, e2y) = (z[2 * w] - z[2 * u], z[2 * w + 1] - z[2 * u + 1]);
            let (dvx, dvy, dwx, dwy) = (tsc * e2y, -tsc * e2x, -tsc * e1y, tsc * e1x);
            add_row(
                &[2 * v, 2 * v + 1, 2 * w, 2 * w + 1, 2 * u, 2 * u + 1],
                &[dvx, dvy, dwx, dwy, -dvx - dwx, -dvy - dwy],
                tsc * (e1x * e2y - e1y * e2x - ta),
            );
        }
        let a2 = self.anchor * self.anchor;
        for c in 0..(2 * self.n) {
            let rc = self.anchor * (z[c] - self.x0[c]);
            sum_sq += rc * rc;
            if let Some(k) = free_map.map_or(Some(c), |m| m[c]) {
                g[k] += a2 * (z[c] - self.x0[c]);
                if build_jtj {
                    diag[k] += a2;
                    triplets.push(Triplet::new(k, k, a2));
                }
            }
        }
        (triplets, diag, g, 0.5 * sum_sq)
    }
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

/// Minimise `1/2 |r(z)|^2` via sparse Cholesky Gauss-Newton normal equations `(J^T J + mu D) dz = -J^T r`,
/// reusing the Cholesky factor on chord steps once relative cost drop falls below `freeze = 1e-3`.
pub fn gauss_newton(
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
        let (mut base_trips, mut diag, g, _) = setup.assemble(&z, None, ndof, llt.is_none() || !frozen);
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
                let (t, d, _, _) = setup.assemble(&z, None, ndof, true);
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
    let (trips, diag, g, f) = setup.assemble(z, free_map.as_deref(), ndof, true);
    let rhs = Mat::from_fn(ndof, 1, |row, _| -g[row]);
    for mu in [0.0, 1e-8, 1e-5, 1e-2] {
        if let Some(llt) = build_spd_cholesky(ndof, &trips, &diag, mu) {
            let sol = llt.solve(&rhs);
            return Some(((0..ndof).map(|k| sol[(k, 0)]).collect(), f));
        }
    }
    None
}

/// Gauss-Newton with backtracking restricted to `free` coordinate indices (holding pinned rim fixed).
pub fn relax_pinned(
    setup: &RelaxSetup,
    z0: &[f64],
    free: &[usize],
    max_iter: usize,
    rtol: f64,
) -> (Vec<f64>, usize) {
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

/// Planar-guarded Gauss-Newton relaxation: accepts a step only if cost drops and `badness(z)` does not grow.
pub fn relax_guarded<F: FnMut(&[f64]) -> usize>(
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

pub(crate) fn z_to_pos(z: &[f64]) -> Vec<[f64; 2]> {
    z.as_chunks::<2>().0.to_vec()
}

pub(crate) fn pos_to_z(pos: &[[f64; 2]]) -> Vec<f64> {
    pos.iter().flat_map(|p| *p).collect()
}

/// Relax `crystal.pos` in place using `faer` sparse Cholesky Gauss-Newton normal equations (`freeze = 1e-3`)
/// and mark any remaining stretched bonds (`> 1.8x`) as `ghost_stub`s.
pub fn relax_picture(crystal: &mut BuiltCrystal) {
    let x0 = crystal.pos.clone();
    let setup = RelaxSetup::from_crystal(crystal, &x0);
    if !setup.springs.is_empty() {
        let (z, _, _, _) = gauss_newton(&setup, &setup.x0, 500, 1e-10, 1e-3);
        crystal.pos = z_to_pos(&z);
    }
    mark_long_bonds(crystal);
}
