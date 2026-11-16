//! Cheap preview relaxation for interactive frames (live pin drag): a few matrix-free Gauss-Newton steps on
//! the same spring + corner-area + weak-anchor energy as `relax_picture`, each solved approximately by
//! conjugate gradients preconditioned with the 2x2 site blocks of `J^T J`, so no sparse factorisation or
//! symbolic analysis is needed. The graph is untouched; only `pos` and the `ghost_stub` marks change.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::defect::surgery::mark_long_bonds;
use crate::plane::relax::{RelaxSetup, z_to_pos};
use crate::types::BuiltCrystal;

/// Most Gauss-Newton steps of the preview.
pub const PREVIEW_STEPS: usize = 4;
/// Most conjugate-gradient iterations per Gauss-Newton step.
pub const PREVIEW_CG_ITERS: usize = 60;
const CG_RTOL: f64 = 1e-3;
const STOP_RTOL: f64 = 1e-4;

/// What `relax_preview` did: Gauss-Newton steps taken, conjugate-gradient iterations, the energy before and
/// after (the `relax_picture` energy, anchored at the input positions), and wall time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PreviewReport {
    pub steps: usize,
    pub cg_iters: usize,
    pub cost_start: f64,
    pub cost_end: f64,
    pub ms: f64,
}

/// Residual gradients at `z`: springs `(gx, gy)` (`+g` on `u`, `-g` on `v`) and corner triangles
/// `(dvx, dvy, dwx, dwy)` (the apex `u` takes minus their sum); `pre` holds the inverted 2x2 site blocks.
struct Linear {
    spring: Vec<[f64; 2]>,
    tri: Vec<[f64; 4]>,
    pre: Vec<[f64; 3]>,
}

impl Linear {
    fn at(setup: &RelaxSetup, z: &[f64], g: &mut [f64]) -> Self {
        let a2 = setup.anchor * setup.anchor;
        let mut blk = vec![[a2, 0.0, a2]; setup.n];
        for c in 0..g.len() {
            g[c] = a2 * (z[c] - setup.x0[c]);
        }
        let add = |g: &mut [f64], blk: &mut [[f64; 3]], s: usize, d: [f64; 2], r: f64| {
            g[2 * s] += d[0] * r;
            g[2 * s + 1] += d[1] * r;
            let b = &mut blk[s];
            b[0] += d[0] * d[0];
            b[1] += d[0] * d[1];
            b[2] += d[1] * d[1];
        };
        let mut spring = Vec::with_capacity(setup.springs.len());
        for &(u, v, rest, wt) in &setup.springs {
            let (dx, dy) = (z[2 * u] - z[2 * v], z[2 * u + 1] - z[2 * v + 1]);
            let ln = dx.hypot(dy);
            let inv = wt / ln.max(1e-12);
            let d = [inv * dx, inv * dy];
            let r = wt * (ln - rest);
            add(g, &mut blk, u, d, r);
            add(g, &mut blk, v, [-d[0], -d[1]], r);
            spring.push(d);
        }
        let mut tri = Vec::with_capacity(setup.tris.len());
        for &(u, v, w, ta, tsc) in &setup.tris {
            let (e1x, e1y) = (z[2 * v] - z[2 * u], z[2 * v + 1] - z[2 * u + 1]);
            let (e2x, e2y) = (z[2 * w] - z[2 * u], z[2 * w + 1] - z[2 * u + 1]);
            let d = [tsc * e2y, -tsc * e2x, -tsc * e1y, tsc * e1x];
            let r = tsc * (e1x * e2y - e1y * e2x - ta);
            add(g, &mut blk, v, [d[0], d[1]], r);
            add(g, &mut blk, w, [d[2], d[3]], r);
            add(g, &mut blk, u, [-d[0] - d[2], -d[1] - d[3]], r);
            tri.push(d);
        }
        let pre = blk
            .into_iter()
            .map(|[a, b, c]| {
                let det = a * c - b * b;
                if det > 1e-300 { [c / det, -b / det, a / det] } else { [1.0 / a.max(1e-300), 0.0, 1.0 / c.max(1e-300)] }
            })
            .collect();
        Self { spring, tri, pre }
    }

    /// `out = (J^T J + a^2 I) x`.
    fn apply(&self, setup: &RelaxSetup, x: &[f64], out: &mut [f64]) {
        let a2 = setup.anchor * setup.anchor;
        for (o, &xi) in out.iter_mut().zip(x) {
            *o = a2 * xi;
        }
        for (&(u, v, _, _), &[gx, gy]) in setup.springs.iter().zip(&self.spring) {
            let jx = gx * (x[2 * u] - x[2 * v]) + gy * (x[2 * u + 1] - x[2 * v + 1]);
            out[2 * u] += gx * jx;
            out[2 * u + 1] += gy * jx;
            out[2 * v] -= gx * jx;
            out[2 * v + 1] -= gy * jx;
        }
        for (&(u, v, w, _, _), d) in setup.tris.iter().zip(&self.tri) {
            let (dux, duy) = (-d[0] - d[2], -d[1] - d[3]);
            let jx = d[0] * x[2 * v] + d[1] * x[2 * v + 1] + d[2] * x[2 * w] + d[3] * x[2 * w + 1] + dux * x[2 * u] + duy * x[2 * u + 1];
            out[2 * v] += d[0] * jx;
            out[2 * v + 1] += d[1] * jx;
            out[2 * w] += d[2] * jx;
            out[2 * w + 1] += d[3] * jx;
            out[2 * u] += dux * jx;
            out[2 * u + 1] += duy * jx;
        }
    }

    fn precondition(&self, r: &[f64], out: &mut [f64]) {
        for (s, p) in self.pre.iter().enumerate() {
            let (rx, ry) = (r[2 * s], r[2 * s + 1]);
            out[2 * s] = p[0] * rx + p[1] * ry;
            out[2 * s + 1] = p[1] * rx + p[2] * ry;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Preconditioned conjugate gradients for `(J^T J + a^2 I) x = -g`; returns `(x, iterations)`.
fn pcg(setup: &RelaxSetup, lin: &Linear, g: &[f64], max_iter: usize) -> (Vec<f64>, usize) {
    let n = g.len();
    let mut x = vec![0.0; n];
    let mut r: Vec<f64> = g.iter().map(|&gi| -gi).collect();
    let b_norm = dot(&r, &r).sqrt();
    if b_norm == 0.0 {
        return (x, 0);
    }
    let mut zv = vec![0.0; n];
    lin.precondition(&r, &mut zv);
    let mut p = zv.clone();
    let mut rz = dot(&r, &zv);
    let mut ap = vec![0.0; n];
    let mut it = 0;
    while it < max_iter {
        it += 1;
        lin.apply(setup, &p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        if dot(&r, &r).sqrt() <= CG_RTOL * b_norm {
            break;
        }
        lin.precondition(&r, &mut zv);
        let rz_new = dot(&r, &zv);
        let beta = rz_new / rz;
        rz = rz_new;
        for k in 0..n {
            p[k] = zv[k] + beta * p[k];
        }
    }
    (x, it)
}

/// Fast, non-final relaxation of `crystal.pos` for interactive frames: same graph and same energy as
/// `relax_picture`, at most `PREVIEW_STEPS` Gauss-Newton steps of at most `PREVIEW_CG_ITERS` preconditioned
/// conjugate-gradient iterations each. After the first step, no new step starts once `max_ms_hint`
/// milliseconds have elapsed (pass `f64::INFINITY` for a result that does not depend on timing).
/// Over-stretched bonds are marked as `ghost_stub`s as in `relax_picture`. Run `relax_picture` for the
/// final picture (the preview may keep a few crossings where the surgery picture is far from relaxed).
pub fn relax_preview(crystal: &mut BuiltCrystal, max_ms_hint: f64) -> PreviewReport {
    let t0 = Instant::now();
    let x0 = crystal.pos.clone();
    let setup = RelaxSetup::from_crystal(crystal, &x0);
    let mut rep = PreviewReport::default();
    if !setup.springs.is_empty() {
        let mut z = setup.x0.clone();
        let mut f = setup.cost(&z);
        rep.cost_start = f;
        let mut g = vec![0.0; 2 * setup.n];
        for step in 0..PREVIEW_STEPS {
            if step > 0 && t0.elapsed().as_secs_f64() * 1000.0 > max_ms_hint {
                break;
            }
            let lin = Linear::at(&setup, &z, &mut g);
            let (dz, it) = pcg(&setup, &lin, &g, PREVIEW_CG_ITERS);
            rep.cg_iters += it;
            let mut a = 1.0;
            let mut accepted = None;
            for _ in 0..8 {
                let z2: Vec<f64> = z.iter().zip(&dz).map(|(&zi, &di)| zi + a * di).collect();
                let f2 = setup.cost(&z2);
                if f2 < f {
                    accepted = Some((z2, f2));
                    break;
                }
                a *= 0.5;
            }
            let Some((z2, f2)) = accepted else { break };
            let drop = f - f2;
            (z, f) = (z2, f2);
            rep.steps += 1;
            if drop <= STOP_RTOL * f {
                break;
            }
        }
        rep.cost_end = f;
        crystal.pos = z_to_pos(&z);
    }
    mark_long_bonds(crystal);
    rep.ms = t0.elapsed().as_secs_f64() * 1000.0;
    rep
}
