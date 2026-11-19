//! Homogenised GSV moduli (`#041`, `#090`): Rust port of `papertilling/defects/gsv_elastic.py`
//! (`cauchy_born_tensor`, `acoustic_tensor`, `barnett_lothe_tensor`, `dipole_prelog_coeff`) and a rotation-invariant
//! symmetry classification of `C_{iajb}`.
//!
//! `C[i][a][j][b] = K4[2i + a][2j + b]`, `K4 = J_eff^T J_eff / area`, `J_eff = J_G + J_eta eta_G` with the Born-Huang
//! internal shifts `eta_G = -(J_eta^T J_eta)^+ J_eta^T J_G` (minimum-norm least squares, same as NumPy `lstsq`; `J_eff`
//! does not depend on the null-space choice). The pseudoinverse uses a Jacobi eigen-decomposition with NumPy's default
//! cut-off `eps * max(m, n) * lambda_max`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::types::Cell2D;

pub type Tensor4 = [[[[f64; 2]; 2]; 2]; 2];

/// Symmetric eigen-decomposition by cyclic Jacobi: `(eigenvalues, eigenvectors as columns)`.
pub(crate) fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        let diag: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-32 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let th = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let t = if th == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

type NbrList = Vec<(usize, (i32, i32), [f64; 2])>;

type BondKey = ((i32, i32, usize), (i32, i32, usize));

fn bond_rows(cell: &Cell2D) -> (Vec<[f64; 4]>, Vec<Vec<f64>>, usize) {
    let ns = cell.basis.len();
    let n_eta = 2 * ns.saturating_sub(1);
    let mut nbrs: Vec<NbrList> = vec![Vec::new(); ns];
    let mut seen: BTreeSet<BondKey> = BTreeSet::new();
    let mut bonds_cell = Vec::new();
    for b in &cell.bonds {
        let (i, j, n1, n2) = (b.i, b.j, b.n1, b.n2);
        let canon = ((0, 0, i), (n1, n2, j)).min(((0, 0, j), (-n1, -n2, i)));
        let (f1, f2) = (f64::from(n1), f64::from(n2));
        let dv = [
            cell.basis[j][0] + f1 * cell.a1[0] + f2 * cell.a2[0] - cell.basis[i][0],
            cell.basis[j][1] + f1 * cell.a1[1] + f2 * cell.a2[1] - cell.basis[i][1],
        ];
        for (s_from, s_to, sh, vec) in [(i, j, (n1, n2), dv), (j, i, (-n1, -n2), [-dv[0], -dv[1]])] {
            if !nbrs[s_from].iter().any(|&(jj, ss, _)| jj == s_to && ss == sh) {
                nbrs[s_from].push((s_to, sh, vec));
            }
        }
        if seen.insert(canon) {
            bonds_cell.push((i, j, dv));
        }
    }
    let (mut rows_g, mut rows_eta) = (Vec::new(), Vec::new());
    let add = |me: &mut Vec<f64>, s: usize, w: [f64; 2], sign: f64| {
        if s > 0 {
            me[2 * (s - 1)] += sign * w[0];
            me[2 * (s - 1) + 1] += sign * w[1];
        }
    };
    for &(i, j, dv) in &bonds_cell {
        let l = dv[0].hypot(dv[1]).max(1e-12);
        let u = [dv[0] / l, dv[1] / l];
        rows_g.push([u[0] * dv[0], u[0] * dv[1], u[1] * dv[0], u[1] * dv[1]]);
        let mut me = vec![0.0; n_eta];
        add(&mut me, j, u, 1.0);
        add(&mut me, i, u, -1.0);
        rows_eta.push(me);
    }
    for (s, lst) in nbrs.iter().enumerate() {
        for ia in 0..lst.len() {
            let (ja, _, da) = lst[ia];
            let la = da[0].hypot(da[1]).max(1e-12);
            for &(jb, _, db) in &lst[ia + 1..] {
                let lb = db[0].hypot(db[1]).max(1e-12);
                let c = 0.5 / (la * lb).sqrt();
                rows_g.push([
                    c * 2.0 * da[0] * db[0], c * (da[0] * db[1] + db[0] * da[1]),
                    c * (da[1] * db[0] + db[1] * da[0]), c * 2.0 * da[1] * db[1],
                ]);
                let mut me = vec![0.0; n_eta];
                add(&mut me, jb, da, c);
                add(&mut me, ja, db, c);
                add(&mut me, s, [da[0] + db[0], da[1] + db[1]], -c);
                rows_eta.push(me);
            }
        }
    }
    (rows_g, rows_eta, n_eta)
}

/// Continuum stiffness `C_{iajb}` of the GSV bond model with Born-Huang sublattice relaxation
/// (port of `gsv_elastic.py::cauchy_born_tensor`).
pub fn cauchy_born_tensor(cell: &Cell2D) -> Tensor4 {
    cauchy_born_tensor_opt(cell, true)
}

/// As [`cauchy_born_tensor`]; `relax = false` freezes the internal shifts (`eta = 0`).
pub fn cauchy_born_tensor_opt(cell: &Cell2D, relax: bool) -> Tensor4 {
    let (mut jg, je, n_eta) = bond_rows(cell);
    if relax && n_eta > 0 {
        let mut n = vec![vec![0.0; n_eta]; n_eta];
        for r in &je {
            for p in 0..n_eta {
                for q in 0..n_eta {
                    n[p][q] += r[p] * r[q];
                }
            }
        }
        let (lam, v) = jacobi_eigen(n);
        let lmax = lam.iter().copied().fold(0.0, f64::max);
        let cut = f64::EPSILON * n_eta as f64 * lmax;
        // rhs = J_eta^T J_G (n_eta x 4); eta = -N^+ rhs; J_G += J_eta eta
        let mut rhs = vec![[0.0; 4]; n_eta];
        for (r, g) in je.iter().zip(&jg) {
            for (rp, &x) in rhs.iter_mut().zip(r) {
                for (slot, gc) in rp.iter_mut().zip(g) {
                    *slot += x * gc;
                }
            }
        }
        let mut eta = vec![[0.0; 4]; n_eta];
        for (k, &l) in lam.iter().enumerate() {
            if l <= cut {
                continue;
            }
            for c in 0..4 {
                let proj: f64 = (0..n_eta).map(|p| v[p][k] * rhs[p][c]).sum::<f64>() / l;
                for (p, e) in eta.iter_mut().enumerate() {
                    e[c] -= v[p][k] * proj;
                }
            }
        }
        for (r, g) in je.iter().zip(jg.iter_mut()) {
            for (c, gc) in g.iter_mut().enumerate() {
                *gc += (0..n_eta).map(|p| r[p] * eta[p][c]).sum::<f64>();
            }
        }
    }
    let area = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs();
    let mut c4 = [[[[0.0; 2]; 2]; 2]; 2];
    for (p, slot) in [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))] {
        for (q, slot2) in [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))] {
            c4[slot.0][slot.1][slot2.0][slot2.1] = jg.iter().map(|r| r[p] * r[q]).sum::<f64>() / area;
        }
    }
    c4
}

/// `M_ij = C_{iajb} x_a y_b`.
pub(crate) fn contract(c: &Tensor4, x: [f64; 2], y: [f64; 2]) -> [[f64; 2]; 2] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| c[i][k / 2][j][k % 2] * x[k / 2] * y[k % 2]).sum::<f64>()))
}

/// Acoustic tensor `Gamma_ij(theta) = C_{iajb} n_a n_b`, `n = (cos theta, sin theta)`.
pub fn acoustic_tensor(c: &Tensor4, theta: f64) -> [[f64; 2]; 2] {
    let n = [theta.cos(), theta.sin()];
    contract(c, n, n)
}

/// Barnett-Lothe prelogarithmic tensor `B = <mm - mn nn^-1 nm>` over `n_quad` angles (port of `barnett_lothe_tensor`).
///
/// Normalisation: `<.>` is the angular mean, so `B = 4 pi B_BL` with Barnett & Lothe's
/// `B_BL = (1/8 pi^2) int_0^{2pi} (...) d omega` (Hirth & Lothe 1982 ch. 13); the energy per length of one
/// dislocation is `b.B.b ln(R/r0) / (4 pi)`. Isotropic check (Hirth & Lothe 1982 ch. 3): `B = 2 mu (lambda + mu) /
/// (lambda + 2 mu) I = mu / (1 - nu) I`, i.e. the edge energy `mu b^2 ln(R/r0) / (4 pi (1 - nu))`.
pub fn barnett_lothe_tensor(c: &Tensor4, n_quad: usize) -> [[f64; 2]; 2] {
    let mut b = [[0.0; 2]; 2];
    for k in 0..n_quad {
        let th = 2.0 * std::f64::consts::PI * k as f64 / n_quad as f64;
        let m = [th.cos(), th.sin()];
        let n = [-th.sin(), th.cos()];
        let (mm, nn, mn) = (contract(c, m, m), contract(c, n, n), contract(c, m, n));
        let det = nn[0][0] * nn[1][1] - nn[0][1] * nn[1][0];
        let inv = [[nn[1][1] / det, -nn[0][1] / det], [-nn[1][0] / det, nn[0][0] / det]];
        // (mn nn^-1 mn^T)_ij; mn^T = nm by the major symmetry C_iajb = C_jbia
        let s = |i: usize, j: usize| (0..4).map(|k| mn[i][k / 2] * inv[k / 2][k % 2] * mn[j][k % 2]).sum::<f64>();
        for (i, row) in b.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x += mm[i][j] - s(i, j);
            }
        }
    }
    b.map(|row| row.map(|x| x / n_quad as f64))
}

/// Dipole prelogarithmic slope `K = b^T B b / (2 pi)` (port of `dipole_prelog_coeff`).
pub fn dipole_prelog_coeff(c: &Tensor4, b: [f64; 2], n_quad: usize) -> f64 {
    let bt = barnett_lothe_tensor(c, n_quad);
    let s: f64 = (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| b[i] * bt[i][j] * b[j]).sum();
    s / (2.0 * std::f64::consts::PI)
}

/// Point symmetry class of a 2D stiffness tensor (rotation-invariant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElasticClass {
    Isotropic,
    /// Square (tetragonal / "cubic") symmetry: invariant under 90-degree rotation.
    SquareCubic,
    /// Has a mirror line (rectangular).
    Orthotropic,
    /// No mirror line (oblique).
    Monoclinic,
}

/// Classify from the harmonics of `E(phi) = C_{iajb} e_i e_a e_j e_b`, `e = (cos phi, sin phi)`: a 2D stiffness with
/// major and minor symmetry splits into two isotropic scalars, one order-2 and one order-4 harmonic, all seen by `E`.
/// Zero order-2 and order-4 parts: isotropic; only order-4: square; both with a common mirror (`psi4 = 2 psi2 mod pi`):
/// orthotropic; otherwise monoclinic.
pub fn classify_elastic(c: &Tensor4) -> ElasticClass {
    let n = 16;
    let (mut a2, mut b2, mut a4, mut b4, mut scale) = (0.0, 0.0, 0.0, 0.0, 0.0f64);
    for k in 0..n {
        let ph = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let e = [ph.cos(), ph.sin()];
        let m = contract(c, e, e);
        let val = e[0] * (m[0][0] * e[0] + m[0][1] * e[1]) + e[1] * (m[1][0] * e[0] + m[1][1] * e[1]);
        scale = scale.max(val.abs());
        a2 += val * (2.0 * ph).cos();
        b2 += val * (2.0 * ph).sin();
        a4 += val * (4.0 * ph).cos();
        b4 += val * (4.0 * ph).sin();
    }
    let tol = 1e-8 * scale.max(1e-300) * n as f64;
    let (d2, d4) = (a2.hypot(b2), a4.hypot(b4));
    match (d2 <= tol, d4 <= tol) {
        (true, true) => ElasticClass::Isotropic,
        (true, false) => ElasticClass::SquareCubic,
        (false, true) => ElasticClass::Orthotropic,
        (false, false) => {
            let (p2, p4) = (b2.atan2(a2), b4.atan2(a4));
            if (p4 - 2.0 * p2).sin().abs() <= 1e-7 { ElasticClass::Orthotropic } else { ElasticClass::Monoclinic }
        }
    }
}

/// `C'_{iajb} = R_ip R_aq R_jr R_bs C_{pqrs}` (rotation by `phi`).
pub fn rotate_tensor(c: &Tensor4, phi: f64) -> Tensor4 {
    let (s, co) = phi.sin_cos();
    let r = [[co, -s], [s, co]];
    // index k in 0..16 runs over (p, q, rr, ss) = bits (3, 2, 1, 0)
    let comp = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        (0..16).map(|k| {
            let (p, q, rr, ss) = (k >> 3, (k >> 2) & 1, (k >> 1) & 1, k & 1);
            r[i][p] * r[a][q] * r[j][rr] * r[b][ss] * c[p][q][rr][ss]
        }).sum()
    };
    std::array::from_fn(|i| std::array::from_fn(|a| std::array::from_fn(|j| std::array::from_fn(|b| comp(i, a, j, b)))))
}
