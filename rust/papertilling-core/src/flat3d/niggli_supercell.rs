//! Proper `det = +1` 3D Krivy-Gruber Niggli reduction, exact `CELLTRAN` 2D/3D
//! supercell builders, and finite/infinitesimal cell strain tensors.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::bieberbach::{IntMat, invert_unimodular, smith_normal_form};
use crate::types::{Bond2D, Cell2D};

pub use super::boundary_bcs::{
    BoundaryBond2d, BoundaryBond3d, BoundaryCondition, BoundaryLattice2d, BoundaryLattice3d,
    apply_boundary_conditions_2d, apply_boundary_conditions_3d,
};
use super::{Cell3D, canon_bond_3d, det3_f64, dot3, frac_to_cart_3d, inv3_f64, lattice_parameters_3d, transpose3};

/// Result of a proper `det(p_mat) == +1` 3D Krivy-Gruber Niggli reduction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NiggliReduction3d {
    pub reduced: [[f64; 3]; 3],
    pub p_mat: [[i32; 3]; 3],
    pub lengths: [f64; 3],
    pub angles_deg: [f64; 3],
    pub niggli_type: u8,
    pub volume: f64,
    pub iterations: usize,
}

fn det3_i32(m: [[i32; 3]; 3]) -> i32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Krivy-Gruber 3D Niggli reduction on row-vector lattice `[a1, a2, a3]`,
/// returning `reduced = p_mat * lat` with strictly proper `det(p_mat) == +1`.
///
/// Steps A1-A8 follow I. Krivy & B. Gruber, Acta Cryst. A32 (1976) 297-298, with the
/// epsilon-tolerant comparisons of R. W. Grosse-Kunstleve, N. K. Sauter & P. D. Adams,
/// Acta Cryst. A60 (2004) 1-6. Swaps are done without the sign change of A1/A2 and the
/// handedness is restored at the end by negating all three vectors (metric unchanged).
///
/// # Errors
/// Returns an error for a non-finite or singular basis, integer overflow, or non-convergence.
pub fn niggli_reduce_3d(lat: [[f64; 3]; 3]) -> Result<NiggliReduction3d, String> {
    if lat.iter().flatten().any(|v| !v.is_finite()) {
        return Err("non-finite 3D lattice vector".to_string());
    }
    let vol0 = det3_f64(lat).abs();
    let max_n2 = dot3(lat[0], lat[0]).max(dot3(lat[1], lat[1])).max(dot3(lat[2], lat[2]));
    if vol0 < 1e-12 * max_n2.powf(1.5).max(1e-12) {
        return Err("degenerate or singular 3D lattice basis".to_string());
    }
    let mut v = lat;
    let mut p = [[1_i32, 0, 0], [0, 1, 0], [0, 0, 1]];
    let eps = 1e-6 * vol0.powf(2.0 / 3.0);

    let metrics = |v: &[[f64; 3]; 3]| -> (f64, f64, f64, f64, f64, f64) {
        (dot3(v[0], v[0]), dot3(v[1], v[1]), dot3(v[2], v[2]), 2.0 * dot3(v[1], v[2]), 2.0 * dot3(v[0], v[2]), 2.0 * dot3(v[0], v[1]))
    };
    let sgn = |x: f64| -> i32 { if x > eps { 1 } else if x < -eps { -1 } else { 0 } };
    let mut iterations = 0usize;
    let mut converged = false;

    for it in 0..10_000 {
        iterations = it;
        let (a, b, _c, xi, eta, _zeta) = metrics(&v);
        if a > b + eps || ((a - b).abs() <= eps && xi.abs() > eta.abs() + eps) {
            v.swap(0, 1);
            p.swap(0, 1);
            continue;
        }
        let (_a, b, c, _xi, eta, zeta) = metrics(&v);
        if b > c + eps || ((b - c).abs() <= eps && eta.abs() > zeta.abs() + eps) {
            v.swap(1, 2);
            p.swap(1, 2);
            continue;
        }
        let (_a, _b, _c, xi, eta, zeta) = metrics(&v);
        let lmn = [sgn(xi), sgn(eta), sgn(zeta)];
        let signs = if lmn[0] * lmn[1] * lmn[2] == 1 {
            // A3: make xi, eta, zeta all positive.
            lmn
        } else {
            // A4 (Krivy-Gruber): i = j = k = 1; flip each vector whose product is > 0; remember the
            // last zero product; if i*j*k = -1, flip that one too (keeps det = +1).
            let mut r = [1_i32; 3];
            for (ri, &sv) in r.iter_mut().zip(&lmn) {
                if sv > 0 {
                    *ri = -1;
                }
            }
            let last_zero = lmn.iter().rposition(|&sv| sv == 0);
            if r[0] * r[1] * r[2] < 0
                && let Some(k) = last_zero
            {
                r[k] = -1;
            }
            r
        };
        for i in 0..3 {
            if signs[i] < 0 {
                v[i] = [-v[i][0], -v[i][1], -v[i][2]];
                p[i] = [-p[i][0], -p[i][1], -p[i][2]];
            }
        }
        let (a, b, _c, xi, eta, zeta) = metrics(&v);
        if xi.abs() > b + eps || ((xi - b).abs() <= eps && 2.0 * eta < zeta - eps) || ((xi + b).abs() <= eps && zeta < -eps) {
            let s = if xi > 0.0 { 1 } else { -1 };
            for k in 0..3 {
                v[2][k] -= f64::from(s) * v[1][k];
                p[2][k] = p[2][k].checked_sub(s * p[1][k]).ok_or("Niggli integer overflow")?;
            }
            continue;
        }
        if eta.abs() > a + eps || ((eta - a).abs() <= eps && 2.0 * xi < zeta - eps) || ((eta + a).abs() <= eps && zeta < -eps) {
            let s = if eta > 0.0 { 1 } else { -1 };
            for k in 0..3 {
                v[2][k] -= f64::from(s) * v[0][k];
                p[2][k] = p[2][k].checked_sub(s * p[0][k]).ok_or("Niggli integer overflow")?;
            }
            continue;
        }
        if zeta.abs() > a + eps || ((zeta - a).abs() <= eps && 2.0 * xi < eta - eps) || ((zeta + a).abs() <= eps && eta < -eps) {
            let s = if zeta > 0.0 { 1 } else { -1 };
            for k in 0..3 {
                v[1][k] -= f64::from(s) * v[0][k];
                p[1][k] = p[1][k].checked_sub(s * p[0][k]).ok_or("Niggli integer overflow")?;
            }
            continue;
        }
        let s_all = xi + eta + zeta + a + b;
        if s_all < -eps || (s_all.abs() <= eps && 2.0 * (a + eta) + zeta > eps) {
            for k in 0..3 {
                v[2][k] += v[0][k] + v[1][k];
                p[2][k] = p[2][k].checked_add(p[0][k] + p[1][k]).ok_or("Niggli integer overflow")?;
            }
            continue;
        }
        converged = true;
        break;
    }
    if !converged {
        return Err("Niggli reduction did not converge in 10000 steps".to_string());
    }

    if det3_i32(p) == -1 {
        for r in 0..3 {
            for c in 0..3 {
                v[r][c] = -v[r][c];
                p[r][c] = -p[r][c];
            }
        }
    }
    if det3_i32(p) != 1 {
        return Err("Niggli transformation lost unimodularity".to_string());
    }
    let (_, _, _, xi, eta, zeta) = metrics(&v);
    let niggli_type = if xi > eps && eta > eps && zeta > eps { 1 } else { 2 };
    let (lengths, angles_deg) = lattice_parameters_3d(&v);
    Ok(NiggliReduction3d { reduced: v, p_mat: p, lengths, angles_deg, niggli_type, volume: det3_f64(v).abs(), iterations })
}

fn adj3_i128(m: [[i128; 3]; 3]) -> [[i128; 3]; 3] {
    [
        [m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]],
        [m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]],
        [m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]],
    ]
}

fn mat3_vec_i128(m: [[i128; 3]; 3], v: [i128; 3]) -> [i128; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/// Build an exact 3D `CELLTRAN` supercell (`new_lat = celltran * old_lat`, `det(celltran) > 0`).
///
/// # Errors
/// Returns an error if `det(celltran) <= 0`.
pub fn build_supercell_3d(cell: &Cell3D, celltran: [[i32; 3]; 3]) -> Result<Cell3D, String> {
    let det = det3_i32(celltran);
    if det <= 0 {
        return Err(format!("3D CELLTRAN must have det > 0, got {det}"));
    }
    let det128 = i128::from(det);
    let m_arr = [0, 1, 2].map(|r| [0, 1, 2].map(|c| i128::from(celltran[c][r])));
    let adj = adj3_i128(m_arr);
    let m_int = IntMat::from_rows(&m_arr.iter().map(|r| r.to_vec()).collect::<Vec<_>>());
    let snf = smith_normal_form(&m_int, true);
    let u_inv = invert_unimodular(&snf.u);
    let norm_coset = |o: [i128; 3]| -> ([i128; 3], [i32; 3]) {
        let ao = mat3_vec_i128(adj, o);
        let q = ao.map(|x| x.div_euclid(det128));
        let rep = mat3_vec_i128(m_arr, ao.map(|x| x.rem_euclid(det128))).map(|x| x / det128);
        (rep, [q[0] as i32, q[1] as i32, q[2] as i32])
    };
    let mut offsets = Vec::with_capacity(det as usize);
    let mut off_map = HashMap::with_capacity(det as usize);
    for c0 in 0..snf.diag[0] {
        for c1 in 0..snf.diag[1] {
            for c2 in 0..snf.diag[2] {
                let raw = u_inv.mul_vec(&[c0, c1, c2]);
                let (rep, _) = norm_coset([raw[0], raw[1], raw[2]]);
                off_map.insert(rep, offsets.len());
                offsets.push(rep);
            }
        }
    }
    let new_lat = [0, 1, 2].map(|r| [0, 1, 2].map(|k| f64::from(celltran[r][0]) * cell.lattice[0][k] + f64::from(celltran[r][1]) * cell.lattice[1][k] + f64::from(celltran[r][2]) * cell.lattice[2][k]));
    let m = cell.basis.len();
    let mut new_basis = Vec::with_capacity(offsets.len() * m);
    let mut new_kinds = Vec::with_capacity(offsets.len() * m);
    for &rep in &offsets {
        let sh = frac_to_cart_3d(&cell.lattice, [rep[0] as f64, rep[1] as f64, rep[2] as f64]);
        for (i, &bi) in cell.basis.iter().enumerate() {
            new_basis.push([bi[0] + sh[0], bi[1] + sh[1], bi[2] + sh[2]]);
            new_kinds.push(cell.site_kinds.get(i).copied().unwrap_or(0));
        }
    }
    let mut seen = BTreeSet::new();
    for (oidx, &rep) in offsets.iter().enumerate() {
        for bd in &cell.bonds {
            let (t_rep, q) = norm_coset([rep[0] + i128::from(bd.n1), rep[1] + i128::from(bd.n2), rep[2] + i128::from(bd.n3)]);
            let cb = canon_bond_3d(oidx * m + bd.i, off_map[&t_rep] * m + bd.j, q[0], q[1], q[2], bd.kind);
            seen.insert((cb.i, cb.j, cb.n1, cb.n2, cb.n3, cb.kind));
        }
    }
    let bonds = seen.into_iter().map(|(i, j, n1, n2, n3, kind)| super::Bond3D { i, j, n1, n2, n3, kind }).collect();
    Ok(Cell3D { lattice: new_lat, basis: new_basis, bonds, space_group_num: (det == 1).then_some(cell.space_group_num).flatten(), site_kinds: new_kinds })
}

/// Build an exact 2D `CELLTRAN` supercell (`[b1 | b2] = [a1 | a2] * celltran`, `det(celltran) > 0`).
///
/// # Errors
/// Returns an error if `det(celltran) <= 0`.
pub fn build_supercell_2d(cell: &Cell2D, celltran: [[i32; 2]; 2]) -> Result<Cell2D, String> {
    let det = celltran[0][0] * celltran[1][1] - celltran[0][1] * celltran[1][0];
    if det <= 0 {
        return Err(format!("2D CELLTRAN must have det > 0, got {det}"));
    }
    let d128 = i128::from(det);
    let s = [[i128::from(celltran[0][0]), i128::from(celltran[0][1])], [i128::from(celltran[1][0]), i128::from(celltran[1][1])]];
    let adj = [[s[1][1], -s[0][1]], [-s[1][0], s[0][0]]];
    let s_int = IntMat::from_rows(&[vec![s[0][0], s[0][1]], vec![s[1][0], s[1][1]]]);
    let snf = smith_normal_form(&s_int, true);
    let u_inv = invert_unimodular(&snf.u);
    let norm_coset = |o: [i128; 2]| -> ([i128; 2], [i32; 2]) {
        let ao = [adj[0][0] * o[0] + adj[0][1] * o[1], adj[1][0] * o[0] + adj[1][1] * o[1]];
        let q = [ao[0].div_euclid(d128), ao[1].div_euclid(d128)];
        let u = [ao[0].rem_euclid(d128), ao[1].rem_euclid(d128)];
        ([(s[0][0] * u[0] + s[0][1] * u[1]) / d128, (s[1][0] * u[0] + s[1][1] * u[1]) / d128], [q[0] as i32, q[1] as i32])
    };
    let mut offsets = Vec::with_capacity(det as usize);
    let mut off_map = HashMap::with_capacity(det as usize);
    for c0 in 0..snf.diag[0] {
        for c1 in 0..snf.diag[1] {
            let raw = u_inv.mul_vec(&[c0, c1]);
            let (rep, _) = norm_coset([raw[0], raw[1]]);
            off_map.insert(rep, offsets.len());
            offsets.push(rep);
        }
    }
    let b1 = [f64::from(celltran[0][0]) * cell.a1[0] + f64::from(celltran[1][0]) * cell.a2[0], f64::from(celltran[0][0]) * cell.a1[1] + f64::from(celltran[1][0]) * cell.a2[1]];
    let b2 = [f64::from(celltran[0][1]) * cell.a1[0] + f64::from(celltran[1][1]) * cell.a2[0], f64::from(celltran[0][1]) * cell.a1[1] + f64::from(celltran[1][1]) * cell.a2[1]];
    let m = cell.basis.len();
    let mut new_basis = Vec::with_capacity(offsets.len() * m);
    for &rep in &offsets {
        let sh = [rep[0] as f64 * cell.a1[0] + rep[1] as f64 * cell.a2[0], rep[0] as f64 * cell.a1[1] + rep[1] as f64 * cell.a2[1]];
        for &p in &cell.basis {
            new_basis.push([p[0] + sh[0], p[1] + sh[1]]);
        }
    }
    let mut seen = BTreeSet::new();
    for (oidx, &rep) in offsets.iter().enumerate() {
        for bd in &cell.bonds {
            let (t_rep, q) = norm_coset([rep[0] + i128::from(bd.n1), rep[1] + i128::from(bd.n2)]);
            let (u, v) = (oidx * m + bd.i, off_map[&t_rep] * m + bd.j);
            seen.insert(if (u, q[0], q[1]) <= (v, -q[0], -q[1]) { (u, v, q[0], q[1], bd.kind) } else { (v, u, -q[0], -q[1], bd.kind) });
        }
    }
    let bonds = seen.into_iter().map(|(i, j, n1, n2, kind)| Bond2D { i, j, n1, n2, kind }).collect();
    Ok(Cell2D { a1: b1, a2: b2, basis: new_basis, bonds })
}

/// Deformation gradient and finite/infinitesimal strain tensors between two 3D cells.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellStrain3d {
    pub deformation_gradient: [[f64; 3]; 3],
    pub green_lagrange: [[f64; 3]; 3],
    pub infinitesimal_strain: [[f64; 3]; 3],
    pub volumetric_strain: f64,
    pub von_mises_equivalent: f64,
}

/// Compute deformation gradient `F`, Green-Lagrange strain `E = 0.5*(F^T F - I)`,
/// and infinitesimal strain `eps = 0.5*(F + F^T) - I` between `reference` and `deformed`.
///
/// # Errors
/// Returns an error if `reference` is singular.
pub fn cell_strain_tensor(reference: [[f64; 3]; 3], deformed: [[f64; 3]; 3]) -> Result<CellStrain3d, String> {
    let a_ref_inv = inv3_f64(transpose3(reference))?;
    let a_def = transpose3(deformed);
    let mut f = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            f[i][j] = (0..3).map(|k| a_def[i][k] * a_ref_inv[k][j]).sum();
        }
    }
    let mut e_gl = [[0.0_f64; 3]; 3];
    let mut eps = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let ftf: f64 = (0..3).map(|k| f[k][i] * f[k][j]).sum();
            let id = if i == j { 1.0 } else { 0.0 };
            e_gl[i][j] = 0.5 * (ftf - id);
            eps[i][j] = 0.5 * (f[i][j] + f[j][i]) - id;
        }
    }
    // Von Mises equivalent strain sqrt(2/3 e':e') of the deviatoric Green-Lagrange strain e'
    // (e.g. Hill, *The Mathematical Theory of Plasticity*; uniaxial incompressible strain e -> |e|).
    let tr_3 = (e_gl[0][0] + e_gl[1][1] + e_gl[2][2]) / 3.0;
    let mut dev_sq = 0.0;
    for (i, row) in e_gl.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let d = val - if i == j { tr_3 } else { 0.0 };
            dev_sq += d * d;
        }
    }
    Ok(CellStrain3d {
        deformation_gradient: f,
        green_lagrange: e_gl,
        infinitesimal_strain: eps,
        volumetric_strain: det3_f64(f) - 1.0,
        von_mises_equivalent: (2.0 * dev_sq / 3.0).sqrt(),
    })
}
