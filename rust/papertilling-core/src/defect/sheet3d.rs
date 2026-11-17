//! Discrete 3D elastic sheet relaxation for crystal patches with defects (`Defects` and `Combine` tabs).
//!
//! Relaxes 3D vertex coordinates `X_i = (x_i, y_i, z_i)` under exact G-gauge-invariant GSV
//! metric elasticity (bond stretch + interior-face corner shear) and orientation-safe /
//! Z_2-twisted dihedral hinge bending.

#[cfg(test)]
use std::collections::HashSet;

use crate::defect::charge::{loop_charge, total_frank};
#[cfg(test)]
use crate::defect::cone_geometry::{add2, norm2, same_iso, sub2};
use crate::defect::gsv_elastic::{
    Elastic2DOptions, canonical_align_2d, relax_elastic_2d_with_model,
};
use crate::defect::gsv_mobius::seed_mobius_3d;
use crate::defect::gsv_model::{GsvModel, compute_gsv_diagnostics, eval_gsv_fg};
use crate::defect::sheet3d_energy::{
    bond_stretch_energy_3d, build_hinges, compute_strains, compute_vertex_omega, dist3_at,
    eval_gsv_sheet_fg, eval_hinge_bending_fg,
};
#[cfg(test)]
use crate::defect::sheet3d_energy::{Spring, eval_sheet_fg};
use crate::defect::sheet3d_opt::{center_and_align_horizontal, lbfgs_minimize};
#[cfg(test)]
use crate::types::Isometry2;
use crate::types::{BuiltCrystal, Cell2D};

use crate::defect::sheet3d_topology as topology;
#[cfg(test)]
use topology::extract_interior_faces;
use topology::validate_inputs;
pub use topology::validate_crystal as validate_sheet_crystal;

/// Elastic parameters and solver budget for [`relax_sheet_3d`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sheet3DOptions {
    /// Dimensionless bending modulus `kappa_b` (scaled internally by `a0^2`).
    pub kappa_b: f64,
    /// Maximum 3D L-BFGS iterations (`240` full quality, `90` fast drag preview).
    /// The separately diagnosed planar reference uses at most 120 iterations.
    pub max_iters: usize,
    /// RMS gradient convergence threshold.
    pub grad_tol: f64,
}

impl Default for Sheet3DOptions {
    fn default() -> Self {
        Self {
            kappa_b: 0.02,
            max_iters: 240,
            grad_tol: 1e-6,
        }
    }
}

/// Relaxed 3D elastic sheet geometry and per-bond / per-vertex diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct Sheet3DResult {
    pub xyz: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    /// Relative strain `(||X_u - X_v|| - l0) / l0` aligned with `crystal.edges`.
    pub bond_strain: Vec<f64>,
    /// Physical rest length `l0` aligned with every bond in `crystal.edges`, including display ghosts.
    pub bond_rest_len: Vec<f64>,
    /// Mean signed relative bond strain on each interior face in `faces`.
    pub face_strain: Vec<f64>,
    /// Discrete Gaussian curvature / angle deficit `2*pi - sum theta_{f,i}` (radians; `0.0` on boundary).
    pub vertex_omega: Vec<f64>,
    /// Marks vertices belonging to the outer boundary walk.
    pub is_boundary: Vec<bool>,
    /// Mean pristine rest bond length `a0`.
    pub mean_bond_len: f64,
    /// Nearest-neighbour bond-stretching energy `0.5 * sum_{e in E} (||X_i - X_j|| - l0)^2` in 3D.
    pub bond_stretch_energy: f64,
    /// Total GSV stretching + corner-shear energy in 3D.
    pub stretch_energy: f64,
    /// Hinge bending energy `0.5 * kappa_b * a0^2 * sum E_bend` in 3D.
    pub bend_energy: f64,
    /// Nearest-neighbour bond-stretching energy of the forced-2D (`z = 0`) relaxed embedding.
    pub planar_2d_bond_energy: f64,
    /// Total GSV energy of the forced-2D (`z = 0`) relaxed embedding.
    pub planar_2d_stretch_energy: f64,
    /// Per-site total GSV energy (`0.5 * sum_e E_s + sum_{corners} E_shear`) in 3D.
    pub site_gsv_energy: Vec<f64>,
    /// Per-bond total GSV energy (`E_s(e) + 0.5 * sum_{corners ni e} E_shear`) in 3D.
    pub bond_gsv_energy: Vec<f64>,
    /// Per-bond dimensionless RMS shear strain across incident face corners.
    pub bond_shear_strain: Vec<f64>,
    pub iterations: usize,
    /// Whether the 3D objective reached the requested RMS gradient tolerance.
    pub converged: bool,
    /// Whether the separately optimized planar reference reached its RMS gradient tolerance.
    pub planar_2d_converged: bool,
    /// Invalid input, unsupported topology and incomplete optimization are never silent.
    pub diagnostics: Vec<String>,
}

impl Sheet3DResult {
    /// Guard numerical output before rendering, including independently supplied/modified results.
    pub fn has_finite_geometry(&self) -> bool {
        self.xyz
            .iter()
            .flatten()
            .chain(&self.bond_strain)
            .chain(&self.bond_rest_len)
            .chain(&self.face_strain)
            .chain(&self.vertex_omega)
            .chain(&self.site_gsv_energy)
            .chain(&self.bond_gsv_energy)
            .chain(&self.bond_shear_strain)
            .all(|x| x.is_finite())
            && [
                self.mean_bond_len,
                self.bond_stretch_energy,
                self.stretch_energy,
                self.bend_energy,
                self.planar_2d_bond_energy,
                self.planar_2d_stretch_energy,
            ]
            .iter()
            .all(|x| x.is_finite())
            && self.mean_bond_len > 0.0
            && self.bond_rest_len.iter().all(|&l| l > 0.0)
            && self.is_boundary.len() == self.xyz.len()
            && self.vertex_omega.len() == self.xyz.len()
            && self.site_gsv_energy.len() == self.xyz.len()
            && self.bond_gsv_energy.len() == self.bond_rest_len.len()
            && self.bond_shear_strain.len() == self.bond_rest_len.len()
            && self.face_strain.len() == self.faces.len()
            && self
                .faces
                .iter()
                .all(|f| f.len() >= 3 && f.iter().all(|&u| u < self.xyz.len()))
    }
}

/// Relax `crystal` in `R^3` using exact GSV metric elasticity and dihedral hinge bending.
pub fn relax_sheet_3d(
    crystal: &BuiltCrystal,
    cell: &Cell2D,
    opts: &Sheet3DOptions,
    warm_z: Option<&[f64]>,
) -> Sheet3DResult {
    if let Err(message) = validate_inputs(crystal, cell, opts, warm_z) {
        return unavailable_result(message);
    }
    let n = crystal.pos.len();
    if n == 0 {
        return empty_result();
    }

    let model = match GsvModel::from_crystal(crystal, Some(cell)) {
        Ok(m) => m,
        Err(message) => return unavailable_result(message),
    };
    let a0 = model.mean_bond_len;
    let kappa_eff = opts.kappa_b * a0 * a0;
    if !a0.is_finite() || !kappa_eff.is_finite() {
        return unavailable_result("nonfinite physical rest lengths or bending scale".into());
    }
    let mut diagnostics = model.diagnostics.clone();
    let warm_z = warm_z.filter(|wz| wz.iter().any(|z| z.abs() > 1e-4));
    let hinges = build_hinges(&model.faces);

    let mut x_flat = vec![0.0_f64; 3 * n];
    for i in 0..n {
        x_flat[3 * i] = model.chart_2d[i][0];
        x_flat[3 * i + 1] = model.chart_2d[i][1];
    }
    let max_total_init_strain = model
        .stretches
        .iter()
        .map(|s| (dist3_at(&x_flat, s.u, s.v) - s.l0).abs() / s.l0)
        .fold(0.0_f64, f64::max);

    let mut x_2d = vec![0.0_f64; 2 * n];
    if !crystal.is_non_orientable
        && max_total_init_strain >= 1e-7
        && let Ok(r2) = relax_elastic_2d_with_model(&model, &Elastic2DOptions::default())
    {
        for i in 0..n {
            x_2d[2 * i] = r2.pos[i][0];
            x_2d[2 * i + 1] = r2.pos[i][1];
        }
    } else {
        for i in 0..n {
            x_2d[2 * i] = model.chart_2d[i][0];
            x_2d[2 * i + 1] = model.chart_2d[i][1];
        }
    }
    let _ = lbfgs_minimize(&mut x_2d, 200, 1e-10, |pos2, g2| {
        let (es, eh) = eval_gsv_fg::<2>(pos2, g2, &model.stretches, &model.shears_3d);
        es + eh
    });
    let mut g2 = vec![0.0; 2 * n];
    let (es2, eh2) = eval_gsv_fg::<2>(&x_2d, &mut g2, &model.stretches, &model.shears_3d);
    let planar_2d_stretch_energy = es2 + eh2;
    let planar_2d_bond_energy: f64 = model
        .stretches
        .iter()
        .map(|s| {
            let diff = (x_2d[2 * s.u] - x_2d[2 * s.v]).hypot(x_2d[2 * s.u + 1] - x_2d[2 * s.v + 1]) - s.l0;
            0.5 * diff * diff
        })
        .sum();
    let planar_2d_converged = gradient_rms(&g2) <= 1e-6;
    if !planar_2d_converged {
        diagnostics.push("planar reference not converged; strain comparison is provisional".into());
    }

    let mut x3 = vec![0.0_f64; 3 * n];
    if crystal.is_non_orientable && warm_z.is_none() {
        for (i, p) in seed_mobius_3d(crystal).into_iter().enumerate() {
            x3[3 * i] = p[0];
            x3[3 * i + 1] = p[1];
            x3[3 * i + 2] = p[2];
        }
    } else if let Some(wz) = warm_z {
        for i in 0..n {
            x3[3 * i] = x_2d[2 * i];
            x3[3 * i + 1] = x_2d[2 * i + 1];
            x3[3 * i + 2] = wz[i];
        }
    } else {
        let mut pts_2d: Vec<[f64; 2]> = x_2d.as_chunks::<2>().0.to_vec();
        canonical_align_2d(&mut pts_2d);
        for i in 0..n {
            x3[3 * i] = pts_2d[i][0];
            x3[3 * i + 1] = pts_2d[i][1];
        }
        if max_total_init_strain >= 1e-7 {
            let frank_deg = if model.has_valid_frames {
                total_frank(crystal)
            } else if !crystal.is_non_orientable {
                model
                    .faces
                    .iter()
                    .map(|f| {
                        let h = loop_charge(crystal, f).0;
                        if h.det() > 0.0 { h.m[1][0].atan2(h.m[0][0]).to_degrees() } else { 0.0 }
                    })
                    .sum()
            } else {
                0.0
            };
            seed_out_of_plane(&mut x3, n, frank_deg, a0, max_total_init_strain);
        }
    }

    let mut centroids_buf = vec![[0.0_f64; 3]; model.faces.len()];
    let mut centroid_grad = vec![[0.0_f64; 3]; model.faces.len()];
    let (_, iterations) = lbfgs_minimize(&mut x3, opts.max_iters, opts.grad_tol, |pos, grad| {
        eval_gsv_sheet_fg(
            pos,
            grad,
            &model,
            &hinges,
            kappa_eff,
            &mut centroids_buf,
            &mut centroid_grad,
        )
    });
    let mut g3 = vec![0.0; 3 * n];
    let energy = eval_gsv_sheet_fg(
        &x3,
        &mut g3,
        &model,
        &hinges,
        kappa_eff,
        &mut centroids_buf,
        &mut centroid_grad,
    );
    if !energy.is_finite() || x3.iter().chain(&g3).any(|x| !x.is_finite()) {
        return unavailable_result("nonfinite sheet objective or geometry".into());
    }
    let rms = gradient_rms(&g3);
    let converged = rms <= opts.grad_tol;
    if !converged {
        diagnostics.push(format!(
            "3D solver not converged (RMS gradient {rms:.2e}, tolerance {:.2e})",
            opts.grad_tol
        ));
    }

    let mut xyz: Vec<[f64; 3]> = (0..n)
        .map(|i| [x3[3 * i], x3[3 * i + 1], x3[3 * i + 2]])
        .collect();
    center_and_align_horizontal(&mut xyz, None);
    for i in 0..n {
        x3[3 * i] = xyz[i][0];
        x3[3 * i + 1] = xyz[i][1];
        x3[3 * i + 2] = xyz[i][2];
    }

    let bond_stretch_energy = bond_stretch_energy_3d(&x3, &model.stretches);
    let diag = compute_gsv_diagnostics::<3>(&x3, &model, &model.shears_3d);
    let stretch_energy = diag.stretch_energy + diag.shear_energy;
    let mut g_dummy = vec![0.0; 3 * n];
    let bend_energy = eval_hinge_bending_fg(
        &x3,
        &mut g_dummy,
        &model.faces,
        &hinges,
        kappa_eff,
        &mut centroids_buf,
        &mut centroid_grad,
    );
    let bend_energy = if bend_energy.is_finite() { bend_energy.max(0.0) } else { bend_energy };
    let (bond_strain, face_strain) = compute_strains(&xyz, crystal, &model.edge_rest, &model.faces);
    let vertex_omega = compute_vertex_omega(&xyz, &model.faces, &model.is_boundary);

    let result = Sheet3DResult {
        xyz,
        faces: model.faces,
        bond_strain,
        bond_rest_len: model.edge_rest,
        face_strain,
        vertex_omega,
        is_boundary: model.is_boundary,
        mean_bond_len: a0,
        bond_stretch_energy,
        stretch_energy,
        bend_energy,
        planar_2d_bond_energy,
        planar_2d_stretch_energy,
        site_gsv_energy: diag.site_gsv_energy,
        bond_gsv_energy: diag.bond_gsv_energy,
        bond_shear_strain: diag.bond_shear_strain,
        iterations,
        converged,
        planar_2d_converged,
        diagnostics,
    };
    if result.has_finite_geometry() {
        result
    } else {
        unavailable_result("nonfinite sheet diagnostics".into())
    }
}

fn empty_result() -> Sheet3DResult {
    Sheet3DResult {
        xyz: Vec::new(),
        faces: Vec::new(),
        bond_strain: Vec::new(),
        bond_rest_len: Vec::new(),
        face_strain: Vec::new(),
        vertex_omega: Vec::new(),
        is_boundary: Vec::new(),
        mean_bond_len: 1.0,
        bond_stretch_energy: 0.0,
        stretch_energy: 0.0,
        bend_energy: 0.0,
        planar_2d_bond_energy: 0.0,
        planar_2d_stretch_energy: 0.0,
        site_gsv_energy: Vec::new(),
        bond_gsv_energy: Vec::new(),
        bond_shear_strain: Vec::new(),
        iterations: 0,
        converged: true,
        planar_2d_converged: true,
        diagnostics: Vec::new(),
    }
}

fn unavailable_result(message: String) -> Sheet3DResult {
    Sheet3DResult {
        converged: false,
        planar_2d_converged: false,
        diagnostics: vec![message],
        ..empty_result()
    }
}

fn gradient_rms(gradient: &[f64]) -> f64 {
    if gradient.iter().any(|x| !x.is_finite()) {
        return f64::INFINITY;
    }
    gradient.iter().fold(0.0_f64, |norm, &g| norm.hypot(g)) / (gradient.len().max(1) as f64).sqrt()
}

#[cfg(test)]
fn add_bulk_tile_chord_springs(
    crystal: &BuiltCrystal,
    faces: &[Vec<usize>],
    has_valid_frames: bool,
    springs: &mut Vec<Spring>,
) {
    if !has_valid_frames {
        return;
    }
    let mut seen_pair: HashSet<(usize, usize)> = springs
        .iter()
        .map(|s| (s.u.min(s.v), s.u.max(s.v)))
        .collect();

    for f in faces {
        if f.len() < 4 {
            continue;
        }
        let (h, loop_s) = loop_charge(crystal, f);
        if !same_iso(&h, &Isometry2::IDENTITY, 1e-4) || norm2(loop_s) > 1e-4 {
            continue;
        }
        let mut q = Vec::with_capacity(f.len());
        let mut cur = [0.0_f64, 0.0];
        let mut dm = Isometry2::IDENTITY;
        let mut ok = true;
        for m in 0..f.len() {
            q.push(cur);
            let (u, v) = (f[m], f[(m + 1) % f.len()]);
            let Some(e) = crystal.adj[u].iter().find(|b| b.to == v) else {
                ok = false;
                break;
            };
            cur = add2(cur, dm.apply_linear(e.vec));
            let g = e.seam.unwrap_or(Isometry2::IDENTITY);
            dm = dm.compose(&g.inverse());
        }
        if !ok || norm2(cur) > 1e-4 {
            continue;
        }
        for m in 0..f.len() {
            let prev = (m + f.len() - 1) % f.len();
            let next = (m + 1) % f.len();
            let (u, w) = (f[prev], f[next]);
            if seen_pair.insert((u.min(w), u.max(w))) {
                let l0 = norm2(sub2(q[next], q[prev]));
                if l0 > 1e-6 {
                    springs.push(Spring {
                        u,
                        v: w,
                        l0,
                        weight: 0.45,
                        is_primary_bond: false,
                    });
                }
            }
        }
    }
}

fn seed_out_of_plane(
    x3: &mut [f64],
    n: usize,
    frank_deg: f64,
    a0: f64,
    max_strain: f64,
) {
    let inv_n = 1.0 / (n as f64);
    let cx = (0..n).map(|i| x3[3 * i]).sum::<f64>() * inv_n;
    let cy = (0..n).map(|i| x3[3 * i + 1]).sum::<f64>() * inv_n;
    let r_max = (0..n)
        .map(|i| (x3[3 * i] - cx).hypot(x3[3 * i + 1] - cy))
        .fold(a0, f64::max);
    let scale = (max_strain * 12.0).clamp(0.05, 1.0);

    for i in 0..n {
        let dx = x3[3 * i] - cx;
        let dy = x3[3 * i + 1] - cy;
        let r = dx.hypot(dy);
        let theta = dy.atan2(dx);
        let h = (7.31 * (dx / a0) + 11.17 * (dy / a0)).sin();
        x3[3 * i + 2] = if frank_deg > 3.0 {
            let omega_frac = (frank_deg / 360.0).clamp(0.02, 0.45);
            let slope = (2.0 * omega_frac).sqrt().clamp(0.25, 0.95);
            scale * (0.65 * slope * (r_max - r) + 0.02 * a0 * (2.0 * theta).cos() + 1e-3 * a0 * h)
        } else if frank_deg < -3.0 {
            let omega_frac = (-frank_deg / 360.0).clamp(0.02, 0.65);
            let amp = omega_frac.sqrt().clamp(0.25, 0.75);
            scale * (amp * r * (2.0 * theta).cos() + 0.03 * r * (3.0 * theta).sin() + 1e-3 * a0 * h)
        } else {
            let env = (-r * r / (12.0 * a0 * a0).max(0.25 * r_max * r_max)).exp();
            scale
                * a0
                * (0.45 * env * (1.0 + 0.4 * theta.cos())
                    + 0.18 * (r / r_max) * (2.0 * theta).cos()
                    + 2e-3 * h)
        };
    }
}

#[cfg(test)]
#[inline]
fn det_hash_unit(mut k: u64) -> f64 {
    k = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    k = (k ^ (k >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    k = (k ^ (k >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    k ^= k >> 31;
    ((k >> 11) as f64) * (2.0 / ((1u64 << 53) as f64)) - 1.0
}

#[cfg(test)]
#[path = "sheet3d_tests.rs"]
mod tests;
