//! Discrete 3D elastic sheet relaxation for crystal patches with defects (`Defects` and `Combine` tabs).
//!
//! Relaxes 3D vertex coordinates `X_i = (x_i, y_i, z_i)` under nearest-neighbour bond stretching
//! (plus unfrustrated bulk-tile corner-chord shear rigidity) and orientation-safe dihedral hinge bending:
//!
//!   E = 1/2 sum_{(i,j)} (||X_i - X_j|| - l_{ij}^0)^2 + (kappa_b / 2) sum_{hinges} E_bend(f_1, f_2).

use std::collections::{HashMap, HashSet};

use crate::defect::charge::{check_frames, loop_charge, total_frank};
use crate::defect::cone_geometry::{add2, norm2, same_iso, sub2};
use crate::defect::sheet3d_energy::{
    Spring, build_hinges, compute_strains, compute_vertex_omega, dist3_at, eval_2d_energies,
    eval_2d_stretch_fg, eval_energies, eval_sheet_fg,
};
use crate::defect::sheet3d_opt::{center_and_align_horizontal, lbfgs_minimize};
use crate::types::{BuiltCrystal, Cell2D, Isometry2};

#[path = "sheet3d_topology.rs"]
mod topology;
pub use topology::validate_crystal as validate_sheet_crystal;
use topology::{extract_interior_faces, prepare_planar_chart, validate_inputs};

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
    /// Total stretching (bond + bulk-tile shear chord) energy in 3D.
    pub stretch_energy: f64,
    /// Hinge bending energy `0.5 * kappa_b * a0^2 * sum E_bend` in 3D.
    pub bend_energy: f64,
    /// Nearest-neighbour bond-stretching energy of the forced-2D (`z = 0`) relaxed embedding.
    pub planar_2d_bond_energy: f64,
    /// Total stretching energy of the forced-2D (`z = 0`) relaxed embedding.
    pub planar_2d_stretch_energy: f64,
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
            && self.face_strain.len() == self.faces.len()
            && self
                .faces
                .iter()
                .all(|f| f.len() >= 3 && f.iter().all(|&u| u < self.xyz.len()))
    }
}

/// Relax `crystal` in `R^3` using rest bond lengths from `cell` (with optional `warm_z` seed per vertex).
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

    let kind_len = rest_lengths_from_cell(cell);
    let mut edge_rest = vec![0.0_f64; crystal.edges.len()];
    let mut springs = Vec::with_capacity(crystal.edges.len() * 2);
    let mut len_sum = 0.0_f64;
    let mut len_cnt = 0_usize;

    for (e_idx, &(u, v)) in crystal.edges.iter().enumerate() {
        let d_opt = crystal
            .adj
            .get(u)
            .and_then(|row| row.iter().find(|d| d.to == v));
        let l0 = match d_opt {
            Some(d) if !d.across => kind_len
                .get(&d.kind)
                .copied()
                .unwrap_or_else(|| norm2(d.vec))
                .max(1e-6),
            Some(d) => {
                let vlen = norm2(d.vec);
                if vlen > 1e-6 {
                    vlen
                } else {
                    kind_len.get(&d.kind).copied().unwrap_or(1.0).max(1e-6)
                }
            }
            None => kind_len.values().next().copied().unwrap_or(1.0).max(1e-6),
        };
        edge_rest[e_idx] = l0;
        len_sum += l0;
        len_cnt += 1;
        springs.push(Spring {
            u,
            v,
            l0,
            weight: 1.0,
            is_primary_bond: true,
        });
    }
    let a0 = if len_cnt > 0 {
        len_sum / (len_cnt as f64)
    } else {
        1.0
    };
    let kappa_eff = opts.kappa_b * a0 * a0;
    if !a0.is_finite()
        || !kappa_eff.is_finite()
        || edge_rest.iter().any(|l| !l.is_finite() || *l <= 0.0)
    {
        return unavailable_result("nonfinite physical rest lengths or bending scale".into());
    }
    let mut diagnostics = Vec::new();
    let warm_z = warm_z.filter(|wz| wz.iter().any(|z| z.abs() > 1e-4));

    let max_init_rel_strain = springs
        .iter()
        .map(|s| {
            let d = (crystal.pos[s.u][0] - crystal.pos[s.v][0])
                .hypot(crystal.pos[s.u][1] - crystal.pos[s.v][1]);
            (d - s.l0).abs() / s.l0
        })
        .fold(0.0_f64, f64::max);

    let has_valid_frames =
        !crystal.is_non_orientable && crystal.wall_sigma.is_none() && check_frames(crystal) == 0;
    // The drawing checker uses integer spatial bins. Do not send extreme finite coordinates through
    // those bins; numerical overflow in the objective is diagnosed independently below.
    let safe_chart_repair = crystal.pos.iter().flatten().all(|x| x.abs() < 1e12);
    let chart_2d: Vec<[f64; 2]> =
        if max_init_rel_strain < 1e-7 || crystal.is_non_orientable || !safe_chart_repair {
            crystal.pos.clone()
        } else {
            prepare_planar_chart(crystal, has_valid_frames)
        };
    if chart_2d.iter().flatten().any(|x| !x.is_finite()) {
        return unavailable_result("nonfinite planar chart".into());
    }
    let (faces, is_boundary) = if crystal.is_non_orientable {
        diagnostics.push("Möbius face/hinge topology unsupported: spring-only preview retaining every seam bond, not a physical elastic sheet".into());
        (Vec::new(), vec![true; n])
    } else {
        let (faces, boundary, notes) = extract_interior_faces(crystal, &chart_2d, has_valid_frames);
        diagnostics.extend(notes);
        (faces, boundary)
    };
    add_bulk_tile_chord_springs(crystal, &faces, has_valid_frames, &mut springs);
    if springs.iter().any(|s| !s.l0.is_finite() || s.l0 <= 0.0) {
        return unavailable_result("nonfinite tile rest geometry".into());
    }
    let hinges = build_hinges(&faces);

    let mut x_flat = vec![0.0_f64; 3 * n];
    for i in 0..n {
        x_flat[3 * i] = chart_2d[i][0];
        x_flat[3 * i + 1] = chart_2d[i][1];
    }
    let max_total_init_strain = springs
        .iter()
        .map(|s| (dist3_at(&x_flat, s.u, s.v) - s.l0).abs() / s.l0)
        .fold(0.0_f64, f64::max);

    let mut x_2d = vec![0.0_f64; 2 * n];
    for i in 0..n {
        x_2d[2 * i] = chart_2d[i][0];
        x_2d[2 * i + 1] = chart_2d[i][1];
    }
    let _ = lbfgs_minimize(&mut x_2d, 120, 1e-6, |pos2, g2| {
        eval_2d_stretch_fg(pos2, g2, &springs)
    });
    let (planar_2d_bond_energy, planar_2d_stretch_energy) = eval_2d_energies(&x_2d, &springs);
    let mut g2 = vec![0.0; 2 * n];
    eval_2d_stretch_fg(&x_2d, &mut g2, &springs);
    let planar_2d_converged = gradient_rms(&g2) <= 1e-6;
    if !planar_2d_converged {
        diagnostics.push("planar reference not converged; strain comparison is provisional".into());
    }

    let mut x3 = vec![0.0_f64; 3 * n];
    for i in 0..n {
        x3[3 * i] = x_2d[2 * i];
        x3[3 * i + 1] = x_2d[2 * i + 1];
    }
    if let Some(wz) = warm_z {
        for i in 0..n {
            x3[3 * i + 2] = wz[i];
        }
    } else if max_total_init_strain >= 1e-7 {
        seed_out_of_plane(
            &mut x3,
            crystal,
            has_valid_frames,
            a0,
            max_total_init_strain,
        );
    }

    let mut centroids_buf = vec![[0.0_f64; 3]; faces.len()];
    let mut centroid_grad = vec![[0.0_f64; 3]; faces.len()];
    let (_, iterations) = lbfgs_minimize(&mut x3, opts.max_iters, opts.grad_tol, |pos, grad| {
        eval_sheet_fg(
            pos,
            grad,
            &faces,
            &springs,
            &hinges,
            kappa_eff,
            &mut centroids_buf,
            &mut centroid_grad,
        )
    });
    let mut g3 = vec![0.0; 3 * n];
    let energy = eval_sheet_fg(
        &x3,
        &mut g3,
        &faces,
        &springs,
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

    let (bond_stretch_energy, stretch_energy, bend_energy) =
        eval_energies(&x3, &faces, &springs, &hinges, kappa_eff);
    let (bond_strain, face_strain) = compute_strains(&xyz, crystal, &edge_rest, &faces);
    let vertex_omega = compute_vertex_omega(&xyz, &faces, &is_boundary);

    let result = Sheet3DResult {
        xyz,
        faces,
        bond_strain,
        bond_rest_len: edge_rest,
        face_strain,
        vertex_omega,
        is_boundary,
        mean_bond_len: a0,
        bond_stretch_energy,
        stretch_energy,
        bend_energy,
        planar_2d_bond_energy,
        planar_2d_stretch_energy,
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

fn rest_lengths_from_cell(cell: &Cell2D) -> HashMap<usize, f64> {
    let mut sums: HashMap<usize, (f64, usize)> = HashMap::new();
    for b in &cell.bonds {
        if let (Some(&pi), Some(&pj)) = (cell.basis.get(b.i), cell.basis.get(b.j)) {
            let dx = f64::from(b.n1) * cell.a1[0] + f64::from(b.n2) * cell.a2[0] + pj[0] - pi[0];
            let dy = f64::from(b.n1) * cell.a1[1] + f64::from(b.n2) * cell.a2[1] + pj[1] - pi[1];
            let len = dx.hypot(dy);
            if len > 1e-9 {
                let ent = sums.entry(b.kind).or_insert((0.0, 0));
                ent.0 += len;
                ent.1 += 1;
            }
        }
    }
    sums.into_iter()
        .map(|(k, (s, c))| (k, s / (c as f64)))
        .collect()
}

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
    crystal: &BuiltCrystal,
    has_valid_frames: bool,
    a0: f64,
    max_strain: f64,
) {
    let n = crystal.pos.len();
    let inv_n = 1.0 / (n as f64);
    let cx = (0..n).map(|i| x3[3 * i]).sum::<f64>() * inv_n;
    let cy = (0..n).map(|i| x3[3 * i + 1]).sum::<f64>() * inv_n;
    let r_max = (0..n)
        .map(|i| (x3[3 * i] - cx).hypot(x3[3 * i + 1] - cy))
        .fold(a0, f64::max);

    let frank_deg = if has_valid_frames {
        total_frank(crystal)
    } else {
        0.0
    };
    let scale = (max_strain * 12.0).clamp(0.05, 1.0);

    for i in 0..n {
        let dx = x3[3 * i] - cx;
        let dy = x3[3 * i + 1] - cy;
        let r = dx.hypot(dy);
        let theta = dy.atan2(dx);
        let h = det_hash_unit(i as u64);
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
