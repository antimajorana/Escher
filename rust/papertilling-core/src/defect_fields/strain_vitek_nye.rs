//! Local deformation gradient and strains, Vitek differential displacements, and the discrete Nye tensor
//! (`#040`, `#043`, `#044`).
//!
//! - `F_i` maps reference bond vectors to current ones, `dx ~= F_i dX`, by least squares over the bonds of site `i`:
//!   `F_i = (sum dx dX^T)(sum dX dX^T)^-1`.
//! - Linear strain `eps_i = (F_i + F_i^T)/2 - I`; Green-Lagrange `E_i = (F_i^T F_i - I)/2`; displacement `u_i = x_i - X_i`.
//! - Vitek arrow of bond `(i, j)`: `du_ij = u_j - u_i`; its component along `b` wrapped into `(-|b|/2, |b|/2]`.
//! - Nye: `alpha_{z beta}(f) = (1/A_f) sum_{e in boundary f, CCW} G_{e, beta}` with snapped, exactly antisymmetric `G`,
//!   so `sum_f alpha(f) A_f` over a region equals the closure of its boundary (`burgers_circuit` sign). This is the
//!   discrete form of Nye's `int_S alpha_{ij} n_i dS = B_j` (Nye 1953, Acta Metall. 1, 153; lattice-correspondence
//!   version Hartley & Mishin 2005, Acta Mater. 53, 1313) with `alpha_{ij}` = line direction `i`, Burgers component `j`.
//!   With the FS/RH Burgers vector of Hirth & Lothe (line direction `+z`) the Nye tensor is `-alpha` (see `nye_loops`).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::nye_loops::{default_cutoff, fit_frame_angle, g_dir, region_boundary_walk, region_frank, snap_edges};
use super::topo_detect::topological_charges;
use super::{BondGraph, PlanarFaces, RefLattice, norm, sub};
use crate::types::{Cell2D, Frame2D};

/// Per-site deformation data; `f` is `None` when the site's reference bonds do not span the plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalStrain {
    pub u: [f64; 2],
    pub f: Option<[[f64; 2]; 2]>,
    pub eps: Option<[[f64; 2]; 2]>,
    pub green: Option<[[f64; 2]; 2]>,
}

/// Least-squares `F_i`, linear and Green-Lagrange strain at every site from reference / current positions and bonds.
pub fn local_deformation_and_strain(ref_pos: &[[f64; 2]], pos: &[[f64; 2]], edges: &[(usize, usize)]) -> Result<Vec<LocalStrain>, String> {
    if ref_pos.len() != pos.len() {
        return Err(format!("ref_pos has {} sites, pos has {}", ref_pos.len(), pos.len()));
    }
    let n = pos.len();
    let (mut a, mut m) = (vec![[[0.0; 2]; 2]; n], vec![[[0.0; 2]; 2]; n]);
    for &(i, j) in edges {
        if i >= n || j >= n {
            return Err(format!("edge ({i}, {j}) out of range"));
        }
        let (dx, dxr) = (sub(pos[j], pos[i]), sub(ref_pos[j], ref_pos[i]));
        for s in [i, j] {
            for r in 0..2 {
                for c in 0..2 {
                    a[s][r][c] += dx[r] * dxr[c];
                    m[s][r][c] += dxr[r] * dxr[c];
                }
            }
        }
    }
    Ok((0..n).map(|s| {
        let u = sub(pos[s], ref_pos[s]);
        let mm = m[s];
        let det = mm[0][0] * mm[1][1] - mm[0][1] * mm[1][0];
        let scale = (mm[0][0] + mm[1][1]).max(1e-300);
        if det.abs() <= 1e-10 * scale * scale {
            return LocalStrain { u, f: None, eps: None, green: None };
        }
        let inv = [[mm[1][1] / det, -mm[0][1] / det], [-mm[1][0] / det, mm[0][0] / det]];
        let mut f = [[0.0; 2]; 2];
        for (r, row) in f.iter_mut().enumerate() {
            for (c, x) in row.iter_mut().enumerate() {
                *x = a[s][r][0] * inv[0][c] + a[s][r][1] * inv[1][c];
            }
        }
        let mut eps = [[0.0; 2]; 2];
        let mut green = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                let id = if r == c { 1.0 } else { 0.0 };
                eps[r][c] = 0.5 * (f[r][c] + f[c][r]) - id;
                green[r][c] = 0.5 * (f[0][r] * f[0][c] + f[1][r] * f[1][c] - id);
            }
        }
        LocalStrain { u, f: Some(f), eps: Some(eps), green: Some(green) }
    }).collect())
}

/// One Vitek differential-displacement arrow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VitekArrow {
    pub i: usize,
    pub j: usize,
    pub du: [f64; 2],
    /// `du . b_hat`.
    pub along_b: f64,
    /// `along_b` wrapped into `(-|b|/2, |b|/2]`.
    pub wrapped: f64,
}

/// Wrap `x` into `(-p/2, p/2]`.
pub fn wrap_period(x: f64, p: f64) -> f64 {
    x - p * (x / p - 0.5).ceil()
}

/// Vitek differential displacements `u_j - u_i` of every bond, component along `b` wrapped modulo `|b|`.
pub fn vitek_differential(ref_pos: &[[f64; 2]], pos: &[[f64; 2]], edges: &[(usize, usize)], b: [f64; 2]) -> Result<Vec<VitekArrow>, String> {
    let lb = norm(b);
    if lb < 1e-12 {
        return Err("Vitek map needs a non-zero period vector b".to_string());
    }
    if ref_pos.len() != pos.len() {
        return Err("ref_pos and pos differ in length".to_string());
    }
    let bh = [b[0] / lb, b[1] / lb];
    Ok(edges.iter().map(|&(i, j)| {
        let du = sub(sub(pos[j], ref_pos[j]), sub(pos[i], ref_pos[i]));
        let along_b = du[0] * bh[0] + du[1] * bh[1];
        VitekArrow { i, j, du, along_b, wrapped: wrap_period(along_b, lb) }
    }).collect())
}

/// Discrete Nye tensor on the bounded faces and its integral over a face region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NyeField {
    pub faces: Vec<Vec<usize>>,
    pub areas: Vec<f64>,
    /// `alpha_{z1}, alpha_{z2}` per face.
    pub alpha: Vec<[f64; 2]>,
    /// Frame angle (degrees) fitted on the outer boundary ring.
    pub frame_angle_deg: f64,
    pub region: Vec<usize>,
    /// Counter-clockwise boundary walk of the region.
    pub boundary_walk: Vec<usize>,
    pub net_frank_deg: f64,
    /// `sum_f alpha(f) A_f` over the region (`None` when the net Frank angle is non-zero).
    pub total: Option<[f64; 2]>,
    /// `sum G` along `boundary_walk` (same snapping), for the telescoping check.
    pub boundary_closure: [f64; 2],
    pub reason: String,
}

/// Nye tensor from coordinates in one far-field frame (fitted on the outer boundary ring). `region = None` integrates
/// over all bounded faces; the total is refused when the region encloses a net Frank angle.
pub fn nye_tensor(
    pos: &[[f64; 2]], cell: &Cell2D, frame: Frame2D, edges: Option<&[(usize, usize)]>, cutoff: Option<f64>, region: Option<&[usize]>,
) -> Result<NyeField, String> {
    let rl = RefLattice::new(cell, frame);
    let g = match edges {
        Some(e) => BondGraph::from_edges(pos, e),
        None => BondGraph::from_cutoff(pos, cutoff.unwrap_or_else(|| default_cutoff(cell, &rl))),
    };
    let pf = PlanarFaces::new(&g);
    if pf.faces.is_empty() {
        return Err("REFUSED: no bounded faces (the bond graph encloses no area)".to_string());
    }
    let rim_edges: Vec<(usize, usize)> = g.edges.iter().copied().filter(|&(u, v)| pf.rim[u] && pf.rim[v]).collect();
    let th = fit_frame_angle(&g, &rl, &rim_edges, rl.frame_angle);
    let snap = snap_edges(&g, &rl, th);
    let alpha: Vec<[f64; 2]> = pf.faces.iter().zip(&pf.areas).map(|(f, &a)| {
        let mut s = [0.0, 0.0];
        for k in 0..f.len() {
            let e = g_dir(&snap, f[k], f[(k + 1) % f.len()]);
            s = [s[0] + e[0], s[1] + e[1]];
        }
        [s[0] / a, s[1] / a]
    }).collect();
    let reg: BTreeSet<usize> = match region {
        Some(r) => {
            if let Some(bad) = r.iter().find(|&&f| f >= pf.faces.len()) {
                return Err(format!("REFUSED: region face {bad} does not exist ({} bounded faces)", pf.faces.len()));
            }
            r.iter().copied().collect()
        }
        None => (0..pf.faces.len()).collect(),
    };
    let walk = region_boundary_walk(&pf, &reg);
    let tc = topological_charges(&g, &pf, &rl);
    let net = region_frank(&pf, &reg, &walk, &tc.vertex_frank, &tc.face_frank);
    let mut closure = [0.0, 0.0];
    for w in walk.windows(2) {
        let e = g_dir(&snap, w[0], w[1]);
        closure = [closure[0] + e[0], closure[1] + e[1]];
    }
    let (total, reason) = if net == 0.0 {
        let mut t = [0.0, 0.0];
        for &f in &reg {
            t = [t[0] + alpha[f][0] * pf.areas[f], t[1] + alpha[f][1] * pf.areas[f]];
        }
        (Some(t), "[PROVED] antisymmetric snapping telescopes: sum_f alpha A_f = boundary closure".to_string())
    } else {
        (None, format!("[PROVED] region encloses net Frank angle {net} deg: the Burgers vector depends on the frame"))
    };
    Ok(NyeField {
        faces: pf.faces, areas: pf.areas, alpha, frame_angle_deg: th.to_degrees(), region: reg.into_iter().collect(),
        boundary_walk: walk, net_frank_deg: net, total, boundary_closure: closure, reason,
    })
}
