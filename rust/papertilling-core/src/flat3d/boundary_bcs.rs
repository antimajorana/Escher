//! Per-direction boundary conditions (`Open`, `Periodic`, `Antiperiodic`,
//! `Twisted { phase_rad }`) on 2D and 3D supercell clusters with accumulated
//! Peierls phases for spin/electronic Hamiltonians.

use std::collections::BTreeSet;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::types::Cell2D;

use super::{Cell3D, frac_to_cart_3d};

/// Per-axis boundary condition for finite/periodic/twisted Hamiltonian lattices.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryCondition {
    Open,
    Periodic,
    Antiperiodic,
    Twisted { phase_rad: f64 },
}

impl BoundaryCondition {
    pub(crate) fn phase_step(self, wrap: i32) -> Option<f64> {
        match self {
            Self::Open if wrap != 0 => None,
            Self::Open | Self::Periodic => Some(0.0),
            Self::Antiperiodic => Some(f64::from(wrap) * PI),
            Self::Twisted { phase_rad } => Some(f64::from(wrap) * phase_rad),
        }
    }
}

/// Bond with accumulated boundary Peierls phase `phase_rad` in a 3D cluster.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryBond3d {
    pub u: usize,
    pub v: usize,
    pub kind: usize,
    pub wrap: [i32; 3],
    pub phase_rad: f64,
    pub weight_real: f64,
}

/// Finite/periodic/twisted 3D supercell lattice ready for Hamiltonian export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryLattice3d {
    pub positions: Vec<[f64; 3]>,
    pub site_kinds: Vec<u8>,
    pub bonds: Vec<BoundaryBond3d>,
    pub bcs: [BoundaryCondition; 3],
}

/// Apply per-direction `BoundaryCondition`s to an `(nx, ny, nz)` supercell of `cell`.
pub fn apply_boundary_conditions_3d(
    cell: &Cell3D,
    repeats: [usize; 3],
    bcs: [BoundaryCondition; 3],
) -> Result<BoundaryLattice3d, String> {
    if repeats.contains(&0) {
        return Err("all repeat counts must be >= 1".to_string());
    }
    let [nx, ny, nz] = repeats;
    let m = cell.basis.len();
    let sid = |b: usize, x: usize, y: usize, z: usize| ((b * nx + x) * ny + y) * nz + z;
    let mut positions = vec![[0.0_f64; 3]; m * nx * ny * nz];
    let mut site_kinds = vec![0_u8; m * nx * ny * nz];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let sh = frac_to_cart_3d(&cell.lattice, [ix as f64, iy as f64, iz as f64]);
                for (b, &pos) in cell.basis.iter().enumerate() {
                    let id = sid(b, ix, iy, iz);
                    positions[id] = [pos[0] + sh[0], pos[1] + sh[1], pos[2] + sh[2]];
                    site_kinds[id] = cell.site_kinds.get(b).copied().unwrap_or(0);
                }
            }
        }
    }
    let mut seen = BTreeSet::new();
    let mut bonds = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for bd in &cell.bonds {
                    let u = sid(bd.i, ix, iy, iz);
                    let t = [ix as i32 + bd.n1, iy as i32 + bd.n2, iz as i32 + bd.n3];
                    let w = [t[0].div_euclid(nx as i32), t[1].div_euclid(ny as i32), t[2].div_euclid(nz as i32)];
                    let Some(p0) = bcs[0].phase_step(w[0]) else { continue };
                    let Some(p1) = bcs[1].phase_step(w[1]) else { continue };
                    let Some(p2) = bcs[2].phase_step(w[2]) else { continue };
                    let v = sid(bd.j, t[0].rem_euclid(nx as i32) as usize, t[1].rem_euclid(ny as i32) as usize, t[2].rem_euclid(nz as i32) as usize);
                    let (a, b, wr, ph) = if (u, w) <= (v, [-w[0], -w[1], -w[2]]) {
                        (u, v, w, p0 + p1 + p2)
                    } else {
                        (v, u, [-w[0], -w[1], -w[2]], -(p0 + p1 + p2))
                    };
                    if seen.insert((a, b, wr)) {
                        bonds.push(BoundaryBond3d { u: a, v: b, kind: bd.kind, wrap: wr, phase_rad: ph, weight_real: ph.cos() });
                    }
                }
            }
        }
    }
    Ok(BoundaryLattice3d { positions, site_kinds, bonds, bcs })
}

/// Bond with accumulated boundary Peierls phase `phase_rad` in a 2D cluster.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryBond2d {
    pub u: usize,
    pub v: usize,
    pub kind: usize,
    pub wrap: [i32; 2],
    pub phase_rad: f64,
    pub weight_real: f64,
}

/// Finite/periodic/twisted 2D supercell lattice ready for Hamiltonian export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryLattice2d {
    pub positions: Vec<[f64; 2]>,
    pub bonds: Vec<BoundaryBond2d>,
    pub bcs: [BoundaryCondition; 2],
}

/// Apply per-direction `BoundaryCondition`s to an `(nx, ny)` supercell of `cell`.
pub fn apply_boundary_conditions_2d(
    cell: &Cell2D,
    repeats: [usize; 2],
    bcs: [BoundaryCondition; 2],
) -> Result<BoundaryLattice2d, String> {
    if repeats.contains(&0) {
        return Err("all repeat counts must be >= 1".to_string());
    }
    let [nx, ny] = repeats;
    let m = cell.basis.len();
    let sid = |b: usize, x: usize, y: usize| (b * nx + x) * ny + y;
    let mut positions = vec![[0.0_f64; 2]; m * nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            let sx = ix as f64 * cell.a1[0] + iy as f64 * cell.a2[0];
            let sy = ix as f64 * cell.a1[1] + iy as f64 * cell.a2[1];
            for (b, &p) in cell.basis.iter().enumerate() {
                positions[sid(b, ix, iy)] = [p[0] + sx, p[1] + sy];
            }
        }
    }
    let mut seen = BTreeSet::new();
    let mut bonds = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for bd in &cell.bonds {
                let u = sid(bd.i, ix, iy);
                let t = [ix as i32 + bd.n1, iy as i32 + bd.n2];
                let w = [t[0].div_euclid(nx as i32), t[1].div_euclid(ny as i32)];
                let Some(p0) = bcs[0].phase_step(w[0]) else { continue };
                let Some(p1) = bcs[1].phase_step(w[1]) else { continue };
                let v = sid(bd.j, t[0].rem_euclid(nx as i32) as usize, t[1].rem_euclid(ny as i32) as usize);
                let (a, b, wr, ph) = if (u, w) <= (v, [-w[0], -w[1]]) {
                    (u, v, w, p0 + p1)
                } else {
                    (v, u, [-w[0], -w[1]], -(p0 + p1))
                };
                if seen.insert((a, b, wr)) {
                    bonds.push(BoundaryBond2d { u: a, v: b, kind: bd.kind, wrap: wr, phase_rad: ph, weight_real: ph.cos() });
                }
            }
        }
    }
    Ok(BoundaryLattice2d { positions, bonds, bcs })
}
