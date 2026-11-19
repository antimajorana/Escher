//! Flat 3D non-compact manifold builders: `TorusSlab` (`T^2 x R`),
//! `KleinSlab` (`K^2 x R`), `MoebiusTube` (`Moebius x S^1`),
//! `CylinderCrossS1` (`Cylinder x S^1`), and `CylinderChimney` (`S^1 x R^2`).

use std::collections::BTreeSet;
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use super::slabs3d_nonorient::{build_non_orientable_slab, find_inplane_glide};
use super::{Cell3D, cart_to_frac_3d, dot3, frac_to_cart_3d, norm3};

/// Non-compact flat 3D manifold topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlatSlabKind {
    TorusSlab,
    KleinSlab,
    MoebiusTube,
    CylinderCrossS1,
    CylinderChimney,
}

impl FlatSlabKind {
    /// Parse a slab topology name.
    pub fn from_name(name: &str) -> Result<Self, String> {
        let norm: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).flat_map(char::to_lowercase).collect();
        match norm.as_str() {
            "torusslab" | "t2xr" | "torus" => Ok(Self::TorusSlab),
            "kleinslab" | "k2xr" | "klein" => Ok(Self::KleinSlab),
            "moebiustube" | "mobiustube" | "moebiusxs1" | "mobiusxs1" => Ok(Self::MoebiusTube),
            "cylindercrosss1" | "cylinderxs1" | "annulusxs1" => Ok(Self::CylinderCrossS1),
            "cylinderchimney" | "chimney" | "s1xr2" => Ok(Self::CylinderChimney),
            _ => Err(format!("unknown flat 3D slab kind {name:?}")),
        }
    }
}

/// Finite 3D crystal patch realized on a flat 3D non-compact manifold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlabPatch3d {
    pub kind: FlatSlabKind,
    pub status: String,
    pub reason: String,
    pub manifold_dim: usize,
    pub compact_dims: usize,
    pub noncompact_dims: usize,
    pub orientable: bool,
    pub h1_label: String,
    pub b1: usize,
    pub repeats: [usize; 3],
    pub positions: Vec<[f64; 3]>,
    pub flat_positions: Vec<[f64; 3]>,
    pub bonds: Vec<(usize, usize, bool, usize)>,
    pub site_kinds: Vec<u8>,
    pub boundary_sites: Vec<usize>,
    pub periodic_vectors: Vec<[f64; 3]>,
}

pub(crate) fn slab_meta(kind: FlatSlabKind) -> (usize, usize, bool, &'static str, usize) {
    match kind {
        FlatSlabKind::TorusSlab => (2, 1, true, "Z^2", 2),
        FlatSlabKind::KleinSlab => (2, 1, false, "Z_2 x Z", 1),
        FlatSlabKind::MoebiusTube => (2, 1, false, "Z^2", 2),
        FlatSlabKind::CylinderCrossS1 => (2, 1, true, "Z^2", 2),
        FlatSlabKind::CylinderChimney => (1, 2, true, "Z", 1),
    }
}

fn empty_impossible(kind: FlatSlabKind, repeats: [usize; 3], reason: String) -> SlabPatch3d {
    let (cd, nd, orientable, h1, b1) = slab_meta(kind);
    SlabPatch3d {
        kind,
        status: "IMPOSSIBLE".to_string(),
        reason,
        manifold_dim: 3,
        compact_dims: cd,
        noncompact_dims: nd,
        orientable,
        h1_label: h1.to_string(),
        b1,
        repeats,
        positions: Vec::new(),
        flat_positions: Vec::new(),
        bonds: Vec::new(),
        site_kinds: Vec::new(),
        boundary_sites: Vec::new(),
        periodic_vectors: Vec::new(),
    }
}

/// Build a 3D crystal patch on one of the 5 flat 3D non-compact manifold topologies.
///
/// `H_1` of each (`slab_meta`): `T^2 x R` -> `Z^2`; `K x R` -> `Z + Z_2` (Klein bottle `K`);
/// `Moebius x S^1` -> `Z^2` (the Moebius band retracts to `S^1`); `Cylinder x S^1` -> `Z^2`;
/// `S^1 x R^2` -> `Z` (homotopy invariance and cellular homology, e.g. Hatcher, *Algebraic Topology*, Section 2.2).
///
/// # Errors
/// Returns an error for zero repeats, an empty basis, or a singular lattice.
pub fn build_flat_slab_3d(
    cell: &Cell3D,
    kind: FlatSlabKind,
    repeats: [usize; 3],
) -> Result<SlabPatch3d, String> {
    if repeats.contains(&0) {
        return Err("all slab repeat counts must be >= 1".to_string());
    }
    if cell.basis.is_empty() {
        return Err("cell has no basis sites".to_string());
    }
    let frac_basis: Vec<[f64; 3]> = cell.basis.iter().map(|&p| cart_to_frac_3d(&cell.lattice, p)).collect::<Result<_, _>>()?;
    if matches!(kind, FlatSlabKind::KleinSlab | FlatSlabKind::MoebiusTube) {
        let Some(glide) = find_inplane_glide(cell, &frac_basis) else {
            return Ok(empty_impossible(
                kind,
                repeats,
                format!(
                    "IMPOSSIBLE [PROVED]: {kind:?} is glued by a z-preserving in-plane mirror/glide (x, y, z) -> (M(x, y) + t, z) of the crystal; exhaustive search over M with entries in [-2, 2] in the given a1, a2 basis (a3 fixed) finds none for this cell setting"
                ),
            ));
        };
        return Ok(build_non_orientable_slab(cell, &frac_basis, kind, repeats, &glide));
    }

    let [l1, l2, l3] = repeats;
    let m = frac_basis.len();
    let sid = |b: usize, x: usize, y: usize, z: usize| ((b * l1 + x) * l2 + y) * l3 + z;
    let n_sites = m * l1 * l2 * l3;
    let mut flat_positions = vec![[0.0_f64; 3]; n_sites];
    let mut site_kinds = vec![0_u8; n_sites];
    let (mid_y, mid_z) = (0.5 * l2 as f64, 0.5 * l3 as f64);
    for ix in 0..l1 {
        for iy in 0..l2 {
            for iz in 0..l3 {
                for (b, fb) in frac_basis.iter().enumerate() {
                    let u = sid(b, ix, iy, iz);
                    let fy = if matches!(kind, FlatSlabKind::CylinderCrossS1 | FlatSlabKind::CylinderChimney) { fb[1] + iy as f64 - mid_y } else { fb[1] + iy as f64 };
                    let fz = if matches!(kind, FlatSlabKind::TorusSlab | FlatSlabKind::CylinderChimney) { fb[2] + iz as f64 - mid_z } else { fb[2] + iz as f64 };
                    flat_positions[u] = frac_to_cart_3d(&cell.lattice, [fb[0] + ix as f64, fy, fz]);
                    site_kinds[u] = cell.site_kinds.get(b).copied().unwrap_or(0);
                }
            }
        }
    }

    let n1 = norm3(cell.lattice[0]).max(1e-12);
    let e1 = [cell.lattice[0][0] / n1, cell.lattice[0][1] / n1, cell.lattice[0][2] / n1];
    let c1 = (l1 as f64) * n1;
    let r_cyl = (c1 / TAU).max(0.5);
    let a2p = [0, 1, 2].map(|k| cell.lattice[1][k] - dot3(cell.lattice[1], e1) * e1[k]);
    let n2p = norm3(a2p).max(1e-12);
    let e2 = [a2p[0] / n2p, a2p[1] / n2p, a2p[2] / n2p];
    let e3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let positions: Vec<[f64; 3]> = flat_positions
        .iter()
        .map(|&p| match kind {
            FlatSlabKind::TorusSlab => p,
            FlatSlabKind::CylinderCrossS1 | FlatSlabKind::CylinderChimney => {
                let th = (dot3(p, e1) / c1) * TAU;
                let rad = (r_cyl + dot3(p, e2)).max(0.15);
                [rad * th.cos(), rad * th.sin(), dot3(p, e3)]
            }
            _ => unreachable!(),
        })
        .collect();

    let wrap_coord = |val: i32, len: usize, periodic: bool| -> Option<(usize, bool)> {
        if periodic { Some((val.rem_euclid(len as i32) as usize, val < 0 || val >= len as i32)) } else if val < 0 || val >= len as i32 { None } else { Some((val as usize, false)) }
    };
    let (px, py, pz) = match kind {
        FlatSlabKind::TorusSlab => (true, true, false),
        FlatSlabKind::CylinderCrossS1 => (true, false, true),
        FlatSlabKind::CylinderChimney => (true, false, false),
        _ => unreachable!(),
    };

    let mut seen_bonds = BTreeSet::new();
    let mut bonds = Vec::new();
    let mut deg = vec![0usize; n_sites];
    for ix in 0..l1 {
        for iy in 0..l2 {
            for iz in 0..l3 {
                for bd in &cell.bonds {
                    let u = sid(bd.i, ix, iy, iz);
                    let Some((vx, sx)) = wrap_coord(ix as i32 + bd.n1, l1, px) else { continue };
                    let Some((vy, sy)) = wrap_coord(iy as i32 + bd.n2, l2, py) else { continue };
                    let Some((vz, sz)) = wrap_coord(iz as i32 + bd.n3, l3, pz) else { continue };
                    let v = sid(bd.j, vx, vy, vz);
                    if u != v {
                        let pair = (u.min(v), u.max(v));
                        if seen_bonds.insert(pair) {
                            bonds.push((pair.0, pair.1, sx || sy || sz, bd.kind));
                            deg[pair.0] += 1;
                            deg[pair.1] += 1;
                        }
                    }
                }
            }
        }
    }
    let max_d = deg.iter().copied().max().unwrap_or(0);
    let boundary_sites = (0..n_sites).filter(|&u| deg[u] < max_d).collect();
    let (cd, nd, orientable, h1, b1) = slab_meta(kind);
    let mut periodic_vectors = Vec::new();
    for (flag, row, len) in [(px, cell.lattice[0], l1), (py, cell.lattice[1], l2), (pz, cell.lattice[2], l3)] {
        if flag {
            periodic_vectors.push([row[0] * len as f64, row[1] * len as f64, row[2] * len as f64]);
        }
    }
    Ok(SlabPatch3d {
        kind,
        status: "OK".to_string(),
        reason: format!("OK: {kind:?} with {n_sites} sites, {} bonds", bonds.len()),
        manifold_dim: 3,
        compact_dims: cd,
        noncompact_dims: nd,
        orientable,
        h1_label: h1.to_string(),
        b1,
        repeats,
        positions,
        flat_positions,
        bonds,
        site_kinds,
        boundary_sites,
        periodic_vectors,
    })
}
