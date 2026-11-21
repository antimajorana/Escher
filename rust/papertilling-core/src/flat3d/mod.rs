//! Stage 4 Core: Exact torsion-free space-group classification, 10 closed flat 3D
//! platycosms, flat 3D non-compact slabs/chimneys, chiral `(n, m)` nanotubes and
//! nanoribbons, symmetry-centred flakes, proper `det = +1` 3D Niggli reduction,
//! `CELLTRAN` supercells, strain tensors, and per-direction boundary conditions.

pub mod boundary_bcs;
pub mod flakes;
mod lattice_symmetry;
pub mod nanotubes_ribbons;
pub mod niggli_supercell;
mod platycosm_data;
pub mod platycosms;
pub mod slabs3d;
mod slabs3d_nonorient;
pub mod torsion_free;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::space3d::motifs::{NetMotif, net_motif};
use crate::space3d::nets3d::CanonicalNet3d;

pub use flakes::{SymmetricFlake, build_symmetric_flake, build_wigner_seitz_flake};
pub use nanotubes_ribbons::{
    NanoribbonPatch, NanotubeMesh, build_nanoribbon, build_nanotube, build_nanotube_opts,
    find_orthogonal_period,
};
pub use niggli_supercell::{
    BoundaryBond2d, BoundaryBond3d, BoundaryCondition, BoundaryLattice2d, BoundaryLattice3d,
    CellStrain3d, NiggliReduction3d, apply_boundary_conditions_2d, apply_boundary_conditions_3d,
    build_supercell_2d, build_supercell_3d, cell_strain_tensor, niggli_reduce_3d,
};
pub use platycosms::{
    PLATYCOSMS, PlatycosmCompatibility, PlatycosmInfo, PlatycosmQuotient3d, QuotientBond3d,
    QuotientVertex3d, build_platycosm_quotient, check_platycosm_compatibility, list_platycosms,
    lookup_platycosm,
};
pub use slabs3d::{FlatSlabKind, SlabPatch3d, build_flat_slab_3d};
pub use torsion_free::{
    FixedPointObstruction, TorsionFreeReport, analyze_torsion_free, classify_all_2d_torsion_free,
    classify_all_3d_torsion_free, is_torsion_free_conv_ops, is_torsion_free_exact,
};

/// Periodic 3D bond connecting basis site `i` in cell `(0,0,0)` to `j` in `(n1,n2,n3)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bond3D {
    pub i: usize,
    pub j: usize,
    pub n1: i32,
    pub n2: i32,
    pub n3: i32,
    pub kind: usize,
}

/// Canonical 3D periodic crystal cell with row-vector lattice `[a1, a2, a3]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cell3D {
    pub lattice: [[f64; 3]; 3],
    pub basis: Vec<[f64; 3]>,
    pub bonds: Vec<Bond3D>,
    #[serde(default)]
    pub space_group_num: Option<u16>,
    #[serde(default)]
    pub site_kinds: Vec<u8>,
}

/// Canonicalize undirected 3D bond orientation so `(i, j, n1, n2, n3) <= (j, i, -n1, -n2, -n3)`.
#[must_use]
pub fn canon_bond_3d(
    i: usize,
    j: usize,
    n1: i32,
    n2: i32,
    n3: i32,
    kind: usize,
) -> Bond3D {
    if (i, j, n1, n2, n3) <= (j, i, -n1, -n2, -n3) {
        Bond3D { i, j, n1, n2, n3, kind }
    } else {
        Bond3D { i: j, j: i, n1: -n1, n2: -n2, n3: -n3, kind }
    }
}

pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

pub(crate) fn det3_f64(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub(crate) fn inv3_f64(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], String> {
    let det = det3_f64(m);
    if !det.is_finite() || det.abs() < 1e-14 {
        return Err("singular or non-finite 3x3 matrix".to_string());
    }
    let inv = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

pub(crate) fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub(crate) fn frac_to_cart_3d(lat: &[[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    [
        f[0] * lat[0][0] + f[1] * lat[1][0] + f[2] * lat[2][0],
        f[0] * lat[0][1] + f[1] * lat[1][1] + f[2] * lat[2][1],
        f[0] * lat[0][2] + f[1] * lat[1][2] + f[2] * lat[2][2],
    ]
}

pub(crate) fn cart_to_frac_3d(lat: &[[f64; 3]; 3], p: [f64; 3]) -> Result<[f64; 3], String> {
    let a_inv = inv3_f64(transpose3(*lat))?;
    Ok([dot3(a_inv[0], p), dot3(a_inv[1], p), dot3(a_inv[2], p)])
}

/// Extract lattice lengths `(a, b, c)` and interaxial angles `(alpha, beta, gamma)` in degrees.
#[must_use]
pub fn lattice_parameters_3d(lat: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
    let a = norm3(lat[0]);
    let b = norm3(lat[1]);
    let c = norm3(lat[2]);
    let ang = |u: [f64; 3], lu: f64, v: [f64; 3], lv: f64| -> f64 {
        if lu < 1e-15 || lv < 1e-15 {
            90.0
        } else {
            (dot3(u, v) / (lu * lv)).clamp(-1.0, 1.0).acos().to_degrees()
        }
    };
    let alpha = ang(lat[1], b, lat[2], c);
    let beta = ang(lat[0], a, lat[2], c);
    let gamma = ang(lat[0], a, lat[1], b);
    ([a, b, c], [alpha, beta, gamma])
}

impl Cell3D {
    /// Construct a `Cell3D` from fractional site positions, periodic bonds, and lattice rows.
    #[must_use]
    pub fn from_fractional(
        lattice: [[f64; 3]; 3],
        frac_sites: &[([f64; 3], u8)],
        bonds: Vec<Bond3D>,
        space_group_num: Option<u16>,
    ) -> Self {
        let basis = frac_sites
            .iter()
            .map(|&(f, _)| frac_to_cart_3d(&lattice, f))
            .collect();
        let site_kinds = frac_sites.iter().map(|&(_, k)| k).collect();
        Self { lattice, basis, bonds, space_group_num, site_kinds }
    }

    /// Build a periodic `Cell3D` matching any of the 10 canonical nets (motif table: `space3d::motifs`).
    #[must_use]
    pub fn from_canonical_net(net: CanonicalNet3d) -> Self {
        let NetMotif { lattice: lat, sites: motif, bond_cutoff: cutoff } = net_motif(net);
        let c2 = cutoff * cutoff;
        let mut seen = BTreeSet::new();
        for (i, &(fi, _)) in motif.iter().enumerate() {
            let pi = frac_to_cart_3d(&lat, fi);
            for (j, &(fj0, _)) in motif.iter().enumerate().skip(i) {
                for n1 in -2_i32..=2 {
                    for n2 in -2_i32..=2 {
                        for n3 in -2_i32..=2 {
                            if i == j && n1 == 0 && n2 == 0 && n3 == 0 {
                                continue;
                            }
                            let fj = [fj0[0] + f64::from(n1), fj0[1] + f64::from(n2), fj0[2] + f64::from(n3)];
                            let pj = frac_to_cart_3d(&lat, fj);
                            let dx = pj[0] - pi[0];
                            let dy = pj[1] - pi[1];
                            let dz = pj[2] - pi[2];
                            let d2 = dx * dx + dy * dy + dz * dz;
                            if d2 > 1e-4 && d2 <= c2 {
                                let cb = canon_bond_3d(i, j, n1, n2, n3, 0);
                                seen.insert((cb.i, cb.j, cb.n1, cb.n2, cb.n3, cb.kind));
                            }
                        }
                    }
                }
            }
        }
        let bonds = seen
            .into_iter()
            .map(|(i, j, n1, n2, n3, kind)| Bond3D { i, j, n1, n2, n3, kind })
            .collect();
        Self::from_fractional(lat, &motif, bonds, Some(net.space_group_num()))
    }
}

/// Parse a `CanonicalNet3d` variant name (`"SimpleCubic"`, `"BetaQuartz"`, etc.).
///
/// # Errors
/// Returns an error for an unknown net name.
pub fn parse_canonical_net_3d(name: &str) -> Result<CanonicalNet3d, String> {
    let norm: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    match norm.as_str() {
        "simplecubic" | "sc" => Ok(CanonicalNet3d::SimpleCubic),
        "bcc" => Ok(CanonicalNet3d::BCC),
        "fcc" => Ok(CanonicalNet3d::FCC),
        "diamond" => Ok(CanonicalNet3d::Diamond),
        "hcp" => Ok(CanonicalNet3d::HCP),
        "tetragonal123" | "tetragonal" => Ok(CanonicalNet3d::Tetragonal123),
        "rutile136" | "rutile" => Ok(CanonicalNet3d::Rutile136),
        "perovskite" => Ok(CanonicalNet3d::Perovskite),
        "pyrochlore" => Ok(CanonicalNet3d::Pyrochlore),
        "betaquartz" | "quartz" => Ok(CanonicalNet3d::BetaQuartz),
        _ => Err(format!("unknown canonical 3D net {name:?}")),
    }
}
