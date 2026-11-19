//! Solver export (Section D): extended XYZ writers and exact neighbour-shell / space-group bond-orbit
//! classification of a periodic cell.
//!
//! Conventions shared with the pure-Python fallback `papertilling/solver_export.py` (keep them in step):
//! * an unordered periodic bond `{(i, 0), (j, n)}` is stored canonically as `min((i, j, n1, n2), (j, i, -n1, -n2))`;
//! * shells are distance classes of all site pairs (bonded or not) with `|d - d_s| <= 1e-5 * d_1`;
//! * the symmetry used is the bond-preserving symmetry of the cell (`Cell2D::symmetry_frac_ops(true)`), which is the
//!   wallpaper group of the bonded lattice (it can be larger than the family group on a special locus);
//! * orbits inside a shell are listed in increasing order of their smallest canonical bond.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::defect::point_defects::default_site_species;
use crate::types::{BuiltCrystal, Cell2D, FracOp2D, SurfaceMesh};

/// Vacuum padding (Angstrom-like units) added to the bounding box of open clusters.
pub const OPEN_BOX_PADDING: f64 = 10.0;
/// Out-of-plane lattice vector length used for periodic 2D cells.
pub const SLAB_C: f64 = 20.0;
const PROPS: &str = "Properties=species:S:1:pos:R:3:sublattice:I:1:coordination:I:1";

/// `{:.8}` with `-0.0` (and anything rounding to `-0.00000000`) printed as `0.00000000`.
pub fn fmt_f8(x: f64) -> String {
    let v = if x.abs() < 5e-13 { 0.0 } else { x };
    let s = format!("{v:.8}");
    if s == "-0.00000000" { "0.00000000".into() } else { s }
}

fn write_xyz(lattice: [f64; 9], pbc: &str, rows: &[(String, [f64; 3], usize, usize)]) -> String {
    let mut out = format!("{}\n", rows.len());
    let lat: Vec<String> = lattice.iter().map(|&x| fmt_f8(x)).collect();
    out.push_str(&format!("Lattice=\"{}\" {PROPS} pbc=\"{pbc}\"\n", lat.join(" ")));
    for (sp, p, sub, coord) in rows {
        out.push_str(&format!("{sp} {} {} {} {sub} {coord}\n", fmt_f8(p[0]), fmt_f8(p[1]), fmt_f8(p[2])));
    }
    out
}

fn pick_species(n: usize, labels: &[usize], species: Option<&[String]>) -> Result<Vec<String>, String> {
    match species {
        Some(s) if s.len() != n => Err(format!("species has {} entries, expected {n}", s.len())),
        Some(s) => {
            for sym in s {
                crate::defect::point_defects::validate_element_symbol(sym)?;
            }
            Ok(s.to_vec())
        }
        None => Ok(default_site_species(labels)),
    }
}

fn open_box(pts: &[[f64; 3]]) -> [f64; 9] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let ext = |k: usize| if pts.is_empty() { OPEN_BOX_PADDING } else { hi[k] - lo[k] + OPEN_BOX_PADDING };
    [ext(0), 0.0, 0.0, 0.0, ext(1), 0.0, 0.0, 0.0, ext(2)]
}

/// Number of bonds at each basis site of a periodic cell (a self-bond `(i, i, n)` counts twice).
pub fn cell_coordination(cell: &Cell2D) -> Vec<usize> {
    let mut c = vec![0usize; cell.basis.len()];
    for b in &cell.bonds {
        c[b.i] += 1;
        c[b.j] += 1;
    }
    c
}

/// Extended XYZ of the periodic `nx x ny` supercell of `cell` (`pbc="T T F"`, `c = 20`).
/// Site order: `index = (iy * nx + ix) * n_basis + s`. `species` (one per basis site) defaults to
/// [`default_site_species`] of the basis index.
pub fn export_cell_extxyz(cell: &Cell2D, nx: usize, ny: usize, species: Option<&[String]>) -> Result<String, String> {
    if nx == 0 || ny == 0 {
        return Err("supercell sizes nx, ny must be >= 1".into());
    }
    let nb = cell.basis.len();
    if cell.bonds.iter().any(|b| b.i >= nb || b.j >= nb) {
        return Err("cell bond site index out of range".into());
    }
    let labels: Vec<usize> = (0..nb).collect();
    let sp = pick_species(nb, &labels, species)?;
    let coord = cell_coordination(cell);
    let mut rows = Vec::with_capacity(nx * ny * nb);
    for iy in 0..ny {
        for ix in 0..nx {
            for s in 0..nb {
                let (fx, fy) = (ix as f64, iy as f64);
                let p = cell.basis[s];
                let x = p[0] + fx * cell.a1[0] + fy * cell.a2[0];
                let y = p[1] + fx * cell.a1[1] + fy * cell.a2[1];
                rows.push((sp[s].clone(), [x, y, 0.0], s, coord[s]));
            }
        }
    }
    let (sx, sy) = (nx as f64, ny as f64);
    let lat = [sx * cell.a1[0], sx * cell.a1[1], 0.0, sy * cell.a2[0], sy * cell.a2[1], 0.0, 0.0, 0.0, SLAB_C];
    Ok(write_xyz(lat, "T T F", &rows))
}

/// Extended XYZ of an open crystal patch (`pbc="F F F"`, box = bounding box + 10 on each axis).
/// `z` defaults to 0 and `species` to [`default_site_species`] of `basis_idx`; coordination = `adj[i].len()`.
pub fn export_crystal_extxyz(crystal: &BuiltCrystal, z: Option<&[f64]>, species: Option<&[String]>) -> Result<String, String> {
    let n = crystal.pos.len();
    if crystal.basis_idx.len() != n || crystal.adj.len() != n {
        return Err("inconsistent BuiltCrystal: pos / basis_idx / adj lengths differ".into());
    }
    if let Some(zz) = z && zz.len() != n {
        return Err(format!("z has {} entries, expected {n}", zz.len()));
    }
    let sp = pick_species(n, &crystal.basis_idx, species)?;
    let pts: Vec<[f64; 3]> = (0..n).map(|i| [crystal.pos[i][0], crystal.pos[i][1], z.map_or(0.0, |zz| zz[i])]).collect();
    let rows: Vec<_> = (0..n).map(|i| (sp[i].clone(), pts[i], crystal.basis_idx[i], crystal.adj[i].len())).collect();
    Ok(write_xyz(open_box(&pts), "F F F", &rows))
}

/// Extended XYZ of a 3D surface mesh (`xyz`, bonds) as an open cluster (`pbc="F F F"`).
/// Sublattice = first entry of `mesh.sites[i]`; coordination counts mesh bonds (a loop counts twice).
pub fn export_surface_extxyz(mesh: &SurfaceMesh, species: Option<&[String]>) -> Result<String, String> {
    let n = mesh.xyz.len();
    let labels: Vec<usize> = (0..n).map(|i| mesh.sites.get(i).map_or(0, |s| s.0)).collect();
    let sp = pick_species(n, &labels, species)?;
    let mut coord = vec![0usize; n];
    for &(u, v, _) in &mesh.bonds {
        if u < n && v < n {
            coord[u] += 1;
            coord[v] += 1;
        }
    }
    let rows: Vec<_> = (0..n).map(|i| (sp[i].clone(), mesh.xyz[i], labels[i], coord[i])).collect();
    Ok(write_xyz(open_box(&mesh.xyz), "F F F", &rows))
}

/// Canonical unordered periodic bond `{(i, 0), (j, (n1, n2))}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PeriodicBond {
    pub i: usize,
    pub j: usize,
    pub n1: i32,
    pub n2: i32,
}

impl PeriodicBond {
    pub fn canonical(i: usize, j: usize, n1: i32, n2: i32) -> Self {
        if (i, j, n1, n2) <= (j, i, -n1, -n2) { Self { i, j, n1, n2 } } else { Self { i: j, j: i, n1: -n1, n2: -n2 } }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellOrbit {
    pub representative: PeriodicBond,
    pub bonds_per_cell: usize,
    pub coordination_by_basis: Vec<usize>,
    pub mean_coordination: f64,
    pub bonds: Vec<PeriodicBond>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellInfo {
    pub shell: usize,
    pub distance: f64,
    pub total_directed_per_cell: usize,
    pub coordination_by_basis: Vec<usize>,
    pub mean_coordination: f64,
    pub orbits: Vec<ShellOrbit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellClassification {
    pub group: String,
    pub n_basis: usize,
    pub n_symmetry_ops: usize,
    pub shells: Vec<ShellInfo>,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub provenance: crate::defect::burgers_indices::StaticStr,
}

/// Fractional coordinates (unreduced, matching `Cell2D::symmetry_frac_ops`).
pub(crate) fn frac_sites(cell: &Cell2D) -> Result<Vec<[f64; 2]>, String> {
    let (a, b, c, d) = (cell.a1[0], cell.a2[0], cell.a1[1], cell.a2[1]);
    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        return Err("singular cell".into());
    }
    Ok(cell.basis.iter().map(|p| [(d * p[0] - b * p[1]) / det, (-c * p[0] + a * p[1]) / det]).collect())
}

/// Site permutation `perm[s]` and integer cell shift `shift[s]` of a fractional op: `op(f_s) = f_perm[s] + shift[s]`.
pub(crate) fn op_site_action(op: &FracOp2D, fr: &[[f64; 2]]) -> Option<(Vec<usize>, Vec<[i32; 2]>)> {
    let mut perm = Vec::with_capacity(fr.len());
    let mut shift = Vec::with_capacity(fr.len());
    for f in fr {
        let m = op.mf;
        let g = [
            f64::from(m[0][0]) * f[0] + f64::from(m[0][1]) * f[1] + op.t[0],
            f64::from(m[1][0]) * f[0] + f64::from(m[1][1]) * f[1] + op.t[1],
        ];
        let hit = fr.iter().enumerate().find_map(|(j, fj)| {
            let (dx, dy) = (g[0] - fj[0], g[1] - fj[1]);
            let (rx, ry) = (dx.round(), dy.round());
            ((dx - rx).abs() < 1e-6 && (dy - ry).abs() < 1e-6).then_some((j, [rx as i32, ry as i32]))
        })?;
        perm.push(hit.0);
        shift.push(hit.1);
    }
    Some((perm, shift))
}

/// Bond-preserving symmetry operation paired with its basis permutation and cell shifts.
pub(crate) type CellOpAction = (FracOp2D, Vec<usize>, Vec<[i32; 2]>);

/// Bond-preserving symmetry ops of the cell with their site actions.
pub(crate) fn cell_ops_with_action(cell: &Cell2D) -> Result<Vec<CellOpAction>, String> {
    let fr = frac_sites(cell)?;
    let ops = cell.symmetry_frac_ops(true);
    if ops.is_empty() {
        return Err("cell has no symmetry operations (not even the identity)".into());
    }
    ops.into_iter()
        .map(|op| op_site_action(&op, &fr).map(|(p, s)| (op, p, s)).ok_or_else(|| "symmetry op does not permute the basis".to_string()))
        .collect()
}

fn map_bond(b: &PeriodicBond, mf: [[i32; 2]; 2], perm: &[usize], shift: &[[i32; 2]]) -> PeriodicBond {
    let n1 = mf[0][0] * b.n1 + mf[0][1] * b.n2 + shift[b.j][0] - shift[b.i][0];
    let n2 = mf[1][0] * b.n1 + mf[1][1] * b.n2 + shift[b.j][1] - shift[b.i][1];
    PeriodicBond::canonical(perm[b.i], perm[b.j], n1, n2)
}

/// Directed pair displacements `(i, j, n1, n2, d)` with `1e-8 < d <= r_max`.
fn pair_distances(cell: &Cell2D, k: i32, r_max: f64) -> Vec<(usize, usize, i32, i32, f64)> {
    let nb = cell.basis.len();
    let mut out = Vec::new();
    for i in 0..nb {
        for j in 0..nb {
            for n1 in -k..=k {
                for n2 in -k..=k {
                    let (f1, f2) = (f64::from(n1), f64::from(n2));
                    let dx = cell.basis[j][0] + f1 * cell.a1[0] + f2 * cell.a2[0] - cell.basis[i][0];
                    let dy = cell.basis[j][1] + f1 * cell.a1[1] + f2 * cell.a2[1] - cell.basis[i][1];
                    let d = dx.hypot(dy);
                    if d > 1e-8 && d <= r_max {
                        out.push((i, j, n1, n2, d));
                    }
                }
            }
        }
    }
    out
}

/// Distance classes `d_1 < d_2 < ...` (each the smallest member) with tolerance `1e-5 * d_1`.
fn cluster_distances(ds: &mut [f64]) -> Vec<f64> {
    ds.sort_by(f64::total_cmp);
    let mut shells: Vec<f64> = Vec::new();
    let Some(&d1) = ds.first() else { return shells };
    let tol = 1e-5 * d1;
    for &d in ds.iter() {
        if shells.last().is_none_or(|&s| d - s > tol) {
            shells.push(d);
        }
    }
    shells
}

/// Neighbour shells `1..=max_shells` of a periodic cell, each split into exact space-group bond orbits.
pub fn classify_cell_shells(group: &str, cell: &Cell2D, max_shells: usize) -> Result<ShellClassification, String> {
    let nb = cell.basis.len();
    if nb == 0 || max_shells == 0 {
        return Err("need at least one basis site and max_shells >= 1".into());
    }
    let ops = cell_ops_with_action(cell)?;
    let det = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs();
    let h = det / cell.a1[0].hypot(cell.a1[1]).max(cell.a2[0].hypot(cell.a2[1]));
    let mut diam = 0.0_f64;
    for p in &cell.basis {
        for q in &cell.basis {
            diam = diam.max((p[0] - q[0]).hypot(p[1] - q[1]));
        }
    }
    let mut k = 2i32;
    let (pairs, dist) = loop {
        let r_safe = f64::from(k) * h - diam;
        let pairs = if r_safe > 0.0 { pair_distances(cell, k, r_safe) } else { Vec::new() };
        let mut ds: Vec<f64> = pairs.iter().map(|p| p.4).collect();
        let dist = cluster_distances(&mut ds);
        if dist.len() > max_shells {
            break (pairs, dist);
        }
        if k > 4096 {
            return Err("could not resolve the requested number of shells".into());
        }
        k *= 2;
    };
    let tol = 1e-5 * dist[0];
    let mut shells = Vec::with_capacity(max_shells);
    for (s, &ds) in dist.iter().take(max_shells).enumerate() {
        let members: Vec<_> = pairs.iter().filter(|p| (p.4 - ds).abs() <= tol).collect();
        let mut coord = vec![0usize; nb];
        for p in &members {
            coord[p.0] += 1;
        }
        let bonds: Vec<PeriodicBond> =
            members.iter().map(|p| PeriodicBond::canonical(p.0, p.1, p.2, p.3)).collect::<BTreeSet<_>>().into_iter().collect();
        let mut orbit_id: Vec<Option<usize>> = vec![None; bonds.len()];
        let mut orbits = Vec::new();
        for start in 0..bonds.len() {
            if orbit_id[start].is_some() {
                continue;
            }
            let oid = orbits.len();
            let mut members_o = BTreeSet::new();
            for (op, perm, shift) in &ops {
                let img = map_bond(&bonds[start], op.mf, perm, shift);
                let idx = bonds.binary_search(&img).map_err(|_| "symmetry image left the shell".to_string())?;
                orbit_id[idx] = Some(oid);
                members_o.insert(img);
            }
            let ob: Vec<PeriodicBond> = members_o.into_iter().collect();
            let mut oc = vec![0usize; nb];
            for b in &ob {
                oc[b.i] += 1;
                oc[b.j] += 1;
            }
            orbits.push(ShellOrbit {
                representative: ob[0],
                bonds_per_cell: ob.len(),
                mean_coordination: oc.iter().sum::<usize>() as f64 / nb as f64,
                coordination_by_basis: oc,
                bonds: ob,
            });
        }
        orbits.sort_by_key(|a| a.representative);
        shells.push(ShellInfo {
            shell: s + 1,
            distance: (ds * 1e10).round() / 1e10,
            total_directed_per_cell: members.len(),
            mean_coordination: members.len() as f64 / nb as f64,
            coordination_by_basis: coord,
            orbits,
        });
    }
    Ok(ShellClassification {
        group: group.to_string(),
        n_basis: nb,
        n_symmetry_ops: ops.len(),
        shells,
        provenance: "[PROVED: exact integer orbit closure under the bond-preserving cell symmetry]",
    })
}
