//! Symmetry-centred finite open flakes (`BuiltCrystal`) invariant under the
//! site-symmetry point group at `centre`, supporting disk cuts, regular `n`-fold
//! polygonal cuts, and Bravais Wigner-Seitz Voronoi cuts.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::plane::cell::{mat2_det, mat2_inv, mat2_vec};
use crate::plane::groups::group_spec;
use crate::plane::voronoi_cells::voronoi_cells;
use crate::types::{BuiltCrystal, Cell2D, DirectedBondData};

/// Finite open crystal flake with exact site-symmetry permutations and orbits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymmetricFlake {
    pub group: String,
    pub centre: [f64; 2],
    pub radius: f64,
    pub n_fold_polygon: Option<usize>,
    pub point_group_order: usize,
    pub is_dihedral: bool,
    pub crystal: BuiltCrystal,
    pub site_orbits: Vec<Vec<usize>>,
    pub vertex_perms: Vec<Vec<usize>>,
    pub boundary_sites: Vec<usize>,
}

fn norm2(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn mat_close(a: [[f64; 2]; 2], b: [[f64; 2]; 2], tol: f64) -> bool {
    (0..2).all(|r| (0..2).all(|c| (a[r][c] - b[r][c]).abs() <= tol))
}

fn max_rot_order_of_group(group: &str) -> Result<(bool, bool, usize), String> {
    let spec = group_spec(group)?;
    let has_mirrors = !spec.chiral && !matches!(spec.symbol, "pg" | "pgg");
    let max_rot = match spec.symbol {
        "p1" | "pm" | "pg" | "cm" => 1,
        "p2" | "pmm" | "pmg" | "pgg" | "cmm" => 2,
        "p3" | "p3m1" | "p31m" => 3,
        "p4" | "p4m" | "p4g" => 4,
        "p6" | "p6m" => 6,
        _ => 6,
    };
    Ok((has_mirrors, spec.chiral, max_rot))
}

fn mat_mul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

fn site_symmetry_mats(
    group: &str,
    cell: &Cell2D,
    centre: [f64; 2],
) -> Result<Vec<[[f64; 2]; 2]>, String> {
    let a = cell.mat_a();
    let a_inv = mat2_inv(a)?;
    let (has_mirrors, chiral_req, max_rot) = if group.trim().is_empty() {
        (true, false, 6)
    } else {
        max_rot_order_of_group(group)?
    };
    let ops = cell.symmetry_ops(false);
    let mut rots: Vec<[[f64; 2]; 2]> = vec![[[1.0, 0.0], [0.0, 1.0]]];
    let mut refls: Vec<[[f64; 2]; 2]> = Vec::new();
    for op in ops {
        let det = op.m[0][0] * op.m[1][1] - op.m[0][1] * op.m[1][0];
        if (chiral_req || !has_mirrors) && det < 0.0 {
            continue;
        }
        let mc = mat2_vec(op.m, centre);
        let l_req = [centre[0] - (mc[0] + op.t[0]), centre[1] - (mc[1] + op.t[1])];
        let f = mat2_vec(a_inv, l_req);
        if (f[0] - f[0].round()).abs() >= 1e-4 || (f[1] - f[1].round()).abs() >= 1e-4 {
            continue;
        }
        if det > 0.0 {
            let ang = op.m[1][0].atan2(op.m[0][0]).abs();
            let ord = if ang < 1e-5 { 1 } else { (TAU / ang).round() as usize };
            if ord > 1 && !max_rot.is_multiple_of(ord) {
                continue;
            }
            if !rots.iter().any(|&m| mat_close(m, op.m, 1e-5)) {
                rots.push(op.m);
            }
        } else if !refls.iter().any(|&m| mat_close(m, op.m, 1e-5)) {
            refls.push(op.m);
        }
    }
    // Keep only a closed mirror coset R * m0 (products m * m0 must lie in `rots`).
    if let Some(&m0) = refls.first() {
        for m in refls {
            let prod = mat_mul2(m, m0);
            if rots.iter().any(|&r| mat_close(prod, r, 1e-4)) && !rots.iter().any(|&k| mat_close(k, m, 1e-5)) {
                rots.push(m);
            }
        }
    }
    Ok(rots)
}

fn halfplanes_for_n_gon(
    n_sides: usize,
    radius: f64,
    theta0: f64,
) -> Vec<([f64; 2], f64)> {
    (0..n_sides)
        .map(|k| {
            let th = theta0 + (k as f64) * TAU / (n_sides as f64);
            ([th.cos(), th.sin()], radius)
        })
        .collect()
}

fn halfplanes_for_poly(poly: &[[f64; 2]], scale: f64) -> Vec<([f64; 2], f64)> {
    let m = poly.len();
    let mut out = Vec::with_capacity(m);
    for k in 0..m {
        let p = [poly[k][0] * scale, poly[k][1] * scale];
        let q = [poly[(k + 1) % m][0] * scale, poly[(k + 1) % m][1] * scale];
        let e = [q[0] - p[0], q[1] - p[1]];
        let len = norm2(e).max(1e-12);
        // Outward normal for CCW polygon.
        let nrm = [e[1] / len, -e[0] / len];
        let d = dot2(nrm, p);
        out.push((nrm, d));
    }
    out
}

fn build_flake_from_halfplanes(
    group: &str,
    cell: &Cell2D,
    centre: [f64; 2],
    radius: f64,
    n_fold_polygon: Option<usize>,
    halfplanes: Option<&[([f64; 2], f64)]>,
    sym_mats: Vec<[[f64; 2]; 2]>,
) -> Result<SymmetricFlake, String> {
    let a = cell.mat_a();
    let a_inv = mat2_inv(a)?;
    let max_b = cell.basis.iter().map(|&b| norm2(b)).fold(0.0_f64, f64::max);
    let r_search = radius + max_b + norm2(cell.a1) + norm2(cell.a2);
    let len_inv = norm2([a_inv[0][0], a_inv[0][1]]).max(norm2([a_inv[1][0], a_inv[1][1]]));
    let n_max = (r_search * len_inv + 3.0).ceil() as i32;
    let cf = mat2_vec(a_inv, centre);
    let (cx0, cy0) = (cf[0].round() as i32, cf[1].round() as i32);

    let inside = |rel: [f64; 2]| -> bool {
        if let Some(hps) = halfplanes {
            hps.iter().all(|&(u, d)| dot2(rel, u) <= d + 1e-6)
        } else {
            norm2(rel) <= radius + 1e-6
        }
    };

    // Filter sym_mats to those preserving the cut boundary.
    let active_mats: Vec<[[f64; 2]; 2]> = if let Some(hps) = halfplanes {
        sym_mats
            .into_iter()
            .filter(|&m| {
                hps.iter().all(|&(u, d)| {
                    let mu = mat2_vec(m, u);
                    hps.iter().any(|&(u2, d2)| norm2([mu[0] - u2[0], mu[1] - u2[1]]) < 1e-4 && (d - d2).abs() < 1e-4)
                })
            })
            .collect()
    } else {
        sym_mats
    };

    let mut kept_map: BTreeMap<(usize, i32, i32), [f64; 2]> = BTreeMap::new();
    for n1 in (cx0 - n_max)..=(cx0 + n_max) {
        for n2 in (cy0 - n_max)..=(cy0 + n_max) {
            let sx = f64::from(n1) * cell.a1[0] + f64::from(n2) * cell.a2[0];
            let sy = f64::from(n1) * cell.a1[1] + f64::from(n2) * cell.a2[1];
            for (i, &bi) in cell.basis.iter().enumerate() {
                let p = [bi[0] + sx, bi[1] + sy];
                let rel = [p[0] - centre[0], p[1] - centre[1]];
                if inside(rel) {
                    kept_map.insert((i, n1, n2), p);
                }
            }
        }
    }

    // Close kept_map under active_mats to eliminate any boundary-tolerance asymmetry.
    let initial_pts: Vec<[f64; 2]> = kept_map.values().copied().collect();
    for p in initial_pts {
        let rel = [p[0] - centre[0], p[1] - centre[1]];
        for &m in &active_mats {
            let mr = mat2_vec(m, rel);
            let q = [centre[0] + mr[0], centre[1] + mr[1]];
            for (j, &bj) in cell.basis.iter().enumerate() {
                let f = mat2_vec(a_inv, [q[0] - bj[0], q[1] - bj[1]]);
                let (r0, r1) = (f[0].round(), f[1].round());
                if (f[0] - r0).abs() < 1e-4 && (f[1] - r1).abs() < 1e-4 {
                    let key = (j, r0 as i32, r1 as i32);
                    kept_map.entry(key).or_insert([
                        bj[0] + r0 * cell.a1[0] + r1 * cell.a2[0],
                        bj[1] + r0 * cell.a1[1] + r1 * cell.a2[1],
                    ]);
                    break;
                }
            }
        }
    }

    let keys: Vec<(usize, i32, i32)> = kept_map.keys().copied().collect();
    let idx_of: HashMap<(usize, i32, i32), usize> = keys.iter().enumerate().map(|(u, &k)| (k, u)).collect();
    let n = keys.len();
    let pos: Vec<[f64; 2]> = keys.iter().map(|k| kept_map[k]).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|k| k.0).collect();

    let mut seen_edges = BTreeSet::new();
    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for (u, &(i, n1, n2)) in keys.iter().enumerate() {
        for bd in &cell.bonds {
            for (src, dst, d1, d2) in [(bd.i, bd.j, bd.n1, bd.n2), (bd.j, bd.i, -bd.n1, -bd.n2)] {
                if src != i {
                    continue;
                }
                if let Some(&v) = idx_of.get(&(dst, n1 + d1, n2 + d2))
                    && u != v
                {
                    let vec_uv = [pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]];
                    adj[u].push(DirectedBondData {
                        to: v,
                        vec: vec_uv,
                        kind: bd.kind,
                        seam: None,
                        ang: None,
                        across: false,
                        ghost_stub: None,
                    });
                    if seen_edges.insert((u.min(v), u.max(v))) {
                        edges.push((u.min(v), u.max(v)));
                    }
                }
            }
        }
    }

    // Compute exact site permutations & orbits under active_mats.
    let mut vertex_perms = Vec::with_capacity(active_mats.len());
    for &m in &active_mats {
        let mut perm = vec![0usize; n];
        for (u, &pu) in pos.iter().enumerate() {
            let mr = mat2_vec(m, [pu[0] - centre[0], pu[1] - centre[1]]);
            let q = [centre[0] + mr[0], centre[1] + mr[1]];
            let v = (0..n)
                .find(|&w| norm2([pos[w][0] - q[0], pos[w][1] - q[1]]) < 1e-4)
                .ok_or_else(|| format!("symmetry operation did not map flake site {u} onto flake"))?;
            perm[u] = v;
        }
        vertex_perms.push(perm);
    }
    let mut orb_of = vec![usize::MAX; n];
    let mut site_orbits = Vec::new();
    for u in 0..n {
        if orb_of[u] != usize::MAX {
            continue;
        }
        let oid = site_orbits.len();
        let mut orb: Vec<usize> = vertex_perms.iter().map(|p| p[u]).collect();
        orb.sort_unstable();
        orb.dedup();
        for &w in &orb {
            orb_of[w] = oid;
        }
        site_orbits.push(orb);
    }

    let max_d = adj.iter().map(Vec::len).max().unwrap_or(0);
    let boundary_sites = (0..n).filter(|&u| adj[u].len() < max_d).collect();
    let is_dihedral = active_mats.iter().any(|&m| mat2_det(m) < 0.0);
    Ok(SymmetricFlake {
        group: group.to_string(),
        centre,
        radius,
        n_fold_polygon,
        point_group_order: active_mats.len(),
        is_dihedral,
        crystal: BuiltCrystal {
            pos: pos.clone(),
            ref_pos: pos,
            basis_idx,
            side: vec![None; n],
            cone_sites: Vec::new(),
            edges,
            adj,
            wall_sigma: None,
            is_non_orientable: false,
        },
        site_orbits,
        vertex_perms,
        boundary_sites,
    })
}

/// Cut a finite open `BuiltCrystal` flake around `centre`, invariant under its site-symmetry
/// point group (disk cut when `n_fold_polygon` is `None`, or regular `n`-gon of apothem `radius`).
///
/// # Errors
/// Returns an error for a non-positive radius, an empty basis, `n_fold_polygon < 3`, an unknown
/// group, or a symmetry operation that fails to map the flake onto itself.
pub fn build_symmetric_flake(
    group: &str,
    cell: &Cell2D,
    centre: [f64; 2],
    n_fold_polygon: Option<usize>,
    radius: f64,
) -> Result<SymmetricFlake, String> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err("flake radius must be positive and finite".to_string());
    }
    if cell.basis.is_empty() {
        return Err("cell has no basis sites".to_string());
    }
    let sym_mats = site_symmetry_mats(group, cell, centre)?;
    let hps = if let Some(n_sides) = n_fold_polygon {
        if n_sides < 3 {
            return Err("n_fold_polygon must be >= 3".to_string());
        }
        // Align polygon normal theta0 with the first mirror line eigenvector (+1) if present.
        let theta0 = sym_mats
            .iter()
            .find(|&&m| mat2_det(m) < 0.0)
            .map(|&m| {
                let c0 = [m[0][0] + 1.0, m[1][0]];
                let c1 = [m[0][1], m[1][1] + 1.0];
                let v = if norm2(c0) >= norm2(c1) { c0 } else { c1 };
                v[1].atan2(v[0])
            })
            .unwrap_or(0.0);
        Some(halfplanes_for_n_gon(n_sides, radius, theta0))
    } else {
        None
    };
    build_flake_from_halfplanes(group, cell, centre, radius, n_fold_polygon, hps.as_deref(), sym_mats)
}

/// Cut a finite open `BuiltCrystal` flake inside the scaled Bravais Wigner-Seitz cell
/// around `centre` (automatically invariant under the full site-symmetry group).
///
/// # Errors
/// Returns an error for a non-positive scale, an unknown group, or a degenerate Voronoi cell.
pub fn build_wigner_seitz_flake(
    group: &str,
    cell: &Cell2D,
    centre: [f64; 2],
    scale: f64,
) -> Result<SymmetricFlake, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("Wigner-Seitz scale must be positive and finite".to_string());
    }
    let sym_mats = site_symmetry_mats(group, cell, centre)?;
    let ws = voronoi_cells(&[[0.0, 0.0]], cell.a1, cell.a2);
    let vc = ws.first().ok_or("failed to build Bravais Wigner-Seitz Voronoi cell")?;
    let hps = halfplanes_for_poly(&vc.vertices, scale);
    let eff_r = vc.vertices.iter().map(|&p| norm2(p) * scale).fold(0.0_f64, f64::max);
    build_flake_from_halfplanes(group, cell, centre, eff_r, Some(vc.vertices.len()), Some(&hps), sym_mats)
}
