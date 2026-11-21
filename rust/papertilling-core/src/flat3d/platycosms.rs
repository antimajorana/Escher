//! 3D crystal-net compatibility & `[PROVED]` obstruction diagnostics, and finite
//! quotient 3-net + 3-complex (`chi_3 = V - E + F - C = 0`) builder for the 10
//! Conway-Rossetti closed flat 3D platycosms.
//!
//! Status values: `"OK"` (quotient built), `"IMPOSSIBLE"` (`[PROVED]` holohedry obstruction: no
//! symmetry subgroup of the crystal can have the platycosm's point group), `"NOT_FOUND"` (refused:
//! the finite search over supercells / origin shifts / the ITA standard setting relative to the
//! given cell axes found no free action; `[COMPUTATIONAL EVIDENCE]` only, since other settings are
//! not excluded).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::lattice_symmetry::{full_translation_lattice, holohedry_obstruction, lattice_holohedry};
use super::platycosm_data::{PlatyOp, ops_for_ita, verify_cw_euler_zero};
pub use super::platycosm_data::{PLATYCOSMS, PlatycosmInfo, list_platycosms, lookup_platycosm};
use super::{Cell3D, cart_to_frac_3d, dot3, frac_to_cart_3d, lattice_parameters_3d};

/// Result of testing whether a 3D crystal cell can form a quotient on a platycosm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlatycosmCompatibility {
    pub platycosm_code: String,
    pub status: String,
    pub compatible: bool,
    pub ita_number: Option<u16>,
    pub ita_symbol: Option<String>,
    pub supercell: Option<[usize; 3]>,
    pub origin_shift: Option<[f64; 3]>,
    pub reason: String,
}

/// Vertex in a finite platycosm quotient 3-net.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuotientVertex3d {
    pub id: usize,
    pub basis_idx: usize,
    pub site_kind: u8,
    pub frac: [f64; 3],
    pub pos: [f64; 3],
    pub orbit_size: usize,
}

/// Undirected edge in a finite platycosm quotient 3-net.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuotientBond3d {
    pub u: usize,
    pub v: usize,
    pub vec: [f64; 3],
    pub kind: usize,
    pub across_seam: bool,
}

/// Finite 3D crystal quotient on a closed flat 3D platycosm with verified `chi_3 = 0`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlatycosmQuotient3d {
    pub platycosm_code: String,
    pub platycosm_name: String,
    pub status: String,
    pub reason: String,
    pub orientable: bool,
    pub ita_number: Option<u16>,
    pub ita_symbol: Option<String>,
    pub h1_label: String,
    pub b1: usize,
    pub supercell: [usize; 3],
    pub vertices: Vec<QuotientVertex3d>,
    pub bonds: Vec<QuotientBond3d>,
    pub cw_v: usize,
    pub cw_e: usize,
    pub cw_f: usize,
    pub cw_c: usize,
    pub euler_chi_3: i64,
}

fn scaled_lat(lat: &[[f64; 3]; 3], l_sup: [usize; 3]) -> [[f64; 3]; 3] {
    [
        [lat[0][0] * l_sup[0] as f64, lat[0][1] * l_sup[0] as f64, lat[0][2] * l_sup[0] as f64],
        [lat[1][0] * l_sup[1] as f64, lat[1][1] * l_sup[1] as f64, lat[1][2] * l_sup[1] as f64],
        [lat[2][0] * l_sup[2] as f64, lat[2][1] * l_sup[2] as f64, lat[2][2] * l_sup[2] as f64],
    ]
}

fn metric_compatible(lat: &[[f64; 3]; 3], l_sup: [usize; 3], ops: &[PlatyOp]) -> bool {
    let s_lat = scaled_lat(lat, l_sup);
    let mut g = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            g[i][j] = dot3(s_lat[i], s_lat[j]);
        }
    }
    let tol = 1e-5 * g[0][0].max(g[1][1]).max(g[2][2]).max(1e-12);
    ops.iter().all(|op| {
        (0..3).all(|i| {
            (0..3).all(|j| {
                let val: f64 = (0..3)
                    .flat_map(|a| (0..3).map(move |b| f64::from(op.r[a][i]) * g[a][b] * f64::from(op.r[b][j])))
                    .sum();
                (val - g[i][j]).abs() <= tol
            })
        })
    })
}

fn metric_refusal_reason(info: &PlatycosmInfo, lat: &[[f64; 3]; 3], l_sup: [usize; 3]) -> String {
    let s_lat = scaled_lat(lat, l_sup);
    let ([a, b, c], [alpha, beta, gamma]) = lattice_parameters_3d(&s_lat);
    format!(
        "NOT_FOUND [COMPUTATIONAL EVIDENCE]: {} ({}) point operations in the ITA standard setting do not preserve the metric of any searched supercell of the given cell axes (e.g. {l_sup:?}: a={a:.4}, b={b:.4}, c={c:.4}, alpha={alpha:.1}, beta={beta:.1}, gamma={gamma:.1} deg); other settings are not excluded",
        info.code, info.ita_symbols[0]
    )
}

fn shift_op(op: &PlatyOp, s: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| {
        op.t[r] + s[r] - (f64::from(op.r[r][0]) * s[0] + f64::from(op.r[r][1]) * s[1] + f64::from(op.r[r][2]) * s[2])
    })
}

fn map_shift(r: [[i32; 3]; 3], s: [i32; 3], sv: [i32; 3], su: [i32; 3]) -> [i32; 3] {
    [0, 1, 2].map(|k| r[k][0] * s[0] + r[k][1] * s[1] + r[k][2] * s[2] + sv[k] - su[k])
}

type FreeActionData = (Vec<Vec<usize>>, Vec<(usize, usize, [f64; 3], usize)>);

fn try_free_action(
    cell: &Cell3D,
    frac_basis: &[[f64; 3]],
    l_sup: [usize; 3],
    ops: &[PlatyOp],
    shift: [f64; 3],
    ignore_site_kinds: bool,
) -> Option<FreeActionData> {
    let (m, [l1, l2, l3]) = (frac_basis.len(), l_sup);
    let n_sites = m * l1 * l2 * l3;
    let site_id = |b: usize, x: usize, y: usize, z: usize| ((b * l1 + x) * l2 + y) * l3 + z;
    let site_kind = |b: usize| cell.site_kinds.get(b).copied().unwrap_or(0);

    let mut perms = vec![vec![0usize; n_sites]; ops.len()];
    let mut cell_shifts = vec![vec![[0_i32; 3]; n_sites]; ops.len()];
    let mut x_coords = vec![[0.0_f64; 3]; n_sites];
    for (b, fb) in frac_basis.iter().enumerate() {
        for ix in 0..l1 {
            for iy in 0..l2 {
                for iz in 0..l3 {
                    x_coords[site_id(b, ix, iy, iz)] =
                        [(ix as f64 + fb[0]) / (l1 as f64), (iy as f64 + fb[1]) / (l2 as f64), (iz as f64 + fb[2]) / (l3 as f64)];
                }
            }
        }
    }

    for (oidx, op) in ops.iter().enumerate() {
        let t_eff = shift_op(op, shift);
        let is_id = op.r == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] && t_eff.iter().all(|&v| (v - v.round()).abs() < 1e-5);
        for (sid, &xp) in x_coords.iter().enumerate() {
            let k_src = site_kind(sid / (l1 * l2 * l3));
            let trg = [0, 1, 2].map(|r| f64::from(op.r[r][0]) * xp[0] + f64::from(op.r[r][1]) * xp[1] + f64::from(op.r[r][2]) * xp[2] + t_eff[r]);
            let matched = (0..m).find_map(|b2| {
                if !ignore_site_kinds && site_kind(b2) != k_src {
                    return None;
                }
                let u = [trg[0] * l1 as f64 - frac_basis[b2][0], trg[1] * l2 as f64 - frac_basis[b2][1], trg[2] * l3 as f64 - frac_basis[b2][2]];
                let r = u.map(f64::round);
                if (0..3).any(|k| (u[k] - r[k]).abs() >= 1e-4) {
                    return None;
                }
                let iv = r.map(|v| v as i32);
                let tid = site_id(b2, iv[0].rem_euclid(l1 as i32) as usize, iv[1].rem_euclid(l2 as i32) as usize, iv[2].rem_euclid(l3 as i32) as usize);
                if !is_id && tid == sid {
                    return None;
                }
                Some((tid, [iv[0].div_euclid(l1 as i32), iv[1].div_euclid(l2 as i32), iv[2].div_euclid(l3 as i32)]))
            })?;
            (perms[oidx][sid], cell_shifts[oidx][sid]) = matched;
        }
    }

    let mut dart_set = HashSet::new();
    let mut all_bonds = Vec::new();
    for ix in 0..l1 {
        for iy in 0..l2 {
            for iz in 0..l3 {
                for bd in &cell.bonds {
                    let u = site_id(bd.i, ix, iy, iz);
                    let (tx, ty, tz) = (ix as i32 + bd.n1, iy as i32 + bd.n2, iz as i32 + bd.n3);
                    let v = site_id(bd.j, tx.rem_euclid(l1 as i32) as usize, ty.rem_euclid(l2 as i32) as usize, tz.rem_euclid(l3 as i32) as usize);
                    let s = [tx.div_euclid(l1 as i32), ty.div_euclid(l2 as i32), tz.div_euclid(l3 as i32)];
                    dart_set.insert((u, v, s));
                    dart_set.insert((v, u, [-s[0], -s[1], -s[2]]));
                    all_bonds.push((u, v, s, bd.kind));
                }
            }
        }
    }
    for (oidx, op) in ops.iter().enumerate() {
        for &(u, v, s, _) in &all_bonds {
            let rs = map_shift(op.r, s, cell_shifts[oidx][v], cell_shifts[oidx][u]);
            if !dart_set.contains(&(perms[oidx][u], perms[oidx][v], rs)) {
                return None;
            }
        }
    }

    let mut site_orbit = vec![usize::MAX; n_sites];
    let mut orbits_out = Vec::new();
    for s in 0..n_sites {
        if site_orbit[s] != usize::MAX {
            continue;
        }
        let oid = orbits_out.len();
        let mut imgs: Vec<usize> = perms.iter().map(|p| p[s]).collect();
        imgs.sort_unstable();
        imgs.dedup();
        if imgs.len() != ops.len() {
            return None;
        }
        for &x in &imgs {
            site_orbit[x] = oid;
        }
        orbits_out.push(imgs);
    }

    let s_lat = scaled_lat(&cell.lattice, l_sup);
    let mut seen_b = HashSet::new();
    let mut q_bonds = Vec::new();
    for (u, v, s, kind) in all_bonds {
        let mut canon = if (u, s) <= (v, [-s[0], -s[1], -s[2]]) { (u, v, s) } else { (v, u, [-s[0], -s[1], -s[2]]) };
        for (oidx, op) in ops.iter().enumerate() {
            let (pu, pv) = (perms[oidx][u], perms[oidx][v]);
            let rs = map_shift(op.r, s, cell_shifts[oidx][v], cell_shifts[oidx][u]);
            let k1 = if (pu, rs) <= (pv, [-rs[0], -rs[1], -rs[2]]) { (pu, pv, rs) } else { (pv, pu, [-rs[0], -rs[1], -rs[2]]) };
            if k1 < canon {
                canon = k1;
            }
        }
        if seen_b.insert(canon) {
            let vec = frac_to_cart_3d(&s_lat, [0, 1, 2].map(|k| x_coords[v][k] + f64::from(s[k]) - x_coords[u][k]));
            q_bonds.push((site_orbit[u], site_orbit[v], vec, kind));
        }
    }
    Some((orbits_out, q_bonds))
}

type EmbedMatch = (u16, String, usize, [usize; 3], [f64; 3], Vec<Vec<usize>>, Vec<(usize, usize, [f64; 3], usize)>);

/// Refusal of an embedding search: `(status, reason)` with status `"IMPOSSIBLE"` or `"NOT_FOUND"`.
type Refusal = (&'static str, String);

fn find_compatible_embedding(
    cell: &Cell3D,
    info: &PlatycosmInfo,
    supercell: Option<[usize; 3]>,
    ignore_site_kinds: bool,
) -> Result<Result<EmbedMatch, Refusal>, String> {
    if cell.basis.is_empty() {
        return Err("cell has no basis sites".to_string());
    }
    let frac_basis: Vec<[f64; 3]> = cell.basis.iter().map(|&p| cart_to_frac_3d(&cell.lattice, p)).collect::<Result<_, _>>()?;
    if let Some(reason) = full_translation_lattice(cell, &frac_basis, ignore_site_kinds)
        .and_then(|lat| lattice_holohedry(&lat))
        .and_then(|h| holohedry_obstruction(info.code, &h))
    {
        return Ok(Err(("IMPOSSIBLE", reason)));
    }
    let l_cands: Vec<[usize; 3]> = match supercell {
        Some([a, b, c]) if a > 0 && b > 0 && c > 0 => vec![[a, b, c]],
        Some(_) => return Err("supercell repeats must all be >= 1".to_string()),
        None => vec![[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 6], [2, 2, 1], [2, 2, 2], [2, 2, 4], [2, 2, 6]],
    };
    let mut shifts = Vec::with_capacity(64);
    for &sz in &[0.0, 0.5, 0.25, 0.75] {
        for &sx in &[0.0, 0.5, 0.25, 0.75] {
            for &sy in &[0.0, 0.5, 0.25, 0.75] {
                shifts.push([sx, sy, sz]);
            }
        }
    }
    let axis_modes: &[usize] = if info.code == "c2" { &[0, 1, 2] } else { &[0] };
    let mut any_metric_ok = false;

    for &l_sup in &l_cands {
        for (&ita, &sym) in info.ita_numbers.iter().zip(info.ita_symbols) {
            for &ax in axis_modes {
                let ops = ops_for_ita(ita, ax)?;
                if !metric_compatible(&cell.lattice, l_sup, &ops) {
                    continue;
                }
                any_metric_ok = true;
                for &sh in &shifts {
                    if let Some((orbits, bonds)) = try_free_action(cell, &frac_basis, l_sup, &ops, sh, ignore_site_kinds) {
                        return Ok(Ok((ita, sym.to_string(), ax, l_sup, sh, orbits, bonds)));
                    }
                }
            }
        }
    }
    if !any_metric_ok {
        return Ok(Err(("NOT_FOUND", metric_refusal_reason(info, &cell.lattice, l_cands[0]))));
    }
    Ok(Err((
        "NOT_FOUND",
        format!(
            "NOT_FOUND [COMPUTATIONAL EVIDENCE]: {} ({}) metric holds for some searched supercell of {l_cands:?}, but no origin shift on the 1/4 grid gives a free {}-action by crystal symmetries; the lattice holohedry does not exclude {}",
            info.code, info.ita_symbols[0], info.ita_symbols[0], info.code
        ),
    )))
}

/// Test whether `cell` is compatible with platycosm `code`: `"OK"`, `"IMPOSSIBLE"` with a
/// `[PROVED]` holohedry obstruction, or `"NOT_FOUND"` with `[COMPUTATIONAL EVIDENCE]`.
///
/// # Errors
/// Returns an error for an unknown platycosm code or a singular/empty cell.
pub fn check_platycosm_compatibility(
    cell: &Cell3D,
    code: &str,
    supercell: Option<[usize; 3]>,
    ignore_site_kinds: bool,
) -> Result<PlatycosmCompatibility, String> {
    let info = lookup_platycosm(code)?;
    match find_compatible_embedding(cell, info, supercell, ignore_site_kinds)? {
        Ok((ita, sym, _ax, l_sup, sh, _, _)) => Ok(PlatycosmCompatibility {
            platycosm_code: info.code.to_string(),
            status: "OK".to_string(),
            compatible: true,
            ita_number: Some(ita),
            ita_symbol: Some(sym),
            supercell: Some(l_sup),
            origin_shift: Some(sh),
            reason: format!("compatible with {} ({}) on supercell {l_sup:?}", info.code, info.name),
        }),
        Err((status, reason)) => Ok(PlatycosmCompatibility {
            platycosm_code: info.code.to_string(),
            status: status.to_string(),
            compatible: false,
            ita_number: None,
            ita_symbol: None,
            supercell,
            origin_shift: None,
            reason,
        }),
    }
}

/// Build the finite quotient 3-net of `cell` on platycosm `code` and verify `chi_3 = 0`.
///
/// # Errors
/// Returns an error for an unknown platycosm code, a singular/empty cell, or a failed CW check.
pub fn build_platycosm_quotient(
    cell: &Cell3D,
    code: &str,
    supercell: Option<[usize; 3]>,
    ignore_site_kinds: bool,
) -> Result<PlatycosmQuotient3d, String> {
    let info = lookup_platycosm(code)?;
    match find_compatible_embedding(cell, info, supercell, ignore_site_kinds)? {
        Ok((ita, sym, ax, l_sup, _sh, orbits, raw_bonds)) => {
            let ops = ops_for_ita(ita, ax)?;
            let (cw_v, cw_e, cw_f, cw_c) = verify_cw_euler_zero(info.code, &ops, l_sup)?;
            let frac_basis: Vec<[f64; 3]> = cell.basis.iter().map(|&p| cart_to_frac_3d(&cell.lattice, p)).collect::<Result<_, _>>()?;
            let [l1, l2, l3] = l_sup;
            let s_lat = scaled_lat(&cell.lattice, l_sup);
            let vertices: Vec<QuotientVertex3d> = orbits
                .iter()
                .enumerate()
                .map(|(vid, orb)| {
                    let rep = orb[0];
                    let (b, rem) = (rep / (l1 * l2 * l3), rep % (l1 * l2 * l3));
                    let (ix, iy, iz) = (rem / (l2 * l3), (rem / l3) % l2, rem % l3);
                    let frac = [(ix as f64 + frac_basis[b][0]) / l1 as f64, (iy as f64 + frac_basis[b][1]) / l2 as f64, (iz as f64 + frac_basis[b][2]) / l3 as f64];
                    QuotientVertex3d { id: vid, basis_idx: b, site_kind: cell.site_kinds.get(b).copied().unwrap_or(0), frac, pos: frac_to_cart_3d(&s_lat, frac), orbit_size: orb.len() }
                })
                .collect();
            let bonds: Vec<QuotientBond3d> = raw_bonds
                .into_iter()
                .map(|(u, v, vec, kind)| {
                    let d = [0, 1, 2].map(|k| vertices[v].pos[k] - vertices[u].pos[k] - vec[k]);
                    QuotientBond3d { u, v, vec, kind, across_seam: u == v || dot3(d, d) > 1e-6 }
                })
                .collect();
            let euler_chi_3 = cw_v as i64 - cw_e as i64 + cw_f as i64 - cw_c as i64;
            Ok(PlatycosmQuotient3d {
                platycosm_code: info.code.to_string(),
                platycosm_name: info.name.to_string(),
                status: "OK".to_string(),
                reason: format!("OK: {} ({}) quotient with {} sites, {} bonds, chi_3={euler_chi_3}", info.code, sym, vertices.len(), bonds.len()),
                orientable: info.orientable,
                ita_number: Some(ita),
                ita_symbol: Some(sym),
                h1_label: info.h1_label.to_string(),
                b1: info.b1,
                supercell: l_sup,
                vertices,
                bonds,
                cw_v,
                cw_e,
                cw_f,
                cw_c,
                euler_chi_3,
            })
        }
        Err((status, reason)) => Ok(PlatycosmQuotient3d {
            platycosm_code: info.code.to_string(),
            platycosm_name: info.name.to_string(),
            status: status.to_string(),
            reason,
            orientable: info.orientable,
            ita_number: None,
            ita_symbol: None,
            h1_label: info.h1_label.to_string(),
            b1: info.b1,
            supercell: supercell.unwrap_or([1, 1, 1]),
            vertices: Vec::new(),
            bonds: Vec::new(),
            cw_v: 0,
            cw_e: 0,
            cw_f: 0,
            cw_c: 0,
            euler_chi_3: 0,
        }),
    }
}
