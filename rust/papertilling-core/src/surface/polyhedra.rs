//! Polyhedral sphere and projective-plane constructions and rotation-centre catalogue.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::pieces::{ConeSurface, LatticeData, glue};
use super::report::symmetry_facts;
pub use super::solids::{SolidNet, solid_net};
use super::solids::{populate_3d, solid_name};
use super::{
    IndexUf, RefusalKind, SurfaceError, add2, cross2, dot2, frac_mod, mat2_inv, mat2_mul, mat2_vec,
    norm2, sub2, symmetry_ops_on_cell,
};
use crate::types::{Bond2D, Cell2D, SurfaceMesh};

/// Rotation-centre equivalence class on the reduced primitive cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentreClass {
    pub id: String,
    pub label: String,
    pub gclass: String,
    pub order: u32,
    pub point: [f64; 2],
    pub location: String,
    pub members: Vec<[f64; 2]>,
}

fn reduce_basis(mut b1: [f64; 2], mut b2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    for _ in 0..200 {
        if dot2(b2, b2) < dot2(b1, b1) - 1e-12 {
            std::mem::swap(&mut b1, &mut b2);
        }
        let k = (dot2(b1, b2) / dot2(b1, b1)).round();
        if k == 0.0 {
            break;
        }
        b2 = [b2[0] - k * b1[0], b2[1] - k * b1[1]];
    }
    if cross2(b1, b2) < 0.0 {
        b2 = [-b2[0], -b2[1]];
    }
    (b1, b2)
}

fn locate_in_cell(p: [f64; 2], b_inv: [[f64; 2]; 2], basis: &[[f64; 2]]) -> Option<(usize, [i32; 2])> {
    for (s, &b) in basis.iter().enumerate() {
        let f = mat2_vec(b_inv, sub2(p, b));
        let r = [f[0].round(), f[1].round()];
        if (f[0] - r[0]).abs() < 1e-6 && (f[1] - r[1]).abs() < 1e-6 {
            return Some((s, [r[0] as i32, r[1] as i32]));
        }
    }
    None
}

fn rebase_cell(cell: &Cell2D, b1: [f64; 2], b2: [f64; 2]) -> Result<Cell2D, String> {
    let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
    let b = [[b1[0], b2[0]], [b1[1], b2[1]]];
    let b_inv = mat2_inv(b)?;
    let mut new_basis: Vec<[f64; 2]> = Vec::new();
    for &p in &cell.basis {
        let f = mat2_vec(b_inv, p);
        let q = mat2_vec(b, [frac_mod(f[0]), frac_mod(f[1])]);
        if !new_basis.iter().any(|&r| norm2(sub2(q, r)) < 1e-6) {
            new_basis.push(q);
        }
    }
    let mut bond_set: Vec<(usize, usize, i32, i32)> = Vec::new();
    for bd in &cell.bonds {
        let p = cell.basis[bd.i];
        let q = add2(cell.basis[bd.j], mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]));
        let (si, ni) = locate_in_cell(p, b_inv, &new_basis).ok_or("rebase site i not found")?;
        let (sj, nj) = locate_in_cell(q, b_inv, &new_basis).ok_or("rebase site j not found")?;
        let (dn0, dn1) = (nj[0] - ni[0], nj[1] - ni[1]);
        let canon = (si, sj, dn0, dn1).min((sj, si, -dn0, -dn1));
        bond_set.push(canon);
    }
    bond_set.sort_unstable();
    bond_set.dedup();
    Ok(Cell2D {
        a1: b1,
        a2: b2,
        basis: new_basis,
        bonds: bond_set.into_iter().map(|(i, j, n1, n2)| Bond2D { i, j, n1, n2, kind: 0 }).collect(),
    })
}

fn primitive_reduced_cell(cell: &Cell2D) -> Result<Cell2D, String> {
    let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
    let a_inv = mat2_inv(a)?;
    let f_basis: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    let bonds: Vec<_> = cell.bonds.iter().map(|b| (b.i, b.j, b.n1, b.n2)).collect();
    let mut darts = HashSet::with_capacity(bonds.len() * 2);
    for &(i, j, n1, n2) in &bonds {
        darts.insert((i, j, n1, n2));
        darts.insert((j, i, -n1, -n2));
    }
    let ts: Vec<[f64; 2]> = symmetry_ops_on_cell(a, a_inv, &f_basis, &bonds, &darts, true)
        .into_iter()
        .map(|(_, t)| t)
        .collect();
    if ts.len() <= 1 {
        let (b1, b2) = reduce_basis(cell.a1, cell.a2);
        if norm2(sub2(b1, cell.a1)) < 1e-8 && norm2(sub2(b2, cell.a2)) < 1e-8 {
            return Ok(cell.clone());
        }
        return rebase_cell(cell, b1, b2);
    }
    let mut cands = Vec::new();
    for t in &ts {
        for n1 in -2..=2 {
            for n2 in -2..=2 {
                let v = mat2_vec(a, [t[0] + n1 as f64, t[1] + n2 as f64]);
                if norm2(v) > 1e-6 {
                    cands.push(v);
                }
            }
        }
    }
    cands.sort_by(|u, v| {
        ((norm2(*u) * 1e9).round() as i64)
            .cmp(&((norm2(*v) * 1e9).round() as i64))
            .then_with(|| (-u[0]).total_cmp(&(-v[0])))
            .then_with(|| (-u[1]).total_cmp(&(-v[1])))
    });
    let area = (a[0][0] * a[1][1] - a[0][1] * a[1][0]).abs() / ts.len() as f64;
    let b1 = cands[0];
    let b2 = *cands
        .iter()
        .find(|&&v| (cross2(b1, v).abs() - area).abs() < 1e-6 * area.max(1.0))
        .ok_or("primitive basis b2 not found")?;
    let (b1, b2) = reduce_basis(b1, b2);
    rebase_cell(cell, b1, b2)
}

fn has_crossings(cell: &Cell2D, a: [[f64; 2]; 2]) -> bool {
    let segs0: Vec<([f64; 2], [f64; 2])> = cell
        .bonds
        .iter()
        .map(|b| (cell.basis[b.i], add2(cell.basis[b.j], mat2_vec(a, [b.n1 as f64, b.n2 as f64]))))
        .collect();
    for n1 in -5..=5 {
        for n2 in -5..=5 {
            let sh = mat2_vec(a, [n1 as f64, n2 as f64]);
            for (k, &(p, q)) in segs0.iter().enumerate() {
                for (l, &(r0, s0)) in segs0.iter().enumerate() {
                    if n1 == 0 && n2 == 0 && l <= k {
                        continue;
                    }
                    let (r, s) = (add2(r0, sh), add2(s0, sh));
                    let (da, db) = (sub2(q, p), sub2(s, r));
                    let den = cross2(da, db);
                    if den.abs() >= 1e-9 {
                        let (u, v) = (cross2(sub2(r, p), db) / den, cross2(sub2(r, p), da) / den);
                        if u > 1e-9 && u < 1.0 - 1e-9 && v > 1e-9 && v < 1.0 - 1e-9 {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

struct CatalogueData {
    a_inv: [[f64; 2]; 2],
    classes: Vec<CentreClass>,
}

impl CatalogueData {
    fn same_point(&self, p: [f64; 2], q: [f64; 2]) -> bool {
        let f = mat2_vec(self.a_inv, sub2(p, q));
        (f[0] - f[0].round()).abs() < 1e-4 && (f[1] - f[1].round()).abs() < 1e-4
    }

    fn lookup(&self, c: [f64; 2]) -> Option<(String, String)> {
        self.classes
            .iter()
            .find(|r| r.members.iter().any(|&q| self.same_point(c, q)))
            .map(|r| (r.label.clone(), r.location.clone()))
    }
}

fn build_catalogue(cell: &Cell2D) -> Result<CatalogueData, String> {
    let p_cell = primitive_reduced_cell(cell)?;
    let a = [[p_cell.a1[0], p_cell.a2[0]], [p_cell.a1[1], p_cell.a2[1]]];
    let a_inv = mat2_inv(a)?;
    let f_basis: Vec<[f64; 2]> = p_cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    let bonds: Vec<_> = p_cell.bonds.iter().map(|b| (b.i, b.j, b.n1, b.n2)).collect();
    let mut darts = HashSet::with_capacity(bonds.len() * 2);
    for &(i, j, n1, n2) in &bonds {
        darts.insert((i, j, n1, n2));
        darts.insert((j, i, -n1, -n2));
    }
    let opsf = symmetry_ops_on_cell(a, a_inv, &f_basis, &bonds, &darts, false);
    let ops: Vec<([[f64; 2]; 2], [f64; 2])> = opsf
        .iter()
        .map(|&(mf, t)| {
            let mf_f = [[mf[0][0] as f64, mf[0][1] as f64], [mf[1][0] as f64, mf[1][1] as f64]];
            (mat2_mul(mat2_mul(a, mf_f), a_inv), mat2_vec(a, t))
        })
        .collect();
    let plus: Vec<_> = ops.iter().copied().filter(|&(m, _)| m[0][0] * m[1][1] - m[0][1] * m[1][0] > 0.0).collect();
    let reduce_pt = |pt: [f64; 2]| -> [f64; 2] {
        let f = mat2_vec(a_inv, pt);
        let mut g = [f[0] - (f[0] + 1e-6).floor(), f[1] - (f[1] + 1e-6).floor()];
        for x in &mut g {
            if x.abs() < 1e-9 || (*x - 1.0).abs() < 1e-6 {
                *x = 0.0;
            }
        }
        mat2_vec(a, g)
    };
    let same_pt = |p: [f64; 2], q: [f64; 2]| -> bool {
        let f = mat2_vec(a_inv, sub2(p, q));
        (f[0] - f[0].round()).abs() < 1e-4 && (f[1] - f[1].round()).abs() < 1e-4
    };
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for &(m, t) in &plus {
        if (m[0][0] - 1.0).abs() < 1e-6 && m[0][1].abs() < 1e-6 && m[1][0].abs() < 1e-6 && (m[1][1] - 1.0).abs() < 1e-6 {
            continue;
        }
        let Ok(im_inv) = mat2_inv([[1.0 - m[0][0], -m[0][1]], [-m[1][0], 1.0 - m[1][1]]]) else {
            continue;
        };
        for m1 in -2..=2 {
            for m2 in -2..=2 {
                let c = reduce_pt(mat2_vec(im_inv, add2(t, mat2_vec(a, [m1 as f64, m2 as f64]))));
                if !pts.iter().any(|&q| same_pt(c, q)) {
                    pts.push(c);
                }
            }
        }
    }
    let orders: Vec<u32> = pts
        .iter()
        .map(|&c| plus.iter().filter(|&&(m, t)| same_pt(add2(mat2_vec(m, c), t), c)).count() as u32)
        .collect();
    let orbits = |sym: &[([[f64; 2]; 2], [f64; 2])]| -> Vec<Vec<usize>> {
        let mut uf = IndexUf::new(pts.len());
        for (i, &c) in pts.iter().enumerate() {
            for &(m, t) in sym {
                let im = reduce_pt(add2(mat2_vec(m, c), t));
                if let Some(j) = pts.iter().position(|&q| same_pt(im, q)) {
                    uf.union(i, j);
                }
            }
        }
        let roots = uf.roots();
        roots.into_iter().map(|r| (0..pts.len()).filter(|&i| uf.find(i) == r).collect()).collect()
    };
    let (plus_orbs, full_orbs) = (orbits(&plus), orbits(&ops));
    let mut gid = vec![0usize; pts.len()];
    for (k, o) in full_orbs.iter().enumerate() {
        for &i in o {
            gid[i] = k;
        }
    }
    let frac_key = |c: [f64; 2]| -> (i64, i64) {
        let f = mat2_vec(a_inv, c);
        ((f[0] * 10000.0).round().rem_euclid(10000.0) as i64, (f[1] * 10000.0).round().rem_euclid(10000.0) as i64)
    };
    let mut raw = Vec::with_capacity(plus_orbs.len());
    for o in plus_orbs {
        let rep = o.iter().map(|&i| pts[i]).min_by_key(|&c| frac_key(c)).unwrap();
        let members: Vec<[f64; 2]> = o.iter().map(|&i| pts[i]).collect();
        raw.push((rep, frac_key(rep), orders[o[0]], gid[o[0]], members));
    }
    raw.sort_by_key(|r| (-(r.2 as i32), r.1));
    let planar = !has_crossings(&p_cell, a);
    let lmax = p_cell.bonds.iter().fold(0.0_f64, |acc, b| {
        acc.max(norm2(sub2(add2(p_cell.basis[b.j], mat2_vec(a, [b.n1 as f64, b.n2 as f64])), p_cell.basis[b.i])))
    });
    let min_cell_len = norm2(p_cell.a1).min(norm2(p_cell.a2)).max(1e-6);
    let classify = |c: [f64; 2]| -> String {
        if locate_in_cell(c, a_inv, &p_cell.basis).is_some() {
            return "site".into();
        }
        let fc = mat2_vec(a_inv, c);
        let (n00, n01) = (fc[0].round() as i32, fc[1].round() as i32);
        let k_box = (lmax / min_cell_len).ceil() as i32 + 2;
        for d1 in -k_box..=k_box {
            for d2 in -k_box..=k_box {
                let o = mat2_vec(a, [(n00 + d1) as f64, (n01 + d2) as f64]);
                for b in &p_cell.bonds {
                    let p = add2(o, p_cell.basis[b.i]);
                    let q = add2(add2(o, p_cell.basis[b.j]), mat2_vec(a, [b.n1 as f64, b.n2 as f64]));
                    let d = sub2(q, p);
                    let s = dot2(sub2(c, p), d) / dot2(d, d);
                    if s > -1e-9 && s < 1.0 + 1e-9 && norm2(sub2([p[0] + s * d[0], p[1] + s * d[1]], c)) < 1e-6 {
                        return if (s - 0.5).abs() < 1e-6 { "bond midpoint".into() } else { "on a bond".into() };
                    }
                }
            }
        }
        if !planar { "elsewhere".into() } else { "face centre".into() }
    };
    let (mut glabel, mut letters, mut gcount, mut seen) =
        (HashMap::new(), HashMap::<u32, usize>::new(), HashMap::<usize, usize>::new(), HashMap::<usize, usize>::new());
    for &(_, _, ord, gi, _) in &raw {
        *gcount.entry(gi).or_insert(0) += 1;
        glabel.entry(gi).or_insert_with(|| {
            let idx = *letters.entry(ord).and_modify(|x| *x += 1).or_insert(0);
            let ch = b"ABCDEFGH"[idx.min(7)] as char;
            format!("order-{ord} class {ch}")
        });
    }
    let classes = raw
        .into_iter()
        .enumerate()
        .map(|(k, (point, _, order, gi, members))| {
            let gclass = glabel[&gi].clone();
            let s = *seen.entry(gi).and_modify(|x| *x += 1).or_insert(1);
            let label = if gcount[&gi] == 1 { gclass.clone() } else { format!("{gclass}/{s}") };
            CentreClass { id: format!("C{}", k + 1), label, gclass, order, point, location: classify(point), members }
        })
        .collect();
    Ok(CatalogueData { a_inv, classes })
}

/// Proper rotation-centre classes of `cell`, sorted descending by order and `"face centre"` before `"site"`.
pub fn candidate_centres(_group: &str, cell: &Cell2D) -> Vec<CentreClass> {
    let Ok(cat) = build_catalogue(cell) else {
        return Vec::new();
    };
    let loc_rank = |loc: &str| match loc {
        "face centre" => 0,
        "site" => 1,
        "bond midpoint" => 2,
        "on a bond" => 3,
        _ => 4,
    };
    let mut indexed: Vec<(usize, CentreClass)> = cat.classes.into_iter().enumerate().collect();
    indexed.sort_by_key(|(idx, r)| (-(r.order as i32), loc_rank(&r.location), *idx));
    indexed.into_iter().map(|(_, r)| r).collect()
}

pub(crate) fn build_polyhedron(
    group: &str,
    cell: &Cell2D,
    size: Option<(i32, i32)>,
    centre_filter: Option<&str>,
    projective: bool,
) -> Result<ConeSurface, SurfaceError> {
    let what = if projective { "projective plane" } else { "sphere" };
    let facts = symmetry_facts(group);
    if let Some(reason) = facts.impossible_reason(what) {
        return Err(SurfaceError::impossible(reason));
    }
    let lat = LatticeData::new(group, cell).map_err(SurfaceError::invalid)?;
    let cat = build_catalogue(cell).map_err(SurfaceError::invalid)?;
    let n = cat.classes.iter().map(|r| r.order).max().unwrap_or(1);
    if n == 1 {
        return Err(SurfaceError::not_found(format!("no rotation centre found for {group}"), String::new(), false));
    }
    let solid = solid_name(n)?;
    let sz = size.unwrap_or(if projective || n == 2 { (1, 1) } else { (1, 0) });
    let recs: Vec<&CentreClass> = cat.classes.iter().filter(|r| r.order == n).collect();
    let (cands, skipped): (Vec<&CentreClass>, Vec<&CentreClass>) = if let Some(cf) = centre_filter {
        let hit: Vec<&CentreClass> = recs
            .iter()
            .copied()
            .filter(|r| cf == r.id || cf == r.label || cf == r.location || cf == r.gclass)
            .collect();
        if hit.is_empty() {
            return Err(SurfaceError::not_found(format!("no order-{n} rotation centre class {cf:?} in {group}"), String::new(), true));
        }
        (hit, Vec::new())
    } else {
        let mut usable: Vec<&CentreClass> = recs.iter().copied().filter(|r| r.location == "face centre" || r.location == "site").collect();
        usable.sort_by_key(|r| usize::from(r.location != "face centre"));
        let skip = recs.iter().copied().filter(|r| r.location != "face centre" && r.location != "site").collect();
        (usable, skip)
    };
    if cands.is_empty() {
        let mut err = SurfaceError::not_found(
            format!("{what} not found for {group}: every order-{n} centre is at a bond midpoint"),
            String::new(),
            false,
        );
        err.tried.push(format!("centres {} (bond midpoints)", skipped.iter().map(|r| r.label.as_str()).collect::<Vec<_>>().join(", ")));
        return Err(err);
    }
    let mut errors: Vec<(String, SurfaceError)> = Vec::new();
    for rec in cands {
        let net_res = solid_net(&lat, solid, sz, rec.point, projective);
        let (pieces, pairs, f3, v3) = match net_res {
            Ok(v) => v,
            Err(e) => {
                if matches!(e.kind, RefusalKind::InvalidArg) {
                    return Err(e);
                }
                errors.push((rec.label.clone(), e));
                continue;
            }
        };
        let kind_name = if projective {
            if solid == "pillow" { "hemi-pillow (projective plane)".into() } else { format!("hemi-{solid} (projective plane)") }
        } else {
            format!("{solid} (sphere)")
        };
        let lookup = |pt: [f64; 2]| cat.lookup(pt);
        match glue(&lat, &pieces, &pairs, &kind_name, Some(&lookup), true) {
            Ok(mut surf) => {
                let ch = surf.checks();
                if (projective && surf.mesh.orientable) || !ch.ok || surf.mesh.chi != if projective { 1 } else { 2 } {
                    return Err(SurfaceError::invalid(format!("{kind_name} internal check failed")));
                }
                populate_3d(&mut surf, &lat, &pieces, f3.as_deref(), v3.as_deref(), projective && solid == "pillow", projective)?;
                return Ok(surf);
            }
            Err(e) => errors.push((rec.label.clone(), e)),
        }
    }
    let retry = errors.iter().any(|(_, e)| matches!(e.kind, RefusalKind::NotFound { retry: true }));
    let ladder = if solid == "pillow" { [(1, 1), (1, 2), (2, 1), (2, 2)] } else { [(1, 0), (1, 1), (2, 0), (2, 1)] };
    let other: Vec<_> = ladder.into_iter().filter(|&z| z != sz).take(2).map(|z| format!("size={z:?}")).collect();
    let mut err = SurfaceError::not_found(
        format!("{what} of {group} not found: {solid} at size {sz:?} refused at every order-{n} centre class tried"),
        String::new(),
        retry,
    )
    .with_hint(format!("try {}", other.join(" or ")));
    err.tried = errors.iter().map(|(lab, e)| format!("size {sz:?}, centre {lab}: {}", e.reason)).collect();
    err.centres = errors.iter().map(|(lab, _)| lab.clone()).collect();
    err.causes = errors.into_iter().map(|(_, e)| e.to_string()).collect();
    Err(err)
}

/// `sphere` (`projective == false`) or `projective_plane`, keeping the refusal's kind (IMPOSSIBLE / NOT FOUND /
/// invalid argument) instead of flattening it to a string.
pub fn polyhedron(
    group: &str,
    cell: &Cell2D,
    size: Option<(i32, i32)>,
    centre_filter: Option<&str>,
    projective: bool,
) -> Result<SurfaceMesh, SurfaceError> {
    build_polyhedron(group, cell, size, centre_filter, projective).map(|s| s.mesh)
}

pub fn sphere(group: &str, cell: &Cell2D, size: Option<(i32, i32)>, centre_filter: Option<&str>) -> Result<SurfaceMesh, String> {
    build_polyhedron(group, cell, size, centre_filter, false).map(|s| s.mesh).map_err(|e| e.to_string())
}

pub fn projective_plane(group: &str, cell: &Cell2D, size: Option<(i32, i32)>, centre_filter: Option<&str>) -> Result<SurfaceMesh, String> {
    build_polyhedron(group, cell, size, centre_filter, true).map(|s| s.mesh).map_err(|e| e.to_string())
}
