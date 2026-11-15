//! Closed orientable surfaces of genus `g >= 1`: branched double covers of a flat torus (`genus.py`)
//! with a smooth rotatable 3D genus-`g` handlebody embedding.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use crate::surface::checks::check_surface;
use crate::surface::flat::torus;
use crate::surface::quotient::{
    add2, cross2, dom_edge, face_centre_2d, mat2_inv, mat2_vec, median, point_seg_dist, scale2,
    seg_seg_dist, sub2, vec2_norm, CellGeom,
};
use crate::types::{Cell2D, ConeMark, SurfaceMesh};

/// Build a closed orientable surface of genus `g` (`g == 1` returns `6x6` torus; `g >= 2` branched double cover).
pub fn genus_surface(cell: &Cell2D, g: usize) -> Result<SurfaceMesh, String> {
    if g < 1 {
        return Err("g must be an integer >= 1 (genus 0 is the sphere)".to_string());
    }
    genus_surface_with_options(cell, g, [6, 0], [0, 6], None)
}

/// Build a genus-`g` surface trying `report.py` default torus sizes `(6, 6)`, `(8, 8)`, `(10, 10)`.
pub fn default_genus_surface(cell: &Cell2D, g: usize) -> Result<SurfaceMesh, String> {
    if g <= 1 {
        return genus_surface(cell, g);
    }
    let mut last_err = String::new();
    for sz in [6, 8, 10] {
        match genus_surface_with_options(cell, g, [sz, 0], [0, sz], None) {
            Ok(mesh) if check_surface(&mesh).ok => return Ok(mesh),
            Ok(_) => last_err = format!("NOT FOUND: checks failed on ({sz}, {sz})"),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// Build a genus-`g` surface over torus periods `(u_vec, v_vec)` and optional explicit branch slit pairs.
pub fn genus_surface_with_options(
    cell: &Cell2D,
    g: usize,
    u_vec: [i32; 2],
    v_vec: [i32; 2],
    branch: Option<&[([f64; 2], [f64; 2])]>,
) -> Result<SurfaceMesh, String> {
    if g < 1 {
        return Err("g must be an integer >= 1".to_string());
    }
    let t_mesh = torus(cell, u_vec, v_vec)?;
    if g == 1 {
        return Ok(t_mesh);
    }
    if !check_surface(&t_mesh).ok {
        return Err(format!("NOT FOUND: the torus {u_vec:?}, {v_vec:?} is too small for this lattice"));
    }

    let geom = CellGeom::from_cell(cell)?;
    let v1 = mat2_vec(geom.a, [u_vec[0] as f64, u_vec[1] as f64]);
    let v2 = mat2_vec(geom.a, [v_vec[0] as f64, v_vec[1] as f64]);
    let u_mat = [[v1[0], v2[0]], [v1[1], v2[1]]];
    let u_inv = mat2_inv(u_mat).ok_or_else(|| "singular torus periods".to_string())?;
    let v0 = t_mesh.sites.len();

    let uv0: Vec<[f64; 2]> = t_mesh
        .uv
        .iter()
        .map(|&p| {
            let f = mat2_vec(u_inv, p);
            mat2_vec(u_mat, [f[0] - (f[0] + 1e-9).floor(), f[1] - (f[1] + 1e-9).floor()])
        })
        .collect();

    let mut vec_map: HashMap<(usize, usize), [f64; 2]> = HashMap::with_capacity(t_mesh.bonds.len() * 2);
    for &(a, b, v) in &t_mesh.bonds {
        vec_map.insert((a, b), v);
        vec_map.insert((b, a), [-v[0], -v[1]]);
    }
    let blen = median(t_mesh.bonds.iter().map(|&(_, _, v)| vec2_norm(v)).collect());
    let tol = 0.05 * blen;

    let mut centres = Vec::with_capacity(t_mesh.faces.len());
    let mut shape_centre: HashMap<Vec<i64>, [f64; 2]> = HashMap::new();
    for fc in &t_mesh.faces {
        let mut poly = Vec::with_capacity(fc.len());
        let mut cur = uv0[fc[0]];
        for k in 0..fc.len() {
            poly.push(cur);
            cur = add2(cur, vec_map[&(fc[k], fc[(k + 1) % fc.len()])]);
        }
        if vec2_norm(sub2(cur, poly[0])) > 1e-6 {
            return Err("a face of the torus does not close in the plane".to_string());
        }
        let key: Vec<i64> = poly
            .iter()
            .flat_map(|&p| {
                [
                    ((p[0] - poly[0][0]) * 1e6).round_ties_even() as i64,
                    ((p[1] - poly[0][1]) * 1e6).round_ties_even() as i64,
                ]
            })
            .collect();
        let rel = *shape_centre.entry(key).or_insert_with(|| sub2(face_centre_2d(&poly), poly[0]));
        let c = add2(poly[0], rel);
        let fr = mat2_vec(u_inv, c);
        let shift = mat2_vec(u_mat, [(fr[0] + 1e-9).floor(), (fr[1] + 1e-9).floor()]);
        centres.push(sub2(c, shift));
    }

    let shifts: Vec<[f64; 2]> = (-2..=2)
        .flat_map(|i| (-2..=2).map(move |j| mat2_vec(u_mat, [i as f64, j as f64])))
        .collect();
    let sites_all: Vec<[f64; 2]> = shifts
        .iter()
        .flat_map(|&s| uv0.iter().map(move |&p| add2(p, s)))
        .collect();

    let min_image = |d: [f64; 2]| -> [f64; 2] {
        let fr = mat2_vec(u_inv, d);
        let mut best: Option<((i64, i32), [f64; 2])> = None;
        for i in -3_i32..=3 {
            for j in -3_i32..=3 {
                let c = mat2_vec(u_mat, [fr[0] + i as f64, fr[1] + j as f64]);
                let key = ((vec2_norm(c) * 1e9).round_ties_even() as i64, i.abs() + j.abs());
                if best.as_ref().is_none_or(|(bk, _)| key < *bk) {
                    best = Some((key, c));
                }
            }
        }
        best.unwrap().1
    };

    let problem = |p: [f64; 2], q: [f64; 2], fp: usize, fq: usize, chosen: &[([f64; 2], [f64; 2], usize, usize)]| -> Option<String> {
        if fp == fq {
            return Some("both branch points are in the same face".to_string());
        }
        if chosen.iter().any(|x| fp == x.2 || fp == x.3 || fq == x.2 || fq == x.3) {
            return Some("a face already holds a branch point".to_string());
        }
        let dmin = sites_all.iter().map(|&pt| point_seg_dist(pt, p, q)).fold(f64::INFINITY, f64::min);
        if dmin < tol {
            return Some("the slit passes through a site".to_string());
        }
        for &(p2, q2, _, _) in chosen {
            for &s in &shifts {
                if seg_seg_dist(p, q, add2(p2, s), add2(q2, s)) < blen {
                    return Some("the slit comes closer than one bond length to another slit".to_string());
                }
            }
        }
        None
    };

    let n_slits = g - 1;
    let mut chosen: Vec<([f64; 2], [f64; 2], usize, usize)> = Vec::with_capacity(n_slits);
    if let Some(pairs) = branch {
        if pairs.len() != n_slits {
            return Err(format!("branch: give g - 1 = {n_slits} pairs of points"));
        }
        let nearest_face = |pt: [f64; 2]| -> usize {
            centres
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| vec2_norm(min_image(sub2(**a, pt))).total_cmp(&vec2_norm(min_image(sub2(**b, pt)))))
                .unwrap()
                .0
        };
        for &(x, y) in pairs {
            let (fp, fq) = (nearest_face(x), nearest_face(y));
            let p = centres[fp];
            let q = add2(p, min_image(sub2(centres[fq], p)));
            if let Some(why) = problem(p, q, fp, fq, &chosen) {
                return Err(format!("NOT FOUND: slit {x:?} -> {y:?}: {why}"));
            }
            chosen.push((p, q, fp, fq));
        }
    } else {
        let fcen: Vec<[f64; 2]> = centres.iter().map(|&c| mat2_vec(u_inv, c)).collect();
        let face_vsets: Vec<HashSet<usize>> =
            t_mesh.faces.iter().map(|fc| fc.iter().copied().collect()).collect();
        for k in 0..n_slits {
            let yk = (k as f64 + 0.5) / (n_slits as f64);
            let (tp, tq) = (mat2_vec(u_mat, [1.0 / 3.0, yk]), mat2_vec(u_mat, [2.0 / 3.0, yk]));
            let mut order_p: Vec<(f64, usize)> = centres.iter().enumerate().map(|(i, &c)| (vec2_norm(sub2(c, tp)), i)).collect();
            let mut order_q: Vec<(f64, usize)> = centres.iter().enumerate().map(|(i, &c)| (vec2_norm(sub2(c, tq)), i)).collect();
            order_p.sort_by(|a, b| a.0.total_cmp(&b.0));
            order_q.sort_by(|a, b| a.0.total_cmp(&b.0));
            order_p.truncate(12);
            order_q.truncate(12);
            let mut combos: Vec<(f64, usize, usize)> = order_p
                .iter()
                .flat_map(|&(dp, fp)| order_q.iter().map(move |&(dq, fq)| (dp + dq, fp, fq)))
                .collect();
            combos.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut picked = false;
            for (_, fp, fq) in combos {
                let (cp, cq) = (fcen[fp], fcen[fq]);
                if !(cp[0] > 0.02 && cp[1] > 0.02 && cp[0] < 0.98 && cp[1] < 0.98
                    && cq[0] > 0.02 && cq[1] > 0.02 && cq[0] < 0.98 && cq[1] < 0.98)
                {
                    continue;
                }
                let mut used = vec![fp, fq];
                for &(_, _, up, uq) in &chosen {
                    used.push(up);
                    used.push(uq);
                }
                let shares_site = (0..used.len()).any(|ia| {
                    (ia + 1..used.len()).any(|ib| !face_vsets[used[ia]].is_disjoint(&face_vsets[used[ib]]))
                });
                if shares_site {
                    continue;
                }
                if problem(centres[fp], centres[fq], fp, fq, &chosen).is_none() {
                    chosen.push((centres[fp], centres[fq], fp, fq));
                    picked = true;
                    break;
                }
            }
            if !picked {
                return Err(format!("NOT FOUND: no room for {n_slits} slits on torus {u_vec:?}, {v_vec:?}"));
            }
        }
    }

    let mut pmap: HashMap<(usize, usize), usize> = HashMap::with_capacity(t_mesh.bonds.len() * 2);
    for &(a, b, v) in &t_mesh.bonds {
        let (p0, p1) = (uv0[a], add2(uv0[a], v));
        let mut cross_cnt = 0usize;
        for &(p, q, _, _) in &chosen {
            for &s in &shifts {
                let (s0, s1) = (add2(p, s), add2(q, s));
                let (ds, db) = (sub2(s1, s0), sub2(p1, p0));
                if cross2(ds, sub2(p0, s0)) * cross2(ds, sub2(p1, s0)) < 0.0
                    && cross2(db, sub2(s0, p0)) * cross2(db, sub2(s1, p0)) < 0.0
                {
                    cross_cnt += 1;
                }
            }
        }
        let par = cross_cnt % 2;
        pmap.insert((a, b), par);
        pmap.insert((b, a), par);
    }

    let mut branch_faces: HashSet<usize> = HashSet::with_capacity(2 * n_slits);
    for &(_, _, fp, fq) in &chosen {
        branch_faces.insert(fp);
        branch_faces.insert(fq);
    }
    for (k, fc) in t_mesh.faces.iter().enumerate() {
        let odd = (0..fc.len()).map(|j| pmap[&(fc[j], fc[(j + 1) % fc.len()])]).sum::<usize>() % 2 == 1;
        if odd != branch_faces.contains(&k) {
            return Err("slit crossing parity mismatch at torus faces".to_string());
        }
    }

    let sites: Vec<(usize, i32, i32, usize)> = (0..=1)
        .flat_map(|s| t_mesh.sites.iter().map(move |&(i, a, b, _)| (i, a, b, s)))
        .collect();
    let offset = scale2(1.15, v1);
    let uv: Vec<[f64; 2]> = uv0.iter().copied().chain(uv0.iter().map(|&p| add2(p, offset))).collect();
    let bonds: Vec<(usize, usize, [f64; 2])> = (0..=1)
        .flat_map(|s| {
            let pmap_ref = &pmap;
            t_mesh.bonds.iter().map(move |&(a, b, v)| (a + s * v0, b + (s ^ pmap_ref[&(a, b)]) * v0, v))
        })
        .collect();

    let mut faces = Vec::new();
    let mut seen_faces: HashSet<Vec<usize>> = HashSet::new();
    let mut cones = Vec::new();
    for (k, fc) in t_mesh.faces.iter().enumerate() {
        for s0 in 0..=1 {
            let (mut ids, mut s) = (Vec::new(), s0);
            for _ in 0..2 {
                for j in 0..fc.len() {
                    ids.push(fc[j] + s * v0);
                    s ^= pmap[&(fc[j], fc[(j + 1) % fc.len()])];
                }
                if s == s0 {
                    break;
                }
            }
            let mut key = ids.clone();
            key.sort_unstable();
            key.dedup();
            if !seen_faces.insert(key) {
                continue;
            }
            let face_idx = faces.len();
            if branch_faces.contains(&k) {
                cones.push(ConeMark {
                    angle_deg: 360.0 * (1.0 - (ids.len() as f64) / (fc.len() as f64)),
                    site: None,
                    face: Some(face_idx),
                    class_label: None,
                    location: "face centre".to_string(),
                });
            }
            faces.push(ids);
        }
    }

    let plane_degree: Vec<usize> = t_mesh.plane_degree.iter().chain(&t_mesh.plane_degree).copied().collect();
    let v12 = add2(v1, v2);
    let domain_polygons = vec![
        vec![[0.0, 0.0], v1, v12, v2],
        vec![offset, add2(offset, v1), add2(offset, v12), add2(offset, v2)],
    ];
    let domain_edges = vec![
        dom_edge([0.0, 0.0], v1, "1", 0), dom_edge(v2, v12, "1", 0),
        dom_edge([0.0, 0.0], v2, "2", 1), dom_edge(v1, v12, "2", 1),
        dom_edge(offset, add2(offset, v1), "1'", 0), dom_edge(add2(offset, v2), add2(offset, v12), "1'", 0),
        dom_edge(offset, add2(offset, v2), "2'", 1), dom_edge(add2(offset, v1), add2(offset, v12), "2'", 1),
    ];
    let mut slits = Vec::with_capacity(2 * n_slits);
    for (k, &(p, q, _, _)) in chosen.iter().enumerate() {
        let label = char::from_u32(65 + (k as u32 % 26)).unwrap_or('A').to_string();
        slits.push((p, q, label.clone()));
        slits.push((add2(p, offset), add2(q, offset), label));
    }

    let (xyz, bond_curves_3d) = build_handlebody_3d(&uv0, u_inv, &chosen, &bonds);

    Ok(SurfaceMesh {
        name: format!("genus {g} surface"),
        sites,
        bonds,
        faces,
        plane_degree,
        edge_sites: Vec::new(),
        chi: 2 - 2 * (g as i32),
        orientable: true,
        closed: true,
        cones,
        uv,
        domain_polygons,
        domain_edges,
        slits,
        xyz,
        bond_curves_3d,
        cover: None,
    })
}

/// Smooth rotatable 3D genus-`g` handlebody: upper & lower horizontal tori joined by `g - 1` smooth
/// vertical cylindrical necks at the branch slits, relaxed with 40 projected Laplacian/spring steps.
fn build_handlebody_3d(
    uv0: &[[f64; 2]],
    u_inv: [[f64; 2]; 2],
    chosen: &[([f64; 2], [f64; 2], usize, usize)],
    bonds: &[(usize, usize, [f64; 2])],
) -> (Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>) {
    const R_MAJ: f64 = 2.4;
    const R_TUBE: f64 = 0.50;
    const Z_SEP: f64 = 0.85;
    const R_NECK: f64 = 0.36;
    const BETA: f64 = 8.0;

    let necks: Vec<([f64; 2], [f64; 2])> = chosen
        .iter()
        .map(|&(p, q, _, _)| {
            let f = mat2_vec(u_inv, scale2(0.5, add2(p, q)));
            let f1 = f[0].rem_euclid(1.0);
            let th = 2.0 * PI * f[1].rem_euclid(1.0);
            let w = R_MAJ + R_TUBE * (2.0 * PI * (f1 - 0.5)).sin();
            ([w * th.cos(), w * th.sin()], [th.cos(), th.sin()])
        })
        .collect();

    let sdf = |pt: [f64; 3]| -> f64 {
        let rho = (pt[0] * pt[0] + pt[1] * pt[1] + 1e-12).sqrt();
        let d_top = ((rho - R_MAJ).hypot(pt[2] - Z_SEP)) - R_TUBE;
        let d_bot = ((rho - R_MAJ).hypot(pt[2] + Z_SEP)) - R_TUBE;
        let mut m = d_top.min(d_bot);
        for &(nc, _) in &necks {
            let zc = pt[2].clamp(-Z_SEP, Z_SEP);
            let dn = ((pt[0] - nc[0]).powi(2) + (pt[1] - nc[1]).powi(2) + (pt[2] - zc).powi(2)).sqrt() - R_NECK;
            m = m.min(dn);
        }
        let mut sum = (-BETA * (d_top - m)).exp() + (-BETA * (d_bot - m)).exp();
        for &(nc, _) in &necks {
            let zc = pt[2].clamp(-Z_SEP, Z_SEP);
            let dn = ((pt[0] - nc[0]).powi(2) + (pt[1] - nc[1]).powi(2) + (pt[2] - zc).powi(2)).sqrt() - R_NECK;
            sum += (-BETA * (dn - m)).exp();
        }
        m - sum.ln() / BETA
    };

    let project = |mut pt: [f64; 3]| -> [f64; 3] {
        if pt[2].abs() <= Z_SEP {
            for &(nc, dir) in &necks {
                let (dx, dy) = (pt[0] - nc[0], pt[1] - nc[1]);
                let r2 = dx * dx + dy * dy;
                if r2 < (0.25 * R_NECK).powi(2) {
                    let r = r2.sqrt();
                    let (ux, uy) = if r > 1e-9 { (dx / r, dy / r) } else { (dir[0], dir[1]) };
                    pt[0] = nc[0] + R_NECK * ux;
                    pt[1] = nc[1] + R_NECK * uy;
                }
            }
        }
        let h = 1e-5;
        for _ in 0..3 {
            let val = sdf(pt);
            if val.abs() < 1e-7 {
                break;
            }
            let gx = (sdf([pt[0] + h, pt[1], pt[2]]) - sdf([pt[0] - h, pt[1], pt[2]])) / (2.0 * h);
            let gy = (sdf([pt[0], pt[1] + h, pt[2]]) - sdf([pt[0], pt[1] - h, pt[2]])) / (2.0 * h);
            let gz = (sdf([pt[0], pt[1], pt[2] + h]) - sdf([pt[0], pt[1], pt[2] - h])) / (2.0 * h);
            let g2 = gx * gx + gy * gy + gz * gz + 1e-12;
            let step = (val / g2).clamp(-0.35, 0.35);
            pt = [pt[0] - step * gx, pt[1] - step * gy, pt[2] - step * gz];
        }
        pt
    };

    let v0 = uv0.len();
    let v_tot = 2 * v0;
    let mut xyz = Vec::with_capacity(v_tot);
    for s in 0..=1 {
        let sgn = if s == 0 { 1.0 } else { -1.0 };
        for &p in uv0 {
            let f = mat2_vec(u_inv, p);
            let psi = 2.0 * PI * (f[0].rem_euclid(1.0) - 0.5);
            let th = 2.0 * PI * f[1].rem_euclid(1.0);
            let w = R_MAJ + R_TUBE * psi.sin();
            xyz.push(project([w * th.cos(), w * th.sin(), sgn * (Z_SEP - R_TUBE * psi.cos())]));
        }
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); v_tot];
    for &(a, b, _) in bonds {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut next_xyz = xyz.clone();
    for _ in 0..40 {
        for u in 0..v_tot {
            if adj[u].is_empty() {
                next_xyz[u] = xyz[u];
                continue;
            }
            let inv_d = 1.0 / (adj[u].len() as f64);
            let mut avg = [0.0; 3];
            for &nb in &adj[u] {
                avg = [avg[0] + xyz[nb][0], avg[1] + xyz[nb][1], avg[2] + xyz[nb][2]];
            }
            let blended = [
                0.65 * xyz[u][0] + 0.35 * inv_d * avg[0],
                0.65 * xyz[u][1] + 0.35 * inv_d * avg[1],
                0.65 * xyz[u][2] + 0.35 * inv_d * avg[2],
            ];
            next_xyz[u] = project(blended);
        }
        std::mem::swap(&mut xyz, &mut next_xyz);
    }

    let k_sub = if bonds.len() <= 1500 { 6 } else if bonds.len() <= 6000 { 3 } else { 1 };
    let curves = bonds
        .iter()
        .map(|&(a, b, _)| {
            if k_sub <= 1 {
                return vec![xyz[a], xyz[b]];
            }
            (0..=k_sub)
                .map(|j| match j {
                    0 => xyz[a],
                    _ if j == k_sub => xyz[b],
                    _ => {
                        let t = j as f64 / (k_sub as f64);
                        project([
                            xyz[a][0] + t * (xyz[b][0] - xyz[a][0]),
                            xyz[a][1] + t * (xyz[b][1] - xyz[a][1]),
                            xyz[a][2] + t * (xyz[b][2] - xyz[a][2]),
                        ])
                    }
                })
                .collect()
        })
        .collect();
    (xyz, curves)
}

#[cfg(test)]
mod tests {
    use super::{genus_surface, genus_surface_with_options};
    use crate::surface::checks::check_surface;
    use crate::types::{Bond2D, Cell2D};

    #[test]
    fn explicit_slits_and_handlebody_geometry() {
        let sq = Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                Bond2D { i: 0, j: 0, n1: -1, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 0, n1: 0, n2: -1, kind: 0 },
            ],
        };
        let g1 = genus_surface(&sq, 1).unwrap();
        assert_eq!((g1.sites.len(), g1.chi, g1.cones.len()), (36, 0, 0));

        let g2 = genus_surface_with_options(&sq, 2, [5, 0], [0, 5], Some(&[([1.5, 2.5], [3.5, 2.5])])).unwrap();
        assert!(check_surface(&g2).ok);
        assert_eq!((g2.sites.len(), g2.bonds.len(), g2.faces.len(), g2.chi), (50, 100, 48, -2));
        assert_eq!(g2.cones.len(), 2);
        assert_eq!(g2.xyz.len(), 50);
        assert_eq!(g2.bond_curves_3d.len(), 100);
    }
}
