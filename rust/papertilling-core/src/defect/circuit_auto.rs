//! Automatic dual-face Burgers-circuit finder matching `circuit.py:auto_burgers_circuit`.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use crate::defect::charge::{frame_faces, infer_patch_disk, loop_charge, outer_face_index};
use crate::defect::circuit::{CircuitResult, burgers_circuit, mobius_seam_circuit};
use crate::defect::combine_types::CoreFaceSummary;
use crate::defect::core_gallery::resolve_patch_rim;
use crate::defect::core_graph::{
    LatticeCoreInfo, TWO_PI, irregular_sites_with_rim, norm2, patch_rim, sub2,
};
use crate::defect::equivalence::inspect_crystal_cores;
use crate::defect::surgery::{DefectSpec, patch_centre_and_radius};
use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

fn mean_pts(pts: &[[f64; 2]], idxs: &[usize]) -> [f64; 2] {
    let n = idxs.len() as f64;
    [idxs.iter().map(|&i| pts[i][0]).sum::<f64>() / n, idxs.iter().map(|&i| pts[i][1]).sum::<f64>() / n]
}

fn point_in_poly(pt: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let (x, y, mut inside, n) = (pt[0], pt[1], false, poly.len());
    for i in 0..n {
        let ([x1, y1], [x2, y2]) = (poly[i], poly[(i + 1) % n]);
        if (y1 > y) != (y2 > y) && x < x1 + (y - y1) * (x2 - x1) / (y2 - y1) {
            inside = !inside;
        }
    }
    inside
}

fn winding(pt: [f64; 2], poly: &[[f64; 2]]) -> i32 {
    let (px, py, n) = (pt[0], pt[1], poly.len());
    let mut ang = 0.0;
    for i in 0..n {
        let ([ax, ay], [bx, by]) = ([poly[i][0] - px, poly[i][1] - py], [poly[(i + 1) % n][0] - px, poly[(i + 1) % n][1] - py]);
        if ax.hypot(ay) < 1e-9 || bx.hypot(by) < 1e-9 { return 0; }
        ang += (ax * by - ay * bx).atan2(ax * bx + ay * by);
    }
    (ang / TWO_PI).round() as i32
}

fn match_point_to_face(
    pt: [f64; 2],
    crystal: &BuiltCrystal,
    lmax: f64,
    cores: &[CoreFaceSummary],
    int_faces: &[Vec<usize>],
    f_charges: &[(Isometry2, [f64; 2])],
) -> (Vec<usize>, usize, [f64; 2]) {
    let (p, x_ref) = (&crystal.pos, &crystal.ref_pos);
    let best_core = cores.iter().filter_map(|c| {
        let dp = norm2(sub2(mean_pts(p, &c.sites), pt));
        let dr = if x_ref.len() == p.len() { norm2(sub2(mean_pts(x_ref, &c.sites), pt)) } else { f64::INFINITY };
        let dm = dp.min(dr);
        (dm <= 1.2 * lmax).then_some((dm, c))
    }).min_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, c)) = best_core {
        if let Some(sv) = c.site {
            let inc: Vec<usize> = int_faces.iter().enumerate().filter(|(_, f)| f.contains(&sv)).map(|(i, _)| i).collect();
            let rot_inc: Vec<usize> = inc.iter().copied().filter(|&fi| {
                let m = f_charges[fi].0.m;
                (m[0][0] - 1.0).abs() > 1e-5 || m[0][1].abs() > 1e-5 || m[1][0].abs() > 1e-5 || (m[1][1] - 1.0).abs() > 1e-5
            }).collect();
            return (inc.clone(), *rot_inc.first().unwrap_or(&inc[0]), p[sv]);
        }
        let tset: BTreeSet<usize> = c.sites.iter().copied().collect();
        for (fi, f) in int_faces.iter().enumerate() {
            if f.iter().copied().collect::<BTreeSet<_>>() == tset { return (vec![fi], fi, mean_pts(p, f)); }
        }
    }
    for (coords, use_p) in [(p.as_slice(), true), (x_ref.as_slice(), false)] {
        if coords.len() == p.len() {
            for (fi, f) in int_faces.iter().enumerate() {
                let poly: Vec<[f64; 2]> = f.iter().map(|&v| coords[v]).collect();
                if point_in_poly(pt, &poly) { return (vec![fi], fi, if use_p { pt } else { mean_pts(p, f) }); }
            }
        }
    }
    let fi = (0..int_faces.len()).min_by(|&a, &b| norm2(sub2(mean_pts(p, &int_faces[a]), pt)).total_cmp(&norm2(sub2(mean_pts(p, &int_faces[b]), pt)))).unwrap_or(0);
    (vec![fi], fi, mean_pts(p, &int_faces[fi]))
}

fn extract_boundary_cycle(int_faces: &[Vec<usize>], region: &BTreeSet<usize>) -> Option<Vec<usize>> {
    let mut dir_edges = BTreeSet::new();
    for &fi in region {
        let f = &int_faces[fi];
        for k in 0..f.len() { dir_edges.insert((f[k], f[(k + 1) % f.len()])); }
    }
    let bnd: Vec<(usize, usize)> = dir_edges.iter().copied().filter(|&(u, v)| !dir_edges.contains(&(v, u))).collect();
    if bnd.is_empty() { return None; }
    let mut out_map = BTreeMap::new();
    for &(u, v) in &bnd { if out_map.insert(u, v).is_some() { return None; } }
    let start = bnd[0].0;
    let (mut cyc, mut seen) = (vec![start], HashSet::from([start]));
    let mut cur = out_map[&start];
    while cur != start {
        if !seen.insert(cur) { return None; }
        cyc.push(cur);
        cur = *out_map.get(&cur)?;
    }
    if cyc.len() == bnd.len() { cyc.push(start); Some(cyc) } else { None }
}

fn connect_seed(seed: &BTreeSet<usize>, dual_adj: &[BTreeSet<usize>], forbidden: &BTreeSet<usize>) -> Option<BTreeSet<usize>> {
    let Some(&first) = seed.iter().next() else { return Some(BTreeSet::new()); };
    if seed.len() == 1 { return Some(seed.clone()); }
    let (mut rem, mut comp, mut dq) = (seed.clone(), BTreeSet::from([first]), VecDeque::from([first]));
    rem.remove(&first);
    while let Some(u) = dq.pop_front() {
        for &nb in &dual_adj[u] { if rem.remove(&nb) && comp.insert(nb) { dq.push_back(nb); } }
    }
    while !rem.is_empty() {
        let mut parent: BTreeMap<usize, Option<usize>> = comp.iter().map(|&fi| (fi, None)).collect();
        let mut dq2: VecDeque<usize> = comp.iter().copied().collect();
        let mut hit = None;
        while hit.is_none() && let Some(fi) = dq2.pop_front() {
            for &nb in &dual_adj[fi] {
                if !forbidden.contains(&nb) && !parent.contains_key(&nb) {
                    parent.insert(nb, Some(fi));
                    if rem.contains(&nb) { hit = Some(nb); break; }
                    dq2.push_back(nb);
                }
            }
        }
        let mut cur = Some(hit?);
        while let Some(node) = cur { comp.insert(node); rem.remove(&node); cur = parent[&node]; }
    }
    Some(comp)
}

fn fill_holes(n_faces: usize, dual_adj: &[BTreeSet<usize>], outer_adj: &BTreeSet<usize>, reg: &BTreeSet<usize>, forbidden: &BTreeSet<usize>) -> Option<BTreeSet<usize>> {
    let mut outside: BTreeSet<usize> = outer_adj.iter().copied().filter(|fi| !reg.contains(fi)).collect();
    let mut dq: VecDeque<usize> = outside.iter().copied().collect();
    while let Some(u) = dq.pop_front() {
        for &nb in &dual_adj[u] { if !reg.contains(&nb) && outside.insert(nb) { dq.push_back(nb); } }
    }
    let mut out = reg.clone();
    for fi in 0..n_faces {
        if !reg.contains(&fi) && !outside.contains(&fi) {
            if forbidden.contains(&fi) { return None; }
            out.insert(fi);
        }
    }
    Some(out)
}

pub fn auto_burgers_circuit(
    crystal: &BuiltCrystal,
    cell_info: Option<(&str, &Cell2D, Frame2D)>,
    target: &[[f64; 2]],
    exclude: &[[f64; 2]],
    radius_clearance: usize,
    specs: &[DefectSpec],
) -> Result<CircuitResult, String> {
    if crystal.is_non_orientable || crystal.adj.iter().flatten().any(|e| e.seam.is_some_and(|g| g.det() < 0.0)) {
        return mobius_seam_circuit(crystal, cell_info);
    }
    let lat_info = cell_info.map(|(_, c, _)| LatticeCoreInfo::new(c));
    let faces = frame_faces(crystal);
    let k_out = outer_face_index(&faces);
    let int_faces: Vec<Vec<usize>> = faces.iter().enumerate().filter(|&(i, _)| i != k_out).map(|(_, f)| f.clone()).collect();
    let f_charges: Vec<(Isometry2, [f64; 2])> = int_faces.iter().map(|f| loop_charge(crystal, f)).collect();
    let mut edge_owner = BTreeMap::new();
    for (fi, f) in int_faces.iter().enumerate() {
        for k in 0..f.len() { edge_owner.insert((f[k], f[(k + 1) % f.len()]), fi); }
    }
    let (mut dual_adj, mut v2f, mut outer_adj) = (vec![BTreeSet::new(); int_faces.len()], BTreeMap::<usize, BTreeSet<usize>>::new(), BTreeSet::new());
    for (fi, f) in int_faces.iter().enumerate() {
        for k in 0..f.len() {
            v2f.entry(f[k]).or_default().insert(fi);
            if let Some(&nb) = edge_owner.get(&(f[(k + 1) % f.len()], f[k])) { dual_adj[fi].insert(nb); } else { outer_adj.insert(fi); }
        }
    }
    let rim: HashSet<usize> = match &lat_info {
        Some(li) if !specs.is_empty() => resolve_patch_rim(crystal, li, specs),
        Some(li) => { let (c, r) = infer_patch_disk(crystal); patch_rim(crystal, li, c, r) }
        None => faces[k_out].iter().copied().collect(),
    };
    let rim_faces: BTreeSet<usize> = rim.iter().flat_map(|v| v2f.get(v).into_iter().flatten().copied()).collect();
    let (cores, lmax, p_all) = (inspect_crystal_cores(crystal), lat_info.as_ref().map_or(1.5, |li| li.lmax), &crystal.pos);
    let irreg: BTreeSet<usize> = if crystal.adj.iter().flatten().any(|e| e.across) {
        let ctr = lat_info.as_ref().filter(|_| !specs.is_empty())
            .and_then(|li| patch_centre_and_radius(li, specs, None, None).ok().map(|(c, _)| c))
            .unwrap_or_else(|| infer_patch_disk(crystal).0);
        let mut s: BTreeSet<usize> = cores.iter().flat_map(|c| c.sites.iter().copied()).filter(|&v| norm2(sub2(p_all[v], ctr)) <= 3.0 * lmax).collect();
        if s.is_empty() { s = (0..p_all.len()).filter(|&i| norm2(sub2(p_all[i], ctr)) <= 1.2 * lmax && !rim.contains(&i)).collect(); }
        s
    } else if let Some(ref li) = lat_info {
        irregular_sites_with_rim(crystal, &rim, li).into_iter().collect()
    } else {
        cores.iter().flat_map(|c| c.sites.iter().copied()).collect()
    };
    let (mut t_faces, mut t_strict, mut t_chk) = (BTreeSet::new(), BTreeSet::new(), Vec::new());
    if target.is_empty() {
        t_faces = irreg.iter().flat_map(|v| v2f.get(v).into_iter().flatten().copied()).collect();
        if t_faces.is_empty() {
            let mean_all = [p_all.iter().map(|q| q[0]).sum::<f64>() / p_all.len() as f64, p_all.iter().map(|q| q[1]).sum::<f64>() / p_all.len() as f64];
            let best = (0..int_faces.len()).min_by(|&a, &b| norm2(sub2(mean_pts(p_all, &int_faces[a]), mean_all)).total_cmp(&norm2(sub2(mean_pts(p_all, &int_faces[b]), mean_all)))).unwrap_or(0);
            t_faces.insert(best);
        }
        t_strict = t_faces.clone();
        t_chk = if irreg.is_empty() { vec![mean_pts(p_all, &int_faces[*t_faces.iter().next().unwrap()])] } else { irreg.iter().map(|&v| p_all[v]).collect() };
    } else {
        for &pt in target {
            let (inc, sing, pchk) = match_point_to_face(pt, crystal, lmax, &cores, &int_faces, &f_charges);
            t_faces.extend(inc); t_strict.insert(sing); t_chk.push(pchk);
        }
    }
    let (mut x_faces, mut x_strict, mut x_chk) = (BTreeSet::new(), BTreeSet::new(), Vec::new());
    for &pt in exclude {
        let (inc, sing, pchk) = match_point_to_face(pt, crystal, lmax, &cores, &int_faces, &f_charges);
        x_faces.extend(inc); x_strict.insert(sing); x_chk.push(pchk);
    }
    for clr in (0..=radius_clearance).rev() {
        let seed = if clr > 0 { &t_faces } else { &t_strict };
        let mut forb: BTreeSet<usize> = rim_faces.union(if clr > 0 { &x_faces } else { &x_strict }).copied().collect();
        if clr > 0 && !x_faces.is_empty() {
            let mut xv: BTreeSet<usize> = x_faces.iter().flat_map(|&fi| int_faces[fi].iter().copied()).collect();
            for _ in 0..(clr - 1) { xv = xv.iter().flat_map(|v| v2f.get(v).into_iter().flatten().flat_map(|&fi| int_faces[fi].iter().copied())).collect(); }
            for v in &xv { if let Some(fs) = v2f.get(v) { forb.extend(fs); } }
        }
        if !seed.is_disjoint(&forb) { continue; }
        let Some(mut reg) = connect_seed(seed, &dual_adj, &forb) else { continue; };
        for _ in 0..clr {
            let verts: Vec<usize> = reg.iter().flat_map(|&fi| int_faces[fi].iter().copied()).collect();
            for v in verts { if let Some(fs) = v2f.get(&v) { for &fi in fs { if !forb.contains(&fi) { reg.insert(fi); } } } }
        }
        let Some(filled) = fill_holes(int_faces.len(), &dual_adj, &outer_adj, &reg, if clr == 0 { &x_strict } else { &x_faces }) else { continue; };
        let Some(cyc) = extract_boundary_cycle(&int_faces, &filled) else { continue; };
        let cyc_verts = &cyc[..cyc.len() - 1];
        let poly: Vec<[f64; 2]> = cyc_verts.iter().map(|&v| p_all[v]).collect();
        let on_irreg = clr == 0 && cyc_verts.iter().any(|&v| irreg.contains(&v) && t_chk.iter().any(|&pt| norm2(sub2(p_all[v], pt)) < 1e-6));
        if (!on_irreg && !t_chk.iter().all(|&pt| winding(pt, &poly) == 1)) || x_chk.iter().any(|&pt| winding(pt, &poly) != 0) {
            continue;
        }
        return burgers_circuit(crystal, &cyc, cell_info);
    }
    Err("could not find a clean Burgers circuit separating target from exclude without hitting rim".to_string())
}
