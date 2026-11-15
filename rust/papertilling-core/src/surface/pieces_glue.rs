//! Gluing planar lattice cells, bonds, faces, and disclination cones across a `Net2D`.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use super::pieces::{
    CentreLookup, ConeSurface, LatticeData, Net2D, PairingSpec, TOL, pairing_test_with,
};
use super::{IndexUf, SiteLabel, SurfaceError, add2, cross2, dot2, mat2_mul, mat2_vec, norm2, sub2};
use crate::types::{ConeMark, DomainEdge2D, FracOp2D, SurfaceMesh};

pub fn glue(
    lat: &LatticeData,
    raw_pieces: &[Vec<[f64; 2]>],
    pairs: &[PairingSpec],
    kind_name: &str,
    centre_lookup: Option<CentreLookup<'_>>,
    strict: bool,
) -> Result<ConeSurface, SurfaceError> {
    let net = Net2D::new(raw_pieces, pairs).map_err(SurfaceError::invalid)?;
    let bad: Vec<_> = pairing_test_with(lat, &net).into_iter().filter(|r| !r.ok).collect();
    if !bad.is_empty() {
        let detail: Vec<_> = bad
            .iter()
            .take(4)
            .map(|r| format!("sides {:?} -> {:?}: {} ({})", r.pair.0, r.pair.1, r.isometry, r.reason))
            .collect();
        return Err(SurfaceError::not_found(
            format!("{kind_name} refused for {}: {} of {} gluings of this net are not symmetries of the lattice", lat.group, bad.len(), net.pairs.len()),
            detail.join("; "),
            true,
        )
        .with_hint("try another size or centre"));
    }
    let mut gf_map = HashMap::with_capacity(net.glue_map.len());
    for (&side, &(_, _, iso)) in &net.glue_map {
        let (mf, t) = lat.frac(&iso);
        let mi = [[mf[0][0].round() as i32, mf[0][1].round() as i32], [mf[1][0].round() as i32, mf[1][1].round() as i32]];
        gf_map.insert(side, FracOp2D { mf: mi, t });
    }
    let (mut s_keys, mut s_map) = (Vec::<(usize, SiteLabel, [f64; 2])>::new(), HashMap::<(usize, SiteLabel), usize>::new());
    let (mut b_keys, mut b_map) =
        (Vec::<(usize, [SiteLabel; 2], [f64; 2], (SiteLabel, SiteLabel))>::new(), HashMap::<(usize, [SiteLabel; 2]), usize>::new());
    let (mut f_keys, mut f_map) = (Vec::<(usize, Vec<SiteLabel>, [f64; 2])>::new(), HashMap::<(usize, Vec<SiteLabel>), usize>::new());
    let margin = lat.lmax + lat.fmax + lat.bmax + 1.0;
    for (k, piece) in net.pieces.iter().enumerate() {
        let lo = [
            piece.corners.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - margin,
            piece.corners.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - margin,
        ];
        let hi = [
            piece.corners.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max) + margin,
            piece.corners.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max) + margin,
        ];
        let fr = [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]].map(|c| mat2_vec(lat.a_inv, c));
        let r1 = (fr.iter().map(|f| f[0]).fold(f64::INFINITY, f64::min).floor() as i32 - 1)
            ..=(fr.iter().map(|f| f[0]).fold(f64::NEG_INFINITY, f64::max).ceil() as i32 + 1);
        let r2 = (fr.iter().map(|f| f[1]).fold(f64::INFINITY, f64::min).floor() as i32 - 1)
            ..=(fr.iter().map(|f| f[1]).fold(f64::NEG_INFINITY, f64::max).ceil() as i32 + 1);
        for n1 in r1 {
            for n2 in r2.clone() {
                for i in 0..lat.basis.len() {
                    let (lab, x) = ((i, n1, n2), lat.pos((i, n1, n2)));
                    if piece.inside(x, TOL) && !s_map.contains_key(&(k, lab)) {
                        s_map.insert((k, lab), s_keys.len());
                        s_keys.push((k, lab, x));
                    }
                }
                for &(i, j, d1, d2) in &lat.bonds {
                    let (a, b) = ((i, n1, n2), (j, n1 + d1, n2 + d2));
                    let (pa, pb) = (lat.pos(a), lat.pos(b));
                    let x = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                    if piece.inside(x, TOL) {
                        if piece.corner_at(x, 1e-6).is_some() {
                            return Err(SurfaceError::not_found(
                                format!("{kind_name} refused for {}: a cone point sits on the midpoint of a bond ([{:.4}, {:.4}])", lat.group, x[0], x[1]),
                                String::new(),
                                true,
                            )
                            .with_hint("try another centre"));
                        }
                        let bkey = if a <= b { [a, b] } else { [b, a] };
                        if let std::collections::hash_map::Entry::Vacant(e) = b_map.entry((k, bkey)) {
                            e.insert(b_keys.len());
                            b_keys.push((k, bkey, x, (a, b)));
                        }
                    }
                }
                for (labs, cen) in &lat.faces {
                    let x = add2(*cen, mat2_vec(lat.a, [n1 as f64, n2 as f64]));
                    if piece.inside(x, TOL) {
                        let mut fset: Vec<SiteLabel> = labs.iter().map(|&(i, l1, l2)| (i, l1 + n1, l2 + n2)).collect();
                        fset.sort_unstable();
                        fset.dedup();
                        if let Entry::Vacant(e) = f_map.entry((k, fset.clone())) {
                            e.insert(f_keys.len());
                            f_keys.push((k, fset, x));
                        }
                    }
                }
            }
        }
    }
    let (mut uf_s, mut uf_b, mut uf_f) = (IndexUf::new(s_keys.len()), IndexUf::new(b_keys.len()), IndexUf::new(f_keys.len()));
    let mut links: Vec<Vec<(usize, [[f64; 2]; 2])>> = vec![Vec::new(); s_keys.len()];
    let mut glue_sides: Vec<_> = net.glue_map.keys().copied().collect();
    glue_sides.sort_unstable();
    for (k, s) in glue_sides {
        let ((l, _, iso), gf) = (net.glue_map[&(k, s)], &gf_map[&(k, s)]);
        for (idx, &(kk, key, x)) in s_keys.iter().enumerate() {
            if kk == k && net.pieces[k].on_side(x, s, TOL) {
                let key2 = lat.apply_frac(gf, key).map_err(SurfaceError::invalid)?;
                let idx2 = *s_map.get(&(l, key2)).ok_or_else(|| SurfaceError::invalid("missing glued site copy"))?;
                uf_s.union(idx, idx2);
                links[idx].push((idx2, iso.m));
                links[idx2].push((idx, iso.inverse().m));
            }
        }
        for (idx, &(kk, key, x, _)) in b_keys.iter().enumerate() {
            if kk == k && net.pieces[k].on_side(x, s, TOL) {
                let a2 = lat.apply_frac(gf, key[0]).map_err(SurfaceError::invalid)?;
                let b2 = lat.apply_frac(gf, key[1]).map_err(SurfaceError::invalid)?;
                let key2 = if a2 <= b2 { [a2, b2] } else { [b2, a2] };
                let idx2 = *b_map.get(&(l, key2)).ok_or_else(|| SurfaceError::invalid("missing glued bond copy"))?;
                uf_b.union(idx, idx2);
            }
        }
        for (idx, (kk, key, x)) in f_keys.iter().enumerate() {
            if *kk == k && net.pieces[k].on_side(*x, s, TOL) {
                let mut key2: Vec<SiteLabel> =
                    key.iter().map(|&a| lat.apply_frac(gf, a)).collect::<Result<_, _>>().map_err(SurfaceError::invalid)?;
                key2.sort_unstable();
                key2.dedup();
                let idx2 = *f_map.get(&(l, key2)).ok_or_else(|| SurfaceError::invalid("missing glued face copy"))?;
                uf_f.union(idx, idx2);
            }
        }
    }
    let corner_list: Vec<(usize, usize)> = net
        .pieces
        .iter()
        .enumerate()
        .flat_map(|(k, p)| (0..p.corners.len()).map(move |j| (k, j)))
        .collect();
    let corner_ids: HashMap<_, _> = corner_list.iter().copied().enumerate().map(|(i, c)| (c, i)).collect();
    let mut cuf = IndexUf::new(corner_list.len());
    for (&(k, s), &(l, _, iso)) in &net.glue_map {
        let len_k = net.pieces[k].corners.len();
        for j in [s, (s + 1) % len_k] {
            if let Some(j2) = net.pieces[l].corner_at(iso.apply(net.pieces[k].corners[j]), 1e-6) {
                cuf.union(corner_ids[&(k, j)], corner_ids[&(l, j2)]);
            }
        }
    }
    let s_roots = uf_s.roots();
    let sid_of_root: HashMap<_, _> = s_roots.iter().copied().enumerate().map(|(n, r)| (r, n)).collect();
    let site_of: Vec<usize> = (0..s_keys.len()).map(|i| sid_of_root[&uf_s.find(i)]).collect();
    let mut site_copies = vec![Vec::new(); s_roots.len()];
    for (i, &(k, lab, _)) in s_keys.iter().enumerate() {
        site_copies[site_of[i]].push((k, lab));
    }
    let s_root_set: HashSet<usize> = s_roots.iter().copied().collect();
    let sites: Vec<_> = s_roots.iter().map(|&r| (s_keys[r].1.0, s_keys[r].1.1, s_keys[r].1.2, 0)).collect();
    let uv: Vec<[f64; 2]> = s_roots.iter().map(|&r| s_keys[r].2).collect();
    let mut lrep: Vec<Option<[[f64; 2]; 2]>> = vec![None; s_keys.len()];
    for &r in &s_roots {
        lrep[r] = Some([[1.0, 0.0], [0.0, 1.0]]);
        let mut stack = vec![r];
        while let Some(c) = stack.pop() {
            let lc = lrep[c].unwrap();
            for &(c2, r_mat) in &links[c] {
                if lrep[c2].is_none() {
                    lrep[c2] = Some(mat2_mul(lc, [[r_mat[0][0], r_mat[1][0]], [r_mat[0][1], r_mat[1][1]]]));
                    stack.push(c2);
                }
            }
        }
    }
    let b_roots = uf_b.roots();
    let (mut bonds, mut bond_mid) = (Vec::with_capacity(b_roots.len()), Vec::with_capacity(b_roots.len()));
    for &r in &b_roots {
        let (k, _, x, (a, b)) = b_keys[r];
        let (ma, _, la_labs, la) = net.walk(&gf_map, k, x, lat.pos(a), vec![a], lat)?;
        let (mb, _, lb_labs, lb) = net.walk(&gf_map, k, x, lat.pos(b), vec![b], lat)?;
        let ca = *s_map.get(&(ma, la_labs[0])).ok_or_else(|| SurfaceError::invalid("bond end A missing"))?;
        let cb = *s_map.get(&(mb, lb_labs[0])).ok_or_else(|| SurfaceError::invalid("bond end B missing"))?;
        let d = sub2(lat.pos(b), lat.pos(a));
        if s_root_set.contains(&cb) && !s_root_set.contains(&ca) {
            let v = mat2_vec(mat2_mul(lrep[cb].unwrap_or([[1.0, 0.0], [0.0, 1.0]]), lb), d);
            bonds.push((site_of[cb], site_of[ca], [-v[0], -v[1]]));
        } else {
            let v = mat2_vec(mat2_mul(lrep[ca].unwrap_or([[1.0, 0.0], [0.0, 1.0]]), la), d);
            bonds.push((site_of[ca], site_of[cb], v));
        }
        bond_mid.push((k, x, a, b));
    }
    let f_roots = uf_f.roots();
    let fid_of_root: HashMap<_, _> = f_roots.iter().copied().enumerate().map(|(n, r)| (r, n)).collect();
    let mut face_sides = vec![Vec::new(); f_roots.len()];
    for (e, &(k, x, a, b)) in bond_mid.iter().enumerate() {
        for (p, q) in [(a, b), (b, a)] {
            let &(f_idx, (o1, o2)) = lat
                .dart_face
                .get(&(p.0, q.0, q.1 - p.1, q.2 - p.2))
                .ok_or_else(|| SurfaceError::invalid("missing dart face"))?;
            let ((labs, cen), sh) = (&lat.faces[f_idx], (p.1 - o1, p.2 - o2));
            let ls: Vec<SiteLabel> = labs.iter().map(|&(i, l1, l2)| (i, l1 + sh.0, l2 + sh.1)).collect();
            let (m, _, mut ls2, _) = net.walk(&gf_map, k, x, add2(*cen, mat2_vec(lat.a, [sh.0 as f64, sh.1 as f64])), ls, lat)?;
            ls2.sort_unstable();
            ls2.dedup();
            let fc = *f_map.get(&(m, ls2)).ok_or_else(|| SurfaceError::invalid("face beside bond missing after walk"))?;
            face_sides[fid_of_root[&uf_f.find(fc)]].push(e);
        }
    }
    let faces: Vec<Vec<usize>> = face_sides.iter().map(|fs| face_cycle(fs, &bonds)).collect();
    let c_roots = cuf.roots();
    let mut cones = Vec::new();
    for &vr in &c_roots {
        let cs: Vec<_> = (0..corner_list.len()).filter(|&i| cuf.find(i) == vr).map(|i| corner_list[i]).collect();
        let tot: f64 = cs
            .iter()
            .map(|&(k, j)| {
                let p = &net.pieces[k].corners;
                let (u1, u2) = (sub2(p[(j + 1) % p.len()], p[j]), sub2(p[(j + p.len() - 1) % p.len()], p[j]));
                cross2(u1, u2).atan2(dot2(u1, u2)).to_degrees()
            })
            .sum();
        let frank = 360.0 - tot;
        if frank.abs() < 1e-6 {
            continue;
        }
        let (k0, j0) = cs[0];
        let pt = net.pieces[k0].corners[j0];
        let site = s_keys.iter().enumerate().find(|(_, (kk, _, pos))| *kk == k0 && norm2(sub2(*pos, pt)) < 1e-6).map(|(i, _)| site_of[i]);
        let face = f_keys
            .iter()
            .enumerate()
            .find(|(_, (kk, _, pos))| *kk == k0 && norm2(sub2(*pos, pt)) < 1e-6)
            .map(|(i, _)| fid_of_root[&uf_f.find(i)]);
        let (class_label, location) = centre_lookup.and_then(|f| f(pt)).map(|(lbl, loc)| (Some(lbl), loc)).unwrap_or((None, String::new()));
        cones.push(ConeMark { angle_deg: (frank * 1e9).round() / 1e9, site, face, class_label, location });
    }
    let net_chi = c_roots.len() as i32 - net.pairs.len() as i32 + net.pieces.len() as i32;
    let (orientable, domain_edges) = net_topology_and_edges(&net);
    let surf = ConeSurface {
        mesh: SurfaceMesh {
            name: kind_name.to_string(),
            sites,
            bonds,
            faces,
            plane_degree: lat.plane_degree.clone(),
            edge_sites: Vec::new(),
            chi: net_chi,
            orientable,
            closed: true,
            cones,
            uv,
            domain_polygons: net.pieces.iter().map(|p| p.corners.clone()).collect(),
            domain_edges,
            slits: Vec::new(),
            xyz: Vec::new(),
            bond_curves_3d: Vec::new(),
            cover: None,
        },
        face_sides,
        net_chi,
        site_copies,
    };
    if strict {
        let ch = surf.checks();
        let bad: Vec<_> = [("no_loops", ch.no_loops), ("no_double_bonds", ch.no_double_bonds), ("faces_at_least_3", ch.faces_at_least_3)]
            .into_iter()
            .filter_map(|(nm, ok)| (!ok).then_some(nm))
            .collect();
        if !bad.is_empty() {
            return Err(SurfaceError::not_found(
                format!("{kind_name} refused for {}: the cones fold the lattice ({} failed)", lat.group, bad.join(", ")),
                String::new(),
                true,
            )
            .with_hint("try a larger size or another centre"));
        }
    }
    Ok(surf)
}

fn net_topology_and_edges(net: &Net2D) -> (bool, Vec<DomainEdge2D>) {
    let mut sign: Vec<Option<i32>> = vec![None; net.pieces.len()];
    sign[0] = Some(1);
    let (mut orientable, mut stack) = (true, vec![0usize]);
    while let Some(k) = stack.pop() {
        let sk = sign[k].unwrap();
        for s in 0..net.pieces[k].corners.len() {
            let (l, _, iso) = net.glue_map[&(k, s)];
            let want = sk * if iso.det() > 0.0 { 1 } else { -1 };
            match sign[l] {
                None => {
                    sign[l] = Some(want);
                    stack.push(l);
                }
                Some(prev) if prev != want => orientable = false,
                _ => {}
            }
        }
    }
    let (mut edges, mut num) = (Vec::new(), 0usize);
    for &(a, b, flip) in &net.pairs {
        let g = &net.glue_map[&a].2;
        if (g.m[0][0] - 1.0).abs() < 1e-6
            && g.m[0][1].abs() < 1e-6
            && g.m[1][0].abs() < 1e-6
            && (g.m[1][1] - 1.0).abs() < 1e-6
            && norm2(g.t) < 1e-9
        {
            continue;
        }
        num += 1;
        let color_idx = usize::from(g.det() <= 0.0);
        for ((k, s), fl) in [(a, false), (b, flip)] {
            let (p, q) = net.pieces[k].side(s);
            let (p, q) = if fl { (q, p) } else { (p, q) };
            edges.push(DomainEdge2D { p, q, label: num.to_string(), color_idx });
        }
    }
    (orientable, edges)
}

fn face_cycle(sides: &[usize], bonds: &[(usize, usize, [f64; 2])]) -> Vec<usize> {
    let Some((&first, tail)) = sides.split_first() else {
        return Vec::new();
    };
    let mut rest = tail.to_vec();
    let (a, b, _) = bonds[first];
    let (mut cyc, mut cur) = (vec![a], b);
    while !rest.is_empty() {
        if let Some(pos) = rest.iter().position(|&x| bonds[x].0 == cur || bonds[x].1 == cur) {
            let nxt = rest.remove(pos);
            cyc.push(cur);
            cur = if bonds[nxt].0 == cur { bonds[nxt].1 } else { bonds[nxt].0 };
        } else {
            cyc.push(cur);
            cyc.extend(rest.iter().map(|&x| bonds[x].0));
            return cyc;
        }
    }
    cyc
}
