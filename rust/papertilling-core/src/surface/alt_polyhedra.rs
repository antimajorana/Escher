//! Alternating-centre hemi-cube construction for split order-4 `G`-orbits on `RP2` (`p4g`).

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::defect::catalogue::{CentreClassEntry, DefectCatalogue};

use super::pieces::{ConeSurface, LatticeData, Net2D, PairingSpec, glue, pairing_test_with};
use super::solids::{SolidNet, populate_3d};
use super::{SurfaceError, add2, dot2, mat2_vec, norm2, rot2, sub2};

pub(super) fn find_split_order4_pair(cat: &DefectCatalogue) -> Option<(&CentreClassEntry, &CentreClassEntry)> {
    let recs: Vec<&CentreClassEntry> = cat
        .centre_classes
        .iter()
        .filter(|r| r.order == 4 && !r.gclass.is_empty())
        .collect();
    for (i, &c1) in recs.iter().enumerate() {
        for &c2 in &recs[(i + 1)..] {
            if c1.gclass == c2.gclass {
                return Some((c1, c2));
            }
        }
    }
    None
}

pub(super) fn alternating_matches_centre(
    pair: Option<(&CentreClassEntry, &CentreClassEntry)>,
    centre_filter: Option<&str>,
) -> bool {
    let Some((c1, c2)) = pair else {
        return false;
    };
    let Some(cf) = centre_filter else {
        return true;
    };
    let pair_label = format!("{}+{}", c1.label, c2.label);
    cf == c1.gclass
        || cf == pair_label
        || (cf == "face centre" && c1.location == "face centre" && c2.location == "face centre")
}

fn basis_4(lat: &LatticeData) -> Result<([f64; 2], [f64; 2]), SurfaceError> {
    let mut vs: Vec<[f64; 2]> = (-6..=6)
        .flat_map(|a| (-6..=6).filter(move |&b| a != 0 || b != 0).map(move |b| mat2_vec(lat.a, [a as f64, b as f64])))
        .collect();
    vs.sort_by(|u, v| {
        let ang = |w: [f64; 2]| {
            let mut x = (w[1] + 0.0).atan2(w[0] + 0.0).rem_euclid(std::f64::consts::TAU);
            if x >= std::f64::consts::TAU - 1e-12 {
                x = 0.0;
            }
            x
        };
        ((norm2(*u) * 1e9).round() as i64)
            .cmp(&((norm2(*v) * 1e9).round() as i64))
            .then_with(|| ang(*u).total_cmp(&ang(*v)))
    });
    let b1 = vs[0];
    let b2 = mat2_vec(rot2(std::f64::consts::FRAC_PI_2), b1);
    let f = mat2_vec(lat.a_inv, b2);
    if (f[0] - f[0].round()).abs() > 1e-6 || (f[1] - f[1].round()).abs() > 1e-6 {
        return Err(SurfaceError::not_found(
            format!("{}: b1 turned by 90 deg is not a translation of the lattice", lat.group),
            String::new(),
            false,
        ));
    }
    Ok((b1, b2))
}

fn minimal_d0_candidates(c1: [f64; 2], c2: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> Vec<[f64; 2]> {
    let diff = sub2(c2, c1);
    let mut cands = Vec::new();
    for m in -2..=2 {
        for n in -2..=2 {
            let d = add2(diff, [m as f64 * b1[0] + n as f64 * b2[0], m as f64 * b1[1] + n as f64 * b2[1]]);
            let nm = norm2(d);
            if nm > 1e-7 {
                cands.push((nm, d));
            }
        }
    }
    let Some(min_norm) = cands.iter().map(|&(nm, _)| nm).reduce(f64::min) else {
        return Vec::new();
    };
    let tol = 1e-6 * min_norm.max(1.0);
    let mut short: Vec<[f64; 2]> = Vec::new();
    for (nm, d) in cands {
        if (nm - min_norm).abs() <= tol && !short.iter().any(|&q| norm2(sub2(d, q)) < 1e-6) {
            short.push(d);
        }
    }
    let b12 = add2(b1, b2);
    short.sort_by(|&a, &b| {
        let qa = dot2(a, b1) > 1e-9 && dot2(a, b2) > 1e-9;
        let qb = dot2(b, b1) > 1e-9 && dot2(b, b2) > 1e-9;
        qb.cmp(&qa)
            .then_with(|| dot2(b, b12).total_cmp(&dot2(a, b12)))
            .then_with(|| (-a[0]).total_cmp(&(-b[0])))
            .then_with(|| (-a[1]).total_cmp(&(-b[1])))
    });
    short
}

fn cube_data() -> (Vec<[f64; 3]>, Vec<Vec<usize>>, Vec<usize>) {
    let mut v: Vec<[f64; 3]> = Vec::with_capacity(8);
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                v.push([x, y, z]);
            }
        }
    }
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(6);
    for ax in [0usize, 1, 2] {
        for s in [-1.0_f64, 1.0] {
            faces.push((0..8).filter(|&i| (v[i][ax] - s).abs() < 1e-9).collect());
        }
    }
    for f in &mut faces {
        let c = f.iter().fold([0.0; 3], |acc, &i| [acc[0] + v[i][0] / 4.0, acc[1] + v[i][1] / 4.0, acc[2] + v[i][2] / 4.0]);
        let u = [v[f[0]][0] - c[0], v[f[0]][1] - c[1], v[f[0]][2] - c[2]];
        let w = [c[1] * u[2] - c[2] * u[1], c[2] * u[0] - c[0] * u[2], c[0] * u[1] - c[1] * u[0]];
        f.sort_by(|&i, &j| {
            let ang = |idx: usize| {
                let d = [v[idx][0] - c[0], v[idx][1] - c[1], v[idx][2] - c[2]];
                (d[0] * w[0] + d[1] * w[1] + d[2] * w[2]).atan2(d[0] * u[0] + d[1] * u[1] + d[2] * u[2])
            };
            ang(i).total_cmp(&ang(j))
        });
    }
    let anti = (0..8).map(|i| 7 - i).collect();
    (v, faces, anti)
}

fn hemi_cube_net(c1_pt: [f64; 2], t_vec: [f64; 2]) -> SolidNet {
    let (v3, faces, anti) = cube_data();
    let u = mat2_vec(rot2(std::f64::consts::FRAC_PI_2), t_vec);
    let first = vec![c1_pt, add2(c1_pt, t_vec), add2(add2(c1_pt, t_vec), u), add2(c1_pt, u)];
    let anti_face = |f: usize| -> usize {
        let mut s: Vec<usize> = faces[f].iter().map(|&v| anti[v]).collect();
        s.sort_unstable();
        (0..faces.len())
            .find(|&g| {
                let mut sg = faces[g].clone();
                sg.sort_unstable();
                sg == s
            })
            .unwrap()
    };
    let mut pos: BTreeMap<usize, HashMap<usize, [f64; 2]>> = BTreeMap::new();
    pos.insert(0, faces[0].iter().copied().zip(first).collect());
    let mut todo = VecDeque::from([0usize]);
    while let Some(f) = todo.pop_front() {
        let cyc = &faces[f];
        for s in 0..4 {
            let (u_v, v_v) = (cyc[s], cyc[(s + 1) % 4]);
            let g = (0..faces.len()).find(|&h_i| h_i != f && faces[h_i].contains(&u_v) && faces[h_i].contains(&v_v)).unwrap();
            if pos.contains_key(&g) || pos.keys().any(|&pf| anti_face(g) == pf) {
                continue;
            }
            let cg = &faces[g];
            let j = cg.iter().position(|&x| x == v_v).unwrap();
            let order: Vec<usize> = (0..4).map(|idx| cg[(j + idx) % 4]).collect();
            let mut p_pts = vec![pos[&f][&v_v], pos[&f][&u_v]];
            for _ in 0..2 {
                let (p2, p1) = (p_pts[p_pts.len() - 2], p_pts[p_pts.len() - 1]);
                p_pts.push(add2(p1, mat2_vec(rot2(std::f64::consts::FRAC_PI_2), sub2(p1, p2))));
            }
            pos.insert(g, order.into_iter().zip(p_pts).collect());
            todo.push_back(g);
        }
    }
    let ids: Vec<usize> = pos.keys().copied().collect();
    type SideMap = HashMap<[usize; 2], Vec<(usize, usize, usize, usize)>>;
    let mut pieces = Vec::with_capacity(ids.len());
    let mut side_keys: Vec<[usize; 2]> = Vec::new();
    let mut side_map: SideMap = HashMap::new();
    for (p_idx, &f) in ids.iter().enumerate() {
        let cyc = &faces[f];
        pieces.push(cyc.iter().map(|v| pos[&f][v]).collect());
        for s in 0..4 {
            let (u_v, v_v) = (cyc[s], cyc[(s + 1) % 4]);
            let key = [u_v.min(v_v), u_v.max(v_v)].min([anti[u_v].min(anti[v_v]), anti[u_v].max(anti[v_v])]);
            if !side_map.contains_key(&key) {
                side_keys.push(key);
            }
            side_map.entry(key).or_default().push((p_idx, s, u_v, v_v));
        }
    }
    let pairs: Vec<PairingSpec> = side_keys
        .into_iter()
        .map(|key| {
            let lst = &side_map[&key];
            let ((k1, s1, u1, _), (k2, s2, u2, v2)) = (lst[0], lst[1]);
            let tgt = if u1 == u2 || u1 == v2 { u1 } else { anti[u1] };
            ((k1, s1), (k2, s2), tgt == v2)
        })
        .collect();
    (pieces, pairs, Some(ids.into_iter().map(|f| faces[f].clone()).collect()), Some(v3))
}

pub(super) fn build_alternating_rp2(
    lat: &LatticeData,
    cat: &DefectCatalogue,
    size: (i32, i32),
    centre_filter: Option<&str>,
    errors: &mut Vec<(String, SurfaceError)>,
) -> Option<ConeSurface> {
    let pair = find_split_order4_pair(cat);
    if !alternating_matches_centre(pair, centre_filter) {
        return None;
    }
    let (c1, c2) = pair.unwrap();
    let pair_label = format!("{}+{}", c1.label, c2.label);
    let (h, k) = size;
    if h < 1 || k < 1 {
        errors.push((pair_label, SurfaceError::not_found("alternating-centre hemi-cube requires h >= 1 and k >= 1", String::new(), true)));
        return None;
    }
    let (b1, b2) = match basis_4(lat) {
        Ok(b) => b,
        Err(e) => {
            errors.push((pair_label, e));
            return None;
        }
    };
    let d0_cands = minimal_d0_candidates(c1.point, c2.point, b1, b2);
    let d0 = d0_cands.into_iter().find(|&cand| {
        let (pieces0, pairs0, _, _) = hemi_cube_net(c1.point, cand);
        Net2D::new(&pieces0, &pairs0).is_ok_and(|net0| pairing_test_with(lat, &net0).iter().all(|r| r.ok))
    });
    let Some(d0) = d0 else {
        errors.push((pair_label, SurfaceError::not_found("no minimal alternating offset passes pairing_test at (1, 1)", String::new(), true)));
        return None;
    };
    let t_vec = add2(d0, [(h - 1) as f64 * b1[0] + (k - 1) as f64 * b2[0], (h - 1) as f64 * b1[1] + (k - 1) as f64 * b2[1]]);
    if norm2(t_vec) < 1e-9 {
        errors.push((pair_label, SurfaceError::not_found("zero alternating edge vector", String::new(), true)));
        return None;
    }
    let (pieces, pairs, f3, v3) = hemi_cube_net(c1.point, t_vec);
    let lookup = |pt: [f64; 2]| cat.centre_class_of(pt).map(|r| (r.label.clone(), r.location.clone()));
    match glue(lat, &pieces, &pairs, "hemi-cube (projective plane)", Some(&lookup), true) {
        Ok(mut surf) => {
            let ch = surf.checks();
            if surf.mesh.orientable || !ch.ok || surf.mesh.chi != 1 {
                errors.push((pair_label, SurfaceError::not_found("alternating hemi-cube checks failed", String::new(), true)));
                return None;
            }
            if let Err(e) = populate_3d(&mut surf, lat, &pieces, f3.as_deref(), v3.as_deref(), false, true) {
                errors.push((pair_label, e));
                return None;
            }
            Some(surf)
        }
        Err(e) => {
            errors.push((pair_label, e));
            None
        }
    }
}
