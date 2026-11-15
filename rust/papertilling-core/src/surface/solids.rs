//! Unfolded polyhedral and pillow nets, `(h, k)` bases, and 3D unit-sphere / RP2 embeddings.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use super::pieces::{ConeSurface, LatticeData, PairingSpec};
use super::{SurfaceError, add2, cross2, dot2, mat2_inv, mat2_vec, norm2, rot2, sub2};
use crate::types::SurfaceCover3D;

pub type SolidNet = (Vec<Vec<[f64; 2]>>, Vec<PairingSpec>, Option<Vec<Vec<usize>>>, Option<Vec<[f64; 3]>>);

pub(super) fn solid_name(n: u32) -> Result<&'static str, SurfaceError> {
    match n {
        6 => Ok("icosahedron"),
        4 => Ok("cube"),
        3 => Ok("octahedron"),
        2 => Ok("pillow"),
        _ => Err(SurfaceError::not_found(format!("unsupported rotation order {n}"), String::new(), false)),
    }
}

type SolidGeom = (Vec<[f64; 3]>, Vec<Vec<usize>>, Option<Vec<usize>>);
type SideIncidenceMap = HashMap<[usize; 2], Vec<(usize, usize, usize, usize)>>;

fn solid_data(name: &str) -> SolidGeom {
    let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
    let (v, m) = match name {
        "icosahedron" => {
            let mut pts = Vec::with_capacity(12);
            for a in [-1.0, 1.0] {
                for b in [-phi, phi] {
                    pts.extend([[0.0, a, b], [a, b, 0.0], [b, 0.0, a]]);
                }
            }
            (pts, 3usize)
        }
        "octahedron" => (
            vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            3,
        ),
        "cube" => {
            let mut pts = Vec::with_capacity(8);
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        pts.push([x, y, z]);
                    }
                }
            }
            (pts, 4)
        }
        _ => (vec![[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], 3),
    };
    let dist3 = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let v_ref = &v;
    let e = (0..v.len())
        .flat_map(|i| (0..v_ref.len()).filter(move |&j| i != j).map(move |j| dist3(v_ref[i], v_ref[j])))
        .fold(f64::INFINITY, f64::min);
    let mut faces = Vec::new();
    if m == 3 {
        for a in 0..v.len() {
            for b in (a + 1)..v.len() {
                for c in (b + 1)..v.len() {
                    if (dist3(v[a], v[b]) - e).abs() < 1e-6 && (dist3(v[b], v[c]) - e).abs() < 1e-6 && (dist3(v[a], v[c]) - e).abs() < 1e-6 {
                        faces.push(vec![a, b, c]);
                    }
                }
            }
        }
    } else {
        for ax in 0..3 {
            for s in [-1.0, 1.0] {
                faces.push((0..v.len()).filter(|&i| (v[i][ax] - s).abs() < 1e-9).collect());
            }
        }
    }
    for f in &mut faces {
        let len = f.len() as f64;
        let c = f.iter().fold([0.0; 3], |acc, &i| [acc[0] + v[i][0] / len, acc[1] + v[i][1] / len, acc[2] + v[i][2] / len]);
        let u = [v[f[0]][0] - c[0], v[f[0]][1] - c[1], v[f[0]][2] - c[2]];
        let w = [c[1] * u[2] - c[2] * u[1], c[2] * u[0] - c[0] * u[2], c[0] * u[1] - c[1] * u[0]];
        f.sort_by(|&i, &j| {
            let ang = |idx: usize| {
                let d = [v[idx][0] - c[0], v[idx][1] - c[1], v[idx][2] - c[2]];
                let (dw, du) = (d[0] * w[0] + d[1] * w[1] + d[2] * w[2] + 0.0, d[0] * u[0] + d[1] * u[1] + d[2] * u[2] + 0.0);
                dw.atan2(du)
            };
            ang(i).total_cmp(&ang(j))
        });
    }
    let anti = (name != "pillow").then(|| {
        (0..v.len())
            .map(|i| {
                (0..v.len())
                    .min_by(|&a, &b| dist3(v[a], [-v[i][0], -v[i][1], -v[i][2]]).total_cmp(&dist3(v[b], [-v[i][0], -v[i][1], -v[i][2]])))
                    .unwrap()
            })
            .collect()
    });
    (v, faces, anti)
}

fn short_vectors(lat: &LatticeData) -> Vec<[f64; 2]> {
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
    vs
}

fn hk_basis(lat: &LatticeData, n: u32) -> Result<([f64; 2], [f64; 2]), SurfaceError> {
    let vs = short_vectors(lat);
    let b1 = vs[0];
    if n == 2 {
        let b2 = *vs.iter().find(|&&v| cross2(b1, v) > 1e-9).ok_or_else(|| SurfaceError::invalid("collinear lattice"))?;
        return Ok((b1, b2));
    }
    let deg = if n == 4 { 90.0_f64 } else { 60.0_f64 };
    let b2 = mat2_vec(rot2(deg.to_radians()), b1);
    let f = mat2_vec(lat.a_inv, b2);
    if (f[0] - f[0].round()).abs() > 1e-6 || (f[1] - f[1].round()).abs() > 1e-6 {
        return Err(SurfaceError::not_found(
            format!("{}: b1 turned by {deg:.0} deg is not a translation of the lattice", lat.group),
            String::new(),
            false,
        ));
    }
    Ok((b1, b2))
}

fn anti_face(faces: &[Vec<usize>], anti: &[usize], f: usize) -> usize {
    let mut s: Vec<usize> = faces[f].iter().map(|&v| anti[v]).collect();
    s.sort_unstable();
    (0..faces.len())
        .find(|&g| {
            let mut sg = faces[g].clone();
            sg.sort_unstable();
            sg == s
        })
        .unwrap()
}

pub fn solid_net(
    lat: &LatticeData,
    solid: &str,
    size: (i32, i32),
    c0: [f64; 2],
    projective: bool,
) -> Result<SolidNet, SurfaceError> {
    let (h, k) = size;
    if h < 0 || k < 0 || (solid == "pillow" && (h < 1 || k < 1)) {
        return Err(SurfaceError::invalid(format!("invalid size ({h}, {k})")));
    }
    if solid == "pillow" {
        return pillow_net(lat, c0, h, k, projective);
    }
    let (v3, faces, anti) = solid_data(solid);
    let n = match solid {
        "icosahedron" => 6,
        "cube" => 4,
        _ => 3,
    };
    let (b1, b2) = hk_basis(lat, n)?;
    let t_vec = [h as f64 * b1[0] + k as f64 * b2[0], h as f64 * b1[1] + k as f64 * b2[1]];
    if norm2(t_vec) < 1e-9 {
        return Err(SurfaceError::invalid("size (0, 0)"));
    }
    let first = if solid == "cube" {
        let u = mat2_vec(rot2(std::f64::consts::FRAC_PI_2), t_vec);
        vec![c0, add2(c0, t_vec), add2(add2(c0, t_vec), u), add2(c0, u)]
    } else {
        vec![c0, add2(c0, t_vec), add2(c0, mat2_vec(rot2(std::f64::consts::FRAC_PI_3), t_vec))]
    };
    let m = faces[0].len();
    let mut pos: BTreeMap<usize, HashMap<usize, [f64; 2]>> = BTreeMap::new();
    pos.insert(0, faces[0].iter().copied().zip(first).collect());
    let mut todo = VecDeque::from([0usize]);
    while let Some(f) = todo.pop_front() {
        let cyc = &faces[f];
        for s in 0..m {
            let (u, v) = (cyc[s], cyc[(s + 1) % m]);
            let g = (0..faces.len()).find(|&h_idx| h_idx != f && faces[h_idx].contains(&u) && faces[h_idx].contains(&v)).unwrap();
            if pos.contains_key(&g)
                || (projective && anti.as_ref().is_some_and(|an| pos.keys().any(|&pf| anti_face(&faces, an, g) == pf)))
            {
                continue;
            }
            let cg = &faces[g];
            let j = cg.iter().position(|&x| x == v).unwrap();
            let order: Vec<usize> = (0..m).map(|idx| cg[(j + idx) % m]).collect();
            let mut p_pts = vec![pos[&f][&v], pos[&f][&u]];
            for _ in 0..(m - 2) {
                let (p2, p1) = (p_pts[p_pts.len() - 2], p_pts[p_pts.len() - 1]);
                p_pts.push(add2(p1, mat2_vec(rot2(std::f64::consts::TAU / m as f64), sub2(p1, p2))));
            }
            pos.insert(g, order.into_iter().zip(p_pts).collect());
            todo.push_back(g);
        }
    }
    let ids: Vec<usize> = pos.keys().copied().collect();
    let mut pieces = Vec::with_capacity(ids.len());
    let mut side_keys: Vec<[usize; 2]> = Vec::new();
    let mut side_map: SideIncidenceMap = HashMap::new();
    for (p_idx, &f) in ids.iter().enumerate() {
        let cyc = &faces[f];
        pieces.push(cyc.iter().map(|v| pos[&f][v]).collect());
        for s in 0..cyc.len() {
            let (u, v) = (cyc[s], cyc[(s + 1) % cyc.len()]);
            let mut key = [u.min(v), u.max(v)];
            if projective && let Some(an) = &anti {
                key = key.min([an[u].min(an[v]), an[u].max(an[v])]);
            }
            if !side_map.contains_key(&key) {
                side_keys.push(key);
            }
            side_map.entry(key).or_default().push((p_idx, s, u, v));
        }
    }
    let mut pairs = Vec::with_capacity(side_keys.len());
    for key in side_keys {
        let lst = &side_map[&key];
        let ((k1, s1, u1, _), (k2, s2, u2, v2)) = (lst[0], lst[1]);
        let tgt = if u1 == u2 || u1 == v2 { u1 } else { anti.as_ref().unwrap()[u1] };
        pairs.push(((k1, s1), (k2, s2), tgt == v2));
    }
    Ok((pieces, pairs, Some(ids.into_iter().map(|f| faces[f].clone()).collect()), Some(v3)))
}

fn pillow_net(lat: &LatticeData, c0: [f64; 2], h: i32, k: i32, projective: bool) -> Result<SolidNet, SurfaceError> {
    if !projective {
        let (b1, b2) = hk_basis(lat, 2)?;
        let (u, w) = ([h as f64 * b1[0], h as f64 * b1[1]], [k as f64 * b2[0], k as f64 * b2[1]]);
        let (u2, w2, uw) = ([2.0 * u[0], 2.0 * u[1]], [2.0 * w[0], 2.0 * w[1]], add2(u, w));
        let pieces = vec![
            vec![c0, add2(c0, u), add2(c0, w)],
            vec![add2(c0, u), add2(c0, u2), add2(c0, uw)],
            vec![add2(c0, w), add2(c0, uw), add2(c0, w2)],
            vec![add2(c0, u), add2(c0, uw), add2(c0, w)],
        ];
        let ids = vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0], vec![1, 3, 2]];
        let (v3, _, _) = solid_data("pillow");
        let mut keys = Vec::new();
        let mut sides: SideIncidenceMap = HashMap::new();
        for (p, cyc) in ids.iter().enumerate() {
            for s in 0..3 {
                let (a, b) = (cyc[s], cyc[(s + 1) % 3]);
                let key = [a.min(b), a.max(b)];
                if !sides.contains_key(&key) {
                    keys.push(key);
                }
                sides.entry(key).or_default().push((p, s, a, b));
            }
        }
        let pairs = keys.into_iter().map(|key| {
            let (a, b) = (sides[&key][0], sides[&key][1]);
            ((a.0, a.1), (b.0, b.1), a.2 == b.3)
        }).collect();
        return Ok((pieces, pairs, Some(ids), Some(v3)));
    }
    let mut vecs: Vec<[i32; 2]> = (-4..=4).flat_map(|a| (-4..=4).filter(move |&b| a != 0 || b != 0).map(move |b| [a, b])).collect();
    vecs.sort_by(|u, v| norm2(mat2_vec(lat.a, [u[0] as f64, u[1] as f64])).total_cmp(&norm2(mat2_vec(lat.a, [v[0] as f64, v[1] as f64]))));
    let mut seen = HashSet::new();
    let mut along_vecs: Vec<[f64; 2]> = Vec::new();
    for &(mf, _) in &lat.ops {
        if mf[0][0] * mf[1][1] - mf[0][1] * mf[1][0] != -1 || !seen.insert(mf) {
            continue;
        }
        if let Some(&v) = vecs.iter().find(|&&v| mf[0][0] * v[0] + mf[0][1] * v[1] == v[0] && mf[1][0] * v[0] + mf[1][1] * v[1] == v[1]) {
            along_vecs.push(mat2_vec(lat.a, [v[0] as f64, v[1] as f64]));
        }
    }
    let mut pick = None;
    'outer: for i in 0..along_vecs.len() {
        for j in (i + 1)..along_vecs.len() {
            if dot2(along_vecs[i], along_vecs[j]).abs() < 1e-9 {
                pick = Some((along_vecs[i], along_vecs[j]));
                break 'outer;
            }
        }
    }
    let Some((a1, a2)) = pick else {
        return Err(SurfaceError::not_found(
            format!("{}: the hemi-pillow needs two perpendicular mirror or glide lines", lat.group),
            String::new(),
            false,
        ));
    };
    let u = [0.5 * h as f64 * a1[0], 0.5 * h as f64 * a1[1]];
    let mut w = [0.5 * k as f64 * a2[0], 0.5 * k as f64 * a2[1]];
    if cross2(u, w) < 0.0 {
        w = [-w[0], -w[1]];
    }
    let pieces = vec![vec![c0, add2(c0, u), add2(add2(c0, u), w), add2(c0, w)]];
    let pairs = vec![((0, 0), (0, 2), false), ((0, 1), (0, 3), false)];
    Ok((pieces, pairs, None, None))
}

pub(super) fn populate_3d(
    surf: &mut ConeSurface,
    lat: &LatticeData,
    pieces: &[Vec<[f64; 2]>],
    f3: Option<&[Vec<usize>]>,
    v3: Option<&[[f64; 3]]>,
    hemi_pillow: bool,
    projective: bool,
) -> Result<(), SurfaceError> {
    let n = surf.mesh.sites.len();
    let mut xyz = vec![[0.0; 3]; n];
    if hemi_pillow {
        let p = &pieces[0];
        let (u, w) = (sub2(p[1], p[0]), sub2(p[3], p[0]));
        let m_inv = mat2_inv([[u[0], w[0]], [u[1], w[1]]]).map_err(SurfaceError::invalid)?;
        for (s, cps) in surf.site_copies.iter().enumerate() {
            let &(_, lab) = cps.iter().min().unwrap();
            let uv = mat2_vec(m_inv, sub2(lat.pos(lab), p[0]));
            let (x, y) = (2.0 * uv[0] - 1.0, 2.0 * uv[1] - 1.0);
            let (px, py) = (x * (1.0 - 0.5 * y * y).max(0.0).sqrt(), y * (1.0 - 0.5 * x * x).max(0.0).sqrt());
            xyz[s] = [px, py, (1.0 - px * px - py * py).max(0.0).sqrt()];
        }
    } else {
        let (faces3d, verts3d) = (f3.unwrap(), v3.unwrap());
        for (s, cps) in surf.site_copies.iter().enumerate() {
            let &(k, lab) = cps.iter().min_by_key(|&&(k, _)| k).unwrap();
            let p = &pieces[k];
            let (d1, d2) = (sub2(p[1], p[0]), sub2(p[2], p[0]));
            let m_inv = mat2_inv([[d1[0], d2[0]], [d1[1], d2[1]]]).map_err(SurfaceError::invalid)?;
            let ab = mat2_vec(m_inv, sub2(lat.pos(lab), p[0]));
            let (q0, q1, q2) = (verts3d[faces3d[k][0]], verts3d[faces3d[k][1]], verts3d[faces3d[k][2]]);
            let x = [
                q0[0] + ab[0] * (q1[0] - q0[0]) + ab[1] * (q2[0] - q0[0]),
                q0[1] + ab[0] * (q1[1] - q0[1]) + ab[1] * (q2[1] - q0[1]),
                q0[2] + ab[0] * (q1[2] - q0[2]) + ab[1] * (q2[2] - q0[2]),
            ];
            let len = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt().max(1e-12);
            xyz[s] = [x[0] / len, x[1] / len, x[2] / len];
        }
    }
    let dist3_sq = |a: [f64; 3], b: [f64; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    surf.mesh.bond_curves_3d = surf
        .mesh
        .bonds
        .iter()
        .map(|&(a, b, _)| {
            let xa = xyz[a];
            let xb = if projective && dist3_sq(xa, [-xyz[b][0], -xyz[b][1], -xyz[b][2]]) < dist3_sq(xa, xyz[b]) {
                [-xyz[b][0], -xyz[b][1], -xyz[b][2]]
            } else {
                xyz[b]
            };
            (0..=16)
                .map(|step| {
                    let t = step as f64 / 16.0;
                    let p = [(1.0 - t) * xa[0] + t * xb[0], (1.0 - t) * xa[1] + t * xb[1], (1.0 - t) * xa[2] + t * xb[2]];
                    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-12);
                    [p[0] / l, p[1] / l, p[2] / l]
                })
                .collect()
        })
        .collect();
    if projective {
        let mut cover_xyz = Vec::with_capacity(2 * n);
        cover_xyz.extend_from_slice(&xyz);
        cover_xyz.extend(xyz.iter().map(|&p| [-p[0], -p[1], -p[2]]));
        let lift = |a: usize, b: usize| -> usize {
            let xa = cover_xyz[a];
            if dist3_sq(xa, xyz[b]) <= dist3_sq(xa, [-xyz[b][0], -xyz[b][1], -xyz[b][2]]) { b } else { b + n }
        };
        let mut cbonds = Vec::with_capacity(surf.mesh.bonds.len() * 2);
        for &(a, b, _) in &surf.mesh.bonds {
            cbonds.push((a, lift(a, b)));
            cbonds.push((a + n, lift(a + n, b)));
        }
        let mut cfaces = Vec::with_capacity(surf.mesh.faces.len() * 2);
        for f in &surf.mesh.faces {
            for start in [f[0], f[0] + n] {
                let mut cyc = vec![start];
                for &b in &f[1..] {
                    cyc.push(lift(*cyc.last().unwrap(), b));
                }
                cfaces.push(cyc);
            }
        }
        let mut site_rgba = vec![[0.0_f32; 4]; 2 * n];
        for i in 0..n {
            let rgba = hsv_to_rgba(i as f32 / n.max(1) as f32);
            site_rgba[i] = rgba;
            site_rgba[i + n] = rgba;
        }
        let mut cone_sites: Vec<usize> = surf.mesh.cones.iter().filter_map(|c| c.site).flat_map(|s| [s, s + n]).collect();
        cone_sites.sort_unstable();
        cone_sites.dedup();
        let mut cone_faces: Vec<usize> = surf.mesh.cones.iter().filter_map(|c| c.face).flat_map(|f| [2 * f, 2 * f + 1]).collect();
        cone_faces.sort_unstable();
        cone_faces.dedup();
        surf.mesh.cover = Some(Box::new(SurfaceCover3D { xyz: cover_xyz, bonds: cbonds, faces: cfaces, site_rgba, cone_sites, cone_faces }));
    }
    surf.mesh.xyz = xyz;
    Ok(())
}

fn hsv_to_rgba(h: f32) -> [f32; 4] {
    let s = (h.rem_euclid(1.0)) * 6.0;
    let (i, f) = (s.floor() as i32, s - s.floor());
    let (q, t) = (1.0 - f, f);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (1.0, t, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, t),
        3 => (0.0, q, 1.0),
        4 => (t, 0.0, 1.0),
        _ => (1.0, 0.0, q),
    };
    [r, g, b, 1.0]
}
