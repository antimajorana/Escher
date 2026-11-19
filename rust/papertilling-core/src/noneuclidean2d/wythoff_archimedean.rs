//! Wythoff operations on `{p, q}` across `S^2`, `E^2`, and `H^2`.
//!
//! The generating point lies in the Schwarz triangle `(pi/p, pi/q, pi/2)` with mirrors `0` (face
//! centre - edge midpoint), `1` (face centre - vertex), `2` (vertex - edge midpoint); the seven ringed
//! operations and the snub follow Coxeter, Longuet-Higgins & Miller, "Uniform polyhedra", Phil. Trans.
//! R. Soc. A 246 (1954) 401-450. Spot checks used in the tests: truncated icosahedron 5.6.6 (60, 90, 32),
//! rhombicosidodecahedron 3.4.5.4 (60, 120, 62), snub cube 3^4.4 (24, 60, 38), snub dodecahedron
//! 3^4.5 (60, 150, 92) (ibid., Table of uniform polyhedra; Cromwell, *Polyhedra*, 1997, ch. 2).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::GeometryKind;

use super::unified_pq::{build_unified_pq, delta_pq};
use super::wythoff_h2::build_hyperbolic_wythoff;
use super::wythoff_planar::{assemble_wythoff_result, build_euclidean_wythoff, solve3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnubHandedness {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WythoffOp {
    Regular,
    Dual,
    Rectified,
    Truncated,
    Bitruncated,
    Cantellated,
    Omnitruncated,
    Snub { chirality: SnubHandedness },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WythoffTiling {
    pub p: usize,
    pub q: usize,
    pub op_name: String,
    pub chirality: Option<String>,
    pub vertex_symbol: String,
    pub geometry_kind: GeometryKind,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    pub euler_chi: i64,
    pub face_type_counts: BTreeMap<usize, usize>,
    pub coords_3d: Vec<[f64; 3]>,
    pub coords_2d: Vec<[f64; 2]>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub provenance: String,
}

/// Realise any of the 8 Wythoff operations on `{p, q}` across `S^2`, `E^2`, and `H^2`.
pub fn build_wythoff(p: usize, q: usize, op: WythoffOp, layers: usize) -> Result<WythoffTiling, String> {
    let delta = delta_pq(p, q)?;
    if p == 2 || q == 2 {
        return build_digon_collapsed_wythoff(p, q, op, layers);
    }
    if delta > 0 {
        return build_spherical_wythoff(p, q, op);
    }
    if delta == 0 {
        return build_euclidean_wythoff(p, q, op, layers);
    }
    build_hyperbolic_wythoff(p, q, op, layers)
}

fn build_digon_collapsed_wythoff(
    p: usize,
    q: usize,
    op: WythoffOp,
    layers: usize,
) -> Result<WythoffTiling, String> {
    match op {
        WythoffOp::Regular => {
            let u = build_unified_pq(p, q, layers)?;
            Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, u.coords_3d, u.coords_2d, u.faces))
        }
        WythoffOp::Dual => {
            let u = build_unified_pq(q, p, layers)?;
            Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, u.coords_3d, u.coords_2d, u.faces))
        }
        WythoffOp::Rectified => {
            let k = p.max(q);
            if k < 3 {
                return Err("{2,2} rectified collapses all digons to a 2-gon (no polygon mesh)".to_string());
            }
            let u = build_unified_pq(k, 2, layers)?;
            Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, u.coords_3d, u.coords_2d, u.faces))
        }
        WythoffOp::Truncated if q == 2 => {
            let u = build_unified_pq(2 * p, 2, layers)?;
            Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, u.coords_3d, u.coords_2d, u.faces))
        }
        WythoffOp::Bitruncated if p == 2 => {
            let u = build_unified_pq(2 * q, 2, layers)?;
            Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, u.coords_3d, u.coords_2d, u.faces))
        }
        WythoffOp::Truncated | WythoffOp::Bitruncated | WythoffOp::Cantellated | WythoffOp::Omnitruncated => {
            let k = if matches!(op, WythoffOp::Omnitruncated) { 2 * p.max(q) } else { p.max(q) };
            if k < 3 {
                return Err(format!("digon collapse on {{{p}, {q}}} requires order >= 3"));
            }
            build_prism(p, q, k, op)
        }
        WythoffOp::Snub { chirality } => {
            let k = p.max(q);
            if k < 3 {
                return Err(format!("antiprism digon collapse on {{{p}, {q}}} requires order >= 3"));
            }
            build_antiprism(p, q, k, op, chirality)
        }
    }
}

fn build_prism(p: usize, q: usize, k: usize, op: WythoffOp) -> Result<WythoffTiling, String> {
    let z = (PI / k as f64).sin() / (1.0 + (PI / k as f64).sin().powi(2)).sqrt();
    let r = (1.0 - z * z).sqrt();
    let mut coords_3d = Vec::with_capacity(2 * k);
    for &zz in &[z, -z] {
        for j in 0..k {
            let th = 2.0 * PI * (j as f64) / (k as f64);
            coords_3d.push([r * th.cos(), r * th.sin(), zz]);
        }
    }
    let mut faces = vec![(0..k).collect::<Vec<_>>(), (0..k).rev().map(|j| k + j).collect()];
    for j in 0..k {
        let jn = (j + 1) % k;
        faces.push(vec![j, k + j, k + jn, jn]);
    }
    let coords_2d = stereo_safe_2d(&coords_3d);
    Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, coords_3d, coords_2d, faces))
}

fn build_antiprism(
    p: usize,
    q: usize,
    k: usize,
    op: WythoffOp,
    chirality: SnubHandedness,
) -> Result<WythoffTiling, String> {
    // Uniform k-antiprism on the unit sphere: rings at z = +-h, bottom ring turned by pi/k. Equal
    // edges 4 r^2 sin^2(pi/k) = 2 r^2 (1 - cos(pi/k)) + 4 h^2 with r^2 = 1 - h^2 give
    // h^2 = A / (4 + A), A = 2 (1 + c - 2 c^2), c = cos(pi/k) (k = 3: h^2 = 1/3, the octahedron).
    let c = (PI / k as f64).cos();
    let a_coef = 2.0 * (1.0 + c - 2.0 * c * c);
    let z = (a_coef / (4.0 + a_coef)).sqrt();
    let r = (1.0 - z * z).sqrt();
    let shift = PI / k as f64;
    let mut coords_3d = Vec::with_capacity(2 * k);
    for j in 0..k {
        let th = 2.0 * PI * (j as f64) / (k as f64);
        coords_3d.push([r * th.cos(), r * th.sin(), z]);
    }
    for j in 0..k {
        let th = 2.0 * PI * (j as f64) / (k as f64) + shift;
        coords_3d.push([r * th.cos(), r * th.sin(), -z]);
    }
    let mut faces = vec![(0..k).collect::<Vec<_>>(), (0..k).rev().map(|j| k + j).collect()];
    for j in 0..k {
        let jn = (j + 1) % k;
        faces.push(vec![j, k + j, jn]);
        faces.push(vec![jn, k + j, k + jn]);
    }
    if chirality == SnubHandedness::Right {
        // Mirror image (y -> -y) with reversed cycles, as for the other snubs.
        for pt in &mut coords_3d {
            pt[1] = -pt[1];
        }
        for f in &mut faces {
            f.reverse();
        }
    }
    let coords_2d = stereo_safe_2d(&coords_3d);
    Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, coords_3d, coords_2d, faces))
}

fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-15);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn mat_vec3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

fn mat_mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    })
}

fn refl_s2(n: [f64; 3]) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| (if i == j { 1.0 } else { 0.0 }) - 2.0 * n[i] * n[j]))
}

fn stereo_safe_2d(pts: &[[f64; 3]]) -> Vec<[f64; 2]> {
    let candidates = [[0.0, 0.0, 1.0], norm3([0.23, 0.37, 0.90]), norm3([0.41, -0.29, 0.86])];
    let pole = *candidates
        .iter()
        .find(|&&pol| pts.iter().all(|&v| dot3(v, pol) < 0.995))
        .unwrap_or(&candidates[1]);
    let ref_v = if pole[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = norm3(cross3(pole, ref_v));
    let e2 = cross3(pole, e1);
    pts.iter()
        .map(|&pt| {
            let d = (1.0 - dot3(pt, pole)).max(1e-9);
            [dot3(pt, e1) / d, dot3(pt, e2) / d]
        })
        .collect()
}

fn build_spherical_wythoff(p: usize, q: usize, op: WythoffOp) -> Result<WythoffTiling, String> {
    let (ap, aq) = (PI / p as f64, PI / q as f64);
    let n0 = [0.0, 1.0, 0.0];
    let n1 = [ap.sin(), -ap.cos(), 0.0];
    let n2x = -aq.cos() / ap.sin();
    let n2 = [n2x, 0.0, (1.0 - n2x * n2x).max(0.0).sqrt()];
    let s = [refl_s2(n0), refl_s2(n1), refl_s2(n2)];
    let (rf, rv, re) = (mat_mul3(&s[1], &s[0]), mat_mul3(&s[2], &s[1]), mat_mul3(&s[0], &s[2]));
    let rv_inv = mat_mul3(&s[1], &s[2]);
    let seed_sys = |mask: [f64; 3]| -> [f64; 3] {
        norm3(solve3([
            [n0[0], n0[1], n0[2], mask[0]],
            [n1[0], n1[1], n1[2], mask[1]],
            [n2[0], n2[1], n2[2], mask[2]],
        ]))
    };
    let (cf, ce, cv) = ([0.0, 0.0, 1.0], seed_sys([0.0, 1.0, 0.0]), seed_sys([1.0, 0.0, 0.0]));

    let (seed, is_snub, right_hand) = match op {
        WythoffOp::Regular => (cv, false, false),
        WythoffOp::Dual => (cf, false, false),
        WythoffOp::Rectified => (ce, false, false),
        WythoffOp::Truncated => (seed_sys([1.0, 1.0, 0.0]), false, false),
        WythoffOp::Bitruncated => (seed_sys([0.0, 1.0, 1.0]), false, false),
        WythoffOp::Cantellated => (seed_sys([1.0, 0.0, 1.0]), false, false),
        WythoffOp::Omnitruncated => (seed_sys([1.0, 1.0, 1.0]), false, false),
        WythoffOp::Snub { chirality } => {
            let pt = |u: f64, v: f64| {
                let (w0, w1, w2) = (1.0, u.exp(), v.exp());
                norm3([
                    w0 * cf[0] + w1 * ce[0] + w2 * cv[0],
                    w0 * cf[1] + w1 * ce[1] + w2 * cv[1],
                    w0 * cf[2] + w1 * ce[2] + w2 * cv[2],
                ])
            };
            let d3 = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let res = |u: f64, v: f64| {
                let pt0 = pt(u, v);
                (d3(pt0, mat_vec3(&rf, pt0)) - d3(pt0, mat_vec3(&re, pt0)), d3(pt0, mat_vec3(&rv, pt0)) - d3(pt0, mat_vec3(&re, pt0)))
            };
            let (mut u, mut v) = (0.0_f64, 0.0_f64);
            for _ in 0..35 {
                let (r0, r1) = res(u, v);
                if r0 * r0 + r1 * r1 < 1e-24 {
                    break;
                }
                // Central differences of the residual: Newton step on the 2x2 system.
                let h = 1e-6;
                let (up, um, vp, vm) = (res(u + h, v), res(u - h, v), res(u, v + h), res(u, v - h));
                let (j00, j10, j01, j11) = ((up.0 - um.0) / (2.0 * h), (up.1 - um.1) / (2.0 * h), (vp.0 - vm.0) / (2.0 * h), (vp.1 - vm.1) / (2.0 * h));
                let det = j00 * j11 - j01 * j10;
                if !det.is_finite() || det.abs() < 1e-18 {
                    break;
                }
                u -= (j11 * r0 - j01 * r1) / det;
                v -= (-j10 * r0 + j00 * r1) / det;
            }
            let (r0, r1) = res(u, v);
            let resid = r0.hypot(r1);
            if resid.is_nan() || resid >= 1e-9 {
                return Err(format!(
                    "snub {{{p}, {q}}}: equal-edge generating point not found (residual {resid:.3e}); refusing"
                ));
            }
            (pt(u, v), true, chirality == SnubHandedness::Right)
        }
    };

    let qmat = |m: &[[f64; 3]; 3]| -> [i32; 9] {
        let mut out = [0; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[3 * i + j] = (m[i][j] * 1e5).round() as i32;
            }
        }
        out
    };
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut group = vec![id];
    let mut seen_g = HashSet::from([qmat(&id)]);
    let mut idx = 0;
    while idx < group.len() {
        let g = group[idx];
        idx += 1;
        for g_op in &s {
            let ng = mat_mul3(&g, g_op);
            if seen_g.insert(qmat(&ng)) {
                group.push(ng);
            }
        }
    }
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let active_group: Vec<[[f64; 3]; 3]> = if is_snub {
        group.iter().copied().filter(|g| det3(g) > 0.0).collect()
    } else {
        group.clone()
    };
    let qpt = |v: [f64; 3]| [(v[0] * 1e5).round() as i64, (v[1] * 1e5).round() as i64, (v[2] * 1e5).round() as i64];
    let mut vmap: HashMap<[i64; 3], usize> = HashMap::new();
    let mut coords_3d = Vec::new();
    let gen_subgroup = |gens: &[&[[f64; 3]; 3]]| -> Vec<[[f64; 3]; 3]> {
        let mut sub = vec![id];
        let mut s_seen = HashSet::from([qmat(&id)]);
        let mut k = 0;
        while k < sub.len() {
            let cur = sub[k];
            k += 1;
            for g_op in gens {
                let nx = mat_mul3(&cur, g_op);
                if s_seen.insert(qmat(&nx)) {
                    sub.push(nx);
                }
            }
        }
        sub
    };
    let sub_f = if is_snub { gen_subgroup(&[&rf]) } else { gen_subgroup(&[&s[0], &s[1]]) };
    let sub_v = if is_snub { gen_subgroup(&[&rv]) } else { gen_subgroup(&[&s[1], &s[2]]) };
    let sub_e = if is_snub { Vec::new() } else { gen_subgroup(&[&s[0], &s[2]]) };

    let mut faces = Vec::new();
    let mut seen_faces: BTreeSet<Vec<usize>> = BTreeSet::new();
    for (sub, centre0) in [(&sub_f, cf), (&sub_v, cv), (&sub_e, ce)] {
        if sub.is_empty() {
            continue;
        }
        for g in &active_group {
            let mut cyc = Vec::new();
            for h in sub {
                let pt = mat_vec3(g, mat_vec3(h, seed));
                let vid = *vmap.entry(qpt(pt)).or_insert_with(|| {
                    let id = coords_3d.len();
                    coords_3d.push(pt);
                    id
                });
                if !cyc.contains(&vid) {
                    cyc.push(vid);
                }
            }
            if cyc.len() >= 3 {
                let mut key = cyc.clone();
                key.sort_unstable();
                if seen_faces.insert(key) {
                    let c = mat_vec3(g, centre0);
                    let u = norm3(cross3(c, if c[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] }));
                    let w = cross3(c, u);
                    cyc.sort_by(|&a, &b| {
                        let aa = dot3(coords_3d[a], w).atan2(dot3(coords_3d[a], u));
                        let ab = dot3(coords_3d[b], w).atan2(dot3(coords_3d[b], u));
                        aa.total_cmp(&ab)
                    });
                    faces.push(cyc);
                }
            }
        }
    }
    if is_snub {
        for g in &active_group {
            let mut tri = [0usize; 3];
            for (slot, pt) in tri.iter_mut().zip([
                mat_vec3(g, seed),
                mat_vec3(g, mat_vec3(&rf, seed)),
                mat_vec3(g, mat_vec3(&rv_inv, seed)),
            ]) {
                *slot = *vmap.entry(qpt(pt)).or_insert_with(|| {
                    let id = coords_3d.len();
                    coords_3d.push(pt);
                    id
                });
            }
            let mut key = tri.to_vec();
            key.sort_unstable();
            if seen_faces.insert(key) {
                let (a, b, c) = (coords_3d[tri[0]], coords_3d[tri[1]], coords_3d[tri[2]]);
                let cen = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
                let nrm = cross3([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
                faces.push(if dot3(nrm, cen) >= 0.0 { tri.to_vec() } else { vec![tri[0], tri[2], tri[1]] });
            }
        }
    }
    if right_hand {
        for pt in &mut coords_3d {
            pt[1] = -pt[1];
        }
        for f in &mut faces {
            f.reverse();
        }
    }
    let coords_2d = stereo_safe_2d(&coords_3d);
    Ok(assemble_wythoff_result(p, q, op, GeometryKind::Spherical, coords_3d, coords_2d, faces))
}
