//! Combinatorial face walks, local-frame checks, Gauss-Bonnet Frank angles, and outer holonomy charge.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::defect::cone_geometry::{add2, cross2, dot2, mat2_inv, mat2_vec, norm2, same_iso, sub2};
use crate::defect::core_graph::TWO_PI;
use crate::types::{BuiltCrystal, Isometry2};

pub(crate) fn corners_order(crystal: &BuiltCrystal) -> Vec<Vec<(f64, usize)>> {
    crystal
        .adj
        .iter()
        .map(|edges| {
            let mut lst: Vec<(f64, usize)> = edges
                .iter()
                .map(|e| (e.ang.unwrap_or_else(|| e.vec[1].atan2(e.vec[0]).rem_euclid(TWO_PI)), e.to))
                .collect();
            lst.sort_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
            lst
        })
        .collect()
}

/// Combinatorial face walks read strictly from bond reference frames (`ang` / `vec`).
pub fn frame_faces(crystal: &BuiltCrystal) -> Vec<Vec<usize>> {
    let corners = corners_order(crystal);
    let nbr: Vec<Vec<usize>> = corners.iter().map(|lst| lst.iter().map(|&(_, v)| v).collect()).collect();
    let idx_map: Vec<HashMap<usize, usize>> =
        nbr.iter().map(|lst| lst.iter().copied().enumerate().map(|(k, v)| (v, k)).collect()).collect();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut faces = Vec::new();
    for (u, lst) in nbr.iter().enumerate() {
        for &v in lst {
            if seen.contains(&(u, v)) {
                continue;
            }
            let mut f = Vec::new();
            let (mut a, mut b) = (u, v);
            while seen.insert((a, b)) {
                f.push(a);
                let blst = &nbr[b];
                let k = idx_map[b][&a];
                a = b;
                b = blst[if k == 0 { blst.len() - 1 } else { k - 1 }];
            }
            faces.push(f);
        }
    }
    faces
}

pub(crate) fn outer_face_index(faces: &[Vec<usize>]) -> usize {
    let mut best = 0;
    for (i, f) in faces.iter().enumerate() {
        if f.len() > faces[best].len() {
            best = i;
        }
    }
    best
}

/// Count directed bonds failing local-frame compatibility (`0` = globally consistent crystal).
pub fn check_frames(crystal: &BuiltCrystal) -> usize {
    let mut bad = 0usize;
    for (u, edges) in crystal.adj.iter().enumerate() {
        for e in edges {
            let Some(back) = crystal.adj[e.to].iter().find(|b| b.to == u) else {
                bad += 1;
                continue;
            };
            let g = e.seam.unwrap_or(Isometry2::IDENTITY);
            let bg = back.seam.unwrap_or(Isometry2::IDENTITY);
            let ok = norm2(sub2(g.apply(add2(crystal.ref_pos[u], e.vec)), crystal.ref_pos[e.to])) < 1e-5
                && same_iso(&bg.compose(&g), &Isometry2::IDENTITY, 1e-5)
                && norm2(add2(back.vec, g.apply_linear(e.vec))) < 1e-5;
            if !ok {
                bad += 1;
            }
        }
    }
    bad
}

/// Discrete Gauss-Bonnet Frank angles (in degrees) for each combinatorial face walk.
pub fn frank_angles(crystal: &BuiltCrystal, faces: &[Vec<usize>]) -> Vec<f64> {
    let corners = corners_order(crystal);
    let idx_map: Vec<HashMap<usize, usize>> =
        corners.iter().map(|lst| lst.iter().enumerate().map(|(k, &(_, v))| (v, k)).collect()).collect();
    faces
        .iter()
        .map(|f| {
            let n = f.len();
            let mut tot = 0.0;
            for k in 0..n {
                let (prev, curr) = (f[(k + n - 1) % n], f[k]);
                let lst = &corners[curr];
                let i = idx_map[curr][&prev];
                tot += if i > 0 { lst[i].0 - lst[i - 1].0 } else { lst[0].0 + TWO_PI - lst[lst.len() - 1].0 };
            }
            (tot - (n as f64 - 2.0) * PI).to_degrees()
        })
        .collect()
}

/// Unreduced sum of interior Frank angles (in degrees) on an open crystal patch.
pub fn total_frank(crystal: &BuiltCrystal) -> f64 {
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return 0.0;
    }
    let k_out = outer_face_index(&faces);
    frank_angles(crystal, &faces).into_iter().enumerate().filter(|&(i, _)| i != k_out).map(|(_, a)| a).sum()
}

/// Holonomy `(H, S)` accumulated round a closed vertex loop.
pub fn loop_charge(crystal: &BuiltCrystal, loop_verts: &[usize]) -> (Isometry2, [f64; 2]) {
    let mut dm = Isometry2::IDENTITY;
    let mut h = Isometry2::IDENTITY;
    let mut s = [0.0, 0.0];
    for k in 0..loop_verts.len() {
        let (u, v) = (loop_verts[k], loop_verts[(k + 1) % loop_verts.len()]);
        let e = crystal.adj[u].iter().find(|b| b.to == v).expect("loop bond must exist");
        s = add2(s, dm.apply_linear(e.vec));
        let g = e.seam.unwrap_or(Isometry2::IDENTITY);
        dm = dm.compose(&g.inverse());
        h = g.compose(&h);
    }
    (h, s)
}

/// Topological charge measured along the outer CCW boundary of an open crystal patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeReport {
    pub kind: String,
    pub frank_deg: f64,
    pub holonomy: Isometry2,
    pub burgers: Option<[f64; 2]>,
    pub centre: Option<[f64; 2]>,
    pub loop_sum: [f64; 2],
}

/// Compute outer-boundary CCW holonomy and classify (`vacuum`, `dislocation`, `disclination`, `whole_turn`).
pub fn total_charge(crystal: &BuiltCrystal) -> Result<ChargeReport, String> {
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return Err("crystal has no combinatorial faces".to_string());
    }
    let k_out = outer_face_index(&faces);
    let ccw_outer: Vec<usize> = faces[k_out].iter().copied().rev().collect();
    let (h, s) = loop_charge(crystal, &ccw_outer);
    let frank = total_frank(crystal);
    let rot_deg = h.m[1][0].atan2(h.m[0][0]).to_degrees();
    if ((frank - rot_deg + 180.0).rem_euclid(360.0) - 180.0).abs() > 1e-3 {
        return Err(format!("Frank angle {frank:.4} disagrees with holonomy rotation {rot_deg:.4}"));
    }
    let frank_deg = (frank * 1e6).round() / 1e6;
    let is_trans = (h.m[0][0] - 1.0).abs() <= 1e-6
        && h.m[0][1].abs() <= 1e-6
        && h.m[1][0].abs() <= 1e-6
        && (h.m[1][1] - 1.0).abs() <= 1e-6;
    if is_trans {
        let mut b = [-h.t[0], -h.t[1]];
        for x in &mut b {
            if x.abs() < 1e-9 {
                *x = 0.0;
            }
        }
        let kind = if frank.abs() < 1e-6 {
            if norm2(b) < 1e-6 { "vacuum" } else { "dislocation" }
        } else {
            "whole_turn"
        };
        return Ok(ChargeReport { kind: kind.to_string(), frank_deg, holonomy: h, burgers: Some(b), centre: None, loop_sum: s });
    }
    let c = mat2_vec(mat2_inv([[1.0 - h.m[0][0], -h.m[0][1]], [-h.m[1][0], 1.0 - h.m[1][1]]]), h.t);
    Ok(ChargeReport { kind: "disclination".to_string(), frank_deg, holonomy: h, burgers: None, centre: Some(c), loop_sum: s })
}

/// Infer the enclosing reference disk `(centre, radius)` of `crystal.ref_pos` via convex-hull minimum enclosing circle.
pub fn infer_patch_disk(crystal: &BuiltCrystal) -> ([f64; 2], f64) {
    let mut pts = crystal.ref_pos.clone();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|a, b| norm2(sub2(*a, *b)) < 1e-9);
    if pts.is_empty() {
        return ([0.0, 0.0], 0.0);
    }
    if pts.len() == 1 {
        return (pts[0], 0.0);
    }
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for &p in &pts {
        while hull.len() >= 2 && cross2(sub2(hull[hull.len() - 1], hull[hull.len() - 2]), sub2(p, hull[hull.len() - 2])) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len && cross2(sub2(hull[hull.len() - 1], hull[hull.len() - 2]), sub2(p, hull[hull.len() - 2])) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    let mut best = ([0.0, 0.0], f64::INFINITY);
    let covers = |c: [f64; 2], r: f64| hull.iter().all(|&q| norm2(sub2(q, c)) <= r + 1e-7);
    for i in 0..hull.len() {
        for j in (i + 1)..hull.len() {
            let c = [0.5 * (hull[i][0] + hull[j][0]), 0.5 * (hull[i][1] + hull[j][1])];
            let r = 0.5 * norm2(sub2(hull[i], hull[j]));
            if r < best.1 && covers(c, r) {
                best = (c, r);
            }
        }
    }
    for i in 0..hull.len() {
        for j in (i + 1)..hull.len() {
            for k in (j + 1)..hull.len() {
                let (a, b, c0) = (hull[i], hull[j], hull[k]);
                let d = 2.0 * (a[0] * (b[1] - c0[1]) + b[0] * (c0[1] - a[1]) + c0[0] * (a[1] - b[1]));
                if d.abs() < 1e-12 {
                    continue;
                }
                let (aa, bb, cc) = (dot2(a, a), dot2(b, b), dot2(c0, c0));
                let cen = [
                    (aa * (b[1] - c0[1]) + bb * (c0[1] - a[1]) + cc * (a[1] - b[1])) / d,
                    (aa * (c0[0] - b[0]) + bb * (a[0] - c0[0]) + cc * (b[0] - a[0])) / d,
                ];
                let r = norm2(sub2(a, cen));
                if r < best.1 && covers(cen, r) {
                    best = (cen, r);
                }
            }
        }
    }
    best
}
