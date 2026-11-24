//! Oriented 3D Newell tangent projection and ear-clipping triangulation for concave sheet faces.
//!
//! Convex and centroid-star-shaped faces (`k >= 4`) preserve the documented centroid-fan
//! semantics on real lattice sites without redistributing virtual-centroid curvature.
//! Concave faces whose centroid falls outside (or on the boundary of) their kernel are
//! compatibly triangulated in their right-handed Newell tangent plane.

use std::f64::consts::PI;

use crate::defect::sheet3d_energy::{cross3, dot3, norm3, scale3, sub3};

const TWO_PI: f64 = 2.0 * PI;

/// Project `face` onto its right-handed 3D Newell tangent plane `(t1, t2)` with `t1 x t2 = +n_newell`.
///
/// Because `A_vec . (t1 x t2) = ||A_vec|| > 0`, the returned 2D coordinates are centered at `[0, 0]`
/// and oriented counter-clockwise regardless of 3D sheet orientation or cyclic walk handedness.
pub(crate) fn project_face_newell_ccw(xyz: &[[f64; 3]], face: &[usize]) -> Option<Vec<[f64; 2]>> {
    let k = face.len();
    if k < 3 || face.iter().any(|&u| u >= xyz.len()) {
        return None;
    }
    let inv = 1.0 / (k as f64);
    let mut c = [0.0_f64; 3];
    for &u in face {
        c[0] += xyz[u][0] * inv;
        c[1] += xyz[u][1] * inv;
        c[2] += xyz[u][2] * inv;
    }
    let mut n_raw = [0.0_f64; 3];
    for m in 0..k {
        let cr = cross3(sub3(xyz[face[m]], c), sub3(xyz[face[(m + 1) % k]], c));
        n_raw[0] += cr[0];
        n_raw[1] += cr[1];
        n_raw[2] += cr[2];
    }
    let a2 = norm3(n_raw);
    if !a2.is_finite() || a2 <= 1e-15 {
        return None;
    }
    let n_hat = scale3(n_raw, 1.0 / a2);

    let i = (0..3).min_by(|&a, &b| n_hat[a].abs().total_cmp(&n_hat[b].abs()))?;
    let mut axis = [0.0; 3];
    axis[i] = 1.0;
    let tangent = sub3(axis, scale3(n_hat, n_hat[i]));
    let t1 = scale3(tangent, 1.0 / norm3(tangent));
    let t2 = cross3(n_hat, t1);

    Some(
        face.iter()
            .map(|&u| {
                let rel = sub3(xyz[u], c);
                [dot3(rel, t1), dot3(rel, t2)]
            })
            .collect(),
    )
}

/// True iff the CCW 2D polygon `q` is strictly star-shaped with respect to its centroid.
pub(crate) fn is_centroid_star_shaped_ccw(q: &[[f64; 2]]) -> bool {
    let k = q.len();
    if k < 3 {
        return false;
    }
    let inv = 1.0 / (k as f64);
    let mut c = [0.0_f64; 2];
    for p in q {
        c[0] += p[0] * inv;
        c[1] += p[1] * inv;
    }
    let mut a2 = 0.0_f64;
    for m in 0..k {
        let r0 = [q[m][0] - c[0], q[m][1] - c[1]];
        let r1 = [q[(m + 1) % k][0] - c[0], q[(m + 1) % k][1] - c[1]];
        a2 += r0[0] * r1[1] - r0[1] * r1[0];
    }
    if a2 <= 1e-15 {
        return false;
    }
    let min_spoke_area2 = 64.0 * f64::EPSILON * a2;
    let mut winding = 0.0_f64;
    for m in 0..k {
        let r0 = [q[m][0] - c[0], q[m][1] - c[1]];
        let r1 = [q[(m + 1) % k][0] - c[0], q[(m + 1) % k][1] - c[1]];
        let cr = r0[0] * r1[1] - r0[1] * r1[0];
        if cr <= min_spoke_area2 {
            return false;
        }
        winding += cr.atan2(r0[0] * r1[0] + r0[1] * r1[1]);
    }
    (winding - TWO_PI).abs() < 1e-5
}

/// Ear-clipping triangulation of a CCW 2D polygon `q`, returning local index triples `[ia, ib, ic]`.
pub(crate) fn ear_clip_ccw(q: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let k = q.len();
    if k < 3 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..k).collect();
    let mut tris = Vec::with_capacity(k - 2);
    let cross2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let dist2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);

    while idx.len() > 3 {
        let m = idx.len();
        let mut best: Option<(usize, f64)> = None;
        for strict in [true, false] {
            for t in 0..m {
                let (ip, ic, inx) = (idx[(t + m - 1) % m], idx[t], idx[(t + 1) % m]);
                let (a, b, c) = (q[ip], q[ic], q[inx]);
                let perim2 = (dist2(a, b) + dist2(b, c) + dist2(c, a)).max(1e-24);
                let cr = cross2(a, b, c);
                if (strict && cr <= 1e-12 * perim2) || (!strict && cr < -1e-12 * perim2) {
                    continue;
                }
                let eps = if strict {
                    -1e-12 * cr.abs()
                } else {
                    1e-12 * perim2
                };
                let contains_other = idx.iter().any(|&iu| {
                    if iu == ip || iu == ic || iu == inx {
                        return false;
                    }
                    let p = q[iu];
                    cross2(a, b, p) >= eps && cross2(b, c, p) >= eps && cross2(c, a, p) >= eps
                });
                if !contains_other {
                    let qual = cr / perim2;
                    if best.is_none_or(|(_, bq)| qual > bq + 32.0 * f64::EPSILON) {
                        best = Some((t, qual));
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        let pick = best.map(|(t, _)| t).unwrap_or_else(|| {
            (0..m)
                .max_by(|&t1, &t2| {
                    let c1 = cross2(q[idx[(t1 + m - 1) % m]], q[idx[t1]], q[idx[(t1 + 1) % m]]);
                    let c2 = cross2(q[idx[(t2 + m - 1) % m]], q[idx[t2]], q[idx[(t2 + 1) % m]]);
                    c1.total_cmp(&c2)
                })
                .unwrap_or(0)
        });
        tris.push([idx[(pick + m - 1) % m], idx[pick], idx[(pick + 1) % m]]);
        idx.remove(pick);
    }
    tris.push([idx[0], idx[1], idx[2]]);
    tris
}

/// Returns `Some(local_ear_tris)` iff `face` (`k >= 4`) is concave (not centroid-star-shaped)
/// in its oriented Newell tangent plane.
pub(crate) fn concave_face_triangulation(
    xyz: &[[f64; 3]],
    face: &[usize],
) -> Option<Vec<[usize; 3]>> {
    if face.len() < 4 {
        return None;
    }
    let q = project_face_newell_ccw(xyz, face)?;
    (!is_centroid_star_shaped_ccw(&q)).then(|| ear_clip_ccw(&q))
}

#[inline]
fn add_tri_angles(angle_sum: &mut [f64], xyz: &[[f64; 3]], tri: [usize; 3]) {
    for m in 0..3 {
        let prev = tri[(m + 2) % 3];
        let curr = tri[m];
        let next = tri[(m + 1) % 3];
        let e1 = sub3(xyz[prev], xyz[curr]);
        let e2 = sub3(xyz[next], xyz[curr]);
        let d = (norm3(e1) * norm3(e2)).max(1e-15);
        angle_sum[curr] += (dot3(e1, e2) / d).clamp(-1.0, 1.0).acos();
    }
}

/// Discrete Gaussian curvature / angle deficit `2*pi - sum theta_{f,i}` on real interior sites.
///
/// Preserves the centroid-fan angle sum on convex / centroid-star-shaped `k >= 4` faces without
/// redistributing virtual-centroid curvature, and ear-clips concave faces in their oriented Newell plane.
pub(crate) fn compute_vertex_omega(
    xyz: &[[f64; 3]],
    faces: &[Vec<usize>],
    is_boundary: &[bool],
    material_xyz: &[[f64; 3]],
) -> Vec<f64> {
    let mut angle_sum = vec![0.0_f64; xyz.len()];
    for f in faces {
        let k = f.len();
        if k < 3 {
            continue;
        }
        if k == 3 {
            add_tri_angles(&mut angle_sum, xyz, [f[0], f[1], f[2]]);
        } else if let Some(tris) = concave_face_triangulation(material_xyz, f) {
            for [ia, ib, ic] in tris {
                add_tri_angles(&mut angle_sum, xyz, [f[ia], f[ib], f[ic]]);
            }
        } else {
            let inv = 1.0 / (k as f64);
            let mut c = [0.0_f64; 3];
            for &u in f {
                c[0] += xyz[u][0] * inv;
                c[1] += xyz[u][1] * inv;
                c[2] += xyz[u][2] * inv;
            }
            for m in 0..k {
                let prev = f[(m + k - 1) % k];
                let curr = f[m];
                let next = f[(m + 1) % k];
                let e_prev = sub3(xyz[prev], xyz[curr]);
                let e_cent = sub3(c, xyz[curr]);
                let e_next = sub3(xyz[next], xyz[curr]);
                let d1 = (norm3(e_prev) * norm3(e_cent)).max(1e-15);
                let d2 = (norm3(e_cent) * norm3(e_next)).max(1e-15);
                angle_sum[curr] += (dot3(e_prev, e_cent) / d1).clamp(-1.0, 1.0).acos()
                    + (dot3(e_cent, e_next) / d2).clamp(-1.0, 1.0).acos();
            }
        }
    }
    angle_sum
        .into_iter()
        .enumerate()
        .map(|(u, s)| {
            if is_boundary[u] || s == 0.0 {
                0.0
            } else {
                TWO_PI - s
            }
        })
        .collect()
}

fn is_simple_cycle(points: &[[f64; 2]]) -> bool {
    let n = points.len();
    let turn = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        for j in i + 2..n {
            if (j + 1) % n == i {
                continue;
            }
            let (c, d) = (points[j], points[(j + 1) % n]);
            let boxes_meet = (0..2)
                .all(|k| a[k].min(b[k]) <= c[k].max(d[k]) && c[k].min(d[k]) <= a[k].max(b[k]));
            if boxes_meet
                && turn(a, b, c) * turn(a, b, d) <= 0.0
                && turn(c, d, a) * turn(c, d, b) <= 0.0
            {
                return false;
            }
        }
    }
    true
}

/// Reconstruct interior polygon angles (radians) of a flat concave polygon from spoke & edge lengths
/// when unsigned central angles fail `sum(gamma) == 2*pi`.
pub(crate) fn reconstruct_concave_planar_angles(sp: &[f64], el: &[f64]) -> Option<Vec<f64>> {
    let d = sp.len();
    if !(4..=12).contains(&d) || el.len() != d || sp.iter().any(|&s| s <= 1e-9) {
        return None;
    }
    let mut gammas = Vec::with_capacity(d);
    for m in 0..d {
        let (a, b, side) = (sp[m], sp[(m + 1) % d], el[m]);
        let cos = (a * a + b * b - side * side) / (2.0 * a * b);
        if !cos.is_finite() || cos.abs() > 1.0 + 32.0 * f64::EPSILON {
            return None;
        }
        gammas.push(cos.clamp(-1.0, 1.0).acos());
    }
    let scale = sp.iter().copied().fold(0.0, f64::max);
    let mut result: Option<Vec<f64>> = None;
    for mask in 0..(1usize << (d - 1)) {
        let mut phis = Vec::with_capacity(d + 1);
        phis.push(0.0_f64);
        for m in 0..d {
            let sgn = if m == 0 || ((mask >> (m - 1)) & 1) != 0 {
                1.0
            } else {
                -1.0
            };
            phis.push(phis[m] + sgn * gammas[m]);
        }
        let rem = ((phis[d] + PI).rem_euclid(TWO_PI) - PI).abs();
        if rem > 1e-8 * d as f64 {
            continue;
        }
        let cx: f64 = (0..d).map(|m| sp[m] * phis[m].cos()).sum::<f64>() / (d as f64);
        let cy: f64 = (0..d).map(|m| sp[m] * phis[m].sin()).sum::<f64>() / (d as f64);
        if cx.hypot(cy) <= 1e-8 * scale {
            let mut pts: Vec<[f64; 2]> = (0..d)
                .map(|m| [sp[m] * phis[m].cos(), sp[m] * phis[m].sin()])
                .collect();
            let area2: f64 = (0..d)
                .map(|m| pts[m][0] * pts[(m + 1) % d][1] - pts[m][1] * pts[(m + 1) % d][0])
                .sum();
            if area2.abs() <= 1e-12 * scale * scale || !is_simple_cycle(&pts) {
                continue;
            }
            if area2 < 0.0 {
                for p in &mut pts {
                    p[1] = -p[1];
                }
            }
            let angles: Vec<_> = (0..d)
                .map(|m| {
                    let (prev, curr, next) = (pts[(m + d - 1) % d], pts[m], pts[(m + 1) % d]);
                    let u = [prev[0] - curr[0], prev[1] - curr[1]];
                    let v = [next[0] - curr[0], next[1] - curr[1]];
                    let ang = (v[0] * u[1] - v[1] * u[0]).atan2(v[0] * u[0] + v[1] * u[1]);
                    if ang < 0.0 { ang + TWO_PI } else { ang }
                })
                .collect();
            // Distances alone need not specify an embedding. Accept only an
            // unambiguous set of interior angles among the simple reconstructions.
            if let Some(previous) = &result {
                if previous
                    .iter()
                    .zip(&angles)
                    .any(|(a, b)| (a - b).abs() > 1e-7)
                {
                    return None;
                }
            } else {
                result = Some(angles);
            }
        }
    }
    result
}
