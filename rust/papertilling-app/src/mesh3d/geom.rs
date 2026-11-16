//! Small geometry helpers for `mesh3d`: normals, centroids, markers and wires.

use std::collections::HashMap;

use super::Vertex3d;

pub(super) fn cone_rgba(angle_deg: f64) -> [f32; 4] {
    if angle_deg >= 0.0 { [1.0, 0.26, 0.16, 1.0] } else { [1.0, 0.80, 0.14, 1.0] }
}

pub(super) fn tint_towards(base: [f32; 4], target: [f32; 4], t: f32) -> [f32; 4] {
    [
        base[0] * (1.0 - t) + target[0] * t,
        base[1] * (1.0 - t) + target[1] * t,
        base[2] * (1.0 - t) + target[2] * t,
        base[3],
    ]
}

pub(super) fn mean_chord_len(pts: &[[f32; 3]], bonds: impl Iterator<Item = (usize, usize)>) -> f32 {
    let mut sum = 0.0_f32;
    let mut cnt = 0_usize;
    for (u, v) in bonds {
        if u < pts.len() && v < pts.len() {
            let d = dist3(pts[u], pts[v]);
            if d > 1e-6 {
                sum += d;
                cnt += 1;
            }
        }
    }
    if cnt > 0 { (sum / cnt as f32).clamp(0.06, 0.55) } else { 0.22 }
}

pub(super) fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

pub(super) fn polygon_centroid_idx(pts: &[[f32; 3]], face: &[usize]) -> Option<[f32; 3]> {
    if face.is_empty() || face.iter().any(|&u| u >= pts.len()) {
        return None;
    }
    let inv = 1.0 / (face.len() as f32);
    let mut c = [0.0_f32; 3];
    for &u in face {
        c[0] += pts[u][0] * inv;
        c[1] += pts[u][1] * inv;
        c[2] += pts[u][2] * inv;
    }
    Some(c)
}

pub(super) fn polygon_centroid_pts(ring: &[[f32; 3]]) -> [f32; 3] {
    let inv = 1.0 / (ring.len().max(1) as f32);
    let mut c = [0.0_f32; 3];
    for p in ring {
        c[0] += p[0] * inv;
        c[1] += p[1] * inv;
        c[2] += p[2] * inv;
    }
    c
}

/// Point at parameter `t` on the great-circle arc from `a` to `b` (both near the unit sphere).
pub(super) fn sph_arc(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
    [p[0] / l, p[1] / l, p[2] / l]
}

/// Number of straight pieces for a sphere arc of chord `chord`: enough to follow the sphere, at most 16.
pub(super) fn arc_subdivisions(chord: f32) -> usize {
    ((chord / 0.06).ceil() as usize).clamp(1, 16)
}

/// The closed 3D outline of `face` following its bonds' curves, or `None` when a bond has no curve.
pub(super) fn build_curved_face_ring(
    face: &[usize],
    curves: &[Vec<[f32; 3]>],
    lookup: &HashMap<(usize, usize), (usize, bool)>,
) -> Option<Vec<[f32; 3]>> {
    if curves.is_empty() {
        return None;
    }
    let mut ring = Vec::new();
    for m in 0..face.len() {
        let u = face[m];
        let v = face[(m + 1) % face.len()];
        let &(e_idx, rev) = lookup.get(&(u, v))?;
        let c = &curves[e_idx];
        if rev {
            ring.extend(c.iter().rev().take(c.len().saturating_sub(1)).copied());
        } else {
            ring.extend(c.iter().take(c.len().saturating_sub(1)).copied());
        }
    }
    (ring.len() >= 3).then_some(ring)
}

pub(super) fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-7 { [nx / len, ny / len, nz / len] } else { [0.0, 0.0, 1.0] }
}

/// Vertex normals averaged over the incident fan triangles, each flipped to agree with the running sum, so the
/// result does not depend on the face winding (non-orientable surfaces have no consistent winding).
pub(super) fn compute_smooth_vertex_normals(pts: &[[f32; 3]], faces: &[Vec<usize>]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0_f32; 3]; pts.len()];
    for face in faces {
        let Some(c) = polygon_centroid_idx(pts, face) else { continue };
        for m in 0..face.len() {
            let u0 = face[m];
            let u1 = face[(m + 1) % face.len()];
            let n = tri_normal(c, pts[u0], pts[u1]);
            for u in [u0, u1] {
                let dot = acc[u][0] * n[0] + acc[u][1] * n[1] + acc[u][2] * n[2];
                let sign = if dot < 0.0 { -1.0 } else { 1.0 };
                acc[u][0] += sign * n[0];
                acc[u][1] += sign * n[1];
                acc[u][2] += sign * n[2];
            }
        }
    }
    for n in &mut acc {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        *n = if len > 1e-6 { [n[0] / len, n[1] / len, n[2] / len] } else { [0.0, 0.0, 1.0] };
    }
    acc
}

pub(super) fn blend_normal(face_n: [f32; 3], vert_n: [f32; 3], t: f32) -> [f32; 3] {
    if t <= 0.0 {
        return face_n;
    }
    let dot = face_n[0] * vert_n[0] + face_n[1] * vert_n[1] + face_n[2] * vert_n[2];
    let vn = if dot < 0.0 { [-vert_n[0], -vert_n[1], -vert_n[2]] } else { vert_n };
    let bx = face_n[0] * (1.0 - t) + vn[0] * t;
    let by = face_n[1] * (1.0 - t) + vn[1] * t;
    let bz = face_n[2] * (1.0 - t) + vn[2] * t;
    let len = (bx * bx + by * by + bz * bz).sqrt();
    if len > 1e-6 { [bx / len, by / len, bz / len] } else { face_n }
}

pub(super) fn push_wire(out: &mut Vec<Vertex3d>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    out.push(Vertex3d { position: a, normal: [0.0, 0.0, 1.0], color });
    out.push(Vertex3d { position: b, normal: [0.0, 0.0, 1.0], color });
}

pub(super) fn append_octahedron(out: &mut Vec<Vertex3d>, c: [f32; 3], r: f32, color: [f32; 4]) {
    let v = [
        [c[0] + r, c[1], c[2]], [c[0] - r, c[1], c[2]],
        [c[0], c[1] + r, c[2]], [c[0], c[1] - r, c[2]],
        [c[0], c[1], c[2] + r], [c[0], c[1], c[2] - r],
    ];
    let tris = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ];
    for [i, j, k] in tris {
        let n = tri_normal(v[i], v[j], v[k]);
        for p in [v[i], v[j], v[k]] {
            out.push(Vertex3d { position: p, normal: n, color });
        }
    }
}

pub(super) fn append_icosahedron(out: &mut Vec<Vertex3d>, c: [f32; 3], r: f32, color: [f32; 4]) {
    let phi = 0.5 * (1.0 + 5.0_f32.sqrt());
    let s = r / (1.0 + phi * phi).sqrt();
    let p = phi * s;
    let raw = [
        [-s, p, 0.0], [s, p, 0.0], [-s, -p, 0.0], [s, -p, 0.0],
        [0.0, -s, p], [0.0, s, p], [0.0, -s, -p], [0.0, s, -p],
        [p, 0.0, -s], [p, 0.0, s], [-p, 0.0, -s], [-p, 0.0, s],
    ];
    let v: [[f32; 3]; 12] = core::array::from_fn(|i| [c[0] + raw[i][0], c[1] + raw[i][1], c[2] + raw[i][2]]);
    let tris = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for [i, j, k] in tris {
        let n = tri_normal(v[i], v[j], v[k]);
        for pos in [v[i], v[j], v[k]] {
            out.push(Vertex3d { position: pos, normal: n, color });
        }
    }
}

pub(super) fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}
