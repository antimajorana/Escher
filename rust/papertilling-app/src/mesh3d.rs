use std::collections::HashMap;

use papertilling_core::types::SurfaceMesh;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3d {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex3d {
    pub const ZERO: Self = Self { position: [0.0; 3], normal: [0.0, 0.0, 1.0], color: [0.0; 4] };
}

#[derive(Clone, Debug, Default)]
pub struct GpuMeshData {
    pub faces: Vec<Vertex3d>,
    pub wires: Vec<Vertex3d>,
    pub center: [f32; 3],
    pub radius: f32,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
}

pub fn face_size_rgba(sides: usize, orientable: bool) -> [f32; 4] {
    let alpha = if orientable { 0.86 } else { 0.72 };
    match sides {
        3 => [0.94, 0.73, 0.24, alpha],
        4 => [0.29, 0.63, 0.90, alpha],
        5 => [0.92, 0.39, 0.34, alpha],
        6 => [0.32, 0.76, 0.65, alpha],
        7 => [0.67, 0.45, 0.88, alpha],
        8 => [0.86, 0.43, 0.70, alpha],
        _ => [0.66, 0.75, 0.35, alpha],
    }
}

pub fn mesh_from_surface(surf: &SurfaceMesh, show_cover: bool) -> GpuMeshData {
    let cover_opt = if show_cover { surf.cover.as_deref() } else { None };
    let raw_xyz: &[[f64; 3]] = match cover_opt {
        Some(cov) => &cov.xyz,
        None => &surf.xyz,
    };
    if raw_xyz.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }

    let mut min_p = [f64::INFINITY; 3];
    let mut max_p = [f64::NEG_INFINITY; 3];
    let mut absorb = |p: [f64; 3]| {
        if p.iter().all(|v| v.is_finite()) {
            for k in 0..3 {
                min_p[k] = min_p[k].min(p[k]);
                max_p[k] = max_p[k].max(p[k]);
            }
        }
    };
    for &p in raw_xyz {
        absorb(p);
    }
    if cover_opt.is_none() {
        for curve in &surf.bond_curves_3d {
            for &p in curve {
                absorb(p);
            }
        }
    }

    let center_f64: [f64; 3] = core::array::from_fn(|k| 0.5 * (min_p[k] + max_p[k]));
    let mut max_r = 0.0_f64;
    let dist_from_center = |p: [f64; 3]| {
        ((p[0] - center_f64[0]).powi(2) + (p[1] - center_f64[1]).powi(2) + (p[2] - center_f64[2]).powi(2)).sqrt()
    };
    for &p in raw_xyz {
        max_r = max_r.max(dist_from_center(p));
    }
    if cover_opt.is_none() {
        for curve in &surf.bond_curves_3d {
            for &p in curve {
                max_r = max_r.max(dist_from_center(p));
            }
        }
    }
    let radius_f64 = max_r.max(1e-6);
    let norm_pt = |p: [f64; 3]| -> [f32; 3] {
        core::array::from_fn(|k| ((p[k] - center_f64[k]) / radius_f64) as f32)
    };

    let pts: Vec<[f32; 3]> = raw_xyz.iter().copied().map(norm_pt).collect();
    let curves: Vec<Vec<[f32; 3]>> = if cover_opt.is_none() {
        surf.bond_curves_3d
            .iter()
            .map(|c| c.iter().copied().map(norm_pt).collect())
            .collect()
    } else {
        Vec::new()
    };

    let avg_bond = match cover_opt {
        Some(cov) => mean_chord_len(&pts, cov.bonds.iter().copied()),
        None => mean_chord_len(&pts, surf.bonds.iter().map(|&(u, v, _)| (u, v))),
    };
    let site_r = (avg_bond * 0.095).clamp(0.013, 0.034);
    let cone_r = (avg_bond * 0.19).clamp(0.028, 0.064);

    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();

    if let Some(cov) = cover_opt {
        let mut is_cone_site = vec![false; pts.len()];
        for &s in &cov.cone_sites {
            if s < is_cone_site.len() {
                is_cone_site[s] = true;
            }
        }
        let mut is_cone_face = vec![false; cov.faces.len()];
        for &f in &cov.cone_faces {
            if f < is_cone_face.len() {
                is_cone_face[f] = true;
            }
        }

        for (u, &pos) in pts.iter().enumerate() {
            if is_cone_site[u] {
                append_icosahedron(&mut face_vertices, pos, cone_r, [1.0, 0.28, 0.18, 1.0]);
            } else {
                let c = cov.site_rgba.get(u).copied().unwrap_or([0.22, 0.28, 0.36, 1.0]);
                append_octahedron(&mut face_vertices, pos, site_r, [c[0], c[1], c[2], 1.0]);
            }
        }
        for (f_idx, face) in cov.faces.iter().enumerate() {
            if is_cone_face[f_idx] && let Some(fc) = polygon_centroid_idx(&pts, face) {
                append_icosahedron(&mut face_vertices, fc, cone_r, [1.0, 0.32, 0.16, 1.0]);
            }
        }

        let sph_arc = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
            [p[0] / l, p[1] / l, p[2] / l]
        };
        const COV_SUB: usize = 16;

        for (f_idx, face) in cov.faces.iter().enumerate() {
            if face.len() < 3 || face.iter().any(|&u| u >= pts.len()) {
                continue;
            }
            let Some(raw_c) = polygon_centroid_idx(&pts, face) else { continue };
            let centroid = sph_arc(raw_c, raw_c, 0.0);
            let fallback = face_size_rgba(face.len(), true);
            let mut mean_col = [0.0_f32; 4];
            for &u in face {
                let cu = cov.site_rgba.get(u).copied().unwrap_or(fallback);
                for ch in 0..4 {
                    mean_col[ch] += cu[ch] / (face.len() as f32);
                }
            }
            mean_col[3] = mean_col[3].clamp(0.75, 0.92);
            if is_cone_face[f_idx] {
                mean_col = tint_towards(mean_col, [1.0, 0.45, 0.25, 0.92], 0.25);
            }
            for m in 0..face.len() {
                let u0 = face[m];
                let u1 = face[(m + 1) % face.len()];
                let mut c0 = cov.site_rgba.get(u0).copied().unwrap_or(fallback);
                let mut c1 = cov.site_rgba.get(u1).copied().unwrap_or(fallback);
                c0[3] = mean_col[3];
                c1[3] = mean_col[3];
                for s in 0..COV_SUB {
                    let (t0, t1) = (s as f32 / COV_SUB as f32, (s + 1) as f32 / COV_SUB as f32);
                    let (p0, p1) = (sph_arc(pts[u0], pts[u1], t0), sph_arc(pts[u0], pts[u1], t1));
                    let col0 = tint_towards(c0, c1, t0);
                    let col1 = tint_towards(c0, c1, t1);
                    let n = tri_normal(centroid, p0, p1);
                    face_vertices.push(Vertex3d { position: centroid, normal: n, color: mean_col });
                    face_vertices.push(Vertex3d { position: p0, normal: n, color: col0 });
                    face_vertices.push(Vertex3d { position: p1, normal: n, color: col1 });
                }
            }
        }

        for &(u, v) in &cov.bonds {
            if u < pts.len() && v < pts.len() {
                let cu = cov.site_rgba.get(u).copied().unwrap_or([0.9, 0.94, 0.98, 1.0]);
                let cv = cov.site_rgba.get(v).copied().unwrap_or([0.9, 0.94, 0.98, 1.0]);
                let wc = [
                    0.45 * 0.5 * (cu[0] + cv[0]) + 0.55 * 0.95,
                    0.45 * 0.5 * (cu[1] + cv[1]) + 0.55 * 0.97,
                    0.45 * 0.5 * (cu[2] + cv[2]) + 0.55 * 1.00,
                    0.96,
                ];
                for s in 0..COV_SUB {
                    let (t0, t1) = (s as f32 / COV_SUB as f32, (s + 1) as f32 / COV_SUB as f32);
                    push_wire(&mut wire_vertices, sph_arc(pts[u], pts[v], t0), sph_arc(pts[u], pts[v], t1), wc);
                }
            }
        }

        return GpuMeshData {
            faces: face_vertices,
            wires: wire_vertices,
            center: [center_f64[0] as f32, center_f64[1] as f32, center_f64[2] as f32],
            radius: radius_f64 as f32,
            vertex_count: pts.len(),
            edge_count: cov.bonds.len(),
            face_count: cov.faces.len(),
        };
    }

    let mut cone_site_angle: Vec<Option<f64>> = vec![None; pts.len()];
    let mut cone_face_angle: Vec<Option<f64>> = vec![None; surf.faces.len()];
    for cone in &surf.cones {
        if let Some(s) = cone.site && s < cone_site_angle.len() {
            cone_site_angle[s] = Some(cone.angle_deg);
        }
        if let Some(f) = cone.face && f < cone_face_angle.len() {
            cone_face_angle[f] = Some(cone.angle_deg);
        }
    }

    for (u, &pos) in pts.iter().enumerate() {
        if let Some(ang) = cone_site_angle[u] {
            append_icosahedron(&mut face_vertices, pos, cone_r, cone_rgba(ang));
        } else {
            append_octahedron(&mut face_vertices, pos, site_r, [0.20, 0.25, 0.33, 1.0]);
        }
    }
    for (f_idx, face) in surf.faces.iter().enumerate() {
        if let Some(ang) = cone_face_angle[f_idx]
            && let Some(fc) = polygon_centroid_idx(&pts, face)
        {
            append_icosahedron(&mut face_vertices, fc, cone_r, cone_rgba(ang));
        }
    }

    let mut curve_lookup: HashMap<(usize, usize), (usize, bool)> = HashMap::new();
    if !curves.is_empty() {
        for (e_idx, &(u, v, _)) in surf.bonds.iter().enumerate() {
            if curves.get(e_idx).is_some_and(|c| c.len() >= 2) {
                curve_lookup.entry((u, v)).or_insert((e_idx, false));
                curve_lookup.entry((v, u)).or_insert((e_idx, true));
            }
        }
    }

    let smooth_normals = compute_smooth_vertex_normals(&pts, &surf.faces);
    for (f_idx, face) in surf.faces.iter().enumerate() {
        if face.len() < 3 || face.iter().any(|&u| u >= pts.len()) {
            continue;
        }
        let mut color = face_size_rgba(face.len(), surf.orientable);
        if let Some(ang) = cone_face_angle[f_idx] {
            color = tint_towards(color, cone_rgba(ang), 0.22);
        }

        if let Some(ring) = build_curved_face_ring(face, &curves, &curve_lookup) {
            let centroid = polygon_centroid_pts(&ring);
            for m in 0..ring.len() {
                let b0 = ring[m];
                let b1 = ring[(m + 1) % ring.len()];
                let n = tri_normal(centroid, b0, b1);
                face_vertices.push(Vertex3d { position: centroid, normal: n, color });
                face_vertices.push(Vertex3d { position: b0, normal: n, color });
                face_vertices.push(Vertex3d { position: b1, normal: n, color });
            }
        } else if let Some(centroid) = polygon_centroid_idx(&pts, face) {
            let smooth_mix = if surf.closed && surf.cones.is_empty() { 0.35 } else { 0.0 };
            for m in 0..face.len() {
                let u0 = face[m];
                let u1 = face[(m + 1) % face.len()];
                let fnrm = tri_normal(centroid, pts[u0], pts[u1]);
                let n0 = blend_normal(fnrm, smooth_normals[u0], smooth_mix);
                let n1 = blend_normal(fnrm, smooth_normals[u1], smooth_mix);
                face_vertices.push(Vertex3d { position: centroid, normal: fnrm, color });
                face_vertices.push(Vertex3d { position: pts[u0], normal: n0, color });
                face_vertices.push(Vertex3d { position: pts[u1], normal: n1, color });
            }
        }
    }

    let mut is_edge_site = vec![false; pts.len()];
    for &s in &surf.edge_sites {
        if s < is_edge_site.len() {
            is_edge_site[s] = true;
        }
    }
    for (e_idx, &(u, v, _)) in surf.bonds.iter().enumerate() {
        let is_rim = u < pts.len() && v < pts.len() && is_edge_site[u] && is_edge_site[v];
        let wc = if is_rim {
            [1.0, 0.86, 0.44, 1.0]
        } else {
            [0.91, 0.95, 0.99, 0.96]
        };
        if let Some(c) = curves.get(e_idx) && c.len() >= 2 {
            for seg in c.windows(2) {
                push_wire(&mut wire_vertices, seg[0], seg[1], wc);
            }
        } else if u < pts.len() && v < pts.len() {
            push_wire(&mut wire_vertices, pts[u], pts[v], wc);
        }
    }

    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: [center_f64[0] as f32, center_f64[1] as f32, center_f64[2] as f32],
        radius: radius_f64 as f32,
        vertex_count: pts.len(),
        edge_count: surf.bonds.len(),
        face_count: surf.faces.len(),
    }
}

fn cone_rgba(angle_deg: f64) -> [f32; 4] {
    if angle_deg >= 0.0 { [1.0, 0.26, 0.16, 1.0] } else { [1.0, 0.80, 0.14, 1.0] }
}

fn tint_towards(base: [f32; 4], target: [f32; 4], t: f32) -> [f32; 4] {
    [
        base[0] * (1.0 - t) + target[0] * t,
        base[1] * (1.0 - t) + target[1] * t,
        base[2] * (1.0 - t) + target[2] * t,
        base[3],
    ]
}

fn mean_chord_len(pts: &[[f32; 3]], bonds: impl Iterator<Item = (usize, usize)>) -> f32 {
    let mut sum = 0.0_f32;
    let mut cnt = 0_usize;
    for (u, v) in bonds {
        if u < pts.len() && v < pts.len() {
            let dx = pts[v][0] - pts[u][0];
            let dy = pts[v][1] - pts[u][1];
            let dz = pts[v][2] - pts[u][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 1e-6 {
                sum += d;
                cnt += 1;
            }
        }
    }
    if cnt > 0 { (sum / cnt as f32).clamp(0.06, 0.55) } else { 0.22 }
}

fn polygon_centroid_idx(pts: &[[f32; 3]], face: &[usize]) -> Option<[f32; 3]> {
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

fn polygon_centroid_pts(ring: &[[f32; 3]]) -> [f32; 3] {
    let inv = 1.0 / (ring.len().max(1) as f32);
    let mut c = [0.0_f32; 3];
    for p in ring {
        c[0] += p[0] * inv;
        c[1] += p[1] * inv;
        c[2] += p[2] * inv;
    }
    c
}

fn build_curved_face_ring(
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

fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-7 { [nx / len, ny / len, nz / len] } else { [0.0, 0.0, 1.0] }
}

fn compute_smooth_vertex_normals(pts: &[[f32; 3]], faces: &[Vec<usize>]) -> Vec<[f32; 3]> {
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

fn blend_normal(face_n: [f32; 3], vert_n: [f32; 3], t: f32) -> [f32; 3] {
    if t <= 0.0 {
        return face_n;
    }
    let dot = face_n[0] * vert_n[0] + face_n[1] * vert_n[1] + face_n[2] * vert_n[2];
    let vn = if dot < 0.0 { [-vert_n[0], -vert_n[1], -vert_n[2]] } else { vert_n };
    let bx = face_n[0] * (1.0 - t) + vn[0] * t;
    let by = face_n[1] * (1.0 - t) + vn[1] * t;
    let bz = face_n[2] * (1.0 - t) + vn[2] * t;
    let len = (bx * bx + by * by + bz * bz).sqrt().max(1e-6);
    [bx / len, by / len, bz / len]
}

fn push_wire(out: &mut Vec<Vertex3d>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    out.push(Vertex3d { position: a, normal: [0.0, 0.0, 1.0], color });
    out.push(Vertex3d { position: b, normal: [0.0, 0.0, 1.0], color });
}

fn append_octahedron(out: &mut Vec<Vertex3d>, c: [f32; 3], r: f32, color: [f32; 4]) {
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

fn append_icosahedron(out: &mut Vec<Vertex3d>, c: [f32; 3], r: f32, color: [f32; 4]) {
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

pub fn orbit_rotation(yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let ((sy, cy), (sp, cp)) = (yaw.sin_cos(), pitch.sin_cos());
    let ry = [[cy, 0.0, -sy, 0.0], [0.0, 1.0, 0.0, 0.0], [sy, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let rx = [[1.0, 0.0, 0.0, 0.0], [0.0, cp, sp, 0.0], [0.0, -sp, cp, 0.0], [0.0, 0.0, 0.0, 1.0]];
    mat4_mul(&rx, &ry)
}

pub fn build_mvp(aspect: f32, distance: f32, rot: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let f = 1.0 / (0.5 * 45.0_f32.to_radians()).tan();
    let (near, far) = (0.15_f32, 30.0_f32);
    let proj = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ];
    let mut view_model = *rot;
    view_model[3][2] = -distance;
    mat4_mul(&proj, &view_model)
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AppBackend, SurfaceBuildParamsUi, SurfaceTargetUi};

    #[test]
    fn face_size_palette_distinguishes_3_4_5_6_7_rings() {
        let rings = [3, 4, 5, 6, 7];
        for i in 0..rings.len() {
            for j in (i + 1)..rings.len() {
                assert_ne!(face_size_rgba(rings[i], true), face_size_rgba(rings[j], true));
            }
        }
    }

    #[test]
    fn mesh_from_surface_torus_rp2_cover_and_genus2() {
        let backend = AppBackend::new();
        let kagome = backend.find_named("kagome").unwrap();
        let params = SurfaceBuildParamsUi::default();

        let (torus, rep_t) = backend.build_surface(kagome, SurfaceTargetUi::Torus, &params).unwrap();
        assert!(rep_t.ok && torus.chi == 0);
        let gpu_torus = mesh_from_surface(&torus, false);
        assert!(gpu_torus.faces.len() > torus.faces.len() * 3 && gpu_torus.wires.len() > torus.bonds.len() * 2);
        assert!(gpu_torus.faces.iter().all(|v| (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt() <= 1.15));

        let (rp2, rep_p) = backend.build_surface(kagome, SurfaceTargetUi::ProjectivePlane, &params).unwrap();
        assert!(rep_p.ok && rp2.chi == 1 && rp2.cover.is_some());
        let (gpu_quot, gpu_cov) = (mesh_from_surface(&rp2, false), mesh_from_surface(&rp2, true));
        assert_eq!(gpu_cov.vertex_count, 2 * gpu_quot.vertex_count);
        assert_eq!(gpu_cov.face_count, 2 * gpu_quot.face_count);
        assert!(gpu_cov.faces.len() > gpu_quot.faces.len());

        let (g2, rep_g2) = backend.build_surface(kagome, SurfaceTargetUi::Genus2, &params).unwrap();
        assert!(rep_g2.ok && g2.chi == -2);
        let gpu_g2 = mesh_from_surface(&g2, false);
        assert_eq!(gpu_g2.vertex_count, g2.xyz.len());
        assert!(gpu_g2.faces.len() > 100 && gpu_g2.wires.len() >= g2.bonds.len() * 2);
    }

    #[test]
    fn perspective_maps_origin_inside_wgpu_unit_depth() {
        let rot = orbit_rotation(0.4, -0.25);
        let mvp = build_mvp(1.5, 3.4, &rot);
        let ndc_z = mvp[3][2] / mvp[3][3];
        assert!((0.0..=1.0).contains(&ndc_z));
    }
}
