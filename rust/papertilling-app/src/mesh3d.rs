use std::collections::HashMap;

use papertilling_core::types::{SurfaceCover3D, SurfaceMesh};

pub mod atlas_mesh;
mod geom;
mod rp2;
pub mod sheet;
pub mod sheet_colors;
pub mod sheet_modes;
mod sort;
mod surface_elastic;
#[cfg(test)]
mod tests;

use geom::{
    append_icosahedron, append_octahedron, arc_subdivisions, blend_normal, build_curved_face_ring, compute_smooth_vertex_normals,
    cone_rgba, dist3, mat4_mul, mean_chord_len, polygon_centroid_idx, polygon_centroid_pts, push_wire, sph_arc, tint_towards,
    tri_normal,
};
pub use sheet::{
    DefectViewMode, Sheet3dColorMode, Sheet3dViewOpts, mesh_from_sheet3d_with_spectral,
    sheet3d_caption, show_sheet3d_controls,
};
pub use sort::back_to_front_indices;
pub use surface_elastic::{SurfaceColor3d, elastic_surface_gpu_mesh};

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

/// Triangles and lines for the 3D viewport. `faces[..opaque_vertex_count]` are the opaque site and cone markers; the
/// rest are translucent face triangles, drawn sorted back to front (`back_to_front_indices`).
#[derive(Clone, Debug, Default)]
pub struct GpuMeshData {
    pub faces: Vec<Vertex3d>,
    pub wires: Vec<Vertex3d>,
    pub center: [f32; 3],
    pub radius: f32,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub opaque_vertex_count: usize,
}

/// Face colour by ring size: the RGB of the 2D ring palette (`scene2d::ring_tint_rgba`), so a k-gon has the same hue
/// in the 2D pictures and in 3D; more transparent on non-orientable surfaces (they pass through themselves in 3D).
pub fn face_size_rgba(sides: usize, orientable: bool) -> [f32; 4] {
    let alpha = if orientable { 0.86 } else { 0.72 };
    let c = crate::scene2d::ring_tint_rgba(sides);
    [f32::from(c[0]) / 255.0, f32::from(c[1]) / 255.0, f32::from(c[2]) / 255.0, alpha]
}

/// The projective plane drawn once (`show_cover == false`) uses the hemisphere model built from its double cover;
/// `show_cover` draws the whole antipodal double cover (a sphere).
pub fn mesh_from_surface(surf: &SurfaceMesh, show_cover: bool) -> GpuMeshData {
    if let Some(cov) = surf.cover.as_deref() {
        if show_cover {
            return cover_mesh(cov);
        }
        if let Some(m) = rp2::rp2_hemisphere_mesh(surf, cov) {
            return m;
        }
    }
    quotient_mesh(surf)
}

pub(crate) fn normalise(raw: &[[f64; 3]], extra: &[Vec<[f64; 3]>]) -> ([f64; 3], f64) {
    let mut min_p = [f64::INFINITY; 3];
    let mut max_p = [f64::NEG_INFINITY; 3];
    let finite = || raw.iter().chain(extra.iter().flatten()).filter(|p| p.iter().all(|v| v.is_finite()));
    for p in finite() {
        for k in 0..3 {
            min_p[k] = min_p[k].min(p[k]);
            max_p[k] = max_p[k].max(p[k]);
        }
    }
    if min_p[0] > max_p[0] {
        return ([0.0; 3], 1.0);
    }
    let center: [f64; 3] = core::array::from_fn(|k| 0.5 * (min_p[k] + max_p[k]));
    let dist = |p: &[f64; 3]| ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2)).sqrt();
    let radius = finite().map(dist).fold(0.0_f64, f64::max).max(1e-6);
    (center, radius)
}

fn cover_mesh(cov: &SurfaceCover3D) -> GpuMeshData {
    if cov.xyz.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center_f64, radius_f64) = normalise(&cov.xyz, &[]);
    let pts: Vec<[f32; 3]> =
        cov.xyz.iter().map(|p| core::array::from_fn(|k| ((p[k] - center_f64[k]) / radius_f64) as f32)).collect();
    let avg_bond = mean_chord_len(&pts, cov.bonds.iter().copied());
    let site_r = (avg_bond * 0.095).clamp(0.013, 0.034);
    let cone_r = (avg_bond * 0.19).clamp(0.028, 0.064);
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();

    let mut is_cone_site = vec![false; pts.len()];
    for &s in cov.cone_sites.iter().filter(|&&s| s < pts.len()) {
        is_cone_site[s] = true;
    }
    let mut is_cone_face = vec![false; cov.faces.len()];
    for &f in cov.cone_faces.iter().filter(|&&f| f < cov.faces.len()) {
        is_cone_face[f] = true;
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
            append_icosahedron(&mut face_vertices, sph_arc(fc, fc, 0.0), cone_r, [1.0, 0.32, 0.16, 1.0]);
        }
    }
    let opaque_vertex_count = face_vertices.len();

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
            let (u0, u1) = (face[m], face[(m + 1) % face.len()]);
            let mut c0 = cov.site_rgba.get(u0).copied().unwrap_or(fallback);
            let mut c1 = cov.site_rgba.get(u1).copied().unwrap_or(fallback);
            c0[3] = mean_col[3];
            c1[3] = mean_col[3];
            let k = arc_subdivisions(dist3(pts[u0], pts[u1]));
            for s in 0..k {
                let (t0, t1) = (s as f32 / k as f32, (s + 1) as f32 / k as f32);
                let (p0, p1) = (sph_arc(pts[u0], pts[u1], t0), sph_arc(pts[u0], pts[u1], t1));
                let n = tri_normal(centroid, p0, p1);
                face_vertices.push(Vertex3d { position: centroid, normal: n, color: mean_col });
                face_vertices.push(Vertex3d { position: p0, normal: n, color: tint_towards(c0, c1, t0) });
                face_vertices.push(Vertex3d { position: p1, normal: n, color: tint_towards(c0, c1, t1) });
            }
        }
    }

    for &(u, v) in cov.bonds.iter().filter(|&&(u, v)| u < pts.len() && v < pts.len()) {
        let cu = cov.site_rgba.get(u).copied().unwrap_or([0.9, 0.94, 0.98, 1.0]);
        let cv = cov.site_rgba.get(v).copied().unwrap_or([0.9, 0.94, 0.98, 1.0]);
        let wc = [
            0.45 * 0.5 * (cu[0] + cv[0]) + 0.55 * 0.95,
            0.45 * 0.5 * (cu[1] + cv[1]) + 0.55 * 0.97,
            0.45 * 0.5 * (cu[2] + cv[2]) + 0.55 * 1.00,
            0.96,
        ];
        let k = arc_subdivisions(dist3(pts[u], pts[v]));
        for s in 0..k {
            let (t0, t1) = (s as f32 / k as f32, (s + 1) as f32 / k as f32);
            push_wire(&mut wire_vertices, sph_arc(pts[u], pts[v], t0), sph_arc(pts[u], pts[v], t1), wc);
        }
    }

    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: center_f64.map(|x| x as f32),
        radius: radius_f64 as f32,
        vertex_count: pts.len(),
        edge_count: cov.bonds.len(),
        face_count: cov.faces.len(),
        opaque_vertex_count,
    }
}

fn quotient_mesh(surf: &SurfaceMesh) -> GpuMeshData {
    if surf.xyz.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center_f64, radius_f64) = normalise(&surf.xyz, &surf.bond_curves_3d);
    let norm_pt = |p: [f64; 3]| -> [f32; 3] { core::array::from_fn(|k| ((p[k] - center_f64[k]) / radius_f64) as f32) };
    let pts: Vec<[f32; 3]> = surf.xyz.iter().copied().map(norm_pt).collect();
    let curves: Vec<Vec<[f32; 3]>> =
        surf.bond_curves_3d.iter().map(|c| c.iter().copied().map(norm_pt).collect()).collect();
    let avg_bond = mean_chord_len(&pts, surf.bonds.iter().map(|&(u, v, _)| (u, v)));
    let site_r = (avg_bond * 0.095).clamp(0.013, 0.034);
    let cone_r = (avg_bond * 0.19).clamp(0.028, 0.064);
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();

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
        match cone_site_angle[u] {
            Some(ang) => append_icosahedron(&mut face_vertices, pos, cone_r, cone_rgba(ang)),
            None => append_octahedron(&mut face_vertices, pos, site_r, [0.20, 0.25, 0.33, 1.0]),
        }
    }

    let mut curve_lookup: HashMap<(usize, usize), (usize, bool)> = HashMap::new();
    for (e_idx, &(u, v, _)) in surf.bonds.iter().enumerate() {
        if curves.get(e_idx).is_some_and(|c| c.len() >= 2) {
            curve_lookup.entry((u, v)).or_insert((e_idx, false));
            curve_lookup.entry((v, u)).or_insert((e_idx, true));
        }
    }
    let rings: Vec<Option<Vec<[f32; 3]>>> = surf
        .faces
        .iter()
        .map(|face| {
            if face.len() < 3 || face.iter().any(|&u| u >= pts.len()) {
                return None;
            }
            build_curved_face_ring(face, &curves, &curve_lookup).or_else(|| Some(face.iter().map(|&u| pts[u]).collect()))
        })
        .collect();
    for (f_idx, ring) in rings.iter().enumerate() {
        if let (Some(ang), Some(ring)) = (cone_face_angle[f_idx], ring) {
            append_icosahedron(&mut face_vertices, polygon_centroid_pts(ring), cone_r, cone_rgba(ang));
        }
    }
    let opaque_vertex_count = face_vertices.len();

    let smooth_normals = compute_smooth_vertex_normals(&pts, &surf.faces);
    let smooth_mix = if surf.closed && surf.cones.is_empty() { 0.35 } else { 0.0 };
    for (f_idx, face) in surf.faces.iter().enumerate() {
        let Some(ring) = &rings[f_idx] else { continue };
        let mut color = face_size_rgba(face.len(), surf.orientable);
        if let Some(ang) = cone_face_angle[f_idx] {
            color = tint_towards(color, cone_rgba(ang), 0.22);
        }
        let centroid = polygon_centroid_pts(ring);
        let straight = ring.len() == face.len();
        for m in 0..ring.len() {
            let (b0, b1) = (ring[m], ring[(m + 1) % ring.len()]);
            let fnrm = tri_normal(centroid, b0, b1);
            let (n0, n1) = if straight {
                let (u0, u1) = (face[m], face[(m + 1) % face.len()]);
                (blend_normal(fnrm, smooth_normals[u0], smooth_mix), blend_normal(fnrm, smooth_normals[u1], smooth_mix))
            } else {
                (fnrm, fnrm)
            };
            face_vertices.push(Vertex3d { position: centroid, normal: fnrm, color });
            face_vertices.push(Vertex3d { position: b0, normal: n0, color });
            face_vertices.push(Vertex3d { position: b1, normal: n1, color });
        }
    }

    let mut is_edge_site = vec![false; pts.len()];
    for &s in surf.edge_sites.iter().filter(|&&s| s < pts.len()) {
        is_edge_site[s] = true;
    }
    for (e_idx, &(u, v, _)) in surf.bonds.iter().enumerate() {
        let is_rim = u < pts.len() && v < pts.len() && is_edge_site[u] && is_edge_site[v];
        let wc = if is_rim { [1.0, 0.86, 0.44, 1.0] } else { [0.91, 0.95, 0.99, 0.96] };
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
        center: center_f64.map(|x| x as f32),
        radius: radius_f64 as f32,
        vertex_count: pts.len(),
        edge_count: surf.bonds.len(),
        face_count: surf.faces.len(),
        opaque_vertex_count,
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
    let near = (distance - 1.35).clamp(0.04, 0.15);
    let far = (distance + 6.0).max(30.0);
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
