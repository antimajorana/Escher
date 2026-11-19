//! 3D GPU mesh builder for Föppl–von Kármán relaxed closed/open `SurfaceMesh`es and spectral overlays.

use std::collections::HashMap;

use papertilling_core::surface::elastic3d::SurfaceElasticResult;
use papertilling_core::types::SurfaceMesh;

use super::geom::{
    append_icosahedron, append_octahedron, blend_normal, compute_smooth_vertex_normals, cone_rgba,
    mean_chord_len, polygon_centroid_idx, push_wire, tint_towards, tri_normal,
};
use super::color_scales::{buckle_span, spectral_vmax};
use super::sheet_colors::{diverging_rgba, height_rgba, sequential_heat_rgba, strain_rgba};
use super::{GpuMeshData, Vertex3d, face_size_rgba, normalise};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurfaceColor3d {
    #[default]
    FaceRingSize,
    BondStrain,
    RadialBuckling,
    SpectralLdos,
    SpectralCharge,
}

impl SurfaceColor3d {
    pub const NON_SPECTRAL: [Self; 3] = [
        Self::FaceRingSize,
        Self::BondStrain,
        Self::RadialBuckling,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::FaceRingSize => "Face ring size",
            Self::BondStrain => "Bond strain",
            Self::RadialBuckling => "Radial buckling r - R_mean",
            Self::SpectralLdos => "LDOS |ψ|² (multiplet)",
            Self::SpectralCharge => "Spectral charge δρ(μ)",
        }
    }
}

/// Build `GpuMeshData` from `surf` using relaxed FvK coordinates `el_res` and coloring `color_mode`.
pub fn elastic_surface_gpu_mesh(
    surf: &SurfaceMesh,
    el_res: Option<&SurfaceElasticResult>,
    color_mode: SurfaceColor3d,
    show_cover: bool,
    spectral_scalar: Option<(&[f64], bool)>,
) -> GpuMeshData {
    let use_cover = show_cover && surf.cover.is_some();
    let (raw_xyz, bonds, faces, orientable) = if use_cover && let Some(cov) = surf.cover.as_deref() {
        let xyz = el_res.and_then(|r| r.cover_xyz.as_deref()).unwrap_or(&cov.xyz);
        (xyz, cov.bonds.clone(), cov.faces.as_slice(), true)
    } else {
        let xyz = el_res.map_or(surf.xyz.as_slice(), |r| r.xyz.as_slice());
        let bs: Vec<(usize, usize)> = surf.bonds.iter().map(|&(a, b, _)| (a, b)).collect();
        (xyz, bs, surf.faces.as_slice(), surf.orientable)
    };
    if raw_xyz.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }

    let (center, radius) = normalise(raw_xyz, &[]);
    let pts: Vec<[f32; 3]> = raw_xyz
        .iter()
        .map(|p| core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32))
        .collect();
    let avg_bond = mean_chord_len(&pts, bonds.iter().copied());
    let site_r = (avg_bond * 0.095).clamp(0.013, 0.034);
    let cone_r = (avg_bond * 0.19).clamp(0.028, 0.064);

    let nv = pts.len();
    let mut cone_site_angle: Vec<Option<f64>> = vec![None; nv];
    let mut cone_face_angle: Vec<Option<f64>> = vec![None; faces.len()];
    if use_cover && let Some(cov) = surf.cover.as_deref() {
        for &s in cov.cone_sites.iter().filter(|&&s| s < nv) {
            cone_site_angle[s] = Some(60.0);
        }
        for &f in cov.cone_faces.iter().filter(|&&f| f < faces.len()) {
            cone_face_angle[f] = Some(60.0);
        }
    } else {
        for cone in &surf.cones {
            if let Some(s) = cone.site && s < nv { cone_site_angle[s] = Some(cone.angle_deg); }
            if let Some(f) = cone.face && f < faces.len() { cone_face_angle[f] = Some(cone.angle_deg); }
        }
    }

    let spec_vmax = spectral_scalar.map_or(1.0, |(vals, _)| spectral_vmax(vals));
    let buckle_span = buckle_span(el_res);

    let site_color_of = |u: usize, fallback: [f32; 4]| -> [f32; 4] {
        match (color_mode, el_res, spectral_scalar) {
            (SurfaceColor3d::SpectralLdos | SurfaceColor3d::SpectralCharge, _, Some((vals, div))) => {
                let ui = if use_cover && !surf.sites.is_empty() { u % surf.sites.len() } else { u };
                let val = vals.get(ui).copied().unwrap_or(0.0);
                if div { diverging_rgba(val, spec_vmax, fallback[3]) } else { sequential_heat_rgba(val.max(0.0), spec_vmax, fallback[3]) }
            }
            (SurfaceColor3d::RadialBuckling, Some(r), _) => {
                let ui = if use_cover && !surf.sites.is_empty() { u % surf.sites.len() } else { u };
                height_rgba(r.radial_buckling.get(ui).copied().unwrap_or(0.0), buckle_span, fallback[3])
            }
            (SurfaceColor3d::BondStrain, Some(r), _) => {
                let ui = if use_cover && !surf.sites.is_empty() { u % surf.sites.len() } else { u };
                strain_rgba(r.site_strain.get(ui).copied().unwrap_or(0.0), fallback[3])
            }
            _ => fallback,
        }
    };

    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();
    for (u, &pos) in pts.iter().enumerate() {
        if let Some(ang) = cone_site_angle[u] && !matches!(color_mode, SurfaceColor3d::SpectralLdos | SurfaceColor3d::SpectralCharge) {
            append_icosahedron(&mut face_vertices, pos, cone_r, cone_rgba(ang));
        } else {
            let col = site_color_of(u, [0.20, 0.25, 0.33, 1.0]);
            append_octahedron(&mut face_vertices, pos, site_r, col);
        }
    }
    for (f_idx, face) in faces.iter().enumerate() {
        if let Some(ang) = cone_face_angle[f_idx] && let Some(fc) = polygon_centroid_idx(&pts, face) {
            append_icosahedron(&mut face_vertices, fc, cone_r, cone_rgba(ang));
        }
    }
    let opaque_vertex_count = face_vertices.len();

    let alpha = if orientable { 0.88 } else { 0.74 };
    let smooth_normals = compute_smooth_vertex_normals(&pts, faces);
    let smooth_mix = if surf.closed && surf.cones.is_empty() { 0.35 } else { 0.0 };
    for (f_idx, face) in faces.iter().enumerate() {
        let Some(centroid) = polygon_centroid_idx(&pts, face) else { continue };
        let base_face = face_size_rgba(face.len(), orientable);
        let mut c_cent = [0.0_f32; 4];
        let mut scols = Vec::with_capacity(face.len());
        for &u in face {
            let cu = site_color_of(u, base_face);
            for k in 0..4 { c_cent[k] += cu[k] / (face.len() as f32); }
            scols.push(cu);
        }
        c_cent[3] = alpha;
        if let Some(ang) = cone_face_angle[f_idx] && color_mode == SurfaceColor3d::FaceRingSize {
            c_cent = tint_towards(c_cent, cone_rgba(ang), 0.22);
        }
        for m in 0..face.len() {
            let (u0, u1) = (face[m], face[(m + 1) % face.len()]);
            let fnrm = tri_normal(centroid, pts[u0], pts[u1]);
            let n0 = blend_normal(fnrm, smooth_normals[u0], smooth_mix);
            let n1 = blend_normal(fnrm, smooth_normals[u1], smooth_mix);
            let (c0, c1) = if color_mode == SurfaceColor3d::FaceRingSize { (c_cent, c_cent) } else { (scols[m], scols[(m + 1) % face.len()]) };
            face_vertices.push(Vertex3d { position: centroid, normal: fnrm, color: c_cent });
            face_vertices.push(Vertex3d { position: pts[u0], normal: n0, color: c0 });
            face_vertices.push(Vertex3d { position: pts[u1], normal: n1, color: c1 });
        }
    }

    let mut bond_map: HashMap<(usize, usize), f64> = HashMap::new();
    if let Some(r) = el_res {
        let vs = surf.sites.len().max(1);
        for (bi, &(a, b, _)) in surf.bonds.iter().enumerate() {
            let eps = r.bond_strain.get(bi).copied().unwrap_or(0.0);
            bond_map.entry((a.min(b), a.max(b))).or_insert(eps);
            if use_cover {
                bond_map.entry(((a % vs).min(b % vs), (a % vs).max(b % vs))).or_insert(eps);
            }
        }
    }
    let vs = surf.sites.len().max(1);
    for &(u, v) in bonds.iter().filter(|&&(u, v)| u < nv && v < nv) {
        let wc = if color_mode == SurfaceColor3d::BondStrain && el_res.is_some() {
            let key = if use_cover { ((u % vs).min(v % vs), (u % vs).max(v % vs)) } else { (u.min(v), u.max(v)) };
            strain_rgba(bond_map.get(&key).copied().unwrap_or(0.0), 0.98)
        } else {
            [0.91, 0.95, 0.99, 0.96]
        };
        push_wire(&mut wire_vertices, pts[u], pts[v], wc);
    }

    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: nv,
        edge_count: bonds.len(),
        face_count: faces.len(),
        opaque_vertex_count,
    }
}
