//! 3D mesh builders for the Atlas tab: Frieze Cylinder/Mobius ribbons,
//! 3D crystal nets & 80 layer group slabs (`Net3dPatch`), and 3D Hyperboloid `H^2` bowl.

use papertilling_core::hyperbolic::HyperbolicTiling;
use papertilling_core::space3d::nets3d::Net3dPatch;
use papertilling_core::subperiodic::FriezePatch;

use super::geom::{append_icosahedron, append_octahedron, polygon_centroid_idx, push_wire, tri_normal};
use super::{face_size_rgba, normalise, GpuMeshData, Vertex3d};

fn site_rgba_3d(kind: u8) -> [f32; 4] {
    match kind % 4 {
        0 => [0.22, 0.44, 0.78, 1.0],
        1 => [0.85, 0.36, 0.20, 1.0],
        2 => [0.18, 0.64, 0.44, 1.0],
        _ => [0.58, 0.36, 0.72, 1.0],
    }
}

/// Convert a 3D crystal net or layer slab patch into a GPU mesh.
pub fn mesh_from_net3d(net: &Net3dPatch) -> GpuMeshData {
    if net.positions.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center, radius) = normalise(&net.positions, &[]);
    let pts: Vec<[f32; 3]> = net
        .positions
        .iter()
        .map(|p| core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32))
        .collect();
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();

    for (u, &pos) in pts.iter().enumerate() {
        let kind = net.site_kinds.get(u).copied().unwrap_or(0);
        append_octahedron(&mut face_vertices, pos, 0.032, site_rgba_3d(kind));
    }
    if let Some((a, b)) = net.defect_axis {
        let pa: [f32; 3] = core::array::from_fn(|k| ((a[k] - center[k]) / radius) as f32);
        let pb: [f32; 3] = core::array::from_fn(|k| ((b[k] - center[k]) / radius) as f32);
        append_icosahedron(&mut face_vertices, pa, 0.045, [0.95, 0.16, 0.16, 1.0]);
        append_icosahedron(&mut face_vertices, pb, 0.045, [0.95, 0.16, 0.16, 1.0]);
        push_wire(&mut wire_vertices, pa, pb, [0.95, 0.16, 0.16, 1.0]);
    }
    let opaque_vertex_count = face_vertices.len();

    for &(u, v, b_kind) in &net.bonds {
        if u < pts.len() && v < pts.len() {
            let col = if b_kind == 1 {
                [0.95, 0.32, 0.14, 1.0]
            } else {
                [0.18, 0.24, 0.34, 0.92]
            };
            push_wire(&mut wire_vertices, pts[u], pts[v], col);
        }
    }
    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: pts.len(),
        edge_count: net.bonds.len(),
        face_count: 0,
        opaque_vertex_count,
    }
}

/// Convert a 1D frieze Cylinder or Mobius ribbon into a GPU mesh.
pub fn mesh_from_frieze3d(patch: &FriezePatch) -> GpuMeshData {
    if patch.pos3d.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center, radius) = normalise(&patch.pos3d, &[]);
    let pts: Vec<[f32; 3]> = patch
        .pos3d
        .iter()
        .map(|p| core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32))
        .collect();
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();
    for &p in &pts {
        append_octahedron(&mut face_vertices, p, 0.018, [0.22, 0.42, 0.72, 1.0]);
    }
    let opaque_vertex_count = face_vertices.len();
    let col = if patch.topology == papertilling_core::subperiodic::FriezeTopology::Mobius {
        [0.74, 0.56, 0.88, 0.85]
    } else {
        [0.52, 0.78, 0.88, 0.85]
    };
    for q in &patch.quads {
        for &[i0, i1, i2] in &[[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let n = tri_normal(pts[i0], pts[i1], pts[i2]);
            face_vertices.push(Vertex3d { position: pts[i0], normal: n, color: col });
            face_vertices.push(Vertex3d { position: pts[i1], normal: n, color: col });
            face_vertices.push(Vertex3d { position: pts[i2], normal: n, color: col });
        }
    }
    for &(u, v, ek) in &patch.edges {
        if u < pts.len() && v < pts.len() {
            let ec = match ek {
                1 => [0.92, 0.24, 0.16, 1.0],
                2 => [0.16, 0.64, 0.32, 0.9],
                _ => [0.15, 0.18, 0.24, 0.9],
            };
            push_wire(&mut wire_vertices, pts[u], pts[v], ec);
        }
    }
    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: pts.len(),
        edge_count: patch.edges.len(),
        face_count: patch.quads.len(),
        opaque_vertex_count,
    }
}

/// Convert a hyperbolic tiling on the upper-sheet Minkowski hyperboloid into a 3D GPU mesh.
pub fn mesh_from_hyperboloid(hyp: &HyperbolicTiling) -> GpuMeshData {
    if hyp.hyperboloid_pts.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center, radius) = normalise(&hyp.hyperboloid_pts, &[]);
    let pts: Vec<[f32; 3]> = hyp
        .hyperboloid_pts
        .iter()
        .map(|p| core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32))
        .collect();
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();
    for (u, &p) in pts.iter().enumerate() {
        let is_core = hyp.wedge_delta != 0 && hyp.faces.first().is_some_and(|f| f.contains(&u));
        if is_core {
            append_icosahedron(&mut face_vertices, p, 0.025, [0.95, 0.25, 0.16, 1.0]);
        } else {
            append_octahedron(&mut face_vertices, p, 0.015, [0.18, 0.28, 0.45, 1.0]);
        }
    }
    let opaque_vertex_count = face_vertices.len();

    for (f_idx, face) in hyp.faces.iter().enumerate() {
        let Some(c) = polygon_centroid_idx(&pts, face) else { continue };
        let mut fc = face_size_rgba(face.len(), true);
        if f_idx == 0 && hyp.wedge_delta != 0 {
            fc = [0.96, 0.42, 0.26, 0.90];
        } else {
            fc[3] = 0.85;
        }
        for m in 0..face.len() {
            let (u0, u1) = (face[m], face[(m + 1) % face.len()]);
            let n = tri_normal(c, pts[u0], pts[u1]);
            face_vertices.push(Vertex3d { position: c, normal: n, color: fc });
            face_vertices.push(Vertex3d { position: pts[u0], normal: n, color: fc });
            face_vertices.push(Vertex3d { position: pts[u1], normal: n, color: fc });
        }
    }
    for &(u, v) in &hyp.edges {
        if u < pts.len() && v < pts.len() {
            push_wire(&mut wire_vertices, pts[u], pts[v], [0.92, 0.95, 0.99, 0.95]);
        }
    }
    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: pts.len(),
        edge_count: hyp.edges.len(),
        face_count: hyp.faces.len(),
        opaque_vertex_count,
    }
}
