//! 3D mesh builders for the workbench stage panels (S2..S6): ball-and-stick nets and fan-triangulated polygon
//! complexes. They live under `mesh3d/` because the geometry helpers in `geom` are `pub(super)`.
//!
//! Both builders normalise the points into the unit ball (as every other mesh builder does) and skip, never panic
//! on, out-of-range indices, self-loops and faces with fewer than 3 vertices.
#![allow(dead_code)] // P0 prewire: stage panels S2..S6 call these builders as they land.

use papertilling_core::space3d::nets3d::Net3dPatch;

use super::atlas_mesh::mesh_from_net3d;
use super::geom::{polygon_centroid_idx, push_wire, tri_normal};
use super::{GpuMeshData, Vertex3d, face_size_rgba, normalise};

const EDGE_RGBA: [f32; 4] = [0.18, 0.24, 0.34, 0.92];

/// Ball-and-stick mesh: an octahedron per point (coloured by `site_kinds[i] % 4`, 0 when missing) and a wire per
/// edge `(u, v, kind)` (kind 1 is drawn red, e.g. seam bonds). Packs a `Net3dPatch` for `mesh_from_net3d`.
pub fn mesh_from_points_edges(points: &[[f64; 3]], edges: &[(usize, usize, u8)], site_kinds: &[u8]) -> GpuMeshData {
    let n = points.len();
    let net = Net3dPatch {
        name: String::from("scene"),
        space_group_num: 0,
        defect_label: String::new(),
        positions: points.to_vec(),
        bonds: edges.iter().copied().filter(|&(u, v, _)| u < n && v < n && u != v).collect(),
        site_kinds: (0..n).map(|i| site_kinds.get(i).copied().unwrap_or(0)).collect(),
        defect_axis: None,
        info: String::new(),
    };
    mesh_from_net3d(&net)
}

/// Put solid site markers in the mesh's opaque prefix, in its existing model coordinates.
pub(crate) fn add_site_markers(mesh: &mut GpuMeshData, sites: &[[f64; 3]]) {
    let mut markers = Vec::with_capacity(24 * sites.len());
    for site in sites {
        let point = core::array::from_fn(|i| (site[i] as f32 - mesh.center[i]) / mesh.radius);
        super::geom::append_octahedron(&mut markers, point, 0.007, [0.10, 0.32, 0.60, 1.0]);
    }
    let count = markers.len();
    mesh.faces.splice(mesh.opaque_vertex_count..mesh.opaque_vertex_count, markers);
    mesh.opaque_vertex_count += count;
}

/// Draw bonds in the model frame used for the cell mesh.
pub(crate) fn add_site_bonds(mesh: &mut GpuMeshData, sites: &[[f64; 3]], bonds: &[[usize; 2]]) {
    let point = |u: usize| core::array::from_fn(|i| (sites[u][i] as f32 - mesh.center[i]) / mesh.radius);
    for &[u, v] in bonds {
        push_wire(&mut mesh.wires, point(u), point(v), [0.10, 0.32, 0.60, 1.0]);
    }
    mesh.edge_count += bonds.len();
}

/// Default face tint: the ring-size palette (`face_size_rgba`), slightly transparent.
pub fn wb_face_tint(face: &[usize]) -> [f32; 4] {
    face_size_rgba(face.len(), true)
}

/// Polygon complex: every face fan-triangulated about its centroid (one triangle per side) and tinted by
/// `tint(face)`, plus a wire per edge. Opaque faces (alpha >= 0.999) come first, as the renderer expects.
pub fn mesh_from_polygons(
    points: &[[f64; 3]],
    faces: &[Vec<usize>],
    edges: &[(usize, usize)],
    tint: impl Fn(&[usize]) -> [f32; 4],
) -> GpuMeshData {
    if points.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center, radius) = normalise(points, &[]);
    let pts: Vec<[f32; 3]> =
        points.iter().map(|p| core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32)).collect();
    let n = pts.len();
    let finite = |i: usize| i < n && pts[i].iter().all(|v| v.is_finite());
    let mut opaque = Vec::new();
    let mut translucent = Vec::new();
    let mut face_count = 0;
    for face in faces {
        if face.len() < 3 || !face.iter().all(|&i| finite(i)) {
            continue;
        }
        let Some(c) = polygon_centroid_idx(&pts, face) else { continue };
        let col = tint(face);
        let out = if col[3] >= 0.999 { &mut opaque } else { &mut translucent };
        for m in 0..face.len() {
            let (u0, u1) = (face[m], face[(m + 1) % face.len()]);
            let nrm = tri_normal(c, pts[u0], pts[u1]);
            out.push(Vertex3d { position: c, normal: nrm, color: col });
            out.push(Vertex3d { position: pts[u0], normal: nrm, color: col });
            out.push(Vertex3d { position: pts[u1], normal: nrm, color: col });
        }
        face_count += 1;
    }
    let opaque_vertex_count = opaque.len();
    opaque.extend(translucent);
    let mut wires = Vec::new();
    let mut edge_count = 0;
    for &(u, v) in edges {
        if finite(u) && finite(v) && u != v {
            push_wire(&mut wires, pts[u], pts[v], EDGE_RGBA);
            edge_count += 1;
        }
    }
    GpuMeshData {
        faces: opaque,
        wires,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: n,
        edge_count,
        face_count,
        opaque_vertex_count,
    }
}

/// Placeholder shown by a stub 3D panel (never empty, so the smoke stage of that view can paint and advance):
/// a regular tetrahedron as ball-and-stick, 4 sites and 6 edges.
pub fn placeholder_mesh() -> GpuMeshData {
    let pts = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
    let edges = [(0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 2, 0), (1, 3, 0), (2, 3, 0)];
    mesh_from_points_edges(&pts, &edges, &[0, 1, 2, 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known answers by hand count: an octahedron has 8 triangles = 24 vertices; a wire has 2 vertices; a k-gon
    // fanned about its centroid gives k triangles = 3k vertices.

    #[test]
    fn tetrahedron_ball_and_stick_counts() {
        let m = placeholder_mesh();
        assert_eq!(m.faces.len(), 4 * 24);
        assert_eq!(m.wires.len(), 6 * 2);
        assert_eq!((m.vertex_count, m.edge_count), (4, 6));
        assert!((m.radius - 3f32.sqrt()).abs() < 1e-5, "circumradius of (+-1,+-1,+-1) tetrahedron is sqrt(3)");
    }

    #[test]
    fn points_edges_skip_bad_edges() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let m = mesh_from_points_edges(&pts, &[(0, 1, 1), (0, 0, 0), (1, 7, 0)], &[]);
        assert_eq!(m.wires.len(), 2);
        assert_eq!(m.edge_count, 1);
        assert_eq!(m.faces.len(), 2 * 24);
    }

    #[test]
    fn cube_polygons_fan_counts() {
        let pts: Vec<[f64; 3]> = (0..8).map(|i| [f64::from(i & 1), f64::from((i >> 1) & 1), f64::from((i >> 2) & 1)]).collect();
        let faces = vec![
            vec![0, 1, 3, 2], vec![4, 5, 7, 6], vec![0, 1, 5, 4],
            vec![2, 3, 7, 6], vec![0, 2, 6, 4], vec![1, 3, 7, 5],
        ];
        let mut edges = Vec::new();
        for a in 0..8usize {
            for b in a + 1..8 {
                if (a ^ b).count_ones() == 1 {
                    edges.push((a, b));
                }
            }
        }
        assert_eq!(edges.len(), 12, "a cube has 12 edges");
        let m = mesh_from_polygons(&pts, &faces, &edges, wb_face_tint);
        assert_eq!(m.face_count, 6);
        assert_eq!(m.faces.len(), 6 * 4 * 3, "6 squares x 4 fan triangles x 3 vertices");
        assert_eq!(m.wires.len(), 12 * 2);
        assert_eq!(m.opaque_vertex_count, 0, "ring-size tints are translucent");
        let opaque = mesh_from_polygons(&pts, &faces, &edges, |_| [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(opaque.opaque_vertex_count, 72);
    }

    #[test]
    fn polygons_skip_degenerate_faces_and_indices() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![vec![0, 1], vec![0, 1, 9], vec![0, 1, 2]];
        let m = mesh_from_polygons(&pts, &faces, &[(0, 9), (2, 2), (1, 2)], wb_face_tint);
        assert_eq!(m.face_count, 1);
        assert_eq!(m.faces.len(), 9);
        assert_eq!(m.edge_count, 1);
        assert!(mesh_from_polygons(&[], &faces, &[], wb_face_tint).faces.is_empty());
    }
}
