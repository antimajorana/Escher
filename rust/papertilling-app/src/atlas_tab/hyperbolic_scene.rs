//! Scenes of the H² view: the 2D disk (Poincare geodesic arcs or Klein chords, scaled by the view zoom) and the 3D
//! hyperboloid sheet, both of the panned tiling, with the picked face or vertex highlighted in each.

use std::f64::consts::PI;

use papertilling_core::hyperbolic::HyperbolicTiling;

use super::hyperbolic_ui::{DiskProjection, HypPick};
use crate::mesh3d::atlas_mesh::mesh_from_hyperboloid;
use crate::mesh3d::{GpuMeshData, Vertex3d};
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, Scene2D, ScenePrimitives, compute_bbox, ring_tint_rgba};

/// Highlight colour of the picked element (2D fill / ring and 3D faces / marker).
pub const HYP_PICK_RGBA8: [u8; 4] = [255, 196, 0, 150];
pub const HYP_PICK_RGBA: [f32; 4] = [1.0, 0.77, 0.0, 0.97];

fn line(a: [f64; 2], b: [f64; 2], color: [u8; 4], width: f32, dashed: bool) -> LinePrim {
    LinePrim { a: [a[0] as f32, a[1] as f32], b: [b[0] as f32, b[1] as f32], color, width, dashed, arrow_mid: false }
}

pub fn empty_scene(title: &str, note: &str) -> Scene2D {
    Scene2D::VectorScene {
        primitives: ScenePrimitives {
            polygons: Vec::new(),
            lines: Vec::new(),
            circles: Vec::new(),
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: title.to_string(),
            note: note.to_string(),
            bbox: ([-1.0, -1.0], [1.0, 1.0]),
            site_count: 0,
            bond_count: 0,
        },
    }
}

/// The 2D disk of `hyp` (already panned) in projection `proj`, all coordinates times `zoom`; the bounding box stays
/// the unit disk's, so the zoom magnifies.
pub fn disk_scene(hyp: &HyperbolicTiling, proj: DiskProjection, zoom: f64, picked: Option<HypPick>) -> Scene2D {
    let s = |p: [f64; 2]| [p[0] * zoom, p[1] * zoom];
    let mut lines = Vec::new();
    let rim = 72usize;
    for k in 0..rim {
        let t0 = 2.0 * PI * (k as f64) / (rim as f64);
        let t1 = 2.0 * PI * ((k + 1) as f64) / (rim as f64);
        lines.push(line(s([t0.cos(), t0.sin()]), s([t1.cos(), t1.sin()]), [110, 130, 160, 180], 1.2, true));
    }
    let pts_2d: Vec<[f64; 2]> = match proj {
        DiskProjection::Poincare => &hyp.poincare_pts,
        DiskProjection::Klein => &hyp.klein_pts,
    }
    .iter()
    .map(|&p| s(p))
    .collect();
    let face_pts = |f: &[usize]| -> Vec<[f32; 2]> {
        f.iter().filter_map(|&u| pts_2d.get(u).map(|p| [p[0] as f32, p[1] as f32])).collect()
    };
    let mut polygons: Vec<PolyPrim> = hyp
        .faces
        .iter()
        .enumerate()
        .filter_map(|(fi, f)| {
            let points = face_pts(f);
            (points.len() >= 3).then(|| PolyPrim {
                fill: if fi == 0 && hyp.wedge_delta != 0 { [245, 130, 96, 150] } else { ring_tint_rgba(f.len()) },
                stroke: None,
                points,
            })
        })
        .collect();
    match proj {
        DiskProjection::Poincare => {
            for arc in &hyp.edge_arcs {
                for w in arc.windows(2) {
                    lines.push(line(s(w[0]), s(w[1]), [32, 48, 76, 230], 1.4, false));
                }
            }
        }
        DiskProjection::Klein => {
            for &(u, v) in &hyp.edges {
                lines.push(line(pts_2d[u], pts_2d[v], [32, 48, 76, 230], 1.4, false));
            }
        }
    }
    let mut circles: Vec<CirclePrim> = pts_2d
        .iter()
        .map(|p| CirclePrim {
            center: [p[0] as f32, p[1] as f32],
            radius_world: 0.018,
            min_px: 1.5,
            max_px: 5.5,
            fill: [28, 78, 152, 230],
            stroke: ([255, 255, 255, 220], 0.5),
        })
        .collect();
    match picked {
        Some(HypPick::Face(f)) if f < hyp.faces.len() => {
            polygons.push(PolyPrim {
                points: face_pts(&hyp.faces[f]),
                fill: HYP_PICK_RGBA8,
                stroke: Some(([230, 140, 0, 255], 2.5)),
            });
        }
        Some(HypPick::Vertex(v)) if v < pts_2d.len() => {
            circles.push(CirclePrim {
                center: [pts_2d[v][0] as f32, pts_2d[v][1] as f32],
                radius_world: 0.04,
                min_px: 6.0,
                max_px: 13.0,
                fill: HYP_PICK_RGBA8,
                stroke: ([230, 140, 0, 255], 2.5),
            });
        }
        _ => {}
    }
    let zoom_note = if (zoom - 1.0).abs() > 1e-9 { format!(", view zoom {zoom:.2}x") } else { String::new() };
    Scene2D::VectorScene {
        primitives: ScenePrimitives {
            polygons,
            lines,
            circles,
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: format!("Hyperbolic {} ({:?})", hyp.symbol_label, proj),
            note: format!(
                "κ(D) = {:.4}, faces = {}, vertices = {}{zoom_note}",
                hyp.curvature_kappa,
                hyp.faces.len(),
                hyp.poincare_pts.len()
            ),
            bbox: compute_bbox([[-1.05, -1.05], [1.05, 1.05]].into_iter()),
            site_count: hyp.poincare_pts.len(),
            bond_count: hyp.edges.len(),
        },
    }
}

/// Mesh-space positions of the sheet vertices of a mesh normalised with `(center, radius)`.
pub fn sheet_points(hyp: &HyperbolicTiling, center: [f32; 3], radius: f32) -> Vec<[f32; 3]> {
    let r = radius.max(1e-6);
    hyp.hyperboloid_pts.iter().map(|p| core::array::from_fn(|k| (p[k] as f32 - center[k]) / r)).collect()
}

/// The hyperboloid sheet of `hyp` (already panned) with the picked face (a fan of `p` triangles, `3 p` vertices in
/// [`HYP_PICK_RGBA`], lifted slightly off the sheet towards its concave side) or vertex (a marker) highlighted.
pub fn sheet_mesh(hyp: &HyperbolicTiling, picked: Option<HypPick>) -> GpuMeshData {
    let mut mesh = mesh_from_hyperboloid(hyp);
    if hyp.hyperboloid_pts.is_empty() {
        return mesh;
    }
    let pts = sheet_points(hyp, mesh.center, mesh.radius);
    // Euclidean normal of t^2 - x^2 - y^2 = 1 towards the bowl's inside: (-x, -y, t), normalised.
    let lift = |k: usize, p: [f32; 3]| -> [f32; 3] {
        let h = hyp.hyperboloid_pts[k];
        let n = [-h[0], -h[1], h[2]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        core::array::from_fn(|i| p[i] + 0.006 * (n[i] / l) as f32)
    };
    match picked {
        Some(HypPick::Face(f)) if f < hyp.faces.len() => {
            let face = &hyp.faces[f];
            let lifted: Vec<[f32; 3]> = face.iter().filter(|&&u| u < pts.len()).map(|&u| lift(u, pts[u])).collect();
            if lifted.len() >= 3 {
                let c: [f32; 3] =
                    core::array::from_fn(|i| lifted.iter().map(|p| p[i]).sum::<f32>() / lifted.len() as f32);
                for m in 0..lifted.len() {
                    let (a, b) = (lifted[m], lifted[(m + 1) % lifted.len()]);
                    let u = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
                    let v = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
                    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                    let normal = [n[0] / l, n[1] / l, n[2] / l];
                    for position in [c, a, b] {
                        mesh.faces.push(Vertex3d { position, normal, color: HYP_PICK_RGBA });
                    }
                    for position in [a, b] {
                        mesh.wires.push(Vertex3d { position, normal: [0.0, 0.0, 1.0], color: [1.0, 0.55, 0.0, 1.0] });
                    }
                }
            }
        }
        Some(HypPick::Vertex(v)) if v < pts.len() => {
            // an opaque marker: insert into the opaque prefix (an octahedron, 8 triangles)
            let (c, r) = (pts[v], 0.04_f32);
            let ax = [[r, 0.0, 0.0], [-r, 0.0, 0.0], [0.0, r, 0.0], [0.0, -r, 0.0], [0.0, 0.0, r], [0.0, 0.0, -r]];
            let vtx = |i: usize| -> [f32; 3] { core::array::from_fn(|k| c[k] + ax[i][k]) };
            let mut marker = Vec::with_capacity(24);
            for (i, j) in [(0, 2), (2, 1), (1, 3), (3, 0)] {
                for top in [4, 5] {
                    let n: [f32; 3] = core::array::from_fn(|k| ax[i][k] + ax[j][k] + ax[top][k]);
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                    let normal = n.map(|x| x / l);
                    let color = [HYP_PICK_RGBA[0], HYP_PICK_RGBA[1], HYP_PICK_RGBA[2], 1.0];
                    for k in [vtx(i), vtx(j), vtx(top)] {
                        marker.push(Vertex3d { position: k, normal, color });
                    }
                }
            }
            let at = mesh.opaque_vertex_count.min(mesh.faces.len());
            let added = marker.len();
            mesh.faces.splice(at..at, marker);
            mesh.opaque_vertex_count = at + added;
        }
        _ => {}
    }
    mesh
}
