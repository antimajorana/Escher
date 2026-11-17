use std::collections::HashSet;

use papertilling_core::types::SurfaceMesh;

use super::{
    CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, compute_bbox, edge_glue_palette,
    ring_tint_rgba,
};

pub(super) fn compile_surface_net(mesh: &SurfaceMesh) -> ScenePrimitives {
    let mut polygons = Vec::new();
    let mut lines = Vec::new();
    let mut circles = Vec::new();
    let mut labels = Vec::new();

    for poly in &mesh.domain_polygons {
        if poly.len() >= 3 {
            polygons.push(PolyPrim {
                points: poly.iter().map(|p| [p[0] as f32, p[1] as f32]).collect(),
                fill: [244, 238, 218, 195],
                stroke: Some(([190, 180, 158, 180], 1.0)),
            });
        }
    }

    // Colour chart-local rings with the same palette as the 3D faces. Do not fill across a
    // quotient seam: those bonds are shown as stubs, not as a polygon spanning the chart.
    let mut vectors = std::collections::HashMap::new();
    for &(a, b, v) in &mesh.bonds {
        vectors.insert((a, b), v);
        vectors.insert((b, a), [-v[0], -v[1]]);
    }
    for face in &mesh.faces {
        if face.len() < 3 || face.iter().any(|&v| v >= mesh.uv.len()) {
            continue;
        }
        let local = (0..face.len()).all(|i| {
            let (a, b) = (face[i], face[(i + 1) % face.len()]);
            vectors.get(&(a, b)).is_some_and(|v| {
                (mesh.uv[b][0] - mesh.uv[a][0] - v[0])
                    .hypot(mesh.uv[b][1] - mesh.uv[a][1] - v[1]) < 1e-6
            })
        });
        if local {
            polygons.push(PolyPrim {
                points: face.iter().map(|&v| [mesh.uv[v][0] as f32, mesh.uv[v][1] as f32]).collect(),
                fill: ring_tint_rgba(face.len()),
                stroke: None,
            });
        }
    }

    let mut edge_line_indices = vec![None; mesh.bonds.len()];
    for (e_idx, &(a, b, v)) in mesh.bonds.iter().enumerate() {
        if a >= mesh.uv.len() || b >= mesh.uv.len() {
            continue;
        }
        let pa = mesh.uv[a];
        let pb = mesh.uv[b];
        let dx = (pb[0] - pa[0]) - v[0];
        let dy = (pb[1] - pa[1]) - v[1];
        let v_len2 = v[0] * v[0] + v[1] * v[1];
        if v_len2 < 1e-10 || dx * dx + dy * dy < 1e-3 {
            edge_line_indices[e_idx] = Some(lines.len());
            lines.push(LinePrim {
                a: [pa[0] as f32, pa[1] as f32],
                b: [pb[0] as f32, pb[1] as f32],
                color: [38, 92, 168, 240],
                width: 1.75,
                dashed: false,
                arrow_mid: false,
            });
        } else {
            let stub_a = [(pa[0] + 0.52 * v[0]) as f32, (pa[1] + 0.52 * v[1]) as f32];
            let stub_b = [(pb[0] - 0.52 * v[0]) as f32, (pb[1] - 0.52 * v[1]) as f32];
            for (s0, s1) in [([pa[0] as f32, pa[1] as f32], stub_a), ([pb[0] as f32, pb[1] as f32], stub_b)] {
                lines.push(LinePrim {
                    a: s0,
                    b: s1,
                    color: [110, 148, 206, 185],
                    width: 1.35,
                    dashed: true,
                    arrow_mid: false,
                });
            }
        }
    }

    for edge in &mesh.domain_edges {
        let p = [edge.p[0] as f32, edge.p[1] as f32];
        let q = [edge.q[0] as f32, edge.q[1] as f32];
        let is_open = edge.label.is_empty() || edge.label == "open";
        let col = if is_open {
            [132, 138, 148, 225]
        } else {
            edge_glue_palette(edge.color_idx)
        };
        lines.push(LinePrim {
            a: p,
            b: q,
            color: col,
            width: if is_open { 2.0 } else { 2.4 },
            dashed: is_open,
            arrow_mid: !is_open,
        });
        if !edge.label.is_empty() {
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            let len = (dx * dx + dy * dy).sqrt().max(1e-4);
            let nx = -dy / len;
            let ny = dx / len;
            labels.push(TextPrim {
                pos: [0.5 * (p[0] + q[0]) + 0.14 * nx, 0.5 * (p[1] + q[1]) + 0.14 * ny],
                text: edge.label.clone(),
                color: col,
                size_px: 12.0,
            });
        }
    }

    for (p, q, label) in &mesh.slits {
        let a = [p[0] as f32, p[1] as f32];
        let b = [q[0] as f32, q[1] as f32];
        lines.push(LinePrim {
            a,
            b,
            color: [156, 39, 176, 245],
            width: 2.8,
            dashed: true,
            arrow_mid: true,
        });
        for end in [a, b] {
            circles.push(CirclePrim {
                center: end,
                radius_world: 0.09,
                min_px: 3.8,
                max_px: 7.5,
                fill: [171, 71, 188, 255],
                stroke: ([74, 20, 140, 255], 1.4),
            });
        }
        if !label.is_empty() {
            labels.push(TextPrim {
                pos: [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]) + 0.16],
                text: label.clone(),
                color: [123, 31, 162, 255],
                size_px: 11.5,
            });
        }
    }

    let edge_set: HashSet<usize> = mesh.edge_sites.iter().copied().collect();
    let mut site_circle_indices = vec![None; mesh.uv.len()];
    for (u, uv) in mesh.uv.iter().enumerate() {
        let is_rim = edge_set.contains(&u);
        site_circle_indices[u] = Some(circles.len());
        circles.push(CirclePrim {
            center: [uv[0] as f32, uv[1] as f32],
            radius_world: 0.065,
            min_px: 2.4,
            max_px: 6.2,
            fill: if is_rim { [226, 152, 28, 255] } else { [36, 46, 62, 255] },
            stroke: ([16, 22, 32, 255], 1.0),
        });
    }

    for cone in &mesh.cones {
        let pos_opt = if let Some(s) = cone.site {
            mesh.uv.get(s).map(|p| [p[0] as f32, p[1] as f32])
        } else if let Some(f) = cone.face {
            mesh.faces.get(f).and_then(|fc| net_face_uv_center(&mesh.uv, fc))
        } else {
            None
        };
        if let Some(pos) = pos_opt {
            let col = if cone.angle_deg >= 0.0 {
                [222, 42, 32, 255]
            } else {
                [218, 142, 12, 255]
            };
            circles.push(CirclePrim {
                center: pos,
                radius_world: 0.13,
                min_px: 5.0,
                max_px: 10.5,
                fill: [col[0], col[1], col[2], 70],
                stroke: (col, 2.0),
            });
            labels.push(TextPrim {
                pos: [pos[0], pos[1] + 0.18],
                text: format!("{:+.0}°", cone.angle_deg),
                color: col,
                size_px: 11.0,
            });
        }
    }

    let bbox = compute_bbox(
        polygons
            .iter()
            .flat_map(|p| p.points.iter().copied())
            .chain(circles.iter().map(|c| c.center))
            .chain(lines.iter().flat_map(|l| [l.a, l.b])),
    );
    ScenePrimitives {
        site_count: mesh.uv.len(),
        bond_count: mesh.bonds.len(),
        polygons,
        lines,
        circles,
        labels,
        site_circle_indices,
        edge_line_indices,
        title: format!("{} — 2D net", mesh.name),
        note: String::new(),
        bbox,
    }
}

fn net_face_uv_center(uv: &[[f64; 2]], face: &[usize]) -> Option<[f32; 2]> {
    let first = *face.first()?;
    let p0 = *uv.get(first)?;
    let mut sum = [0.0_f64; 2];
    let mut cnt = 0_usize;
    for &u in face {
        if let Some(&pu) = uv.get(u)
            && (pu[0] - p0[0]).hypot(pu[1] - p0[1]) < 2.2
        {
            sum[0] += pu[0];
            sum[1] += pu[1];
            cnt += 1;
        }
    }
    (cnt > 0).then(|| [(sum[0] / cnt as f64) as f32, (sum[1] / cnt as f64) as f32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_rings_use_the_3d_palette_without_filling_across_seams() {
        let mut mesh = crate::surfaces_tab::empty_net("square chart");
        mesh.uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        mesh.faces = vec![vec![0, 1, 2, 3]];
        mesh.bonds = vec![(0, 1, [1.0, 0.0]), (1, 2, [0.0, 1.0]), (2, 3, [-1.0, 0.0]), (3, 0, [0.0, -1.0])];
        let drawn = compile_surface_net(&mesh);
        assert_eq!(drawn.polygons.len(), 1);
        assert_eq!(drawn.polygons[0].fill, ring_tint_rgba(4));
        mesh.bonds[0].2 = [-3.0, 0.0]; // this edge wraps around the quotient instead
        assert!(compile_surface_net(&mesh).polygons.is_empty());
    }
}
