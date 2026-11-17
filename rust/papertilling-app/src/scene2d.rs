use std::collections::{HashMap, HashSet};

use papertilling_core::types::{BuiltCrystal, Cell2D, SurfaceMesh};
use crate::scene_crystal::compile_crystal_patch;

#[derive(Clone)]
pub enum Scene2D {
    PeriodicCell {
        cell: Cell2D,
        repeat: i32,
        show_unit_cell: bool,
    },
    SurfaceNet {
        mesh: SurfaceMesh,
    },
    CrystalPatch {
        crystal: BuiltCrystal,
        title: String,
        note: String,
    },
    VectorScene {
        primitives: ScenePrimitives,
    },
}

#[derive(Clone, Debug)]
pub struct PolyPrim {
    pub points: Vec<[f32; 2]>,
    pub fill: [u8; 4],
    pub stroke: Option<([u8; 4], f32)>,
}

#[derive(Clone, Debug)]
pub struct LinePrim {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub color: [u8; 4],
    pub width: f32,
    pub dashed: bool,
    pub arrow_mid: bool,
}

#[derive(Clone, Debug)]
pub struct CirclePrim {
    pub center: [f32; 2],
    pub radius_world: f32,
    pub min_px: f32,
    pub max_px: f32,
    pub fill: [u8; 4],
    pub stroke: ([u8; 4], f32),
}

#[derive(Clone, Debug)]
pub struct TextPrim {
    pub pos: [f32; 2],
    pub text: String,
    pub color: [u8; 4],
    pub size_px: f32,
}

#[derive(Clone, Debug)]
pub struct ScenePrimitives {
    pub polygons: Vec<PolyPrim>,
    pub lines: Vec<LinePrim>,
    pub circles: Vec<CirclePrim>,
    pub labels: Vec<TextPrim>,
    pub site_circle_indices: Vec<Option<usize>>,
    pub edge_line_indices: Vec<Option<usize>>,
    pub title: String,
    pub note: String,
    pub bbox: ([f32; 2], [f32; 2]),
    pub site_count: usize,
    pub bond_count: usize,
}

pub(crate) fn compile_scene(scene: &Scene2D) -> ScenePrimitives {
    match scene {
        Scene2D::PeriodicCell {
            cell,
            repeat,
            show_unit_cell,
        } => compile_periodic_cell(cell, *repeat, *show_unit_cell),
        Scene2D::SurfaceNet { mesh } => compile_surface_net(mesh),
        Scene2D::CrystalPatch {
            crystal,
            title,
            note,
        } => compile_crystal_patch(crystal, title, note),
        Scene2D::VectorScene { primitives } => primitives.clone(),
    }
}

pub(crate) fn bond_palette(kind: usize) -> [u8; 4] {
    match kind % 5 {
        0 => [44, 62, 80, 255],
        1 => [192, 57, 43, 255],
        2 => [34, 139, 94, 255],
        3 => [130, 68, 173, 255],
        _ => [196, 125, 24, 255],
    }
}

pub(crate) fn site_palette(orbit: usize) -> [u8; 4] {
    match orbit % 5 {
        0 => [31, 119, 180, 255],
        1 => [230, 97, 35, 255],
        2 => [27, 158, 119, 255],
        3 => [117, 112, 179, 255],
        _ => [214, 48, 114, 255],
    }
}

fn edge_glue_palette(idx: usize) -> [u8; 4] {
    match idx % 6 {
        0 => [214, 45, 32, 245],
        1 => [25, 108, 198, 245],
        2 => [28, 152, 92, 245],
        3 => [212, 132, 18, 245],
        4 => [142, 58, 178, 245],
        _ => [18, 152, 168, 245],
    }
}

pub(crate) fn ring_tint_rgba(sides: usize) -> [u8; 4] {
    match sides {
        3 => [244, 196, 68, 145],
        4 => [92, 168, 235, 145],
        5 => [238, 104, 88, 160],
        6 => [88, 198, 168, 140],
        7 => [174, 122, 228, 160],
        8 => [224, 116, 182, 155],
        _ => [172, 194, 92, 150],
    }
}

pub(crate) fn compute_bbox(pts: impl Iterator<Item = [f32; 2]>) -> ([f32; 2], [f32; 2]) {
    let mut min_p = [f32::INFINITY; 2];
    let mut max_p = [f32::NEG_INFINITY; 2];
    let mut any = false;
    for p in pts {
        if p[0].is_finite() && p[1].is_finite() {
            min_p[0] = min_p[0].min(p[0]);
            min_p[1] = min_p[1].min(p[1]);
            max_p[0] = max_p[0].max(p[0]);
            max_p[1] = max_p[1].max(p[1]);
            any = true;
        }
    }
    if any {
        (min_p, max_p)
    } else {
        ([-1.0, -1.0], [1.0, 1.0])
    }
}

fn compile_periodic_cell(cell: &Cell2D, repeat: i32, show_unit_cell: bool) -> ScenePrimitives {
    let rep = repeat.clamp(1, 9);
    let n_lo = -(rep / 2);
    let n_hi = n_lo + rep;

    let mut polygons = Vec::new();
    let mut lines = Vec::new();
    let mut circles = Vec::new();
    let mut labels = Vec::new();

    let a1 = [cell.a1[0] as f32, cell.a1[1] as f32];
    let a2 = [cell.a2[0] as f32, cell.a2[1] as f32];

    if show_unit_cell {
        let p00 = [0.0_f32, 0.0];
        let p10 = a1;
        let p11 = [a1[0] + a2[0], a1[1] + a2[1]];
        let p01 = a2;
        polygons.push(PolyPrim {
            points: vec![p00, p10, p11, p01],
            fill: [218, 234, 252, 95],
            stroke: None,
        });
        for (p, q) in [(p00, p10), (p10, p11), (p11, p01), (p01, p00)] {
            lines.push(LinePrim {
                a: p,
                b: q,
                color: [32, 94, 176, 225],
                width: 1.6,
                dashed: true,
                arrow_mid: false,
            });
        }
        labels.push(TextPrim {
            pos: [0.5 * a1[0], 0.5 * a1[1] - 0.08],
            text: String::from("a1"),
            color: [24, 72, 145, 245],
            size_px: 12.0,
        });
        labels.push(TextPrim {
            pos: [0.5 * a2[0] - 0.08, 0.5 * a2[1]],
            text: String::from("a2"),
            color: [24, 72, 145, 245],
            size_px: 12.0,
        });
    }

    let site_pos = |u: i32, v: i32, idx: usize| -> [f32; 2] {
        let b = cell.basis.get(idx).copied().unwrap_or([0.0, 0.0]);
        [
            (u as f64 * cell.a1[0] + v as f64 * cell.a2[0] + b[0]) as f32,
            (u as f64 * cell.a1[1] + v as f64 * cell.a2[1] + b[1]) as f32,
        ]
    };

    let mut seen_sites: HashMap<(i32, i32, usize), [f32; 2]> = HashMap::new();
    for u in n_lo..n_hi {
        for v in n_lo..n_hi {
            for idx in 0..cell.basis.len() {
                seen_sites.insert((u, v, idx), site_pos(u, v, idx));
            }
            for b in &cell.bonds {
                if b.i >= cell.basis.len() || b.j >= cell.basis.len() {
                    continue;
                }
                let p = site_pos(u, v, b.i);
                let q = site_pos(u + b.n1, v + b.n2, b.j);
                seen_sites.insert((u + b.n1, v + b.n2, b.j), q);
                lines.push(LinePrim {
                    a: p,
                    b: q,
                    color: bond_palette(b.kind),
                    width: 1.85,
                    dashed: false,
                    arrow_mid: false,
                });
            }
        }
    }

    let mut ordered_sites: Vec<((i32, i32, usize), [f32; 2])> = seen_sites.into_iter().collect();
    ordered_sites.sort_by_key(|&(k, _)| k);
    for ((_, _, idx), pos) in &ordered_sites {
        circles.push(CirclePrim {
            center: *pos,
            radius_world: 0.075,
            min_px: 2.6,
            max_px: 7.2,
            fill: site_palette(*idx),
            stroke: ([18, 36, 56, 255], 1.15),
        });
    }

    let bbox = compute_bbox(
        circles
            .iter()
            .map(|c| c.center)
            .chain(lines.iter().flat_map(|l| [l.a, l.b])),
    );
    ScenePrimitives {
        site_count: circles.len(),
        bond_count: lines.len(),
        polygons,
        lines,
        circles,
        labels,
        site_circle_indices: Vec::new(),
        edge_line_indices: Vec::new(),
        title: String::new(),
        note: String::new(),
        bbox,
    }
}

fn compile_surface_net(mesh: &SurfaceMesh) -> ScenePrimitives {
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

