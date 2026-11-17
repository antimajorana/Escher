use std::collections::HashMap;

use papertilling_core::types::Cell2D;

use super::{
    CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, bond_palette, compute_bbox,
    site_palette,
};

pub(super) fn compile_periodic_cell(cell: &Cell2D, repeat: i32, show_unit_cell: bool) -> ScenePrimitives {
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
