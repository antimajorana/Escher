//! Schematic layout for reconnections whose straight bonds cross in the original cavity geometry.

use std::f64::consts::TAU;

use papertilling_core::fusion::{PlanarMap, VertexKind};

use crate::scene2d::polygon::cross;

fn point_on_bond(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    cross(a, b, p).abs() <= 1e-9
        && (p[0] - a[0]) * (p[0] - b[0]) + (p[1] - a[1]) * (p[1] - b[1]) <= 1e-9
}

fn crosses_or_overlaps(map: &PlanarMap) -> bool {
    let opposite = |x: f64, y: f64| (x > 1e-9 && y < -1e-9) || (x < -1e-9 && y > 1e-9);
    let edges: Vec<_> = map
        .edges
        .iter()
        .filter(|e| e.alive)
        .map(|e| e.ends)
        .collect();
    for (i, &[u, v]) in edges.iter().enumerate() {
        let [a, b] = [map.verts[u].pos, map.verts[v].pos];
        for &[x, y] in &edges[i + 1..] {
            let [c, d] = [map.verts[x].pos, map.verts[y].pos];
            if (opposite(cross(a, b, c), cross(a, b, d))
                && opposite(cross(c, d, a), cross(c, d, b)))
                || (x != u && x != v && point_on_bond(c, a, b))
                || (y != u && y != v && point_on_bond(d, a, b))
                || (u != x && u != y && point_on_bond(a, c, d))
                || (v != x && v != y && point_on_bond(b, c, d))
            {
                return true;
            }
        }
    }
    false
}

fn drawing_matches_map(map: &PlanarMap) -> bool {
    map.verts.iter().enumerate().all(|(v, vertex)| {
        if vertex.rot.len() < 3 {
            return true;
        }
        let drawn = map.drawn_order(v);
        let order = map.map_order(v);
        drawn
            .iter()
            .position(|label| *label == order[0])
            .is_some_and(|start| {
                order
                    .iter()
                    .eq(drawn.iter().cycle().skip(start).take(drawn.len()))
            })
    }) && !crosses_or_overlaps(map)
}

/// Convex boundary layout for a generated chord core or one-interior-vertex core.
pub(super) fn schematic_if_needed(map: &PlanarMap, ccw_cycle: &[usize]) -> Option<PlanarMap> {
    if drawing_matches_map(map) {
        return None;
    }

    let mut drawing = map.clone();
    let n = ccw_cycle.len();
    let step = TAU / n as f64;
    let positions: Vec<_> = ccw_cycle.iter().map(|&v| map.verts[v].pos).collect();
    let [cx, cy] = positions.iter().fold([0.0, 0.0], |[sx, sy], &[x, y]| {
        [sx + x / n as f64, sy + y / n as f64]
    });
    let spacing = positions
        .iter()
        .zip(positions.iter().cycle().skip(1))
        .map(|(&a, &b)| (b[0] - a[0]).hypot(b[1] - a[1]))
        .sum::<f64>()
        / n as f64;
    let radius = spacing / (2.0 * (step * 0.5).sin());
    let start_angle = (positions[0][1] - cy).atan2(positions[0][0] - cx);
    let point = |r, angle: f64| [cx + r * angle.cos(), cy + r * angle.sin()];
    for (i, &v) in ccw_cycle.iter().enumerate() {
        let angle = start_angle + i as f64 * step;
        drawing.verts[v].pos = point(radius, angle);
        let previous = ccw_cycle[(i + n - 1) % n];
        let rotations = &map.verts[v].rot;
        let start = rotations
            .iter()
            .position(|dart| map.vertex_of(*dart ^ 1) == previous)?;
        let stubs: Vec<_> = rotations
            .iter()
            .cycle()
            .skip(start)
            .take(rotations.len())
            .map(|dart| map.vertex_of(*dart ^ 1))
            .filter(|&u| map.verts[u].kind == VertexKind::Boundary && map.verts[u].rot.len() == 1)
            .collect();
        let intervals = stubs.len() + 1;
        for (j, stub) in stubs.into_iter().enumerate() {
            let spread = (j + 1) as f64 / intervals as f64 - 0.5;
            drawing.verts[stub].pos = point(radius + 0.45 * spacing, angle + spread * step * 0.5);
        }
    }
    if let Some((v, vertex)) = map
        .verts
        .iter()
        .enumerate()
        .find(|(v, u)| u.kind == VertexKind::Complete && !ccw_cycle.contains(v))
    {
        let n = vertex.rot.len() as f64;
        drawing.verts[v].pos = vertex
            .rot
            .iter()
            .map(|dart| drawing.verts[drawing.vertex_of(*dart ^ 1)].pos)
            .fold([0.0, 0.0], |[sx, sy], [x, y]| [sx + x / n, sy + y / n]);
    }
    Some(drawing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use papertilling_core::fusion::enumerate_cavity_cores;

    #[test]
    fn concave_cavity_reconnection_draws_without_crossing_in_the_schematic() {
        let boundary = [[0., 0.], [2., 0.], [2., 1.], [1., 1.], [1., 2.], [0., 2.]];
        let stubs = [
            [-1., -1.],
            [3., -1.],
            [3., 1.5],
            [2., 2.],
            [1.5, 3.],
            [-1., 3.],
        ];
        let vertices: Vec<_> = boundary
            .into_iter()
            .map(|p| ("site", p, VertexKind::Complete))
            .chain(stubs.into_iter().map(|p| ("stub", p, VertexKind::Boundary)))
            .collect();
        let cycle = [0, 1, 2, 3, 4, 5];
        let edges: Vec<_> = (0..6)
            .flat_map(|i| [(i as u32 + 1, i, (i + 1) % 6), (i as u32 + 7, i, i + 6)])
            .collect();
        let map = PlanarMap::from_drawing(&vertices, &edges, 7).unwrap();
        assert!(schematic_if_needed(&map, &cycle).is_none());
        let result = enumerate_cavity_cores(&map, &cycle, 1, 0, false, 13, 0).unwrap();
        // The chord from (2,1) to (1,2) crosses the external stub from (1,1) to (2,2).
        let core = result
            .cores
            .iter()
            .find(|c| c.chords == vec![(2, 4)])
            .unwrap();
        assert!(crosses_or_overlaps(&core.map));
        let drawing = schematic_if_needed(&core.map, &cycle).unwrap();
        assert!(drawing_matches_map(&drawing));
        assert_eq!(drawing.census(), core.census);

        // The barycenter of these three spokes is (4/3, 4/3), outside the original L-shaped cavity.
        let result = enumerate_cavity_cores(&map, &cycle, 3, 1, false, 13, 0).unwrap();
        let core = result
            .cores
            .iter()
            .find(|c| c.chords == vec![(2, 12), (3, 12), (4, 12)])
            .unwrap();
        assert!(crosses_or_overlaps(&core.map));
        let drawing = schematic_if_needed(&core.map, &cycle).unwrap();
        assert!(drawing_matches_map(&drawing));
        assert_eq!(drawing.census(), core.census);
    }
}
