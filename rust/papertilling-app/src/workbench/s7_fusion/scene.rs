//! One lattice of a fusion sequence as a 2D vector scene: faces tinted by their number of sides, vertices
//! coloured by their number of edges, curved edges, loops and dangling edges drawn as they are in the map.

use papertilling_core::fusion::svg::{LOOP_RADIUS, label_point, loop_centre};
use papertilling_core::fusion::{FaceWalk, FusionStep, PlanarMap, VertexKind};

use crate::canvas2d::Scene2D;
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, compute_bbox, ring_tint_rgba};
use crate::workbench::scene_helpers::{EDGE_RGBA, line_prim};

const HIGHLIGHT_RGBA: [u8; 4] = [214, 96, 20, 255];
const UNDER_RGBA: [u8; 4] = [198, 40, 40, 235];
const OVER_RGBA: [u8; 4] = [106, 27, 154, 235];
const INK_RGBA: [u8; 4] = [42, 48, 58, 235];
const WHITE: [u8; 4] = [255, 255, 255, 255];
const CURVE_SEGMENTS: usize = 12;

fn f32p(p: [f64; 2]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

/// Points of the drawn curve of non-loop `edge`, from its end 0 to its end 1.
fn edge_points(m: &PlanarMap, edge: usize) -> Vec<[f64; 2]> {
    let e = &m.edges[edge];
    let (p, q) = (m.verts[e.ends[0]].pos, m.verts[e.ends[1]].pos);
    if e.ctrl.is_none() {
        return vec![p, q];
    }
    let c = m.control_point(edge);
    (0..=CURVE_SEGMENTS)
        .map(|k| {
            let t = k as f64 / CURVE_SEGMENTS as f64;
            let (a, b, w) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t);
            [a * p[0] + b * c[0] + w * q[0], a * p[1] + b * c[1] + w * q[1]]
        })
        .collect()
}

/// Outline of a face as a simple polygon: curves are sampled, a dangling edge (walked out and back) and a
/// loop (it bounds its own one-sided face) are dropped from the outline.
pub(super) fn face_outline(m: &PlanarMap, face: &FaceWalk) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for &dart in &face.darts {
        if m.is_loop(dart / 2) {
            continue;
        }
        let mut seg = edge_points(m, dart / 2);
        if dart % 2 == 1 {
            seg.reverse();
        }
        pts.extend(&seg[..seg.len() - 1]);
    }
    let same = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) < 1e-9;
    loop {
        let n = pts.len();
        if n < 3 {
            return pts;
        }
        // A point equal to its successor, or a point whose two neighbours coincide (a spike), is removed.
        let drop = (0..n).find(|&i| same(pts[i], pts[(i + 1) % n]) || same(pts[(i + n - 1) % n], pts[(i + 1) % n]));
        match drop {
            Some(i) => {
                pts.remove(i);
            }
            None => return pts,
        }
    }
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Ear-clipping triangulation of a simple polygon (either orientation). The canvas fills polygons as fans
/// from their centroid, which spills outside a non-convex face such as the seven-sided one.
pub(super) fn triangulate(poly: &[[f64; 2]]) -> Vec<[[f64; 2]; 3]> {
    let area2: f64 = (0..poly.len()).map(|i| cross([0.0, 0.0], poly[i], poly[(i + 1) % poly.len()])).sum();
    let mut idx: Vec<usize> = if area2 >= 0.0 { (0..poly.len()).collect() } else { (0..poly.len()).rev().collect() };
    let mut out = Vec::new();
    while idx.len() > 3 {
        let n = idx.len();
        let ear = (0..n).find(|&k| {
            let (a, b, c) = (poly[idx[(k + n - 1) % n]], poly[idx[k]], poly[idx[(k + 1) % n]]);
            cross(a, b, c) > 1e-12
                && !idx.iter().any(|&j| {
                    let p = poly[j];
                    let corner = [a, b, c].iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 1e-12);
                    !corner && cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
                })
        });
        // No ear (a degenerate outline): fill what is left as a fan so that no face is left blank.
        let Some(k) = ear else {
            out.extend((1..n - 1).map(|j| [poly[idx[0]], poly[idx[j]], poly[idx[j + 1]]]));
            return out;
        };
        out.push([poly[idx[(k + n - 1) % n]], poly[idx[k]], poly[idx[(k + 1) % n]]]);
        idx.remove(k);
    }
    if idx.len() == 3 {
        out.push([poly[idx[0]], poly[idx[1]], poly[idx[2]]]);
    }
    out
}

fn centroid(poly: &[[f64; 2]]) -> [f64; 2] {
    let n = poly.len().max(1) as f64;
    [poly.iter().map(|p| p[0]).sum::<f64>() / n, poly.iter().map(|p| p[1]).sum::<f64>() / n]
}

fn text(pos: [f64; 2], s: String, color: [u8; 4], size_px: f32) -> TextPrim {
    TextPrim { pos: f32p(pos), text: s, color, size_px }
}

fn disc(center: [f64; 2], radius: f64, fill: [u8; 4], stroke: ([u8; 4], f32)) -> CirclePrim {
    CirclePrim { center: f32p(center), radius_world: radius as f32, min_px: 3.0, max_px: 1.0e4, fill, stroke }
}

/// The scene of one step. `numbers`: edge numbers; `names`: vertex names.
pub(super) fn step_scene(step: &FusionStep, index: usize, total: usize, numbers: bool, names: bool) -> Scene2D {
    let m = &step.map;
    let c = m.census();
    let op = if step.operation.is_empty() { String::new() } else { format!(" — {}", step.operation) };
    let mut p = ScenePrimitives {
        title: format!("Step {index} of {}: {}{op}", total.saturating_sub(1), step.title),
        note: format!(
            "{} inner vertices, {} edges (qubits), {} inner faces. Number in a face: its sides. Number on a vertex: its edge-ends.",
            c.vertices, c.edges, c.faces
        ),
        ..Default::default()
    };
    for face in m.bounded_faces() {
        let sides = face.sides();
        let tint = ring_tint_rgba(sides);
        let centre = if sides == 1 && m.is_loop(face.darts[0] / 2) {
            let lc = loop_centre(m, face.darts[0] / 2);
            p.circles.push(disc(lc, LOOP_RADIUS, tint, (tint, 0.0)));
            lc
        } else {
            let outline = face_outline(m, &face);
            for tri in triangulate(&outline) {
                p.polygons.push(PolyPrim { points: tri.iter().map(|&q| f32p(q)).collect(), fill: tint, stroke: None });
            }
            centroid(&outline)
        };
        if sides != 4 {
            p.labels.push(text(centre, sides.to_string(), INK_RGBA, 15.0));
        }
    }
    for (e, edge) in m.edges.iter().enumerate().filter(|(_, e)| e.alive) {
        let hot = step.highlight.contains(&edge.label);
        let (color, width) = if hot { (HIGHLIGHT_RGBA, 3.2) } else { (EDGE_RGBA, 1.6) };
        let leg = edge.ends.iter().any(|&v| m.verts[v].kind == VertexKind::Boundary && m.verts[v].rot.len() == 1);
        if m.is_loop(e) {
            p.circles.push(disc(loop_centre(m, e), LOOP_RADIUS, [0, 0, 0, 0], (color, width)));
        } else {
            for w in edge_points(m, e).windows(2) {
                p.lines.push(LinePrim { dashed: leg, ..line_prim(w[0], w[1], color, width) });
            }
        }
        if numbers {
            // Beside the edge, not on it: shifted a little along the edge's normal.
            let mut at = label_point(m, e);
            if !m.is_loop(e) {
                let (a, b) = (m.verts[edge.ends[0]].pos, m.verts[edge.ends[1]].pos);
                let len = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-9);
                at = [at[0] - 0.13 * (b[1] - a[1]) / len, at[1] + 0.13 * (b[0] - a[0]) / len];
            }
            p.labels.push(text(at, edge.label.to_string(), if hot { HIGHLIGHT_RGBA } else { INK_RGBA }, 13.0));
        }
    }
    for v in m.verts.iter().filter(|v| v.kind != VertexKind::Removed) {
        match (v.kind, v.rot.len()) {
            (VertexKind::Complete, 4) => p.circles.push(disc(v.pos, 0.07, EDGE_RGBA, (WHITE, 1.0))),
            (VertexKind::Complete, k) => {
                p.circles.push(disc(v.pos, 0.13, if k < 4 { UNDER_RGBA } else { OVER_RGBA }, (WHITE, 1.0)));
                p.labels.push(text(v.pos, k.to_string(), WHITE, 12.0));
            }
            _ => p.circles.push(disc(v.pos, 0.045, WHITE, (EDGE_RGBA, 1.2))),
        }
        if names && v.kind == VertexKind::Complete {
            p.labels.push(text([v.pos[0] - 0.2, v.pos[1] + 0.2], v.name.clone(), INK_RGBA, 13.0));
        }
    }
    p.site_count = c.vertices;
    p.bond_count = c.edges;
    let rims = p.circles.iter().flat_map(|c| {
        let r = c.radius_world;
        [[c.center[0] - r, c.center[1] - r], [c.center[0] + r, c.center[1] + r]]
    });
    let pts = p.lines.iter().flat_map(|l| [l.a, l.b]).chain(p.labels.iter().map(|t| t.pos)).chain(rims);
    let (lo, hi) = compute_bbox(pts);
    p.bbox = ([lo[0] - 0.3, lo[1] - 0.3], [hi[0] + 0.3, hi[1] + 0.3]);
    Scene2D::VectorScene { primitives: p }
}
