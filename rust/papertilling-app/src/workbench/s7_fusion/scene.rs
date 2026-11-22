//! One lattice of a fusion sequence or core connection as a 2D vector scene: faces tinted by their number
//! of sides, vertices coloured by their number of edges, curved edges, loops and dangling edges drawn as
//! they are in the map.

use papertilling_core::fusion::svg::{LOOP_RADIUS, label_point, loop_centre};
use papertilling_core::fusion::{CoreConnection, FaceWalk, FusionStep, PlanarMap, VertexKind};

use crate::canvas2d::Scene2D;
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, compute_bbox, ring_tint_rgba};
use crate::workbench::scene_helpers::{EDGE_RGBA, line_prim};

const HIGHLIGHT_RGBA: [u8; 4] = [214, 96, 20, 255];
const UNDER_RGBA: [u8; 4] = [198, 40, 40, 235];
const OVER_RGBA: [u8; 4] = [106, 27, 154, 235];
const INK_RGBA: [u8; 4] = [20, 24, 30, 255];
const WHITE: [u8; 4] = [255, 255, 255, 255];
const CURVE_SEGMENTS: usize = 12;

fn f32p(p: [f64; 2]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

fn same_pt(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) < 1e-9
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
    loop {
        let n = pts.len();
        if n < 3 {
            return pts;
        }
        // A point equal to its successor, or a point whose two neighbours coincide (a spike), is removed.
        let drop = (0..n).find(|&i| same_pt(pts[i], pts[(i + 1) % n]) || same_pt(pts[(i + n - 1) % n], pts[(i + 1) % n]));
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

/// Minimum interior angle (radians) of a triangle; used to prefer fat ears and avoid razor slivers.
fn min_tri_angle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let verts = [a, b, c];
    let mut best = std::f64::consts::PI;
    for i in 0..3 {
        let (o, p, q) = (verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3]);
        let (ux, uy) = (p[0] - o[0], p[1] - o[1]);
        let (vx, vy) = (q[0] - o[0], q[1] - o[1]);
        let denom = ux.hypot(uy) * vx.hypot(vy);
        if denom < 1e-18 {
            return 0.0;
        }
        let cos = ((ux * vx + uy * vy) / denom).clamp(-1.0, 1.0);
        best = best.min(cos.acos());
    }
    best
}

fn orient_ccw(poly: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let area2: f64 = (0..poly.len()).map(|i| cross([0.0, 0.0], poly[i], poly[(i + 1) % poly.len()])).sum();
    if area2 >= 0.0 {
        poly.to_vec()
    } else {
        poly.iter().copied().rev().collect()
    }
}

fn is_convex_ccw(poly: &[[f64; 2]]) -> bool {
    let n = poly.len();
    n >= 3 && (0..n).all(|i| cross(poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]) >= -1e-9)
}

/// Ear-clipping triangulation of a simple polygon (either orientation), picking the ear with the largest
/// minimum angle at each step so razor-thin slivers (which cause miter spikes in egui's AA) are avoided.
pub(super) fn triangulate(poly: &[[f64; 2]]) -> Vec<[[f64; 2]; 3]> {
    let ccw = orient_ccw(poly);
    let mut idx: Vec<usize> = (0..ccw.len()).collect();
    let mut out = Vec::new();
    while idx.len() > 3 {
        let n = idx.len();
        let mut best: Option<(usize, f64)> = None;
        for k in 0..n {
            let (a, b, c) = (ccw[idx[(k + n - 1) % n]], ccw[idx[k]], ccw[idx[(k + 1) % n]]);
            if cross(a, b, c) <= 1e-12 {
                continue;
            }
            let empty = !idx.iter().any(|&j| {
                let p = ccw[j];
                let corner = [a, b, c].iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 1e-12);
                !corner && cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
            });
            if empty {
                let score = min_tri_angle(a, b, c);
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((k, score));
                }
            }
        }
        // No ear (a degenerate outline): fill what is left as a fan so that no face is left blank.
        let Some((k, _)) = best else {
            out.extend((1..n - 1).map(|j| [ccw[idx[0]], ccw[idx[j]], ccw[idx[j + 1]]]));
            return out;
        };
        out.push([ccw[idx[(k + n - 1) % n]], ccw[idx[k]], ccw[idx[(k + 1) % n]]]);
        idx.remove(k);
    }
    if idx.len() == 3 {
        out.push([ccw[idx[0]], ccw[idx[1]], ccw[idx[2]]]);
    }
    out
}

/// Partition a simple polygon into minimal convex polygons (Hertel–Mehlhorn merging over fat ear-clipping).
/// Convex faces emit a single polygon with zero interior diagonals, eliminating dark-theme seams.
pub(super) fn convex_partition(poly: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
    if poly.len() < 3 {
        return Vec::new();
    }
    let ccw = orient_ccw(poly);
    if is_convex_ccw(&ccw) {
        return vec![ccw];
    }
    let mut pieces: Vec<Vec<[f64; 2]>> = triangulate(&ccw).into_iter().map(|t| t.to_vec()).collect();
    loop {
        let mut merged_pair: Option<(usize, usize, Vec<[f64; 2]>)> = None;
        'outer: for i in 0..pieces.len() {
            for j in (i + 1)..pieces.len() {
                let (pi, pj) = (&pieces[i], &pieces[j]);
                let (ni, nj) = (pi.len(), pj.len());
                for a in 0..ni {
                    let (u, v) = (pi[a], pi[(a + 1) % ni]);
                    for b in 0..nj {
                        if same_pt(pj[b], v) && same_pt(pj[(b + 1) % nj], u) {
                            let mut cand = Vec::with_capacity(ni + nj - 2);
                            cand.extend_from_slice(&pi[..=a]);
                            for step in 2..nj {
                                cand.push(pj[(b + step) % nj]);
                            }
                            cand.extend_from_slice(&pi[a + 1..]);
                            if is_convex_ccw(&cand) {
                                merged_pair = Some((i, j, cand));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        match merged_pair {
            Some((i, j, merged)) => {
                pieces[i] = merged;
                pieces.remove(j);
            }
            None => return pieces,
        }
    }
}

fn point_in_poly_f64(pt: [f64; 2], poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if (yi > pt[1]) != (yj > pt[1]) {
            let x_int = (xj - xi) * (pt[1] - yi) / (yj - yi) + xi;
            if pt[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn dist_to_seg_f64(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-18 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    (p[0] - (a[0] + t * dx)).hypot(p[1] - (a[1] + t * dy))
}

/// Interior label anchor maximizing clearance to face boundary segments and vertex discs.
fn face_label_pos(poly: &[[f64; 2]], m: &PlanarMap, scale: f64) -> [f64; 2] {
    let n = poly.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    if n < 3 {
        return [poly.iter().map(|p| p[0]).sum::<f64>() / n as f64, poly.iter().map(|p| p[1]).sum::<f64>() / n as f64];
    }
    let discs: Vec<([f64; 2], f64)> = m
        .verts
        .iter()
        .filter(|v| v.kind != VertexKind::Removed)
        .map(|v| {
            let r = if v.kind == VertexKind::Complete { 0.17 * scale } else { 0.07 * scale };
            (v.pos, r)
        })
        .collect();
    let clearance = |pt: [f64; 2]| -> f64 {
        let edge_d = (0..n)
            .map(|i| dist_to_seg_f64(pt, poly[i], poly[(i + 1) % n]))
            .fold(f64::INFINITY, f64::min);
        let vert_d = discs
            .iter()
            .map(|&(vp, r)| (pt[0] - vp[0]).hypot(pt[1] - vp[1]) - r)
            .fold(f64::INFINITY, f64::min);
        edge_d.min(vert_d)
    };

    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &p in poly {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let mut best_pt = triangulate(poly)
        .first()
        .map_or([0.5 * (x0 + x1), 0.5 * (y0 + y1)], |t| [(t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0]);
    let mut best_d = if point_in_poly_f64(best_pt, poly) { clearance(best_pt) } else { f64::NEG_INFINITY };

    let grid = 20usize;
    let (mut hx, mut hy) = (0.5 * (x1 - x0), 0.5 * (y1 - y0));
    let mut center = [0.5 * (x0 + x1), 0.5 * (y0 + y1)];
    for round in 0..3 {
        let steps = if round == 0 { grid } else { 8 };
        for ix in 0..=steps {
            let fx = (ix as f64 / steps as f64) * 2.0 - 1.0;
            for iy in 0..=steps {
                let fy = (iy as f64 / steps as f64) * 2.0 - 1.0;
                let cand = [center[0] + fx * hx, center[1] + fy * hy];
                if point_in_poly_f64(cand, poly) {
                    let d = clearance(cand);
                    if d > best_d {
                        best_d = d;
                        best_pt = cand;
                    }
                }
            }
        }
        center = best_pt;
        hx *= 0.25;
        hy *= 0.25;
    }
    best_pt
}

fn text(pos: [f64; 2], s: String, color: [u8; 4], size_px: f32) -> TextPrim {
    TextPrim { pos: f32p(pos), text: s, color, size_px }
}

fn disc(center: [f64; 2], radius: f64, fill: [u8; 4], stroke: ([u8; 4], f32)) -> CirclePrim {
    CirclePrim { center: f32p(center), radius_world: radius as f32, min_px: 3.0, max_px: 1.0e4, fill, stroke }
}

/// Position of an edge's number label: outward along a curved edge's bend so it never overlaps a 2-gon face
/// label inside a lens, or shifted slightly along the normal of a straight edge.
fn edge_label_pos(m: &PlanarMap, e: usize, scale: f64) -> [f64; 2] {
    let at = label_point(m, e);
    if m.is_loop(e) {
        return at;
    }
    let edge = &m.edges[e];
    let (a, b) = (m.verts[edge.ends[0]].pos, m.verts[edge.ends[1]].pos);
    if let Some(ctrl) = edge.ctrl {
        let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let (dx, dy) = (ctrl[0] - mid[0], ctrl[1] - mid[1]);
        let len = dx.hypot(dy);
        if len > 1e-6 {
            return [at[0] + 0.15 * scale * dx / len, at[1] + 0.15 * scale * dy / len];
        }
    }
    let len = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-9);
    [at[0] - 0.13 * scale * (b[1] - a[1]) / len, at[1] + 0.13 * scale * (b[0] - a[0]) / len]
}

#[derive(Clone, Copy)]
struct MapStyle {
    regular_sides: Option<usize>,
    regular_valence: Option<usize>,
    scale: f64,
}

const WANG_STYLE: MapStyle = MapStyle { regular_sides: Some(4), regular_valence: Some(4), scale: 1.0 };

fn median_edge_scale(m: &PlanarMap) -> f64 {
    let mut lens: Vec<f64> = m
        .edges
        .iter()
        .enumerate()
        .filter(|&(i, e)| e.alive && !m.is_loop(i))
        .map(|(_, e)| {
            let (a, b) = (m.verts[e.ends[0]].pos, m.verts[e.ends[1]].pos);
            (b[0] - a[0]).hypot(b[1] - a[1])
        })
        .filter(|&d| d.is_finite() && d > 1e-6)
        .collect();
    if lens.is_empty() {
        return 1.0;
    }
    lens.sort_by(f64::total_cmp);
    lens[lens.len() / 2].clamp(0.45, 2.5)
}

fn build_map_scene(
    m: &PlanarMap,
    highlight: &[u32],
    title: String,
    style: MapStyle,
    numbers: bool,
    names: bool,
    show_gate_labels: bool,
) -> Scene2D {
    let c = m.census();
    let edges_word = if show_gate_labels { "edges (qubits)" } else { "edges" };
    let mut p = ScenePrimitives {
        title,
        note: format!(
            "{} inner vertices, {} {edges_word}, {} inner faces. Number in a face: its sides. Number on a vertex: its edge-ends.",
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
            for poly in convex_partition(&outline) {
                p.polygons.push(PolyPrim { points: poly.iter().map(|&q| f32p(q)).collect(), fill: tint, stroke: None });
            }
            face_label_pos(&outline, m, style.scale)
        };
        if style.regular_sides.is_none_or(|reg| sides != reg) {
            p.labels.push(text(centre, sides.to_string(), INK_RGBA, 15.0));
        }
    }
    for (e, edge) in m.edges.iter().enumerate().filter(|(_, e)| e.alive) {
        let hot = highlight.contains(&edge.label);
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
            // Always use crisp dark charcoal ink so highlighted edge numbers stay legible over tinted faces.
            p.labels.push(text(edge_label_pos(m, e, style.scale), edge.label.to_string(), INK_RGBA, 13.0));
        }
    }
    for v in m.verts.iter().filter(|v| v.kind != VertexKind::Removed) {
        match (v.kind, style.regular_valence, v.rot.len()) {
            (VertexKind::Complete, Some(reg), k) if k == reg => {
                p.circles.push(disc(v.pos, 0.07 * style.scale, EDGE_RGBA, (WHITE, 1.0)));
            }
            (VertexKind::Complete, Some(reg), k) => {
                let fill = if k < reg { UNDER_RGBA } else { OVER_RGBA };
                p.circles.push(disc(v.pos, 0.13 * style.scale, fill, (WHITE, 1.0)));
                p.labels.push(text(v.pos, k.to_string(), WHITE, 12.0));
            }
            (VertexKind::Complete, None, k) => {
                p.circles.push(disc(v.pos, 0.11 * style.scale, EDGE_RGBA, (WHITE, 1.0)));
                p.labels.push(text(v.pos, k.to_string(), WHITE, 11.0));
            }
            _ => p.circles.push(disc(v.pos, 0.045 * style.scale, WHITE, (EDGE_RGBA, 1.2))),
        }
        if names && v.kind == VertexKind::Complete {
            p.labels.push(text(
                [v.pos[0] - 0.2 * style.scale, v.pos[1] + 0.2 * style.scale],
                v.name.clone(),
                INK_RGBA,
                13.0,
            ));
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
    let pad = (0.3 * style.scale) as f32;
    p.bbox = ([lo[0] - pad, lo[1] - pad], [hi[0] + pad, hi[1] + pad]);
    Scene2D::VectorScene { primitives: p }
}

/// Arbitrary planar cavity map scene (for live Place & fuse cavities on any 2D lattice).
#[allow(clippy::too_many_arguments)]
pub(super) fn map_scene(
    m: &PlanarMap,
    highlight: &[u32],
    title: String,
    regular_sides: Option<usize>,
    regular_valence: Option<usize>,
    numbers: bool,
    names: bool,
    show_gate_labels: bool,
) -> Scene2D {
    let style = MapStyle { regular_sides, regular_valence, scale: median_edge_scale(m) };
    build_map_scene(m, highlight, title, style, numbers, names, show_gate_labels)
}

/// The scene of one step along a route.
pub(super) fn step_scene(
    step: &FusionStep,
    index: usize,
    total: usize,
    numbers: bool,
    names: bool,
    show_gate_labels: bool,
) -> Scene2D {
    let op = if show_gate_labels && !step.operation.is_empty() {
        format!(" — {}", step.operation)
    } else {
        String::new()
    };
    let title = format!("Step {index} of {}: {}{op}", total.saturating_sub(1), step.title);
    build_map_scene(&step.map, &step.highlight, title, WANG_STYLE, numbers, names, show_gate_labels)
}

/// The scene of one enumerated core connection for the fixed surrounding cavity.
pub(super) fn core_scene(
    core: &CoreConnection,
    total_cores: usize,
    numbers: bool,
    names: bool,
    show_gate_labels: bool,
) -> Scene2D {
    let title = format!("Core {} of {total_cores}: {}", core.id + 1, core.title);
    build_map_scene(&core.map, &[11, 12], title, WANG_STYLE, numbers, names, show_gate_labels)
}

/// The unfused Fig. 8(a) start lattice shown for comparison inside the Core connections sub-view.
pub(super) fn start_comparison_scene(
    start: &FusionStep,
    numbers: bool,
    names: bool,
    show_gate_labels: bool,
) -> Scene2D {
    let title = String::from("Start lattice (Fig. 8(a)): unfused core with 2-valent vertex s");
    build_map_scene(&start.map, &start.highlight, title, WANG_STYLE, numbers, names, show_gate_labels)
}
