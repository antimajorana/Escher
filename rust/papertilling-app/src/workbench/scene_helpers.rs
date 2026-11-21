//! 2D scene builders shared by the workbench stage panels: points / edges / faces and polylines turned into a
//! `Scene2D::VectorScene`, a unit-circle outline and a diverging colour ramp.
//!
//! Every builder skips (never panics on) out-of-range indices, faces with fewer than 3 vertices and non-finite
//! coordinates.
#![allow(dead_code)] // P0 prewire: stage panels S2..S6 call these helpers as they land.

use crate::canvas2d::Scene2D;
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, ScenePrimitives, compute_bbox, ring_tint_rgba, site_palette};

pub(crate) const EDGE_RGBA: [u8; 4] = [44, 62, 80, 255];
/// Colour of non-finite values in [`heat_rgba`].
pub(crate) const NAN_RGBA: [u8; 4] = [150, 150, 150, 255];

fn finite2(p: [f64; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

fn f32p(p: [f64; 2]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

pub(crate) fn line_prim(a: [f64; 2], b: [f64; 2], color: [u8; 4], width: f32) -> LinePrim {
    LinePrim { a: f32p(a), b: f32p(b), color, width, dashed: false, arrow_mid: false }
}

/// Face tint by ring size (the same palette as the crystal pictures and the 3D `face_size_rgba`).
pub(crate) fn ring_size_tint(sides: usize) -> [u8; 4] {
    ring_tint_rgba(sides)
}

fn scene(primitives: ScenePrimitives) -> Scene2D {
    Scene2D::VectorScene { primitives }
}

fn empty_prims(title: &str, note: &str) -> ScenePrimitives {
    ScenePrimitives { title: title.to_string(), note: note.to_string(), bbox: ([-1.0, -1.0], [1.0, 1.0]), ..Default::default() }
}

/// Sites (circles), edges (lines) and faces (filled polygons, tinted by `face_tint(face_index)`).
pub(crate) fn vector_scene(
    points: &[[f64; 2]],
    edges: &[(usize, usize)],
    faces: &[Vec<usize>],
    face_tint: impl Fn(usize) -> [u8; 4],
    title: &str,
    note: &str,
) -> Scene2D {
    let ok = |i: usize| i < points.len() && finite2(points[i]);
    let mut p = empty_prims(title, note);
    for (fi, face) in faces.iter().enumerate() {
        if face.len() >= 3 && face.iter().all(|&i| ok(i)) {
            let pts = face.iter().map(|&i| f32p(points[i])).collect();
            p.polygons.push(PolyPrim { points: pts, fill: face_tint(fi), stroke: None });
        }
    }
    let mut len_sum = 0.0_f64;
    p.edge_line_indices = edges
        .iter()
        .map(|&(u, v)| {
            (ok(u) && ok(v) && u != v).then(|| {
                let (a, b) = (points[u], points[v]);
                len_sum += ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
                p.lines.push(line_prim(a, b, EDGE_RGBA, 1.4));
                p.lines.len() - 1
            })
        })
        .collect();
    let drawn_edges = p.lines.len();
    p.bbox = compute_bbox(points.iter().filter(|q| finite2(**q)).map(|q| f32p(*q)));
    let r = if drawn_edges > 0 {
        0.12 * len_sum / drawn_edges as f64
    } else {
        let (lo, hi) = p.bbox;
        0.01 * f64::from((hi[0] - lo[0]).hypot(hi[1] - lo[1])).max(1e-3)
    };
    p.site_circle_indices = (0..points.len())
        .map(|i| {
            ok(i).then(|| {
                p.circles.push(CirclePrim {
                    center: f32p(points[i]),
                    radius_world: r as f32,
                    min_px: 2.0,
                    max_px: 7.0,
                    fill: site_palette(0),
                    stroke: ([255, 255, 255, 255], 1.0),
                });
                p.circles.len() - 1
            })
        })
        .collect();
    p.site_count = p.circles.len();
    p.bond_count = drawn_edges;
    scene(p)
}

/// Open polylines (e.g. geodesic arcs, Schlegel edges) as line segments in one colour.
pub(crate) fn polyline_scene(polylines: &[Vec<[f64; 2]>], color: [u8; 4], title: &str, note: &str) -> Scene2D {
    let mut p = empty_prims(title, note);
    for pl in polylines {
        for w in pl.windows(2) {
            if finite2(w[0]) && finite2(w[1]) {
                p.lines.push(line_prim(w[0], w[1], color, 1.2));
            }
        }
    }
    p.bbox = compute_bbox(p.lines.iter().flat_map(|l| [l.a, l.b]));
    p.bond_count = p.lines.len();
    scene(p)
}

/// The unit circle (Poincare / Klein disk boundary) as `segments` chords (at least 3).
pub(crate) fn unit_circle_lines(segments: usize, color: [u8; 4]) -> Vec<LinePrim> {
    let n = segments.max(3);
    let at = |k: usize| {
        let t = std::f64::consts::TAU * k as f64 / n as f64;
        [t.cos(), t.sin()]
    };
    (0..n).map(|k| line_prim(at(k), at(k + 1), color, 1.0)).collect()
}

/// Append `extra` lines to a vector scene and grow its bounding box over them (other scenes are returned as is).
pub(crate) fn with_lines(scene2d: Scene2D, extra: Vec<LinePrim>) -> Scene2D {
    match scene2d {
        Scene2D::VectorScene { mut primitives } => {
            let had_any = !primitives.lines.is_empty() || !primitives.polygons.is_empty() || !primitives.circles.is_empty();
            primitives.lines.extend(extra);
            let (lo, hi) = primitives.bbox;
            let old = if had_any { vec![lo, hi] } else { Vec::new() };
            primitives.bbox = compute_bbox(old.into_iter().chain(primitives.lines.iter().flat_map(|l| [l.a, l.b])));
            Scene2D::VectorScene { primitives }
        }
        other => other,
    }
}

/// Diverging blue - white - red ramp: `vmin` blue, the midpoint white, `vmax` red; clamped outside the range.
/// A non-finite value is grey ([`NAN_RGBA`]); an empty or reversed range maps everything to the midpoint.
pub(crate) fn heat_rgba(v: f64, vmin: f64, vmax: f64) -> [u8; 4] {
    if !v.is_finite() {
        return NAN_RGBA;
    }
    let t = if vmax > vmin && vmin.is_finite() && vmax.is_finite() { ((v - vmin) / (vmax - vmin)).clamp(0.0, 1.0) } else { 0.5 };
    const LO: [f64; 3] = [33.0, 102.0, 172.0];
    const MID: [f64; 3] = [247.0, 247.0, 247.0];
    const HI: [f64; 3] = [178.0, 24.0, 43.0];
    let (a, b, s) = if t < 0.5 { (LO, MID, 2.0 * t) } else { (MID, HI, 2.0 * t - 1.0) };
    let c = |k: usize| (a[k] + (b[k] - a[k]) * s).round().clamp(0.0, 255.0) as u8;
    [c(0), c(1), c(2), 255]
}
