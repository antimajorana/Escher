//! A plain SVG picture of a fusion sequence: one panel per step. Faces that are not squares and vertices
//! that do not have four edges are coloured and carry their number of sides / edges.

use std::fmt::Write as _;

use super::map::{FaceWalk, PlanarMap, VertexKind};
use super::wang::FusionStep;

const SCALE: f64 = 84.0;
const PANEL_W: f64 = 5.4;
const PANEL_H: f64 = 5.6;
/// Radius of a drawn loop, in lattice units.
pub const LOOP_RADIUS: f64 = 0.2;

fn px(origin: [f64; 2], p: [f64; 2]) -> (f64, f64) {
    (origin[0] + (p[0] + 0.7) * SCALE, origin[1] + (3.6 - p[1]) * SCALE)
}

/// Centre of the circle drawn for loop `edge`.
pub fn loop_centre(m: &PlanarMap, edge: usize) -> [f64; 2] {
    let p = m.verts[m.edges[edge].ends[0]].pos;
    let a = m.edges[edge].loop_dir_deg.to_radians();
    [p[0] + LOOP_RADIUS * a.cos(), p[1] + LOOP_RADIUS * a.sin()]
}

/// Where the label of `edge` goes: on the curve's middle, or on the far side of a loop.
pub fn label_point(m: &PlanarMap, edge: usize) -> [f64; 2] {
    if m.is_loop(edge) {
        let (p, c) = (m.verts[m.edges[edge].ends[0]].pos, loop_centre(m, edge));
        return [2.6 * c[0] - 1.6 * p[0], 2.6 * c[1] - 1.6 * p[1]];
    }
    let e = &m.edges[edge];
    let (p, q, c) = (m.verts[e.ends[0]].pos, m.verts[e.ends[1]].pos, m.control_point(edge));
    [0.25 * p[0] + 0.5 * c[0] + 0.25 * q[0], 0.25 * p[1] + 0.5 * c[1] + 0.25 * q[1]]
}

fn face_path(m: &PlanarMap, face: &FaceWalk, origin: [f64; 2]) -> String {
    let mut d = String::new();
    for (k, &dart) in face.darts.iter().enumerate() {
        let edge = dart / 2;
        let (x, y) = px(origin, m.verts[face.vertices[k]].pos);
        if k == 0 {
            let _ = write!(d, "M{x:.1},{y:.1}");
        }
        if m.is_loop(edge) {
            continue;
        }
        let (cx, cy) = px(origin, m.control_point(edge));
        let (ex, ey) = px(origin, m.verts[m.vertex_of(dart ^ 1)].pos);
        let _ = write!(d, " Q{cx:.1},{cy:.1} {ex:.1},{ey:.1}");
    }
    d.push('Z');
    d
}

fn face_centre(m: &PlanarMap, face: &FaceWalk) -> [f64; 2] {
    if face.sides() == 1 && m.is_loop(face.darts[0] / 2) {
        return loop_centre(m, face.darts[0] / 2);
    }
    let pts: Vec<[f64; 2]> = face
        .darts
        .iter()
        .zip(&face.vertices)
        .flat_map(|(&dart, &v)| {
            let on_edge = if m.is_loop(dart / 2) { m.verts[v].pos } else { label_point(m, dart / 2) };
            [m.verts[v].pos, on_edge]
        })
        .collect();
    let n = pts.len() as f64;
    [pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n]
}

fn panel(out: &mut String, st: &FusionStep, index: usize, origin: [f64; 2]) {
    let m = &st.map;
    let c = m.census();
    let (tx, ty) = (origin[0] + 14.0, origin[1] + 22.0);
    let op = if st.operation.is_empty() { String::new() } else { format!(" · {}", st.operation) };
    let _ = write!(out, r#"<text x="{tx}" y="{ty}" class="t">{index}. {}{op}</text>"#, st.title);
    let _ = write!(
        out,
        r#"<text x="{tx}" y="{}" class="s">V = {}, E = {}, F = {} inside the patch</text>"#,
        ty + 18.0, c.vertices, c.edges, c.faces
    );
    for face in m.bounded_faces() {
        let odd = face.sides() != 4;
        let fill = if odd { "#f6c58f" } else { "#e9edf2" };
        if face.sides() == 1 && m.is_loop(face.darts[0] / 2) {
            let (x, y) = px(origin, loop_centre(m, face.darts[0] / 2));
            let _ = write!(out, r#"<circle cx="{x:.1}" cy="{y:.1}" r="{:.1}" fill="{fill}"/>"#, LOOP_RADIUS * SCALE);
        } else {
            let _ = write!(out, r#"<path d="{}" fill="{fill}" fill-rule="evenodd"/>"#, face_path(m, &face, origin));
        }
    }
    for (e, edge) in m.edges.iter().enumerate().filter(|(_, e)| e.alive) {
        let hot = st.highlight.contains(&edge.label);
        let leg = edge.ends.iter().any(|&v| m.verts[v].kind == VertexKind::Boundary && m.verts[v].rot.len() == 1);
        let style = format!(
            r#"fill="none" stroke="{}" stroke-width="{}"{}"#,
            if hot { "#c2410c" } else { "#1f2937" },
            if hot { 3.2 } else { 1.8 },
            if leg { r#" stroke-dasharray="5 4""# } else { "" }
        );
        if m.is_loop(e) {
            let (x, y) = px(origin, loop_centre(m, e));
            let _ = write!(out, r#"<circle cx="{x:.1}" cy="{y:.1}" r="{:.1}" {style}/>"#, LOOP_RADIUS * SCALE);
        } else {
            let (x0, y0) = px(origin, m.verts[edge.ends[0]].pos);
            let (cx, cy) = px(origin, m.control_point(e));
            let (x1, y1) = px(origin, m.verts[edge.ends[1]].pos);
            let _ = write!(out, r#"<path d="M{x0:.1},{y0:.1} Q{cx:.1},{cy:.1} {x1:.1},{y1:.1}" {style}/>"#);
        }
        let (lx, ly) = px(origin, label_point(m, e));
        let _ = write!(
            out,
            r##"<circle cx="{lx:.1}" cy="{ly:.1}" r="9" fill="#fff" opacity="0.85"/><text x="{lx:.1}" y="{:.1}" class="l">{}</text>"##,
            ly + 4.0, edge.label
        );
    }
    for face in m.bounded_faces().iter().filter(|f| f.sides() != 4) {
        let (x, y) = px(origin, face_centre(m, face));
        let _ = write!(out, r#"<text x="{x:.1}" y="{:.1}" class="n">{}</text>"#, y + 5.0, face.sides());
    }
    for v in m.verts.iter().filter(|v| v.kind != VertexKind::Removed) {
        let (x, y) = px(origin, v.pos);
        match v.kind {
            VertexKind::Complete if v.rot.len() != 4 => {
                let _ = write!(
                    out,
                    r##"<circle cx="{x:.1}" cy="{y:.1}" r="11" fill="#b91c1c"/><text x="{x:.1}" y="{:.1}" class="v">{}</text>"##,
                    y + 4.5, v.rot.len()
                );
            }
            VertexKind::Complete => {
                let _ = write!(out, r##"<circle cx="{x:.1}" cy="{y:.1}" r="6" fill="#1f2937"/>"##);
            }
            _ => {
                let _ = write!(out, r##"<circle cx="{x:.1}" cy="{y:.1}" r="4" fill="#fff" stroke="#6b7280" stroke-width="1.5"/>"##);
            }
        }
    }
}

/// One SVG with the steps laid out `columns` to a row.
pub fn steps_to_svg(steps: &[FusionStep], columns: usize) -> String {
    let cols = columns.clamp(1, steps.len().max(1));
    let rows = steps.len().div_ceil(cols);
    let (w, h) = (cols as f64 * PANEL_W * SCALE, rows as f64 * PANEL_H * SCALE);
    let mut out = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {h:.0}" font-family="sans-serif"><style>.t{{font-size:15px;font-weight:600;fill:#111827}}.s{{font-size:12.5px;fill:#4b5563}}.l{{font-size:11.5px;text-anchor:middle;fill:#111827}}.n{{font-size:17px;font-weight:600;text-anchor:middle;fill:#9a3412}}.v{{font-size:12.5px;font-weight:600;text-anchor:middle;fill:#fff}}</style><rect width="100%" height="100%" fill="#fff"/>"##
    );
    for (k, st) in steps.iter().enumerate() {
        let origin = [(k % cols) as f64 * PANEL_W * SCALE, (k / cols) as f64 * PANEL_H * SCALE + 26.0];
        panel(&mut out, st, k, origin);
    }
    out.push_str("</svg>");
    out
}
