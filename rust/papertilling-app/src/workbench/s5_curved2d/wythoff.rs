//! Wythoff family of {p,q}: the seven ringed operations and the snub (both hands) on S², E² and H² via
//! `build_wythoff`; vertex symbol and face-type counts in the inspector.

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::noneuclidean2d::{SnubHandedness, WythoffOp, build_wythoff};
use serde_json::{Value, json};

use super::draw::{DiskModel, Picture, e2_picture, h2_picture, mesh_with_arcs, s2_picture, slerp};
use super::{S5Out, check_symbol, delta_badge, geom_label, get_usize, pq_controls};
use crate::workbench::scene_helpers::{EDGE_RGBA, ring_size_tint};

/// The nine choices of the operation box (snub split by hand).
pub(crate) const OPS: [(WythoffOp, &str); 9] = [
    (WythoffOp::Regular, "regular t0 {p,q}"),
    (WythoffOp::Dual, "dual t2 {q,p}"),
    (WythoffOp::Rectified, "rectified t1"),
    (WythoffOp::Truncated, "truncated t0,1"),
    (WythoffOp::Bitruncated, "bitruncated t1,2"),
    (WythoffOp::Cantellated, "cantellated t0,2"),
    (WythoffOp::Omnitruncated, "omnitruncated t0,1,2"),
    (WythoffOp::Snub { chirality: SnubHandedness::Left }, "snub sr, left hand"),
    (WythoffOp::Snub { chirality: SnubHandedness::Right }, "snub sr, right hand"),
];

/// Layers honoured by the Wythoff builders on E² and H² (S² is closed).
pub(crate) const MAX_LAYERS: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WyJob {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) op: usize,
    pub(crate) layers: usize,
}

pub(crate) struct WyState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) op: usize,
    pub(crate) layers: usize,
}

impl Default for WyState {
    /// Truncated {3,5}: the truncated icosahedron 5.6.6.
    fn default() -> Self {
        Self { p: 3, q: 5, op: 3, layers: 2 }
    }
}

impl WyState {
    pub(crate) fn job(&self) -> WyJob {
        WyJob { p: self.p, q: self.q, op: self.op, layers: self.layers }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = pq_controls(ui, &mut self.p, &mut self.q, 2);
        ui.label(delta_badge(self.p, self.q));
        ui.horizontal(|ui| {
            ui.label("Operation:");
            egui::ComboBox::from_id_salt("wb_s5_wythoff_op").selected_text(OPS[self.op.min(8)].1).show_ui(ui, |ui| {
                for (i, (_, label)) in OPS.iter().enumerate() {
                    rebuild |= ui.selectable_value(&mut self.op, i, *label).changed();
                }
            });
        });
        if !matches!(check_symbol(self.p, self.q), Ok(GeometryKind::Spherical)) {
            ui.horizontal(|ui| {
                ui.label("Layers of the base patch:");
                rebuild |= ui.add(egui::DragValue::new(&mut self.layers).range(1..=MAX_LAYERS)).changed();
            });
        }
        rebuild
    }

    pub(super) fn save(&self) -> Value {
        json!({ "p": self.p, "q": self.q, "op": self.op, "layers": self.layers })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 2, super::MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 2, super::MAX_PQ) {
            self.q = x;
        }
        if let Some(x) = get_usize(v, "op", 0, OPS.len() - 1) {
            self.op = x;
        }
        if let Some(x) = get_usize(v, "layers", 1, MAX_LAYERS) {
            self.layers = x;
        }
    }
}

pub(crate) fn build(j: &WyJob) -> Result<S5Out, String> {
    let kind = check_symbol(j.p, j.q)?;
    let (op, op_label) = *OPS.get(j.op).ok_or_else(|| format!("unknown Wythoff operation #{}", j.op))?;
    if !(1..=MAX_LAYERS).contains(&j.layers) {
        return Err(format!("{} layers refused: allowed 1..={MAX_LAYERS}", j.layers));
    }
    let t = build_wythoff(j.p, j.q, op, j.layers)?;
    let title = format!("S5 Wythoff {op_label} of {{{},{}}}", j.p, j.q);
    let tint = |fi: usize| ring_size_tint(t.faces[fi].len());
    let (scene2d, dropped, arcs3d) = match kind {
        GeometryKind::Spherical => {
            let note = "stereographic projection from the centre of face 0 (face 0 is the outside)";
            let arcs: Option<Vec<Vec<[f64; 3]>>> = t.edges.iter().map(|&(u, v)| slerp(t.coords_3d[u], t.coords_3d[v], 10)).collect();
            let (s, d) = match s2_picture(&t.coords_3d, &t.faces, &title, note) {
                Some(x) => x,
                None => straight_picture(&t.coords_2d, &t.edges, &title),
            };
            (s, d, arcs)
        }
        GeometryKind::Euclidean => {
            let (s, d) = e2_picture(&t.coords_2d, &t.faces, &title, "Euclidean plane");
            (s, d, None)
        }
        GeometryKind::Hyperbolic => {
            let note = "Poincaré disk; edges are sampled geodesic arcs";
            let (s, d) = h2_picture(&t.coords_2d, &t.faces, tint, DiskModel::Poincare, true, &title, note);
            (s, d, None)
        }
    };
    let mesh = mesh_with_arcs(&t.coords_3d, &t.faces, arcs3d.as_deref());
    let closed = kind == GeometryKind::Spherical;
    let mut rows = vec![
        (String::from("Geometry"), geom_label(kind).to_string()),
        (String::from("Operation"), format!("{}{}", t.op_name, t.chirality.as_ref().map(|c| format!(" ({c})")).unwrap_or_default())),
        (String::from("Vertex symbol"), t.vertex_symbol.clone()),
        (String::from("V / E / F"), format!("{} / {} / {}", t.n_vertices, t.n_edges, t.n_faces)),
        (String::from(if closed { "χ (closed surface)" } else { "χ (finite patch)" }), t.euler_chi.to_string()),
    ];
    if !closed {
        rows.push((String::from("Base patch layers"), j.layers.to_string()));
    }
    let mut notes = vec![format!("Core: {}", t.provenance)];
    if closed {
        notes.push(String::from("Uniform polyhedron on S²: V − E + F = 2 (Euler) [PROVED]"));
    }
    if dropped > 0 {
        notes.push(format!("{dropped} face(s) outside the 2D clip window were not drawn."));
    }
    let table_rows = t.face_type_counts.iter().map(|(s, c)| vec![format!("{s}-gon"), c.to_string()]).collect();
    let caption = format!("Wythoff {} of {{{},{}}}: {} faces", t.op_name, j.p, j.q, t.n_faces);
    Ok(S5Out { scene2d, mesh, caption, rows, notes, table: Some((vec!["Face type", "Count"], table_rows)), svg: None })
}

/// Fallback for S² results whose vertices are not on the unit sphere: the core's stereographic coordinates with
/// straight edges (labelled as such).
fn straight_picture(pts: &[[f64; 2]], edges: &[(usize, usize)], title: &str) -> (crate::canvas2d::Scene2D, usize) {
    let mut pic = Picture::new(title, "core stereographic coordinates, edges drawn as straight chords");
    for &(u, v) in edges {
        if let (Some(a), Some(b)) = (pts.get(u), pts.get(v)) {
            pic.polyline(&[*a, *b], EDGE_RGBA, 1.2);
        }
    }
    pic.finish(pts.len())
}
