//! Wythoff family of {p,q}: the seven ringed operations and the snub (both hands) on S², E² and H² via
//! `build_wythoff`; vertex symbol, face-type counts, `WythoffTiling::stopped` banner, and base-patch layer table.

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::hyperbolic::inflation::{HypPointSet, grow_vertex_completion, hyp_dist};
use papertilling_core::noneuclidean2d::s2_seeds::grow_s2;
use papertilling_core::noneuclidean2d::wythoff_h2::WYTHOFF_TILE_BUDGET;
use papertilling_core::noneuclidean2d::{SnubHandedness, WythoffOp, build_unified_pq, build_wythoff};
use serde_json::{Value, json};

use super::draw::{DiskModel, Picture, e2_picture, h2_picture, mesh_with_arcs, s2_picture, slerp};
use super::growth_table::{GROWTH_HEADERS, build_e2_regular_seeded, non_disk_flags, patch_layer_stats, stats_to_rows};
use super::{S5Out, check_symbol, delta_badge, euler_note, geom_label, get_usize, pq_controls};
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

/// 0-based layer range offered for Wythoff base patches (`0..=6`; capped by `WYTHOFF_TILE_BUDGET` on H²).
pub(crate) const MAX_LAYERS: usize = 6;

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
        WyJob { p: self.p, q: self.q, op: self.op, layers: self.layers.min(MAX_LAYERS) }
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
        if check_symbol(self.p, self.q).is_ok() {
            let mut lay = self.layers.min(MAX_LAYERS);
            ui.horizontal(|ui| {
                ui.label("Layers:");
                if ui.add(egui::Slider::new(&mut lay, 0..=MAX_LAYERS)).changed() {
                    self.layers = lay;
                    rebuild = true;
                }
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
        if let Some(x) = get_usize(v, "layers", 0, MAX_LAYERS) {
            self.layers = x;
        }
    }
}

fn base_growth_readout(p: usize, q: usize, kind: GeometryKind, layers: usize) -> Option<(Vec<Vec<String>>, Vec<String>)> {
    match kind {
        GeometryKind::Spherical => {
            let u = build_unified_pq(p, q, layers).ok()?;
            let g = grow_s2(p, &u.faces, SeedKind::Cell, layers).ok()?;
            let closed = match g.stopped {
                Some(Stopped::Closed { layer }) => Some(layer as usize),
                _ => None,
            };
            Some((stats_to_rows(&g.stats), non_disk_flags(&g.stats, closed)))
        }
        GeometryKind::Euclidean => {
            let (_, stats) = build_e2_regular_seeded(p, q, layers.clamp(1, 3), SeedKind::Cell).ok()?;
            Some((stats_to_rows(&stats), non_disk_flags(&stats, None)))
        }
        GeometryKind::Hyperbolic => {
            let vc = grow_vertex_completion(p, q, layers, WYTHOFF_TILE_BUDGET).ok()?;
            let mut set = HypPointSet::new(hyp_dist(vc.base.vertices[0], vc.base.vertices[1]));
            let faces: Vec<Vec<usize>> = (0..vc.faces.len())
                .map(|i| vc.face_vertices(i).into_iter().map(|v| set.insert(v).0).collect())
                .collect();
            let max_l = vc.shell_sizes.len().saturating_sub(1);
            let stats = patch_layer_stats(&faces, &vc.shell, max_l);
            Some((stats_to_rows(&stats), non_disk_flags(&stats, None)))
        }
    }
}

pub(crate) fn build(j: &WyJob) -> Result<S5Out, String> {
    let kind = check_symbol(j.p, j.q)?;
    let (op, op_label) = *OPS.get(j.op).ok_or_else(|| format!("unknown Wythoff operation #{}", j.op))?;
    if j.layers > MAX_LAYERS {
        return Err(format!("{} layers refused: allowed 0..={MAX_LAYERS}", j.layers));
    }
    let t = build_wythoff(j.p, j.q, op, j.layers)?;
    if t.faces.is_empty() {
        return Err(format!(
            "{op_label} of {{{},{}}} at layer 0 has no surrounded {}-valent base vertex yet; increase Layers to >= 1 [PROVED]",
            j.p, j.q, j.q
        ));
    }
    if let Some(f) = t.faces.iter().find(|f| f.len() < 3) {
        return Err(format!(
            "{op_label} of {{{},{}}} has {}-gon (digon) faces, which this view cannot draw or count as a polygon mesh; the Regular {{p,q}} row draws hosohedra and dihedra",
            j.p,
            j.q,
            f.len()
        ));
    }
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
    let stopped = match kind {
        GeometryKind::Hyperbolic => t.stopped.clone().map(|s| format!("Stopped: {s} [COMPUTATIONAL EVIDENCE]")),
        GeometryKind::Euclidean if j.layers > 3 => Some(format!(
            "Stopped at layer 3 (requested {}): E² Wythoff caps the base patch at 3 layers [COMPUTATIONAL EVIDENCE].",
            j.layers
        )),
        GeometryKind::Euclidean if j.layers == 0 => Some(String::from(
            "Base layer 0 promoted to layer 1 on E² so every Wythoff vertex figure closes [COMPUTATIONAL EVIDENCE].",
        )),
        _ => None,
    };

    let seed = if matches!(op, WythoffOp::Snub { .. }) { "; the snub generating point is a numerical root" } else { "" };
    let mut notes = vec![format!(
        "Wythoff construction (Coxeter, Longuet-Higgins & Miller 1954){seed}; the built vertex coordinates are floating point{} [COMPUTATIONAL EVIDENCE]",
        if closed { "" } else { " and the patch is cut at the requested layers" }
    )];
    notes.push(euler_note(closed, t.euler_chi));
    if let Some(s) = &stopped {
        notes.push(s.clone());
    }
    let extra_table = if let Some((g_rows, flags)) = base_growth_readout(j.p, j.q, kind, j.layers) {
        notes.extend(flags);
        (!g_rows.is_empty()).then(|| (GROWTH_HEADERS.to_vec(), g_rows))
    } else {
        None
    };
    if dropped > 0 {
        notes.push(format!("{dropped} face(s) outside the 2D clip window were not drawn."));
    }
    let table_rows = t.face_type_counts.iter().map(|(s, c)| vec![format!("{s}-gon"), c.to_string()]).collect();
    let caption = format!("Wythoff {} of {{{},{}}}: {} faces", t.op_name, j.p, j.q, t.n_faces);
    Ok(S5Out {
        scene2d,
        mesh,
        caption,
        rows,
        notes,
        stopped,
        table: Some((vec!["Face type", "Count"], table_rows)),
        extra_table,
        svg: None,
    })
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
