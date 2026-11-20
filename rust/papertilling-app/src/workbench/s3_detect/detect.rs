//! "Detect defects": the bond graph rebuilt from bare coordinates (or the crystal's own bonds), per-face Frank charges,
//! clusters, surrounding loops and Burgers vectors (`detect_defects_from_coords`).

use papertilling_core::defect_fields::nye_loops::default_cutoff;
use papertilling_core::defect_fields::{BondGraph, CoordDetection, RefLattice, detect_defects_from_coords};
use papertilling_core::types::Frame2D;

use super::graph_check::check_plane;
use super::{SCENE_TITLE, Settings, Source};
use crate::canvas2d::Scene2D;
use crate::scene2d::LinePrim;
use crate::workbench::scene_helpers::{vector_scene, with_lines};

pub(crate) const POS_TINT: [u8; 4] = [231, 76, 60, 120];
pub(crate) const NEG_TINT: [u8; 4] = [52, 152, 219, 120];
const LOOP_RGBA: [u8; 4] = [192, 57, 43, 255];
const LOOP_SELECTED_RGBA: [u8; 4] = [120, 20, 20, 255];

pub(crate) struct DetectOut {
    pub(crate) det: CoordDetection,
    pub(crate) pos: Vec<[f64; 2]>,
    /// Longest ideal bond of the cell (length unit of the controls).
    pub(crate) lmax: f64,
    pub(crate) graph_source: String,
    /// Whether the cutoff graph equals the crystal's own bonds (`None` when the own bonds were used).
    pub(crate) graph_matches_crystal: Option<bool>,
}

/// Longest ideal bond of the source cell (identity frame: `gallery.cell()` already carries the lattice frame).
pub(crate) fn ref_lattice(src: &Source) -> RefLattice {
    RefLattice::new(&src.cell, Frame2D::default())
}

pub(crate) fn build(src: &Source, set: &Settings) -> Result<DetectOut, String> {
    if src.pos.len() < 3 {
        return Err(format!("REFUSED: the source crystal has {} sites (need at least 3)", src.pos.len()));
    }
    if src.pos.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
        return Err(String::from("REFUSED: the source crystal has non-finite coordinates"));
    }
    let rl = ref_lattice(src);
    if !(rl.lmax.is_finite() && rl.lmax > 0.0) {
        return Err(String::from("REFUSED: the lattice cell has no bonds (no ideal bond star to snap to)"));
    }
    // the detector's own graph, rebuilt here first: a drawing that is not plane is refused before any face walk
    let pre = if set.own_bonds {
        BondGraph::from_edges(&src.pos, &src.edges)
    } else {
        BondGraph::from_cutoff(&src.pos, if set.auto_cutoff { default_cutoff(&src.cell, &rl) } else { set.cutoff_rel * rl.lmax })
    };
    check_plane(&src.pos, &pre.edges).map_err(|e| if set.own_bonds { e } else { format!("{e}; lower the cutoff or use the automatic one") })?;
    let own = set.own_bonds.then_some(src.edges.as_slice());
    let cutoff = (!set.own_bonds && !set.auto_cutoff).then_some(set.cutoff_rel * rl.lmax);
    let det = detect_defects_from_coords(&src.pos, &src.group, &src.cell, Frame2D::default(), own, cutoff, set.clearance)?;
    let graph_source = if set.own_bonds {
        String::from("the crystal's own bonds")
    } else {
        format!("distance cutoff {:.4} ({:.3} x longest bond, {})", det.cutoff, det.cutoff / rl.lmax, if set.auto_cutoff { "automatic" } else { "manual" })
    };
    let graph_matches_crystal = (!set.own_bonds).then(|| {
        let mut ce: Vec<(usize, usize)> = src.edges.iter().map(|&(u, v)| (u.min(v), u.max(v))).filter(|(u, v)| u != v).collect();
        ce.sort_unstable();
        ce.dedup();
        ce == det.edges
    });
    Ok(DetectOut { det, pos: src.pos.clone(), lmax: rl.lmax, graph_source, graph_matches_crystal })
}

/// The picked defect, or (automatic) the first one with a non-zero Burgers vector, else 0.
pub(crate) fn resolve(det: &CoordDetection, sel: Option<usize>) -> usize {
    sel.unwrap_or_else(|| det.defects.iter().position(|d| d.burgers.is_some_and(|b| b[0].hypot(b[1]) > 1e-9)).unwrap_or(0))
}

/// Index of the detected defect whose centre is nearest to `w`, within `max_d`.
pub(crate) fn nearest_defect(det: &CoordDetection, w: [f64; 2], max_d: f64) -> Option<usize> {
    let d = |k: usize| (det.defects[k].centre[0] - w[0]).hypot(det.defects[k].centre[1] - w[1]);
    (0..det.defects.len()).filter(|&k| d(k) <= max_d).min_by(|&a, &b| d(a).total_cmp(&d(b)))
}

/// The loops of every detected defect as line primitives (the selected one darker and thicker).
pub(crate) fn loop_lines(o: &DetectOut, selected: usize) -> Vec<LinePrim> {
    let mut out = Vec::new();
    for (k, d) in o.det.defects.iter().enumerate() {
        let (col, w) = if k == selected { (LOOP_SELECTED_RGBA, 3.0) } else { (LOOP_RGBA, 1.8) };
        for s in d.loop_path.windows(2) {
            if let (Some(a), Some(b)) = (o.pos.get(s[0]), o.pos.get(s[1])) {
                out.push(crate::workbench::scene_helpers::line_prim(*a, *b, col, w));
            }
        }
    }
    out
}

/// Crystal with charged faces tinted (red = positive Frank angle, blue = negative) and the defect loops in red.
pub(crate) fn scene(o: &DetectOut, selected: usize) -> Scene2D {
    let charged: Vec<usize> = (0..o.det.faces.len()).filter(|&f| o.det.face_frank[f] != 0.0).collect();
    let faces: Vec<Vec<usize>> = charged.iter().map(|&f| o.det.faces[f].clone()).collect();
    let tint = |k: usize| if o.det.face_frank[charged[k]] > 0.0 { POS_TINT } else { NEG_TINT };
    let note = format!(
        "{} sites, {} bonds, {} charged faces, {} defects; red face = positive Frank angle, blue = negative",
        o.pos.len(),
        o.det.edges.len(),
        charged.len(),
        o.det.defects.len()
    );
    with_lines(vector_scene(&o.pos, &o.det.edges, &faces, tint, SCENE_TITLE, &note), loop_lines(o, selected))
}

pub(crate) fn fmt_v(v: [f64; 2]) -> String {
    format!("({:.4}, {:.4})", v[0] + 0.0, v[1] + 0.0)
}

pub(crate) fn show_inspector(ui: &mut egui::Ui, o: &DetectOut, selected: usize) {
    let det = &o.det;
    ui.label(format!("Bond graph: {} ({} bonds, {} bounded faces)", o.graph_source, det.edges.len(), det.faces.len()));
    if let Some(m) = o.graph_matches_crystal {
        ui.label(format!(
            "[COMPUTATIONAL EVIDENCE] cutoff graph {} the builder's bonds",
            if m { "equals" } else { "DIFFERS from" }
        ));
    }
    ui.label("[COMPUTATIONAL EVIDENCE] plane drawing checked: no two bonds cross, touch or overlap (relative tolerance 1e-9)");
    let net_v: f64 = det.vertex_frank.iter().sum();
    let net_f: f64 = det.face_frank.iter().sum();
    let nz = det.face_frank.iter().filter(|q| **q != 0.0).count();
    ui.label(format!(
        "[PROVED] Frank charges of this graph from snapped corner angles: {nz} charged faces, sum over faces {net_f:+.0}°, \
         over interior vertices {net_v:+.0}°"
    ));
    ui.label(format!("Defective clusters: {}; detected defects: {}", det.clusters.len(), det.defects.len()));
    ui.small(
        "b = sum of snapped ideal bonds round the counter-clockwise loop, in the chart of the loop's first site; \
         the FS/RH vector (Hirth & Lothe, line +z) is -b.",
    );
    ui.separator();
    egui::ScrollArea::vertical().id_salt("wb_s3_defects").max_height(260.0).show(ui, |ui| {
        egui::Grid::new("wb_s3_defects_grid").striped(true).show(ui, |ui| {
            for h in ["#", "kind", "Frank°", "b", "FS/RH", "[uv]", "Weber", "centre"] {
                ui.strong(h);
            }
            ui.end_row();
            for (k, d) in det.defects.iter().enumerate() {
                ui.label(if k == selected { format!("▶{k}") } else { k.to_string() });
                ui.label(d.kind.as_str());
                ui.label(format!("{:+.0}", d.frank_deg));
                ui.label(d.burgers.map_or_else(|| String::from("-"), fmt_v));
                ui.label(d.burgers_fs_rh.map_or_else(|| String::from("-"), fmt_v));
                ui.label(d.uv.map_or_else(|| String::from("-"), |u| format!("[{} {}]", u[0], u[1])));
                ui.label(d.index.as_ref().and_then(|i| i.weber_str.clone()).unwrap_or_else(|| String::from("-")));
                ui.label(fmt_v(d.centre));
                ui.end_row();
            }
        });
    });
    for (k, d) in det.defects.iter().enumerate() {
        ui.small(format!("#{k}: {}", d.reason));
    }
}
