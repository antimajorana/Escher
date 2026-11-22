//! Live cavity core-reconnection inspection for arbitrary 2D defects sent from `Place & fuse`,
//! plus shared geometry-table and formatting helpers for `s7_fusion`.

use std::collections::BTreeMap;

use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::fusion::{
    CavityExtraction, CoreEnumeration, PlanarMap, enumerate_cavity_cores, enumerate_defect_cavity_reconnections,
};
use papertilling_core::types::{BuiltCrystal, Cell2D, Frame2D};

use super::scene;
use crate::canvas2d::Scene2D;
use crate::theme::ThemeTokens;
use crate::workbench::scene_helpers::polyline_scene;

/// Safety cap for interactive live cavity core enumeration.
const LIVE_MAX_CORES: usize = 256;

/// Request sent from `Place & fuse` (`CombineTabState`) to inspect cavity core reconnections in `s7_fusion`.
#[derive(Clone, Debug)]
pub(crate) struct CavityRequest {
    pub group: String,
    pub cell: Cell2D,
    pub frame: Frame2D,
    pub lattice_name: String,
    pub specs: Vec<DefectSpec>,
    pub existing_crystal: Option<BuiltCrystal>,
}

/// Live cavity extraction & core enumeration state inside `s7_fusion::Panel`.
pub(crate) struct LoadedCavity {
    pub(crate) request: CavityRequest,
    pub(crate) extra_rings: usize,
    pub(crate) inner_edges: usize,
    pub(crate) max_inner_verts: usize,
    pub(crate) regular_sides: Option<usize>,
    pub(crate) regular_valence: Option<usize>,
    pub(crate) extraction: Option<CavityExtraction>,
    pub(crate) enumeration: Option<CoreEnumeration>,
    /// `0` = reference filled cavity, `1..=cores.len()` = enumerated core `selected - 1`.
    pub(crate) selected: usize,
    pub(crate) error: Option<String>,
}

pub(super) fn op_text(letter: char, labels: &[u32]) -> String {
    labels.iter().map(|l| format!("{letter}{l}")).collect::<Vec<_>>().join(" ")
}

pub(super) fn edge_list_text(labels: &[u32]) -> String {
    labels.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
}

pub(super) fn signed(d: i64) -> String {
    if d == 0 { String::from("no change") } else { format!("{d:+}") }
}

pub(super) fn fmt_sorted(mut xs: Vec<usize>) -> String {
    xs.sort_unstable();
    xs.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

pub(super) fn slide_reachability_note(slide_connected_to_regular: Option<bool>) -> &'static str {
    match slide_connected_to_regular {
        Some(true) => {
            "[COMPUTATIONAL EVIDENCE] Slide-move BFS verified every enumerated core is reachable from a regular square core."
        }
        Some(false) => {
            "[COMPUTATIONAL EVIDENCE] Slide-move BFS found at least one enumerated core not reachable from a regular square core."
        }
        None => {
            "[COMPUTATIONAL EVIDENCE] Slide-move reachability not tested (no regular square core in this set, or more than 32 cores)."
        }
    }
}

/// Rows of the counts table: (what, how many now, change since the reference lattice).
pub(super) fn count_rows(
    now: &PlanarMap,
    start: &PlanarMap,
    show_gate_labels: bool,
) -> [(&'static str, usize, String); 3] {
    let (a, b) = (now.census(), start.census());
    let row = |name, x: usize, y: usize| (name, x, signed(x as i64 - y as i64));
    let edges_label = if show_gate_labels { "Edges (qubits)" } else { "Edges" };
    [
        row("Inner vertices", a.vertices, b.vertices),
        row(edges_label, a.edges, b.edges),
        row("Inner faces", a.faces, b.faces),
    ]
}

pub(super) fn counts_grid(
    ui: &mut egui::Ui,
    now: &PlanarMap,
    start: &PlanarMap,
    ref_col: &str,
    show_gate_labels: bool,
) {
    egui::Grid::new("wb_s7_counts").num_columns(3).striped(true).show(ui, |ui| {
        ui.strong("In this picture");
        ui.strong("now");
        ui.strong(ref_col);
        ui.end_row();
        for (name, x, change) in count_rows(now, start, show_gate_labels) {
            ui.label(name);
            ui.label(x.to_string());
            ui.label(change);
            ui.end_row();
        }
    });
}

pub(super) fn show_geometry_tables(
    ui: &mut egui::Ui,
    m: &PlanarMap,
    tokens: &ThemeTokens,
    regular_valence: Option<usize>,
    regular_sides: Option<usize>,
    show_gate_labels: bool,
) {
    ui.strong("Vertices");
    let v_cols = if show_gate_labels { 3 } else { 2 };
    egui::Grid::new("wb_s7_vertices").num_columns(v_cols).striped(true).show(ui, |ui| {
        for v in m.complete_vertices() {
            let k = m.verts[v].rot.len();
            ui.label(&m.verts[v].name);
            let edges = egui::RichText::new(format!("{k} edges"));
            let regular = regular_valence.is_some_and(|reg| k == reg);
            ui.label(if regular || regular_valence.is_none() { edges } else { edges.color(tokens.status_warn) });
            if show_gate_labels {
                ui.monospace(op_text('X', &m.vertex_op(v)));
            }
            ui.end_row();
        }
    });
    ui.strong("Faces");
    egui::Grid::new("wb_s7_faces").num_columns(2).striped(true).show(ui, |ui| {
        for f in m.bounded_faces() {
            let sides = egui::RichText::new(format!("{} sides", f.sides()));
            let regular = regular_sides.is_some_and(|reg| f.sides() == reg);
            ui.label(if regular || regular_sides.is_none() { sides } else { sides.color(tokens.status_warn) });
            if show_gate_labels {
                ui.monospace(op_text('Z', &PlanarMap::face_op(&f)));
            } else {
                ui.small(format!("edges {}", edge_list_text(&f.labels)));
            }
            ui.end_row();
        }
    });
    if show_gate_labels {
        ui.small("X…: vertex term; Z…: face term. An edge met twice (a loop at a vertex, a dangling edge in a face) drops out.");
    }
}

fn infer_regular_valence(cell: &Cell2D) -> Option<usize> {
    let info = LatticeCoreInfo::new(cell);
    let first = *info.perfect_degree.first()?;
    info.perfect_degree.iter().all(|&d| d == first).then_some(first)
}

fn infer_regular_sides(regular_valence: Option<usize>, reference: &PlanarMap) -> Option<usize> {
    let faces = reference.census().face_sides;
    if faces.is_empty() {
        return None;
    }
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &s in &faces {
        *counts.entry(s).or_insert(0) += 1;
    }
    let expected = match regular_valence {
        Some(3) => Some(6),
        Some(4) => Some(4),
        Some(6) => Some(3),
        _ => None,
    };
    if let Some(exp) = expected
        && counts.contains_key(&exp)
    {
        return Some(exp);
    }
    let (&mode_s, &mode_cnt) = counts.iter().max_by_key(|&(_, c)| c)?;
    (2 * mode_cnt > faces.len()).then_some(mode_s)
}

impl LoadedCavity {
    pub(crate) fn from_request(request: CavityRequest) -> Self {
        let regular_valence = infer_regular_valence(&request.cell);
        let mut live = Self {
            request,
            extra_rings: 0,
            inner_edges: 1,
            max_inner_verts: 1,
            regular_sides: None,
            regular_valence,
            extraction: None,
            enumeration: None,
            selected: 0,
            error: None,
        };
        live.reextract_and_enumerate();
        live
    }

    pub(crate) fn reextract_and_enumerate(&mut self) {
        let req = &self.request;
        match enumerate_defect_cavity_reconnections(
            &req.group,
            &req.cell,
            req.frame,
            &req.specs,
            req.existing_crystal.as_ref(),
            self.extra_rings,
            None,
            self.max_inner_verts,
            LIVE_MAX_CORES,
        ) {
            Ok((ext, en)) => {
                self.regular_sides = infer_regular_sides(self.regular_valence, &ext.reference_filled);
                self.inner_edges = en.inner_edges;
                self.extraction = Some(ext);
                self.enumeration = Some(en);
                self.selected = 0;
                self.error = None;
            }
            Err(err) => {
                self.extraction = None;
                self.enumeration = None;
                self.selected = 0;
                self.error = Some(err);
            }
        }
    }

    pub(crate) fn reenumerate_only(&mut self) {
        let Some(ext) = &self.extraction else {
            self.reextract_and_enumerate();
            return;
        };
        let first_label = ext.cavity_map.edges.iter().map(|e| e.label).max().unwrap_or(0) + 1;
        match enumerate_cavity_cores(
            &ext.cavity_map,
            &ext.cavity_cycle,
            self.inner_edges,
            self.max_inner_verts,
            false,
            first_label,
            LIVE_MAX_CORES,
        ) {
            Ok(mut en) => {
                en.surrounding_name = ext.title.clone();
                self.selected = self.selected.min(en.cores.len());
                self.enumeration = Some(en);
                self.error = None;
            }
            Err(err) => {
                self.enumeration = None;
                self.selected = 0;
                self.error = Some(err);
            }
        }
    }

    /// Returns `(rebuild_scene, switch_to_wang)`.
    pub(super) fn show_controls(
        &mut self,
        ui: &mut egui::Ui,
        show_numbers: &mut bool,
        show_names: &mut bool,
        show_gate_labels: &mut bool,
    ) -> (bool, bool) {
        let mut rebuild = false;
        let mut switch_to_wang = false;

        ui.strong(format!("Live cavity: {} ({})", self.request.lattice_name, self.request.group));
        if let Some(ext) = &self.extraction {
            ui.small(format!("Source: {}", ext.far_field_source));
            ui.small(format!("Algebraic total charge: {}", ext.algebraic_charge));
            ui.small(format!(
                "{}-gon cavity · original interior: {} vert(s), {} edge(s)",
                ext.cavity_cycle.len(),
                ext.orig_inner_verts,
                ext.orig_inner_edges
            ));
        }
        if ui.button("Switch to Wang Fig. 8(a) example").clicked() {
            switch_to_wang = true;
        }
        ui.separator();

        let mut rings_changed = false;
        ui.horizontal(|ui| {
            ui.label("Extra rings:");
            for r in 0..=2_usize {
                if ui.selectable_label(self.extra_rings == r, r.to_string()).clicked() && self.extra_rings != r {
                    self.extra_rings = r;
                    rings_changed = true;
                }
            }
        });
        if rings_changed {
            self.reextract_and_enumerate();
            rebuild = true;
        }

        let mut enum_changed = false;
        let max_edges = self.extraction.as_ref().map_or(8, |e| (e.orig_inner_edges + 4).max(6));
        ui.horizontal(|ui| {
            ui.label("Interior edges:");
            if ui.add(egui::DragValue::new(&mut self.inner_edges).range(1..=max_edges)).changed() {
                enum_changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Max interior vertices:");
            for v in 0..=1_usize {
                if ui.selectable_label(self.max_inner_verts == v, v.to_string()).clicked()
                    && self.max_inner_verts != v
                {
                    self.max_inner_verts = v;
                    enum_changed = true;
                }
            }
        });
        if enum_changed {
            self.reenumerate_only();
            rebuild = true;
        }

        ui.separator();
        if let Some(ext) = &self.extraction {
            let mut sel = self.selected;
            let ref_label = format!(
                "0. Reference filled cavity (V_in={}, E_in={})",
                ext.orig_inner_verts, ext.orig_inner_edges
            );
            if ui.selectable_label(sel == 0, ref_label).clicked() {
                sel = 0;
            }
            if let Some(en) = &self.enumeration {
                ui.add_space(4.0);
                ui.strong(format!(
                    "{} labelled core(s) in {} symmetry orbit(s) (cavity symmetry order {})",
                    en.cores.len(),
                    en.symmetry_orbit_count,
                    en.cavity_symmetry_order
                ));
                if en.truncated {
                    ui.small(format!("Truncated at safety cap ({LIVE_MAX_CORES} cores)."));
                }
                if en.cores.is_empty() {
                    ui.small("No valid lattice cores match this interior-edge / vertex budget.");
                } else {
                    egui::ScrollArea::vertical().id_salt("s7_live_cores_list").max_height(220.0).show(ui, |ui| {
                        for core in &en.cores {
                            let idx = core.id + 1;
                            let line = format!(
                                "{idx}. {} [orbit #{} (size {}) | V_in={}]",
                                core.title,
                                core.orbit_id + 1,
                                core.orbit_size,
                                core.inner_verts
                            );
                            let tip = if core.is_regular_square { "regular square core" } else { &core.description };
                            if ui.selectable_label(sel == idx, line).on_hover_text(tip).clicked() {
                                sel = idx;
                            }
                        }
                    });
                }
            }
            if sel != self.selected {
                self.selected = sel;
                rebuild = true;
            }
        }

        ui.separator();
        rebuild |= ui.checkbox(show_numbers, "Edge numbers").changed();
        rebuild |= ui.checkbox(show_names, "Vertex names").changed();
        rebuild |= ui.checkbox(show_gate_labels, "Show toric-code / gate annotations").changed();
        (rebuild, switch_to_wang)
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui, tokens: &ThemeTokens, show_gate_labels: bool) {
        let Some(ext) = &self.extraction else {
            ui.strong(format!("Live cavity: {} ({})", self.request.lattice_name, self.request.group));
            ui.label("No cavity extracted for the current defect configuration.");
            return;
        };
        let selected_core =
            self.selected.checked_sub(1).and_then(|i| self.enumeration.as_ref()?.cores.get(i));
        let active_map = selected_core.map_or(&ext.reference_filled, |c| &c.map);

        if let Some(core) = selected_core {
            ui.strong(format!("Core {} — {}", core.id + 1, core.title));
            ui.label(format!(
                "orbit #{} (size {}), V_inner = {}",
                core.orbit_id + 1,
                core.orbit_size,
                core.inner_verts
            ));
            ui.label(&core.description);
        } else {
            ui.strong(format!("Reference filled cavity: {}", ext.title));
            ui.label(&ext.note);
        }
        ui.separator();
        counts_grid(ui, active_map, &ext.reference_filled, "vs reference", show_gate_labels);

        if let Some(en) = &self.enumeration {
            ui.separator();
            ui.strong(format!(
                "{} labelled core(s) · {} symmetry orbit(s) (cavity symmetry order {})",
                en.cores.len(),
                en.symmetry_orbit_count,
                en.cavity_symmetry_order
            ));
            if en.truncated {
                ui.colored_label(
                    tokens.status_warn,
                    format!("Search truncated at safety cap ({LIVE_MAX_CORES} cores)."),
                );
            }
            ui.small(slide_reachability_note(en.slide_connected_to_regular));
            egui::ScrollArea::vertical().id_salt("s7_live_cores_tbl").max_height(180.0).show(ui, |ui| {
                egui::Grid::new("wb_s7_live_table").num_columns(5).striped(true).show(ui, |ui| {
                    ui.strong("Core");
                    ui.strong("Orbit");
                    ui.strong("V/E/F (V_in)");
                    ui.strong("Valences");
                    ui.strong("Faces");
                    ui.end_row();
                    let rc = ext.reference_filled.census();
                    let r_vals = fmt_sorted(rc.valences.iter().map(|&(_, k)| k).collect());
                    let r_faces = fmt_sorted(rc.face_sides.clone());
                    ui.label("0. Reference");
                    ui.monospace("ref");
                    ui.monospace(format!("{}/{}/{} ({})", rc.vertices, rc.edges, rc.faces, ext.orig_inner_verts));
                    ui.monospace(format!("{{{r_vals}}}"));
                    ui.monospace(format!("{{{r_faces}}}"));
                    ui.end_row();
                    for core in &en.cores {
                        let c = &core.census;
                        let vals = fmt_sorted(c.valences.iter().map(|&(_, k)| k).collect());
                        let faces = fmt_sorted(c.face_sides.clone());
                        let name = if core.is_regular_square {
                            egui::RichText::new(format!("{}. {}", core.id + 1, core.title))
                                .strong()
                                .color(tokens.status_ok)
                        } else {
                            egui::RichText::new(format!("{}. {}", core.id + 1, core.title))
                        };
                        ui.label(name);
                        ui.monospace(format!("orbit #{} (size {})", core.orbit_id + 1, core.orbit_size));
                        ui.monospace(format!("{}/{}/{} ({})", c.vertices, c.edges, c.faces, core.inner_verts));
                        ui.monospace(format!("{{{vals}}}"));
                        ui.monospace(format!("{{{faces}}}"));
                        ui.end_row();
                    }
                });
            });
            ui.separator();
            show_geometry_tables(ui, active_map, tokens, self.regular_valence, self.regular_sides, show_gate_labels);
            ui.separator();
            ui.small(format!("[COMPUTATIONAL EVIDENCE] {}", en.provenance));
        } else {
            ui.separator();
            show_geometry_tables(ui, active_map, tokens, self.regular_valence, self.regular_sides, show_gate_labels);
        }
        ui.small(&ext.note);
    }

    pub(super) fn scene(&self, show_numbers: bool, show_names: bool, show_gate_labels: bool) -> Scene2D {
        let Some(ext) = &self.extraction else {
            let note = self.error.as_deref().unwrap_or("no cavity extracted");
            return polyline_scene(&[], [0, 0, 0, 0], "Live cavity", note);
        };
        let base_max_label = ext.cavity_map.edges.iter().map(|e| e.label).max().unwrap_or(0);
        let interior_labels = |m: &PlanarMap| -> Vec<u32> {
            m.edges.iter().filter(|e| e.alive && e.label > base_max_label).map(|e| e.label).collect()
        };
        if let Some(i) = self.selected.checked_sub(1)
            && let Some(en) = &self.enumeration
            && let Some(core) = en.cores.get(i)
        {
            let hi = interior_labels(&core.map);
            let title = format!(
                "Core {} of {} (orbit #{}): {}",
                core.id + 1,
                en.cores.len(),
                core.orbit_id + 1,
                core.title
            );
            return scene::map_scene(
                &core.map,
                &hi,
                title,
                self.regular_sides,
                self.regular_valence,
                show_numbers,
                show_names,
                show_gate_labels,
            );
        }
        let hi = interior_labels(&ext.reference_filled);
        let title = format!("Reference cavity: {}", ext.title);
        scene::map_scene(
            &ext.reference_filled,
            &hi,
            title,
            self.regular_sides,
            self.regular_valence,
            show_numbers,
            show_names,
            show_gate_labels,
        )
    }
}
