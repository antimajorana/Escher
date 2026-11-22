//! Core connections & fusion routes: explicit planar lattice graphs for one fixed surrounding cavity
//! (`papertilling_core::fusion`). First example: Wang Fig. 8(a)'s 8-gon cavity (12 fixed edges).
//!
//! Geometry is shown by default (vertex valences, face sizes, core connections, routes between cores).
//! Toric-code stabilizer terms (`X`/`Z`) and gate notation (`CX`/`H`) are an optional annotation, off by default.

use papertilling_core::fusion::{
    CoreConnection, CoreEnumeration, FusionStep, PlanarMap, gate_route, hamiltonian_route, wang_core_connections,
};
use serde_json::{Value, json};

use super::{WbCtx, WbPanel, WbScenes};
use crate::theme::ThemeTokens;

mod scene;
#[cfg(test)]
mod tests;

/// Sub-row cards exposed to the Command Deck (matches `realms.rs` `0..=1`).
const DECK_LABELS: [&str; 2] = ["Core connections", "Routes between cores"];
const VIEW_LABELS: [&str; 3] = [
    "Core connections",
    "Route 1: edge slides (5 steps)",
    "Route 2: merge & add (2 steps)",
];
const SOURCE: &str =
    "Start lattice and edge numbers: Wang, Liu, Lu, Phys. Rev. B 109, 195165 (2024), Fig. 8(a), Eq. (56). Vertex names are ours.";
const ROUTE_SOURCES: [&str; 2] = [
    "Route 1 (edge slides): 5 edge-preserving moves ending at Eq. (58) of the paper with edge 15 renamed 12 and edge 16 renamed 11.",
    "Route 2 (merge & add): the two steps of the paper, Eq. (57), Fig. 8(b) and Eq. (58), Fig. 8(c).",
];
const UNIQUENESS_PROVED: &str = "[COMPUTATIONAL EVIDENCE, exhaustive search] With 2 interior edges and valence ≥ 3, sides ≥ 3, all 5 lattice cores on {c,d,g,h} are listed; even over all 120 cavity chord pairs, {c–h, d–g} is the unique regular square core.";

const ROUTE1_GEOM_LABELS: [&str; 6] = [
    "Start (four dislocations)",
    "Slide edge 12 along 11",
    "Leaf 11 -> loop",
    "Slide loop 11 along 12",
    "Slide edge 11 along 3",
    "Slide edge 11 along 9",
];
const ROUTE2_GEOM_LABELS: [&str; 3] = [
    "Start (four dislocations)",
    "Merge edges 11, 12 -> 15",
    "Add edge 16 (h to c)",
];

pub(crate) struct Panel {
    /// Active sub-view: 0 = Core connections, 1 = Route 1 (edge slides), 2 = Route 2 (merge & add).
    route: usize,
    /// Selected index per sub-view: [core_index (0..5 cores, 5 = Fig. 8a start), route1_step, route2_step].
    step: [usize; 3],
    show_numbers: bool,
    show_names: bool,
    show_gate_labels: bool,
    enumeration: CoreEnumeration,
    routes: [Vec<FusionStep>; 2],
    error: Option<String>,
}

impl Default for Panel {
    fn default() -> Self {
        let mut error = None;
        let mut take = |r: Result<Vec<FusionStep>, String>| {
            r.unwrap_or_else(|e| {
                error = Some(match error.take() {
                    Some(first) => format!("{first}; {e}"),
                    None => e,
                });
                Vec::new()
            })
        };
        let routes = [take(gate_route()), take(hamiltonian_route())];
        let enumeration = wang_core_connections();
        Self {
            route: 0,
            step: [0, 0, 0],
            show_numbers: true,
            show_names: true,
            show_gate_labels: false,
            enumeration,
            routes,
            error,
        }
    }
}

fn op_text(letter: char, labels: &[u32]) -> String {
    labels.iter().map(|l| format!("{letter}{l}")).collect::<Vec<_>>().join(" ")
}

fn edge_list_text(labels: &[u32]) -> String {
    labels.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
}

fn signed(d: i64) -> String {
    if d == 0 { String::from("no change") } else { format!("{d:+}") }
}

fn fmt_sorted(mut xs: Vec<usize>) -> String {
    xs.sort_unstable();
    xs.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn step_list_label(route_idx: usize, step_idx: usize, st: &FusionStep, show_gate_labels: bool) -> String {
    if show_gate_labels {
        return if st.operation.is_empty() {
            format!("{step_idx}. Start")
        } else {
            format!("{step_idx}. {}", st.operation)
        };
    }
    let geom = match route_idx {
        0 => ROUTE1_GEOM_LABELS.get(step_idx).copied(),
        1 => ROUTE2_GEOM_LABELS.get(step_idx).copied(),
        _ => None,
    };
    format!("{step_idx}. {}", geom.unwrap_or(&st.title))
}

fn step_explanation(route_idx: usize, step_idx: usize, st: &FusionStep, show_gate_labels: bool) -> &str {
    if show_gate_labels {
        return &st.explanation;
    }
    match (route_idx, step_idx) {
        (0, 5) => "The end of edge 11 at g slides along edge 9 to vertex h. Every inner face has four sides and every inner vertex four edges again, reaching the unique regular square core {c–h, d–g}.",
        (1, 1) => "The two edges meeting at the two-valent vertex s merge into one edge 15 from d to g, and s is removed. One edge fewer (E = 13); the seven-sided face now has six sides.",
        (1, 2) => "A new edge 16 from h to c splits the six-sided face into two four-sided faces, restoring E = 14 and reaching the same regular square core.",
        _ => &st.explanation,
    }
}

impl Panel {
    fn start_step(&self) -> Option<&FusionStep> {
        self.routes[0].first()
    }

    fn item_count(&self, view: usize) -> usize {
        match view {
            0 => self.enumeration.cores.len() + usize::from(self.start_step().is_some()),
            1 => self.routes[0].len(),
            2 => self.routes[1].len(),
            _ => 0,
        }
    }

    fn active_route(&self) -> Option<(usize, &[FusionStep])> {
        match self.route {
            1 => Some((0, &self.routes[0])),
            2 => Some((1, &self.routes[1])),
            _ => None,
        }
    }

    fn selected_core(&self) -> Option<&CoreConnection> {
        (self.route == 0).then(|| self.enumeration.cores.get(self.step[0])).flatten()
    }

    fn current_step(&self) -> Option<(usize, usize, &FusionStep)> {
        let (r, steps) = self.active_route()?;
        let k = self.step[self.route];
        steps.get(k).map(|st| (r, k, st))
    }

    fn set_step(&mut self, k: usize) -> bool {
        let k = k.min(self.item_count(self.route).saturating_sub(1));
        let changed = self.step[self.route] != k;
        self.step[self.route] = k;
        changed
    }

    /// Rows of the counts table: (what, how many now, change since the start lattice).
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

    fn counts_grid(ui: &mut egui::Ui, now: &PlanarMap, start: &PlanarMap, ref_col: &str, show_gate_labels: bool) {
        egui::Grid::new("wb_s7_counts").num_columns(3).striped(true).show(ui, |ui| {
            ui.strong("In this picture");
            ui.strong("now");
            ui.strong(ref_col);
            ui.end_row();
            for (name, x, change) in Self::count_rows(now, start, show_gate_labels) {
                ui.label(name);
                ui.label(x.to_string());
                ui.label(change);
                ui.end_row();
            }
        });
    }

    fn show_geometry_tables(ui: &mut egui::Ui, m: &PlanarMap, tokens: &ThemeTokens, show_gate_labels: bool) {
        ui.strong("Vertices");
        let v_cols = if show_gate_labels { 3 } else { 2 };
        egui::Grid::new("wb_s7_vertices").num_columns(v_cols).striped(true).show(ui, |ui| {
            for v in m.complete_vertices() {
                let k = m.verts[v].rot.len();
                ui.label(&m.verts[v].name);
                let edges = egui::RichText::new(format!("{k} edges"));
                ui.label(if k == 4 { edges } else { edges.color(tokens.status_warn) });
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
                ui.label(if f.sides() == 4 { sides } else { sides.color(tokens.status_warn) });
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

    fn show_cores_comparison_table(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("All 5 lattice cores for this fixed surrounding");
        egui::Grid::new("wb_s7_cores_table").num_columns(4).striped(true).show(ui, |ui| {
            ui.strong("Core");
            ui.strong("V/E/F");
            ui.strong("Valences");
            ui.strong("Faces");
            ui.end_row();
            for core in &self.enumeration.cores {
                let c = &core.census;
                let vals = fmt_sorted(c.valences.iter().map(|&(_, k)| k).collect());
                let faces = fmt_sorted(c.face_sides.clone());
                let name = if core.is_regular_square {
                    egui::RichText::new(format!("{}. {}", core.id + 1, core.title)).strong().color(tokens.status_ok)
                } else {
                    egui::RichText::new(format!("{}. {}", core.id + 1, core.title))
                };
                ui.label(name);
                ui.monospace(format!("{}/{}/{}", c.vertices, c.edges, c.faces));
                ui.monospace(format!("{{{vals}}}"));
                ui.monospace(format!("{{{faces}}}"));
                ui.end_row();
            }
        });
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        &DECK_LABELS
    }

    fn sub_index(&self) -> usize {
        if self.route == 0 { 0 } else { 1 }
    }

    fn select_sub(&mut self, i: usize) -> bool {
        match i {
            0 if self.route != 0 => {
                self.route = 0;
                true
            }
            1 if self.route == 0 => {
                self.route = 1;
                true
            }
            _ => false,
        }
    }

    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        let mut rebuild = false;
        ui.label("How one fixed 8-gon cavity (12 fixed edges) can be connected inside, and two routes between cores.");
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for (i, label) in VIEW_LABELS.iter().enumerate() {
                if ui.selectable_label(self.route == i, *label).clicked() && self.route != i {
                    self.route = i;
                    rebuild = true;
                }
            }
        });
        ui.separator();

        if self.route == 0 {
            let mut sel = self.step[0];
            ui.strong("Lattice cores (valence ≥ 3, sides ≥ 3, E = 14)");
            for core in &self.enumeration.cores {
                let vals = fmt_sorted(core.census.valences.iter().map(|&(_, k)| k).collect());
                let faces = fmt_sorted(core.census.face_sides.clone());
                let tag = if core.is_regular_square { "regular" } else { "3,4,5" };
                let label = format!("{}. {}  [{} | {}]", core.id + 1, core.title, vals, faces);
                if ui.selectable_label(sel == core.id, label).on_hover_text(tag).clicked() {
                    sel = core.id;
                }
            }
            if self.start_step().is_some() {
                ui.add_space(4.0);
                let start_idx = self.enumeration.cores.len();
                if ui
                    .selectable_label(sel == start_idx, "Start: Fig. 8(a) (unfused, 2-valent vertex s)")
                    .clicked()
                {
                    sel = start_idx;
                }
            }
            rebuild |= self.set_step(sel);
        } else if let Some((route_idx, steps)) = self.active_route() {
            let last = steps.len().saturating_sub(1);
            let mut k = self.step[self.route];
            ui.horizontal(|ui| {
                if ui.add_enabled(k > 0, egui::Button::new("◀ Back")).clicked() {
                    k -= 1;
                }
                if ui.add_enabled(k < last, egui::Button::new("Next ▶")).clicked() {
                    k += 1;
                }
                ui.label(format!("step {k} of {last}"));
            });
            for (i, st) in steps.iter().enumerate() {
                let line = step_list_label(route_idx, i, st, self.show_gate_labels);
                if ui.selectable_label(i == k, line).on_hover_text(&st.title).clicked() {
                    k = i;
                }
            }
            rebuild |= self.set_step(k);
        }

        ui.separator();
        rebuild |= ui.checkbox(&mut self.show_numbers, "Edge numbers").changed();
        rebuild |= ui.checkbox(&mut self.show_names, "Vertex names").changed();
        rebuild |= ui
            .checkbox(&mut self.show_gate_labels, "Show toric-code / gate labels (CX, H, X/Z)")
            .changed();
        ui.add_space(4.0);
        ui.label("Blue faces have four sides. Dark dots are four-valent vertices. Everything else is coloured and numbered.");
        ui.label("Dashed edges and hollow dots continue outside the picture; hollow dots are not counted.");
        ui.label("A dangling edge counts twice in its face, a loop twice at its vertex.");
        if self.show_gate_labels {
            ui.label("CX(c -> t): one end of edge t slides along edge c. H(e): Hadamard on edge e.");
        }
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, _cx: &WbCtx, tokens: &ThemeTokens) {
        let Some(start) = self.start_step() else {
            ui.label("No lattice to show.");
            return;
        };
        if self.route == 0 {
            if let Some(core) = self.selected_core() {
                ui.strong(&core.title);
                ui.label(&core.description);
                ui.separator();
                Self::counts_grid(ui, &core.map, &start.map, "vs Fig. 8(a)", self.show_gate_labels);
                ui.separator();
                self.show_cores_comparison_table(ui, tokens);
                ui.separator();
                Self::show_geometry_tables(ui, &core.map, tokens, self.show_gate_labels);
            } else {
                ui.strong("Start lattice: Fig. 8(a) (unfused)");
                ui.label("Unfused comparison map with 2-valent interior vertex s (not in the valence ≥ 3 lattice-core list).");
                ui.separator();
                Self::counts_grid(ui, &start.map, &start.map, "vs Fig. 8(a)", self.show_gate_labels);
                ui.separator();
                self.show_cores_comparison_table(ui, tokens);
                ui.separator();
                Self::show_geometry_tables(ui, &start.map, tokens, self.show_gate_labels);
            }
            ui.separator();
            ui.small(UNIQUENESS_PROVED);
            ui.small(format!("[COMPUTATIONAL EVIDENCE] {}", self.enumeration.provenance));
            ui.small(SOURCE);
            return;
        }

        let Some((route_idx, step_idx, st)) = self.current_step() else {
            ui.label("No step to show.");
            return;
        };
        ui.strong(&st.title);
        if self.show_gate_labels && !st.operation.is_empty() {
            ui.monospace(&st.operation);
        }
        ui.label(step_explanation(route_idx, step_idx, st, self.show_gate_labels));
        ui.separator();
        Self::counts_grid(ui, &st.map, &start.map, "since step 0", self.show_gate_labels);
        ui.separator();
        Self::show_geometry_tables(ui, &st.map, tokens, self.show_gate_labels);
        ui.separator();
        ui.small("[COMPUTATIONAL EVIDENCE] Counts, valences and face sizes are computed directly from the planar map of each step and checked against Eqs. (56)-(58) and hand counts.");
        if self.show_gate_labels {
            ui.small("Signs and couplings are not shown: which anyon the fused state holds is not read off this view.");
        }
        ui.small("The positions are a drawing; only which edge ends where, and in which cyclic order round a vertex, is computed.");
        ui.small(SOURCE);
        ui.small(ROUTE_SOURCES[route_idx.min(1)]);
    }

    fn submit(&mut self, _cx: &WbCtx, _repaint: Option<&egui::Context>) {}

    fn poll(&mut self) -> bool {
        false
    }

    fn scenes(&self) -> WbScenes {
        let scene2d = if self.route == 0 {
            if let Some(core) = self.selected_core() {
                scene::core_scene(
                    core,
                    self.enumeration.cores.len(),
                    self.show_numbers,
                    self.show_names,
                    self.show_gate_labels,
                )
            } else if let Some(start) = self.start_step() {
                scene::start_comparison_scene(start, self.show_numbers, self.show_names, self.show_gate_labels)
            } else {
                super::scene_helpers::polyline_scene(&[], [0, 0, 0, 0], "Core connections", "no core to show")
            }
        } else if let Some((_, k, st)) = self.current_step() {
            let total = self.item_count(self.route);
            scene::step_scene(st, k, total, self.show_numbers, self.show_names, self.show_gate_labels)
        } else {
            super::scene_helpers::polyline_scene(&[], [0, 0, 0, 0], "Fusion routes", "no step to show")
        };
        WbScenes { scene2d, mesh: None, caption: String::new(), overlay: None }
    }

    fn is_busy(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    fn save(&self) -> Value {
        json!({
            "route": self.route,
            "step": self.step,
            "numbers": self.show_numbers,
            "names": self.show_names,
            "gate_labels": self.show_gate_labels,
        })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("Fusion steps state is not an object (ignored)"));
            return;
        };
        let as_index = |x: &Value| x.as_u64().and_then(|n| usize::try_from(n).ok());
        // A session saved before the core list existed has two steps and numbers its routes from 0:
        // its route r is view r + 1 now.
        let legacy = o.get("step").and_then(Value::as_array).is_some_and(|s| s.len() == 2);
        match o.get("route").map(as_index).map(|r| r.and_then(|r| r.checked_add(usize::from(legacy)))) {
            Some(Some(r)) if r < VIEW_LABELS.len() => self.route = r,
            Some(_) => notes.push(String::from("Fusion steps: unknown route (kept current)")),
            None => {}
        }
        if let Some(steps) = o.get("step").and_then(Value::as_array) {
            if steps.len() == 2 {
                // Backward compatibility with legacy 2-route session JSON: [gate_step, ham_step].
                for (old_r, x) in steps.iter().enumerate() {
                    let target = old_r + 1;
                    match as_index(x) {
                        Some(k) if k < self.item_count(target) => self.step[target] = k,
                        _ => notes.push(format!("Fusion steps: step of route {old_r} is out of range (kept current)")),
                    }
                }
            } else {
                for (r, x) in steps.iter().enumerate().take(VIEW_LABELS.len()) {
                    match as_index(x) {
                        Some(k) if k < self.item_count(r) => self.step[r] = k,
                        _ => notes.push(format!("Fusion steps: step of route {r} is out of range (kept current)")),
                    }
                }
            }
        }
        if let Some(b) = o.get("numbers").and_then(Value::as_bool) {
            self.show_numbers = b;
        }
        if let Some(b) = o.get("names").and_then(Value::as_bool) {
            self.show_names = b;
        }
        if let Some(b) = o.get("gate_labels").and_then(Value::as_bool) {
            self.show_gate_labels = b;
        }
    }
}
