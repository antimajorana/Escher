//! Fusion steps: a defect fusion shown as a short sequence of explicit lattice graphs
//! (`papertilling_core::fusion`). First example: four dislocations of the square-lattice toric code.
//!
//! Nothing here depends on the Gallery lattice and nothing runs in the background: the sequences are
//! built once, a step is a lookup.

use papertilling_core::fusion::{FusionStep, PlanarMap, gate_route, hamiltonian_route};
use serde_json::{Value, json};

use super::{WbCtx, WbPanel, WbScenes};
use crate::theme::ThemeTokens;

mod scene;
#[cfg(test)]
mod tests;

const ROUTE_LABELS: [&str; 2] = ["Fusion by rewiring gates", "Fusion by Hamiltonian changes"];
const SOURCE: &str =
    "Start lattice and edge numbers: Wang, Liu, Lu, Phys. Rev. B 109, 195165 (2024), Fig. 8(a), Eq. (56). Vertex names are ours.";
const ROUTE_SOURCES: [&str; 2] = [
    "Gate sequence: the author's note \"Unitary fusion of four dislocations\". Its last lattice is Eq. (58) of the paper with edge 15 renamed 12 and edge 16 renamed 11.",
    "The two steps of the paper: Eq. (57), Fig. 8(b) and Eq. (58), Fig. 8(c).",
];

pub(crate) struct Panel {
    route: usize,
    /// Current step of each route.
    step: [usize; 2],
    show_numbers: bool,
    show_names: bool,
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
        Self { route: 0, step: [0, 0], show_numbers: true, show_names: true, routes, error }
    }
}

fn op_text(letter: char, labels: &[u32]) -> String {
    labels.iter().map(|l| format!("{letter}{l}")).collect::<Vec<_>>().join(" ")
}

fn signed(d: i64) -> String {
    if d == 0 { String::from("no change") } else { format!("{d:+}") }
}

impl Panel {
    fn steps(&self) -> &[FusionStep] {
        &self.routes[self.route]
    }

    fn current(&self) -> Option<&FusionStep> {
        self.steps().get(self.step[self.route])
    }

    fn set_step(&mut self, k: usize) -> bool {
        let k = k.min(self.steps().len().saturating_sub(1));
        let changed = self.step[self.route] != k;
        self.step[self.route] = k;
        changed
    }

    /// Rows of the counts table: (what, how many now, change since the start lattice).
    pub(super) fn count_rows(now: &PlanarMap, start: &PlanarMap) -> [(&'static str, usize, String); 3] {
        let (a, b) = (now.census(), start.census());
        let row = |name, x: usize, y: usize| (name, x, signed(x as i64 - y as i64));
        [
            row("Inner vertices", a.vertices, b.vertices),
            row("Edges (qubits)", a.edges, b.edges),
            row("Inner faces", a.faces, b.faces),
        ]
    }

    fn counts_grid(ui: &mut egui::Ui, now: &PlanarMap, start: &PlanarMap) {
        egui::Grid::new("wb_s7_counts").num_columns(3).striped(true).show(ui, |ui| {
            ui.strong("In this picture");
            ui.strong("now");
            ui.strong("since step 0");
            ui.end_row();
            for (name, x, change) in Self::count_rows(now, start) {
                ui.label(name);
                ui.label(x.to_string());
                ui.label(change);
                ui.end_row();
            }
        });
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        &ROUTE_LABELS
    }

    fn sub_index(&self) -> usize {
        self.route
    }

    fn select_sub(&mut self, i: usize) -> bool {
        if i < ROUTE_LABELS.len() && i != self.route {
            self.route = i;
            return true;
        }
        false
    }

    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        let mut rebuild = false;
        ui.label("Four dislocations of the square-lattice toric code, brought together and fused.");
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for (i, label) in ROUTE_LABELS.iter().enumerate() {
                if ui.selectable_label(self.route == i, *label).clicked() {
                    rebuild |= self.select_sub(i);
                }
            }
        });
        ui.separator();
        let last = self.steps().len().saturating_sub(1);
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
        for (i, st) in self.steps().iter().enumerate() {
            let line = if st.operation.is_empty() { format!("{i}. Start") } else { format!("{i}. {}", st.operation) };
            if ui.selectable_label(i == k, line).on_hover_text(&st.title).clicked() {
                k = i;
            }
        }
        rebuild |= self.set_step(k);
        ui.separator();
        rebuild |= ui.checkbox(&mut self.show_numbers, "Edge numbers").changed();
        rebuild |= ui.checkbox(&mut self.show_names, "Vertex names").changed();
        ui.add_space(4.0);
        ui.label("Blue faces have four sides. Dark dots are vertices with four edges. Everything else is coloured and numbered.");
        ui.label("Dashed edges and hollow dots continue outside the picture; hollow dots are not counted.");
        ui.label("A dangling edge counts twice in its face, a loop twice at its vertex.");
        ui.label("CX(c -> t): one end of edge t slides along edge c. H(e): Hadamard on edge e.");
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, _cx: &WbCtx, tokens: &ThemeTokens) {
        let (Some(st), Some(start)) = (self.current(), self.steps().first()) else {
            ui.label("No step to show.");
            return;
        };
        ui.strong(&st.title);
        if !st.operation.is_empty() {
            ui.monospace(&st.operation);
        }
        ui.label(&st.explanation);
        ui.separator();
        Self::counts_grid(ui, &st.map, &start.map);
        ui.separator();
        let m = &st.map;
        ui.strong("Vertices");
        egui::Grid::new("wb_s7_vertices").num_columns(3).striped(true).show(ui, |ui| {
            for v in m.complete_vertices() {
                let k = m.verts[v].rot.len();
                ui.label(&m.verts[v].name);
                let edges = egui::RichText::new(format!("{k} edges"));
                ui.label(if k == 4 { edges } else { edges.color(tokens.status_warn) });
                ui.monospace(op_text('X', &m.vertex_op(v)));
                ui.end_row();
            }
        });
        ui.strong("Faces");
        egui::Grid::new("wb_s7_faces").num_columns(2).striped(true).show(ui, |ui| {
            for f in m.bounded_faces() {
                let sides = egui::RichText::new(format!("{} sides", f.sides()));
                ui.label(if f.sides() == 4 { sides } else { sides.color(tokens.status_warn) });
                ui.monospace(op_text('Z', &PlanarMap::face_op(&f)));
                ui.end_row();
            }
        });
        ui.small("X…: the toric-code vertex term; Z…: the face term. An edge met twice (a loop at a vertex, a dangling edge in a face) drops out.");
        ui.separator();
        ui.small("[COMPUTATIONAL EVIDENCE] Counts, valences, face sizes and terms are computed from the graph of each step. They are tested against Eqs. (56)-(58) where the paper gives them, against hand counts at the steps in between, and against the gate-conjugated stabilizers up to signs.");
        ui.small("Signs and couplings are not shown: which anyon the fused state holds is not read off this view.");
        ui.small("The positions are a drawing; only which edge ends where, and in which order round a vertex, is computed.");
        ui.small(SOURCE);
        ui.small(ROUTE_SOURCES[self.route.min(1)]);
    }

    fn submit(&mut self, _cx: &WbCtx, _repaint: Option<&egui::Context>) {}

    fn poll(&mut self) -> bool {
        false
    }

    fn scenes(&self) -> WbScenes {
        let scene2d = match self.current() {
            Some(st) => scene::step_scene(st, self.step[self.route], self.steps().len(), self.show_numbers, self.show_names),
            None => super::scene_helpers::polyline_scene(&[], [0, 0, 0, 0], "Fusion steps", "no step to show"),
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
        json!({ "route": self.route, "step": self.step, "numbers": self.show_numbers, "names": self.show_names })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("Fusion steps state is not an object (ignored)"));
            return;
        };
        let as_index = |x: &Value| x.as_u64().and_then(|n| usize::try_from(n).ok());
        match o.get("route").map(as_index) {
            Some(Some(r)) if r < ROUTE_LABELS.len() => self.route = r,
            Some(_) => notes.push(String::from("Fusion steps: unknown route (kept current)")),
            None => {}
        }
        if let Some(steps) = o.get("step").and_then(Value::as_array) {
            for (r, x) in steps.iter().enumerate().take(ROUTE_LABELS.len()) {
                match as_index(x) {
                    Some(k) if k < self.routes[r].len() => self.step[r] = k,
                    _ => notes.push(format!("Fusion steps: step of route {r} is out of range (kept current)")),
                }
            }
        }
        if let Some(b) = o.get("numbers").and_then(Value::as_bool) {
            self.show_numbers = b;
        }
        if let Some(b) = o.get("names").and_then(Value::as_bool) {
            self.show_names = b;
        }
    }
}
