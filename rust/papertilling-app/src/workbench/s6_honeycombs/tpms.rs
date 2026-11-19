//! Triply periodic P / D / G: labyrinth nets (ball-and-stick over `reps`³ conventional cells, with ghost end
//! sites for bonds that leave the block) and periodic surfaces (nodal marching-tetrahedra meshes or the exact
//! Petrie-Coxeter complexes), unwrapped from the 3-torus face by face. Clean-room core: no net database.

use std::collections::HashMap;

use papertilling_core::honeycomb3d::{
    PeriodicSurface, TpmsKind, TriplyPeriodicNet, nodal_surface_mesh, petrie_coxeter_surface, tpms_net,
};
use serde_json::{Value, json};

use super::present::{Built, CoreOut, projection_scene, square_outline, tiles_scene};
use crate::mesh3d::wb_mesh::{mesh_from_points_edges, mesh_from_polygons, wb_face_tint};
use crate::scene2d::PolyPrim;
use crate::workbench::scene_helpers::heat_rgba;

pub(crate) const KINDS: [TpmsKind; 3] = [TpmsKind::P, TpmsKind::D, TpmsKind::G];
/// Nodal grid resolution offered in the UI (the core accepts 4..=96).
pub(crate) const NODAL_N: std::ops::RangeInclusive<usize> = 8..=32;
pub(crate) const MAX_REPS: u32 = 2;
/// Above this nodal grid size only one cell is drawn (n = 32 over 2³ cells would be about 2 million mesh vertices).
pub(crate) const NODAL_REPS_MAX_N: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum TpmsMode {
    #[default]
    Net,
    Nodal,
    PetrieCoxeter,
}

impl TpmsMode {
    pub(crate) const ALL: [Self; 3] = [Self::Net, Self::Nodal, Self::PetrieCoxeter];

    fn label(self) -> &'static str {
        match self {
            Self::Net => "Labyrinth net",
            Self::Nodal => "Nodal surface mesh",
            Self::PetrieCoxeter => "Petrie-Coxeter polyhedron (exact)",
        }
    }
}

pub(crate) fn kind_label(k: TpmsKind) -> &'static str {
    match k {
        TpmsKind::P => "Schwarz P",
        TpmsKind::D => "Schwarz D",
        TpmsKind::G => "Schoen gyroid G",
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TpmsState {
    pub(crate) kind: TpmsKind,
    pub(crate) mode: TpmsMode,
    pub(crate) labyrinths: u32,
    pub(crate) n: usize,
    pub(crate) reps: u32,
}

impl Default for TpmsState {
    fn default() -> Self {
        Self { kind: TpmsKind::D, mode: TpmsMode::Net, labyrinths: 1, n: 16, reps: 1 }
    }
}

impl TpmsState {
    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.horizontal(|ui| {
            for k in KINDS {
                ch |= ui.radio_value(&mut self.kind, k, kind_label(k)).changed();
            }
        });
        egui::ComboBox::from_id_salt("wb_s6_tpms_mode").selected_text(self.mode.label()).show_ui(ui, |ui| {
            for m in TpmsMode::ALL {
                ch |= ui.selectable_value(&mut self.mode, m, m.label()).changed();
            }
        });
        match self.mode {
            TpmsMode::Net => {
                ui.horizontal(|ui| {
                    ui.label("Labyrinths:");
                    ch |= ui.radio_value(&mut self.labyrinths, 1, "one").changed();
                    ch |= ui.radio_value(&mut self.labyrinths, 2, "both").changed();
                });
            }
            TpmsMode::Nodal => {
                ui.horizontal(|ui| {
                    ui.label("Grid n (n³ cubes):");
                    ch |= ui.add(egui::Slider::new(&mut self.n, NODAL_N)).changed();
                });
            }
            TpmsMode::PetrieCoxeter => {
                ui.label("Exact skew polyhedra exist on P ({6,4|4}) and D ({6,6|3}); none lies on G.");
            }
        }
        ui.horizontal(|ui| {
            ui.label("Cells per side:");
            ch |= ui.add(egui::Slider::new(&mut self.reps, 1..=MAX_REPS)).changed();
        });
        if self.effective_reps() < self.reps {
            ui.label(format!("Grid n > {NODAL_REPS_MAX_N}: one cell drawn (keeps the mesh small)."));
        }
        ch
    }

    /// Cells per side actually drawn: a fine nodal mesh is drawn in one cell only.
    pub(crate) fn effective_reps(&self) -> u32 {
        if self.mode == TpmsMode::Nodal && self.n > NODAL_REPS_MAX_N { 1 } else { self.reps }
    }

    pub(crate) fn save(&self) -> Value {
        json!({"kind": format!("{:?}", self.kind), "mode": format!("{:?}", self.mode), "labyrinths": self.labyrinths, "n": self.n, "reps": self.reps})
    }

    pub(crate) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        if let Some(s) = v.get("kind").and_then(Value::as_str) {
            match KINDS.into_iter().find(|k| format!("{k:?}") == s) {
                Some(k) => self.kind = k,
                None => notes.push(format!("S6 panel: unknown periodic surface {s:?} (kept current)")),
            }
        }
        if let Some(s) = v.get("mode").and_then(Value::as_str) {
            match TpmsMode::ALL.into_iter().find(|m| format!("{m:?}") == s) {
                Some(m) => self.mode = m,
                None => notes.push(format!("S6 panel: unknown periodic mode {s:?} (kept current)")),
            }
        }
        let get = |k: &str| v.get(k).and_then(Value::as_u64);
        if let Some(x) = get("labyrinths") {
            self.labyrinths = x.clamp(1, 2) as u32;
        }
        if let Some(x) = get("n") {
            self.n = (x.min(1000) as usize).clamp(*NODAL_N.start(), *NODAL_N.end());
        }
        if let Some(x) = get("reps") {
            self.reps = x.clamp(1, u64::from(MAX_REPS)) as u32;
        }
    }
}

fn cells(reps: u32) -> Vec<[i32; 3]> {
    let r = reps as i32;
    let mut out = Vec::new();
    for a in 0..r {
        for b in 0..r {
            for c in 0..r {
                out.push([a, b, c]);
            }
        }
    }
    out
}

/// Ball-and-stick points of a net over `reps`³ cells: `(points, kinds, edges, real_sites)`; bonds leaving the block
/// end on ghost sites (listed after the real ones).
type NetDrawing = (Vec<[f64; 3]>, Vec<u8>, Vec<(usize, usize, u8)>, usize);

pub(crate) fn net_drawing(net: &TriplyPeriodicNet, reps: u32) -> NetDrawing {
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut kinds: Vec<u8> = Vec::new();
    let mut ids: HashMap<(usize, [i32; 3]), usize> = HashMap::new();
    let mut site = |j: usize, t: [i32; 3], pts: &mut Vec<[f64; 3]>, kinds: &mut Vec<u8>| -> usize {
        *ids.entry((j, t)).or_insert_with(|| {
            let x = net.positions[j];
            pts.push([0, 1, 2].map(|k| x[k] + f64::from(t[k])));
            kinds.push((net.labyrinth_of[j] % 2) as u8);
            pts.len() - 1
        })
    };
    let block = cells(reps);
    for &t in &block {
        for j in 0..net.positions.len() {
            site(j, t, &mut pts, &mut kinds);
        }
    }
    let real = pts.len();
    let mut edges = Vec::new();
    for &t in &block {
        for &(i, j, off) in &net.edges {
            let u = site(i, t, &mut pts, &mut kinds);
            let v = site(j, [t[0] + off[0], t[1] + off[1], t[2] + off[2]], &mut pts, &mut kinds);
            edges.push((u, v, (net.labyrinth_of[i] % 2) as u8));
        }
    }
    (pts, kinds, edges, real)
}

pub(crate) fn build_net(kind: TpmsKind, labyrinths: u32, reps: u32) -> Result<Built, String> {
    let net = tpms_net(kind, labyrinths)?;
    let (pts, kinds, edges, real) = net_drawing(&net, reps);
    let mesh = mesh_from_points_edges(&pts, &edges, &kinds);
    let r = f64::from(reps);
    let title = format!("{}: xy projection", net.name);
    let note = "fractional coordinates of the conventional cubic cell (a = 1); grey square = cell outline";
    let scene2d = projection_scene(&pts, edges.iter().map(|e| (e.0, e.1)), square_outline(r), &title, note);
    let mut coord: Vec<usize> = net.coordination.clone();
    coord.sort_unstable();
    coord.dedup();
    let rows = vec![
        (String::from("Net"), net.name.clone()),
        (String::from("Space group"), format!("{} (#{})", net.space_group, net.space_group_number)),
        (String::from("Sites per conventional cell"), net.positions.len().to_string()),
        (String::from("Bonds per conventional cell"), net.edges.len().to_string()),
        (String::from("Coordination"), format!("{coord:?}")),
        (String::from("Bond length (a = 1)"), format!("{:.6}", net.bond_length)),
        (String::from("Drawn: sites + ghost ends"), format!("{real} + {}", pts.len() - real)),
    ];
    let notes = vec![
        String::from("[COMPUTATIONAL EVIDENCE: nearest-neighbour search on the exact site coordinates; classical values: pcu 6-coordinated (bond 1), dia 4 (√3/4), srs 3 (√2/4)]"),
        String::from("Clean-room: sites are International Tables Wyckoff positions (221 1a, 227 8a, 214 8a); net names are labels only; no net database is used."),
        String::from("Second labyrinth (red bonds): P and D shifted by (½,½,½); G mirrored x ↦ −x."),
    ];
    let caption = format!("{} — {}³ cells, {real} sites", net.name, reps);
    Ok(Built { core: CoreOut::Net(Box::new(net)), scene2d, mesh, caption, rows, notes })
}

/// Faces unwrapped from the torus, replicated over `reps`³ cells. Each vertex is placed at the minimum image of
/// the previous one along the cycle (face edges are shorter than half a cell; a whole face may not be, e.g. the
/// {6,4|4} hexagons span 0.71 cell).
pub(crate) fn unwrap_faces(s: &PeriodicSurface, reps: u32) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let mut pts = Vec::new();
    let mut faces = Vec::new();
    for t in cells(reps) {
        for f in &s.faces {
            let Some(&first) = f.first() else { continue };
            let mut prev = s.vertices[first];
            let start = pts.len();
            for &v in f {
                let x = s.vertices[v];
                prev = [0, 1, 2].map(|k| {
                    let d = x[k] - prev[k];
                    prev[k] + d - d.round()
                });
                pts.push([0, 1, 2].map(|k| prev[k] + f64::from(t[k])));
            }
            faces.push((start..pts.len()).collect());
        }
    }
    (pts, faces)
}

/// Heat map of the nodal function on the slice z = 0 (`g` tiles per cell side).
fn nodal_slice(kind: TpmsKind, reps: u32, g: usize) -> Vec<PolyPrim> {
    let side = g * reps as usize;
    let h = 1.0 / g as f64;
    let vals: Vec<f64> =
        (0..side * side).map(|k| kind.nodal(&[((k % side) as f64 + 0.5) * h, ((k / side) as f64 + 0.5) * h, 0.0])).collect();
    let vmax = vals.iter().fold(1e-9_f64, |m, v| m.max(v.abs()));
    vals.iter()
        .enumerate()
        .map(|(k, &v)| {
            let (x, y) = ((k % side) as f64 * h, (k / side) as f64 * h);
            let sq = [[x, y], [x + h, y], [x + h, y + h], [x, y + h]].map(|p| [p[0] as f32, p[1] as f32]);
            PolyPrim { points: sq.to_vec(), fill: heat_rgba(v, -vmax, vmax), stroke: None }
        })
        .collect()
}

pub(crate) fn build_surface(kind: TpmsKind, mode: TpmsMode, n: usize, reps: u32) -> Result<Built, String> {
    let s = match mode {
        TpmsMode::PetrieCoxeter => petrie_coxeter_surface(kind)?,
        _ => nodal_surface_mesh(kind, n)?,
    };
    let (pts, faces) = unwrap_faces(&s, reps);
    let exact = mode == TpmsMode::PetrieCoxeter;
    let mesh = if exact {
        let mut edges = Vec::new();
        for f in &faces {
            for m in 0..f.len() {
                edges.push((f[m], f[(m + 1) % f.len()]));
            }
        }
        mesh_from_polygons(&pts, &faces, &edges, wb_face_tint)
    } else {
        mesh_from_polygons(&pts, &faces, &[], |_| [0.30, 0.62, 0.86, 1.0])
    };
    let title = format!("{}: nodal function on the slice z = 0", kind_label(kind));
    let note = "blue < 0 < red; the surface meets this slice where the colour is white";
    let scene2d = tiles_scene(nodal_slice(kind, reps, 40), square_outline(f64::from(reps)), &title, note);
    let rows = vec![
        (String::from("Surface"), s.name.clone()),
        (String::from("V, E, F on the 3-torus"), format!("{}, {}, {}", s.vertices.len(), s.edges.len(), s.faces.len())),
        (String::from("χ per conventional cell"), s.euler_conventional.to_string()),
        (String::from("Side-preserving cells"), s.side_preserving_cells.to_string()),
        (String::from("χ per primitive cell"), s.euler_primitive.to_string()),
        (String::from("Genus per primitive cell"), s.genus_primitive.to_string()),
        (String::from("Connected components"), s.connected_components.to_string()),
        (String::from("Closed 2-manifold"), if s.closed_manifold { "yes" } else { "no" }.to_string()),
    ];
    let notes = vec![
        s.evidence.clone(),
        String::from("Classical value for the minimal surfaces: genus 3 per primitive cell for P, D and G (Schoen, NASA TN D-5541, 1970)."),
        String::from(if exact {
            "Exact combinatorial complex; its skew faces are drawn fan-triangulated."
        } else {
            "The nodal level set approximates, but is not, the minimal surface: only its topology is compared."
        }),
    ];
    let caption = format!("{} — {} faces in {}³ cells", s.name, faces.len(), reps);
    Ok(Built { core: CoreOut::Surface(Box::new(s)), scene2d, mesh, caption, rows, notes })
}
