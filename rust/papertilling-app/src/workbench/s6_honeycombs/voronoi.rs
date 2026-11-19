//! Voronoi (Wigner-Seitz) cells of the SC / FCC / BCC lattices, and Dirichlet domains of closed 3-manifolds with
//! paired faces in matching tints plus the quotient's cell counts, χ and H₁.

use std::collections::{BTreeMap, HashMap};

use papertilling_core::honeycomb3d::{dirichlet_manifold, lattice_voronoi_cell};
use serde_json::{Value, json};

use super::present::{Built, COUNTS_EVIDENCE, CoreOut, palette_rgba, projection_scene};
use crate::mesh3d::wb_mesh::{mesh_from_polygons, wb_face_tint};

/// Core lattice name and display label.
pub(crate) const LATTICES: [(&str, &str); 3] =
    [("SC", "SC: cube"), ("FCC", "FCC: rhombic dodecahedron"), ("BCC", "BCC: truncated octahedron")];

/// Core manifold name and display label.
pub(crate) const MANIFOLDS: [(&str, &str); 5] = [
    ("seifert_weber", "Seifert-Weber dodecahedral space (H³)"),
    ("poincare_homology_sphere", "Poincaré homology sphere (S³)"),
    ("cubic_3_torus", "cubic 3-torus (E³)"),
    ("fcc_3_torus", "FCC 3-torus (E³)"),
    ("bcc_3_torus", "BCC 3-torus (E³)"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum VorMode {
    #[default]
    LatticeCell,
    Manifold,
}

#[derive(Clone, Debug)]
pub(crate) struct VorState {
    pub(crate) mode: VorMode,
    pub(crate) lattice: usize,
    pub(crate) manifold: usize,
}

impl Default for VorState {
    fn default() -> Self {
        Self { mode: VorMode::LatticeCell, lattice: 2, manifold: 0 }
    }
}

fn pick(ui: &mut egui::Ui, id: &str, sel: &mut usize, items: &[(&str, &str)]) -> bool {
    let mut ch = false;
    egui::ComboBox::from_id_salt(id).selected_text(items[*sel].1).show_ui(ui, |ui| {
        for (i, it) in items.iter().enumerate() {
            ch |= ui.selectable_value(sel, i, it.1).changed();
        }
    });
    ch
}

impl VorState {
    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.horizontal(|ui| {
            ch |= ui.radio_value(&mut self.mode, VorMode::LatticeCell, "Lattice Voronoi cell").changed();
            ch |= ui.radio_value(&mut self.mode, VorMode::Manifold, "Closed 3-manifold").changed();
        });
        ch |= match self.mode {
            VorMode::LatticeCell => pick(ui, "wb_s6_vor_lattice", &mut self.lattice, &LATTICES),
            VorMode::Manifold => pick(ui, "wb_s6_vor_manifold", &mut self.manifold, &MANIFOLDS),
        };
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({"mode": format!("{:?}", self.mode), "lattice": LATTICES[self.lattice].0, "manifold": MANIFOLDS[self.manifold].0})
    }

    pub(crate) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        match v.get("mode").and_then(Value::as_str) {
            Some("LatticeCell") => self.mode = VorMode::LatticeCell,
            Some("Manifold") => self.mode = VorMode::Manifold,
            Some(s) => notes.push(format!("S6 panel: unknown Voronoi mode {s:?} (kept current)")),
            None => {}
        }
        for (key, list, slot) in [("lattice", &LATTICES[..], &mut self.lattice), ("manifold", &MANIFOLDS[..], &mut self.manifold)] {
            if let Some(s) = v.get(key).and_then(Value::as_str) {
                match list.iter().position(|it| it.0 == s) {
                    Some(i) => *slot = i,
                    None => notes.push(format!("S6 panel: unknown {key} {s:?} (kept current)")),
                }
            }
        }
    }
}

fn histogram(sizes: impl Iterator<Item = usize>) -> String {
    let mut h: BTreeMap<usize, usize> = BTreeMap::new();
    for s in sizes {
        *h.entry(s).or_default() += 1;
    }
    h.iter().map(|(k, n)| format!("{n} × {k}-gon")).collect::<Vec<_>>().join(", ")
}

pub(crate) fn build_lattice_cell(lattice: &str) -> Result<Built, String> {
    let label = LATTICES.iter().find(|l| l.0 == lattice).map_or(lattice, |l| l.1);
    let cell = lattice_voronoi_cell(lattice)?;
    let edges: Vec<(usize, usize)> = cell.edges.iter().map(|e| (e[0], e[1])).collect();
    let mesh = mesh_from_polygons(&cell.vertices, &cell.faces, &edges, wb_face_tint);
    let title = format!("Voronoi cell of {label}: xy projection");
    let scene2d = projection_scene(&cell.vertices, edges.iter().copied(), Vec::new(), &title, "conventional cube side 2 for FCC and BCC");
    let (v, e, f) = cell.counts();
    let rows = vec![
        (String::from("Lattice"), label.to_string()),
        (String::from("V, E, F"), format!("{v}, {e}, {f}")),
        (String::from("V − E + F"), (v as i64 - e as i64 + f as i64).to_string()),
        (String::from("Faces"), histogram(cell.faces.iter().map(Vec::len))),
    ];
    let notes = vec![
        String::from("[COMPUTATIONAL EVIDENCE: half-space clipping by perpendicular bisectors in floating point; classical cells: cube (8, 12, 6), rhombic dodecahedron (14, 24, 12), truncated octahedron (24, 36, 14)]"),
        String::from("Expected V − E + F = 2 for any convex polyhedron [PROVED: Euler's polyhedron formula]."),
    ];
    let caption = format!("Voronoi cell of {label} — {v} V, {e} E, {f} F");
    Ok(Built { core: CoreOut::Cell(Box::new(cell)), scene2d, mesh, caption, rows, notes })
}

pub(crate) fn build_manifold(name: &str) -> Result<Built, String> {
    let label = MANIFOLDS.iter().find(|m| m.0 == name).map_or(name, |m| m.1);
    let dom = dirichlet_manifold(name)?;
    let q = dom.quotient()?;
    let mut pair_of: HashMap<&[usize], usize> = HashMap::new();
    for (k, p) in dom.pairings.iter().enumerate() {
        for f in [p.face, p.partner] {
            if let Some(face) = dom.faces.get(f) {
                pair_of.insert(face.as_slice(), k);
            }
        }
    }
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for f in &dom.faces {
        for m in 0..f.len() {
            let (a, b) = (f[m], f[(m + 1) % f.len()]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mesh = mesh_from_polygons(&dom.vertices, &dom.faces, &edges, |face| {
        pair_of.get(face).map_or([0.6, 0.6, 0.6, 0.5], |&k| palette_rgba(k, 0.55))
    });
    let title = format!("{label}: domain, xy projection");
    let scene2d = projection_scene(&dom.vertices, edges.iter().copied(), Vec::new(), &title, &dom.note);
    let rows = vec![
        (String::from("Manifold"), label.to_string()),
        (String::from("Geometry"), dom.geometry.clone()),
        (String::from("Domain V, E, F"), format!("{}, {}, {}", dom.vertices.len(), edges.len(), dom.faces.len())),
        (String::from("Face pairs"), dom.pairings.len().to_string()),
        (String::from("Quotient C, F, E, V"), format!("{}, {}, {}, {}", q.cells, q.faces, q.edges, q.vertices)),
        (String::from("Quotient χ"), q.euler.to_string()),
        (String::from("Edge class sizes"), format!("{:?}", q.edge_class_sizes)),
        (String::from("Vertex class sizes"), format!("{:?}", q.vertex_class_sizes)),
        (String::from("H₁"), q.h1_label.clone()),
    ];
    let mut notes = vec![
        q.evidence.clone(),
        String::from("Expected χ = 0 for every closed 3-manifold [PROVED: Poincaré duality]."),
        format!("Domain: {}", dom.note),
        String::from("Paired faces share a tint; domain counts are of the polyhedron before gluing."),
    ];
    if dom.geometry == "S3" {
        notes.push(String::from(
            "Combinatorial model only: the drawn dodecahedron is not the spherical one (dihedral angle 2π/3); the counts and H₁ depend only on the gluing.",
        ));
    } else if dom.geometry == "H3" {
        notes.push(String::from("Drawn in the Poincaré ball model of H³ (conformal, not isometric)."));
    }
    notes.push(String::from(COUNTS_EVIDENCE));
    let caption = format!("{label} — Dirichlet domain, {} face pairs", dom.pairings.len());
    Ok(Built { core: CoreOut::Manifold(Box::new(dom), Box::new(q)), scene2d, mesh, caption, rows, notes })
}
