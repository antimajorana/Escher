//! Regular {p,q}: one switch over S² (δ > 0), E² (δ = 0) and H² (δ < 0) via `build_unified_pq`.
//! 2D: stereographic picture (S²), the plane (E²) or the Poincaré disk with geodesic edges (H²).
//! 3D: the unit sphere, the plane z = 0, or the upper sheet of the hyperboloid x² + y² − z² = −1 (exact lift of the
//! faces within Poincaré radius [`LIFT_R_MAX`]; the outer faces are counted, never squashed onto a rim).

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling};
use papertilling_core::noneuclidean2d::{build_unified_pq, poincare_to_hyperboloid_exact};
use serde_json::{Value, json};

use super::draw::{DiskModel, arcs_picture, e2_picture, h2_picture, mesh_with_arcs, s2_picture};
use super::{S5Out, check_symbol, delta_badge, euler_note, excess_fraction, geom_label, get_usize, pq_controls};
use crate::workbench::scene_helpers::ring_size_tint;

/// Layer range the core honours for each geometry (S² is closed: layers are not used).
pub(crate) fn layer_range(k: GeometryKind) -> (usize, usize) {
    match k {
        GeometryKind::Spherical => (1, 1),
        GeometryKind::Euclidean => (1, 6),
        GeometryKind::Hyperbolic => (1, 4),
    }
}

/// Poincaré radius within which H² faces are lifted to the hyperboloid: height t = (1 + r²)/(1 − r²) ≈ 19.5 at 0.95.
/// Faces reaching farther are left out of the 3D view and counted (they would set the scale and shrink the rest).
pub(crate) const LIFT_R_MAX: f64 = 0.95;

/// Exact hyperboloid lift of the H² faces whose vertices all lie within the radius, and of the geodesic edge samples
/// within it. Points are compacted, so the faces left out (`dropped`) do not set the 3D scale.
pub(crate) struct H2Lift {
    pub(crate) pts: Vec<[f64; 3]>,
    pub(crate) faces: Vec<Vec<usize>>,
    pub(crate) arcs: Vec<Vec<[f64; 3]>>,
    pub(crate) dropped: usize,
}

pub(crate) fn h2_lift(coords: &[[f64; 2]], faces: &[Vec<usize>], arcs: &[Vec<[f64; 2]>], r_max: f64) -> Result<H2Lift, String> {
    let inside = |z: &[f64; 2]| z[0].hypot(z[1]) <= r_max;
    let mut map: Vec<Option<usize>> = vec![None; coords.len()];
    let (mut pts, mut kept, mut dropped) = (Vec::new(), Vec::new(), 0);
    for f in faces {
        if !f.iter().all(|&v| coords.get(v).is_some_and(inside)) {
            dropped += 1;
            continue;
        }
        let mut ring = Vec::with_capacity(f.len());
        for &v in f {
            let id = match map[v] {
                Some(id) => id,
                None => {
                    pts.push(poincare_to_hyperboloid_exact(coords[v])?);
                    map[v] = Some(pts.len() - 1);
                    pts.len() - 1
                }
            };
            ring.push(id);
        }
        kept.push(ring);
    }
    let arcs = arcs
        .iter()
        .filter(|a| a.iter().all(inside))
        .map(|a| a.iter().map(|&z| poincare_to_hyperboloid_exact(z)).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(H2Lift { pts, faces: kept, arcs, dropped })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PqJob {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
}

pub(crate) struct PqState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
}

impl Default for PqState {
    /// The dodecahedron {5,3} on S²: 20 vertices, 30 edges, 12 faces.
    fn default() -> Self {
        Self { p: 5, q: 3, layers: 2 }
    }
}

impl PqState {
    pub(crate) fn set(&mut self, p: usize, q: usize, layers: usize) {
        (self.p, self.q, self.layers) = (p, q, layers);
    }

    pub(crate) fn job(&self) -> PqJob {
        PqJob { p: self.p, q: self.q, layers: self.layers }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = pq_controls(ui, &mut self.p, &mut self.q, 2);
        ui.label(delta_badge(self.p, self.q));
        match check_symbol(self.p, self.q) {
            Ok(GeometryKind::Spherical) => {
                ui.small("Closed surface: the whole tiling is built (no layers).");
            }
            Ok(k) => {
                let (lo, hi) = layer_range(k);
                ui.horizontal(|ui| {
                    ui.label("Layers (vertex-completion shells):");
                    rebuild |= ui.add(egui::DragValue::new(&mut self.layers).range(lo..=hi)).changed();
                });
                if k == GeometryKind::Hyperbolic {
                    ui.small(
                        "H² patch: whole vertex-completion shells while the total stays within 1000 faces; faces centred beyond Poincaré radius 0.998 are dropped, so the outer shell can be partial (see the inspector).",
                    );
                }
            }
            Err(_) => {}
        }
        rebuild
    }

    pub(super) fn save(&self) -> Value {
        json!({ "p": self.p, "q": self.q, "layers": self.layers })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 2, super::MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 2, super::MAX_PQ) {
            self.q = x;
        }
        if let Some(x) = get_usize(v, "layers", 1, 6) {
            self.layers = x;
        }
    }
}

pub(crate) fn build(j: &PqJob) -> Result<S5Out, String> {
    let kind = check_symbol(j.p, j.q)?;
    let (lo, hi) = layer_range(kind);
    if kind != GeometryKind::Spherical && !(lo..=hi).contains(&j.layers) {
        return Err(format!("{} layers refused on {}: allowed {lo}..={hi}", j.layers, geom_label(kind)));
    }
    let t = build_unified_pq(j.p, j.q, j.layers)?;
    let title = format!("S5 regular {{{},{}}}", j.p, j.q);
    let digon = j.p == 2 || j.q == 2;
    let (scene2d, dropped, view) = match kind {
        GeometryKind::Spherical => {
            let note = "stereographic projection from the centre of face 0 (face 0 is the outside)";
            let s2 = if digon { None } else { s2_picture(&t.coords_3d, &t.faces, &title, note) };
            match s2 {
                Some((s, d)) => (s, d, note),
                None => {
                    let note = "stereographic projection of the core's sampled great-circle edges; digon faces not filled";
                    let (s, d) = arcs_picture(&t.edge_arcs_2d, &title, note);
                    (s, d, note)
                }
            }
        }
        GeometryKind::Euclidean => {
            let note = "Euclidean plane, unit edges";
            let (s, d) = e2_picture(&t.coords_2d, &t.faces, &title, note);
            (s, d, note)
        }
        GeometryKind::Hyperbolic => {
            let note = "Poincaré disk; edges are sampled geodesic arcs";
            let (s, d) = h2_picture(&t.coords_2d, &t.faces, |_| ring_size_tint(j.p), DiskModel::Poincare, true, &title, note);
            (s, d, note)
        }
    };
    // H²: the core's own shell bookkeeping (same builder and caps as `build_unified_pq`, so the same faces).
    let shells = if kind == GeometryKind::Hyperbolic {
        let h = build_hyperbolic_tiling(j.p, j.q, HyperbolicStyle::RegularPQ, j.layers, 0)?;
        if h.faces.len() != t.n_faces {
            return Err(format!("core builds disagree on {{{},{}}}: {} vs {} faces", j.p, j.q, h.faces.len(), t.n_faces));
        }
        Some((h.complete_through_shell, h.growth_rule))
    } else {
        None
    };
    let (mesh, model3d, lift_note) = if kind == GeometryKind::Hyperbolic {
        let l = h2_lift(&t.coords_2d, &t.faces, &t.edge_arcs_2d, LIFT_R_MAX)?;
        let note = format!(
            "3D: exact lift z ↦ (2z, 1 + |z|²)/(1 − |z|²) of the {} face(s) whose vertices lie within Poincaré radius {LIFT_R_MAX} (height ≤ 19.5); {} outer face(s) are left out of the 3D view and drawn in 2D only [COMPUTATIONAL EVIDENCE]",
            l.faces.len(),
            l.dropped
        );
        let model = format!("hyperboloid sheet (Minkowski model), faces within Poincaré radius {LIFT_R_MAX}");
        (mesh_with_arcs(&l.pts, &l.faces, Some(&l.arcs)), model, Some(note))
    } else {
        let model = if kind == GeometryKind::Spherical { "unit sphere" } else { "plane z = 0" };
        (mesh_with_arcs(&t.coords_3d, &t.faces, Some(&t.edge_arcs_3d)), model.to_string(), None)
    };
    let chi_label = if kind == GeometryKind::Spherical { "χ = V − E + F (sphere)" } else { "χ = V − E + F (finite patch)" };
    let mut rows = vec![
        (String::from("Geometry"), geom_label(kind).to_string()),
        (String::from("δ = 4 − (p−2)(q−2)"), t.delta.to_string()),
        (String::from("1/p + 1/q − 1/2 = δ/(2pq)"), excess_fraction(j.p, j.q).unwrap_or_default()),
        (String::from("V / E / F"), format!("{} / {} / {}", t.n_vertices, t.n_edges, t.n_faces)),
        (String::from(chi_label), t.euler_chi.to_string()),
        (String::from("2D picture"), view.to_string()),
        (String::from("3D model"), model3d.clone()),
    ];
    match (&shells, kind) {
        (Some((complete, _)), _) => {
            rows.insert(4, (String::from("Layers requested (core range 1..=4)"), t.layers.to_string()));
            rows.insert(5, (String::from("Complete through shell"), complete.to_string()));
        }
        (None, GeometryKind::Euclidean) => rows.insert(4, (String::from("Layers"), t.layers.to_string())),
        (None, _) => {}
    }
    let mut notes = vec![String::from(
        "Geometry from the sign of δ (Coxeter, Regular Polytopes, §1.7, §4.5) [PROVED]; the built vertex coordinates are floating point [COMPUTATIONAL EVIDENCE]",
    )];
    notes.push(euler_note(kind == GeometryKind::Spherical, t.euler_chi));
    if let Some((_, rule)) = &shells {
        notes.push(format!("Patch: {rule} [COMPUTATIONAL EVIDENCE]"));
    }
    notes.extend(lift_note);
    if j.p == 2 {
        notes.push(String::from("Hosohedron: its digon faces cannot be filled as flat polygons; only the great-circle edges are drawn."));
    } else if j.q == 2 {
        notes.push(String::from("Dihedron: its two p-gon faces are the two hemispheres; in 3D they are drawn as two coincident flat disks."));
    }
    if dropped > 0 {
        notes.push(format!("{dropped} face(s) outside the 2D clip window were not drawn."));
    }
    let caption = format!("{{{},{}}} on the {model3d}: {} faces", j.p, j.q, t.n_faces);
    Ok(S5Out { scene2d, mesh, caption, rows, notes, table: None, svg: None })
}
