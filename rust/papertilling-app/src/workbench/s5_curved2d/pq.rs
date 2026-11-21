//! Regular {p,q}: one switch over S² (δ > 0), E² (δ = 0) and H² (δ < 0) with 0-based seeded layer growth.
//! 2D: stereographic picture (S²), the plane (E²) or the Poincaré disk with geodesic edges (H²).
//! 3D: the unit sphere, the plane z = 0, or the upper sheet of the hyperboloid x² + y² − z² = −1 (exact lift of the
//! faces within Poincaré radius [`LIFT_R_MAX`]; the outer faces are counted, never squashed onto a rim).

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling_seeded};
use papertilling_core::noneuclidean2d::s2_seeds::grow_s2;
use papertilling_core::noneuclidean2d::{build_unified_pq_seeded, poincare_to_hyperboloid_exact};
use serde_json::{Value, json};

use super::draw::{DiskModel, arcs_picture, e2_picture, h2_picture, mesh_with_arcs, s2_picture};
use super::growth_table::{
    GROWTH_HEADERS, build_e2_regular_seeded, non_disk_flags, parse_seed_2d, patch_layer_stats, seed_controls,
    seed_label_2d, stats_to_rows,
};
use super::{S5Out, check_symbol, delta_badge, euler_note, excess_fraction, geom_label, get_usize, pq_controls};
use crate::workbench::scene_helpers::ring_size_tint;

/// 0-based layer range offered for each geometry: S² `0..=6` (closes), H² `0..=8` (cap-reported), E² `0..=12`.
pub(crate) fn layer_range(k: GeometryKind) -> (usize, usize) {
    match k {
        GeometryKind::Spherical => (0, 6),
        GeometryKind::Hyperbolic => (0, 8),
        GeometryKind::Euclidean => (0, 12),
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
    pub(crate) seed: SeedKind,
}

pub(crate) struct PqState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
    pub(crate) seed: SeedKind,
}

impl Default for PqState {
    /// The dodecahedron {5,3} on S²: 20 vertices, 30 edges, 12 faces; Vertex seed default (D19).
    fn default() -> Self {
        Self { p: 5, q: 3, layers: 2, seed: SeedKind::Vertex }
    }
}

impl PqState {
    pub(crate) fn set(&mut self, p: usize, q: usize, layers: usize) {
        (self.p, self.q, self.layers) = (p, q, layers);
    }

    pub(crate) fn job(&self) -> PqJob {
        let layers = match check_symbol(self.p, self.q) {
            Ok(k) => {
                let (lo, hi) = layer_range(k);
                self.layers.clamp(lo, hi)
            }
            Err(_) => self.layers,
        };
        PqJob { p: self.p, q: self.q, layers, seed: self.seed }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = pq_controls(ui, &mut self.p, &mut self.q, 2);
        ui.label(delta_badge(self.p, self.q));
        if let Ok(k) = check_symbol(self.p, self.q) {
            rebuild |= seed_controls(ui, &mut self.seed);
            let (lo, hi) = layer_range(k);
            let mut lay = self.layers.clamp(lo, hi);
            ui.horizontal(|ui| {
                ui.label("Layers:");
                if ui.add(egui::Slider::new(&mut lay, lo..=hi)).changed() {
                    self.layers = lay;
                    rebuild = true;
                }
            });
            match k {
                GeometryKind::Spherical => {
                    ui.small("S²: whole solid drawn; table tracks touching-rule layers 0..=K until closure (χ = 2).");
                }
                GeometryKind::Hyperbolic => {
                    ui.small(
                        "H²: whole shells within 1000 faces and Poincaré centre radius 0.998; stops early with a banner when capped.",
                    );
                }
                GeometryKind::Euclidean => {
                    ui.small("E²: touching-rule growth from the chosen seed on the regular tiling.");
                }
            }
        }
        rebuild
    }

    pub(super) fn save(&self) -> Value {
        json!({ "p": self.p, "q": self.q, "layers": self.layers, "seed": format!("{:?}", self.seed) })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 2, super::MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 2, super::MAX_PQ) {
            self.q = x;
        }
        if let Some(x) = get_usize(v, "layers", 0, 12) {
            self.layers = x;
        }
        if let Some(s) = v.get("seed").and_then(Value::as_str).and_then(parse_seed_2d) {
            self.seed = s;
        }
    }
}

pub(crate) fn build(j: &PqJob) -> Result<S5Out, String> {
    let kind = check_symbol(j.p, j.q)?;
    let (lo, hi) = layer_range(kind);
    if !(lo..=hi).contains(&j.layers) {
        return Err(format!("{} layers refused on {}: allowed {lo}..={hi}", j.layers, geom_label(kind)));
    }
    let (t, stats, closed_layer, mut stopped, h2_shells) = match kind {
        GeometryKind::Spherical => {
            let t = build_unified_pq_seeded(j.p, j.q, j.layers, j.seed)?;
            let g = grow_s2(j.p, &t.faces, j.seed, j.layers)?;
            let closed = match g.stopped {
                Some(Stopped::Closed { layer }) => Some(layer as usize),
                _ => None,
            };
            let stop_msg = closed.map(|l| format!("Closed at layer {l}: S_{l} is the whole sphere (χ = 2) [PROVED]"));
            (t, g.stats, closed, stop_msg, None)
        }
        GeometryKind::Euclidean => {
            let (t, stats) = build_e2_regular_seeded(j.p, j.q, j.layers, j.seed)?;
            (t, stats, None, None, None)
        }
        GeometryKind::Hyperbolic => {
            let h = build_hyperbolic_tiling_seeded(j.p, j.q, HyperbolicStyle::RegularPQ, j.layers, 0, j.seed)?;
            let t = build_unified_pq_seeded(j.p, j.q, j.layers, j.seed)?;
            if h.faces.len() != t.n_faces {
                return Err(format!("core builds disagree on {{{},{}}}: {} vs {} faces", j.p, j.q, h.faces.len(), t.n_faces));
            }
            let stats = patch_layer_stats(&h.faces, &h.face_shell, h.complete_through_shell);
            let stop_msg = (h.complete_through_shell < j.layers).then(|| {
                format!(
                    "Stopped at layer {} (requested {}): {} [COMPUTATIONAL EVIDENCE]",
                    h.complete_through_shell, j.layers, h.growth_rule
                )
            });
            (t, stats, None, stop_msg, Some((h.complete_through_shell, h.growth_rule)))
        }
    };

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
        (String::from("Seed"), seed_label_2d(j.seed).to_string()),
        (String::from("V / E / F"), format!("{} / {} / {}", t.n_vertices, t.n_edges, t.n_faces)),
        (String::from(chi_label), t.euler_chi.to_string()),
        (String::from("2D picture"), view.to_string()),
        (String::from("3D model"), model3d.clone()),
    ];
    match (&h2_shells, kind) {
        (Some((complete, _)), _) => {
            rows.insert(5, (String::from("Layers requested (0..=8)"), t.layers.to_string()));
            rows.insert(6, (String::from("Complete through shell"), complete.to_string()));
        }
        (None, _) => {
            rows.insert(5, (String::from("Layers"), j.layers.to_string()));
        }
    }

    let mut notes = vec![String::from(
        "Geometry from the sign of δ (Coxeter, Regular Polytopes, §1.7, §4.5) [PROVED]; the built vertex coordinates are floating point [COMPUTATIONAL EVIDENCE]",
    )];
    notes.push(euler_note(kind == GeometryKind::Spherical, t.euler_chi));
    let flags = non_disk_flags(&stats, closed_layer);
    if stopped.is_none()
        && let Some(first_flag) = flags.first()
    {
        stopped = Some(first_flag.clone());
    }
    if let Some(s) = &stopped {
        notes.push(s.clone());
    }
    for fl in flags {
        if stopped.as_ref() != Some(&fl) {
            notes.push(fl);
        }
    }
    if let Some((_, rule)) = &h2_shells {
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
    let table = (!stats.is_empty()).then(|| (GROWTH_HEADERS.to_vec(), stats_to_rows(&stats)));
    let caption = format!("{{{},{}}} on the {model3d}: {} faces", j.p, j.q, t.n_faces);
    Ok(S5Out { scene2d, mesh, caption, rows, notes, stopped, table, extra_table: None, svg: None })
}
