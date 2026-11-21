//! Flat 3D non-compact slabs: a canonical 3D net on one of the five topologies `T² × R`, `K² × R`, Moebius strip × S¹,
//! cylinder × S¹ and the `S¹ × R²` chimney. A core `IMPOSSIBLE` status is a refusal (its reason is shown verbatim and
//! carries the core's `[PROVED]` stamp); no geometry is drawn for it.

use papertilling_core::flat3d::{Cell3D, FlatSlabKind, SlabPatch3d, build_flat_slab_3d};
use papertilling_core::space3d::nets3d::CanonicalNet3d;
use serde_json::{Value, json};

use super::{Scenes, net_combo, parse_net, points_scene, repeats_row};
use crate::mesh3d::wb_mesh::mesh_from_points_edges;

pub(super) const SCENE_TITLE: &str = "S4 flat 3D slab (xy of the flat cover)";
pub(super) const MAX_REPEAT: usize = 6;

pub(super) const KINDS: [FlatSlabKind; 5] = [
    FlatSlabKind::TorusSlab,
    FlatSlabKind::KleinSlab,
    FlatSlabKind::MoebiusTube,
    FlatSlabKind::CylinderCrossS1,
    FlatSlabKind::CylinderChimney,
];

pub(super) fn kind_label(k: FlatSlabKind) -> &'static str {
    match k {
        FlatSlabKind::TorusSlab => "T² × R (torus slab)",
        FlatSlabKind::KleinSlab => "K² × R (Klein slab)",
        FlatSlabKind::MoebiusTube => "Moebius strip × S¹",
        FlatSlabKind::CylinderCrossS1 => "Cylinder × S¹",
        FlatSlabKind::CylinderChimney => "S¹ × R² (chimney)",
    }
}

/// True for the two kinds the core glues by an in-plane glide: their repeats run along the glide axis, across it
/// and along a3, not along a1, a2, a3.
pub(super) fn glide_glued(k: FlatSlabKind) -> bool {
    matches!(k, FlatSlabKind::KleinSlab | FlatSlabKind::MoebiusTube)
}

/// Repeat-row label for a slab kind.
pub(super) fn repeats_label(k: FlatSlabKind) -> &'static str {
    if glide_glued(k) { "Slab repeats (along glide, across glide, a3):" } else { "Slab repeats (a1, a2, a3):" }
}

/// The glide axes of a glide-glued slab, read back from the core's periodic vectors.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct GlideAxes {
    /// One repeat along the glide axis (Cartesian).
    pub(super) along_cart: [f64; 3],
    /// That vector as an integer combination of the net's a1, a2 (`None` if it is not one to 1e-6).
    pub(super) along: Option<[i64; 2]>,
    /// One repeat across the axis in (a1, a2); the core reports it only for the Klein slab.
    pub(super) across: Option<[i64; 2]>,
}

/// Integer `(c1, c2)` with `v = c1 a1 + c2 a2` (the Gram system of the net's a1, a2), if one fits to 1e-6 |v|.
pub(super) fn lattice_coords(lat: &[[f64; 3]; 3], v: [f64; 3]) -> Option<[i64; 2]> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a1, a2) = (lat[0], lat[1]);
    let (g11, g12, g22) = (dot(a1, a1), dot(a1, a2), dot(a2, a2));
    let det = g11 * g22 - g12 * g12;
    if det.abs() < 1e-12 {
        return None;
    }
    let (r1, r2) = (dot(a1, v), dot(a2, v));
    let c = [((g22 * r1 - g12 * r2) / det).round(), ((g11 * r2 - g12 * r1) / det).round()];
    let back = [0, 1, 2].map(|k| c[0] * a1[k] + c[1] * a2[k] - v[k]);
    (dot(back, back).sqrt() <= 1e-6 * dot(v, v).sqrt().max(1e-12)).then_some([c[0] as i64, c[1] as i64])
}

/// For a glide-glued slab built `OK`: its glide axes (periodic vector k divided by its repeat count).
pub(super) fn glide_axes(net: CanonicalNet3d, p: &SlabPatch3d) -> Option<GlideAxes> {
    if !glide_glued(p.kind) || p.status != "OK" {
        return None;
    }
    let per = |k: usize| -> Option<[f64; 3]> {
        let v = p.periodic_vectors.get(k)?;
        let r = p.repeats[k].max(1) as f64;
        Some([v[0] / r, v[1] / r, v[2] / r])
    };
    let lat = Cell3D::from_canonical_net(net).lattice;
    let along_cart = per(0)?;
    let across = if p.kind == FlatSlabKind::KleinSlab { per(1).and_then(|v| lattice_coords(&lat, v)) } else { None };
    Some(GlideAxes { along_cart, along: lattice_coords(&lat, along_cart), across })
}

fn axis_text(c: Option<[i64; 2]>) -> String {
    c.map_or_else(|| String::from("not an a1, a2 combination"), |[x, y]| format!("{x}·a1 + {y}·a2"))
}

/// Build the slab; a non-`OK` core status is returned as `Err(reason)`.
pub(super) fn build(net: CanonicalNet3d, kind: FlatSlabKind, repeats: [usize; 3]) -> Result<SlabPatch3d, String> {
    let cell = Cell3D::from_canonical_net(net);
    let p = build_flat_slab_3d(&cell, kind, repeats)?;
    if p.status != "OK" {
        return Err(p.reason);
    }
    if p.positions.is_empty() {
        return Err(String::from("the slab patch has no sites (refused, nothing drawn)"));
    }
    Ok(p)
}

pub(super) struct SlabState {
    pub(super) net: CanonicalNet3d,
    pub(super) kind: FlatSlabKind,
    pub(super) repeats: [usize; 3],
    pub(super) out: Option<SlabPatch3d>,
    /// Glide axes of `out` (glide-glued kinds only).
    pub(super) glide: Option<GlideAxes>,
}

impl Default for SlabState {
    fn default() -> Self {
        Self { net: CanonicalNet3d::SimpleCubic, kind: FlatSlabKind::TorusSlab, repeats: [3, 3, 3], out: None, glide: None }
    }
}

impl SlabState {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.repeats.iter().any(|&r| r == 0 || r > MAX_REPEAT) {
            return Err(format!("slab repeats must lie in 1..={MAX_REPEAT}"));
        }
        Ok(())
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = net_combo(ui, "wb_s4_slab_net", &mut self.net);
        egui::ComboBox::from_id_salt("wb_s4_slab_kind").selected_text(kind_label(self.kind)).show_ui(ui, |ui| {
            for k in KINDS {
                ch |= ui.selectable_value(&mut self.kind, k, kind_label(k)).changed();
            }
        });
        ch |= repeats_row(ui, repeats_label(self.kind), &mut self.repeats, MAX_REPEAT);
        ui.small("The compact directions are glued; the others stay open (boundary sites).");
        ch
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(p) = &self.out else {
            ui.label("No slab built yet.");
            return;
        };
        ui.strong(kind_label(p.kind));
        ui.label(format!(
            "[PROVED] H1 = {}, b1 = {}, {} (homotopy invariance + cellular homology, cited in core slabs3d)",
            p.h1_label,
            p.b1,
            if p.orientable { "orientable" } else { "non-orientable" }
        ));
        ui.label(format!("Compact / open directions: {} / {} (manifold dim {})", p.compact_dims, p.noncompact_dims, p.manifold_dim));
        let seams = p.bonds.iter().filter(|b| b.2).count();
        ui.label(format!(
            "[COMPUTATIONAL EVIDENCE] built patch: {} sites, {} bonds ({} across a seam, red), {} boundary sites",
            p.positions.len(),
            p.bonds.len(),
            seams,
            p.boundary_sites.len()
        ));
        if glide_glued(p.kind) {
            ui.label(format!("Repeats {:?} along (glide axis, across it, a3); status {}", p.repeats, p.status));
            match &self.glide {
                Some(g) => {
                    let [x, y, z] = g.along_cart;
                    ui.label(format!("Glide axis: one repeat = {} = ({x:.4}, {y:.4}, {z:.4})", axis_text(g.along)));
                    if p.kind == FlatSlabKind::KleinSlab {
                        ui.label(format!("Across the axis: one repeat = {}", axis_text(g.across)));
                    } else {
                        ui.label("Across the axis: direction not reported by the core");
                    }
                }
                None => {
                    ui.label("Glide axis: not reported by the core");
                }
            }
        } else {
            ui.label(format!("Repeats {:?} along (a1, a2, a3); status {}", p.repeats, p.status));
        }
        ui.small(&p.reason);
    }

    pub(super) fn scenes(&self) -> Option<Scenes> {
        let p = self.out.as_ref()?;
        let xy: Vec<[f64; 2]> = p.flat_positions.iter().map(|q| [q[0], q[1]]).collect();
        let flat: Vec<(usize, usize)> = p.bonds.iter().filter(|b| !b.2).map(|b| (b.0, b.1)).collect();
        let scene = points_scene(&xy, &flat, &p.boundary_sites, SCENE_TITLE, "seam bonds are drawn in 3D only; ringed = boundary");
        let edges: Vec<(usize, usize, u8)> = p.bonds.iter().map(|b| (b.0, b.1, u8::from(b.2))).collect();
        let mesh = mesh_from_points_edges(&p.positions, &edges, &p.site_kinds);
        let cap = format!("{} of {:?}: {} sites (seam bonds red)", kind_label(p.kind), self.net, p.positions.len());
        Some((scene, Some(mesh), cap))
    }

    pub(super) fn save(&self) -> Value {
        json!({"net": format!("{:?}", self.net), "kind": format!("{:?}", self.kind), "repeats": self.repeats})
    }

    pub(super) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        if let Some(n) = parse_net(v.get("net"), notes) {
            self.net = n;
        }
        if let Some(s) = v.get("kind").and_then(Value::as_str) {
            match FlatSlabKind::from_name(s) {
                Ok(k) => self.kind = k,
                Err(e) => notes.push(format!("S4 slab: {e} (kept current)")),
            }
        }
        if let Some(r) = super::read_triple(v.get("repeats"), MAX_REPEAT) {
            self.repeats = r;
        }
    }
}
