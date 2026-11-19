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
}

impl Default for SlabState {
    fn default() -> Self {
        Self { net: CanonicalNet3d::SimpleCubic, kind: FlatSlabKind::TorusSlab, repeats: [3, 3, 3], out: None }
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
        ch |= repeats_row(ui, "Slab repeats (a1, a2, a3):", &mut self.repeats, MAX_REPEAT);
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
        ui.label(format!("Repeats {:?}; status {}", p.repeats, p.status));
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
