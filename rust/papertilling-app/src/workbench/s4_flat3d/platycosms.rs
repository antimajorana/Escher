//! Closed flat 3D platycosms: the 10-row Conway-Rossetti table (a static lookup, shown synchronously) and the finite
//! quotient 3-net of a canonical net on the chosen platycosm (built on the worker).
//!
//! The core answers `OK`, `IMPOSSIBLE` (a `[PROVED]` holohedry obstruction: the crystal's lattice holohedry lacks the
//! needed rotation) or `NOT_FOUND` (`[COMPUTATIONAL EVIDENCE]`: the bounded search found no free action). The two
//! refusals are shown verbatim and never drawn; NOT_FOUND is never presented as impossible.

use papertilling_core::flat3d::{Cell3D, PlatycosmQuotient3d, build_platycosm_quotient, list_platycosms, lookup_platycosm};
use papertilling_core::space3d::nets3d::CanonicalNet3d;
use serde_json::{Value, json};

use super::{Scenes, net_combo, parse_net, points_scene, repeats_row};
use crate::mesh3d::wb_mesh::mesh_from_points_edges;

pub(super) const SCENE_TITLE: &str = "S4 platycosm quotient (xy of the fundamental domain)";
pub(super) const MAX_SUPERCELL: usize = 6;

/// Build the quotient; a non-`OK` core status is returned as `Err(reason)`.
pub(super) fn build(net: CanonicalNet3d, code: &str, supercell: Option<[usize; 3]>, ignore_kinds: bool) -> Result<PlatycosmQuotient3d, String> {
    let cell = Cell3D::from_canonical_net(net);
    let q = build_platycosm_quotient(&cell, code, supercell, ignore_kinds)?;
    if q.status != "OK" {
        return Err(q.reason);
    }
    if q.vertices.is_empty() {
        return Err(String::from("the quotient has no vertices (refused, nothing drawn)"));
    }
    Ok(q)
}

pub(super) struct PlatyState {
    pub(super) net: CanonicalNet3d,
    pub(super) code: String,
    pub(super) auto_supercell: bool,
    pub(super) supercell: [usize; 3],
    pub(super) ignore_kinds: bool,
    pub(super) out: Option<PlatycosmQuotient3d>,
}

impl Default for PlatyState {
    fn default() -> Self {
        Self {
            net: CanonicalNet3d::SimpleCubic,
            code: String::from("c2"),
            auto_supercell: true,
            supercell: [1, 1, 2],
            ignore_kinds: false,
            out: None,
        }
    }
}

impl PlatyState {
    pub(super) fn supercell_opt(&self) -> Option<[usize; 3]> {
        (!self.auto_supercell).then_some(self.supercell)
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        lookup_platycosm(&self.code).map_err(|e| format!("platycosm: {e}"))?;
        if !self.auto_supercell && self.supercell.iter().any(|&r| r == 0 || r > MAX_SUPERCELL) {
            return Err(format!("supercell repeats must lie in 1..={MAX_SUPERCELL}"));
        }
        Ok(())
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = net_combo(ui, "wb_s4_platy_net", &mut self.net);
        let shown = lookup_platycosm(&self.code).map_or_else(|_| self.code.clone(), |p| format!("{} {}", p.code, p.name));
        egui::ComboBox::from_id_salt("wb_s4_platy_code").selected_text(shown).show_ui(ui, |ui| {
            for p in list_platycosms() {
                let sel = self.code == p.code;
                if ui.selectable_label(sel, format!("{} {}", p.code, p.name)).clicked() && !sel {
                    self.code = p.code.to_string();
                    ch = true;
                }
            }
        });
        ch |= ui.checkbox(&mut self.auto_supercell, "Search the supercell automatically").changed();
        if !self.auto_supercell {
            ch |= repeats_row(ui, "Supercell (a1, a2, a3):", &mut self.supercell, MAX_SUPERCELL);
        }
        ch |= ui.checkbox(&mut self.ignore_kinds, "Ignore site colours (uncoloured net)").changed();
        ch
    }

    pub(super) fn show_inspector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.label("[PROVED] the ten closed flat 3-manifolds (Conway-Rossetti 2003; H1 re-derived as G^ab in the core tests):");
        egui::ScrollArea::horizontal().id_salt("wb_s4_platy_scroll").show(ui, |ui| {
            egui::Grid::new("wb_s4_platy_grid").striped(true).show(ui, |ui| {
                for h in ["code", "name", "orient.", "ITA", "H1", "b1"] {
                    ui.strong(h);
                }
                ui.end_row();
                for p in list_platycosms() {
                    let sel = self.code == p.code;
                    if ui.selectable_label(sel, format!("[{}]", p.code)).clicked() && !sel {
                        self.code = p.code.to_string();
                        ch = true;
                    }
                    ui.label(p.name);
                    ui.label(if p.orientable { "yes" } else { "no" });
                    let ita: Vec<String> = p.ita_numbers.iter().zip(p.ita_symbols).map(|(n, s)| format!("#{n} {s}")).collect();
                    ui.label(ita.join(", "));
                    ui.label(p.h1_label);
                    ui.label(p.b1.to_string());
                    ui.end_row();
                }
            });
        });
        ui.separator();
        match &self.out {
            None => {
                ui.label("No quotient built for these settings.");
            }
            Some(q) => {
                ui.strong(format!("{} ({}) on {:?}", q.platycosm_code, q.platycosm_name, self.net));
                ui.label(format!(
                    "[COMPUTATIONAL EVIDENCE] free action found: ITA #{} {} on supercell {:?}",
                    q.ita_number.map_or_else(|| String::from("?"), |n| n.to_string()),
                    q.ita_symbol.as_deref().unwrap_or("?"),
                    q.supercell
                ));
                ui.label(format!(
                    "[PROVED] χ₃ = V − E + F − C = {} − {} + {} − {} = {} (exact CW count of the quotient complex)",
                    q.cw_v, q.cw_e, q.cw_f, q.cw_c, q.euler_chi_3
                ));
                ui.label(format!("[PROVED] H1 = {}, b1 = {} (published table)", q.h1_label, q.b1));
                let loops = q.bonds.iter().filter(|b| b.u == b.v).count();
                let seams = q.bonds.iter().filter(|b| b.across_seam).count();
                ui.label(format!(
                    "Quotient net: {} vertices, {} bonds ({} across a seam, red; {} loops not drawn)",
                    q.vertices.len(),
                    q.bonds.len(),
                    seams,
                    loops
                ));
                ui.small(&q.reason);
            }
        }
        ch
    }

    pub(super) fn scenes(&self) -> Option<Scenes> {
        let q = self.out.as_ref()?;
        let pts3: Vec<[f64; 3]> = q.vertices.iter().map(|v| v.pos).collect();
        let xy: Vec<[f64; 2]> = pts3.iter().map(|p| [p[0], p[1]]).collect();
        let inner: Vec<(usize, usize)> = q.bonds.iter().filter(|b| !b.across_seam).map(|b| (b.u, b.v)).collect();
        let scene = points_scene(&xy, &inner, &[], SCENE_TITLE, "bonds across a seam are drawn in 3D only");
        let edges: Vec<(usize, usize, u8)> = q.bonds.iter().map(|b| (b.u, b.v, u8::from(b.across_seam))).collect();
        let kinds: Vec<u8> = q.vertices.iter().map(|v| v.site_kind).collect();
        let mesh = mesh_from_points_edges(&pts3, &edges, &kinds);
        let cap = format!("{} quotient of {:?}: {} vertices (seam bonds red)", q.platycosm_code, self.net, q.vertices.len());
        Some((scene, Some(mesh), cap))
    }

    pub(super) fn save(&self) -> Value {
        json!({
            "net": format!("{:?}", self.net),
            "code": self.code,
            "auto_supercell": self.auto_supercell,
            "supercell": self.supercell,
            "ignore_kinds": self.ignore_kinds,
        })
    }

    pub(super) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        if let Some(n) = parse_net(v.get("net"), notes) {
            self.net = n;
        }
        if let Some(c) = v.get("code").and_then(Value::as_str) {
            match lookup_platycosm(c) {
                Ok(p) => self.code = p.code.to_string(),
                Err(_) => notes.push(format!("S4 platycosm: unknown code {c:?} (kept current)")),
            }
        }
        if let Some(b) = v.get("auto_supercell").and_then(Value::as_bool) {
            self.auto_supercell = b;
        }
        if let Some(r) = super::read_triple(v.get("supercell"), MAX_SUPERCELL) {
            self.supercell = r;
        }
        if let Some(b) = v.get("ignore_kinds").and_then(Value::as_bool) {
            self.ignore_kinds = b;
        }
    }
}
