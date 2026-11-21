//! CSL symmetric-tilt grain boundaries: the exact coincidence-site list (Σ, θ) of hexagonal and square cells, and a
//! finite bicrystal built at a listed CSL angle or at a free angle (not an exact CSL boundary).

use papertilling_core::defect::{BicrystalOptions, BicrystalResult, CslGrainBoundaryEntry, build_symmetric_tilt_bicrystal, csl_family, list_csl_grain_boundaries};
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;

pub(super) const SCENE_TITLE: &str = "S2 tilt bicrystal";

pub(crate) struct GbOut {
    pub(crate) result: BicrystalResult,
}

pub(crate) fn build(group: &str, cell: &Cell2D, theta: f64, sigma: Option<u32>, radius: f64) -> Result<GbOut, String> {
    if !(theta.is_finite() && theta > 0.0 && theta < 180.0) {
        return Err(format!("tilt angle {theta} deg refused: it must lie strictly between 0 and 180 deg"));
    }
    let opts = BicrystalOptions { radius, ..BicrystalOptions::default() };
    Ok(GbOut { result: build_symmetric_tilt_bicrystal(group, cell, theta, sigma, &opts)? })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GbChoice {
    Csl(usize),
    Free,
}

pub(crate) struct GbState {
    pub(crate) max_sigma: u32,
    pub(crate) list: Vec<CslGrainBoundaryEntry>,
    list_key: String,
    pub(crate) family: Option<&'static str>,
    pub(crate) choice: GbChoice,
    /// The user chose the free angle (kept across lattices); otherwise the first listed CSL boundary is the default.
    want_free: bool,
    pub(crate) free_theta: f64,
    pub(crate) out: Option<GbOut>,
}

impl Default for GbState {
    fn default() -> Self {
        Self {
            max_sigma: 19,
            list: Vec::new(),
            list_key: String::new(),
            family: None,
            choice: GbChoice::Free,
            want_free: false,
            free_theta: 15.0,
            out: None,
        }
    }
}

impl GbState {
    pub(super) fn clear_for_new_lattice(&mut self) {
        self.list_key.clear();
        self.out = None;
    }

    /// Recompute the CSL list (a tiny exact enumeration) when the lattice or `max_sigma` changed.
    pub(super) fn ensure_list(&mut self, group: &str, cell: &Cell2D, lattice_key: &str) {
        let key = format!("{lattice_key}|{}", self.max_sigma);
        if key == self.list_key {
            return;
        }
        let keep_id = match self.choice {
            GbChoice::Csl(i) => self.list.get(i).map(|e| e.id.clone()),
            GbChoice::Free => None,
        };
        self.list_key = key;
        self.family = csl_family(group, cell);
        self.list = list_csl_grain_boundaries(group, cell, self.max_sigma);
        let kept = keep_id.and_then(|id| self.list.iter().position(|e| e.id == id));
        self.choice = if self.want_free || self.list.is_empty() { GbChoice::Free } else { GbChoice::Csl(kept.unwrap_or(0)) };
    }

    /// `(theta_deg, Some(sigma))` for a listed CSL boundary, `(free_theta, None)` otherwise.
    pub(crate) fn params(&self) -> (f64, Option<u32>) {
        match self.choice {
            GbChoice::Csl(i) => self.list.get(i).map_or((self.free_theta, None), |e| (e.theta_deg, Some(e.sigma))),
            GbChoice::Free => (self.free_theta, None),
        }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = false;
        match self.family {
            Some(f) => ui.label(format!("CSL family: {f} cell")),
            None => ui.label("No CSL family for this group and cell metric: free tilt angle only."),
        };
        ui.horizontal(|ui| {
            ui.label("Max Σ:");
            rebuild |= ui.add(egui::DragValue::new(&mut self.max_sigma).range(3..=49)).changed();
        });
        egui::ScrollArea::vertical().id_salt("wb_s2_csl_list").max_height(220.0).show(ui, |ui| {
            for (i, e) in self.list.iter().enumerate() {
                let text = format!("{}  θ = {:.2}°  (m, n) = ({}, {})", e.id, e.theta_deg, e.m, e.n);
                if ui.selectable_label(self.choice == GbChoice::Csl(i), text).clicked() && self.choice != GbChoice::Csl(i) {
                    self.choice = GbChoice::Csl(i);
                    self.want_free = false;
                    rebuild = true;
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.selectable_label(self.choice == GbChoice::Free, "Free tilt angle").clicked() && self.choice != GbChoice::Free {
                self.choice = GbChoice::Free;
                self.want_free = true;
                rebuild = true;
            }
            let ch = ui.add(egui::DragValue::new(&mut self.free_theta).range(0.5..=179.5).speed(0.1).suffix("°")).changed();
            if ch && self.choice == GbChoice::Free {
                rebuild = true;
            }
        });
        ui.small("Twin walls (mirror boundaries) of this lattice are listed under the all-defects table.");
        rebuild
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        match self.family {
            Some("hexagonal") => {
                ui.small("Σ = (m² + mn + n²) / gcd(m² + mn + n², 3), cos θ = (m² + 4mn + n²) / (2 (m² + mn + n²)) [PROVED]");
            }
            Some(_) => {
                ui.small("Σ = (m² + n²) / gcd(m² + n², 2), θ = 2 atan(n / m) [PROVED]");
            }
            None => {}
        }
        let Some(o) = &self.out else {
            ui.label("Building the bicrystal ...");
            return;
        };
        let r = &o.result;
        let sigma = r.sigma.map_or_else(|| String::from("none"), |s| s.to_string());
        ui.label(format!("θ = {:.4}°, Σ = {sigma}, exact CSL: {}", r.theta_deg, r.exact_csl));
        ui.label(format!("Badge: {}", r.provenance));
        ui.label(format!("Sites {}, bonds {}", r.crystal.pos.len(), r.crystal.edges.len()));
        ui.small("Grain A (x <= 0) is rotated by -θ/2, grain B (x >= 0) by +θ/2.");
    }

    pub(super) fn scene(&self) -> Scene2D {
        match &self.out {
            Some(o) => {
                let s = o.result.sigma.map_or_else(|| String::from("free angle"), |s| format!("Σ{s}"));
                Scene2D::CrystalPatch {
                    crystal: o.result.crystal.clone(),
                    title: String::from(SCENE_TITLE),
                    note: format!("{s}, θ = {:.2}°", o.result.theta_deg),
                }
            }
            None => empty_scene(SCENE_TITLE, "building ..."),
        }
    }

    pub(super) fn save(&self) -> Value {
        let theta = if self.free_theta.is_finite() { self.free_theta } else { 15.0 };
        let csl = match self.choice {
            GbChoice::Csl(i) => self.list.get(i).map(|e| e.id.clone()),
            GbChoice::Free => None,
        };
        json!({ "max_sigma": self.max_sigma, "free_theta": theta, "csl": csl })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(m) = v.get("max_sigma").and_then(Value::as_u64) {
            self.max_sigma = u32::try_from(m.clamp(3, 49)).unwrap_or(19);
        }
        if let Some(t) = v.get("free_theta").and_then(Value::as_f64).filter(|t| t.is_finite()) {
            self.free_theta = t.clamp(0.5, 179.5);
        }
        match v.get("csl") {
            Some(Value::Null) => {
                self.choice = GbChoice::Free;
                self.want_free = true;
            }
            Some(Value::String(id)) => {
                // re-resolved against the list of the current lattice by `ensure_list`
                self.list = vec![CslGrainBoundaryEntry {
                    id: id.clone(),
                    sigma: 0,
                    m: 0,
                    n: 0,
                    theta_deg: 0.0,
                    family: "",
                    provenance: "",
                }];
                self.choice = GbChoice::Csl(0);
                self.want_free = false;
                self.list_key.clear();
            }
            _ => {}
        }
    }
}
