//! Shared interactive Burgers circuit and 2D overlay selector for `Defects` and `Combine`.

use papertilling_core::defect::circuit::{
    CircuitResult, auto_burgers_circuit, burgers_circuit, shift_circuit_basepoint,
};
use papertilling_core::types::{BuiltCrystal, Cell2D, Frame2D};

use crate::canvas2d::Canvas2D;
use crate::canvas2d::circuit::{CanvasCircuitOverlay, CircuitTargetMode, CircuitUiState};
use crate::canvas2d::gauge_overlay::{
    CrystalOverlayData, CrystalOverlayMode, SeamConnectionMark,
};
use crate::canvas2d::overlay_legend::seam_legend_chips;
use crate::theme::ThemeTokens;

/// Interactive inspection state shared by `DefectsTabState` and `CombineTabState`.
pub struct DefectInspectors {
    pub circuit: CircuitUiState,
    pub overlay_mode: CrystalOverlayMode,
}

impl Default for DefectInspectors {
    fn default() -> Self {
        Self::new()
    }
}

impl DefectInspectors {
    pub fn new() -> Self {
        Self {
            circuit: CircuitUiState::default(),
            overlay_mode: CrystalOverlayMode::Standard,
        }
    }

    pub fn clear_for_switch(&mut self) {
        self.circuit.overlay = None;
        self.circuit.error = None;
        self.circuit.base_v0 = None;
        self.circuit.step_k = 0;
    }

    /// Recompute the active Burgers circuit overlay cleanly against `crystal` and `cell`.
    pub fn refresh_circuit(
        &mut self,
        group: &str,
        cell: &Cell2D,
        frame: Frame2D,
        crystal: &BuiltCrystal,
        pin_centers: &[[f64; 2]],
    ) {
        if !self.circuit.enabled || crystal.pos.is_empty() {
            self.circuit.overlay = None;
            self.circuit.error = None;
            return;
        }
        if self.circuit.pin_mask.len() != pin_centers.len() {
            self.circuit.pin_mask = vec![true; pin_centers.len()];
        }
        let (target, exclude): (Vec<[f64; 2]>, Vec<[f64; 2]>) = match self.circuit.target_mode {
            CircuitTargetMode::SelectedPin if !pin_centers.is_empty() => {
                let idx = self.circuit.pin_idx.min(pin_centers.len() - 1);
                let ex = pin_centers.iter().enumerate().filter(|&(i, _)| i != idx).map(|(_, &p)| p).collect();
                (vec![pin_centers[idx]], ex)
            }
            CircuitTargetMode::PinSubset if !pin_centers.is_empty() => {
                let mut tgt = Vec::new();
                let mut ex = Vec::new();
                for (i, &p) in pin_centers.iter().enumerate() {
                    if self.circuit.pin_mask.get(i).copied().unwrap_or(false) { tgt.push(p); } else { ex.push(p); }
                }
                (tgt, ex)
            }
            _ => (Vec::new(), Vec::new()),
        };
        let mut res: Result<CircuitResult, String> = auto_burgers_circuit(
            crystal,
            Some((group, cell, frame)),
            &target,
            &exclude,
            2,
            &[],
        );
        if res.is_err() && !exclude.is_empty() {
            res = auto_burgers_circuit(crystal, Some((group, cell, frame)), &target, &[], 2, &[]);
        }
        if let Ok(mut cr) = res {
            if let Some(v0) = self.circuit.base_v0
                && cr.path.contains(&v0)
                && let Ok(rot) = shift_circuit_basepoint(&cr.path, v0)
                && let Ok(shifted) = burgers_circuit(crystal, &rot, Some((group, cell, frame)))
            {
                cr = shifted;
            } else {
                self.circuit.base_v0 = cr.path.first().copied();
            }
            res = Ok(cr);
        }
        match res {
            Ok(cr) => {
                let world_pts = cr.path.iter().filter_map(|&v| crystal.pos.get(v).copied()).collect::<Vec<_>>();
                let mut across_steps = Vec::with_capacity(cr.path.len().saturating_sub(1));
                for w in cr.path.windows(2) {
                    let across = crystal.adj.get(w[0]).is_some_and(|r| r.iter().any(|e| e.to == w[1] && e.across));
                    across_steps.push(across);
                }
                let k_max = cr.ref_path.len().saturating_sub(1);
                self.circuit.step_k = self.circuit.step_k.min(k_max);
                self.circuit.overlay = Some(CanvasCircuitOverlay {
                    path: cr.path.clone(),
                    world_pts,
                    across_steps,
                    result: cr,
                    step_k: self.circuit.step_k,
                });
                self.circuit.error = None;
            }
            Err(err) => {
                self.circuit.overlay = None;
                self.circuit.error = Some(err);
            }
        }
    }

    pub fn build_crystal_overlay(
        &self,
        crystal: &BuiltCrystal,
        ao_override: Option<CrystalOverlayData>,
    ) -> Option<CrystalOverlayData> {
        match self.overlay_mode {
            CrystalOverlayMode::Standard => None,
            CrystalOverlayMode::SeamConnections => {
                let mut seams = Vec::new();
                for &(u, v) in &crystal.edges {
                    if let Some(e) = crystal.adj.get(u).and_then(|r| r.iter().find(|b| b.to == v))
                        && (e.seam.is_some() || e.across)
                    {
                        let (det_neg, lbl) = if let Some(g) = e.seam {
                            let ang = g.m[1][0].atan2(g.m[0][0]).to_degrees().round() as i32;
                            if g.det() < 0.0 {
                                (true, String::from("det=-1"))
                            } else if ang != 0 {
                                (false, format!("{ang:+}°"))
                            } else {
                                (false, format!("t=({:.1},{:.1})", g.t[0], g.t[1]))
                            }
                        } else {
                            (false, String::from("wall"))
                        };
                        seams.push(SeamConnectionMark { a: crystal.pos[u], b: crystal.pos[v], label: lbl, det_neg, has_ge: e.seam.is_some() });
                    }
                }
                let chips = seam_legend_chips(&seams);
                Some(CrystalOverlayData { seams, chips, ..Default::default() })
            }
            CrystalOverlayMode::GsvEnergyShear
            | CrystalOverlayMode::BondStretchStrain
            | CrystalOverlayMode::VertexAngleShear
            | CrystalOverlayMode::PlaquetteFlux => ao_override,
        }
    }

    pub fn sync_to_canvas(&self, canvas: &mut Canvas2D, crystal: &BuiltCrystal, ao_override: Option<CrystalOverlayData>) {
        canvas.set_circuit_overlay(if self.circuit.enabled { self.circuit.overlay.clone() } else { None });
        canvas.set_gauge_overlay(self.build_crystal_overlay(crystal, ao_override));
    }

    /// Render the circuit controls and 2D overlay dropdown.
    /// Returns `(circuit_changed, overlay_changed)`.
    pub fn show_controls_ui(
        &mut self,
        ui: &mut egui::Ui,
        id_prefix: &str,
        num_pins: usize,
        _tokens: &ThemeTokens,
    ) -> (bool, bool) {
        let mut circ_dirty = false;
        let mut ov_dirty = false;

        ui.separator();
        ui.strong("Burgers Circuit & Parallel Transport");
        ui.horizontal_wrapped(|ui| {
            if ui.checkbox(&mut self.circuit.enabled, "Burgers Circuit").changed() {
                circ_dirty = true;
            }
            if self.circuit.enabled && num_pins > 1 {
                if ui.selectable_label(self.circuit.target_mode == CircuitTargetMode::AllDefects, "All").clicked()
                    && self.circuit.target_mode != CircuitTargetMode::AllDefects
                {
                    self.circuit.target_mode = CircuitTargetMode::AllDefects;
                    self.circuit.base_v0 = None;
                    circ_dirty = true;
                }
                if ui.selectable_label(self.circuit.target_mode == CircuitTargetMode::SelectedPin, "Pin").clicked()
                    && self.circuit.target_mode != CircuitTargetMode::SelectedPin
                {
                    self.circuit.target_mode = CircuitTargetMode::SelectedPin;
                    self.circuit.base_v0 = None;
                    circ_dirty = true;
                }
                if ui.selectable_label(self.circuit.target_mode == CircuitTargetMode::PinSubset, "Subset").clicked()
                    && self.circuit.target_mode != CircuitTargetMode::PinSubset
                {
                    self.circuit.target_mode = CircuitTargetMode::PinSubset;
                    if self.circuit.pin_mask.len() != num_pins {
                        self.circuit.pin_mask = vec![true; num_pins];
                    }
                    self.circuit.base_v0 = None;
                    circ_dirty = true;
                }
                if self.circuit.target_mode == CircuitTargetMode::SelectedPin {
                    for k in 0..num_pins {
                        if ui.selectable_label(self.circuit.pin_idx == k, format!("#{}", k + 1)).clicked() {
                            self.circuit.pin_idx = k;
                            self.circuit.base_v0 = None;
                            circ_dirty = true;
                        }
                    }
                } else if self.circuit.target_mode == CircuitTargetMode::PinSubset {
                    if self.circuit.pin_mask.len() != num_pins {
                        self.circuit.pin_mask = vec![true; num_pins];
                    }
                    for k in 0..num_pins {
                        let on = self.circuit.pin_mask[k];
                        if ui.selectable_label(on, format!("#{}", k + 1)).clicked() {
                            self.circuit.pin_mask[k] = !on;
                            self.circuit.base_v0 = None;
                            circ_dirty = true;
                        }
                    }
                }
            }
        });
        if self.circuit.enabled {
            let mut next_k = None;
            if let Some(ov) = &self.circuit.overlay {
                let k_max = ov.result.ref_path.len().saturating_sub(1);
                ui.horizontal(|ui| {
                    ui.small("Step:");
                    if ui.add(egui::Slider::new(&mut self.circuit.step_k, 0..=k_max)).changed() {
                        next_k = Some(self.circuit.step_k.min(k_max));
                        ov_dirty = true;
                    }
                    if let Some(v0) = self.circuit.base_v0 {
                        ui.small(format!("v0={v0}"));
                    }
                });
                let cum_a = ov.result.cum_angle_deg.get(self.circuit.step_k).copied().unwrap_or(0.0);
                let ch_str = ov
                    .result
                    .charge
                    .as_ref()
                    .map(|c| format!(" | Class: {} (b=({:+.2},{:+.2}))", c.label, ov.result.closure_vector[0], ov.result.closure_vector[1]))
                    .unwrap_or_default();
                ui.small(format!(
                    "Circuit ({} edges): orientable={}, cum={cum_a:+.1}°, |b|={:.3}{ch_str} (click loop site to shift v0)",
                    k_max, ov.result.orientation_preserving, ov.result.burgers_norm
                ));
            } else if let Some(err) = &self.circuit.error {
                ui.small(format!("Circuit unavailable: {err}"));
            }
            if let (Some(k), Some(ov)) = (next_k, &mut self.circuit.overlay) {
                ov.step_k = k;
            }
        }

        ui.horizontal(|ui| {
            ui.small("2D Overlay:");
            let prev = self.overlay_mode;
            egui::ComboBox::from_id_salt(format!("{id_prefix}_2d_ov"))
                .selected_text(self.overlay_mode.label())
                .show_ui(ui, |ui| {
                    for m in CrystalOverlayMode::NON_SPECTRAL {
                        ui.selectable_value(&mut self.overlay_mode, m, m.label());
                    }
                });
            if self.overlay_mode != prev {
                ov_dirty = true;
            }
        });
        (circ_dirty, ov_dirty)
    }
}
