//! Equivalence verdict and measurements for the selected candidate.

use crate::theme::ThemeTokens;

use super::FuseTabState;

impl FuseTabState {
    pub fn show_inspector_panel(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("Equivalence");
        let bundle = match &self.last_bundle {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.small("Comparing the pair with single-defect cores…");
                });
                return;
            }
            Some(Err(err)) => {
                let headline = if self.last_combined.refused.is_some() {
                    "Could not build this pair"
                } else {
                    "Comparison unavailable"
                };
                ui.colored_label(tokens.status_warn, headline);
                ui.label(err);
                return;
            }
            Some(Ok(bundle)) => bundle,
        };
        let rep = &bundle.report;
        let (headline, color) = match (rep.kind.as_str(), rep.equivalent, rep.match_tier.as_str()) {
            ("vacuum", _, _) if bundle.delta_v.is_some_and(|dv| dv != 0) => {
                ("Vacuum charge · site count differs", tokens.status_warn)
            }
            ("vacuum", true, _) if bundle.delta_v == Some(0) => ("Vacuum · regular core", tokens.status_ok),
            ("vacuum", true, _) => ("Vacuum charge · regular core", tokens.status_warn),
            ("vacuum", false, _) => ("Vacuum charge · irregular core", tokens.status_warn),
            (_, true, "volterra") => ("Volterra core match", tokens.status_ok),
            (_, true, "completion") => ("Healed core match", tokens.accent_primary),
            (_, true, _) => ("Equivalent to a single defect", tokens.status_ok),
            (_, false, _) if rep.single.is_none() => ("No single elementary defect with this charge", tokens.status_warn),
            (_, false, _) => ("No core match in this scan", tokens.status_warn),
        };
        egui::Frame::NONE
            .fill(tokens.bg_card)
            .stroke(egui::Stroke::new(1.5, color))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.colored_label(color, egui::RichText::new(headline).strong());
                if let Some(dv) = bundle.delta_v {
                    ui.small(format!("Site excess ΔV = {dv:+}"));
                }
            });

        ui.add_space(6.0);
        ui.strong("Measured charge");
        ui.label(&rep.charge.label);
        ui.separator();
        let candidate = self.active_single_candidate();
        let circuit_results = [&self.combined_inspectors, &self.single_inspectors]
            .map(|inspector| inspector.circuit.overlay.as_ref().map(|overlay| &overlay.result));
        egui::Grid::new("fuse_side_by_side_grid")
            .num_columns(3)
            .spacing([8.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Measure");
                ui.strong("Pair");
                ui.strong("Candidate");
                ui.end_row();

                ui.label("Core graph");
                ui.monospace(format!("{}V, {}E", rep.core_nodes, rep.core_edges));
                ui.monospace(candidate.map_or_else(|| "—".into(), |c| format!("{}V, {}E", c.core_nodes, c.core_edges)));
                ui.end_row();

                ui.label("Core rings");
                ui.monospace(rep.config_core.as_ref().map_or_else(|| "regular".into(), |c| format!("{:?}", c.rings)));
                ui.monospace(candidate.map_or_else(|| "—".into(), |c| {
                    if c.kind == "vacuum" { "regular".into() } else { format!("{:?}", c.rings) }
                }));
                ui.end_row();

                ui.label("Circuit |b|");
                for result in circuit_results {
                    ui.monospace(result
                        .filter(|r| r.frank_angle_deg.is_some_and(|angle| angle.abs() < 1e-6))
                        .map_or_else(|| "—".into(), |r| format!("{:.3}", r.burgers_norm)));
                }
                ui.end_row();

                ui.label("Circuit Ω");
                for result in circuit_results {
                    ui.monospace(result.and_then(|r| r.frank_angle_deg)
                        .map_or_else(|| "—".into(), |angle| format!("{angle:+.1}°")));
                }
                ui.end_row();
            });
        if let Some(c) = candidate {
            ui.add_space(4.0);
            ui.small(&c.description);
        }
        if rep.equivalent && rep.kind != "vacuum" && !bundle.candidates.iter().any(|c| c.is_match) {
            ui.small("The matching core has no full-patch drawing.");
        }
        egui::CollapsingHeader::new("Calculation details").show(ui, |ui| {
            ui.small(&rep.verdict);
            if rep.kind == "vacuum" && bundle.delta_v.is_none() {
                ui.small("Site excess unavailable.");
            }
            ui.small(&self.last_combined.charge_detail);
            if let Some(ab) = &bundle.abelian_summary {
                ui.small(ab);
            }
            ui.small(&self.last_combined.core_summary);
        });
    }
}
