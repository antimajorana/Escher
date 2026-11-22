//! Right inspector panel for `Fuse & compare`: visual fusion equation (`A + B -> Total Charge`),
//! color-coded equivalence verdict banner, side-by-side core & Burgers census table, and handoff to `Place & fuse`.

use crate::backend::CatalogueItemUi;
use crate::theme::ThemeTokens;

use super::{FuseTabState, choice_label};

impl FuseTabState {
    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        _item: &CatalogueItemUi,
        tokens: &ThemeTokens,
    ) {
        ui.strong("Fusion Reaction & Equivalence");
        ui.add_space(3.0);

        // 1. Visual Equation Card: A + B -> Total Charge
        egui::Frame::NONE
            .fill(tokens.bg_card)
            .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                if let Some(ca) = self.choices.get(self.choice_a) {
                    ui.label(format!("A: {}", choice_label(&self.info, ca)));
                }
                if let Some(cb) = self.choices.get(self.choice_b) {
                    ui.label(format!("+ B: {}", choice_label(&self.info, cb)));
                }
                let dx = self.pos_b[0] - self.pos_a[0];
                let dy = self.pos_b[1] - self.pos_a[1];
                ui.small(format!("Offset Δ = ({:+.3}, {:+.3}),  |Δ| = {:.3}", dx, dy, dx.hypot(dy)));
                ui.separator();
                ui.strong("Total Topological Charge");
                if self.last_combined.charge_label.is_empty() {
                    ui.small("Computing…");
                } else {
                    ui.label(&self.last_combined.charge_label);
                    if !self.last_combined.charge_detail.is_empty() {
                        ui.small(&self.last_combined.charge_detail);
                    }
                }
            });

        ui.add_space(6.0);

        // 2. Equivalence Verdict Card
        let mut cand_picked: Option<usize> = None;
        match &self.last_bundle {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.small("Evaluating single-defect equivalence…");
                });
            }
            Some(Err(err)) => {
                egui::Frame::NONE
                    .fill(tokens.bg_card)
                    .stroke(egui::Stroke::new(1.5, tokens.status_warn))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.colored_label(
                            tokens.status_warn,
                            egui::RichText::new("SURGERY REFUSED AT THIS OFFSET").strong(),
                        );
                        ui.small(err);
                        ui.small("Click 'Farther ▶' to separate overlapping defect cores.");
                    });
            }
            Some(Ok(bundle)) => {
                let rep = &bundle.report;
                let (bg, border, fg, headline) = match (rep.equivalent, rep.match_tier.as_str()) {
                    (true, "volterra") => (
                        tokens.bg_card_selected,
                        tokens.status_ok,
                        tokens.status_ok,
                        "EQUIVALENT TO SINGLE DEFECT (VOLTERRA CORE)",
                    ),
                    (true, "completion") => (
                        tokens.accent_soft,
                        tokens.accent_primary,
                        tokens.accent_primary,
                        "EQUIVALENT TO SINGLE DEFECT (HEALED / COMPLETION CORE)",
                    ),
                    (true, _) if rep.kind == "vacuum" => (
                        tokens.bg_card_selected,
                        tokens.status_ok,
                        tokens.status_ok,
                        "ANNIHILATED TO PERFECT LATTICE (VACUUM)",
                    ),
                    (true, _) => (
                        tokens.bg_card_selected,
                        tokens.status_ok,
                        tokens.status_ok,
                        "EQUIVALENT TO SINGLE DEFECT",
                    ),
                    (false, _) if rep.single.is_none() && rep.kind != "vacuum" => (
                        tokens.bg_card,
                        tokens.status_warn,
                        tokens.status_warn,
                        "NO SINGLE ELEMENTARY DEFECT CARRIES THIS CHARGE",
                    ),
                    (false, _) => (
                        tokens.bg_card,
                        tokens.status_warn,
                        tokens.status_warn,
                        "SAME TOPOLOGICAL CHARGE — ATOMIC CORE DIFFERS",
                    ),
                };

                egui::Frame::NONE
                    .fill(bg)
                    .stroke(egui::Stroke::new(1.5, border))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.colored_label(fg, egui::RichText::new(headline).strong());
                        ui.small(&rep.verdict);
                        if let Some(dv) = bundle.delta_v {
                            ui.small(format!("Vacuum site excess ΔV = {dv:+} site(s)"));
                        }
                        if let Some(ab) = &bundle.abelian_summary {
                            ui.small(ab);
                        }
                    });

                if !bundle.candidates.is_empty() {
                    ui.add_space(6.0);
                    ui.separator();
                    ui.strong(format!("Single-Defect Core Candidates ({})", bundle.candidates.len()));
                    ui.horizontal_wrapped(|ui| {
                        for (idx, c) in bundle.candidates.iter().enumerate() {
                            let star = if c.is_match { " ★" } else { "" };
                            if ui
                                .selectable_label(self.selected_candidate == idx, format!("{}{star}", c.name))
                                .on_hover_text(&c.description)
                                .clicked()
                                && self.selected_candidate != idx
                            {
                                cand_picked = Some(idx);
                            }
                        }
                    });
                    if let Some(sc) = bundle.candidates.get(self.selected_candidate) {
                        ui.small(format!("{}: {}", sc.name, sc.description));
                    }
                }

                // 3. Side-by-side Core & Burgers Comparison Table
                ui.add_space(6.0);
                ui.separator();
                ui.strong("Side-by-Side Invariants");
                let single_cand = self.active_single_candidate();
                let comb_circ = self.combined_inspectors.circuit.overlay.as_ref().map(|o| &o.result);
                let sing_circ = self.single_inspectors.circuit.overlay.as_ref().map(|o| &o.result);

                egui::Grid::new("fuse_side_by_side_grid")
                    .num_columns(3)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Property");
                        ui.strong("Combined (A+B)");
                        ui.strong("Single C");
                        ui.end_row();

                        ui.label("Core graph");
                        ui.monospace(format!("{}V, {}E", rep.core_nodes, rep.core_edges));
                        if let Some(sc) = single_cand {
                            ui.monospace(format!("{}V, {}E", sc.core_nodes, sc.core_edges));
                        } else {
                            ui.monospace("—");
                        }
                        ui.end_row();

                        ui.label("Core rings");
                        let comb_rings = rep
                            .config_core
                            .as_ref()
                            .map_or_else(|| String::from("regular"), |c| format!("{:?}", c.rings));
                        ui.monospace(comb_rings);
                        if let Some(sc) = single_cand {
                            let sr = if sc.rings.is_empty() {
                                sc.kind.clone()
                            } else {
                                format!("{:?}", sc.rings)
                            };
                            ui.monospace(sr);
                        } else {
                            ui.monospace("—");
                        }
                        ui.end_row();

                        ui.label("Burgers |b|");
                        ui.monospace(comb_circ.map_or_else(|| String::from("—"), |c| format!("{:.3}", c.burgers_norm)));
                        ui.monospace(sing_circ.map_or_else(|| String::from("—"), |c| format!("{:.3}", c.burgers_norm)));
                        ui.end_row();

                        ui.label("Frank Ω");
                        ui.monospace(comb_circ.map_or_else(|| String::from("—"), |c| format!("{:+.1}°", c.frank_angle_deg.unwrap_or(0.0))));
                        ui.monospace(sing_circ.map_or_else(|| String::from("—"), |c| format!("{:+.1}°", c.frank_angle_deg.unwrap_or(0.0))));
                        ui.end_row();
                    });

                if !self.last_combined.core_summary.is_empty() {
                    ui.add_space(4.0);
                    ui.small(format!("Combined patch: {}", self.last_combined.core_summary));
                }
            }
        }

        if let Some(new_idx) = cand_picked {
            self.selected_candidate = new_idx;
            self.sync_single_canvas_only(_item, false);
        }

        ui.add_space(8.0);
        ui.separator();
        if ui
            .button("Open pair in Place & fuse")
            .on_hover_text("Transfer Defect A and Defect B into Place & fuse to add a 3rd defect or wall")
            .clicked()
        {
            self.open_in_combine = Some(self.current_specs().to_vec());
        }
    }
}
