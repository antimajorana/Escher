//! Instant `[PROVED]` algebraic fusion rules, universal abelianizations `(G+)^ab` / `G^ab`,
//! and improper reflection/glide holonomy sector renderer for `Defects > Combinations`.

use papertilling_core::defect::algebraic_fusion::{
    AlgebraicFusionOutcome, AlgebraicFusionRow, ImproperSectorInfo, SpaceGroupAlgebraReport,
};

use super::combos::CombosState;
use crate::backend::CatalogueItemUi;
use crate::gallery_tab::filter::group_with_orbifold;
use crate::theme::ThemeTokens;

const PANEL_TITLE: &str = "[PROVED] Algebraic fusion rules (all separations) & improper sectors";

impl CombosState {
    /// Render the instant `[PROVED]` space-group algebra, improper sector classification,
    /// and all-separation lifted `G~+` fusion table.
    pub(crate) fn draw_algebra_panel(
        &mut self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        tokens: &ThemeTokens,
    ) {
        self.ensure_algebra(item);

        egui::CollapsingHeader::new(egui::RichText::new(PANEL_TITLE).strong())
            .id_salt("combos_algebra_panel_hdr")
            .default_open(true)
            .show(ui, |ui| {
                let Some(res) = &self.algebra else {
                    return;
                };
                match res {
                    Err(err) => {
                        ui.colored_label(
                            tokens.status_err,
                            format!("Algebraic fusion analysis unavailable: {err}"),
                        );
                    }
                    Ok(report) => {
                        draw_summary_card(ui, report, tokens);
                        if !report.improper_sectors.is_empty() {
                            ui.add_space(4.0);
                            draw_improper_sectors(ui, &report.improper_sectors, tokens);
                        }
                        ui.add_space(4.0);
                        draw_fusion_channels(ui, &report.fusion_rows, &mut self.algebra_filter, tokens);
                    }
                }
            });
    }
}

fn draw_summary_card(
    ui: &mut egui::Ui,
    report: &SpaceGroupAlgebraReport,
    tokens: &ThemeTokens,
) {
    let symm_kind = if report.symmorphic { "symmorphic" } else { "nonsymmorphic" };
    egui::Frame::NONE
        .fill(tokens.bg_card)
        .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!(
                    "{} ({symm_kind}, |P| = {}, |P+| = {})",
                    group_with_orbifold(&report.group),
                    report.point_group_order,
                    report.proper_order,
                ));
                ui.separator();
                ui.strong(format!("(G+)^ab = {}", report.ab_proper.label));
                ui.separator();
                ui.strong(format!("G^ab = {}", report.ab_full.label));
            });
            ui.colored_label(
                tokens.text_secondary,
                egui::RichText::new(&report.provenance).small(),
            );
        });
}

fn draw_improper_sectors(
    ui: &mut egui::Ui,
    sectors: &[ImproperSectorInfo],
    tokens: &ThemeTokens,
) {
    let title = format!(
        "Reflection & glide holonomy sectors (Q_R // Z_P(R)) — {}",
        sectors.len()
    );
    egui::CollapsingHeader::new(egui::RichText::new(title).strong())
        .id_salt("combos_improper_sectors_hdr")
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("combos_improper_sectors_grid")
                .striped(true)
                .spacing([12.0, 4.0])
                .show(ui, |ui| {
                    ui.strong("Sector");
                    ui.strong("Quotient Q_R");
                    ui.strong("Mirror / glide status");
                    ui.strong("Pair / square proper channel");
                    ui.end_row();

                    for sec in sectors {
                        let axis = format!("[{}, {}]", sec.parallel_axis[0], sec.parallel_axis[1]);
                        ui.label(format!("{} ({axis})", sec.label))
                            .on_hover_text(&sec.centralizer_summary);
                        ui.label(&sec.quotient_label);
                        draw_mirror_status(ui, sec, tokens);
                        ui.label(&sec.square_proper_channel);
                        ui.end_row();
                    }
                });
        });
}

fn draw_mirror_status(ui: &mut egui::Ui, sec: &ImproperSectorInfo, tokens: &ThemeTokens) {
    let (text, color) = match (sec.has_pure_mirror, sec.fixed_point_free_centralizer) {
        (true, false) => ("Pure mirror (has Z_P(R) fixed points)", tokens.status_ok),
        (true, true) => ("Pure mirror (fixed-point-free Z_P(R))", tokens.accent_primary),
        (false, true) => ("Fixed-point-free glide (no pure mirror)", tokens.status_warn),
        (false, false) => ("Pure glide (no pure mirror)", tokens.status_warn),
    };
    ui.colored_label(color, text);
}

fn draw_fusion_channels(
    ui: &mut egui::Ui,
    rows: &[AlgebraicFusionRow],
    filter: &mut String,
    tokens: &ThemeTokens,
) {
    let title = format!("All-separation fusion channels (lifted G~+) — {} pairs", rows.len());
    egui::CollapsingHeader::new(egui::RichText::new(title).strong())
        .id_salt("combos_fusion_channels_hdr")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small("Filter defects:");
                ui.add(
                    egui::TextEdit::singleline(filter)
                        .id_salt("combos_algebra_filter_input")
                        .desired_width(180.0)
                        .hint_text("e.g. 1a, T1, +90"),
                );
                if !filter.is_empty() && ui.small_button("Clear").clicked() {
                    filter.clear();
                }
            });

            let needle = filter.trim().to_lowercase();
            let filtered: Vec<&AlgebraicFusionRow> = if needle.is_empty() {
                rows.iter().collect()
            } else {
                rows.iter().filter(|r| row_matches(r, &needle)).collect()
            };

            if !needle.is_empty() {
                ui.small(format!("Showing {} of {} pairs", filtered.len(), rows.len()));
            }
            if filtered.is_empty() {
                ui.small("No algebraic fusion pairs match the current filter.");
                return;
            }

            ui.add_space(2.0);
            egui::ScrollArea::vertical()
                .id_salt("combos_fusion_channels_scroll")
                .max_height(260.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    egui::Grid::new("combos_fusion_channels_grid")
                        .striped(true)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.strong("Defect A");
                            ui.strong("Defect B");
                            ui.strong("Abelian sum in (G+)^ab");
                            ui.strong("Allowed output channels [PROVED]");
                            ui.strong("Frank angle");
                            ui.strong("Single defect");
                            ui.end_row();

                            for row in filtered {
                                for out in &row.outcomes {
                                    ui.label(&row.defect_a);
                                    ui.label(&row.defect_b);
                                    ui.small(&row.abelian_sum_proper);
                                    ui.label(format_outcome_summary(out));
                                    ui.label(format_frank_angle(out.frank_angle_deg));
                                    if out.allowed_single_defect {
                                        ui.colored_label(tokens.status_ok, "Yes");
                                    } else {
                                        ui.colored_label(tokens.status_warn, "No");
                                    }
                                    ui.end_row();
                                }
                            }
                        });
                });
        });
}

fn row_matches(row: &AlgebraicFusionRow, needle_lower: &str) -> bool {
    row.defect_a.to_lowercase().contains(needle_lower)
        || row.defect_b.to_lowercase().contains(needle_lower)
        || row.abelian_sum_proper.to_lowercase().contains(needle_lower)
        || row
            .outcomes
            .iter()
            .any(|o| o.summary.to_lowercase().contains(needle_lower))
}

fn format_frank_angle(angle_deg: f64) -> String {
    if (angle_deg - angle_deg.round()).abs() < 1e-6 {
        format!("{:+.0}°", angle_deg.round())
    } else {
        format!("{:+.1}°", angle_deg)
    }
}

fn format_outcome_summary(out: &AlgebraicFusionOutcome) -> String {
    if out.shortest_burgers_frac.is_empty() {
        out.summary.clone()
    } else {
        let min_b: Vec<String> = out
            .shortest_burgers_frac
            .iter()
            .take(4)
            .map(|b| format!("({}, {})", b[0], b[1]))
            .collect();
        let suffix = if out.shortest_burgers_frac.len() > 4 { ", …" } else { "" };
        format!("{} · shortest b: {}{suffix}", out.summary, min_b.join(", "))
    }
}
