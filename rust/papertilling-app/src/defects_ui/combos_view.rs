//! Left controls, central table, core thumbnails, and right inspector for Defects > Combinations.

use papertilling_core::defect::combinations_types::{
    ChargeGroup, CombinationEntry, CombinationsTable, CoreGroup, SingleMatchStatus,
};
use papertilling_core::defect::surgery::DefectSpec;

use super::combos::{ComboModeKind, CombosState};
use crate::backend::{CatalogueItemUi, DefectCatalogueUi};
use crate::core_panel::{CoreInspectorModal, compile_core_card, paint_card_primitives};
use crate::gallery_tab::filter::group_with_orbifold;
use crate::theme::ThemeTokens;

const THUMB_SIDE: f32 = 128.0;

impl CombosState {
    pub(crate) fn show_left_controls(
        &mut self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        cat: &DefectCatalogueUi,
        tokens: &ThemeTokens,
    ) {
        ui.strong("Defect Combinations");
        ui.add_space(2.0);

        let modes = ComboModeKind::available_for(cat);
        let mut changed = false;
        ui.label("Family:");
        for m in modes {
            if ui.selectable_label(self.mode_kind == m, m.label()).clicked() && self.mode_kind != m {
                self.mode_kind = m;
                changed = true;
            }
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Max sep (a):");
            let slider = egui::Slider::new(&mut self.max_sep_scale, 0.5..=2.5).step_by(0.1);
            if ui.add(slider).changed() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Core radius:");
            let slider = egui::Slider::new(&mut self.core_radius, 1..=3);
            if ui.add(slider).changed() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Search grid:");
            for g in [1usize, 2] {
                if ui.selectable_label(self.scan_grid == g, format!("{g}×{g}")).clicked()
                    && self.scan_grid != g
                {
                    self.scan_grid = g;
                    changed = true;
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Recompute").clicked() {
                self.submit(item);
                changed = false;
            }
            if self.is_busy() && ui.button("Cancel").clicked() {
                self.cancel_current(item);
                changed = false;
            }
        });
        if changed {
            self.submit(item);
        }

        if let Some(err) = &self.error {
            ui.add_space(4.0);
            ui.colored_label(tokens.status_err, err);
        }

        ui.add_space(4.0);
        ui.small(
            "Enumerates symmetry-inequivalent defect placements within the chosen separation bound, grouped by total topological charge and distinct core graph.",
        );
    }

    /// Central combinations table; returns `Some(specs)` when the user clicks `"Open in Place & fuse"`.
    pub(crate) fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        tokens: &ThemeTokens,
    ) -> Option<Vec<DefectSpec>> {
        let mut open_specs: Option<Vec<DefectSpec>> = None;

        if self.is_busy() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.strong("Computing defect combinations...");
                if ui.button("Cancel").clicked() {
                    self.cancel_current(item);
                }
            });
            ui.add_space(4.0);
        }

        if let Some(err) = &self.error {
            ui.colored_label(tokens.status_err, format!("Combinations error: {err}"));
            return None;
        }

        let Some(table) = self.table.clone() else {
            if !self.is_busy() {
                if self.cancelled_by_user {
                    ui.label("Combination search cancelled. Click Recompute to run again.");
                } else {
                    ui.label("Preparing defect combinations...");
                }
            }
            ui.add_space(4.0);
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("combos_algebra_pending_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.draw_algebra_panel(ui, item, tokens);
                });
            return None;
        };

        self.draw_bounds_banner(ui, item, &table, tokens);
        ui.separator();

        egui::ScrollArea::vertical()
            .id_salt("combos_table_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.draw_algebra_panel(ui, item, tokens);
                ui.add_space(4.0);
                ui.separator();

                if table.charge_groups.is_empty() && table.refused.is_empty() {
                    ui.label(format!(
                        "No combinations found for {} within max separation {:.2} a.",
                        self.mode_kind.label(),
                        self.max_sep_scale
                    ));
                    if self.mode_kind == ComboModeKind::DisclinationPairs {
                        ui.small("Lattices without rotation centres have no disclinations; try Dislocation pairs.");
                    }
                    return;
                }
                let single_indices: Vec<usize> = table
                    .charge_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| g.has_single_defect)
                    .map(|(i, _)| i)
                    .collect();
                let multi_indices: Vec<usize> = table
                    .charge_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| !g.has_single_defect)
                    .map(|(i, _)| i)
                    .collect();

                ui.strong("Charges carried by a single defect (and vacuum)");
                ui.add_space(3.0);
                if single_indices.is_empty() {
                    ui.small("None within the current separation bound.");
                } else {
                    for cg_i in single_indices {
                        if let Some(specs) = self.draw_charge_group(ui, cg_i, &table.charge_groups[cg_i], tokens) {
                            open_specs = Some(specs);
                        }
                    }
                }

                ui.add_space(8.0);
                ui.separator();
                ui.strong("Charges no single defect carries");
                ui.add_space(3.0);
                if multi_indices.is_empty() {
                    ui.small("None within the current separation bound.");
                } else {
                    for cg_i in multi_indices {
                        if let Some(specs) = self.draw_charge_group(ui, cg_i, &table.charge_groups[cg_i], tokens) {
                            open_specs = Some(specs);
                        }
                    }
                }

                if !table.refused.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    let hdr = format!("Refused combinations ({})", table.refused.len());
                    egui::CollapsingHeader::new(hdr)
                        .id_salt("combos_refused_hdr")
                        .default_open(false)
                        .show(ui, |ui| {
                            for r in &table.refused {
                                ui.horizontal_wrapped(|ui| {
                                    ui.strong(&r.label);
                                    ui.small(format!("(d = {:.3})", r.separation));
                                    if let Some(alg) = &r.algebraic_charge {
                                        ui.small(format!("Algebraic total charge: {alg}"));
                                    }
                                    ui.colored_label(tokens.status_warn, &r.reason);
                                });
                            }
                        });
                }
            });

        if let Some(mut modal) = self.inspector_modal.take()
            && modal.show(ui.ctx(), tokens)
        {
            self.inspector_modal = Some(modal);
        }

        open_specs
    }

    fn draw_bounds_banner(
        &self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        table: &CombinationsTable,
        tokens: &ThemeTokens,
    ) {
        let total_cores: usize = table.charge_groups.iter().map(|g| g.core_groups.len()).sum();
        let max_sep_a = table.max_sep / self.a_scale.max(1e-9);
        egui::Frame::NONE
            .fill(tokens.bg_card)
            .stroke(egui::Stroke::new(1.0, tokens.border_subtle))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.strong(format!(
                    "{} · {}  |  {}  |  max sep = {:.2} a ({:.3})  |  |b| <= {:.1} b_min  |  core radius = {}  |  grid = {}×{}",
                    item.display_with_orig(),
                    group_with_orbifold(&table.group),
                    self.mode_kind.label(),
                    max_sep_a,
                    table.max_sep,
                    table.bmax,
                    table.radius,
                    table.grid,
                    table.grid,
                ));
                ui.small(format!(
                    "Candidates: {} · Built: {} · Refused: {} · Charge groups: {} · Distinct cores: {} · {:.0} ms",
                    table.total_candidates,
                    table.built_count,
                    table.refused_count,
                    table.charge_groups.len(),
                    total_cores,
                    table.elapsed_ms,
                ));
                ui.colored_label(
                    tokens.text_secondary,
                    egui::RichText::new(format!(
                        "[COMPUTATIONAL EVIDENCE] Bounds: sep <= {:.2} a, |b| <= {:.1} b_min, core radius {}, {}×{} search grid. {} Core grouping reflects the built core of each candidate's working build order.",
                        max_sep_a, table.bmax, table.radius, table.grid, table.grid, table.symmetry_note
                    ))
                    .small(),
                );
                if table.cancelled {
                    ui.colored_label(
                        tokens.status_warn,
                        "Computation stopped early after cancellation; showing partial results.",
                    );
                }
            });
    }

    fn draw_charge_group(
        &mut self,
        ui: &mut egui::Ui,
        cg_i: usize,
        cg: &ChargeGroup,
        tokens: &ThemeTokens,
    ) -> Option<Vec<DefectSpec>> {
        let mut open_specs = None;
        let pair_count: usize = cg.core_groups.iter().map(|c| c.members.len()).sum();
        let title = format!(
            "{} — {} distinct core{}, {} placement{}",
            cg.charge_label,
            cg.core_groups.len(),
            plural(cg.core_groups.len()),
            pair_count,
            plural(pair_count),
        );
        egui::CollapsingHeader::new(egui::RichText::new(title).strong())
            .id_salt(("combos_cg", cg_i, &cg.charge_label))
            .default_open(true)
            .show(ui, |ui| {
                for (core_i, core) in cg.core_groups.iter().enumerate() {
                    if let Some(specs) = self.draw_core_group(ui, cg_i, core_i, &cg.charge_label, core, tokens) {
                        open_specs = Some(specs);
                    }
                    ui.add_space(4.0);
                }
            });
        open_specs
    }

    fn draw_core_group(
        &mut self,
        ui: &mut egui::Ui,
        cg_i: usize,
        core_i: usize,
        charge_label: &str,
        core: &CoreGroup,
        tokens: &ThemeTokens,
    ) -> Option<Vec<DefectSpec>> {
        let mut open_specs = None;
        let card_border = if core.status == SingleMatchStatus::MatchesSingle {
            egui::Stroke::new(1.5, tokens.status_ok)
        } else {
            egui::Stroke::new(1.0, tokens.border_subtle)
        };
        egui::Frame::NONE
            .fill(tokens.bg_card)
            .stroke(card_border)
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    self.draw_thumbnail(ui, cg_i, core_i, charge_label, core, tokens);
                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        ui.horizontal_wrapped(|ui| {
                            draw_status_badge(ui, core, tokens);
                            ui.small(format!("{} sites, {} bonds", core.core_nodes, core.core_edges));
                            if let Some(dv) = core.delta_v {
                                ui.separator();
                                ui.strong(format_delta_v(dv));
                            }
                        });
                        ui.add_space(3.0);
                        for (mem_i, m) in core.members.iter().enumerate() {
                            let is_sel = self.selected == Some((cg_i, core_i, mem_i));
                            if let Some(specs) = self.draw_member_row(ui, cg_i, core_i, mem_i, m, is_sel) {
                                open_specs = Some(specs);
                            }
                        }
                    });
                });
            });
        open_specs
    }

    fn draw_thumbnail(
        &mut self,
        ui: &mut egui::Ui,
        cg_i: usize,
        core_i: usize,
        charge_label: &str,
        core: &CoreGroup,
        tokens: &ThemeTokens,
    ) {
        let size = egui::vec2(THUMB_SIDE, THUMB_SIDE);
        if let Some(entry) = &core.thumbnail {
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
            if ui.is_rect_visible(rect) {
                let key = format!("{cg_i}:{core_i}");
                let prims = self
                    .thumb_cache
                    .entry(key)
                    .or_insert_with(|| compile_core_card(entry));
                paint_card_primitives(&ui.painter_at(rect), rect, prims, tokens);
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
            }
            if resp.on_hover_text("Click to enlarge core graph").clicked() {
                let is_match = core.status == SingleMatchStatus::MatchesSingle;
                self.inspector_modal = Some(CoreInspectorModal::from_entry(entry, charge_label, is_match));
            }
        } else {
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, tokens.bg_canvas_2d);
            painter.rect_stroke(
                rect,
                4.0,
                egui::Stroke::new(1.0, tokens.border_subtle),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Perfect lattice\n(0 core nodes)",
                egui::FontId::proportional(11.0),
                tokens.text_secondary,
            );
        }
    }

    fn draw_member_row(
        &mut self,
        ui: &mut egui::Ui,
        cg_i: usize,
        core_i: usize,
        mem_i: usize,
        member: &CombinationEntry,
        is_selected: bool,
    ) -> Option<Vec<DefectSpec>> {
        let mut open_specs = None;
        ui.push_id(("combo_mem", cg_i, core_i, mem_i), |ui| {
            ui.horizontal_wrapped(|ui| {
                let ids = member.defect_ids.join(" + ");
                let disp = format_displacements(&member.displacements);
                let row_text = format!("{ids}  ·  d = {:.3}  ·  Δ = {disp}", member.separation);
                if ui.selectable_label(is_selected, row_text).clicked() {
                    self.selected = Some((cg_i, core_i, mem_i));
                }
                if ui.small_button("Open in Place & fuse").clicked() {
                    self.selected = Some((cg_i, core_i, mem_i));
                    open_specs = Some(member.specs.clone());
                }
            });
        });
        open_specs
    }

    /// Right inspector panel for the currently selected combination.
    pub(crate) fn show_inspector(
        &mut self,
        ui: &mut egui::Ui,
        _item: &CatalogueItemUi,
        tokens: &ThemeTokens,
    ) -> Option<Vec<DefectSpec>> {
        ui.strong("Combination Details");
        ui.add_space(3.0);

        let mut open_specs = None;
        if let Some(table) = &self.table
            && let Some((cg_i, core_i, mem_i)) = self.selected
            && let Some(cg) = table.charge_groups.get(cg_i)
            && let Some(core) = cg.core_groups.get(core_i)
            && let Some(member) = core.members.get(mem_i)
        {
            ui.label(format!("Defects: {}", member.defect_ids.join(" + ")));
            ui.label(format!(
                "Separation: {:.4} ({:.2} a)",
                member.separation,
                member.separation / self.a_scale.max(1e-9)
            ));
            ui.small(format!("Displacements: {}", format_displacements(&member.displacements)));
            ui.small(format!("Build order: {:?}", member.build_order));
            ui.separator();
            ui.strong("Total Charge & Core");
            ui.label(&cg.charge_label);
            draw_status_badge(ui, core, tokens);
            ui.small(format!(
                "Core graph: {} sites, {} bonds (tier: {})",
                core.core_nodes, core.core_edges, core.match_tier
            ));
            if !core.volterra_matches.is_empty() {
                ui.small(format!("Volterra matches: {}", core.volterra_matches.join(", ")));
            }
            if !core.completion_matches.is_empty() {
                ui.small(format!("Completion matches: {}", core.completion_matches.join(", ")));
            }
            if let Some(dv) = member.delta_v {
                ui.strong(format!("Vacuum site excess: {}", format_delta_v(dv)));
            }
            ui.add_space(6.0);
            if ui.button("Open in Place & fuse").clicked() {
                open_specs = Some(member.specs.clone());
            }
        } else if self.is_busy() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small("Computing combinations...");
            });
        } else {
            ui.small("Click any combination row in the table to inspect its charge, core match, and build order.");
        }

        ui.add_space(8.0);
        ui.separator();
        ui.small(
            "[COMPUTATIONAL EVIDENCE] Placements are reduced modulo proper symmetry G+ (with co-rotated dislocation offsets) within the shown bounds; core grouping reflects the built core of each candidate's working build order.",
        );
        open_specs
    }
}

fn draw_status_badge(ui: &mut egui::Ui, core: &CoreGroup, tokens: &ThemeTokens) {
    match core.status {
        SingleMatchStatus::MatchesSingle => {
            let names = if core.single_matches.is_empty() {
                String::from("single defect")
            } else {
                core.single_matches.join(", ")
            };
            ui.colored_label(
                tokens.status_ok,
                egui::RichText::new(format!("Matches single: {names}")).strong(),
            );
        }
        SingleMatchStatus::CoreDiffers => {
            ui.colored_label(tokens.status_warn, egui::RichText::new("Core differs").strong());
        }
        SingleMatchStatus::NoSingleDefect => {
            ui.colored_label(tokens.text_secondary, egui::RichText::new("No single defect").strong());
        }
    }
}

fn format_delta_v(dv: i32) -> String {
    match dv {
        0 => String::from("ΔV = 0 sites"),
        1 => String::from("ΔV = +1 site"),
        -1 => String::from("ΔV = -1 site"),
        n if n > 0 => format!("ΔV = +{n} sites"),
        n => format!("ΔV = {n} sites"),
    }
}

fn format_displacements(disps: &[[f64; 2]]) -> String {
    disps
        .iter()
        .map(|d| {
            let x = if d[0].abs() < 1e-9 { 0.0 } else { d[0] };
            let y = if d[1].abs() < 1e-9 { 0.0 } else { d[1] };
            format!("({x:.2}, {y:.2})")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
