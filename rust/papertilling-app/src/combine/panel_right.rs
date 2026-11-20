use egui_extras::{Column, TableBuilder};
use papertilling_core::defect::surgery::DefectSpec;

use super::CombineTabState;
use super::history::{CutDragResponse, show_cut_angle_drag_value};
#[cfg(test)]
use crate::backend::AppBackend;
use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::live_combine::{
    cuts::{
        default_cut_angle_deg, has_coincident_same_kind_pins, merge_coincident_pins,
        pin_explicit_cut_vec, set_pin_cut_angle,
    },
    spec_short_label, spec_world_pos,
};
use crate::theme::ThemeTokens;

impl CombineTabState {
    #[cfg(test)]
    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
        let tokens = ThemeTokens::paper();
        if self.show_left_panel(ui, gallery_item, backend, canvas, &tokens, true) {
            ui.separator();
            self.show_inspector_panel(ui, gallery_item, canvas, &tokens, true);
        }
    }

    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
        #[allow(unused_variables)] advanced: bool,
    ) {
        if self.is_initializing() || self.init_error.is_some() {
            return;
        }
        let mut build_preview: Option<bool> = None;
        ui.strong(format!("Placed Defects ({})", self.specs.len()));
        if self.specs.is_empty() {
            ui.small("Empty — use a preset or Click-to-Place.");
        } else {
            let outcome = self.placed_table_ui(ui);
            if let Some(idx) = outcome.select_idx
                && idx < self.specs.len()
            {
                if self.commit_cut_edit_anchor() {
                    build_preview = Some(false);
                }
                self.selected_pin = Some(idx);
                self.popover_open = true;
                self.sync_selection_to_canvas(canvas);
            }
            if let Some(i) = outcome.flip_idx {
                let _ = self.flip_pin(i, gallery_item, Some(canvas));
            }
            if let Some(i) = outcome.remove_idx {
                let _ = self.delete_pin(i, gallery_item, Some(canvas));
            }
            if let Some(prev) = outcome.build_preview {
                self.sync_selection_to_canvas(canvas);
                build_preview = Some(build_preview.unwrap_or(true) && prev);
            }
            if has_coincident_same_kind_pins(&self.info, &self.specs)
                && ui
                    .small_button("Merge coincident pins")
                    .on_hover_text("Collapse superposed same-point pins permanently")
                    .clicked()
            {
                self.commit_cut_edit_anchor();
                self.push_undo();
                merge_coincident_pins(&self.info, &mut self.specs);
                self.clear_selection();
                self.sync_selection_to_canvas(canvas);
                build_preview = Some(false);
            }
        }
        if let Some(preview_only) = build_preview {
            self.request_build(gallery_item, preview_only, !preview_only && self.auto_compare, false);
        }

        ui.separator();
        self.equivalence_ui(ui, tokens);
        self.show_inspectors_ui(ui, gallery_item, canvas, tokens);
    }

    /// The placed-defects table; returns requested actions without double-building.
    fn placed_table_ui(&mut self, ui: &mut egui::Ui) -> TableOutcome {
        let mut out = TableOutcome::default();
        let mut cut_edit: Option<(usize, Option<f64>)> = None;
        let mut cut_drag: Option<(usize, CutDragResponse)> = None;
        ui.push_id("placed_specs_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(108.0)
                .column(Column::exact(28.0))
                .column(Column::exact(62.0))
                .column(Column::remainder())
                .column(Column::exact(58.0))
                .column(Column::exact(24.0))
                .column(Column::exact(22.0))
                .header(17.0, |mut h| {
                    for t in ["#", "Kind", "(x, y)", "Cut", "±", ""] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, sp) in self.specs.iter().enumerate() {
                        let pos = spec_world_pos(&self.info, idx, sp);
                        let kstr = spec_short_label(&self.info, sp);
                        let cut_opt = pin_explicit_cut_vec(sp);
                        let is_selected = self.selected_pin == Some(idx);
                        body.row(18.0, |mut row| {
                            row.set_selected(is_selected);
                            row.col(|ui| {
                                if ui.selectable_label(is_selected, format!("#{}", idx + 1)).clicked() {
                                    out.select_idx = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(kstr);
                            });
                            row.col(|ui| {
                                ui.small(format!("({:+.2}, {:+.2})", pos[0], pos[1]));
                            });
                            row.col(|ui| {
                                if let Some(c) = cut_opt {
                                    let deg = c[1].atan2(c[0]).to_degrees();
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 2.0;
                                        let ev = show_cut_angle_drag_value(ui, deg, 2.0);
                                        if ev.started || ev.changed || ev.finished {
                                            cut_drag = Some((idx, ev));
                                        }
                                        if ui.small_button("a").on_hover_text("Reset direction to auto").clicked() {
                                            cut_edit = Some((idx, None));
                                        }
                                    });
                                } else {
                                    let lbl = if matches!(sp, DefectSpec::HalfWall { .. }) { "ray" } else { "auto" };
                                    if ui.small_button(lbl).on_hover_text("Lock to explicit angle").clicked() {
                                        let deg = default_cut_angle_deg(
                                            &self.info,
                                            &self.specs,
                                            idx,
                                            &self.last_build.applied_cuts,
                                        );
                                        cut_edit = Some((idx, Some(deg)));
                                    }
                                }
                            });
                            row.col(|ui| {
                                if ui.small_button("±").clicked() {
                                    out.flip_idx = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                if ui.small_button("×").clicked() {
                                    out.remove_idx = Some(idx);
                                }
                            });
                        });
                    }
                });
        });
        if let Some((i, ev)) = cut_drag
            && let Some(prev) = self.apply_cut_drag_response(i, ev)
        {
            out.build_preview = Some(prev);
        }
        if let Some((i, ang_opt)) = cut_edit {
            self.commit_cut_edit_anchor();
            self.push_undo();
            set_pin_cut_angle(&mut self.specs[i], ang_opt);
            out.build_preview = Some(false);
        }
        out
    }

    fn equivalence_ui(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("Total Charge & Equivalence");
        ui.label(&self.last_build.charge_label);
        ui.small(&self.last_build.charge_detail);
        if self.last_build.preview {
            ui.small("Core: (read out when the drag ends)");
        } else {
            ui.small(format!(
                "Core: {} nodes, {} edges | {}",
                self.last_build.core_nodes, self.last_build.core_edges, self.last_build.core_summary
            ));
        }
        if !self.last_build.local_cores.is_empty() {
            let lc: Vec<&str> = self.last_build.local_cores.iter().map(|c| c.label.as_str()).collect();
            ui.small(format!("Cores: {}", lc.join(", ")));
        }
        if self.is_comparing() || self.dragging || self.cut_dragging {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small(if self.dragging || self.cut_dragging {
                    "Comparison runs when the drag ends"
                } else {
                    "Comparing cores…"
                });
            });
        }
        let stale = self.compare_stale || self.is_comparing() || self.dragging || self.cut_dragging;
        match &self.compare_result {
            Some(Ok(rep)) => {
                let (bg, fg, tag) = match (rep.equivalent, rep.match_tier.as_str()) {
                    (true, "volterra" | "exact") => (
                        tokens.bg_card_selected,
                        tokens.status_ok,
                        "EQUIVALENT (VOLTERRA CORE)",
                    ),
                    (true, _) => (
                        tokens.accent_soft,
                        tokens.accent_primary,
                        "EQUIVALENT (COMPLETION / HEALED CORE)",
                    ),
                    (false, _) => (
                        tokens.bg_card,
                        tokens.status_warn,
                        "DISTINCT FROM SINGLE",
                    ),
                };
                let (bg, fg, stroke_col) = if stale {
                    (tokens.bg_panel, tokens.text_muted, tokens.border_subtle)
                } else {
                    (bg, fg, fg)
                };
                egui::Frame::NONE
                    .fill(bg)
                    .stroke(egui::Stroke::new(1.0, stroke_col))
                    .inner_margin(4.0)
                    .corner_radius(3.0)
                    .show(ui, |ui| {
                        let tag = if stale {
                            format!("{tag} (previous configuration)")
                        } else {
                            tag.to_string()
                        };
                        ui.colored_label(fg, egui::RichText::new(tag).strong().size(11.0));
                        ui.small(&rep.verdict);
                    });
                if !stale {
                    ui.small(format!(
                        "compared in {:.1} ms (scan cache {} hit(s), {} miss(es))",
                        self.compare_ms, self.cache_stats.0, self.cache_stats.1
                    ));
                }
            }
            Some(Err(err)) => {
                ui.small(format!("Single-defect check: {err}"));
            }
            None => {}
        }
    }
}

#[derive(Default)]
struct TableOutcome {
    select_idx: Option<usize>,
    flip_idx: Option<usize>,
    remove_idx: Option<usize>,
    build_preview: Option<bool>,
}
