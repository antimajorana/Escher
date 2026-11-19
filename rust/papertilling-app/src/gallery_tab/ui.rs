//! Left-rail browsing and rename bar of the Gallery tab.

use egui_extras::{Column, TableBuilder};

use super::{BrowseMode, GalleryTabState};
use crate::backend::{AppBackend, CatalogueRuleUi};
use crate::canvas2d::Canvas2D;

impl GalleryTabState {
    /// Inline rename controls (Enter to save, Esc to cancel, Reset name when a custom alias is active), added
    /// straight into the caller's row (the HUD, after the lattice segment that shows the name and its `±`). Their
    /// width is bounded: the catalogue id is a hover on the field and long notes truncate, so the HUD can keep the
    /// whole bar on its row.
    pub fn show_rename_bar(&mut self, ui: &mut egui::Ui, backend: &mut AppBackend, canvas: &mut Canvas2D, id_salt: &str) {
        self.rename_has_focus = false;
        if self.renaming {
            let resp = ui
                .add(
                    egui::TextEdit::singleline(&mut self.rename_buf)
                        .id_salt(id_salt)
                        .desired_width(165.0)
                        .hint_text(&self.active_item.default_label),
                )
                .on_hover_text(format!("Catalogue id: {}", self.active_item.default_label));
            if self.rename_focus_pending {
                resp.request_focus();
                self.rename_focus_pending = false;
            }
            self.rename_has_focus = resp.has_focus();
            let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            if esc {
                self.cancel_rename();
            } else if (enter && (resp.lost_focus() || resp.has_focus())) || ui.small_button("Save").clicked() {
                self.commit_rename(backend, canvas);
            } else if ui.small_button("Cancel").clicked() {
                self.cancel_rename();
            }
        } else {
            if self.readout.chiral {
                let tip = if self.readout.topologically_chiral {
                    "± Chiral enantiomorph pair in 2D (chiral bare graph, merged under Euclidean normalizer)"
                } else {
                    "± Chiral enantiomorph pair in 2D (merged under Euclidean normalizer)"
                };
                ui.small("±").on_hover_text(tip);
            }
            if self.active_item.has_custom_name() {
                small_truncated(ui, format!("[{}]", self.active_item.default_label), 120.0);
            }
            if ui.small_button("✎ Rename").on_hover_text("Give this lattice a persistent custom name").clicked() {
                self.begin_rename();
            }
            if self.active_item.has_custom_name()
                && ui.small_button("Reset name").on_hover_text("Restore default catalogue name").clicked()
            {
                self.reset_active_lattice_name(backend, canvas);
            }
        }
        if let Some(note) = &self.alias_note {
            small_truncated(ui, note.clone(), 160.0);
        }
    }

    /// Draw the full side panel (browsing + inspector) for legacy callers/tests.
    #[cfg(test)]
    pub fn show_side_panel(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let lattice_changed = self.show_left_panel(ui, backend, canvas);
        ui.separator();
        self.show_inspector_panel(ui, backend, canvas, &crate::theme::ThemeTokens::paper());
        lattice_changed
    }

    /// Switch the browse mode as its left-rail tab does (also used by the Command Deck); true = the lattice changed.
    pub(crate) fn set_browse_mode(&mut self, m: BrowseMode, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        match m {
            BrowseMode::ByGroup => {
                self.select_group(self.selected_group_idx, backend, canvas);
                true
            }
            BrowseMode::Named21 => {
                self.select_named(self.selected_named_idx, backend, canvas);
                true
            }
            BrowseMode::CustomBuilder => {
                self.mode = BrowseMode::CustomBuilder;
                false
            }
        }
    }

    /// Left rail: browse mode tabs, group/catalogue/named tables, live search, and custom builder.
    pub fn show_left_panel(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let mut lattice_changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.strong("Browse:");
            for (m, label) in [
                (BrowseMode::ByGroup, "17 Groups"),
                (BrowseMode::Named21, "Named (21)"),
                (BrowseMode::CustomBuilder, "+ Custom"),
            ] {
                if ui.selectable_label(self.mode == m, label).clicked() {
                    lattice_changed |= self.set_browse_mode(m, backend, canvas);
                }
            }
        });
        ui.add_space(4.0);

        match self.mode {
            BrowseMode::ByGroup => {
                self.filter_bar(ui);
                ui.add_space(3.0);
                lattice_changed |= self.group_tables(ui, backend, canvas);
                ui.separator();
                lattice_changed |= self.lattice_list(ui, backend, canvas);
                ui.separator();
                let grp = backend.groups()[self.selected_group_idx].symbol;
                lattice_changed |= self.live_search_section(ui, backend, canvas, grp);
            }
            BrowseMode::Named21 => {
                self.filter_bar(ui);
                ui.add_space(3.0);
                lattice_changed |= self.lattice_list(ui, backend, canvas);
            }
            BrowseMode::CustomBuilder => {
                lattice_changed |= self.custom_builder_panel(ui, backend, canvas);
            }
        }
        lattice_changed
    }

    fn group_tables(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Rule:");
            for r in CatalogueRuleUi::ALL {
                let desc = match r {
                    CatalogueRuleUi::Bonds => "Standard lattice: minimal site and bond orbits; bonds may lower point symmetry",
                    CatalogueRuleUi::Sheet => "Atomic crystals: bare site positions alone already have exact group G",
                    CatalogueRuleUi::Points => "Delaunay lattice: bare site positions have exact group G; bonds are Voronoi-neighbour dual edges",
                };
                if ui
                    .selectable_label(self.rule == r, r.short_label())
                    .on_hover_text(format!("{} — {desc}", r.label()))
                    .clicked()
                {
                    self.select_rule(r, backend, canvas);
                    changed = true;
                }
            }
        });
        ui.add_space(3.0);

        let groups = backend.groups();
        let rows: Vec<usize> = (0..groups.len()).filter(|&i| self.filter.group_visible(&groups[i])).collect();
        let mut picked_grp: Option<usize> = None;
        ui.push_id("gallery_groups_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(135.0)
                .column(Column::exact(46.0))
                .column(Column::exact(28.0))
                .column(Column::exact(84.0))
                .column(Column::exact(34.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["Grp", "ITA", "System", "|P|", "List N"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for &idx in &rows {
                        let g = &groups[idx];
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let tip = format!(
                                    "{} ({}) — max {}-fold, {}",
                                    g.symbol,
                                    g.crystal_system,
                                    g.max_rotation,
                                    if g.has_reflection_or_glide { "achiral" } else { "chiral" }
                                );
                                if ui.selectable_label(self.selected_group_idx == idx, g.symbol).on_hover_text(tip).clicked() {
                                    picked_grp = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(super::filter::ita_number(g.symbol).map_or_else(String::new, |n| n.to_string()));
                            });
                            row.col(|ui| {
                                ui.label(g.crystal_system);
                            });
                            row.col(|ui| {
                                ui.label(g.point_order.to_string());
                            });
                            row.col(|ui| {
                                ui.label(g.counts[self.rule.index()].to_string());
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_grp {
            self.select_group(idx, backend, canvas);
            changed = true;
        }
        changed
    }
}

/// A small label cut to `max_w` with an ellipsis (egui shows the whole text on hover when it is cut).
fn small_truncated(ui: &mut egui::Ui, text: String, max_w: f32) {
    ui.scope(|ui| {
        ui.set_max_width(max_w);
        ui.add(egui::Label::new(egui::RichText::new(text).small()).truncate());
    });
}
