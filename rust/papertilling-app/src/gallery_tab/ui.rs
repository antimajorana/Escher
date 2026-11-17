//! Left-rail browsing and rename bar of the Gallery tab.

use egui_extras::{Column, TableBuilder};

use super::{BrowseMode, GalleryTabState};
use crate::backend::{AppBackend, CatalogueRuleUi};
use crate::canvas2d::Canvas2D;

impl GalleryTabState {
    /// Inline rename control bar (Enter to save, Esc to cancel, Reset name when custom alias is active).
    pub fn show_rename_bar(&mut self, ui: &mut egui::Ui, backend: &mut AppBackend, canvas: &mut Canvas2D, id_salt: &str) {
        ui.push_id(id_salt, |ui| {
            ui.horizontal_wrapped(|ui| {
                if self.renaming {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.rename_buf)
                            .desired_width(165.0)
                            .hint_text(&self.active_item.default_label),
                    );
                    if self.rename_focus_pending {
                        resp.request_focus();
                        self.rename_focus_pending = false;
                    }
                    let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if esc {
                        self.cancel_rename();
                    } else if (enter && (resp.lost_focus() || resp.has_focus())) || ui.small_button("Save").clicked() {
                        self.commit_rename(backend, canvas);
                    } else if ui.small_button("Cancel").clicked() {
                        self.cancel_rename();
                    }
                    ui.small(format!("(id: {})", self.active_item.default_label));
                } else {
                    let title_resp = ui
                        .add(
                            egui::Label::new(egui::RichText::new(&self.active_item.label).strong())
                                .sense(egui::Sense::click()),
                        )
                        .on_hover_text("Double-click to give this lattice a custom name");
                    if title_resp.double_clicked() {
                        self.begin_rename();
                    }
                    if self.readout.chiral {
                        let tip = if self.readout.topologically_chiral {
                            "± Chiral enantiomorph pair in 2D (chiral bare graph, merged under Euclidean normalizer)"
                        } else {
                            "± Chiral enantiomorph pair in 2D (merged under Euclidean normalizer)"
                        };
                        ui.small("±").on_hover_text(tip);
                    }
                    if self.active_item.has_custom_name() {
                        ui.small(format!("[{}]", self.active_item.default_label));
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
            });
            if let Some(note) = &self.alias_note {
                ui.small(note);
            }
        });
    }

    /// Draw the full side panel (browsing + inspector) for legacy callers/tests.
    #[cfg(test)]
    pub fn show_side_panel(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let lattice_changed = self.show_left_panel(ui, backend, canvas);
        ui.separator();
        self.show_inspector_panel(ui, backend, canvas, &crate::theme::ThemeTokens::paper());
        lattice_changed
    }

    /// Left rail: browse mode tabs, group/catalogue/named tables, live search, and custom builder.
    pub fn show_left_panel(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let mut lattice_changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.strong("Browse:");
            if ui.selectable_label(self.mode == BrowseMode::ByGroup, "17 Groups").clicked() {
                self.select_group(self.selected_group_idx, backend, canvas);
                lattice_changed = true;
            }
            if ui.selectable_label(self.mode == BrowseMode::Named21, "Named (21)").clicked() {
                self.select_named(self.selected_named_idx, backend, canvas);
                lattice_changed = true;
            }
            if ui.selectable_label(self.mode == BrowseMode::CustomBuilder, "+ Custom").clicked() {
                self.mode = BrowseMode::CustomBuilder;
            }
        });
        ui.add_space(4.0);

        match self.mode {
            BrowseMode::ByGroup => {
                lattice_changed |= self.group_tables(ui, backend, canvas);
                let grp = backend.groups()[self.selected_group_idx].symbol;
                lattice_changed |= self.live_search_section(ui, backend, canvas, grp);
            }
            BrowseMode::Named21 => {
                lattice_changed |= self.named_table(ui, backend, canvas);
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
        let mut picked_grp: Option<usize> = None;
        ui.push_id("gallery_groups_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(135.0)
                .column(Column::exact(46.0))
                .column(Column::exact(84.0))
                .column(Column::exact(34.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["Grp", "System", "|P|", "List N"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, g) in groups.iter().enumerate() {
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

        ui.separator();
        let grp = groups.get(self.selected_group_idx).map_or("?", |g| g.symbol);
        ui.strong(format!("Lattices in {grp} ({}, {} entries)", self.rule.label(), self.catalogue().len()));
        if self.catalogue().is_empty() {
            ui.small("No stored lattice for this group under this rule.");
        }
        let mut picked_cat: Option<usize> = None;
        ui.push_id("gallery_cat_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(96.0)
                .column(Column::exact(26.0))
                .column(Column::remainder())
                .column(Column::exact(38.0))
                .column(Column::exact(44.0))
                .header(18.0, |mut h| {
                    for t in ["#", "Lattice", "Sites", "Coord"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, it) in self.catalogue().iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                ui.label(it.index.to_string());
                            });
                            row.col(|ui| {
                                let txt = it.display_with_orig();
                                if it.chiral {
                                    ui.horizontal(|ui| {
                                        if ui.selectable_label(self.selected_cat_idx == idx, txt).clicked() {
                                            picked_cat = Some(idx);
                                        }
                                        let tip = if it.topologically_chiral {
                                            "± Chiral enantiomorph pair in 2D (chiral bare graph)"
                                        } else {
                                            "± Chiral enantiomorph pair in 2D (merged under Euclidean normalizer)"
                                        };
                                        ui.small("±").on_hover_text(tip);
                                    });
                                } else if ui.selectable_label(self.selected_cat_idx == idx, txt).clicked() {
                                    picked_cat = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(it.sites_per_cell.to_string());
                            });
                            row.col(|ui| {
                                ui.label(&it.coordination);
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_cat {
            self.select_catalogue_entry(idx, backend, canvas);
            changed = true;
        }
        changed
    }

    fn named_table(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        ui.strong("21 Named Lattices (18 sheet + 3 variants)");
        let named = backend.named_lattices();
        let mut picked_named: Option<usize> = None;
        ui.push_id("named_21_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(185.0)
                .column(Column::remainder())
                .column(Column::exact(44.0))
                .column(Column::exact(44.0))
                .column(Column::exact(40.0))
                .header(18.0, |mut h| {
                    for t in ["Name", "Grp", "Sheet", "Sites"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, it) in named.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let txt = it.display_with_orig();
                                if it.chiral {
                                    ui.horizontal(|ui| {
                                        if ui.selectable_label(self.selected_named_idx == idx, txt).clicked() {
                                            picked_named = Some(idx);
                                        }
                                        ui.small("±").on_hover_text(
                                            "± Chiral enantiomorph pair in 2D (merged under Euclidean normalizer)",
                                        );
                                    });
                                } else if ui.selectable_label(self.selected_named_idx == idx, txt).clicked() {
                                    picked_named = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(&it.group);
                            });
                            row.col(|ui| {
                                ui.label(it.sheet_group.as_deref().unwrap_or("-"));
                            });
                            row.col(|ui| {
                                ui.label(it.sites_per_cell.to_string());
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_named {
            self.select_named(idx, backend, canvas);
            return true;
        }
        false
    }
}
