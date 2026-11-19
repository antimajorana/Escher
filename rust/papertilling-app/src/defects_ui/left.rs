use egui_extras::{Column, TableBuilder};

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::defects_tab::{DefectSubTab, DefectsTabState};
use crate::live_ops_ao::Defect2dMode;
use crate::theme::ThemeTokens;

impl DefectsTabState {
    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) -> bool {
        if self.stale {
            self.rebuild_crystal(active_item, backend, canvas, true);
        }
        self.poll_worker(ui.ctx(), canvas);

        let mut dirty = false;
        let mut reset_view = false;

        ui.strong(format!("Active lattice: {} ({})", active_item.display_with_orig(), active_item.group));
        if self.initializing {
            ui.spinner();
            ui.label("Preparing the defect catalogue and picture in the background…");
            return false;
        }
        if let Some(err) = &self.error {
            ui.colored_label(tokens.status_err, err);
        }
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            for st in DefectSubTab::ALL {
                if ui.selectable_label(self.sub_tab == st, st.label()).clicked() {
                    self.sub_tab = st;
                    dirty = true;
                    reset_view = true;
                }
            }
        });
        ui.separator();

        match self.sub_tab {
            DefectSubTab::Dislocations => {
                if let Some(idx) = self.dislocations_table(ui) {
                    self.selected_disloc = idx;
                    dirty = true;
                }
            }
            DefectSubTab::Disclinations => {
                if let Some(idx) = self.disclinations_table(ui, tokens) {
                    self.selected_disclin = idx;
                    dirty = true;
                }
            }
            DefectSubTab::Walls => {
                if let Some(idx) = self.walls_table(ui, tokens) {
                    self.selected_wall = idx;
                    dirty = true;
                }
                if let Some(w) = self.catalogue.walls.get(self.selected_wall) {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Span:");
                        if ui.selectable_label(!self.wall_half || !w.can_end_orientable, "Full wall").clicked() && self.wall_half {
                            self.wall_half = false;
                            dirty = true;
                        }
                        let hw = ui.add_enabled(
                            w.can_end_orientable,
                            egui::Button::selectable(w.can_end_orientable && self.wall_half, "Half-wall (terminated)"),
                        ).on_hover_text(&w.ending_summary);
                        if hw.clicked() && !self.wall_half {
                            self.wall_half = true;
                            dirty = true;
                        }
                    });
                    ui.small(&w.ending_summary);
                }
            }
        }

        ui.separator();
        ui.strong("Patch & 2D Drawing Controls");
        ui.horizontal(|ui| {
            ui.label("Radius R:");
            if ui.add(egui::Slider::new(&mut self.radius, 2.4..=8.0).step_by(0.2)).changed() {
                dirty = true;
                reset_view = true;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("2D Drawing:");
            for dm in Defect2dMode::ALL {
                if ui.selectable_label(self.draw_mode == dm, dm.label()).clicked() && self.draw_mode != dm {
                    self.draw_mode = dm;
                    self.relaxed = dm != Defect2dMode::Raw;
                    dirty = true;
                }
            }
        });
        dirty |= super::growth::show(ui, self, active_item);

        if dirty {
            self.rebuild_crystal(active_item, backend, canvas, reset_view);
        }
        true
    }

    fn dislocations_table(&self, ui: &mut egui::Ui) -> Option<usize> {
        ui.strong(format!("Dislocation Classes ({} orbits)", self.catalogue.dislocations.len()));
        let mut picked = None;
        ui.push_id("disloc_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(130.0)
                .column(Column::exact(38.0))
                .column(Column::exact(68.0))
                .column(Column::exact(48.0))
                .column(Column::exact(42.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["ID", "b=(m,n)", "|b|/a", "|Orb|", "Mirror"] {
                        h.col(|ui| { ui.strong(t); });
                    }
                })
                .body(|mut body| {
                    for (idx, d) in self.catalogue.dislocations.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                if ui.selectable_label(self.selected_disloc == idx, &d.id).clicked() {
                                    picked = Some(idx);
                                }
                            });
                            row.col(|ui| { ui.label(format!("({}, {})", d.burgers_mn.0, d.burgers_mn.1)); });
                            row.col(|ui| { ui.label(format!("{:.2}", d.norm)); });
                            row.col(|ui| { ui.label(d.orbit_size.to_string()); });
                            row.col(|ui| { ui.label(d.mirror_partner.as_deref().unwrap_or("self")); });
                        });
                    }
                });
        });
        picked
    }

    fn disclinations_table(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) -> Option<usize> {
        if self.catalogue.centres.is_empty() {
            ui.colored_label(
                tokens.status_warn,
                "No rotational centres in this group (|P+| = 1): zero disclination classes.",
            );
            return None;
        }
        ui.strong("Rotation Centre Classes (C1..)");
        ui.push_id("centres_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(78.0)
                .column(Column::exact(38.0))
                .column(Column::exact(52.0))
                .column(Column::exact(46.0))
                .column(Column::remainder())
                .header(17.0, |mut h| {
                    for t in ["ID", "Wyck", "Order", "Location"] {
                        h.col(|ui| { ui.strong(t); });
                    }
                })
                .body(|mut body| {
                    for c in &self.catalogue.centres {
                        body.row(17.0, |mut row| {
                            row.col(|ui| { ui.strong(&c.id); });
                            row.col(|ui| { ui.label(&c.wyckoff); });
                            row.col(|ui| { ui.label(format!("{}-fold", c.order)); });
                            row.col(|ui| { ui.label(&c.location); });
                        });
                    }
                });
        });
        ui.add_space(4.0);
        ui.strong(format!(
            "Disclinations D(α, C) ({} classes, +180° != -180°)",
            self.catalogue.disclinations.len()
        ));
        let mut picked = None;
        ui.push_id("disclin_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(145.0)
                .column(Column::exact(102.0))
                .column(Column::exact(48.0))
                .column(Column::exact(52.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["Class", "Centre", "Frank", "Loc"] {
                        h.col(|ui| { ui.strong(t); });
                    }
                })
                .body(|mut body| {
                    for (idx, d) in self.catalogue.disclinations.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                if ui.selectable_label(self.selected_disclin == idx, &d.id).clicked() {
                                    picked = Some(idx);
                                }
                            });
                            row.col(|ui| { ui.label(&d.centre_id).on_hover_text(&d.wyckoff); });
                            row.col(|ui| { ui.label(format!("{:+}°", d.frank_angle_deg)); });
                            row.col(|ui| { ui.label(&d.location); });
                        });
                    }
                });
        });
        picked
    }

    fn walls_table(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) -> Option<usize> {
        ui.strong("Four Mirror / Glide Wall Kinds (Wa - Wd)");
        let mut picked = None;
        ui.push_id("walls_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(130.0)
                .column(Column::exact(38.0))
                .column(Column::exact(96.0))
                .column(Column::exact(68.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["ID", "Kind", "Ending", "Axis / Action"] {
                        h.col(|ui| { ui.strong(t); });
                    }
                })
                .body(|mut body| {
                    for (idx, w) in self.catalogue.walls.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let dir_tip = format!("dir = ({}, {})\n{}", w.direction_mn.0, w.direction_mn.1, w.ending_summary);
                                if ui.selectable_label(self.selected_wall == idx, &w.id).on_hover_text(dir_tip).clicked() {
                                    picked = Some(idx);
                                }
                            });
                            row.col(|ui| { ui.label(w.kind.label()); });
                            row.col(|ui| {
                                let (txt, col) = if w.can_end_orientable {
                                    ("orientable", tokens.status_ok)
                                } else {
                                    ("non-orient.", tokens.status_warn)
                                };
                                ui.colored_label(col, egui::RichText::new(txt).size(11.0)).on_hover_text(&w.ending_summary);
                            });
                            row.col(|ui| { ui.small(&w.description); });
                        });
                    }
                });
        });
        picked
    }
}
