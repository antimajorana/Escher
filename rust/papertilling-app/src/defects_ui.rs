//! Side panel and central 2D / 3D / Split viewports of the `Defects` tab.

use egui_extras::{Column, Size, StripBuilder, TableBuilder};

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::defects_tab::{DefectSubTab, DefectsTabState};
use crate::mesh3d::{DefectViewMode, show_sheet3d_controls};
use crate::view3d::View3dState;

impl DefectsTabState {
    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
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
            return;
        }
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::DARK_RED, err);
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
                if let Some(idx) = self.disclinations_table(ui) {
                    self.selected_disclin = idx;
                    dirty = true;
                }
            }
            DefectSubTab::Walls => {
                if let Some(idx) = self.walls_table(ui) {
                    self.selected_wall = idx;
                    dirty = true;
                }
            }
        }

        ui.separator();
        ui.strong("Patch, 2D & 3D Sheet Controls");
        ui.horizontal(|ui| {
            ui.label("View:");
            for vm in DefectViewMode::ALL {
                if ui.selectable_label(self.view_mode == vm, vm.label()).clicked() {
                    self.view_mode = vm;
                    self.view3d_dirty = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Radius R:");
            if ui.add(egui::Slider::new(&mut self.radius, 2.4..=6.2).step_by(0.2)).changed() {
                dirty = true;
                reset_view = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("2D Drawing:");
            if ui.selectable_label(!self.relaxed, "Raw surgery").clicked() {
                self.relaxed = false;
                dirty = true;
            }
            if ui.selectable_label(self.relaxed, "Relaxed / Embedded").clicked() {
                self.relaxed = true;
                dirty = true;
            }
        });

        let (kappa_changed, vis_changed) = show_sheet3d_controls(ui, &mut self.sheet_opts, "def_side_col");
        if vis_changed {
            self.view3d_dirty = true;
        }

        if dirty {
            self.rebuild_crystal(active_item, backend, canvas, reset_view);
        } else if kappa_changed {
            self.rebuild_sheet3d_only(active_item);
        }

        ui.separator();
        ui.strong("Core Charge & 3D Buckling Readout");
        if self.is_busy() || self.error.is_some() {
            ui.horizontal(|ui| {
                if self.is_busy() {
                    ui.spinner();
                }
                ui.label("Previous readout — stale");
            });
        }
        ui.label(&self.readout.charge_summary);
        ui.small(&self.readout.holonomy_detail);
        ui.small(format!("Core structure: {}", self.readout.core_rings));
        if !self.last_sheet.xyz.is_empty() {
            let z_min = self.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
            let z_max = self.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
            let dz_a = (z_max - z_min) / self.last_sheet.mean_bond_len.max(1e-9);
            ui.small(format!(
                "3D Sheet: Δz/a = {dz_a:.2} | E_stretch = {:.4}, E_bend = {:.4}",
                self.last_sheet.stretch_energy, self.last_sheet.bend_energy
            ));
        }
        if !self.readout.note.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(168, 92, 14),
                format!("Note: {}", self.readout.note),
            );
        }
    }

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        self.poll_worker(ui.ctx(), canvas);
        ui.horizontal_wrapped(|ui| {
            ui.strong("View:");
            for vm in DefectViewMode::ALL {
                if ui.selectable_label(self.view_mode == vm, vm.label()).clicked() {
                    self.view_mode = vm;
                    self.view3d_dirty = true;
                }
            }
            if self.is_busy() {
                ui.spinner();
            }
        });
        if self.view_mode.uses_3d() {
            let (kc, vc) = show_sheet3d_controls(ui, &mut self.sheet_opts, "def_top_col");
            if vc {
                self.view3d_dirty = true;
            }
            if kc {
                self.rebuild_sheet3d_only(active_item);
            }
        }
        if self.view3d_dirty && self.view_mode.uses_3d() {
            self.sync_view3d(view3d);
        }
        ui.add_space(2.0);

        match self.view_mode {
            DefectViewMode::Planar2D => {
                ui.strong("2D Surgical Crystal Patch (irregular core rings, cones, A|B domains)");
                ui.add_space(2.0);
                canvas.show(ui);
            }
            DefectViewMode::Sheet3D => {
                ui.strong("3D Elastic Sheet Geometry (buckling relieves in-plane defect strain)");
                ui.add_space(2.0);
                view3d.show(ui);
            }
            DefectViewMode::Split2D3D => {
                StripBuilder::new(ui)
                    .clip(true)
                    .sizes(Size::remainder().at_least(180.0), 2)
                    .horizontal(|mut strip| {
                        strip.cell(|ui| {
                            ui.strong("2D Surgical Crystal Patch");
                            ui.add_space(2.0);
                            canvas.show(ui);
                        });
                        strip.cell(|ui| {
                            ui.strong("3D Elastic Sheet Geometry (orbit & zoom)");
                            ui.add_space(2.0);
                            view3d.show(ui);
                        });
                    });
            }
        }
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

    fn disclinations_table(&self, ui: &mut egui::Ui) -> Option<usize> {
        if self.catalogue.centres.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(176, 82, 24),
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

    fn walls_table(&self, ui: &mut egui::Ui) -> Option<usize> {
        ui.strong("Four Mirror / Glide Wall Kinds (Wa - Wd)");
        let mut picked = None;
        ui.push_id("walls_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(130.0)
                .column(Column::exact(42.0))
                .column(Column::exact(118.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["ID", "Kind", "Axis / Action"] {
                        h.col(|ui| { ui.strong(t); });
                    }
                })
                .body(|mut body| {
                    for (idx, w) in self.catalogue.walls.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let dir_tip = format!("dir = ({}, {})", w.direction_mn.0, w.direction_mn.1);
                                if ui.selectable_label(self.selected_wall == idx, &w.id).on_hover_text(dir_tip).clicked() {
                                    picked = Some(idx);
                                }
                            });
                            row.col(|ui| { ui.label(w.kind.label()); });
                            row.col(|ui| { ui.small(&w.description); });
                        });
                    }
                });
        });
        picked
    }
}
