use egui_extras::{Column, TableBuilder};

use crate::backend::{
    AppBackend, CatalogueItemUi, DefectCatalogueUi, DefectReadoutUi, DefectSelectionUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefectSubTab {
    Dislocations,
    Disclinations,
    Walls,
}

impl DefectSubTab {
    pub const ALL: [Self; 3] = [Self::Dislocations, Self::Disclinations, Self::Walls];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dislocations => "Dislocations",
            Self::Disclinations => "Disclinations",
            Self::Walls => "Walls (Wa-Wd)",
        }
    }
}

pub struct DefectsTabState {
    pub sub_tab: DefectSubTab,
    catalogue: DefectCatalogueUi,
    selected_disloc: usize,
    selected_disclin: usize,
    selected_wall: usize,
    pub radius: f64,
    pub relaxed: bool,
    pub readout: DefectReadoutUi,
}

impl DefectsTabState {
    pub fn new(item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let catalogue = backend.defect_catalogue(item);
        let radius = 3.8;
        let relaxed = true;
        let (_, readout) =
            backend.build_defect(item, &DefectSelectionUi::Dislocation(0), radius, relaxed);
        Self {
            sub_tab: DefectSubTab::Dislocations,
            catalogue,
            selected_disloc: 0,
            selected_disclin: 0,
            selected_wall: 0,
            radius,
            relaxed,
            readout,
        }
    }

    fn current_selection(&self) -> DefectSelectionUi {
        match self.sub_tab {
            DefectSubTab::Dislocations => DefectSelectionUi::Dislocation(self.selected_disloc),
            DefectSubTab::Disclinations => DefectSelectionUi::Disclination(self.selected_disclin),
            DefectSubTab::Walls => DefectSelectionUi::Wall(self.selected_wall),
        }
    }

    pub fn reload_for_item(
        &mut self,
        item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        sync_canvas: bool,
    ) {
        self.catalogue = backend.defect_catalogue(item);
        self.selected_disloc = 0;
        self.selected_disclin = 0;
        self.selected_wall = 0;
        if sync_canvas {
            self.rebuild_crystal(item, backend, canvas, true);
        }
    }

    pub fn rebuild_crystal(
        &mut self,
        item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        reset_view: bool,
    ) {
        let sel = self.current_selection();
        let (crystal, readout) = backend.build_defect(item, &sel, self.radius, self.relaxed);
        canvas.set_scene(
            Scene2D::CrystalPatch {
                crystal,
                title: readout.title.clone(),
                note: readout.note.clone(),
            },
            reset_view,
        );
        self.readout = readout;
    }

    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
        let mut dirty = false;
        let mut reset_view = false;

        ui.strong(format!("Active lattice: {} ({})", active_item.label, active_item.group));
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            for st in DefectSubTab::ALL {
                if ui
                    .selectable_label(self.sub_tab == st, st.label())
                    .clicked()
                {
                    self.sub_tab = st;
                    dirty = true;
                    reset_view = true;
                }
            }
        });
        ui.separator();

        match self.sub_tab {
            DefectSubTab::Dislocations => {
                ui.strong(format!(
                    "Dislocation Classes ({} orbits)",
                    self.catalogue.dislocations.len()
                ));
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
                                h.col(|ui| {
                                    ui.strong(t);
                                });
                            }
                        })
                        .body(|mut body| {
                            for (idx, d) in self.catalogue.dislocations.iter().enumerate() {
                                body.row(18.0, |mut row| {
                                    row.col(|ui| {
                                        if ui
                                            .selectable_label(self.selected_disloc == idx, &d.id)
                                            .clicked()
                                        {
                                            picked = Some(idx);
                                        }
                                    });
                                    row.col(|ui| {
                                        ui.label(format!("({}, {})", d.burgers_mn.0, d.burgers_mn.1));
                                    });
                                    row.col(|ui| {
                                        ui.label(format!("{:.2}", d.norm));
                                    });
                                    row.col(|ui| {
                                        ui.label(d.orbit_size.to_string());
                                    });
                                    row.col(|ui| {
                                        ui.label(d.mirror_partner.as_deref().unwrap_or("self"));
                                    });
                                });
                            }
                        });
                });
                if let Some(idx) = picked {
                    self.selected_disloc = idx;
                    dirty = true;
                }
            }
            DefectSubTab::Disclinations => {
                if self.catalogue.centres.is_empty() {
                    ui.colored_label(
                        egui::Color32::from_rgb(176, 82, 24),
                        "No rotational centres in this group (|P+| = 1): zero disclination classes.",
                    );
                } else {
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
                                    h.col(|ui| {
                                        ui.strong(t);
                                    });
                                }
                            })
                            .body(|mut body| {
                                for c in &self.catalogue.centres {
                                    body.row(17.0, |mut row| {
                                        row.col(|ui| {
                                            ui.strong(&c.id);
                                        });
                                        row.col(|ui| {
                                            ui.label(&c.wyckoff);
                                        });
                                        row.col(|ui| {
                                            ui.label(format!("{}-fold", c.order));
                                        });
                                        row.col(|ui| {
                                            ui.label(&c.location);
                                        });
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
                                    h.col(|ui| {
                                        ui.strong(t);
                                    });
                                }
                            })
                            .body(|mut body| {
                                for (idx, d) in self.catalogue.disclinations.iter().enumerate() {
                                    body.row(18.0, |mut row| {
                                        row.col(|ui| {
                                            if ui
                                                .selectable_label(self.selected_disclin == idx, &d.id)
                                                .clicked()
                                            {
                                                picked = Some(idx);
                                            }
                                        });
                                        row.col(|ui| {
                                            ui.label(&d.centre_id).on_hover_text(&d.wyckoff);
                                        });
                                        row.col(|ui| {
                                            ui.label(format!("{:+}°", d.frank_angle_deg));
                                        });
                                        row.col(|ui| {
                                            ui.label(&d.location);
                                        });
                                    });
                                }
                            });
                    });
                    if let Some(idx) = picked {
                        self.selected_disclin = idx;
                        dirty = true;
                    }
                }
            }
            DefectSubTab::Walls => {
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
                                h.col(|ui| {
                                    ui.strong(t);
                                });
                            }
                        })
                        .body(|mut body| {
                            for (idx, w) in self.catalogue.walls.iter().enumerate() {
                                body.row(18.0, |mut row| {
                                    row.col(|ui| {
                                        let dir_tip = format!(
                                            "dir = ({}, {})",
                                            w.direction_mn.0, w.direction_mn.1
                                        );
                                        if ui
                                            .selectable_label(self.selected_wall == idx, &w.id)
                                            .on_hover_text(dir_tip)
                                            .clicked()
                                        {
                                            picked = Some(idx);
                                        }
                                    });
                                    row.col(|ui| {
                                        ui.label(w.kind.label());
                                    });
                                    row.col(|ui| {
                                        ui.small(&w.description);
                                    });
                                });
                            }
                        });
                });
                if let Some(idx) = picked {
                    self.selected_wall = idx;
                    dirty = true;
                }
            }
        }

        ui.separator();
        ui.strong("Patch & Relaxation Controls");
        ui.horizontal(|ui| {
            ui.label("Radius R:");
            if ui
                .add(egui::Slider::new(&mut self.radius, 2.4..=6.2).step_by(0.2))
                .changed()
            {
                dirty = true;
                reset_view = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Drawing:");
            if ui.selectable_label(!self.relaxed, "Raw surgery").clicked() {
                self.relaxed = false;
                dirty = true;
            }
            if ui
                .selectable_label(self.relaxed, "Relaxed / Embedded")
                .clicked()
            {
                self.relaxed = true;
                dirty = true;
            }
        });

        if dirty {
            self.rebuild_crystal(active_item, backend, canvas, reset_view);
        }

        ui.separator();
        ui.strong("Core Charge & Topology Readout");
        ui.label(&self.readout.charge_summary);
        ui.small(&self.readout.holonomy_detail);
        ui.small(format!("Core structure: {}", self.readout.core_rings));
        if !self.readout.note.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(168, 92, 14),
                format!("Note: {}", self.readout.note),
            );
        }
    }
}
