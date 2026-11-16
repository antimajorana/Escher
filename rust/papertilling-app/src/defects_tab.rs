use egui_extras::{Column, TableBuilder};

use crate::backend::{
    AppBackend, CatalogueItemUi, DefectCatalogueUi, DefectReadoutUi, DefectSelectionUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_worker::LatestWorker;
use papertilling_core::types::{BuiltCrystal, Cell2D};

struct BuildJob {
    item: CatalogueItemUi,
    selection: DefectSelectionUi,
    radius: f64,
    relaxed: bool,
}

type BuildOutput = (DefectCatalogueUi, BuiltCrystal, DefectReadoutUi);

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
    /// The lattice changed while another tab was shown: catalogue and readout are rebuilt when this tab draws.
    stale: bool,
    initializing: bool,
    worker: LatestWorker<BuildJob, BuildOutput>,
    last_scene: Option<Scene2D>,
    reset_on_result: bool,
    error: Option<String>,
}

impl DefectsTabState {
    pub fn new(item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let catalogue = backend.defect_catalogue(item);
        let radius = 3.8;
        let relaxed = true;
        let (crystal, readout) =
            backend.build_defect(item, &DefectSelectionUi::Dislocation(0), radius, relaxed);
        let last_scene = Some(Scene2D::CrystalPatch {
            crystal, title: readout.title.clone(), note: readout.note.clone(),
        });
        Self {
            sub_tab: DefectSubTab::Dislocations,
            catalogue,
            selected_disloc: 0,
            selected_disclin: 0,
            selected_wall: 0,
            radius,
            relaxed,
            readout,
            stale: false,
            initializing: false,
            worker: LatestWorker::spawn("defects-build", (), |_: &mut (), job: BuildJob| {
                let catalogue = crate::live_ops::catalogue_for_lattice(&job.item);
                let (crystal, readout) = crate::live_ops::build_defect_patch(
                    &job.item, &job.selection, job.radius, job.relaxed,
                );
                (catalogue, crystal, readout)
            }),
            last_scene,
            reset_on_result: false,
            error: None,
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
        self.selected_disloc = 0;
        self.selected_disclin = 0;
        self.selected_wall = 0;
        self.worker.cancel();
        self.last_scene = None;
        self.initializing = true;
        self.stale = true;
        if sync_canvas {
            self.rebuild_crystal(item, backend, canvas, true);
        }
    }

    pub fn rebuild_crystal(
        &mut self,
        item: &CatalogueItemUi,
        _backend: &AppBackend,
        canvas: &mut Canvas2D,
        reset_view: bool,
    ) {
        self.worker.submit(BuildJob {
            item: item.clone(), selection: self.current_selection(),
            radius: self.radius, relaxed: self.relaxed,
        });
        self.stale = false;
        self.error = None;
        self.reset_on_result |= reset_view;
        let mut scene = self.last_scene.clone().unwrap_or_else(|| Scene2D::PeriodicCell {
            cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: Vec::new(), bonds: Vec::new() },
            repeat: 1, show_unit_cell: false,
        });
        if let Scene2D::CrystalPatch { note, .. } = &mut scene {
            *note = String::from("Previous result — stale while the new defect builds in the background.");
        }
        canvas.set_scene(scene, reset_view);
    }

    pub(crate) fn is_busy(&self) -> bool { self.stale || self.worker.is_busy() }

    fn take_result(&mut self, result: Result<BuildOutput, String>, canvas: &mut Canvas2D) {
        self.initializing = false;
        match result {
            Ok((catalogue, crystal, readout)) => {
                self.catalogue = catalogue;
                let scene = Scene2D::CrystalPatch {
                    crystal, title: readout.title.clone(), note: readout.note.clone(),
                };
                canvas.set_scene(scene.clone(), self.reset_on_result);
                self.last_scene = Some(scene);
                self.readout = readout;
            }
            Err(err) => self.error = Some(err),
        }
        self.reset_on_result = false;
    }

    #[cfg(test)]
    pub(crate) fn wait_idle(&mut self, canvas: &mut Canvas2D) {
        if let Some(result) = self.worker.wait(std::time::Duration::from_secs(120)) {
            self.take_result(result, canvas);
        }
        assert!(!self.is_busy(), "defect build did not complete");
    }

    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
        self.worker.set_repaint_ctx(ui.ctx());
        if self.stale {
            self.rebuild_crystal(active_item, backend, canvas, true);
        }
        if let Some(result) = self.worker.poll() { self.take_result(result, canvas); }
        let mut dirty = false;
        let mut reset_view = false;

        ui.strong(format!("Active lattice: {} ({})", active_item.label, active_item.group));
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
        if self.is_busy() || self.error.is_some() {
            ui.horizontal(|ui| {
                if self.is_busy() { ui.spinner(); }
                ui.label("Previous readout — stale");
            });
        }
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

#[cfg(test)]
#[path = "defects_tab_tests.rs"]
pub(crate) mod tests;
