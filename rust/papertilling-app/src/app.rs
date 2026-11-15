use egui_extras::{Size, StripBuilder};
use papertilling_core::types::Cell2D;

use crate::backend::{AppBackend, SurfaceTargetUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::defects_tab::DefectsTabState;
use crate::gallery_tab::GalleryTabState;
use crate::surfaces_tab::SurfacesTabState;
use crate::view3d::View3dState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavTab {
    Gallery,
    Surfaces,
    Defects,
}

impl NavTab {
    const ALL: [Self; 3] = [Self::Gallery, Self::Surfaces, Self::Defects];

    fn label(self) -> &'static str {
        match self {
            Self::Gallery => "Gallery (17 Groups & 21 Named)",
            Self::Surfaces => "Surfaces (2D Net + 3D)",
            Self::Defects => "Defects (T, D, Wa-Wd)",
        }
    }
}

pub struct PapertillingApp {
    smoke_test: bool,
    smoke_reported: bool,
    last_smoke_stage: u32,
    adapter_name: String,
    backend_name: String,
    active_tab: NavTab,
    export_status: String,
    backend: AppBackend,
    canvas2d: Canvas2D,
    view3d: View3dState,
    gallery: GalleryTabState,
    surfaces: SurfacesTabState,
    defects: DefectsTabState,
}

impl PapertillingApp {
    pub fn new(smoke_test: bool, adapter_name: String, backend_name: String) -> Self {
        let backend = AppBackend::new();
        let placeholder = Scene2D::PeriodicCell {
            cell: Cell2D {
                a1: [1.0, 0.0],
                a2: [0.0, 1.0],
                basis: vec![[0.0, 0.0]],
                bonds: Vec::new(),
            },
            repeat: 3,
            show_unit_cell: true,
        };
        let mut canvas2d = Canvas2D::new(placeholder);
        let mut view3d = View3dState::default();
        let gallery = GalleryTabState::new(&backend, &mut canvas2d);
        let mut surfaces = SurfacesTabState::new();
        let defects = DefectsTabState::new(gallery.active_item(), &backend);

        let active_tab = if smoke_test {
            surfaces.activate_and_sync(
                gallery.active_item(),
                &backend,
                &mut canvas2d,
                &mut view3d,
            );
            NavTab::Surfaces
        } else {
            NavTab::Gallery
        };

        Self {
            smoke_test,
            smoke_reported: false,
            last_smoke_stage: 0,
            adapter_name,
            backend_name,
            active_tab,
            export_status: String::from("Ready (SVG / high-DPI PNG)"),
            backend,
            canvas2d,
            view3d,
            gallery,
            surfaces,
            defects,
        }
    }

    fn switch_tab(&mut self, tab: NavTab) {
        if self.active_tab == tab {
            return;
        }
        self.active_tab = tab;
        match tab {
            NavTab::Gallery => {
                self.gallery
                    .sync_canvas(&self.backend, &mut self.canvas2d, true);
            }
            NavTab::Surfaces => {
                self.surfaces.activate_and_sync(
                    self.gallery.active_item(),
                    &self.backend,
                    &mut self.canvas2d,
                    &mut self.view3d,
                );
            }
            NavTab::Defects => {
                self.defects.rebuild_crystal(
                    self.gallery.active_item(),
                    &self.backend,
                    &mut self.canvas2d,
                    true,
                );
            }
        }
    }

    fn export_dialog(&mut self) {
        let default_name = match self.active_tab {
            NavTab::Gallery => "lattice_gallery.svg",
            NavTab::Surfaces => "surface_net.svg",
            NavTab::Defects => "defect_crystal.svg",
        };
        let dialog = rfd::FileDialog::new()
            .set_title("Export active 2D scene (SVG or PNG)")
            .set_file_name(default_name)
            .add_filter("SVG vector", &["svg"])
            .add_filter("PNG image", &["png"]);
        if let Some(path) = dialog.save_file() {
            self.export_status = match self.canvas2d.export_to_path(&path) {
                Ok(msg) => msg,
                Err(err) => format!("Export error: {err}"),
            };
        }
    }

    fn show_top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(format!("papertilling {}", papertilling_core::VERSION));
                ui.separator();
                for tab in NavTab::ALL {
                    if ui
                        .selectable_label(self.active_tab == tab, tab.label())
                        .clicked()
                    {
                        self.switch_tab(tab);
                    }
                }
                ui.separator();
                if ui.button("Reset views").clicked() {
                    self.canvas2d.reset();
                    self.view3d.reset();
                }
                if ui.button("Export SVG / PNG...").clicked() && !self.smoke_test {
                    self.export_dialog();
                }
            });
        });
    }

    fn show_side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("side_panel")
            .default_size(355.0)
            .min_size(290.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.active_tab {
                        NavTab::Gallery => {
                            let changed = self.gallery.show_side_panel(
                                ui,
                                &self.backend,
                                &mut self.canvas2d,
                            );
                            if changed {
                                self.defects.reload_for_item(
                                    self.gallery.active_item(),
                                    &self.backend,
                                    &mut self.canvas2d,
                                    false,
                                );
                            }
                        }
                        NavTab::Surfaces => {
                            let item = self.gallery.active_item().clone();
                            if let Some(new_grp) = self.surfaces.show_side_panel(
                                ui,
                                &item,
                                &self.backend,
                                &mut self.canvas2d,
                                &mut self.view3d,
                            ) {
                                self.gallery.select_group_by_symbol(
                                    &new_grp,
                                    &self.backend,
                                    &mut self.canvas2d,
                                );
                                self.defects.reload_for_item(
                                    self.gallery.active_item(),
                                    &self.backend,
                                    &mut self.canvas2d,
                                    false,
                                );
                                self.surfaces.activate_and_sync(
                                    self.gallery.active_item(),
                                    &self.backend,
                                    &mut self.canvas2d,
                                    &mut self.view3d,
                                );
                            }
                        }
                        NavTab::Defects => {
                            let item = self.gallery.active_item().clone();
                            self.defects.show_side_panel(
                                ui,
                                &item,
                                &self.backend,
                                &mut self.canvas2d,
                            );
                        }
                    }

                    ui.separator();
                    if ui.button("Export 2D Scene (SVG / PNG)...").clicked() && !self.smoke_test {
                        self.export_dialog();
                    }
                    ui.small(&self.export_status);
                    let (sites, bonds) = self.canvas2d.counts();
                    let (prep, paint) = self.view3d.counters.counts();
                    ui.small(format!(
                        "2D: {sites} sites, {bonds} lines | GPU ({}) prep={prep} paint={paint} gen={}",
                        self.backend_name,
                        self.view3d.mesh_generation()
                    ));
                });
            });
    }

    fn show_central_viewports(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| match self.active_tab {
            NavTab::Gallery => {
                ui.strong(format!(
                    "2D Periodic Lattice — {} ({}) [drag: pan, scroll: zoom, double-click: fit]",
                    self.gallery.active_item().label,
                    self.gallery.active_item().group
                ));
                ui.add_space(2.0);
                self.canvas2d.show(ui);
            }
            NavTab::Surfaces => {
                StripBuilder::new(ui)
                    .clip(true)
                    .sizes(Size::remainder().at_least(180.0), 2)
                    .horizontal(|mut strip| {
                        strip.cell(|ui| {
                            ui.strong("2D Fundamental Chart / Gluing Net (drag pan, scroll zoom)");
                            ui.add_space(2.0);
                            self.canvas2d.show(ui);
                        });
                        strip.cell(|ui| {
                            ui.strong("3D wgpu Surface Viewport (two-sided lit faces + bond curves)");
                            ui.add_space(2.0);
                            self.view3d.show(ui);
                        });
                    });
            }
            NavTab::Defects => {
                ui.strong("2D Surgical Crystal Patch (irregular core rings, cones, A|B domains)");
                ui.add_space(2.0);
                self.canvas2d.show(ui);
            }
        });
    }
}

impl eframe::App for PapertillingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.smoke_test {
            let (prepared, painted) = self.view3d.counters.counts();
            if prepared >= 5 && painted >= 5 {
                if !self.smoke_reported {
                    self.smoke_reported = true;
                    println!(
                        "PAPERTILLING_SMOKE_OK adapter={} backend={} frames={painted} uploads={}",
                        self.adapter_name,
                        self.backend_name,
                        self.view3d.counters.uploads()
                    );
                }
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            if painted > self.last_smoke_stage && painted < 5 {
                self.last_smoke_stage = painted;
                let (next_target, cov) = match painted {
                    1 => (SurfaceTargetUi::Torus, false),
                    2 => (SurfaceTargetUi::ProjectivePlane, true),
                    3 => (SurfaceTargetUi::KleinBottle, false),
                    _ => (SurfaceTargetUi::Genus2, false),
                };
                self.surfaces.target = next_target;
                self.surfaces.show_cover = cov;
                self.surfaces.activate_and_sync(
                    self.gallery.active_item(),
                    &self.backend,
                    &mut self.canvas2d,
                    &mut self.view3d,
                );
            }
            self.view3d.yaw += 0.14;
            ui.ctx().request_repaint();
        }

        self.show_top_bar(ui);
        self.show_side_panel(ui);
        self.show_central_viewports(ui);
    }
}
