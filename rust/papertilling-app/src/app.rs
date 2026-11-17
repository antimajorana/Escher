use std::sync::Arc;
use std::sync::atomic::AtomicU8;

use egui_extras::{Size, StripBuilder};
use papertilling_core::types::{Cell2D, Frame2D, Recipe};

use crate::backend::{AppBackend, CatalogueItemUi, SurfaceTargetUi};
use crate::aliases::LatticeAliasKey;
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_tab::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::gallery_tab::GalleryTabState;
use crate::surfaces_tab::SurfacesTabState;
use crate::view3d::View3dState;

#[cfg(test)]
mod rename_tests;
mod smoke;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavTab {
    Gallery,
    Surfaces,
    Defects,
    Combine,
    Atlas,
}

impl NavTab {
    const ALL: [Self; 5] = [Self::Gallery, Self::Surfaces, Self::Defects, Self::Combine, Self::Atlas];

    fn label(self) -> &'static str {
        match self {
            Self::Gallery => "Gallery (17 Groups & 21 Named)",
            Self::Surfaces => "Surfaces (2D Net + 3D)",
            Self::Defects => "Defects (T, D, Wa-Wd)",
            Self::Combine => "Combine (Live Fusion & Cores)",
            Self::Atlas => "Atlas (0D–3D & H²)",
        }
    }
}

/// What the Defects and Combine tabs were built for: the lattice and its (slider-tuned) recipe.
type ItemKey = (LatticeAliasKey, String, Recipe, Frame2D);

fn item_key(item: &CatalogueItemUi) -> ItemKey {
    (item.alias_key.clone(), item.group.clone(), item.recipe.clone(), item.frame)
}

pub struct PapertillingApp {
    defects_seen: ItemKey,
    combine_seen: ItemKey,
    smoke_test: bool,
    smoke_reported: bool,
    last_smoke_stage: u32,
    smoke_frame_at_switch: u32,
    smoke_gen_at_switch: u64,
    smoke_frames: u32,
    smoke_outcome: Arc<AtomicU8>,
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
    combine: CombineTabState,
    atlas: crate::atlas_tab::AtlasTabState,
}

impl PapertillingApp {
    pub fn new(smoke_test: bool, adapter_name: String, backend_name: String) -> Self {
        let backend = AppBackend::new_with_os_config(smoke_test);
        backend.prefetch_surface_verdicts();
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
        let combine = CombineTabState::new(gallery.active_item(), &backend);
        let atlas = crate::atlas_tab::AtlasTabState::new();

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
            defects_seen: item_key(gallery.active_item()),
            combine_seen: item_key(gallery.active_item()),
            smoke_test,
            smoke_reported: false,
            last_smoke_stage: 0,
            smoke_frame_at_switch: 0,
            smoke_gen_at_switch: 1,
            smoke_frames: 0,
            smoke_outcome: Arc::new(AtomicU8::new(0)),
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
            combine,
            atlas,
        }
    }

    pub fn with_smoke_outcome(mut self, outcome: Arc<AtomicU8>) -> Self {
        self.smoke_outcome = outcome;
        self
    }

    pub(crate) fn frame_ui(&mut self, ui: &mut egui::Ui) -> bool {
        if self.smoke_test && self.smoke_step(ui.ctx()) {
            return true;
        }
        self.show_top_bar(ui);
        self.show_side_panel(ui);
        self.show_central_viewports(ui);
        false
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
                let item = self.gallery.active_item().clone();
                if item_key(&item) == self.defects_seen {
                    self.defects.rebuild_crystal(&item, &self.backend, &mut self.canvas2d, true);
                } else {
                    self.defects_seen = item_key(&item);
                    self.defects.reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                }
                self.defects.sync_view3d(&mut self.view3d);
            }
            NavTab::Combine => {
                let item = self.gallery.active_item().clone();
                if item_key(&item) == self.combine_seen {
                    self.combine.ensure_compared(&item, &self.backend);
                    self.combine.sync_to_canvas(&item, &mut self.canvas2d, true);
                } else {
                    self.combine_seen = item_key(&item);
                    self.combine.reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                }
                self.combine.sync_view3d(&item, &mut self.view3d);
            }
            NavTab::Atlas => {
                self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d);
            }
        }
    }

    fn export_dialog(&mut self) {
        if self.active_tab == NavTab::Defects && self.defects.is_busy() {
            self.export_status = String::from("Wait for the defect build before exporting.");
            return;
        }
        if self.active_tab == NavTab::Combine && (self.combine.is_building() || self.combine.dragging) {
            self.export_status = String::from("Finish moving the pins and wait for the crystal build before exporting.");
            return;
        }
        let default_name = match self.active_tab {
            NavTab::Gallery => "lattice_gallery.svg",
            NavTab::Surfaces => "surface_net.svg",
            NavTab::Defects => "defect_crystal.svg",
            NavTab::Combine => "combined_defects.svg",
            NavTab::Atlas => "atlas_scene.svg",
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

    fn export_obj_dialog(&mut self) {
        if self.active_tab == NavTab::Gallery {
            self.export_status = String::from("Open Surfaces, Defects, or Combine to export a 3D .obj mesh.");
            return;
        }
        if self.active_tab == NavTab::Defects && self.defects.is_busy() {
            self.export_status = String::from("Wait for the defect build before exporting .obj.");
            return;
        }
        if self.active_tab == NavTab::Combine && (self.combine.is_building() || self.combine.dragging) {
            self.export_status = String::from("Finish moving pins and wait for the crystal build before exporting .obj.");
            return;
        }
        let default_name = match self.active_tab {
            NavTab::Gallery => "mesh.obj",
            NavTab::Surfaces => "surface_mesh.obj",
            NavTab::Defects => "defect_sheet.obj",
            NavTab::Combine => "combined_sheet.obj",
            NavTab::Atlas => "atlas_mesh.obj",
        };
        let dialog = rfd::FileDialog::new()
            .set_title("Export active 3D mesh (.obj)")
            .set_file_name(default_name)
            .add_filter("Wavefront OBJ", &["obj"]);
        if let Some(path) = dialog.save_file() {
            let res = match self.active_tab {
                NavTab::Gallery => unreachable!(),
                NavTab::Surfaces => self.surfaces.export_active_obj(&path),
                NavTab::Defects => self.defects.export_active_obj(&path),
                NavTab::Combine => self.combine.export_active_obj(self.gallery.active_item(), &path),
                NavTab::Atlas => self.atlas.export_active_obj(&path, &self.view3d),
            };
            self.export_status = match res {
                Ok(msg) => msg,
                Err(err) => format!("OBJ export error: {err}"),
            };
        }
    }

    fn sync_active_tab_after_rename(&mut self) {
        let item = self.gallery.active_item().clone();
        if let Some(override_item) = &mut self.combine.override_item
            && override_item.alias_key == item.alias_key
        {
            override_item.label = self.backend.resolve_alias(override_item);
        }
        match self.active_tab {
            NavTab::Gallery => {}
            NavTab::Surfaces => {
                self.surfaces.activate_and_sync(&item, &self.backend, &mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Defects => {
                self.defects_seen = item_key(&item);
                self.defects.rebuild_crystal(&item, &self.backend, &mut self.canvas2d, false);
            }
            NavTab::Combine => {
                self.combine_seen = item_key(&item);
                self.combine.sync_to_canvas(&item, &mut self.canvas2d, false);
                self.combine.view3d_dirty = true;
            }
            NavTab::Atlas => {}
        }
    }

    fn show_top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
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
                let rev0 = self.gallery.revision();
                self.gallery.show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "top_rename");
                if self.gallery.revision() != rev0 {
                    self.sync_active_tab_after_rename();
                }
                ui.separator();
                if ui.button("Reset views").clicked() {
                    self.canvas2d.reset();
                    self.view3d.reset();
                }
                if ui.button("Export SVG / PNG...").clicked() && !self.smoke_test {
                    self.export_dialog();
                }
                if ui.button("Export .obj (3D)...").clicked() && !self.smoke_test {
                    self.export_obj_dialog();
                }
                if ui
                    .button("Import .json...")
                    .on_hover_text("Import a Notebook 3 defect configuration (.json)")
                    .clicked()
                    && !self.smoke_test
                {
                    self.switch_tab(NavTab::Combine);
                    self.combine
                        .import_json_dialog(&self.backend, &mut self.canvas2d);
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
                            let revision = self.gallery.revision();
                            self.gallery.show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "side_rename");
                            if revision != self.gallery.revision() {
                                self.sync_active_tab_after_rename();
                            }
                            ui.separator();
                            // Defects and Combine reload lazily on their next activation
                            // (`switch_tab` compares the active item's key, sliders included).
                            let _ = self.gallery.show_side_panel(
                                ui,
                                &self.backend,
                                &mut self.canvas2d,
                            );
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
                                self.gallery.rule = crate::backend::CatalogueRuleUi::Points;
                                self.gallery.select_group_by_symbol(
                                    &new_grp,
                                    &self.backend,
                                    &mut self.canvas2d,
                                );
                                let active = self.gallery.active_item().clone();
                                self.surfaces.activate_and_sync(
                                    &active,
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
                        NavTab::Combine => {
                            let item = self.gallery.active_item().clone();
                            self.combine.show_side_panel(
                                ui,
                                &item,
                                &self.backend,
                                &mut self.canvas2d,
                            );
                        }
                        NavTab::Atlas => {
                            self.atlas.show_side_panel(
                                ui,
                                &mut self.canvas2d,
                                &mut self.view3d,
                            );
                        }
                    }

                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Export 2D Scene (SVG / PNG)...").clicked() && !self.smoke_test {
                            self.export_dialog();
                        }
                        if ui.button("Export .obj (3D)...").clicked() && !self.smoke_test {
                            self.export_obj_dialog();
                        }
                    });
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
                    "2D Periodic Lattice — {} ({}) [drag: pan, scroll/pinch/+/-: zoom, 0/dbl-click: fit]",
                    self.gallery.active_item().display_with_orig(),
                    self.gallery.active_item().group
                ));
                ui.add_space(2.0);
                self.canvas2d.show(ui);
            }
            NavTab::Surfaces => {
                self.surfaces.poll_background(ui.ctx(), &mut self.canvas2d, &mut self.view3d);
                StripBuilder::new(ui)
                    .clip(true)
                    .sizes(Size::remainder().at_least(180.0), 2)
                    .horizontal(|mut strip| {
                        strip.cell(|ui| {
                            ui.strong("2D Fundamental Chart / Gluing Net (drag pan, scroll/pinch zoom)");
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
                let item = self.gallery.active_item().clone();
                self.defects
                    .show_central(ui, &item, &mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Combine => {
                let item = self.gallery.active_item().clone();
                self.combine
                    .show_central(ui, &item, &self.backend, &mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Atlas => {
                self.atlas.show_central(ui, &mut self.canvas2d, &mut self.view3d);
            }
        });
    }
}

impl eframe::App for PapertillingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let _ = self.frame_ui(ui);
    }
}
