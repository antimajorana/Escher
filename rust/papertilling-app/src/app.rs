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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavTab {
    Gallery,
    Surfaces,
    Defects,
    Combine,
}

impl NavTab {
    const ALL: [Self; 4] = [Self::Gallery, Self::Surfaces, Self::Defects, Self::Combine];

    fn label(self) -> &'static str {
        match self {
            Self::Gallery => "Gallery (17 Groups & 21 Named)",
            Self::Surfaces => "Surfaces (2D Net + 3D)",
            Self::Defects => "Defects (T, D, Wa-Wd)",
            Self::Combine => "Combine (Live Fusion & Cores)",
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
                    "2D Periodic Lattice — {} ({}) [drag: pan, scroll/pinch/+/-: zoom, 0/dbl-click: fit]",
                    self.gallery.active_item().display_with_orig(),
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
        });
    }
}

impl eframe::App for PapertillingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let _ = self.frame_ui(ui);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combine_tab::ToolMode;
    use crate::live_combine::spec_world_pos;
    use papertilling_core::defect::combine_types::{
        ClassifiedCharge, EquivalentSingleReport, NotebookConfigJson,
    };
    use papertilling_core::types::Isometry2;

    #[test]
    fn combine_pin_drag_snap_json_roundtrip_and_app_tabs() {
        let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
        app.switch_tab(NavTab::Combine);
        assert_eq!(app.active_tab, NavTab::Combine);
        assert!(app.canvas2d.counts().0 > 0);

        let backend = AppBackend::new();
        let square = backend.find_named("square").unwrap();
        let oblique = backend.find_named("oblique").unwrap();
        let mut canvas = Canvas2D::new(Scene2D::PeriodicCell {
            cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
            repeat: 2,
            show_unit_cell: false,
        });

        let mut st = CombineTabState::new(square, &backend);
        st.clear_defects(square, &backend, &mut canvas);

        // Place +90 disclination near 1b (0.5, 0.5), then drag/snap to (1.0, 0.0)
        st.tool_mode = ToolMode::PlaceDisclination;
        st.selected_angle_deg = 90;
        st.place_at_world([0.48, 0.53], square, &backend, &mut canvas);
        assert_eq!(st.specs.len(), 1);
        assert!((spec_world_pos(&st.info, 0, &st.specs[0])[0] - 0.5).abs() < 1e-6);
        assert!(!st.move_pin_to(0, [0.49, 0.51], square, &backend, &mut canvas));
        assert!(st.move_pin_to(0, [1.03, -0.04], square, &backend, &mut canvas));
        assert!((spec_world_pos(&st.info, 0, &st.specs[0])[0] - 1.0).abs() < 1e-6);
        assert!(!st.move_pin_to(99, [0.0, 0.0], square, &backend, &mut canvas));

        // Filter to 2c (2-fold bond midpoint) auto-snaps angle to 180 and point to (0.5, 0)
        st.centre_filter_idx = 3;
        st.place_at_world([0.47, 0.02], square, &backend, &mut canvas);
        assert_eq!(st.specs.len(), 2);
        assert!((spec_world_pos(&st.info, 1, &st.specs[1])[0] - 0.5).abs() < 1e-6);

        // Dislocation placement and continuous drag
        st.tool_mode = ToolMode::PlaceDislocation;
        st.custom_mn = (1, 0);
        st.place_at_world([0.25, -0.35], square, &backend, &mut canvas);
        assert!(st.move_pin_to(2, [0.65, -0.15], square, &backend, &mut canvas));
        st.finish_drag(square);
        st.wait_idle(square, Some(&mut canvas));
        assert!(!st.is_busy() && !st.last_build.preview);

        // NotebookConfigJson round-trip handling both Err and Ok compare_result gracefully
        st.compare_result = Some(Err(String::from("pending")));
        let raw_err = st.to_notebook_json(square).to_json_pretty().unwrap();
        let cfg_err = NotebookConfigJson::from_json_str(&raw_err).unwrap();
        assert!(cfg_err.summary.is_none() && cfg_err.format == "papertilling.config.v1");

        st.compare_result = Some(Ok(EquivalentSingleReport {
            charge: ClassifiedCharge {
                kind: String::from("dislocation"), angle: 0.0, holonomy: Isometry2::IDENTITY,
                b: Some([1.0, 0.0]), lattice_vector: Some(true), b_rep: Some([1.0, 0.0]),
                length: Some(1.0), centre: None, centre_class: None, label: String::from("T1 (1,0)"),
                single: None, loop_sum: [1.0, 0.0],
            },
            kind: String::from("dislocation"), single: None, candidates: Vec::new(),
            matches: vec![String::from("V1")], equivalent: true, verdict: String::from("matches V1"),
            scan: None, core_nodes: 4, core_edges: 4, config_core: None, gallery: Vec::new(),
        }));
        let raw_ok = st.to_notebook_json(square).to_json_pretty().unwrap();
        let cfg_ok = NotebookConfigJson::from_json_str(&raw_ok).unwrap();
        assert!(cfg_ok.summary.as_ref().is_some_and(|s| s.equivalent && s.matches == ["V1"]));

        let mut st2 = CombineTabState::new(oblique, &backend);
        assert!(st2.load_notebook_json(&cfg_ok, &backend, &mut canvas).is_ok());
        assert_eq!(st2.effective_item(oblique).group, "p4m");
        assert_eq!(st2.specs.len(), 3);

        let mut bad_cfg = cfg_ok;
        bad_cfg.recipe.group = String::from("invalid_group");
        assert!(st2.load_notebook_json(&bad_cfg, &backend, &mut canvas).is_err());
        assert!(NotebookConfigJson::from_json_str("{not valid json").is_err());
    }

    /// Regression: a Gallery change used to rebuild Defects and Combine synchronously on every
    /// click; now each reloads once, when it is shown, for the lattice the Gallery shows then.
    #[test]
    fn tabs_reload_lazily_for_the_current_gallery_lattice() {
        let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
        app.gallery.select_group_by_symbol("p4m", &app.backend, &mut app.canvas2d);
        let p4m = app.gallery.active_item().clone();
        assert_ne!(item_key(&p4m), app.combine_seen, "the start lattice is p6m kagome");
        app.switch_tab(NavTab::Combine);
        assert_eq!(app.combine_seen, item_key(&p4m));
        app.combine.wait_idle(&p4m, Some(&mut app.canvas2d));
        assert_eq!(app.combine.info.cell.basis.len(), p4m.sites_per_cell);
        assert!(app.combine.specs.iter().all(|s| matches!(s, papertilling_core::defect::surgery::DefectSpec::Disclination { angle_deg, .. } if (angle_deg.abs() - 90.0).abs() < 1e-9)),
            "p4m pair preset uses its 4-fold centres (±90°)");
        app.switch_tab(NavTab::Defects);
        assert_eq!(app.defects_seen, item_key(&p4m));
    }

    #[test]
    fn tuned_gallery_geometry_reaches_surfaces_defects_and_combine() {
        let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
        let idx = app.backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("oblique")).unwrap();
        app.gallery.select_named(idx, &app.backend, &mut app.canvas2d);
        app.gallery.set_param("b", 0.8, &mut app.canvas2d).unwrap();
        app.gallery.set_param("gamma", 60.0, &mut app.canvas2d).unwrap();
        let item = app.gallery.active_item().clone();
        assert_eq!((item.recipe.b, item.recipe.gamma), (Some(0.8), Some(60.0)));

        // By hand: a2 = 0.8 (cos 60°, sin 60°) = (0.4, 0.4 √3).
        app.switch_tab(NavTab::Combine);
        app.combine.wait_idle(&item, Some(&mut app.canvas2d));
        let a2 = app.combine.info.cell.a2;
        assert!((a2[0] - 0.4).abs() < 1e-10 && (a2[1] - 0.4 * 3.0_f64.sqrt()).abs() < 1e-10);

        app.switch_tab(NavTab::Defects);
        app.defects.wait_idle(&mut app.canvas2d);
        let Scene2D::CrystalPatch { crystal, .. } = app.canvas2d.scene() else { panic!("crystal"); };
        assert!(crystal.adj.iter().flatten().any(|e| (e.vec[0].hypot(e.vec[1]) - 0.8).abs() < 1e-9));

        app.surfaces.target = SurfaceTargetUi::Torus;
        app.surfaces.auto_size = false;
        (app.surfaces.params.h, app.surfaces.params.k) = (3, 4);
        app.switch_tab(NavTab::Surfaces);
        let Scene2D::SurfaceNet { mesh } = app.canvas2d.scene() else { panic!("surface net"); };
        assert_eq!(mesh.sites.len(), 12, "one square-network site per cell on a 3×4 torus");
        assert!(mesh.bonds.iter().any(|(_, _, v)| (v[0].hypot(v[1]) - 0.8).abs() < 1e-9));
    }
}
