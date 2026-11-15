use egui_extras::{Size, StripBuilder};
use papertilling_core::types::Cell2D;

use crate::backend::{AppBackend, SurfaceTargetUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_tab::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::gallery_tab::GalleryTabState;
use crate::surfaces_tab::SurfacesTabState;
use crate::view3d::View3dState;

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
    combine: CombineTabState,
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
            combine,
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
            NavTab::Combine => {
                let item = self.gallery.active_item();
                self.combine.ensure_compared(item, &self.backend);
                self.combine.sync_to_canvas(item, &mut self.canvas2d, true);
            }
        }
    }

    fn export_dialog(&mut self) {
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
                if ui
                    .button("Import .json...")
                    .on_hover_text("Import a Notebook 3 defect configuration (.json)")
                    .clicked()
                    && !self.smoke_test
                {
                    self.active_tab = NavTab::Combine;
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
                            let changed = self.gallery.show_side_panel(
                                ui,
                                &self.backend,
                                &mut self.canvas2d,
                            );
                            if changed {
                                let active = self.gallery.active_item().clone();
                                self.defects.reload_for_item(
                                    &active,
                                    &self.backend,
                                    &mut self.canvas2d,
                                    false,
                                );
                                self.combine.reload_for_item(
                                    &active,
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
                                let active = self.gallery.active_item().clone();
                                self.defects.reload_for_item(
                                    &active,
                                    &self.backend,
                                    &mut self.canvas2d,
                                    false,
                                );
                                self.combine.reload_for_item(
                                    &active,
                                    &self.backend,
                                    &mut self.canvas2d,
                                    false,
                                );
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
            NavTab::Combine => {
                let item = self.gallery.active_item().clone();
                self.combine
                    .show_central(ui, &item, &self.backend, &mut self.canvas2d);
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
}
