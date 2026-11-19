use std::sync::Arc;
use std::sync::atomic::AtomicU8;

use papertilling_core::types::{Cell2D, Frame2D, Recipe};

use crate::aliases::LatticeAliasKey;
use crate::analysis_mode::AnalysisModeState;
use crate::backend::{AppBackend, CatalogueItemUi, SurfaceTargetUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::gallery_tab::GalleryTabState;
use crate::surfaces_tab::SurfacesTabState;
use crate::theme::StudioTheme;
use crate::view3d::{OffscreenGpuContext, View3dState};
use crate::workbench::WorkbenchState;

mod export3d;
mod modes;
mod preset_apply;
#[cfg(test)]
mod preset_tests;
#[cfg(test)]
mod export3d_tests;
#[cfg(test)]
mod rename_tests;
pub(crate) mod session_hooks;
mod session_migrate;
#[cfg(test)]
mod session_tests;
mod smoke;
#[cfg(test)]
mod tests;
mod workbench_hooks;

pub use modes::{AppMode, DefectsView, LatticesView, NavTab};

/// What the Defects and Combine tabs were built for: the lattice and its (slider-tuned) recipe.
type ItemKey = (LatticeAliasKey, String, Recipe, Frame2D);

fn item_key(item: &CatalogueItemUi) -> ItemKey {
    (
        item.alias_key.clone(),
        item.group.clone(),
        item.recipe.clone(),
        item.frame,
    )
}

#[cfg(test)]
pub(crate) struct TestLayoutRects {
    pub top: egui::Rect,
    pub status: egui::Rect,
    pub left: egui::Rect,
    pub right: egui::Rect,
    pub center: egui::Rect,
}

#[cfg(test)]
impl Default for TestLayoutRects {
    fn default() -> Self {
        Self {
            top: egui::Rect::NOTHING,
            status: egui::Rect::NOTHING,
            left: egui::Rect::NOTHING,
            right: egui::Rect::NOTHING,
            center: egui::Rect::NOTHING,
        }
    }
}

pub struct PapertillingApp {
    defects_seen: ItemKey,
    combine_seen: ItemKey,
    pub(crate) smoke_test: bool,
    smoke_reported: bool,
    last_smoke_stage: u32,
    smoke_frame_at_switch: u32,
    smoke_gen_at_switch: u64,
    smoke_frames: u32,
    smoke_outcome: Arc<AtomicU8>,
    adapter_name: String,
    pub(crate) backend_name: String,
    pub(crate) active_mode: AppMode,
    pub(crate) lattices_view: LatticesView,
    pub(crate) defects_view: DefectsView,
    pub(crate) active_tab: NavTab,
    pub(crate) export_status: String,
    /// Pixel scale of a 2D PNG export (1, 2 or 4: 1024, 2048 or 4096 px a side), chosen in the Export menu.
    pub(crate) png_scale: u8,
    /// GPU handles for the offscreen 3D PNG export (`None` headless).
    pub(crate) offscreen_gpu: Option<OffscreenGpuContext>,
    /// The 3D PNG being composited and written on a worker thread (joined when dropped).
    pub(crate) export3d_job: Option<export3d::Export3dJob>,
    pub(crate) theme: StudioTheme,
    pub(crate) left_rail_open: bool,
    pub(crate) left_rail_width: f32,
    pub(crate) right_inspector_open: bool,
    pub(crate) right_inspector_width: f32,
    pub(crate) viewport_split_ratio: f32,
    #[cfg(test)]
    pub(crate) test_rects: TestLayoutRects,
    pub(crate) backend: AppBackend,
    pub(crate) canvas2d: Canvas2D,
    pub(crate) view3d: View3dState,
    pub(crate) gallery: GalleryTabState,
    pub(crate) surfaces: SurfacesTabState,
    pub(crate) defects: DefectsTabState,
    pub(crate) combine: CombineTabState,
    pub(crate) atlas: crate::atlas_tab::AtlasTabState,
    pub(crate) analysis: AnalysisModeState,
    /// Stage S2..S6 panels behind `NavTab::Workbench`.
    pub(crate) workbench: WorkbenchState,
    pub(crate) switcher: crate::ui::lattice_switcher::LatticeSwitcher,
    /// Follow-up work of the last preset waiting on a tab's background worker (Combine pins or Defects spectrum).
    pub(crate) preset_pending: Option<preset_apply::PendingPreset>,
    pub(crate) session: session_hooks::SessionHost,
    /// The Command Deck overlay and its HUD hooks (`crate::game_menu`).
    pub(crate) deck: crate::game_menu::DeckState,
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
        let analysis = AnalysisModeState::new();

        let (active_mode, lattices_view, defects_view, active_tab) = if smoke_test {
            surfaces.activate_and_sync(
                gallery.active_item(),
                &backend,
                &mut canvas2d,
                &mut view3d,
            );
            (
                AppMode::Lattices,
                LatticesView::Manifolds,
                DefectsView::default(),
                NavTab::Surfaces,
            )
        } else {
            (
                AppMode::Lattices,
                LatticesView::Plane2D,
                DefectsView::default(),
                NavTab::Gallery,
            )
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
            active_mode,
            lattices_view,
            defects_view,
            active_tab,
            export_status: String::from("Ready (SVG / high-DPI PNG)"),
            png_scale: 2,
            offscreen_gpu: None,
            export3d_job: None,
            theme: StudioTheme::Paper,
            left_rail_open: true,
            left_rail_width: 300.0,
            right_inspector_open: true,
            right_inspector_width: 320.0,
            viewport_split_ratio: 0.5,
            #[cfg(test)]
            test_rects: TestLayoutRects::default(),
            backend,
            canvas2d,
            view3d,
            gallery,
            surfaces,
            defects,
            combine,
            atlas,
            analysis,
            workbench: WorkbenchState::new(),
            switcher: Default::default(),
            preset_pending: None,
            session: session_hooks::SessionHost::for_mode(smoke_test),
            deck: Default::default(),
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
        let ctx = ui.ctx().clone();
        self.poll_export_3d();
        self.pump_preset_pending(Some(&ctx));
        self.gallery.thumbs.begin_frame(ctx.cumulative_pass_nr());
        self.handle_global_shortcuts(&ctx);
        let on_gallery = self.active_tab == NavTab::Gallery;
        let canvas = if on_gallery { Some(&mut self.canvas2d) } else { None };
        if self.gallery.poll_retune(canvas) && !on_gallery {
            // a background parameter move finished after leaving the Gallery: the active tab follows it
            self.resync_active_tab_for_lattice();
        }
        self.show_hud_bar(ui);
        self.show_status_bar(ui);
        self.show_left_rail(ui);
        self.show_right_inspector(ui);
        self.show_viewport_pane(ui);
        self.show_switcher_modal(&ctx);
        if self.deck.open {
            self.show_command_deck(&ctx);
        }
        self.switcher.push_recent(self.gallery.base_item());
        false
    }

    /// Select `item` (a stored lattice at its default values) from anywhere, e.g. the Quick Lattice Switcher:
    /// the Gallery selection follows it and the active tab reloads for it at once, without a tab round-trip.
    pub(crate) fn select_lattice_and_sync(&mut self, item: CatalogueItemUi) {
        if self.select_lattice_quiet(item) {
            self.resync_active_tab_for_lattice();
        }
    }

    /// Select `item` in the Gallery (and Recent) without reloading the active tab; true = the lattice changed.
    /// For callers that switch the view right after (its activation loads the new lattice once).
    pub(crate) fn select_lattice_quiet(&mut self, item: CatalogueItemUi) -> bool {
        if item.alias_key == self.gallery.base_item().alias_key {
            // already selected: keep its tuned parameters and every tab's work (e.g. Ctrl+K, Enter on Recent #1)
            return false;
        }
        self.gallery.cancel_rename();
        self.gallery.select_external(item, &self.backend, &mut self.canvas2d);
        self.switcher.push_recent(self.gallery.base_item());
        true
    }

    /// Reload the active tab for the Gallery's current `active_item()` (another lattice or parameter values).
    pub(crate) fn resync_active_tab_for_lattice(&mut self) {
        let item = self.gallery.active_item().clone();
        match self.active_tab {
            NavTab::Gallery => {}
            NavTab::Surfaces => {
                self.surfaces
                    .activate_and_sync(&item, &self.backend, &mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Defects if item_key(&item) != self.defects_seen => {
                self.defects_seen = item_key(&item);
                self.defects
                    .reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                self.defects.sync_view3d(&mut self.view3d);
            }
            NavTab::Combine if item_key(&item) != self.combine_seen => {
                self.combine_seen = item_key(&item);
                self.combine
                    .reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                self.combine.sync_view3d(&item, &mut self.view3d);
            }
            NavTab::Defects | NavTab::Combine => {}
            // the Gallery redraw above replaced the Atlas 2D scene: restore it
            NavTab::Atlas => self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d),
            NavTab::Analysis => {
                self.analysis.reset_transient_for_session();
                self.ensure_analysis_sources_synced();
                self.analysis.sync_canvas(&self.surfaces, &self.defects, &self.combine, &mut self.canvas2d);
                self.analysis.sync_view3d(&self.surfaces, &self.defects, &self.combine, &mut self.view3d);
            }
            // the Gallery redraw replaced the workbench scene: rebuild (lattice-dependent panels) and re-push it
            NavTab::Workbench => self.workbench_lattice_changed(),
        }
    }

    /// On the Gallery, apply a pending background parameter move before exporting what the sliders show.
    pub(crate) fn finish_gallery_moves(&mut self) {
        if self.active_tab == NavTab::Gallery {
            self.gallery.finish_pending_retune(&mut self.canvas2d);
        }
    }

    pub(crate) fn export_dialog(&mut self) {
        if self.export3d_job.is_some() {
            // its finishing status would otherwise overwrite this export's result
            self.export_status = String::from("A 3D PNG export is still being written.");
            return;
        }
        self.finish_gallery_moves();
        if self.active_tab == NavTab::Defects && self.defects.is_busy() {
            self.export_status = String::from("Wait for the defect build before exporting.");
            return;
        }
        if self.active_tab == NavTab::Surfaces && self.surfaces.is_building() {
            self.export_status = String::from("Wait for the surface build before exporting.");
            return;
        }
        if self.active_tab == NavTab::Combine
            && (self.combine.is_building() || self.combine.dragging || self.combine.cut_dragging)
        {
            self.export_status =
                String::from("Finish moving the pins and wait for the crystal build before exporting.");
            return;
        }
        if self.active_tab == NavTab::Analysis && (self.analysis.is_busy() || self.analysis_source_busy()) {
            self.export_status = String::from("Wait for the spectrum or source build before exporting.");
            return;
        }
        if self.active_tab == NavTab::Workbench && self.workbench_busy() {
            self.export_status = String::from("Wait for the scene build before exporting.");
            return;
        }
        let wb_name = format!("{}_scene.svg", self.workbench.view.file_stem());
        let default_name = match self.active_tab {
            NavTab::Gallery => "lattice_gallery.svg",
            NavTab::Surfaces => "surface_net.svg",
            NavTab::Defects => "defect_crystal.svg",
            NavTab::Combine => "combined_defects.svg",
            NavTab::Atlas => "atlas_scene.svg",
            NavTab::Analysis => "analysis_spectrum.svg",
            NavTab::Workbench => wb_name.as_str(),
        };
        let dialog = rfd::FileDialog::new()
            .set_title("Export active 2D scene (SVG or PNG)")
            .set_file_name(default_name)
            .add_filter("SVG vector", &["svg"])
            .add_filter("PNG image", &["png"]);
        if let Some(path) = dialog.save_file() {
            self.export_status = match self.canvas2d.export_to_path(&path, u32::from(self.png_scale)) {

                Ok(msg) => msg,
                Err(err) => format!("Export error: {err}"),
            };
        }
    }

    pub(crate) fn export_obj_dialog(&mut self) {
        if self.export3d_job.is_some() {
            // its finishing status would otherwise overwrite this export's result
            self.export_status = String::from("A 3D PNG export is still being written.");
            return;
        }
        self.finish_gallery_moves();
        if self.active_tab == NavTab::Gallery {
            self.export_status =
                String::from("Open Surfaces, Defects, Combine or Atlas to export a 3D .obj mesh.");
            return;
        }
        if self.active_tab == NavTab::Analysis && (self.analysis.is_busy() || self.analysis_source_busy()) {
            self.export_status = String::from("Wait for the spectrum or source build before exporting .obj.");
            return;
        }
        if self.active_tab == NavTab::Workbench && self.workbench_busy() {
            self.export_status = String::from("Wait for the scene build before exporting .obj.");
            return;
        }
        if self.active_tab == NavTab::Defects && self.defects.is_busy() {
            self.export_status = String::from("Wait for the defect build before exporting .obj.");
            return;
        }
        if self.active_tab == NavTab::Surfaces && self.surfaces.is_building() {
            self.export_status = String::from("Wait for the surface build before exporting .obj.");
            return;
        }
        if self.active_tab == NavTab::Combine
            && (self.combine.is_building() || self.combine.dragging || self.combine.cut_dragging)
        {
            self.export_status =
                String::from("Finish moving pins and wait for the crystal build before exporting .obj.");
            return;
        }
        let wb_name = format!("{}_mesh.obj", self.workbench.view.file_stem());
        let default_name = match self.active_tab {
            NavTab::Gallery => "mesh.obj",
            NavTab::Surfaces => "surface_mesh.obj",
            NavTab::Defects => "defect_sheet.obj",
            NavTab::Combine => "combined_sheet.obj",
            NavTab::Atlas => "atlas_mesh.obj",
            NavTab::Analysis => "analysis_mesh.obj",
            NavTab::Workbench => wb_name.as_str(),
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
                NavTab::Analysis => self.analysis.export_active_obj(
                    self.gallery.active_item(),
                    &self.surfaces,
                    &self.defects,
                    &self.combine,
                    &path,
                ),
                NavTab::Workbench => self.workbench.export_active_obj(&path, &self.view3d),
            };
            self.export_status = match res {
                Ok(msg) => msg,
                Err(err) => format!("OBJ export error: {err}"),
            };
        }
    }

    pub(crate) fn sync_active_tab_after_rename(&mut self) {
        self.switcher.refresh_labels(&self.backend);
        let item = self.gallery.active_item().clone();
        if let Some(override_item) = &mut self.combine.override_item
            && override_item.alias_key == item.alias_key
        {
            override_item.label = self.backend.resolve_alias(override_item);
        }
        match self.active_tab {
            NavTab::Gallery => {}
            NavTab::Surfaces => {
                self.surfaces
                    .activate_and_sync(&item, &self.backend, &mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Defects => {
                self.defects_seen = item_key(&item);
                self.defects
                    .rebuild_crystal(&item, &self.backend, &mut self.canvas2d, false);
            }
            NavTab::Combine => {
                self.combine_seen = item_key(&item);
                self.combine.sync_to_canvas(&item, &mut self.canvas2d, false);
                self.combine.view3d_dirty = true;
            }
            // the renamed Gallery cell was redrawn into the shared canvas: restore the Atlas scene
            NavTab::Atlas => self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d),
            NavTab::Analysis => {
                self.analysis.sync_canvas(&self.surfaces, &self.defects, &self.combine, &mut self.canvas2d);
                self.analysis.sync_view3d(&self.surfaces, &self.defects, &self.combine, &mut self.view3d);
            }
            // the renamed Gallery cell was redrawn into the shared canvas: restore the workbench scene
            NavTab::Workbench => self.workbench_lattice_changed(),
        }
    }
}

impl eframe::App for PapertillingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let _ = self.frame_ui(ui);
    }

    fn on_exit(&mut self) {
        self.save_session_on_exit();
    }
}
