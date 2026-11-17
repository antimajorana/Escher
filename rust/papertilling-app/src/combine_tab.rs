use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

use papertilling_core::defect::combine_types::EquivalentSingleReport;
use papertilling_core::defect::kinematics::AbsorptionPreview;
use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasCutRay, CanvasMarker, MarkerKind, Scene2D};
use crate::combine_half_wall_ui::CombineHalfWallState;
use crate::combine_worker::{
    BuildJob, BuildWorkerOutput, CombineSheet3dBundle, CompareJob, CompareOutput, LatestWorker,
    build_sheet_bundle_sync, empty_crystal, spawn_build_worker, spawn_compare_worker,
    spawn_lattice_worker,
};
use crate::core_panel::CorePanelState;
use crate::defects_inspectors::DefectInspectors;
use crate::live_combine::{
    CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo, build_combined_crystal,
    burgers_from_mn, cuts::build_cut_rays, preset_neighbour_pair, preset_shortest_dislocation,
    spec_short_label, spec_world_pos,
};
use crate::mesh3d::{
    DefectViewMode, GpuMeshData, Sheet3dViewOpts, mesh_from_sheet3d_with_spectral, sheet3d_caption,
};
use crate::view3d::View3dState;

/// Manual patch radius range (slider and `.json` import): covers Python's default radii.
pub(crate) const MANUAL_R_MIN: f64 = 2.4;
pub(crate) const MANUAL_R_MAX: f64 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    PanAndDrag,
    PlaceDisclination,
    PlaceDislocation,
    PlaceHalfWall,
}

/// Combine tab. Building the crystal and comparing it with the single defect run on two background
/// workers (`combine_worker`): the UI submits jobs, polls results every frame and never blocks.
pub struct CombineTabState {
    pub tool_mode: ToolMode,
    pub(crate) override_item: Option<CatalogueItemUi>,
    pub(crate) info: Arc<CombineLatticeInfo>,
    pub specs: Vec<DefectSpec>,
    pub(crate) selected_angle_deg: i32,
    pub(crate) centre_filter_idx: usize,
    pub(crate) selected_disloc_idx: usize,
    pub custom_mn: (i32, i32),
    pub opts: CombineBuildOptions,
    pub auto_compare: bool,
    pub grid: usize,
    pub show_branch_cuts: bool,
    pub show_glide_lines: bool,
    pub motion_mode: MotionMode,
    pub kinematic_status: String,
    pub(crate) drag_anchor: Option<(usize, Vec<DefectSpec>)>,
    pub(crate) pending_disc_step: Option<kinematics::PendingDiscStep>,
    pub(crate) pending_absorb: Option<AbsorptionPreview>,
    pub(crate) undo_stack: Vec<Vec<DefectSpec>>,
    pub view_mode: DefectViewMode,
    pub sheet_opts: Sheet3dViewOpts,
    pub(crate) last_build: CombineBuildOutput,
    pub(crate) last_sheet3d: CombineSheet3dBundle,
    pub(crate) view3d_dirty: bool,
    pub compare_result: Option<Result<EquivalentSingleReport, String>>,
    /// `compare_result` belongs to an earlier configuration (a newer comparison is pending or due).
    pub(crate) compare_stale: bool,
    pub(crate) compare_ms: f64,
    pub(crate) cache_stats: (usize, usize),
    compare_after_build: bool,
    reset_view_on_build: bool,
    canvas_dirty: bool,
    preview_queued: bool,
    preview_dirty: bool,
    pub(crate) dragging: bool,
    pub(crate) last_hover_world: Option<[f64; 2]>,
    pub json_status: String,
    pub(crate) init_error: Option<String>,
    pub core_panel: CorePanelState,
    pub half_wall: CombineHalfWallState,
    pub inspectors: DefectInspectors,
    builder: LatestWorker<BuildJob, BuildWorkerOutput>,
    comparer: LatestWorker<CompareJob, CompareOutput>,
    initializer: LatestWorker<CatalogueItemUi, Arc<CombineLatticeInfo>>,
}

impl CombineTabState {
    pub fn new(gallery_item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let info = Arc::new(backend.combine_lattice_info(gallery_item));
        let specs = preset_neighbour_pair(&info);
        let opts = CombineBuildOptions {
            draw_mode: CombineDrawMode::FastRelaxed,
            full_planar_repair: false,
            auto_radius: true,
            manual_radius: info.r_auto_base,
            core_radius: 2,
        };
        let sheet_opts = Sheet3dViewOpts::default();
        let last_build = build_combined_crystal(gallery_item, &info, &[], &opts, false);
        let last_sheet3d = build_sheet_bundle_sync(&info, &last_build.crystal, last_build.applied_radius, sheet_opts.kappa_b);
        let mut st = Self {
            tool_mode: ToolMode::PanAndDrag,
            override_item: None,
            selected_angle_deg: info.allowed_angles.first().copied().unwrap_or(90),
            centre_filter_idx: 0,
            selected_disloc_idx: 0,
            custom_mn: info.dislocations.first().map_or((1, 0), |d| d.burgers_mn),
            info,
            specs,
            opts,
            auto_compare: true,
            grid: 1,
            show_branch_cuts: false,
            show_glide_lines: true,
            motion_mode: MotionMode::Conserved,
            kinematic_status: String::new(),
            drag_anchor: None,
            pending_disc_step: None,
            pending_absorb: None,
            undo_stack: Vec::new(),
            view_mode: DefectViewMode::Split2D3D,
            sheet_opts,
            last_build,
            last_sheet3d,
            view3d_dirty: true,
            compare_result: None,
            compare_stale: false,
            compare_ms: 0.0,
            cache_stats: (0, 0),
            compare_after_build: false,
            reset_view_on_build: true,
            canvas_dirty: false,
            preview_queued: false,
            preview_dirty: false,
            dragging: false,
            last_hover_world: None,
            json_status: String::new(),
            init_error: None,
            core_panel: CorePanelState::default(),
            half_wall: CombineHalfWallState::from_item(gallery_item),
            inspectors: DefectInspectors::new(),
            builder: spawn_build_worker(),
            comparer: spawn_compare_worker(),
            initializer: spawn_lattice_worker(),
        };
        st.request_build(gallery_item, false, false, true);
        st
    }

    pub fn effective_item<'a>(&'a self, gallery_item: &'a CatalogueItemUi) -> &'a CatalogueItemUi {
        self.override_item.as_ref().unwrap_or(gallery_item)
    }

    /// A build or a comparison is running or queued.
    pub fn is_busy(&self) -> bool {
        self.initializer.is_busy() || self.builder.is_busy() || self.comparer.is_busy() || self.compare_after_build
    }

    pub(crate) fn is_building(&self) -> bool {
        self.initializer.is_busy() || self.builder.is_busy()
    }

    fn lattice_unavailable(&self) -> bool {
        self.initializer.is_busy() || self.init_error.is_some()
    }

    pub(crate) fn is_initializing(&self) -> bool { self.initializer.is_busy() }

    pub(crate) fn is_comparing(&self) -> bool {
        self.comparer.is_busy() || self.compare_after_build
    }

    fn reset_lattice_defaults(&mut self, gallery_item: &CatalogueItemUi, info: CombineLatticeInfo) {
        self.clear_kinematic_drag_state(true);
        self.inspectors.clear_for_switch();
        self.half_wall.refresh(gallery_item);
        self.init_error = None;
        self.selected_angle_deg = info.allowed_angles.first().copied().unwrap_or(90);
        self.centre_filter_idx = 0;
        self.selected_disloc_idx = 0;
        self.custom_mn = info.dislocations.first().map_or((1, 0), |d| d.burgers_mn);
        self.opts.manual_radius = info.r_auto_base;
        self.info = Arc::new(info);
        // Never show the previous lattice while the new build runs: a defect-free disk is instant.
        self.last_build = build_combined_crystal(gallery_item, &self.info, &[], &self.opts, false);
        self.last_sheet3d = build_sheet_bundle_sync(&self.info, &self.last_build.crystal, self.last_build.applied_radius, self.sheet_opts.kappa_b);
        self.canvas_dirty = true;
        self.view3d_dirty = true;
        self.compare_result = None;
        self.comparer.cancel();
    }

    pub fn reload_for_item(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, canvas: &mut Canvas2D, sync_canvas: bool) {
        self.clear_kinematic_drag_state(true);
        self.inspectors.clear_for_switch();
        self.override_item = None;
        self.init_error = None;
        self.builder.cancel();
        self.comparer.cancel();
        self.compare_after_build = false;
        self.compare_stale = false;
        self.compare_result = None;
        self.preview_queued = false;
        self.preview_dirty = false;
        self.specs.clear();
        self.dragging = false;
        self.initializer.submit(gallery_item.clone());
        self.last_build.crystal = empty_crystal();
        self.last_build.refused = None;
        self.last_sheet3d = CombineSheet3dBundle::empty();
        self.last_build.charge_label = String::from("Preparing lattice…");
        self.last_build.note = String::from("Building the defect catalogue in the background");
        self.canvas_dirty = true;
        self.view3d_dirty = true;
        self.reset_view_on_build = true;
        if sync_canvas {
            self.sync_to_canvas(gallery_item, canvas, true);
        }
    }

    /// Submit a build of the current defects; `then_compare`: compare once it lands (final builds only).
    pub(crate) fn request_build(&mut self, gallery_item: &CatalogueItemUi, preview: bool, then_compare: bool, reset_view: bool) {
        if self.lattice_unavailable() {
            return;
        }
        // Let an in-flight preview finish before submitting the next one. Invalidating it on every
        // mouse move starves the canvas when builds take longer than a pointer frame.
        if preview && self.preview_queued && self.builder.is_busy() {
            self.preview_dirty = true;
            return;
        }
        let item = self.effective_item(gallery_item).clone();
        let job = BuildJob {
            item,
            info: Arc::clone(&self.info),
            specs: self.specs.clone(),
            opts: self.opts.clone(),
            preview,
            kappa_b: self.sheet_opts.kappa_b,
            reuse_2d: None,
        };
        self.builder.submit(job);
        self.view3d_dirty = true;
        self.reset_view_on_build |= reset_view;
        self.preview_queued = preview;
        self.preview_dirty = false;
        self.compare_after_build = !preview && then_compare && !self.specs.is_empty();
        self.comparer.cancel();
        if self.specs.is_empty() {
            self.compare_result = None;
        }
        self.compare_stale = self.compare_result.is_some();
    }

    fn submit_compare(&mut self, gallery_item: &CatalogueItemUi) {
        if self.specs.is_empty() {
            self.compare_result = None;
            return;
        }
        if let Some(reason) = &self.last_build.refused {
            // The crystal is a defect-free stand-in: comparing it would report "perfect lattice".
            self.compare_result = Some(Err(format!("not compared: build refused ({reason})")));
            self.compare_stale = false;
            return;
        }
        let job = CompareJob {
            item: self.effective_item(gallery_item).clone(),
            info: Arc::clone(&self.info),
            specs: self.specs.clone(),
            crystal: self.last_build.crystal.clone(),
            core_radius: self.opts.core_radius,
            grid: self.grid,
        };
        self.comparer.submit(job);
        self.compare_stale = self.compare_result.is_some();
    }

    /// Compare now if the current build is final and up to date, else right after it lands.
    pub fn run_compare(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() { return; }
        if self.builder.is_busy() || self.last_build.preview {
            self.compare_after_build = true;
            if !self.builder.is_busy() {
                self.request_build(gallery_item, false, true, false);
            }
        } else {
            self.submit_compare(gallery_item);
        }
    }

    pub fn ensure_compared(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend) {
        let fresh = self.compare_result.is_some() && !self.compare_stale;
        if self.auto_compare && !self.specs.is_empty() && !fresh && !self.is_comparing() {
            self.run_compare(gallery_item);
        }
    }

    /// Collect finished jobs (call every frame while the tab is shown); `ctx` lets workers wake the UI.
    pub fn poll(&mut self, ctx: Option<&egui::Context>, gallery_item: &CatalogueItemUi, mut canvas: Option<&mut Canvas2D>) {
        if let Some(ctx) = ctx {
            self.initializer.set_repaint_ctx(ctx);
            self.builder.set_repaint_ctx(ctx);
            self.comparer.set_repaint_ctx(ctx);
        }
        if let Some(result) = self.initializer.poll() { self.take_init(result, gallery_item); }
        if let Some(res) = self.builder.poll() {
            self.take_build(res);
            self.on_combine_build_landed(gallery_item);
        }
        self.poll_spectrum_inspector(ctx, gallery_item, canvas.as_deref_mut());
        if self.compare_after_build && !self.builder.is_busy() && !self.last_build.preview {
            self.compare_after_build = false;
            self.submit_compare(gallery_item);
        }
        if let Some(res) = self.comparer.poll() {
            let (rep, ms, stats) = res.unwrap_or_else(|err| (Err(err), 0.0, self.cache_stats));
            self.compare_result = Some(rep);
            (self.compare_ms, self.cache_stats, self.compare_stale) = (ms, stats, false);
        }
        if self.canvas_dirty && let Some(canvas) = canvas {
            let reset = std::mem::take(&mut self.reset_view_on_build) && !self.dragging;
            self.sync_to_canvas(gallery_item, canvas, reset);
            self.canvas_dirty = false;
        }
        if self.dragging && self.preview_dirty && !self.builder.is_busy() {
            self.request_build(gallery_item, true, false, false);
        }
    }

    /// Block until every queued job has landed (tests).
    #[cfg(test)]
    pub fn wait_idle(&mut self, gallery_item: &CatalogueItemUi, mut canvas: Option<&mut Canvas2D>) {
        for _ in 0..8 {
            if let Some(result) = self.initializer.wait(Duration::from_secs(120)) {
                self.take_init(result, gallery_item);
            }
            if let Some(res) = self.builder.wait(Duration::from_secs(120)) {
                self.take_build(res);
                self.on_combine_build_landed(gallery_item);
            }
            self.poll(None, gallery_item, canvas.as_deref_mut());
            if let Some(res) = self.comparer.wait(Duration::from_secs(300)) {
                let (rep, ms, stats) = res.unwrap_or_else(|err| (Err(err), 0.0, self.cache_stats));
                self.compare_result = Some(rep);
                (self.compare_ms, self.cache_stats, self.compare_stale) = (ms, stats, false);
            }
            self.poll(None, gallery_item, canvas.as_deref_mut());
            if !self.is_busy() {
                return;
            }
        }
    }

    pub fn request_sheet3d_only(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() {
            return;
        }
        // A stiffness edit must not supersede a newer geometry job with the last completed crystal.
        if self.builder.is_busy() || self.last_build.preview {
            let compare = self.compare_after_build || self.auto_compare;
            self.request_build(gallery_item, self.dragging, compare, false);
            return;
        }
        let reuse = (!self.last_build.crystal.pos.is_empty()).then(|| self.last_build.clone());
        let job = BuildJob {
            item: self.effective_item(gallery_item).clone(),
            info: Arc::clone(&self.info),
            specs: self.specs.clone(),
            opts: self.opts.clone(),
            preview: false,
            kappa_b: self.sheet_opts.kappa_b,
            reuse_2d: reuse,
        };
        self.builder.submit(job);
        self.view3d_dirty = true;
    }

    pub fn sync_view3d(&mut self, gallery_item: &CatalogueItemUi, view3d: &mut View3dState) {
        if self.is_initializing() {
            view3d.set_mesh(GpuMeshData::default(), String::from("Preparing lattice…"));
            self.view3d_dirty = false;
            return;
        }
        if let Some(err) = &self.init_error {
            view3d.set_mesh(GpuMeshData::default(), format!("3D unavailable: {err}"));
            self.view3d_dirty = false;
            return;
        }
        let item = self.effective_item(gallery_item);
        let title = format!("{} ({}) — {} defect(s) | {}", item.display_with_orig(), item.group, self.specs.len(), self.last_build.charge_label);
        let (c, s) = if self.sheet_opts.show_flat_baseline {
            (&self.last_sheet3d.flat_crystal, &self.last_sheet3d.flat_sheet)
        } else {
            (&self.last_build.crystal, &self.last_sheet3d.sheet)
        };
        let gpu = mesh_from_sheet3d_with_spectral(
            c,
            s,
            &self.sheet_opts,
            self.inspectors.spectral_3d_for(self.sheet_opts.color_mode),
        );
        let mut cap = sheet3d_caption(&title, s, &self.sheet_opts);
        if self.is_building() { cap.push_str(" | previous result — stale"); }
        view3d.set_mesh(gpu, cap);
        self.view3d_dirty = false;
    }

    fn take_build(&mut self, res: Result<BuildWorkerOutput, String>) {
        match res {
            Ok((out, sheet3d)) => {
                self.last_build = out;
                self.last_sheet3d = sheet3d;
            }
            Err(err) => {
                self.last_build.note = err.clone();
                self.last_build.charge_label = String::from("Build failed");
                self.last_build.charge_detail = err.clone();
                self.last_build.refused = Some(err.clone());
                self.compare_after_build = false;
                self.comparer.cancel();
                self.compare_result = Some(Err(format!("not compared: build failed ({err})")));
                self.compare_stale = false;
            }
        }
        self.canvas_dirty = true;
        self.view3d_dirty = true;
    }

    #[cfg(test)]
    pub(crate) fn fail_init_for_test(&mut self, err: &str, gallery_item: &CatalogueItemUi) {
        self.initializer.cancel();
        self.take_init(Err(err.to_owned()), gallery_item);
    }

    pub(crate) fn take_init(&mut self, result: Result<Arc<CombineLatticeInfo>, String>, gallery_item: &CatalogueItemUi) {
        match result {
            Ok(info) => {
                self.reset_lattice_defaults(gallery_item, (*info).clone());
                self.specs = preset_neighbour_pair(&self.info);
                self.request_build(gallery_item, false, self.auto_compare, true);
            }
            Err(err) => {
                self.init_error = Some(err.clone());
                self.last_build.charge_label = String::from("Lattice load failed");
                self.last_build.charge_detail = err.clone();
                self.last_build.refused = Some(err.clone());
                self.last_build.note = err;
                self.canvas_dirty = true;
                self.view3d_dirty = true;
            }
        }
    }

    pub fn current_cut_rays(&self) -> Vec<CanvasCutRay> {
        if self.last_build.refused.is_some() {
            return Vec::new();
        }
        let cuts: Vec<_> = if self.show_branch_cuts {
            self.last_build.applied_cuts.clone()
        } else {
            self.last_build.applied_cuts.iter().copied().filter(|ac| ac.is_half_wall).collect()
        };
        build_cut_rays(
            &cuts,
            self.last_build.patch_centre,
            self.last_build.applied_radius,
        )
    }

    pub fn sync_to_canvas(&self, gallery_item: &CatalogueItemUi, canvas: &mut Canvas2D, reset_view: bool) {
        let item = self.effective_item(gallery_item);
        let title = format!("{} ({}) — {} defect(s) | {}", item.display_with_orig(), item.group, self.specs.len(), self.last_build.charge_label);
        canvas.set_scene(Scene2D::CrystalPatch { crystal: self.last_build.crystal.clone(), title, note: self.last_build.note.clone() }, reset_view);
        canvas.set_export_markers(&self.build_markers());
        canvas.set_cut_rays(&self.current_cut_rays());
        canvas.set_kinematic_overlay(self.current_kinematic_overlay());
        self.sync_inspectors_to_canvas(gallery_item, canvas);
    }

    pub fn apply_preset_pair(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.lattice_unavailable() { return; }
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.specs = preset_neighbour_pair(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn apply_preset_dislocation(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.lattice_unavailable() { return; }
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.specs = preset_shortest_dislocation(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn clear_defects(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.lattice_unavailable() { return; }
        if !self.specs.is_empty() {
            self.push_undo();
        }
        self.clear_kinematic_drag_state(false);
        self.specs.clear();
        self.request_build(gallery_item, false, false, true);
    }

    pub(crate) fn active_centre_and_angle(&self) -> (Option<&str>, f64) {
        if let Some(c) = self.centre_filter_idx.checked_sub(1).and_then(|k| self.info.centres.get(k)) {
            let want = self.selected_angle_deg as f64;
            let ang = if c.angles.iter().any(|&a| (a - want).abs() <= 0.5) {
                want
            } else {
                c.angles.first().copied().unwrap_or(want)
            };
            (Some(c.id.as_str()), ang)
        } else {
            (None, self.selected_angle_deg as f64)
        }
    }

    pub fn place_at_world(&mut self, world: [f64; 2], gallery_item: &CatalogueItemUi, backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.lattice_unavailable() || self.tool_mode == ToolMode::PanAndDrag {
            return;
        }
        match self.tool_mode {
            ToolMode::PanAndDrag => return,
            ToolMode::PlaceDisclination => {
                if self.info.centres.is_empty() {
                    return;
                }
                self.push_undo();
                self.clear_kinematic_drag_state(false);
                let (cid, ang) = self.active_centre_and_angle();
                let item = self.effective_item(gallery_item);
                let (snapped, _) = backend.snap_disclination(item, &self.info, ang, cid, world);
                self.specs.push(DefectSpec::disclination(ang, snapped));
            }
            ToolMode::PlaceDislocation => {
                self.push_undo();
                self.clear_kinematic_drag_state(false);
                if self.custom_mn == (0, 0) {
                    self.custom_mn = (1, 0);
                }
                let b = burgers_from_mn(&self.info, self.custom_mn.0, self.custom_mn.1);
                self.specs.push(DefectSpec::dislocation(b, Some(world)));
            }
            ToolMode::PlaceHalfWall => {
                self.place_half_wall_at(world);
            }
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
    }

    /// Move pin `idx` during a drag (respecting active `MotionMode` and `shift_held` overrides).
    #[cfg(test)]
    pub fn move_pin_to(&mut self, idx: usize, world: [f64; 2], gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        self.move_pin_with_mods(idx, world, false, gallery_item, backend, canvas)
    }

    /// Drag released: apply kinematic transition / parallel transport, then full build + compare.
    #[cfg(test)]
    pub fn finish_drag(&mut self, gallery_item: &CatalogueItemUi) {
        self.finish_drag_kinematics(gallery_item, None);
    }

    /// A comparison result that matches the current defects (for the `.json` summary).
    fn current_report(&self) -> Option<&EquivalentSingleReport> {
        let fresh = !self.compare_stale && !self.is_busy() && !self.last_build.preview;
        self.compare_result.as_ref().filter(|_| fresh).and_then(|r| r.as_ref().ok())
    }

    pub(crate) fn build_markers(&self) -> Vec<CanvasMarker> {
        self.specs.iter().enumerate().map(|(idx, s)| CanvasMarker {
            index: idx,
            pos: spec_world_pos(&self.info, idx, s),
            kind: match *s {
                DefectSpec::Disclination { angle_deg, .. } if angle_deg >= 0.0 => MarkerKind::DisclinationPlus,
                DefectSpec::Disclination { .. } => MarkerKind::DisclinationMinus,
                DefectSpec::Dislocation { .. } | DefectSpec::HalfWall { .. } => MarkerKind::Dislocation,
            },
            label: spec_short_label(&self.info, s),
        }).collect()
    }
}

#[path = "combine_json.rs"]
mod json;

#[path = "combine_kinematics.rs"]
mod kinematics;
pub use kinematics::MotionMode;

#[cfg(test)]
#[path = "combine_tab_tests.rs"]
mod tests;
