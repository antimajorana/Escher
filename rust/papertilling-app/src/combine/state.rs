use std::sync::Arc;

use papertilling_core::defect::combine_types::EquivalentSingleReport;
use papertilling_core::defect::kinematics::AbsorptionPreview;
use papertilling_core::defect::surgery::DefectSpec;

use super::kinematics::{MotionMode, PendingDiscStep};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::combine_half_wall_ui::CombineHalfWallState;
use crate::combine_worker::{
    BuildJob, BuildWorkerOutput, CombineSheet3dBundle, CompareJob, CompareOutput, LatestWorker,
    build_sheet_bundle_sync, empty_crystal, spawn_build_worker, spawn_compare_worker,
    spawn_lattice_worker,
};
use crate::core_panel::CorePanelState;
use crate::defects_inspectors::DefectInspectors;
use crate::live_combine::{
    CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo,
    build_combined_crystal, preset_neighbour_pair,
};
use crate::mesh3d::{DefectViewMode, Sheet3dViewOpts};

/// Manual patch radius range (slider and `.json` import): covers Python's default radii.
pub(crate) const MANUAL_R_MIN: f64 = 2.4;
pub(crate) const MANUAL_R_MAX: f64 = 30.0;

/// The build options of a fresh Combine tab (also what a curated preset restores): fast relaxed drawing, automatic
/// radius (manual radius `r_auto_base` when the user unticks it), core radius 2.
pub(crate) fn default_build_opts(r_auto_base: f64) -> CombineBuildOptions {
    CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: true,
        manual_radius: r_auto_base,
        core_radius: 2,
    }
}

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
    pub(crate) pending_disc_step: Option<PendingDiscStep>,
    pub(crate) pending_absorb: Option<AbsorptionPreview>,
    pub(crate) undo_stack: Vec<Vec<DefectSpec>>,
    pub(crate) redo_stack: Vec<Vec<DefectSpec>>,
    pub selected_pin: Option<usize>,
    pub popover_open: bool,
    pub(crate) cut_dragging: bool,
    pub(crate) cut_drag_anchor: Option<(usize, Vec<DefectSpec>)>,
    pub(crate) cut_edit_anchor: Option<(usize, Vec<DefectSpec>)>,
    #[cfg(test)]
    pub(crate) last_popover_rect: Option<egui::Rect>,
    pub view_mode: DefectViewMode,
    pub sheet_opts: Sheet3dViewOpts,
    pub(crate) last_build: CombineBuildOutput,
    pub(crate) last_sheet3d: CombineSheet3dBundle,
    pub(crate) view3d_dirty: bool,
    pub compare_result: Option<Result<EquivalentSingleReport, String>>,
    /// `compare_result` belongs to an earlier configuration (a newer comparison is pending or due).
    pub(crate) compare_stale: bool,
    pub(crate) compare_ms: f64,
    #[cfg(test)]
    pub(crate) cache_stats: (usize, usize),
    pub(super) compare_after_build: bool,
    pub(super) reset_view_on_build: bool,
    pub(super) canvas_dirty: bool,
    pub(super) preview_queued: bool,
    pub(super) preview_dirty: bool,
    pub(crate) dragging: bool,
    pub(crate) last_hover_world: Option<[f64; 2]>,
    pub json_status: String,
    pub(crate) init_error: Option<String>,
    pub core_panel: CorePanelState,
    pub half_wall: CombineHalfWallState,
    pub inspectors: DefectInspectors,
    pub(super) builder: LatestWorker<BuildJob, BuildWorkerOutput>,
    pub(super) comparer: LatestWorker<CompareJob, CompareOutput>,
    pub(super) initializer: LatestWorker<CatalogueItemUi, Arc<CombineLatticeInfo>>,
}

impl CombineTabState {
    pub fn new(gallery_item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let info = Arc::new(backend.combine_lattice_info(gallery_item));
        let specs = preset_neighbour_pair(&info);
        let opts = default_build_opts(info.r_auto_base);
        let sheet_opts = Sheet3dViewOpts::default();
        let last_build = build_combined_crystal(gallery_item, &info, &[], &opts, false);
        let last_sheet3d = build_sheet_bundle_sync(
            &info,
            &last_build.crystal,
            last_build.applied_radius,
            sheet_opts.kappa_b,
        );
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
            redo_stack: Vec::new(),
            selected_pin: None,
            popover_open: false,
            cut_dragging: false,
            cut_drag_anchor: None,
            cut_edit_anchor: None,
            #[cfg(test)]
            last_popover_rect: None,
            view_mode: DefectViewMode::Planar2D,
            sheet_opts,
            last_build,
            last_sheet3d,
            view3d_dirty: true,
            compare_result: None,
            compare_stale: false,
            compare_ms: 0.0,
            #[cfg(test)]
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
        self.initializer.is_busy()
            || self.builder.is_busy()
            || self.comparer.is_busy()
            || self.compare_after_build
    }

    pub(crate) fn is_building(&self) -> bool {
        self.initializer.is_busy() || self.builder.is_busy()
    }

    pub(super) fn lattice_unavailable(&self) -> bool {
        self.initializer.is_busy() || self.init_error.is_some()
    }

    pub(crate) fn is_initializing(&self) -> bool {
        self.initializer.is_busy()
    }

    pub(crate) fn is_comparing(&self) -> bool {
        self.comparer.is_busy() || self.compare_after_build
    }

    pub(super) fn reset_lattice_defaults(&mut self, gallery_item: &CatalogueItemUi, info: CombineLatticeInfo) {
        self.clear_kinematic_drag_state(true);
        self.clear_selection();
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
        self.last_sheet3d = build_sheet_bundle_sync(
            &self.info,
            &self.last_build.crystal,
            self.last_build.applied_radius,
            self.sheet_opts.kappa_b,
        );
        self.canvas_dirty = true;
        self.view3d_dirty = true;
        self.compare_result = None;
        self.comparer.cancel();
    }

    pub fn reload_for_item(
        &mut self,
        gallery_item: &CatalogueItemUi,
        _backend: &AppBackend,
        canvas: &mut Canvas2D,
        sync_canvas: bool,
    ) {
        self.clear_kinematic_drag_state(true);
        self.clear_selection();
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

    /// A comparison result that matches the current defects (for the `.json` summary).
    pub(super) fn current_report(&self) -> Option<&EquivalentSingleReport> {
        let fresh = !self.compare_stale
            && !self.is_busy()
            && !self.last_build.preview
            && self.last_build.refused.is_none();
        self.compare_result.as_ref().filter(|_| fresh).and_then(|r| r.as_ref().ok())
    }
}
