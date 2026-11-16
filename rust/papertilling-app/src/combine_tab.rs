use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

use papertilling_core::defect::combine_types::{
    ConfigOptionsJson, ConfigSummaryJson, EquivalentSingleReport, NotebookConfigJson,
};
use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasMarker, MarkerKind, Scene2D};
use crate::combine_worker::{
    BuildJob, CompareJob, CompareOutput, LatestWorker, spawn_build_worker, spawn_compare_worker, spawn_lattice_worker,
};
use crate::core_panel::CorePanelState;
use crate::live_combine::{
    CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo, build_combined_crystal,
    burgers_from_mn, preset_neighbour_pair, preset_shortest_dislocation, spec_short_label, spec_world_pos,
};

/// Manual patch radius range (slider and `.json` import): covers Python's default radii.
pub(crate) const MANUAL_R_MIN: f64 = 2.4;
pub(crate) const MANUAL_R_MAX: f64 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    PanAndDrag,
    PlaceDisclination,
    PlaceDislocation,
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
    pub(crate) last_build: CombineBuildOutput,
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
    pub core_panel: CorePanelState,
    builder: LatestWorker<BuildJob, CombineBuildOutput>,
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
        let last_build = build_combined_crystal(gallery_item, &info, &[], &opts, false);
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
            last_build,
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
            core_panel: CorePanelState::default(),
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

    pub(crate) fn is_initializing(&self) -> bool { self.initializer.is_busy() }

    pub(crate) fn is_comparing(&self) -> bool {
        self.comparer.is_busy() || self.compare_after_build
    }

    fn reset_lattice_defaults(&mut self, gallery_item: &CatalogueItemUi, info: CombineLatticeInfo) {
        self.selected_angle_deg = info.allowed_angles.first().copied().unwrap_or(90);
        self.centre_filter_idx = 0;
        self.selected_disloc_idx = 0;
        self.custom_mn = info.dislocations.first().map_or((1, 0), |d| d.burgers_mn);
        self.opts.manual_radius = info.r_auto_base;
        self.info = Arc::new(info);
        // Never show the previous lattice while the new build runs: a defect-free disk is instant.
        self.last_build = build_combined_crystal(gallery_item, &self.info, &[], &self.opts, false);
        self.canvas_dirty = true;
        self.compare_result = None;
        self.comparer.cancel();
    }

    pub fn reload_for_item(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, canvas: &mut Canvas2D, sync_canvas: bool) {
        self.override_item = None;
        self.builder.cancel();
        self.comparer.cancel();
        self.compare_after_build = false;
        self.compare_result = None;
        self.specs.clear();
        self.dragging = false;
        self.initializer.submit(gallery_item.clone());
        let c = &mut self.last_build.crystal;
        c.pos.clear();
        c.ref_pos.clear();
        c.adj.clear();
        c.edges.clear();
        c.basis_idx.clear();
        c.cone_sites.clear();
        c.side.clear();
        self.last_build.charge_label = String::from("Preparing lattice…");
        self.last_build.note = String::from("Building the defect catalogue in the background");
        self.canvas_dirty = true;
        self.reset_view_on_build = true;
        if sync_canvas {
            self.sync_to_canvas(gallery_item, canvas, true);
        }
    }

    /// Submit a build of the current defects; `then_compare`: compare once it lands (final builds only).
    pub(crate) fn request_build(&mut self, gallery_item: &CatalogueItemUi, preview: bool, then_compare: bool, reset_view: bool) {
        if self.initializer.is_busy() {
            return;
        }
        // Let an in-flight preview finish before submitting the next one. Invalidating it on every
        // mouse move starves the canvas when builds take longer than a pointer frame.
        if preview && self.preview_queued && self.builder.is_busy() {
            self.preview_dirty = true;
            return;
        }
        let item = self.effective_item(gallery_item).clone();
        let job = BuildJob { item, info: Arc::clone(&self.info), specs: self.specs.clone(), opts: self.opts.clone(), preview };
        self.builder.submit(job);
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
        if self.initializer.is_busy() { return; }
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
    pub fn poll(&mut self, ctx: Option<&egui::Context>, gallery_item: &CatalogueItemUi, canvas: Option<&mut Canvas2D>) {
        if let Some(ctx) = ctx {
            self.initializer.set_repaint_ctx(ctx);
            self.builder.set_repaint_ctx(ctx);
            self.comparer.set_repaint_ctx(ctx);
        }
        if let Some(result) = self.initializer.poll() { self.take_init(result, gallery_item); }
        if let Some(res) = self.builder.poll() {
            self.take_build(res);
        }
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

    fn take_build(&mut self, res: Result<CombineBuildOutput, String>) {
        match res {
            Ok(out) => self.last_build = out,
            Err(err) => {
                self.last_build.note = err.clone();
                self.last_build.charge_label = String::from("Build failed");
                self.last_build.charge_detail = err;
                self.compare_after_build = false;
            }
        }
        self.canvas_dirty = true;
    }

    fn take_init(&mut self, result: Result<Arc<CombineLatticeInfo>, String>, gallery_item: &CatalogueItemUi) {
        match result {
            Ok(info) => {
                self.reset_lattice_defaults(gallery_item, (*info).clone());
                self.specs = preset_neighbour_pair(&self.info);
                self.request_build(gallery_item, false, self.auto_compare, true);
            }
            Err(err) => self.last_build.note = err,
        }
    }

    pub fn sync_to_canvas(&self, gallery_item: &CatalogueItemUi, canvas: &mut Canvas2D, reset_view: bool) {
        let item = self.effective_item(gallery_item);
        let title = format!("{} ({}) — {} defect(s) | {}", item.label, item.group, self.specs.len(), self.last_build.charge_label);
        canvas.set_scene(Scene2D::CrystalPatch { crystal: self.last_build.crystal.clone(), title, note: self.last_build.note.clone() }, reset_view);
        canvas.set_export_markers(&self.build_markers());
    }

    pub fn apply_preset_pair(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.initializer.is_busy() { return; }
        self.specs = preset_neighbour_pair(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn apply_preset_dislocation(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
        if self.initializer.is_busy() { return; }
        self.specs = preset_shortest_dislocation(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn clear_defects(&mut self, gallery_item: &CatalogueItemUi, _backend: &AppBackend, _canvas: &mut Canvas2D) {
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
        if self.initializer.is_busy() { return; }
        let item = self.effective_item(gallery_item);
        match self.tool_mode {
            ToolMode::PanAndDrag => return,
            ToolMode::PlaceDisclination => {
                if self.info.centres.is_empty() {
                    return;
                }
                let (cid, ang) = self.active_centre_and_angle();
                let (snapped, _) = backend.snap_disclination(item, &self.info, ang, cid, world);
                self.specs.push(DefectSpec::disclination(ang, snapped));
            }
            ToolMode::PlaceDislocation => {
                if self.custom_mn == (0, 0) {
                    self.custom_mn = (1, 0);
                }
                let b = burgers_from_mn(&self.info, self.custom_mn.0, self.custom_mn.1);
                self.specs.push(DefectSpec::dislocation(b, Some(world)));
            }
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
    }

    /// Move pin `idx` during a drag: cheap preview build; `finish_drag` does the full one.
    pub fn move_pin_to(&mut self, idx: usize, world: [f64; 2], gallery_item: &CatalogueItemUi, backend: &AppBackend, _canvas: &mut Canvas2D) -> bool {
        if self.initializer.is_busy() { return false; }
        let item = self.effective_item(gallery_item);
        let Some(old_pos) = self.specs.get(idx).map(|s| spec_world_pos(&self.info, idx, s)) else {
            return false;
        };
        let new_spec = match self.specs[idx] {
            DefectSpec::Disclination { angle_deg, cut, .. } => {
                let (snapped, _) = backend.snap_disclination(item, &self.info, angle_deg, None, world);
                if (snapped[0] - old_pos[0]).hypot(snapped[1] - old_pos[1]) < 1e-6 {
                    return false;
                }
                DefectSpec::Disclination { angle_deg, at: snapped, cut }
            }
            DefectSpec::Dislocation { b, cut, .. } => {
                if (world[0] - old_pos[0]).hypot(world[1] - old_pos[1]) < 1e-4 {
                    return false;
                }
                DefectSpec::Dislocation { b, at: Some(world), cut }
            }
        };
        self.specs[idx] = new_spec;
        self.dragging = true;
        self.request_build(gallery_item, true, false, false);
        true
    }

    /// Drag released: full build, then compare when automatic comparison is on.
    pub fn finish_drag(&mut self, gallery_item: &CatalogueItemUi) {
        if std::mem::take(&mut self.dragging) {
            self.request_build(gallery_item, false, self.auto_compare, false);
        }
    }

    /// A comparison result that matches the current defects (for the `.json` summary).
    fn current_report(&self) -> Option<&EquivalentSingleReport> {
        let fresh = !self.compare_stale && !self.is_busy() && !self.last_build.preview;
        self.compare_result.as_ref().filter(|_| fresh).and_then(|r| r.as_ref().ok())
    }

    pub fn to_notebook_json(&self, gallery_item: &CatalogueItemUi) -> NotebookConfigJson {
        let item = self.effective_item(gallery_item);
        let summary = self.current_report().map(|rep| ConfigSummaryJson {
            charge: rep.charge.label.clone(),
            kind: rep.kind.clone(),
            equivalent: rep.equivalent,
            matches: rep.matches.clone(),
            verdict: rep.verdict.clone(),
        });
        NotebookConfigJson {
            format: String::from("papertilling.config.v1"),
            lattice_label: item.label.clone(),
            group: item.group.clone(),
            recipe: item.recipe.clone(),
            frame: item.frame,
            defects: self.specs.clone(),
            options: ConfigOptionsJson {
                r_patch: (!self.opts.auto_radius).then_some(self.opts.manual_radius),
                picture: if self.opts.draw_mode == CombineDrawMode::Raw { "raw" } else { "relaxed" }.into(),
                grid: self.grid,
                radius: self.opts.core_radius,
            },
            summary,
        }
    }

    pub fn load_notebook_json(&mut self, cfg: &NotebookConfigJson, backend: &AppBackend, canvas: &mut Canvas2D) -> Result<String, String> {
        if cfg.format != papertilling_core::defect::combine_types::CONFIG_FORMAT {
            return Err(format!("unsupported configuration format {:?}", cfg.format));
        }
        let imported = backend.item_from_notebook_json(cfg)?;
        self.initializer.cancel();
        self.reset_lattice_defaults(&imported, backend.combine_lattice_info(&imported));
        let o = &cfg.options;
        self.opts.auto_radius = o.r_patch.is_none();
        self.opts.manual_radius = o.r_patch.unwrap_or(self.info.r_auto_base).clamp(MANUAL_R_MIN, MANUAL_R_MAX);
        self.opts.draw_mode = if o.picture == "raw" { CombineDrawMode::Raw } else { CombineDrawMode::FastRelaxed };
        self.grid = o.grid.clamp(1, 3);
        self.opts.core_radius = o.radius.clamp(1, 4);
        self.specs = cfg.defects.clone();
        let mut clamped = Vec::new();
        if o.r_patch.is_some_and(|r| (r - self.opts.manual_radius).abs() > 1e-9) {
            clamped.push(format!("r_patch {:?} -> {}", o.r_patch.unwrap_or_default(), self.opts.manual_radius));
        }
        if o.grid != self.grid {
            clamped.push(format!("grid {} -> {}", o.grid, self.grid));
        }
        if o.radius != self.opts.core_radius {
            clamped.push(format!("radius {} -> {}", o.radius, self.opts.core_radius));
        }
        let mut msg = format!("Imported {} ({}) with {} defect(s)", imported.label, imported.group, self.specs.len());
        if !clamped.is_empty() {
            msg.push_str(&format!(" (outside the app's range: {})", clamped.join(", ")));
        }
        self.override_item = Some(imported.clone());
        self.request_build(&imported, false, self.auto_compare || cfg.summary.is_some(), true);
        self.sync_to_canvas(&imported, canvas, true);
        self.json_status = msg.clone();
        Ok(msg)
    }

    pub fn export_json_dialog(&mut self, gallery_item: &CatalogueItemUi) {
        if self.specs.is_empty() || self.initializer.is_busy() {
            self.json_status = String::from("Place at least one defect before exporting a Notebook 3 configuration.");
            return;
        }
        let dlg = rfd::FileDialog::new().set_title("Export Notebook 3 defect configuration (.json)").set_file_name("defect_config.json").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.save_file() {
            let cfg = self.to_notebook_json(gallery_item);
            let partial = if cfg.summary.is_none() && !self.specs.is_empty() { " (no summary: comparison not finished)" } else { "" };
            let res = cfg.to_json_pretty().and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()));
            self.json_status = res.map_or_else(|e| format!("JSON export error: {e}"), |()| format!("Exported {}{partial}", path.file_name().and_then(|s| s.to_str()).unwrap_or("config.json")));
        }
    }

    pub fn import_json_dialog(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        let dlg = rfd::FileDialog::new().set_title("Import Notebook 3 defect configuration (.json)").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.pick_file()
            && let Err(err) = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|raw| NotebookConfigJson::from_json_str(&raw)).and_then(|cfg| self.load_notebook_json(&cfg, backend, canvas))
        {
            self.json_status = format!("JSON import error: {err}");
        }
    }

    pub(crate) fn build_markers(&self) -> Vec<CanvasMarker> {
        self.specs.iter().enumerate().map(|(idx, s)| CanvasMarker {
            index: idx,
            pos: spec_world_pos(&self.info, idx, s),
            kind: match *s {
                DefectSpec::Disclination { angle_deg, .. } if angle_deg >= 0.0 => MarkerKind::DisclinationPlus,
                DefectSpec::Disclination { .. } => MarkerKind::DisclinationMinus,
                DefectSpec::Dislocation { .. } => MarkerKind::Dislocation,
            },
            label: spec_short_label(&self.info, s),
        }).collect()
    }
}

#[cfg(test)]
#[path = "combine_tab_tests.rs"]
mod tests;
