use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

use super::CombineTabState;
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::combine_worker::{BuildJob, BuildWorkerOutput, CompareJob};
use crate::live_combine::{CombineLatticeInfo, preset_neighbour_pair};
use crate::mesh3d::{GpuMeshData, mesh_from_sheet3d_with_spectral, sheet3d_caption};
use crate::view3d::View3dState;

impl CombineTabState {
    /// Submit a build of the current defects; `then_compare`: compare once it lands (final builds only).
    pub(crate) fn request_build(
        &mut self,
        gallery_item: &CatalogueItemUi,
        preview: bool,
        then_compare: bool,
        reset_view: bool,
    ) {
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

    pub(super) fn submit_compare(&mut self, gallery_item: &CatalogueItemUi) {
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
        if self.lattice_unavailable() {
            return;
        }
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

    pub(super) fn set_compare_grid(&mut self, grid: usize) {
        self.grid = grid;
        self.comparer.cancel();
        self.compare_after_build = false;
        self.compare_stale = self.compare_result.is_some();
    }

    /// Collect finished jobs (call every frame while the tab is shown); `ctx` lets workers wake the UI.
    /// Returns `true` when an initializer or build result landed.
    pub fn poll(
        &mut self,
        ctx: Option<&egui::Context>,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) -> bool {
        let mut landed = false;
        if let Some(ctx) = ctx {
            self.initializer.set_repaint_ctx(ctx);
            self.builder.set_repaint_ctx(ctx);
            self.comparer.set_repaint_ctx(ctx);
        }
        if let Some(result) = self.initializer.poll() {
            self.take_init(result, gallery_item);
            landed = true;
        }
        if let Some(res) = self.builder.poll() {
            self.take_build(res);
            self.on_combine_build_landed(gallery_item);
            landed = true;
        }
        if self.compare_after_build && !self.builder.is_busy() && !self.last_build.preview {
            self.compare_after_build = false;
            self.submit_compare(gallery_item);
        }
        if let Some(res) = self.comparer.poll() {
            match res {
                Ok((rep, ms, _stats)) => {
                    self.compare_result = Some(rep);
                    (self.compare_ms, self.compare_stale) = (ms, false);
                    #[cfg(test)]
                    {
                        self.cache_stats = _stats;
                    }
                }
                Err(err) => {
                    self.compare_result = Some(Err(err));
                    (self.compare_ms, self.compare_stale) = (0.0, false);
                }
            }
        }
        if self.canvas_dirty
            && let Some(canvas) = canvas
        {
            let reset = std::mem::take(&mut self.reset_view_on_build) && !self.dragging && !self.cut_dragging;
            self.sync_to_canvas(gallery_item, canvas, reset);
            self.canvas_dirty = false;
        }
        if (self.dragging || self.cut_dragging) && self.preview_dirty && !self.builder.is_busy() {
            self.request_build(gallery_item, true, false, false);
        }
        landed
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

    #[cfg(test)]
    pub fn request_sheet3d_only(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() {
            return;
        }
        // A stiffness edit must not supersede a newer geometry job with the last completed crystal.
        if self.builder.is_busy() || self.last_build.preview {
            let compare = self.compare_after_build || self.auto_compare;
            self.request_build(gallery_item, self.dragging || self.cut_dragging, compare, false);
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
        let title = format!(
            "{} ({}) — {} defect(s) | {}",
            item.display_with_orig(),
            item.group,
            self.specs.len(),
            self.last_build.charge_label
        );
        let (c, s) = if self.sheet_opts.show_flat_baseline {
            (&self.last_sheet3d.flat_crystal, &self.last_sheet3d.flat_sheet)
        } else {
            (&self.last_build.crystal, &self.last_sheet3d.sheet)
        };
        let gpu = mesh_from_sheet3d_with_spectral(c, s, &self.sheet_opts, None);
        let legend = crate::mesh3d::color_scales::sheet_color_legend(c, s, &self.sheet_opts, None);
        let mut cap = sheet3d_caption(&title, s, &self.sheet_opts);
        if let Some(reason) = &self.last_build.refused {
            cap = if s.xyz.is_empty() {
                format!("3D sheet unavailable: {reason}")
            } else {
                format!("Previous 3D sheet | Build refused: {reason}")
            };
        } else if self.is_building() {
            cap.push_str(" | previous result — stale");
        }
        view3d.set_mesh(gpu, cap);
        view3d.set_legend(legend);
        self.view3d_dirty = false;
    }

    pub(super) fn take_build(&mut self, res: Result<BuildWorkerOutput, String>) {
        match res {
            Ok((mut out, sheet3d)) => {
                let prev_was_standin = self.last_build.refused.is_some()
                    && self.last_build.note.ends_with(crate::live_combine::cuts::REFUSAL_STANDIN_SUFFIX);
                let preserve_prev = out.refused.is_some()
                    && !prev_was_standin
                    && !self.last_build.crystal.pos.is_empty()
                    && self.last_build.lattice_key == out.lattice_key;
                if preserve_prev {
                    self.last_build.refused = out.refused;
                    self.last_build.note = crate::live_combine::cuts::with_refusal_canvas_notice(out.note, true);
                    self.last_build.charge_label = out.charge_label;
                    self.last_build.charge_detail = out.charge_detail;
                    self.last_build.elapsed_ms = out.elapsed_ms;
                    self.last_build.preview = out.preview;
                    self.last_build.applied_cuts = out.applied_cuts;
                    self.last_build.coalesced_count = 0;
                    self.last_build.core_summary = out.core_summary;
                    self.last_build.local_cores = out.local_cores;
                } else {
                    if out.refused.is_some() && !out.crystal.pos.is_empty() {
                        out.note = crate::live_combine::cuts::with_refusal_canvas_notice(out.note, false);
                    }
                    self.last_build = out;
                    self.last_sheet3d = sheet3d;
                }
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

    pub(crate) fn take_init(
        &mut self,
        result: Result<Arc<CombineLatticeInfo>, String>,
        gallery_item: &CatalogueItemUi,
    ) {
        match result {
            Ok(info) => {
                self.reset_lattice_defaults(gallery_item, (*info).clone());
                self.clear_selection();
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
}
