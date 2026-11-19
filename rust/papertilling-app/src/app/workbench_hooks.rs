//! App-side glue of the Workbench tab (`crate::workbench`): building the panel context from disjoint app fields,
//! routing sub-views to workbench views, the busy rule shared by exports and the smoke test, and keeping the S3
//! defect source (the Defects-tab crystal) in sync with the Gallery lattice.

use super::{PapertillingApp, item_key};
use crate::canvas2d::Canvas2D;
use crate::view3d::View3dState;
use crate::workbench::{ModeView, WbCtx, WorkbenchState, WorkbenchView};

impl PapertillingApp {
    /// Run `f` on the workbench with a fresh [`WbCtx`] and the shared canvas / 3D view.
    pub(crate) fn workbench_call<R>(
        &mut self,
        f: impl FnOnce(&mut WorkbenchState, &WbCtx<'_>, &mut Canvas2D, &mut View3dState) -> R,
    ) -> R {
        let item = self.gallery.active_item().clone();
        let cell = self.gallery.cell().clone();
        let cx = WbCtx {
            item: &item,
            cell: &cell,
            smoke: self.smoke_test,
            backend: &self.backend,
            defect_crystal: Some(&self.defects.last_crystal),
        };
        f(&mut self.workbench, &cx, &mut self.canvas2d, &mut self.view3d)
    }

    /// Point the workbench at `wv` (a sub-view's workbench view, if any); true when that changed it.
    pub(super) fn route_workbench_view(&mut self, wv: Option<WorkbenchView>) -> bool {
        match wv {
            Some(wv) if self.workbench.view != wv => {
                self.workbench.view = wv;
                self.workbench.dirty = true;
                true
            }
            _ => false,
        }
    }

    /// Show stage panel `view` (at sub-row `sub`, if given) through its mode sub-view. `dirty` makes
    /// `activate_nav_tab` enter the panel again even when the view did not change.
    pub(crate) fn enter_workbench(&mut self, view: WorkbenchView, sub: Option<usize>) {
        if let Some(s) = sub {
            self.workbench.select(view, s);
        } else {
            self.workbench.view = view;
            self.workbench.dirty = true;
        }
        match view.mode_view() {
            ModeView::Lattices(v) => self.select_lattices_view(v),
            ModeView::Defects(v) => self.select_defects_view(v),
        }
    }

    /// The workbench panel, or (on S3) the Defects-tab build it reads, is still running.
    pub(crate) fn workbench_busy(&self) -> bool {
        self.workbench.is_busy() || (self.workbench.view == WorkbenchView::S3Detect && self.defects.is_busy())
    }

    /// S3 reads `defects.last_crystal`: rebuild it for the Gallery's current lattice when it was built for another.
    pub(crate) fn workbench_defect_source_synced(&mut self) {
        if self.workbench.view != WorkbenchView::S3Detect {
            return;
        }
        let item = self.gallery.active_item().clone();
        let k = item_key(&item);
        if k != self.defects_seen {
            self.defects_seen = k;
            self.defects.reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
            self.defects.submit_rebuild(&item, true);
        }
    }

    /// Each Workbench frame: drain the Defects worker on S3 (quietly, the canvas belongs to the workbench) and ask
    /// the panel to rebuild when a new crystal arrived.
    pub(crate) fn workbench_pump_sources(&mut self, ctx: &egui::Context) {
        if self.workbench.view == WorkbenchView::S3Detect && self.defects.poll_worker_quiet(ctx) {
            self.workbench.source_changed();
        }
    }

    /// The Gallery lattice changed while the Workbench is shown.
    pub(super) fn workbench_lattice_changed(&mut self) {
        self.workbench_defect_source_synced();
        self.workbench_call(|wb, cx, canvas, view3d| wb.on_lattice_changed(cx, canvas, view3d));
    }
}
