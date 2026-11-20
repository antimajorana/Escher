//! Surface builds from the left-panel controls (target, custom `(h, k)`, centre mode, auto size) run off the UI
//! thread: the slowest offered size takes about 2 s (sphere (8, 8) on a p6 lattice). A change is debounced
//! ([`DEBOUNCE`]) so dragging a size value queues one build, and a `LatestWorker` keeps only the newest job.
//! `activate_and_sync` (tab switch, lattice change, verdict-table clicks, tests, smoke) stays synchronous and
//! cancels a pending background build.

use std::time::{Duration, Instant};

use papertilling_core::types::{SurfaceCheckReport, SurfaceMesh};

use super::SurfacesTabState;
use super::bc::{self, BcControls, BcNotes};
use crate::backend::{CatalogueItemUi, SurfaceBuildParamsUi, SurfaceTargetUi};
use crate::canvas2d::Canvas2D;
use crate::combine_worker::LatestWorker;
use crate::live_ops;
use crate::view3d::View3dState;

pub const DEBOUNCE: Duration = Duration::from_millis(120);

pub(super) type Built = Result<(SurfaceMesh, SurfaceCheckReport, BcNotes), String>;

pub(super) struct SurfaceJob {
    pub item: CatalogueItemUi,
    pub target: SurfaceTargetUi,
    pub params: SurfaceBuildParamsUi,
    pub auto_size: bool,
    /// The boundary-condition options to build from (`None`: the table sizes or the legacy surface builders).
    pub bc: Option<BcControls>,
}

/// Build the surface of `job` (as `activate_and_sync` always did).
pub(super) fn run_job(job: &SurfaceJob) -> Built {
    if let Some(ctl) = &job.bc {
        return bc::build_bc(&job.item, job.target, ctl);
    }
    let built = if job.auto_size {
        live_ops::build_surface_mesh_auto(&job.item, job.target, job.params.centre_mode, job.params.genus_g)
    } else {
        live_ops::build_surface_mesh(&job.item, job.target, &job.params)
    };
    built.map(|(mesh, rep)| {
        let notes = bc::notes_for_built(&mesh, job.target);
        (mesh, rep, notes)
    })
}

pub(super) type SurfaceWorker = LatestWorker<SurfaceJob, (String, Built)>;

fn spawn() -> SurfaceWorker {
    LatestWorker::spawn("surface-build", (), |_: &mut (), job: SurfaceJob| (job.item.group.clone(), run_job(&job)))
}

impl SurfacesTabState {
    /// Queue a build for the current controls (debounced, off the UI thread).
    pub fn request_build(&mut self, item: &CatalogueItemUi, ctx: &egui::Context) {
        let job = self.prepare_build(item);
        self.queued_build = Some((Instant::now(), job));
        ctx.request_repaint_after(DEBOUNCE);
    }

    /// A control change is waiting for its debounce or its build.
    pub fn is_building(&self) -> bool {
        self.queued_build.is_some() || self.builder.as_ref().is_some_and(LatestWorker::is_busy)
    }

    pub(super) fn cancel_build(&mut self) {
        self.queued_build = None;
        if let Some(w) = self.builder.as_mut() {
            w.cancel();
        }
    }

    /// Submit a debounced job once it is due, and apply a finished build. Returns true when a build was applied.
    pub(super) fn pump_build(&mut self, ctx: &egui::Context, canvas: &mut Canvas2D, view3d: &mut View3dState) -> bool {
        if let Some((t0, _)) = &self.queued_build {
            let waited = t0.elapsed();
            if waited >= DEBOUNCE {
                if let Some((_, job)) = self.queued_build.take() {
                    let w = self.builder.get_or_insert_with(spawn);
                    w.set_repaint_ctx(ctx);
                    w.submit(job);
                }
            } else {
                ctx.request_repaint_after(DEBOUNCE - waited);
            }
        }
        let Some(res) = self.builder.as_mut().and_then(LatestWorker::poll) else { return false };
        if self.queued_build.is_some() {
            // superseded by a request still in its debounce: never show this mesh
            return false;
        }
        match res {
            Ok((group, built)) => self.apply_built(built, &group, canvas, view3d),
            Err(e) => self.apply_built(Err(e), "", canvas, view3d),
        }
        true
    }

    /// Block until the queued or running background build is applied (tests).
    #[cfg(test)]
    pub(crate) fn wait_build(&mut self, ctx: &egui::Context, canvas: &mut Canvas2D, view3d: &mut View3dState) -> bool {
        let t0 = Instant::now();
        while self.is_building() && t0.elapsed().as_secs_f64() < 30.0 {
            if self.pump_build(ctx, canvas, view3d) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }
}
