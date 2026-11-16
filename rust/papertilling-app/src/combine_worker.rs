//! Background workers for the Combine tab: the egui thread only submits jobs and polls results, so
//! building a crystal or comparing cores never freezes the window. Each worker keeps the newest job
//! only (a generation counter drops superseded ones) and wakes the UI when a result is ready.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use papertilling_core::defect::combine_api::{SingleDefectCache, equivalent_single_cached};
use papertilling_core::defect::combine_types::EquivalentSingleReport;
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::types::BuiltCrystal;

use crate::backend::CatalogueItemUi;
use crate::live_combine::{CombineBuildOptions, CombineBuildOutput, CombineLatticeInfo, build_combined_crystal};

type Shared<T> = Arc<Mutex<Option<T>>>;

/// A thread running `run(state, job)` on the newest submitted job only.
pub struct LatestWorker<J, R> {
    jobs: Sender<(u64, J)>,
    results: Receiver<(u64, Result<R, String>)>,
    latest: Arc<AtomicU64>,
    repaint: Shared<egui::Context>,
    submitted: u64,
    received: u64,
}

impl<J, R> Drop for LatestWorker<J, R> {
    fn drop(&mut self) {
        // Queued jobs are no longer useful once the tab/state is gone. Never join here: a join on
        // the UI thread would reintroduce the stall this worker is intended to prevent.
        self.latest.store(u64::MAX, Ordering::Release);
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| String::from("unknown panic"))
}

impl<J: Send + 'static, R: Send + 'static> LatestWorker<J, R> {
    pub fn spawn<S: Send + 'static>(name: &str, mut state: S, mut run: impl FnMut(&mut S, J) -> R + Send + 'static) -> Self {
        let (job_tx, job_rx) = channel::<(u64, J)>();
        let (res_tx, res_rx) = channel();
        let latest = Arc::new(AtomicU64::new(0));
        let repaint: Shared<egui::Context> = Arc::new(Mutex::new(None));
        let (latest_w, repaint_w) = (Arc::clone(&latest), Arc::clone(&repaint));
        thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(mut job) = job_rx.recv() {
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    if job.0 < latest_w.load(Ordering::Acquire) {
                        continue;
                    }
                    let out = catch_unwind(AssertUnwindSafe(|| run(&mut state, job.1)))
                        .map_err(|p| format!("internal error (recovered): {}", panic_text(p.as_ref())));
                    if res_tx.send((job.0, out)).is_err() {
                        break;
                    }
                    if let Some(ctx) = repaint_w.lock().ok().and_then(|g| g.clone()) {
                        ctx.request_repaint();
                    }
                }
            })
            .expect("spawn Combine worker thread");
        Self { jobs: job_tx, results: res_rx, latest, repaint, submitted: 0, received: 0 }
    }

    /// Context to wake when a result lands (set once from the UI thread).
    pub fn set_repaint_ctx(&self, ctx: &egui::Context) {
        if let Ok(mut g) = self.repaint.lock()
            && g.is_none()
        {
            *g = Some(ctx.clone());
        }
    }

    /// Queue `job`, superseding every earlier one; returns its generation.
    pub fn submit(&mut self, job: J) -> u64 {
        self.submitted += 1;
        self.latest.store(self.submitted, Ordering::Release);
        let _ = self.jobs.send((self.submitted, job));
        self.submitted
    }

    /// Drop every queued or running job: their results will be ignored.
    pub fn cancel(&mut self) {
        self.submitted += 1;
        self.latest.store(self.submitted, Ordering::Release);
        self.received = self.submitted;
    }

    /// True while the newest job has not returned.
    pub fn is_busy(&self) -> bool {
        self.received < self.submitted
    }

    /// The result of the newest job, once (older results are discarded).
    pub fn poll(&mut self) -> Option<Result<R, String>> {
        let mut newest = None;
        while let Ok((gen_id, res)) = self.results.try_recv() {
            if gen_id == self.submitted && self.received < gen_id {
                self.received = gen_id;
                newest = Some(res);
            }
        }
        newest
    }

    /// Block until the newest job returns (tests).
    #[cfg(test)]
    pub fn wait(&mut self, timeout: Duration) -> Option<Result<R, String>> {
        let t0 = Instant::now();
        while self.is_busy() && t0.elapsed() < timeout {
            match self.results.recv_timeout(Duration::from_millis(5)) {
                Ok((gen_id, res)) if gen_id == self.submitted => {
                    self.received = gen_id;
                    return Some(res);
                }
                _ => {}
            }
        }
        None
    }
}

/// Build the combined crystal (`preview`: cheap picture during a drag, see `build_combined_crystal`).
pub struct BuildJob {
    pub item: CatalogueItemUi,
    pub info: Arc<CombineLatticeInfo>,
    pub specs: Vec<DefectSpec>,
    pub opts: CombineBuildOptions,
    pub preview: bool,
}

/// Compare a built configuration against the single defect of the same charge.
pub struct CompareJob {
    pub item: CatalogueItemUi,
    pub info: Arc<CombineLatticeInfo>,
    pub specs: Vec<DefectSpec>,
    pub crystal: BuiltCrystal,
    pub core_radius: usize,
    pub grid: usize,
}

pub type CompareOutput = (Result<EquivalentSingleReport, String>, f64, (usize, usize));

pub fn spawn_lattice_worker() -> LatestWorker<CatalogueItemUi, Arc<CombineLatticeInfo>> {
    LatestWorker::spawn("combine-lattice", (), |_: &mut (), item: CatalogueItemUi| {
        Arc::new(crate::live_combine::combine_lattice_info(&item))
    })
}

pub fn spawn_build_worker() -> LatestWorker<BuildJob, CombineBuildOutput> {
    LatestWorker::spawn("combine-build", (), |_: &mut (), job: BuildJob| {
        build_combined_crystal(&job.item, &job.info, &job.specs, &job.opts, job.preview)
    })
}

/// Comparison worker; its `SingleDefectCache` lives on the worker thread (no locking).
pub fn spawn_compare_worker() -> LatestWorker<CompareJob, CompareOutput> {
    LatestWorker::spawn("combine-compare", SingleDefectCache::default(), |cache: &mut SingleDefectCache, job: CompareJob| {
        let t0 = Instant::now();
        let rep = equivalent_single_cached(
            &job.item.group, &job.info.cell, job.item.frame, &job.specs, &job.crystal,
            job.core_radius, job.grid, true, cache,
        );
        (rep, t0.elapsed().as_secs_f64() * 1000.0, cache.stats())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_worker_drops_superseded_jobs_and_recovers_from_panics() {
        let mut w = LatestWorker::spawn("test", 0_u32, |runs: &mut u32, x: u32| {
            *runs += 1;
            thread::sleep(Duration::from_millis(30));
            assert!(x != 13, "boom");
            (x, *runs)
        });
        for x in 1..=5 {
            w.submit(x);
        }
        assert!(w.is_busy());
        let (x, runs) = w.wait(Duration::from_secs(5)).expect("result").expect("no panic");
        assert_eq!(x, 5, "only the newest job's result is delivered");
        assert!(runs <= 2, "superseded jobs are skipped, ran {runs}");
        assert!(!w.is_busy() && w.poll().is_none());

        w.submit(13);
        let err = w.wait(Duration::from_secs(5)).expect("result").expect_err("panic becomes an error");
        assert!(err.contains("boom"));
        w.submit(7);
        assert_eq!(w.wait(Duration::from_secs(5)).unwrap().unwrap().0, 7, "worker survives a panic");

        w.submit(8);
        w.cancel();
        assert!(!w.is_busy());
        thread::sleep(Duration::from_millis(80));
        assert!(w.poll().is_none(), "cancelled result is ignored");
    }
}
