//! Background worker for live minimal-lattice discovery at `N > 7` (`N in 1..=25`).

use std::sync::mpsc::{self, Receiver, Sender};

use papertilling_core::plane::live_search::{search_minimal_lattices, DiscoveredLattice};

#[derive(Clone, Debug)]
pub struct SearchJob {
    pub rev: u64,
    pub group: String,
    pub n_min: usize,
    pub n_max: usize,
    pub min_degree: usize,
    pub max_results: usize,
}

#[derive(Debug)]
pub struct SearchDone {
    pub rev: u64,
    pub group: String,
    pub result: Result<Vec<DiscoveredLattice>, String>,
}

pub struct SearchWorker {
    tx: Sender<(SearchJob, Option<egui::Context>)>,
    rx: Receiver<SearchDone>,
    pub latest_rev: u64,
    pub busy: bool,
}

impl SearchWorker {
    pub fn new() -> Self {
        let (tx, req_rx) = mpsc::channel::<(SearchJob, Option<egui::Context>)>();
        let (res_tx, rx) = mpsc::channel::<SearchDone>();
        std::thread::Builder::new()
            .name("gallery-live-search".to_string())
            .spawn(move || {
                while let Ok((mut latest_job, mut latest_ctx)) = req_rx.recv() {
                    while let Ok((newer, nctx)) = req_rx.try_recv() {
                        latest_job = newer;
                        if nctx.is_some() {
                            latest_ctx = nctx;
                        }
                    }
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        search_minimal_lattices(
                            &latest_job.group,
                            latest_job.n_min,
                            latest_job.n_max,
                            latest_job.min_degree,
                            latest_job.max_results,
                        )
                    }))
                    .unwrap_or_else(|_| Err("live search panicked".to_string()));
                    let _ = res_tx.send(SearchDone {
                        rev: latest_job.rev,
                        group: latest_job.group,
                        result: res,
                    });
                    if let Some(ctx) = latest_ctx {
                        ctx.request_repaint();
                    }
                }
            })
            .expect("spawn gallery live search worker");
        Self { tx, rx, latest_rev: 0, busy: false }
    }

    pub fn submit(
        &mut self,
        group: String,
        n_min: usize,
        n_max: usize,
        min_degree: usize,
        ctx: Option<egui::Context>,
    ) -> u64 {
        self.latest_rev = self.latest_rev.saturating_add(1);
        self.busy = true;
        let _ = self.tx.send((
            SearchJob {
                rev: self.latest_rev,
                group,
                n_min,
                n_max,
                min_degree,
                max_results: 30,
            },
            ctx,
        ));
        self.latest_rev
    }

    pub fn poll(&mut self) -> Option<SearchDone> {
        let mut latest = None;
        while let Ok(done) = self.rx.try_recv() {
            if done.rev == self.latest_rev {
                self.busy = false;
                latest = Some(done);
            }
        }
        latest
    }
}
