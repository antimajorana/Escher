//! Slider moves of slow families (one move > `LIVE_MOVE_SECS`, e.g. 0.54 s for the 20 sliders of `straight_p1`)
//! run off the UI thread: `drag_to` (which recomputes the coupled allowed intervals), the slider rows and the
//! retuned lattice are computed by a `LatestWorker` that keeps only the newest job. Moves made while a job runs are
//! queued and replayed on top of the last applied `SliderState`, so no move is lost when several sliders are
//! dragged in a row. `set_param`, `drag_param` and `reset_params` stay synchronous (and cancel a pending job).

use std::time::Instant;

use papertilling_core::plane::tuning::SliderState;

use super::GalleryTabState;
use crate::backend::{CatalogueItemUi, SliderParamUi};
use crate::canvas2d::Canvas2D;
use crate::combine_worker::LatestWorker;
use crate::live_plane::{TunedLattice, slider_params_ui, tune_item};

pub(super) struct RetuneJob {
    base: CatalogueItemUi,
    state: SliderState,
    moves: Vec<(String, f64)>,
}

pub(super) struct RetuneOut {
    state: SliderState,
    sliders: Vec<SliderParamUi>,
    tuned: TunedLattice,
    secs: f64,
}

pub(super) type RetuneWorker = LatestWorker<RetuneJob, RetuneOut>;

fn spawn() -> RetuneWorker {
    LatestWorker::spawn("gallery-retune", (), |_: &mut (), job: RetuneJob| {
        let t0 = Instant::now();
        let mut state = job.state;
        for (name, raw) in &job.moves {
            let _ = state.drag_to(name, *raw);
        }
        let sliders = slider_params_ui(&state);
        let tuned = tune_item(&job.base, Some(&state));
        RetuneOut { state, sliders, tuned, secs: t0.elapsed().as_secs_f64() }
    })
}

impl GalleryTabState {
    /// Queue a pointer move of parameter `name` to `raw` for the background worker (the slider shows `raw` until
    /// the result is applied by [`Self::poll_retune`]).
    pub(crate) fn drag_param_async(&mut self, name: &str, raw: f64, ctx: &egui::Context) {
        let Some(state) = self.slider_state.clone() else { return };
        // consecutive moves of one slider collapse; moves of different sliders keep their order (coupled limits)
        match self.pending_moves.last_mut() {
            Some((n, v)) if n == name => *v = raw,
            _ => self.pending_moves.push((name.to_owned(), raw)),
        }
        let w = self.retune.get_or_insert_with(spawn);
        w.set_repaint_ctx(ctx);
        w.submit(RetuneJob { base: self.base_item.clone(), state, moves: self.pending_moves.clone() });
    }

    /// The value a slider should show: a queued move, else the stored value.
    pub(crate) fn pending_value(&self, name: &str) -> Option<f64> {
        self.pending_moves.iter().rev().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// A background move is queued or running.
    pub fn retune_pending(&self) -> bool {
        self.retune.as_ref().is_some_and(LatestWorker::is_busy)
    }

    /// Replay the queued moves on the UI thread, in order, and ignore the running job (before a synchronous move,
    /// so a typed value or Reset never discards another slider's drag). Returns true when moves were replayed;
    /// the caller retunes.
    pub(super) fn flush_pending_moves(&mut self) -> bool {
        let moves = std::mem::take(&mut self.pending_moves);
        if let Some(w) = self.retune.as_mut() {
            w.cancel();
        }
        let Some(st) = self.slider_state.as_mut() else { return false };
        for (name, raw) in &moves {
            let _ = st.drag_to(name, *raw);
        }
        !moves.is_empty()
    }

    /// Apply a pending background move synchronously (leaving the Gallery: the next tab gets the final lattice).
    pub(crate) fn finish_pending_retune(&mut self, canvas: &mut Canvas2D) {
        if self.flush_pending_moves() {
            self.retune(canvas, false);
        }
    }

    /// Drop the queued moves and ignore a running job (another lattice is selected).
    pub(super) fn cancel_retune(&mut self) {
        if let Some(w) = self.retune.as_mut() {
            w.cancel();
        }
        self.pending_moves.clear();
    }

    /// Apply the newest finished background move. `canvas` is `None` off the Gallery tab (the cell is then
    /// redrawn when the Gallery is shown again). Returns true when `active_item()` changed.
    pub(crate) fn poll_retune(&mut self, canvas: Option<&mut Canvas2D>) -> bool {
        let Some(res) = self.retune.as_mut().and_then(LatestWorker::poll) else { return false };
        self.pending_moves.clear();
        let out = match res {
            Ok(out) => out,
            Err(e) => {
                self.readout.status_ok = false;
                self.readout.status_message = format!("Parameter move failed: {e}");
                return false;
            }
        };
        let rev0 = self.revision;
        let mut out = out;
        // the job carried the lattice as it was named when submitted: keep a rename made meanwhile
        if let Some(it) = out.tuned.item.as_mut() {
            it.label.clone_from(&self.base_item.label);
        }
        self.slider_state = Some(out.state);
        self.last_move_secs = out.secs;
        self.apply_tuned(out.sliders, out.tuned);
        if let Some(canvas) = canvas {
            self.sync_scene(canvas, false);
        }
        self.revision != rev0
    }

    /// Block until a pending background move is applied (tests).
    #[cfg(test)]
    pub(crate) fn wait_retune(&mut self, canvas: &mut Canvas2D) -> bool {
        let t0 = Instant::now();
        while self.retune_pending() && t0.elapsed().as_secs_f64() < 30.0 {
            if self.poll_retune(Some(canvas)) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        self.poll_retune(Some(canvas))
    }
}
