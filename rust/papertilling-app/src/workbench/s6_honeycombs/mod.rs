//! S6 workbench panel (Lattices -> "Honeycombs & TPMS"): regular `{p,q,r}` honeycombs of S³ / E³ / H³ with a live
//! regime classification, the four Euclidean honeycombs, clean-room triply periodic P / D / G nets and surfaces, and
//! lattice Voronoi cells plus Dirichlet domains of closed 3-manifolds (quotient counts, χ, H₁).
//!
//! Every geometry build runs on one lazily spawned [`LatestWorker`] (meshes and 2D scenes are made there too); only
//! `classify_schlafli` runs on the UI thread. Each sub-row keeps its own last result keyed by the job's inputs, so
//! switching sub-rows does not rebuild. A refused input (hyperideal or degenerate symbol, a surface the core does not
//! have) drops that sub-row's result and shows a placeholder with the reason: an old mesh never sits under it.
//!
//! Stale results: a stored result is shown only while its key equals the sub-row's current job key; a job whose
//! inputs are no longer wanted (cache hit, refusal, another sub-row) is cancelled, and a late result whose key is not
//! the wanted one is dropped.

use serde_json::{Value, json};

use papertilling_core::honeycomb3d::{EuclideanKind, TpmsKind};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::combine_worker::LatestWorker;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

mod honeycomb;
mod present;
mod tpms;
mod voronoi;
#[cfg(test)]
mod tests;

use present::Built;

/// The panel's inner sub-row. Labels are unique on screen (kittest clicks them by exact label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum S6Sub {
    #[default]
    Schlafli,
    Euclidean,
    Tpms,
    Voronoi,
}

impl S6Sub {
    pub(crate) const ALL: [Self; 4] = [Self::Schlafli, Self::Euclidean, Self::Tpms, Self::Voronoi];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Schlafli => "Schläfli {p,q,r}",
            Self::Euclidean => "Euclidean E³ patches",
            Self::Tpms => "TPMS nets & surfaces",
            Self::Voronoi => "Voronoi & Dirichlet",
        }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug)]
pub(crate) enum S6Job {
    Schlafli { p: u32, q: u32, r: u32, shells: u32, layers: u32 },
    Euclid { kind: EuclideanKind, shells: u32 },
    Net { kind: TpmsKind, labyrinths: u32, reps: u32 },
    Surface { kind: TpmsKind, mode: tpms::TpmsMode, n: usize, reps: u32 },
    LatticeCell { lattice: &'static str },
    Manifold { name: &'static str },
}

pub(crate) fn run_job(job: S6Job) -> Result<Built, String> {
    match job {
        S6Job::Schlafli { p, q, r, shells, layers } => honeycomb::build_schlafli(p, q, r, shells, layers),
        S6Job::Euclid { kind, shells } => honeycomb::build_euclid(kind, shells),
        S6Job::Net { kind, labyrinths, reps } => tpms::build_net(kind, labyrinths, reps),
        S6Job::Surface { kind, mode, n, reps } => tpms::build_surface(kind, mode, n, reps),
        S6Job::LatticeCell { lattice } => voronoi::build_lattice_cell(lattice),
        S6Job::Manifold { name } => voronoi::build_manifold(name),
    }
}

type S6Worker = LatestWorker<S6Job, Result<Built, String>>;

#[derive(Default)]
pub(crate) struct Panel {
    sub: S6Sub,
    schlafli: honeycomb::SchlafliState,
    euclid: honeycomb::EuclidState,
    tpms: tpms::TpmsState,
    vor: voronoi::VorState,
    worker: Option<S6Worker>,
    /// Sub-row and input key of the job in flight.
    pending: Option<(S6Sub, String)>,
    /// Each sub-row's last result and its input key.
    built: [Option<(String, Built)>; 4],
    errors: [Option<String>; 4],
}

impl Panel {
    fn job(&self) -> S6Job {
        self.job_for(self.sub)
    }

    /// The job the inputs of sub-row `sub` currently ask for.
    fn job_for(&self, sub: S6Sub) -> S6Job {
        match sub {
            S6Sub::Schlafli => {
                let (p, q, r, shells, layers) = self.schlafli.params();
                S6Job::Schlafli { p, q, r, shells, layers }
            }
            S6Sub::Euclidean => S6Job::Euclid { kind: self.euclid.kind, shells: self.euclid.shells },
            S6Sub::Tpms => {
                let t = &self.tpms;
                match t.mode {
                    tpms::TpmsMode::Net => S6Job::Net { kind: t.kind, labyrinths: t.labyrinths, reps: t.reps },
                    m => {
                        let n = if m == tpms::TpmsMode::Nodal { t.n } else { 0 };
                        S6Job::Surface { kind: t.kind, mode: m, n, reps: t.effective_reps() }
                    }
                }
            }
            S6Sub::Voronoi => match self.vor.mode {
                voronoi::VorMode::LatticeCell => S6Job::LatticeCell { lattice: voronoi::LATTICES[self.vor.lattice].0 },
                voronoi::VorMode::Manifold => S6Job::Manifold { name: voronoi::MANIFOLDS[self.vor.manifold].0 },
            },
        }
    }

    /// Refusals decided on the UI thread (no worker round trip): a non-buildable Schläfli symbol.
    fn refusal(&self, sub: S6Sub) -> Option<String> {
        if sub != S6Sub::Schlafli {
            return None;
        }
        let c = self.schlafli.classify();
        (!c.is_buildable()).then_some(c.reason)
    }

    fn accept(&mut self, res: Result<Result<Built, String>, String>) {
        let Some((sub, key)) = self.pending.take() else { return };
        // a late result for inputs that are no longer wanted is dropped, never stored under the current controls
        if self.refusal(sub).is_some() || key != format!("{:?}", self.job_for(sub)) {
            if sub == self.sub {
                // defensive: the shown sub-row then asks for its current inputs (no-op when built or refused)
                self.request(None);
            }
            return;
        }
        let i = sub.idx();
        match res.and_then(|r| r) {
            Ok(b) => {
                self.errors[i] = None;
                self.built[i] = Some((key, b));
            }
            Err(e) => {
                self.errors[i] = Some(e);
                self.built[i] = None;
            }
        }
    }

    /// The shown sub-row's result, only when it was built from the current inputs (else `None`: "Building…").
    fn current(&self) -> Option<&Built> {
        let i = self.sub.idx();
        if self.errors[i].is_some() {
            return None;
        }
        let key = format!("{:?}", self.job());
        self.built[i].as_ref().filter(|(k, _)| *k == key).map(|(_, b)| b)
    }

    /// Build the shown sub-row's current inputs unless they are refused, built or already in flight.
    fn request(&mut self, repaint: Option<&egui::Context>) {
        let i = self.sub.idx();
        if let Some(reason) = self.refusal(self.sub) {
            self.errors[i] = Some(reason);
            self.built[i] = None;
            self.drop_pending();
            return;
        }
        let job = self.job();
        let key = format!("{job:?}");
        if self.pending.as_ref().is_some_and(|(s, k)| *s == self.sub && *k == key) {
            return;
        }
        if self.built[i].as_ref().is_some_and(|(k, _)| *k == key) {
            // cache hit: a job for other inputs (this or another sub-row) is no longer wanted
            self.drop_pending();
            return;
        }
        self.errors[i] = None;
        let worker = self.worker.get_or_insert_with(|| LatestWorker::spawn("wb-s6", (), |_: &mut (), job| run_job(job)));
        if let Some(ctx) = repaint {
            worker.set_repaint_ctx(ctx);
        }
        worker.submit(job);
        self.pending = Some((self.sub, key));
    }

    /// Cancel the job in flight (any sub-row): its result, if it still lands, is ignored by the worker.
    fn drop_pending(&mut self) {
        if self.pending.take().is_some()
            && let Some(w) = self.worker.as_mut()
        {
            w.cancel();
        }
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        const LABELS: [&str; 4] = ["Schläfli {p,q,r}", "Euclidean E³ patches", "TPMS nets & surfaces", "Voronoi & Dirichlet"];
        &LABELS
    }
    fn sub_index(&self) -> usize {
        self.sub as usize
    }
    /// The body of the sub-row click branch in `show_controls`.
    fn select_sub(&mut self, i: usize) -> bool {
        match S6Sub::ALL.get(i) {
            Some(&s) if self.sub != s => {
                self.sub = s;
                true
            }
            _ => false,
        }
    }
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        let mut rebuild = false;
        ui.horizontal_wrapped(|ui| {
            for s in S6Sub::ALL {
                if ui.selectable_label(self.sub == s, s.label()).clicked() && self.sub != s {
                    self.sub = s;
                    rebuild = true;
                }
            }
        });
        ui.separator();
        rebuild |= match self.sub {
            S6Sub::Schlafli => self.schlafli.show_controls(ui),
            S6Sub::Euclidean => self.euclid.show_controls(ui),
            S6Sub::Tpms => self.tpms.show_controls(ui),
            S6Sub::Voronoi => self.vor.show_controls(ui),
        };
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, _cx: &WbCtx, _tokens: &ThemeTokens) {
        match self.current() {
            Some(b) => present::show_built(ui, b, "wb_s6_inspector_grid"),
            None if self.errors[self.sub.idx()].is_some() => {
                ui.label("Nothing is drawn for refused input: the 3D pane shows a placeholder.");
            }
            None => {
                ui.label("Building…");
            }
        }
    }

    fn submit(&mut self, _cx: &WbCtx, repaint: Option<&egui::Context>) {
        self.request(repaint);
    }

    fn poll(&mut self) -> bool {
        let Some(res) = self.worker.as_mut().and_then(LatestWorker::poll) else { return false };
        self.accept(res);
        true
    }

    fn scenes(&self) -> WbScenes {
        if let Some(b) = self.current() {
            return WbScenes { scene2d: b.scene2d.clone(), mesh: Some(b.mesh.clone()), caption: b.caption.clone() };
        }
        let (title, note, caption) = match &self.errors[self.sub.idx()] {
            Some(e) => ("S6: input refused", e.as_str(), "S6 placeholder tetrahedron (input refused, see inspector)"),
            None => ("S6: building", "the result appears when the background build finishes", "S6 placeholder tetrahedron (building)"),
        };
        WbScenes { scene2d: empty_scene(title, note), mesh: Some(placeholder_mesh()), caption: String::from(caption) }
    }

    /// Busy only for the shown sub-row's own job (a job for another sub-row never holds back the scene sync).
    fn is_busy(&self) -> bool {
        self.pending.as_ref().is_some_and(|(s, _)| *s == self.sub) && self.worker.as_ref().is_some_and(LatestWorker::is_busy)
    }

    fn error(&self) -> Option<&str> {
        self.errors[self.sub.idx()].as_deref()
    }

    fn save(&self) -> Value {
        json!({
            "sub": format!("{:?}", self.sub),
            "schlafli": self.schlafli.save(),
            "euclid": self.euclid.save(),
            "tpms": self.tpms.save(),
            "voronoi": self.vor.save(),
        })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("S6 panel state is not an object (ignored)"));
            return;
        };
        if let Some(s) = o.get("sub").and_then(Value::as_str) {
            match S6Sub::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) => self.sub = x,
                None => notes.push(format!("S6 panel: unknown sub-view {s:?} (kept current)")),
            }
        }
        if let Some(x) = o.get("schlafli") {
            self.schlafli.restore(x);
        }
        if let Some(x) = o.get("euclid") {
            self.euclid.restore(x, notes);
        }
        if let Some(x) = o.get("tpms") {
            self.tpms.restore(x, notes);
        }
        if let Some(x) = o.get("voronoi") {
            self.vor.restore(x, notes);
        }
    }

    /// The smoke stage of this view: the cubic honeycomb {4,3,4}, one shell (32 V, 60 E, 36 F, 7 cells).
    fn smoke_defaults(&mut self) {
        self.sub = S6Sub::Schlafli;
        self.schlafli = honeycomb::SchlafliState::default();
    }
}
