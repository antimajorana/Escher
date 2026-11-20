//! S3 workbench panel (Defects -> "Detect & Fields"): defects detected from the bare coordinates of the Defects-tab
//! crystal (Frank charges, surrounding loops, Burgers vectors), per-site strain / per-face Nye / Vitek maps, the GSV
//! moduli with the Barnett-Lothe tensor and the Stroh dislocation field, and a motion GIF.
//!
//! Every build runs on one lazily spawned [`LatestWorker`]. Each sub-row keeps its last result keyed by the job's
//! inputs (a fingerprint of the source crystal included), so switching sub-rows or display-only choices does not
//! rebuild, and a new Defects crystal (the host's `source_changed`) does. The source is read in the cell frame
//! (`gallery.cell()`, which already carries the lattice frame), so the detector gets the identity frame.

use std::sync::Arc;

use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::CanvasInteraction;
use crate::combine_worker::LatestWorker;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

mod detect;
mod elastic;
mod fields;
mod gif;
mod graph_check;
mod settings;
#[cfg(test)]
mod tests;

pub(crate) use settings::{FieldKind, Settings};

pub(crate) const SCENE_TITLE: &str = "S3 detection";

/// The panel's inner sub-row. Labels are unique on screen (kittest clicks them by exact label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum S3Sub {
    #[default]
    Detect,
    Fields,
    Elastic,
    Gif,
}

impl S3Sub {
    pub(crate) const ALL: [Self; 4] = [Self::Detect, Self::Fields, Self::Elastic, Self::Gif];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Detect => "Detect defects",
            Self::Fields => "Strain / Nye",
            Self::Elastic => "Elastic & Stroh",
            Self::Gif => "Motion GIF",
        }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

/// Snapshot of the Defects-tab crystal plus the lattice it was built on.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) pos: Vec<[f64; 2]>,
    pub(crate) ref_pos: Vec<[f64; 2]>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) group: String,
    pub(crate) cell: Cell2D,
    /// Hash of positions, edges and lattice: part of every job key.
    pub(crate) fp: u64,
}

impl Source {
    pub(crate) fn new(pos: Vec<[f64; 2]>, ref_pos: Vec<[f64; 2]>, edges: Vec<(usize, usize)>, group: &str, cell: &Cell2D) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for p in pos.iter().chain(&ref_pos) {
            p[0].to_bits().hash(&mut h);
            p[1].to_bits().hash(&mut h);
        }
        edges.hash(&mut h);
        group.hash(&mut h);
        format!("{:?}{:?}{:?}{:?}", cell.a1, cell.a2, cell.basis, cell.bonds).hash(&mut h);
        Self { pos, ref_pos, edges, group: group.to_string(), cell: cell.clone(), fp: h.finish() }
    }

    fn from_ctx(cx: &WbCtx) -> Option<Self> {
        let c = cx.defect_crystal.filter(|c| !c.pos.is_empty())?;
        Some(Self::new(c.pos.clone(), c.ref_pos.clone(), c.edges.clone(), &cx.item.group, cx.cell))
    }
}

pub(crate) enum S3Out {
    Detect(Box<detect::DetectOut>),
    Fields(Box<fields::FieldsOut>),
    Elastic(Box<elastic::ElasticOut>),
    Gif(Box<gif::GifOut>),
}

#[derive(Debug)]
pub(crate) struct S3Job {
    pub(crate) sub: S3Sub,
    pub(crate) src: Arc<Source>,
    pub(crate) set: Settings,
}

/// The worker body (also called directly by the tests).
pub(crate) fn run_job(job: &S3Job) -> Result<S3Out, String> {
    job.set.validate(job.sub)?;
    let det = detect::build(&job.src, &job.set)?;
    Ok(match job.sub {
        S3Sub::Detect => S3Out::Detect(Box::new(det)),
        S3Sub::Fields => S3Out::Fields(Box::new(fields::build(&job.src, &job.set, det)?)),
        S3Sub::Elastic => S3Out::Elastic(Box::new(elastic::build(&job.src, &job.set, det)?)),
        S3Sub::Gif => S3Out::Gif(Box::new(gif::build(&job.src, &job.set, det)?)),
    })
}

type S3Worker = LatestWorker<S3Job, Result<S3Out, String>>;

#[derive(Default)]
pub(crate) struct Panel {
    pub(crate) sub: S3Sub,
    pub(crate) set: Settings,
    pub(crate) detect: Option<Box<detect::DetectOut>>,
    pub(crate) fields: Option<Box<fields::FieldsOut>>,
    pub(crate) elastic: Option<Box<elastic::ElasticOut>>,
    pub(crate) gif: Option<Box<gif::GifOut>>,
    /// Status line of the last GIF export.
    gif_status: String,
    worker: Option<S3Worker>,
    /// Sub-row and input key of the job in flight.
    pending: Option<(S3Sub, String)>,
    /// Per sub-row, the key the current inputs ask for (set by `submit`); a result with another key is stale.
    wanted: [Option<String>; 4],
    built: [Option<String>; 4],
    errors: [Option<String>; 4],
    /// Why nothing was submitted (no source crystal yet); never an error.
    pub(crate) waiting: Option<String>,
}

impl Panel {
    pub(crate) fn job_key(&self, src: &Source) -> String {
        format!("{:?}|{}|{}", self.sub, src.fp, self.set.key(self.sub))
    }

    /// Forget the job in flight: the worker drops its late result (it cannot interrupt the running job).
    fn forget_pending(&mut self) {
        if self.pending.take().is_some()
            && let Some(w) = self.worker.as_mut()
        {
            w.cancel();
        }
    }

    fn accept(&mut self, res: Result<Result<S3Out, String>, String>) {
        let Some((sub, key)) = self.pending.take() else { return };
        if self.wanted[sub.idx()].as_deref() != Some(key.as_str()) {
            return; // stale: the inputs changed (or the source went away) after this job was submitted
        }
        match res.and_then(|r| r) {
            Ok(out) => {
                self.errors[sub.idx()] = None;
                self.built[sub.idx()] = Some(key);
                match out {
                    S3Out::Detect(o) => self.detect = Some(o),
                    S3Out::Fields(o) => self.fields = Some(o),
                    S3Out::Elastic(o) => self.elastic = Some(o),
                    S3Out::Gif(o) => self.gif = Some(o),
                }
            }
            Err(e) => {
                self.errors[sub.idx()] = Some(e);
                self.built[sub.idx()] = None;
                match sub {
                    S3Sub::Detect => self.detect = None,
                    S3Sub::Fields => self.fields = None,
                    S3Sub::Elastic => self.elastic = None,
                    S3Sub::Gif => self.gif = None,
                }
            }
        }
    }

    /// The detection shown for the current sub-row (each result carries its own).
    fn current_detection(&self) -> Option<&detect::DetectOut> {
        match self.sub {
            S3Sub::Detect => self.detect.as_deref(),
            S3Sub::Fields => self.fields.as_deref().map(|f| &f.det),
            S3Sub::Elastic => self.elastic.as_deref().map(|e| &e.det),
            S3Sub::Gif => self.gif.as_deref().map(|g| &g.det),
        }
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        const LABELS: [&str; 4] = ["Detect defects", "Strain / Nye", "Elastic & Stroh", "Motion GIF"];
        &LABELS
    }
    fn sub_index(&self) -> usize {
        self.sub as usize
    }
    /// The body of the sub-row click branch in `show_controls`.
    fn select_sub(&mut self, i: usize) -> bool {
        match S3Sub::ALL.get(i) {
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
            for s in S3Sub::ALL {
                if ui.selectable_label(self.sub == s, s.label()).clicked() && self.sub != s {
                    self.sub = s;
                    rebuild = true;
                }
            }
        });
        ui.separator();
        rebuild |= self.set.show_common(ui);
        rebuild |= match self.sub {
            S3Sub::Detect => false,
            S3Sub::Fields => self.set.show_fields(ui),
            S3Sub::Elastic => self.set.show_elastic(ui),
            S3Sub::Gif => self.set.show_gif(ui),
        };
        if let Some(e) = self.set.validate(self.sub).err() {
            ui.colored_label(egui::Color32::from_rgb(192, 57, 43), format!("Input refused: {e}"));
        }
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, _tokens: &ThemeTokens) {
        ui.label(format!("Lattice: {} ({})", cx.item.label, cx.item.group));
        match cx.defect_crystal {
            Some(c) if !c.pos.is_empty() => ui.label(format!("Source: Defects-tab crystal, {} sites, {} bonds", c.pos.len(), c.edges.len())),
            _ => ui.label("Source: the Defects-tab crystal is empty (still building or refused)"),
        };
        if let Some(w) = &self.waiting {
            ui.label(w.as_str());
            return; // the stored results belong to a crystal that is gone
        }
        ui.separator();
        match self.sub {
            S3Sub::Detect => {
                if let Some(o) = &self.detect {
                    detect::show_inspector(ui, o, detect::resolve(&o.det, self.set.selected));
                }
            }
            S3Sub::Fields => {
                if let Some(o) = &self.fields {
                    fields::show_inspector(ui, o, &self.set);
                }
            }
            S3Sub::Elastic => {
                if let Some(o) = &self.elastic {
                    elastic::show_inspector(ui, o);
                }
            }
            S3Sub::Gif => {
                if let Some(o) = &self.gif
                    && let Some(msg) = gif::show_inspector(ui, o, cx.smoke)
                {
                    self.gif_status = msg;
                }
                if !self.gif_status.is_empty() {
                    ui.small(self.gif_status.as_str());
                }
            }
        }
    }

    fn submit(&mut self, cx: &WbCtx, repaint: Option<&egui::Context>) {
        let Some(src) = Source::from_ctx(cx) else {
            self.waiting = Some(String::from("Waiting for the Defects-tab crystal (build a defect there, or wait for it)."));
            self.forget_pending();
            self.wanted = Default::default();
            return;
        };
        self.waiting = None;
        let key = self.job_key(&src);
        let i = self.sub.idx();
        self.wanted[i] = Some(key.clone());
        // Some(true): this key is in flight; Some(false): another key of this sub-row is in flight (stale)
        let pending_same = self.pending.as_ref().filter(|(s, _)| *s == self.sub).map(|(_, k)| *k == key);
        if pending_same == Some(true) {
            return;
        }
        if self.built[i].as_deref() == Some(key.as_str()) {
            if pending_same == Some(false) {
                self.forget_pending();
            }
            return;
        }
        self.errors[i] = None;
        let worker = self.worker.get_or_insert_with(|| LatestWorker::spawn("wb-s3", (), |_: &mut (), job: S3Job| run_job(&job)));
        if let Some(ctx) = repaint {
            worker.set_repaint_ctx(ctx);
        }
        worker.submit(S3Job { sub: self.sub, src: Arc::new(src), set: self.set.clone() });
        self.pending = Some((self.sub, key));
    }

    fn poll(&mut self) -> bool {
        let Some(res) = self.worker.as_mut().and_then(LatestWorker::poll) else { return false };
        self.accept(res);
        true
    }

    fn scenes(&self) -> WbScenes {
        let placeholder = |note: &str| empty_scene(SCENE_TITLE, note);
        let scene2d = match (self.sub, &self.waiting) {
            (_, Some(w)) => placeholder(w),
            (S3Sub::Detect, _) => self.detect.as_deref().map_or_else(|| placeholder("building ..."), |o| detect::scene(o, detect::resolve(&o.det, self.set.selected))),
            (S3Sub::Fields, _) => self.fields.as_deref().map_or_else(|| placeholder("building ..."), |o| fields::scene(o, &self.set)),
            (S3Sub::Elastic, _) => self.elastic.as_deref().map_or_else(|| placeholder("building ..."), elastic::scene),
            (S3Sub::Gif, _) => self.gif.as_deref().map_or_else(|| placeholder("building ..."), gif::scene),
        };
        WbScenes { scene2d, mesh: None, caption: String::new(), overlay: None }
    }

    fn is_busy(&self) -> bool {
        self.worker.as_ref().is_some_and(LatestWorker::is_busy)
    }

    fn error(&self) -> Option<&str> {
        if self.waiting.is_some() {
            return None;
        }
        self.errors[self.sub.idx()].as_deref()
    }

    fn lattice_dependent(&self) -> bool {
        true
    }

    /// A click selects the detected defect whose centre is nearest (Fields / Elastic then use its Burgers vector).
    fn on_canvas(&mut self, i: &CanvasInteraction, _cx: &WbCtx) -> bool {
        let Some(w) = i.clicked_world else { return false };
        if self.waiting.is_some() {
            return false;
        }
        let Some(det) = self.current_detection() else { return false };
        let Some(k) = detect::nearest_defect(&det.det, w, 3.0 * det.lmax) else { return false };
        if self.set.selected == Some(k) {
            return false;
        }
        self.set.selected = Some(k);
        true
    }

    fn save(&self) -> Value {
        json!({ "sub": format!("{:?}", self.sub), "settings": self.set.save() })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("S3 panel state is not an object (ignored)"));
            return;
        };
        if let Some(s) = o.get("sub").and_then(Value::as_str) {
            match S3Sub::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) => self.sub = x,
                None => notes.push(format!("S3 panel: unknown sub-view {s:?} (kept current)")),
            }
        }
        if let Some(s) = o.get("settings") {
            self.set.restore(s);
        }
    }
}
