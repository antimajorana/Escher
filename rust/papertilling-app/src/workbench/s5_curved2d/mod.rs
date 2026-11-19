//! S5 workbench panel (Lattices -> "{p,q} on S² / E² / H²"): regular {p,q} tilings of constant curvature, the
//! Wythoff family, large H² patches (three kernels), H² tilings placed on surfaces in R³ with papercraft nets, and
//! Conway orbifolds / vertex configurations / compact triangle-group quotients (TG cells).
//!
//! Every geometry build runs on one lazily spawned [`LatestWorker`], which also builds the 2D scene and the 3D mesh,
//! so `scenes()` only clones. Tiny exact lookups (δ = 4 − (p−2)(q−2), orbifold and vertex-configuration parsing) run
//! on the UI thread. Each sub-row keeps its own last result, keyed by the job's inputs; a refused job clears that
//! sub-row's result (a refused geometry is never replaced by an approximation).

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::noneuclidean2d::{delta_pq, geometry_kind_pq};
use serde_json::{Value, json};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;
use crate::combine_worker::LatestWorker;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

mod draw;
mod large_h2;
mod orbifolds;
mod pq;
mod surfaces;
#[cfg(test)]
mod tests;
mod wythoff;

/// Largest p and q offered (the H² vertex-completion builder supports polygons up to 12 sides).
pub(crate) const MAX_PQ: usize = 12;

/// The panel's inner sub-row. Labels are unique on screen (kittest clicks them by exact label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum S5Sub {
    #[default]
    Regular,
    Wythoff,
    LargeH2,
    Surfaces,
    Orbifolds,
}

impl S5Sub {
    pub(crate) const ALL: [Self; 5] = [Self::Regular, Self::Wythoff, Self::LargeH2, Self::Surfaces, Self::Orbifolds];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Regular => "Regular {p,q}",
            Self::Wythoff => "Wythoff family",
            Self::LargeH2 => "Large H² patch",
            Self::Surfaces => "Surfaces & nets",
            Self::Orbifolds => "Orbifolds & TG cells",
        }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

/// A finished build: the pictures plus the inspector content (key/value rows, stamped notes, an optional table).
#[derive(Clone)]
pub(crate) struct S5Out {
    pub(crate) scene2d: Scene2D,
    pub(crate) mesh: GpuMeshData,
    pub(crate) caption: String,
    pub(crate) rows: Vec<(String, String)>,
    pub(crate) notes: Vec<String>,
    pub(crate) table: Option<(Vec<&'static str>, Vec<Vec<String>>)>,
    /// Papercraft net as SVG text (Surfaces sub-row).
    pub(crate) svg: Option<String>,
}

pub(crate) fn geom_label(k: GeometryKind) -> &'static str {
    match k {
        GeometryKind::Spherical => "S² (spherical)",
        GeometryKind::Euclidean => "E² (Euclidean)",
        GeometryKind::Hyperbolic => "H² (hyperbolic)",
    }
}

/// Geometry of {p,q}, refused with a reason outside 2..=MAX_PQ.
pub(crate) fn check_symbol(p: usize, q: usize) -> Result<GeometryKind, String> {
    if p > MAX_PQ || q > MAX_PQ {
        return Err(format!("{{{p},{q}}} refused: this panel offers p, q <= {MAX_PQ}"));
    }
    geometry_kind_pq(p, q)
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// Exact 1/p + 1/q − 1/2 = δ / (2pq) as a reduced fraction.
pub(crate) fn excess_fraction(p: usize, q: usize) -> Option<String> {
    let d = delta_pq(p, q).ok()?;
    let den = 2 * (p * q) as i64;
    let g = gcd(d, den);
    Some(if d == 0 { String::from("0") } else { format!("{}/{}", d / g, den / g) })
}

/// Live badge: δ and the geometry it selects (exact integer arithmetic).
pub(crate) fn delta_badge(p: usize, q: usize) -> String {
    match (delta_pq(p, q), check_symbol(p, q)) {
        (Ok(d), Ok(k)) => format!("δ = 4 − (p−2)(q−2) = {d}: {} [PROVED]", geom_label(k)),
        (_, Err(e)) | (Err(e), _) => format!("refused: {e}"),
    }
}

/// Euler-characteristic note, stamped by what it rests on: Euler's formula for a closed sphere tiling, or the
/// count of the built patch (which says so when it is not the expected value).
pub(crate) fn euler_note(closed: bool, chi: i64) -> String {
    match (closed, chi) {
        (true, 2) => String::from("Closed tiling of S²: V − E + F = 2 (Euler) [PROVED]"),
        (true, c) => format!("V − E + F = {c} ≠ 2: the built complex is not a valid sphere tiling [COMPUTATIONAL EVIDENCE]"),
        (false, 1) => String::from("V − E + F = 1, as for a disk-shaped patch [COMPUTATIONAL EVIDENCE]"),
        (false, c) => format!("V − E + F = {c} ≠ 1: the patch is not a disk [COMPUTATIONAL EVIDENCE]"),
    }
}

/// p and q DragValues (2..=MAX_PQ, or 3..=MAX_PQ for H²-only rows); true when changed.
pub(crate) fn pq_controls(ui: &mut egui::Ui, p: &mut usize, q: &mut usize, lo: usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("p (polygon sides):");
        changed |= ui.add(egui::DragValue::new(p).range(lo..=MAX_PQ)).changed();
        ui.label("q (at a vertex):");
        changed |= ui.add(egui::DragValue::new(q).range(lo..=MAX_PQ)).changed();
    });
    changed
}

pub(crate) fn get_usize(v: &Value, k: &str, lo: usize, hi: usize) -> Option<usize> {
    v.get(k)?.as_u64().map(|x| usize::try_from(x).unwrap_or(hi).clamp(lo, hi))
}

#[derive(Debug)]
enum S5Job {
    Regular(pq::PqJob),
    Wythoff(wythoff::WyJob),
    Large(large_h2::LargeJob),
    Surface(surfaces::SurfJob),
    Tg(String),
}

fn run_job(job: S5Job) -> Result<S5Out, String> {
    match job {
        S5Job::Regular(j) => pq::build(&j),
        S5Job::Wythoff(j) => wythoff::build(&j),
        S5Job::Large(j) => large_h2::build(&j),
        S5Job::Surface(j) => surfaces::build(&j),
        S5Job::Tg(name) => orbifolds::build_tg(&name),
    }
}

type S5Worker = LatestWorker<S5Job, Result<S5Out, String>>;

#[derive(Default)]
pub(crate) struct Panel {
    sub: S5Sub,
    pq: pq::PqState,
    wy: wythoff::WyState,
    large: large_h2::LargeState,
    surf: surfaces::SurfState,
    orb: orbifolds::OrbState,
    worker: Option<S5Worker>,
    pending: Option<(S5Sub, String)>,
    built: [Option<String>; 5],
    outs: [Option<S5Out>; 5],
    errors: [Option<String>; 5],
}

impl Panel {
    fn job(&self) -> S5Job {
        match self.sub {
            S5Sub::Regular => S5Job::Regular(self.pq.job()),
            S5Sub::Wythoff => S5Job::Wythoff(self.wy.job()),
            S5Sub::LargeH2 => S5Job::Large(self.large.job()),
            S5Sub::Surfaces => S5Job::Surface(self.surf.job()),
            S5Sub::Orbifolds => S5Job::Tg(self.orb.tg_name().to_string()),
        }
    }

    fn accept(&mut self, res: Result<Result<S5Out, String>, String>) {
        let Some((sub, key)) = self.pending.take() else { return };
        let i = sub.idx();
        match res.and_then(|r| r) {
            Ok(out) => {
                self.errors[i] = None;
                self.built[i] = Some(key);
                self.outs[i] = Some(out);
            }
            Err(e) => {
                self.errors[i] = Some(e);
                self.built[i] = None;
                self.outs[i] = None;
            }
        }
    }

    fn show_out(&mut self, ui: &mut egui::Ui, cx: &WbCtx) {
        let i = self.sub.idx();
        let Some(out) = &self.outs[i] else {
            if self.errors[i].is_none() {
                ui.label("Building ...");
            }
            return;
        };
        egui::Grid::new("wb_s5_rows").striped(true).num_columns(2).show(ui, |ui| {
            for (k, v) in &out.rows {
                ui.label(k.as_str());
                ui.label(v.as_str());
                ui.end_row();
            }
        });
        for n in &out.notes {
            ui.small(n.as_str());
        }
        if let Some((head, rows)) = &out.table {
            ui.separator();
            ui.push_id("wb_s5_table", |ui| {
                egui::ScrollArea::vertical().id_salt("wb_s5_table_scroll").max_height(220.0).show(ui, |ui| {
                    egui::Grid::new("wb_s5_table_grid").striped(true).show(ui, |ui| {
                        for h in head {
                            ui.strong(*h);
                        }
                        ui.end_row();
                        for r in rows {
                            for c in r {
                                ui.label(c.as_str());
                            }
                            ui.end_row();
                        }
                    });
                });
            });
        }
        if self.sub == S5Sub::Surfaces {
            let svg = out.svg.clone();
            self.surf.show_export(ui, cx, svg.as_deref());
        }
    }
}

impl WbPanel for Panel {
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        let mut rebuild = false;
        ui.horizontal_wrapped(|ui| {
            for s in S5Sub::ALL {
                if ui.selectable_label(self.sub == s, s.label()).clicked() && self.sub != s {
                    self.sub = s;
                    rebuild = true;
                }
            }
        });
        ui.separator();
        rebuild |= match self.sub {
            S5Sub::Regular => self.pq.show_controls(ui),
            S5Sub::Wythoff => self.wy.show_controls(ui),
            S5Sub::LargeH2 => self.large.show_controls(ui),
            S5Sub::Surfaces => self.surf.show_controls(ui),
            S5Sub::Orbifolds => self.orb.show_controls(ui),
        };
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, _tokens: &ThemeTokens) {
        if self.sub == S5Sub::Orbifolds {
            self.orb.show_cards(ui);
            ui.separator();
        }
        self.show_out(ui, cx);
    }

    fn submit(&mut self, _cx: &WbCtx, repaint: Option<&egui::Context>) {
        let job = self.job();
        let key = format!("{job:?}");
        let i = self.sub.idx();
        let in_flight = self.pending.as_ref().is_some_and(|(s, k)| *s == self.sub && *k == key);
        if in_flight || self.built[i].as_deref() == Some(key.as_str()) {
            return;
        }
        self.errors[i] = None;
        let worker = self.worker.get_or_insert_with(|| LatestWorker::spawn("wb-s5", (), |_: &mut (), job| run_job(job)));
        if let Some(ctx) = repaint {
            worker.set_repaint_ctx(ctx);
        }
        worker.submit(job);
        self.pending = Some((self.sub, key));
    }

    fn poll(&mut self) -> bool {
        let Some(res) = self.worker.as_mut().and_then(LatestWorker::poll) else { return false };
        self.accept(res);
        true
    }

    fn scenes(&self) -> WbScenes {
        let i = self.sub.idx();
        match &self.outs[i] {
            Some(o) => WbScenes { scene2d: o.scene2d.clone(), mesh: Some(o.mesh.clone()), caption: o.caption.clone() },
            None => {
                let note = match &self.errors[i] {
                    Some(e) => format!("refused: {e}"),
                    None => String::from("building ..."),
                };
                WbScenes {
                    scene2d: empty_scene(&format!("S5 {}", self.sub.label()), &note),
                    mesh: Some(placeholder_mesh()),
                    caption: String::from("S5: no result yet (placeholder tetrahedron)"),
                }
            }
        }
    }

    fn is_busy(&self) -> bool {
        self.worker.as_ref().is_some_and(LatestWorker::is_busy)
    }

    fn error(&self) -> Option<&str> {
        self.errors[self.sub.idx()].as_deref()
    }

    fn save(&self) -> Value {
        json!({
            "sub": format!("{:?}", self.sub),
            "pq": self.pq.save(),
            "wythoff": self.wy.save(),
            "large": self.large.save(),
            "surfaces": self.surf.save(),
            "orbifolds": self.orb.save(),
        })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("S5 panel state is not an object (ignored)"));
            return;
        };
        if let Some(s) = o.get("sub").and_then(Value::as_str) {
            match S5Sub::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) => self.sub = x,
                None => notes.push(format!("S5 panel: unknown sub-view {s:?} (kept current)")),
            }
        }
        if let Some(x) = o.get("pq") {
            self.pq.restore(x);
        }
        if let Some(x) = o.get("wythoff") {
            self.wy.restore(x);
        }
        if let Some(x) = o.get("large") {
            self.large.restore(x);
        }
        if let Some(x) = o.get("surfaces") {
            self.surf.restore(x);
        }
        if let Some(x) = o.get("orbifolds") {
            self.orb.restore(x);
        }
    }

    /// {7,3} with 2 layers on the regular sub-row: a small H² patch with a non-empty hyperboloid mesh.
    fn smoke_defaults(&mut self) {
        self.sub = S5Sub::Regular;
        self.pq.set(7, 3, 2);
    }
}
