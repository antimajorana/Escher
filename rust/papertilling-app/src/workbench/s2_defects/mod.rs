//! S2 workbench panel (Defects -> "Point, Stone-Wales & GB"): point defects, the Stone-Wales bond rotation, CSL
//! symmetric-tilt grain boundaries, an all-defects table with Burgers indices, and solver export.
//!
//! Every geometry build runs on one lazily spawned [`LatestWorker`]; tiny lookups (CSL list, Burgers indices) run on
//! the UI thread. Each sub-row keeps its own last result, keyed by the job's inputs, so switching sub-rows does not
//! rebuild. Site and bond indices picked by the user refer to the host disk of the current lattice and radius; they are
//! cleared when either changes and are never saved in the session.

use std::collections::HashSet;

use papertilling_core::defect::{perfect_disk, rim_sites};
use papertilling_core::defect::point_defects::PointDefectOp;
use papertilling_core::types::{BuiltCrystal, Cell2D};
use serde_json::{Value, json};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::backend::CatalogueItemUi;
use crate::canvas2d::CanvasInteraction;
use crate::combine_worker::LatestWorker;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

mod boundary;
mod point;
mod solver;
mod stone_wales;
mod table;
#[cfg(test)]
mod tests;

/// The panel's inner sub-row. Labels are unique on screen (kittest clicks them by exact label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum S2Sub {
    #[default]
    Point,
    StoneWales,
    GrainBoundary,
    AllDefects,
    Solver,
}

impl S2Sub {
    pub(crate) const ALL: [Self; 5] = [Self::Point, Self::StoneWales, Self::GrainBoundary, Self::AllDefects, Self::Solver];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Point => "Point defects",
            Self::StoneWales => "Stone-Wales",
            Self::GrainBoundary => "CSL / twin GB",
            Self::AllDefects => "All defects",
            Self::Solver => "Solver export",
        }
    }

    fn idx(self) -> usize {
        self as usize
    }

    fn uses_disk(self) -> bool {
        matches!(self, Self::Point | Self::StoneWales | Self::GrainBoundary)
    }
}

/// Perfect host disk of the current lattice and radius, with its rim and mean bond length (for picking).
#[derive(Clone, Debug)]
pub(crate) struct Host {
    pub(crate) crystal: BuiltCrystal,
    pub(crate) rim: HashSet<usize>,
    pub(crate) bond: f64,
}

pub(crate) fn cell_scale(cell: &Cell2D) -> f64 {
    cell.a1[0].hypot(cell.a1[1]).max(1e-9)
}

/// Host disk centred at the origin; refused when it is too small to hold an interior site.
pub(crate) fn make_host(cell: &Cell2D, radius: f64) -> Result<Host, String> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(String::from("disk radius must be finite and > 0"));
    }
    let crystal = perfect_disk(cell, [0.0, 0.0], radius);
    if crystal.edges.is_empty() {
        return Err(String::from("the host disk has no bonds: increase the radius"));
    }
    let rim = rim_sites(&crystal);
    if rim.len() >= crystal.pos.len() {
        return Err(String::from("every site of the host disk is on its rim: increase the radius"));
    }
    let len = |&(u, v): &(usize, usize)| {
        let (p, q) = (crystal.pos[u], crystal.pos[v]);
        (p[0] - q[0]).hypot(p[1] - q[1])
    };
    let bond = crystal.edges.iter().map(len).sum::<f64>() / crystal.edges.len() as f64;
    Ok(Host { crystal, rim, bond })
}

#[derive(Debug)]
enum S2Job {
    Point { cell: Cell2D, radius: f64, ops: Vec<PointDefectOp> },
    StoneWales { cell: Cell2D, radius: f64, edge: Option<usize>, relax: bool, layers: u32 },
    Boundary { group: String, cell: Cell2D, theta: f64, sigma: Option<u32>, radius: f64 },
    Table { item: Box<CatalogueItemUi>, cell: Cell2D, max_sigma: u32 },
    Solver { group: String, cell: Cell2D, nx: usize, ny: usize, shells: usize },
}

enum S2Out {
    Point(point::PointOut),
    StoneWales(stone_wales::SwOut),
    Boundary(boundary::GbOut),
    Table(table::TableOut),
    Solver(Box<solver::SolverOut>),
}

fn run_job(job: S2Job) -> Result<S2Out, String> {
    Ok(match job {
        S2Job::Point { cell, radius, ops } => S2Out::Point(point::build(&cell, radius, ops)?),
        S2Job::StoneWales { cell, radius, edge, relax, layers } => S2Out::StoneWales(stone_wales::build_grown(&cell, radius, edge, relax, layers)?),
        S2Job::Boundary { group, cell, theta, sigma, radius } => S2Out::Boundary(boundary::build(&group, &cell, theta, sigma, radius)?),
        S2Job::Table { item, cell, max_sigma } => S2Out::Table(table::build(&item, &cell, max_sigma)),
        S2Job::Solver { group, cell, nx, ny, shells } => S2Out::Solver(Box::new(solver::build(&group, &cell, nx, ny, shells)?)),
    })
}

type S2Worker = LatestWorker<S2Job, Result<S2Out, String>>;

fn lattice_key(cx: &WbCtx) -> String {
    format!("{}|{:?}|{:?}|{:?}|{:?}", cx.item.group, cx.cell.a1, cx.cell.a2, cx.cell.basis, cx.cell.bonds)
}

pub(crate) struct Panel {
    sub: S2Sub,
    /// Host disk radius in units of |a1|.
    radius_cells: f64,
    host_key: String,
    lattice_key: String,
    point: point::PointState,
    sw: stone_wales::SwState,
    gb: boundary::GbState,
    table: table::TableState,
    solver: solver::SolverState,
    worker: Option<S2Worker>,
    /// Sub-row and input key of the job in flight.
    pending: Option<(S2Sub, String)>,
    /// Input key of each sub-row's stored result.
    built: [Option<String>; 5],
    errors: [Option<String>; 5],
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            sub: S2Sub::default(),
            radius_cells: 3.0,
            host_key: String::new(),
            lattice_key: String::new(),
            point: Default::default(),
            sw: Default::default(),
            gb: Default::default(),
            table: Default::default(),
            solver: Default::default(),
            worker: None,
            pending: None,
            built: Default::default(),
            errors: Default::default(),
        }
    }
}

impl Panel {
    fn radius(&self, cell: &Cell2D) -> f64 {
        self.radius_cells * cell_scale(cell)
    }

    /// Clear lattice- or radius-bound picks when the host changed (called first in `submit`).
    fn track_host(&mut self, cx: &WbCtx) {
        let lk = lattice_key(cx);
        let hk = format!("{lk}|r={}", self.radius_cells);
        if hk != self.host_key {
            self.host_key = hk;
            self.point.clear_for_new_host();
            self.sw.clear_for_new_host();
            for s in [S2Sub::Point, S2Sub::StoneWales] {
                self.built[s.idx()] = None;
            }
        }
        if lk != self.lattice_key {
            // results of the old lattice must neither show nor be exported while the new ones build
            self.lattice_key = lk;
            self.gb.clear_for_new_lattice();
            self.table.out = None;
            self.table.order.clear();
            self.solver.out = None;
            for s in [S2Sub::GrainBoundary, S2Sub::AllDefects, S2Sub::Solver] {
                self.built[s.idx()] = None;
            }
        }
        self.gb.ensure_list(&cx.item.group, cx.cell, &self.lattice_key);
    }

    fn job(&self, cx: &WbCtx) -> S2Job {
        let (cell, radius) = (cx.cell.clone(), self.radius(cx.cell));
        match self.sub {
            S2Sub::Point => S2Job::Point { cell, radius, ops: self.point.ops.clone() },
            S2Sub::StoneWales => S2Job::StoneWales { cell, radius, edge: self.sw.edge, relax: self.sw.relax, layers: self.sw.layers },
            S2Sub::GrainBoundary => {
                let (theta, sigma) = self.gb.params();
                S2Job::Boundary { group: cx.item.group.clone(), cell, theta, sigma, radius }
            }
            S2Sub::AllDefects => S2Job::Table { item: Box::new(cx.item.clone()), cell, max_sigma: self.gb.max_sigma },
            S2Sub::Solver => {
                let s = &self.solver;
                S2Job::Solver { group: cx.item.group.clone(), cell, nx: s.nx, ny: s.ny, shells: s.shells }
            }
        }
    }

    fn accept(&mut self, res: Result<Result<S2Out, String>, String>) {
        let Some((sub, key)) = self.pending.take() else { return };
        match res.and_then(|r| r) {
            Ok(out) => {
                self.errors[sub.idx()] = None;
                self.built[sub.idx()] = Some(key);
                match out {
                    S2Out::Point(o) => self.point.out = Some(o),
                    S2Out::StoneWales(o) => self.sw.out = Some(o),
                    S2Out::Boundary(o) => self.gb.out = Some(o),
                    S2Out::Table(o) => self.table.set_out(o),
                    S2Out::Solver(o) => self.solver.out = Some(o),
                }
            }
            Err(e) => {
                self.errors[sub.idx()] = Some(e);
                self.built[sub.idx()] = None;
            }
        }
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        const LABELS: [&str; 5] = ["Point defects", "Stone-Wales", "CSL / twin GB", "All defects", "Solver export"];
        &LABELS
    }
    fn sub_index(&self) -> usize {
        self.sub as usize
    }
    /// The body of the sub-row click branch in `show_controls`.
    fn select_sub(&mut self, i: usize) -> bool {
        match S2Sub::ALL.get(i) {
            Some(&s) if self.sub != s => {
                self.sub = s;
                true
            }
            _ => false,
        }
    }
    fn show_controls(&mut self, ui: &mut egui::Ui, cx: &WbCtx) -> bool {
        let mut rebuild = false;
        ui.horizontal_wrapped(|ui| {
            for s in S2Sub::ALL {
                if ui.selectable_label(self.sub == s, s.label()).clicked() && self.sub != s {
                    self.sub = s;
                    rebuild = true;
                }
            }
        });
        ui.separator();
        if self.sub.uses_disk() {
            ui.horizontal(|ui| {
                ui.label("Host disk radius (cells):");
                rebuild |= ui.add(egui::DragValue::new(&mut self.radius_cells).range(2.0..=6.0).speed(0.05)).changed();
            });
        }
        rebuild |= match self.sub {
            S2Sub::Point => self.point.show_controls(ui),
            S2Sub::StoneWales => self.sw.show_controls(ui),
            S2Sub::GrainBoundary => self.gb.show_controls(ui),
            S2Sub::AllDefects => self.table.show_controls(ui, cx),
            S2Sub::Solver => self.solver.show_controls(ui),
        };
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, _tokens: &ThemeTokens) {
        ui.label(format!("Host lattice: {} ({})", cx.item.label, cx.item.group));
        ui.separator();
        match self.sub {
            S2Sub::Point => self.point.show_inspector(ui),
            S2Sub::StoneWales => self.sw.show_inspector(ui),
            S2Sub::GrainBoundary => self.gb.show_inspector(ui),
            S2Sub::AllDefects => self.table.show_inspector(ui),
            S2Sub::Solver => self.solver.show_inspector(ui, cx, self.point.out.as_ref()),
        }
    }

    fn submit(&mut self, cx: &WbCtx, repaint: Option<&egui::Context>) {
        self.track_host(cx);
        let job = self.job(cx);
        let key = format!("{job:?}");
        let i = self.sub.idx();
        let in_flight = self.pending.as_ref().is_some_and(|(s, k)| *s == self.sub && *k == key);
        if in_flight || self.built[i].as_deref() == Some(key.as_str()) {
            return;
        }
        self.errors[i] = None;
        let worker = self.worker.get_or_insert_with(|| LatestWorker::spawn("wb-s2", (), |_: &mut (), job| run_job(job)));
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
        let (scene2d, mesh, caption) = match self.sub {
            S2Sub::Point => self.point.scenes(),
            S2Sub::StoneWales => (self.sw.scene(), None, String::new()),
            S2Sub::GrainBoundary => (self.gb.scene(), None, String::new()),
            S2Sub::AllDefects => (empty_scene(table::SCENE_TITLE, "the sortable table is in the inspector"), None, String::new()),
            S2Sub::Solver => (self.solver.scene(), None, String::new()),
        };
        WbScenes { scene2d, mesh, caption }
    }

    fn is_busy(&self) -> bool {
        self.worker.as_ref().is_some_and(LatestWorker::is_busy)
    }

    fn error(&self) -> Option<&str> {
        self.errors[self.sub.idx()].as_deref()
    }

    fn lattice_dependent(&self) -> bool {
        true
    }

    fn on_canvas(&mut self, i: &CanvasInteraction, _cx: &WbCtx) -> bool {
        let Some(w) = i.clicked_world else { return false };
        match self.sub {
            S2Sub::Point => self.point.click(w),
            S2Sub::StoneWales => self.sw.click(w),
            _ => false,
        }
    }

    fn save(&self) -> Value {
        json!({
            "sub": format!("{:?}", self.sub),
            "radius_cells": self.radius_cells,
            "sw_relax": self.sw.relax,
            "sw_layers": self.sw.layers,
            "gb": self.gb.save(),
            "table": self.table.save(),
            "solver": self.solver.save(),
        })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("S2 panel state is not an object (ignored)"));
            return;
        };
        if let Some(s) = o.get("sub").and_then(Value::as_str) {
            match S2Sub::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) => self.sub = x,
                None => notes.push(format!("S2 panel: unknown sub-view {s:?} (kept current)")),
            }
        }
        if let Some(r) = o.get("radius_cells").and_then(Value::as_f64).filter(|r| r.is_finite()) {
            self.radius_cells = r.clamp(2.0, 6.0);
        }
        if let Some(b) = o.get("sw_relax").and_then(Value::as_bool) {
            self.sw.relax = b;
        }
        if let Some(n) = o.get("sw_layers").and_then(Value::as_u64) {
            self.sw.layers = n.min(u64::from(crate::defects_ui::growth::MAX_LAYERS)) as u32;
        }
        if let Some(g) = o.get("gb") {
            self.gb.restore(g);
        }
        if let Some(t) = o.get("table") {
            self.table.restore(t);
        }
        if let Some(s) = o.get("solver") {
            self.solver.restore(s);
        }
    }
}
