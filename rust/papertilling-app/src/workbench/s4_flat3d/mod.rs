//! S4 workbench panel (Lattices -> "Flat 3D: Tubes, Ribbons, Flakes"): flat 3D slabs on the five non-compact
//! topologies, finite quotients on the ten closed platycosms, chiral nanotubes and nanoribbons and symmetry-centred
//! flakes of the Gallery lattice, and the exact torsion-free classification of the 230 space groups.
//!
//! Every build runs on one lazily spawned [`LatestWorker`]; only the static platycosm table is read on the UI thread.
//! Each sub-row keeps its last result keyed by the job's inputs; a late result for inputs no longer selected is
//! dropped (going back to the shown inputs cancels the job in flight). A refusal (bad input, or a core `IMPOSSIBLE` /
//! `NOT_FOUND` / error) clears that sub-row's result: the reason is shown verbatim and no geometry is drawn.

use papertilling_core::flat3d::parse_canonical_net_3d;
use papertilling_core::space3d::nets3d::CanonicalNet3d;
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::{CanvasInteraction, Scene2D};
use crate::combine_worker::LatestWorker;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::theme::ThemeTokens;
use crate::workbench::scene_helpers::vector_scene;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

mod flakes;
mod jobs;
mod platycosms;
mod slabs;
#[cfg(test)]
mod tests;
mod torsion_free;
mod tubes;

use jobs::{S4Job, S4Out, run_job};

/// 2D scene, optional 3D mesh and caption of one sub-row.
type Scenes = (Scene2D, Option<GpuMeshData>, String);

/// The panel's inner sub-row. Labels are unique on screen (kittest clicks them by exact label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum S4Sub {
    Slabs,
    Platycosms,
    #[default]
    Nanotubes,
    Ribbons,
    Flakes,
    TorsionFree,
}

impl S4Sub {
    pub(crate) const ALL: [Self; 6] = [Self::Slabs, Self::Platycosms, Self::Nanotubes, Self::Ribbons, Self::Flakes, Self::TorsionFree];
    /// The sub-rows the user can reach. C1 (D19): 3D slabs and platycosms embed a 3D net on a 3D manifold (torus,
    /// Klein, Mobius tube, cylinder x S¹, chimney, the ten closed platycosms), which leaves the UI in favour of
    /// growth by inflation; their code and the indices in `ALL` stay. Torsion-free groups is Bieberbach group data.
    pub(crate) const SHOWN: [Self; 4] = [Self::Nanotubes, Self::Ribbons, Self::Flakes, Self::TorsionFree];

    pub(crate) fn is_shown(self) -> bool {
        Self::SHOWN.contains(&self)
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Slabs => "3D slabs",
            Self::Platycosms => "Platycosms",
            Self::Nanotubes => "Nanotubes",
            Self::Ribbons => "Nanoribbons",
            Self::Flakes => "Flakes",
            Self::TorsionFree => "Torsion-free groups",
        }
    }

    fn idx(self) -> usize {
        self as usize
    }

    /// Built from the Gallery's 2D lattice (rebuilt on a lattice change).
    fn uses_lattice(self) -> bool {
        matches!(self, Self::Nanotubes | Self::Ribbons | Self::Flakes)
    }
}

// ---- small shared UI / scene helpers used by the sub-row files ----

fn net_combo(ui: &mut egui::Ui, id: &str, net: &mut CanonicalNet3d) -> bool {
    let mut ch = false;
    egui::ComboBox::from_id_salt(id).selected_text(net.label()).show_ui(ui, |ui| {
        for n in CanonicalNet3d::ALL {
            ch |= ui.selectable_value(net, n, n.label()).changed();
        }
    });
    ch
}

fn parse_net(v: Option<&Value>, notes: &mut Vec<String>) -> Option<CanonicalNet3d> {
    let s = v?.as_str()?;
    match parse_canonical_net_3d(s) {
        Ok(n) => Some(n),
        Err(e) => {
            notes.push(format!("S4 panel: {e} (kept current)"));
            None
        }
    }
}

fn int_row(ui: &mut egui::Ui, label: &str, v: &mut usize, lo: usize, hi: usize) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(v).range(lo..=hi)).changed()
    })
    .inner
}

fn repeats_row(ui: &mut egui::Ui, label: &str, r: &mut [usize; 3], hi: usize) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let mut ch = false;
        for x in r.iter_mut() {
            ch |= ui.add(egui::DragValue::new(x).range(1..=hi)).changed();
        }
        ch
    })
    .inner
}

/// A JSON `[a, b, c]` of integers clamped to `1..=hi`.
fn read_triple(v: Option<&Value>, hi: usize) -> Option<[usize; 3]> {
    let a = v?.as_array().filter(|a| a.len() == 3)?;
    let c = |x: &Value| x.as_u64().map(|k| (k as usize).clamp(1, hi));
    Some([c(&a[0])?, c(&a[1])?, c(&a[2])?])
}

/// Sites and bonds in the plane; `ringed` sites get an orange outline.
fn points_scene(xy: &[[f64; 2]], edges: &[(usize, usize)], ringed: &[usize], title: &str, note: &str) -> Scene2D {
    let mut s = vector_scene(xy, edges, &[], |_| [0, 0, 0, 0], title, note);
    if let Scene2D::VectorScene { primitives } = &mut s {
        for &i in ringed {
            if let Some(Some(c)) = primitives.site_circle_indices.get(i).copied() {
                primitives.circles[c].stroke = ([230, 126, 34, 255], 2.0);
            }
        }
    }
    s
}

/// Scenes of a sub-row with no result: its refusal, or a building placeholder.
fn no_result_scenes(sub: S4Sub, error: Option<&str>) -> Scenes {
    match error {
        Some(e) => (empty_scene(sub.label(), &format!("Refused, nothing drawn: {e}")), None, String::new()),
        None if sub == S4Sub::TorsionFree => (empty_scene(torsion_free::SCENE_TITLE, "the table is in the inspector"), None, String::new()),
        None => (empty_scene(sub.label(), "building..."), Some(placeholder_mesh()), format!("{}: building (placeholder tetrahedron)", sub.label())),
    }
}

fn lattice_key(cx: &WbCtx) -> String {
    format!("{}|{:?}|{:?}|{:?}|{:?}|{:?}", cx.item.group, cx.cell.a1, cx.cell.a2, cx.cell.basis, cx.cell.bonds, cx.item.frame.origin)
}

type S4Worker = LatestWorker<S4Job, Result<S4Out, String>>;

#[derive(Default)]
pub(crate) struct Panel {
    sub: S4Sub,
    slab: slabs::SlabState,
    platy: platycosms::PlatyState,
    tube: tubes::TubeState,
    ribbon: tubes::RibbonState,
    flake: flakes::FlakeState,
    torsion: torsion_free::TorsionState,
    lattice_key: String,
    /// A table row was clicked in the inspector: rebuild on the next controls frame.
    inspector_rebuild: bool,
    worker: Option<S4Worker>,
    /// Sub-row and input key of the job in flight.
    pending: Option<(S4Sub, String)>,
    /// Per sub-row, the input key last asked for by `submit`; a worker result for any other key is dropped.
    wanted: [Option<String>; 6],
    built: [Option<String>; 6],
    errors: [Option<String>; 6],
}

impl Panel {
    fn validate(&self) -> Result<(), String> {
        match self.sub {
            S4Sub::Slabs => self.slab.validate(),
            S4Sub::Platycosms => self.platy.validate(),
            S4Sub::Nanotubes => self.tube.validate(),
            S4Sub::Ribbons => self.ribbon.validate(),
            S4Sub::Flakes => self.flake.validate(),
            S4Sub::TorsionFree => Ok(()),
        }
    }

    fn job(&self, cx: &WbCtx) -> S4Job {
        let cell: Cell2D = cx.cell.clone();
        match self.sub {
            S4Sub::Slabs => S4Job::Slab { net: self.slab.net, kind: self.slab.kind, repeats: self.slab.repeats },
            S4Sub::Platycosms => S4Job::Platy {
                net: self.platy.net,
                code: self.platy.code.clone(),
                supercell: self.platy.supercell_opt(),
                ignore_kinds: self.platy.ignore_kinds,
            },
            S4Sub::Nanotubes => S4Job::Tube { cell, p: self.tube.p.clone() },
            S4Sub::Ribbons => S4Job::Ribbon { cell, p: self.ribbon.p.clone() },
            S4Sub::Flakes => {
                let centre = self.flake.p.centre.resolve(cx.cell, cx.item.frame.origin);
                S4Job::Flake { group: cx.item.group.clone(), cell, centre, p: self.flake.p.clone() }
            }
            S4Sub::TorsionFree => S4Job::Torsion,
        }
    }

    fn clear_out(&mut self, sub: S4Sub) {
        match sub {
            S4Sub::Slabs => self.slab.out = None,
            S4Sub::Platycosms => self.platy.out = None,
            S4Sub::Nanotubes => self.tube.out = None,
            S4Sub::Ribbons => self.ribbon.out = None,
            S4Sub::Flakes => self.flake.out = None,
            S4Sub::TorsionFree => self.torsion.out = None,
        }
    }

    /// Results built from another Gallery lattice must neither show nor be exported (called first in `submit`).
    fn track_lattice(&mut self, cx: &WbCtx) {
        let lk = lattice_key(cx);
        if lk == self.lattice_key {
            return;
        }
        self.lattice_key = lk;
        self.flake.clear_for_new_lattice();
        if self.pending.as_ref().is_some_and(|(s, _)| s.uses_lattice()) {
            self.forget_pending();
        }
        for s in S4Sub::ALL.into_iter().filter(|s| s.uses_lattice()) {
            self.clear_out(s);
            self.built[s.idx()] = None;
            self.errors[s.idx()] = None;
            self.wanted[s.idx()] = None;
        }
    }

    /// Cancel the job in flight: its late result is ignored by the worker and, with `pending` cleared, by `accept`.
    fn forget_pending(&mut self) {
        self.pending = None;
        if let Some(w) = self.worker.as_mut() {
            w.cancel();
        }
    }

    fn refuse(&mut self, reason: String) {
        let i = self.sub.idx();
        if self.pending.as_ref().is_some_and(|(s, _)| *s == self.sub) {
            self.forget_pending();
        }
        self.clear_out(self.sub);
        self.built[i] = None;
        self.errors[i] = Some(reason);
        self.wanted[i] = None;
    }

    fn accept(&mut self, res: Result<Result<S4Out, String>, String>) {
        let Some((sub, key)) = self.pending.take() else { return };
        let i = sub.idx();
        if self.wanted[i].as_deref() != Some(key.as_str()) {
            return; // built for inputs that are no longer selected: never shown
        }
        match res.and_then(|r| r) {
            Ok(out) => {
                self.errors[i] = None;
                self.built[i] = Some(key);
                match out {
                    S4Out::Slab(o, glide) => {
                        self.slab.out = Some(*o);
                        self.slab.glide = glide;
                    }
                    S4Out::Platy(o) => self.platy.out = Some(*o),
                    S4Out::Tube(o) => self.tube.out = Some(*o),
                    S4Out::Ribbon(o) => self.ribbon.out = Some(*o),
                    S4Out::Flake(o, shape) => {
                        self.flake.out = Some(*o);
                        self.flake.out_shape = shape;
                    }
                    S4Out::Torsion(o) => self.torsion.out = Some(o),
                }
            }
            Err(e) => {
                self.clear_out(sub);
                self.built[i] = None;
                self.errors[i] = Some(e);
            }
        }
    }
}

impl WbPanel for Panel {
    fn sub_labels(&self) -> &'static [&'static str] {
        const LABELS: [&str; 6] = ["3D slabs", "Platycosms", "Nanotubes", "Nanoribbons", "Flakes", "Torsion-free groups"];
        &LABELS
    }
    fn sub_index(&self) -> usize {
        self.sub as usize
    }
    /// The body of the sub-row click branch in `show_controls`.
    fn select_sub(&mut self, i: usize) -> bool {
        match S4Sub::ALL.get(i) {
            Some(&s) if self.sub != s && s.is_shown() => {
                self.sub = s;
                true
            }
            _ => false,
        }
    }
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        let mut rebuild = std::mem::take(&mut self.inspector_rebuild);
        ui.horizontal_wrapped(|ui| {
            for s in S4Sub::SHOWN {
                if ui.selectable_label(self.sub == s, s.label()).clicked() && self.sub != s {
                    self.sub = s;
                    rebuild = true;
                }
            }
        });
        ui.separator();
        rebuild |= match self.sub {
            S4Sub::Slabs => self.slab.show_controls(ui),
            S4Sub::Platycosms => self.platy.show_controls(ui),
            S4Sub::Nanotubes => self.tube.show_controls(ui),
            S4Sub::Ribbons => self.ribbon.show_controls(ui),
            S4Sub::Flakes => self.flake.show_controls(ui),
            S4Sub::TorsionFree => {
                self.torsion.show_controls(ui);
                false
            }
        };
        if self.sub.uses_lattice() {
            ui.small("Built from the Gallery's current 2D lattice.");
        }
        rebuild
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, _tokens: &ThemeTokens) {
        if self.sub.uses_lattice() {
            ui.label(format!("Source lattice: {} ({})", cx.item.label, cx.item.group));
            ui.separator();
        }
        match self.sub {
            S4Sub::Slabs => self.slab.show_inspector(ui),
            S4Sub::Platycosms => self.inspector_rebuild |= self.platy.show_inspector(ui),
            S4Sub::Nanotubes => self.tube.show_inspector(ui),
            S4Sub::Ribbons => self.ribbon.show_inspector(ui),
            S4Sub::Flakes => self.flake.show_inspector(ui),
            S4Sub::TorsionFree => self.torsion.show_inspector(ui),
        }
        if self.inspector_rebuild {
            ui.ctx().request_repaint();
        }
    }

    fn submit(&mut self, cx: &WbCtx, repaint: Option<&egui::Context>) {
        self.track_lattice(cx);
        if let Err(e) = self.validate() {
            self.refuse(e);
            return;
        }
        let job = self.job(cx);
        let key = format!("{job:?}");
        let i = self.sub.idx();
        self.wanted[i] = Some(key.clone());
        if self.pending.as_ref().is_some_and(|(s, k)| *s == self.sub && *k == key) {
            return; // already in flight
        }
        if self.built[i].as_deref() == Some(key.as_str()) {
            // back to the shown result: a job for other inputs of this sub-row must not replace it when it lands
            if self.pending.as_ref().is_some_and(|(s, _)| *s == self.sub) {
                self.forget_pending();
            }
            return;
        }
        self.errors[i] = None;
        let worker = self.worker.get_or_insert_with(|| LatestWorker::spawn("wb-s4", (), |_: &mut (), job| run_job(job)));
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
        let got = match self.sub {
            S4Sub::Slabs => self.slab.scenes(),
            S4Sub::Platycosms => self.platy.scenes(),
            S4Sub::Nanotubes => self.tube.scenes(),
            S4Sub::Ribbons => self.ribbon.scenes(),
            S4Sub::Flakes => self.flake.scenes(),
            S4Sub::TorsionFree => None,
        };
        let (scene2d, mesh, caption) = got.unwrap_or_else(|| no_result_scenes(self.sub, self.error()));
        WbScenes { scene2d, mesh, caption }
    }

    fn is_busy(&self) -> bool {
        self.worker.as_ref().is_some_and(LatestWorker::is_busy)
    }

    fn error(&self) -> Option<&str> {
        self.errors[self.sub.idx()].as_deref()
    }

    fn lattice_dependent(&self) -> bool {
        self.sub.uses_lattice()
    }

    fn on_canvas(&mut self, i: &CanvasInteraction, _cx: &WbCtx) -> bool {
        match (self.sub, i.clicked_world) {
            (S4Sub::Flakes, Some(w)) => self.flake.click(w),
            _ => false,
        }
    }

    fn save(&self) -> Value {
        json!({
            "sub": format!("{:?}", self.sub),
            "slab": self.slab.save(),
            "platy": self.platy.save(),
            "tube": self.tube.save(),
            "ribbon": self.ribbon.save(),
            "flake": self.flake.save(),
            "torsion": self.torsion.save(),
        })
    }

    fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let Some(o) = v.as_object() else {
            notes.push(String::from("S4 panel state is not an object (ignored)"));
            return;
        };
        if let Some(s) = o.get("sub").and_then(Value::as_str) {
            match S4Sub::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) if !x.is_shown() => {
                    self.sub = S4Sub::default();
                    notes.push(format!("S4 panel: {} is hidden (3D embeddings left the app); showing {}", x.label(), self.sub.label()));
                }
                Some(x) => self.sub = x,
                None => notes.push(format!("S4 panel: unknown sub-view {s:?} (kept current)")),
            }
        }
        if let Some(x) = o.get("slab") {
            self.slab.restore(x, notes);
        }
        if let Some(x) = o.get("platy") {
            self.platy.restore(x, notes);
        }
        if let Some(x) = o.get("tube") {
            self.tube.restore(x);
        }
        if let Some(x) = o.get("ribbon") {
            self.ribbon.restore(x);
        }
        if let Some(x) = o.get("flake") {
            self.flake.restore(x, notes);
        }
        if let Some(x) = o.get("torsion") {
            self.torsion.restore(x);
        }
    }

    fn smoke_defaults(&mut self) {
        // C1: the slab sub-rows are hidden, so the smoke stage builds the default nanotube of the Gallery lattice
        self.sub = S4Sub::Nanotubes;
    }
}
