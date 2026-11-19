//! Host of the five stage panels (S2..S6) behind the [`WbPanel`] trait. `NavTab::Workbench` is only the internal
//! route: the user reaches every sub-row through its Command Deck card (`crate::game_menu`) and never sees the name.
//!
//! The shared app files route here once (P0 prewire). After that, each stage only replaces the files inside its own
//! `sN_*` folder: the panel type there is always `Panel` (with `Default`) implementing [`WbPanel`].
//!
//! Life cycle of the active panel, driven by this host:
//! - entering the view (`on_enter`): `submit`, then the scenes are pushed (a cached result or a placeholder);
//! - a lattice change (`on_lattice_changed`): `submit` again only if `lattice_dependent()`, then the scenes are
//!   re-pushed (the Gallery redraw overwrote the shared canvas);
//! - every frame (`show_central`): `poll`; when a result arrived (or the view is dirty and idle) the scenes are
//!   pushed and `dirty` is cleared; `dirty` stays true while the panel is busy;
//! - `show_controls` returning true, or `on_canvas` returning true: `submit` (sync build or worker job).

use std::path::Path;

use papertilling_core::types::{BuiltCrystal, Cell2D};
use serde_json::Value;

use crate::app::{DefectsView, LatticesView};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasInteraction, Scene2D};
use crate::mesh3d::GpuMeshData;
use crate::session::{name_of, parse_named};
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::show_split_2d_3d;
use crate::view3d::View3dState;

mod s2_defects;
mod s3_detect;
mod s4_flat3d;
mod s5_curved2d;
mod s6_honeycombs;
pub(crate) mod scene_helpers;
#[cfg(test)]
mod tests;

/// Caption of the empty 3D mesh pushed for 2D-only views (so a stale 3D mesh is never exported).
pub(crate) const CAPTION_2D_ONLY: &str = "2D view only";

/// The five stage panels hosted by the Workbench tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum WorkbenchView {
    #[default]
    S2Defects,
    S3Detect,
    S4Flat3d,
    S5Curved2d,
    S6Honeycombs,
}

/// The mode sub-view that routes to a workbench view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModeView {
    Lattices(LatticesView),
    Defects(DefectsView),
}

impl WorkbenchView {
    pub(crate) const ALL: [Self; 5] = [Self::S2Defects, Self::S3Detect, Self::S4Flat3d, Self::S5Curved2d, Self::S6Honeycombs];

    pub(crate) fn mode_view(self) -> ModeView {
        match self {
            Self::S2Defects => ModeView::Defects(DefectsView::PointAndBoundary),
            Self::S3Detect => ModeView::Defects(DefectsView::DetectFields),
            Self::S4Flat3d => ModeView::Lattices(LatticesView::Flat3D),
            Self::S5Curved2d => ModeView::Lattices(LatticesView::Curved2D),
            Self::S6Honeycombs => ModeView::Lattices(LatticesView::Honeycombs3D),
        }
    }

    /// Whether the view always shows a 3D pane (S2 may still add one, e.g. for adatoms, through `WbScenes::mesh`).
    pub(crate) fn uses_3d(self) -> bool {
        matches!(self, Self::S4Flat3d | Self::S5Curved2d | Self::S6Honeycombs)
    }

    /// Key of this panel's entry in the session JSON.
    fn key(self) -> &'static str {
        match self {
            Self::S2Defects => "s2",
            Self::S3Detect => "s3",
            Self::S4Flat3d => "s4",
            Self::S5Curved2d => "s5",
            Self::S6Honeycombs => "s6",
        }
    }

    /// Realm-native heading of the panel (never equal to a sub-row or card label: kittest `get_by_label` needs those
    /// unique; never names the internal "Workbench" route).
    pub(crate) fn heading(self) -> &'static str {
        match self {
            Self::S2Defects => "Defects Forge › point defects, Stone-Wales, grain boundaries",
            Self::S3Detect => "Defects Forge › detection and elastic fields",
            Self::S4Flat3d => "Manifolds › flat 3D slabs, tubes, flakes, platycosms",
            Self::S5Curved2d => "Geometries › {p,q} tilings on S², E² and H²",
            Self::S6Honeycombs => "Geometries › 3D honeycombs, TPMS and Dirichlet cells",
        }
    }
}

impl LatticesView {
    pub(crate) fn workbench_view(self) -> Option<WorkbenchView> {
        WorkbenchView::ALL.into_iter().find(|w| w.mode_view() == ModeView::Lattices(self))
    }
}

impl DefectsView {
    pub(crate) fn workbench_view(self) -> Option<WorkbenchView> {
        WorkbenchView::ALL.into_iter().find(|w| w.mode_view() == ModeView::Defects(self))
    }
}

/// What a panel may read from the rest of the app (built per call from disjoint `PapertillingApp` fields).
#[allow(dead_code)] // P0 prewire: the stub panels read only some fields; stage panels S2..S6 read the rest.
pub(crate) struct WbCtx<'a> {
    /// The Gallery's active lattice (slider-tuned recipe).
    pub item: &'a CatalogueItemUi,
    /// Its periodic cell (`gallery.cell()`).
    pub cell: &'a Cell2D,
    /// `--smoke-test` run: never open a file dialog.
    pub smoke: bool,
    pub backend: &'a AppBackend,
    /// The Defects-tab crystal (`defects.last_crystal`), kept in sync for S3 by `workbench_defect_source_synced`.
    pub defect_crystal: Option<&'a BuiltCrystal>,
}

/// The scenes a panel shows: `mesh: None` means a 2D-only view (the host pushes an empty "2D view only" mesh).
pub(crate) struct WbScenes {
    pub scene2d: Scene2D,
    pub mesh: Option<GpuMeshData>,
    pub caption: String,
}

/// One stage panel. A 3D panel must never return an empty mesh while idle (return the previous mesh or
/// `wb_mesh::placeholder_mesh()`); captions must never contain "unavailable", "Build failed" or "stale".
pub(crate) trait WbPanel {
    /// Left rail controls; true = rebuild (the host then calls `submit`).
    fn show_controls(&mut self, ui: &mut egui::Ui, cx: &WbCtx) -> bool;
    fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, tokens: &ThemeTokens);
    /// Start a build: synchronous for tiny calls, otherwise a lazily spawned `LatestWorker` job.
    fn submit(&mut self, cx: &WbCtx, repaint: Option<&egui::Context>);
    /// True when a worker result (or error) arrived since the last call.
    fn poll(&mut self) -> bool;
    fn scenes(&self) -> WbScenes;
    fn is_busy(&self) -> bool;
    fn error(&self) -> Option<&str>;
    /// Rebuild on a Gallery lattice change.
    fn lattice_dependent(&self) -> bool {
        false
    }
    /// A 2D canvas click / hover (2D-only views); true = rebuild.
    fn on_canvas(&mut self, _i: &CanvasInteraction, _cx: &WbCtx) -> bool {
        false
    }
    fn save(&self) -> Value {
        Value::Null
    }
    fn restore(&mut self, _v: &Value, _notes: &mut Vec<String>) {}
    /// Small, fast settings for the `--smoke-test` stage of this view.
    fn smoke_defaults(&mut self) {}
    /// Labels of the panel's sub-rows, in order (one Command Deck card each).
    fn sub_labels(&self) -> &'static [&'static str] {
        &[]
    }
    fn sub_index(&self) -> usize {
        0
    }
    /// Select sub-row `i` exactly as clicking its selectable label does; true if it changed.
    fn select_sub(&mut self, _i: usize) -> bool {
        false
    }
    /// Breadcrumb geometry text (S5 only).
    fn breadcrumb_geometry(&self) -> Option<&'static str> {
        None
    }
}

pub(crate) struct WorkbenchState {
    pub(crate) view: WorkbenchView,
    /// The shared canvas / 3D view do not show this panel's current scenes yet (or its build is running).
    pub(crate) dirty: bool,
    /// A source changed outside the panel (e.g. the Defects crystal for S3): submit on the next central frame.
    pending_submit: bool,
    reset_view: bool,
    has_mesh: bool,
    s2: s2_defects::Panel,
    s3: s3_detect::Panel,
    s4: s4_flat3d::Panel,
    s5: s5_curved2d::Panel,
    s6: s6_honeycombs::Panel,
}

impl WorkbenchState {
    /// Spawns no thread: panels create their workers lazily on the first `submit`.
    pub(crate) fn new() -> Self {
        Self {
            view: WorkbenchView::default(),
            dirty: true,
            pending_submit: false,
            reset_view: true,
            has_mesh: false,
            s2: Default::default(),
            s3: Default::default(),
            s4: Default::default(),
            s5: Default::default(),
            s6: Default::default(),
        }
    }

    fn panel(&self, v: WorkbenchView) -> &dyn WbPanel {
        match v {
            WorkbenchView::S2Defects => &self.s2,
            WorkbenchView::S3Detect => &self.s3,
            WorkbenchView::S4Flat3d => &self.s4,
            WorkbenchView::S5Curved2d => &self.s5,
            WorkbenchView::S6Honeycombs => &self.s6,
        }
    }

    fn panel_mut(&mut self, v: WorkbenchView) -> &mut dyn WbPanel {
        match v {
            WorkbenchView::S2Defects => &mut self.s2,
            WorkbenchView::S3Detect => &mut self.s3,
            WorkbenchView::S4Flat3d => &mut self.s4,
            WorkbenchView::S5Curved2d => &mut self.s5,
            WorkbenchView::S6Honeycombs => &mut self.s6,
        }
    }

    /// Point the workbench at `view`, sub-row `sub` (as its left-rail click would), and mark it for a rebuild.
    pub(crate) fn select(&mut self, view: WorkbenchView, sub: usize) {
        self.view = view;
        self.panel_mut(view).select_sub(sub);
        self.dirty = true;
    }

    pub(crate) fn sub_labels(&self, view: WorkbenchView) -> &'static [&'static str] {
        self.panel(view).sub_labels()
    }

    pub(crate) fn sub_index(&self, view: WorkbenchView) -> usize {
        self.panel(view).sub_index()
    }

    /// Breadcrumb geometry text of `view` (S5: from its current symbol), if the panel knows one.
    pub(crate) fn breadcrumb_geometry(&self, view: WorkbenchView) -> Option<&'static str> {
        self.panel(view).breadcrumb_geometry()
    }

    pub(crate) fn active(&self) -> &dyn WbPanel {
        self.panel(self.view)
    }

    pub(crate) fn active_mut(&mut self) -> &mut dyn WbPanel {
        self.panel_mut(self.view)
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.active().is_busy()
    }

    pub(crate) fn error(&self) -> Option<&str> {
        self.active().error()
    }

    /// Entering the Workbench tab (or another workbench view): build, then push the scenes at once.
    pub(crate) fn on_enter(&mut self, cx: &WbCtx, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        self.pending_submit = false;
        self.active_mut().submit(cx, None);
        self.reset_view = true;
        self.sync_scenes(canvas, view3d);
    }

    /// The Gallery lattice changed while the Workbench is shown.
    pub(crate) fn on_lattice_changed(&mut self, cx: &WbCtx, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        if self.active().lattice_dependent() {
            self.active_mut().submit(cx, None);
            self.reset_view = true;
        }
        self.sync_scenes(canvas, view3d);
    }

    /// A source outside the panel changed (S3: the Defects crystal arrived): rebuild on the next central frame.
    pub(crate) fn source_changed(&mut self) {
        self.pending_submit = true;
        self.dirty = true;
    }

    /// Push the active panel's scenes into the shared canvas and 3D view; `dirty` stays true while it is busy.
    pub(crate) fn sync_scenes(&mut self, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        let sc = self.active().scenes();
        canvas.set_scene(sc.scene2d, std::mem::take(&mut self.reset_view));
        match sc.mesh {
            Some(mesh) => {
                self.has_mesh = true;
                view3d.set_mesh(mesh, sc.caption);
            }
            None => {
                self.has_mesh = false;
                view3d.set_mesh(GpuMeshData { radius: 1.0, ..Default::default() }, CAPTION_2D_ONLY);
            }
        }
        self.dirty = self.active().is_busy();
    }

    pub(crate) fn show_left_panel(&mut self, ui: &mut egui::Ui, cx: &WbCtx) {
        ui.strong(self.view.heading());
        ui.separator();
        if self.active_mut().show_controls(ui, cx) {
            self.active_mut().submit(cx, Some(ui.ctx()));
            self.dirty = true;
        }
    }

    pub(crate) fn show_inspector_panel(&mut self, ui: &mut egui::Ui, cx: &WbCtx, tokens: &ThemeTokens) {
        if let Some(err) = self.error() {
            ui.colored_label(egui::Color32::from_rgb(192, 57, 43), format!("Refused: {err}"));
            ui.separator();
        }
        self.active_mut().show_inspector(ui, cx, tokens);
    }

    pub(crate) fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        cx: &WbCtx,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        if self.pending_submit {
            self.pending_submit = false;
            self.active_mut().submit(cx, Some(ui.ctx()));
            self.dirty = true;
        }
        let arrived = self.active_mut().poll();
        if arrived || (self.dirty && !self.is_busy()) {
            self.sync_scenes(canvas, view3d);
        }
        let busy = self.is_busy();
        ui.horizontal(|ui| {
            ui.strong(self.view.heading());
            if busy {
                ui.spinner();
            }
        });
        ui.add_space(2.0);
        let mut inter: Option<CanvasInteraction> = None;
        if self.has_mesh || self.view.uses_3d() {
            show_split_2d_3d(
                ui,
                split_ratio,
                "workbench_split",
                tokens,
                |ui| inter = Some(canvas.show_interactive(ui, &[], true, None, tokens)),
                |ui| {
                    view3d.show(ui, tokens);
                },
            );
        } else {
            inter = Some(canvas.show_interactive(ui, &[], true, None, tokens));
        }
        if let Some(i) = inter
            && self.active_mut().on_canvas(&i, cx)
        {
            self.active_mut().submit(cx, Some(ui.ctx()));
            self.dirty = true;
        }
        if busy || self.dirty {
            ui.ctx().request_repaint();
        }
    }

    /// Smoke-test settings of the active view; marks it dirty so the smoke stage waits for its first build.
    pub(crate) fn smoke_defaults(&mut self) {
        self.active_mut().smoke_defaults();
        self.dirty = true;
    }

    /// Session JSON: `{"view": "<WorkbenchView>", "s2": .., ...}` (panels returning `null` are left out).
    pub(crate) fn capture(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert(String::from("view"), Value::String(name_of(self.view)));
        for v in WorkbenchView::ALL {
            let s = self.panel(v).save();
            if !s.is_null() {
                m.insert(String::from(v.key()), s);
            }
        }
        Value::Object(m)
    }

    /// Restore panel states (call before routing to a view, so the routed view wins and enters restored state).
    pub(crate) fn restore(&mut self, v: Option<&Value>, notes: &mut Vec<String>) {
        let Some(v) = v else { return };
        let Some(obj) = v.as_object() else {
            notes.push(String::from("workbench state is not an object (ignored)"));
            return;
        };
        let name = obj.get("view").and_then(Value::as_str);
        match parse_named(&WorkbenchView::ALL, name) {
            Some(w) => self.view = w,
            None if obj.contains_key("view") => notes.push(String::from("unknown workbench view (kept current)")),
            None => {}
        }
        for w in WorkbenchView::ALL {
            if let Some(s) = obj.get(w.key()) {
                self.panel_mut(w).restore(s, notes);
            }
        }
        self.dirty = true;
    }

    pub(crate) fn export_active_obj(&self, path: &Path, view3d: &View3dState) -> Result<String, String> {
        if !self.has_mesh {
            return Err(String::from("This workbench view is 2D only: there is no 3D mesh to export."));
        }
        crate::export_obj::write_gpu_mesh_obj(path, view3d.mesh_data())
    }
}
