//! Hyperbolic `H^2` `{p, q}` tilings, `(p.q.p.q)` kagome, Poincare geodesics, Klein chords,
//! 3D Minkowski hyperboloid bowl, wedge disclinations, 0-based seeded layer growth, and `kappa(D) < 0` inspector.
//!
//! Interactive view (rendering only; the tiling is cached and rebuilt only when `p, q`, style, layers, seed or wedge
//! change): a hyperbolic pan (a disk isometry applied to the 2D disk and the 3D sheet, so the picture stays a valid
//! `{p,q}` tiling), a view zoom of the 2D disk, and face / vertex picking with a readout (`hyperbolic_pick`).

use papertilling_core::growth::{LayerStats, SeedKind};
use papertilling_core::hyperbolic::holonomy::{ReflectionGroup, wedge_angle_mismatch};
use papertilling_core::hyperbolic::{HyperbolicStyle, HyperbolicTiling, build_hyperbolic_tiling_seeded};

use super::hyperbolic_charge_ui::HyperbolicChargeUi;
use super::hyperbolic_scene::{disk_scene, empty_scene, sheet_mesh};
use super::hyperbolic_shells::{SHELL_CACHE_MAX, ShellTable, drawn_patch_stats, shell_table_seeded};
use super::hyperbolic_view::{DiskIsometry, vertex_generations, view_tiling};
use crate::mesh3d::GpuMeshData;
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;

#[cfg(test)]
#[path = "hyperbolic_growth_tests.rs"]
mod hyperbolic_growth_tests;

/// Largest 0-based layer index offered in the Atlas H² slider (`0..=8`).
pub const MAX_UI_H2_LAYERS: usize = 8;

const SEED_FORCED_HOVER: &str =
    "Kagome cluster patches and Δp ≠ 0 wedge cores grow from their own central core (Cell seed only).";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskProjection {
    Poincare,
    Klein,
}

/// A picked element of the patch: a face or a vertex (indices into `HyperbolicTiling::faces` / `poincare_pts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypPick {
    Face(usize),
    Vertex(usize),
}

type TilingKey = (usize, usize, HyperbolicStyle, usize, i32, SeedKind);

/// The origin-frame tiling of a key with its face/vertex generations and complete-shell topological readout.
pub struct CachedTiling {
    pub key: TilingKey,
    pub base: HyperbolicTiling,
    pub face_gen: Vec<usize>,
    pub vertex_gen: Vec<usize>,
    pub drawn_stats: Vec<LayerStats>,
}

pub struct HyperbolicPanelState {
    pub pristine_only: bool,
    pub p: usize,
    pub q: usize,
    pub style: HyperbolicStyle,
    pub layers: usize,
    pub seed: SeedKind,
    pub wedge_delta: i32,
    pub disk_proj: DiskProjection,
    pub status: String,
    /// Hyperbolic pan: the isometry taking the tiling's own frame to the view.
    pub pan: DiskIsometry,
    /// View scale of the 2D disk (1 = the unit disk fills the view).
    pub disk_zoom: f32,
    pub picked: Option<HypPick>,
    pub hovered: Option<HypPick>,
    /// A view option (pan, zoom) or the pick changed: refresh the scenes without resetting the 2D view.
    pub view_changed: bool,
    pub(super) cache: Option<CachedTiling>,
    /// The panned tiling of the last build (what is drawn and picked).
    pub(super) view: Option<HyperbolicTiling>,
    /// Effective `{p, q}` whose shell table the inspector shows.
    pub shell_key: Option<(usize, usize)>,
    /// Uncapped BDF vertex-completion shell counts, at most [`SHELL_CACHE_MAX`] `(p, q, seed)` keys
    /// (oldest dropped first), so Regular <-> Dual and seed toggles do not recompute.
    pub shell_cache: Vec<ShellCacheEntry>,
    /// Which shells the drawn patch contains (from the tiling's `growth_rule`).
    pub drawn_patch_note: String,
    pub(super) charges: HyperbolicChargeUi,
}

pub type ShellCacheEntry = ((usize, usize, SeedKind), Result<ShellTable, String>);

impl Default for HyperbolicPanelState {
    fn default() -> Self {
        Self {
            pristine_only: false,
            p: 7,
            q: 3,
            style: HyperbolicStyle::RegularPQ,
            layers: 2,
            seed: SeedKind::Cell,
            wedge_delta: 0,
            disk_proj: DiskProjection::Poincare,
            status: String::new(),
            pan: DiskIsometry::IDENTITY,
            disk_zoom: 1.0,
            picked: None,
            hovered: None,
            view_changed: false,
            cache: None,
            view: None,
            shell_key: None,
            shell_cache: Vec::new(),
            drawn_patch_note: String::new(),
            charges: HyperbolicChargeUi::default(),
        }
    }
}

impl HyperbolicPanelState {
    pub fn effective_wedge(&self) -> i32 {
        if self.pristine_only { 0 } else { self.wedge_delta }
    }

    /// Active seed: forced to [`SeedKind::Cell`] when `RectifiedKagome` or `effective_wedge() != 0` is active.
    pub fn effective_seed(&self) -> SeedKind {
        if self.style == HyperbolicStyle::RectifiedKagome || self.effective_wedge() != 0 {
            SeedKind::Cell
        } else {
            match self.seed {
                SeedKind::Vertex | SeedKind::Edge | SeedKind::Cell => self.seed,
                SeedKind::Face | SeedKind::Dirichlet => SeedKind::Cell,
            }
        }
    }

    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal_wrapped(|ui| {
            ui.small("Presets:");
            for (lp, lq, st, lbl) in [
                (7, 3, HyperbolicStyle::RegularPQ, "{7,3}"),
                (3, 7, HyperbolicStyle::RegularPQ, "{3,7}"),
                (7, 3, HyperbolicStyle::RectifiedKagome, "(3.7.3.7) Kagome"),
                (5, 4, HyperbolicStyle::RegularPQ, "{5,4}"),
                (8, 3, HyperbolicStyle::RegularPQ, "{8,3}"),
            ] {
                if ui.small_button(lbl).clicked() {
                    self.p = lp;
                    self.q = lq;
                    self.style = st;
                    dirty = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.small("p:");
            dirty |= ui.add(egui::DragValue::new(&mut self.p).range(3..=12)).changed();
            ui.small("q:");
            dirty |= ui.add(egui::DragValue::new(&mut self.q).range(3..=12)).changed();
        });
        let mut lay = self.layers.min(MAX_UI_H2_LAYERS);
        ui.horizontal(|ui| {
            ui.small("Layers:");
            if ui.add(egui::Slider::new(&mut lay, 0..=MAX_UI_H2_LAYERS)).changed() {
                self.layers = lay;
                dirty = true;
            }
        });
        let seed_allowed = self.style != HyperbolicStyle::RectifiedKagome && self.effective_wedge() == 0;
        let eff = self.effective_seed();
        ui.horizontal_wrapped(|ui| {
            ui.small("Seed:");
            for (s, lbl) in [(SeedKind::Vertex, "Vertex"), (SeedKind::Edge, "Edge")] {
                let resp = ui.add_enabled(seed_allowed, egui::Button::selectable(eff == s, lbl));
                if !seed_allowed {
                    resp.on_disabled_hover_text(SEED_FORCED_HOVER);
                } else if resp.clicked() && self.seed != s {
                    self.seed = s;
                    dirty = true;
                }
            }
            let cell_resp = ui.selectable_label(eff == SeedKind::Cell, "Cell");
            if !seed_allowed {
                cell_resp.on_hover_text(SEED_FORCED_HOVER);
            } else if cell_resp.clicked() && self.seed != SeedKind::Cell {
                self.seed = SeedKind::Cell;
                dirty = true;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.small("Tiling:");
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::RegularPQ, "Regular {p,q}").changed();
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::RectifiedKagome, "Kagome").changed();
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::DualQP, "Dual {q,p}").changed();
        });
        ui.horizontal_wrapped(|ui| {
            ui.small("2D Disk:");
            dirty |= ui.selectable_value(&mut self.disk_proj, DiskProjection::Poincare, "Poincare Geodesics").changed();
            dirty |= ui.selectable_value(&mut self.disk_proj, DiskProjection::Klein, "Klein Chords").changed();
        });
        if !self.pristine_only {
            ui.horizontal(|ui| {
                ui.small("Central Wedge Δp:");
                for d in [-1, 0, 1] {
                    dirty |= ui.selectable_value(&mut self.wedge_delta, d, format!("{d:+}")).changed();
                }
            });
        }
        self.show_view_controls(ui);
        dirty
    }

    fn show_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("Disk View: Hyperbolic Pan, Zoom & Picking");
        ui.small("Drag the disk: hyperbolic pan (Mobius). Click: pick a face or vertex.");
        let mut changed = false;
        let r = 0.125_f64.tanh();
        ui.horizontal(|ui| {
            ui.small("Pan:");
            for (lbl, d) in [("←", [-r, 0.0]), ("→", [r, 0.0]), ("↑", [0.0, r]), ("↓", [0.0, -r])] {
                if ui.small_button(lbl).clicked() {
                    self.pan = self.pan.then(DiskIsometry::from_origin(d));
                    changed = true;
                }
            }
            if ui.small_button("Reset").on_hover_text("Undo the hyperbolic pan and zoom").clicked() {
                self.pan = DiskIsometry::IDENTITY;
                self.disk_zoom = 1.0;
                changed = true;
            }
        });
        changed |= ui.add(egui::Slider::new(&mut self.disk_zoom, 0.5..=4.0).logarithmic(true).text("Disk zoom")).changed();
        ui.horizontal(|ui| {
            if self.picked.is_some()
                && ui.small_button("Center on pick").clicked()
                && let Some(z) = self.picked_view_point()
            {
                self.pan = self.pan.then(DiskIsometry::to_origin(z));
                changed = true;
            }
            if self.picked.is_some() && ui.small_button("Clear pick").clicked() {
                self.picked = None;
                changed = true;
            }
        });
        self.view_changed |= changed;
    }

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("Hyperbolic H² Metric & Status");
        let is_hyperbolic = papertilling_core::hyperbolic::inflation::is_hyperbolic(self.p, self.q);
        if !self.status.is_empty() {
            let col = if is_hyperbolic { tokens.status_ok } else { tokens.status_err };
            ui.colored_label(col, egui::RichText::new(&self.status).small());
        }
        if let Some(cache) = &self.cache
            && cache.base.complete_through_shell < self.layers
        {
            ui.colored_label(
                tokens.status_warn,
                egui::RichText::new(format!(
                    "Stopped at layer {} (requested {}): {}",
                    cache.base.complete_through_shell, self.layers, cache.base.growth_rule
                ))
                .small()
                .strong(),
            );
        }
        let (p, q) = if self.style == HyperbolicStyle::DualQP { (self.q, self.p) } else { (self.p, self.q) };
        if self.pristine_only {
            ui.small(format!(
                "Schläfli {{{}, {}}} | Seed: {:?} | Layers: {}",
                p,
                q,
                self.effective_seed(),
                self.layers
            ));
        } else {
            ui.small(format!(
                "Schläfli {{{}, {}}} | Seed: {:?} | Layers: {} | Wedge Δp: {:+}",
                p,
                q,
                self.effective_seed(),
                self.layers,
                self.effective_wedge()
            ));
        }

        if is_hyperbolic {
            if self.effective_wedge() == 0 && self.style != HyperbolicStyle::RectifiedKagome {
                let orbit = if p > q { "face centres" } else { "vertices" };
                ui.small(format!("Blue sites lie at {orbit}. One site is shared by {} reflection chambers.", 2 * p.max(q)));
            }
            ui.small("Intrinsic curvature K = −1.");
            if !self.pristine_only {
                let delta = self.effective_wedge();
                let core = (p as i32 + delta).clamp(3, 12) as usize;
                if delta != 0 && let Some(mismatch) = wedge_angle_mismatch(p, q, core) {
                    ui.small(format!("Attempted {q}-fold join · angular mismatch {mismatch:+.4} rad"));
                    ui.small("Positive means a gap. Negative means overlap. The wedge sketch shows unglued polygon copies.");
                }
                ui.separator();
                self.charges.show(ui, ReflectionGroup::h2(p, q), false);
            }
        }
        self.show_pick_readout(ui, tokens);
        self.show_drawn_growth_table(ui, tokens);
        self.show_shell_table(ui);
    }

    fn key(&self) -> TilingKey {
        (self.p, self.q, self.style, self.layers, self.effective_wedge(), self.effective_seed())
    }

    /// Build (or reuse) the origin-frame tiling of the current key and update the shell table cache;
    /// a new key clears the pick.
    fn ensure_tiling(&mut self) -> Result<(), String> {
        let (pe, qe) = if self.style == HyperbolicStyle::DualQP { (self.q, self.p) } else { (self.p, self.q) };
        let eff_seed = self.effective_seed();
        if !self.shell_cache.iter().any(|(k, _)| *k == (pe, qe, eff_seed)) {
            let res = shell_table_seeded(pe, qe, eff_seed);
            if self.shell_cache.len() >= SHELL_CACHE_MAX {
                self.shell_cache.drain(..1);
            }
            self.shell_cache.push(((pe, qe, eff_seed), res));
        }
        self.shell_key = Some((pe, qe));
        let key = self.key();
        if self.cache.as_ref().is_some_and(|c| c.key == key) {
            return Ok(());
        }
        self.cache = None;
        self.picked = None;
        self.hovered = None;
        let base = match build_hyperbolic_tiling_seeded(
            self.p,
            self.q,
            self.style,
            self.layers,
            self.effective_wedge(),
            eff_seed,
        ) {
            Ok(b) => b,
            Err(e) => {
                self.drawn_patch_note.clear();
                return Err(e);
            }
        };
        self.drawn_patch_note = base.growth_rule.clone();
        let face_gen = base.face_shell.clone();
        let vertex_gen = vertex_generations(base.poincare_pts.len(), &base.faces, &face_gen);
        let drawn_stats = drawn_patch_stats(&base.faces, &base.face_shell, base.complete_through_shell);
        self.cache = Some(CachedTiling { key, base, face_gen, vertex_gen, drawn_stats });
        Ok(())
    }

    pub fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        if let Err(e) = self.ensure_tiling() {
            self.status = e;
            self.view = None;
        }
        let Some(cache) = self.cache.as_ref() else {
            return (
                empty_scene("Non-hyperbolic (p, q)", &self.status),
                GpuMeshData { radius: 1.0, ..Default::default() },
                self.status.clone(),
            );
        };
        let base = &cache.base;
        self.status = format!(
            "{} | κ(D) = {:.4} ({:?}) | PSU(1,1) |Tr/2|={:.3}",
            base.symbol_label, base.curvature_kappa, base.geometry_kind, base.psu11_half_trace
        );
        let hyp = view_tiling(base, self.pan);
        let scene = disk_scene(&hyp, self.disk_proj, f64::from(self.disk_zoom), self.picked);
        let mesh = sheet_mesh(&hyp, self.picked);
        let cap = format!("Hyperboloid H² {} ({} faces)", hyp.symbol_label, hyp.faces.len());
        self.view = Some(hyp);
        (scene, mesh, cap)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use papertilling_core::hyperbolic::inflation::is_hyperbolic;

    use super::*;

    /// Full scene build (tiling + 2D primitives + hyperboloid mesh) for every pair, layer count and
    /// wedge, timed: the worst case is what one slider step costs.
    #[test]
    fn build_scenes_all_pairs_timed() {
        let t = Instant::now();
        let mut worst = (0.0, (0, 0, 0, 0_i32), 0, 0);
        for p in 3..=12 {
            for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
                for layers in 1..=4 {
                    for wedge_delta in [0, -1, 1] {
                        let mut st = HyperbolicPanelState { p, q, layers, wedge_delta, ..Default::default() };
                        let t1 = Instant::now();
                        let (scene, _mesh, _cap) = st.build_scenes();
                        let dt = t1.elapsed().as_secs_f64();
                        assert!(st.cache.is_some(), "({p},{q},{layers},{wedge_delta}) failed: {}", st.status);
                        let Scene2D::VectorScene { primitives } = scene else { panic!("vector scene") };
                        if dt > worst.0 {
                            worst = (dt, (p, q, layers, wedge_delta), primitives.lines.len(), primitives.polygons.len());
                        }
                    }
                }
            }
        }
        println!(
            "# time build_scenes 92 pairs x 4 layers x 3 wedges: {:?}; worst {:.1} ms at {:?} ({} line prims, {} polygons)",
            t.elapsed(),
            1e3 * worst.0,
            worst.1,
            worst.2,
            worst.3
        );
    }
}
