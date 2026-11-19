//! Hyperbolic `H^2` `{p, q}` tilings, `(p.q.p.q)` kagome, Poincare geodesics, Klein chords,
//! 3D Minkowski hyperboloid bowl, wedge disclinations, and `kappa(D) < 0` inspector.
//!
//! Interactive view (rendering only; the tiling is cached and rebuilt only when `p, q`, style, layers or wedge
//! change): a hyperbolic pan (a disk isometry applied to the 2D disk and the 3D sheet, so the picture stays a valid
//! `{p,q}` tiling), a view zoom of the 2D disk, and face / vertex picking with a readout (`hyperbolic_pick`).

use papertilling_core::hyperbolic::{HyperbolicStyle, HyperbolicTiling, build_hyperbolic_tiling};

use super::hyperbolic_scene::{disk_scene, empty_scene, sheet_mesh};
use super::hyperbolic_shells::{SHELL_CACHE_MAX, ShellTable, shell_table};
use super::hyperbolic_view::{DiskIsometry, vertex_generations, view_tiling};
use crate::mesh3d::GpuMeshData;
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;

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

type TilingKey = (usize, usize, HyperbolicStyle, usize, i32);

/// The origin-frame tiling of a key with its face and vertex generations.
pub struct CachedTiling {
    pub key: TilingKey,
    pub base: HyperbolicTiling,
    pub face_gen: Vec<usize>,
    pub vertex_gen: Vec<usize>,
}

pub struct HyperbolicPanelState {
    pub p: usize,
    pub q: usize,
    pub style: HyperbolicStyle,
    pub layers: usize,
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
    /// Uncapped BDF vertex-completion shell counts, at most [`SHELL_CACHE_MAX`] `{p, q}` keys
    /// (oldest dropped first), so Regular <-> Dual toggles and revisits do not recompute.
    pub shell_cache: Vec<((usize, usize), Result<ShellTable, String>)>,
    /// Which shells the drawn patch contains (from the tiling's `growth_rule`).
    pub drawn_patch_note: String,
}

impl Default for HyperbolicPanelState {
    fn default() -> Self {
        Self {
            p: 7,
            q: 3,
            style: HyperbolicStyle::RegularPQ,
            layers: 2,
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
        }
    }
}

impl HyperbolicPanelState {
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
            ui.small("Layers:");
            dirty |= ui.add(egui::DragValue::new(&mut self.layers).range(1..=4)).changed();
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
        ui.horizontal(|ui| {
            ui.small("Central Wedge Δp:");
            for d in [-1, 0, 1] {
                dirty |= ui.selectable_value(&mut self.wedge_delta, d, format!("{d:+}")).changed();
            }
        });
        self.show_view_controls(ui);
        dirty
    }

    fn show_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("Disk View: Hyperbolic Pan, Zoom & Picking");
        ui.small("Drag the disk: hyperbolic pan (Mobius). Click: pick a face or vertex.");
        let mut changed = false;
        // one step = hyperbolic distance 0.25 (Poincare radius tanh(0.125))
        let r = 0.125_f64.tanh();
        ui.horizontal(|ui| {
            ui.small("Pan:");
            for (lbl, d) in [("←", [-r, 0.0]), ("→", [r, 0.0]), ("↑", [0.0, r]), ("↓", [0.0, -r])] {
                if ui.small_button(lbl).clicked() {
                    // move the content by `d`: the view origin's content goes to `d`
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
        let is_hyperbolic = (self.p.saturating_sub(2)) * (self.q.saturating_sub(2)) > 4;
        if !self.status.is_empty() {
            let col = if is_hyperbolic { tokens.status_ok } else { tokens.status_err };
            ui.colored_label(col, egui::RichText::new(&self.status).small());
        }
        ui.small(format!(
            "Schläfli {{{}, {}}} | Layers: {} | Wedge Δp: {:+}",
            self.p, self.q, self.layers, self.wedge_delta
        ));
        ui.small(format!("Projection: {:?} | Style: {:?}", self.disk_proj, self.style));
        self.show_pick_readout(ui, tokens);
        self.show_shell_table(ui);
    }

    fn key(&self) -> TilingKey {
        (self.p, self.q, self.style, self.layers, self.wedge_delta)
    }

    /// Build (or reuse) the origin-frame tiling of the current key and update the shell table cache;
    /// a new key clears the pick.
    fn ensure_tiling(&mut self) -> Result<(), String> {
        let (pe, qe) = if self.style == HyperbolicStyle::DualQP { (self.q, self.p) } else { (self.p, self.q) };
        if !self.shell_cache.iter().any(|(k, _)| *k == (pe, qe)) {
            let res = shell_table(pe, qe);
            if self.shell_cache.len() >= SHELL_CACHE_MAX {
                self.shell_cache.drain(..1); // drop the oldest table
            }
            self.shell_cache.push(((pe, qe), res));
        }
        self.shell_key = Some((pe, qe));
        let key = self.key();
        if self.cache.as_ref().is_some_and(|c| c.key == key) {
            return Ok(());
        }
        self.cache = None;
        self.picked = None;
        self.hovered = None;
        let base = match build_hyperbolic_tiling(self.p, self.q, self.style, self.layers, self.wedge_delta) {
            Ok(b) => b,
            Err(e) => {
                self.drawn_patch_note.clear();
                return Err(e);
            }
        };
        self.drawn_patch_note = base.growth_rule.clone();
        let face_gen = base.face_shell.clone();
        let vertex_gen = vertex_generations(base.poincare_pts.len(), &base.faces, &face_gen);
        self.cache = Some(CachedTiling { key, base, face_gen, vertex_gen });
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
