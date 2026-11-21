//! Lazily built 2 x 2 unit-cell previews for the lattice cards and the Quick Lattice Switcher.
//!
//! The geometry of a preview depends only on the lattice (default parameter values), so it is cached by the
//! lattice's alias key; the theme is applied when painting (`ThemeTokens::remap_scene_ink`), which keeps one
//! cache entry per lattice for both themes. Builds are capped per frame so a first scroll through hundreds of
//! cards never stalls the UI: a card whose preview is not built yet shows a placeholder and asks for a repaint.

use std::collections::HashMap;
use std::sync::Arc;

use crate::aliases::LatticeAliasKey;
use crate::backend::CatalogueItemUi;
use crate::canvas2d::Scene2D;
use crate::live_plane::build_cell_for_item;
use crate::scene2d::compile_scene;
use crate::theme::ThemeTokens;

/// Previews built per frame at most (each costs well under a millisecond for the stored catalogues).
pub const BUILDS_PER_FRAME: usize = 12;

pub struct ThumbGeom {
    pub lines: Vec<([f32; 2], [f32; 2], [u8; 4])>,
    pub sites: Vec<([f32; 2], [u8; 4])>,
    /// The unit cell at the origin `(0, a1, a1 + a2, a2)`.
    pub cell: [[f32; 2]; 4],
    pub bbox: ([f32; 2], [f32; 2]),
    /// Sites and bonds of one unit cell (`cell.basis.len()`, `cell.bonds.len()`).
    pub sites_per_cell: usize,
    pub bonds_per_cell: usize,
}

impl ThumbGeom {
    pub fn build(item: &CatalogueItemUi) -> Self {
        let (cell, _) = build_cell_for_item(item, &[]);
        let prims = compile_scene(&Scene2D::PeriodicCell { cell: cell.clone(), repeat: 2, show_unit_cell: false });
        let a1 = [cell.a1[0] as f32, cell.a1[1] as f32];
        let a2 = [cell.a2[0] as f32, cell.a2[1] as f32];
        let o = central_cell_origin(a1, a2, prims.bbox);
        Self {
            lines: prims.lines.iter().map(|l| (l.a, l.b, l.color)).collect(),
            sites: prims.circles.iter().map(|c| (c.center, c.fill)).collect(),
            cell: [o, [o[0] + a1[0], o[1] + a1[1]], [o[0] + a1[0] + a2[0], o[1] + a1[1] + a2[1]], [o[0] + a2[0], o[1] + a2[1]]],
            bbox: prims.bbox,
            sites_per_cell: cell.basis.len(),
            bonds_per_cell: cell.bonds.len(),
        }
    }

    /// Paint the preview fitted (aspect kept, 6 % margin) into `rect`, y axis up.
    pub fn paint(&self, painter: &egui::Painter, rect: egui::Rect, tokens: &ThemeTokens) {
        let (lo, hi) = self.bbox;
        let w = (hi[0] - lo[0]).max(1e-3);
        let h = (hi[1] - lo[1]).max(1e-3);
        if !(w.is_finite() && h.is_finite()) {
            return;
        }
        let s = 0.88 * (rect.width() / w).min(rect.height() / h);
        let c = rect.center();
        let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
        let to = |p: [f32; 2]| egui::pos2(c.x + s * (p[0] - mid[0]), c.y - s * (p[1] - mid[1]));
        let cell: Vec<egui::Pos2> = self.cell.iter().map(|&p| to(p)).collect();
        painter.add(egui::Shape::convex_polygon(cell.clone(), tokens.unit_cell_fill, egui::Stroke::NONE));
        painter.add(egui::Shape::closed_line(cell, egui::Stroke::new(0.8, tokens.unit_cell_vec)));
        let lw = (0.045 * s).clamp(0.7, 1.4);
        for &(a, b, col) in &self.lines {
            painter.line_segment([to(a), to(b)], egui::Stroke::new(lw, tokens.remap_scene_ink(col)));
        }
        let r = (0.07 * s).clamp(1.3, 2.8);
        for &(p, col) in &self.sites {
            painter.circle_filled(to(p), r, tokens.remap_scene_ink(col));
        }
    }
}

/// Origin `n1 a1 + n2 a2` (integers) of the unit cell whose centre is nearest the middle of `bbox` (the patch may
/// sit at negative cell indices, depending on how the bonds are canonicalised).
pub(crate) fn central_cell_origin(a1: [f32; 2], a2: [f32; 2], bbox: ([f32; 2], [f32; 2])) -> [f32; 2] {
    let det = a1[0] * a2[1] - a1[1] * a2[0];
    let mid = [0.5 * (bbox.0[0] + bbox.1[0]), 0.5 * (bbox.0[1] + bbox.1[1])];
    if det.abs() < 1e-9 || !det.is_finite() || !(mid[0].is_finite() && mid[1].is_finite()) {
        return [0.0, 0.0];
    }
    // fractional coordinates of the middle, then the cell [n, n + 1) whose centre n + 1/2 is nearest
    let f1 = (mid[0] * a2[1] - mid[1] * a2[0]) / det;
    let f2 = (a1[0] * mid[1] - a1[1] * mid[0]) / det;
    let (n1, n2) = ((f1 - 0.5).round(), (f2 - 0.5).round());
    [n1 * a1[0] + n2 * a2[0], n1 * a1[1] + n2 * a2[1]]
}

#[derive(Default)]
pub struct LatticeThumbCache {
    map: HashMap<LatticeAliasKey, Arc<ThumbGeom>>,
    frame: u64,
    built_in_frame: usize,
    /// A preview was requested but deferred by the per-frame cap.
    pub deferred: bool,
}

impl LatticeThumbCache {
    /// Start a frame (resets the per-frame build budget).
    pub fn begin_frame(&mut self, frame: u64) {
        if frame != self.frame {
            self.frame = frame;
            self.built_in_frame = 0;
            self.deferred = false;
        }
    }

    /// The preview of `item`, building it if the frame budget allows (`None` = deferred to a later frame).
    pub fn get_or_build(&mut self, item: &CatalogueItemUi) -> Option<Arc<ThumbGeom>> {
        if let Some(g) = self.map.get(&item.alias_key) {
            return Some(Arc::clone(g));
        }
        if self.built_in_frame >= BUILDS_PER_FRAME {
            self.deferred = true;
            return None;
        }
        self.built_in_frame += 1;
        let g = Arc::new(ThumbGeom::build(item));
        self.map.insert(item.alias_key.clone(), Arc::clone(&g));
        Some(g)
    }

    /// The cached preview of `key`, without building it.
    pub fn peek(&self, key: &LatticeAliasKey) -> Option<&ThumbGeom> {
        self.map.get(key).map(Arc::as_ref)
    }

    /// Number of cached previews.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}
