use std::path::Path;

use crate::export::{ExportLegend, export_scene_full, export_scene_layers};
pub use crate::scene2d::Scene2D;
use crate::scene2d::{ScenePrimitives, compile_scene};
use crate::theme::ThemeTokens;

pub mod circuit;
mod gallery_paint;
mod cuts;
mod draw_helpers;
pub mod gauge_overlay;
pub mod overlay_legend;
pub mod overlay_scales;
mod hit;
mod input;
mod render;
#[cfg(test)]
mod gallery_overlay_tests;
#[cfg(test)]
mod tests;

pub use circuit::CanvasCircuitOverlay;
pub use cuts::{CanvasCutRay, CanvasKinematicOverlay, CanvasReactionArrow};
pub(crate) use cuts::cut_ray_handle_screen_pos;
pub(crate) use draw_helpers::{draw_arrow_mid_egui, draw_dashed_egui, draw_overlays, marker_colors, to_color32};
pub use gauge_overlay::CrystalOverlayData;
pub(crate) use hit::take_keyboard_zoom_action;
pub use hit::{
    ViewTransform, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP_FACTOR, ZoomKeyAction,
    input_zoom_factor, keyboard_zoom_action, zoom_by_factor_about,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    DisclinationPlus,
    DisclinationMinus,
    Dislocation,
}

#[derive(Clone, Debug)]
pub struct CanvasMarker {
    pub index: usize,
    pub pos: [f64; 2],
    pub kind: MarkerKind,
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct CanvasInteraction {
    /// A click (press and release without dragging) on empty canvas, in world coordinates.
    pub clicked_world: Option<[f64; 2]>,
    /// An interactive click that hit neither a pin marker, nor a cut-ray handle, nor a Burgers circuit vertex.
    pub clicked_empty: bool,
    pub dragged_marker: Option<(usize, [f64; 2])>,
    pub drag_released: bool,
    /// Live cut-ray handle drag: `(spec_index, angle_deg)`.
    pub dragged_cut_ray: Option<(usize, f64)>,
    pub cut_drag_released: bool,
    pub hovered_world: Option<[f64; 2]>,
    pub shift_held: bool,
    /// A click that landed on a pin (its `CanvasMarker::index`); `clicked_world` is then `None`, so a click
    /// on an existing pin never places a second defect on top of it.
    pub clicked_marker: Option<usize>,
    /// A click that landed on a vertex of the active Burgers circuit loop.
    pub clicked_circuit_vertex: Option<usize>,
    /// With `Canvas2D::drag_reports_world`: this frame's drag as `(from, to)` world points instead of a view pan.
    pub drag_world: Option<([f64; 2], [f64; 2])>,
}

pub struct Canvas2D {
    pub pan: egui::Vec2,
    pub zoom: f32,
    pub(super) fit_zoom: f32,
    pub(super) world_center: [f32; 2],
    pub(super) auto_fit_pending: bool,
    pub(super) active_drag_pin: Option<usize>,
    pub(super) active_drag_cut: Option<(usize, [f64; 2])>,
    pub(super) selected_marker: Option<usize>,
    pub(super) scene: Scene2D,
    pub(super) compiled: ScenePrimitives,
    pub(super) export_markers: Vec<CanvasMarker>,
    pub(super) cut_rays: Vec<CanvasCutRay>,
    pub(super) kinematic_overlay: CanvasKinematicOverlay,
    pub(super) circuit_overlay: Option<CanvasCircuitOverlay>,
    pub(super) gauge_overlay: Option<CrystalOverlayData>,
    /// Gallery symmetry / Voronoi overlay: `under` merged into `compiled`, `over` painted and exported last.
    pub(super) gallery_overlay: Option<crate::scene2d::overlays::GalleryOverlay>,
    pub(super) export_title: Option<String>,
    /// Off by default: a background drag pans the view. On (set only around one interactive `show` by a caller that
    /// moves the content itself), the drag is reported in `CanvasInteraction::drag_world` and the view stays.
    pub(crate) drag_reports_world: bool,
    pub(crate) last_rect: egui::Rect,
}

impl Canvas2D {
    pub fn new(scene: Scene2D) -> Self {
        let compiled = compile_scene(&scene);
        let (bmin, bmax) = compiled.bbox;
        let world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 56.0,
            fit_zoom: 56.0,
            world_center,
            auto_fit_pending: true,
            active_drag_pin: None,
            active_drag_cut: None,
            selected_marker: None,
            scene,
            compiled,
            export_markers: Vec::new(),
            cut_rays: Vec::new(),
            kinematic_overlay: CanvasKinematicOverlay::default(),
            circuit_overlay: None,
            gauge_overlay: None,
            gallery_overlay: None,
            export_title: None,
            drag_reports_world: false,
            last_rect: egui::Rect::NOTHING,
        }
    }

    pub fn set_scene(&mut self, scene: Scene2D, reset_view: bool) {
        self.set_scene_with_title(scene, None, reset_view);
    }

    pub fn set_scene_with_title(&mut self, scene: Scene2D, export_title: Option<String>, reset_view: bool) {
        self.export_markers.clear();
        self.cut_rays.clear();
        self.selected_marker = None;
        self.kinematic_overlay = CanvasKinematicOverlay::default();
        self.circuit_overlay = None;
        self.gauge_overlay = None;
        self.gallery_overlay = None;
        self.export_title = export_title;
        self.compiled = compile_scene(&scene);
        self.scene = scene;
        if reset_view {
            let (bmin, bmax) = self.compiled.bbox;
            self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            self.auto_fit_pending = true;
            self.active_drag_pin = None;
            self.active_drag_cut = None;
        }
    }

    pub fn reset(&mut self) {
        let (bmin, bmax) = self.compiled.bbox;
        self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        self.pan = egui::Vec2::ZERO;
        self.auto_fit_pending = true;
        self.active_drag_pin = None;
        self.active_drag_cut = None;
    }

    pub fn zoom_at(&mut self, rect: egui::Rect, factor: f32, anchor: Option<egui::Pos2>) {
        (self.pan, self.zoom) = zoom_by_factor_about(self.pan, self.zoom, factor, rect.center(), anchor);
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.compiled.site_count, self.compiled.bond_count)
    }

    #[cfg(test)]
    pub(crate) fn scene(&self) -> &Scene2D {
        &self.scene
    }

    #[cfg(test)]
    pub(crate) fn gauge_overlay(&self) -> Option<&CrystalOverlayData> {
        self.gauge_overlay.as_ref()
    }

    /// Export the scene as shown (overlays, pins, legend) to `path` (`.svg` / `.png`); a PNG is drawn at the
    /// integer pixel scale `png_scale` (1, 2 or 4: 1024, 2048 or 4096 px a side), the SVG ignores it.
    pub fn export_to_path(&self, path: &Path, png_scale: u32) -> Result<String, String> {
        if self.circuit_overlay.is_some() || self.gauge_overlay.is_some() || self.gallery_overlay.is_some() {
            let mut prims = self.compiled.clone();
            if let Some(ov) = &self.gauge_overlay {
                gauge_overlay::append_gauge_export_prims(&mut prims, ov);
            }
            if let Some(circ) = &self.circuit_overlay {
                circuit::append_circuit_export_prims(&mut prims, circ);
            }
            let title = self.export_title.as_deref();
            let legend = self.gauge_overlay.as_ref().and_then(ExportLegend::from_overlay);
            // over layer as two layers so the rotation symbols sit on the mirror / glide lines (as on the canvas)
            let tops: Vec<crate::scene2d::ScenePrimitives> = self.gallery_overlay.as_ref().map_or_else(Vec::new, |g| {
                let lines = crate::scene2d::ScenePrimitives { lines: g.over.lines.clone(), bbox: g.over.bbox, ..Default::default() };
                let symbols = crate::scene2d::ScenePrimitives { polygons: g.over.polygons.clone(), bbox: g.over.bbox, ..Default::default() };
                vec![lines, symbols]
            });
            export_scene_layers(&prims, &tops, &self.export_markers, title, legend.as_ref(), (path, png_scale))
        } else {
            export_scene_full(&self.scene, &self.export_markers, self.export_title.as_deref(), path, png_scale)

        }
    }

    pub(crate) fn set_circuit_overlay(&mut self, circuit: Option<CanvasCircuitOverlay>) {
        self.circuit_overlay = circuit;
    }

    /// Show the Gallery overlay (`None` clears it); the scene is recompiled with the Voronoi layer underneath.
    pub(crate) fn set_gallery_overlay(&mut self, ov: Option<crate::scene2d::overlays::GalleryOverlay>) {
        self.compiled = compile_scene(&self.scene);
        if let Some(data) = &self.gauge_overlay {
            gauge_overlay::apply_site_and_bond_colors(&mut self.compiled, data);
        }
        if let Some(g) = &ov {
            let mut polys = g.under.polygons.clone();
            polys.append(&mut self.compiled.polygons);
            self.compiled.polygons = polys;
            // the overlay window can reach past the drawn sites: fit (canvas and export) covers both
            let (lo, hi) = self.compiled.bbox;
            let mut corners = vec![lo, hi];
            if !g.under.polygons.is_empty() {
                corners.extend([g.under.bbox.0, g.under.bbox.1]);
            }
            if !g.over.polygons.is_empty() || !g.over.lines.is_empty() {
                corners.extend([g.over.bbox.0, g.over.bbox.1]);
            }
            self.compiled.bbox = crate::scene2d::compute_bbox(corners.into_iter());
            if self.auto_fit_pending {
                let (bmin, bmax) = self.compiled.bbox;
                self.world_center = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
            }
        }
        self.gallery_overlay = ov;
    }

    /// Why the Gallery overlay could not be built (if any), for the inspector.
    pub(crate) fn gallery_overlay_note(&self) -> Option<&str> {
        self.gallery_overlay.as_ref().and_then(|g| g.note.as_deref())
    }

    pub(crate) fn set_gauge_overlay(&mut self, ov: Option<CrystalOverlayData>) {
        self.gauge_overlay = ov;
        self.compiled = compile_scene(&self.scene);
        if let Some(data) = &self.gauge_overlay {
            gauge_overlay::apply_site_and_bond_colors(&mut self.compiled, data);
        }
    }

    pub(crate) fn set_export_markers(&mut self, markers: &[CanvasMarker]) {
        self.export_markers.clear();
        self.export_markers.extend_from_slice(markers);
    }

    pub(crate) fn set_selected_marker(&mut self, idx: Option<usize>) {
        self.selected_marker = idx;
    }

    #[cfg(test)]
    pub(crate) fn selected_marker(&self) -> Option<usize> {
        self.selected_marker
    }

    pub(crate) fn set_cut_rays(&mut self, rays: &[CanvasCutRay]) {
        self.cut_rays.clear();
        self.cut_rays.extend_from_slice(rays);
    }

    pub(crate) fn set_kinematic_overlay(&mut self, overlay: CanvasKinematicOverlay) {
        self.kinematic_overlay = overlay;
    }

    pub(crate) fn cut_rays(&self) -> &[CanvasCutRay] {
        &self.cut_rays
    }

    #[cfg(test)]
    pub(crate) fn kinematic_overlay(&self) -> &CanvasKinematicOverlay {
        &self.kinematic_overlay
    }

    /// The world <-> screen map of the current view inside `rect`.
    pub fn view_at(&self, rect: egui::Rect) -> ViewTransform {
        ViewTransform { origin: rect.center() + self.pan, zoom: self.zoom, world_center: self.world_center }
    }

    pub fn world_to_screen_at(&self, rect: egui::Rect, p: [f32; 2]) -> egui::Pos2 {
        self.view_at(rect).world_to_screen(p)
    }

    pub fn screen_to_world_at(&self, rect: egui::Rect, pos: egui::Pos2) -> [f64; 2] {
        self.view_at(rect).screen_to_world(pos)
    }

    pub fn show(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.export_markers.clear();
        let _ = self.show_inner(ui, &[], false, None, false, tokens);
    }

    pub fn show_interactive(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        preview_world: Option<[f64; 2]>,
        tokens: &ThemeTokens,
    ) -> CanvasInteraction {
        self.set_export_markers(markers);
        self.show_inner(ui, markers, click_to_place, preview_world, true, tokens)
    }

    fn show_inner(
        &mut self,
        ui: &mut egui::Ui,
        markers: &[CanvasMarker],
        click_to_place: bool,
        preview_world: Option<[f64; 2]>,
        interactive: bool,
        tokens: &ThemeTokens,
    ) -> CanvasInteraction {
        let (rect, has_hud, hud_rect, out) =
            self.handle_canvas_input(ui, markers, click_to_place, interactive);
        self.last_rect = rect;
        self.render_canvas(ui, rect, has_hud.then_some(hud_rect), markers, preview_world, tokens);
        out
    }
}
