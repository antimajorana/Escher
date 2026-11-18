//! Picking in the H² view (2D disk and 3D sheet), the Mobius drag pan, and the pick readout: hyperbolic distance
//! from the view origin `d(0, z) = 2 artanh |z|` (Poincare `z` of the vertex, or of the face's hyperbolic centre),
//! the Schläfli symbol, the generation (layer) index and the polar angle `Δφ = arg z` in the current view.

use super::hyperbolic_scene::sheet_points;
use super::hyperbolic_ui::{DiskProjection, HypPick, HyperbolicPanelState};
use super::hyperbolic_view::{
    C, DiskIsometry, cabs, dist_from_origin, klein_to_poincare, lorentz_centroid, point_in_convex, poincare_to_klein,
};
use super::space3d_view::PICK_RADIUS_PX;
use crate::canvas2d::CanvasInteraction;
use crate::theme::ThemeTokens;
use crate::view3d::{View3dState, pick_nearest, project_point};

fn cross2(a: egui::Pos2, b: egui::Pos2, c: egui::Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Depth (`w`) of `pos` inside screen triangle `(a, b, c)` with clip depths, or `None`.
fn tri_hit_depth(a: (egui::Pos2, f32), b: (egui::Pos2, f32), c: (egui::Pos2, f32), pos: egui::Pos2) -> Option<f32> {
    let area2 = cross2(a.0, b.0, c.0);
    if area2.abs() < 1e-4 {
        return None;
    }
    let u = cross2(pos, b.0, c.0) / area2;
    let v = cross2(a.0, pos, c.0) / area2;
    let t = 1.0 - u - v;
    if u >= -1e-4 && v >= -1e-4 && t >= -1e-4 {
        let inv_w = u / a.1 + v / b.1 + t / c.1;
        Some(if inv_w > 1e-6 { 1.0 / inv_w } else { u * a.1 + v * b.1 + t * c.1 })
    } else {
        None
    }
}

impl HyperbolicPanelState {
    /// Poincare point (current view) of a pick: the vertex, or the face's hyperbolic centre.
    fn view_point(&self, pick: HypPick) -> Option<C> {
        let hyp = self.view.as_ref()?;
        match pick {
            HypPick::Vertex(v) => hyp.poincare_pts.get(v).copied(),
            HypPick::Face(f) => {
                let pts: Vec<C> = hyp.faces.get(f)?.iter().filter_map(|&u| hyp.poincare_pts.get(u).copied()).collect();
                (!pts.is_empty()).then(|| lorentz_centroid(&pts))
            }
        }
    }

    pub(super) fn picked_view_point(&self) -> Option<C> {
        self.picked.and_then(|p| self.view_point(p))
    }

    /// Disk point (Poincare, current view) under the 2D world point `world`, or `None` outside the disk.
    fn world_to_poincare(&self, world: [f64; 2]) -> Option<C> {
        let z = f64::from(self.disk_zoom.max(1e-3));
        let d = [world[0] / z, world[1] / z];
        let zp = match self.disk_proj {
            DiskProjection::Poincare => d,
            DiskProjection::Klein => {
                if cabs(d) >= 1.0 {
                    return None;
                }
                klein_to_poincare(d)
            }
        };
        (cabs(zp) < 0.999).then_some(zp)
    }

    /// The element under the 2D world point `world`: a vertex within `radius_px` (canvas `zoom` px per world unit)
    /// first, else the face containing it (convex polygons in the Klein model, where geodesics are chords).
    pub fn pick_2d(&self, world: [f64; 2], canvas_zoom: f32, radius_px: f32) -> Option<HypPick> {
        let hyp = self.view.as_ref()?;
        let dz = f64::from(self.disk_zoom.max(1e-3));
        let pts = match self.disk_proj {
            DiskProjection::Poincare => &hyp.poincare_pts,
            DiskProjection::Klein => &hyp.klein_pts,
        };
        let r_world = f64::from(radius_px / canvas_zoom.max(1e-3));
        let near = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p[0] * dz - world[0]).hypot(p[1] * dz - world[1])))
            .filter(|&(_, d)| d <= r_world)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = near {
            return Some(HypPick::Vertex(i));
        }
        let k = poincare_to_klein(self.world_to_poincare(world)?);
        hyp.faces
            .iter()
            .position(|f| {
                let poly: Vec<C> = f.iter().filter_map(|&u| hyp.klein_pts.get(u).copied()).collect();
                point_in_convex(&poly, k)
            })
            .map(HypPick::Face)
    }

    /// The element under a 3D-view screen point: the nearest projected sheet vertex within `PICK_RADIUS_PX`, else the
    /// front-most face triangle containing `pos` (with a small centroid fallback for grazing angles).
    pub(crate) fn pick_3d(&self, view3d: &View3dState, rect: egui::Rect, pos: egui::Pos2) -> Option<HypPick> {
        let hyp = self.view.as_ref()?;
        let mesh = view3d.mesh_data();
        let pts = sheet_points(hyp, mesh.center, mesh.radius);
        let mvp = view3d.mvp_for_rect(rect);
        let all = vec![true; pts.len()];
        if let Some(v) = pick_nearest(&pts, &all, &mvp, rect, pos, PICK_RADIUS_PX) {
            return Some(HypPick::Vertex(v));
        }
        let proj_pts: Vec<Option<(egui::Pos2, f32)>> = pts.iter().map(|&p| project_point(&mvp, rect, p)).collect();
        let mut best_face: Option<(usize, f32)> = None;
        let mut centres = Vec::with_capacity(hyp.faces.len());
        let mut valid = Vec::with_capacity(hyp.faces.len());
        for (fi, face) in hyp.faces.iter().enumerate() {
            if face.len() < 3 || face.iter().any(|&u| u >= pts.len()) {
                centres.push([0.0; 3]);
                valid.push(false);
                continue;
            }
            let c: [f32; 3] =
                core::array::from_fn(|k| face.iter().map(|&u| pts[u][k]).sum::<f32>() / face.len() as f32);
            centres.push(c);
            valid.push(true);
            let Some(pc) = project_point(&mvp, rect, c) else { continue };
            for m in 0..face.len() {
                let (Some(pa), Some(pb)) = (proj_pts[face[m]], proj_pts[face[(m + 1) % face.len()]]) else {
                    continue;
                };
                if let Some(w) = tri_hit_depth(pc, pa, pb, pos)
                    && best_face.is_none_or(|(_, bw)| w < bw)
                {
                    best_face = Some((fi, w));
                }
            }
        }
        if let Some((fi, _)) = best_face {
            return Some(HypPick::Face(fi));
        }
        pick_nearest(&centres, &valid, &mvp, rect, pos, PICK_RADIUS_PX).map(HypPick::Face)
    }

    /// Drag pan, hover and click picks of this frame; sets `view_changed` when the pan or the pick changed.
    pub fn handle_pointer(&mut self, canvas: Option<&CanvasInteraction>, canvas_zoom: f32, view3d: &View3dState) {
        if let Some((from, to)) = canvas.and_then(|c| c.drag_world)
            && let (Some(z0), Some(z1)) = (self.world_to_poincare(from), self.world_to_poincare(to))
            && cabs([z1[0] - z0[0], z1[1] - z0[1]]) > 1e-9
        {
            // the isometry taking z0 to z1 (through the origin), after the current pan
            self.pan = self.pan.then(DiskIsometry::to_origin(z0)).then(DiskIsometry::from_origin(z1));
            self.view_changed = true;
        }
        let mut hovered = canvas.and_then(|c| c.hovered_world).and_then(|w| self.pick_2d(w, canvas_zoom, PICK_RADIUS_PX));
        let mut clicked = canvas.and_then(|c| c.clicked_world).map(|w| self.pick_2d(w, canvas_zoom, PICK_RADIUS_PX));
        if let Some(ptr) = view3d.pointer() {
            hovered = hovered.or_else(|| ptr.hover.and_then(|p| self.pick_3d(view3d, ptr.rect, p)));
            if let Some(p) = ptr.click {
                clicked = Some(self.pick_3d(view3d, ptr.rect, p));
            }
        }
        self.hovered = hovered;
        if let Some(c) = clicked
            && c != self.picked
        {
            self.picked = c;
            self.view_changed = true;
        }
    }

    pub(super) fn show_pick_readout(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.separator();
        ui.strong("Picked Element");
        let Some(pick) = self.picked.or(self.hovered) else {
            ui.small("None: click a face or vertex in the disk or on the sheet. Drag the disk to pan.");
            return;
        };
        let (Some(cache), Some(z)) = (self.cache.as_ref(), self.view_point(pick)) else { return };
        let tag = if self.picked == Some(pick) { "picked" } else { "hover" };
        let (what, generation, z0) = match pick {
            HypPick::Vertex(v) => (format!("Vertex #{v}"), cache.vertex_gen.get(v), cache.base.poincare_pts.get(v).copied()),
            HypPick::Face(f) => {
                let Some(face) = cache.base.faces.get(f) else { return };
                let base_pts: Vec<C> =
                    face.iter().filter_map(|&u| cache.base.poincare_pts.get(u).copied()).collect();
                (
                    format!("Face #{f} ({}-gon)", face.len()),
                    cache.face_gen.get(f),
                    (!base_pts.is_empty()).then(|| lorentz_centroid(&base_pts)),
                )
            }
        };
        let (p, q) = (cache.base.p, cache.base.q);
        ui.colored_label(tokens.status_ok, format!("{what} ({tag})"));
        ui.small(format!(
            "Schläfli {{{p},{q}}}: (p-2)(q-2) = {} > 4 | style {}",
            (p - 2) * (q - 2),
            cache.base.symbol_label
        ));
        let gen_txt = generation.filter(|&&g| g != usize::MAX).map_or("-".to_string(), |g| g.to_string());
        ui.small(format!("Layer (generation): {gen_txt}"));
        ui.small(format!(
            "z = ({:+.4}, {:+.4}) | d(0,z) = 2 artanh|z| = {:.4}",
            z[0],
            z[1],
            dist_from_origin(z)
        ));
        let phi_txt = if cabs(z) < 1e-9 {
            String::from("Δφ = — (at view origin)")
        } else {
            format!("Δφ = arg z = {:+.2}° (view origin, +x axis)", z[1].atan2(z[0]).to_degrees())
        };
        ui.small(phi_txt);
        if let Some(z0) = z0 {
            ui.small(format!("Distance from the tiling centre (unpanned): {:.4}", dist_from_origin(z0)));
        }
        if let (Some(h), Some(pk)) = (self.hovered, self.picked)
            && h != pk
        {
            ui.small(format!("Hover: {h:?}"));
        }
    }
}
