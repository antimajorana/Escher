//! Interactive part of the Atlas 3D space-group view: unit-cell box and axes toggles, z-slice slab, depth cue,
//! and click / hover picking of a site in the 3D view (through the camera's MVP) or in its 2D xy projection, with
//! the site readout for the Right Inspector.

use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d, Net3dPatch, build_net_3d};

use super::space3d_geom::{NetFrame, coordination, net_frame, slab_mask, z_fractions};
use super::space3d_ui::Space3dPanelState;
use crate::canvas2d::CanvasInteraction;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::atlas_net_view::{NetMeshView, mesh_from_net_view, net_mesh_points};
use crate::scene2d::{CirclePrim, LinePrim, Scene2D, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;
use crate::view3d::{View3dState, pick_nearest};

type Mat4 = [[f32; 4]; 4];

/// Pick radius in screen pixels (2D and 3D).
pub const PICK_RADIUS_PX: f32 = 9.0;

/// View options and pick state of the 3D space-group view (rendering only: never part of the net's identity).
pub struct Space3dView {
    pub show_cell: bool,
    pub show_axes: bool,
    /// Slab in fractional z of the drawn block (`[0, 1]` = every site).
    pub z_slice_range: [f32; 2],
    pub depth_cue: bool,
    pub picked: Option<usize>,
    pub hovered: Option<usize>,
    /// A view option or the pick changed: the scenes need a refresh (without resetting the 2D view).
    pub changed: bool,
    net: Option<((CanonicalNet3d, LineDefect3d), Net3dPatch)>,
    /// Rotation the current mesh's depth cue was computed for.
    depth_rot_used: Option<Mat4>,
}

impl Default for Space3dView {
    fn default() -> Self {
        Self {
            show_cell: false,
            show_axes: false,
            z_slice_range: [0.0, 1.0],
            depth_cue: false,
            picked: None,
            hovered: None,
            changed: false,
            net: None,
            depth_rot_used: None,
        }
    }
}

/// Site within `radius` (world units) of `world` in the xy projection, the topmost (largest z: the projection is
/// seen from +z) among visible sites.
pub fn pick_xy(positions: &[[f64; 3]], visible: &[bool], world: [f64; 2], radius: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, p) in positions.iter().enumerate() {
        if visible.get(i).copied().unwrap_or(false)
            && (p[0] - world[0]).hypot(p[1] - world[1]) <= radius
            && best.is_none_or(|(_, z)| p[2] > z)
        {
            best = Some((i, p[2]));
        }
    }
    best.map(|b| b.0)
}

fn species_label(net: CanonicalNet3d, kind: u8) -> &'static str {
    match (net, kind) {
        (CanonicalNet3d::Rutile136, 0) => "Ti (cation)",
        (CanonicalNet3d::Rutile136, _) => "O (anion)",
        (CanonicalNet3d::Perovskite, 0) => "A site",
        (CanonicalNet3d::Perovskite, 1) => "B site",
        (CanonicalNet3d::Perovskite, _) => "X (anion)",
        (CanonicalNet3d::Diamond, 0) => "sublattice A (fcc)",
        (CanonicalNet3d::Diamond, _) => "sublattice B (fcc + 1/4)",
        (CanonicalNet3d::BCC | CanonicalNet3d::Tetragonal123, 0) => "corner",
        (CanonicalNet3d::BCC | CanonicalNet3d::Tetragonal123, _) => "body centre",
        (CanonicalNet3d::Pyrochlore, 0) => "tetrahedron vertex",
        (CanonicalNet3d::Pyrochlore, _) => "body centre",
        _ => "site",
    }
}

impl Space3dPanelState {
    fn net_key(&self) -> (CanonicalNet3d, LineDefect3d) {
        (self.selected_net, self.selected_defect)
    }

    /// The net of the current (net, defect) choice, rebuilt (and the pick cleared) when the choice changed.
    pub fn ensure_net(&mut self) -> &Net3dPatch {
        let key = self.net_key();
        if self.view.net.as_ref().is_none_or(|(k, _)| *k != key) {
            self.view.net = Some((key, build_net_3d(key.0, key.1)));
            self.view.picked = None;
            self.view.hovered = None;
        }
        &self.view.net.as_ref().expect("built above").1
    }

    pub fn frame(&self) -> NetFrame {
        net_frame(self.selected_net, self.selected_defect)
    }

    pub fn visible_mask(&mut self) -> Vec<bool> {
        let range = self.view.z_slice_range;
        slab_mask(&self.ensure_net().positions, range)
    }

    /// Whether the depth cue of the current mesh was computed for another camera rotation.
    pub fn needs_depth_refresh(&self, rot: &Mat4) -> bool {
        self.view.depth_cue && self.view.depth_rot_used.as_ref() != Some(rot)
    }

    pub fn show_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("View: Cell, Axes, Slab & Picking");
        let v = &mut self.view;
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            changed |= ui.checkbox(&mut v.show_cell, "Unit cell").changed();
            changed |= ui.checkbox(&mut v.show_axes, "Axes").changed();
            changed |= ui.checkbox(&mut v.depth_cue, "Depth cue").on_hover_text("Farther sites fainter").changed();
        });
        ui.small("z-slice slab (fractional z of the drawn block):");
        if ui.add(egui::Slider::new(&mut v.z_slice_range[0], 0.0..=1.0).text("z min")).changed() {
            v.z_slice_range[1] = v.z_slice_range[1].max(v.z_slice_range[0]);
            changed = true;
        }
        if ui.add(egui::Slider::new(&mut v.z_slice_range[1], 0.0..=1.0).text("z max")).changed() {
            v.z_slice_range[0] = v.z_slice_range[0].min(v.z_slice_range[1]);
            changed = true;
        }
        ui.horizontal(|ui| {
            if ui.small_button("Full slab").clicked() {
                v.z_slice_range = [0.0, 1.0];
                changed = true;
            }
            if v.picked.is_some() && ui.small_button("Clear pick").clicked() {
                v.picked = None;
                changed = true;
            }
        });
        ui.small("Click a site in the 2D or 3D view to inspect it.");
        v.changed |= changed;
    }

    /// Hover / click picks from the 2D projection (`canvas`, with the canvas zoom in px per world unit) and the 3D
    /// view. Sets `view.changed` when the picked site changed.
    pub fn handle_pointer(&mut self, canvas: Option<&CanvasInteraction>, canvas_zoom: f32, view3d: &View3dState) {
        let range = self.view.z_slice_range;
        let net = self.ensure_net();
        let visible = slab_mask(&net.positions, range);
        let r_world = f64::from(PICK_RADIUS_PX / canvas_zoom.max(1e-3));
        let mut hovered =
            canvas.and_then(|c| c.hovered_world).and_then(|w| pick_xy(&net.positions, &visible, w, r_world));
        let mut clicked = canvas.and_then(|c| c.clicked_world).map(|w| pick_xy(&net.positions, &visible, w, r_world));
        if let Some(ptr) = view3d.pointer() {
            let pts = net_mesh_points(net);
            let mvp = view3d.mvp_for_rect(ptr.rect);
            let pick = |p: egui::Pos2| pick_nearest(&pts, &visible, &mvp, ptr.rect, p, PICK_RADIUS_PX);
            hovered = hovered.or_else(|| ptr.hover.and_then(pick));
            if let Some(p) = ptr.click {
                clicked = Some(pick(p));
            }
        }
        self.view.hovered = hovered;
        if let Some(c) = clicked
            && c != self.view.picked
        {
            self.view.picked = c;
            self.view.changed = true;
        }
    }

    /// 2D projection, 3D mesh and caption of the current net with the view options applied.
    pub fn build_view_scenes(&mut self, rot: &Mat4) -> (Scene2D, GpuMeshData, String) {
        let visible = self.visible_mask();
        if self.view.picked.is_some_and(|p| !visible.get(p).copied().unwrap_or(false)) {
            self.view.picked = None; // the picked site left the slab
        }
        let frame = self.frame();
        let (show_cell, show_axes, depth_cue, picked) =
            (self.view.show_cell, self.view.show_axes, self.view.depth_cue, self.view.picked);
        self.view.depth_rot_used = depth_cue.then_some(*rot);
        let net = self.ensure_net();
        let mut extra_wires = Vec::new();
        if show_cell {
            extra_wires.extend(frame.cell_edges().into_iter().map(|(a, b)| (a, b, [0.55, 0.60, 0.70, 0.95])));
        }
        if show_axes {
            let len = frame.lat.iter().map(|a| a.iter().map(|x| x * x).sum::<f64>().sqrt()).fold(0.0, f64::max);
            let cols = [[0.90, 0.20, 0.20, 1.0], [0.20, 0.70, 0.25, 1.0], [0.20, 0.40, 0.95, 1.0]];
            extra_wires.extend(frame.axes(len).into_iter().zip(cols).map(|((a, b), c)| (a, b, c)));
        }
        let view = NetMeshView { visible: &visible, extra_wires, depth_rot: depth_cue.then_some(*rot), highlight: picked };
        let mesh = mesh_from_net_view(net, &view);
        let scene = net_projection_scene(net, &visible, depth_cue, picked);
        (scene, mesh, net.info.clone())
    }

    /// Right Inspector readout of the picked (else hovered) site.
    pub fn show_pick_readout(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.separator();
        ui.strong("Picked Site");
        let (picked, hovered) = (self.view.picked, self.view.hovered);
        let Some(i) = picked.or(hovered) else {
            ui.small("None: click a site in the 2D projection or the 3D view.");
            return;
        };
        let frame = self.frame();
        let (net_kind, defect) = self.net_key();
        let zf = z_fractions(&self.ensure_net().positions);
        let net = self.ensure_net();
        let Some(&p) = net.positions.get(i) else { return };
        let kind = net.site_kinds.get(i).copied().unwrap_or(0);
        let coord = coordination(net.positions.len(), &net.bonds)[i];
        let f = frame.fractional(p);
        let tag = if picked == Some(i) { "picked" } else { "hover" };
        ui.colored_label(tokens.status_ok, format!("Site #{i} ({tag}) | species {kind}: {}", species_label(net_kind, kind)));
        ui.small(format!("Cartesian: ({:+.4}, {:+.4}, {:+.4})", p[0], p[1], p[2]));
        let nominal = if defect == LineDefect3d::None { "" } else { " (nominal: defect-warped)" };
        ui.small(format!("Fractional: ({:.4}, {:.4}, {:.4}){nominal}", f[0], f[1], f[2]));
        ui.small(format!("Coordination: {coord} bonds | slab z = {:.3}", zf.get(i).copied().unwrap_or(0.0)));
        if let (Some(h), Some(pk)) = (hovered, picked)
            && h != pk
        {
            ui.small(format!("Hover: site #{h}"));
        }
    }
}

fn net_projection_scene(net: &Net3dPatch, visible: &[bool], depth_cue: bool, picked: Option<usize>) -> Scene2D {
    let zf = z_fractions(&net.positions);
    let shown = |i: usize| visible.get(i).copied().unwrap_or(true);
    // seen from +z: the top layer (z fraction 1) is nearest
    let fade = |i: usize| if depth_cue { 0.25 + 0.75 * zf[i] as f32 } else { 1.0 };
    let alpha = |a: u8, f: f32| (f32::from(a) * f).round() as u8;
    let mut order: Vec<usize> = (0..net.positions.len()).filter(|&i| shown(i)).collect();
    // Draw lower-z sites first so the topmost site (matching `pick_xy`) renders on top.
    order.sort_by(|&a, &b| net.positions[a][2].total_cmp(&net.positions[b][2]).then(a.cmp(&b)));
    let site_count = order.len();
    let mut circles: Vec<CirclePrim> = order
        .into_iter()
        .map(|i| {
            let p = net.positions[i];
            let rgb = match net.site_kinds.get(i).copied().unwrap_or(0) % 3 {
                0 => [42, 92, 168],
                1 => [214, 92, 52],
                _ => [42, 148, 96],
            };
            CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.055,
                min_px: 2.5,
                max_px: 8.5,
                fill: [rgb[0], rgb[1], rgb[2], alpha(255, fade(i))],
                stroke: ([255, 255, 255, alpha(255, fade(i))], 0.8),
            }
        })
        .collect();
    if let Some(h) = picked.filter(|&h| h < net.positions.len() && shown(h)) {
        let p = net.positions[h];
        circles.push(CirclePrim {
            center: [p[0] as f32, p[1] as f32],
            radius_world: 0.11,
            min_px: 6.0,
            max_px: 14.0,
            fill: [255, 209, 20, 90],
            stroke: ([255, 209, 20, 255], 2.5),
        });
    }
    let lines: Vec<LinePrim> = net
        .bonds
        .iter()
        .filter(|&&(u, v, _)| shown(u) && shown(v))
        .map(|&(u, v, bk)| LinePrim {
            a: [net.positions[u][0] as f32, net.positions[u][1] as f32],
            b: [net.positions[v][0] as f32, net.positions[v][1] as f32],
            color: if bk == 1 { [224, 58, 32, 255] } else { [44, 62, 88, alpha(190, fade(u).min(fade(v)))] },
            width: if bk == 1 { 2.4 } else { 1.3 },
            dashed: false,
            arrow_mid: false,
        })
        .collect();
    // bbox of the whole block, so the slab never re-frames the view
    let pts = net.positions.iter().map(|p| [p[0] as f32, p[1] as f32]);
    let bond_count = lines.len();
    Scene2D::VectorScene {
        primitives: ScenePrimitives {
            polygons: Vec::new(),
            lines,
            circles,
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: format!("3D Net (xy-projection): {}", net.name),
            note: net.defect_label.clone(),
            bbox: compute_bbox(pts),
            site_count,
            bond_count,
        },
    }
}
