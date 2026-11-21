//! View-only BDF / cluster layer-growth patch and inspector controls for the Gallery tab (C1 WP-U2).

use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use papertilling_core::plane::inflation::{
    CentreSet, InflationResult, InflationRule, InflationSeed, PlaneTiling, ShellStats, VertexId, inflate_with,
};
use papertilling_core::types::Cell2D;

use crate::scene2d::periodic::edge_label_pos;
use crate::scene2d::{
    CirclePrim, LinePrim, PolyPrim, Scene2D, ScenePrimitives, TextPrim, bond_palette, compute_bbox, site_palette,
};
use crate::theme::ThemeTokens;
use crate::theme::tokens::{SCENE_UNIT_CELL_EDGE, SCENE_UNIT_CELL_FILL, SCENE_UNIT_CELL_LABEL};

/// Largest 0-based layer index offered by the Gallery `Layers` slider.
pub const MAX_E2_UI_LAYERS: usize = 16;
/// Maximum number of faces drawn in a Gallery open patch; layers exceeding this cap are omitted whole.
pub const MAX_E2_DRAW_FACES: usize = 10_000;

const SHELL_FILL_PALETTE: [[u8; 4]; 6] = [
    [244, 196, 68, 135],
    [92, 168, 235, 130],
    [88, 198, 168, 130],
    [238, 116, 98, 130],
    [174, 128, 226, 130],
    [166, 194, 88, 130],
];
const SHELL_BORDER_RGBA: [u8; 4] = [100, 112, 128, 90];

/// Cross-lattice safe seed choice for 2D Gallery growth (indices clamp to the active cell's ranges).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum E2SeedChoice {
    /// Vertex star of `PlaneTiling::most_symmetric_vertex()` (`PlaneTiling::default_seed()`, plan Q3).
    #[default]
    MostSymmetricVertex,
    /// Vertex star of basis site `s` (clamped to `0..cell.basis.len()`).
    Vertex(usize),
    /// Edge seed on `cell.bonds[b]` (clamped to `0..cell.bonds.len()`).
    Edge(usize),
    /// Single tile of face orbit `f` (clamped to `0..tiling.orbit_count()`).
    Tile(usize),
    /// Repaired Wigner-Seitz cell about `PlaneTiling::most_symmetric_point()`.
    Dirichlet,
}

impl E2SeedChoice {
    /// Resolve against `cell` and `tiling`, clamping indices so switching lattices never panics or errors on range.
    pub fn resolve(self, cell: &Cell2D, tiling: &PlaneTiling) -> Result<InflationSeed, String> {
        match self {
            Self::MostSymmetricVertex => Ok(tiling.default_seed()),
            Self::Vertex(s) => {
                if cell.basis.is_empty() {
                    return Err("cell has no basis sites".to_string());
                }
                Ok(InflationSeed::VertexStar(s.min(cell.basis.len() - 1)))
            }
            Self::Edge(b) => {
                if cell.bonds.is_empty() {
                    return Err("cell has no bonds".to_string());
                }
                let bond = &cell.bonds[b.min(cell.bonds.len() - 1)];
                Ok(InflationSeed::Edge((bond.i, [0, 0]), (bond.j, [bond.n1, bond.n2])))
            }
            Self::Tile(f) => {
                let n = tiling.orbit_count();
                if n == 0 {
                    return Err("tiling has no face orbits".to_string());
                }
                Ok(InflationSeed::Face(f.min(n - 1)))
            }
            Self::Dirichlet => Ok(InflationSeed::Dirichlet),
        }
    }
}

/// View-only layer-growth state for the Gallery tab (never changes `active_item` or `revision`).
#[derive(Clone, Debug, PartialEq)]
pub struct GalleryInflationState {
    pub enabled: bool,
    pub rule: InflationRule,
    pub seed_choice: E2SeedChoice,
    /// Optional raw `InflationSeed` override (used when callers/tests pass an unclamped `InflationSeed` directly).
    pub seed_override: Option<InflationSeed>,
    pub steps: usize,
    pub max_draw_faces: usize,
    pub last_shells: Vec<ShellStats>,
    pub last_readout: String,
    pub last_stopped: Option<String>,
    pub last_error: Option<String>,
}

impl Default for GalleryInflationState {
    fn default() -> Self {
        Self {
            enabled: true,
            rule: InflationRule::Bdf,
            seed_choice: E2SeedChoice::MostSymmetricVertex,
            seed_override: None,
            steps: 4,
            max_draw_faces: MAX_E2_DRAW_FACES,
            last_shells: Vec::new(),
            last_readout: String::new(),
            last_stopped: None,
            last_error: None,
        }
    }
}

/// Boundary vertex count `∂V` of a shell: every vertex with incomplete patch star is tallied in `boundary_vertex_types`.
pub fn shell_boundary_vertices(s: &ShellStats) -> usize {
    s.boundary_vertex_types.values().sum()
}

impl GalleryInflationState {
    pub(super) fn clear_status(&mut self) {
        self.last_shells.clear();
        self.last_readout.clear();
        self.last_stopped = None;
        self.last_error = None;
    }

    pub(super) fn build_scene(&mut self, cell: &Cell2D, show_unit_cell: bool) -> Result<(Scene2D, i32), String> {
        match build_inflated_patch(
            cell,
            self.seed_choice,
            self.seed_override,
            &self.rule,
            self.steps,
            self.max_draw_faces,
            show_unit_cell,
        ) {
            Ok((scene, overlay_repeat, shells, readout, stopped)) => {
                self.last_shells = shells;
                self.last_readout = readout;
                self.last_stopped = stopped;
                self.last_error = None;
                Ok((scene, overlay_repeat))
            }
            Err(e) => {
                self.last_error = Some(e.clone());
                self.last_shells.clear();
                self.last_readout.clear();
                self.last_stopped = None;
                Err(e)
            }
        }
    }

    /// Plain default controls: `Layers` slider (`0..=16`), stop/error/non-disk banners, and per-layer readout table.
    pub(super) fn show_default_controls(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) -> bool {
        let mut changed = false;
        if self.enabled {
            let mut shown_steps = self.steps.min(MAX_E2_UI_LAYERS);
            ui.horizontal(|ui| {
                ui.label("Layers:");
                if ui.add(egui::Slider::new(&mut shown_steps, 0..=MAX_E2_UI_LAYERS)).changed() {
                    self.steps = shown_steps;
                    changed = true;
                }
            });
            if let Some(err) = &self.last_error {
                ui.colored_label(tokens.status_err, egui::RichText::new(err).small());
                return changed;
            }
            if let Some(stop) = &self.last_stopped {
                ui.colored_label(tokens.status_warn, egui::RichText::new(stop).small());
            }
            for s in &self.last_shells {
                if s.topology.chi != 1 {
                    ui.colored_label(
                        tokens.status_warn,
                        egui::RichText::new(format!("Layer {} is not a disk (χ = {})", s.n, s.topology.chi)).small(),
                    );
                }
            }
            if !self.last_shells.is_empty() {
                egui::ScrollArea::vertical().id_salt("gallery_inflation_scroll").max_height(135.0).show(ui, |ui| {
                    egui::Grid::new("gallery_inflation_shell_table")
                        .num_columns(6)
                        .spacing([8.0, 2.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for h in ["Layer", "Cells", "New", "∂V", "∂E", "χ"] {
                                ui.small(egui::RichText::new(h).strong());
                            }
                            ui.end_row();
                            for s in &self.last_shells {
                                ui.small(s.n.to_string());
                                ui.small(s.topology.faces.to_string());
                                ui.small(s.new_tiles.to_string());
                                ui.small(shell_boundary_vertices(s).to_string());
                                ui.small(s.topology.boundary_edges.to_string());
                                ui.small(s.topology.chi.to_string());
                                ui.end_row();
                            }
                        });
                });
            }
        }
        changed
    }

    /// Collapsible `"More"` section (`default_open(false)`): `Seed`, `Rule`, and `View` (`Open patch` vs `Periodic grid`).
    /// Returns `(inflation_changed, mode_toggled_to, repeat_changed_to)`.
    pub(super) fn show_more_controls(
        &mut self,
        ui: &mut egui::Ui,
        cell: &Cell2D,
        current_repeat: i32,
    ) -> (bool, Option<bool>, Option<i32>) {
        let mut infl_changed = false;
        let mut mode_toggle = None;
        let mut new_repeat = None;
        egui::CollapsingHeader::new("More").id_salt("gallery_growth_more").default_open(false).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("View:");
                let mut en = self.enabled;
                let mut m = false;
                m |= ui.selectable_value(&mut en, true, "Open patch").changed();
                m |= ui.selectable_value(&mut en, false, "Periodic grid").changed();
                if m {
                    mode_toggle = Some(en);
                }
            });
            if !self.enabled || self.last_error.is_some() {
                ui.horizontal(|ui| {
                    ui.label("Patch repeat:");
                    let mut r = current_repeat;
                    if ui.add(egui::Slider::new(&mut r, 1..=6)).changed() {
                        new_repeat = Some(r);
                    }
                });
            }
            if self.enabled {
                let mut sc = self.seed_choice;
                let mut seed_clicked = false;
                ui.horizontal_wrapped(|ui| {
                    ui.label("Seed:");
                    let is_v = matches!(sc, E2SeedChoice::MostSymmetricVertex | E2SeedChoice::Vertex(_))
                        && self.seed_override.is_none();
                    if ui.selectable_label(is_v, "Vertex (most symmetric)").clicked() {
                        sc = E2SeedChoice::MostSymmetricVertex;
                        seed_clicked = true;
                    }
                    let is_e = matches!(sc, E2SeedChoice::Edge(_)) && self.seed_override.is_none();
                    if ui.selectable_label(is_e, "Edge").clicked() {
                        sc = E2SeedChoice::Edge(0);
                        seed_clicked = true;
                    }
                    let is_t = matches!(sc, E2SeedChoice::Tile(_)) && self.seed_override.is_none();
                    if ui.selectable_label(is_t, "Tile").clicked() {
                        sc = E2SeedChoice::Tile(0);
                        seed_clicked = true;
                    }
                    let is_d = sc == E2SeedChoice::Dirichlet && self.seed_override.is_none();
                    if ui.selectable_label(is_d, "Dirichlet").clicked() {
                        sc = E2SeedChoice::Dirichlet;
                        seed_clicked = true;
                    }
                });
                if matches!(sc, E2SeedChoice::MostSymmetricVertex | E2SeedChoice::Vertex(_)) && cell.basis.len() > 1 {
                    ui.horizontal_wrapped(|ui| {
                        ui.small("Vertex orbit:");
                        if ui
                            .selectable_label(sc == E2SeedChoice::MostSymmetricVertex, "Most symmetric")
                            .clicked()
                        {
                            sc = E2SeedChoice::MostSymmetricVertex;
                            seed_clicked = true;
                        }
                        for s in 0..cell.basis.len() {
                            if ui.selectable_label(sc == E2SeedChoice::Vertex(s), format!("#{s}")).clicked() {
                                sc = E2SeedChoice::Vertex(s);
                                seed_clicked = true;
                            }
                        }
                    });
                }
                if let E2SeedChoice::Edge(mut b) = sc
                    && cell.bonds.len() > 1
                {
                    b = b.min(cell.bonds.len() - 1);
                    ui.horizontal(|ui| {
                        ui.small("Bond #:");
                        if ui.add(egui::DragValue::new(&mut b).range(0..=cell.bonds.len() - 1)).changed() {
                            sc = E2SeedChoice::Edge(b);
                            seed_clicked = true;
                        }
                    });
                }
                if let E2SeedChoice::Tile(mut f) = sc
                    && let Ok(t) = PlaneTiling::new(cell)
                    && t.orbit_count() > 1
                {
                    f = f.min(t.orbit_count() - 1);
                    ui.horizontal(|ui| {
                        ui.small("Tile orbit #:");
                        if ui.add(egui::DragValue::new(&mut f).range(0..=t.orbit_count() - 1)).changed() {
                            sc = E2SeedChoice::Tile(f);
                            seed_clicked = true;
                        }
                    });
                }
                if seed_clicked {
                    self.seed_choice = sc;
                    self.seed_override = None;
                    infl_changed = true;
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("Rule:");
                    infl_changed |= ui.selectable_value(&mut self.rule, InflationRule::Bdf, "Touching").changed();
                    infl_changed |= ui
                        .selectable_value(&mut self.rule, InflationRule::Cluster(CentreSet::MaxPolygon), "Cluster")
                        .changed();
                    infl_changed |= ui.selectable_value(&mut self.rule, InflationRule::Auto, "Auto").changed();
                });
                if !self.last_readout.is_empty() {
                    ui.small(&self.last_readout);
                }
            }
        });
        (infl_changed, mode_toggle, new_repeat)
    }
}

type InflatedPatchOut = (Scene2D, i32, Vec<ShellStats>, String, Option<String>);

fn build_inflated_patch(
    cell: &Cell2D,
    seed_choice: E2SeedChoice,
    seed_override: Option<InflationSeed>,
    rule: &InflationRule,
    steps: usize,
    max_faces: usize,
    show_unit_cell: bool,
) -> Result<InflatedPatchOut, String> {
    let tiling = PlaneTiling::new(cell)?;
    let seed = match seed_override {
        Some(s) => s,
        None => seed_choice.resolve(cell, &tiling)?,
    };
    let mut res = match inflate_with(cell, seed, rule, steps) {
        Ok(r) => r,
        Err(e) if e.contains("is not a closed disk") => {
            let (sf, _) = tiling.seed_faces_with_status(seed)?;
            tiling.grow(sf, rule, steps)?
        }
        Err(e) => return Err(e),
    };
    let mut stopped = None;
    if let Some(k) = res.shells.iter().position(|s| s.topology.faces > max_faces) {
        let needed = res.shells[k].topology.faces;
        if k == 0 {
            return Err(format!("layer 0 ({needed} cells) exceeds the {max_faces}-cell drawing cap"));
        }
        stopped = Some(format!(
            "Stopped before layer {k}: layer {k} has {needed} cells, exceeding the {max_faces}-cell cap (showing layers 0..={}).",
            k - 1
        ));
        res.shells.truncate(k);
        res.faces.retain(|f| f.shell < k);
    }
    let (scene, overlay_repeat) = compile_inflation_scene(cell, &tiling, &res, show_unit_cell)?;
    Ok((scene, overlay_repeat, res.shells, res.readout, stopped))
}

fn compile_inflation_scene(
    cell: &Cell2D,
    tiling: &PlaneTiling,
    res: &InflationResult,
    show_unit_cell: bool,
) -> Result<(Scene2D, i32), String> {
    let mut polygons = Vec::with_capacity(res.faces.len() + usize::from(show_unit_cell));
    let mut unique_sites: BTreeMap<VertexId, [f32; 2]> = BTreeMap::new();
    let mut unique_edges: BTreeSet<(VertexId, VertexId)> = BTreeSet::new();
    let mut max_abs = 0_i32;

    for face in &res.faces {
        let verts = tiling.face_vertices((face.orbit, face.offset));
        let pts: Vec<[f32; 2]> = face.polygon.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
        if pts.len() >= 3 {
            polygons.push(PolyPrim {
                points: pts,
                fill: SHELL_FILL_PALETTE[face.shell % SHELL_FILL_PALETTE.len()],
                stroke: Some((SHELL_BORDER_RGBA, 0.8)),
            });
        }
        for &(site, off) in &verts {
            max_abs = max_abs.max(off[0].saturating_abs()).max(off[1].saturating_abs());
            if let Entry::Vacant(e) = unique_sites.entry((site, off)) {
                let Some(&b) = cell.basis.get(site) else {
                    return Err(format!("vertex site {site} out of range"));
                };
                let pos = [
                    (b[0] + f64::from(off[0]) * cell.a1[0] + f64::from(off[1]) * cell.a2[0]) as f32,
                    (b[1] + f64::from(off[0]) * cell.a1[1] + f64::from(off[1]) * cell.a2[1]) as f32,
                ];
                e.insert(pos);
            }
        }
        for k in 0..verts.len() {
            let (u, v) = (verts[k], verts[(k + 1) % verts.len()]);
            if u != v {
                unique_edges.insert(if u < v { (u, v) } else { (v, u) });
            }
        }
    }

    let mut lines = Vec::with_capacity(unique_edges.len() + if show_unit_cell { 4 } else { 0 });
    for (u, v) in &unique_edges {
        if let (Some(&pa), Some(&pb)) = (unique_sites.get(u), unique_sites.get(v)) {
            lines.push(LinePrim {
                a: pa,
                b: pb,
                color: bond_palette(0),
                width: 1.85,
                dashed: false,
                arrow_mid: false,
            });
        }
    }
    let bond_count = lines.len();

    let mut labels = Vec::new();
    if show_unit_cell {
        let a1 = [cell.a1[0] as f32, cell.a1[1] as f32];
        let a2 = [cell.a2[0] as f32, cell.a2[1] as f32];
        let (p00, p10, p11, p01) = ([0.0_f32, 0.0], a1, [a1[0] + a2[0], a1[1] + a2[1]], a2);
        polygons.push(PolyPrim { points: vec![p00, p10, p11, p01], fill: SCENE_UNIT_CELL_FILL, stroke: None });
        for (p, q) in [(p00, p10), (p10, p11), (p11, p01), (p01, p00)] {
            lines.push(LinePrim {
                a: p,
                b: q,
                color: SCENE_UNIT_CELL_EDGE,
                width: 1.6,
                dashed: true,
                arrow_mid: false,
            });
        }
        for (text, pos) in [("a1", edge_label_pos(a1, a2)), ("a2", edge_label_pos(a2, a1))] {
            labels.push(TextPrim { pos, text: String::from(text), color: SCENE_UNIT_CELL_LABEL, size_px: 12.0 });
        }
    }

    let circles: Vec<CirclePrim> = unique_sites
        .iter()
        .map(|(&(site, _), &center)| CirclePrim {
            center,
            radius_world: 0.075,
            min_px: 2.6,
            max_px: 7.2,
            fill: site_palette(site),
            stroke: ([18, 36, 56, 255], 1.15),
        })
        .collect();
    let site_count = circles.len();

    let bbox = compute_bbox(
        polygons
            .iter()
            .flat_map(|p| p.points.iter().copied())
            .chain(lines.iter().flat_map(|l| [l.a, l.b]))
            .chain(circles.iter().map(|c| c.center)),
    );
    let scene = Scene2D::VectorScene {
        primitives: ScenePrimitives {
            polygons,
            lines,
            circles,
            labels,
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: String::new(),
            note: String::new(),
            bbox,
            site_count,
            bond_count,
        },
    };
    let overlay_repeat = (2 * max_abs + 1).clamp(1, 9);
    Ok((scene, overlay_repeat))
}
