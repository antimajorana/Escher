//! View-only BDF / cluster shell inflation overlay and inspector controls for the Gallery tab.

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

const SHELL_FILL_PALETTE: [[u8; 4]; 6] = [
    [244, 196, 68, 135],
    [92, 168, 235, 130],
    [88, 198, 168, 130],
    [238, 116, 98, 130],
    [174, 128, 226, 130],
    [166, 194, 88, 130],
];
const SHELL_BORDER_RGBA: [u8; 4] = [100, 112, 128, 90];

/// View-only shell inflation state for the Gallery tab (never changes `active_item` or `revision`).
#[derive(Clone, Debug, PartialEq)]
pub struct GalleryInflationState {
    pub enabled: bool,
    pub rule: InflationRule,
    pub seed: InflationSeed,
    pub steps: usize,
    pub last_shells: Vec<ShellStats>,
    pub last_readout: String,
    pub last_error: Option<String>,
}

impl Default for GalleryInflationState {
    fn default() -> Self {
        Self {
            enabled: false,
            rule: InflationRule::Auto,
            seed: InflationSeed::Dirichlet,
            steps: 2,
            last_shells: Vec::new(),
            last_readout: String::new(),
            last_error: None,
        }
    }
}

impl GalleryInflationState {
    pub(super) fn clear_status(&mut self) {
        self.last_shells.clear();
        self.last_readout.clear();
        self.last_error = None;
    }

    pub(super) fn build_scene(&mut self, cell: &Cell2D, show_unit_cell: bool) -> Result<(Scene2D, i32), String> {
        match build_inflated_patch(cell, self.seed, &self.rule, self.steps, show_unit_cell) {
            Ok((scene, overlay_repeat, shells, readout)) => {
                self.last_shells = shells;
                self.last_readout = readout;
                self.last_error = None;
                Ok((scene, overlay_repeat))
            }
            Err(e) => {
                self.last_error = Some(e.clone());
                self.last_shells.clear();
                self.last_readout.clear();
                Err(e)
            }
        }
    }

    /// Render the inflation rule, seed, step controls, readout, and compact shell table; returns `true` if changed.
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Rule:");
            changed |= ui.selectable_value(&mut self.rule, InflationRule::Auto, "Auto").changed();
            changed |= ui.selectable_value(&mut self.rule, InflationRule::Bdf, "Plain BDF").changed();
            changed |= ui
                .selectable_value(&mut self.rule, InflationRule::Cluster(CentreSet::MaxPolygon), "Cluster")
                .changed();
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Seed:");
            changed |= ui.selectable_value(&mut self.seed, InflationSeed::Dirichlet, "Dirichlet").changed();
            changed |= ui.selectable_value(&mut self.seed, InflationSeed::Face(0), "Face 0").changed();
            changed |= ui.selectable_value(&mut self.seed, InflationSeed::VertexStar(0), "Vertex 0").changed();
        });
        ui.horizontal(|ui| {
            ui.label("Shells n:");
            changed |= ui.add(egui::Slider::new(&mut self.steps, 0..=5)).changed();
        });
        if let Some(err) = &self.last_error {
            ui.colored_label(tokens.status_err, egui::RichText::new(err).small());
            return changed;
        }
        if !self.last_readout.is_empty() {
            ui.small(&self.last_readout);
        }
        if !self.last_shells.is_empty() {
            egui::Grid::new("gallery_inflation_shell_table")
                .num_columns(6)
                .spacing([8.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    for h in ["n", "|S~_n|", "N_n", "ratio", "filled", "χ"] {
                        ui.small(egui::RichText::new(h).strong());
                    }
                    ui.end_row();
                    for s in &self.last_shells {
                        ui.small(s.n.to_string());
                        ui.small(s.topology.faces.to_string());
                        ui.small(s.new_tiles.to_string());
                        ui.small(s.growth.map_or("-".to_string(), |r| format!("{r:.3}")));
                        ui.small(s.filled.to_string());
                        ui.small(s.topology.chi.to_string());
                        ui.end_row();
                    }
                });
        }
        changed
    }
}

fn build_inflated_patch(
    cell: &Cell2D,
    seed: InflationSeed,
    rule: &InflationRule,
    steps: usize,
    show_unit_cell: bool,
) -> Result<(Scene2D, i32, Vec<ShellStats>, String), String> {
    let res = inflate_with(cell, seed, rule, steps)?;
    let tiling = PlaneTiling::new(cell)?;
    let (scene, overlay_repeat) = compile_inflation_scene(cell, &tiling, &res, show_unit_cell)?;
    Ok((scene, overlay_repeat, res.shells, res.readout))
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
