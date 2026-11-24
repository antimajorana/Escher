//! What one S6 build produces (core result + both scenes + inspector rows), and the shared drawing helpers:
//! 2D xy projections of edge lists, a pair palette for 3D faces, and the inspector table.

use papertilling_core::growth::{LayerStats, Stopped};
use papertilling_core::honeycomb3d::{
    CellComplex3d, DirichletDomain3d, DirichletQuotient, ParacompactBall, PeriodicSurface, TriplyPeriodicNet,
    VoronoiPolyhedron3d,
};

use crate::canvas2d::Scene2D;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{LinePrim, PolyPrim, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;
use crate::workbench::scene_helpers::{EDGE_RGBA, line_prim, polyline_scene};

/// Per-layer growth readout attached to a [`CoreOut::Growth`] build.
#[derive(Clone, Debug)]
pub(crate) struct GrowthMeta {
    pub(crate) stats: Vec<LayerStats>,
    pub(crate) stopped: Option<Stopped>,
    pub(crate) boundary_only: bool,
    pub(crate) banners: Vec<String>,
}

/// The core object behind a build (kept for the inspector and the tests).
pub(crate) enum CoreOut {
    Growth(Box<CellComplex3d>, GrowthMeta),
    Ball(Box<ParacompactBall>),
    Net(Box<TriplyPeriodicNet>),
    Surface(Box<PeriodicSurface>),
    Cell(Box<VoronoiPolyhedron3d>),
    Manifold(Box<DirichletDomain3d>, Box<DirichletQuotient>),
}

impl CoreOut {
    /// Inspector heading: what was built.
    pub(crate) fn heading(&self) -> String {
        match self {
            Self::Growth(cx, _) => format!("Cell complex: {}", cx.name),
            Self::Ball(b) => format!("Paracompact ball: {} (R = {:.2})", b.name, b.radius),
            Self::Net(n) => format!("Net: {}", n.name),
            Self::Surface(s) => format!("Surface: {}", s.name),
            Self::Cell(c) => format!("Convex cell with {} faces", c.faces.len()),
            Self::Manifold(d, q) => format!("Closed 3-manifold: {} (H₁ = {})", d.name, q.h1_label),
        }
    }

    pub(crate) fn complex(&self) -> Option<&CellComplex3d> {
        match self {
            Self::Growth(cx, _) => Some(cx),
            Self::Ball(b) => Some(&b.complex),
            _ => None,
        }
    }
}

/// One finished build: scenes are made on the worker, so `scenes()` only clones.
pub(crate) struct Built {
    pub(crate) core: CoreOut,
    pub(crate) scene2d: Scene2D,
    pub(crate) mesh: GpuMeshData,
    pub(crate) caption: String,
    /// Inspector table: (quantity, value).
    pub(crate) rows: Vec<(String, String)>,
    /// Inspector notes: evidence stamps, provenance, model caveats (shown verbatim).
    pub(crate) notes: Vec<String>,
}

/// Stamp for counts of a generated finite complex that the core does not stamp itself.
pub(crate) const COUNTS_EVIDENCE: &str =
    "[COMPUTATIONAL EVIDENCE: exact counts of the generated finite complex; vertices are merged within a floating-point tolerance]";

/// Edges `[u, v]` of 3D points as 2-point polylines of their xy projection.
pub(crate) fn xy_edges(points: &[[f64; 3]], edges: impl Iterator<Item = (usize, usize)>) -> Vec<Vec<[f64; 2]>> {
    edges
        .filter(|&(u, v)| u < points.len() && v < points.len())
        .map(|(u, v)| vec![[points[u][0], points[u][1]], [points[v][0], points[v][1]]])
        .collect()
}

/// xy projection scene of an edge list (`extra` lines, e.g. a unit circle or cell outline, are added on top).
pub(crate) fn projection_scene(
    points: &[[f64; 3]],
    edges: impl Iterator<Item = (usize, usize)>,
    extra: Vec<LinePrim>,
    title: &str,
    note: &str,
) -> Scene2D {
    let scene = polyline_scene(&xy_edges(points, edges), EDGE_RGBA, title, note);
    if extra.is_empty() { scene } else { crate::workbench::scene_helpers::with_lines(scene, extra) }
}

/// Unit-square outline (the conventional cell seen along z), `reps` cells per side.
pub(crate) fn square_outline(reps: f64) -> Vec<LinePrim> {
    let c = [[0.0, 0.0], [reps, 0.0], [reps, reps], [0.0, reps]];
    (0..4).map(|k| line_prim(c[k], c[(k + 1) % 4], [120, 120, 120, 255], 1.0)).collect()
}

/// Filled cells (heat-map tiles etc.) as a vector scene without sites.
pub(crate) fn tiles_scene(tiles: Vec<PolyPrim>, extra: Vec<LinePrim>, title: &str, note: &str) -> Scene2D {
    let bbox = compute_bbox(tiles.iter().flat_map(|t| t.points.iter().copied()).chain(extra.iter().flat_map(|l| [l.a, l.b])));
    let primitives = ScenePrimitives {
        polygons: tiles,
        lines: extra,
        title: title.to_string(),
        note: note.to_string(),
        bbox,
        ..Default::default()
    };
    Scene2D::VectorScene { primitives }
}

/// Distinct translucent tints for paired faces / shells (cycled).
pub(crate) fn palette_rgba(k: usize, alpha: f32) -> [f32; 4] {
    const P: [[f32; 3]; 8] = [
        [0.90, 0.49, 0.13],
        [0.16, 0.50, 0.73],
        [0.15, 0.68, 0.38],
        [0.56, 0.27, 0.68],
        [0.91, 0.30, 0.24],
        [0.95, 0.77, 0.06],
        [0.10, 0.74, 0.61],
        [0.50, 0.55, 0.55],
    ];
    let c = P[k % P.len()];
    [c[0], c[1], c[2], alpha]
}

/// Inspector body of a build: prominent banners, key-value grid, optional per-layer readout table, and notes.
pub(crate) fn show_built(ui: &mut egui::Ui, b: &Built, tokens: &ThemeTokens, grid_id: &str) {
    ui.strong(b.core.heading());
    match &b.core {
        CoreOut::Growth(_, meta) => {
            let col = if matches!(meta.stopped, Some(Stopped::Closed { .. })) {
                tokens.accent_primary
            } else {
                tokens.status_warn
            };
            for banner in &meta.banners {
                ui.colored_label(col, egui::RichText::new(banner).small().strong());
            }
        }
        CoreOut::Ball(ball) => {
            ui.colored_label(tokens.accent_primary, egui::RichText::new(ball.reason).small());
        }
        _ => {}
    }
    let column_width = ((ui.available_width() - ui.spacing().item_spacing.x) * 0.5).max(1.0);
    egui::Grid::new(grid_id).num_columns(2).max_col_width(column_width).striped(true).show(ui, |ui| {
        for (k, v) in &b.rows {
            ui.add(egui::Label::new(k).wrap());
            ui.add(egui::Label::new(v).wrap());
            ui.end_row();
        }
    });
    if let CoreOut::Growth(_, meta) = &b.core
        && !meta.stats.is_empty()
    {
        ui.add_space(4.0);
        ui.strong(if meta.boundary_only {
            "Layer growth readout (3D draws boundary only)"
        } else {
            "Layer growth readout"
        });
        egui::ScrollArea::vertical().id_salt("wb_s6_growth_scroll").max_height(160.0).show(ui, |ui| {
            egui::Grid::new("wb_s6_growth_layer_table")
                .num_columns(6)
                .spacing([8.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    for h in ["Layer", "Cells", "New", "∂V", "∂E", "χ"] {
                        ui.small(egui::RichText::new(h).strong());
                    }
                    ui.end_row();
                    for (k, s) in meta.stats.iter().enumerate() {
                        ui.small(k.to_string());
                        ui.small(s.cells.to_string());
                        ui.small(s.new_cells.to_string());
                        ui.small(s.boundary_vertices.to_string());
                        ui.small(s.boundary_edges.to_string());
                        ui.small(s.euler.to_string());
                        ui.end_row();
                    }
                });
        });
    }
    egui::CollapsingHeader::new("Geometry and numerical limits").show(ui, |ui| {
        for note in &b.notes {
            ui.add(egui::Label::new(egui::RichText::new(note).small()).wrap());
        }
    });
}
