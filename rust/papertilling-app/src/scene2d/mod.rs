use papertilling_core::types::{BuiltCrystal, Cell2D, SurfaceMesh};

use crate::scene_crystal::compile_crystal_patch;

mod overlays;
pub(crate) mod periodic;
mod surface_net;

#[derive(Clone)]
pub enum Scene2D {
    PeriodicCell {
        cell: Cell2D,
        repeat: i32,
        show_unit_cell: bool,
    },
    SurfaceNet {
        mesh: SurfaceMesh,
    },
    CrystalPatch {
        crystal: BuiltCrystal,
        title: String,
        note: String,
    },
    VectorScene {
        primitives: ScenePrimitives,
    },
}

#[derive(Clone, Debug)]
pub struct PolyPrim {
    pub points: Vec<[f32; 2]>,
    pub fill: [u8; 4],
    pub stroke: Option<([u8; 4], f32)>,
}

#[derive(Clone, Debug)]
pub struct LinePrim {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub color: [u8; 4],
    pub width: f32,
    pub dashed: bool,
    pub arrow_mid: bool,
}

#[derive(Clone, Debug)]
pub struct CirclePrim {
    pub center: [f32; 2],
    pub radius_world: f32,
    pub min_px: f32,
    pub max_px: f32,
    pub fill: [u8; 4],
    pub stroke: ([u8; 4], f32),
}

#[derive(Clone, Debug)]
pub struct TextPrim {
    pub pos: [f32; 2],
    pub text: String,
    pub color: [u8; 4],
    pub size_px: f32,
}

#[derive(Clone, Debug)]
pub struct ScenePrimitives {
    pub polygons: Vec<PolyPrim>,
    pub lines: Vec<LinePrim>,
    pub circles: Vec<CirclePrim>,
    pub labels: Vec<TextPrim>,
    pub site_circle_indices: Vec<Option<usize>>,
    pub edge_line_indices: Vec<Option<usize>>,
    pub title: String,
    pub note: String,
    pub bbox: ([f32; 2], [f32; 2]),
    pub site_count: usize,
    pub bond_count: usize,
}

pub(crate) fn compile_scene(scene: &Scene2D) -> ScenePrimitives {
    match scene {
        Scene2D::PeriodicCell {
            cell,
            repeat,
            show_unit_cell,
        } => periodic::compile_periodic_cell(cell, *repeat, *show_unit_cell),
        Scene2D::SurfaceNet { mesh } => surface_net::compile_surface_net(mesh),
        Scene2D::CrystalPatch {
            crystal,
            title,
            note,
        } => compile_crystal_patch(crystal, title, note),
        Scene2D::VectorScene { primitives } => primitives.clone(),
    }
}

pub(crate) fn bond_palette(kind: usize) -> [u8; 4] {
    match kind % 5 {
        0 => [44, 62, 80, 255],
        1 => [192, 57, 43, 255],
        2 => [34, 139, 94, 255],
        3 => [130, 68, 173, 255],
        _ => [196, 125, 24, 255],
    }
}

pub(crate) fn site_palette(orbit: usize) -> [u8; 4] {
    match orbit % 5 {
        0 => [31, 119, 180, 255],
        1 => [230, 97, 35, 255],
        2 => [27, 158, 119, 255],
        3 => [117, 112, 179, 255],
        _ => [214, 48, 114, 255],
    }
}

pub(super) fn edge_glue_palette(idx: usize) -> [u8; 4] {
    match idx % 6 {
        0 => [214, 45, 32, 245],
        1 => [25, 108, 198, 245],
        2 => [28, 152, 92, 245],
        3 => [212, 132, 18, 245],
        4 => [142, 58, 178, 245],
        _ => [18, 152, 168, 245],
    }
}

pub(crate) fn ring_tint_rgba(sides: usize) -> [u8; 4] {
    match sides {
        3 => [244, 196, 68, 145],
        4 => [92, 168, 235, 145],
        5 => [238, 104, 88, 160],
        6 => [88, 198, 168, 140],
        7 => [174, 122, 228, 160],
        8 => [224, 116, 182, 155],
        _ => [172, 194, 92, 150],
    }
}

pub(crate) fn compute_bbox(pts: impl Iterator<Item = [f32; 2]>) -> ([f32; 2], [f32; 2]) {
    let mut min_p = [f32::INFINITY; 2];
    let mut max_p = [f32::NEG_INFINITY; 2];
    let mut any = false;
    for p in pts {
        if p[0].is_finite() && p[1].is_finite() {
            min_p[0] = min_p[0].min(p[0]);
            min_p[1] = min_p[1].min(p[1]);
            max_p[0] = max_p[0].max(p[0]);
            max_p[1] = max_p[1].max(p[1]);
            any = true;
        }
    }
    if any {
        (min_p, max_p)
    } else {
        ([-1.0, -1.0], [1.0, 1.0])
    }
}
