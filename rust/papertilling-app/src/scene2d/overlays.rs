//! Gallery scientific overlays: the ITA symmetry elements of the lattice's plane group (mirrors solid, glides
//! dashed with chevrons along the glide, 2/3/4/6-fold centres as lens / triangle / square / hexagon) and the
//! Voronoi cells of the sites, both over the same patch of cells as the drawn lattice.
//!
//! Two layers: Voronoi cells are drawn under the bonds and sites (`under`), symmetry elements over them (`over`),
//! since sites often sit on rotation centres and mirror lines. Colours are the `Paper` tokens (the export draws
//! them as compiled); the live `Studio` canvas maps them through `ThemeTokens`.

use papertilling_core::plane::{RotationCentre2D, SymLineKind, SymmetryElements2D, compute_symmetry_elements, voronoi_cells_of_cell};
use papertilling_core::types::{Cell2D, Frame2D};

use super::{LinePrim, PolyPrim, ScenePrimitives, compute_bbox};
use crate::theme::tokens::{SCENE_ITA_GLIDE, SCENE_ITA_MIRROR, SCENE_ITA_ROTATION, SCENE_VORONOI_EDGE, SCENE_VORONOI_FILL};

/// Which overlays the Gallery shows (view-only state: never changes the lattice).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GalleryOverlayToggles {
    pub symmetry: bool,
    pub voronoi: bool,
}

impl GalleryOverlayToggles {
    pub fn any(self) -> bool {
        self.symmetry || self.voronoi
    }
}

/// Overlay geometry of one drawn Gallery cell.
#[derive(Clone, Debug, Default)]
pub struct GalleryOverlay {
    /// Rotation centres in the patch (for the Wyckoff hover read-out).
    pub centres: Vec<RotationCentre2D>,
    /// Drawn under the lattice (Voronoi cells).
    pub under: ScenePrimitives,
    /// Drawn over the lattice (mirrors, glides, rotation symbols).
    pub over: ScenePrimitives,
    /// Why an overlay could not be built (shown in the inspector), if any.
    pub note: Option<String>,
}

/// World size of the rotation symbols: a fixed fraction of the shorter cell vector.
fn symbol_size(cell: &Cell2D) -> f32 {
    let l1 = cell.a1[0].hypot(cell.a1[1]);
    let l2 = cell.a2[0].hypot(cell.a2[1]);
    (0.07 * l1.min(l2)) as f32
}

/// Build the overlays of `cell` (group `group`, frame `frame`) over the patch of `repeat` x `repeat` cells drawn by
/// `compile_periodic_cell`.
pub fn build_gallery_overlay(cell: &Cell2D, group: &str, frame: Frame2D, repeat: i32, show: GalleryOverlayToggles) -> GalleryOverlay {
    let rep = repeat.clamp(1, 9);
    let mut out = GalleryOverlay::default();
    let mut notes = Vec::new();
    if show.voronoi {
        match voronoi_cells_of_cell(cell) {
            Ok(cells) => {
                let n_lo = -(rep / 2);
                for u in n_lo..n_lo + rep {
                    for v in n_lo..n_lo + rep {
                        let sh = [
                            (f64::from(u) * cell.a1[0] + f64::from(v) * cell.a2[0]) as f32,
                            (f64::from(u) * cell.a1[1] + f64::from(v) * cell.a2[1]) as f32,
                        ];
                        for vc in &cells {
                            let points = vc.vertices.iter().map(|p| [p[0] as f32 + sh[0], p[1] as f32 + sh[1]]).collect();
                            out.under.polygons.push(PolyPrim { points, fill: SCENE_VORONOI_FILL, stroke: Some((SCENE_VORONOI_EDGE, 0.9)) });
                        }
                    }
                }
            }
            Err(e) => notes.push(format!("Voronoi cells unavailable: {e}")),
        }
    }
    if show.symmetry {
        match compute_symmetry_elements(group, cell, frame, [rep, rep]) {
            Ok(el) => {
                push_symmetry(&mut out.over, &el, symbol_size(cell));
                out.centres = el.centres;
            }
            Err(e) => notes.push(format!("Symmetry elements unavailable: {e}")),
        }
    }
    for layer in [&mut out.under, &mut out.over] {
        layer.bbox = compute_bbox(layer.polygons.iter().flat_map(|p| p.points.iter().copied()).chain(layer.lines.iter().flat_map(|l| [l.a, l.b])));
    }
    out.note = (!notes.is_empty()).then(|| notes.join("; "));
    out
}

fn f32p(p: [f64; 2]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

fn push_symmetry(layer: &mut ScenePrimitives, el: &SymmetryElements2D, size: f32) {
    for ln in &el.lines {
        let (a, b) = (f32p(ln.a), f32p(ln.b));
        match ln.kind {
            SymLineKind::Mirror => {
                layer.lines.push(LinePrim { a, b, color: SCENE_ITA_MIRROR, width: 2.2, dashed: false, arrow_mid: false });
            }
            SymLineKind::Glide => {
                // one chevron per glide translation along the line (at most 12 per segment)
                let len = (b[0] - a[0]).hypot(b[1] - a[1]);
                let g = (ln.glide[0].hypot(ln.glide[1]) as f32).max(len / 12.0).max(1e-6);
                let n = ((len / g).round() as usize).clamp(1, 12);
                for k in 0..n {
                    let (s, t) = (k as f32 / n as f32, (k + 1) as f32 / n as f32);
                    let p = [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])];
                    let q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    layer.lines.push(LinePrim { a: p, b: q, color: SCENE_ITA_GLIDE, width: 1.8, dashed: true, arrow_mid: true });
                }
            }
        }
    }
    for c in &el.centres {
        let points = rotation_symbol(f32p(c.pos), c.order, size);
        if points.len() >= 3 {
            layer.polygons.push(PolyPrim { points, fill: SCENE_ITA_ROTATION, stroke: Some(([255, 255, 255, 230], 1.0)) });
        }
    }
}

/// ITA graphical symbol of an n-fold centre at `p`: lens (2), triangle (3), square (4), hexagon (6).
pub fn rotation_symbol(p: [f32; 2], order: u8, size: f32) -> Vec<[f32; 2]> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let ring = |n: usize, r: f32, phase: f32| -> Vec<[f32; 2]> {
        (0..n).map(|k| {
            let t = phase + 2.0 * PI * k as f32 / n as f32;
            [p[0] + r * t.cos(), p[1] + r * t.sin()]
        }).collect()
    };
    match order {
        // lens: two circular arcs of half-angle 60 degrees meeting at (+-size, 0)
        2 => {
            let r = size / (PI / 3.0).sin();
            let h = r * (PI / 3.0).cos();
            let arc = |cy: f32, from: f32, to: f32| -> Vec<[f32; 2]> {
                (0..=8).map(|k| {
                    let t = from + (to - from) * k as f32 / 8.0;
                    [p[0] + r * t.cos(), p[1] + cy + r * t.sin()]
                }).collect::<Vec<_>>()
            };
            let mut pts = arc(-h, FRAC_PI_2 - PI / 3.0, FRAC_PI_2 + PI / 3.0);
            pts.extend(arc(h, -FRAC_PI_2 - PI / 3.0, -FRAC_PI_2 + PI / 3.0).into_iter().skip(1).take(7));
            pts
        }
        3 => ring(3, 1.15 * size, FRAC_PI_2),
        4 => ring(4, 1.05 * size, PI / 4.0),
        6 => ring(6, 1.05 * size, 0.0),
        _ => Vec::new(),
    }
}

/// The rotation centre within `radius` (world units) of `world`, nearest first (for the hover read-out).
pub fn centre_near(centres: &[RotationCentre2D], world: [f64; 2], radius: f64) -> Option<&RotationCentre2D> {
    centres
        .iter()
        .map(|c| ((c.pos[0] - world[0]).hypot(c.pos[1] - world[1]), c))
        .filter(|(d, _)| *d <= radius)
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .map(|(_, c)| c)
}
