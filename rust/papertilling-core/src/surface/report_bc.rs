//! Boundary conditions of a 2D lattice as one entry point (C1 WP-B1, plan §2.5 and §5 T3): open, periodic on 1, 2
//! or 3 pairs of sides, twisted (Möbius, Klein), with reasoned refusals and the view flags that WP-U3 renders.
//! The 8-row `surface_table` is unchanged; `Open` and `TorusHex` are targets here only.

use super::flat::{NO_GLIDE, klein_bottle, mobius_band, torus};
use super::flat_bc::{cylinder_wrap, open_patch, simple_quotient, torus_hex_domain};
use super::glides::{glide_power, glide_search};
use super::{RefusalKind, SurfaceError, check_surface};
use crate::types::{Cell2D, SurfaceMesh};

/// Label of every 3D picture that is not an isometric embedding (same wording as the frieze ribbons, WP-B2).
pub const NOT_ISOMETRIC_NOTE: &str = "topological picture, not isometric";

/// A boundary condition on a 2D lattice. Vectors are integer, in units of the cell vectors `a1, a2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// No gluing: the vertex star of the most symmetric vertex grown `layers` times by the touching rule (a disk).
    Open { layers: usize },
    /// 1 pair, by the translation `wrap` (any non-zero lattice vector, e.g. a (6, 0) tube), `rows` rows long.
    Cylinder { wrap: [i32; 2], rows: i32 },
    /// 2 pairs: the torus `R^2 / span(t1, t2)` on its parallelogram.
    Torus { t1: [i32; 2], t2: [i32; 2] },
    /// 3 pairs: the same torus drawn on the Wigner-Seitz cell of `span(t1, t2)`.
    TorusHex { t1: [i32; 2], t2: [i32; 2] },
    /// 1 twisted pair (open width): glide class `glide`, closed after `steps` glide steps, `width` rows each side.
    Mobius { glide: usize, steps: i32, width: i32 },
    /// 1 twisted pair and 1 periodic pair: glide class `glide`, `steps` glide steps, `across` translations across.
    Klein { glide: usize, steps: i32, across: i32 },
}

/// A built boundary condition: the mesh plus the view flags (plan §2.5, "no silent non-isometric pictures").
#[derive(Clone, Debug)]
pub struct BcSurface {
    pub mesh: SurfaceMesh,
    /// The default view is the exact flat domain with its edge identifications (always true for flat surfaces).
    pub exact_view: bool,
    /// Caption required on the 3D picture, or `None` when the 3D picture is isometric (the cylinder) or flat.
    pub picture_note: Option<&'static str>,
    /// Plain-words read-out lines (surface, χ, boundary, and any warning such as "3 pairs reduce to 2").
    pub readout: Vec<String>,
}

/// Caption of the 3D picture of `bc` ([`NOT_ISOMETRIC_NOTE`] for the torus, Möbius band and Klein bottle: a flat
/// torus has no C² isometric embedding in R³, a Klein bottle no embedding at all, and the twisted strip is not
/// shown to be developable; `None` for the rolled cylinder, which is isometric, and for the flat open patch).
pub fn picture_note(bc: &BoundaryCondition) -> Option<&'static str> {
    match bc {
        BoundaryCondition::Open { .. } | BoundaryCondition::Cylinder { .. } => None,
        _ => Some(NOT_ISOMETRIC_NOTE),
    }
}

/// Whether the default view is the exact flat domain (true for every boundary condition here).
pub fn exact_view(_bc: &BoundaryCondition) -> bool {
    true
}

fn classify(msg: String) -> SurfaceError {
    if let Some(rest) = msg.strip_prefix("IMPOSSIBLE (lattice): ") {
        SurfaceError::impossible_lattice(rest)
    } else if msg.starts_with("NOT FOUND") {
        SurfaceError::not_found(msg, String::new(), true)
    } else {
        SurfaceError::invalid(msg)
    }
}

/// Build `bc` on `cell`, refusing with a reason: `IMPOSSIBLE (lattice)` for a twisted gluing on a decorated lattice
/// with no mirror and no glide ([`glide_search`] is complete), `NOT FOUND (too small)` for a quotient that is not a
/// simple graph, invalid arguments otherwise. Every built mesh also passes [`check_surface`].
pub fn build_boundary(cell: &Cell2D, bc: &BoundaryCondition) -> Result<BcSurface, SurfaceError> {
    let mut readout = Vec::new();
    let mesh = match *bc {
        BoundaryCondition::Open { layers } => {
            let m = open_patch(cell, layers).map_err(classify)?;
            if m.chi != 1 {
                readout.push(format!("layer is not a disk (chi = {})", m.chi));
            }
            m
        }
        BoundaryCondition::Cylinder { wrap, rows } => cylinder_wrap(cell, wrap, rows).map_err(classify)?,
        BoundaryCondition::Torus { t1, t2 } => {
            simple_quotient(cell, &[t1, t2], None).map_err(classify)?;
            torus(cell, t1, t2).map_err(classify)?
        }
        BoundaryCondition::TorusHex { t1, t2 } => {
            let (m, rect) = torus_hex_domain(cell, t1, t2).map_err(classify)?;
            if rect {
                readout.push(
                    "3 pairs reduce to 2 for this supercell: its Wigner-Seitz cell is a rectangle; choose a \
                     non-rectangular supercell for a hexagonal domain"
                        .to_string(),
                );
            }
            m
        }
        BoundaryCondition::Mobius { glide, steps, width } => {
            let g = twisted_glide(cell, glide, steps)?;
            simple_quotient(cell, &[[steps * g.tau[0], steps * g.tau[1]]], Some((g.m, g.t))).map_err(classify)?;
            mobius_band(cell, glide, steps, width).map_err(classify)?
        }
        BoundaryCondition::Klein { glide, steps, across } => {
            let g = twisted_glide(cell, glide, steps)?;
            let gens = [[steps * g.tau[0], steps * g.tau[1]], [across * g.across[0], across * g.across[1]]];
            simple_quotient(cell, &gens, Some((g.m, g.t))).map_err(classify)?;
            klein_bottle(cell, glide, steps, across).map_err(classify)?
        }
    };
    let rep = check_surface(&mesh);
    if !rep.ok {
        return Err(SurfaceError::not_found("NOT FOUND: the built surface fails its checks", format!("{rep:?}"), true));
    }
    let (v, e, f) = (mesh.sites.len(), mesh.bonds.len(), mesh.faces.len());
    readout.insert(0, format!("{}: V = {v}, E = {e}, F = {f}, chi = {}", mesh.name, v as i64 - e as i64 + f as i64));
    Ok(BcSurface { picture_note: picture_note(bc), exact_view: exact_view(bc), mesh, readout })
}

/// The class element used by a twisted gluing, or the refusal: `IMPOSSIBLE (lattice)` when the decorated lattice
/// has no mirror or glide (proved by the complete search), invalid when `steps` is not allowed for the class.
fn twisted_glide(cell: &Cell2D, glide: usize, steps: i32) -> Result<super::GlideAxis, SurfaceError> {
    let gl = glide_search(cell).map_err(classify)?;
    if gl.is_empty() {
        return Err(classify(NO_GLIDE.to_string()));
    }
    glide_power(&gl[glide % gl.len()], steps).map_err(SurfaceError::invalid)
}

/// How the pairs of sides of a fundamental polygon are glued (for [`pairs_refusal`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairsGluing {
    /// Every pair by a translation.
    Translations,
    /// One pair by a glide (Möbius or Klein), the others by translations.
    OneTwisted,
    /// Every pair by a glide (two pairs: the projective plane).
    AllTwisted,
}

/// Refusal of a gluing of `pairs` pairs of sides that no flat boundary condition here realises, or `None` when one
/// does (1 to 3 translation pairs, 1 or 2 pairs with one twisted). Only the projective plane is `IMPOSSIBLE`
/// (proved); 4 or more pairs are refused as not offered, with the proved (χ = 0) and cited (parallelogons) reasons.
pub fn pairs_refusal(pairs: u32, gluing: PairsGluing) -> Option<SurfaceError> {
    let flat_chi0 = "[PROVED] a closed flat surface has chi = 0 (Gauss-Bonnet: its total curvature 2 pi chi is 0), so it \
                     is a torus or a Klein bottle, both reached with 2 pairs of sides";
    match (pairs, gluing) {
        (0, _) => Some(SurfaceError::invalid("a gluing needs at least one pair of sides (0 pairs is the open patch)")),
        (2, PairsGluing::AllTwisted) => Some(SurfaceError::impossible_lattice(
            "[PROVED] both pairs twisted give the projective plane, chi = 1, which has no flat metric (Gauss-Bonnet); \
             it exists only as the cone polyhedron: a cone metric with curvature +360 deg at its cone points, never flat",
        )),
        (3, PairsGluing::Translations) | (1 | 2, _) => None,
        (3, _) => Some(SurfaceError::invalid(
            "hexagonal (3-pair) domains carry only the periodic torus in C1; Moebius and Klein gluings use square or \
             rectangular domains (user decision Q7)",
        )),
        (_, PairsGluing::Translations) => Some(SurfaceError::invalid(format!(
            "{pairs} pairs are not offered: {flat_chi0}; and a convex domain that tiles the plane by a lattice is a parallelogram (2 pairs) or a centrally symmetric hexagon (3 pairs) [CITED: Fedorov 1885, as stated by Yang and Zong, arXiv:1711.02514; primary source not checked]"
        ))),
        (_, _) => Some(SurfaceError::invalid(format!(
            "{pairs} pairs are not offered: {flat_chi0}; a gluing that gives any other closed surface carries the lattice only with cone points (a cone metric, never flat)"
        ))),
    }
}

/// `true` when `e` is a proved refusal (`IMPOSSIBLE`).
pub fn is_proved(e: &SurfaceError) -> bool {
    matches!(e.kind, RefusalKind::Impossible)
}
