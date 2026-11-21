//! Frieze strip patches: the open strip, the cylinder and the Möbius strip as true quotients.
//!
//! Lattice: vertex columns one glide (half a period `a`) apart, 3 rows (row 1 is the strip axis).
//! The decoration (kind-2 diagonals, period 2 columns) realises exactly the frieze group named.
//! A closed strip of `N` columns glues column `N` back to column 0 by `g = (x + N/2, ±y)`, which
//! must be an element of the group (C3: the seam is glued, so no two vertices coincide).
//!
//! C2 [PROVED]: a Möbius strip is `strip / <g>` with `g` reversing the strip's transverse direction
//! and acting freely, i.e. `g(x, y) = (x + L, -y)` with `L != 0`. Frieze elements that flip `y` are
//! horizontal mirrors (fixed axis), glides, mirror∘translation, and half-turns (fixed point). So
//! `g` exists iff the group has a horizontal mirror or a glide: p11m, p11g, p2mm, p2mg. In p11m and
//! p2mm, `L` is a whole number of periods; in p11g and p2mg, `L` is an odd number of glides
//! (`L = (k + 1/2) a`). p1, p2 and p1m1 have no such element and are refused.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::analyze_frieze_group;
use crate::bieberbach::BieberbachSummary;

/// Label of every non-isometric 3D frieze picture (the Möbius ribbon), §2.5 of the C1 plan.
pub const NOT_ISOMETRIC_NOTE: &str = "topological picture, not isometric";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FriezeTopology {
    OpenStrip,
    Cylinder,
    Mobius,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FriezeDefect {
    None,
    HalfTurn180,
    GlideDispiration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FriezePatch {
    pub symbol: String,
    pub iucr_name: String,
    pub cells: usize,
    pub topology: FriezeTopology,
    pub defect: FriezeDefect,
    pub summary: BieberbachSummary,
    /// Vertex columns (3 vertices each). Open: `2 cells + 1`; closed: the glued circumference.
    pub columns: usize,
    /// Strip length in periods `a`: open `cells`; cylinder and mirror Möbius `cells`;
    /// glide Möbius `cells + 1/2` (an odd number `2 cells + 1` of glides).
    pub length_periods: f64,
    pub pos2d: Vec<[f64; 2]>,
    pub pos3d: Vec<[f64; 3]>,
    /// `(u, v, kind)`: kind 0 lattice, 1 defect seam, 2 decoration.
    pub edges: Vec<(usize, usize, u8)>,
    pub quads: Vec<[usize; 4]>,
    /// Indices into `edges` of the edges that cross the gluing seam (closed strips only). A flat
    /// 2D view of the fundamental domain draws them as identifications, not as segments.
    pub wrap_edges: Vec<usize>,
    /// Indices into `quads` of the quads that cross the gluing seam.
    pub wrap_quads: Vec<usize>,
    /// `Some(NOT_ISOMETRIC_NOTE)` for the Möbius ribbon; the open strip and the cylinder are isometric.
    pub picture_note: Option<String>,
    /// How the strip was closed, in words.
    pub glue_note: String,
}

/// How a Möbius strip of `cells` periods is glued for `symbol`, or the [PROVED] refusal.
/// Returns the number of vertex columns (glides) of the circumference.
pub fn mobius_columns(symbol: &str, cells: usize) -> Result<usize, String> {
    match symbol {
        "p11m" | "p2mm" => Ok(2 * cells),
        "p11g" | "p2mg" => Ok(2 * cells + 1),
        "p1" | "p2" | "p1m1" => Err(format!(
            "[PROVED] {symbol} cannot be closed into a Möbius strip: that needs a symmetry that flips the \
             strip across its axis without a fixed point (a horizontal mirror or a glide), and {symbol} has none"
        )),
        _ => Err(format!("unknown frieze group {symbol:?}")),
    }
}

/// Decoration diagonals of quad column `c` (between vertex columns `c` and `c+1`), as
/// `(left_row, right_row)`. Rows 0, 1, 2 bottom to top; "T" = rows 1-2, "B" = rows 0-1.
/// Period 2 columns; each pattern realises exactly its frieze group (checked by brute force in tests).
pub fn frieze_decoration(symbol: &str, c: usize) -> &'static [(usize, usize)] {
    const T_UP: (usize, usize) = (1, 2);
    const T_DN: (usize, usize) = (2, 1);
    const B_UP: (usize, usize) = (0, 1);
    const B_DN: (usize, usize) = (1, 0);
    let even = c.is_multiple_of(2);
    match (symbol, even) {
        ("p1", true) => &[T_UP],
        ("p11m", true) => &[T_UP, B_DN],
        ("p1m1", true) | ("p11g", true) | ("p2", true) => &[T_UP],
        ("p1m1", false) => &[T_DN],
        ("p11g", false) => &[B_DN],
        ("p2", false) => &[B_UP],
        ("p2mm", true) => &[T_UP, B_DN],
        ("p2mm", false) => &[T_DN, B_UP],
        ("p2mg", true) => &[T_UP, B_UP],
        ("p2mg", false) => &[T_DN, B_DN],
        _ => &[],
    }
}

const NY: usize = 3;
const WIDTH: f64 = 1.4;

/// Build an open strip, cylinder or Möbius strip for a frieze group, with an optional strip defect.
/// Refuses (with the [PROVED] reason) a Möbius strip for p1, p2 and p1m1.
pub fn build_frieze_patch(
    symbol: &str,
    cells: usize,
    topology: FriezeTopology,
    defect: FriezeDefect,
) -> Result<FriezePatch, String> {
    let (summary, iucr) = analyze_frieze_group(symbol)?;
    let n_cells = cells.clamp(4, 24);
    let (nx, closed, flip) = match topology {
        FriezeTopology::OpenStrip => (2 * n_cells + 1, false, false),
        FriezeTopology::Cylinder => (2 * n_cells, true, false),
        FriezeTopology::Mobius => (mobius_columns(symbol, n_cells)?, true, true),
    };
    // Quad columns: open nx-1, closed nx (the last one crosses the seam).
    let n_quad_cols = if closed { nx } else { nx - 1 };
    let length = n_quad_cols as f64 / 2.0;
    let spacing = 0.5;
    let vid = |ix: usize, iy: usize| ix * NY + iy;
    let rho = |iy: usize| if flip { NY - 1 - iy } else { iy };
    // Isometric cylinder: circumference = strip length. The Möbius radius only keeps the ribbon
    // off its axis (it is labelled non-isometric anyway).
    let r_cyl = length / (2.0 * PI);
    let r_mob = r_cyl.max(WIDTH);

    let mut pos2d = Vec::with_capacity(nx * NY);
    let mut pos3d = Vec::with_capacity(nx * NY);
    for ix in 0..nx {
        let x2 = ix as f64 * spacing - 0.5 * length;
        let th = 2.0 * PI * (ix as f64) / (nx as f64);
        let right = ix > nx / 2;
        let d_sh = if defect == FriezeDefect::GlideDispiration && right { 0.5 } else { 0.0 };
        for iy in 0..NY {
            let v = (iy as f64) / ((NY - 1) as f64) - 0.5;
            let y_sign = if defect == FriezeDefect::HalfTurn180 && right { -1.0 } else { 1.0 };
            let y2 = v * WIDTH * y_sign;
            pos2d.push([x2, y2]);
            pos3d.push(match topology {
                FriezeTopology::OpenStrip => [x2, y2, d_sh * 0.25],
                FriezeTopology::Cylinder => {
                    let r = r_cyl + d_sh * 0.15;
                    [r * th.cos(), v * WIDTH, r * th.sin()]
                }
                FriezeTopology::Mobius => {
                    let tw = 0.5 * th;
                    let r = r_mob + v * WIDTH * tw.cos();
                    [r * th.cos(), v * WIDTH * tw.sin(), r * th.sin()]
                }
            });
        }
    }

    let mut edges = Vec::new();
    let mut quads = Vec::new();
    let mut wrap_edges = Vec::new();
    let mut wrap_quads = Vec::new();
    let seam_defect = matches!(defect, FriezeDefect::HalfTurn180 | FriezeDefect::GlideDispiration);
    for ix in 0..nx {
        for iy in 0..(NY - 1) {
            edges.push((vid(ix, iy), vid(ix, iy + 1), 0));
        }
    }
    for c in 0..n_quad_cols {
        let wrap = c + 1 == nx;
        // Right-hand vertex of row `iy` in quad column `c`.
        let right = |iy: usize| if wrap { vid(0, rho(iy)) } else { vid(c + 1, iy) };
        let kind = if seam_defect && c == nx / 2 { 1_u8 } else { 0_u8 };
        for iy in 0..NY {
            if wrap {
                wrap_edges.push(edges.len());
            }
            edges.push((vid(c, iy), right(iy), kind));
        }
        for &(l, r) in frieze_decoration(symbol, c) {
            if wrap {
                wrap_edges.push(edges.len());
            }
            edges.push((vid(c, l), right(r), 2));
        }
        for iy in 0..(NY - 1) {
            if wrap {
                wrap_quads.push(quads.len());
            }
            quads.push([vid(c, iy), right(iy), right(iy + 1), vid(c, iy + 1)]);
        }
    }

    let (length_periods, glue_note) = match topology {
        FriezeTopology::OpenStrip => (n_cells as f64, "open strip".to_string()),
        FriezeTopology::Cylinder => (n_cells as f64, format!("cylinder: x ~ x + {n_cells} a")),
        FriezeTopology::Mobius if nx % 2 == 1 => (
            n_cells as f64 + 0.5,
            format!("Möbius: (x, y) ~ (x + {n_cells}.5 a, -y), an odd number ({nx}) of glides"),
        ),
        FriezeTopology::Mobius => (
            n_cells as f64,
            format!("Möbius: (x, y) ~ (x + {n_cells} a, -y), the horizontal mirror after {n_cells} periods"),
        ),
    };
    let picture_note = (topology == FriezeTopology::Mobius).then(|| NOT_ISOMETRIC_NOTE.to_string());
    Ok(FriezePatch {
        symbol: symbol.to_string(),
        iucr_name: iucr.to_string(),
        cells: n_cells,
        topology,
        defect,
        summary,
        columns: nx,
        length_periods,
        pos2d,
        pos3d,
        edges,
        quads,
        wrap_edges,
        wrap_quads,
        picture_note,
        glue_note,
    })
}
