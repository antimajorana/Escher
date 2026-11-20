//! Subperiodic 0D rosettes (`C_n, D_n`) and all 7 1D frieze groups (`p1 .. p2mg`):
//! Bieberbach cohomology, abelianization, cylinder / Möbius strip quotients (glued seams, [PROVED] refusals), and defects.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::bieberbach::{analyze_subperiodic_summary, BieberbachSummary, ConvOp, Rat};

mod frieze_patch;
pub use frieze_patch::{
    build_frieze_patch, frieze_decoration, mobius_columns, FriezeDefect, FriezePatch, FriezeTopology,
    NOT_ISOMETRIC_NOTE,
};

pub const FRIEZE_SYMBOLS: [&str; 7] = ["p1", "p11m", "p1m1", "p11g", "p2", "p2mm", "p2mg"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RosettePatch {
    pub symbol: String,
    pub dihedral: bool,
    pub n: usize,
    pub wedge_delta: i32,
    pub mirror_wall: bool,
    pub point_order: usize,
    pub ab_label: String,
    pub positions: Vec<[f64; 2]>,
    pub edges: Vec<(usize, usize, u8)>,
}

/// Build a 0D rosette `C_n` or `D_n` ring+spoke graph with optional wedge disclination or mirror wall.
pub fn build_rosette(dihedral: bool, n: usize, wedge_delta: i32, mirror_wall: bool) -> RosettePatch {
    let n_orig = n.clamp(2, 16);
    let eff_n = (n_orig as i32 + wedge_delta).clamp(2, 20) as usize;
    let symbol = if dihedral { format!("D{n_orig}") } else { format!("C{n_orig}") };
    let ab_label = if !dihedral {
        format!("Z_{n_orig}")
    } else if n_orig.is_multiple_of(2) {
        "Z_2^2".to_string()
    } else {
        "Z_2".to_string()
    };
    let point_order = if dihedral { 2 * n_orig } else { n_orig };

    let mut positions = Vec::with_capacity(2 * eff_n + 1);
    positions.push([0.0, 0.0]);
    for k in 0..eff_n {
        let th = 2.0 * PI * (k as f64) / (eff_n as f64);
        let asym = if dihedral { 0.0 } else { 0.22 };
        positions.push([1.0 * th.cos(), 1.0 * th.sin()]);
        positions.push([2.0 * (th + asym).cos(), 2.0 * (th + asym).sin()]);
    }
    let mut edges = Vec::new();
    for k in 0..eff_n {
        let kn = (k + 1) % eff_n;
        let i_in = 1 + 2 * k;
        let i_out = 2 + 2 * k;
        let j_in = 1 + 2 * kn;
        let j_out = 2 + 2 * kn;
        let kind_ring = if wedge_delta != 0 && k + 1 == eff_n { 1_u8 } else { 0_u8 };
        edges.push((0, i_in, 0));
        edges.push((i_in, i_out, 0));
        edges.push((i_in, j_in, kind_ring));
        edges.push((i_out, j_out, kind_ring));
        if dihedral && !mirror_wall {
            edges.push((i_out, j_in, 0));
        }
    }
    if mirror_wall && dihedral {
        edges.push((2, 1 + 2 * (eff_n / 2), 2));
    }
    RosettePatch {
        symbol,
        dihedral,
        n: n_orig,
        wedge_delta,
        mirror_wall,
        point_order,
        ab_label,
        positions,
        edges,
    }
}

pub fn analyze_frieze_group(symbol: &str) -> Result<(BieberbachSummary, &'static str), String> {
    let i2 = vec![vec![1_i64, 0], vec![0, 1]];
    let mx = vec![vec![-1_i64, 0], vec![0, 1]];
    let my = vec![vec![1_i64, 0], vec![0, -1]];
    let c2 = vec![vec![-1_i64, 0], vec![0, -1]];
    let half = Rat::new(1, 2);
    let zero2 = vec![Rat::ZERO, Rat::ZERO];
    let half_x = vec![half, Rat::ZERO];
    let tb1 = vec![vec![Rat::ONE]];

    let (num, canon, iucr, gens): (i32, &str, &str, Vec<ConvOp>) = match symbol {
        "p1" => (1, "p1", "p 1 (f1)", vec![]),
        "p11m" => (2, "p11m", "p 1 1 m (f2)", vec![ConvOp { r: my, t: zero2.clone() }]),
        "p11g" => (3, "p11g", "p 1 1 g (f3)", vec![ConvOp { r: my, t: half_x.clone() }]),
        "p1m1" => (4, "p1m1", "p 1 m 1 (f4)", vec![ConvOp { r: mx, t: zero2.clone() }]),
        "p2" => (5, "p2", "p 2 1 1 (f5)", vec![ConvOp { r: c2, t: zero2.clone() }]),
        "p2mm" => (6, "p2mm", "p 2 m m (f6)", vec![ConvOp { r: mx, t: zero2.clone() }, ConvOp { r: my, t: zero2.clone() }]),
        "p2mg" => (7, "p2mg", "p 2 m g (f7)", vec![ConvOp { r: c2, t: zero2.clone() }, ConvOp { r: my, t: half_x.clone() }]),
        _ => return Err(format!("unknown frieze group {symbol:?}")),
    };
    let mut all_gens = gens;
    all_gens.push(ConvOp { r: i2, t: vec![Rat::ONE, Rat::ZERO] });
    let summary = analyze_subperiodic_summary(1, 2, num, canon, &all_gens, &tb1)?;
    Ok((summary, iucr))
}

#[cfg(test)]
mod tests;
