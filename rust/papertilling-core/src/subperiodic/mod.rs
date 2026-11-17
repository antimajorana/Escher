//! Subperiodic 0D rosettes (`C_n, D_n`) and all 7 1D frieze groups (`p1 .. p2mg`):
//! Bieberbach cohomology, abelianization, cylinder / Mobius strip quotients, and defects.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::bieberbach::{analyze_subperiodic_summary, BieberbachSummary, ConvOp, Rat};

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
    pub pos2d: Vec<[f64; 2]>,
    pub pos3d: Vec<[f64; 3]>,
    pub edges: Vec<(usize, usize, u8)>,
    pub quads: Vec<[usize; 4]>,
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

/// Build a 2D strip / 3D cylinder / Mobius ribbon patch for a 1D frieze group with optional strip defect.
pub fn build_frieze_patch(
    symbol: &str,
    cells: usize,
    topology: FriezeTopology,
    defect: FriezeDefect,
) -> Result<FriezePatch, String> {
    let (summary, iucr) = analyze_frieze_group(symbol)?;
    let n_cells = cells.clamp(4, 24);
    let nx = n_cells * 2 + 1;
    let ny = 3usize;
    let mut pos2d = Vec::with_capacity(nx * ny);
    let mut pos3d = Vec::with_capacity(nx * ny);

    let radius = (n_cells as f64) / PI;
    let width = 1.4_f64;
    let length = n_cells as f64;

    for ix in 0..nx {
        let u = (ix as f64) / (nx.saturating_sub(1) as f64); // 0..1
        let x2 = (u - 0.5) * length;
        for iy in 0..ny {
            let v = (iy as f64) / ((ny - 1) as f64) - 0.5; // -0.5..0.5
            let defect_shift = if defect == FriezeDefect::GlideDispiration && ix > nx / 2 {
                0.5
            } else {
                0.0
            };
            let y2 = v * width + if defect == FriezeDefect::HalfTurn180 && ix > nx / 2 { -2.0 * v * width } else { 0.0 };
            pos2d.push([x2, y2]);

            let p3 = match topology {
                FriezeTopology::OpenStrip => [x2, y2, defect_shift * 0.25],
                FriezeTopology::Cylinder => {
                    let th = 2.0 * PI * u;
                    let r_eff = radius + defect_shift * 0.15;
                    [r_eff * th.cos(), v * width, r_eff * th.sin()]
                }
                FriezeTopology::Mobius => {
                    let th = 2.0 * PI * u;
                    let tw = 0.5 * th;
                    let r = radius + v * width * tw.cos();
                    let z = v * width * tw.sin();
                    [r * th.cos(), z, r * th.sin()]
                }
            };
            pos3d.push(p3);
        }
    }

    let vid = |ix: usize, iy: usize| ix * ny + iy;
    let mut edges = Vec::new();
    let mut quads = Vec::new();
    for ix in 0..(nx - 1) {
        let seam_col = matches!(defect, FriezeDefect::HalfTurn180 | FriezeDefect::GlideDispiration) && ix == nx / 2;
        for iy in 0..ny {
            let k = if seam_col { 1_u8 } else { 0_u8 };
            edges.push((vid(ix, iy), vid(ix + 1, iy), k));
        }
        for iy in 0..(ny - 1) {
            edges.push((vid(ix, iy), vid(ix, iy + 1), 0));
            if matches!(symbol, "p11g" | "p2mg") && ix % 2 == 0 {
                edges.push((vid(ix, iy), vid(ix + 1, iy + 1), 2));
            }
            if matches!(symbol, "p1m1" | "p2mm" | "p2") && iy == 0 {
                edges.push((vid(ix, 0), vid(ix, ny - 1), 2));
            }
            quads.push([vid(ix, iy), vid(ix + 1, iy), vid(ix + 1, iy + 1), vid(ix, iy + 1)]);
        }
    }
    for iy in 0..(ny - 1) {
        edges.push((vid(nx - 1, iy), vid(nx - 1, iy + 1), 0));
    }
    Ok(FriezePatch {
        symbol: symbol.to_string(),
        iucr_name: iucr.to_string(),
        cells: n_cells,
        topology,
        defect,
        summary,
        pos2d,
        pos3d,
        edges,
        quads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_friezes_have_exact_five_split_and_two_nonsymmorphic() {
        let mut split = 0usize;
        let mut nonsym = Vec::new();
        for &sym in &FRIEZE_SYMBOLS {
            let (s, _) = analyze_frieze_group(sym).unwrap();
            if s.is_split {
                split += 1;
            } else {
                nonsym.push(sym);
            }
        }
        assert_eq!(split, 5);
        assert_eq!(nonsym, vec!["p11g", "p2mg"]);
        assert_eq!(analyze_frieze_group("p11m").unwrap().0.ab_full_label, "Z_2 x Z");
        assert_eq!(analyze_frieze_group("p11g").unwrap().0.ab_full_label, "Z");
        let mob = build_frieze_patch("p2mg", 10, FriezeTopology::Mobius, FriezeDefect::GlideDispiration).unwrap();
        assert!(!mob.pos3d.is_empty() && !mob.quads.is_empty());
    }
}
