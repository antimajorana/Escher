//! Dimension-n Bieberbach space-group cohomology (`H^2(P, Z^n)`), abelianizations
//! (`(G^+)^ab`, `G^ab`), Master Affine-Centralizer sector classification (`Q_R // Z_P(R)`),
//! finite-torus conjugacy class counts, and unbased defect fusion channels.

pub mod analyzer;
pub mod rat;
pub mod sectors;
pub mod snf;
pub mod subperiodic_ext;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::plane::groups::{canonical_symbol, group_spec, is_centred, rational_ops};

pub use analyzer::{AbelianGroup, AbelianizationData, CohomologyH2, ConvOp, SpaceGroupAnalyzer};
pub use rat::Rat;
pub use sectors::{BieberbachOrbit, FusionChannel, SectorClass, is_in_image_i_minus_r};
pub use snf::{IntMat, SnfDecomp, hnf_compress_rows, invariant_factors_compressed, invert_unimodular, smith_normal_form, solve_int_linear};
pub use subperiodic_ext::analyze_subperiodic_summary;

/// Compact serializable summary of a crystallographic space group's Bieberbach invariants.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BieberbachSummary {
    pub dim: usize,
    pub number: i32,
    pub symbol: String,
    pub point_order: usize,
    pub h2_torsion: Vec<i128>,
    pub h2_label: String,
    pub is_split: bool,
    pub ab_plus_torsion: Vec<i128>,
    pub ab_plus_free: usize,
    pub ab_plus_label: String,
    pub ab_full_torsion: Vec<i128>,
    pub ab_full_free: usize,
    pub ab_full_label: String,
    pub torus_counts: Vec<(i64, usize)>,
    pub sectors_full: Vec<SectorBrief>,
}

/// Compact sector summary for UI and Python interop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectorBrief {
    pub det: i64,
    pub order: usize,
    pub class_size_p: usize,
    pub r_rank: usize,
    pub t_r_label: String,
    pub num_orbits: usize,
    pub num_pure_classes: usize,
    pub num_elem_glide_orbits: usize,
    pub alpha_r: Option<String>,
}

/// Build a `SpaceGroupAnalyzer` for any of the 17 2D wallpaper groups (`"p1" ..= "p6m"`).
pub fn from_wallpaper_group(group: &str) -> Result<SpaceGroupAnalyzer, String> {
    let canon = canonical_symbol(group)?;
    let spec = group_spec(canon)?;
    let ops = rational_ops(canon)?;
    let centred = is_centred(canon)?;
    let mut gens = Vec::with_capacity(ops.len() + 2);
    for op in ops {
        gens.push(ConvOp {
            r: vec![
                vec![i64::from(op.m[0][0]), i64::from(op.m[0][1])],
                vec![i64::from(op.m[1][0]), i64::from(op.m[1][1])],
            ],
            t: vec![
                Rat::new(i64::from(op.t_num[0]), i64::from(op.t_den)),
                Rat::new(i64::from(op.t_num[1]), i64::from(op.t_den)),
            ],
        });
    }
    let id2 = vec![vec![1_i64, 0], vec![0, 1]];
    gens.push(ConvOp { r: id2.clone(), t: vec![Rat::ONE, Rat::ZERO] });
    gens.push(ConvOp { r: id2, t: vec![Rat::ZERO, Rat::ONE] });
    let tb_rows = if centred {
        // Primitive cell columns = [[1/2, -1/2], [1/2, 1/2]] -> rows of TB are columns^T.
        vec![
            vec![Rat::new(1, 2), Rat::new(1, 2)],
            vec![Rat::new(-1, 2), Rat::new(1, 2)],
        ]
    } else {
        vec![vec![Rat::ONE, Rat::ZERO], vec![Rat::ZERO, Rat::ONE]]
    };
    SpaceGroupAnalyzer::new(2, i32::from(spec.ita_number), canon, &gens, &tb_rows)
}

/// Compute a full `BieberbachSummary` for any `SpaceGroupAnalyzer`.
pub fn summarize_analyzer(sg: &SpaceGroupAnalyzer, torus_ns: &[i64]) -> BieberbachSummary {
    let h2 = sg.compute_cohomology_h2();
    let (ab_plus, ab_full) = sg.abelianizations();
    let non_id: Vec<usize> = (1..sg.p_ops.len()).collect();
    let sectors = sg.classify_sectors(&sg.all_indices(), &non_id);
    let sectors_brief = sectors
        .into_iter()
        .map(|s| SectorBrief {
            det: s.det,
            order: s.order,
            class_size_p: s.class_size_p,
            r_rank: s.r_rank,
            t_r_label: AbelianGroup { torsion: s.t_r, free_rank: 0 }.format(),
            num_orbits: s.num_orbits,
            num_pure_classes: s.num_pure_classes,
            num_elem_glide_orbits: s.num_elem_glide_orbits,
            alpha_r: s.alpha_r.map(|r| r.to_string()),
        })
        .collect();
    let torus_counts = torus_ns
        .iter()
        .map(|&n| (n, sg.count_torus_conjugacy_classes(n).0))
        .collect();
    BieberbachSummary {
        dim: sg.dim,
        number: sg.number,
        symbol: sg.symbol.clone(),
        point_order: sg.p_ops.len(),
        h2_label: h2.format(),
        h2_torsion: h2.torsion,
        is_split: h2.is_split,
        ab_plus_label: ab_plus.format(),
        ab_plus_torsion: ab_plus.torsion,
        ab_plus_free: ab_plus.free_rank,
        ab_full_label: ab_full.format(),
        ab_full_torsion: ab_full.torsion,
        ab_full_free: ab_full.free_rank,
        torus_counts,
        sectors_full: sectors_brief,
    }
}

/// Analyze any of the 17 wallpaper groups into a `BieberbachSummary`.
pub fn analyze_wallpaper_group(group: &str) -> Result<BieberbachSummary, String> {
    let sg = from_wallpaper_group(group)?;
    Ok(summarize_analyzer(&sg, &[2, 3, 4, 6]))
}
