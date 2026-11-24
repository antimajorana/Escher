//! Complete 230 3D space groups (`space_groups_230.json`) and 80 layer groups
//! (`layer_groups_80.json`), analyzed dynamically via the Bieberbach engine.

pub mod holonomy;
pub mod layer_slab;
pub mod motifs;
pub mod nets3d;
pub mod points3d;
pub mod screw;

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::bieberbach::{
    analyze_subperiodic_summary, summarize_analyzer, BieberbachSummary, ConvOp, Rat,
    SpaceGroupAnalyzer,
};

const SPACE_230_JSON: &str = include_str!("../../../../papertilling/data/space_groups_230.json");
const LAYER_80_JSON: &str = include_str!("../../../../papertilling/data/layer_groups_80.json");

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawOp3d {
    pub r: Vec<Vec<i64>>,
    pub t: Vec<[i64; 2]>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpaceGroup3dEntry {
    pub number: u16,
    pub hall_number: u16,
    pub symbol: String,
    pub full_symbol: String,
    pub schoenflies: String,
    pub pointgroup: String,
    pub system: String,
    pub centring: String,
    pub tb_rows: Vec<Vec<[i64; 2]>>,
    pub point_order: usize,
    pub is_split: bool,
    pub h2_label: String,
    pub ab_full_label: String,
    pub ops: Vec<RawOp3d>,
}

#[derive(Deserialize)]
struct SpaceGroups230File {
    groups: Vec<SpaceGroup3dEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawLayerOp {
    pub r: Vec<Vec<i64>>,
    pub t2: Vec<[i64; 2]>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LayerGroupEntry {
    pub number: u8,
    pub hall_source: u16,
    pub symbol: String,
    pub pointgroup: String,
    pub system: String,
    pub centred: bool,
    pub point_order: usize,
    pub ops: Vec<RawLayerOp>,
}

#[derive(Deserialize)]
struct LayerGroups80File {
    groups: Vec<LayerGroupEntry>,
}

static SPACE_GROUPS_230: LazyLock<Vec<SpaceGroup3dEntry>> = LazyLock::new(|| {
    let parsed: SpaceGroups230File =
        serde_json::from_str(SPACE_230_JSON).expect("valid embedded space_groups_230.json");
    parsed.groups
});

static LAYER_GROUPS_80: LazyLock<Vec<LayerGroupEntry>> = LazyLock::new(|| {
    let parsed: LayerGroups80File =
        serde_json::from_str(LAYER_80_JSON).expect("valid embedded layer_groups_80.json");
    parsed.groups
});

pub fn list_space_groups_3d() -> &'static [SpaceGroup3dEntry] {
    &SPACE_GROUPS_230
}

pub fn list_layer_groups() -> &'static [LayerGroupEntry] {
    &LAYER_GROUPS_80
}

pub fn space_group_3d_entry(num: u16) -> Result<&'static SpaceGroup3dEntry, String> {
    SPACE_GROUPS_230
        .iter()
        .find(|e| e.number == num)
        .ok_or_else(|| format!("unknown 3D space group #{num} (expected 1..=230)"))
}

pub fn layer_group_entry(num: u8) -> Result<&'static LayerGroupEntry, String> {
    LAYER_GROUPS_80
        .iter()
        .find(|e| e.number == num)
        .ok_or_else(|| format!("unknown layer group #{num} (expected 1..=80)"))
}

pub fn build_analyzer_3d(num: u16) -> Result<SpaceGroupAnalyzer, String> {
    let e = space_group_3d_entry(num)?;
    let tb_rows: Vec<Vec<Rat>> = e
        .tb_rows
        .iter()
        .map(|row| row.iter().map(|&[n, d]| Rat::new(n, d)).collect())
        .collect();
    let mut gens: Vec<ConvOp> = e
        .ops
        .iter()
        .map(|op| ConvOp {
            r: op.r.clone(),
            t: op.t.iter().map(|&[n, d]| Rat::new(n, d)).collect(),
        })
        .collect();
    let i3 = vec![vec![1_i64, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
    for row in &tb_rows {
        gens.push(ConvOp { r: i3.clone(), t: row.clone() });
    }
    SpaceGroupAnalyzer::new(3, i32::from(e.number), &e.symbol, &gens, &tb_rows)
}

pub fn analyze_space_group_3d(num: u16) -> Result<BieberbachSummary, String> {
    let sg = build_analyzer_3d(num)?;
    Ok(summarize_analyzer(&sg, &[2]))
}

pub(crate) fn layer_translation_basis(centred: bool) -> Vec<Vec<Rat>> {
    if centred {
        vec![
            vec![Rat::new(1, 2), Rat::new(1, 2)],
            vec![Rat::new(-1, 2), Rat::new(1, 2)],
        ]
    } else {
        vec![vec![Rat::ONE, Rat::ZERO], vec![Rat::ZERO, Rat::ONE]]
    }
}

pub fn analyze_layer_group(num: u8) -> Result<BieberbachSummary, String> {
    let e = layer_group_entry(num)?;
    let tb_lat = layer_translation_basis(e.centred);
    let mut gens: Vec<ConvOp> = e
        .ops
        .iter()
        .map(|op| ConvOp {
            r: op.r.clone(),
            t: op.t2.iter().map(|&[n, d]| Rat::new(n, d)).collect(),
        })
        .collect();
    let i3 = vec![vec![1_i64, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
    for row in &tb_lat {
        gens.push(ConvOp { r: i3.clone(), t: row.clone() });
    }
    analyze_subperiodic_summary(2, 3, i32::from(e.number), &e.symbol, &gens, &tb_lat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_230_space_groups_and_80_layer_groups_verify() {
        assert_eq!(list_space_groups_3d().len(), 230);
        assert_eq!(list_layer_groups().len(), 80);

        let mut split = 0usize;
        let mut nonsym = 0usize;
        for g in list_space_groups_3d() {
            let sg = build_analyzer_3d(g.number).unwrap();
            let h2 = sg.compute_cohomology_h2();
            assert_eq!(h2.is_split, g.is_split, "#{}", g.number);
            if h2.is_split {
                split += 1;
            } else {
                nonsym += 1;
            }
        }
        assert_eq!(split, 73);
        assert_eq!(nonsym, 157);

        let s123 = analyze_space_group_3d(123).unwrap();
        let s136 = analyze_space_group_3d(136).unwrap();
        assert!(s123.is_split && s123.ab_full_label == "Z_2^5");
        assert!(!s136.is_split && s136.ab_full_label == "Z_2^2 x Z_4");

        for lg in list_layer_groups() {
            let sum = analyze_layer_group(lg.number).unwrap();
            assert_eq!(sum.point_order, lg.point_order, "layer #{}", lg.number);
        }
    }
}
