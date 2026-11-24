use papertilling_core::defect::combine_types::CoreFaceSummary;
#[cfg(test)]
use papertilling_core::defect::combine_types::NotebookConfigJson;
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::equivalence::CentreSnapper;
use papertilling_core::defect::surgery::AppliedCut;
#[cfg(test)]
use papertilling_core::defect::surgery::DefectSpec;
#[cfg(test)]
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{BuiltCrystal, Cell2D};

#[cfg(test)]
use crate::backend::CatalogueItemUi;

mod builder;
pub mod cuts;
mod snap;

pub(crate) use builder::align_crystal_orientation;
pub use builder::{build_combined_crystal, item_from_notebook_json};
pub use snap::{
    burgers_from_mn, combine_lattice_info, map_drawn_to_ref, map_ref_to_drawn, mn_from_burgers,
    preset_neighbour_pair, preset_shortest_dislocation, snap_disclination_centre,
    spec_core_ref_pos, spec_short_label, spec_world_pos,
};
#[cfg(test)]
use snap::{centre_class_at, iso_rotation_about, same_mod_cell};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineDrawMode {
    FastRelaxed,
    Raw,
    #[allow(dead_code)] // C1 hidden: the AO elastic view left the UI
    ElasticAoMetric,
}

#[derive(Clone, Debug)]
pub struct CombineCentreOption {
    pub id: String,
    pub label: String,
    pub order: usize,
    pub location: String,
    pub point: [f64; 2],
    pub angles: Vec<f64>,
    pub unit_seeds: Vec<[f64; 2]>,
}

#[derive(Clone, Debug)]
pub struct CombineDislocOption {
    pub id: String,
    pub burgers_mn: (i32, i32),
    pub b: [f64; 2],
    pub at: [f64; 2],
    pub members: Vec<(i32, i32)>,
}

#[derive(Clone, Debug)]
pub struct CombineLatticeInfo {
    pub cell: Cell2D,
    pub lat_info: LatticeCoreInfo,
    pub centres: Vec<CombineCentreOption>,
    pub allowed_angles: Vec<i32>,
    pub dislocations: Vec<CombineDislocOption>,
    pub r_auto_base: f64,
    /// The catalogue writes Burgers coefficients in its reduced primitive basis, not the drawn cell.
    pub burgers_basis: [[f64; 2]; 2],
    pub snapper: CentreSnapper,
}

#[derive(Clone, Debug)]
pub struct CombineBuildOptions {
    pub draw_mode: CombineDrawMode,
    pub full_planar_repair: bool,
    pub auto_radius: bool,
    pub manual_radius: f64,
    pub core_radius: usize,
}

#[derive(Clone, Debug)]
pub struct CombineBuildOutput {
    pub lattice_key: (String, Cell2D),
    pub crystal: BuiltCrystal,
    pub elapsed_ms: f64,
    pub applied_radius: f64,
    pub charge_label: String,
    pub charge_detail: String,
    pub core_summary: String,
    pub local_cores: Vec<CoreFaceSummary>,
    pub patch_centre: [f64; 2],
    pub applied_cuts: Vec<AppliedCut>,
    pub coalesced_count: usize,
    pub note: String,
    /// Cheap in-drag picture: no planarity check and no core readouts (they stay at zero / empty).
    pub preview: bool,
    /// The surgery refused the defects (the reason); `crystal` is then a defect-free stand-in disk.
    pub refused: Option<String>,
}

#[cfg(test)]
mod json_tests;
#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests {
    use papertilling_core::defect::surgery::DefectSpec;

    use super::*;
    use crate::backend::{AppBackend, CatalogueRuleUi};

    #[test]
    fn snap_centres_burgers_and_refusal_fallback() {
        let backend = AppBackend::new();
        let kagome = backend.find_named("kagome").unwrap();
        let info = combine_lattice_info(kagome);
        assert_eq!(info.centres.len(), 3); // 1a (size 1), 2b (size 2), 3c (size 3)
        assert_eq!(info.centres[0].unit_seeds.len(), 1);
        assert_eq!(info.centres[1].unit_seeds.len(), 2);
        assert_eq!(info.centres[2].unit_seeds.len(), 3);

        let b21 = burgers_from_mn(&info, 2, -1);
        assert_eq!(mn_from_burgers(&info, b21), Some((2, -1)));

        // p3 split 3-fold classes C2 (1b) and C3 (1c) stay distinct under class filter
        let p3_first = &backend.catalogue_for_group("p3", CatalogueRuleUi::Points)[0];
        let p3_info = combine_lattice_info(p3_first);
        assert_eq!(p3_info.centres.len(), 3);
        let (p_c2, _) = snap_disclination_centre(p3_first, &p3_info, 120.0, Some("C2"), p3_info.centres[2].point);
        let (p_c3, _) = snap_disclination_centre(p3_first, &p3_info, 120.0, Some("C3"), p3_info.centres[2].point);
        assert!((p_c2[0] - p_c3[0]).hypot(p_c2[1] - p_c3[1]) > 0.1);

        // Invalid rotation angle at generic point falls back cleanly without panicking
        let opts = CombineBuildOptions {
            draw_mode: CombineDrawMode::FastRelaxed,
            full_planar_repair: false,
            auto_radius: true,
            manual_radius: info.r_auto_base,
            core_radius: 2,
        };
        let bad = [DefectSpec::disclination(73.0, [0.123, 0.456])];
        let out_bad = build_combined_crystal(kagome, &info, &bad, &opts, false);
        assert!(!out_bad.crystal.pos.is_empty() && !out_bad.note.is_empty());
    }
}
