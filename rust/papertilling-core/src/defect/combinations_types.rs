//! Data types for the general defect Combinations ("Equivalence") table engine.

use serde::{Deserialize, Serialize};

use crate::defect::combine_types::{ClassifiedCharge, CoreGalleryEntry, DislocationScanInfo};
use crate::defect::surgery::DefectSpec;

/// Which family of multi-defect combinations to enumerate and classify.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CombinationMode {
    /// All symmetry-inequivalent pairs of catalogue disclinations within `max_sep`.
    DisclinationPairs,
    /// All symmetry-inequivalent pairs of catalogue dislocations within `max_sep`.
    DislocationPairs,
    /// All symmetry-inequivalent (disclination, dislocation) pairs within `max_sep`.
    MixedPairs,
    /// All symmetry-inequivalent placements of three named catalogue defects within `max_sep`.
    Triple { ids: [String; 3] },
    /// Caller-supplied list of explicit defect combinations.
    Custom { candidates: Vec<CombinationCandidate> },
}

/// Tuning parameters for building a [`CombinationsTable`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CombinationsOptions {
    pub max_sep: f64,
    pub radius: usize,
    pub grid: usize,
    pub draw_thumbnails: bool,
    pub include_single_gallery: bool,
    pub bmax: f64,
}

impl Default for CombinationsOptions {
    fn default() -> Self {
        Self {
            max_sep: 1.5,
            radius: 2,
            grid: 1,
            draw_thumbnails: true,
            include_single_gallery: false,
            bmax: 2.0,
        }
    }
}

/// One candidate multi-defect placement prior to surgical building.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CombinationCandidate {
    pub label: String,
    pub defect_ids: Vec<String>,
    pub separation: f64,
    pub displacements: Vec<[f64; 2]>,
    pub specs: Vec<DefectSpec>,
}

/// Whether a built combination's core matches a single defect of the same total charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SingleMatchStatus {
    MatchesSingle,
    CoreDiffers,
    NoSingleDefect,
}

/// One successfully built combination member inside a [`CoreGroup`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CombinationEntry {
    pub label: String,
    pub defect_ids: Vec<String>,
    pub separation: f64,
    pub displacements: Vec<[f64; 2]>,
    /// Defect specifications in the working surgical build order.
    pub specs: Vec<DefectSpec>,
    /// Permutation of the candidate's original spec indices that succeeded.
    pub build_order: Vec<usize>,
    /// Exact site-count excess `V_crystal - V_perfect` for vacuum (`total charge 0`) crystals.
    pub delta_v: Option<i32>,
}

/// Distinct degree-labelled core graph class within a [`ChargeGroup`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreGroup {
    pub core_nodes: usize,
    pub core_edges: usize,
    pub status: SingleMatchStatus,
    pub match_tier: String,
    pub single_matches: Vec<String>,
    pub volterra_matches: Vec<String>,
    pub completion_matches: Vec<String>,
    pub delta_v: Option<i32>,
    pub thumbnail: Option<CoreGalleryEntry>,
    pub members: Vec<CombinationEntry>,
}

/// Group of built combinations sharing the same total topological charge in `G~+`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeGroup {
    pub charge: ClassifiedCharge,
    pub charge_label: String,
    pub has_single_defect: bool,
    pub single_scan: Option<DislocationScanInfo>,
    pub single_gallery: Vec<CoreGalleryEntry>,
    pub core_groups: Vec<CoreGroup>,
}

/// Candidate combination that could not be built on an open disk patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefusedCombination {
    pub label: String,
    pub defect_ids: Vec<String>,
    pub separation: f64,
    pub displacements: Vec<[f64; 2]>,
    pub specs: Vec<DefectSpec>,
    pub reason: String,
}

/// Complete Combinations ("Equivalence") table grouped by total charge and distinct core graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CombinationsTable {
    pub group: String,
    pub max_sep: f64,
    pub radius: usize,
    pub grid: usize,
    pub total_candidates: usize,
    pub built_count: usize,
    pub refused_count: usize,
    pub cancelled: bool,
    pub elapsed_ms: f64,
    pub charge_groups: Vec<ChargeGroup>,
    pub refused: Vec<RefusedCombination>,
}
