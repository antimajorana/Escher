//! Shared Stage 5 data structures for defect configuration equivalence,
//! Volterra/completion core galleries, and Notebook 3 `.json` import/export.

use serde::{Deserialize, Serialize};

use crate::defect::surgery::DefectSpec;
use crate::types::{BuiltCrystal, Frame2D, Isometry2, Recipe};

fn default_format() -> String {
    "papertilling.config.v1".to_string()
}

fn default_picture() -> String {
    "relaxed".to_string()
}

fn default_grid() -> usize {
    2
}

fn default_radius() -> usize {
    2
}

/// Topological charge classified in the lifted proper wallpaper group `G~+`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifiedCharge {
    pub kind: String,
    pub angle: f64,
    pub holonomy: Isometry2,
    pub b: Option<[f64; 2]>,
    pub lattice_vector: Option<bool>,
    pub b_rep: Option<[f64; 2]>,
    pub length: Option<f64>,
    pub centre: Option<[f64; 2]>,
    pub centre_class: Option<String>,
    pub label: String,
    pub single: Option<DefectSpec>,
    pub loop_sum: [f64; 2],
}

/// Local non-trivial holonomy face/site core inside a built crystal (`Configuration.cores()`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreFaceSummary {
    pub kind: String,
    pub sites: Vec<usize>,
    pub ring: Option<usize>,
    pub coordination: Vec<usize>,
    pub b: Option<[f64; 2]>,
    pub angle: Option<f64>,
    pub site: Option<usize>,
    pub label: String,
}

/// Summary metadata of a Volterra + completion dislocation core scan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DislocationScanInfo {
    pub grid: usize,
    pub r_patch: f64,
    pub builds: usize,
    pub anomalies: usize,
    pub capped: usize,
    pub max_candidates: usize,
    pub lab_cores: usize,
    pub other_class: usize,
    pub skipped_rim: usize,
    pub distinct: usize,
}

/// Candidate entry returned by `dislocation_core_list` and `equivalent_single`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreCandidateInfo {
    pub name: String,
    pub volterra: Option<bool>,
    pub rings: Vec<usize>,
    pub under: Vec<usize>,
    pub added: usize,
    pub lattice_bonds: usize,
    pub reconstruction_bonds: usize,
    pub bipartite: Option<bool>,
    pub core_point: Option<[f64; 2]>,
    pub cut: Option<[f64; 2]>,
}

/// Drawable cropped crystal panel for a configuration's core or a single-dislocation Volterra/completion core.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreGalleryEntry {
    pub name: String,
    /// `"Volterra"`, `"completion"`, or `"configuration"`.
    pub kind: String,
    pub crystal: BuiltCrystal,
    pub core_sites: Vec<usize>,
    pub chords: Vec<(usize, usize)>,
    pub under_sites: Vec<usize>,
    pub rings: Vec<usize>,
    pub faces: Vec<Vec<usize>>,
    pub under: Vec<usize>,
    pub added: usize,
    pub description: String,
    pub also: Vec<String>,
    pub faces_ok: bool,
    pub iso_ok: Option<bool>,
    pub perfect_rings: Vec<usize>,
}

/// Full result of `equivalent_single` comparing a multi-defect configuration against single defects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EquivalentSingleReport {
    pub charge: ClassifiedCharge,
    pub kind: String,
    pub single: Option<DefectSpec>,
    pub candidates: Vec<CoreCandidateInfo>,
    pub matches: Vec<String>,
    pub equivalent: bool,
    pub verdict: String,
    pub scan: Option<DislocationScanInfo>,
    pub core_nodes: usize,
    pub core_edges: usize,
    pub config_core: Option<CoreGalleryEntry>,
    pub gallery: Vec<CoreGalleryEntry>,
}

/// Pairwise equivalence comparison of two configurations on the same lattice (`config.compare`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigComparison {
    pub same_charge: bool,
    pub same_charge_up_to_mirror: bool,
    pub same_core: bool,
    pub charge1: String,
    pub charge2: String,
    pub core1: (usize, usize),
    pub core2: (usize, usize),
}

/// Build/comparison options saved in a Notebook 3 `.json` configuration file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigOptionsJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r_patch: Option<f64>,
    #[serde(default = "default_picture")]
    pub picture: String,
    #[serde(default = "default_grid")]
    pub grid: usize,
    #[serde(default = "default_radius")]
    pub radius: usize,
}

impl Default for ConfigOptionsJson {
    fn default() -> Self {
        Self {
            r_patch: None,
            picture: default_picture(),
            grid: default_grid(),
            radius: default_radius(),
        }
    }
}

/// Optional human-readable readout saved alongside a Notebook 3 `.json` configuration file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigSummaryJson {
    pub charge: String,
    pub kind: String,
    pub equivalent: bool,
    pub matches: Vec<String>,
    pub verdict: String,
}

/// Portable Notebook 3 `.json` defect configuration (`recipe` + `frame` + `defects` + `options` + `summary`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NotebookConfigJson {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub lattice_label: String,
    pub group: String,
    pub recipe: Recipe,
    #[serde(default)]
    pub frame: Frame2D,
    pub defects: Vec<DefectSpec>,
    #[serde(default)]
    pub options: ConfigOptionsJson,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ConfigSummaryJson>,
}

impl NotebookConfigJson {
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn from_json_str(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}
