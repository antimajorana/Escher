//! Public Stage 5 API entry points for live defect combination, charge comparison,
//! Volterra/completion core galleries, and single-defect equivalence.

use crate::defect::combine_types::{
    ClassifiedCharge, ConfigComparison, CoreFaceSummary, CoreGalleryEntry, DislocationScanInfo,
    EquivalentSingleReport,
};
use crate::defect::surgery::DefectSpec;
use crate::types::{BuiltCrystal, Cell2D, Frame2D};

/// Snap `near` to the nearest allowed Wyckoff rotation centre of `G` permitting `angle_deg`,
/// returning `(snapped_cartesian_point, centre_label)`.
pub fn nearest_allowed_centre(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    angle_deg: f64,
    near: [f64; 2],
) -> Result<([f64; 2], String), String> {
    crate::defect::equivalence::nearest_allowed_centre(group, cell, frame, angle_deg, near)
}

/// Classify the outer-boundary holonomy and unreduced Frank angle of `crystal` in `G~+`.
pub fn classify_crystal_charge(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    crystal: &BuiltCrystal,
) -> Result<ClassifiedCharge, String> {
    crate::defect::equivalence::classify_crystal_charge(group, cell, frame, crystal)
}

/// Check whether `q1` and `q2` lie in the same conjugacy class of `G~+` (or `G~` when `mirror = true`).
pub fn same_charge(
    group: &str,
    cell: &Cell2D,
    q1: &ClassifiedCharge,
    q2: &ClassifiedCharge,
    mirror: bool,
) -> bool {
    crate::defect::equivalence::same_charge(group, cell, q1, q2, mirror)
}

/// Extract per-core local face/site summaries (`Configuration.cores()`).
pub fn inspect_crystal_cores(crystal: &BuiltCrystal) -> Vec<CoreFaceSummary> {
    crate::defect::equivalence::inspect_crystal_cores(crystal)
}

/// Scan Volterra and non-crossing completion cores for Burgers vector `b`.
pub fn core_gallery(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    b: [f64; 2],
    radius: usize,
    grid: usize,
    draw: bool,
) -> Result<(Vec<CoreGalleryEntry>, DislocationScanInfo), String> {
    crate::defect::core_gallery::core_gallery(group, cell, frame, b, radius, grid, draw)
}

/// Compare a built configuration against the single defect(s) carrying the same total charge.
#[allow(clippy::too_many_arguments)]
pub fn equivalent_single(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    crystal: &BuiltCrystal,
    radius: usize,
    grid: usize,
    draw_gallery: bool,
) -> Result<EquivalentSingleReport, String> {
    crate::defect::core_gallery::equivalent_single(
        group,
        cell,
        frame,
        specs,
        crystal,
        radius,
        grid,
        draw_gallery,
    )
}

/// Compare two built configurations on the same lattice (`same_charge`, `same_charge_up_to_mirror`, `same_core`).
pub fn compare_configurations(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    c1: &BuiltCrystal,
    c2: &BuiltCrystal,
    radius: usize,
) -> Result<ConfigComparison, String> {
    crate::defect::equivalence::compare_configurations(group, cell, frame, c1, c2, radius)
}

/// Group built configurations on the same lattice into equivalence classes (`0, 1, ...`).
pub fn classify_configurations(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    crystals: &[BuiltCrystal],
    radius: usize,
) -> Result<Vec<usize>, String> {
    crate::defect::equivalence::classify_configurations(group, cell, frame, crystals, radius)
}
