//! The 230-group points catalogue (C1 WP-U4, plan §5 T4 with Q1, Q2, Q10): the loader of the offline
//! sweep `papertilling/data/points3d.json` (built by `tools/build_points3d.py` from `papertilling/points3d.py`)
//! and its coverage report, and the growth of a catalogue entry by inflation on its Dirichlet complex.
//!
//! - **Entries.** A union of Wyckoff orbits whose exact group (spglib, symprec 1e-5) is the space group, bonded
//!   by the shells-until-connected rule (connected and 3-periodic). Per group the first `n` = 7 entries are
//!   shown and the rest are listed as extra. [COMPUTATIONAL EVIDENCE] The sweep samples 48 parameter sets with
//!   seed 0; two entries are merged when they share the space group, the sites per primitive cell and the
//!   per-orbit coordination sequences to depth 12, which is not an exact identity of nets (the coverage note
//!   says so).
//! - **Data layout.** `A`: lattice rows (Cartesian, primitive cell), `F`: fractional sites (unreduced; a bond
//!   `(i, j, t)` joins site i of cell n to site j of cell n + t, with t relative to `F` as stored).
//! - **Growth.** Never wrapped on a torus: the Dirichlet (Voronoi) cells of the sites (plan Q2) are grown by
//!   the touching rule ([`crate::growth::periodic_net`]) and the entry's bonds are drawn between grown cells.

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use super::nets3d::Net3dPatch;
use crate::growth::periodic_net::{NetCell, PeriodicNet};
use crate::growth::{grow, GrowthCaps, LayerStats, SeedKind, Stopped};

const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points3d.json");
const COVERAGE_JSON: &str = include_str!("../../../../papertilling/data/points3d_coverage.json");

/// One catalogue lattice (fields of `points3d.json` that the app uses; the key and sequences are skipped).
#[derive(Clone, Debug, Deserialize)]
pub struct PointsEntry {
    /// Wyckoff letters of the orbits (with multiplicity), e.g. `["8a"]`.
    pub wyckoff: Vec<String>,
    /// Sites per primitive cell.
    pub sites: usize,
    /// Distinct site degrees (bonds per site).
    pub degrees: Vec<usize>,
    /// Share of the 48 samples in this phase.
    pub share: f64,
    #[serde(rename = "A")]
    pub lattice: [[f64; 3]; 3],
    #[serde(rename = "F")]
    pub frac: Vec<[f64; 3]>,
    /// Orbit index of each site.
    pub labels: Vec<usize>,
    /// `(i, j, t)`: site i of cell n to site j of cell n + t, each bond once.
    pub bonds: Vec<(usize, usize, [i64; 3])>,
    /// Other Wyckoff choices that gave the same lattice.
    #[serde(default)]
    pub also: Vec<Vec<String>>,
}

/// The entries of one space group.
#[derive(Clone, Debug, Deserialize)]
pub struct PointsGroup {
    pub group: u16,
    pub symbol: String,
    /// How many entries are shown (7); the rest are extra.
    pub n: usize,
    pub entries: Vec<PointsEntry>,
}

impl PointsGroup {
    /// The shown entries (the first `n`).
    pub fn shown(&self) -> &[PointsEntry] {
        &self.entries[..self.n.min(self.entries.len())]
    }
}

/// Coverage of one group: entries found, the deepest tier `[orbits, sites per cell]` searched completely.
#[derive(Clone, Debug, Deserialize)]
pub struct PointsCoverage {
    pub group: u16,
    pub entries: usize,
    pub deepest_complete_tier: Vec<usize>,
    pub timed_out: bool,
    pub collisions: usize,
}

#[derive(Deserialize)]
struct PointsFile {
    note: String,
    partial: bool,
    groups: HashMap<String, PointsGroup>,
}

#[derive(Deserialize)]
struct CoverageFile {
    rule: String,
    status: String,
    groups: Vec<PointsCoverage>,
}

static POINTS: LazyLock<PointsFile> =
    LazyLock::new(|| serde_json::from_str(POINTS_JSON).expect("valid embedded points3d.json"));
static COVERAGE: LazyLock<CoverageFile> =
    LazyLock::new(|| serde_json::from_str(COVERAGE_JSON).expect("valid embedded points3d_coverage.json"));

/// The catalogue entries of space group `num` (1..=230).
pub fn points_group(num: u16) -> Result<&'static PointsGroup, String> {
    POINTS
        .groups
        .get(&num.to_string())
        .ok_or_else(|| format!("no points-catalogue entry for space group #{num} (expected 1..=230)"))
}

/// The coverage row of space group `num`.
pub fn points_coverage(num: u16) -> Option<&'static PointsCoverage> {
    COVERAGE.groups.iter().find(|c| c.group == num)
}

/// The catalogue caveat shown with every entry: evidence status, rule and the duplicate-key note.
pub fn points_note() -> String {
    let partial = if POINTS.partial { " Partial sweep: some groups are missing." } else { "" };
    format!("[{}] {}. {}{partial}", COVERAGE.status, COVERAGE.rule, POINTS.note)
}

/// "k of 7 found" readout of a group (with the stopping tier when below `n`).
pub fn coverage_line(num: u16) -> String {
    let Ok(g) = points_group(num) else { return format!("#{num}: not in the catalogue") };
    let shown = g.shown().len();
    let extra = g.entries.len() - shown;
    let mut s = format!("{shown} of {} found", g.n);
    if let Some(c) = points_coverage(num) {
        if shown < g.n {
            s += &format!(" (search stopped at tier {:?}{})", c.deepest_complete_tier, if c.timed_out { ", timed out" } else { "" });
        }
        if c.collisions > 0 {
            s += &format!(", {} key collisions", c.collisions);
        }
    }
    if extra > 0 {
        s += &format!(", {extra} more listed as extra");
    }
    s
}

/// A grown catalogue entry: the drawing, the frame of its lattice and the per-layer readout.
#[derive(Clone, Debug)]
pub struct PointsGrowth {
    pub patch: Net3dPatch,
    /// Lattice rows of the entry and the Cartesian point of fractional (0, 0, 0) in the drawn patch.
    pub lattice: [[f64; 3]; 3],
    pub origin: [f64; 3],
    pub stats: Vec<LayerStats>,
    pub stopped: Option<Stopped>,
    /// Grown cells (one site each).
    pub cells: usize,
}

/// Grow entry `idx` (0-based, among all entries) of space group `num` by `layers` touching-rule layers from
/// `seed`, with at most `max_cells` cells. Refused (with the reason) when the Dirichlet complex cannot be
/// certified or the seed does not exist.
pub fn grow_points3d(num: u16, idx: usize, seed: SeedKind, layers: u32, max_cells: usize) -> Result<PointsGrowth, String> {
    let g = points_group(num)?;
    let e = g.entries.get(idx).ok_or_else(|| format!("#{num} {} has {} entries, not {}", g.symbol, g.entries.len(), idx + 1))?;
    let l = e.lattice;
    let cart = |f: [f64; 3], n: [i64; 3]| -> [f64; 3] {
        let a = [f[0] + n[0] as f64, f[1] + n[1] as f64, f[2] + n[2] as f64];
        [0, 1, 2].map(|k| a[0] * l[0][k] + a[1] * l[1][k] + a[2] * l[2][k])
    };
    let sites: Vec<[f64; 3]> = e.frac.iter().map(|&f| cart(f, [0; 3])).collect();
    let mut net = PeriodicNet::new(&l, &sites)?;
    let s0 = net.seed_cells(seed)?;
    let gr = grow(&mut net, s0, GrowthCaps { max_layers: layers, max_cells });
    let cells: Vec<NetCell> = gr.layers.iter().flatten().copied().collect();
    let index: HashMap<NetCell, usize> = cells.iter().enumerate().map(|(k, c)| (*c, k)).collect();
    let mut positions: Vec<[f64; 3]> = cells.iter().map(|c| cart(e.frac[c.0 as usize], c.1)).collect();
    let m = positions.len().max(1) as f64;
    let centre = positions.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / m, a[1] + p[1] / m, a[2] + p[2] / m]);
    positions.iter_mut().for_each(|p| *p = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]]);
    let mut bonds = Vec::new();
    for (ci, c) in cells.iter().enumerate() {
        for &(_, j, t) in e.bonds.iter().filter(|b| b.0 == c.0 as usize) {
            let partner = (j as u32, [c.1[0] + t[0], c.1[1] + t[1], c.1[2] + t[2]]);
            if let Some(&cj) = index.get(&partner) {
                bonds.push((ci, cj, 0_u8));
            }
        }
    }
    let site_kinds: Vec<u8> = cells.iter().map(|c| e.labels.get(c.0 as usize).copied().unwrap_or(0).min(255) as u8).collect();
    let layer = gr.stats.len().saturating_sub(1);
    let name = format!("#{num} {} lattice {} ({})", g.symbol, idx + 1, e.wyckoff.join(" + "));
    let info = format!(
        "{name} | {} sites/cell, degree {:?} | layer {layer} from a {seed:?} seed: {} cells, {} bonds",
        e.sites,
        e.degrees,
        cells.len(),
        bonds.len()
    );
    let patch = Net3dPatch {
        name,
        space_group_num: num,
        defect_label: "Grown by inflation (Dirichlet cells), not wrapped on a torus [COMPUTATIONAL EVIDENCE]".into(),
        positions,
        bonds,
        site_kinds,
        defect_axis: None,
        info,
    };
    Ok(PointsGrowth { patch, lattice: l, origin: [-centre[0], -centre[1], -centre[2]], stats: gr.stats, stopped: gr.stopped, cells: cells.len() })
}

#[cfg(test)]
#[path = "points3d_tests.rs"]
mod tests;
