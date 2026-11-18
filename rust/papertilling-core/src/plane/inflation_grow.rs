//! Growth rules and results of plane patch inflation. A child module of `plane::inflation` (declared there with
//! `#[path]`).
//!
//! * [`InflationRule::Bdf`]: `S~_0 = FillHoles(seed)`, `S~_{n+1} = FillHoles(S~_n U U(S~_n))` with `U` the vertex
//!   completion (Boyle, Dickens & Flicker 2020, closed under FillHoles).
//! * [`InflationRule::Cluster`] (author decision 1, 2026-11-18): a set of centre faces (a tile class) grows by
//!   vertex completion restricted to centres, `P_{n+1} = P_n U {centre faces sharing >= 1 vertex with P_n}`, and
//!   the shell is the centres decorated by their edge neighbours, `S_n = FillHoles(P_n U {non-centre faces sharing
//!   an EDGE with a face of P_n})`. Seed handling: `P_0` = the centre faces of `FillHoles(seed)`; if there are none,
//!   the centre faces sharing a vertex with `FillHoles(seed)`, and so on: the centres of the first plain vertex
//!   completion `R_k` that has any (`R_0 = FillHoles(seed)`, `R_{k+1} = R_k U U(R_k)`, no FillHoles; needed when
//!   a seed face touches no centre). So `S_0` is built from `P_0`, not from the seed:
//!   seed faces that are neither centres nor edge neighbours of `P_0` are dropped (a kagome hexagon seed gives the
//!   hexagon and its 6 triangles). Centre faces that FillHoles adds to a shell do not join `P`; `P` grows only by
//!   the vertex rule (they are in every later shell anyway, FillHoles being monotone). When every face is a centre
//!   the rule is `S_n = FillHoles(P_n)` with `P_{n+1} = P_n U U(P_n)`: plain BDF whenever no hole ever appears.
//!
//! Each cluster step decorates exactly the new centres `P_{n+1} \ P_n`; `S_{n+1} = FillHoles(S_n U dec(new))` equals
//! `FillHoles(dec(P_{n+1}))` because FillHoles is monotone and idempotent, and the incremental FillHoles start set
//! (faces added this step) is exact because `S_n` is hole-free.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use super::{EMPTY_BOX, FaceId, InflationSeed, MAX_STEPS, PatchTopology, PlaneTiling, Tracker, grow_box};
use crate::types::Cell2D;

/// Which faces are centres for [`InflationRule::Cluster`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentreSet {
    /// Every face of maximal polygon size (a union of tile classes).
    MaxPolygon,
    /// Every face of these translation face orbits.
    Orbits(Vec<usize>),
    /// Every face of these tile classes ([`PlaneTiling::tile_classes`], the `G`-orbits of faces).
    TileClasses(Vec<usize>),
}

/// Growth rule of [`inflate_with`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InflationRule {
    /// BDF vertex completion closed under FillHoles.
    Bdf,
    /// Cluster rule with these centre faces.
    Cluster(CentreSet),
}

/// Per-shell statistics of `S~_n`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellStats {
    pub n: usize,
    pub topology: PatchTopology,
    /// `N_n = |S~_n| - |S~_{n-1}|` (`N_0 = |S~_0|`).
    pub new_tiles: usize,
    /// `N_n / N_{n-1}` for `n >= 1`.
    pub growth: Option<f64>,
    /// Faces added by FillHoles at this step (included in `new_tiles`).
    pub filled: usize,
    /// Boundary vertex types: `t -> number of boundary vertices lying on exactly t patch face corners`.
    pub boundary_vertex_types: BTreeMap<usize, usize>,
    /// `|P_n|` for the cluster rule.
    pub centres: Option<usize>,
}

/// One face of the final patch, drawn in Cartesian coordinates, with the shell that first contained it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InflationFace {
    pub orbit: usize,
    pub offset: [i32; 2],
    pub shell: usize,
    pub polygon: Vec<[f64; 2]>,
}

/// Shells `S~_0 .. S~_{n_max}` and the faces of the final patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InflationResult {
    pub shells: Vec<ShellStats>,
    pub faces: Vec<InflationFace>,
}

/// BDF inflation ([`InflationRule::Bdf`]) of `seed` for `n_max` steps; see [`inflate_with`].
pub fn inflate(cell: &Cell2D, seed: InflationSeed, n_max: usize) -> Result<InflationResult, String> {
    inflate_with(cell, seed, &InflationRule::Bdf, n_max)
}

/// Grow `seed` by `rule` for `n_max` steps. Errs when `S~_0` is not a closed disk (`chi != 1` or a pinch vertex);
/// later shells are reported as they are ([`PlaneTiling::grow`] skips the `S~_0` check).
pub fn inflate_with(cell: &Cell2D, seed: InflationSeed, rule: &InflationRule, n_max: usize) -> Result<InflationResult, String> {
    let tiling = PlaneTiling::new(cell)?;
    let faces = tiling.seed_faces(seed)?;
    tiling.run(faces, rule, n_max, Some(seed))
}

impl PlaneTiling {
    /// Grow the face set `seed` by `rule` for `n_max` steps, without requiring `S~_0` to be a disk.
    pub fn grow(&self, seed: HashSet<FaceId>, rule: &InflationRule, n_max: usize) -> Result<InflationResult, String> {
        self.run(seed, rule, n_max, None)
    }

    /// `centre[f]`: whether face orbit `f` is a centre orbit.
    pub fn centre_mask(&self, centres: &CentreSet) -> Result<Vec<bool>, String> {
        let n = self.orbits.len();
        let mask: Vec<bool> = match centres {
            CentreSet::MaxPolygon => {
                let big = self.orbits.iter().map(Vec::len).max().unwrap_or(0);
                self.orbits.iter().map(|o| o.len() == big).collect()
            }
            CentreSet::Orbits(list) => {
                if let Some(f) = list.iter().find(|&&f| f >= n) {
                    return Err(format!("centre face orbit {f} out of range ({n} orbits)"));
                }
                (0..n).map(|f| list.contains(&f)).collect()
            }
            CentreSet::TileClasses(list) => {
                let classes = self.tile_classes();
                let nc = classes.iter().max().map_or(0, |m| m + 1);
                if let Some(c) = list.iter().find(|&&c| c >= nc) {
                    return Err(format!("tile class {c} out of range ({nc} classes)"));
                }
                classes.iter().map(|c| list.contains(c)).collect()
            }
        };
        if !mask.iter().any(|&b| b) {
            return Err("the centre set is empty".to_string());
        }
        Ok(mask)
    }

    /// Centre faces not in `p` sharing a vertex with a face of `front`.
    fn centre_completion(&self, mask: &[bool], p: &HashSet<FaceId>, front: &[FaceId]) -> BTreeSet<FaceId> {
        self.completion_from(p, front).into_iter().filter(|g| mask[g.0]).collect()
    }

    /// `centres` plus their non-centre edge neighbours, minus the faces already in `patch`.
    fn decorate(&self, mask: &[bool], centres: &BTreeSet<FaceId>, patch: &HashSet<FaceId>) -> Vec<FaceId> {
        let mut out: BTreeSet<FaceId> = centres.iter().copied().collect();
        for &c in centres {
            out.extend(self.across(c).filter(|g| !mask[g.0]));
        }
        out.into_iter().filter(|g| !patch.contains(g)).collect()
    }

    fn run(&self, seed: HashSet<FaceId>, rule: &InflationRule, n_max: usize, strict: Option<InflationSeed>) -> Result<InflationResult, String> {
        if n_max > MAX_STEPS {
            return Err(format!("n_max = {n_max} exceeds the limit {MAX_STEPS}"));
        }
        let mask = match rule {
            InflationRule::Bdf => None,
            InflationRule::Cluster(c) => Some(self.centre_mask(c)?),
        };
        let mut patch = seed;
        // Cluster state: the centre set P and the centres added at the last step.
        let (mut p, mut pfront): (HashSet<FaceId>, Vec<FaceId>) = (HashSet::new(), Vec::new());
        let mut frontier: Vec<FaceId> = if let Some(mask) = &mask {
            self.fill_holes(&mut patch);
            let mut p0: BTreeSet<FaceId> = patch.iter().copied().filter(|g| mask[g.0]).collect();
            let mut ring = patch.clone();
            for _ in 0..MAX_STEPS {
                if !p0.is_empty() || ring.is_empty() {
                    break;
                }
                let more = self.completion(&ring);
                ring.extend(more);
                p0 = ring.iter().copied().filter(|g| mask[g.0]).collect();
            }
            if p0.is_empty() {
                return Err(format!("no centre face within {MAX_STEPS} vertex completions of the seed"));
            }
            patch = HashSet::new();
            let dec = self.decorate(mask, &p0, &patch);
            patch.extend(dec.iter().copied());
            (p, pfront) = (p0.iter().copied().collect(), p0.into_iter().collect());
            dec
        } else {
            patch.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()
        };
        let filled0 = self.fill_holes(&mut patch);
        frontier.extend(filled0.iter().copied());
        let mut tracker = Tracker::default();
        tracker.add(self, &frontier, &patch);
        let top = tracker.stats();
        if let Some(seed) = strict
            && (top.chi != 1 || top.pinch_vertices != 0)
        {
            let (chi, pinch) = (top.chi, top.pinch_vertices);
            return Err(format!("S~_0 of {seed:?} is not a closed disk (chi = {chi}, {pinch} pinch vertices)"));
        }
        let mut bbox = EMPTY_BOX;
        grow_box(&mut bbox, &frontier);
        let mut order: Vec<(FaceId, usize)> = frontier.iter().map(|&f| (f, 0)).collect();
        let centres = |p: &HashSet<FaceId>| mask.as_ref().map(|_| p.len());
        let mut shells = vec![ShellStats {
            n: 0,
            topology: top,
            new_tiles: patch.len(),
            growth: None,
            filled: filled0.len(),
            boundary_vertex_types: tracker.vertex_types(),
            centres: centres(&p),
        }];
        for n in 1..=n_max {
            let grown: Vec<FaceId> = match &mask {
                None => self.completion_from(&patch, &frontier).into_iter().collect(),
                Some(mask) => {
                    let dp = self.centre_completion(mask, &p, &pfront);
                    p.extend(dp.iter().copied());
                    let grown = self.decorate(mask, &dp, &patch);
                    pfront = dp.into_iter().collect();
                    grown
                }
            };
            patch.extend(grown.iter().copied());
            grow_box(&mut bbox, &grown);
            let filled = self.fill_holes_from(&mut patch, &grown, bbox);
            frontier = grown;
            frontier.extend(filled.iter().copied());
            tracker.add(self, &frontier, &patch);
            order.extend(frontier.iter().map(|&f| (f, n)));
            let (new_tiles, prev) = (frontier.len(), shells[n - 1].new_tiles);
            shells.push(ShellStats {
                n,
                topology: tracker.stats(),
                new_tiles,
                growth: (prev > 0).then(|| new_tiles as f64 / prev as f64),
                filled: filled.len(),
                boundary_vertex_types: tracker.vertex_types(),
                centres: centres(&p),
            });
        }
        let faces = order
            .into_iter()
            .map(|(face, shell)| InflationFace { orbit: face.0, offset: face.1, shell, polygon: self.polygon(face) })
            .collect();
        Ok(InflationResult { shells, faces })
    }
}
