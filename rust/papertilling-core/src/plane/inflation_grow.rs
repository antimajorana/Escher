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

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::{EMPTY_BOX, FaceId, InflationSeed, MAX_STEPS, Off, PatchTopology, PlaneTiling, Tracker, add, grow_box, sub};
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

/// Growth rule of [`inflate`] and [`inflate_with`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum InflationRule {
    /// Automatic: cluster on max-polygon centres when connected in 2D and covering every other tile by an edge;
    /// otherwise plain BDF (stated in [`InflationResult::readout`]).
    #[default]
    Auto,
    /// Hand override: BDF vertex completion closed under FillHoles.
    Bdf,
    /// Hand override: cluster rule with these centre faces.
    Cluster(CentreSet),
}

/// Eligibility of `CentreSet::MaxPolygon` for automatic cluster inflation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoEligibility {
    pub use_cluster: bool,
    pub reason: &'static str,
}

type ResolvedRule = (InflationRule, Option<Vec<bool>>, Option<AutoEligibility>);

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
    /// Effective rule applied (`Cluster(MaxPolygon)` or `Bdf` when `Auto` was requested; otherwise the override).
    #[serde(default)]
    pub rule_used: InflationRule,
    /// `true` when a Dirichlet seed was minimally repaired to remove pinches or disconnects.
    #[serde(default)]
    pub seed_repaired: bool,
    /// Human-readable summary of rule, fallback reason (if any), and Dirichlet seed repair (if any).
    #[serde(default)]
    pub readout: String,
}

/// Inflate `seed` for `n_max` steps using [`InflationRule::Auto`]; see [`inflate_with`].
pub fn inflate(cell: &Cell2D, seed: InflationSeed, n_max: usize) -> Result<InflationResult, String> {
    inflate_with(cell, seed, &InflationRule::Auto, n_max)
}

/// Grow `seed` by `rule` for `n_max` steps. Errs when `S~_0` is not a closed disk (`chi != 1` or a pinch vertex);
/// later shells are reported as they are ([`PlaneTiling::grow`] skips the `S~_0` check).
pub fn inflate_with(cell: &Cell2D, seed: InflationSeed, rule: &InflationRule, n_max: usize) -> Result<InflationResult, String> {
    let tiling = PlaneTiling::new(cell)?;
    let (faces, repair) = tiling.seed_faces_detail(seed)?;
    tiling.run_with(faces, rule, n_max, Some(seed), repair)
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn format_readout(rule_used: &InflationRule, auto_elig: Option<AutoEligibility>, repair: Option<(usize, usize)>) -> String {
    use std::fmt::Write;
    let mut s = match (auto_elig, rule_used) {
        (Some(elig), InflationRule::Cluster(_)) if elig.use_cluster => {
            "cluster (auto: max-polygon centres)".to_string()
        }
        (Some(elig), _) => format!("plain BDF (auto fallback: {})", elig.reason),
        (None, InflationRule::Bdf) => "plain BDF (override)".to_string(),
        (None, InflationRule::Cluster(c)) => format!("cluster (override: {c:?})"),
        (None, InflationRule::Auto) => "auto".to_string(),
    };
    if let Some((added, order)) = repair {
        let _ = write!(s, "; Dirichlet seed repaired (+{added} faces, order-{order} point symmetry kept)");
    }
    s
}

impl PlaneTiling {
    /// Grow the face set `seed` by `rule` for `n_max` steps, without requiring `S~_0` to be a disk.
    pub fn grow(&self, seed: HashSet<FaceId>, rule: &InflationRule, n_max: usize) -> Result<InflationResult, String> {
        self.run_with(seed, rule, n_max, None, None)
    }

    /// Check whether `CentreSet::MaxPolygon` centres form one 2D connected component in `R^2` (via quotient
    /// connectivity + unit-index cycle lattice in `Z^2`) and cover every non-centre orbit by an edge.
    pub fn auto_centre_eligibility(&self) -> AutoEligibility {
        let n = self.orbits.len();
        let Ok(mask) = self.centre_mask(&CentreSet::MaxPolygon) else {
            return AutoEligibility { use_cluster: false, reason: "the centre set is empty" };
        };
        let Some(f0) = (0..n).find(|&f| mask[f]) else {
            return AutoEligibility { use_cluster: false, reason: "the centre set is empty" };
        };
        let mut adj: Vec<BTreeSet<(usize, Off)>> = vec![BTreeSet::new(); n];
        for (u, edges) in adj.iter_mut().enumerate() {
            if !mask[u] {
                continue;
            }
            for c in &self.orbits[u] {
                for (v, doff) in self.vertex_faces((c.site, c.rel)) {
                    if mask[v] && !(v == u && doff == [0, 0]) {
                        edges.insert((v, doff));
                    }
                }
            }
        }
        let mut rel: Vec<Option<Off>> = vec![None; n];
        rel[f0] = Some([0, 0]);
        let mut q = VecDeque::from([f0]);
        while let Some(u) = q.pop_front() {
            let ru = rel[u].unwrap_or([0, 0]);
            for &(v, doff) in &adj[u] {
                if rel[v].is_none() {
                    rel[v] = Some(add(ru, doff));
                    q.push_back(v);
                }
            }
        }
        if (0..n).any(|f| mask[f] && rel[f].is_none()) {
            return AutoEligibility {
                use_cluster: false,
                reason: "max-polygon centres do not form one connected component",
            };
        }
        let mut cycle_set = BTreeSet::new();
        for (u, edges) in adj.iter().enumerate() {
            let Some(ru) = rel[u] else { continue };
            for &(v, doff) in edges {
                let Some(rv) = rel[v] else { continue };
                let d = sub(add(ru, doff), rv);
                if d != [0, 0] {
                    cycle_set.insert(d);
                }
            }
        }
        if cycle_set.is_empty() {
            return AutoEligibility {
                use_cluster: false,
                reason: "max-polygon centres do not form one connected component",
            };
        }
        let cycles: Vec<Off> = cycle_set.into_iter().collect();
        let mut det_gcd = 0i64;
        for (i, &a) in cycles.iter().enumerate() {
            for &b in &cycles[i + 1..] {
                let det = i64::from(a[0]) * i64::from(b[1]) - i64::from(a[1]) * i64::from(b[0]);
                det_gcd = gcd_i64(det_gcd, det);
            }
        }
        if det_gcd == 0 {
            return AutoEligibility {
                use_cluster: false,
                reason: "max-polygon centres only connect along 1D strips",
            };
        }
        if det_gcd != 1 {
            return AutoEligibility {
                use_cluster: false,
                reason: "max-polygon centres split into disjoint 2D sublattices",
            };
        }
        let covers_all = (0..n).filter(|&f| !mask[f]).all(|f| self.across((f, [0, 0])).any(|g| mask[g.0]));
        if !covers_all {
            return AutoEligibility {
                use_cluster: false,
                reason: "some non-centre tile shares no edge with a max-polygon centre",
            };
        }
        AutoEligibility {
            use_cluster: true,
            reason: "max-polygon centres form a connected 2D network covering every other tile by an edge",
        }
    }

    /// `(effective_rule, optional_centre_mask, auto_eligibility)` resolved from a requested [`InflationRule`].
    pub(super) fn resolve_rule(&self, rule: &InflationRule) -> Result<ResolvedRule, String> {
        match rule {
            InflationRule::Auto => {
                let elig = self.auto_centre_eligibility();
                if elig.use_cluster {
                    Ok((InflationRule::Cluster(CentreSet::MaxPolygon), Some(self.centre_mask(&CentreSet::MaxPolygon)?), Some(elig)))
                } else {
                    Ok((InflationRule::Bdf, None, Some(elig)))
                }
            }
            InflationRule::Bdf => Ok((InflationRule::Bdf, None, None)),
            InflationRule::Cluster(c) => Ok((InflationRule::Cluster(c.clone()), Some(self.centre_mask(c)?), None)),
        }
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

    fn run_with(
        &self,
        seed: HashSet<FaceId>,
        rule: &InflationRule,
        n_max: usize,
        strict: Option<InflationSeed>,
        repair: Option<(usize, usize)>,
    ) -> Result<InflationResult, String> {
        if n_max > MAX_STEPS {
            return Err(format!("n_max = {n_max} exceeds the limit {MAX_STEPS}"));
        }
        let (rule_used, mask, auto_elig) = self.resolve_rule(rule)?;
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
        let readout = format_readout(&rule_used, auto_elig, repair);
        Ok(InflationResult {
            shells,
            faces,
            rule_used,
            seed_repaired: repair.is_some(),
            readout,
        })
    }
}
