//! Torus mode: the same local growth steps on the `L1 x L2` torus quotient of the tiling (face and vertex offsets
//! taken mod `(L1, L2)`), reporting the wrap step = the first `n` whose shell is not a closed disk (`chi != 1` or a
//! pinch vertex). There is no outside on a torus, so no FillHoles. A child module of `plane::inflation`.
//!
//! An edge is keyed by its lattice dart `(start vertex mod L, end site, bond offset)` up to reversal, so a small
//! torus on which two bonds join the same pair of vertex classes still counts them apart. The quotient is a
//! cellular map only when every face's corners stay distinct mod `L`; use `L` at least a few face diameters.

use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use super::{FaceId, InflationRule, Off, PlaneTiling, VertexId, add, sub};

/// Topology of one torus shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TorusShell {
    pub n: usize,
    pub faces: usize,
    pub vertices: usize,
    pub edges: usize,
    pub chi: i64,
    pub pinch_vertices: usize,
}

/// Shells on the torus and the first step whose shell is not a disk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TorusResult {
    pub shells: Vec<TorusShell>,
    pub wrap_step: Option<usize>,
}

impl PlaneTiling {
    /// Grow `seed` by `rule` on the `l[0] x l[1]` torus for `n_max` steps (no FillHoles).
    pub fn inflate_torus(&self, seed: &HashSet<FaceId>, rule: &InflationRule, l: Off, n_max: usize) -> Result<TorusResult, String> {
        if l[0] < 1 || l[1] < 1 {
            return Err(format!("torus size {l:?} must be positive"));
        }
        let w = |o: Off| [o[0].rem_euclid(l[0]), o[1].rem_euclid(l[1])];
        let wf = |f: FaceId| (f.0, w(f.1));
        let near = |s: &HashSet<FaceId>| -> BTreeSet<FaceId> {
            let fs = s.iter().flat_map(|&f| self.face_vertices(f));
            fs.flat_map(|v| self.vertex_faces(v)).map(wf).filter(|g| !s.contains(g)).collect()
        };
        let (_, mask, _) = self.resolve_rule(rule)?;
        let mut s: HashSet<FaceId> = seed.iter().map(|&f| wf(f)).collect();
        let mut p: HashSet<FaceId> = HashSet::new();
        if let Some(m) = &mask {
            p = s.iter().copied().filter(|g| m[g.0]).collect();
            let mut ring = s.clone();
            while p.is_empty() && ring.len() < self.orbits.len() * (l[0] * l[1]) as usize {
                ring.extend(near(&ring));
                p = ring.iter().copied().filter(|g| m[g.0]).collect();
            }
        }
        let decorate = |p: &HashSet<FaceId>, m: &[bool]| -> HashSet<FaceId> {
            let edge_nbrs = p.iter().flat_map(|&c| self.across(c)).map(wf).filter(|g| !m[g.0]);
            p.iter().copied().chain(edge_nbrs.collect::<Vec<_>>()).collect()
        };
        let mut shells = Vec::new();
        for n in 0..=n_max {
            if let Some(m) = &mask {
                if n > 0 {
                    let more: Vec<FaceId> = near(&p).into_iter().filter(|g| m[g.0]).collect();
                    p.extend(more);
                }
                s = decorate(&p, m);
            } else if n > 0 {
                let more = near(&s);
                s.extend(more);
            }
            shells.push(self.torus_topology(&s, n, &w));
        }
        let wrap_step = shells.iter().find(|t| t.chi != 1 || t.pinch_vertices != 0).map(|t| t.n);
        Ok(TorusResult { shells, wrap_step })
    }

    fn torus_topology(&self, s: &HashSet<FaceId>, n: usize, w: &impl Fn(Off) -> Off) -> TorusShell {
        let wv = |v: VertexId| (v.0, w(v.1));
        let (mut verts, mut edges) = (HashSet::new(), HashSet::new());
        for &(f, o) in s {
            let cs = &self.orbits[f];
            for k in 0..cs.len() {
                let (a, b) = (cs[k], cs[(k + 1) % cs.len()]);
                let (u, v) = ((a.site, add(o, a.rel)), (b.site, add(o, b.rel)));
                verts.insert(wv(u));
                let d = sub(v.1, u.1);
                edges.insert(std::cmp::min((wv(u), b.site, d), (wv(v), a.site, sub([0, 0], d))));
            }
        }
        let pinch = verts
            .iter()
            .filter(|&&v| {
                let ring: Vec<bool> = self.vertex_faces(v).iter().map(|&f| s.contains(&(f.0, w(f.1)))).collect();
                (0..ring.len()).filter(|&k| ring[k] && !ring[(k + ring.len() - 1) % ring.len()]).count() >= 2
            })
            .count();
        let (faces, vertices, edges) = (s.len(), verts.len(), edges.len());
        let chi = vertices as i64 - edges as i64 + faces as i64;
        TorusShell { n, faces, vertices, edges, chi, pinch_vertices: pinch }
    }
}
