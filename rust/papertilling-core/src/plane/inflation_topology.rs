//! Topology of a finite face set of a [`PlaneTiling`]: the public [`PatchTopology`] counts and the incremental
//! [`Tracker`] that `inflate` keeps while a patch grows. A child module of `plane::inflation` (declared there with
//! `#[path]`), so it reads the tiling's private rotation tables; `PatchTopology` is re-exported from there.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{FaceId, PlaneTiling, VertexId, add};

/// Topology of a finite face set, read off the closed 2-complex it spans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchTopology {
    pub faces: usize,
    pub vertices: usize,
    pub edges: usize,
    /// Edges with exactly one of their two sides on a patch face.
    pub boundary_edges: usize,
    /// `V - E + F` of the closed 2-complex.
    pub chi: i64,
    /// Patch vertices around which the patch faces form two or more separate arcs.
    pub pinch_vertices: usize,
    /// Lattice bonds joining two patch vertices that lie on no patch face.
    pub dangling_bonds: usize,
}

impl PlaneTiling {
    /// Euler characteristic, boundary, pinch and dangling-bond counts of the closed 2-complex spanned by `patch`.
    pub fn topology(&self, patch: &HashSet<FaceId>) -> PatchTopology {
        let mut tracker = Tracker::default();
        tracker.add(self, patch, patch);
        tracker.stats()
    }
}

/// Running [`PatchTopology`] of a growing patch. Adding faces changes the pinch status of, and the dangling status
/// of bonds at, only the vertices of the added faces, so [`Tracker::add`] rechecks just those.
#[derive(Default)]
pub(super) struct Tracker {
    faces: usize,
    verts: HashSet<VertexId>,
    sides: HashMap<(VertexId, VertexId), usize>,
    boundary: usize,
    pinch: HashSet<VertexId>,
    dangling: HashSet<(VertexId, VertexId)>,
}

fn edge(u: VertexId, w: VertexId) -> (VertexId, VertexId) {
    if u < w { (u, w) } else { (w, u) }
}

impl Tracker {
    /// Record the faces `new` (distinct, not recorded before), which `patch` must already contain.
    pub(super) fn add<'a>(&mut self, t: &PlaneTiling, new: impl IntoIterator<Item = &'a FaceId>, patch: &HashSet<FaceId>) {
        let mut touched = HashSet::new();
        for &face in new {
            self.faces += 1;
            let vs = t.face_vertices(face);
            for k in 0..vs.len() {
                touched.insert(vs[k]);
                self.verts.insert(vs[k]);
                let n = self.sides.entry(edge(vs[k], vs[(k + 1) % vs.len()])).or_insert(0);
                *n += 1;
                // An edge is a boundary edge while exactly one of its sides is a patch face.
                self.boundary = self.boundary + usize::from(*n == 1) - usize::from(*n == 2);
            }
        }
        for v in touched {
            let ring: Vec<bool> = t.vertex_faces(v).iter().map(|f| patch.contains(f)).collect();
            let arcs = (0..ring.len()).filter(|&k| ring[k] && !ring[(k + ring.len() - 1) % ring.len()]).count();
            toggle(&mut self.pinch, v, arcs >= 2);
            for c in &t.rotation[v.0] {
                let w = (c.to_site, add(v.1, c.to_off));
                toggle(&mut self.dangling, edge(v, w), self.verts.contains(&w) && !self.sides.contains_key(&edge(v, w)));
            }
        }
    }

    pub(super) fn stats(&self) -> PatchTopology {
        let (faces, vertices, edges) = (self.faces, self.verts.len(), self.sides.len());
        let chi = vertices as i64 - edges as i64 + faces as i64;
        let (boundary_edges, pinch_vertices, dangling_bonds) = (self.boundary, self.pinch.len(), self.dangling.len());
        PatchTopology { faces, vertices, edges, boundary_edges, chi, pinch_vertices, dangling_bonds }
    }
}

fn toggle<T: std::hash::Hash + Eq>(set: &mut HashSet<T>, x: T, on: bool) {
    let _ = if on { set.insert(x) } else { set.remove(&x) };
}
