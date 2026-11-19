//! Topology of a finite face set of a [`PlaneTiling`]: the public [`PatchTopology`] counts and the incremental
//! [`Tracker`] that `inflate` keeps while a patch grows. A child module of `plane::inflation` (declared there with
//! `#[path]`), so it reads the tiling's private rotation tables; `PatchTopology` is re-exported from there.

use std::collections::{BTreeMap, HashMap, HashSet};

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
    /// Patch faces with at least one boundary edge (a face touching the boundary only at a vertex is not one).
    pub boundary_tiles: usize,
}

impl PlaneTiling {
    /// Euler characteristic, boundary, pinch and dangling-bond counts of the closed 2-complex spanned by `patch`.
    pub fn topology(&self, patch: &HashSet<FaceId>) -> PatchTopology {
        let mut tracker = Tracker::default();
        tracker.add(self, patch, patch);
        tracker.stats()
    }

    /// Boundary vertex types of `patch`: for each vertex with at least one non-patch face corner around it, `t` =
    /// the number of patch face corners there; returns `t -> number of such vertices`.
    pub fn boundary_vertex_types(&self, patch: &HashSet<FaceId>) -> BTreeMap<usize, usize> {
        let mut tracker = Tracker::default();
        tracker.add(self, patch, patch);
        tracker.vertex_types()
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
    /// Boundary vertex -> number of patch face corners at it.
    btype: HashMap<VertexId, usize>,
    /// Patch faces with a non-patch face across one of their edges.
    bfaces: HashSet<FaceId>,
}

fn edge(u: VertexId, w: VertexId) -> (VertexId, VertexId) {
    if u < w { (u, w) } else { (w, u) }
}

impl Tracker {
    /// Record the faces `new` (distinct, not recorded before), which `patch` must already contain.
    pub(super) fn add<'a>(&mut self, t: &PlaneTiling, new: impl IntoIterator<Item = &'a FaceId>, patch: &HashSet<FaceId>) {
        let mut touched = HashSet::new();
        let mut recheck = HashSet::new();
        for &face in new {
            self.faces += 1;
            recheck.insert(face);
            recheck.extend(t.across(face).filter(|g| patch.contains(g)));
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
            let inside = ring.iter().filter(|&&b| b).count();
            if inside < ring.len() { self.btype.insert(v, inside) } else { self.btype.remove(&v) };
            for c in &t.rotation[v.0] {
                let w = (c.to_site, add(v.1, c.to_off));
                toggle(&mut self.dangling, edge(v, w), self.verts.contains(&w) && !self.sides.contains_key(&edge(v, w)));
            }
        }
        for f in recheck {
            let open = t.across(f).any(|g| !patch.contains(&g));
            toggle(&mut self.bfaces, f, open);
        }
    }

    /// `t -> number of boundary vertices with t patch face corners`.
    pub(super) fn vertex_types(&self) -> BTreeMap<usize, usize> {
        let mut out = BTreeMap::new();
        for &t in self.btype.values() {
            *out.entry(t).or_insert(0) += 1;
        }
        out
    }

    pub(super) fn stats(&self) -> PatchTopology {
        let (faces, vertices, edges) = (self.faces, self.verts.len(), self.sides.len());
        let chi = vertices as i64 - edges as i64 + faces as i64;
        let (boundary_edges, pinch_vertices, dangling_bonds) = (self.boundary, self.pinch.len(), self.dangling.len());
        let boundary_tiles = self.bfaces.len();
        PatchTopology { faces, vertices, edges, boundary_edges, chi, pinch_vertices, dangling_bonds, boundary_tiles }
    }
}

fn toggle<T: std::hash::Hash + Eq>(set: &mut HashSet<T>, x: T, on: bool) {
    let _ = if on { set.insert(x) } else { set.remove(&x) };
}
