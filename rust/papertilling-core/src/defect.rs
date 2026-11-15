//! Geometry-agnostic holonomy classification and bond-graph surgery types (Stages 4–5).

use petgraph::algo::is_isomorphic_matching;
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};

/// Crystalline defect families classified by holonomy and mirror/glide surgery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectKind {
    Dislocation,
    Disclination,
    DomainWall,
    TwinWall,
    MobiusSeam,
    GlideWall,
}

/// Defect core construction mode used for equivalence checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreKind {
    Volterra,
    Completion,
}

/// Abstract lifted proper symmetry group acting on bond holonomies across cuts.
pub trait HolonomyGroup {
    type Element: Clone + Eq;

    fn identity(&self) -> Self::Element;
    fn compose(&self, left: &Self::Element, right: &Self::Element) -> Self::Element;
    fn inverse(&self, elem: &Self::Element) -> Self::Element;

    fn conjugate(&self, elem: &Self::Element, by: &Self::Element) -> Self::Element {
        let by_inv = self.inverse(by);
        self.compose(&self.compose(by, elem), &by_inv)
    }
}

/// Directed bond in a surgically cut-and-reglued crystal graph, carrying accumulated holonomy `H`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurgicalBond<H> {
    pub tail: usize,
    pub head: usize,
    pub holonomy: H,
    pub bond_type: u32,
}

/// Geometry-free result of cut-and-reglue surgery on a crystal bond graph prior to spatial relaxation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BondGraphSurgery<H> {
    pub num_sites: usize,
    pub bonds: Vec<SurgicalBond<H>>,
}

impl<H> BondGraphSurgery<H> {
    pub fn new(num_sites: usize, bonds: Vec<SurgicalBond<H>>) -> Self {
        Self { num_sites, bonds }
    }
}

/// Labelled VF2 graph isomorphism check for comparing defect core graphs.
pub fn cores_isomorphic<N: PartialEq, E: PartialEq>(
    left: &UnGraph<N, E>,
    right: &UnGraph<N, E>,
) -> bool {
    is_isomorphic_matching(left, right, |a, b| a == b, |a, b| a == b)
}
