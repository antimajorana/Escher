//! Seeded layer growth on a cell complex (C1, track T2; plan `c1-plan.md` §2.1-§2.3).
//!
//! A tiling or honeycomb is a cell complex: top cells (tiles in 2D, cells in 3D) and
//! vertices, with incidence given by [`CellComplex`]. Growth starts from a seed set S₀
//! and applies the touching rule (BDF vertex completion):
//! S_{k+1} = ⋃ { cells_at(v) : v ∈ vertices(S_k) }.
//! Layer 0 is the seed set. A layer is built whole or not at all; when growth stops,
//! [`Growth::stopped`] says why ([`Stopped`]).
//!
//! This file is frozen after WP-C0: Wave 1 packages fill the submodules and add their
//! public items there, not here.
//!
//! JSON forms: `SeedKind` is `"vertex"`, `"edge"`, ...; `Stopped` and `GrowthError` are
//! tagged by `kind` in snake_case, e.g. `{"kind": "cell_cap", "layer": 3, "partial": 61000}`.

use std::hash::Hash;

use serde::{Deserialize, Serialize};

pub mod e3_complex;
pub mod engine;
pub mod face_graph;
pub mod finite;
pub mod h3_complex;
pub mod periodic_net;
pub mod stats;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_face_graph;
#[cfg(test)]
mod tests_h3;
#[cfg(test)]
mod tests_periodic_net;

/// A cell complex grown lazily: adapters may build cells on demand.
///
/// `Key` is a canonical key for an edge or (3D) a face. An edge is keyed by itself,
/// not by its vertex pair, because in a hosohedron {2,q} two vertices bound q edges.
pub trait CellComplex {
    type Cell: Copy + Eq + Hash + Ord;
    type Vertex: Copy + Eq + Hash + Ord;
    type Key: Copy + Eq + Hash + Ord;

    /// Vertices of a top cell.
    fn vertices_of(&self, c: Self::Cell) -> Vec<Self::Vertex>;
    /// Canonical edge keys of a top cell.
    fn edges_of(&self, c: Self::Cell) -> Vec<Self::Key>;
    /// Canonical face keys of a top cell (3D only; empty in 2D).
    fn faces_of(&self, c: Self::Cell) -> Vec<Self::Key>;
    /// True when `v` lies on the rim of a finite crystal (T7), where cells are missing
    /// because the crystal ends, not because growth stopped.
    fn on_rim(&self, _v: Self::Vertex) -> bool {
        false
    }
    /// All top cells containing `v`. `Err` when that set is infinite (an ideal vertex)
    /// or cannot be built ([`GrowthError`]).
    fn cells_at(&mut self, v: Self::Vertex) -> Result<Vec<Self::Cell>, GrowthError>;
    /// All top cells containing the edge `e` (2 in 2D, r around an edge of {p,q,r}).
    fn cells_on_edge(&mut self, e: Self::Key) -> Result<Vec<Self::Cell>, GrowthError>;
    /// The top cells on the face spanned by `f` (3D only; 2 cells).
    fn cells_on_face(&mut self, f: &[Self::Vertex]) -> Result<Vec<Self::Cell>, GrowthError>;
}

/// Why an adapter could not answer an incidence query.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrowthError {
    /// The answer is an infinite set (for example the cells at an ideal vertex of a
    /// paracompact honeycomb, or the vertices of an infinite {6,3} cell).
    Infinite { reason: &'static str },
    /// Two distinct keys came closer than the merge tolerance allows (§2.3).
    Precision { gap: f64 },
    /// The complex holds no cells beyond its stored range (a finite crystal or table).
    OutOfRange { reason: &'static str },
}

impl GrowthError {
    /// The stop reported when this error occurs while building layer `layer`.
    pub fn stopped_at(&self, layer: u32) -> Stopped {
        match *self {
            GrowthError::Infinite { reason } => Stopped::Infinite { layer, reason },
            GrowthError::Precision { gap } => Stopped::Precision { layer, gap },
            GrowthError::OutOfRange { .. } => Stopped::Rim { layer },
        }
    }
}

/// The seed kinds of §2.1. Vertex is the default (D19).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeedKind {
    /// The cells at one vertex (vertex star).
    #[default]
    Vertex,
    /// The cells on one edge, keyed by the edge.
    Edge,
    /// The 2 cells on one face (3D only; in 2D a face is a tile).
    Face,
    /// One top cell.
    Cell,
    /// The cells meeting the Dirichlet (Wigner-Seitz) cell of a point, ties included (D18).
    Dirichlet,
}

impl SeedKind {
    /// Every seed kind, in menu order.
    pub const ALL: [SeedKind; 5] = [
        SeedKind::Vertex,
        SeedKind::Edge,
        SeedKind::Face,
        SeedKind::Cell,
        SeedKind::Dirichlet,
    ];
}

/// Count-based caps (§2.2). Layers run 0..=max_layers; a layer whose union would pass
/// `max_cells` is not built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GrowthCaps {
    pub max_layers: u32,
    pub max_cells: usize,
}

/// Why growth stopped before `max_layers`, or that it closed up.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Stopped {
    /// Building `layer` would pass the cell cap; `partial` cells were counted before
    /// stopping. The last layer shown is `layer - 1`.
    CellCap { layer: u32, partial: usize },
    /// `layer` would be infinite (paracompact H³, §1 Q4).
    Infinite { layer: u32, reason: &'static str },
    /// Duplicate merging lost its margin while building `layer` (§2.3).
    Precision { layer: u32, gap: f64 },
    /// S_{layer+1} = S_layer is the whole complex (S², S³).
    Closed { layer: u32 },
    /// Growth reached the rim of a finite crystal at `layer`.
    Rim { layer: u32 },
}

impl Stopped {
    /// The layer index this stop refers to.
    pub fn layer(&self) -> u32 {
        match *self {
            Stopped::CellCap { layer, .. }
            | Stopped::Infinite { layer, .. }
            | Stopped::Precision { layer, .. }
            | Stopped::Closed { layer }
            | Stopped::Rim { layer } => layer,
        }
    }
}

/// Readout for one layer S_k (§2.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LayerStats {
    /// |S_k|.
    pub cells: usize,
    /// |S_k \ S_{k-1}| (the seed size at k = 0).
    pub new_cells: usize,
    pub boundary_vertices: usize,
    pub boundary_edges: usize,
    /// χ of the union: V − E + F in 2D, V − E + F − C in 3D. Measured, never asserted (§2.1).
    pub euler: i64,
}

/// Result of [`grow`]. `layers[k]` holds the cells first added at layer k
/// (`layers[0]` is the deduplicated seed), so S_k is the union of `layers[0..=k]`.
/// `stats[k]` describes S_k.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Growth<C> {
    pub layers: Vec<Vec<C>>,
    pub stats: Vec<LayerStats>,
    pub stopped: Option<Stopped>,
}

/// Grow `seed` layer by layer under `caps` with the touching rule.
pub fn grow<X: CellComplex>(x: &mut X, seed: Vec<X::Cell>, caps: GrowthCaps) -> Growth<X::Cell> {
    engine::grow(x, seed, caps)
}
