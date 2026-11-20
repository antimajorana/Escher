//! Fusion of defects shown as a sequence of explicit lattice graphs.
//!
//! The Volterra surgery of `defect::surgery` assumes perfect lattice round every core, so it cannot
//! build the lattices that appear while cores merge. Here a lattice patch is a planar map written down
//! directly (`map`), changed by local moves (`moves`). `wang` is the first worked example.

pub mod map;
pub mod moves;
pub mod svg;
pub mod wang;

#[cfg(test)]
mod tests;

pub use map::{Census, Dart, FaceWalk, MapEdge, MapVertex, PlanarMap, VertexKind};
pub use moves::Side;
pub use wang::{FusionStep, fig_8a, gate_route, hamiltonian_route};
