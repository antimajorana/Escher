//! Per-layer readout (§2.1): the running vertex, edge and face sets of the union S_k, and
//! χ of the union.
//!
//! Definitions used by [`super::engine::grow`]:
//! - a **boundary vertex** of S_k is a vertex of S_k whose cells (`cells_at`) are not all in
//!   S_k, or whose cells could not be listed (an `Err` counts as "not all in");
//! - a **boundary edge** is an edge of S_k whose cells (`cells_on_edge`) are not all in S_k.
//!
//! This is the growth frontier, in 2D and 3D alike (the trait has no face-to-edge incidence,
//! so a 3D boundary edge is not defined through boundary faces). On a 2D disk patch the
//! boundary vertices and edges form one cycle (V_b = E_b); on a 3D ball they are the
//! vertices and edges of the boundary sphere. Only vertices and edges first seen in layer k
//! can be on the boundary of S_k: every cell at an older vertex was added by layer k.
//!
//! χ = V − E + F in 2D (the top cells are the faces) and V − E + F − C in 3D; a complex
//! is 3D when its cells report faces.

use std::collections::HashSet;

use super::{CellComplex, LayerStats};

/// Vertices and edges first seen in the layer just added, sorted (so Rust and Python visit
/// them in the same order).
pub struct Fresh<V, K> {
    pub vertices: Vec<V>,
    pub edges: Vec<K>,
}

/// The vertex, edge and face sets of the union so far.
pub struct Tally<X: CellComplex> {
    verts: HashSet<X::Vertex>,
    edges: HashSet<X::Key>,
    faces: HashSet<X::Key>,
}

impl<X: CellComplex> Default for Tally<X> {
    fn default() -> Self {
        Self { verts: HashSet::new(), edges: HashSet::new(), faces: HashSet::new() }
    }
}

impl<X: CellComplex> Tally<X> {
    /// Add the cells of one layer; returns the vertices and edges seen for the first time.
    pub fn add_cells(&mut self, x: &X, cells: &[X::Cell]) -> Fresh<X::Vertex, X::Key> {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        for &c in cells {
            for v in x.vertices_of(c) {
                if self.verts.insert(v) {
                    vertices.push(v);
                }
            }
            for e in x.edges_of(c) {
                if self.edges.insert(e) {
                    edges.push(e);
                }
            }
            self.faces.extend(x.faces_of(c));
        }
        vertices.sort_unstable();
        edges.sort_unstable();
        Fresh { vertices, edges }
    }

    /// χ of a union with `cells` top cells.
    pub fn euler(&self, cells: usize) -> i64 {
        let (v, e, f, c) = (self.verts.len() as i64, self.edges.len() as i64, self.faces.len() as i64, cells as i64);
        if self.faces.is_empty() {
            v - e + c
        } else {
            v - e + f - c
        }
    }

    /// The readout for S_k.
    pub fn stats(&self, cells: usize, new_cells: usize, boundary_vertices: usize, boundary_edges: usize) -> LayerStats {
        LayerStats { cells, new_cells, boundary_vertices, boundary_edges, euler: self.euler(cells) }
    }
}
