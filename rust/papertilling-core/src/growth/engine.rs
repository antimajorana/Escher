//! The layer grower (touching rule, plan §2.1-§2.2) and seed sets (§2.1 table).
//!
//! Step k (S_k already built):
//! 1. tally the vertices and edges first seen in layer k, sorted;
//! 2. list the cells at each of those vertices (older vertices have all their cells in S_k)
//!    and the cells on each new edge; count the boundary ([`super::stats`]);
//! 3. stop when k = `max_layers`; else stop with `Rim { k+1 }` if a new vertex is `on_rim`
//!    (its true star is unknown, so S_{k+1} cannot be built whole), or with the first query
//!    error (in sorted order) mapped by [`GrowthError::stopped_at`]`(k+1)`;
//! 4. S_{k+1} = S_k ∪ the listed stars. If the union would pass `max_cells` the layer is not
//!    built: `CellCap { k+1, partial }` with `partial = max_cells + 1`, the count at which the
//!    build stopped (a lower bound on |S_{k+1}|, independent of visiting order). If no cell
//!    is new, `Closed { k }`: S_k is a whole connected complex.
//!
//! Layers hold their cells sorted. An empty seed gives an empty growth; a seed larger than
//! `max_cells` gives `CellCap { layer: 0, partial: seed size }` and no layers.

use std::collections::HashSet;

use super::stats::Tally;
use super::{CellComplex, Growth, GrowthCaps, GrowthError, Stopped};

/// Where a seed sits (§2.1): the cells at a vertex, on an edge, on a face (3D), or one cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeedAt<C, V, K> {
    Vertex(V),
    Edge(K),
    Face(Vec<V>),
    Cell(C),
}

/// The seed set S₀ for a seed position.
pub fn seed_cells<X: CellComplex>(
    x: &mut X,
    at: &SeedAt<X::Cell, X::Vertex, X::Key>,
) -> Result<Vec<X::Cell>, GrowthError> {
    match at {
        SeedAt::Vertex(v) => x.cells_at(*v),
        SeedAt::Edge(e) => x.cells_on_edge(*e),
        SeedAt::Face(f) => x.cells_on_face(f),
        SeedAt::Cell(c) => Ok(vec![*c]),
    }
}

/// Grow `seed` under `caps` with the touching rule (module docs give each step).
pub fn grow<X: CellComplex>(x: &mut X, seed: Vec<X::Cell>, caps: GrowthCaps) -> Growth<X::Cell> {
    let mut seed = seed;
    seed.sort_unstable();
    seed.dedup();
    let mut g = Growth { layers: Vec::new(), stats: Vec::new(), stopped: None };
    if seed.is_empty() {
        return g;
    }
    if seed.len() > caps.max_cells {
        g.stopped = Some(Stopped::CellCap { layer: 0, partial: seed.len() });
        return g;
    }
    let mut in_s: HashSet<X::Cell> = seed.iter().copied().collect();
    let mut tally = Tally::<X>::default();
    g.layers.push(seed);
    let mut k: u32 = 0;
    loop {
        let fresh = tally.add_cells(x, &g.layers[k as usize]);
        let mut err: Option<GrowthError> = None;
        let mut rim = false;
        let mut stars: Vec<Vec<X::Cell>> = Vec::with_capacity(fresh.vertices.len());
        let mut bv = 0;
        for &v in &fresh.vertices {
            rim |= x.on_rim(v);
            match x.cells_at(v) {
                Ok(cs) => {
                    bv += usize::from(cs.iter().any(|c| !in_s.contains(c)));
                    stars.push(cs);
                }
                Err(e) => {
                    bv += 1;
                    err.get_or_insert(e);
                }
            }
        }
        let mut be = 0;
        for &e in &fresh.edges {
            match x.cells_on_edge(e) {
                Ok(cs) => be += usize::from(cs.iter().any(|c| !in_s.contains(c))),
                Err(e2) => {
                    be += 1;
                    err.get_or_insert(e2);
                }
            }
        }
        let n = in_s.len();
        g.stats.push(tally.stats(n, g.layers[k as usize].len(), bv, be));
        if k >= caps.max_layers {
            break;
        }
        if rim {
            g.stopped = Some(Stopped::Rim { layer: k + 1 });
            break;
        }
        if let Some(e) = err {
            g.stopped = Some(e.stopped_at(k + 1));
            break;
        }
        let mut next = Vec::new();
        let mut seen = HashSet::new();
        let mut capped = false;
        'stars: for cs in &stars {
            for &c in cs {
                if !in_s.contains(&c) && seen.insert(c) {
                    next.push(c);
                    if n + next.len() > caps.max_cells {
                        capped = true;
                        break 'stars;
                    }
                }
            }
        }
        if capped {
            g.stopped = Some(Stopped::CellCap { layer: k + 1, partial: n + next.len() });
            break;
        }
        if next.is_empty() {
            g.stopped = Some(Stopped::Closed { layer: k });
            break;
        }
        next.sort_unstable();
        in_s.extend(next.iter().copied());
        g.layers.push(next);
        k += 1;
    }
    g
}
