//! Stage 6: regular `{p,q,r}` honeycombs of S³ / E³ / H³, 3D Voronoi cells, Dirichlet domains of
//! closed 3-manifolds, and clean-room triply-periodic P / D / G nets and surfaces.
//!
//! Everything here is derived from first principles (Coxeter reflection groups, perpendicular
//! bisectors, classical nodal approximations); no external net database is used.

pub mod coxeter_gram;
pub mod coxeter_ldl;
pub mod coxeter_orbit;
pub mod dirichlet_quotient;
pub mod growth_build;
pub mod honeycombs_e3;
pub mod honeycombs_h3;
pub mod key_index;
pub mod paracompact_ball;
pub mod polychora_s3;
pub mod tpms_nets;
pub mod tpms_surfaces;
pub mod voronoi3d_dirichlet;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_geometry;
#[cfg(test)]
mod tests_growth;

use std::collections::{HashMap, HashSet};

use serde::Serialize;

pub use coxeter_gram::{classify_schlafli, Classification, HoneycombRegime};
pub use growth_build::{grow_honeycomb, GrowthRule, HoneycombGrowth, HoneycombKind};
pub use honeycombs_e3::{build_euclidean_honeycomb, EuclideanKind};
pub use paracompact_ball::{paracompact_ball, ParacompactBall};
pub use honeycombs_h3::{build_honeycomb, build_hyperbolic_honeycomb};
pub use polychora_s3::build_polychoron;
pub use tpms_nets::{tpms_net, TpmsKind, TriplyPeriodicNet};
pub use tpms_surfaces::{nodal_surface_mesh, petrie_coxeter_surface, PeriodicSurface};
pub use voronoi3d_dirichlet::{
    dirichlet_manifold, lattice_voronoi_cell, voronoi_cell_3d, DirichletDomain3d, DirichletQuotient,
    VoronoiPolyhedron3d,
};

/// A finite 3D cell complex (a patch of a honeycomb, or a whole polychoron).
#[derive(Clone, Debug, Default, Serialize)]
pub struct CellComplex3d {
    pub name: String,
    pub regime: String,
    /// Ambient coordinates: unit vectors of R⁴ (S³), hyperboloid / light-cone vectors with
    /// coordinate 0 as time (H³), or `(x, y, z, 1)` (E³).
    pub coords4: Vec<[f64; 4]>,
    /// Render coordinates: stereographic image (S³), Poincaré ball (H³), Euclidean (E³).
    pub points: Vec<[f64; 3]>,
    /// Klein-ball coordinates (H³ only; empty otherwise).
    pub klein: Vec<[f64; 3]>,
    /// Vertex lies on the ideal sphere at infinity (H³ with Euclidean vertex figure).
    pub is_ideal: Vec<bool>,
    pub edges: Vec<[usize; 2]>,
    /// Faces as ordered vertex cycles.
    pub faces: Vec<Vec<usize>>,
    /// Cells as lists of face indices.
    pub cells: Vec<Vec<usize>>,
    /// Face-adjacency distance of each cell from the central cell.
    pub cell_shell: Vec<u32>,
    /// Cells that are infinite (horospherical `{4,4}` / `{6,3}`) and were truncated to a patch.
    pub cell_truncated: Vec<bool>,
}

impl CellComplex3d {
    /// `(V, E, F, C)`.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (self.points.len(), self.edges.len(), self.faces.len(), self.cells.len())
    }

    /// `V - E + F - C`.
    pub fn euler(&self) -> i64 {
        let (v, e, f, c) = self.counts();
        v as i64 - e as i64 + f as i64 - c as i64
    }
}

/// Tolerance-based point index (spatial hash, neighbouring buckets searched).
#[derive(Clone, Debug)]
pub struct PointIndex<const D: usize> {
    tol: f64,
    buckets: HashMap<[i64; D], Vec<usize>>,
    pts: Vec<[f64; D]>,
}

impl<const D: usize> PointIndex<D> {
    pub fn new(tol: f64) -> Self {
        Self { tol, buckets: HashMap::new(), pts: Vec::new() }
    }

    fn key(&self, x: &[f64; D]) -> [i64; D] {
        let mut k = [0i64; D];
        for (ki, xi) in k.iter_mut().zip(x) {
            *ki = (xi / (4.0 * self.tol)).floor() as i64;
        }
        k
    }

    pub fn find(&self, x: &[f64; D]) -> Option<usize> {
        let k = self.key(x);
        let n = 3usize.pow(D as u32);
        for code in 0..n {
            let mut kk = k;
            let mut c = code;
            for ki in kk.iter_mut() {
                *ki += (c % 3) as i64 - 1;
                c /= 3;
            }
            if let Some(ids) = self.buckets.get(&kk) {
                for &i in ids {
                    let d2: f64 = self.pts[i].iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum();
                    if d2 <= self.tol * self.tol {
                        return Some(i);
                    }
                }
            }
        }
        None
    }

    /// Return `(index, inserted)`.
    pub fn insert(&mut self, x: [f64; D]) -> (usize, bool) {
        if let Some(i) = self.find(&x) {
            return (i, false);
        }
        let i = self.pts.len();
        let k = self.key(&x);
        self.buckets.entry(k).or_default().push(i);
        self.pts.push(x);
        (i, true)
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    pub fn points(&self) -> &[[f64; D]] {
        &self.pts
    }
}

/// Incremental builder: deduplicates faces by their sorted vertex set and derives edges.
#[derive(Default)]
pub struct ComplexBuilder {
    pub out: CellComplex3d,
    face_ids: HashMap<Vec<usize>, usize>,
    edge_ids: HashSet<[usize; 2]>,
}

impl ComplexBuilder {
    pub fn new(name: &str, regime: &str) -> Self {
        let out = CellComplex3d { name: name.into(), regime: regime.into(), ..Default::default() };
        Self { out, face_ids: HashMap::new(), edge_ids: HashSet::new() }
    }

    /// Add a face given as an ordered vertex cycle; returns its id.
    pub fn add_face(&mut self, cycle: Vec<usize>) -> usize {
        let mut key = cycle.clone();
        key.sort_unstable();
        if let Some(&f) = self.face_ids.get(&key) {
            return f;
        }
        let n = cycle.len();
        for i in 0..n {
            let (a, b) = (cycle[i], cycle[(i + 1) % n]);
            let e = [a.min(b), a.max(b)];
            if self.edge_ids.insert(e) {
                self.out.edges.push(e);
            }
        }
        let f = self.out.faces.len();
        self.face_ids.insert(key, f);
        self.out.faces.push(cycle);
        f
    }

    pub fn add_cell(&mut self, faces: Vec<usize>, shell: u32, truncated: bool) {
        self.out.cells.push(faces);
        self.out.cell_shell.push(shell);
        self.out.cell_truncated.push(truncated);
    }

    pub fn finish(self) -> CellComplex3d {
        self.out
    }
}

pub(crate) fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub(crate) fn norm3(a: &[f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
