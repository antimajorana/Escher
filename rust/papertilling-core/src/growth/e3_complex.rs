//! The four E³ honeycombs as exact integer complexes, for the touching-rule grower.
//!
//! Coordinates are integers at scale [`E3_SCALE`] = 2 (real = integer / 2), the geometry of
//! [`crate::honeycomb3d::build_euclidean_honeycomb`]:
//! - cubic: cells at the integer points, cubes of side 1;
//! - bitruncated cubic (Kelvin): cells at the BCC points (all coordinates even or all odd),
//!   truncated octahedra with vertices at the permutations of (0, ±½, ±1) about the centre;
//! - rhombic dodecahedral: cells at the FCC points (even coordinate sum), vertices at
//!   (±1, 0, 0) (in 6 cells) and (±½, ±½, ±½) (in 4 cells) about the centre;
//! - alternated cubic (octet): vertices at the odd-sum integer points, octahedra centred at the
//!   even-sum points, tetrahedra centred at the half-integer points (two orientations).
//!
//! A cell is its scaled centre. Its class (the octet has three: octahedron and two tetrahedra)
//! follows from the centre's residues, and its vertices are the centre plus the class offsets.
//! Edges join the vertex pairs of a cell at the smallest distance (true for all five
//! polyhedra); faces are the vertex sets maximising n·x for the class's face normals.
//! An edge or face is keyed by [`E3Key`] = (vertex count, sum of its scaled vertices): exact,
//! and unique because distinct faces of a face-to-face convex tiling have distinct centroids.
//! So `cells_on_edge` and `cells_on_face` solve key = n·c + (offset sum) for c directly.
//!
//! Seed representatives (scaled), chosen as the most symmetric of each kind:
//! - vertex: cubic (1,1,1); Kelvin (0,1,2) (one orbit); rhombic (2,0,0), the vertex in 6 cells
//!   (site symmetry m-3m), not (1,1,1); octet (2,0,0) (one orbit);
//! - edge: cubic (1,1,±1); Kelvin (1,2,0)-(2,1,0), between two hexagons; rhombic
//!   (2,0,0)-(1,1,1) (one orbit); octet (2,0,0)-(0,2,0) (one orbit);
//! - face: cubic x = 1; Kelvin the square x = 2 (not a hexagon); rhombic the rhombus with
//!   normal (1,1,0); octet the triangle (2,0,0),(0,2,0),(0,0,2);
//! - cell: the cell at the origin (the octahedron for the octet).

use serde::Serialize;

use super::engine::SeedAt;
use super::{CellComplex, GrowthError, SeedKind};
use crate::honeycomb3d::EuclideanKind;

/// Integer coordinates are this many times the real ones.
pub const E3_SCALE: i32 = 2;

/// A scaled point.
pub type P3 = [i32; 3];

/// An edge (n = 2) or face (n ≥ 3) key: vertex count and sum of the scaled vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct E3Key {
    pub n: u8,
    pub sum: P3,
}

struct Class {
    offsets: Vec<P3>,
    edges: Vec<(usize, usize)>,
    faces: Vec<Vec<usize>>,
    edge_sums: Vec<P3>,
    face_sums: Vec<E3Key>,
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn key_of(vs: &[P3]) -> E3Key {
    let sum = vs.iter().fold([0; 3], |s, v| add(s, *v));
    E3Key { n: vs.len() as u8, sum }
}

/// All integer points with these absolute values in some order (and every sign).
fn perms_of(abs_sorted: [i32; 3]) -> Vec<P3> {
    let mut out = Vec::new();
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                let mut s = [i32::abs(a), i32::abs(b), i32::abs(c)];
                s.sort_unstable();
                if s == abs_sorted {
                    out.push([a, b, c]);
                }
            }
        }
    }
    out
}

fn corners() -> Vec<P3> {
    perms_of([1, 1, 1])
}

fn class(offsets: Vec<P3>, normals: &[P3]) -> Class {
    let d2 = |a: P3, b: P3| sub(a, b).iter().map(|t| t * t).sum::<i32>();
    let pairs = (0..offsets.len()).flat_map(|i| (i + 1..offsets.len()).map(move |j| (i, j)));
    let m = pairs.clone().map(|(i, j)| d2(offsets[i], offsets[j])).min().unwrap_or(0);
    let edges: Vec<(usize, usize)> = pairs.filter(|&(i, j)| d2(offsets[i], offsets[j]) == m).collect();
    let faces: Vec<Vec<usize>> = normals
        .iter()
        .map(|n| {
            let dots: Vec<i32> = offsets.iter().map(|o| n[0] * o[0] + n[1] * o[1] + n[2] * o[2]).collect();
            let top = *dots.iter().max().unwrap_or(&0);
            (0..offsets.len()).filter(|&i| dots[i] == top).collect()
        })
        .collect();
    let edge_sums = edges.iter().map(|&(i, j)| add(offsets[i], offsets[j])).collect();
    let face_sums = faces.iter().map(|f| key_of(&f.iter().map(|&i| offsets[i]).collect::<Vec<_>>())).collect();
    Class { offsets, edges, faces, edge_sums, face_sums }
}

/// One E³ honeycomb kind as an exact, unbounded complex.
pub struct E3Complex {
    kind: EuclideanKind,
    classes: Vec<Class>,
}

impl E3Complex {
    pub fn new(kind: EuclideanKind) -> Self {
        let axes = perms_of([0, 0, 1]);
        let classes = match kind {
            EuclideanKind::Cubic => vec![class(corners(), &axes)],
            EuclideanKind::BitruncatedCubic => {
                let normals: Vec<P3> = axes.iter().copied().chain(corners()).collect();
                vec![class(perms_of([0, 1, 2]), &normals)]
            }
            EuclideanKind::RhombicDodecahedral => {
                let offsets = perms_of([0, 0, 2]).into_iter().chain(corners()).collect();
                vec![class(offsets, &perms_of([0, 1, 1]))]
            }
            EuclideanKind::AlternatedCubic => {
                let tet = |r: i32| -> Class {
                    let o: Vec<P3> = corners().into_iter().filter(|c| (c[0] + c[1] + c[2]).rem_euclid(4) == r).collect();
                    let normals: Vec<P3> = o.iter().map(|c| [-c[0], -c[1], -c[2]]).collect();
                    class(o, &normals)
                };
                vec![class(perms_of([0, 0, 2]), &corners()), tet(1), tet(3)]
            }
        };
        Self { kind, classes }
    }

    pub fn parse(kind: &str) -> Result<Self, String> {
        EuclideanKind::parse(kind).map(Self::new)
    }

    pub fn kind(&self) -> EuclideanKind {
        self.kind
    }

    /// The class of the cell centred at `c`, or `None` when no cell is centred there.
    pub fn class_of(&self, c: P3) -> Option<usize> {
        let even = c.iter().all(|t| t % 2 == 0);
        let sum4 = (c[0] + c[1] + c[2]).rem_euclid(4);
        match self.kind {
            EuclideanKind::Cubic => even.then_some(0),
            EuclideanKind::BitruncatedCubic => {
                let r = c[0].rem_euclid(4);
                (r % 2 == 0 && c.iter().all(|t| t.rem_euclid(4) == r)).then_some(0)
            }
            EuclideanKind::RhombicDodecahedral => (even && sum4 == 0).then_some(0),
            EuclideanKind::AlternatedCubic => {
                if even {
                    (sum4 == 0).then_some(0)
                } else if c.iter().all(|t| t.rem_euclid(2) == 1) {
                    Some(if sum4 == 1 { 1 } else { 2 })
                } else {
                    None
                }
            }
        }
    }

    /// Whether `v` is a vertex of the honeycomb (an independent residue test).
    pub fn is_vertex(&self, v: P3) -> bool {
        let odd = v.iter().filter(|t| t.rem_euclid(2) == 1).count();
        let even2 = odd == 0 && (v[0] + v[1] + v[2]).rem_euclid(4) == 2;
        match self.kind {
            EuclideanKind::Cubic => odd == 3,
            EuclideanKind::BitruncatedCubic => {
                let ev: Vec<i32> = v.iter().filter(|t| t.rem_euclid(2) == 0).map(|t| t.rem_euclid(4)).collect();
                odd == 1 && ev[0] != ev[1]
            }
            EuclideanKind::RhombicDodecahedral => even2 || odd == 3,
            EuclideanKind::AlternatedCubic => even2,
        }
    }

    fn verts(&self, c: P3) -> Option<(usize, Vec<P3>)> {
        let ci = self.class_of(c)?;
        Some((ci, self.classes[ci].offsets.iter().map(|o| add(c, *o)).collect()))
    }

    /// Cells `c` whose class has `key - n·c` among its relative keys (`n` = key.n).
    fn cells_with(&self, key: E3Key, faces: bool) -> Vec<P3> {
        let n = i32::from(key.n);
        let mut out = Vec::new();
        for (ci, cl) in self.classes.iter().enumerate() {
            let rel: Vec<P3> = if faces {
                cl.face_sums.iter().filter(|k| k.n == key.n).map(|k| k.sum).collect()
            } else {
                cl.edge_sums.clone()
            };
            for t in rel {
                let d = sub(key.sum, t);
                if d.iter().all(|q| q % n == 0) {
                    let c = [d[0] / n, d[1] / n, d[2] / n];
                    if self.class_of(c) == Some(ci) {
                        out.push(c);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The cells on the face with key `key` (a face key from [`CellComplex::faces_of`]).
    pub fn cells_on_face_key(&self, key: E3Key) -> Vec<P3> {
        if key.n < 3 {
            return Vec::new();
        }
        self.cells_with(key, true)
    }

    /// The seed position of a seed kind (module docs); Dirichlet is refused with a reason.
    pub fn default_seed(&self, seed: SeedKind) -> Result<SeedAt<P3, P3, E3Key>, String> {
        let (v, e, f): (P3, [P3; 2], Vec<P3>) = match self.kind {
            EuclideanKind::Cubic => ([1, 1, 1], [[1, 1, 1], [1, 1, -1]], vec![[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]]),
            EuclideanKind::BitruncatedCubic => {
                ([0, 1, 2], [[1, 2, 0], [2, 1, 0]], vec![[2, 0, 1], [2, 0, -1], [2, 1, 0], [2, -1, 0]])
            }
            EuclideanKind::RhombicDodecahedral => {
                ([2, 0, 0], [[2, 0, 0], [1, 1, 1]], vec![[2, 0, 0], [0, 2, 0], [1, 1, 1], [1, 1, -1]])
            }
            EuclideanKind::AlternatedCubic => ([2, 0, 0], [[2, 0, 0], [0, 2, 0]], vec![[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        };
        Ok(match seed {
            SeedKind::Vertex => SeedAt::Vertex(v),
            SeedKind::Edge => SeedAt::Edge(key_of(&e)),
            SeedKind::Face => SeedAt::Face(f),
            SeedKind::Cell => SeedAt::Cell([0, 0, 0]),
            SeedKind::Dirichlet => {
                return Err("Dirichlet seed is not offered for E3 honeycombs: the Dirichlet cell of a lattice point \
                            is the cell itself (use cell)"
                    .into())
            }
        })
    }

    /// Vertex, edge and face counts of the cell class `ci` (for checks).
    pub fn class_counts(&self, ci: usize) -> Option<(usize, usize, usize)> {
        let cl = self.classes.get(ci)?;
        Some((cl.offsets.len(), cl.edges.len(), cl.faces.len()))
    }

    /// Vertex counts of the faces of class `ci`, in normal order (for checks).
    pub fn class_face_sizes(&self, ci: usize) -> Vec<usize> {
        self.classes.get(ci).map(|cl| cl.faces.iter().map(Vec::len).collect()).unwrap_or_default()
    }
}

impl CellComplex for E3Complex {
    type Cell = P3;
    type Vertex = P3;
    type Key = E3Key;

    /// Empty for a point that is not a cell centre.
    fn vertices_of(&self, c: P3) -> Vec<P3> {
        self.verts(c).map(|(_, v)| v).unwrap_or_default()
    }
    fn edges_of(&self, c: P3) -> Vec<E3Key> {
        self.verts(c)
            .map(|(ci, v)| self.classes[ci].edges.iter().map(|&(i, j)| key_of(&[v[i], v[j]])).collect())
            .unwrap_or_default()
    }
    fn faces_of(&self, c: P3) -> Vec<E3Key> {
        self.verts(c)
            .map(|(ci, v)| self.classes[ci].faces.iter().map(|f| key_of(&f.iter().map(|&i| v[i]).collect::<Vec<_>>())).collect())
            .unwrap_or_default()
    }
    fn cells_at(&mut self, v: P3) -> Result<Vec<P3>, GrowthError> {
        if !self.is_vertex(v) {
            return Err(GrowthError::OutOfRange { reason: "not a vertex of this honeycomb" });
        }
        let mut out = Vec::new();
        for (ci, cl) in self.classes.iter().enumerate() {
            for o in &cl.offsets {
                let c = sub(v, *o);
                if self.class_of(c) == Some(ci) {
                    out.push(c);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }
    fn cells_on_edge(&mut self, e: E3Key) -> Result<Vec<P3>, GrowthError> {
        if e.n != 2 {
            return Err(GrowthError::OutOfRange { reason: "not an edge key" });
        }
        Ok(self.cells_with(e, false))
    }
    fn cells_on_face(&mut self, f: &[P3]) -> Result<Vec<P3>, GrowthError> {
        if f.len() < 3 {
            return Err(GrowthError::OutOfRange { reason: "a face needs at least 3 vertices" });
        }
        Ok(self.cells_on_face_key(key_of(f)))
    }
}
