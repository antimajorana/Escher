//! Euclidean 2D patch inflation by simply-connected chamber vertex completion.
//!
//! Boyle, Dickens & Flicker, "Conformal Quasicrystals and Holography", PRX 10, 011009 (2020), grow a patch by
//! vertex completion `S_{n+1} = S_n U {faces sharing >= 1 vertex with S_n}`. Here every shell is also closed
//! under FillHoles: `S~_{n+1} = FillHoles(S~_n U U(S~_n))`, so that each shell is a closed topological disk.
//!
//! Everything is exact and combinatorial. A face of the infinite periodic tiling is `(face orbit, cell offset of
//! its first corner)` with the orbits enumerated by `surface::quotient::plane_faces`; a vertex is `(site,
//! offset)`. Floats enter only through the cell geometry of `plane_faces` (angular order at a site), the
//! planarity and signed-area checks in [`PlaneTiling::new`], the generous FillHoles box margin, and the polygons
//! for drawing. [`PatchTopology`] and the incremental shell bookkeeping live in `inflation_topology.rs`.
//!
//! FillHoles uses SHARED-EDGE adjacency between non-patch faces: the open edge between two non-patch faces lies
//! outside the closed patch, so edge-adjacency components are exactly the connected components of the
//! complement `R^2 \ |S|`. Two non-patch faces meeting only at a patch vertex are separated by that vertex, so
//! vertex adjacency would leave pinched pockets (a ring whose ends touch at one vertex) unfilled.
//!
//! [`inflate`] / [`inflate_with`] err when `S~_0` has `chi != 1` or a pinch vertex. Each step works only on the
//! faces the previous step added (completion, FillHoles starts, topology update). Child modules: the growth rules
//! and results (`inflation_grow.rs`), the Dirichlet seed (`inflation_dirichlet.rs`) and tile classes / symmetry
//! (`inflation_symmetry.rs`), and the torus mode (`inflation_torus.rs`).

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::plane::tuning;
use crate::surface::quotient::{CellGeom, Dart2D, plane_faces};
use crate::types::{Bond2D, Cell2D, FracOp2D};

#[path = "inflation_topology.rs"]
mod topology;
pub use topology::PatchTopology;
use topology::Tracker;
#[path = "inflation_grow.rs"]
mod grow;
pub use grow::{CentreSet, InflationFace, InflationResult, InflationRule, ShellStats, inflate, inflate_with};
#[path = "inflation_dirichlet.rs"]
mod dirichlet;
pub use dirichlet::{CentreKind, SymmetricPoint};
#[path = "inflation_symmetry.rs"]
mod symmetry;
#[path = "inflation_torus.rs"]
mod torus;
pub use torus::{TorusResult, TorusShell};

/// Integer cell offset `(n1, n2)` in units of `(a1, a2)`.
pub type Off = [i32; 2];
/// A face of the infinite tiling: `(face orbit, cell offset of the orbit's first corner)`.
pub type FaceId = (usize, Off);
/// A vertex of the infinite tiling: `(basis site, cell offset)`.
pub type VertexId = (usize, Off);

/// Largest number of inflation steps accepted by [`inflate`].
pub const MAX_STEPS: usize = 64;

fn add(a: Off, b: Off) -> Off { [a[0] + b[0], a[1] + b[1]] }
fn sub(a: Off, b: Off) -> Off { [a[0] - b[0], a[1] - b[1]] }

/// Initial face set (before FillHoles).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum InflationSeed {
    /// The single face of this orbit at offset `(0, 0)`.
    Face(usize),
    /// All faces around basis site `i` at offset `(0, 0)`.
    VertexStar(usize),
    /// Faces whose centroids lie in the closed Wigner-Seitz cell of the lattice about the most symmetric point
    /// ([`PlaneTiling::most_symmetric_point`]), boundary ties included.
    Dirichlet,
    /// Faces whose centroids lie in the closed Wigner-Seitz cell about this Cartesian point.
    DirichletAt([f64; 2]),
}

/// One corner of a face orbit: the corner's site, its offset relative to the first corner, and the face across
/// the edge that leaves this corner (orbit, offset shift relative to this face).
#[derive(Clone, Copy, Debug)]
struct Corner {
    site: usize,
    rel: Off,
    across: (usize, Off),
}

/// One face corner around a site, in rotation order: the face containing vertex `(site, p)` at this corner is
/// `(orbit, p + shift)`, and the bond leaving along this corner ends at `(to_site, p + to_off)`.
#[derive(Clone, Copy, Debug)]
struct SiteCorner {
    orbit: usize,
    shift: Off,
    to_site: usize,
    to_off: Off,
}

/// Exact combinatorial description of the periodic plane tiling of a `Cell2D`.
#[derive(Clone, Debug)]
pub struct PlaneTiling {
    a1: [f64; 2],
    a2: [f64; 2],
    basis: Vec<[f64; 2]>,
    orbits: Vec<Vec<Corner>>,
    rotation: Vec<Vec<SiteCorner>>,
    pad: Off,
    /// Coset representatives `x_frac -> mf x_frac + t` of the cell's uncoloured symmetry group modulo translations.
    ops: Vec<FracOp2D>,
}

impl PlaneTiling {
    /// Build the face orbits, cross-edge neighbours and per-site rotation from `cell`. Errs unless the face cycles
    /// form a cellular torus map (`V - E + F = 0` per cell), the straight bonds are a plane embedding (no two
    /// periodic bond images cross, no site lies on a bond it is not an end of), and the orbit polygons tile the
    /// cell with one orientation.
    ///
    /// The Euler and area checks alone are not enough: a cell with one face orbit has summed signed area
    /// `degree x det` whatever the drawing, so a crossing drawing of a valid torus map can pass both. The
    /// planarity check is the core's criterion `!has_crossings && clearance > 1e-6` (as in
    /// `custom_builder`). [`tuning::has_crossings`] is exhaustive: for each bond pair it scans every integer
    /// translate whose fractional bounding box meets the other bond's, and a proper crossing needs that.
    /// [`tuning::clearance`] scans site translates within 2 cells of each bond's offset range. Both checks run on
    /// [`wrapped`] (the same drawing, every site in `[0, 1)^2`), where that covers every site on a bond: a site image
    /// `f_k + s` in the fractional box of bond `(i, j, n)` then has `min(0, n) <= s <= max(0, n)` per axis.
    pub fn new(cell: &Cell2D) -> Result<Self, String> {
        let geom = CellGeom::from_cell(cell)?;
        let (faces, _deg) = plane_faces(cell, &geom)?;
        let (nv, ne, nf) = (cell.basis.len(), cell.bonds.len(), faces.len());
        if nv + nf != ne {
            return Err(format!("crossing bonds: V - E + F = {nv} - {ne} + {nf} != 0 per cell (not a torus map)"));
        }
        let unit = wrapped(cell, &geom);
        if tuning::has_crossings(&unit) {
            return Err("crossing bonds: two periodic bond images cross properly".to_string());
        }
        let clearance = tuning::clearance(&unit);
        if clearance <= 1e-6 {
            return Err(format!("crossing bonds: a site lies on a bond it is not an end of (clearance {clearance:e})"));
        }
        let pos: HashMap<Dart2D, (usize, usize)> =
            faces.iter().enumerate().flat_map(|(f, cyc)| cyc.iter().enumerate().map(move |(k, &d)| (d, (f, k)))).collect();
        let rev = |d: Dart2D| (d.1, d.0, -d.2, -d.3);
        let mut rels: Vec<Vec<Off>> = Vec::with_capacity(faces.len());
        for (f, cyc) in faces.iter().enumerate() {
            let mut r = [0, 0];
            let mut v = Vec::with_capacity(cyc.len());
            for &(_, _, n1, n2) in cyc {
                v.push(r);
                r = add(r, [n1, n2]);
            }
            if r != [0, 0] {
                return Err(format!("face orbit {f} does not close in the plane (net offset {r:?})"));
            }
            rels.push(v);
        }
        let mut orbits = Vec::with_capacity(faces.len());
        for (f, cyc) in faces.iter().enumerate() {
            let mut corners = Vec::with_capacity(cyc.len());
            for (k, &d) in cyc.iter().enumerate() {
                let (g, l) = *pos.get(&rev(d)).ok_or("reverse dart missing from the face cycles")?;
                // Dart k of (f, o) ends at (j, o + r_k + n); dart l of (g, o') starts there.
                let across = sub(add(rels[f][k], [d.2, d.3]), rels[g][l]);
                corners.push(Corner { site: d.0, rel: rels[f][k], across: (g, across) });
            }
            orbits.push(corners);
        }
        let mut rotation = vec![Vec::new(); cell.basis.len()];
        for (s, rot) in rotation.iter_mut().enumerate() {
            let out: Vec<Dart2D> = pos.keys().copied().filter(|d| d.0 == s).collect();
            let Some(&x0) = out.iter().min() else {
                return Err(format!("site {s} has no incident bonds"));
            };
            let mut x = x0;
            loop {
                let (g, l) = pos[&x];
                rot.push(SiteCorner { orbit: g, shift: sub([0, 0], rels[g][l]), to_site: x.1, to_off: [x.2, x.3] });
                // Rotation successor of outgoing dart x: the dart that follows rev(x) in its face cycle.
                let (h, m) = pos[&rev(x)];
                x = faces[h][(m + 1) % faces[h].len()];
                if x == x0 {
                    break;
                }
                if rot.len() > out.len() {
                    return Err(format!("rotation at site {s} does not close"));
                }
            }
            if rot.len() != out.len() {
                return Err(format!("rotation at site {s} visits {} of {} bonds", rot.len(), out.len()));
            }
        }
        // FillHoles box margin: spread of corner fractional positions frac(site) + rel over all orbits.
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (c, ax) in orbits.iter().flatten().flat_map(|c| [(c, 0), (c, 1)]) {
            let x = geom.frac_basis[c.site][ax] + c.rel[ax] as f64;
            (lo[ax], hi[ax]) = (lo[ax].min(x), hi[ax].max(x));
        }
        let pad = [(hi[0] - lo[0]).ceil() as i32 + 1, (hi[1] - lo[1]).ceil() as i32 + 1];
        let ops = cell.symmetry_frac_ops(false);
        let t = Self { a1: cell.a1, a2: cell.a2, basis: cell.basis.clone(), orbits, rotation, pad, ops };
        let det = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs();
        let areas: Vec<f64> = (0..nf).map(|g| signed_area(&t.polygon((g, [0, 0])))).collect();
        // Backstop for multi-orbit cells; after the planarity check it should not fire (each face is a simple
        // polygon of a plane embedding, and the faces of one cell tile it once).
        let one_sign = areas.iter().all(|&a| a > 0.0) || areas.iter().all(|&a| a < 0.0);
        if !one_sign || (areas.iter().map(|a| a.abs()).sum::<f64>() - det).abs() > 1e-6 * det {
            return Err(format!("crossing bonds: face areas {areas:?} do not tile the cell (area {det})"));
        }
        Ok(t)
    }

    /// Number of face orbits (faces per primitive cell).
    pub fn orbit_count(&self) -> usize {
        self.orbits.len()
    }

    /// Number of sides of each face orbit.
    pub fn face_sizes(&self) -> Vec<usize> {
        self.orbits.iter().map(Vec::len).collect()
    }

    /// Vertices of face `(f, o)` in cyclic order.
    pub fn face_vertices(&self, face: FaceId) -> Vec<VertexId> {
        self.orbits[face.0].iter().map(|c| (c.site, add(face.1, c.rel))).collect()
    }

    /// Faces around vertex `(s, p)` in rotation order (a face appears once per corner it has there).
    pub fn vertex_faces(&self, v: VertexId) -> Vec<FaceId> {
        self.rotation[v.0].iter().map(|c| (c.orbit, add(v.1, c.shift))).collect()
    }

    /// Cartesian polygon of face `(f, o)`.
    pub fn polygon(&self, face: FaceId) -> Vec<[f64; 2]> {
        let at = |s: usize, p: Off, ax: usize| self.basis[s][ax] + p[0] as f64 * self.a1[ax] + p[1] as f64 * self.a2[ax];
        self.face_vertices(face).into_iter().map(|(s, p)| [at(s, p, 0), at(s, p, 1)]).collect()
    }

    /// Faces of the initial patch for `seed`, before FillHoles.
    pub fn seed_faces(&self, seed: InflationSeed) -> Result<HashSet<FaceId>, String> {
        let star = |s: usize| self.vertex_faces((s, [0, 0]));
        match seed {
            InflationSeed::Face(f) if f < self.orbits.len() => Ok(HashSet::from([(f, [0, 0])])),
            InflationSeed::Face(f) => Err(format!("face orbit {f} out of range ({} orbits)", self.orbits.len())),
            InflationSeed::VertexStar(s) if s < self.rotation.len() => Ok(star(s).into_iter().collect()),
            InflationSeed::VertexStar(s) => Err(format!("site {s} out of range ({} sites)", self.rotation.len())),
            InflationSeed::Dirichlet => Ok(self.dirichlet_faces(self.most_symmetric_point().point)),
            InflationSeed::DirichletAt(c) => Ok(self.dirichlet_faces(c)),
        }
    }

    /// Vertex completion `U(S)`: every face not in `patch` that shares at least one vertex with it.
    pub fn completion(&self, patch: &HashSet<FaceId>) -> BTreeSet<FaceId> {
        self.completion_from(patch, patch)
    }

    /// Faces not in `patch` sharing a vertex with a `frontier` face. For `patch = S~_n` and `frontier` the faces
    /// added at step `n` this is all of `U(S~_n)`: every face around a vertex of `S~_{n-1}` is already in `S~_n`.
    fn completion_from<'a>(&self, patch: &HashSet<FaceId>, frontier: impl IntoIterator<Item = &'a FaceId>) -> BTreeSet<FaceId> {
        let mut out = BTreeSet::new();
        for &(f, o) in frontier {
            for c in &self.orbits[f] {
                out.extend(self.vertex_faces((c.site, add(o, c.rel))).into_iter().filter(|g| !patch.contains(g)));
            }
        }
        out
    }

    fn across(&self, face: FaceId) -> impl Iterator<Item = FaceId> + '_ {
        self.orbits[face.0].iter().map(move |c| (c.across.0, add(face.1, c.across.1)))
    }

    /// Fill every bounded component of the complement of `patch` (shared-edge adjacency); returns the faces added.
    pub fn fill_holes(&self, patch: &mut HashSet<FaceId>) -> BTreeSet<FaceId> {
        if patch.is_empty() {
            return BTreeSet::new();
        }
        let all: Vec<FaceId> = patch.iter().copied().collect();
        let mut bbox = EMPTY_BOX;
        grow_box(&mut bbox, &all);
        self.fill_holes_from(patch, &all, bbox)
    }

    /// FillHoles started only from complement faces edge-adjacent to `frontier`; `bbox` bounds the patch anchors.
    /// Exact when the patch minus `frontier` was hole-free: a bounded complement component touching no frontier
    /// face would have been a hole of that smaller patch.
    ///
    /// A hole face lies in the convex hull of the patch, hence in the box of patch-corner fractional positions, so
    /// its anchor offset is within `pad` of the box of patch anchors. A complement component that reaches an
    /// anchor outside that padded box, or a face already known to be outer, is the unbounded component.
    fn fill_holes_from(&self, patch: &mut HashSet<FaceId>, frontier: &[FaceId], bbox: (Off, Off)) -> BTreeSet<FaceId> {
        let mut added = BTreeSet::new();
        let (lo, hi) = (sub(bbox.0, self.pad), add(bbox.1, self.pad));
        let outside = |o: Off| (0..2).any(|ax| o[ax] < lo[ax] || o[ax] > hi[ax]);
        let mut outer: HashSet<FaceId> = HashSet::new();
        let starts: BTreeSet<FaceId> =
            frontier.iter().flat_map(|&f| self.across(f)).filter(|g| !patch.contains(g)).collect();
        for start in starts {
            if outer.contains(&start) || added.contains(&start) {
                continue;
            }
            let (mut comp, mut seen, mut head, mut is_outer) = (vec![start], HashSet::from([start]), 0, false);
            while head < comp.len() && !is_outer {
                let face = comp[head];
                head += 1;
                is_outer = outside(face.1);
                for g in self.across(face).filter(|g| !patch.contains(g)) {
                    is_outer |= outer.contains(&g);
                    if seen.insert(g) {
                        comp.push(g);
                    }
                }
            }
            if is_outer { outer.extend(comp) } else { added.extend(comp) }
        }
        patch.extend(added.iter().copied());
        added
    }
}

/// Copy of `cell` with site `k` moved by `-fl_k` cells, `fl_k = floor(frac_k)`, into `[0, 1)^2`, and each bond
/// `(i, j, n)` given offset `n + fl_j - fl_i`: every bond keeps its vector, so the periodic drawing is unchanged.
fn wrapped(cell: &Cell2D, geom: &CellGeom) -> Cell2D {
    let fl: Vec<Off> = geom.frac_basis.iter().map(|f| [f[0].floor() as i32, f[1].floor() as i32]).collect();
    let at = |p: [f64; 2], k: Off, ax: usize| p[ax] - k[0] as f64 * cell.a1[ax] - k[1] as f64 * cell.a2[ax];
    let basis = cell.basis.iter().zip(&fl).map(|(&p, &k)| [at(p, k, 0), at(p, k, 1)]).collect();
    let shift = |b: &Bond2D| Bond2D { n1: b.n1 + fl[b.j][0] - fl[b.i][0], n2: b.n2 + fl[b.j][1] - fl[b.i][1], ..*b };
    Cell2D { a1: cell.a1, a2: cell.a2, basis, bonds: cell.bonds.iter().map(shift).collect() }
}

/// `(min, max)` box of face anchors containing nothing; grow it before use.
const EMPTY_BOX: (Off, Off) = ([i32::MAX; 2], [i32::MIN; 2]);

fn grow_box(b: &mut (Off, Off), faces: &[FaceId]) {
    for &(_, o) in faces {
        *b = ([b.0[0].min(o[0]), b.0[1].min(o[1])], [b.1[0].max(o[0]), b.1[1].max(o[1])]);
    }
}

fn signed_area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    0.5 * (0..n).map(|k| poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1]).sum::<f64>()
}
