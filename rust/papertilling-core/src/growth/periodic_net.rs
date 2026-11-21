//! Dirichlet (Voronoi) cell complex of a 2- or 3-periodic point set, for the touching-rule grower
//! (C1 WP-L1; plan Q2: a cell is the Dirichlet cell of a site, and two cells touch when they share
//! a Voronoi vertex). Mirrored by `papertilling/_periodic_net.py`.
//!
//! Sites: m points (Cartesian) of one cell of the lattice spanned by d = 2 or 3 vectors (for d = 2
//! they lie in the plane z = 0). A cell is (site, n): the site translated by n · lattice.
//!
//! **Vertices.** A Voronoi vertex is keyed by the maximal set T of sites on an empty sphere (no site
//! strictly inside): its Delaunay polytope, translation-normalised ([`delaunay::canonical`]).
//! - [`NetMode::Flat`] (d = 2, one height): the minimal Voronoi faces are vertical lines over the 2D
//!   Voronoi vertices, so T = the cocircular sets (circles through 3 sites).
//! - [`NetMode::Slab`] (d = 2, several heights): finite spheres through 4 non-coplanar sites (every
//!   minimal face is a vertex since the sites span R³); the cells are unbounded above and below.
//! - [`NetMode::Space`] (d = 3): as Slab, with bounded cells.
//!
//! **Search and certificate.** A sphere through site i and `dim` others among its N nearest (N = 20
//! at first) is accepted when empty and when its footprint (its intersection with the slab, or the
//! ball) lies inside the searched neighbourhood of radius R, so every site that could lie inside it
//! was checked. R starts at 2.5 h (h = (cell area or volume / m)^(1/2 or 1/3)); R and N grow by 1.5
//! until the polytopes found (one per translation class) fill the cell: area (Flat), area × thickness
//! (Slab: the hull of a 2-periodic slab is the slab between its lowest and highest sites), volume
//! (Space), to relative 1e-6. [PROVED] The Delaunay polytopes tile the convex hull with disjoint
//! interiors, so a matching total means none is missing; a polytope is the same whichever search
//! found it, so Rust and Python agree whenever both certify. Otherwise the build is refused.
//!
//! **Edges and faces** (boundary counts and χ):
//! - Flat: an edge is two cells sharing 2 vertices (a Voronoi edge); no faces; χ = V − E + F.
//! - Space: an edge is two vertices whose polytopes share ≥ 3 sites (a Voronoi edge); a face is two
//!   cells sharing ≥ 3 vertices (a Voronoi facet); χ = V − E + F − C. [PROVED] two Voronoi cells meet
//!   in a common face, so ≥ 3 shared vertices means a facet; ≥ 3 shared (cospherical, so
//!   non-collinear) sites means a Voronoi edge, which has exactly 2 vertices.
//! - Slab: as Space over bounded faces only. The cells are unbounded, so χ of their union is not
//!   defined: [`PeriodicNet::euler_defined`] is false and callers must not show it.

pub mod delaunay;

use std::collections::{BTreeMap, BTreeSet};

pub use delaunay::{Member, NetMode};

use super::{CellComplex, GrowthError, SeedKind};

/// A cell: (site, lattice shift).
pub type NetCell = Member;
/// A vertex: (polytope class, lattice shift).
pub type NetVertex = Member;
/// An edge (Flat: two cells; Space and Slab: two vertices) or a face (two cells), sorted.
pub type NetKey = (Member, Member);

fn add(x: Member, n: [i64; 3]) -> Member {
    (x.0, [x.1[0] + n[0], x.1[1] + n[1], x.1[2] + n[2]])
}

fn pair(a: Member, b: Member) -> NetKey {
    if a <= b { (a, b) } else { (b, a) }
}

/// The Dirichlet complex (module docs).
pub struct PeriodicNet {
    mode: NetMode,
    polytopes: Vec<Vec<Member>>,
    inc: Vec<Vec<NetVertex>>,
    edge_t: Vec<Vec<NetKey>>,
    face_t: Vec<Vec<NetCell>>,
}

impl PeriodicNet {
    /// Build from d = 2 or 3 lattice vectors and the sites of one cell.
    pub fn new(lattice: &[[f64; 3]], sites: &[[f64; 3]]) -> Result<Self, String> {
        let d = lattice.len();
        if !(2..=3).contains(&d) || sites.is_empty() {
            return Err("need 2 or 3 lattice vectors and at least one site".into());
        }
        let mut l3 = [[0.0; 3]; 3];
        l3[..d].copy_from_slice(lattice);
        let zmin = sites.iter().map(|p| p[2]).fold(f64::MAX, f64::min);
        let zmax = sites.iter().map(|p| p[2]).fold(f64::MIN, f64::max);
        let lmax = lattice.iter().flatten().fold(1.0_f64, |a, x| a.max(x.abs()));
        let mode = if d == 3 {
            NetMode::Space
        } else if zmax - zmin <= 1e-9 * lmax {
            NetMode::Flat
        } else {
            NetMode::Slab
        };
        let polytopes = delaunay::Periodic { lattice: l3, d, sites, mode, zmin, zmax }.polytopes()?;
        let m = sites.len();
        let mut inc: Vec<Vec<NetVertex>> = vec![Vec::new(); m];
        for (u, t) in polytopes.iter().enumerate() {
            for &(s, n) in t {
                inc[s as usize].push((u as u32, [-n[0], -n[1], -n[2]]));
            }
        }
        inc.iter_mut().for_each(|row| row.sort_unstable());
        let mut net = PeriodicNet { mode, polytopes, inc, edge_t: Vec::new(), face_t: Vec::new() };
        for i in 0..m {
            let c0: NetCell = (i as u32, [0; 3]);
            let vs = net.vertices_of(c0);
            let stars: BTreeMap<NetVertex, BTreeSet<NetCell>> =
                vs.iter().map(|&v| (v, net.star(v).into_iter().collect())).collect();
            let mut cnt: BTreeMap<NetCell, usize> = BTreeMap::new();
            for s in stars.values() {
                for &c in s.iter().filter(|&&c| c != c0) {
                    *cnt.entry(c).or_default() += 1;
                }
            }
            if mode == NetMode::Flat {
                net.edge_t.push(cnt.iter().filter(|e| *e.1 >= 2).map(|(&c, _)| pair(c0, c)).collect());
                net.face_t.push(Vec::new());
            } else {
                let mut edges = Vec::new();
                for (a, &va) in vs.iter().enumerate() {
                    for &vb in &vs[a + 1..] {
                        if stars[&va].intersection(&stars[&vb]).count() >= 3 {
                            edges.push((va, vb));
                        }
                    }
                }
                edges.sort_unstable();
                net.edge_t.push(edges);
                net.face_t.push(cnt.iter().filter(|e| *e.1 >= 3).map(|(&c, _)| c).collect());
            }
        }
        Ok(net)
    }

    pub fn mode(&self) -> NetMode {
        self.mode
    }

    /// False for a slab: its cells are unbounded and χ of their union is not defined.
    pub fn euler_defined(&self) -> bool {
        self.mode != NetMode::Slab
    }

    /// The Delaunay polytopes, one per translation class (sorted, translation-normalised).
    pub fn polytopes(&self) -> &[Vec<Member>] {
        &self.polytopes
    }

    fn star(&self, v: NetVertex) -> Vec<NetCell> {
        let mut out: Vec<NetCell> = self.polytopes[v.0 as usize].iter().map(|&t| add(t, v.1)).collect();
        out.sort_unstable();
        out
    }

    /// The seed set of §2.1 at site 0 of cell 0: vertex = the vertex of that cell in most cells (lowest
    /// on ties); edge and face = the first of that cell; cell = that cell. Dirichlet is the cell itself
    /// (a site's Dirichlet cell is its own cell).
    pub fn seed_cells(&self, kind: SeedKind) -> Result<Vec<NetCell>, String> {
        let c0: NetCell = (0, [0; 3]);
        match kind {
            SeedKind::Cell | SeedKind::Dirichlet => Ok(vec![c0]),
            SeedKind::Vertex => {
                let vs = self.vertices_of(c0);
                let mut best = vs[0];
                for &v in &vs {
                    if self.polytopes[v.0 as usize].len() > self.polytopes[best.0 as usize].len() {
                        best = v;
                    }
                }
                Ok(self.star(best))
            }
            SeedKind::Edge => {
                let e = *self.edges_of(c0).first().ok_or("cell 0 has no bounded edge")?;
                Ok(self.edge_cells(e))
            }
            SeedKind::Face => {
                if self.mode == NetMode::Flat {
                    return Err("a face seed needs a 3D Dirichlet complex (in 2D a face is a cell)".into());
                }
                let f = *self.faces_of(c0).first().ok_or("cell 0 has no bounded face")?;
                Ok(vec![f.0, f.1])
            }
        }
    }

    fn edge_cells(&self, e: NetKey) -> Vec<NetCell> {
        if self.mode == NetMode::Flat {
            return vec![e.0, e.1];
        }
        let a: BTreeSet<NetCell> = self.star(e.0).into_iter().collect();
        self.star(e.1).into_iter().filter(|c| a.contains(c)).collect()
    }
}

impl CellComplex for PeriodicNet {
    type Cell = NetCell;
    type Vertex = NetVertex;
    type Key = NetKey;

    fn vertices_of(&self, c: NetCell) -> Vec<NetVertex> {
        let mut out: Vec<NetVertex> = self.inc[c.0 as usize].iter().map(|&v| add(v, c.1)).collect();
        out.sort_unstable();
        out
    }

    fn edges_of(&self, c: NetCell) -> Vec<NetKey> {
        let n = c.1;
        let mut out: Vec<NetKey> = if self.mode == NetMode::Flat {
            self.edge_t[c.0 as usize].iter().map(|&(a, b)| pair(add(a, n), add(b, n))).collect()
        } else {
            self.edge_t[c.0 as usize].iter().map(|&(a, b)| (add(a, n), add(b, n))).collect()
        };
        out.sort_unstable();
        out
    }

    fn faces_of(&self, c: NetCell) -> Vec<NetKey> {
        let mut out: Vec<NetKey> = self.face_t[c.0 as usize].iter().map(|&o| pair(c, add(o, c.1))).collect();
        out.sort_unstable();
        out
    }

    fn cells_at(&mut self, v: NetVertex) -> Result<Vec<NetCell>, GrowthError> {
        Ok(self.star(v))
    }

    fn cells_on_edge(&mut self, e: NetKey) -> Result<Vec<NetCell>, GrowthError> {
        Ok(self.edge_cells(e))
    }

    fn cells_on_face(&mut self, f: &[NetVertex]) -> Result<Vec<NetCell>, GrowthError> {
        let mut it = f.iter();
        let Some(&first) = it.next() else { return Ok(Vec::new()) };
        let mut s: BTreeSet<NetCell> = self.star(first).into_iter().collect();
        for &v in it {
            let t: BTreeSet<NetCell> = self.star(v).into_iter().collect();
            s = s.intersection(&t).copied().collect();
        }
        Ok(s.into_iter().collect())
    }
}
