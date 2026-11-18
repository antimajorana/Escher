//! Dirichlet-cell seed (author decision 2, 2026-11-18): the faces whose centroids lie in the closed Wigner-Seitz
//! (Dirichlet) cell of the lattice about the most symmetric point. A child module of `plane::inflation` (declared
//! there with `#[path]`), so it reads the tiling's private tables.
//!
//! * Centroid of a face = the average of its corners (fixed by every symmetry that fixes the face).
//! * Most symmetric point: among the candidates face-orbit centroids, basis sites and bond midpoints (one cell's
//!   worth, offset `(0, 0)`), the one of largest site-symmetry order = number of coset representatives of the
//!   cell's uncoloured symmetry group modulo translations that fix it modulo the lattice. Ties prefer a face
//!   centroid, then a site, then a bond midpoint; among faces the larger polygon, then the lower orbit index;
//!   among sites and bonds the lower index. The group is that of the given drawing (`symmetry_frac_ops(false)`).
//! * Closed Wigner-Seitz cell about `c`: a face centroid `q` is in it iff `|q - c| <= |q - c - g|` for every lattice
//!   vector `g`, i.e. `q` is a translate of its own orbit's centroid nearest to `c`. Ties (boundary points) are all
//!   included, with one symmetric relative tolerance `1e-6 |b1|^2` on squared distances (`b1` the shortest
//!   lattice vector), so a symmetric tie set is kept or dropped as a whole. Each face orbit has at least one face in
//!   the seed. The lattice is that of the given cell: a non-primitive cell gives the Dirichlet cell of its
//!   superlattice.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{FaceId, Off, PlaneTiling};
use crate::plane::cell::{mat2_inv, mat2_vec, reduce_basis_vectors};

/// What the most symmetric point is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentreKind {
    /// Centroid of face orbit `f` at offset `(0, 0)`.
    Face(usize),
    /// Basis site `s` at offset `(0, 0)`.
    Site(usize),
    /// Midpoint of a bond at basis site `s`: `(s, to_site, to_offset)`.
    BondMidpoint(usize, usize, Off),
}

/// A point of maximal site symmetry and its site-symmetry order.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymmetricPoint {
    pub point: [f64; 2],
    pub order: usize,
    pub kind: CentreKind,
}

/// Tolerance on fractional coordinates when testing whether an operation fixes a point modulo the lattice.
const FIX_TOL: f64 = 1e-5;
/// Relative tolerance on squared distances for Wigner-Seitz boundary ties.
const TIE_TOL: f64 = 1e-6;

impl PlaneTiling {
    fn a_mat(&self) -> [[f64; 2]; 2] {
        [[self.a1[0], self.a2[0]], [self.a1[1], self.a2[1]]]
    }

    /// Fractional coordinates of the Cartesian point `p` in the cell basis `(a1, a2)`.
    pub(super) fn to_frac(&self, p: [f64; 2]) -> [f64; 2] {
        mat2_vec(mat2_inv(self.a_mat()).expect("PlaneTiling::new checked the cell area"), p)
    }

    /// Cartesian centroid (corner average) of face `(f, o)`.
    pub fn centroid(&self, face: FaceId) -> [f64; 2] {
        let poly = self.polygon(face);
        let k = poly.len() as f64;
        [poly.iter().map(|p| p[0]).sum::<f64>() / k, poly.iter().map(|p| p[1]).sum::<f64>() / k]
    }

    /// Site-symmetry order of the Cartesian point `p`: the number of symmetry operations (modulo translations) of
    /// the cell that fix `p` modulo the lattice.
    pub fn site_symmetry_order(&self, p: [f64; 2]) -> usize {
        let x = self.to_frac(p);
        let near_int = |v: f64| (v - v.round()).abs() < FIX_TOL;
        let fixes = |m: &[[i32; 2]; 2], t: [f64; 2]| {
            (0..2).all(|r| near_int(f64::from(m[r][0]) * x[0] + f64::from(m[r][1]) * x[1] + t[r] - x[r]))
        };
        self.ops.iter().filter(|op| fixes(&op.mf, op.t)).count()
    }

    /// The most symmetric point of the tiling (see the module notes for the candidates and the tie-break).
    pub fn most_symmetric_point(&self) -> SymmetricPoint {
        let mut cands: Vec<(usize, SymmetricPoint)> = Vec::new();
        let mut push = |rank: usize, point: [f64; 2], kind: CentreKind| {
            cands.push((rank, SymmetricPoint { point, order: self.site_symmetry_order(point), kind }));
        };
        let mut faces: Vec<usize> = (0..self.orbits.len()).collect();
        faces.sort_by_key(|&f| (std::cmp::Reverse(self.orbits[f].len()), f));
        for f in faces {
            push(0, self.centroid((f, [0, 0])), CentreKind::Face(f));
        }
        for (s, &site) in self.basis.iter().enumerate() {
            push(1, site, CentreKind::Site(s));
        }
        for (s, rot) in self.rotation.iter().enumerate() {
            for c in rot {
                let w = self.vertex_point((c.to_site, c.to_off));
                let mid = [(self.basis[s][0] + w[0]) / 2.0, (self.basis[s][1] + w[1]) / 2.0];
                push(2, mid, CentreKind::BondMidpoint(s, c.to_site, c.to_off));
            }
        }
        // Stable sort: larger order first, then kind rank; insertion order breaks the remaining ties.
        cands.sort_by_key(|(rank, sp)| (std::cmp::Reverse(sp.order), *rank));
        cands[0].1
    }

    /// Cartesian position of vertex `(s, p)`.
    pub(super) fn vertex_point(&self, v: (usize, Off)) -> [f64; 2] {
        let (s, p) = v;
        let at = |ax: usize| self.basis[s][ax] + f64::from(p[0]) * self.a1[ax] + f64::from(p[1]) * self.a2[ax];
        [at(0), at(1)]
    }

    /// Faces whose centroids lie in the closed Wigner-Seitz cell of the lattice about the Cartesian point `c`.
    pub fn dirichlet_faces(&self, c: [f64; 2]) -> HashSet<FaceId> {
        let (b1, b2) = reduce_basis_vectors(self.a1, self.a2);
        // Integer matrix U with (b1, b2) = (a1, a2) U.
        let u: Vec<[i32; 2]> = [b1, b2].iter().map(|&b| self.to_frac(b).map(|v| v.round() as i32)).collect();
        let b_inv = mat2_inv([[b1[0], b2[0]], [b1[1], b2[1]]]).expect("reduced basis of a non-degenerate cell");
        let tie = TIE_TOL * (b1[0] * b1[0] + b1[1] * b1[1]);
        let mut out = HashSet::new();
        for f in 0..self.orbits.len() {
            let q0 = self.centroid((f, [0, 0]));
            let y = mat2_vec(b_inv, [c[0] - q0[0], c[1] - q0[1]]);
            let (k0, k1) = (y[0].round() as i32, y[1].round() as i32);
            let mut cands: Vec<(f64, Off)> = Vec::new();
            for i in k0 - 3..=k0 + 3 {
                for j in k1 - 3..=k1 + 3 {
                    let g = [f64::from(i) * b1[0] + f64::from(j) * b2[0], f64::from(i) * b1[1] + f64::from(j) * b2[1]];
                    let d = [q0[0] + g[0] - c[0], q0[1] + g[1] - c[1]];
                    let off = [i * u[0][0] + j * u[1][0], i * u[0][1] + j * u[1][1]];
                    cands.push((d[0] * d[0] + d[1] * d[1], off));
                }
            }
            let best = cands.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
            out.extend(cands.into_iter().filter(|x| x.0 <= best + tie).map(|x| (f, x.1)));
        }
        out
    }
}
