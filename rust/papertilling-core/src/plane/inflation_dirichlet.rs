//! Dirichlet-cell seed (author decision 2, 2026-11-18): the faces whose centroids lie in the closed Wigner-Seitz
//! (Dirichlet) cell of the lattice about the most symmetric point, minimally repaired to a closed disk when the
//! Wigner-Seitz cell has holes/disconnects or pinch vertices. A child module of `plane::inflation` (declared there
//! with `#[path]`), so it reads the tiling's private tables.
//!
//! * Centroid of a face = the average of its corners (fixed by every symmetry that fixes the face).
//! * Most symmetric point: among the candidates face-orbit centroids, basis sites and bond midpoints (one cell's
//!   worth, offset `(0, 0)`), the one of largest site-symmetry order = number of coset representatives of the
//!   cell's uncoloured symmetry group modulo translations that fix it modulo the lattice (`FIX_TOL = 1e-5`). Ties
//!   prefer a face centroid, then a site, then a bond midpoint; among faces the larger polygon, then the lower orbit
//!   index; among sites and bonds the lower index.
//! * Group & supercells: the group comes from `symmetry_frac_ops(false)` on the entered cell's drawing modulo its own
//!   lattice `(a1, a2)`, and the Wigner-Seitz cell is taken with respect to `(a1, a2)`. Consequently, a non-primitive
//!   or elongated supercell drops point symmetries that do not map `(a1, a2)` to itself and gives the superlattice
//!   Dirichlet cell.
//! * Closed Wigner-Seitz cell about `c`: a face centroid `q` is in it iff `|q - c| <= |q - c - g|` for every lattice
//!   vector `g`, i.e. `q` is a translate of its own orbit's centroid nearest to `c`. Ties (boundary points) are all
//!   included, with symmetric tolerance `TIE_TOL * |b1|^2 = 1e-6 |b1|^2` on squared distances (`b1` the shortest
//!   lattice vector). When `B_0 = FillHoles(D_0)` is disconnected (`chi != 1`) or has pinch vertices,
//!   `dirichlet_faces_repaired` finds the fewest-tile `FillHoles`-closed union of `G_c`-orbits forming a disk, keeping
//!   the full point-symmetry stabilizer `G_c` about `c`.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashSet};

use serde::{Deserialize, Serialize};

use super::{FaceId, Off, PlaneTiling, SeedDetail};
use crate::plane::cell::{mat2_inv, mat2_vec, reduce_basis_vectors};

type AffineOp = ([[i32; 2]; 2], [f64; 2]);

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
        self.stabilizer(p).len()
    }

    /// Exact affine fractional symmetries `(mf, t - s)` fixing the Cartesian point `c` (without lattice wrap).
    fn stabilizer(&self, c: [f64; 2]) -> Vec<AffineOp> {
        let x = self.to_frac(c);
        self.ops
            .iter()
            .filter_map(|op| {
                let mut s = [0.0; 2];
                for (r, sr) in s.iter_mut().enumerate() {
                    let d = f64::from(op.mf[r][0]) * x[0] + f64::from(op.mf[r][1]) * x[1] + op.t[r] - x[r];
                    let rv = d.round();
                    if (d - rv).abs() >= FIX_TOL {
                        return None;
                    }
                    *sr = rv;
                }
                Some((op.mf, [op.t[0] - s[0], op.t[1] - s[1]]))
            })
            .collect()
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

    /// Orbit of `face` under the point-symmetry stabilizer `stab` about `c`.
    fn stabilizer_orbit(&self, stab: &[AffineOp], face: FaceId) -> Result<BTreeSet<FaceId>, String> {
        let mut orb = BTreeSet::from([face]);
        for &(mf, t) in stab {
            let g = self
                .affine_face_image(mf, t, face)
                .ok_or_else(|| format!("point symmetry mapped face {face:?} outside tiling"))?;
            orb.insert(g);
        }
        Ok(orb)
    }

    fn candidate_orbits(
        &self,
        pool: &BTreeSet<FaceId>,
        stab: &[AffineOp],
        base: &BTreeSet<FaceId>,
    ) -> Result<Vec<BTreeSet<FaceId>>, String> {
        let mut seen = HashSet::new();
        let mut orbits = Vec::new();
        for &f in pool {
            let orb = self.stabilizer_orbit(stab, f)?;
            if !orb.is_subset(base) && seen.insert(orb.clone()) {
                orbits.push(orb);
            }
        }
        Ok(orbits)
    }

    /// Best-first Dijkstra over `FillHoles`-closed unions of `G_c`-orbits: the state is the patch set alone,
    /// deduplicated by set and ordered by `|set| - |b0|`, so the first disk popped is a true minimum-tile repair.
    fn search_repair(&self, b0: &BTreeSet<FaceId>, orbits: &[BTreeSet<FaceId>], cap: usize) -> Option<HashSet<FaceId>> {
        let mut heap: BinaryHeap<Reverse<(usize, BTreeSet<FaceId>)>> = BinaryHeap::new();
        let mut seen: HashSet<BTreeSet<FaceId>> = HashSet::from([b0.clone()]);
        heap.push(Reverse((0, b0.clone())));
        let mut expanded = 0usize;
        while let Some(Reverse((_, cur))) = heap.pop() {
            let hs: HashSet<FaceId> = cur.iter().copied().collect();
            let top = self.topology(&hs);
            if top.chi == 1 && top.pinch_vertices == 0 {
                return Some(hs);
            }
            expanded += 1;
            if expanded >= cap {
                return None;
            }
            for orb in orbits {
                if orb.is_subset(&cur) {
                    continue;
                }
                let mut next = hs.clone();
                next.extend(orb.iter().copied());
                self.fill_holes(&mut next);
                let key: BTreeSet<FaceId> = next.into_iter().collect();
                if seen.insert(key.clone()) {
                    heap.push(Reverse((key.len().saturating_sub(b0.len()), key)));
                }
            }
        }
        None
    }

    /// Dirichlet faces about `c`, repaired minimally while keeping the point symmetry `G_c` if `FillHoles(D_0)`
    /// is disconnected or has a pinch vertex. Returns `(faces, Some((added_beyond_b0, order)))` when repaired.
    pub(super) fn dirichlet_faces_repaired(&self, c: [f64; 2]) -> Result<SeedDetail, String> {
        let d0 = self.dirichlet_faces(c);
        let mut b0_hs = d0.clone();
        self.fill_holes(&mut b0_hs);
        let top = self.topology(&b0_hs);
        if top.chi == 1 && top.pinch_vertices == 0 {
            return Ok((d0, None));
        }
        let stab = self.stabilizer(c);
        let ord = stab.len().max(1);
        let b0: BTreeSet<FaceId> = b0_hs.iter().copied().collect();
        let u1 = self.completion(&b0_hs);
        let orb1 = self.candidate_orbits(&u1, &stab, &b0)?;
        if let Some(rep) = self.search_repair(&b0, &orb1, 3000) {
            let added = rep.len().saturating_sub(b0.len());
            return Ok((rep, Some((added, ord))));
        }
        let mut wide = b0_hs;
        wide.extend(u1.iter().copied());
        let mut u2 = u1;
        u2.extend(self.completion(&wide));
        let orb2 = self.candidate_orbits(&u2, &stab, &b0)?;
        if let Some(rep) = self.search_repair(&b0, &orb2, 5000) {
            let added = rep.len().saturating_sub(b0.len());
            return Ok((rep, Some((added, ord))));
        }
        Err(format!(
            "Dirichlet seed is not a disk (chi = {}, pinch_vertices = {}) and could not be repaired within two vertex-completion shells",
            top.chi, top.pinch_vertices
        ))
    }
}
