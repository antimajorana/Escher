//! Tile classes: the orbits of the faces under the cell's uncoloured symmetry group `G` (`symmetry_frac_ops(false)`,
//! the group `Cell2D::dsymbol` is built from). These are the D-symbol tile classes: the tiles of a periodic tiling
//! correspond to the `<s0, s1>`-orbits of its Delaney-Dress symbol, one per `G`-orbit of tiles. A child module of
//! `plane::inflation`.
//!
//! What is missing for a true D-symbol selector: `dsymbol_of_cell` (in `cell.rs`) keeps its chamber-to-dart and
//! dart-to-face maps local, so the index of an `<s0, s1>`-orbit in the D-symbol cannot be matched to a face orbit
//! here. The classes below are numbered by their lowest face orbit instead; their count equals the number of
//! `<s0, s1>`-orbits (checked in the tests on all named lattices).

use std::collections::HashSet;

use super::{FaceId, InflationSeed, Off, PlaneTiling, VertexId, add};
use crate::types::FracOp2D;

const SITE_TOL: f64 = 1e-5;

impl PlaneTiling {
    /// Image of vertex `v` under the affine fractional operation `(mf, t)`, or `None` if it lands on no site.
    fn vertex_image_affine(&self, mf: [[i32; 2]; 2], t: [f64; 2], v: VertexId) -> Option<VertexId> {
        let f = self.to_frac(self.vertex_point(v));
        let y = [
            f64::from(mf[0][0]) * f[0] + f64::from(mf[0][1]) * f[1] + t[0],
            f64::from(mf[1][0]) * f[0] + f64::from(mf[1][1]) * f[1] + t[1],
        ];
        (0..self.basis.len()).find_map(|j| {
            let fj = self.to_frac(self.basis[j]);
            let d = [y[0] - fj[0], y[1] - fj[1]];
            let r: Off = [d[0].round() as i32, d[1].round() as i32];
            ((d[0] - f64::from(r[0])).abs() < SITE_TOL && (d[1] - f64::from(r[1])).abs() < SITE_TOL).then_some((j, r))
        })
    }

    /// Image of face `face` under the exact affine fractional symmetry `(mf, t)`.
    pub(super) fn affine_face_image(&self, mf: [[i32; 2]; 2], t: [f64; 2], face: FaceId) -> Option<FaceId> {
        let img: Option<HashSet<VertexId>> =
            self.face_vertices(face).into_iter().map(|v| self.vertex_image_affine(mf, t, v)).collect();
        let img = img?;
        let w0 = self.vertex_image_affine(mf, t, self.face_vertices(face)[0])?;
        self.vertex_faces(w0)
            .into_iter()
            .find(|&g| self.orbits[g.0].len() == img.len() && self.face_vertices(g).iter().all(|v| img.contains(v)))
    }

    /// Image of face `face` under `op`: the face around the image of its first corner with the same vertex set.
    pub fn face_image(&self, op: &FracOp2D, face: FaceId) -> Option<FaceId> {
        self.affine_face_image(op.mf, op.t, face)
    }

    /// Tile class of every face orbit: `class[f]` is the index of the `G`-orbit of face orbit `f`, classes numbered
    /// in order of their lowest face orbit.
    pub fn tile_classes(&self) -> Vec<usize> {
        let n = self.orbits.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for op in &self.ops {
            for f in 0..n {
                if let Some((g, _)) = self.face_image(op, (f, [0, 0])) {
                    let (a, b) = (root(&mut parent, f), root(&mut parent, g));
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        let mut class = vec![usize::MAX; n];
        let mut next = 0;
        for f in 0..n {
            let r = root(&mut parent, f);
            if class[r] == usize::MAX {
                class[r] = next;
                next += 1;
            }
            class[f] = class[r];
        }
        class
    }

    /// Number of coset representatives of the symmetry group modulo translations held by the tiling.
    pub fn symmetry_op_count(&self) -> usize {
        self.ops.len()
    }

    /// The most symmetric vertex orbit (plan Q3, the default seed): the basis site with the largest site-symmetry
    /// order ([`PlaneTiling::site_symmetry_order`], symmetries modulo translations fixing the site modulo the
    /// lattice), lowest site index on ties. Returns `(site, order)`.
    pub fn most_symmetric_vertex(&self) -> (usize, usize) {
        let mut best = (0, 0);
        for (s, &p) in self.basis.iter().enumerate() {
            let order = self.site_symmetry_order(p);
            if order > best.1 {
                best = (s, order);
            }
        }
        best
    }

    /// The default seed of the touching rule (D19, Q3): the vertex star of [`PlaneTiling::most_symmetric_vertex`].
    pub fn default_seed(&self) -> InflationSeed {
        InflationSeed::VertexStar(self.most_symmetric_vertex().0)
    }

    /// The faces on the bond `u`-`w`: the face left of the dart `u -> w` and the face left of `w -> u` (one face
    /// when both are the same, as on a dangling bond). Errs unless exactly one bond joins `u` and `w`.
    pub(super) fn edge_faces(&self, u: VertexId, w: VertexId) -> Result<HashSet<FaceId>, String> {
        let side = |a: VertexId, b: VertexId| -> Result<FaceId, String> {
            let rot = self.rotation.get(a.0).ok_or_else(|| format!("site {} out of range ({} sites)", a.0, self.rotation.len()))?;
            let hits: Vec<FaceId> =
                rot.iter().filter(|c| (c.to_site, add(a.1, c.to_off)) == b).map(|c| (c.orbit, add(a.1, c.shift))).collect();
            match hits.as_slice() {
                [f] => Ok(*f),
                [] => Err(format!("edge seed: vertices {u:?} and {w:?} are not joined by a bond")),
                _ => Err(format!("edge seed: {} bonds join {u:?} and {w:?}; the edge is ambiguous", hits.len())),
            }
        };
        Ok(HashSet::from([side(u, w)?, side(w, u)?]))
    }
}
