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

use super::{FaceId, Off, PlaneTiling, VertexId};
use crate::types::FracOp2D;

const SITE_TOL: f64 = 1e-5;

impl PlaneTiling {
    /// Image of vertex `v` under the fractional operation `op`, or `None` if it lands on no site.
    fn vertex_image(&self, op: &FracOp2D, v: VertexId) -> Option<VertexId> {
        let f = self.to_frac(self.vertex_point(v));
        let m = op.mf;
        let y = [
            f64::from(m[0][0]) * f[0] + f64::from(m[0][1]) * f[1] + op.t[0],
            f64::from(m[1][0]) * f[0] + f64::from(m[1][1]) * f[1] + op.t[1],
        ];
        (0..self.basis.len()).find_map(|j| {
            let fj = self.to_frac(self.basis[j]);
            let d = [y[0] - fj[0], y[1] - fj[1]];
            let r: Off = [d[0].round() as i32, d[1].round() as i32];
            ((d[0] - f64::from(r[0])).abs() < SITE_TOL && (d[1] - f64::from(r[1])).abs() < SITE_TOL).then_some((j, r))
        })
    }

    /// Image of face `face` under `op`: the face around the image of its first corner with the same vertex set.
    pub fn face_image(&self, op: &FracOp2D, face: FaceId) -> Option<FaceId> {
        let img: Option<HashSet<VertexId>> = self.face_vertices(face).into_iter().map(|v| self.vertex_image(op, v)).collect();
        let img = img?;
        let w0 = self.vertex_image(op, self.face_vertices(face)[0])?;
        self.vertex_faces(w0)
            .into_iter()
            .find(|&g| self.orbits[g.0].len() == img.len() && self.face_vertices(g).iter().all(|v| img.contains(v)))
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
}
