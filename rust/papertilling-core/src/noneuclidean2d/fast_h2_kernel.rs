//! `O(N)` hyperbolic tilings scaling past `10^5` faces/vertices, neighbor queries, and dynamic edits.

use std::collections::HashSet;

use super::h2_builders::{build_dunham_hyperboloid, build_spatial_hash_poincare};
use super::h2_comb_builder::build_graph_combinatorial;
use super::h2_spatial_hash::{
    H2Kernel, H2LargeSummary, PoincarePolarHash, RegionFilter, poincare_to_su_trans, su_mul, su_origin,
};
use super::models_surfaces::{geodesic_arc_poincare, hyp_distance_poincare};

#[derive(Clone, Debug)]
pub struct H2LargePatch {
    pub p: usize,
    pub q: usize,
    pub kernel: H2Kernel,
    pub faces: Vec<Vec<usize>>,
    pub face_shell: Vec<usize>,
    pub face_centres_poincare: Vec<[f64; 2]>,
    pub poincare_pts: Vec<[f64; 2]>,
    pub face_neighbors: Vec<Vec<Option<usize>>>,
    pub vertex_faces: Vec<Vec<usize>>,
    pub active: Vec<bool>,
    pub n_active_faces: usize,
    pub n_vertices: usize,
    pub n_edges: usize,
    face_hash: PoincarePolarHash,
}

impl H2LargePatch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        p: usize,
        q: usize,
        kernel: H2Kernel,
        faces: Vec<Vec<usize>>,
        face_shell: Vec<usize>,
        face_centres_poincare: Vec<[f64; 2]>,
        poincare_pts: Vec<[f64; 2]>,
        face_neighbors: Vec<Vec<Option<usize>>>,
        vertex_faces: Vec<Vec<usize>>,
        n_edges: usize,
        face_hash: PoincarePolarHash,
    ) -> Self {
        let nf = faces.len();
        let nv = poincare_pts.len();
        Self {
            p,
            q,
            kernel,
            faces,
            face_shell,
            face_centres_poincare,
            poincare_pts,
            face_neighbors,
            vertex_faces,
            active: vec![true; nf],
            n_active_faces: nf,
            n_vertices: nv,
            n_edges,
            face_hash,
        }
    }

    pub fn summary(&self) -> H2LargeSummary {
        H2LargeSummary {
            p: self.p,
            q: self.q,
            kernel: self.kernel,
            total_faces: self.faces.len(),
            n_active_faces: self.n_active_faces,
            n_vertices: self.n_vertices,
            n_edges: self.n_edges,
            max_shell: self.face_shell.iter().copied().max().unwrap_or(0),
        }
    }

    pub fn hyperbolic_ball_query(&self, center_poincare: [f64; 2], radius: f64) -> Result<Vec<usize>, String> {
        let c_su = poincare_to_su_trans(center_poincare)
            .ok_or_else(|| format!("ball center {center_poincare:?} outside Poincare disk"))?;
        let mut hits: Vec<usize> = self
            .face_hash
            .ball_query(c_su, radius.max(0.0))
            .into_iter()
            .filter(|&fid| self.active.get(fid).copied().unwrap_or(false))
            .collect();
        hits.sort_unstable();
        Ok(hits)
    }

    pub fn edge_neighbors(&self, face_id: usize) -> Vec<usize> {
        if face_id >= self.faces.len() || !self.active[face_id] {
            return Vec::new();
        }
        self.face_neighbors[face_id]
            .iter()
            .filter_map(|&nb| nb.filter(|&id| self.active[id]))
            .collect()
    }

    pub fn vertex_neighbors(&self, face_id: usize) -> Vec<usize> {
        if face_id >= self.faces.len() || !self.active[face_id] {
            return Vec::new();
        }
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for &vid in &self.faces[face_id] {
            for &fid in &self.vertex_faces[vid] {
                if fid != face_id && self.active[fid] && seen.insert(fid) {
                    out.push(fid);
                }
            }
        }
        out.sort_unstable();
        out
    }

    pub fn remove_face(&mut self, face_id: usize) -> bool {
        if face_id < self.active.len() && self.active[face_id] {
            self.active[face_id] = false;
            self.n_active_faces -= 1;
            true
        } else {
            false
        }
    }

    pub fn restore_face(&mut self, face_id: usize) -> bool {
        if face_id < self.active.len() && !self.active[face_id] {
            self.active[face_id] = true;
            self.n_active_faces += 1;
            true
        } else {
            false
        }
    }

    pub fn add_face_across_edge(&mut self, face_id: usize, side: usize) -> Result<usize, String> {
        if face_id >= self.faces.len() || !self.active[face_id] {
            return Err(format!("face {face_id} is not active"));
        }
        if side >= self.p {
            return Err(format!("side {side} out of range for p = {}", self.p));
        }
        if let Some(nb) = self.face_neighbors[face_id][side] {
            if !self.active[nb] {
                self.active[nb] = true;
                self.n_active_faces += 1;
                return Ok(nb);
            }
            return Err(format!("edge ({face_id}, {side}) already borders active face {nb}"));
        }
        let u = self.faces[face_id][side];
        let v = self.faces[face_id][(side + 1) % self.p];
        let mid = geodesic_arc_poincare(self.poincare_pts[u], self.poincare_pts[v], 2)?[1];
        let t_m = poincare_to_su_trans(mid).ok_or("edge midpoint outside disk")?;
        let t_m_inv = [t_m[0], -t_m[1], -t_m[2], -t_m[3]];
        let half_turn = su_mul(su_mul(t_m, [0.0, 1.0, 0.0, 0.0]), t_m_inv);
        let new_centre_su = su_mul(half_turn, self.face_hash.items[face_id]);
        let new_centre = su_origin(new_centre_su);
        // The half-turn H swaps u = f[side] and v = f[side + 1] and preserves orientation, so the image
        // face reads v, u, H(f[side + 2]), ..., H(f[side - 1]) in the same rotational sense.
        // New vertices are not merged with existing ones (a dynamic edit, not a re-tiling).
        let mut new_cyc = vec![v, u];
        for step in 1..(self.p - 1) {
            let src_idx = (side + 1 + step) % self.p;
            let src_vid = self.faces[face_id][src_idx];
            let src_su = poincare_to_su_trans(self.poincare_pts[src_vid]).ok_or("vertex outside disk")?;
            let dst_pt = su_origin(su_mul(half_turn, src_su));
            let new_vid = self.poincare_pts.len();
            self.poincare_pts.push(dst_pt);
            self.vertex_faces.push(Vec::new());
            new_cyc.push(new_vid);
        }
        self.n_vertices = self.poincare_pts.len();
        self.n_edges += self.p - 1;
        let new_fid = self.faces.len();
        for &vid in &new_cyc {
            self.vertex_faces[vid].push(new_fid);
        }
        let mut new_nbrs = vec![None; self.p];
        new_nbrs[0] = Some(face_id);
        self.face_neighbors[face_id][side] = Some(new_fid);
        self.faces.push(new_cyc);
        self.face_shell.push(self.face_shell[face_id] + 1);
        self.face_centres_poincare.push(new_centre);
        self.face_neighbors.push(new_nbrs);
        self.active.push(true);
        self.n_active_faces += 1;
        self.face_hash.push_unchecked(new_centre_su);
        Ok(new_fid)
    }

    pub fn filter_by_region(&mut self, filter: RegionFilter) -> usize {
        let mut count = 0usize;
        for fid in 0..self.faces.len() {
            let z = self.face_centres_poincare[fid];
            let keep = match filter {
                RegionFilter::PoincareRadiusMax(rmax) => z[0] * z[0] + z[1] * z[1] <= rmax * rmax,
                RegionFilter::HyperbolicRadiusMax(rho_max) => {
                    2.0 * z[0].hypot(z[1]).min(0.999_999_999).atanh() <= rho_max
                }
                RegionFilter::ShellRange(s_lo, s_hi) => {
                    let s = self.face_shell[fid];
                    (s_lo..=s_hi).contains(&s)
                }
                RegionFilter::DiskBall { center, radius } => {
                    hyp_distance_poincare(z, center).is_ok_and(|d| d <= radius)
                }
                RegionFilter::HalfPlane { nx, ny, offset } => nx * z[0] + ny * z[1] >= offset,
            };
            self.active[fid] = keep;
            if keep {
                count += 1;
            }
        }
        self.n_active_faces = count;
        count
    }
}

pub fn build_hyperbolic_large(
    p: usize,
    q: usize,
    target_faces: usize,
    kernel: H2Kernel,
) -> Result<H2LargePatch, String> {
    if p < 3 || q < 3 || (p - 2) * (q - 2) <= 4 {
        return Err(format!("{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)"));
    }
    let target = target_faces.max(1);
    match kernel {
        H2Kernel::GraphCombinatorial => build_graph_combinatorial(p, q, target),
        H2Kernel::DunhamHyperboloid => build_dunham_hyperboloid(p, q, target),
        H2Kernel::SpatialHashPoincare => build_spatial_hash_poincare(p, q, target),
    }
}
