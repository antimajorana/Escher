//! Stage 3 defect fields: defect detection from bare coordinates, order parameters, local strain / Vitek / Nye
//! fields, homogenised elastic moduli with the Barnett-Lothe / Stroh dislocation field, and motion GIF export.
//!
//! Conventions shared by every submodule:
//! - Faces are the bounded faces of the planar embedding of `pos` + `edges`, walked counter-clockwise (signed area
//!   `A_f > 0`); vertices on the unbounded face are the *rim* and never carry a disclination charge.
//! - Topological (Frank) charges in degrees, from reference corner angles (each measured corner snapped to the nearest
//!   corner angle of the ideal lattice, i.e. the angle spanned by a run of consecutive ideal bonds): vertex
//!   `q_v = 360 - sum corners`, face `q_f = sum corners - (p - 2) 180`.
//!   This is the convention of `crate::defect::charge::frank_angles` (a pentagon in honeycomb is `+60`, a heptagon `-60`).
//! - Burgers vector: `b = sum_e G_e` over a counter-clockwise closed walk, `G_e` the ideal bond vector the measured bond
//!   snaps to, i.e. `closure_vector` of `crate::defect::circuit::burgers_circuit` (the convention of
//!   `DefectSpec::dislocation`). The Hirth-Lothe FS/RH vector for line direction `+z` is `-b` (`burgers_fs_rh`), and
//!   the continuum fields of `stroh_field` (displacement jump `+b` counter-clockwise) are FS/RH: see `nye_loops`.
//! - Coordination charge (`voronoi_charge`): `z_ideal - z` from Voronoi neighbours, `6 - z` for triangular crystals.

pub mod anim_gif;
pub mod cna_psi;
pub mod gsv_stroh;
pub mod nye_loops;
pub mod strain_vitek_nye;
pub mod stroh_field;
pub mod topo_detect;
pub mod voronoi_charge;
pub mod wigner_seitz;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_detect;

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::types::{Cell2D, Frame2D};

pub use anim_gif::{encode_gif89a, export_motion_gif, interpolate_crystal_frames, render_frames};
pub use cna_psi::{CnaSignature, cna_2d, psi_n};
pub use gsv_stroh::{ElasticClass, acoustic_tensor, barnett_lothe_tensor, cauchy_born_tensor, classify_elastic, dipole_prelog_coeff};
pub use nye_loops::{CoordDetection, DetectedDefect, detect_defects_from_coords};
pub use strain_vitek_nye::{LocalStrain, NyeField, local_deformation_and_strain, nye_tensor, vitek_differential};
pub use stroh_field::{AntiplaneField, StrohField, antiplane_screw_field, isotropic_dislocation_field, stroh_dislocation_field, stroh_sextic_field};
pub use topo_detect::{DefectCluster, TopoCharges, topological_charges};
pub use voronoi_charge::{VoronoiCoordination, voronoi_coordination};
pub use wigner_seitz::{WignerSeitz, wigner_seitz_occupancy};

/// Ideal-lattice data read from a unit cell: directed bond star, corner angles, coordinations, cell matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefLattice {
    /// Every directed bond vector `+-d` of the cell, A/B labels ignored (duplicates removed), cell Cartesian frame.
    pub star: Vec<[f64; 2]>,
    /// Distinct corner angles (degrees) spanned by runs of angularly consecutive bonds at the basis sites (below 360).
    pub corner_angles: Vec<f64>,
    /// Distinct site coordinations of the ideal lattice.
    pub degrees: Vec<usize>,
    pub lmax: f64,
    pub a1: [f64; 2],
    pub a2: [f64; 2],
    /// Rotation angle (radians) of the frame `q` the crystal is drawn in.
    pub frame_angle: f64,
}

pub(crate) fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn norm(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

pub(crate) fn rot(v: [f64; 2], th: f64) -> [f64; 2] {
    let (s, c) = th.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

/// Angle in `[0, 2 pi)`.
pub(crate) fn ang(v: [f64; 2]) -> f64 {
    v[1].atan2(v[0]).rem_euclid(2.0 * PI)
}

impl RefLattice {
    pub fn new(cell: &Cell2D, frame: Frame2D) -> Self {
        let ns = cell.basis.len();
        let mut per_site: Vec<Vec<[f64; 2]>> = vec![Vec::new(); ns];
        let mut star: Vec<[f64; 2]> = Vec::new();
        let push = |list: &mut Vec<[f64; 2]>, v: [f64; 2]| {
            if norm(v) > 1e-12 && !list.iter().any(|w| norm(sub(*w, v)) < 1e-9) {
                list.push(v);
            }
        };
        for b in &cell.bonds {
            let (bi, bj) = (cell.basis[b.i], cell.basis[b.j]);
            let (n1, n2) = (f64::from(b.n1), f64::from(b.n2));
            let d = [bj[0] + n1 * cell.a1[0] + n2 * cell.a2[0] - bi[0], bj[1] + n1 * cell.a1[1] + n2 * cell.a2[1] - bi[1]];
            let md = [-d[0], -d[1]];
            push(&mut per_site[b.i], d);
            push(&mut per_site[b.j], md);
            push(&mut star, d);
            push(&mut star, md);
        }
        let mut corner_angles: Vec<f64> = Vec::new();
        let mut degrees: Vec<usize> = Vec::new();
        for list in &per_site {
            if !degrees.contains(&list.len()) {
                degrees.push(list.len());
            }
            let mut angs: Vec<f64> = list.iter().map(|&v| ang(v)).collect();
            angs.sort_by(f64::total_cmp);
            // every run of `len` consecutive ideal corners: a site that lost bonds (dislocation core, vacancy rim) has
            // merged corners, e.g. 90 + 90 = 180 at a 3-coordinated square-lattice site
            let m = angs.len();
            for k in 0..m {
                for len in 1..m.max(2) {
                    let end = k + len;
                    let next = if end < m { angs[end] } else { angs[end - m] + 2.0 * PI };
                    let c = (next - angs[k]).to_degrees();
                    if c < 360.0 - 1e-6 && !corner_angles.iter().any(|&x| (x - c).abs() < 1e-6) {
                        corner_angles.push(c);
                    }
                }
            }
        }
        corner_angles.sort_by(f64::total_cmp);
        degrees.sort_unstable();
        let lmax = star.iter().map(|&v| norm(v)).fold(0.0, f64::max);
        let frame_angle = frame.q[1][0].atan2(frame.q[0][0]);
        Self { star, corner_angles, degrees, lmax, a1: cell.a1, a2: cell.a2, frame_angle }
    }

    /// Snap a measured corner angle (degrees) to the nearest ideal corner angle.
    pub fn snap_corner(&self, deg: f64) -> f64 {
        self.corner_angles.iter().copied().min_by(|a, b| (a - deg).abs().total_cmp(&(b - deg).abs())).unwrap_or(deg)
    }

    /// Fractional components `(u, v)` with `b = u a1 + v a2`.
    pub fn frac(&self, b: [f64; 2]) -> [f64; 2] {
        let det = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0];
        [(b[0] * self.a2[1] - b[1] * self.a2[0]) / det, (self.a1[0] * b[1] - self.a1[1] * b[0]) / det]
    }
}

/// Planar bond graph read from bare coordinates; `nbrs[v]` sorted counter-clockwise by bond angle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BondGraph {
    pub pos: Vec<[f64; 2]>,
    pub edges: Vec<(usize, usize)>,
    pub nbrs: Vec<Vec<usize>>,
}

impl BondGraph {
    /// Graph from explicit undirected edges (duplicates and self-loops dropped).
    pub fn from_edges(pos: &[[f64; 2]], edges: &[(usize, usize)]) -> Self {
        let mut set: HashSet<(usize, usize)> = HashSet::new();
        let mut out = Vec::new();
        for &(u, v) in edges {
            if u != v && u < pos.len() && v < pos.len() && set.insert((u.min(v), u.max(v))) {
                out.push((u.min(v), u.max(v)));
            }
        }
        out.sort_unstable();
        let mut nbrs = vec![Vec::new(); pos.len()];
        for &(u, v) in &out {
            nbrs[u].push(v);
            nbrs[v].push(u);
        }
        for (u, lst) in nbrs.iter_mut().enumerate() {
            lst.sort_by(|&a, &b| ang(sub(pos[a], pos[u])).total_cmp(&ang(sub(pos[b], pos[u]))));
        }
        Self { pos: pos.to_vec(), edges: out, nbrs }
    }

    /// Graph of all pairs closer than `cutoff` (spatial hash, `O(N)` for bounded density).
    pub fn from_cutoff(pos: &[[f64; 2]], cutoff: f64) -> Self {
        Self::from_edges(pos, &pairs_within(pos, cutoff))
    }

    pub fn degree(&self, v: usize) -> usize {
        self.nbrs[v].len()
    }
}

/// All index pairs `(i < j)` with `|p_i - p_j| < cutoff`.
pub fn pairs_within(pos: &[[f64; 2]], cutoff: f64) -> Vec<(usize, usize)> {
    let cell = cutoff.max(1e-12);
    let key = |p: [f64; 2]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in pos.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let mut out = Vec::new();
    for (i, &p) in pos.iter().enumerate() {
        let (kx, ky) = key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(kx + dx, ky + dy)).map_or(&[][..], Vec::as_slice) {
                    if j > i && norm(sub(pos[j], p)) < cutoff {
                        out.push((i, j));
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Faces of the planar embedding: bounded CCW faces, their areas, and the rim (vertices of unbounded faces).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanarFaces {
    pub faces: Vec<Vec<usize>>,
    pub areas: Vec<f64>,
    pub rim: Vec<bool>,
    /// Bounded face lying to the left of each directed edge `(u, v)`.
    pub left_face: HashMap<(usize, usize), usize>,
}

pub(crate) fn signed_area(pos: &[[f64; 2]], f: &[usize]) -> f64 {
    let m = f.len();
    0.5 * (0..m).map(|k| {
        let (p, q) = (pos[f[k]], pos[f[(k + 1) % m]]);
        p[0] * q[1] - p[1] * q[0]
    }).sum::<f64>()
}

impl PlanarFaces {
    /// Face walk of `crate::defect::volterra_patch::drawn_faces`: from dart `(a, b)` continue to the neighbour of `b`
    /// just clockwise of `a`; walks with area `> 1e-12` are the bounded faces.
    pub fn new(g: &BondGraph) -> Self {
        let idx: Vec<HashMap<usize, usize>> = g.nbrs.iter().map(|l| l.iter().enumerate().map(|(k, &v)| (v, k)).collect()).collect();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let (mut faces, mut areas, mut rim) = (Vec::new(), Vec::new(), vec![false; g.pos.len()]);
        let mut left_face = HashMap::new();
        for (u, lst) in g.nbrs.iter().enumerate() {
            for &v in lst {
                if seen.contains(&(u, v)) {
                    continue;
                }
                let mut f = Vec::new();
                let (mut a, mut b) = (u, v);
                while seen.insert((a, b)) {
                    f.push(a);
                    let bl = &g.nbrs[b];
                    let k = idx[b][&a];
                    a = b;
                    b = bl[if k == 0 { bl.len() - 1 } else { k - 1 }];
                }
                let area = signed_area(&g.pos, &f);
                if area > 1e-12 {
                    let fi = faces.len();
                    for k in 0..f.len() {
                        left_face.insert((f[k], f[(k + 1) % f.len()]), fi);
                    }
                    faces.push(f);
                    areas.push(area);
                } else {
                    for &x in &f {
                        rim[x] = true;
                    }
                }
            }
        }
        for (v, lst) in g.nbrs.iter().enumerate() {
            if lst.is_empty() {
                rim[v] = true;
            }
        }
        Self { faces, areas, rim, left_face }
    }

    /// Bounded faces sharing an edge with face `fi`.
    pub fn face_neighbours(&self, fi: usize) -> Vec<usize> {
        let f = &self.faces[fi];
        (0..f.len()).filter_map(|k| self.left_face.get(&(f[(k + 1) % f.len()], f[k])).copied()).collect()
    }

    /// Whether a face has a directed edge whose reverse has no bounded face (it touches the unbounded face).
    pub fn on_boundary(&self, fi: usize) -> bool {
        let f = &self.faces[fi];
        (0..f.len()).any(|k| !self.left_face.contains_key(&(f[(k + 1) % f.len()], f[k])))
    }
}
