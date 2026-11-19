//! Point defects on a built crystal: vacancies (optionally ring-reconstructed), substitutions, interstitials and
//! adatoms, keeping every `BuiltCrystal` field consistent (sites removed/appended, `adj` / `edges` / `cone_sites`
//! remapped).
//!
//! `BuiltCrystal` stores no faces or rim: faces are the walks of [`frame_faces`] and the rim is the vertex set of the
//! outer (largest) face walk; both are recomputed from the returned crystal.
//!
//! Site indices in [`PointDefectOp`] refer to the input crystal; appended sites (interstitials, adatoms) get indices
//! `N, N + 1, ...` in op order and can be referenced by later ops.

use std::collections::{BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::charge::{corners_order, frame_faces, outer_face_index};
use crate::defect::cone_geometry::{add2, mat2_vec, norm2, sub2};
use crate::types::{BuiltCrystal, DirectedBondData, Isometry2};

const DEFAULT_SPECIES: [&str; 8] = ["C", "B", "N", "Si", "P", "S", "Ge", "As"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PointDefectOp {
    Vacancy { site: usize, reconstruct_ring: bool },
    Substitutional { site: usize, species: String },
    Interstitial { pos: [f64; 2], species: String, bond_cutoff: f64 },
    Adatom { host_site: usize, height: f64, species: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointDefectedCrystal {
    pub crystal: BuiltCrystal,
    pub species: Vec<String>,
    pub pos3d: Vec<[f64; 3]>,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub holonomy_badge: crate::defect::burgers_indices::StaticStr,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub core_badge: crate::defect::burgers_indices::StaticStr,
    pub summary: String,
}

/// Default chemical symbols by sublattice: `["C", "B", "N", "Si", "P", "S", "Ge", "As"][basis_idx % 8]`.
pub fn default_site_species(basis_idx: &[usize]) -> Vec<String> {
    basis_idx.iter().map(|&b| DEFAULT_SPECIES[b % 8].to_string()).collect()
}

/// Syntactic element-symbol check: 1-2 ASCII letters, uppercase first, optional lowercase second.
pub fn validate_element_symbol(s: &str) -> Result<(), String> {
    let b = s.as_bytes();
    let ok = matches!(b.len(), 1 | 2) && b[0].is_ascii_uppercase() && (b.len() == 1 || b[1].is_ascii_lowercase());
    if ok { Ok(()) } else { Err(format!("invalid element symbol {s:?} (expected e.g. \"C\", \"Si\")")) }
}

/// Vertex set of the outer (largest) face walk of [`frame_faces`].
pub fn rim_sites(crystal: &BuiltCrystal) -> HashSet<usize> {
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return HashSet::new();
    }
    faces[outer_face_index(&faces)].iter().copied().collect()
}

/// Sorted `(u, v)` with `u < v` for every bond of `adj`.
pub(crate) fn edges_from_adj(adj: &[Vec<DirectedBondData>]) -> Vec<(usize, usize)> {
    let set: BTreeSet<(usize, usize)> = adj.iter().enumerate().flat_map(|(u, l)| l.iter().map(move |e| (u.min(e.to), u.max(e.to)))).collect();
    set.into_iter().collect()
}

pub(crate) fn plain_bond(to: usize, vec: [f64; 2], kind: usize) -> DirectedBondData {
    DirectedBondData { to, vec, kind, seam: None, ang: None, across: false, ghost_stub: None }
}

pub(crate) fn segments_cross(p: [f64; 2], q: [f64; 2], r: [f64; 2], s: [f64; 2]) -> bool {
    let c = |a: [f64; 2], b: [f64; 2], o: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let (d1, d2, d3, d4) = (c(r, s, p), c(r, s, q), c(p, q, r), c(p, q, s));
    let eps = 1e-12;
    ((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) && ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps))
}

struct Work {
    pos: Vec<[f64; 2]>,
    ref_pos: Vec<[f64; 2]>,
    basis_idx: Vec<usize>,
    side: Vec<Option<char>>,
    species: Vec<String>,
    z: Vec<f64>,
    alive: Vec<bool>,
    adj: Vec<Vec<DirectedBondData>>,
}

impl Work {
    fn check_site(&self, v: usize) -> Result<(), String> {
        if v < self.alive.len() && self.alive[v] { Ok(()) } else { Err(format!("site {v} does not exist (or was removed)")) }
    }

    fn push_site(&mut self, pos: [f64; 2], ref_pos: [f64; 2], basis: usize, species: String, z: f64) -> usize {
        self.pos.push(pos);
        self.ref_pos.push(ref_pos);
        self.basis_idx.push(basis);
        self.side.push(None);
        self.species.push(species);
        self.z.push(z);
        self.alive.push(true);
        self.adj.push(Vec::new());
        self.pos.len() - 1
    }

    fn has_bond(&self, u: usize, v: usize) -> bool {
        self.adj[u].iter().any(|e| e.to == v)
    }

    fn crosses_any(&self, a: usize, b: usize) -> bool {
        self.adj.iter().enumerate().filter(|(u, _)| self.alive[*u]).any(|(u, l)| {
            l.iter().any(|e| u < e.to && ![a, b].contains(&u) && ![a, b].contains(&e.to)
                && segments_cross(self.pos[a], self.pos[b], self.pos[u], self.pos[e.to]))
        })
    }

    /// Bond `a -> b` through a removed site `v`: `vec = vec(a,v) + g1^-1 vec(v,b)`, `seam = g2 o g1`.
    fn chord(&mut self, a: usize, e_av: &DirectedBondData, e_vb: &DirectedBondData) {
        let g1 = e_av.seam.unwrap_or(Isometry2::IDENTITY);
        let g2 = e_vb.seam.unwrap_or(Isometry2::IDENTITY);
        let vec = add2(e_av.vec, mat2_vec(g1.inverse().m, e_vb.vec));
        let g = g2.compose(&g1);
        let is_id = (g.m[0][0] - 1.0).abs() + g.m[0][1].abs() + g.m[1][0].abs() + (g.m[1][1] - 1.0).abs() + g.t[0].abs() + g.t[1].abs() < 1e-9;
        let seam = if is_id { None } else { Some(g) };
        let b = e_vb.to;
        let back_vec = mat2_vec(g.m, vec);
        self.adj[a].push(DirectedBondData { seam, ..plain_bond(b, vec, e_av.kind) });
        self.adj[b].push(DirectedBondData { seam: seam.map(|g| g.inverse()), ..plain_bond(a, [-back_vec[0], -back_vec[1]], e_av.kind) });
    }

    fn vacancy(&mut self, v: usize, reconstruct: bool) -> usize {
        let order: Vec<usize> = {
            let probe = self.as_crystal();
            corners_order(&probe)[v].iter().map(|&(_, w)| w).collect()
        };
        let out_edges = std::mem::take(&mut self.adj[v]);
        for e in &out_edges {
            self.adj[e.to].retain(|b| b.to != v);
        }
        self.alive[v] = false;
        if !reconstruct {
            return 0;
        }
        let back: Vec<(usize, DirectedBondData)> = order
            .iter()
            .filter_map(|&w| {
                let e_vw = out_edges.iter().find(|e| e.to == w)?.clone();
                Some((w, e_vw))
            })
            .collect();
        // e(a -> v) is the reverse of e(v -> a) (check_frames convention): vec = -g.m vec(v,a), seam = g^-1
        let rev = |e: &DirectedBondData| {
            let g = e.seam.unwrap_or(Isometry2::IDENTITY);
            let v = mat2_vec(g.m, e.vec);
            DirectedBondData { seam: e.seam.map(|g| g.inverse()), ..plain_bond(usize::MAX, [-v[0], -v[1]], e.kind) }
        };
        let mut ring: Vec<usize> = (0..back.len()).collect();
        let mut chords = 0;
        while ring.len() >= 2 {
            let m = ring.len();
            let mut best: Option<(f64, usize)> = None;
            for k in 0..(if m == 2 { 1 } else { m }) {
                let (a, b) = (back[ring[k]].0, back[ring[(k + 1) % m]].0);
                if self.has_bond(a, b) || self.crosses_any(a, b) {
                    continue;
                }
                let len = norm2(sub2(self.pos[a], self.pos[b]));
                if best.is_none_or(|bst| len < bst.0 - 1e-12) {
                    best = Some((len, k));
                }
            }
            let Some((_, k)) = best else { break };
            let (ia, ib) = (ring[k], ring[(k + 1) % m]);
            let e_av = rev(&back[ia].1);
            self.chord(back[ia].0, &e_av, &back[ib].1);
            chords += 1;
            ring.retain(|&x| x != ia && x != ib);
        }
        chords
    }

    fn as_crystal(&self) -> BuiltCrystal {
        BuiltCrystal {
            pos: self.pos.clone(),
            ref_pos: self.ref_pos.clone(),
            basis_idx: self.basis_idx.clone(),
            side: self.side.clone(),
            cone_sites: Vec::new(),
            edges: edges_from_adj(&self.adj),
            adj: self.adj.clone(),
            wall_sigma: None,
            is_non_orientable: false,
        }
    }
}

/// Apply point-defect ops in order (see the module doc for index conventions).
pub fn apply_point_defects(crystal: &BuiltCrystal, ops: &[PointDefectOp]) -> Result<PointDefectedCrystal, String> {
    let n0 = crystal.pos.len();
    if crystal.adj.len() != n0 || crystal.ref_pos.len() != n0 || crystal.basis_idx.len() != n0 || crystal.side.len() != n0 {
        return Err("inconsistent BuiltCrystal: pos / ref_pos / basis_idx / side / adj lengths differ".into());
    }
    let rim = rim_sites(crystal);
    let extra_basis = crystal.basis_idx.iter().copied().max().map_or(0, |m| m + 1);
    let mut w = Work {
        pos: crystal.pos.clone(),
        ref_pos: crystal.ref_pos.clone(),
        basis_idx: crystal.basis_idx.clone(),
        side: crystal.side.clone(),
        species: default_site_species(&crystal.basis_idx),
        z: vec![0.0; n0],
        alive: vec![true; n0],
        adj: crystal.adj.clone(),
    };
    let (mut n_vac, mut n_chord, mut n_sub, mut n_int, mut n_ad) = (0, 0, 0, 0, 0);
    let mut reconstructed = false;
    for (k, op) in ops.iter().enumerate() {
        let ctx = |e: String| format!("op {k}: {e}");
        match op {
            PointDefectOp::Vacancy { site, reconstruct_ring } => {
                w.check_site(*site).map_err(ctx)?;
                if rim.contains(site) {
                    return Err(ctx(format!("site {site} lies on the rim of the patch")));
                }
                n_chord += w.vacancy(*site, *reconstruct_ring);
                reconstructed |= *reconstruct_ring;
                n_vac += 1;
            }
            PointDefectOp::Substitutional { site, species } => {
                w.check_site(*site).map_err(ctx)?;
                validate_element_symbol(species).map_err(ctx)?;
                w.species[*site] = species.clone();
                n_sub += 1;
            }
            PointDefectOp::Interstitial { pos, species, bond_cutoff } => {
                validate_element_symbol(species).map_err(ctx)?;
                if !(bond_cutoff.is_finite() && *bond_cutoff > 0.0 && pos[0].is_finite() && pos[1].is_finite()) {
                    return Err(ctx("interstitial needs a finite position and bond_cutoff > 0".into()));
                }
                let live: Vec<usize> = (0..w.pos.len()).filter(|&i| w.alive[i]).collect();
                let nearest = live.iter().copied().min_by(|&a, &b| norm2(sub2(w.pos[a], *pos)).total_cmp(&norm2(sub2(w.pos[b], *pos))));
                let ref_new = nearest.map_or(*pos, |j| add2(w.ref_pos[j], sub2(*pos, w.pos[j])));
                let v = w.push_site(*pos, ref_new, extra_basis, species.clone(), 0.0);
                for j in live {
                    let d = norm2(sub2(w.pos[j], *pos));
                    if d > 0.0 && d <= *bond_cutoff {
                        let vec = sub2(w.ref_pos[j], ref_new);
                        w.adj[v].push(plain_bond(j, vec, 0));
                        w.adj[j].push(plain_bond(v, [-vec[0], -vec[1]], 0));
                    }
                }
                n_int += 1;
            }
            PointDefectOp::Adatom { host_site, height, species } => {
                w.check_site(*host_site).map_err(ctx)?;
                validate_element_symbol(species).map_err(ctx)?;
                if !(height.is_finite() && height.abs() > 1e-6) {
                    return Err(ctx("adatom height must be finite with |height| > 1e-6".into()));
                }
                let (p, r) = (w.pos[*host_site], w.ref_pos[*host_site]);
                let v = w.push_site(p, r, extra_basis, species.clone(), *height);
                w.adj[v].push(plain_bond(*host_site, [0.0, 0.0], 0));
                w.adj[*host_site].push(plain_bond(v, [0.0, 0.0], 0));
                n_ad += 1;
            }
        }
    }
    // compaction
    let mut new_idx = vec![usize::MAX; w.pos.len()];
    let mut next = 0;
    for (i, &a) in w.alive.iter().enumerate() {
        if a {
            new_idx[i] = next;
            next += 1;
        }
    }
    let keep: Vec<usize> = (0..w.pos.len()).filter(|&i| w.alive[i]).collect();
    let adj: Vec<Vec<DirectedBondData>> = keep
        .iter()
        .map(|&i| w.adj[i].iter().map(|e| DirectedBondData { to: new_idx[e.to], ..e.clone() }).collect())
        .collect();
    let out = BuiltCrystal {
        pos: keep.iter().map(|&i| w.pos[i]).collect(),
        ref_pos: keep.iter().map(|&i| w.ref_pos[i]).collect(),
        basis_idx: keep.iter().map(|&i| w.basis_idx[i]).collect(),
        side: keep.iter().map(|&i| w.side[i]).collect(),
        cone_sites: crystal.cone_sites.iter().filter(|&&c| c < n0 && w.alive[c]).map(|&c| new_idx[c]).collect(),
        edges: edges_from_adj(&adj),
        adj,
        wall_sigma: crystal.wall_sigma,
        is_non_orientable: crystal.is_non_orientable,
    };
    let pos3d = keep.iter().map(|&i| [w.pos[i][0], w.pos[i][1], w.z[i]]).collect();
    let summary = format!(
        "{n_vac} vacancy(ies) ({n_chord} reconstruction chord(s)), {n_sub} substitution(s), {n_int} interstitial(s), \
         {n_ad} adatom(s); net Burgers [0, 0], Frank 0 deg (no lattice cut)"
    );
    Ok(PointDefectedCrystal {
        crystal: out,
        species: keep.iter().map(|&i| w.species[i].clone()).collect(),
        pos3d,
        holonomy_badge: "[PROVED]",
        core_badge: if reconstructed { "[COMPUTATIONAL EVIDENCE]" } else { "[PROVED]" },
        summary,
    })
}
