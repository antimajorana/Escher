//! Coincidence-site-lattice (CSL) grain boundaries and finite symmetric-tilt bicrystals.
//!
//! CSL formulas (coprime `m > n >= 1`):
//! * hexagonal / trigonal: `s = m^2 + mn + n^2`, `Sigma = s / gcd(s, 3)`, `cos(theta) = (m^2 + 4mn + n^2) / (2s)`;
//! * square: `s = m^2 + n^2`, `Sigma = s / gcd(s, 2)`, `theta = 2 atan(n / m)`.
//!
//! The complement (`60 deg - theta` hexagonal, `90 deg - theta` square) is listed too (it has the same Sigma and a
//! different boundary plane); repeated angles from other `(m, n)` are removed.
//!
//! Bicrystal frames: grain A (`side = 'A'`) has local frame angle `-theta/2`, grain B `+theta/2`. `ref_pos` is the
//! unrotated lattice position, `pos` the rotated one, `vec` is expressed in the frame of its tail site, and a bond
//! between grains carries `across = true` and the seam rotation `R(phi_u - phi_w)`, so `check_frames` holds.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::cone_geometry::{norm2, sub2};
use crate::defect::core_graph::PerfectLattice;
use crate::defect::point_defects::{edges_from_adj, plain_bond, segments_cross};
use crate::types::{BuiltCrystal, Cell2D, DirectedBondData, Isometry2};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CslGrainBoundaryEntry {
    pub id: String,
    pub sigma: u32,
    pub m: i32,
    pub n: i32,
    pub theta_deg: f64,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub family: crate::defect::burgers_indices::StaticStr,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub provenance: crate::defect::burgers_indices::StaticStr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BicrystalOptions {
    pub radius: f64,
    pub merge_tol: f64,
    pub max_bond_scale: f64,
}

impl Default for BicrystalOptions {
    fn default() -> Self {
        Self { radius: 8.0, merge_tol: 0.35, max_bond_scale: 1.28 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BicrystalResult {
    pub crystal: BuiltCrystal,
    pub theta_deg: f64,
    pub sigma: Option<u32>,
    pub exact_csl: bool,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub provenance: crate::defect::burgers_indices::StaticStr,
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// `"hexagonal"` / `"square"` when the group and the cell metric fit, else `None`.
pub fn csl_family(group: &str, cell: &Cell2D) -> Option<&'static str> {
    let (l1, l2) = (norm2(cell.a1), norm2(cell.a2));
    if l1 < 1e-12 || (l1 - l2).abs() > 1e-5 * l1 {
        return None;
    }
    let cos_g = (cell.a1[0] * cell.a2[0] + cell.a1[1] * cell.a2[1]) / (l1 * l2);
    match group {
        "p3" | "p3m1" | "p31m" | "p6" | "p6m" if (cos_g.abs() - 0.5).abs() <= 1e-4 => Some("hexagonal"),
        "p4" | "p4m" | "p4g" if cos_g.abs() <= 1e-4 => Some("square"),
        _ => None,
    }
}

/// CSL symmetric-tilt misorientations with `1 < Sigma <= max_sigma`, sorted by `(Sigma, theta)`.
pub fn list_csl_grain_boundaries(group: &str, cell: &Cell2D, max_sigma: u32) -> Vec<CslGrainBoundaryEntry> {
    let Some(family) = csl_family(group, cell) else { return Vec::new() };
    let hex = family == "hexagonal";
    let (period, mult) = if hex { (60.0, 3i64) } else { (90.0, 2i64) };
    let m_max = ((mult * i64::from(max_sigma)) as f64).sqrt().ceil() as i32 + 1;
    let mut raw: Vec<(u32, f64, i32, i32)> = Vec::new();
    for m in 2..=m_max {
        for n in 1..m {
            if gcd(i64::from(m), i64::from(n)) != 1 {
                continue;
            }
            let (mi, ni) = (i64::from(m), i64::from(n));
            let s = if hex { mi * mi + mi * ni + ni * ni } else { mi * mi + ni * ni };
            let sigma = s / gcd(s, mult);
            if sigma <= 1 || sigma > i64::from(max_sigma) {
                continue;
            }
            let theta = if hex {
                (((mi * mi + 4 * mi * ni + ni * ni) as f64) / ((2 * s) as f64)).clamp(-1.0, 1.0).acos().to_degrees()
            } else {
                (2.0 * f64::from(n).atan2(f64::from(m))).to_degrees()
            };
            for t in [theta, period - theta] {
                raw.push((sigma as u32, t, m, n));
            }
        }
    }
    raw.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then((a.2, a.3).cmp(&(b.2, b.3))));
    let mut out: Vec<CslGrainBoundaryEntry> = Vec::new();
    for (sigma, theta, m, n) in raw {
        if out.iter().any(|e| e.sigma == sigma && (e.theta_deg - theta).abs() < 1e-6) {
            continue;
        }
        let k = out.iter().filter(|e| e.sigma == sigma).count();
        let suffix = (b'a' + (k as u8).min(25)) as char;
        out.push(CslGrainBoundaryEntry {
            id: format!("\u{3a3}{sigma}{suffix}"),
            sigma,
            m,
            n,
            theta_deg: theta,
            family,
            provenance: "[PROVED]",
        });
    }
    out
}

fn rot(phi: f64, v: [f64; 2]) -> [f64; 2] {
    let (s, c) = phi.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

/// Perfect open disk of `cell` centred at `centre` (no seams; `ref_pos = pos`).
pub fn perfect_disk(cell: &Cell2D, centre: [f64; 2], radius: f64) -> BuiltCrystal {
    let lat = PerfectLattice::new(cell.clone());
    let keys = lat.sites_in_disk(centre, radius);
    let index: HashMap<_, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| lat.pos(k)).collect();
    let mut adj = vec![Vec::new(); keys.len()];
    for (i, &k) in keys.iter().enumerate() {
        for (k2, kind) in lat.nbrs(k) {
            if let Some(&j) = index.get(&k2) {
                adj[i].push(plain_bond(j, sub2(pos[j], pos[i]), kind));
            }
        }
    }
    BuiltCrystal {
        ref_pos: pos.clone(),
        basis_idx: keys.iter().map(|k| k.2).collect(),
        side: vec![None; pos.len()],
        cone_sites: Vec::new(),
        edges: edges_from_adj(&adj),
        adj,
        pos,
        wall_sigma: None,
        is_non_orientable: false,
    }
}

/// Finite symmetric-tilt bicrystal: grain A (`x <= 0`) rotated by `-theta/2`, grain B (`x >= 0`) by `+theta/2`.
pub fn build_symmetric_tilt_bicrystal(
    group: &str,
    cell: &Cell2D,
    theta_deg: f64,
    sigma: Option<u32>,
    opts: &BicrystalOptions,
) -> Result<BicrystalResult, String> {
    let _ = group;
    if !(theta_deg.is_finite() && opts.radius > 0.0 && opts.merge_tol >= 0.0 && opts.max_bond_scale > 0.0) {
        return Err("need finite theta, radius > 0, merge_tol >= 0, max_bond_scale > 0".into());
    }
    let d_nn = {
        let lat = PerfectLattice::new(cell.clone());
        let mut d = f64::INFINITY;
        for s in 0..cell.basis.len() {
            for (k2, _) in lat.nbrs((0, 0, s)) {
                d = d.min(norm2(sub2(lat.pos(k2), lat.pos((0, 0, s)))));
            }
        }
        if d.is_finite() && d > 1e-9 { d } else { return Err("cell has no bonds".into()) }
    };
    let half = 0.5 * theta_deg.to_radians();
    let disk = perfect_disk(cell, [0.0, 0.0], opts.radius);
    // grain membership: (disk index, side, frame angle)
    let mut pos = Vec::new();
    let mut ref_pos = Vec::new();
    let mut basis_idx = Vec::new();
    let mut side = Vec::new();
    let mut phi = Vec::new();
    let mut map_a = vec![None; disk.pos.len()];
    let mut map_b = vec![None; disk.pos.len()];
    for (g, (ph, tag)) in [(-half, 'A'), (half, 'B')].into_iter().enumerate() {
        for i in 0..disk.pos.len() {
            let p = rot(ph, disk.pos[i]);
            if (g == 0 && p[0] <= 1e-9) || (g == 1 && p[0] >= -1e-9) {
                let idx = pos.len();
                if g == 0 { map_a[i] = Some(idx) } else { map_b[i] = Some(idx) }
                pos.push(p);
                ref_pos.push(disk.pos[i]);
                basis_idx.push(disk.basis_idx[i]);
                side.push(Some(tag));
                phi.push(ph);
            }
        }
    }
    // merge B sites onto A sites within merge_tol * d_nn
    let n_all = pos.len();
    let mut rep: Vec<usize> = (0..n_all).collect();
    let mut taken: HashSet<usize> = HashSet::new();
    let a_ids: Vec<usize> = map_a.iter().flatten().copied().collect();
    for &b in map_b.iter().flatten() {
        let best = a_ids.iter().copied().filter(|a| !taken.contains(a)).map(|a| (norm2(sub2(pos[a], pos[b])), a)).min_by(|x, y| x.0.total_cmp(&y.0));
        if let Some((d, a)) = best && d <= opts.merge_tol * d_nn {
            rep[b] = a;
            taken.insert(a);
        }
    }
    let keep: Vec<usize> = (0..n_all).filter(|&i| rep[i] == i).collect();
    let mut new_idx = vec![usize::MAX; n_all];
    for (k, &i) in keep.iter().enumerate() {
        new_idx[i] = k;
    }
    let fin = |i: usize| new_idx[rep[i]];
    let n = keep.len();
    let p: Vec<[f64; 2]> = keep.iter().map(|&i| pos[i]).collect();
    let fphi: Vec<f64> = keep.iter().map(|&i| phi[i]).collect();
    let mut adj: Vec<Vec<DirectedBondData>> = vec![Vec::new(); n];
    let mut seg: Vec<(usize, usize)> = Vec::new();
    let add = |adj: &mut Vec<Vec<DirectedBondData>>, seg: &mut Vec<(usize, usize)>, u: usize, w: usize, kind: usize, check: bool| {
        if u == w || adj[u].iter().any(|e| e.to == w) {
            return;
        }
        if check && seg.iter().any(|&(a, b)| ![a, b].contains(&u) && ![a, b].contains(&w) && segments_cross(p[u], p[w], p[a], p[b])) {
            return;
        }
        for (x, y) in [(u, w), (w, u)] {
            let vec = rot(-fphi[x], sub2(p[y], p[x]));
            let across = (fphi[x] - fphi[y]).abs() > 1e-12;
            let (s, c) = (fphi[x] - fphi[y]).sin_cos();
            let seam = across.then_some(Isometry2 { m: [[c, -s], [s, c]], t: [0.0, 0.0] });
            adj[x].push(DirectedBondData { seam, across, ..plain_bond(y, vec, kind) });
        }
        seg.push((u, w));
    };
    for (g, map) in [&map_a, &map_b].into_iter().enumerate() {
        for i in 0..disk.pos.len() {
            let Some(gi) = map[i] else { continue };
            for e in &disk.adj[i] {
                if e.to > i && let Some(gj) = map[e.to] {
                    let (u, w) = (fin(gi), fin(gj));
                    add(&mut adj, &mut seg, u, w, e.kind, g == 1 && (rep[gi] != gi || rep[gj] != gj));
                }
            }
        }
    }
    // across-seam bonds, shortest first, without crossings
    let reach = opts.max_bond_scale * d_nn;
    let mut cand: Vec<(f64, usize, usize)> = Vec::new();
    for u in 0..n {
        for w in (u + 1)..n {
            if (fphi[u] - fphi[w]).abs() > 1e-12 && p[u][0].abs() <= reach && p[w][0].abs() <= reach {
                let d = norm2(sub2(p[u], p[w]));
                if d <= reach {
                    cand.push((d, u, w));
                }
            }
        }
    }
    cand.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    for (_, u, w) in cand {
        add(&mut adj, &mut seg, u, w, 0, true);
    }
    let crystal = BuiltCrystal {
        ref_pos: keep.iter().map(|&i| ref_pos[i]).collect(),
        basis_idx: keep.iter().map(|&i| basis_idx[i]).collect(),
        side: keep.iter().map(|&i| side[i]).collect(),
        cone_sites: Vec::new(),
        edges: edges_from_adj(&adj),
        adj,
        pos: p,
        wall_sigma: None,
        is_non_orientable: false,
    };
    Ok(BicrystalResult {
        crystal,
        theta_deg,
        sigma,
        exact_csl: sigma.is_some(),
        provenance: if sigma.is_some() { "[PROVED: exact CSL \u{3a3}]" } else { "[COMPUTATIONAL EVIDENCE: finite symmetric-tilt patch]" },
    })
}
