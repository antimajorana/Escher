//! Archimedean vertex-configuration parser, dihedral canonicalizer, and odd-gon parity checks.
//!
//! Angle defect of `n_1.n_2...n_m`: `Delta = 2 pi - sum_i pi (1 - 2/n_i) = pi (2 - m + 2 sum_i 1/n_i)`;
//! `Delta > 0`, `= 0`, `< 0` selects `S^2`, `E^2`, `H^2` ([PROVED], elementary). Two *necessary*
//! conditions for a vertex-transitive (Archimedean) realisation are checked and refused with a reason:
//! the odd-gon parity rule (Grunbaum & Shephard, *Tilings and Patterns*, 1987, section 2.1: around an
//! odd `k`-gon the neighbouring faces alternate, so `a.k.b` with `a != b` needs another occurrence of
//! `k` to close the cycle) and, on `S^2`, Descartes / Gauss-Bonnet `V * Delta = 4 pi` with `V` an
//! integer. Passing them is not a proof of realisability in `H^2`.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::GeometryKind;

use super::wythoff_archimedean::WythoffOp;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VertexConfigInfo {
    pub input: String,
    pub canonical: String,
    pub sequence: Vec<u32>,
    pub is_chiral: bool,
    pub is_snub_chiral: bool,
    pub angle_defect_rad: f64,
    pub angle_defect_pi_num: i128,
    pub angle_defect_pi_den: i128,
    pub geometry_kind: GeometryKind,
    /// `true` when the necessary local conditions (odd-gon parity, `S^2` Descartes integrality) pass;
    /// configurations failing them are refused with an error instead.
    pub realizable: bool,
    pub provenance: String,
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

pub(crate) fn dihedral_canon_u32(seq: &[u32]) -> (Vec<u32>, bool) {
    let n = seq.len();
    let mut cyc_best: Option<Vec<u32>> = None;
    for r in 0..n {
        let cand: Vec<u32> = (0..n).map(|i| seq[(r + i) % n]).collect();
        if cyc_best.as_ref().is_none_or(|b| cand < *b) {
            cyc_best = Some(cand);
        }
    }
    let mut rev_seq = seq.to_vec();
    rev_seq.reverse();
    let mut rev_best: Option<Vec<u32>> = None;
    for r in 0..n {
        let cand: Vec<u32> = (0..n).map(|i| rev_seq[(r + i) % n]).collect();
        if rev_best.as_ref().is_none_or(|b| cand < *b) {
            rev_best = Some(cand);
        }
    }
    let cyc = cyc_best.unwrap();
    let rev = rev_best.unwrap();
    let is_chiral = cyc != rev;
    (if cyc <= rev { cyc } else { rev }, is_chiral)
}

pub fn canonical_vertex_symbol(seq: &[u32]) -> String {
    let (canon, _) = dihedral_canon_u32(seq);
    canon.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(".")
}

pub(crate) fn wythoff_vertex_symbol(p: usize, q: usize, op: WythoffOp) -> String {
    let (pu, qu) = (p as u32, q as u32);
    if p == 2 || q == 2 {
        let k = if p == 2 { qu } else { pu };
        return match op {
            WythoffOp::Regular => format!("{pu}.{qu}"),
            WythoffOp::Dual => format!("{qu}.{pu}"),
            WythoffOp::Rectified => canonical_vertex_symbol(&[k, k]),
            WythoffOp::Truncated if p == 2 => canonical_vertex_symbol(&[4, 4, qu]),
            WythoffOp::Truncated => canonical_vertex_symbol(&[2 * pu, 2 * pu]),
            WythoffOp::Bitruncated if q == 2 => canonical_vertex_symbol(&[4, 4, pu]),
            WythoffOp::Bitruncated => canonical_vertex_symbol(&[2 * qu, 2 * qu]),
            WythoffOp::Cantellated => canonical_vertex_symbol(&[4, 4, k]),
            WythoffOp::Omnitruncated => canonical_vertex_symbol(&[4, 4, 2 * k]),
            WythoffOp::Snub { .. } => canonical_vertex_symbol(&[3, 3, 3, k]),
        };
    }
    let seq = match op {
        WythoffOp::Regular => vec![pu; q],
        WythoffOp::Dual => vec![qu; p],
        WythoffOp::Rectified => vec![pu, qu, pu, qu],
        WythoffOp::Truncated => vec![qu, 2 * pu, 2 * pu],
        WythoffOp::Bitruncated => vec![pu, 2 * qu, 2 * qu],
        WythoffOp::Cantellated => vec![pu, 4, qu, 4],
        WythoffOp::Omnitruncated => vec![4, 2 * pu, 2 * qu],
        WythoffOp::Snub { .. } => vec![3, 3, pu, 3, qu],
    };
    canonical_vertex_symbol(&seq)
}

fn odd_gon_parity_obstruction(seq: &[u32]) -> Option<String> {
    let m = seq.len();
    let mut odds: Vec<u32> = seq.iter().copied().filter(|&x| x % 2 == 1).collect();
    odds.sort_unstable();
    odds.dedup();
    for k in odds {
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_self_loop = false;
        for i in 0..m {
            if seq[i] == k {
                let a = seq[(i + m - 1) % m];
                let b = seq[(i + 1) % m];
                if a == b {
                    has_self_loop = true;
                    break;
                }
                adj.entry(a).or_default().push(b);
                adj.entry(b).or_default().push(a);
            }
        }
        if has_self_loop {
            continue;
        }
        let mut found_odd = false;
        for &start in adj.keys() {
            let mut dist: HashMap<(u32, u8), u32> = HashMap::new();
            let mut q = VecDeque::new();
            dist.insert((start, 0), 0);
            q.push_back((start, 0_u8));
            while let Some((u, par)) = q.pop_front() {
                let d = dist[&(u, par)];
                if d >= k {
                    continue;
                }
                for &v in adj.get(&u).into_iter().flatten() {
                    let np = par ^ 1;
                    // `d < k` holds here, so this closes an odd walk of length `d + 1 <= k`, which
                    // pads (by back-and-forth steps) to a closed walk of length exactly `k`.
                    if v == start && np == 1 {
                        found_odd = true;
                        break;
                    }
                    if let Entry::Vacant(e) = dist.entry((v, np)) {
                        e.insert(d + 1);
                        q.push_back((v, np));
                    }
                }
                if found_odd {
                    break;
                }
            }
            if found_odd {
                break;
            }
        }
        if !found_odd {
            return Some(format!(
                "[PROVED: odd-gon parity obstruction]: odd {k}-gon cannot close its alternating neighbour cycle"
            ));
        }
    }
    None
}

/// Parse a vertex configuration string like `"3.4.6.4"`, canonicalize under dihedral symmetry,
/// compute exact corner angle defect `Delta`, and reject odd-gon parity obstructions.
pub fn parse_vertex_config(symbol: &str) -> Result<VertexConfigInfo, String> {
    let cleaned = symbol.trim().trim_matches(['(', ')', '[', ']']);
    if cleaned.is_empty() {
        return Err("empty vertex configuration".to_string());
    }
    let mut seq = Vec::new();
    for part in cleaned.split('.') {
        let t = part.trim();
        let val: u32 = t.parse().map_err(|_| format!("invalid face order {t:?} in {symbol:?}"))?;
        if val < 3 {
            return Err(format!("polygon side count must be >= 3 in vertex config, got {val}"));
        }
        seq.push(val);
    }
    if seq.len() < 3 {
        return Err(format!("vertex config requires at least 3 faces, got {}", seq.len()));
    }
    if let Some(err) = odd_gon_parity_obstruction(&seq) {
        return Err(err);
    }
    let (canon_vec, is_chiral) = dihedral_canon_u32(&seq);
    let canonical = canon_vec.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(".");
    // Snubs sr{p,q} have vertex figure 3.3.p.3.q. They are chiral for p != q (snub cube 3^4.4, snub
    // dodecahedron 3^4.5, snub hexagonal 3^4.6). The snub square tiling 3^2.4.3.4 (p = q = 4) is
    // achiral, symmetry p4g = 4*2 (Conway, Burgiel & Goodman-Strauss, *The Symmetries of Things*,
    // 2008, Part I); sr{p,p} is treated likewise (palindromic vertex figure).
    let is_snub_chiral = is_chiral
        || (canon_vec.len() == 5
            && canon_vec[0] == 3
            && canon_vec[1] == 3
            && canon_vec[3] == 3
            && ((canon_vec[2] == 3 && canon_vec[4] >= 4)
                || (canon_vec[2] >= 4 && canon_vec[4] >= 4 && canon_vec[2] != canon_vec[4])));

    let (mut num, mut den) = (2_i128 - seq.len() as i128, 1_i128);
    for &nk in &seq {
        let ni = i128::from(nk);
        let (nn, nd) = (num * ni + 2 * den, den * ni);
        let g = gcd_u128(nn.unsigned_abs(), nd.unsigned_abs()) as i128;
        (num, den) = (nn / g, nd / g);
    }
    let geometry_kind = match num.cmp(&0) {
        std::cmp::Ordering::Greater => GeometryKind::Spherical,
        std::cmp::Ordering::Equal => GeometryKind::Euclidean,
        std::cmp::Ordering::Less => GeometryKind::Hyperbolic,
    };
    let angle_defect_rad = PI * (num as f64) / (den as f64);
    // Descartes: a vertex-transitive spherical polyhedron has V = 4 pi / Delta = 4 den / num vertices.
    if num > 0 && (4 * den) % num != 0 {
        return Err(format!(
            "[PROVED: Descartes obstruction]: {canonical} has angle defect {num}/{den} pi; V = 4 pi / Delta is not an integer"
        ));
    }

    Ok(VertexConfigInfo {
        input: symbol.to_string(),
        canonical,
        sequence: canon_vec,
        is_chiral,
        is_snub_chiral,
        angle_defect_rad,
        angle_defect_pi_num: num,
        angle_defect_pi_den: den,
        geometry_kind,
        realizable: true,
        provenance: "[PROVED] exact angle defect; realisability: necessary local conditions only".to_string(),
    })
}
