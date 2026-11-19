//! 2D common-neighbour analysis (`#038`) and bond-orientational order `psi_n` (`#039`).
//!
//! CNA of a bond `(i, j)`: `n_common` = common neighbours of `i` and `j`, `n_bonds` = bonds among them, `l_chain` =
//! number of bonds in the longest chain (simple path) formed by those bonds. `psi_n(i) = (1/z_i) sum_j exp(i n theta_ij)`,
//! `0` for an isolated site.
//!
//! Sources: CNA signatures after Honeycutt & Andersen, J. Phys. Chem. 91 (1987) 4950 (here restricted to the 2D bond
//! graph); `psi_6` is the bond-orientational order of Halperin & Nelson, Phys. Rev. Lett. 41 (1978) 121. Pristine
//! hand values: triangular `(2, 0, 0)`, kagome `(1, 0, 0)`, square and honeycomb `(0, 0, 0)`; `|psi_n| = 1` exactly on a
//! site whose bond angles are all congruent mod `2 pi / n`.

use std::collections::{BTreeMap, HashSet};


use super::{BondGraph, sub};

/// `(n_common, n_bonds, l_chain)` of one bond.
pub type CnaSignature = (usize, usize, usize);

fn longest_chain(nodes: &[usize], bonds: &[(usize, usize)]) -> usize {
    fn dfs(cur: usize, used: &mut [bool], bonds: &[(usize, usize)]) -> usize {
        let mut best = 0;
        for (k, &(a, b)) in bonds.iter().enumerate() {
            if used[k] || (a != cur && b != cur) {
                continue;
            }
            used[k] = true;
            best = best.max(1 + dfs(if a == cur { b } else { a }, used, bonds));
            used[k] = false;
        }
        best
    }
    let mut used = vec![false; bonds.len()];
    nodes.iter().map(|&s| dfs(s, &mut used, bonds)).max().unwrap_or(0)
}

/// CNA signature of every edge of `g` (same order as `g.edges`) and, per site, the counts of each signature.
pub fn cna_2d(g: &BondGraph) -> (Vec<CnaSignature>, Vec<BTreeMap<CnaSignature, usize>>) {
    let sets: Vec<HashSet<usize>> = g.nbrs.iter().map(|l| l.iter().copied().collect()).collect();
    let edge_set: HashSet<(usize, usize)> = g.edges.iter().copied().collect();
    let mut per_site = vec![BTreeMap::new(); g.pos.len()];
    let sigs = g.edges.iter().map(|&(i, j)| {
        let mut common: Vec<usize> = sets[i].intersection(&sets[j]).copied().collect();
        common.sort_unstable();
        let mut bonds = Vec::new();
        for (a, &x) in common.iter().enumerate() {
            for &y in &common[a + 1..] {
                if edge_set.contains(&(x.min(y), x.max(y))) {
                    bonds.push((x, y));
                }
            }
        }
        let sig = (common.len(), bonds.len(), longest_chain(&common, &bonds));
        *per_site[i].entry(sig).or_insert(0) += 1;
        *per_site[j].entry(sig).or_insert(0) += 1;
        sig
    }).collect();
    (sigs, per_site)
}

/// Bond-orientational order `psi_n` per site as `[re, im]`.
pub fn psi_n(g: &BondGraph, n: u32) -> Vec<[f64; 2]> {
    let nf = f64::from(n);
    g.nbrs.iter().enumerate().map(|(i, lst)| {
        if lst.is_empty() {
            return [0.0, 0.0];
        }
        let (mut re, mut im) = (0.0, 0.0);
        for &j in lst {
            let d = sub(g.pos[j], g.pos[i]);
            let (s, c) = (nf * d[1].atan2(d[0])).sin_cos();
            re += c;
            im += s;
        }
        let z = lst.len() as f64;
        [re / z, im / z]
    }).collect()
}
