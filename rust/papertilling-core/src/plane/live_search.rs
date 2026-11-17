//! Live minimal-lattice enumeration across higher site counts `N > 7` (up to `N = 25`),
//! supporting `min_degree = 2` (decorated / Lieb-like nets) and `min_degree = 3` (tilings).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::dsymbol::DSymbolKey;
use crate::plane::custom_builder::{build_custom_lattice, BondSelection, CustomLatticeSpec};
use crate::plane::groups::{canonical_symbol, wyckoff_table};
use crate::types::{Frame2D, Recipe};

/// One discovered periodic lattice from live Wyckoff/bond-shell enumeration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredLattice {
    pub recipe: Recipe,
    pub frame: Frame2D,
    pub num_sites: usize,
    pub num_bonds: usize,
    pub site_orbits: usize,
    pub bond_orbits: usize,
    pub coordination: String,
    pub dsymbol_key: Option<DSymbolKey>,
    pub network_key: Option<DSymbolKey>,
    pub symmetry_index: usize,
    pub planar: bool,
    pub bond_mode_label: String,
    pub status: String,
}

fn eval_wyckoff_samples(row: &crate::plane::groups::WyckoffRow) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    match row.kind.as_str() {
        "point" => out.push([row.base[0].rem_euclid(1.0), row.base[1].rem_euclid(1.0)]),
        "line" => {
            for t in [0.19, 0.37, 0.13] {
                let x = (row.base[0] + row.d_mat[0][0] * t).rem_euclid(1.0);
                let y = (row.base[1] + row.d_mat[1][0] * t).rem_euclid(1.0);
                out.push([x, y]);
            }
        }
        _ => {
            for (u, v) in [(0.13, 0.29), (0.19, 0.41), (0.27, 0.11)] {
                out.push([u, v]);
            }
        }
    }
    out
}

/// Live search in `group` for connected minimal periodic lattices with site count `N in n_min..=n_max`.
pub fn search_minimal_lattices(
    group: &str,
    n_min: usize,
    n_max: usize,
    min_degree: usize,
    max_results: usize,
) -> Result<Vec<DiscoveredLattice>, String> {
    let canon = canonical_symbol(group)?;
    let n_lo = n_min.max(1);
    let n_hi = n_max.clamp(n_lo, 25);
    let rows = wyckoff_table(canon)?;

    // Build discrete weighted Wyckoff orbit pool `(mult, coord)`.
    let mut pool: Vec<(usize, [f64; 2])> = Vec::new();
    for r in &rows {
        let m = usize::from(r.multiplicity);
        if m > n_hi {
            continue;
        }
        let samples = eval_wyckoff_samples(r);
        let take = if r.kind == "point" { 1 } else { samples.len().min(3) };
        for s in samples.into_iter().take(take) {
            pool.push((m, s));
        }
    }

    // Collect up to ~120 non-decreasing seed combinations whose multiplicity sum falls in n_lo..=n_hi.
    let mut seed_sets: Vec<Vec<[f64; 2]>> = Vec::new();
    fn dfs_combos(
        pool: &[(usize, [f64; 2])],
        start: usize,
        depth: usize,
        cur_n: usize,
        cur_pts: &mut Vec<[f64; 2]>,
        bounds: (usize, usize),
        out: &mut Vec<Vec<[f64; 2]>>,
    ) {
        let (n_lo, n_hi) = bounds;
        if out.len() >= 120 {
            return;
        }
        if cur_n >= n_lo && cur_n <= n_hi && !cur_pts.is_empty() {
            out.push(cur_pts.clone());
        }
        if cur_n >= n_hi || depth >= 4 {
            return;
        }
        for (i, &(m, pt)) in pool.iter().enumerate().skip(start) {
            if cur_n + m <= n_hi && !cur_pts.contains(&pt) {
                cur_pts.push(pt);
                dfs_combos(pool, i + 1, depth + 1, cur_n + m, cur_pts, bounds, out);
                cur_pts.pop();
                if out.len() >= 120 {
                    return;
                }
            }
        }
    }
    let mut cur_pts = Vec::new();
    dfs_combos(&pool, 0, 0, 0, &mut cur_pts, (n_lo, n_hi), &mut seed_sets);
    seed_sets.sort_by_key(Vec::len);

    let bond_modes = [
        ("voronoi", BondSelection::Voronoi),
        ("shell-1", BondSelection::ShellMask(0b001)),
        ("shell-1+2", BondSelection::ShellMask(0b011)),
        ("shell-2", BondSelection::ShellMask(0b010)),
    ];

    let mut seen_net_keys: BTreeSet<DSymbolKey> = BTreeSet::new();
    let mut seen_nonplanar: BTreeSet<(usize, usize, String)> = BTreeSet::new();
    let mut results: Vec<DiscoveredLattice> = Vec::new();

    for seeds in &seed_sets {
        for &(bm_lbl, bond_mode) in &bond_modes {
            if results.len() >= max_results {
                break;
            }
            let spec = CustomLatticeSpec {
                group: canon.to_string(),
                a: 1.0,
                b: None,
                gamma: None,
                seeds: seeds.clone(),
                bond_mode,
                min_degree,
            };
            let Ok(res) = build_custom_lattice(&spec) else { continue };
            if res.num_sites < n_lo || res.num_sites > n_hi {
                continue;
            }
            let is_new = if let Some(nk) = &res.network_key {
                seen_net_keys.insert(nk.clone())
            } else {
                seen_nonplanar.insert((res.num_sites, res.num_bonds, res.coordination.clone()))
            };
            if is_new {
                results.push(DiscoveredLattice {
                    recipe: res.recipe,
                    frame: res.frame,
                    num_sites: res.num_sites,
                    num_bonds: res.num_bonds,
                    site_orbits: res.site_orbits,
                    bond_orbits: res.bond_orbits,
                    coordination: res.coordination,
                    dsymbol_key: res.dsymbol_key,
                    network_key: res.network_key,
                    symmetry_index: res.symmetry_index.unwrap_or(1),
                    planar: res.planar,
                    bond_mode_label: bm_lbl.to_string(),
                    status: res.status,
                });
            }
        }
        if results.len() >= max_results {
            break;
        }
    }

    results.sort_by_key(|d| (!d.planar, d.num_sites, d.bond_orbits, d.num_bonds));
    results.truncate(max_results);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_builder_and_live_search_find_degree2_and_n_gt_7() {
        // Degree-2 Lieb lattice in p4m: (0,0) + (1/2, 0) -> 3 sites, coord "2,4", planar |D|=2.
        let lieb = build_custom_lattice(&CustomLatticeSpec {
            group: "p4m".to_string(),
            a: 1.0,
            b: None,
            gamma: None,
            seeds: vec![[0.0, 0.0], [0.5, 0.0]],
            bond_mode: BondSelection::ShellMask(0b001),
            min_degree: 2,
        })
        .unwrap();
        assert_eq!(lieb.num_sites, 3);
        assert_eq!(lieb.coordination, "2,4");
        assert!(lieb.planar);

        let high_n = search_minimal_lattices("p4m", 8, 12, 2, 12).unwrap();
        assert!(!high_n.is_empty(), "expected discovered lattices in p4m N=8..12");
        assert!(high_n.iter().all(|r| r.num_sites >= 8 && r.num_sites <= 12));

        let hex_n = search_minimal_lattices("p6m", 7, 14, 3, 12).unwrap();
        assert!(!hex_n.is_empty(), "expected discovered degree>=3 lattices in p6m N=7..14");
    }
}
