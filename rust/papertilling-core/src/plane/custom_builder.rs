//! Custom periodic 2D lattice builder from wallpaper group + Wyckoff seed points +
//! Voronoi or distance-shell bond rules (`min_degree` in `{2, 3}` supported).

use std::collections::{BTreeSet, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::dsymbol::{self, DSymbolKey};
use crate::plane::basis_reduce::ReducedBasis2D;
use crate::plane::cell::{canon_bond, mat2_inv, mat2_mul, mat2_vec};
use crate::plane::family::LatticeFamily;
use crate::plane::groups::{conventional_cell, group_spec, prim_ops, prim_u};
use crate::plane::voronoi::voronoi_bonds;
use crate::types::{Cell2D, Frame2D, Recipe};

/// How bonds are generated for a custom site configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondSelection {
    Voronoi,
    ShellMask(u32),
}

/// User specification for building a custom periodic lattice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomLatticeSpec {
    pub group: String,
    pub a: f64,
    pub b: Option<f64>,
    pub gamma: Option<f64>,
    pub seeds: Vec<[f64; 2]>,
    pub bond_mode: BondSelection,
    pub min_degree: usize,
}

/// Result of constructing and analyzing a custom lattice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomLatticeResult {
    pub recipe: Recipe,
    pub cell: Cell2D,
    pub frame: Frame2D,
    pub dsymbol_key: Option<DSymbolKey>,
    pub network_key: Option<DSymbolKey>,
    pub symmetry_index: Option<usize>,
    pub coordination: String,
    pub num_sites: usize,
    pub num_bonds: usize,
    pub site_orbits: usize,
    pub bond_orbits: usize,
    pub shell_distances: Vec<f64>,
    pub planar: bool,
    pub status: String,
}

fn round12(x: f64) -> f64 {
    (x * 1e12).round_ties_even() * 1e-12
}

type BareBond = (usize, usize, [i32; 2]);
type DistanceShell = (f64, Vec<BareBond>);

/// Compute up to `max_shells` shortest unique periodic inter-site distance shells
/// using Lagrange-Gauss basis reduction and scale-invariant tolerances.
pub fn compute_distance_shells(
    prim_sites: &[[f64; 2]],
    a1: [f64; 2],
    a2: [f64; 2],
    max_shells: usize,
) -> Vec<DistanceShell> {
    let n = prim_sites.len();
    if n == 0 {
        return Vec::new();
    }
    let Ok(rb) = ReducedBasis2D::reduce(a1, a2) else {
        return Vec::new();
    };
    let (wrapped, floors) = rb.wrap_sites(prim_sites);
    let cart: Vec<[f64; 2]> = wrapped
        .iter()
        .map(|f| [f[0] * rb.b1[0] + f[1] * rb.b2[0], f[0] * rb.b1[1] + f[1] * rb.b2[1]])
        .collect();
    let (m1_max, m2_max, _) = rb.covering_shift_bounds();
    let (m1_lim, m2_lim) = (m1_max.max(3), m2_max.max(3));
    let l0 = (rb.area() / (n as f64)).sqrt().max(1e-30);
    let min_dist = 1e-5 * l0;
    let shell_tol = 1e-4 * l0;

    let mut seen: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
    let mut raw: Vec<(f64, usize, usize, i32, i32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for m1 in -m1_lim..=m1_lim {
                for m2 in -m2_lim..=m2_lim {
                    let sh = rb.map_shift_to_orig(floors[i], floors[j], [m1, m2]);
                    if j == i && sh == [0, 0] {
                        continue;
                    }
                    let q = [
                        cart[j][0] + f64::from(m1) * rb.b1[0] + f64::from(m2) * rb.b2[0],
                        cart[j][1] + f64::from(m1) * rb.b1[1] + f64::from(m2) * rb.b2[1],
                    ];
                    let dist = (q[0] - cart[i][0]).hypot(q[1] - cart[i][1]);
                    if dist > min_dist {
                        let cb = canon_bond(i, j, sh[0], sh[1], 0);
                        if seen.insert((cb.i, cb.j, cb.n1, cb.n2)) {
                            raw.push((dist, cb.i, cb.j, cb.n1, cb.n2));
                        }
                    }
                }
            }
        }
    }
    raw.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut shells: Vec<DistanceShell> = Vec::new();
    for (d, i, j, n1, n2) in raw {
        if let Some(last) = shells.last_mut()
            && (d - last.0).abs() <= shell_tol
        {
            last.1.push((i, j, [n1, n2]));
            continue;
        }
        if shells.len() >= max_shells {
            break;
        }
        shells.push((d, vec![(i, j, [n1, n2])]));
    }
    shells
}

fn canon_bare(i: usize, j: usize, n1: i32, n2: i32) -> (usize, usize, i32, i32) {
    let b = canon_bond(i, j, n1, n2, 0);
    (b.i, b.j, b.n1, b.n2)
}

/// Group undirected primitive bonds by wallpaper symmetry and emit one conv-frac pair per orbit.
fn collapse_bond_orbits(
    group: &str,
    prim_sites: &[[f64; 2]],
    selected: &[(usize, usize, [i32; 2])],
) -> Result<Vec<[[f64; 2]; 2]>, String> {
    let ops = prim_ops(group)?;
    let u_mat = prim_u(group)?;
    let is_int_v = |d: [f64; 2]| {
        (d[0] - d[0].round_ties_even()).abs() < 1e-5 && (d[1] - d[1].round_ties_even()).abs() < 1e-5
    };
    let mut perm = Vec::with_capacity(ops.len());
    let mut shift = Vec::with_capacity(ops.len());
    for op in &ops {
        let mut pk = Vec::with_capacity(prim_sites.len());
        let mut sk = Vec::with_capacity(prim_sites.len());
        for &fk in prim_sites {
            let g = [
                f64::from(op.mf[0][0]) * fk[0] + f64::from(op.mf[0][1]) * fk[1] + op.t[0],
                f64::from(op.mf[1][0]) * fk[0] + f64::from(op.mf[1][1]) * fk[1] + op.t[1],
            ];
            let (j, c) = prim_sites
                .iter()
                .enumerate()
                .find_map(|(j, &fj)| {
                    let d = [g[0] - fj[0], g[1] - fj[1]];
                    is_int_v(d).then_some((j, [d[0].round_ties_even() as i32, d[1].round_ties_even() as i32]))
                })
                .ok_or_else(|| "site orbit is not closed under group".to_string())?;
            pk.push(j);
            sk.push(c);
        }
        perm.push(pk);
        shift.push(sk);
    }

    let mut covered: HashSet<(usize, usize, i32, i32)> = HashSet::new();
    let mut reps = Vec::new();
    for &(i, j, n) in selected {
        let cb = canon_bare(i, j, n[0], n[1]);
        if covered.contains(&cb) {
            continue;
        }
        let mut orb = Vec::new();
        for (gi, op) in ops.iter().enumerate() {
            let dn0 = op.mf[0][0] * cb.2 + op.mf[0][1] * cb.3 + shift[gi][cb.1][0] - shift[gi][cb.0][0];
            let dn1 = op.mf[1][0] * cb.2 + op.mf[1][1] * cb.3 + shift[gi][cb.1][1] - shift[gi][cb.0][1];
            let target = canon_bare(perm[gi][cb.0], perm[gi][cb.1], dn0, dn1);
            orb.push(target);
        }
        for t in orb {
            covered.insert(t);
        }
        let p = mat2_vec(u_mat, prim_sites[cb.0]);
        let q = mat2_vec(
            u_mat,
            [prim_sites[cb.1][0] + f64::from(cb.2), prim_sites[cb.1][1] + f64::from(cb.3)],
        );
        let sh = [(p[0] + 1e-9).floor(), (p[1] + 1e-9).floor()];
        reps.push([
            [round12(p[0] - sh[0]), round12(p[1] - sh[1])],
            [round12(q[0] - sh[0]), round12(q[1] - sh[1])],
        ]);
    }
    Ok(reps)
}

/// Check quotient connectedness, `min_degree`, and 2D periodic lattice spanning.
pub fn check_periodic_connected(cell: &Cell2D, min_degree: usize) -> Result<String, String> {
    let n = cell.basis.len();
    if n == 0 || cell.bonds.is_empty() {
        return Err("cell has no sites or bonds".to_string());
    }
    let mut adj: Vec<Vec<(usize, [i32; 2])>> = vec![Vec::new(); n];
    for b in &cell.bonds {
        adj[b.i].push((b.j, [b.n1, b.n2]));
        adj[b.j].push((b.i, [-b.n1, -b.n2]));
    }
    let mut degs: BTreeSet<usize> = BTreeSet::new();
    for (u, nbrs) in adj.iter().enumerate() {
        if nbrs.len() < min_degree {
            return Err(format!("site {u} has degree {} < min_degree {min_degree}", nbrs.len()));
        }
        degs.insert(nbrs.len());
    }
    let mut visited = vec![false; n];
    let mut offset = vec![[0_i32; 2]; n];
    let mut q = VecDeque::from([0_usize]);
    visited[0] = true;
    let mut cycle_vecs: Vec<[i32; 2]> = Vec::new();
    while let Some(u) = q.pop_front() {
        for &(v, dv) in &adj[u] {
            let next_off = [offset[u][0] + dv[0], offset[u][1] + dv[1]];
            if !visited[v] {
                visited[v] = true;
                offset[v] = next_off;
                q.push_back(v);
            } else {
                let cyc = [next_off[0] - offset[v][0], next_off[1] - offset[v][1]];
                if cyc != [0, 0] {
                    cycle_vecs.push(cyc);
                }
            }
        }
    }
    if !visited.iter().all(|&v| v) {
        return Err("periodic net has disconnected components".to_string());
    }
    fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let mut det_gcd = 0_i64;
    for i in 0..cycle_vecs.len() {
        for j in (i + 1)..cycle_vecs.len() {
            let det = i64::from(cycle_vecs[i][0]) * i64::from(cycle_vecs[j][1])
                - i64::from(cycle_vecs[i][1]) * i64::from(cycle_vecs[j][0]);
            if det != 0 {
                det_gcd = gcd_i64(det_gcd, det);
            }
        }
    }
    if det_gcd == 0 {
        return Err("periodic net is 0D/1D (does not span 2D cell)".to_string());
    }
    if det_gcd > 1 {
        return Err(format!("periodic net has {det_gcd} disconnected interpenetrating sublattices"));
    }
    Ok(degs.into_iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","))
}

/// Build, validate, and analyze a custom lattice in one call.
pub fn build_custom_lattice(spec: &CustomLatticeSpec) -> Result<CustomLatticeResult, String> {
    let gs = group_spec(&spec.group)?;
    let canon = gs.symbol;
    if spec.seeds.is_empty() {
        return Err("provide at least one seed coordinate".to_string());
    }
    let mut accepted_seeds: Vec<[f64; 2]> = Vec::new();
    for &seed in &spec.seeds {
        let mut cand = accepted_seeds.clone();
        cand.push([seed[0].rem_euclid(1.0), seed[1].rem_euclid(1.0)]);
        let test_rec = Recipe {
            group: canon.to_string(),
            sites: cand.clone(),
            bonds: Vec::new(),
            a: spec.a,
            b: spec.b,
            gamma: spec.gamma,
        };
        if LatticeFamily::from_recipe(&test_rec, None).is_ok() {
            accepted_seeds = cand;
        }
    }
    if accepted_seeds.is_empty() {
        return Err("could not accept any valid seed points".to_string());
    }
    let seed_recipe = Recipe {
        group: canon.to_string(),
        sites: accepted_seeds.clone(),
        bonds: Vec::new(),
        a: spec.a,
        b: spec.b,
        gamma: spec.gamma,
    };
    let fam0 = LatticeFamily::from_recipe(&seed_recipe, None)?;
    let cell0 = fam0.cell();
    let prim_sites = fam0.prim_sites();
    let shells = compute_distance_shells(&prim_sites, cell0.a1, cell0.a2, 6);
    let shell_distances: Vec<f64> = shells.iter().map(|s| s.0).collect();

    let selected_bonds: Vec<(usize, usize, [i32; 2])> = match spec.bond_mode {
        BondSelection::Voronoi => voronoi_bonds(&prim_sites, cell0.a1, cell0.a2, 1e-7),
        BondSelection::ShellMask(mask) => {
            let mut out = Vec::new();
            for (idx, (_, sh_bonds)) in shells.iter().enumerate() {
                if (mask >> idx) & 1 == 1 {
                    out.extend(sh_bonds.iter().copied());
                }
            }
            out
        }
    };
    if selected_bonds.is_empty() {
        return Err("no bonds selected under current rule".to_string());
    }
    let bond_reps = collapse_bond_orbits(canon, &prim_sites, &selected_bonds)?;
    let recipe = Recipe {
        group: canon.to_string(),
        sites: accepted_seeds,
        bonds: bond_reps,
        a: spec.a,
        b: spec.b,
        gamma: spec.gamma,
    };
    let fam = LatticeFamily::from_recipe(&recipe, None)?;
    let cell = fam.cell();
    let coordination = check_periodic_connected(&cell, spec.min_degree.max(1))?;

    let (a1c, a2c) = conventional_cell(canon, spec.a, spec.b, spec.gamma)?;
    let _ = (mat2_mul, mat2_inv, a1c, a2c);
    let (dsymbol_key, network_key, symmetry_index, planar, status) =
        if cell.has_crossings() || cell.clearance() <= 1e-6 {
            (
                None,
                None,
                None,
                false,
                "Connected non-planar (bond crossing or bond through site)".to_string(),
            )
        } else {
            match fam.dsymbol() {
                Ok(ds) => {
                    let ck = ds.canonical_key().map_err(|e| e.to_string())?;
                    let nk = dsymbol::network_key(&ck).map_err(|e| e.to_string())?;
                    let si = dsymbol::symmetry_index(&ck).map_err(|e| e.to_string())?;
                    let msg = format!(
                        "Planar ({}, |D|={}, sym_idx={}, {coordination}-coord)",
                        canon,
                        ds.size,
                        si
                    );
                    (Some(ck), Some(nk), Some(si), true, msg)
                }
                Err(e) => (
                    None,
                    None,
                    None,
                    false,
                    format!("Connected non-planar ({e})"),
                ),
            }
        };
    Ok(CustomLatticeResult {
        num_sites: cell.basis.len(),
        num_bonds: cell.bonds.len(),
        site_orbits: fam.gens.len(),
        bond_orbits: fam.orbits.len(),
        recipe,
        cell,
        frame: Frame2D::default(),
        dsymbol_key,
        network_key,
        symmetry_index,
        coordination,
        shell_distances,
        planar,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bond2D;

    #[test]
    fn interpenetrating_sublattices_and_bond_through_site_are_rejected() {
        let interpenetrating = Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                Bond2D { i: 0, j: 0, n1: 2, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 0, n1: 0, n2: 2, kind: 0 },
            ],
        };
        let err = check_periodic_connected(&interpenetrating, 3).unwrap_err();
        assert!(err.contains("4 disconnected interpenetrating sublattices"), "{err}");

        // pm with 2 mirror sites at (0, 0) and (0.5, 0) and b=0.9: shells are 0.5 (x), 0.9 (y), 1.0 (x skipping 0.5).
        let res = build_custom_lattice(&CustomLatticeSpec {
            group: "pm".to_string(),
            a: 1.0,
            b: Some(0.9),
            gamma: None,
            seeds: vec![[0.0, 0.0], [0.5, 0.0]],
            bond_mode: BondSelection::ShellMask(0b111),
            min_degree: 3,
        })
        .unwrap();
        assert!(res.cell.clearance() <= 1e-6);
        assert!(!res.planar, "bond passing through site must be marked non-planar: {}", res.status);
    }
}
