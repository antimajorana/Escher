//! Site-permutation groups for solvers: the symmetry group of a finite cluster (orthogonal maps about the
//! centroid) and of an `nx x ny` periodic torus (space group x supercell translations), as sorted,
//! closure-checked lists of site permutations.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::export::solver::{PeriodicBond, ShellClassification, cell_ops_with_action, classify_cell_shells};
use crate::types::Cell2D;

/// Position tolerance used when matching mapped cluster sites.
pub const CLUSTER_TOL: f64 = 1e-4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationGroupReport {
    pub n_sites: usize,
    pub order: usize,
    pub permutations: Vec<Vec<usize>>,
    pub group_closed: bool,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub provenance: crate::defect::burgers_indices::StaticStr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SupercellSolverExport {
    pub group: String,
    pub nx: usize,
    pub ny: usize,
    pub n_sites: usize,
    pub lattice: [[f64; 2]; 2],
    pub positions: Vec<[f64; 3]>,
    pub sublattice: Vec<usize>,
    pub site_labels: Vec<usize>,
    pub shell_distances: Vec<f64>,
    /// Per shell: `(a, b, m)` with `a <= b`, `m` = number of directed lattice bonds `a -> b` on the torus.
    pub shell_edges: Vec<Vec<(usize, usize, u32)>>,
    pub shells: ShellClassification,
    pub report: PermutationGroupReport,
}

/// Exact closure check `g o h in G` for all pairs.
pub fn group_closed(perms: &[Vec<usize>]) -> bool {
    let set: HashSet<&[usize]> = perms.iter().map(Vec::as_slice).collect();
    perms.iter().all(|g| perms.iter().all(|h| set.contains(h.iter().map(|&k| g[k]).collect::<Vec<_>>().as_slice())))
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0] * s, a[1] * s, a[2] * s] }
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Orthonormal frame `(e1, e2, e3)` from a non-zero `a` and an optional `b` not parallel to it.
fn frame(a: [f64; 3], b: Option<[f64; 3]>) -> [[f64; 3]; 3] {
    let e1 = scale3(a, 1.0 / norm3(a));
    let helper = b.unwrap_or(if e1[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] });
    let w = sub3(helper, scale3(e1, dot3(helper, e1)));
    let e2 = scale3(w, 1.0 / norm3(w));
    [e1, e2, cross3(e1, e2)]
}

struct Grid {
    cell: f64,
    map: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(pts: &[[f64; 3]], cell: f64) -> Self {
        let mut map: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(Self::key(p, cell)).or_default().push(i);
        }
        Self { cell, map }
    }
    fn key(p: &[f64; 3], cell: f64) -> (i64, i64, i64) {
        ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)
    }
    fn find(&self, pts: &[[f64; 3]], q: [f64; 3], tol: f64) -> Option<usize> {
        let (kx, ky, kz) = Self::key(&q, self.cell);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &i in self.map.get(&(kx + dx, ky + dy, kz + dz)).into_iter().flatten() {
                        let d = norm3(sub3(pts[i], q));
                        if d < tol && best.is_none_or(|b| d < b.0) {
                            best = Some((d, i));
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    }
}

fn edge_sets(n: usize, shell_edge_lists: &[Vec<(usize, usize)>]) -> Result<Vec<HashSet<(usize, usize)>>, String> {
    shell_edge_lists
        .iter()
        .map(|lst| {
            lst.iter()
                .map(|&(a, b)| if a < n && b < n { Ok((a.min(b), a.max(b))) } else { Err(format!("edge ({a}, {b}) out of range")) })
                .collect()
        })
        .collect()
}

/// Every orthogonal map about the centroid that sends `positions` onto themselves (within [`CLUSTER_TOL`]),
/// keeps `site_labels`, and keeps every undirected edge set of `shell_edge_lists`.
/// Planar clusters (all `z = 0`) reduce to `O(2)`; the `z -> -z` partner of each op gives the same permutation.
pub fn cluster_site_permutations(
    positions: &[[f64; 3]],
    site_labels: &[usize],
    shell_edge_lists: &[Vec<(usize, usize)>],
) -> Result<PermutationGroupReport, String> {
    let n = positions.len();
    if n == 0 || site_labels.len() != n {
        return Err(format!("need >= 1 site and one label per site (got {n} sites, {} labels)", site_labels.len()));
    }
    let tol = CLUSTER_TOL;
    let sets = edge_sets(n, shell_edge_lists)?;
    let mut c = [0.0; 3];
    for p in positions {
        for k in 0..3 {
            c[k] += p[k] / n as f64;
        }
    }
    let x: Vec<[f64; 3]> = positions.iter().map(|&p| sub3(p, c)).collect();
    let r: Vec<f64> = x.iter().map(|&v| norm3(v)).collect();
    let cands = |i: usize, pool: &[usize]| -> Vec<usize> {
        pool.iter().copied().filter(|&q| site_labels[q] == site_labels[i] && (r[q] - r[i]).abs() < tol).collect()
    };
    let all: Vec<usize> = (0..n).collect();
    let off: Vec<usize> = (0..n).filter(|&i| r[i] > 10.0 * tol).collect();
    // reference sites with the fewest candidate images (ties: lowest index)
    let p0 = off.iter().copied().min_by_key(|&i| (cands(i, &all).len(), i));
    let p1 = p0.and_then(|i0| {
        off.iter()
            .copied()
            .filter(|&j| norm3(cross3(x[i0], x[j])) > 1e-3 * r[i0] * r[j])
            .min_by_key(|&j| (cands(j, &all).len(), j))
    });
    let grid = Grid::new(&x, 10.0 * tol);
    let mut found: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut try_map = |o: [[f64; 3]; 3]| {
        let mut perm = Vec::with_capacity(n);
        for (i, v) in x.iter().enumerate() {
            let y = [dot3(o[0], *v), dot3(o[1], *v), dot3(o[2], *v)];
            match grid.find(&x, y, tol) {
                Some(j) if site_labels[j] == site_labels[i] => perm.push(j),
                _ => return,
            }
        }
        if perm.iter().collect::<HashSet<_>>().len() != n {
            return;
        }
        let ok = sets.iter().all(|s| s.iter().all(|&(a, b)| s.contains(&(perm[a].min(perm[b]), perm[a].max(perm[b])))));
        if ok {
            found.insert(perm);
        }
    };
    // O = F^T-composed frames: O = sum_k f_k e_k^T
    let build = |e: [[f64; 3]; 3], f: [[f64; 3]; 3], flip: f64| -> [[f64; 3]; 3] {
        let fk = [f[0], f[1], scale3(f[2], flip)];
        let mut o = [[0.0; 3]; 3];
        for (ek, fv) in e.iter().zip(fk.iter()) {
            for a in 0..3 {
                for b in 0..3 {
                    o[a][b] += fv[a] * ek[b];
                }
            }
        }
        o
    };
    match (p0, p1) {
        (None, _) => {
            found.insert((0..n).collect());
        }
        (Some(i0), None) => {
            let e = frame(x[i0], None);
            for q0 in cands(i0, &all) {
                let f = frame(x[q0], None);
                for flip in [1.0, -1.0] {
                    try_map(build(e, f, flip));
                }
            }
        }
        (Some(i0), Some(i1)) => {
            let e = frame(x[i0], Some(x[i1]));
            let g01 = dot3(x[i0], x[i1]);
            let c1 = cands(i1, &all);
            for q0 in cands(i0, &all) {
                for &q1 in c1.iter().filter(|&&q1| q1 != q0 && (dot3(x[q0], x[q1]) - g01).abs() < tol * (1.0 + r[i0] + r[i1])) {
                    let f = frame(x[q0], Some(x[q1]));
                    for flip in [1.0, -1.0] {
                        try_map(build(e, f, flip));
                    }
                }
            }
        }
    }
    let permutations: Vec<Vec<usize>> = found.into_iter().collect();
    let closed = group_closed(&permutations);
    Ok(PermutationGroupReport {
        n_sites: n,
        order: permutations.len(),
        group_closed: closed,
        permutations,
        provenance: "[COMPUTATIONAL EVIDENCE: geometric search, tol 1e-4; closure checked exactly]",
    })
}

/// Site index on the `nx x ny` torus: `(iy * nx + ix) * n_basis + s` (same order as `export_cell_extxyz`).
pub fn torus_index(nx: usize, nb: usize, ix: usize, iy: usize, s: usize) -> usize {
    (iy * nx + ix) * nb + s
}

fn wrap(v: i64, m: usize) -> usize {
    v.rem_euclid(m as i64) as usize
}

/// Space group of `cell` composed with the torus translations, as site permutations of the `nx x ny` torus,
/// verified to keep the site labels (basis-site orbits) and every shell adjacency (with multiplicities).
pub fn periodic_supercell_permutations(
    group: &str,
    cell: &Cell2D,
    nx: usize,
    ny: usize,
    max_shells: usize,
) -> Result<SupercellSolverExport, String> {
    if nx == 0 || ny == 0 {
        return Err("supercell sizes nx, ny must be >= 1".into());
    }
    let shells = classify_cell_shells(group, cell, max_shells)?;
    let ops = cell_ops_with_action(cell)?;
    let nb = cell.basis.len();
    let n = nx * ny * nb;
    // basis-site orbit label: smallest basis index in {perm_g(s) : g in G}
    let site_label: Vec<usize> = (0..nb).map(|s| ops.iter().map(|(_, perm, _)| perm[s]).min().unwrap_or(s)).collect();
    let idx = |ix: i64, iy: i64, s: usize| torus_index(nx, nb, wrap(ix, nx), wrap(iy, ny), s);
    let mut positions = Vec::with_capacity(n);
    let (mut sublattice, mut site_labels) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for iy in 0..ny {
        for ix in 0..nx {
            for s in 0..nb {
                let (fx, fy) = (ix as f64, iy as f64);
                let p = cell.basis[s];
                positions.push([p[0] + fx * cell.a1[0] + fy * cell.a2[0], p[1] + fx * cell.a1[1] + fy * cell.a2[1], 0.0]);
                sublattice.push(s);
                site_labels.push(site_label[s]);
            }
        }
    }
    let mut adjs: Vec<Vec<u32>> = Vec::new();
    let mut shell_edges = Vec::new();
    for sh in &shells.shells {
        let mut a = vec![0u32; n * n];
        let bonds: Vec<PeriodicBond> = sh.orbits.iter().flat_map(|o| o.bonds.iter().copied()).collect();
        for iy in 0..ny as i64 {
            for ix in 0..nx as i64 {
                for b in &bonds {
                    let u = idx(ix, iy, b.i);
                    let v = idx(ix + i64::from(b.n1), iy + i64::from(b.n2), b.j);
                    a[u * n + v] += 1;
                    a[v * n + u] += 1;
                }
            }
        }
        let mut el = Vec::new();
        for u in 0..n {
            for v in u..n {
                if a[u * n + v] > 0 {
                    el.push((u, v, a[u * n + v]));
                }
            }
        }
        shell_edges.push(el);
        adjs.push(a);
    }
    let mut found: BTreeSet<Vec<usize>> = BTreeSet::new();
    for (op, perm, shift) in &ops {
        let m = op.mf;
        // the op must map the torus sublattice (nx a1, ny a2) onto itself
        let compatible = (m[1][0] as i64 * nx as i64).rem_euclid(ny as i64) == 0
            && (m[0][1] as i64 * ny as i64).rem_euclid(nx as i64) == 0;
        if !compatible {
            continue;
        }
        for t2 in 0..ny as i64 {
            for t1 in 0..nx as i64 {
                let mut p = vec![0usize; n];
                for iy in 0..ny as i64 {
                    for ix in 0..nx as i64 {
                        for s in 0..nb {
                            let jx = i64::from(m[0][0]) * ix + i64::from(m[0][1]) * iy + i64::from(shift[s][0]) + t1;
                            let jy = i64::from(m[1][0]) * ix + i64::from(m[1][1]) * iy + i64::from(shift[s][1]) + t2;
                            p[idx(ix, iy, s)] = idx(jx, jy, perm[s]);
                        }
                    }
                }
                let bij = p.iter().collect::<HashSet<_>>().len() == n;
                let labels_ok = (0..n).all(|i| site_labels[p[i]] == site_labels[i]);
                let adj_ok = adjs.iter().all(|a| (0..n).all(|i| (0..n).all(|j| a[p[i] * n + p[j]] == a[i * n + j])));
                if !(bij && labels_ok && adj_ok) {
                    return Err(format!("space-group op {:?} failed the torus invariance check", op.mf));
                }
                found.insert(p);
            }
        }
    }
    let permutations: Vec<Vec<usize>> = found.into_iter().collect();
    let closed = group_closed(&permutations);
    Ok(SupercellSolverExport {
        group: group.to_string(),
        nx,
        ny,
        n_sites: n,
        lattice: [[nx as f64 * cell.a1[0], nx as f64 * cell.a1[1]], [ny as f64 * cell.a2[0], ny as f64 * cell.a2[1]]],
        positions,
        sublattice,
        site_labels,
        shell_distances: shells.shells.iter().map(|s| s.distance).collect(),
        shell_edges,
        shells,
        report: PermutationGroupReport {
            n_sites: n,
            order: permutations.len(),
            group_closed: closed,
            permutations,
            provenance: "[PROVED: integer space-group action, adjacency invariance and closure checked exactly]",
        },
    })
}
