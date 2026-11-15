//! Crystal builders for domain interfaces (`Wa`), reflection/glide walls (`Wb`, `Wd`),
//! and non-orientable Möbius strip seams (`Wc`), plus rooted 1-WL neighbourhood signatures.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::types::{BuiltCrystal, Cell2D, DirectedBondData, Isometry2};

use super::sym_helper::{
    LatticeGroup, PerfectLattice, SiteKey, dot, eigvec_max_sym, mat2_vec, norm, primitive_cell,
    reflection_across, rot90,
};

fn count_components(n: usize, edges: &[(usize, usize)]) -> usize {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &(u, v) in edges {
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        if ru != rv {
            parent[ru.max(rv)] = ru.min(rv);
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}

fn assemble_crystal(
    pos: Vec<[f64; 2]>,
    ref_pos: Vec<[f64; 2]>,
    basis_idx: Vec<usize>,
    side: Vec<Option<char>>,
    directed: BTreeMap<(usize, usize), DirectedBondData>,
    wall_sigma: Option<Isometry2>,
    is_non_orientable: bool,
) -> BuiltCrystal {
    let n = pos.len();
    let mut edges_set = BTreeSet::new();
    let mut adj = vec![Vec::new(); n];
    for ((u, v), d) in directed {
        edges_set.insert((u.min(v), u.max(v)));
        adj[u].push(d);
    }
    for nbrs in &mut adj {
        nbrs.sort_by_key(|b| b.to);
    }
    BuiltCrystal {
        pos,
        ref_pos,
        basis_idx,
        side,
        cone_sites: Vec::new(),
        edges: edges_set.into_iter().collect(),
        adj,
        wall_sigma,
        is_non_orientable,
    }
}

pub(crate) fn build_perfect_disk(cell: &Cell2D, centre: [f64; 2], r: f64) -> (Vec<SiteKey>, BuiltCrystal) {
    let l = PerfectLattice::new(cell.clone());
    let keys = l.sites_in_disk(centre, r);
    let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| l.pos(k)).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|k| k.2).collect();
    let mut directed = BTreeMap::new();
    for (&key, &i) in &index {
        for (k2, typ) in l.nbrs(key) {
            if let Some(&j) = index.get(&k2)
                && j != i
            {
                directed.insert(
                    (i, j),
                    DirectedBondData {
                        to: j,
                        vec: [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]],
                        kind: typ,
                        seam: None,
                        ang: None,
                        across: false,
                        ghost_stub: None,
                    },
                );
            }
        }
    }
    let side = vec![None; pos.len()];
    let crystal = assemble_crystal(pos.clone(), pos, basis_idx, side, directed, None, false);
    (keys, crystal)
}

/// Bare domain interface (`Wa`): side `'A'` (left of `direction`) keeps the lattice, side `'B'`
/// keeps its `sigma`-image; cross-line bonds are cut without reconnection.
pub fn build_interface(
    cell: &Cell2D,
    sigma: Isometry2,
    point: [f64; 2],
    direction: [f64; 2],
    r: f64,
) -> BuiltCrystal {
    let p_cell = primitive_cell(cell, false);
    let l = PerfectLattice::new(p_cell);
    let ld = norm(direction);
    let u = [direction[0] / ld, direction[1] / ld];
    let n = rot90(u);
    let x0 = point;
    let sinv = sigma.inverse();

    let mut pos = Vec::new();
    let mut ref_pos = Vec::new();
    let mut basis_idx = Vec::new();
    let mut side = Vec::new();
    let mut index: HashMap<(SiteKey, char), usize> = HashMap::new();
    let mut site_list: Vec<(SiteKey, char)> = Vec::new();

    for key in l.sites_in_disk(x0, r) {
        let p = l.pos(key);
        if dot([p[0] - x0[0], p[1] - x0[1]], n) >= -1e-9 {
            let idx = site_list.len();
            index.insert((key, 'A'), idx);
            site_list.push((key, 'A'));
            pos.push(p);
            ref_pos.push(p);
            basis_idx.push(key.2);
            side.push(Some('A'));
        }
    }
    let c2 = sinv.apply(x0);
    for key in l.sites_in_disk(c2, r) {
        let p = l.pos(key);
        let q = sigma.apply(p);
        if dot([q[0] - x0[0], q[1] - x0[1]], n) < -1e-9 {
            let idx = site_list.len();
            index.insert((key, 'B'), idx);
            site_list.push((key, 'B'));
            pos.push(q);
            ref_pos.push(p);
            basis_idx.push(key.2);
            side.push(Some('B'));
        }
    }

    let mut directed = BTreeMap::new();
    for (i, &(key, sc)) in site_list.iter().enumerate() {
        for (k2, typ) in l.nbrs(key) {
            if let Some(&j) = index.get(&(k2, sc))
                && j != i
            {
                directed.insert(
                    (i, j),
                    DirectedBondData {
                        to: j,
                        vec: [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]],
                        kind: typ,
                        seam: None,
                        ang: None,
                        across: false,
                        ghost_stub: None,
                    },
                );
            }
        }
    }
    assemble_crystal(pos, ref_pos, basis_idx, side, directed, Some(sigma), false)
}

/// Straight reflection or glide wall (`Wb`, `Wd`) through `point` along `direction`.
/// Side `'A'` is left of `direction`; side `'B'` is its image under `sigma = reflection_across(point, direction, glide)`.
/// Crossing bonds from `u_A` are redirected to `(u_A, u_B)` with `across = true`.
pub fn build_wall(
    cell: &Cell2D,
    point: [f64; 2],
    direction: [f64; 2],
    glide: f64,
    r: f64,
) -> BuiltCrystal {
    let l = PerfectLattice::new(cell.clone());
    let sig = reflection_across(point, direction, glide);
    let ld = norm(direction);
    let u = [direction[0] / ld, direction[1] / ld];
    let n = rot90(u);
    let x0 = point;
    if l.is_symmetry(sig) {
        return build_perfect_disk(cell, x0, r).1;
    }

    let dist = |p: [f64; 2]| dot([p[0] - x0[0], p[1] - x0[1]], n);
    let mut pos = Vec::new();
    let mut ref_pos = Vec::new();
    let mut basis_idx = Vec::new();
    let mut side = Vec::new();
    let mut index: HashMap<(SiteKey, char), usize> = HashMap::new();
    let mut keys_a = Vec::new();

    for key in l.sites_in_disk(x0, r) {
        let p = l.pos(key);
        if dist(p) < -1e-9 {
            continue;
        }
        let idx_a = pos.len();
        index.insert((key, 'A'), idx_a);
        keys_a.push(key);
        pos.push(p);
        ref_pos.push(p);
        basis_idx.push(key.2);
        side.push(Some('A'));

        let q = sig.apply(p);
        if norm([q[0] - p[0], q[1] - p[1]]) > 1e-9 && norm([q[0] - x0[0], q[1] - x0[1]]) <= r + 1e-9 {
            let idx_b = pos.len();
            index.insert((key, 'B'), idx_b);
            pos.push(q);
            ref_pos.push(p);
            basis_idx.push(key.2);
            side.push(Some('B'));
        }
    }

    let mut raw_edges: BTreeMap<(usize, usize), (usize, bool)> = BTreeMap::new();
    for key in keys_a {
        let p = l.pos(key);
        let on_line = !index.contains_key(&(key, 'B')) && dist(p).abs() < 1e-9;
        for (k2, typ) in l.nbrs(key) {
            let p2 = l.pos(k2);
            if dist(p2) >= -1e-9 {
                for sc in ['A', 'B'] {
                    let a_opt = index
                        .get(&(key, sc))
                        .copied()
                        .or_else(|| on_line.then(|| index.get(&(key, 'A')).copied()).flatten());
                    let k2_on = !index.contains_key(&(k2, 'B')) && dist(p2).abs() < 1e-9;
                    let b_opt = index
                        .get(&(k2, sc))
                        .copied()
                        .or_else(|| k2_on.then(|| index.get(&(k2, 'A')).copied()).flatten());
                    if let (Some(a), Some(bq)) = (a_opt, b_opt)
                        && a != bq
                    {
                        raw_edges.insert((a, bq), (typ, false));
                        raw_edges.insert((bq, a), (typ, false));
                    }
                }
            } else if let (Some(&a), Some(&bq)) = (index.get(&(key, 'A')), index.get(&(key, 'B')))
                && !raw_edges.contains_key(&(a, bq))
            {
                raw_edges.insert((a, bq), (typ, true));
                raw_edges.insert((bq, a), (typ, true));
            }
        }
    }

    let mut directed = BTreeMap::new();
    for ((u_i, v_i), (typ, across)) in raw_edges {
        directed.insert(
            (u_i, v_i),
            DirectedBondData {
                to: v_i,
                vec: [pos[v_i][0] - pos[u_i][0], pos[v_i][1] - pos[u_i][1]],
                kind: typ,
                seam: None,
                ang: None,
                across,
                ghost_stub: None,
            },
        );
    }
    assemble_crystal(pos, ref_pos, basis_idx, side, directed, Some(sig), false)
}

/// Non-orientable Möbius strip seam (`Wc`) glued by reflection/glide symmetry `g` extended over `length` periods.
pub fn build_mobius_seam(cell: &Cell2D, g: Isometry2, length: usize, width: f64) -> BuiltCrystal {
    let l = PerfectLattice::new(cell.clone());
    assert!(g.det() < 0.0, "Möbius seam generator must reverse orientation");
    assert!(l.is_symmetry(g), "Möbius seam generator is not a symmetry of the lattice");

    let mut u = eigvec_max_sym(g.m);
    let n = rot90(u);
    let tn = 0.5 * dot(g.t, n);
    let x0 = [n[0] * tn, n[1] * tn];
    let mut glide = dot(g.t, u);

    let mut per: Option<f64> = None;
    for m0 in -6..=6 {
        for m1 in -6..=6 {
            let v = mat2_vec(l.a, [m0 as f64, m1 as f64]);
            let vu = dot(v, u);
            if dot(v, n).abs() < 1e-9 && vu > 1e-9 && per.is_none_or(|p| vu < p) {
                per = Some(vu);
            }
        }
    }
    let per = per.expect("axis lattice period");
    if glide < 0.0 {
        u = [-u[0], -u[1]];
        glide = -glide;
    }
    let ell = if glide > 1e-9 {
        glide + ((length.saturating_sub(1)) as f64) * per
    } else {
        (length as f64) * per
    };
    let mx0 = mat2_vec(g.m, x0);
    let h = Isometry2 {
        m: g.m,
        t: [x0[0] - mx0[0] + ell * u[0], x0[1] - mx0[1] + ell * u[1]],
    };
    let hinv = h.inverse();
    debug_assert!(l.is_symmetry(h));

    let centre = [x0[0] + 0.5 * ell * u[0], x0[1] + 0.5 * ell * u[1]];
    let mut keys = Vec::new();
    for key in l.sites_in_disk(centre, ell + width + 2.0) {
        let p = l.pos(key);
        let dp = [p[0] - x0[0], p[1] - x0[1]];
        let (a, d) = (dot(dp, u), dot(dp, n));
        if (-1e-9..(ell - 1e-9)).contains(&a) && d.abs() <= width + 1e-9 {
            keys.push(key);
        }
    }
    let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| l.pos(k)).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|k| k.2).collect();

    type RawStripEdge = (i32, usize, [f64; 2], Option<[f64; 2]>);
    let mut raw: BTreeMap<(usize, usize), RawStripEdge> = BTreeMap::new();
    for (i, &key) in keys.iter().enumerate() {
        let p = pos[i];
        for (k2, typ) in l.nbrs(key) {
            let q = l.pos(k2);
            let dq = [q[0] - x0[0], q[1] - x0[1]];
            let (a, d) = (dot(dq, u), dot(dq, n));
            if d.abs() > width + 1e-9 {
                continue;
            }
            let (j_cross, img_opt) = if a >= ell - 1e-9 {
                (1, l.locate(hinv.apply(q)))
            } else if a < -1e-9 {
                (-1, l.locate(h.apply(q)))
            } else {
                (0, Some(k2))
            };
            let Some(img) = img_opt else { continue };
            let Some(&v) = index.get(&img) else { continue };
            if v == i {
                continue;
            }
            let bvec = [q[0] - p[0], q[1] - p[1]];
            let ghost = (j_cross != 0).then_some([p[0] + 0.5 * bvec[0], p[1] + 0.5 * bvec[1]]);
            raw.insert((i, v), (j_cross, typ, bvec, ghost));
            raw.entry((v, i))
                .or_insert((-j_cross, typ, [-bvec[0], -bvec[1]], ghost));
        }
    }

    let n_sites = pos.len();
    let mut base_edges = Vec::new();
    let mut cover_edges = Vec::new();
    let mut directed = BTreeMap::new();
    for ((u_i, v_i), (j_cross, typ, bvec, ghost)) in raw {
        if u_i < v_i {
            base_edges.push((u_i, v_i));
        }
        let flip = (j_cross.unsigned_abs() % 2) as usize;
        cover_edges.push((2 * u_i, 2 * v_i + flip));
        cover_edges.push((2 * u_i + 1, 2 * v_i + (1 - flip)));
        let seam = match j_cross {
            1 => Some(hinv),
            -1 => Some(h),
            _ => None,
        };
        directed.insert(
            (u_i, v_i),
            DirectedBondData {
                to: v_i,
                vec: bvec,
                kind: typ,
                seam,
                ang: None,
                across: j_cross != 0,
                ghost_stub: ghost,
            },
        );
    }
    let orientable = count_components(2 * n_sites, &cover_edges) == 2 * count_components(n_sites, &base_edges);
    let side = vec![None; n_sites];
    assemble_crystal(pos.clone(), pos, basis_idx, side, directed, None, !orientable)
}

#[inline]
fn mix128(mut x: u128) -> u128 {
    x ^= x >> 64;
    x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835);
    x ^= x >> 63;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9_94d0_49bb_1331_11eb);
    x ^= x >> 61;
    x
}

/// Canonical `(r + 1)`-iteration 1-WL signature of the induced `r`-ego-graph rooted at `root`.
pub(crate) fn rooted_wl_sig(adj: &[Vec<usize>], root: usize, r: usize) -> Vec<u128> {
    let mut dist: HashMap<usize, usize> = HashMap::new();
    let mut q = VecDeque::new();
    dist.insert(root, 0);
    q.push_back(root);
    while let Some(u) = q.pop_front() {
        let du = dist[&u];
        if du == r {
            continue;
        }
        for &w in &adj[u] {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                e.insert(du + 1);
                q.push_back(w);
            }
        }
    }
    let mut nodes: Vec<usize> = dist.into_keys().collect();
    nodes.sort_unstable();
    let local_of: HashMap<usize, usize> = nodes.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    let m = nodes.len();
    let mut sub_adj = vec![Vec::new(); m];
    for (ui, &u) in nodes.iter().enumerate() {
        for &w in &adj[u] {
            if let Some(&wi) = local_of.get(&w) {
                sub_adj[ui].push(wi);
            }
        }
    }
    let mut col: Vec<u128> = nodes
        .iter()
        .map(|&v| if v == root { 0x1111_2222_3333_4444_5555_6666_7777_8889 } else { 0xaaaa_bbbb_cccc_dddd_eeee_ffff_0123_4567 })
        .collect();
    let mut next_col = vec![0u128; m];
    let mut nbr_cols = Vec::new();
    for _ in 0..=r {
        for ui in 0..m {
            nbr_cols.clear();
            nbr_cols.extend(sub_adj[ui].iter().map(|&wi| col[wi]));
            nbr_cols.sort_unstable();
            let mut h = mix128(col[ui] ^ ((nbr_cols.len() as u128) << 64));
            for &nc in &nbr_cols {
                h = mix128(h.wrapping_add(nc));
            }
            next_col[ui] = h;
        }
        std::mem::swap(&mut col, &mut next_col);
    }
    col.sort_unstable();
    col
}

pub(crate) fn simple_adj(crystal: &BuiltCrystal) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); crystal.pos.len()];
    for &(u, v) in &crystal.edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    adj
}

pub(crate) fn perfect_wl_sigs(grp: &LatticeGroup, r: usize) -> BTreeSet<Vec<u128>> {
    let patch_r = (r as f64 + 2.0) * grp.lmax + 2.0 * grp.scale;
    let (keys, crystal) = build_perfect_disk(&grp.cell, [0.0, 0.0], patch_r);
    let adj = simple_adj(&crystal);
    keys.iter()
        .enumerate()
        .filter(|(_, k)| k.0 == 0 && k.1 == 0)
        .map(|(i, _)| rooted_wl_sig(&adj, i, r))
        .collect()
}

pub(crate) fn count_network_irregular(
    grp: &LatticeGroup,
    crystal: &BuiltCrystal,
    x0: [f64; 2],
    u: [f64; 2],
    period: f64,
    r: usize,
    perfect_sigs: &BTreeSet<Vec<u128>>,
) -> usize {
    let lu = norm(u);
    let uh = [u[0] / lu, u[1] / lu];
    let n = rot90(uh);
    let mut seeds_set = BTreeSet::new();
    for (i, nbrs) in crystal.adj.iter().enumerate() {
        if nbrs.iter().any(|b| b.across) {
            seeds_set.insert(i);
        }
    }
    for (i, &p) in crystal.pos.iter().enumerate() {
        if dot([p[0] - x0[0], p[1] - x0[1]], n).abs() < 1e-7 {
            seeds_set.insert(i);
        }
    }
    let seeds: Vec<usize> = seeds_set
        .into_iter()
        .filter(|&i| {
            let p = crystal.pos[i];
            dot([p[0] - x0[0], p[1] - x0[1]], uh).abs() <= 0.5 * period + 1e-6
        })
        .collect();

    let adj = simple_adj(crystal);
    let mut band = BTreeSet::new();
    let mut q = VecDeque::new();
    for s in seeds {
        let mut dist: HashMap<usize, usize> = HashMap::new();
        dist.insert(s, 0);
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            band.insert(v);
            let dv = dist[&v];
            if dv < r {
                for &w in &adj[v] {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                        e.insert(dv + 1);
                        q.push_back(w);
                    }
                }
            }
        }
    }

    let rd = crystal
        .pos
        .iter()
        .map(|&p| norm([p[0] - x0[0], p[1] - x0[1]]))
        .fold(0.0_f64, f64::max);
    for &i in &band {
        let p = crystal.pos[i];
        assert!(
            norm([p[0] - x0[0], p[1] - x0[1]]) <= rd - (r as f64 + 1.0) * grp.lmax,
            "wall patch too small for the neighbourhood test; increase R"
        );
    }

    band.into_iter()
        .filter(|&i| {
            adj[i].len() != grp.l.rule[crystal.basis_idx[i]].len()
                || !perfect_sigs.contains(&rooted_wl_sig(&adj, i, r))
        })
        .count()
}
