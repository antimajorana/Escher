//! Uniform-weight Tutte barycentric solvers (local disk repair and global convex-rim embedding).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use faer::Mat;
use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};

use crate::defect::core_graph::{TWO_PI, add2, cross2, dot2, norm2, sub2};
use crate::plane::planarity::{component_roots, signed_area};
use crate::types::BuiltCrystal;

/// Solve a sparse symmetric positive-definite Laplacian system `L * x = rhs` via sparse Cholesky.
pub fn solve_tutte_spd(laplacian: &SparseColMat<usize, f64>, rhs: &Mat<f64>) -> Option<Mat<f64>> {
    let llt = laplacian.sp_cholesky(faer::Side::Lower).ok()?;
    Some(llt.solve(rhs))
}

pub(crate) fn undirected_nbrs(crystal: &BuiltCrystal) -> Vec<BTreeSet<usize>> {
    let mut nbr = vec![BTreeSet::new(); crystal.pos.len()];
    for &(u, v) in &crystal.edges {
        nbr[u].insert(v);
        nbr[v].insert(u);
    }
    nbr
}

pub(crate) fn articulation_points(nbr: &[BTreeSet<usize>]) -> Vec<usize> {
    let n = nbr.len();
    let mut disc = vec![0usize; n];
    let mut low = vec![0usize; n];
    let mut timer = 0usize;
    let mut is_art = vec![false; n];

    fn dfs(
        u: usize,
        parent: Option<usize>,
        nbr: &[BTreeSet<usize>],
        disc: &mut [usize],
        low: &mut [usize],
        timer: &mut usize,
        is_art: &mut [bool],
    ) {
        *timer += 1;
        disc[u] = *timer;
        low[u] = *timer;
        let mut children = 0usize;
        for &v in &nbr[u] {
            if Some(v) == parent {
                continue;
            }
            if disc[v] != 0 {
                low[u] = low[u].min(disc[v]);
            } else {
                children += 1;
                dfs(v, Some(u), nbr, disc, low, timer, is_art);
                low[u] = low[u].min(low[v]);
                if parent.is_some() && low[v] >= disc[u] {
                    is_art[u] = true;
                }
            }
        }
        if parent.is_none() && children > 1 {
            is_art[u] = true;
        }
    }

    for i in 0..n {
        if disc[i] == 0 {
            dfs(i, None, nbr, &mut disc, &mut low, &mut timer, &mut is_art);
        }
    }
    (0..n).filter(|&i| is_art[i]).collect()
}

fn hanging_pieces(nbr: &[BTreeSet<usize>], fixed: &[bool]) -> Vec<(usize, HashSet<usize>)> {
    let n = nbr.len();
    let mut pieces: Vec<(usize, HashSet<usize>)> = Vec::new();
    for a in articulation_points(nbr) {
        let mut seen = vec![false; n];
        seen[a] = true;
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut comp = HashSet::from([start]);
            let mut dq = VecDeque::from([start]);
            seen[start] = true;
            let mut touches_fixed = fixed[start];
            let mut touches_a = nbr[a].contains(&start);
            while let Some(u) = dq.pop_front() {
                for &w in &nbr[u] {
                    if !seen[w] {
                        seen[w] = true;
                        comp.insert(w);
                        touches_fixed |= fixed[w];
                        touches_a |= nbr[a].contains(&w);
                        dq.push_back(w);
                    }
                }
            }
            if !touches_fixed && touches_a && comp.len() < n / 2 + 1 {
                pieces.push((a, comp));
            }
        }
    }
    pieces.sort_by_key(|p| std::cmp::Reverse(p.1.len()));
    let mut taken = HashSet::new();
    let mut out = Vec::new();
    for (a, comp) in pieces {
        if !taken.contains(&a) && comp.is_disjoint(&taken) {
            taken.extend(comp.iter().copied());
            out.push((a, comp));
        }
    }
    out
}

/// Uniform-weight Tutte barycentric solve with 1-cut hanging-piece and 2-cut similarity-flap protection.
pub fn tutte_solve(
    crystal: &BuiltCrystal,
    p0: &[[f64; 2]],
    fixed: &[bool],
    anchor: &HashMap<usize, f64>,
) -> Vec<[f64; 2]> {
    let n = p0.len();
    let mut p = p0.to_vec();
    let nbr = undirected_nbrs(crystal);
    let pieces = if fixed.iter().all(|&b| b) { Vec::new() } else { hanging_pieces(&nbr, fixed) };
    let mut gone: HashSet<usize> = pieces.iter().flat_map(|(_, c)| c.iter().copied()).collect();
    let mut flaps: Vec<([usize; 2], HashSet<usize>)> = Vec::new();
    let mut seen = gone.clone();
    for v0 in 0..n {
        if fixed[v0] || seen.contains(&v0) {
            continue;
        }
        let mut comp = HashSet::from([v0]);
        let mut stack = vec![v0];
        let mut att = BTreeSet::new();
        seen.insert(v0);
        while let Some(u) = stack.pop() {
            for &w in &nbr[u] {
                if fixed[w] {
                    att.insert(w);
                } else if !gone.contains(&w) && seen.insert(w) {
                    comp.insert(w);
                    stack.push(w);
                }
            }
        }
        if att.len() == 2 {
            let mut it = att.into_iter();
            gone.extend(comp.iter().copied());
            flaps.push(([it.next().unwrap(), it.next().unwrap()], comp));
        } else if att.is_empty() && !comp.iter().any(|u| anchor.contains_key(u)) {
            gone.extend(comp);
        }
    }

    let free: Vec<usize> = (0..n)
        .filter(|&v| !fixed[v] && !gone.contains(&v) && nbr[v].iter().any(|w| !gone.contains(w)))
        .collect();
    let idx: HashMap<usize, usize> = free.iter().copied().enumerate().map(|(k, v)| (v, k)).collect();
    if !free.is_empty() {
        let m = free.len();
        let mut trips = Vec::new();
        let mut rhs_xy = vec![[0.0, 0.0]; m];
        for (k, &v) in free.iter().enumerate() {
            let live: Vec<usize> = nbr[v].iter().copied().filter(|w| !gone.contains(w)).collect();
            let lam = anchor.get(&v).copied().unwrap_or(0.0);
            trips.push(Triplet::new(k, k, live.len() as f64 + lam));
            rhs_xy[k] = [lam * p0[v][0], lam * p0[v][1]];
            for w in live {
                if let Some(&kw) = idx.get(&w) {
                    trips.push(Triplet::new(k, kw, -1.0));
                } else {
                    rhs_xy[k] = add2(rhs_xy[k], p[w]);
                }
            }
        }
        if let Ok(lap) = SparseColMat::<usize, f64>::try_new_from_triplets(m, m, &trips) {
            let rhs = Mat::from_fn(m, 2, |r, c| rhs_xy[r][c]);
            if let Some(sol) = solve_tutte_spd(&lap, &rhs) {
                for (k, &v) in free.iter().enumerate() {
                    p[v] = [sol[(k, 0)], sol[(k, 1)]];
                }
            }
        }
    }

    let mean_pts = |indices: &[usize], pts: &[[f64; 2]]| -> [f64; 2] {
        if indices.is_empty() {
            return [0.0, 0.0];
        }
        let inv = 1.0 / (indices.len() as f64);
        [
            indices.iter().map(|&i| pts[i][0]).sum::<f64>() * inv,
            indices.iter().map(|&i| pts[i][1]).sum::<f64>() * inv,
        ]
    };
    let fixed_indices: Vec<usize> = (0..n).filter(|&i| fixed[i]).collect();

    for ([a, b], comp) in flaps {
        let mut s_nodes: Vec<usize> = comp.into_iter().collect();
        s_nodes.sort_unstable();
        let (u0, u1) = (sub2(p0[b], p0[a]), sub2(p[b], p[a]));
        let (zr, zi) = if norm2(u0) > 1e-12 {
            let den = dot2(u0, u0);
            (dot2(u1, u0) / den, cross2(u0, u1) / den)
        } else {
            (1.0, 0.0)
        };
        let mut q_vecs: Vec<[f64; 2]> = s_nodes
            .iter()
            .map(|&s| {
                let d = sub2(p0[s], p0[a]);
                [d[0] * zr - d[1] * zi, d[0] * zi + d[1] * zr]
            })
            .collect();
        let ref_pt = if !free.is_empty() { mean_pts(&free, &p) } else { mean_pts(&fixed_indices, &p) };
        let q_mean = [
            q_vecs.iter().map(|q| q[0]).sum::<f64>() / (q_vecs.len() as f64),
            q_vecs.iter().map(|q| q[1]).sum::<f64>() / (q_vecs.len() as f64),
        ];
        if cross2(u1, q_mean) * cross2(u1, sub2(ref_pt, p[a])) < 0.0 {
            let lu1 = norm2(u1).max(1e-300);
            let e = [u1[0] / lu1, u1[1] / lu1];
            for q in &mut q_vecs {
                let proj = 2.0 * dot2(*q, e);
                *q = [proj * e[0] - q[0], proj * e[1] - q[1]];
            }
        }
        for (&s, &qv) in s_nodes.iter().zip(&q_vecs) {
            p[s] = add2(p[a], qv);
        }
    }

    for (a, comp) in pieces {
        let mut s_nodes: Vec<usize> = comp.into_iter().collect();
        s_nodes.sort_unstable();
        let placed: Vec<usize> = nbr[a].iter().copied().filter(|w| !gone.contains(w)).collect();
        let min_len = placed.iter().map(|&w| norm2(sub2(p[w], p[a]))).fold(f64::INFINITY, f64::min);
        let q_vecs: Vec<[f64; 2]> = s_nodes.iter().map(|&s| sub2(p0[s], p0[a])).collect();
        let ext = q_vecs.iter().map(|&q| norm2(q)).fold(0.0_f64, f64::max);
        let ext_eff = if ext > 0.0 { ext } else { 1.0 };
        let ell = if !placed.is_empty() && min_len > 0.0 && min_len.is_finite() { 0.45 * min_len } else { ext_eff };
        let mut ang = 0.0;
        if !placed.is_empty() {
            let mut th: Vec<f64> = placed.iter().map(|&w| (p[w][1] - p[a][1]).atan2(p[w][0] - p[a][0])).collect();
            th.sort_by(f64::total_cmp);
            let mut best_gap = (0usize, -1.0_f64);
            for i in 0..th.len() {
                let mut g = (th[(i + 1) % th.len()] - th[i]).rem_euclid(TWO_PI);
                if g == 0.0 {
                    g = TWO_PI;
                }
                if g > best_gap.1 {
                    best_gap = (i, g);
                }
            }
            ang = th[best_gap.0] + 0.5 * best_gap.1;
        }
        let m_pt = [
            q_vecs.iter().map(|q| q[0]).sum::<f64>() / (q_vecs.len() as f64),
            q_vecs.iter().map(|q| q[1]).sum::<f64>() / (q_vecs.len() as f64),
        ];
        let rot = ang - if norm2(m_pt) > 1e-12 { m_pt[1].atan2(m_pt[0]) } else { 0.0 };
        let (sr, cr) = rot.sin_cos();
        let scale = ell / ext_eff;
        for (&s, &q) in s_nodes.iter().zip(&q_vecs) {
            p[s] = [p[a][0] + scale * (cr * q[0] - sr * q[1]), p[a][1] + scale * (sr * q[0] + cr * q[1])];
        }
    }
    p
}

fn rest_length(crystal: &BuiltCrystal, u: usize, v: usize, x0: &[[f64; 2]]) -> f64 {
    if let Some(e) = crystal.adj[u].iter().find(|b| b.to == v)
        && norm2(e.vec) > 1e-12
    {
        return norm2(e.vec);
    }
    norm2(sub2(x0[v], x0[u]))
}

pub(crate) fn global_tutte(
    crystal: &BuiltCrystal,
    rims: &[Vec<usize>],
    x0: &[[f64; 2]],
    sign: f64,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let n = x0.len();
    let nbr = undirected_nbrs(crystal);
    let roots = component_roots(n, &crystal.edges);
    let mut p = x0.to_vec();
    let mut fixed = vec![false; n];
    for walk in rims {
        let w0: Vec<usize> = walk.iter().copied().filter(|&v| nbr[v].len() > 1).collect();
        let w: Vec<usize> = if w0.len() > 1 {
            (0..w0.len()).filter(|&i| w0[i] != w0[(i + w0.len() - 1) % w0.len()]).map(|i| w0[i]).collect()
        } else {
            w0
        };
        if w.len() < 3 {
            continue;
        }
        let r0 = roots[w[0]];
        let comp_pts: Vec<[f64; 2]> = (0..n).filter(|&v| roots[v] == r0).map(|v| x0[v]).collect();
        let c = [
            comp_pts.iter().map(|q| q[0]).sum::<f64>() / (comp_pts.len() as f64),
            comp_pts.iter().map(|q| q[1]).sum::<f64>() / (comp_pts.len() as f64),
        ];
        let lens: Vec<f64> = (0..w.len()).map(|i| rest_length(crystal, w[i], w[(i + 1) % w.len()], x0)).collect();
        let tot: f64 = lens.iter().sum::<f64>().max(1e-12);
        let rad = tot / TWO_PI;
        let mut acc = 0.0;
        for (i, &v) in w.iter().enumerate() {
            if !fixed[v] {
                let a = sign * TWO_PI * (acc / tot);
                p[v] = [c[0] + rad * a.cos(), c[1] + rad * a.sin()];
                fixed[v] = true;
            }
            acc += lens[i];
        }
    }
    let out = tutte_solve(crystal, &p, &fixed, &HashMap::new());
    let pinned = (0..n).filter(|&i| fixed[i]).collect();
    (out, pinned)
}

pub(crate) fn local_tutte(
    crystal: &BuiltCrystal,
    x0: &[[f64; 2]],
    seeds: &HashSet<usize>,
    k: usize,
    rim: &HashSet<usize>,
) -> (Vec<[f64; 2]>, usize) {
    let nbr = undirected_nbrs(crystal);
    let mut dist: HashMap<usize, usize> = seeds.iter().map(|&s| (s, 0)).collect();
    let mut dq: VecDeque<usize> = seeds.iter().copied().collect();
    while let Some(u) = dq.pop_front() {
        let du = dist[&u];
        if du >= k {
            continue;
        }
        for &w in &nbr[u] {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                e.insert(du + 1);
                dq.push_back(w);
            }
        }
    }
    let mut fixed = vec![true; x0.len()];
    let mut anchor = HashMap::new();
    for &v in dist.keys() {
        fixed[v] = false;
        if rim.contains(&v) {
            anchor.insert(v, 1.0);
        }
    }
    (tutte_solve(crystal, x0, &fixed, &anchor), dist.len())
}

pub(crate) fn drawn_boundary_loop(crystal: &BuiltCrystal, pos: &[[f64; 2]]) -> Vec<usize> {
    let nbr = undirected_nbrs(crystal);
    let mut rot_sys: Vec<Vec<usize>> = Vec::with_capacity(pos.len());
    let mut idx_map: Vec<HashMap<usize, usize>> = Vec::with_capacity(pos.len());
    for (u, set) in nbr.iter().enumerate() {
        let mut lst: Vec<(f64, usize)> = set
            .iter()
            .map(|&v| ((pos[v][1] - pos[u][1]).atan2(pos[v][0] - pos[u][0]).rem_euclid(TWO_PI), v))
            .collect();
        lst.sort_by(|a, b| a.0.total_cmp(&b.0));
        let verts: Vec<usize> = lst.into_iter().map(|(_, v)| v).collect();
        idx_map.push(verts.iter().copied().enumerate().map(|(k, v)| (v, k)).collect());
        rot_sys.push(verts);
    }
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut best: Option<(f64, Vec<usize>)> = None;
    for (u, lst) in rot_sys.iter().enumerate() {
        for &v in lst {
            if seen.contains(&(u, v)) {
                continue;
            }
            let mut f = Vec::new();
            let (mut a, mut b) = (u, v);
            while seen.insert((a, b)) {
                f.push(a);
                let blst = &rot_sys[b];
                let k = idx_map[b][&a];
                a = b;
                b = blst[if k == 0 { blst.len() - 1 } else { k - 1 }];
            }
            let area = signed_area(pos, &f);
            if best.as_ref().is_none_or(|(ba, _)| area < *ba) {
                best = Some((area, f));
            }
        }
    }
    match best {
        Some((a, mut f)) => {
            if a < 0.0 {
                f.reverse();
            }
            f
        }
        None => Vec::new(),
    }
}

pub(crate) fn connectivity_note(crystal: &BuiltCrystal, pinned: &[usize]) -> String {
    let nbr = undirected_nbrs(crystal);
    let cut = articulation_points(&nbr).len();
    let rim: HashSet<usize> = pinned.iter().copied().collect();
    let low = (0..nbr.len()).filter(|&v| nbr[v].len() <= 2 && !rim.contains(&v)).count();
    let comps: HashSet<usize> = component_roots(nbr.len(), &crystal.edges).into_iter().collect();
    if cut == 0 && low == 0 && comps.len() <= 1 {
        "not excluded (no cut site, no interior site of degree <= 2; 2-cuts not checked)".to_string()
    } else {
        format!("no: {} component(s), {cut} cut site(s), {low} interior site(s) of degree <= 2", comps.len())
    }
}
