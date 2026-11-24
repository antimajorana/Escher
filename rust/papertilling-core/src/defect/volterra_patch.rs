//! Three-chart Volterra dislocation surgery, drawn face walks, non-crossing ring completions,
//! and Weisfeiler-Lehman graph hashing (`lab_defects.dislocation` + completion primitives).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use petgraph::graph::UnGraph;

use crate::defect::cone_geometry::{add2, cross2, dot2, mat2_inv, mat2_vec, norm2, sub2};
use crate::defect::core_graph::{LatticeCoreInfo, PerfectLattice, SiteKey, TWO_PI};
use crate::defect::sym_helper::{np_allclose_vec, round_dec};
use crate::types::{BuiltCrystal, DirectedBondData, Isometry2};

/// Result of a 3-chart Volterra cut-and-reglue patch build, retaining lattice `SiteKey`s.
#[derive(Clone, Debug)]
pub struct VolterraPatch {
    pub crystal: BuiltCrystal,
    pub keys: Vec<SiteKey>,
}

/// Build a single Volterra dislocation on `lat_info` using the clean-room 3-chart cut-and-reglue rule.
pub(crate) fn build_volterra_patch(
    lat_info: &LatticeCoreInfo,
    b: [f64; 2],
    core_pt: [f64; 2],
    cut: Option<[f64; 2]>,
    r_patch: f64,
) -> Result<VolterraPatch, String> {
    let lat = &lat_info.lat;
    let fb = mat2_vec(lat.a_inv, b);
    let m_b = [fb[0].round_ties_even() as i32, fb[1].round_ties_even() as i32];
    if norm2(b) < 1e-9
        || (fb[0] - m_b[0] as f64).abs() > 1e-6
        || (fb[1] - m_b[1] as f64).abs() > 1e-6
    {
        return Err(format!("b = {b:?} is not a non-zero lattice translation"));
    }
    let lb = norm2(b);
    let d = cut.unwrap_or([b[1] / lb, -b[0] / lb]);
    if cross2(b, d).abs() < 1e-12 {
        return Err("cut direction is parallel to Burgers vector b".to_string());
    }
    let b_inv = mat2_inv([[b[0], d[0]], [b[1], d[1]]]);
    let all_keys = lat.sites_in_disk(core_pt, r_patch);
    let mut keys = Vec::with_capacity(all_keys.len());
    let mut ts = Vec::with_capacity(all_keys.len());
    let mut ref_pos = Vec::with_capacity(all_keys.len());
    for k in all_keys {
        let p = lat.pos(k);
        let [t, s] = mat2_vec(b_inv, sub2(p, core_pt));
        if s > 1e-9 && (-1e-9..(1.0 - 1e-9)).contains(&t) {
            continue;
        }
        keys.push(k);
        ts.push((t, s));
        ref_pos.push(p);
    }

    let n = keys.len();
    let mut edge_map: BTreeMap<(usize, usize), ([f64; 2], usize)> = BTreeMap::new();
    for chart in 0..3 {
        let mut where_map: HashMap<SiteKey, usize> = HashMap::with_capacity(n);
        for (u, (&k, &(t, s))) in keys.iter().zip(&ts).enumerate() {
            let (ok, sh) = match chart {
                0 => (s <= 1e-9 || t < -1e-9, 0),
                1 => (t >= 1.0 - 1e-9, -1),
                _ => (s > 1e-9, if t >= 1.0 - 1e-9 { -1 } else { 0 }),
            };
            if ok {
                let img = if sh == 0 {
                    k
                } else {
                    (k.0 + sh * m_b[0], k.1 + sh * m_b[1], k.2)
                };
                where_map.insert(img, u);
            }
        }
        for (&img, &u) in &where_map {
            let p_img = lat.pos(img);
            for (k2, kind) in lat.nbrs(img) {
                let Some(&v) = where_map.get(&k2) else { continue };
                if v == u {
                    continue;
                }
                let vec_uv = sub2(lat.pos(k2), p_img);
                for (pair, vec_dir) in [((u, v), vec_uv), ((v, u), [-vec_uv[0], -vec_uv[1]])] {
                    if let Some(&(old_v, _)) = edge_map.get(&pair)
                        && !np_allclose_vec(old_v, vec_dir, 1e-5, 1e-8)
                    {
                        return Err(format!("two charts disagree on bond {pair:?}"));
                    }
                    edge_map.insert(pair, (vec_dir, kind));
                }
            }
        }
    }

    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for ((u, v), (vec, kind)) in edge_map {
        if u < v {
            edges.push((u, v));
        }
        adj[u].push(DirectedBondData {
            to: v,
            vec,
            kind,
            seam: None,
            ang: None,
            across: false,
            ghost_stub: None,
        });
    }
    let sgn = if cross2(d, b) > 0.0 { 1.0 } else { -1.0 };
    let c_mid = [core_pt[0] + 0.5 * b[0], core_pt[1] + 0.5 * b[1]];
    let ld = norm2(d);
    let dh = [d[0] / ld, d[1] / ld];
    let pos: Vec<[f64; 2]> = ref_pos
        .iter()
        .map(|&p| {
            let r = sub2(p, c_mid);
            let ang = (sgn * cross2(dh, r)).atan2(dot2(r, dh)).rem_euclid(TWO_PI);
            let f = 1.0 - ang / TWO_PI;
            [p[0] - f * b[0], p[1] - f * b[1]]
        })
        .collect();
    let basis_idx: Vec<usize> = keys.iter().map(|&k| k.2).collect();
    let crystal = BuiltCrystal {
        pos,
        ref_pos,
        basis_idx,
        side: vec![None; n],
        cone_sites: Vec::new(),
        edges,
        adj,
        wall_sigma: None,
        is_non_orientable: false,
    };
    Ok(VolterraPatch { crystal, keys })
}

/// Public 3-chart Volterra dislocation builder returning `BuiltCrystal`.
pub fn build_volterra_dislocation(
    lat_info: &LatticeCoreInfo,
    b: [f64; 2],
    core_pt: [f64; 2],
    cut: Option<[f64; 2]>,
    r_patch: f64,
) -> Result<BuiltCrystal, String> {
    let mut crystal = build_volterra_patch(lat_info, b, core_pt, cut, r_patch)?.crystal;
    // Bond vectors are expressed in the shifted chart, while reference positions retain
    // their original lattice coordinates. Carry the chart translation on each seam bond.
    for (u, bonds) in crystal.adj.iter_mut().enumerate() {
        for bond in bonds {
            let t = sub2(crystal.ref_pos[bond.to], add2(crystal.ref_pos[u], bond.vec));
            if norm2(t) > 1e-9 {
                bond.seam = Some(Isometry2 { t, ..Isometry2::IDENTITY });
            }
        }
    }
    Ok(crystal)
}

/// Bounded CCW faces read from `crystal.pos` (`FastDefect.faces`), keeping faces with area `> 1e-12`.
pub(crate) fn drawn_faces(crystal: &BuiltCrystal) -> Vec<Vec<usize>> {
    let n = crystal.pos.len();
    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &crystal.edges {
        nbr[u].push(v);
        nbr[v].push(u);
    }
    for (u, lst) in nbr.iter_mut().enumerate() {
        let pu = crystal.pos[u];
        lst.sort_by(|&a, &b| {
            let aa = (crystal.pos[a][1] - pu[1]).atan2(crystal.pos[a][0] - pu[0]);
            let ab = (crystal.pos[b][1] - pu[1]).atan2(crystal.pos[b][0] - pu[0]);
            aa.total_cmp(&ab)
        });
    }
    let idx_map: Vec<HashMap<usize, usize>> = nbr
        .iter()
        .map(|lst| lst.iter().copied().enumerate().map(|(k, v)| (v, k)).collect())
        .collect();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut out = Vec::new();
    for (u, lst) in nbr.iter().enumerate() {
        for &v in lst {
            if seen.contains(&(u, v)) {
                continue;
            }
            let mut f = Vec::new();
            let (mut a, mut b) = (u, v);
            while seen.insert((a, b)) {
                f.push(a);
                let blst = &nbr[b];
                let k = idx_map[b][&a];
                a = b;
                b = blst[if k == 0 { blst.len() - 1 } else { k - 1 }];
            }
            let m = f.len();
            let mut area2 = 0.0;
            for i in 0..m {
                let (p, q) = (crystal.pos[f[i]], crystal.pos[f[(i + 1) % m]]);
                area2 += p[0] * q[1] - p[1] * q[0];
            }
            if 0.5 * area2 > 1e-12 {
                out.push(f);
            }
        }
    }
    out
}

/// Bounded drawn faces whose `vec` loop sum is non-zero (`FastDefect.charged_faces("vec")`).
pub(crate) fn drawn_charged_faces(crystal: &BuiltCrystal) -> Vec<(Vec<usize>, [f64; 2])> {
    let mut out = Vec::new();
    for f in drawn_faces(crystal) {
        let m = f.len();
        let (mut sx, mut sy) = (0.0, 0.0);
        for i in 0..m {
            let (u, v) = (f[i], f[(i + 1) % m]);
            let e = crystal.adj[u].iter().find(|b| b.to == v).expect("face bond");
            sx += e.vec[0];
            sy += e.vec[1];
        }
        if sx.hypot(sy) > 1e-9 {
            out.push((f, [sx, sy]));
        }
    }
    out
}

/// Split `ring` along non-crossing `chords` (`lab_defects._split(..., sizes_only=False)`).
pub(crate) fn split_ring(ring: &[usize], chords: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut rings = vec![ring.to_vec()];
    for &(a, b) in chords {
        if let Some(i) = rings.iter().position(|r| r.contains(&a) && r.contains(&b)) {
            let r = &rings[i];
            let pa = r.iter().position(|&x| x == a).unwrap();
            let pb = r.iter().position(|&x| x == b).unwrap();
            let (ia, ib) = if pa <= pb { (pa, pb) } else { (pb, pa) };
            let r1 = r[ia..=ib].to_vec();
            let mut r2 = r[ib..].to_vec();
            r2.extend_from_slice(&r[..=ia]);
            rings.splice(i..=i, [r1, r2]);
        }
    }
    rings
}

pub(crate) fn first_pos(ring: &[usize]) -> HashMap<usize, usize> {
    let mut pos = HashMap::with_capacity(ring.len());
    for (k, &v) in ring.iter().enumerate() {
        pos.entry(v).or_insert(k);
    }
    pos
}

fn for_each_comb(n: usize, r: usize, mut f: impl FnMut(&[usize])) {
    if r == 0 || r > n {
        return;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    f(&idx);
    while let Some(i) = (0..r).rev().find(|&i| idx[i] != i + n - r) {
        idx[i] += 1;
        for j in (i + 1)..r {
            idx[j] = idx[j - 1] + 1;
        }
        f(&idx);
    }
}

/// Enumerate all non-crossing chord subsets on `ring` respecting `under` (`(subsets, was_capped)`).
pub(crate) fn enumerate_completions(
    has_edge: impl Fn(usize, usize) -> bool,
    ring: &[usize],
    under: &BTreeMap<usize, usize>,
    max_candidates: usize,
) -> (Vec<Vec<(usize, usize)>>, bool) {
    let pos = first_pos(ring);
    let verts: Vec<usize> = under.keys().copied().collect();
    let mut cand = Vec::new();
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            let (u, v) = (verts[i], verts[j]);
            if !has_edge(u, v) {
                cand.push((u, v));
            }
        }
    }
    let capped = cand.len() > max_candidates;
    if capped {
        cand.truncate(max_candidates);
    }
    let crosses = |p: (usize, usize), q: (usize, usize)| -> bool {
        if p.0 == q.0 || p.0 == q.1 || p.1 == q.0 || p.1 == q.1 {
            return false;
        }
        let (p0, p1, q0, q1) = (pos[&p.0], pos[&p.1], pos[&q.0], pos[&q.1]);
        (p0 < q0 && q0 < p1) != (p0 < q1 && q1 < p1)
    };
    let mut out = Vec::new();
    let mut use_cnt: HashMap<usize, usize> = HashMap::new();
    for r in 1..=cand.len() {
        for_each_comb(cand.len(), r, |idxs| {
            use_cnt.clear();
            for &ci in idxs {
                let (u, v) = cand[ci];
                *use_cnt.entry(u).or_default() += 1;
                *use_cnt.entry(v).or_default() += 1;
            }
            if use_cnt.iter().any(|(w, c)| *c > under[w]) {
                return;
            }
            let sub2: Vec<(usize, usize)> = idxs
                .iter()
                .map(|&ci| {
                    let (u, v) = cand[ci];
                    if pos[&u] <= pos[&v] { (u, v) } else { (v, u) }
                })
                .collect();
            for a in 0..sub2.len() {
                for b in (a + 1)..sub2.len() {
                    if crosses(sub2[a], sub2[b]) {
                        return;
                    }
                }
            }
            out.push(idxs.iter().map(|&ci| cand[ci]).collect());
        });
    }
    (out, capped)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn wl_hash_with<N>(g: &UnGraph<N, ()>, init: impl Fn(petgraph::graph::NodeIndex) -> u64) -> u64 {
    let n = g.node_count();
    let mut colors: Vec<u64> = g.node_indices().map(init).collect();
    let mut nbrs = Vec::new();
    for _ in 0..4 {
        let mut next = Vec::with_capacity(n);
        for u in g.node_indices() {
            nbrs.clear();
            nbrs.extend(g.neighbors(u).map(|v| colors[v.index()]));
            nbrs.sort_unstable();
            let mut h = splitmix64(colors[u.index()]);
            for &c in &nbrs {
                h = splitmix64(h ^ c);
            }
            next.push(h);
        }
        colors = next;
    }
    colors.sort_unstable();
    let mut acc = splitmix64((n as u64) << 32 | (g.edge_count() as u64));
    for c in colors {
        acc = splitmix64(acc ^ c);
    }
    acc
}

pub(crate) fn wl_hash_unlabelled(g: &UnGraph<(), ()>) -> u64 {
    wl_hash_with(g, |u| splitmix64(g.neighbors(u).count() as u64 + 1))
}

pub(crate) fn wl_hash_labelled(g: &UnGraph<u16, ()>) -> u64 {
    wl_hash_with(g, |u| splitmix64(((g[u] as u64) << 16) | (g.neighbors(u).count() as u64 + 1)))
}

pub(crate) fn ball_nodes(adj: &[Vec<usize>], sources: &BTreeSet<usize>, radius: usize) -> Vec<usize> {
    let mut dist: HashMap<usize, usize> = sources.iter().map(|&s| (s, 0)).collect();
    let mut dq: VecDeque<usize> = sources.iter().copied().collect();
    while let Some(u) = dq.pop_front() {
        let du = dist[&u];
        if du >= radius {
            continue;
        }
        for &w in &adj[u] {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                e.insert(du + 1);
                dq.push_back(w);
            }
        }
    }
    let mut out: Vec<usize> = dist.into_keys().collect();
    out.sort_unstable();
    out
}

pub(crate) fn ball_ungraph(adj: &[Vec<usize>], sources: &BTreeSet<usize>, radius: usize) -> UnGraph<(), ()> {
    let nodes = ball_nodes(adj, sources, radius);
    let mut g = UnGraph::<(), ()>::new_undirected();
    let mut map = HashMap::with_capacity(nodes.len());
    for &u in &nodes {
        map.insert(u, g.add_node(()));
    }
    for &u in &nodes {
        for &v in &adj[u] {
            if u < v
                && let Some(&nv) = map.get(&v)
            {
                g.add_edge(map[&u], nv, ());
            }
        }
    }
    g
}

pub(crate) fn perfect_ring_sizes(lat: &PerfectLattice, scale: f64, planar: bool) -> BTreeSet<usize> {
    if !planar || lat.rule.is_empty() {
        return BTreeSet::new();
    }
    let keys = lat.sites_in_disk([0.0, 0.0], 3.0 * scale);
    let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| lat.pos(k)).collect();
    let mut edges = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        for (k2, _) in lat.nbrs(k) {
            if let Some(&j) = index.get(&k2)
                && i < j
            {
                edges.push((i, j));
            }
        }
    }
    edges.sort_unstable();
    let dummy = BuiltCrystal {
        ref_pos: pos.clone(),
        pos,
        basis_idx: vec![0; keys.len()],
        side: vec![None; keys.len()],
        cone_sites: Vec::new(),
        edges,
        adj: vec![Vec::new(); keys.len()],
        wall_sigma: None,
        is_non_orientable: false,
    };
    drawn_faces(&dummy).into_iter().map(|f| f.len()).collect()
}

pub(crate) fn candidate_cuts_for_scan(a: [[f64; 2]; 2], b: [f64; 2]) -> Vec<[f64; 2]> {
    let lb = norm2(b);
    let perp = [b[1] / lb, -b[0] / lb];
    let mut cuts = vec![perp, [-perp[0], -perp[1]]];
    let mut others = Vec::new();
    for m0 in -3..=3 {
        for m1 in -3..=3 {
            if (m0, m1) != (0, 0) {
                let v = mat2_vec(a, [m0 as f64, m1 as f64]);
                if cross2(v, b).abs() > 1e-6 {
                    others.push(v);
                }
            }
        }
    }
    others.sort_by(|u, v| round_dec(norm2(*u), 6).total_cmp(&round_dec(norm2(*v), 6)));
    for v in others {
        if cuts.len() >= 6 {
            break;
        }
        let lv = norm2(v);
        let vn = [v[0] / lv, v[1] / lv];
        if !cuts.iter().any(|&c| np_allclose_vec(vn, c, 1e-5, 1e-6)) {
            cuts.push(vn);
        }
    }
    cuts
}
