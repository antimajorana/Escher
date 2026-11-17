//! Lattice reference invariants and degree-labelled defect core graphs.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use petgraph::algo::is_isomorphic_matching;
use petgraph::graph::UnGraph;

pub(crate) use crate::defect::charge::{corners_order, outer_face_index};
pub use crate::defect::charge::{
    ChargeReport, check_frames, frame_faces, frank_angles, infer_patch_disk, loop_charge,
    total_charge, total_frank,
};
pub(crate) use crate::defect::cone_geometry::{
    add2, cross2, dot2, mat2_inv, mat2_vec, norm2, rot_mat, rotation_about, same_iso, sub2,
};
use crate::types::{Bond2D, BuiltCrystal, Cell2D, DirectedBondData, Isometry2};

pub const TWO_PI: f64 = 2.0 * PI;
pub type SiteKey = (i32, i32, usize);
pub type LayerSignature = Vec<(usize, usize, usize, usize, usize)>;

fn frac_mod(f: [f64; 2]) -> [f64; 2] {
    let mut out = [f[0] - (f[0] + 1e-6).floor(), f[1] - (f[1] + 1e-6).floor()];
    for x in &mut out {
        if (*x - 1.0).abs() < 1e-6 || x.abs() < 1e-12 {
            *x = 0.0;
        }
    }
    out
}

fn canon_bond_key(i: usize, j: usize, n1: i32, n2: i32) -> (usize, usize, i32, i32) {
    (i, j, n1, n2).min((j, i, -n1, -n2))
}

fn reduce_basis(mut b1: [f64; 2], mut b2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    for _ in 0..200 {
        if dot2(b2, b2) < dot2(b1, b1) - 1e-12 {
            std::mem::swap(&mut b1, &mut b2);
        }
        let k = (dot2(b1, b2) / dot2(b1, b1)).round_ties_even();
        if k == 0.0 {
            break;
        }
        b2 = [b2[0] - k * b1[0], b2[1] - k * b1[1]];
    }
    if cross2(b1, b2) < 0.0 {
        b2 = [-b2[0], -b2[1]];
    }
    (b1, b2)
}

pub(crate) fn locate_in_basis(
    p: [f64; 2],
    a_inv: [[f64; 2]; 2],
    basis: &[[f64; 2]],
    tol: f64,
) -> Option<(usize, i32, i32)> {
    for (s, &bs) in basis.iter().enumerate() {
        let f = mat2_vec(a_inv, sub2(p, bs));
        let (r0, r1) = (f[0].round_ties_even(), f[1].round_ties_even());
        if (f[0] - r0).abs() <= tol && (f[1] - r1).abs() <= tol {
            return Some((s, r0 as i32, r1 as i32));
        }
    }
    None
}

fn rebase_cell(cell: &Cell2D, b1: [f64; 2], b2: [f64; 2]) -> Cell2D {
    let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
    let b = [[b1[0], b2[0]], [b1[1], b2[1]]];
    let b_inv = mat2_inv(b);
    let mut new_basis: Vec<[f64; 2]> = Vec::new();
    for &p in &cell.basis {
        let q = mat2_vec(b, frac_mod(mat2_vec(b_inv, p)));
        if !new_basis.iter().any(|&r| norm2(sub2(q, r)) < 1e-6) {
            new_basis.push(q);
        }
    }
    let mut seen = HashSet::new();
    let mut bonds = Vec::new();
    for bd in &cell.bonds {
        let p = cell.basis[bd.i];
        let q = add2(cell.basis[bd.j], mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]));
        let (si, ni0, ni1) = locate_in_basis(p, b_inv, &new_basis, 1e-6).expect("locate p");
        let (sj, nj0, nj1) = locate_in_basis(q, b_inv, &new_basis, 1e-6).expect("locate q");
        let key = canon_bond_key(si, sj, nj0 - ni0, nj1 - ni1);
        if seen.insert(key) {
            bonds.push(Bond2D { i: key.0, j: key.1, n1: key.2, n2: key.3, kind: bd.kind });
        }
    }
    bonds.sort_by_key(|bd| (bd.i, bd.j, bd.n1, bd.n2, bd.kind));
    Cell2D { a1: b1, a2: b2, basis: new_basis, bonds }
}

/// Reduce `cell` to its primitive cell ignoring bond colours (`K3`) and drop parallel bonds.
pub fn primitive_uncolored_cell(cell: &Cell2D) -> Cell2D {
    let mut seen_init = HashSet::new();
    let dedup_bonds: Vec<Bond2D> = cell
        .bonds
        .iter()
        .copied()
        .filter(|bd| seen_init.insert(canon_bond_key(bd.i, bd.j, bd.n1, bd.n2)))
        .collect();
    let dedup = Cell2D { a1: cell.a1, a2: cell.a2, basis: cell.basis.clone(), bonds: dedup_bonds };
    if dedup.basis.is_empty() {
        return dedup;
    }
    let a = [[dedup.a1[0], dedup.a2[0]], [dedup.a1[1], dedup.a2[1]]];
    let a_inv = mat2_inv(a);
    let f_coords: Vec<[f64; 2]> = dedup.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    let mut bond_set = HashSet::new();
    for bd in &dedup.bonds {
        bond_set.insert((bd.i, bd.j, bd.n1, bd.n2));
        bond_set.insert((bd.j, bd.i, -bd.n1, -bd.n2));
    }
    let mut trans: Vec<[f64; 2]> = Vec::new();
    for fj in &f_coords {
        let t = frac_mod(sub2(*fj, f_coords[0]));
        let mut perm = Vec::with_capacity(f_coords.len());
        let mut shift = Vec::with_capacity(f_coords.len());
        for fi in &f_coords {
            let tgt = add2(*fi, t);
            if let Some((k, fk)) = f_coords.iter().enumerate().find(|(_, fk)| {
                let (d0, d1) = (tgt[0] - fk[0], tgt[1] - fk[1]);
                (d0 - d0.round_ties_even()).abs() < 1e-6 && (d1 - d1.round_ties_even()).abs() < 1e-6
            }) {
                perm.push(k);
                shift.push([(tgt[0] - fk[0]).round_ties_even() as i32, (tgt[1] - fk[1]).round_ties_even() as i32]);
            } else {
                break;
            }
        }
        if perm.len() != f_coords.len() {
            continue;
        }
        let bonds_ok = dedup.bonds.iter().all(|bd| {
            let dn1 = bd.n1 + shift[bd.j][0] - shift[bd.i][0];
            let dn2 = bd.n2 + shift[bd.j][1] - shift[bd.i][1];
            bond_set.contains(&(perm[bd.i], perm[bd.j], dn1, dn2))
        });
        if bonds_ok
            && !trans.iter().any(|&t2| {
                let (d0, d1) = (t[0] - t2[0], t[1] - t2[1]);
                (d0 - d0.round_ties_even()).abs() < 1e-5 && (d1 - d1.round_ties_even()).abs() < 1e-5
            })
        {
            trans.push(t);
        }
    }
    if trans.len() <= 1 {
        let (b1, b2) = reduce_basis(dedup.a1, dedup.a2);
        if norm2(sub2(b1, dedup.a1)) < 1e-8 && norm2(sub2(b2, dedup.a2)) < 1e-8 {
            return dedup;
        }
        return rebase_cell(&dedup, b1, b2);
    }
    let mut cands = Vec::new();
    for t in &trans {
        for n1 in -2..=2 {
            for n2 in -2..=2 {
                let v = mat2_vec(a, [t[0] + n1 as f64, t[1] + n2 as f64]);
                if norm2(v) > 1e-6 {
                    cands.push(v);
                }
            }
        }
    }
    cands.sort_by(|u, v| {
        (norm2(*u) * 1e9)
            .round()
            .total_cmp(&(norm2(*v) * 1e9).round())
            .then_with(|| (-u[0]).total_cmp(&(-v[0])))
            .then_with(|| (-u[1]).total_cmp(&(-v[1])))
    });
    let area = cross2(dedup.a1, dedup.a2).abs() / (trans.len() as f64);
    let b1 = cands[0];
    let b2 = *cands
        .iter()
        .find(|&&v| (cross2(b1, v).abs() - area).abs() < 1e-6 * area.max(1.0))
        .expect("second primitive basis vector");
    let (b1, b2) = reduce_basis(b1, b2);
    rebase_cell(&dedup, b1, b2)
}

pub(crate) fn has_crossings_cell(cell: &Cell2D) -> bool {
    let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
    let segs: Vec<([f64; 2], [f64; 2])> = cell
        .bonds
        .iter()
        .map(|bd| (cell.basis[bd.i], add2(cell.basis[bd.j], mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]))))
        .collect();
    for n1 in -5..=5 {
        for n2 in -5..=5 {
            let sh = mat2_vec(a, [n1 as f64, n2 as f64]);
            for (k, &(p, q)) in segs.iter().enumerate() {
                for (l, &(r0, s0)) in segs.iter().enumerate() {
                    if n1 == 0 && n2 == 0 && l <= k {
                        continue;
                    }
                    let (r, s) = (add2(r0, sh), add2(s0, sh));
                    let (da, db) = (sub2(q, p), sub2(s, r));
                    let den = cross2(da, db);
                    if den.abs() >= 1e-9 {
                        let rp = sub2(r, p);
                        let (u, v) = (cross2(rp, db) / den, cross2(rp, da) / den);
                        if u > 1e-9 && u < 1.0 - 1e-9 && v > 1e-9 && v < 1.0 - 1e-9 {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Infinite periodic lattice evaluator for site lookup, disk patches, and symmetry probes.
#[derive(Clone, Debug)]
pub struct PerfectLattice {
    pub cell: Cell2D,
    pub a: [[f64; 2]; 2],
    pub a_inv: [[f64; 2]; 2],
    pub rule: Vec<Vec<(i32, i32, usize, usize)>>,
}

impl PerfectLattice {
    pub fn new(cell: Cell2D) -> Self {
        let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
        let mut rule = vec![Vec::new(); cell.basis.len()];
        for bd in &cell.bonds {
            rule[bd.i].push((bd.n1, bd.n2, bd.j, bd.kind));
            rule[bd.j].push((-bd.n1, -bd.n2, bd.i, bd.kind));
        }
        Self { cell, a, a_inv: mat2_inv(a), rule }
    }

    #[inline]
    pub fn pos(&self, key: SiteKey) -> [f64; 2] {
        add2(mat2_vec(self.a, [key.0 as f64, key.1 as f64]), self.cell.basis[key.2])
    }

    #[inline]
    pub fn locate(&self, p: [f64; 2]) -> Option<SiteKey> {
        let (s, n1, n2) = locate_in_basis(p, self.a_inv, &self.cell.basis, 1e-6)?;
        Some((n1, n2, s))
    }

    pub fn nbrs(&self, key: SiteKey) -> impl Iterator<Item = (SiteKey, usize)> + '_ {
        self.rule[key.2].iter().map(move |&(d1, d2, s2, kind)| ((key.0 + d1, key.1 + d2, s2), kind))
    }

    pub fn sites_in_disk(&self, centre: [f64; 2], radius: f64) -> Vec<SiteKey> {
        let fc = mat2_vec(self.a_inv, centre);
        let (n0_0, n0_1) = (fc[0].round_ties_even() as i32, fc[1].round_ties_even() as i32);
        let h = cross2(self.cell.a1, self.cell.a2).abs() / norm2(self.cell.a1).max(norm2(self.cell.a2)).max(1e-12);
        let k_span = (radius / h.max(1e-12)).ceil() as i32 + 2;
        let mut out = Vec::new();
        for n1 in (n0_0 - k_span)..=(n0_0 + k_span) {
            for n2 in (n0_1 - k_span)..=(n0_1 + k_span) {
                for s in 0..self.cell.basis.len() {
                    let key = (n1, n2, s);
                    if norm2(sub2(self.pos(key), centre)) <= radius + 1e-9 {
                        out.push(key);
                    }
                }
            }
        }
        out
    }

    pub fn is_symmetry(&self, g: &Isometry2, probe: f64) -> bool {
        let mut keys = self.sites_in_disk([0.0, 0.0], probe * norm2(self.cell.a1));
        for s in 0..self.cell.basis.len() {
            if !keys.contains(&(0, 0, s)) {
                keys.push((0, 0, s));
            }
        }
        for key in keys {
            let Some(img) = self.locate(g.apply(self.pos(key))) else { return false };
            let mut mine: Vec<SiteKey> = Vec::new();
            for (k2, _) in self.nbrs(key) {
                let Some(loc2) = self.locate(g.apply(self.pos(k2))) else { return false };
                mine.push(loc2);
            }
            let mut target: Vec<SiteKey> = self.nbrs(img).map(|(k2, _)| k2).collect();
            mine.sort_unstable();
            target.sort_unstable();
            if mine != target {
                return false;
            }
        }
        true
    }
}

/// Precomputed perfect-lattice reference invariants used by surgery and core-graph extraction.
#[derive(Clone, Debug)]
pub struct LatticeCoreInfo {
    pub lat: PerfectLattice,
    pub lmax: f64,
    pub lmean: f64,
    pub scale: f64,
    pub planar: bool,
    pub max_ring: usize,
    pub r_det: usize,
    pub perfect_degree: Vec<usize>,
    pub perfect_sigs: BTreeSet<LayerSignature>,
}

impl LatticeCoreInfo {
    pub fn new(cell: &Cell2D) -> Self {
        let prim = primitive_uncolored_cell(cell);
        let planar = !has_crossings_cell(&prim);
        let lat = PerfectLattice::new(prim);
        let mut lens = Vec::new();
        for s in 0..lat.rule.len() {
            let p0 = lat.pos((0, 0, s));
            for (k2, _) in lat.nbrs((0, 0, s)) {
                lens.push(norm2(sub2(lat.pos(k2), p0)));
            }
        }
        let lmax = lens.iter().copied().fold(0.0_f64, f64::max);
        let lmean = if lens.is_empty() { 0.0 } else { lens.iter().sum::<f64>() / (lens.len() as f64) };
        let scale = norm2(lat.cell.a1).max(norm2(lat.cell.a2));
        let mut max_ring = 0usize;
        if planar && !lat.rule.is_empty() {
            let pp = build_perfect_crystal(&lat, 3.0 * scale + 2.0 * lmax);
            let faces = frame_faces(&pp);
            if faces.len() > 1 {
                let k_out = outer_face_index(&faces);
                max_ring = faces.iter().enumerate().filter(|&(i, _)| i != k_out).map(|(_, f)| f.len()).max().unwrap_or(0);
            }
        }
        let r_det = if max_ring > 0 { 2usize.max(max_ring.div_ceil(2)) } else { 3 };
        let keys = lat.sites_in_disk([0.0, 0.0], 9.0 * lmax + 2.0 * scale);
        let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
        let adj: Vec<Vec<usize>> = keys
            .iter()
            .map(|&k| lat.nbrs(k).filter_map(|(k2, _)| index.get(&k2).copied()).collect())
            .collect();
        let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
        let mut perfect_sigs = BTreeSet::new();
        for s in 0..lat.rule.len() {
            if let Some(&v0) = index.get(&(0, 0, s)) {
                perfect_sigs.insert(layer_signature(&adj, &deg, v0, r_det));
            }
        }
        let perfect_degree = lat.rule.iter().map(Vec::len).collect();
        Self { lat, lmax, lmean, scale, planar, max_ring, r_det, perfect_degree, perfect_sigs }
    }
}

fn build_perfect_crystal(lat: &PerfectLattice, radius: f64) -> BuiltCrystal {
    let keys = lat.sites_in_disk([0.0, 0.0], radius);
    let n = keys.len();
    let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| lat.pos(k)).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|&k| k.2).collect();
    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        for (k2, kind) in lat.nbrs(k) {
            if let Some(&j) = index.get(&k2) {
                if i < j {
                    edges.push((i, j));
                }
                adj[i].push(DirectedBondData {
                    to: j, vec: sub2(pos[j], pos[i]), kind, seam: None, ang: None, across: false, ghost_stub: None,
                });
            }
        }
    }
    edges.sort_unstable();
    BuiltCrystal {
        ref_pos: pos.clone(), pos, basis_idx, side: vec![None; n], cone_sites: Vec::new(),
        edges, adj, wall_sigma: None, is_non_orientable: false,
    }
}

/// Layered BFS neighbourhood signature of vertex `v` up to radius `r`.
pub fn layer_signature(adj: &[Vec<usize>], deg: &[usize], v: usize, r: usize) -> LayerSignature {
    let mut dist: HashMap<usize, usize> = HashMap::from([(v, 0)]);
    let mut frontier = vec![v];
    for depth in 1..=r {
        let mut nxt = Vec::new();
        for u in frontier {
            for &w in &adj[u] {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                    e.insert(depth);
                    nxt.push(w);
                }
            }
        }
        frontier = nxt;
    }
    let mut sig = Vec::with_capacity(dist.len());
    for (&u, &du) in &dist {
        let (mut a, mut b, mut c) = (0usize, 0usize, 0usize);
        for &w in &adj[u] {
            if let Some(&dw) = dist.get(&w) {
                if du > 0 && dw == du - 1 { a += 1; } else if dw == du { b += 1; } else { c += 1; }
            }
        }
        sig.push((du, deg[u], a, b, c));
    }
    sig.sort_unstable();
    sig
}

/// Sites on or next to the boundary of an open disk patch `(centre, r_patch)`.
pub fn patch_rim(crystal: &BuiltCrystal, lat_info: &LatticeCoreInfo, centre: [f64; 2], r_patch: f64) -> HashSet<usize> {
    crystal
        .ref_pos
        .iter()
        .zip(&crystal.basis_idx)
        .enumerate()
        .filter_map(|(i, (&x, &s))| {
            let deg = crystal.adj[i].len();
            let r = norm2(sub2(x, centre));
            (r > r_patch - 1.5 * lat_info.lmax || (deg < lat_info.perfect_degree[s] && r > r_patch - 3.0 * lat_info.lmax))
                .then_some(i)
        })
        .collect()
}

fn multi_bfs(adj: &[Vec<usize>], sources: &HashSet<usize>, cutoff: Option<usize>) -> HashMap<usize, usize> {
    let mut dist: HashMap<usize, usize> = sources.iter().map(|&s| (s, 0)).collect();
    let mut dq: VecDeque<usize> = sources.iter().copied().collect();
    while let Some(u) = dq.pop_front() {
        let du = dist[&u];
        if cutoff.is_some_and(|c| du >= c) {
            continue;
        }
        for &w in &adj[u] {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                e.insert(du + 1);
                dq.push_back(w);
            }
        }
    }
    dist
}

/// Vertex indices of irregular sites (strictly inside `rim`, signature not in `perfect_sigs`).
pub(crate) fn irregular_sites_with_rim(
    crystal: &BuiltCrystal,
    rim: &HashSet<usize>,
    lat_info: &LatticeCoreInfo,
) -> HashSet<usize> {
    let adj: Vec<Vec<usize>> = crystal.adj.iter().map(|es| es.iter().map(|e| e.to).collect()).collect();
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let drim = multi_bfs(&adj, rim, None);
    (0..adj.len())
        .filter(|&i| {
            drim.get(&i).copied().unwrap_or(usize::MAX) > lat_info.r_det
                && !lat_info.perfect_sigs.contains(&layer_signature(&adj, &deg, i, lat_info.r_det))
        })
        .collect()
}

/// Extract the degree-labelled core graph given an explicit `rim` vertex set.
pub fn core_graph_with_rim(
    crystal: &BuiltCrystal,
    rim: &HashSet<usize>,
    lat_info: &LatticeCoreInfo,
    radius: usize,
) -> Result<UnGraph<u16, ()>, String> {
    let adj: Vec<Vec<usize>> = crystal.adj.iter().map(|es| es.iter().map(|e| e.to).collect()).collect();
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let drim = multi_bfs(&adj, rim, None);
    let mut irregular = HashSet::new();
    for i in 0..adj.len() {
        if drim.get(&i).copied().unwrap_or(usize::MAX) > lat_info.r_det
            && !lat_info.perfect_sigs.contains(&layer_signature(&adj, &deg, i, lat_info.r_det))
        {
            irregular.insert(i);
        }
    }
    let ball = multi_bfs(&adj, &irregular, Some(radius));
    if ball.keys().any(|i| drim.get(i).copied().unwrap_or(usize::MAX) <= lat_info.r_det) {
        return Err("the core ball reaches the patch rim: build with a larger R".to_string());
    }
    let mut ball_nodes: Vec<usize> = ball.into_keys().collect();
    ball_nodes.sort_unstable();
    let mut g = UnGraph::<u16, ()>::new_undirected();
    let mut node_map = HashMap::with_capacity(ball_nodes.len());
    for &u in &ball_nodes {
        node_map.insert(u, g.add_node(deg[u] as u16));
    }
    for &(u, v) in &crystal.edges {
        if let (Some(&nu), Some(&nv)) = (node_map.get(&u), node_map.get(&v)) {
            g.add_edge(nu, nv, ());
        }
    }
    Ok(g)
}

/// Extract the degree-labelled core graph on an explicit disk patch `(centre, r_patch)`.
pub fn core_graph_on_patch(
    crystal: &BuiltCrystal,
    lat_info: &LatticeCoreInfo,
    centre: [f64; 2],
    r_patch: f64,
    radius: usize,
) -> Result<UnGraph<u16, ()>, String> {
    let rim = patch_rim(crystal, lat_info, centre, r_patch);
    core_graph_with_rim(crystal, &rim, lat_info, radius)
}

/// Extract the degree-labelled core graph on `crystal`, inferring the patch disk from `ref_pos`.
pub fn core_graph(
    crystal: &BuiltCrystal,
    lat_info: &LatticeCoreInfo,
    radius: usize,
) -> Result<UnGraph<u16, ()>, String> {
    let (centre, r_patch) = infer_patch_disk(crystal);
    core_graph_on_patch(crystal, lat_info, centre, r_patch, radius)
}

/// Exact VF2 degree-labelled graph isomorphism on core graphs.
pub fn isomorphic_cores(g1: &UnGraph<u16, ()>, g2: &UnGraph<u16, ()>) -> bool {
    if g1.node_count() != g2.node_count() || g1.edge_count() != g2.edge_count() {
        return false;
    }
    g1.node_count() == 0 || is_isomorphic_matching(g1, g2, |a, b| a == b, |_, _| true)
}
