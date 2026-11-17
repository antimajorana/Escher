//! Disk cuts, component pruning, and seam zipping for terminated half-walls.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::f64::consts::PI;

use crate::defect::charge::frame_faces;
use crate::defect::core_graph::{
    LatticeCoreInfo, PerfectLattice, SiteKey, TWO_PI, add2, cross2, dot2, mat2_inv, mat2_vec,
    norm2, rot_mat, sub2,
};
use crate::types::{BuiltCrystal, DirectedBondData, Isometry2};

#[derive(Clone, Copy, Debug)]
pub(super) struct RawEdge {
    pub(super) vec: [f64; 2],
    pub(super) kind: usize,
    pub(super) seam: Option<Isometry2>,
    pub(super) across: bool,
}

#[derive(Default)]
pub(super) struct RawHalfWall {
    pub(super) sites: Vec<(SiteKey, usize)>,
    pub(super) x_ref: Vec<[f64; 2]>,
    pub(super) phi: Vec<f64>,
    pub(super) p_draw: Vec<[f64; 2]>,
    pub(super) edges: BTreeMap<(usize, usize), RawEdge>,
}

pub(super) struct CutGeom<'a> {
    pub(super) lat: &'a PerfectLattice,
    pub(super) allk: &'a [SiteKey],
    pub(super) pos_all: &'a HashMap<SiteKey, [f64; 2]>,
    pub(super) at: [f64; 2],
    pub(super) ctr: [f64; 2],
    pub(super) u_hat: [f64; 2],
    pub(super) n_hat: [f64; 2],
    pub(super) t_rel: [f64; 2],
    pub(super) r_patch: f64,
}

impl RawHalfWall {
    fn add_undirected(&mut self, u: usize, v: usize, vec_uv: [f64; 2], kind: usize) {
        self.edges.insert((u, v), RawEdge { vec: vec_uv, kind, seam: None, across: false });
        self.edges.insert((v, u), RawEdge { vec: [-vec_uv[0], -vec_uv[1]], kind, seam: None, across: false });
    }
}

fn polar_phi(r: [f64; 2], u_hat: [f64; 2]) -> f64 {
    if norm2(r) < 1e-9 { 0.0 } else { cross2(u_hat, r).atan2(dot2(r, u_hat)).rem_euclid(TWO_PI) }
}

fn seg_cross(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2], tol: f64) -> bool {
    let (a, b) = (sub2(p2, p1), sub2(p4, p3));
    let den = cross2(a, b);
    if den.abs() < 1e-12 { return false; }
    let rp = sub2(p3, p1);
    let (u, v) = (cross2(rp, b) / den, cross2(rp, a) / den);
    u > tol && u < 1.0 - tol && v > tol && v < 1.0 - tol
}

fn in_corner_sector(edges: &BTreeMap<(usize, usize), RawEdge>, face: &[usize], pos: usize, vec_new: [f64; 2]) -> bool {
    let n = face.len();
    let (u, prev_v, next_v) = (face[pos], face[(pos + n - 1) % n], face[(pos + 1) % n]);
    let (ep, en) = (edges[&(u, prev_v)].vec, edges[&(u, next_v)].vec);
    let a_prev = ep[1].atan2(ep[0]).rem_euclid(TWO_PI);
    let a_next = en[1].atan2(en[0]).rem_euclid(TWO_PI);
    let a_new = vec_new[1].atan2(vec_new[0]).rem_euclid(TWO_PI);
    let raw_gap = (a_prev - a_next).rem_euclid(TWO_PI);
    let gap = if raw_gap == 0.0 { TWO_PI } else { raw_gap };
    let rel = (a_new - a_next).rem_euclid(TWO_PI);
    rel > 1e-6 && rel < gap - 1e-6
}

fn add_chart_edges(raw: &mut RawHalfWall, charts: &[HashMap<SiteKey, usize>], g: &CutGeom<'_>) {
    for here in charts {
        let mut items: Vec<(SiteKey, usize)> = here.iter().map(|(&k, &i)| (k, i)).collect();
        items.sort_unstable();
        for (kl, i) in items {
            let pi = g.pos_all[&kl];
            for (k2, kind) in g.lat.nbrs(kl) {
                if let Some(&j) = here.get(&k2) && j != i {
                    raw.add_undirected(i, j, sub2(g.pos_all[&k2], pi), kind);
                }
            }
        }
    }
}

pub(super) fn cut_pure_slip(g: &CutGeom<'_>) -> RawHalfWall {
    let mut raw = RawHalfWall::default();
    let mut idx_map = HashMap::new();
    let has_shift = norm2(g.t_rel) > 1e-7;
    for &k in g.allk {
        let p = g.pos_all[&k];
        let r = sub2(p, g.at);
        let phi = if dot2(r, g.n_hat).abs() < 1e-9 && dot2(r, g.u_hat) >= -1e-9 { 0.0 } else { polar_phi(r, g.u_hat) };
        let pd = [p[0] + (phi / TWO_PI) * g.t_rel[0], p[1] + (phi / TWO_PI) * g.t_rel[1]];
        if norm2(sub2(pd, g.ctr)) <= g.r_patch {
            idx_map.insert(k, raw.sites.len());
            raw.sites.push((k, 0)); raw.x_ref.push(p); raw.phi.push(phi); raw.p_draw.push(pd);
        }
    }
    for (&k, &i) in &idx_map {
        let p1 = raw.x_ref[i];
        for (k2, kind) in g.lat.nbrs(k) {
            let Some(&j) = idx_map.get(&k2) else { continue };
            if j <= i { continue; }
            let p2 = raw.x_ref[j];
            if has_shift {
                let (d1, d2) = (dot2(sub2(p1, g.at), g.n_hat), dot2(sub2(p2, g.at), g.n_hat));
                if (d1 >= -1e-9 && d2 < -1e-9) || (d2 >= -1e-9 && d1 < -1e-9) {
                    let (s1, s2) = (dot2(sub2(p1, g.at), g.u_hat), dot2(sub2(p2, g.at), g.u_hat));
                    if s1 + (d1 / (d1 - d2)) * (s2 - s1) > 1e-9 { continue; }
                }
            }
            raw.add_undirected(i, j, sub2(p2, p1), kind);
        }
    }
    raw
}

pub(super) fn cut_strip(g: &CutGeom<'_>, tn: f64) -> RawHalfWall {
    let binv = mat2_inv([[-g.t_rel[0], g.u_hat[0]], [-g.t_rel[1], g.u_hat[1]]]);
    let ts_all: HashMap<SiteKey, [f64; 2]> = g.allk.iter().map(|&k| (k, mat2_vec(binv, sub2(g.pos_all[&k], g.at)))).collect();
    let mut raw = RawHalfWall::default();
    let charts = if tn > 0.0 {
        for &k in g.allk {
            let [t, s] = ts_all[&k];
            if s > 1e-9 && (-1e-9..(1.0 - 1e-9)).contains(&t) { continue; }
            let p = g.pos_all[&k];
            let mut phi = polar_phi(sub2(p, g.at), g.u_hat);
            if s > 1e-9 && t >= 1.0 - 1e-9 && phi < PI { phi += TWO_PI; }
            let pd = [p[0] + (phi / TWO_PI) * g.t_rel[0], p[1] + (phi / TWO_PI) * g.t_rel[1]];
            if norm2(sub2(pd, g.ctr)) <= g.r_patch {
                raw.sites.push((k, 0)); raw.x_ref.push(p); raw.phi.push(phi); raw.p_draw.push(pd);
            }
        }
        let c0 = raw.sites.iter().enumerate().filter(|&(_, &(k, _))| ts_all[&k][1] <= 1e-9 || ts_all[&k][0] < -1e-9).map(|(i, &(k, _))| (k, i)).collect();
        let c1 = raw.sites.iter().enumerate().filter(|&(_, &(k, _))| ts_all[&k][0] >= 1.0 - 1e-9).map(|(i, &(k, _))| (k, i)).collect();
        vec![c0, c1]
    } else {
        let (mut orig, mut dup) = (HashMap::new(), HashMap::new());
        for &k in g.allk {
            let ([t, s], p) = (ts_all[&k], g.pos_all[&k]);
            let mut phi = polar_phi(sub2(p, g.at), g.u_hat);
            if s > 1e-9 && (-1e-9..(1.0 - 1e-9)).contains(&t) {
                let p_shift = add2(p, g.t_rel);
                if norm2(sub2(p_shift, g.ctr)) <= g.r_patch {
                    orig.insert(k, raw.sites.len());
                    raw.sites.push((k, 0)); raw.x_ref.push(p); raw.phi.push(TWO_PI); raw.p_draw.push(p_shift);
                }
                if norm2(sub2(p, g.ctr)) <= g.r_patch {
                    dup.insert(k, raw.sites.len());
                    raw.sites.push((k, 1)); raw.x_ref.push(p); raw.phi.push(0.0); raw.p_draw.push(p);
                }
            } else {
                if s > 1e-9 && t < -1e-9 && phi < PI { phi += TWO_PI; }
                if s > 1e-9 && t >= 1.0 - 1e-9 && phi > PI { phi -= TWO_PI; }
                let pd = [p[0] + (phi / TWO_PI) * g.t_rel[0], p[1] + (phi / TWO_PI) * g.t_rel[1]];
                if norm2(sub2(pd, g.ctr)) <= g.r_patch {
                    orig.insert(k, raw.sites.len());
                    raw.sites.push((k, 0)); raw.x_ref.push(p); raw.phi.push(phi); raw.p_draw.push(pd);
                }
            }
        }
        let mut ca: HashMap<SiteKey, usize> = orig.iter().filter(|&(&k, _)| ts_all[&k][1] <= 1e-9 || ts_all[&k][0] >= 1.0 - 1e-9).map(|(&k, &i)| (k, i)).collect();
        ca.extend(dup);
        let cb: HashMap<SiteKey, usize> = orig.iter().filter(|&(&k, _)| ts_all[&k][0] < -1e-9 || (ts_all[&k][1] > 1e-9 && ts_all[&k][0] < 1.0 - 1e-9)).map(|(&k, &i)| (k, i)).collect();
        vec![ca, cb]
    };
    add_chart_edges(&mut raw, &charts, g);
    raw
}

pub(super) fn cut_rot(g: &CutGeom<'_>, alpha: f64) -> RawHalfWall {
    let (theta, eps) = (TWO_PI - alpha, 1e-7);
    let th0_cut = g.u_hat[1].atan2(g.u_hat[0]) - 1e-6;
    let (mut raw, mut key_sites, mut core_i) = (RawHalfWall::default(), HashMap::new(), None);
    for &k in g.allk {
        let p = g.pos_all[&k];
        let r = sub2(p, g.at);
        if norm2(r) < 1e-9 {
            if norm2(sub2(g.at, g.ctr)) <= g.r_patch {
                core_i = Some(raw.sites.len());
                key_sites.insert((k, 0usize), raw.sites.len());
                raw.sites.push((k, 0)); raw.x_ref.push(p); raw.phi.push(0.0); raw.p_draw.push(g.at);
            }
            continue;
        }
        let th_a = (r[1].atan2(r[0]) - th0_cut).rem_euclid(TWO_PI);
        let mut m = 0usize;
        while th_a + TWO_PI * (m as f64) < theta - eps {
            let phi = th_a + TWO_PI * (m as f64);
            let rot_r = mat2_vec(rot_mat(phi * (TWO_PI / theta) - phi), r);
            let pd = [g.at[0] + rot_r[0] + (phi / theta) * g.t_rel[0], g.at[1] + rot_r[1] + (phi / theta) * g.t_rel[1]];
            if norm2(sub2(pd, g.ctr)) <= g.r_patch {
                key_sites.insert((k, m), raw.sites.len());
                raw.sites.push((k, m)); raw.x_ref.push(p); raw.phi.push(phi); raw.p_draw.push(pd);
            }
            m += 1;
        }
    }
    for u in 0..raw.sites.len() {
        let ((k, _), p) = (raw.sites[u], raw.x_ref[u]);
        for (k2, kind) in g.lat.nbrs(k) {
            let q = g.lat.pos(k2);
            if norm2(sub2(q, g.at)) < 1e-9 {
                if let Some(ci) = core_i { raw.add_undirected(u, ci, sub2(q, p), kind); }
                continue;
            }
            if Some(u) == core_i { continue; }
            let (a, bb) = (sub2(p, g.at), sub2(q, g.at));
            if cross2(a, bb).abs() < 1e-9 && dot2(a, bb) < 0.0 { continue; }
            let phq = raw.phi[u] + cross2(a, bb).atan2(dot2(a, bb));
            if phq > -eps && phq < theta - eps {
                let lay = ((phq + eps) / TWO_PI).floor() as isize;
                if lay >= 0 && let Some(&v) = key_sites.get(&(k2, lay as usize)) && v != u {
                    raw.add_undirected(u, v, sub2(q, p), kind);
                }
            }
        }
    }
    raw
}

pub(super) fn keep_main_component(raw: RawHalfWall) -> RawHalfWall {
    let n = raw.sites.len();
    if n == 0 { return raw; }
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in raw.edges.keys() { adj[u].push(v); }
    let (mut seen, mut best) = (vec![false; n], Vec::new());
    for s in 0..n {
        if seen[s] { continue; }
        seen[s] = true;
        let (mut comp, mut dq) = (vec![s], VecDeque::from([s]));
        while let Some(u) = dq.pop_front() {
            for &v in &adj[u] { if !seen[v] { seen[v] = true; comp.push(v); dq.push_back(v); } }
        }
        if comp.len() > best.len() { best = comp; }
    }
    if best.len() == n { return raw; }
    best.sort_unstable();
    let remap: HashMap<usize, usize> = best.iter().copied().enumerate().map(|(nw, old)| (old, nw)).collect();
    let edges = raw.edges.into_iter().filter_map(|((u, v), e)| Some(((remap.get(&u).copied()?, remap.get(&v).copied()?), e))).collect();
    RawHalfWall {
        sites: best.iter().map(|&i| raw.sites[i]).collect(),
        x_ref: best.iter().map(|&i| raw.x_ref[i]).collect(),
        phi: best.iter().map(|&i| raw.phi[i]).collect(),
        p_draw: best.iter().map(|&i| raw.p_draw[i]).collect(),
        edges,
    }
}

pub(super) fn build_crystal_from_raw(raw: &RawHalfWall) -> BuiltCrystal {
    let n = raw.sites.len();
    let (mut adj, mut edges) = (vec![Vec::new(); n], Vec::new());
    for (&(u, v), e) in &raw.edges {
        if u < v { edges.push((u, v)); }
        adj[u].push(DirectedBondData { to: v, vec: e.vec, kind: e.kind, seam: e.seam, ang: None, across: e.across, ghost_stub: None });
    }
    edges.sort_unstable();
    BuiltCrystal {
        pos: raw.p_draw.clone(), ref_pos: raw.x_ref.clone(), basis_idx: raw.sites.iter().map(|&(k, _)| k.2).collect(),
        side: vec![None; n], cone_sites: Vec::new(), edges, adj, wall_sigma: None, is_non_orientable: false,
    }
}

pub(super) fn zip_half_wall(lat_info: &LatticeCoreInfo, raw: &mut RawHalfWall, hol: Isometry2, g: &CutGeom<'_>, theta: f64) {
    let n = raw.sites.len();
    let mut deg = vec![0usize; n];
    for &(u, v) in raw.edges.keys() { if u < v { deg[u] += 1; deg[v] += 1; } }
    let (mut active_faces, outer_set): (Vec<Vec<usize>>, BTreeSet<usize>) = if lat_info.planar {
        let faces = frame_faces(&build_crystal_from_raw(raw));
        if faces.is_empty() { (Vec::new(), BTreeSet::new()) } else {
            let f_outer = faces.into_iter().max_by_key(Vec::len).unwrap();
            let oset = f_outer.iter().copied().collect();
            (vec![f_outer], oset)
        }
    } else {
        (Vec::new(), (0..n).collect())
    };
    let lmax = lat_info.lmax;
    let near = |i: usize| {
        let d = sub2(raw.p_draw[i], g.at);
        norm2(d) > 0.2 * lmax && dot2(d, g.u_hat) > -0.2 * lmax && dot2(d, g.n_hat).abs() < 1.35 * lmax
    };
    let bank_a: Vec<usize> = outer_set.iter().copied().filter(|&i| raw.phi[i] < 0.65 * PI && near(i)).collect();
    let bank_b: Vec<usize> = outer_set.iter().copied().filter(|&i| raw.phi[i] > theta - 0.65 * PI && near(i)).collect();
    let (rh, th) = (hol.m, hol.t);
    let mut cands = Vec::new();
    for &u in &bank_b {
        let qu = add2(mat2_vec(rh, raw.x_ref[u]), th);
        for &v in &bank_a {
            if u == v || raw.edges.contains_key(&(u, v)) { continue; }
            let (d_phys, d_draw) = (norm2(sub2(qu, raw.x_ref[v])), norm2(sub2(raw.p_draw[u], raw.p_draw[v])));
            if d_phys <= 1.75 * lmax || d_draw <= 1.75 * lmax { cands.push((d_phys + 0.2 * d_draw, u, v)); }
        }
    }
    cands.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
    let mut existing_segs: Vec<([f64; 2], [f64; 2])> = raw.edges.keys().filter(|&&(a, b)| a < b).map(|&(a, b)| (raw.p_draw[a], raw.p_draw[b])).collect();
    let (g_fwd, g_rev) = (hol, hol.inverse());
    let rh_t = g_rev.m;
    for (_, u, v) in cands {
        if raw.edges.contains_key(&(u, v)) || (deg[u] >= lat_info.perfect_degree[raw.sites[u].0.2] && deg[v] >= lat_info.perfect_degree[raw.sites[v].0.2]) { continue; }
        let (pu, pv) = (raw.p_draw[u], raw.p_draw[v]);
        if existing_segs.iter().any(|&(s1, s2)| seg_cross(pu, pv, s1, s2, 1e-7)) { continue; }
        let vec_uv = sub2(mat2_vec(rh_t, sub2(raw.x_ref[v], th)), raw.x_ref[u]);
        let vec_vu = sub2(add2(mat2_vec(rh, raw.x_ref[u]), th), raw.x_ref[v]);
        if lat_info.planar {
            let Some((fi, pos_u, pos_v)) = active_faces.iter().enumerate().find_map(|(fi, f)| {
                let pu_idx = f.iter().position(|&x| x == u)?;
                let pv_idx = f.iter().position(|&x| x == v)?;
                (in_corner_sector(&raw.edges, f, pu_idx, vec_uv) && in_corner_sector(&raw.edges, f, pv_idx, vec_vu)).then_some((fi, pu_idx, pv_idx))
            }) else { continue; };
            let (iu, iv) = if pos_u < pos_v { (pos_u, pos_v) } else { (pos_v, pos_u) };
            let f = &active_faces[fi];
            let f1: Vec<usize> = f[iu..=iv].to_vec();
            let f2: Vec<usize> = f[iv..].iter().chain(&f[..=iu]).copied().collect();
            active_faces.splice(fi..=fi, [f1, f2]);
        }
        raw.edges.insert((u, v), RawEdge { vec: vec_uv, kind: 0, seam: Some(g_fwd), across: true });
        raw.edges.insert((v, u), RawEdge { vec: vec_vu, kind: 0, seam: Some(g_rev), across: true });
        existing_segs.push((pu, pv));
        deg[u] += 1; deg[v] += 1;
    }
}
