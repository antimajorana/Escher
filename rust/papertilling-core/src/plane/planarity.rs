//! Planar embeddedness diagnostics: crossings, ghost stubs, overlaps, folds, and crowding.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::core_graph::{cross2, dot2, frame_faces, norm2, sub2};
use crate::types::BuiltCrystal;

pub const EPS: f64 = 1e-9;
pub const REL_TOL: f64 = 0.01;
pub const CROWD_REL: f64 = 0.35;
pub const AREA_TOL: f64 = 1e-3;

/// Full planarity and drawing-faithfulness report matching `planarity.py` and `drawing_check.py`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanarityReport {
    pub crossings: usize,
    pub plain_crossings: usize,
    pub stub_crossings: usize,
    pub ghost_bonds: usize,
    pub overlaps: usize,
    pub overlap_pairs: usize,
    pub sites_on_bonds: usize,
    pub folded: Option<usize>,
    pub crowded: usize,
    pub crowded_pairs: usize,
    pub crowded_on: usize,
    pub crowd_tol: f64,
    pub tol: f64,
    pub bad: usize,
    pub embedded: bool,
    pub bad_sites: HashSet<usize>,
    pub crowded_sites: HashSet<usize>,
}

pub(crate) fn median_slice(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(f64::total_cmp);
    let m = vals.len();
    if m % 2 == 1 { vals[m / 2] } else { 0.5 * (vals[m / 2 - 1] + vals[m / 2]) }
}

/// Does `crystal` carry drawing-free bond reference frames (`ang` / `vec`)?
/// Walls (`wall_sigma` / `side`) and non-orientable Möbius seams have `has_frames == false` (`folded = None`).
pub fn has_frames(crystal: &BuiltCrystal) -> bool {
    !crystal.edges.is_empty()
        && crystal.wall_sigma.is_none()
        && !crystal.is_non_orientable
        && crystal.side.iter().all(Option::is_none)
        && crystal.adj.iter().flatten().all(|e| e.ang.is_some() || norm2(e.vec) > 1e-12)
}

pub(crate) fn component_roots(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for &(u, v) in edges {
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        parent[ru] = rv;
    }
    (0..n).map(|i| find(&mut parent, i)).collect()
}

/// Partition combinatorial `frame_faces` into `(inner_faces, outer_faces)` (longest walk per component).
pub fn split_frame_faces(crystal: &BuiltCrystal) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let all = frame_faces(crystal);
    let roots = component_roots(crystal.pos.len(), &crystal.edges);
    let mut outer_by_comp: BTreeMap<usize, usize> = BTreeMap::new();
    for (i, f) in all.iter().enumerate() {
        if f.is_empty() {
            continue;
        }
        let c = roots[f[0]];
        match outer_by_comp.get(&c) {
            Some(&prev) if all[prev].len() >= f.len() => {}
            _ => {
                outer_by_comp.insert(c, i);
            }
        }
    }
    let mut skip: Vec<usize> = outer_by_comp.into_values().collect();
    skip.sort_unstable();
    let skip_set: HashSet<usize> = skip.iter().copied().collect();
    let inner = all.iter().enumerate().filter(|(i, _)| !skip_set.contains(i)).map(|(_, f)| f.clone()).collect();
    let outer = skip.into_iter().map(|i| all[i].clone()).collect();
    (inner, outer)
}

/// Smallest median bond length across bond `kind`s (`1.0` when `lens` is empty).
pub fn bond_unit(lens: &[f64], kinds: &[usize]) -> f64 {
    if lens.is_empty() {
        return 1.0;
    }
    let mut by_kind: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (&l, &k) in lens.iter().zip(kinds) {
        by_kind.entry(k).or_default().push(l);
    }
    by_kind
        .into_values()
        .map(|mut v| median_slice(&mut v))
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn signed_area(pos: &[[f64; 2]], face: &[usize]) -> f64 {
    let n = face.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        s += cross2(pos[face[i]], pos[face[(i + 1) % n]]);
    }
    0.5 * s
}

fn near_segment_pairs(a: &[[f64; 2]], b: &[[f64; 2]], pad: f64, cell_h: f64) -> Vec<(usize, usize)> {
    let n = a.len();
    if n < 2 {
        return Vec::new();
    }
    let h = cell_h.max(1e-6);
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for k in 0..n {
        let lk = [a[k][0].min(b[k][0]) - pad, a[k][1].min(b[k][1]) - pad];
        let hk = [a[k][0].max(b[k][0]) + pad, a[k][1].max(b[k][1]) + pad];
        lo.push(lk);
        hi.push(hk);
        let (x0, x1) = ((lk[0] / h).floor() as i64, (hk[0] / h).floor() as i64);
        let (y0, y1) = ((lk[1] / h).floor() as i64, (hk[1] / h).floor() as i64);
        if (x1 - x0 + 1).saturating_mul(y1 - y0 + 1) > 400 {
            // Pathological giant bond: fall back to direct scan below.
            grid.clear();
            let mut pairs = Vec::new();
            for i in 0..n {
                let li = [a[i][0].min(b[i][0]) - pad, a[i][1].min(b[i][1]) - pad];
                let hii = [a[i][0].max(b[i][0]) + pad, a[i][1].max(b[i][1]) + pad];
                for j in (i + 1)..n {
                    let lj = [a[j][0].min(b[j][0]) - pad, a[j][1].min(b[j][1]) - pad];
                    let hij = [a[j][0].max(b[j][0]) + pad, a[j][1].max(b[j][1]) + pad];
                    if lj[0] <= hii[0] && li[0] <= hij[0] && lj[1] <= hii[1] && li[1] <= hij[1] {
                        pairs.push((i, j));
                    }
                }
            }
            return pairs;
        }
        for cx in x0..=x1 {
            for cy in y0..=y1 {
                grid.entry((cx, cy)).or_default().push(k);
            }
        }
    }
    let mut raw: Vec<(usize, usize)> = Vec::new();
    for bucket in grid.values() {
        for p in 0..bucket.len() {
            for q in (p + 1)..bucket.len() {
                let (u, v) = (bucket[p].min(bucket[q]), bucket[p].max(bucket[q]));
                if u != v && lo[v][0] <= hi[u][0] && lo[u][0] <= hi[v][0] && lo[v][1] <= hi[u][1] && lo[u][1] <= hi[v][1] {
                    raw.push((u, v));
                }
            }
        }
    }
    raw.sort_unstable();
    raw.dedup();
    raw
}

fn crossing_pairs_on_candidates(
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    ea: &[usize],
    eb: &[usize],
    cands: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut hits = Vec::new();
    for &(k, j) in cands {
        if ea[j] == ea[k] || ea[j] == eb[k] || eb[j] == ea[k] || eb[j] == eb[k] {
            continue;
        }
        let (r, q, w) = (sub2(b[k], a[k]), sub2(b[j], a[j]), sub2(a[j], a[k]));
        let den = cross2(r, q);
        if den.abs() <= 1e-14 * norm2(r) * norm2(q) {
            continue;
        }
        let (t, u) = (cross2(w, q) / den, cross2(w, r) / den);
        if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
            hits.push((k, j));
        }
    }
    hits
}

fn self_crossing_face(pos: &[[f64; 2]], face: &[usize]) -> bool {
    let m = face.len();
    if m < 4 {
        return false;
    }
    for k in 0..m {
        let (u1, v1) = (face[k], face[(k + 1) % m]);
        for j in (k + 1)..m {
            let (u2, v2) = (face[j], face[(j + 1) % m]);
            if u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                continue;
            }
            let (r, q, w) = (sub2(pos[v1], pos[u1]), sub2(pos[v2], pos[u2]), sub2(pos[u2], pos[u1]));
            let den = cross2(r, q);
            if den.abs() <= 1e-14 * norm2(r) * norm2(q) {
                continue;
            }
            let (t, u) = (cross2(w, q) / den, cross2(w, r) / den);
            if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
                return true;
            }
        }
    }
    false
}

/// Evaluate planarity on custom vertex positions `pos` (with optional `plain_only` ghost-bond filtering
/// and precomputed `inner_faces` for fast repeated checks inside `embed` / `relax_guarded`).
pub(crate) fn check_positions(
    crystal: &BuiltCrystal,
    pos: &[[f64; 2]],
    plain_only: bool,
    inner_faces_opt: Option<&[Vec<usize>]>,
) -> PlanarityReport {
    let mut plain_edges: Vec<(usize, usize, usize)> = Vec::with_capacity(crystal.edges.len());
    let mut ghost_edges: Vec<(usize, usize)> = Vec::new();
    for &(u, v) in &crystal.edges {
        let eu = crystal.adj[u].iter().find(|e| e.to == v);
        let ev = crystal.adj[v].iter().find(|e| e.to == u);
        let kind = eu.map_or(0, |e| e.kind);
        let is_ghost = plain_only
            && (eu.is_some_and(|e| e.ghost_stub.is_some()) || ev.is_some_and(|e| e.ghost_stub.is_some()));
        if is_ghost {
            ghost_edges.push((u, v));
        } else {
            plain_edges.push((u, v, kind));
        }
    }

    let m = plain_edges.len();
    let mut pa = Vec::with_capacity(m);
    let mut pb = Vec::with_capacity(m);
    let mut pea = Vec::with_capacity(m);
    let mut peb = Vec::with_capacity(m);
    let mut pkinds = Vec::with_capacity(m);
    let mut plens = Vec::with_capacity(m);
    for &(u, v, kind) in &plain_edges {
        pa.push(pos[u]);
        pb.push(pos[v]);
        pea.push(u);
        peb.push(v);
        pkinds.push(kind);
        plens.push(norm2(sub2(pos[v], pos[u])));
    }

    let med_len = if m > 0 {
        let mut tmp = plens.clone();
        median_slice(&mut tmp)
    } else {
        1.0
    };
    let tol = REL_TOL * med_len;
    let unit = bond_unit(&plens, &pkinds);
    let crowd_tol = CROWD_REL * unit;
    let cell_h = med_len.max(crowd_tol).max(1e-3);

    let pairs = near_segment_pairs(&pa, &pb, tol, cell_h);
    let plain_hits = crossing_pairs_on_candidates(&pa, &pb, &pea, &peb, &pairs);
    let plain_crossings = plain_hits.len();

    let mut bad_sites: HashSet<usize> = HashSet::new();
    for &(k, j) in &plain_hits {
        bad_sites.extend([pea[k], peb[k], pea[j], peb[j]]);
    }

    let mut all_cross_ends: HashSet<usize> = bad_sites.clone();
    let stub_crossings = if ghost_edges.is_empty() {
        0
    } else {
        let mut fa = pa.clone();
        let mut fb = pb.clone();
        let mut fea = pea.clone();
        let mut feb = peb.clone();
        for &(u, v) in &ghost_edges {
            fa.push(pos[u]);
            fb.push(pos[v]);
            fea.push(u);
            feb.push(v);
        }
        let fpairs = near_segment_pairs(&fa, &fb, 0.0, cell_h);
        let fhits = crossing_pairs_on_candidates(&fa, &fb, &fea, &feb, &fpairs);
        for &(k, j) in &fhits {
            all_cross_ends.extend([fea[k], feb[k], fea[j], feb[j]]);
        }
        fhits.iter().filter(|&&(k, j)| k >= m || j >= m).count()
    };

    // Collinear segment overlaps among plain edges
    let mut overlap_pairs = 0usize;
    let dirs: Vec<[f64; 2]> = (0..m)
        .map(|k| {
            let l = plens[k].max(1e-300);
            [(pb[k][0] - pa[k][0]) / l, (pb[k][1] - pa[k][1]) / l]
        })
        .collect();
    for &(k, j) in &pairs {
        let dk = dirs[k];
        let (waj, wbj) = (sub2(pa[j], pa[k]), sub2(pb[j], pa[k]));
        if cross2(dk, waj).abs() <= tol && cross2(dk, wbj).abs() <= tol {
            let (proj_a, proj_b) = (dot2(waj, dk), dot2(wbj, dk));
            let common = plens[k].min(proj_a.max(proj_b)) - 0.0_f64.max(proj_a.min(proj_b));
            if common > tol {
                overlap_pairs += 1;
                bad_sites.extend([pea[k], peb[k], pea[j], peb[j]]);
            }
        }
    }

    // Spatial point grid for site-site crowding and site-on-bond checks
    let mut pt_grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in pos.iter().enumerate() {
        pt_grid.entry(((p[0] / cell_h).floor() as i64, (p[1] / cell_h).floor() as i64)).or_default().push(i);
    }

    let mut crowded_sites: HashSet<usize> = HashSet::new();
    let mut crowded_pairs = 0usize;
    let c_rad = (crowd_tol / cell_h).ceil() as i64;
    for (i, &pi) in pos.iter().enumerate() {
        let (cx, cy) = ((pi[0] / cell_h).floor() as i64, (pi[1] / cell_h).floor() as i64);
        for dx in -c_rad..=c_rad {
            for dy in -c_rad..=c_rad {
                if let Some(bucket) = pt_grid.get(&(cx + dx, cy + dy)) {
                    for &j in bucket {
                        if j > i && norm2(sub2(pi, pos[j])) < crowd_tol {
                            crowded_pairs += 1;
                            crowded_sites.extend([i, j]);
                        }
                    }
                }
            }
        }
    }

    let pad_sb = tol.max(crowd_tol);
    let mut sites_on_bonds = 0usize;
    let mut crowded_on = 0usize;
    for k in 0..m {
        let lk = [pa[k][0].min(pb[k][0]) - pad_sb, pa[k][1].min(pb[k][1]) - pad_sb];
        let hk = [pa[k][0].max(pb[k][0]) + pad_sb, pa[k][1].max(pb[k][1]) + pad_sb];
        let (x0, x1) = ((lk[0] / cell_h).floor() as i64, (hk[0] / cell_h).floor() as i64);
        let (y0, y1) = ((lk[1] / cell_h).floor() as i64, (hk[1] / cell_h).floor() as i64);
        let collect_check = |i: usize,
                             sites_on_bonds: &mut usize,
                             crowded_on: &mut usize,
                             bad_sites: &mut HashSet<usize>,
                             crowded_sites: &mut HashSet<usize>| {
            if i == pea[k] || i == peb[k] {
                return;
            }
            let pi = pos[i];
            if pi[0] < lk[0] || pi[0] > hk[0] || pi[1] < lk[1] || pi[1] > hk[1] {
                return;
            }
            let w = sub2(pi, pa[k]);
            let along = dot2(w, dirs[k]);
            let off = cross2(dirs[k], w).abs();
            if off <= tol && along > tol && along < plens[k] - tol {
                *sites_on_bonds += 1;
                bad_sites.extend([i, pea[k], peb[k]]);
            }
            let (da, db) = (norm2(w), norm2(sub2(pi, pb[k])));
            if off < crowd_tol && along >= 0.0 && along <= plens[k] && da >= crowd_tol && db >= crowd_tol {
                *crowded_on += 1;
                crowded_sites.extend([i, pea[k], peb[k]]);
            }
        };
        if (x1 - x0 + 1).saturating_mul(y1 - y0 + 1) > 400 {
            for i in 0..pos.len() {
                collect_check(i, &mut sites_on_bonds, &mut crowded_on, &mut bad_sites, &mut crowded_sites);
            }
        } else {
            for cx in x0..=x1 {
                for cy in y0..=y1 {
                    if let Some(bucket) = pt_grid.get(&(cx, cy)) {
                        for &i in bucket {
                            collect_check(i, &mut sites_on_bonds, &mut crowded_on, &mut bad_sites, &mut crowded_sites);
                        }
                    }
                }
            }
        }
    }

    let owned_inner;
    let inner_ref: Option<&[Vec<usize>]> = match inner_faces_opt {
        Some(f) => Some(f),
        None if has_frames(crystal) => {
            owned_inner = split_frame_faces(crystal).0;
            Some(&owned_inner)
        }
        None => None,
    };

    let folded = inner_ref.map(|faces| {
        if faces.is_empty() {
            return 0;
        }
        let areas: Vec<f64> = faces.iter().map(|f| signed_area(pos, f)).collect();
        let pos_cnt = areas.iter().filter(|&&a| a > 0.0).count();
        let neg_cnt = areas.iter().filter(|&&a| a < 0.0).count();
        let maj_pos = pos_cnt >= neg_cnt;
        let mut abs_areas: Vec<f64> = areas.iter().map(|a| a.abs()).collect();
        let min_area = AREA_TOL * median_slice(&mut abs_areas);
        let mut cnt = 0usize;
        for (f, &a) in faces.iter().zip(&areas) {
            let wrong_sign = if maj_pos { a <= 0.0 } else { a >= 0.0 };
            let tiny = a.abs() < min_area;
            let selfx = !wrong_sign
                && !tiny
                && f.len() >= 4
                && f.iter().any(|v| all_cross_ends.contains(v))
                && self_crossing_face(pos, f);
            if wrong_sign || tiny || selfx {
                cnt += 1;
                bad_sites.extend(f.iter().copied());
            }
        }
        cnt
    });

    let overlaps = overlap_pairs + sites_on_bonds;
    let crowded = crowded_pairs + crowded_on;
    let bad = plain_crossings + overlaps + crowded + folded.unwrap_or(0);
    PlanarityReport {
        crossings: plain_crossings,
        plain_crossings,
        stub_crossings,
        ghost_bonds: ghost_edges.len(),
        overlaps,
        overlap_pairs,
        sites_on_bonds,
        folded,
        crowded,
        crowded_pairs,
        crowded_on,
        crowd_tol,
        tol,
        bad,
        embedded: bad == 0,
        bad_sites,
        crowded_sites,
    }
}

/// Check planar embeddedness of `crystal` on its current `crystal.pos` (`plain_only = true`).
pub fn check(crystal: &BuiltCrystal) -> PlanarityReport {
    check_positions(crystal, &crystal.pos, true, None)
}
