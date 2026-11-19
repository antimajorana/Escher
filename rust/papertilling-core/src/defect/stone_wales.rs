//! Stone-Wales bond rotation: rotate an interior bond `(u, v)` by +90 deg about its midpoint and transfer, at each
//! endpoint `x` with partner `y`, the neighbour that follows `y` counter-clockwise around `x` to `y`. The two faces on
//! either side of the bond lose one site each and the two end faces gain one: `(p1, p2, q1, q2) -> (p1-1, p2-1, q1+1, q2+1)`
//! (a 5-7-7-5 core on the honeycomb). No lattice cut is made, so the holonomy is trivial.
//!
//! Rest lengths: every bond at `u` or `v` gets its direction from the new reference geometry but keeps its original
//! length (the rotated bond keeps its length exactly). Those 8 directed bonds therefore do not satisfy
//! `ref_pos[u] + vec = ref_pos[to]` (`check_frames` counts them) until the crystal is relaxed; this is what makes the
//! optional GSV relaxation non-trivial. GSV relaxation has a free boundary; the result is rigidly re-aligned (Kabsch)
//! onto the input rim, so the rim stays in place only up to the elastic strain it carries.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::charge::{corners_order, frame_faces, outer_face_index};
use crate::defect::cone_geometry::{norm2, sub2};
use crate::defect::gsv_elastic::relax_elastic_2d;
use crate::defect::point_defects::{edges_from_adj, plain_bond};
use crate::types::{BuiltCrystal, Cell2D};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoneWalesResult {
    pub crystal: BuiltCrystal,
    pub flipped_edge: (usize, usize),
    pub before_rings: [usize; 4],
    pub after_rings: [usize; 4],
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub holonomy_badge: crate::defect::burgers_indices::StaticStr,
    #[serde(deserialize_with = "crate::defect::burgers_indices::deser_static_str")]
    pub relax_badge: crate::defect::burgers_indices::StaticStr,
}

fn face_of_dir(faces: &[Vec<usize>]) -> HashMap<(usize, usize), usize> {
    let mut m = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..f.len() {
            m.insert((f[k], f[(k + 1) % f.len()]), fi);
        }
    }
    m
}

fn rot90_about(p: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let d = sub2(p, c);
    [c[0] - d[1], c[1] + d[0]]
}

/// Rigid (proper) Kabsch fit of `moved[idx]` onto `target[idx]`, applied to all of `moved`.
pub(crate) fn kabsch_align(moved: &[[f64; 2]], target: &[[f64; 2]], idx: &[usize]) -> Vec<[f64; 2]> {
    if idx.is_empty() {
        return moved.to_vec();
    }
    let k = idx.len() as f64;
    let (mut cm, mut ct) = ([0.0; 2], [0.0; 2]);
    for &i in idx {
        for a in 0..2 {
            cm[a] += moved[i][a] / k;
            ct[a] += target[i][a] / k;
        }
    }
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for &i in idx {
        let (p, q) = (sub2(moved[i], cm), sub2(target[i], ct));
        sxx += p[0] * q[0] + p[1] * q[1];
        sxy += p[0] * q[1] - p[1] * q[0];
    }
    let th = sxy.atan2(sxx);
    let (s, c) = th.sin_cos();
    moved.iter().map(|p| {
        let d = sub2(*p, cm);
        [ct[0] + c * d[0] - s * d[1], ct[1] + s * d[0] + c * d[1]]
    }).collect()
}

/// Stone-Wales flip of `crystal.edges[edge_idx]` (see the module doc); relaxes with GSV when `cell_for_relax` is given.
pub fn stone_wales_flip(crystal: &BuiltCrystal, edge_idx: usize, cell_for_relax: Option<&Cell2D>) -> Result<StoneWalesResult, String> {
    let &(u, v) = crystal.edges.get(edge_idx).ok_or_else(|| format!("edge index {edge_idx} out of range ({} edges)", crystal.edges.len()))?;
    if crystal.is_non_orientable {
        return Err("Stone-Wales flip needs an orientable crystal".into());
    }
    for x in [u, v] {
        if crystal.adj[x].iter().any(|e| e.seam.is_some()) || crystal.adj.iter().flatten().any(|e| e.to == x && e.seam.is_some()) {
            return Err(format!("site {x} has a bond across a seam; Stone-Wales flips across seams are not supported"));
        }
        if crystal.adj[x].len() < 3 {
            return Err(format!("site {x} has degree {} (< 3)", crystal.adj[x].len()));
        }
    }
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return Err("crystal has no faces".into());
    }
    let k_out = outer_face_index(&faces);
    let rim: HashSet<usize> = faces[k_out].iter().copied().collect();
    if rim.contains(&u) || rim.contains(&v) {
        return Err(format!("edge ({u}, {v}) touches the rim"));
    }
    let corners = corners_order(crystal);
    let next_ccw = |x: usize, y: usize| -> usize {
        let lst = &corners[x];
        let k = lst.iter().position(|&(_, w)| w == y).unwrap_or(0);
        lst[(k + 1) % lst.len()].1
    };
    let (tu, tv) = (next_ccw(u, v), next_ccw(v, u));
    if tu == v || tv == u || tu == tv || crystal.adj[v].iter().any(|e| e.to == tu) || crystal.adj[u].iter().any(|e| e.to == tv) {
        return Err("degenerate neighbourhood around the edge (shared or repeated neighbours)".into());
    }
    let fd = face_of_dir(&faces);
    let fid = |a: usize, b: usize| fd.get(&(a, b)).copied().ok_or_else(|| format!("no face walk through ({a}, {b})"));
    let quad = [fid(u, v)?, fid(v, u)?, fid(u, tu)?, fid(v, tv)?];
    if quad.iter().collect::<HashSet<_>>().len() != 4 || quad.contains(&k_out) {
        return Err("the four faces around the edge must be distinct interior faces".into());
    }
    let before = quad.map(|f| faces[f].len());
    if before[0] < 4 || before[1] < 4 {
        return Err(format!("faces beside the edge have sizes {} and {} (need >= 4)", before[0], before[1]));
    }
    // geometry
    let mut out = crystal.clone();
    let (cp, cr) = ([0.5 * (crystal.pos[u][0] + crystal.pos[v][0]), 0.5 * (crystal.pos[u][1] + crystal.pos[v][1])],
        [0.5 * (crystal.ref_pos[u][0] + crystal.ref_pos[v][0]), 0.5 * (crystal.ref_pos[u][1] + crystal.ref_pos[v][1])]);
    for x in [u, v] {
        out.pos[x] = rot90_about(crystal.pos[x], cp);
        out.ref_pos[x] = rot90_about(crystal.ref_pos[x], cr);
    }
    // rewiring: (u, tu) -> (v, tu), (v, tv) -> (u, tv); rest lengths keep their old values
    let rest = |a: usize, b: usize| crystal.adj[a].iter().find(|e| e.to == b).map(|e| (norm2(e.vec), e.kind));
    let (ru, kind_u) = rest(u, tu).ok_or("missing bond (u, tu)")?;
    let (rv, kind_v) = rest(v, tv).ok_or("missing bond (v, tv)")?;
    for (a, b) in [(u, tu), (v, tv)] {
        out.adj[a].retain(|e| e.to != b);
        out.adj[b].retain(|e| e.to != a);
    }
    out.adj[v].push(plain_bond(tu, [0.0, 0.0], kind_u));
    out.adj[tu].push(plain_bond(v, [0.0, 0.0], kind_u));
    out.adj[u].push(plain_bond(tv, [0.0, 0.0], kind_v));
    out.adj[tv].push(plain_bond(u, [0.0, 0.0], kind_v));
    let mut rest_len: HashMap<(usize, usize), f64> = HashMap::new();
    for x in [u, v] {
        for e in &crystal.adj[x] {
            rest_len.insert((x.min(e.to), x.max(e.to)), norm2(e.vec));
        }
    }
    rest_len.insert((v.min(tu), v.max(tu)), ru);
    rest_len.insert((u.min(tv), u.max(tv)), rv);
    for a in 0..out.adj.len() {
        for e in out.adj[a].iter_mut() {
            if a == u || a == v || e.to == u || e.to == v {
                let d = sub2(out.ref_pos[e.to], out.ref_pos[a]);
                let l = rest_len.get(&(a.min(e.to), a.max(e.to))).copied().unwrap_or(norm2(d));
                let nd = norm2(d).max(1e-12);
                e.vec = [d[0] * l / nd, d[1] * l / nd];
                e.ang = None;
            }
        }
    }
    out.edges = edges_from_adj(&out.adj);
    // verify the ring transform on the new faces
    let new_faces = frame_faces(&out);
    if new_faces.len() != faces.len() || out.edges.len() != crystal.edges.len() {
        return Err("Stone-Wales flip changed the face or edge count".into());
    }
    let old_sets: Vec<HashSet<usize>> = quad.iter().map(|&f| faces[f].iter().copied().collect()).collect();
    let old_keys: HashSet<Vec<usize>> = faces.iter().map(|f| { let mut s = f.clone(); s.sort_unstable(); s }).collect();
    let changed: Vec<&Vec<usize>> = new_faces.iter().filter(|f| { let mut s = (*f).clone(); s.sort_unstable(); !old_keys.contains(&s) }).collect();
    let mut after = [0usize; 4];
    for (k, os) in old_sets.iter().enumerate() {
        let best = changed.iter().max_by_key(|f| f.iter().filter(|x| os.contains(x)).count()).ok_or("no changed faces after the flip")?;
        after[k] = best.len();
    }
    let expect = [before[0] - 1, before[1] - 1, before[2] + 1, before[3] + 1];
    if after != expect || changed.len() != 4 {
        return Err(format!("unexpected ring transform {before:?} -> {after:?} (expected {expect:?})"));
    }
    let relax_badge = match cell_for_relax {
        Some(_) => {
            let res = relax_elastic_2d(&out)?;
            if res.pos.len() != out.pos.len() {
                return Err("relaxation returned a wrong number of sites".into());
            }
            let rim_idx: Vec<usize> = rim.iter().copied().collect();
            out.pos = kabsch_align(&res.pos, &crystal.pos, &rim_idx);
            "[COMPUTATIONAL EVIDENCE]"
        }
        None => "[NOT RELAXED]",
    };
    Ok(StoneWalesResult {
        crystal: out,
        flipped_edge: (u, v),
        before_rings: before,
        after_rings: after,
        holonomy_badge: "[PROVED]",
        relax_badge,
    })
}
