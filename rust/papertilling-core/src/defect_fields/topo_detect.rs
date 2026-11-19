//! Topological (Frank) charges of vertices and faces read from a planar bond graph, and connected defective clusters
//! (`#036`, `#068`). Rim vertices (on the unbounded face) carry no charge.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::{BondGraph, PlanarFaces, RefLattice, ang, sub};

/// Connected set of defective interior vertices and faces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefectCluster {
    pub vertices: Vec<usize>,
    pub faces: Vec<usize>,
    /// Net Frank angle (degrees) of the cluster.
    pub frank_deg: f64,
    pub centre: [f64; 2],
    /// Whether a member face touches the unbounded face (charge then unreliable).
    pub touches_rim: bool,
}

/// Per-vertex and per-face Frank charges (degrees) and the defective clusters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopoCharges {
    pub vertex_frank: Vec<f64>,
    pub face_frank: Vec<f64>,
    pub clusters: Vec<DefectCluster>,
}

fn round6(x: f64) -> f64 {
    let r = (x * 1e6).round() / 1e6;
    if r == 0.0 { 0.0 } else { r }
}

/// Counter-clockwise angle (radians, in `(0, 2 pi]`) at `b` from ray `b -> c` to ray `b -> a`.
pub(crate) fn corner(pos: &[[f64; 2]], a: usize, b: usize, c: usize) -> f64 {
    let d = (ang(sub(pos[a], pos[b])) - ang(sub(pos[c], pos[b]))).rem_euclid(2.0 * PI);
    if d <= 1e-12 { 2.0 * PI } else { d }
}

fn find(p: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while p[r] != r {
        r = p[r];
    }
    let mut y = x;
    while p[y] != r {
        let n = p[y];
        p[y] = r;
        y = n;
    }
    r
}

fn union(p: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(p, a), find(p, b));
    if ra != rb {
        p[ra.max(rb)] = ra.min(rb);
    }
}

/// Vertex charge `360 - sum(snapped corners)` (interior vertices only) and face charge
/// `sum(snapped corners) - (p - 2) 180`, then clusters of defective vertices / faces linked by edges or incidence.
/// A vertex is defective when its charge is non-zero or its coordination is not an ideal one.
pub fn topological_charges(g: &BondGraph, pf: &PlanarFaces, rl: &RefLattice) -> TopoCharges {
    let nv = g.pos.len();
    let mut vertex_frank = vec![0.0; nv];
    let mut bad_v = vec![false; nv];
    for v in 0..nv {
        let lst = &g.nbrs[v];
        if pf.rim[v] || lst.is_empty() {
            continue;
        }
        let m = lst.len();
        let tot: f64 = (0..m).map(|k| rl.snap_corner(corner(&g.pos, lst[(k + 1) % m], v, lst[k]).to_degrees())).sum();
        vertex_frank[v] = round6(360.0 - tot);
        bad_v[v] = vertex_frank[v] != 0.0 || !rl.degrees.contains(&m);
    }
    let face_frank: Vec<f64> = pf.faces.iter().map(|f| {
        let p = f.len();
        let tot: f64 = (0..p).map(|k| rl.snap_corner(corner(&g.pos, f[(k + p - 1) % p], f[k], f[(k + 1) % p]).to_degrees())).sum();
        round6(tot - (p as f64 - 2.0) * 180.0)
    }).collect();
    let nf = pf.faces.len();
    let bad_f: Vec<bool> = face_frank.iter().map(|&q| q != 0.0).collect();
    let mut parent: Vec<usize> = (0..nv + nf).collect();
    for &(u, v) in &g.edges {
        if bad_v[u] && bad_v[v] {
            union(&mut parent, u, v);
        }
    }
    for (fi, f) in pf.faces.iter().enumerate() {
        if !bad_f[fi] {
            continue;
        }
        for &v in f {
            if bad_v[v] {
                union(&mut parent, v, nv + fi);
            }
        }
        for nb in pf.face_neighbours(fi) {
            if bad_f[nb] {
                union(&mut parent, nv + fi, nv + nb);
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, (Vec<usize>, Vec<usize>)> = std::collections::BTreeMap::new();
    for v in (0..nv).filter(|&v| bad_v[v]) {
        groups.entry(find(&mut parent, v)).or_default().0.push(v);
    }
    for fi in (0..nf).filter(|&f| bad_f[f]) {
        groups.entry(find(&mut parent, nv + fi)).or_default().1.push(fi);
    }
    let clusters = groups.into_values().map(|(vertices, faces)| {
        let frank = vertices.iter().map(|&v| vertex_frank[v]).sum::<f64>() + faces.iter().map(|&f| face_frank[f]).sum::<f64>();
        let pts: Vec<[f64; 2]> = vertices.iter().map(|&v| g.pos[v]).chain(faces.iter().map(|&f| face_centre(g, pf, f))).collect();
        let n = pts.len().max(1) as f64;
        let centre = [pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n];
        let touches_rim = faces.iter().any(|&f| pf.faces[f].iter().any(|&v| pf.rim[v]))
            || vertices.iter().any(|&v| g.nbrs[v].iter().any(|&w| pf.rim[w]));
        DefectCluster { vertices, faces, frank_deg: round6(frank), centre, touches_rim }
    }).collect();
    TopoCharges { vertex_frank, face_frank, clusters }
}

pub(crate) fn face_centre(g: &BondGraph, pf: &PlanarFaces, fi: usize) -> [f64; 2] {
    let f = &pf.faces[fi];
    let n = f.len() as f64;
    [f.iter().map(|&v| g.pos[v][0]).sum::<f64>() / n, f.iter().map(|&v| g.pos[v][1]).sum::<f64>() / n]
}
