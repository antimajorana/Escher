//! Unified `{p, q}` constant-curvature switch and `E^2` / `H^2` dispatch.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::GeometryKind;
use crate::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling};

use super::unified_s2::build_spherical_pq;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnifiedPqTiling {
    pub p: usize,
    pub q: usize,
    pub layers: usize,
    pub delta: i64,
    pub geometry_kind: GeometryKind,
    pub curvature_kappa: f64,
    pub euler_chi: i64,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    pub coords_3d: Vec<[f64; 3]>,
    pub coords_2d: Vec<[f64; 2]>,
    pub edges: Vec<(usize, usize)>,
    pub edge_arcs_2d: Vec<Vec<[f64; 2]>>,
    pub edge_arcs_3d: Vec<Vec<[f64; 3]>>,
    pub faces: Vec<Vec<usize>>,
    pub provenance: String,
}

/// Exact integer curvature test `delta = 4 - (p - 2)(q - 2)` for `p >= 2, q >= 2`.
///
/// `delta = 2pq (1/p + 1/q - 1/2)`, so its sign is the sign of the angle excess of the regular
/// `p`-gon with `q` at a vertex: `> 0` spherical, `= 0` Euclidean, `< 0` hyperbolic
/// (Coxeter, *Regular Polytopes*, 3rd ed. 1973, sections 1.7 and 4.5; [PROVED]).
pub fn delta_pq(p: usize, q: usize) -> Result<i64, String> {
    if p < 2 || q < 2 {
        return Err(format!("invalid Schlafli symbol {{{p}, {q}}}: requires p >= 2 and q >= 2"));
    }
    Ok(4 - (p as i64 - 2) * (q as i64 - 2))
}

/// Constant-curvature geometry induced by `delta_pq(p, q)`.
pub fn geometry_kind_pq(p: usize, q: usize) -> Result<GeometryKind, String> {
    let d = delta_pq(p, q)?;
    Ok(if d > 0 {
        GeometryKind::Spherical
    } else if d == 0 {
        GeometryKind::Euclidean
    } else {
        GeometryKind::Hyperbolic
    })
}

/// Realise `{p, q}` in `S^2` (`delta > 0`), `E^2` (`delta == 0`), or `H^2` (`delta < 0`).
pub fn build_unified_pq(p: usize, q: usize, layers: usize) -> Result<UnifiedPqTiling, String> {
    let delta = delta_pq(p, q)?;
    let kappa = (delta as f64) / (2.0 * (p as f64) * (q as f64));
    if delta > 0 {
        return build_spherical_pq(p, q, layers, delta, kappa);
    }
    if delta == 0 {
        return build_euclidean_pq(p, q, layers, kappa);
    }
    let lay = layers.clamp(1, 4);
    let h = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, lay, 0)?;
    let (n_v, n_e, n_f) = (h.poincare_pts.len(), h.edges.len(), h.faces.len());
    let edge_arcs_3d = h
        .edge_arcs
        .iter()
        .map(|arc| arc.iter().map(|&z| poincare_to_hyperboloid_unclamped(z)).collect())
        .collect();
    Ok(UnifiedPqTiling {
        p,
        q,
        layers: h.layers,
        delta,
        geometry_kind: GeometryKind::Hyperbolic,
        curvature_kappa: h.curvature_kappa,
        euler_chi: n_v as i64 - n_e as i64 + n_f as i64,
        n_vertices: n_v,
        n_edges: n_e,
        n_faces: n_f,
        coords_3d: h.hyperboloid_pts,
        coords_2d: h.poincare_pts,
        edges: h.edges,
        edge_arcs_2d: h.edge_arcs,
        edge_arcs_3d,
        faces: h.faces,
        provenance: "[PROVED]".to_string(),
    })
}

fn poincare_to_hyperboloid_unclamped(z: [f64; 2]) -> [f64; 3] {
    let r2 = z[0] * z[0] + z[1] * z[1];
    let d = (1.0 - r2).max(1e-12);
    [2.0 * z[0] / d, 2.0 * z[1] / d, (1.0 + r2) / d]
}

fn qkey2(pt: [f64; 2]) -> (i64, i64) {
    ((pt[0] * 1e6).round() as i64, (pt[1] * 1e6).round() as i64)
}

#[allow(clippy::too_many_arguments)]
fn insert_e2_face(
    c: [f64; 2],
    s: f64,
    base_verts: &[[f64; 2]],
    seen_centres: &mut HashMap<(i64, i64), usize>,
    face_centres: &mut Vec<([f64; 2], f64)>,
    vert_map: &mut HashMap<(i64, i64), usize>,
    coords_2d: &mut Vec<[f64; 2]>,
    faces: &mut Vec<Vec<usize>>,
) -> Option<usize> {
    let ck = qkey2(c);
    if seen_centres.contains_key(&ck) {
        return None;
    }
    let id = faces.len();
    seen_centres.insert(ck, id);
    face_centres.push((c, s));
    let mut cyc = Vec::with_capacity(base_verts.len());
    for &bv in base_verts {
        let pt = [c[0] + s * bv[0], c[1] + s * bv[1]];
        let vid = *vert_map.entry(qkey2(pt)).or_insert_with(|| {
            let nid = coords_2d.len();
            coords_2d.push(pt);
            nid
        });
        cyc.push(vid);
    }
    if s < 0.0 {
        cyc.reverse();
    }
    faces.push(cyc);
    Some(id)
}

fn build_euclidean_pq(p: usize, q: usize, layers: usize, kappa: f64) -> Result<UnifiedPqTiling, String> {
    let lay = layers.clamp(1, 6);
    let rad = 0.5 / (PI / (p as f64)).sin();
    let base_verts: Vec<[f64; 2]> = (0..p)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p as f64) + PI / (p as f64);
            [rad * th.cos(), rad * th.sin()]
        })
        .collect();
    let base_mids: Vec<[f64; 2]> = (0..p)
        .map(|k| {
            let (a, b) = (base_verts[k], base_verts[(k + 1) % p]);
            [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
        })
        .collect();

    let mut vert_map: HashMap<(i64, i64), usize> = HashMap::new();
    let mut coords_2d: Vec<[f64; 2]> = Vec::new();
    let mut face_centres: Vec<([f64; 2], f64)> = Vec::new();
    let mut seen_centres: HashMap<(i64, i64), usize> = HashMap::new();
    let mut faces: Vec<Vec<usize>> = Vec::new();
    let mut shell_bounds = vec![(0usize, 1usize)];

    insert_e2_face(
        [0.0, 0.0],
        1.0,
        &base_verts,
        &mut seen_centres,
        &mut face_centres,
        &mut vert_map,
        &mut coords_2d,
        &mut faces,
    );
    let mut active_verts: HashSet<usize> = faces[0].iter().copied().collect();
    for _ in 0..lay {
        let (start, end) = *shell_bounds.last().unwrap();
        let mut queue: Vec<usize> = (start..end).collect();
        let mut visited_f: HashSet<usize> = queue.iter().copied().collect();
        let mut cursor = 0;
        while cursor < queue.len() {
            let fi = queue[cursor];
            cursor += 1;
            let (c, s) = face_centres[fi];
            // The neighbour across edge k is the point reflection of the face through that edge's
            // midpoint (a symmetry of every regular tiling {3,6}, {4,4}, {6,3}).
            for mid in &base_mids {
                let nc = [c[0] + 2.0 * s * mid[0], c[1] + 2.0 * s * mid[1]];
                let ns = -s;
                let touches_active = base_verts.iter().any(|&bv| {
                    let pt = [nc[0] + ns * bv[0], nc[1] + ns * bv[1]];
                    vert_map.get(&qkey2(pt)).is_some_and(|vid| active_verts.contains(vid))
                });
                if !touches_active {
                    continue;
                }
                if let Some(nid) = insert_e2_face(
                    nc,
                    ns,
                    &base_verts,
                    &mut seen_centres,
                    &mut face_centres,
                    &mut vert_map,
                    &mut coords_2d,
                    &mut faces,
                ) {
                    visited_f.insert(nid);
                    queue.push(nid);
                } else if let Some(&nid) = seen_centres.get(&qkey2(nc))
                    && visited_f.insert(nid)
                {
                    queue.push(nid);
                }
            }
        }
        let next_end = faces.len();
        shell_bounds.push((end, next_end));
        for f in &faces[end..next_end] {
            active_verts.extend(f.iter().copied());
        }
    }

    let mut edge_set = BTreeSet::new();
    for f in &faces {
        for k in 0..f.len() {
            let (u, v) = (f[k], f[(k + 1) % f.len()]);
            edge_set.insert((u.min(v), u.max(v)));
        }
    }
    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let edge_arcs_2d: Vec<Vec<[f64; 2]>> =
        edges.iter().map(|&(u, v)| vec![coords_2d[u], coords_2d[v]]).collect();
    let coords_3d: Vec<[f64; 3]> = coords_2d.iter().map(|&[x, y]| [x, y, 0.0]).collect();
    let edge_arcs_3d: Vec<Vec<[f64; 3]>> =
        edges.iter().map(|&(u, v)| vec![coords_3d[u], coords_3d[v]]).collect();
    let (n_v, n_e, n_f) = (coords_2d.len(), edges.len(), faces.len());
    Ok(UnifiedPqTiling {
        p,
        q,
        layers: lay,
        delta: 0,
        geometry_kind: GeometryKind::Euclidean,
        curvature_kappa: kappa,
        euler_chi: n_v as i64 - n_e as i64 + n_f as i64,
        n_vertices: n_v,
        n_edges: n_e,
        n_faces: n_f,
        coords_3d,
        coords_2d,
        edges,
        edge_arcs_2d,
        edge_arcs_3d,
        faces,
        provenance: "[PROVED]".to_string(),
    })
}
