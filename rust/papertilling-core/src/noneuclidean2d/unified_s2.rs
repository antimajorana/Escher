//! Spherical `S^2` realiser for Platonic solids, hosohedra `{2, q}`, and dihedra `{p, 2}`.
//!
//! Counts `(V, E, F)`: `{3,3}` (4, 6, 4), `{3,4}` (6, 12, 8), `{4,3}` (8, 12, 6), `{3,5}` (12, 30, 20),
//! `{5,3}` (20, 30, 12) (Coxeter, *Regular Polytopes*, 3rd ed. 1973, Table I(i)); `{2,q}` has
//! (2, q, q) and `{p,2}` has (p, p, 2) (ibid. section 1.7). Coordinates: tetrahedron as alternate cube
//! vertices, icosahedron from the golden-ratio rectangles `(0, +-1, +-phi)` (ibid. section 3.7).

use std::collections::BTreeSet;
use std::f64::consts::PI;

use crate::dsymbol::GeometryKind;

use super::unified_pq::{PqGrowth, UnifiedPqTiling};

pub(crate) fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-15);
    [v[0] / l, v[1] / l, v[2] / l]
}

pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn slerp_arc(a: [f64; 3], b: [f64; 3], segs: usize) -> Vec<[f64; 3]> {
    let d = dot3(a, b).clamp(-1.0, 1.0);
    let omega = d.acos();
    if omega < 1e-9 || (PI - omega).abs() < 1e-9 {
        return vec![a, b];
    }
    let sin_o = omega.sin();
    (0..=segs)
        .map(|k| {
            let t = (k as f64) / (segs as f64);
            let w0 = ((1.0 - t) * omega).sin() / sin_o;
            let w1 = (t * omega).sin() / sin_o;
            norm3([w0 * a[0] + w1 * b[0], w0 * a[1] + w1 * b[1], w0 * a[2] + w1 * b[2]])
        })
        .collect()
}

fn stereo_north(p: [f64; 3]) -> [f64; 2] {
    let d = (1.0 - p[2]).max(1e-9);
    [p[0] / d, p[1] / d]
}

fn stereo_pole(p: [f64; 3], pole: [f64; 3], e1: [f64; 3], e2: [f64; 3]) -> [f64; 2] {
    let d = (1.0 - dot3(p, pole)).max(1e-9);
    [dot3(p, e1) / d, dot3(p, e2) / d]
}

pub(crate) fn build_spherical_pq(
    p: usize,
    q: usize,
    layers: usize,
    delta: i64,
    kappa: f64,
) -> Result<UnifiedPqTiling, String> {
    if p == 2 {
        return build_hosohedron(q, layers, delta, kappa);
    }
    if q == 2 {
        return build_dihedron(p, layers, delta, kappa);
    }
    let (coords_3d, faces) = platonic_solid(p, q)?;
    let mut edge_set = BTreeSet::new();
    for f in &faces {
        for k in 0..f.len() {
            let (u, v) = (f[k], f[(k + 1) % f.len()]);
            edge_set.insert((u.min(v), u.max(v)));
        }
    }
    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let edge_arcs_3d: Vec<Vec<[f64; 3]>> =
        edges.iter().map(|&(u, v)| slerp_arc(coords_3d[u], coords_3d[v], 12)).collect();
    let edge_arcs_2d: Vec<Vec<[f64; 2]>> =
        edge_arcs_3d.iter().map(|arc| arc.iter().map(|&pt| stereo_north(pt)).collect()).collect();
    let coords_2d: Vec<[f64; 2]> = coords_3d.iter().map(|&pt| stereo_north(pt)).collect();
    let (n_v, n_e, n_f) = (coords_3d.len(), edges.len(), faces.len());
    Ok(UnifiedPqTiling {
        p,
        q,
        layers,
        delta,
        geometry_kind: GeometryKind::Spherical,
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
        growth: PqGrowth::default(),
    })
}

fn build_hosohedron(q: usize, layers: usize, delta: i64, kappa: f64) -> Result<UnifiedPqTiling, String> {
    let coords_3d = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let pole_ang = PI / (q as f64);
    let pole = [pole_ang.cos(), pole_ang.sin(), 0.0];
    let e1 = [-pole_ang.sin(), pole_ang.cos(), 0.0];
    let e2 = [0.0, 0.0, 1.0];
    let coords_2d: Vec<[f64; 2]> = coords_3d.iter().map(|&pt| stereo_pole(pt, pole, e1, e2)).collect();
    let segs = 16;
    let mut edges = Vec::with_capacity(q);
    let mut edge_arcs_3d = Vec::with_capacity(q);
    let mut edge_arcs_2d = Vec::with_capacity(q);
    let mut faces = Vec::with_capacity(q);
    for k in 0..q {
        edges.push((0, 1));
        faces.push(vec![0, 1]);
        let phi = 2.0 * PI * (k as f64) / (q as f64);
        let arc3: Vec<[f64; 3]> = (0..=segs)
            .map(|s| {
                let t = PI * (s as f64) / (segs as f64);
                [t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos()]
            })
            .collect();
        edge_arcs_2d.push(arc3.iter().map(|&pt| stereo_pole(pt, pole, e1, e2)).collect());
        edge_arcs_3d.push(arc3);
    }
    Ok(UnifiedPqTiling {
        p: 2,
        q,
        layers,
        delta,
        geometry_kind: GeometryKind::Spherical,
        curvature_kappa: kappa,
        euler_chi: 2,
        n_vertices: 2,
        n_edges: q,
        n_faces: q,
        coords_3d,
        coords_2d,
        edges,
        edge_arcs_2d,
        edge_arcs_3d,
        faces,
        provenance: "[PROVED]".to_string(),
        growth: PqGrowth::default(),
    })
}

fn build_dihedron(p: usize, layers: usize, delta: i64, kappa: f64) -> Result<UnifiedPqTiling, String> {
    let coords_3d: Vec<[f64; 3]> = (0..p)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p as f64);
            [th.cos(), th.sin(), 0.0]
        })
        .collect();
    let coords_2d: Vec<[f64; 2]> = coords_3d.iter().map(|&pt| stereo_north(pt)).collect();
    let edges: Vec<(usize, usize)> = (0..p).map(|k| (k, (k + 1) % p)).collect();
    let segs = 12;
    let edge_arcs_3d: Vec<Vec<[f64; 3]>> = (0..p)
        .map(|k| {
            let th0 = 2.0 * PI * (k as f64) / (p as f64);
            let dth = 2.0 * PI / (p as f64);
            (0..=segs)
                .map(|s| {
                    let th = th0 + dth * (s as f64) / (segs as f64);
                    [th.cos(), th.sin(), 0.0]
                })
                .collect()
        })
        .collect();
    let edge_arcs_2d = edge_arcs_3d
        .iter()
        .map(|arc| arc.iter().map(|&pt| stereo_north(pt)).collect())
        .collect();
    let north: Vec<usize> = (0..p).collect();
    let mut south: Vec<usize> = (0..p).collect();
    south.reverse();
    let faces = vec![north, south];
    Ok(UnifiedPqTiling {
        p,
        q: 2,
        layers,
        delta,
        geometry_kind: GeometryKind::Spherical,
        curvature_kappa: kappa,
        euler_chi: 2,
        n_vertices: p,
        n_edges: p,
        n_faces: 2,
        coords_3d,
        coords_2d,
        edges,
        edge_arcs_2d,
        edge_arcs_3d,
        faces,
        provenance: "[PROVED]".to_string(),
        growth: PqGrowth::default(),
    })
}

type Solid = (Vec<[f64; 3]>, Vec<Vec<usize>>);

fn platonic_solid(p: usize, q: usize) -> Result<Solid, String> {
    match (p, q) {
        (3, 3) => {
            let pts: Vec<[f64; 3]> =
                [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
                    .into_iter()
                    .map(norm3)
                    .collect();
            let faces = faces_by_min_edge(&pts);
            Ok((pts, faces))
        }
        (4, 3) => {
            let mut pts = Vec::with_capacity(8);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        pts.push(norm3([sx, sy, sz]));
                    }
                }
            }
            let mut faces = Vec::with_capacity(6);
            for ax in 0..3 {
                for sgn in [-1.0_f64, 1.0] {
                    let cyc: Vec<usize> = (0..8).filter(|&i| (pts[i][ax] * sgn) > 0.0).collect();
                    faces.push(order_ccw(&pts, cyc));
                }
            }
            Ok((pts, faces))
        }
        (3, 4) => {
            let raw = [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ];
            let ez = norm3([1.0, 1.0, 1.0]);
            let ex = norm3([1.0, -1.0, 0.0]);
            let ey = cross3(ez, ex);
            let pts: Vec<[f64; 3]> =
                raw.into_iter().map(|v| [dot3(v, ex), dot3(v, ey), dot3(v, ez)]).collect();
            let faces = faces_by_min_edge(&pts);
            Ok((pts, faces))
        }
        (3, 5) | (5, 3) => {
            let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
            let mut ico = Vec::with_capacity(12);
            for a in [-1.0, 1.0] {
                for b in [-phi, phi] {
                    ico.extend([norm3([0.0, a, b]), norm3([a, b, 0.0]), norm3([b, 0.0, a])]);
                }
            }
            let ico_faces = faces_by_min_edge(&ico);
            if (p, q) == (3, 5) {
                return Ok((ico, ico_faces));
            }
            let dodeca_pts: Vec<[f64; 3]> = ico_faces
                .iter()
                .map(|f| {
                    norm3(f.iter().fold([0.0; 3], |acc, &vi| {
                        [acc[0] + ico[vi][0], acc[1] + ico[vi][1], acc[2] + ico[vi][2]]
                    }))
                })
                .collect();
            let dodeca_faces: Vec<Vec<usize>> = (0..ico.len())
                .map(|vi| {
                    let cyc: Vec<usize> = ico_faces
                        .iter()
                        .enumerate()
                        .filter_map(|(fi, f)| f.contains(&vi).then_some(fi))
                        .collect();
                    order_ccw(&dodeca_pts, cyc)
                })
                .collect();
            Ok((dodeca_pts, dodeca_faces))
        }
        _ => Err(format!("not a spherical regular polyhedron: {{{p}, {q}}}")),
    }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn faces_by_min_edge(pts: &[[f64; 3]]) -> Vec<Vec<usize>> {
    let min_e = (0..pts.len())
        .flat_map(|i| ((i + 1)..pts.len()).map(move |j| dist3(pts[i], pts[j])))
        .fold(f64::INFINITY, f64::min);
    let mut faces = Vec::new();
    for a in 0..pts.len() {
        for b in (a + 1)..pts.len() {
            if (dist3(pts[a], pts[b]) - min_e).abs() > 1e-5 {
                continue;
            }
            for c in (b + 1)..pts.len() {
                if (dist3(pts[a], pts[c]) - min_e).abs() < 1e-5
                    && (dist3(pts[b], pts[c]) - min_e).abs() < 1e-5
                {
                    faces.push(order_ccw(pts, vec![a, b, c]));
                }
            }
        }
    }
    faces
}

fn order_ccw(pts: &[[f64; 3]], mut cyc: Vec<usize>) -> Vec<usize> {
    let n = cyc.len() as f64;
    let c = norm3(cyc.iter().fold([0.0; 3], |acc, &i| {
        [acc[0] + pts[i][0] / n, acc[1] + pts[i][1] / n, acc[2] + pts[i][2] / n]
    }));
    let p0 = pts[cyc[0]];
    let d0 = [p0[0] - dot3(p0, c) * c[0], p0[1] - dot3(p0, c) * c[1], p0[2] - dot3(p0, c) * c[2]];
    let u = norm3(d0);
    let w = cross3(c, u);
    cyc.sort_by(|&i, &j| {
        let ang = |idx: usize| dot3(pts[idx], w).atan2(dot3(pts[idx], u));
        ang(i).total_cmp(&ang(j))
    });
    cyc
}
