//! Hyperbolic `H^2` `(p, q)` tilings, rectified `(p.q.p.q)` kagome, dual `{q, p}`,
//! Poincare disk geodesics, Klein Beltrami chords, upper-sheet 3D hyperboloid bowl,
//! wedge disclinations, and `PSU(1,1)` loop holonomy.

use std::collections::{BTreeSet, VecDeque};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::{DSymbol, GeometryKind};

pub mod inflation;
#[cfg(test)]
mod inflation_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperbolicStyle {
    RegularPQ,
    RectifiedKagome,
    DualQP,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperbolicTiling {
    pub p: usize,
    pub q: usize,
    pub style: HyperbolicStyle,
    pub layers: usize,
    pub wedge_delta: i32,
    pub symbol_label: String,
    pub curvature_kappa: f64,
    pub geometry_kind: GeometryKind,
    pub psu11_half_trace: f64,
    pub poincare_pts: Vec<[f64; 2]>,
    pub klein_pts: Vec<[f64; 2]>,
    pub hyperboloid_pts: Vec<[f64; 3]>,
    pub edges: Vec<(usize, usize)>,
    pub edge_arcs: Vec<Vec<[f64; 2]>>,
    pub faces: Vec<Vec<usize>>,
    /// Shell index of each face: `n` for faces in `S_n \ S_{n-1}` of vertex completion (0 = seed);
    /// BFS depth on the wedge-core path.
    #[serde(default)]
    pub face_shell: Vec<usize>,
    /// Largest `n` with `S_n` fully drawn (0 on the wedge-core path, which is not vertex completion).
    #[serde(default)]
    pub complete_through_shell: usize,
    /// Which growth rule built the patch and which shells are complete or truncated.
    #[serde(default)]
    pub growth_rule: String,
}

#[inline]
fn c_add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] + b[0], a[1] + b[1]] }
#[inline]
fn c_sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
#[inline]
fn c_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]] }
#[inline]
fn c_conj(a: [f64; 2]) -> [f64; 2] { [a[0], -a[1]] }
#[inline]
fn c_div(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let d = b[0] * b[0] + b[1] * b[1];
    [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}
#[inline]
fn c_norm2(a: [f64; 2]) -> f64 { a[0] * a[0] + a[1] * a[1] }

/// Orthogonal circle center & R^2 for Poincare geodesic through A, B (or None if diameter).
fn orthogonal_circle(a: [f64; 2], b: [f64; 2]) -> Option<([f64; 2], f64)> {
    let det = a[0] * b[1] - a[1] * b[0];
    if det.abs() < 1e-10 {
        return None;
    }
    let rhs0 = 0.5 * (1.0 + c_norm2(a));
    let rhs1 = 0.5 * (1.0 + c_norm2(b));
    let cx = (rhs0 * b[1] - a[1] * rhs1) / det;
    let cy = (a[0] * rhs1 - rhs0 * b[0]) / det;
    let r2 = cx * cx + cy * cy - 1.0;
    (r2 > 0.0).then_some(([cx, cy], r2))
}

fn reflect_geodesic(z: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    if let Some((c, r2)) = orthogonal_circle(a, b) {
        let dz = c_conj(c_sub(z, c));
        let denom = c_norm2(dz).max(1e-18);
        [c[0] + r2 * dz[0] / denom, c[1] - r2 * dz[1] / denom]
    } else {
        let len = c_norm2(a).sqrt().max(1e-18);
        let u = [a[0] / len, a[1] / len];
        let u2 = c_mul(u, u);
        c_mul(u2, c_conj(z))
    }
}

fn geodesic_midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    // Mobius shift A -> 0: T(z) = (z - A) / (1 - conj(A)*z)
    let denom = c_sub([1.0, 0.0], c_mul(c_conj(a), b));
    let w = c_div(c_sub(b, a), denom);
    let r = c_norm2(w).sqrt();
    if r < 1e-12 {
        return a;
    }
    let half_r = r / (1.0 + (1.0 - r * r).max(0.0).sqrt());
    let mw = [w[0] * half_r / r, w[1] * half_r / r];
    let inv_d = c_add([1.0, 0.0], c_mul(c_conj(a), mw));
    c_div(c_add(mw, a), inv_d)
}

fn sample_geodesic_arc(a: [f64; 2], b: [f64; 2], segs: usize) -> Vec<[f64; 2]> {
    let Some((c, r2)) = orthogonal_circle(a, b) else {
        return vec![a, b];
    };
    let rad = r2.sqrt();
    let th0 = (a[1] - c[1]).atan2(a[0] - c[0]);
    let th1 = (b[1] - c[1]).atan2(b[0] - c[0]);
    let mut dth = (th1 - th0 + PI).rem_euclid(2.0 * PI) - PI;
    if dth.abs() < 1e-6 {
        dth = 0.0;
    }
    (0..=segs)
        .map(|k| {
            let th = th0 + dth * (k as f64) / (segs as f64);
            [c[0] + rad * th.cos(), c[1] + rad * th.sin()]
        })
        .collect()
}

pub fn to_klein(z: [f64; 2]) -> [f64; 2] {
    let d = 1.0 + c_norm2(z);
    [2.0 * z[0] / d, 2.0 * z[1] / d]
}

pub fn to_hyperboloid(z: [f64; 2]) -> [f64; 3] {
    let n2 = c_norm2(z).min(0.95 * 0.95);
    let scale = (n2 / c_norm2(z).max(1e-18)).sqrt().min(1.0);
    let x = z[0] * scale;
    let y = z[1] * scale;
    let d = (1.0 - (x * x + y * y)).max(0.05);
    [2.0 * x / d, 2.0 * y / d, (1.0 + x * x + y * y) / d]
}

fn insert_pt(pts: &mut Vec<[f64; 2]>, z: [f64; 2]) -> usize {
    if let Some((idx, _)) = pts
        .iter()
        .enumerate()
        .find(|&(_, &p)| (p[0] - z[0]).hypot(p[1] - z[1]) < 2e-4)
    {
        idx
    } else {
        let idx = pts.len();
        pts.push(z);
        idx
    }
}

/// Face cap and Poincare-radius cap (of face centres) for the drawn patch.
const MAX_DRAWN_FACES: usize = 180;
const MAX_CENTRE_RADIUS: f64 = 0.985;
/// Compute budget for growing the drawn patch before the face cap trims it to whole shells.
const DRAW_COMPUTE_BUDGET: usize = 2_000;

/// Drawn patch before edges and projections: Poincare points, faces, per-face shell.
struct Patch {
    pts: Vec<[f64; 2]>,
    faces: Vec<Vec<usize>>,
    face_shell: Vec<usize>,
    complete_through_shell: usize,
    growth_rule: String,
}

/// Regular core (`wedge_delta == 0`): BDF vertex completion `S_0 .. S_layers` from one `p`-gon.
/// Whole shells are kept while the total stays within [`MAX_DRAWN_FACES`]; faces whose centre lies
/// beyond Poincare radius [`MAX_CENTRE_RADIUS`] are dropped (the seed never is), which truncates
/// that shell. `growth_rule` states which shells are complete.
fn vertex_completion_patch(p: usize, q: usize, style: HyperbolicStyle, lay: usize) -> Result<Patch, String> {
    let vc = inflation::grow_vertex_completion(p, q, lay, DRAW_COMPUTE_BUDGET)?;
    let mut keep = 0;
    let mut total = 0;
    for &a in &vc.shell_sizes {
        if total + a > MAX_DRAWN_FACES {
            break;
        }
        total += a;
        keep += 1;
    }
    let mut notes = Vec::new();
    if let Some(&a) = vc.shell_sizes.get(keep) {
        notes.push(format!("shell {keep} ({a} faces) omitted by the {MAX_DRAWN_FACES}-face cap"));
    }
    if let Some(s) = &vc.stopped {
        notes.push(s.clone());
    }
    let kagome = style == HyperbolicStyle::RectifiedKagome;
    let base_pts = if kagome { &vc.base.edge_midpoints } else { &vc.base.vertices };
    let mut set = inflation::HypPointSet::new(inflation::hyp_dist(base_pts[0], base_pts[1]));
    let mut complete = keep - 1;
    let (mut faces, mut face_shell) = (Vec::new(), Vec::new());
    for (i, &s) in vc.shell.iter().enumerate().take_while(|&(_, &s)| s < keep) {
        if i > 0 && c_norm2(inflation::hyperboloid_to_poincare(vc.centre(i))) > MAX_CENTRE_RADIUS.powi(2) {
            complete = complete.min(s - 1);
            continue;
        }
        let pts3 = if kagome { vc.face_edge_midpoints(i) } else { vc.face_vertices(i) };
        faces.push(pts3.into_iter().map(|x| set.insert(x).0).collect());
        face_shell.push(s);
    }
    // The band guard sees `hyp_dist` only, not drift off the hyperboloid; for the drawn range
    // (centres within MAX_CENTRE_RADIUS, p, q <= 12) that drift was measured below 1e-4 in
    // |<x, x> + 1| (2026-11-18), against up to ~2e2 for far vertices of the counting path.
    if set.suspect > 0 {
        return Err(format!(
            "{{{p},{q}}}: f64 precision: {} drawn points have an ambiguous dedup distance",
            set.suspect
        ));
    }
    if complete + 1 < keep {
        notes.push(format!("shell {} truncated by the centre radius cap {MAX_CENTRE_RADIUS}", complete + 1));
    }
    let growth_rule = format!(
        "BDF vertex completion from one {p}-gon: S_0..S_{complete} complete, {} faces drawn (requested S_{lay}){}",
        faces.len(),
        notes.iter().map(|n| format!("; {n}")).collect::<String>()
    );
    let pts = set.pts.iter().map(|&x| inflation::hyperboloid_to_poincare(x)).collect();
    Ok(Patch { pts, faces, face_shell, complete_through_shell: complete, growth_rule })
}

/// Wedge-disclination core (`wedge_delta != 0`): the original edge-BFS construction, not upgraded to
/// vertex completion (corner-only tiles are missed for `q >= 4`; dedup is Euclidean in the disk).
fn wedge_patch(p: usize, q: usize, style: HyperbolicStyle, lay: usize, p_core: usize) -> Patch {
    // Adjust central core radius so its edges maintain nearly equal hyperbolic length.
    let half_edge_hyp = ((PI / (q as f64)).cos() / (PI / (p as f64)).sin()).max(1.0).acosh();
    let rc_hyp = (half_edge_hyp.sinh() / (PI / (p_core as f64)).sin()).asinh();
    let r_core = (0.5 * rc_hyp).tanh().min(0.96);

    let mut poly_verts: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut depths = vec![0_usize];
    let mut seen_centers: Vec<[f64; 2]> = vec![[0.0, 0.0]];
    let central: Vec<[f64; 2]> = (0..p_core)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p_core as f64);
            [r_core * th.cos(), r_core * th.sin()]
        })
        .collect();
    poly_verts.push(central);

    let mut bfs = VecDeque::from([(0_usize, 0_usize)]);
    while let Some((pid, depth)) = bfs.pop_front() {
        if depth >= lay || poly_verts.len() >= MAX_DRAWN_FACES {
            continue;
        }
        let cur_poly = poly_verts[pid].clone();
        let m = cur_poly.len();
        for e in 0..m {
            let a = cur_poly[e];
            let b = cur_poly[(e + 1) % m];
            let next_poly: Vec<[f64; 2]> = cur_poly.iter().map(|&z| reflect_geodesic(z, a, b)).collect();
            let cen = next_poly.iter().fold([0.0, 0.0], |acc, &v| [acc[0] + v[0], acc[1] + v[1]]);
            let cen = [cen[0] / (m as f64), cen[1] / (m as f64)];
            if c_norm2(cen) > MAX_CENTRE_RADIUS * MAX_CENTRE_RADIUS {
                continue;
            }
            if seen_centers.iter().any(|&c| (c[0] - cen[0]).hypot(c[1] - cen[1]) < 1e-3) {
                continue;
            }
            seen_centers.push(cen);
            let nid = poly_verts.len();
            poly_verts.push(next_poly);
            depths.push(depth + 1);
            bfs.push_back((nid, depth + 1));
            if poly_verts.len() >= MAX_DRAWN_FACES {
                break;
            }
        }
    }

    let mut pts = Vec::new();
    let faces: Vec<Vec<usize>> = poly_verts
        .iter()
        .map(|poly| {
            let m = poly.len();
            if style == HyperbolicStyle::RectifiedKagome {
                (0..m).map(|k| insert_pt(&mut pts, geodesic_midpoint(poly[k], poly[(k + 1) % m]))).collect()
            } else {
                poly.iter().map(|&z| insert_pt(&mut pts, z)).collect()
            }
        })
        .collect();
    let growth_rule = format!(
        "edge-BFS around a {p_core}-gon wedge core (not upgraded to vertex completion): {} faces, depth <= {lay}",
        faces.len()
    );
    Patch { pts, faces, face_shell: depths, complete_through_shell: 0, growth_rule }
}

/// Generate a finite patch of hyperbolic `{p, q}` / rectified `(p.q.p.q)` / dual `{q, p}` tiling.
///
/// For `wedge_delta == 0` the patch is BDF vertex completion `S_layers` of one `p`-gon (layers
/// clamped to 1..=4), trimmed to whole shells within 180 faces and to face centres within Poincare
/// radius 0.985 (see `growth_rule`). For `wedge_delta != 0` the original edge-BFS is used.
pub fn build_hyperbolic_tiling(
    p_in: usize,
    q_in: usize,
    style: HyperbolicStyle,
    layers: usize,
    wedge_delta: i32,
) -> Result<HyperbolicTiling, String> {
    let (p, q) = match style {
        HyperbolicStyle::DualQP => (q_in, p_in),
        _ => (p_in, q_in),
    };
    if p < 3 || q < 3 || (p - 2) * (q - 2) <= 4 {
        return Err(format!("{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)"));
    }
    let lay = layers.clamp(1, 4);
    let ds = DSymbol::new(vec![0], vec![0], vec![0], vec![p as u32], vec![q as u32])
        .map_err(|e| format!("D-symbol error: {e:?}"))?;
    let p_core = (p as i32 + wedge_delta).clamp(3, 12) as usize;
    let patch = if wedge_delta == 0 {
        vertex_completion_patch(p, q, style, lay)?
    } else {
        wedge_patch(p, q, style, lay, p_core)
    };
    let Patch { pts: poincare_pts, faces, face_shell, complete_through_shell, growth_rule } = patch;

    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for face_v in &faces {
        let m = face_v.len();
        for k in 0..m {
            let (u, v) = (face_v[k], face_v[(k + 1) % m]);
            if u != v {
                edge_set.insert((u.min(v), u.max(v)));
            }
        }
    }

    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let edge_arcs: Vec<Vec<[f64; 2]>> = edges
        .iter()
        .map(|&(u, v)| sample_geodesic_arc(poincare_pts[u], poincare_pts[v], 8))
        .collect();
    let klein_pts: Vec<[f64; 2]> = poincare_pts.iter().copied().map(to_klein).collect();
    let hyperboloid_pts: Vec<[f64; 3]> = poincare_pts.iter().copied().map(to_hyperboloid).collect();

    let symbol_label = match style {
        HyperbolicStyle::RegularPQ => format!("{{{p_in},{q_in}}}"),
        HyperbolicStyle::RectifiedKagome => format!("({p_in}.{q_in}.{p_in}.{q_in}) kagome"),
        HyperbolicStyle::DualQP => format!("{{{q_in},{p_in}}} dual"),
    };
    let psu11_half_trace = if wedge_delta == 0 {
        1.0
    } else {
        (PI * (wedge_delta as f64).abs() / (p_core as f64)).cos().abs()
    };

    Ok(HyperbolicTiling {
        p,
        q,
        style,
        layers: lay,
        wedge_delta,
        symbol_label,
        curvature_kappa: ds.orbifold_curvature(),
        geometry_kind: ds.geometry_kind(),
        psu11_half_trace,
        poincare_pts,
        klein_pts,
        hyperboloid_pts,
        edges,
        edge_arcs,
        faces,
        face_shell,
        complete_through_shell,
        growth_rule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperbolic_tilings_and_dsymbol_curvatures() {
        let cube = DSymbol::new(vec![0], vec![0], vec![0], vec![4], vec![3]).unwrap();
        assert_eq!(cube.geometry_kind(), GeometryKind::Spherical);
        assert!(cube.orbifold_curvature() > 0.0);

        let sq = DSymbol::new(vec![0], vec![0], vec![0], vec![4], vec![4]).unwrap();
        assert_eq!(sq.geometry_kind(), GeometryKind::Euclidean);
        assert!(sq.is_flat());

        let t73 = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, 0).unwrap();
        assert_eq!(t73.geometry_kind, GeometryKind::Hyperbolic);
        assert!(t73.curvature_kappa < 0.0);
        assert!(t73.faces.len() >= 8);
        assert_eq!(t73.faces[0].len(), 7);
        assert!(t73.poincare_pts.iter().all(|z| z[0] * z[0] + z[1] * z[1] < 1.0));
        assert!(t73.hyperboloid_pts.iter().all(|p| p[2] >= 1.0 - 1e-9));

        let kag = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RectifiedKagome, 2, 0).unwrap();
        assert!(!kag.edges.is_empty());

        let cone6 = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, -1).unwrap();
        assert_eq!(cone6.faces[0].len(), 6);
    }
}
