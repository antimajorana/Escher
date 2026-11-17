//! Hyperbolic `H^2` `(p, q)` tilings, rectified `(p.q.p.q)` kagome, dual `{q, p}`,
//! Poincare disk geodesics, Klein Beltrami chords, upper-sheet 3D hyperboloid bowl,
//! wedge disclinations, and `PSU(1,1)` loop holonomy.

use std::collections::{BTreeSet, VecDeque};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::{DSymbol, GeometryKind};

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

fn regular_hyp_radius(p: usize, q: usize) -> f64 {
    let num = (PI / (p as f64) + PI / (q as f64)).cos();
    let den = (PI / (p as f64) - PI / (q as f64)).cos();
    (num / den).max(0.0).sqrt().min(0.96)
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

/// Generate a finite patch of hyperbolic `{p, q}` / rectified `(p.q.p.q)` / dual `{q, p}` tiling.
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
    let r0 = regular_hyp_radius(p, q);
    let ds = DSymbol::new(vec![0], vec![0], vec![0], vec![p as u32], vec![q as u32])
        .map_err(|e| format!("D-symbol error: {e:?}"))?;

    let p_core = (p as i32 + wedge_delta).clamp(3, 12) as usize;
    let r_core = if wedge_delta == 0 {
        r0
    } else {
        // Adjust central core radius so its edges maintain nearly equal hyperbolic length.
        let half_edge_hyp = ((PI / (q as f64)).cos() / (PI / (p as f64)).sin()).max(1.0).acosh();
        let rc_hyp = (half_edge_hyp.sinh() / (PI / (p_core as f64)).sin()).asinh();
        (0.5 * rc_hyp).tanh().min(0.96)
    };

    let mut poly_verts: Vec<Vec<[f64; 2]>> = Vec::new();
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
        if depth >= lay || poly_verts.len() >= 180 {
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
            if c_norm2(cen) > 0.985 * 0.985 {
                continue;
            }
            if seen_centers.iter().any(|&c| (c[0] - cen[0]).hypot(c[1] - cen[1]) < 1e-3) {
                continue;
            }
            seen_centers.push(cen);
            let nid = poly_verts.len();
            poly_verts.push(next_poly);
            bfs.push_back((nid, depth + 1));
            if poly_verts.len() >= 180 {
                break;
            }
        }
    }

    let mut poincare_pts = Vec::new();
    let mut faces = Vec::new();
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();

    if style == HyperbolicStyle::RectifiedKagome {
        for poly in &poly_verts {
            let m = poly.len();
            let mut face_v = Vec::with_capacity(m);
            for k in 0..m {
                let mid = geodesic_midpoint(poly[k], poly[(k + 1) % m]);
                face_v.push(insert_pt(&mut poincare_pts, mid));
            }
            for k in 0..m {
                let u = face_v[k];
                let v = face_v[(k + 1) % m];
                if u != v {
                    edge_set.insert((u.min(v), u.max(v)));
                }
            }
            faces.push(face_v);
        }
    } else {
        for poly in &poly_verts {
            let m = poly.len();
            let face_v: Vec<usize> = poly.iter().map(|&z| insert_pt(&mut poincare_pts, z)).collect();
            for k in 0..m {
                let u = face_v[k];
                let v = face_v[(k + 1) % m];
                if u != v {
                    edge_set.insert((u.min(v), u.max(v)));
                }
            }
            faces.push(face_v);
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
