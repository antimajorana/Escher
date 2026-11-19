//! Hyperbolic models (Poincare, Klein, UHP, hyperboloid), exact isometries, geodesics,
//! 3D `H^2` surfaces (Bowl, Hemisphere, Pseudosphere, Dini, Kuen), and papercraft net unfolder.
//!
//! Model maps (Beardon, *The Geometry of Discrete Groups*, 1983, ch. 7; Cannon, Floyd, Kenyon
//! & Parry, "Hyperbolic geometry", MSRI Publ. 31, 1997): Cayley `w = i (1 + z) / (1 - z)`, Klein
//! `k = 2 z / (1 + |z|^2)`, hyperboloid `(2z, 1 + |z|^2) / (1 - |z|^2)`, distance
//! `sinh^2(d/2) = |z1 - z2|^2 / ((1 - |z1|^2)(1 - |z2|^2))`. Surfaces of constant Gaussian curvature
//! `K = -1` in Euclidean `R^3`: pseudosphere (tractrix of revolution), Dini's surface with
//! `a^2 + b^2 = 1`, Kuen's surface (Gray, Abbena & Salamon, *Modern Differential Geometry of Curves and
//! Surfaces*, 3rd ed. 2006, chapter on surfaces of constant curvature); checked numerically in tests.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::hyperbolic::inflation::hyperboloid_to_poincare;
use crate::hyperbolic::to_klein;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum H2SurfaceKind {
    HyperboloidBowl,
    Hemisphere,
    Pseudosphere,
    Dini { twist: f64 },
    Kuen,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct H2SurfaceMesh {
    pub kind_name: String,
    /// Gaussian curvature of the metric the surface carries: Euclidean induced metric for
    /// Pseudosphere / Dini / Kuen (`-1`) and Hemisphere (`+1`; the hemisphere *model* of `H^2` is not
    /// isometric); Minkowski induced metric for the HyperboloidBowl (`-1`; its Euclidean curvature is
    /// positive and not constant).
    pub expected_gaussian_curvature: f64,
    pub positions_3d: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PapercraftNet {
    pub pieces: Vec<Vec<[f64; 2]>>,
    pub tree_edges: Vec<(usize, usize, usize, usize)>,
    pub cut_pairings: Vec<(usize, usize, usize, usize)>,
    pub max_edge_length_error: f64,
}

fn check_poincare(z: [f64; 2]) -> Result<f64, String> {
    let r2 = z[0] * z[0] + z[1] * z[1];
    if !z[0].is_finite() || !z[1].is_finite() || r2 >= 1.0 {
        return Err(format!("point {z:?} is outside the open Poincare disk |z| < 1"));
    }
    Ok(r2)
}

fn check_uhp(w: [f64; 2]) -> Result<(), String> {
    if !w[0].is_finite() || !w[1].is_finite() || w[1] <= 0.0 {
        return Err(format!("point {w:?} is outside the Upper Half-Plane Im(w) > 0"));
    }
    Ok(())
}

/// Exact Cayley map from Poincare disk `z` (`|z| < 1`) to Upper Half-Plane `w = i (1 + z) / (1 - z)`.
pub fn poincare_to_uhp(z: [f64; 2]) -> Result<[f64; 2], String> {
    let r2 = check_poincare(z)?;
    let d = (1.0 - z[0]).powi(2) + z[1].powi(2);
    Ok([(-2.0 * z[1]) / d, (1.0 - r2) / d])
}

/// Exact inverse Cayley map from Upper Half-Plane `w` (`Im(w) > 0`) to Poincare disk `z = (w - i) / (w + i)`.
pub fn uhp_to_poincare(w: [f64; 2]) -> Result<[f64; 2], String> {
    check_uhp(w)?;
    let d = w[0] * w[0] + (w[1] + 1.0).powi(2);
    Ok([(w[0] * w[0] + w[1] * w[1] - 1.0) / d, (-2.0 * w[0]) / d])
}

pub fn poincare_to_klein(z: [f64; 2]) -> [f64; 2] {
    to_klein(z)
}

pub fn klein_to_poincare(k: [f64; 2]) -> Result<[f64; 2], String> {
    let r2 = k[0] * k[0] + k[1] * k[1];
    if !k[0].is_finite() || !k[1].is_finite() || r2 >= 1.0 {
        return Err(format!("point {k:?} is outside the open Klein disk |k| < 1"));
    }
    let s = 1.0 + (1.0 - r2).sqrt();
    Ok([k[0] / s, k[1] / s])
}

pub fn poincare_to_hyperboloid_exact(z: [f64; 2]) -> Result<[f64; 3], String> {
    let r2 = check_poincare(z)?;
    let d = 1.0 - r2;
    Ok([2.0 * z[0] / d, 2.0 * z[1] / d, (1.0 + r2) / d])
}

pub fn hyperboloid_to_poincare_exact(x: [f64; 3]) -> [f64; 2] {
    hyperboloid_to_poincare(x)
}

/// Exact hyperbolic distance in the Poincare disk: `2 artanh |(z1 - z2) / (1 - conj(z1) z2)|`.
pub fn hyp_distance_poincare(z1: [f64; 2], z2: [f64; 2]) -> Result<f64, String> {
    let r1 = check_poincare(z1)?;
    let r2 = check_poincare(z2)?;
    let dx = z1[0] - z2[0];
    let dy = z1[1] - z2[1];
    let arg = (dx * dx + dy * dy) / ((1.0 - r1) * (1.0 - r2));
    Ok(2.0 * arg.sqrt().asinh())
}

/// Exact hyperbolic distance in UHP: `acosh(1 + |w1 - w2|^2 / (2 Im(w1) Im(w2)))`.
pub fn hyp_distance_uhp(w1: [f64; 2], w2: [f64; 2]) -> Result<f64, String> {
    check_uhp(w1)?;
    check_uhp(w2)?;
    let dx = w1[0] - w2[0];
    let dy = w1[1] - w2[1];
    let arg = (dx * dx + dy * dy) / (4.0 * w1[1] * w2[1]);
    Ok(2.0 * arg.sqrt().asinh())
}

fn c_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn c_div(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let d = b[0] * b[0] + b[1] * b[1];
    [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}

/// Constant-hyperbolic-speed Poincare geodesic arc from `z1` to `z2` (`steps >= 1`).
pub fn geodesic_arc_poincare(z1: [f64; 2], z2: [f64; 2], steps: usize) -> Result<Vec<[f64; 2]>, String> {
    check_poincare(z1)?;
    check_poincare(z2)?;
    let n = steps.max(1);
    // Mobius shift mapping z1 -> 0: T(z) = (z - z1) / (1 - conj(z1) z)
    let conj_z1 = [z1[0], -z1[1]];
    let num = [z2[0] - z1[0], z2[1] - z1[1]];
    let prod = c_mul(conj_z1, z2);
    let w2 = c_div(num, [1.0 - prod[0], -prod[1]]);
    let r2 = w2[0].hypot(w2[1]);
    if r2 < 1e-14 {
        return Ok(vec![z1; n + 1]);
    }
    let dir = [w2[0] / r2, w2[1] / r2];
    let d_tot = 2.0 * r2.atanh();
    Ok((0..=n)
        .map(|k| {
            if k == 0 {
                return z1;
            }
            if k == n {
                return z2;
            }
            let r = (0.5 * d_tot * (k as f64) / (n as f64)).tanh();
            let wk = [r * dir[0], r * dir[1]];
            // Inverse Mobius: T^{-1}(w) = (w + z1) / (1 + conj(z1) w)
            let p = c_mul(conj_z1, wk);
            c_div([wk[0] + z1[0], wk[1] + z1[1]], [1.0 + p[0], p[1]])
        })
        .collect())
}

/// Constant-hyperbolic-speed UHP geodesic arc from `w1` to `w2`.
pub fn geodesic_arc_uhp(w1: [f64; 2], w2: [f64; 2], steps: usize) -> Result<Vec<[f64; 2]>, String> {
    let z1 = uhp_to_poincare(w1)?;
    let z2 = uhp_to_poincare(w2)?;
    let arc_p = geodesic_arc_poincare(z1, z2, steps)?;
    arc_p.into_iter().map(poincare_to_uhp).collect()
}

/// Evaluate an exact point on a 3D `H^2` surface from intrinsic chart coordinates `(u, v)`.
pub fn surface_point(kind: H2SurfaceKind, u: f64, v: f64) -> [f64; 3] {
    match kind {
        H2SurfaceKind::HyperboloidBowl => {
            let s = v.max(0.0);
            [s.sinh() * u.cos(), s.sinh() * u.sin(), s.cosh()]
        }
        H2SurfaceKind::Hemisphere => {
            let r = v.clamp(0.0, 0.999);
            let z = (1.0 - r * r).sqrt();
            [r * u.cos(), r * u.sin(), z]
        }
        H2SurfaceKind::Pseudosphere => {
            let t = v.max(0.0);
            let r = 1.0 / t.cosh();
            [r * u.cos(), r * u.sin(), t - t.tanh()]
        }
        H2SurfaceKind::Dini { twist } => {
            let t = v.max(0.0);
            let norm = (1.0 + twist * twist).sqrt();
            let (a, b) = (1.0 / norm, twist / norm);
            let r = a / t.cosh();
            [r * u.cos(), r * u.sin(), a * (t - t.tanh()) + b * u]
        }
        H2SurfaceKind::Kuen => {
            let vv = v.clamp(0.05, PI - 0.05);
            let (su, cu) = (u.sin(), u.cos());
            let (sv, cv) = (vv.sin(), vv.cos());
            let w = 1.0 + u * u * sv * sv;
            [
                2.0 * (cu + u * su) * sv / w,
                2.0 * (su - u * cu) * sv / w,
                (0.5 * vv).tan().ln() + 2.0 * cv / w,
            ]
        }
    }
}

/// Place a Poincare-disk point on a 3D surface. Exact isometry only for `HyperboloidBowl`
/// (Minkowski metric); `Hemisphere` is the conformal hemisphere model; `Pseudosphere` / `Dini` use the
/// polar chart `(theta, rho)` and `Kuen` an affine chart, which are visual placements, not isometries.
pub fn map_poincare_to_surface(z: [f64; 2], kind: H2SurfaceKind) -> Result<[f64; 3], String> {
    let r2 = check_poincare(z)?;
    let r = r2.sqrt();
    let theta = z[1].atan2(z[0]);
    let rho = 2.0 * r.atanh();
    Ok(match kind {
        H2SurfaceKind::HyperboloidBowl => poincare_to_hyperboloid_exact(z)?,
        H2SurfaceKind::Hemisphere => {
            let d = 1.0 + r2;
            [2.0 * z[0] / d, 2.0 * z[1] / d, (1.0 - r2) / d]
        }
        H2SurfaceKind::Pseudosphere | H2SurfaceKind::Dini { .. } => surface_point(kind, theta, rho),
        H2SurfaceKind::Kuen => surface_point(kind, 3.2 * z[0], 0.5 * PI + 1.1 * z[1]),
    })
}

/// Build a structured 3D surface quad mesh for any of the 5 `H^2` surface embeddings.
pub fn build_h2_surface_mesh(kind: H2SurfaceKind, u_steps: usize, v_steps: usize) -> Result<H2SurfaceMesh, String> {
    let nu = u_steps.clamp(4, 128);
    let nv = v_steps.clamp(4, 128);
    let (u_range, v_range, kind_name, expected_k) = match kind {
        H2SurfaceKind::HyperboloidBowl => ((0.0, 2.0 * PI), (0.0, 2.2), "HyperboloidBowl", -1.0),
        H2SurfaceKind::Hemisphere => ((0.0, 2.0 * PI), (0.0, 0.98), "Hemisphere", 1.0),
        H2SurfaceKind::Pseudosphere => ((0.0, 2.0 * PI), (0.0, 2.4), "Pseudosphere", -1.0),
        H2SurfaceKind::Dini { .. } => ((0.0, 4.0 * PI), (0.0, 2.2), "Dini", -1.0),
        H2SurfaceKind::Kuen => ((-3.5, 3.5), (0.25, PI - 0.25), "Kuen", -1.0),
    };
    let mut positions_3d = Vec::with_capacity((nu + 1) * (nv + 1));
    for iv in 0..=nv {
        let v = v_range.0 + (v_range.1 - v_range.0) * (iv as f64) / (nv as f64);
        for iu in 0..=nu {
            let u = u_range.0 + (u_range.1 - u_range.0) * (iu as f64) / (nu as f64);
            positions_3d.push(surface_point(kind, u, v));
        }
    }
    let mut faces = Vec::with_capacity(nu * nv);
    for iv in 0..nv {
        for iu in 0..nu {
            let a = iv * (nu + 1) + iu;
            faces.push(vec![a, a + 1, a + nu + 2, a + nu + 1]);
        }
    }
    Ok(H2SurfaceMesh {
        kind_name: kind_name.to_string(),
        expected_gaussian_curvature: expected_k,
        positions_3d,
        faces,
    })
}

pub fn embed_tiling_on_surface(
    coords_poincare: &[[f64; 2]],
    faces: &[Vec<usize>],
    kind: H2SurfaceKind,
) -> Result<H2SurfaceMesh, String> {
    let positions_3d: Vec<[f64; 3]> = coords_poincare
        .iter()
        .map(|&z| map_poincare_to_surface(z, kind))
        .collect::<Result<Vec<_>, _>>()?;
    let (kind_name, expected_k) = match kind {
        H2SurfaceKind::HyperboloidBowl => ("HyperboloidBowl", -1.0),
        H2SurfaceKind::Hemisphere => ("Hemisphere", 1.0),
        H2SurfaceKind::Pseudosphere => ("Pseudosphere", -1.0),
        H2SurfaceKind::Dini { .. } => ("Dini", -1.0),
        H2SurfaceKind::Kuen => ("Kuen", -1.0),
    };
    Ok(H2SurfaceMesh {
        kind_name: kind_name.to_string(),
        expected_gaussian_curvature: expected_k,
        positions_3d,
        faces: faces.to_vec(),
    })
}

fn flatten_face_local(coords_3d: &[[f64; 3]], cyc: &[usize]) -> Vec<[f64; 2]> {
    let m = cyc.len();
    let d3 = |i: usize, j: usize| {
        let (a, b) = (coords_3d[i], coords_3d[j]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let mut pts = Vec::with_capacity(m);
    pts.push([0.0, 0.0]);
    pts.push([d3(cyc[0], cyc[1]), 0.0]);
    for k in 2..m {
        let (a, b, c) = (d3(cyc[0], cyc[k - 1]), d3(cyc[k - 1], cyc[k]), d3(cyc[0], cyc[k]));
        let (p_prev, p0) = (pts[k - 1], pts[0]);
        let ux = (p_prev[0] - p0[0]) / a.max(1e-12);
        let uy = (p_prev[1] - p0[1]) / a.max(1e-12);
        let cos_a = ((a * a + c * c - b * b) / (2.0 * a * c).max(1e-12)).clamp(-1.0, 1.0);
        let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
        // Place vertex k to the CCW left of segment (p0 -> p_prev)
        let dir_x = ux * cos_a - uy * sin_a;
        let dir_y = ux * sin_a + uy * cos_a;
        pts.push([p0[0] + c * dir_x, p0[1] + c * dir_y]);
    }
    pts
}

fn map_piece_across_edge(
    local: &[[f64; 2]],
    side: usize,
    target_u: [f64; 2],
    target_v: [f64; 2],
) -> Vec<[f64; 2]> {
    // Map local[side] -> target_v and local[(side+1)%m] -> target_u (opposite shared edge direction)
    let m = local.len();
    let (su, sv) = (local[side], local[(side + 1) % m]);
    let (dx_s, dy_s) = (sv[0] - su[0], sv[1] - su[1]);
    let (dx_t, dy_t) = (target_u[0] - target_v[0], target_u[1] - target_v[1]);
    let ang = dy_t.atan2(dx_t) - dy_s.atan2(dx_s);
    let (ca, sa) = (ang.cos(), ang.sin());
    local
        .iter()
        .map(|&p| {
            let rx = p[0] - su[0];
            let ry = p[1] - su[1];
            [target_v[0] + ca * rx - sa * ry, target_v[1] + sa * rx + ca * ry]
        })
        .collect()
}

/// Unfold a 3D polygonal surface mesh onto `R^2` along a dual BFS spanning tree.
/// Assumes consistently oriented, planar (or nearly planar) faces; each piece is laid out with exact
/// 3D edge lengths and `max_edge_length_error` reports the residual.
pub fn unfold_papercraft_net(coords_3d: &[[f64; 3]], faces: &[Vec<usize>]) -> Result<PapercraftNet, String> {
    if faces.is_empty() {
        return Err("cannot unfold empty face list".to_string());
    }
    let nf = faces.len();
    let mut edge_inc: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    for (fi, f) in faces.iter().enumerate() {
        if f.len() < 3 {
            return Err(format!("face {fi} has fewer than 3 vertices"));
        }
        for s in 0..f.len() {
            let (u, v) = (f[s], f[(s + 1) % f.len()]);
            edge_inc.entry((u.min(v), u.max(v))).or_default().push((fi, s));
        }
    }
    let local_pieces: Vec<Vec<[f64; 2]>> = faces.iter().map(|f| flatten_face_local(coords_3d, f)).collect();
    let mut placed: Vec<Option<Vec<[f64; 2]>>> = vec![None; nf];
    placed[0] = Some(local_pieces[0].clone());
    let mut q = VecDeque::from([0usize]);
    let mut tree_edges = Vec::new();
    let mut tree_pair_set: HashSet<(usize, usize)> = HashSet::new();

    while let Some(fa) = q.pop_front() {
        let pa = placed[fa].clone().unwrap();
        let f_cyc = &faces[fa];
        for sa in 0..f_cyc.len() {
            let (u, v) = (f_cyc[sa], f_cyc[(sa + 1) % f_cyc.len()]);
            let key = (u.min(v), u.max(v));
            for &(fb, sb) in edge_inc.get(&key).into_iter().flatten() {
                if fb != fa && placed[fb].is_none() {
                    let next_2d = map_piece_across_edge(&local_pieces[fb], sb, pa[sa], pa[(sa + 1) % f_cyc.len()]);
                    placed[fb] = Some(next_2d);
                    tree_edges.push((fa, sa, fb, sb));
                    tree_pair_set.insert((fa.min(fb), fa.max(fb)));
                    q.push_back(fb);
                }
            }
        }
    }

    if placed.iter().any(Option::is_none) {
        return Err("mesh is disconnected; spanning-tree net unfolder requires connected faces".to_string());
    }
    let pieces: Vec<Vec<[f64; 2]>> = placed.into_iter().map(Option::unwrap).collect();
    let mut cut_pairings = Vec::new();
    for inc in edge_inc.values() {
        if inc.len() == 2 {
            let ((fa, sa), (fb, sb)) = (inc[0], inc[1]);
            if !tree_pair_set.contains(&(fa.min(fb), fa.max(fb))) {
                cut_pairings.push((fa, sa, fb, sb));
            }
        }
    }
    let mut max_err = 0.0_f64;
    for (fi, f) in faces.iter().enumerate() {
        let p2 = &pieces[fi];
        for s in 0..f.len() {
            let (a, b) = (f[s], f[(s + 1) % f.len()]);
            let (pa3, pb3) = (coords_3d[a], coords_3d[b]);
            let l3 = ((pa3[0] - pb3[0]).powi(2) + (pa3[1] - pb3[1]).powi(2) + (pa3[2] - pb3[2]).powi(2)).sqrt();
            let (pa2, pb2) = (p2[s], p2[(s + 1) % f.len()]);
            let l2 = (pa2[0] - pb2[0]).hypot(pa2[1] - pb2[1]);
            max_err = max_err.max((l3 - l2).abs());
        }
    }
    Ok(PapercraftNet {
        pieces,
        tree_edges,
        cut_pairings,
        max_edge_length_error: max_err,
    })
}
