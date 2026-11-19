//! `H^2` Wythoff chamber realiser for all 8 Wythoff operations.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use crate::dsymbol::GeometryKind;
use crate::hyperbolic::inflation::{
    HypPointSet, IDENTITY, Mat3, Vec3, grow_vertex_completion, hyp_dist,
    hyperboloid_to_poincare, mat_mul, mat_vec, mink,
};

use super::wythoff_archimedean::{SnubHandedness, WythoffOp, WythoffTiling};
use super::wythoff_planar::{assemble_wythoff_result, solve3};

fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn mink_refl(n: Vec3) -> Mat3 {
    let jn = [n[0], n[1], -n[2]];
    core::array::from_fn(|i| core::array::from_fn(|j| IDENTITY[i][j] - 2.0 * n[i] * jn[j]))
}

fn h2_seed_mink(ns: &[Vec3; 3], mask: [f64; 3]) -> Vec3 {
    let mut sys = [[0.0; 4]; 3];
    for i in 0..3 {
        sys[i] = [ns[i][0], ns[i][1], -ns[i][2], mask[i]];
    }
    let mut p = solve3(sys);
    if p[2] < 0.0 {
        p = [-p[0], -p[1], -p[2]];
    }
    let norm = (-mink(p, p)).sqrt();
    [p[0] / norm, p[1] / norm, p[2] / norm]
}

fn h2_snub_seed(ns: &[Vec3; 3], rf: &Mat3, rv: &Mat3, re: &Mat3) -> Result<Vec3, String> {
    let cf = h2_seed_mink(ns, [0.0, 0.0, 1.0]);
    let ce = h2_seed_mink(ns, [0.0, 1.0, 0.0]);
    let cv = h2_seed_mink(ns, [1.0, 0.0, 0.0]);
    let pt = |u: f64, v: f64| -> Vec3 {
        let (w0, w1, w2) = (1.0, u.exp(), v.exp());
        let x = [
            w0 * cf[0] + w1 * ce[0] + w2 * cv[0],
            w0 * cf[1] + w1 * ce[1] + w2 * cv[1],
            w0 * cf[2] + w1 * ce[2] + w2 * cv[2],
        ];
        let n = (-mink(x, x)).sqrt();
        [x[0] / n, x[1] / n, x[2] / n]
    };
    let res = |u: f64, v: f64| -> (f64, f64) {
        let q = pt(u, v);
        let df = hyp_dist(q, mat_vec(rf, q));
        let dv = hyp_dist(q, mat_vec(rv, q));
        let de = hyp_dist(q, mat_vec(re, q));
        (df - de, dv - de)
    };
    let (mut u, mut v) = (0.0_f64, 0.0_f64);
    for _ in 0..35 {
        let (r0, r1) = res(u, v);
        if r0 * r0 + r1 * r1 < 1e-24 {
            break;
        }
        let h = 1e-6;
        let (u_p, u_m) = (res(u + h, v), res(u - h, v));
        let (v_p, v_m) = (res(u, v + h), res(u, v - h));
        let (dr0u, dr1u) = ((u_p.0 - u_m.0) / (2.0 * h), (u_p.1 - u_m.1) / (2.0 * h));
        let (dr0v, dr1v) = ((v_p.0 - v_m.0) / (2.0 * h), (v_p.1 - v_m.1) / (2.0 * h));
        let det = dr0u * dr1v - dr0v * dr1u;
        if det.abs() < 1e-18 {
            break;
        }
        u -= (dr1v * r0 - dr0v * r1) / det;
        v -= (-dr1u * r0 + dr0u * r1) / det;
    }
    let (r0, r1) = res(u, v);
    let resid = r0.hypot(r1);
    if resid.is_nan() || resid >= 1e-9 {
        return Err(format!("H^2 snub: equal-edge generating point not found (residual {resid:.3e}); refusing"));
    }
    Ok(pt(u, v))
}

pub(super) fn build_hyperbolic_wythoff(
    p: usize,
    q: usize,
    op: WythoffOp,
    layers: usize,
) -> Result<WythoffTiling, String> {
    let lay = layers.clamp(1, 3);
    let vc = grow_vertex_completion(p, q, lay, 600)?;
    let ap = PI / p as f64;
    let (chr, shr) = (vc.base.inradius.cosh(), vc.base.inradius.sinh());
    let ns = [
        [ap.sin(), -ap.cos(), 0.0],
        [0.0, 1.0, 0.0],
        [-chr * ap.cos(), -chr * ap.sin(), -shr],
    ];
    let s = [mink_refl(ns[0]), mink_refl(ns[1]), mink_refl(ns[2])];
    // Rotations (product order opposite to the S^2 code): rf about the face centre, rv about vertex 0,
    // re (half-turn) about the midpoint of edge 0; rf * rv = s0 s2 = re^{-1}.
    let rf = mat_mul(&s[0], &s[1]);
    let rv = mat_mul(&s[1], &s[2]);
    let re = mat_mul(&s[2], &s[0]);
    let rf_inv = mat_mul(&s[1], &s[0]);

    let (seed, is_snub, right_hand) = match op {
        WythoffOp::Regular => (h2_seed_mink(&ns, [1.0, 0.0, 0.0]), false, false),
        WythoffOp::Dual => (h2_seed_mink(&ns, [0.0, 0.0, 1.0]), false, false),
        WythoffOp::Rectified => (h2_seed_mink(&ns, [0.0, 1.0, 0.0]), false, false),
        WythoffOp::Truncated => (h2_seed_mink(&ns, [1.0, 1.0, 0.0]), false, false),
        WythoffOp::Bitruncated => (h2_seed_mink(&ns, [0.0, 1.0, 1.0]), false, false),
        WythoffOp::Cantellated => (h2_seed_mink(&ns, [1.0, 0.0, 1.0]), false, false),
        WythoffOp::Omnitruncated => (h2_seed_mink(&ns, [1.0, 1.0, 1.0]), false, false),
        WythoffOp::Snub { chirality } => (
            h2_snub_seed(&ns, &rf, &rv, &re)?,
            true,
            chirality == SnubHandedness::Right,
        ),
    };
    let min_edge = if is_snub {
        hyp_dist(seed, mat_vec(&rf, seed))
    } else {
        [0, 1, 2]
            .into_iter()
            .map(|i| hyp_dist(seed, mat_vec(&s[i], seed)))
            .filter(|&d| d > 1e-6)
            .fold(f64::INFINITY, f64::min)
    }
    .max(1e-3);

    let mut r_pows = vec![IDENTITY];
    for _ in 1..p {
        r_pows.push(mat_mul(r_pows.last().unwrap(), &rf));
    }
    let mut face_chambers = Vec::new();
    for rp in &r_pows {
        face_chambers.push(*rp);
        if !is_snub {
            face_chambers.push(mat_mul(rp, &s[0]));
        }
    }

    let even_faces: Vec<Mat3> = vc
        .faces
        .iter()
        .map(|g| if is_snub && det3(g) < 0.0 { mat_mul(g, &s[1]) } else { *g })
        .collect();
    let mut base_verts = HypPointSet::new(vc.base.edge_length());
    let mut base_vert_faces: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut edge_map: BTreeMap<(usize, usize), Vec<Vec3>> = BTreeMap::new();
    for (fi, g) in even_faces.iter().enumerate() {
        let mut f_vids = Vec::with_capacity(p);
        for (j, &bv) in vc.base.vertices.iter().enumerate() {
            let (vid, new) = base_verts.insert(mat_vec(g, bv));
            if new {
                base_vert_faces.push(Vec::new());
            }
            base_vert_faces[vid].push((fi, j));
            f_vids.push(vid);
        }
        for k in 0..p {
            let (u, v) = (f_vids[k], f_vids[(k + 1) % p]);
            let mid = mat_vec(g, vc.base.edge_midpoints[k]);
            edge_map.entry((u.min(v), u.max(v))).or_default().push(mid);
        }
    }

    let mut w_verts = HypPointSet::new(min_edge);
    let mut faces_out: Vec<Vec<usize>> = Vec::new();
    let mut face_verts_of_f: Vec<Vec<usize>> = vec![Vec::new(); even_faces.len()];
    for (fi, g) in even_faces.iter().enumerate() {
        let mut local_ids = Vec::new();
        for ch in &face_chambers {
            let pt3 = mat_vec(g, mat_vec(ch, seed));
            let (wid, _) = w_verts.insert(pt3);
            if !local_ids.contains(&wid) {
                local_ids.push(wid);
            }
        }
        face_verts_of_f[fi] = local_ids.clone();
        if local_ids.len() >= 3 {
            let fc = mat_vec(g, [0.0, 0.0, 1.0]);
            faces_out.push(order_ccw_h2(&w_verts.pts, local_ids, fc));
        }
    }

    let rv_seed = hyp_dist(seed, vc.base.vertices[0]);
    let re_seed = hyp_dist(seed, vc.base.edge_midpoints[0]);
    let mut seen_faces: BTreeSet<Vec<usize>> = faces_out
        .iter()
        .map(|f| {
            let mut s = f.clone();
            s.sort_unstable();
            s
        })
        .collect();

    if rv_seed > 1e-6 {
        for (vid, occ) in base_vert_faces.iter().enumerate() {
            if occ.len() != q {
                continue;
            }
            let vc3 = base_verts.pts[vid];
            let mut ring = Vec::new();
            for &(fi, _) in occ {
                for &wid in &face_verts_of_f[fi] {
                    if (hyp_dist(w_verts.pts[wid], vc3) - rv_seed).abs() < 0.25 * min_edge && !ring.contains(&wid) {
                        ring.push(wid);
                    }
                }
            }
            push_unique_h2(&w_verts.pts, &mut seen_faces, &mut faces_out, ring, vc3);
            if is_snub {
                for &(fi, j) in occ {
                    // Snub triangle (x, rf^{-1} x, rv x): its third side has length
                    // d(rf^{-1} x, rv x) = d(x, rf rv x) = d(x, re^{-1} x) = d(x, re x), so all three
                    // sides equal the common edge length of the snub.
                    let gc = mat_mul(&even_faces[fi], &r_pows[j]);
                    let p0 = w_verts.insert(mat_vec(&gc, seed)).0;
                    let p1 = w_verts.insert(mat_vec(&gc, mat_vec(&rf_inv, seed))).0;
                    let p2 = w_verts.insert(mat_vec(&gc, mat_vec(&rv, seed))).0;
                    let c_tri = [
                        (w_verts.pts[p0][0] + w_verts.pts[p1][0] + w_verts.pts[p2][0]) / 3.0,
                        (w_verts.pts[p0][1] + w_verts.pts[p1][1] + w_verts.pts[p2][1]) / 3.0,
                        (w_verts.pts[p0][2] + w_verts.pts[p1][2] + w_verts.pts[p2][2]) / 3.0,
                    ];
                    let n_t = (-mink(c_tri, c_tri)).sqrt();
                    push_unique_h2(
                        &w_verts.pts,
                        &mut seen_faces,
                        &mut faces_out,
                        vec![p0, p1, p2],
                        [c_tri[0] / n_t, c_tri[1] / n_t, c_tri[2] / n_t],
                    );
                }
            }
        }
    }

    if matches!(op, WythoffOp::Cantellated | WythoffOp::Omnitruncated) {
        for mids in edge_map.values() {
            if mids.len() != 2 {
                continue;
            }
            let mid = mids[0];
            let mut sq = Vec::new();
            for (wid, &wpt) in w_verts.pts.iter().enumerate() {
                if (hyp_dist(wpt, mid) - re_seed).abs() < 0.25 * min_edge {
                    sq.push(wid);
                }
            }
            push_unique_h2(&w_verts.pts, &mut seen_faces, &mut faces_out, sq, mid);
        }
    }

    let mut coords_3d = w_verts.pts;
    let mut coords_2d: Vec<[f64; 2]> = coords_3d.iter().map(|&x| hyperboloid_to_poincare(x)).collect();
    if right_hand {
        for p3 in &mut coords_3d {
            p3[1] = -p3[1];
        }
        for p2 in &mut coords_2d {
            p2[1] = -p2[1];
        }
        for f in &mut faces_out {
            f.reverse();
        }
    }
    Ok(assemble_wythoff_result(p, q, op, GeometryKind::Hyperbolic, coords_3d, coords_2d, faces_out))
}

fn push_unique_h2(
    pts: &[Vec3],
    seen_faces: &mut BTreeSet<Vec<usize>>,
    faces_out: &mut Vec<Vec<usize>>,
    cyc: Vec<usize>,
    centre: Vec3,
) {
    if cyc.len() < 3 {
        return;
    }
    let mut key = cyc.clone();
    key.sort_unstable();
    if seen_faces.insert(key) {
        faces_out.push(order_ccw_h2(pts, cyc, centre));
    }
}

fn order_ccw_h2(pts: &[Vec3], mut ids: Vec<usize>, c: Vec3) -> Vec<usize> {
    let p_c = hyperboloid_to_poincare(c);
    let ang_of = |idx: usize| -> f64 {
        let z = hyperboloid_to_poincare(pts[idx]);
        let (dx, dy) = (z[0] - p_c[0], z[1] - p_c[1]);
        let (dr, di) = (1.0 - (p_c[0] * z[0] + p_c[1] * z[1]), -(p_c[0] * z[1] - p_c[1] * z[0]));
        let den = dr * dr + di * di;
        let rx = (dx * dr + dy * di) / den;
        let ry = (dy * dr - dx * di) / den;
        ry.atan2(rx)
    };
    ids.sort_by(|&a, &b| ang_of(a).total_cmp(&ang_of(b)));
    ids
}
