//! `E^2` Wythoff chamber realiser and shared assembly helpers.

use std::collections::{BTreeMap, BTreeSet, HashMap};

// `BTreeMap` (not `HashMap`) for edge-keyed maps so face order is deterministic between runs.
use std::f64::consts::PI;

use crate::dsymbol::GeometryKind;

use super::vertex_config::wythoff_vertex_symbol;
use super::wythoff_archimedean::{SnubHandedness, WythoffOp, WythoffTiling};

pub(super) fn solve3(mut a: [[f64; 4]; 3]) -> [f64; 3] {
    for col in 0..3 {
        let piv = (col..3).max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs())).unwrap();
        a.swap(col, piv);
        let p = a[col][col];
        for v in a[col].iter_mut().skip(col) {
            *v /= p;
        }
        for r in 0..3 {
            if r != col {
                let f = a[r][col];
                for j in col..4 {
                    a[r][j] -= f * a[col][j];
                }
            }
        }
    }
    [a[0][3], a[1][3], a[2][3]]
}

pub(super) fn build_euclidean_wythoff(
    p: usize,
    q: usize,
    op: WythoffOp,
    layers: usize,
) -> Result<WythoffTiling, String> {
    let base_patch = super::unified_pq::build_unified_pq(p, q, layers.clamp(1, 3))?;
    let rad = 0.5 / (PI / p as f64).sin();
    let inrad = rad * (PI / p as f64).cos();
    let cv = [rad, 0.0_f64];
    let ce = [inrad * (PI / p as f64).cos(), inrad * (PI / p as f64).sin()];
    let (ap, sp, cp) = (PI / p as f64, (PI / p as f64).sin(), (PI / p as f64).cos());
    let e2_seed = |mask: [f64; 3]| -> [f64; 2] {
        let sys = [
            [sp, -cp, -mask[0], 0.0],
            [0.0, 1.0, -mask[1], 0.0],
            [-cp, -sp, -mask[2], -inrad],
        ];
        let sol = solve3(sys);
        [sol[0], sol[1]]
    };
    let e2_snub_seed = || -> Result<[f64; 2], String> {
        let (mut u, mut v) = (0.0_f64, 0.0_f64);
        let sq = PI / q as f64;
        let pt = |u: f64, v: f64| -> [f64; 2] {
            let (w0, w1, w2) = (1.0, u.exp(), v.exp());
            let ws = w0 + w1 + w2;
            [(w1 * ce[0] + w2 * cv[0]) / ws, (w1 * ce[1] + w2 * cv[1]) / ws]
        };
        let res = |u: f64, v: f64| -> (f64, f64) {
            let pt0 = pt(u, v);
            let df = ap.sin() * pt0[0].hypot(pt0[1]);
            let dv = sq.sin() * (pt0[0] - cv[0]).hypot(pt0[1] - cv[1]);
            let de = (pt0[0] - ce[0]).hypot(pt0[1] - ce[1]);
            (df - de, dv - de)
        };
        for _ in 0..35 {
            let (r0, r1) = res(u, v);
            if r0 * r0 + r1 * r1 < 1e-24 {
                break;
            }
            let h = 1e-6;
            let (up, um) = (res(u + h, v), res(u - h, v));
            let (vp, vm) = (res(u, v + h), res(u, v - h));
            let (j00, j10) = ((up.0 - um.0) / (2.0 * h), (up.1 - um.1) / (2.0 * h));
            let (j01, j11) = ((vp.0 - vm.0) / (2.0 * h), (vp.1 - vm.1) / (2.0 * h));
            let det = j00 * j11 - j01 * j10;
            if !det.is_finite() || det.abs() < 1e-18 {
                break;
            }
            u -= (j11 * r0 - j01 * r1) / det;
            v -= (-j10 * r0 + j00 * r1) / det;
        }
        let (r0, r1) = res(u, v);
        let resid = r0.hypot(r1);
        if resid.is_nan() || resid >= 1e-9 {
            return Err(format!(
                "snub {{{p}, {q}}}: equal-edge generating point not found (residual {resid:.3e}); refusing"
            ));
        }
        Ok(pt(u, v))
    };
    let (seed, is_snub, right_hand) = match op {
        WythoffOp::Regular => (cv, false, false),
        WythoffOp::Dual => ([0.0, 0.0], false, false),
        WythoffOp::Rectified => (ce, false, false),
        WythoffOp::Truncated => (e2_seed([1.0, 1.0, 0.0]), false, false),
        WythoffOp::Bitruncated => (e2_seed([0.0, 1.0, 1.0]), false, false),
        WythoffOp::Cantellated => (e2_seed([1.0, 0.0, 1.0]), false, false),
        WythoffOp::Omnitruncated => (e2_seed([1.0, 1.0, 1.0]), false, false),
        WythoffOp::Snub { chirality } => (e2_snub_seed()?, true, chirality == SnubHandedness::Right),
    };
    let seed_r = seed[0].hypot(seed[1]);
    let seed_th = seed[1].atan2(seed[0]);
    let qkey = |pt: [f64; 2]| ((pt[0] * 1e6).round() as i64, (pt[1] * 1e6).round() as i64);
    let mut vmap: HashMap<(i64, i64), usize> = HashMap::new();
    let mut coords_2d: Vec<[f64; 2]> = Vec::new();
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); base_patch.coords_2d.len()];
    let mut edge_faces: BTreeMap<(usize, usize), Vec<[f64; 2]>> = BTreeMap::new();
    let mut faces_out = Vec::new();
    let mut face_wverts = vec![Vec::new(); base_patch.faces.len()];

    for (fi, f) in base_patch.faces.iter().enumerate() {
        let fc = f.iter().fold([0.0, 0.0], |acc, &vi| {
            [acc[0] + base_patch.coords_2d[vi][0] / p as f64, acc[1] + base_patch.coords_2d[vi][1] / p as f64]
        });
        for &vi in f {
            vert_faces[vi].push(fi);
        }
        for k in 0..p {
            let (u, v) = (f[k], f[(k + 1) % p]);
            let mid = [
                0.5 * (base_patch.coords_2d[u][0] + base_patch.coords_2d[v][0]),
                0.5 * (base_patch.coords_2d[u][1] + base_patch.coords_2d[v][1]),
            ];
            edge_faces.entry((u.min(v), u.max(v))).or_default().push(mid);
        }
        let v0 = base_patch.coords_2d[f[0]];
        let th0 = (v0[1] - fc[1]).atan2(v0[0] - fc[0]);
        let mut local = Vec::new();
        for k in 0..p {
            let base_th = th0 + 2.0 * PI * (k as f64) / (p as f64);
            let angles = if is_snub { vec![base_th + seed_th] } else { vec![base_th + seed_th, base_th - seed_th] };
            for a in angles {
                let pt = [fc[0] + seed_r * a.cos(), fc[1] + seed_r * a.sin()];
                let wid = *vmap.entry(qkey(pt)).or_insert_with(|| {
                    let id = coords_2d.len();
                    coords_2d.push(pt);
                    id
                });
                if !local.contains(&wid) {
                    local.push(wid);
                }
            }
        }
        face_wverts[fi] = local.clone();
        if local.len() >= 3 {
            faces_out.push(order_ccw_e2(&coords_2d, local, fc));
        }
    }

    let rv_seed = (seed[0] - cv[0]).hypot(seed[1] - cv[1]);
    let re_seed = (seed[0] - ce[0]).hypot(seed[1] - ce[1]);
    let mut seen: BTreeSet<Vec<usize>> = faces_out
        .iter()
        .map(|f| {
            let mut s = f.clone();
            s.sort_unstable();
            s
        })
        .collect();
    if rv_seed > 1e-6 {
        for (vi, occ) in vert_faces.iter().enumerate() {
            if occ.len() != q {
                continue;
            }
            let vc2 = base_patch.coords_2d[vi];
            let mut ring = Vec::new();
            for &fi in occ {
                for &wid in &face_wverts[fi] {
                    if ((coords_2d[wid][0] - vc2[0]).hypot(coords_2d[wid][1] - vc2[1]) - rv_seed).abs() < 1e-4
                        && !ring.contains(&wid)
                    {
                        ring.push(wid);
                    }
                }
            }
            if ring.len() >= 3 {
                let mut key = ring.clone();
                key.sort_unstable();
                if seen.insert(key) {
                    faces_out.push(order_ccw_e2(&coords_2d, ring, vc2));
                }
            }
        }
    }
    if is_snub {
        for _pass in 0..2 {
            let cur_len = faces_out.len();
            let mut extras = Vec::new();
            for cyc in &faces_out[..cur_len] {
                let fc = cyc.iter().fold([0.0, 0.0], |acc, &vi| {
                    [acc[0] + coords_2d[vi][0] / cyc.len() as f64, acc[1] + coords_2d[vi][1] / cyc.len() as f64]
                });
                for k in 0..cyc.len() {
                    let (a, b) = (cyc[k], cyc[(k + 1) % cyc.len()]);
                    let (pa, pb) = (coords_2d[a], coords_2d[b]);
                    let elen = (pa[0] - pb[0]).hypot(pa[1] - pb[1]);
                    let mid = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                    let nrm = [-(pb[1] - pa[1]) / elen, (pb[0] - pa[0]) / elen];
                    let h = 0.5 * 3.0_f64.sqrt() * elen;
                    for sgn in [-1.0_f64, 1.0] {
                        let cand = [mid[0] + sgn * h * nrm[0], mid[1] + sgn * h * nrm[1]];
                        if (cand[0] - fc[0]).hypot(cand[1] - fc[1]) > (mid[0] - fc[0]).hypot(mid[1] - fc[1])
                            && let Some(&c) = vmap.get(&qkey(cand))
                        {
                            let mut key = vec![a, b, c];
                            key.sort_unstable();
                            if seen.insert(key) {
                                let tc = [(pa[0] + pb[0] + cand[0]) / 3.0, (pa[1] + pb[1] + cand[1]) / 3.0];
                                extras.push(order_ccw_e2(&coords_2d, vec![a, b, c], tc));
                            }
                        }
                    }
                }
            }
            faces_out.extend(extras);
        }
    }
    if matches!(op, WythoffOp::Cantellated | WythoffOp::Omnitruncated) {
        for mids in edge_faces.values() {
            if mids.len() != 2 {
                continue;
            }
            let mid = mids[0];
            let mut sq = Vec::new();
            for (wid, &wpt) in coords_2d.iter().enumerate() {
                if ((wpt[0] - mid[0]).hypot(wpt[1] - mid[1]) - re_seed).abs() < 1e-4 {
                    sq.push(wid);
                }
            }
            if sq.len() >= 3 {
                let mut key = sq.clone();
                key.sort_unstable();
                if seen.insert(key) {
                    faces_out.push(order_ccw_e2(&coords_2d, sq, mid));
                }
            }
        }
    }

    if right_hand {
        for pt in &mut coords_2d {
            pt[1] = -pt[1];
        }
        for f in &mut faces_out {
            f.reverse();
        }
    }
    let coords_3d = coords_2d.iter().map(|&[x, y]| [x, y, 0.0]).collect();
    Ok(assemble_wythoff_result(p, q, op, GeometryKind::Euclidean, coords_3d, coords_2d, faces_out))
}

fn order_ccw_e2(pts: &[[f64; 2]], mut ids: Vec<usize>, c: [f64; 2]) -> Vec<usize> {
    ids.sort_by(|&a, &b| {
        let aa = (pts[a][1] - c[1]).atan2(pts[a][0] - c[0]);
        let ab = (pts[b][1] - c[1]).atan2(pts[b][0] - c[0]);
        aa.total_cmp(&ab)
    });
    ids
}

pub(super) fn assemble_wythoff_result(
    p: usize,
    q: usize,
    op: WythoffOp,
    geometry_kind: GeometryKind,
    coords_3d: Vec<[f64; 3]>,
    coords_2d: Vec<[f64; 2]>,
    faces: Vec<Vec<usize>>,
) -> WythoffTiling {
    let mut edge_set = BTreeSet::new();
    let mut face_type_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for f in &faces {
        *face_type_counts.entry(f.len()).or_insert(0) += 1;
        for k in 0..f.len() {
            let (u, v) = (f[k], f[(k + 1) % f.len()]);
            edge_set.insert((u.min(v), u.max(v)));
        }
    }
    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let (n_v, n_e, n_f) = (coords_3d.len(), edges.len(), faces.len());
    // Ringed seeds solve an exact 3x3 linear system; the snub seed is a Newton root (f64).
    let provenance = if matches!(op, WythoffOp::Snub { .. }) {
        "[COMPUTATIONAL EVIDENCE] snub generating point from a Newton solve (residual < 1e-9)"
    } else {
        "[PROVED] Wythoff construction (Coxeter, Longuet-Higgins & Miller 1954)"
    };
    let (op_name, chirality) = match op {
        WythoffOp::Regular => ("Regular", None),
        WythoffOp::Dual => ("Dual", None),
        WythoffOp::Rectified => ("Rectified", None),
        WythoffOp::Truncated => ("Truncated", None),
        WythoffOp::Bitruncated => ("Bitruncated", None),
        WythoffOp::Cantellated => ("Cantellated", None),
        WythoffOp::Omnitruncated => ("Omnitruncated", None),
        WythoffOp::Snub { chirality } => (
            "Snub",
            Some(match chirality {
                SnubHandedness::Left => "Left".to_string(),
                SnubHandedness::Right => "Right".to_string(),
            }),
        ),
    };
    WythoffTiling {
        p,
        q,
        op_name: op_name.to_string(),
        chirality,
        vertex_symbol: wythoff_vertex_symbol(p, q, op),
        geometry_kind,
        n_vertices: n_v,
        n_edges: n_e,
        n_faces: n_f,
        euler_chi: n_v as i64 - n_e as i64 + n_f as i64,
        face_type_counts,
        stopped: None,
        coords_3d,
        coords_2d,
        edges,
        faces,
        provenance: provenance.to_string(),
    }
}
