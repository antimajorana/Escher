//! Stage 4 3D elastic relaxation (spring stretching + orientation-safe hinge bending)
//! for surfaces (`cylinder`, `sphere`, `projective_plane`, `torus`, `genus`) and planar defect sheets.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use faer::{Mat, Side};
use serde::{Deserialize, Serialize};

use crate::types::SurfaceMesh;

/// Result of 3D elastic relaxation matching `papertilling.surfaces.elastic.ElasticResult`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceElasticResult {
    pub xyz: Vec<[f64; 3]>,
    pub cover_xyz: Option<Vec<[f64; 3]>>,
    pub stretch_energy: f64,
    pub bend_energy: f64,
    pub total_energy: f64,
    pub r_mean: f64,
    pub fvk_gamma: f64,
    pub asphericity: f64,
    pub bond_strain: Vec<f64>,
    pub site_strain: Vec<f64>,
    pub radial_buckling: Vec<f64>,
    pub converged: bool,
    pub n_iter: usize,
}

#[derive(Clone, Copy)]
struct Hinge { p: usize, q: usize, a: usize, b: usize, w: f64 }

#[inline] fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline] fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
#[inline] fn norm3(a: [f64; 3]) -> f64 { dot3(a, a).sqrt() }
#[inline] fn scale3(a: [f64; 3], s: f64) -> [f64; 3] { [a[0] * s, a[1] * s, a[2] * s] }
#[inline] fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn lbfgs_solve<F>(x: &mut [f64], max_iters: usize, gtol: f64, mut fg: F) -> usize
where F: FnMut(&[f64], &mut [f64]) -> f64 {
    let n = x.len();
    if n == 0 || max_iters == 0 { return 0; }
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let max_abs = |g: &[f64]| g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let (mut g, mut dir, mut q, mut x_try, mut g_try) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut f = fg(x, &mut g);
    if !f.is_finite() || max_abs(&g) <= gtol { return 0; }
    const M: usize = 10;
    let (mut sh, mut yh, mut rh): (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) = Default::default();
    let (mut alpha, mut iters) = ([0.0; M], 0usize);
    for it in 0..max_iters {
        if max_abs(&g) <= gtol { break; }
        iters = it + 1;
        q.copy_from_slice(&g);
        for i in (0..sh.len()).rev() {
            alpha[i] = rh[i] * dot(&sh[i], &q);
            for (qj, &yj) in q.iter_mut().zip(&yh[i]) { *qj -= alpha[i] * yj; }
        }
        let gamma = match (sh.last(), yh.last()) {
            (Some(s), Some(y)) => { let (sy, yy) = (dot(s, y), dot(y, y)); if yy > 1e-24 && sy > 1e-24 { (sy / yy).clamp(1e-4, 1e4) } else { 1.0 } }
            _ => { let gn = dot(&g, &g).sqrt(); if gn > 1.0 { 1.0 / gn } else { 1.0 } }
        };
        for i in 0..n { dir[i] = q[i] * gamma; }
        for i in 0..sh.len() {
            let b = rh[i] * dot(&yh[i], &dir);
            for (dj, &sj) in dir.iter_mut().zip(&sh[i]) { *dj += (alpha[i] - b) * sj; }
        }
        for d in &mut dir { *d = -*d; }
        let mut dg0 = dot(&dir, &g);
        if dg0 >= -1e-20 || !dg0.is_finite() {
            sh.clear(); yh.clear(); rh.clear();
            let gn = dot(&g, &g).sqrt().max(1.0);
            for i in 0..n { dir[i] = -g[i] / gn; }
            dg0 = dot(&dir, &g);
            if dg0 >= 0.0 { break; }
        }
        let (mut step, mut ok) = (1.0_f64, false);
        for _ in 0..26 {
            for i in 0..n { x_try[i] = x[i] + step * dir[i]; }
            let ft = fg(&x_try, &mut g_try);
            if ft.is_finite() && g_try.iter().all(|v| v.is_finite()) && ft <= f + 1e-4 * step * dg0 { f = ft; ok = true; break; }
            step *= 0.5;
        }
        if !ok { if !sh.is_empty() { sh.clear(); yh.clear(); rh.clear(); continue; } break; }
        let sv: Vec<f64> = x_try.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let yv: Vec<f64> = g_try.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        x.copy_from_slice(&x_try); g.copy_from_slice(&g_try);
        let (step_max, sy, sn, yn) = (max_abs(&sv), dot(&sv, &yv), dot(&sv, &sv), dot(&yv, &yv));
        if sy.is_finite() && sy > 1e-18 * (sn * yn).sqrt().max(1e-18) {
            if sh.len() == M { sh.remove(0); yh.remove(0); rh.remove(0); }
            rh.push(1.0 / sy); sh.push(sv); yh.push(yv);
        }
        if step_max <= 1e-14 { break; }
    }
    iters
}

/// Deterministic proper `SO(3)` principal-inertia gauge `(X_aligned, com, R)`.
pub fn canonical_gauge(x: &[[f64; 3]], is_proj: bool) -> (Vec<[f64; 3]>, [f64; 3], [[f64; 3]; 3]) {
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if x.is_empty() { return (Vec::new(), [0.0; 3], eye); }
    let inv_n = 1.0 / (x.len() as f64);
    let com = if is_proj { [0.0; 3] } else { scale3(x.iter().fold([0.0; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]), inv_n) };
    let pts: Vec<[f64; 3]> = x.iter().map(|p| sub3(*p, com)).collect();
    if pts.len() < 3 { return (pts, com, eye); }
    let mut cov = [[0.0; 3]; 3];
    for p in &pts { for r in 0..3 { for c in 0..3 { cov[r][c] += p[r] * p[c] * inv_n; } } }
    let Ok(ev) = Mat::<f64>::from_fn(3, 3, |r, c| cov[r][c]).self_adjoint_eigen(Side::Lower) else { return (pts, com, eye); };
    let u = ev.U();
    let (mut v0, mut v1) = ([u[(0, 2)], u[(1, 2)], u[(2, 2)]], [u[(0, 1)], u[(1, 1)], u[(2, 1)]]);
    for v in [&mut v0, &mut v1] {
        let vmax = v[0].abs().max(v[1].abs()).max(v[2].abs());
        if vmax > 1e-15 && (0..3).find(|&j| v[j].abs() >= vmax - 1e-12).is_some_and(|j| v[j] < 0.0) { *v = scale3(*v, -1.0); }
    }
    v1 = sub3(v1, scale3(v0, dot3(v1, v0)));
    if norm3(v1) <= 1e-12 {
        let e_min = (0..3).min_by(|&a, &b| v0[a].abs().total_cmp(&v0[b].abs())).unwrap();
        let mut e = [0.0; 3]; e[e_min] = 1.0;
        v1 = sub3(e, scale3(v0, v0[e_min]));
    }
    v1 = scale3(v1, 1.0 / norm3(v1));
    let v2r = cross3(v0, v1);
    let v2 = scale3(v2r, 1.0 / norm3(v2r).max(1e-15));
    let rot = [[v0[0], v1[0], v2[0]], [v0[1], v1[1], v2[1]], [v0[2], v1[2], v2[2]]];
    (pts.iter().map(|&p| [dot3(p, v0), dot3(p, v1), dot3(p, v2)]).collect(), com, rot)
}

/// Centroid-to-corner distances of the flat cyclic polygon with edge lengths `ls`.
pub fn cyclic_polygon_radii(ls: &[f64]) -> Vec<f64> {
    let d = ls.len();
    if d == 0 { return Vec::new(); }
    let (mean, l_min) = (ls.iter().sum::<f64>() / (d as f64), ls.iter().copied().fold(f64::INFINITY, f64::min));
    let (imax, &l_max) = ls.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let reg_r = mean / (2.0 * (PI / (d as f64)).sin());
    if l_max - l_min <= 1e-7 * mean.max(1e-12) { return vec![reg_r; d]; }
    let others: Vec<f64> = ls.iter().enumerate().filter(|&(i, _)| i != imax).map(|(_, &v)| v).collect();
    if l_max >= others.iter().sum::<f64>() - 1e-9 { return vec![reg_r; d]; }
    let r_min = 0.5 * l_max + 1e-9;
    let sum_ang = |r: f64| -> f64 { others.iter().map(|&l| 2.0 * (l / (2.0 * r)).min(1.0).asin()).sum() };
    let outside = sum_ang(r_min) < PI;
    let (mut lo, mut hi) = (r_min, ls.iter().sum::<f64>().max(1.0));
    for _ in 0..72 {
        let (mid, s_max) = (0.5 * (lo + hi), 2.0 * (l_max / (lo + hi)).min(1.0).asin());
        let s_oth = sum_ang(mid);
        if if outside { s_max - s_oth > 0.0 } else { s_max + s_oth > 2.0 * PI } { lo = mid; } else { hi = mid; }
    }
    let (rad, mut th, mut pts) = (0.5 * (lo + hi), 0.0_f64, Vec::with_capacity(d));
    for (m, &l) in ls.iter().enumerate() {
        let da = 2.0 * (l / (2.0 * rad)).min(1.0).asin();
        pts.push([rad * th.cos(), rad * th.sin()]);
        th += if outside && m == imax { -da } else { da };
    }
    let c = pts.iter().fold([0.0; 2], |a, p| [a[0] + p[0], a[1] + p[1]]);
    pts.into_iter().map(|p| (p[0] - c[0] / (d as f64)).hypot(p[1] - c[1] / (d as f64))).collect()
}

/// Flat-template centroid-to-corner spoke rest lengths for every face with `d >= 4`.
pub fn face_spoke_lengths(
    faces: &[Vec<usize>], bmap: &BTreeMap<(usize, usize), f64>,
    bvec: Option<&HashMap<(usize, usize), [f64; 2]>>, uv: Option<&[[f64; 2]]>,
) -> BTreeMap<usize, Vec<f64>> {
    let spokes_from = |pts: &[[f64; 2]]| -> Vec<f64> {
        let (d, c) = (pts.len() as f64, pts.iter().fold([0.0; 2], |a, p| [a[0] + p[0], a[1] + p[1]]));
        pts.iter().map(|p| (p[0] - c[0] / d).hypot(p[1] - c[1] / d)).collect()
    };
    let mut out = BTreeMap::new();
    for (fi, f) in faces.iter().enumerate() {
        let d = f.len();
        if d < 4 { continue; }
        let ls: Vec<f64> = (0..d).map(|m| bmap.get(&(f[m].min(f[(m + 1) % d]), f[m].max(f[(m + 1) % d]))).copied().unwrap_or(1.0)).collect();
        if let Some(bv) = bvec
            && let Some(vecs) = (0..d).map(|m| bv.get(&(f[m], f[(m + 1) % d])).copied()).collect::<Option<Vec<_>>>()
            && vecs.iter().fold([0.0; 2], |a, v| [a[0] + v[0], a[1] + v[1]]).iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-5
        {
            let mut cur = [0.0; 2];
            let pts: Vec<[f64; 2]> = vecs.iter().map(|v| { let p = cur; cur = [cur[0] + v[0], cur[1] + v[1]]; p }).collect();
            out.insert(fi, spokes_from(&pts));
        } else if let Some(uv_pts) = uv
            && (0..d).all(|m| ((uv_pts[f[(m + 1) % d]][0] - uv_pts[f[m]][0]).hypot(uv_pts[f[(m + 1) % d]][1] - uv_pts[f[m]][1]) - ls[m]).abs() < 1e-5)
        {
            let pts: Vec<[f64; 2]> = f.iter().map(|&u| uv_pts[u]).collect();
            out.insert(fi, spokes_from(&pts));
        } else {
            out.insert(fi, cyclic_polygon_radii(&ls));
        }
    }
    out
}

/// Estimate total interior Frank angle (degrees) via Gauss-Bonnet on rest-length triangulated faces.
pub fn estimate_planar_frank_deg(faces: &[Vec<usize>], bmap: &BTreeMap<(usize, usize), f64>, spoke_lengths: &BTreeMap<usize, Vec<f64>>) -> f64 {
    if faces.is_empty() { return 0.0; }
    let (mut edge_cnt, mut v_max): (HashMap<(usize, usize), usize>, usize) = (HashMap::new(), 0);
    for f in faces {
        for m in 0..f.len() {
            let (u, v) = (f[m], f[(m + 1) % f.len()]);
            v_max = v_max.max(u).max(v);
            *edge_cnt.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    let (mut boundary, mut in_face, mut ang_sum) = (vec![false; v_max + 1], vec![false; v_max + 1], vec![0.0; v_max + 1]);
    for (&(u, v), &cnt) in &edge_cnt { if cnt < 2 { boundary[u] = true; boundary[v] = true; } }
    let edge_l = |u: usize, v: usize| bmap.get(&(u.min(v), u.max(v))).copied().unwrap_or(1.0);
    let tri_angle = |a: f64, b: f64, c: f64| ((a * a + b * b - c * c) / (2.0 * a * b).max(1e-15)).clamp(-1.0, 1.0).acos();
    for (fi, f) in faces.iter().enumerate() {
        let d = f.len();
        if d < 3 { continue; }
        for &u in f { in_face[u] = true; }
        if d == 3 {
            for m in 0..3 { ang_sum[f[m]] += tri_angle(edge_l(f[(m + 2) % 3], f[m]), edge_l(f[m], f[(m + 1) % 3]), edge_l(f[(m + 2) % 3], f[(m + 1) % 3])); }
            continue;
        }
        let el: Vec<f64> = (0..d).map(|m| edge_l(f[m], f[(m + 1) % d])).collect();
        let fallback = cyclic_polygon_radii(&el);
        let sp = spoke_lengths.get(&fi).unwrap_or(&fallback);
        let gamma_sum: f64 = (0..d).map(|m| tri_angle(sp[m], sp[(m + 1) % d], el[m])).sum();
        if (gamma_sum - 2.0 * PI).abs() > 1e-4
            && let Some(angs) = crate::defect::sheet3d_polygon::reconstruct_concave_planar_angles(sp, &el)
        {
            for m in 0..d { ang_sum[f[m]] += angs[m]; }
            continue;
        }
        for m in 0..d {
            let (pm, nm) = ((m + d - 1) % d, (m + 1) % d);
            ang_sum[f[m]] += tri_angle(sp[m], edge_l(f[pm], f[m]), sp[pm]) + tri_angle(sp[m], edge_l(f[m], f[nm]), sp[nm]);
        }
    }
    (0..=v_max).filter(|&u| in_face[u] && !boundary[u]).map(|u| 2.0 * PI - ang_sum[u]).sum::<f64>().to_degrees()
}

fn build_hinges(
    faces: &[Vec<usize>],
    vc_map: &BTreeMap<usize, usize>,
    concave_tris: &BTreeMap<usize, Vec<[usize; 3]>>,
    is_proj: bool,
) -> Vec<Hinge> {
    let mut edge_inc: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    let (mut hinges, mult) = (Vec::new(), if is_proj { 0.5 } else { 1.0 });
    for (fi, f) in faces.iter().enumerate() {
        let d = f.len();
        if d < 3 { continue; }
        if let Some(tris) = concave_tris.get(&fi) {
            let mut local_inc: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for &[ia, ib, ic] in tris {
                let (u, v, w) = (f[ia], f[ib], f[ic]);
                for (a, b, opp) in [(u, v, w), (v, w, u), (w, u, v)] {
                    local_inc.entry((a.min(b), a.max(b))).or_default().push(opp);
                }
            }
            for m in 0..d {
                let (u, v) = (f[m], f[(m + 1) % d]);
                if let Some(&opp) = local_inc.get(&(u.min(v), u.max(v))).and_then(|v_o| v_o.first()) {
                    edge_inc.entry((u.min(v), u.max(v))).or_default().push((fi, opp));
                }
            }
            for ((du, dv), opps) in local_inc {
                if opps.len() == 2 && opps[0] != opps[1] {
                    hinges.push(Hinge { p: du, q: dv, a: opps[0], b: opps[1], w: 0.5 * mult });
                }
            }
            continue;
        }
        for m in 0..d {
            let (u, v) = (f[m], f[(m + 1) % d]);
            edge_inc.entry((u.min(v), u.max(v))).or_default().push((fi, if d == 3 { f[(m + 2) % 3] } else { vc_map[&fi] }));
        }
        if d >= 4 {
            for m in 0..d { hinges.push(Hinge { p: vc_map[&fi], q: f[m], a: f[(m + d - 1) % d], b: f[(m + 1) % d], w: 0.5 * mult }); }
        }
    }
    for ((u, v), inc) in edge_inc {
        if inc.len() == 2 && inc[0].0 != inc[1].0 { hinges.push(Hinge { p: u, q: v, a: inc[0].1, b: inc[1].1, w: mult }); }
    }
    hinges
}

fn eval_energy_grad(p: &[[f64; 3]], springs: &[(usize, usize, f64, f64)], hinges: &[Hinge], kappa_eff: f64, grad: &mut [[f64; 3]]) -> (f64, f64) {
    grad.fill([0.0; 3]);
    let mut es = 0.0;
    for &(u, v, l0, w) in springs {
        let dv = sub3(p[u], p[v]);
        let len = (dot3(dv, dv) + 1e-24).sqrt();
        let diff = len - l0;
        es += 0.5 * w * diff * diff;
        let c = w * diff / len;
        for k in 0..3 { grad[u][k] += c * dv[k]; grad[v][k] -= c * dv[k]; }
    }
    let mut eb = 0.0;
    if kappa_eff > 0.0 {
        for h in hinges {
            let (uv, va, vb) = (sub3(p[h.q], p[h.p]), sub3(p[h.a], p[h.p]), sub3(p[h.b], p[h.p]));
            let (na_r, nb_r) = (cross3(uv, va), cross3(vb, uv));
            let (la, lb) = ((dot3(na_r, na_r) + 1e-20).sqrt(), (dot3(nb_r, nb_r) + 1e-20).sqrt());
            let (na, nb) = (scale3(na_r, 1.0 / la), scale3(nb_r, 1.0 / lb));
            let wh = 0.5 * kappa_eff * h.w;
            eb += wh * (1.0 - dot3(na, nb));
            let (wa, wb) = (scale3(nb, -wh), scale3(na, -wh));
            let ga = scale3(sub3(wa, scale3(na, dot3(wa, na))), 1.0 / la);
            let gb = scale3(sub3(wb, scale3(nb, dot3(wb, nb))), 1.0 / lb);
            let (c1, c2, gao, gbo) = (cross3(va, ga), cross3(gb, vb), cross3(ga, uv), cross3(uv, gb));
            let gq = [c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]];
            let gp = [-(gq[0] + gao[0] + gbo[0]), -(gq[1] + gao[1] + gbo[1]), -(gq[2] + gao[2] + gbo[2])];
            for (idx, g) in [(h.p, gp), (h.q, gq), (h.a, gao), (h.b, gbo)] {
                for k in 0..3 { grad[idx][k] += g[k]; }
            }
        }
    }
    (es, eb)
}

/// Core indexed 3D elastic relaxation shared by `relax_surface_3d` and `relax_defect_sheet_3d_json`.
#[allow(clippy::too_many_arguments)]
pub fn relax_indexed_elastic(
    v_count: usize,
    x0: &[[f64; 3]],
    edges: &[(usize, usize)],
    edge_l0: &[f64],
    bond_edge_idx: &[usize],
    faces: &[Vec<usize>],
    spoke_lengths: &BTreeMap<usize, Vec<f64>>,
    kappa_b: f64,
    max_iters: usize,
    is_proj: bool,
    is_open_sheet: bool,
) -> Result<SurfaceElasticResult, String> {
    if !kappa_b.is_finite() || kappa_b < 0.0 {
        return Err("kappa_b must be finite and non-negative".into());
    }
    let n_pts = if is_proj { 2 * v_count } else { v_count };
    if x0.len() != n_pts || edge_l0.len() != edges.len() {
        return Err(format!("inconsistent input lengths: x0={}, edges={}, edge_l0={}", x0.len(), edges.len(), edge_l0.len()));
    }
    if edges.iter().any(|&(u, v)| u >= n_pts || v >= n_pts)
        || bond_edge_idx.iter().any(|&i| i >= edges.len())
        || faces.iter().flatten().any(|&u| u >= n_pts)
    {
        return Err("index out of bounds in edges, bond_edge_idx, or faces".into());
    }
    let l_mean = if edge_l0.is_empty() { 1.0 } else { edge_l0.iter().sum::<f64>() / (edge_l0.len() as f64) };
    let kappa_eff = kappa_b * l_mean * l_mean;
    let sw = if is_proj { 0.5 } else { 1.0 };

    let edge_map: HashMap<(usize, usize), f64> = edges.iter().zip(edge_l0).map(|(&(u, v), &l0)| ((u.min(v), u.max(v)), l0)).collect();
    let mut springs: Vec<(usize, usize, f64, f64)> = edges.iter().zip(edge_l0).map(|(&(u, v), &l0)| (u, v, l0, sw)).collect();
    let (mut vc_map, mut concave_tris, mut c0) = (BTreeMap::new(), BTreeMap::new(), Vec::new());
    for (fi, f) in faces.iter().enumerate() {
        let d = f.len();
        if d >= 4 {
            let el: Vec<f64> = (0..d).map(|m| edge_map.get(&(f[m].min(f[(m + 1) % d]), f[m].max(f[(m + 1) % d]))).copied().unwrap_or(l_mean)).collect();
            let xy_matches = is_open_sheet && (0..d).all(|m| {
                let (a, b) = (f[m], f[(m + 1) % d]);
                ((x0[b][0] - x0[a][0]).hypot(x0[b][1] - x0[a][1]) - el[m]).abs() <= 1e-12 + 1e-9 * el[m]
            });
            let local_pts: Vec<_> = f.iter().map(|&u| if xy_matches { [x0[u][0], x0[u][1], 0.0] } else { x0[u] }).collect();
            let indices: Vec<_> = (0..d).collect();
            if let Some(tris) = crate::defect::sheet3d_polygon::concave_face_triangulation(&local_pts, &indices) {
                concave_tris.insert(fi, tris);
            }
            // Stretching still uses the supplied rest spokes on every face.
            let vc = n_pts + vc_map.len();
            vc_map.insert(fi, vc);
            let s = f.iter().fold([0.0; 3], |a, &u| [a[0] + x0[u][0], a[1] + x0[u][1], a[2] + x0[u][2]]);
            c0.push([s[0] / (d as f64), s[1] / (d as f64), s[2] / (d as f64)]);
            let fallback;
            let sp = match spoke_lengths.get(&fi) {
                Some(v) if v.len() == d => v,
                _ => { fallback = cyclic_polygon_radii(&vec![l_mean; d]); &fallback }
            };
            for (m, &v_idx) in f.iter().enumerate() { springs.push((vc, v_idx, sp[m], sw)); }
        }
    }
    let hinges = build_hinges(faces, &vc_map, &concave_tris, is_proj);

    let mut x = Vec::with_capacity(3 * (v_count + c0.len()));
    for p in x0.iter().take(v_count).chain(c0.iter()) { x.extend_from_slice(p); }
    let (mut p_buf, mut g_buf) = (vec![[0.0; 3]; n_pts + c0.len()], vec![[0.0; 3]; n_pts + c0.len()]);
    let unpack = |flat: &[f64], out: &mut [[f64; 3]]| {
        for i in 0..v_count {
            let pt = [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]];
            out[i] = pt;
            if is_proj { out[i + v_count] = [-pt[0], -pt[1], -pt[2]]; }
        }
        for k in 0..c0.len() { out[n_pts + k] = [flat[3 * (v_count + k)], flat[3 * (v_count + k) + 1], flat[3 * (v_count + k) + 2]]; }
    };
    let mut fg = |flat: &[f64], grad_out: &mut [f64]| -> f64 {
        unpack(flat, &mut p_buf);
        let (es, eb) = eval_energy_grad(&p_buf, &springs, &hinges, kappa_eff, &mut g_buf);
        for i in 0..v_count {
            for c in 0..3 { grad_out[3 * i + c] = if is_proj { g_buf[i][c] - g_buf[i + v_count][c] } else { g_buf[i][c] }; }
        }
        for k in 0..c0.len() { grad_out[3 * (v_count + k)..3 * (v_count + k + 1)].copy_from_slice(&g_buf[n_pts + k]); }
        es + eb
    };

    let n_iter = if max_iters == 0 || v_count == 0 { 0 } else { lbfgs_solve(&mut x, max_iters, 1e-7, &mut fg) };
    let mut g_fin = vec![0.0; x.len()];
    let _ = fg(&x, &mut g_fin);
    let max_g = g_fin.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let converged = max_iters == 0 || v_count == 0 || max_g <= 1e-6;

    unpack(&x, &mut p_buf);
    let (xyz, com, rot) = canonical_gauge(&p_buf[..v_count], is_proj);
    let cover_xyz = is_proj.then(|| xyz.iter().copied().chain(xyz.iter().map(|p| [-p[0], -p[1], -p[2]])).collect::<Vec<_>>());
    if let Some(ref cov) = cover_xyz { p_buf[..2 * v_count].copy_from_slice(cov); } else { p_buf[..v_count].copy_from_slice(&xyz); }
    for k in 0..c0.len() {
        let d = sub3(p_buf[n_pts + k], com);
        p_buf[n_pts + k] = [0, 1, 2].map(|c| d[0] * rot[0][c] + d[1] * rot[1][c] + d[2] * rot[2][c]);
    }
    let (stretch_energy, bend_energy) = eval_energy_grad(&p_buf, &springs, &hinges, kappa_eff, &mut g_buf);
    let r_norms: Vec<f64> = xyz.iter().map(|&p| norm3(p)).collect();
    let r_mean = if v_count > 0 { r_norms.iter().sum::<f64>() / (v_count as f64) } else { 0.0 };
    let asphericity = if v_count > 0 && r_mean > 1e-15 {
        r_norms.iter().map(|&r| (r - r_mean).powi(2)).sum::<f64>() / ((v_count as f64) * r_mean * r_mean)
    } else { 0.0 };
    let fvk_gamma = if kappa_eff > 0.0 && r_mean > 0.0 { (r_mean * r_mean) / kappa_eff } else { f64::INFINITY };
    let edge_strain: Vec<f64> = edges.iter().zip(edge_l0).map(|(&(u, v), &l0)| (norm3(sub3(p_buf[u], p_buf[v])) - l0) / l0.max(1e-12)).collect();
    let bond_strain: Vec<f64> = bond_edge_idx.iter().map(|&idx| edge_strain[idx]).collect();
    let (mut ssum, mut scnt) = (vec![0.0_f64; v_count], vec![0usize; v_count]);
    for (idx, &(u, v)) in edges.iter().enumerate() {
        for w in [u, v] { let wi = if is_proj { w % v_count } else { w }; ssum[wi] += edge_strain[idx]; scnt[wi] += 1; }
    }
    let site_strain = (0..v_count).map(|i| if scnt[i] > 0 { ssum[i] / (scnt[i] as f64) } else { 0.0 }).collect();
    let radial_buckling = if is_open_sheet {
        let z_mean = if v_count > 0 { xyz.iter().map(|p| p[2]).sum::<f64>() / (v_count as f64) } else { 0.0 };
        xyz.iter().map(|p| (p[2] - z_mean) / l_mean.max(1e-12)).collect()
    } else { r_norms.iter().map(|&r| (r - r_mean) / r_mean.max(1e-12)).collect() };

    Ok(SurfaceElasticResult {
        xyz, cover_xyz, stretch_energy, bend_energy, total_energy: stretch_energy + bend_energy,
        r_mean, fvk_gamma, asphericity, bond_strain, site_strain, radial_buckling, converged, n_iter,
    })
}

/// Relax a `SurfaceMesh` in 3D under flat-chart spring stretching and orientation-safe hinge bending.
pub fn relax_surface_3d(mesh: &SurfaceMesh, kappa_b: f64, max_iters: usize) -> Result<SurfaceElasticResult, String> {
    if !kappa_b.is_finite() || kappa_b < 0.0 { return Err("kappa_b must be finite and non-negative".into()); }
    let v_count = mesh.sites.len();
    let is_proj = mesh.cover.is_some() && !mesh.orientable && mesh.chi == 1;
    let mut bmap: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut bvec: HashMap<(usize, usize), [f64; 2]> = HashMap::new();
    let (mut x0, faces_use, uv_opt, bond_edge_idx): (_, _, _, Vec<usize>);

    if is_proj {
        let cov = mesh.cover.as_ref().unwrap();
        if cov.xyz.len() != 2 * v_count { return Err(format!("projective cover xyz len {} != 2*V ({})", cov.xyz.len(), 2 * v_count)); }
        let (mut base_l0, mut base_vec) = (HashMap::new(), HashMap::new());
        for &(a, b, v) in &mesh.bonds {
            let (ua, ub) = (a % v_count, b % v_count);
            base_l0.entry((ua.min(ub), ua.max(ub))).or_insert_with(|| v[0].hypot(v[1]));
            base_vec.insert((ua, ub), v); base_vec.insert((ub, ua), [-v[0], -v[1]]);
        }
        for (idx, &(ca, cb)) in cov.bonds.iter().enumerate() {
            let (ua, ub) = (ca % v_count, cb % v_count);
            let (l0, vo) = match mesh.bonds.get(idx / 2) {
                Some(&(a, b, v)) if ua == a % v_count && ub == b % v_count => (v[0].hypot(v[1]), Some(v)),
                Some(&(a, b, v)) if ua == b % v_count && ub == a % v_count => (v[0].hypot(v[1]), Some([-v[0], -v[1]])),
                _ => (*base_l0.get(&(ua.min(ub), ua.max(ub))).unwrap_or(&1.0), base_vec.get(&(ua, ub)).copied()),
            };
            bmap.entry((ca.min(cb), ca.max(cb))).or_insert(l0);
            if let Some(vv) = vo { bvec.insert((ca, cb), vv); bvec.insert((cb, ca), [-vv[0], -vv[1]]); }
        }
        let e_idx: HashMap<(usize, usize), usize> = bmap.keys().enumerate().map(|(i, &e)| (e, i)).collect();
        bond_edge_idx = mesh.bonds.iter().enumerate().map(|(i, &(a, b, _))| {
            let (ua, ub) = (a % v_count, b % v_count);
            let same = |&(ca, cb): &(usize, usize)| (ca % v_count == ua && cb % v_count == ub) || (ca % v_count == ub && cb % v_count == ua);
            let pair = cov.bonds.get(2 * i).copied().filter(same).unwrap_or_else(|| cov.bonds.iter().copied().find(same).unwrap_or((a, b)));
            e_idx[&(pair.0.min(pair.1), pair.0.max(pair.1))]
        }).collect();
        (faces_use, uv_opt, x0) = (&cov.faces, None, cov.xyz.clone());
    } else {
        for &(a, b, v) in &mesh.bonds {
            bmap.entry((a.min(b), a.max(b))).or_insert_with(|| v[0].hypot(v[1]));
            bvec.insert((a, b), v); bvec.insert((b, a), [-v[0], -v[1]]);
        }
        let e_idx: HashMap<(usize, usize), usize> = bmap.keys().enumerate().map(|(i, &e)| (e, i)).collect();
        bond_edge_idx = mesh.bonds.iter().map(|&(a, b, _)| e_idx[&(a.min(b), a.max(b))]).collect();
        (faces_use, uv_opt) = (&mesh.faces, (mesh.uv.len() == v_count).then_some(mesh.uv.as_slice()));
        x0 = if mesh.xyz.len() == v_count { mesh.xyz.clone() }
            else if mesh.uv.len() == v_count { mesh.uv.iter().map(|&[u, v]| [u, v, 0.05 * u.sin() * v.cos()]).collect() }
            else { return Err(format!("mesh has {} sites but xyz len {} and uv len {}", v_count, mesh.xyz.len(), mesh.uv.len())); };
    }

    let spoke_lengths = face_spoke_lengths(faces_use, &bmap, Some(&bvec), uv_opt);
    let (edges, edge_l0): (Vec<_>, Vec<_>) = (bmap.keys().copied().collect(), bmap.values().copied().collect());
    if !edges.is_empty() && !x0.is_empty() {
        let mean_l0 = edge_l0.iter().sum::<f64>() / (edge_l0.len() as f64);
        let m_init = edges.iter().map(|&(u, v)| norm3(sub3(x0[u], x0[v]))).sum::<f64>() / (edges.len() as f64);
        if m_init > 1e-12 && mean_l0 > 0.0 && (m_init - mean_l0).abs() / mean_l0 > 1e-3 {
            let s = mean_l0 / m_init;
            for p in &mut x0 { *p = [p[0] * s, p[1] * s, p[2] * s]; }
        }
    }
    relax_indexed_elastic(v_count, &x0, &edges, &edge_l0, &bond_edge_idx, faces_use, &spoke_lengths, kappa_b, max_iters, is_proj, false)
}
