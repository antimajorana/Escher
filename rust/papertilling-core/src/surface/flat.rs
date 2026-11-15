//! Flat quotient surfaces: torus, cylinder, Klein bottle, and Möbius band (`flat.py`).

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use crate::surface::checks::check_surface;
use crate::surface::quotient::{
    add2, apply_to_site, arc8_eval, arc8_perimeter, canon_cycle, dom_edge, gcd_i32, mat2_det,
    mat2_inv, mat2_mul, mat2_vec, mat2i_fvec, mat2i_vec, plane_faces, scale2, sub2,
    sublattice_complement, vec2_dot, vec2_norm, CellGeom, Dart2D, QuotientRaw, SubTorus,
};
use crate::types::{Cell2D, SurfaceMesh};

const DEFAULT_SITES: f64 = 220.0;
const K8_HOLE_RATIO: f64 = 2.0;

const HEX_PRIO: [[[i32; 2]; 2]; 6] = [
    [[0, -1], [-1, 0]], [[-1, 1], [0, 1]], [[1, 0], [1, -1]],
    [[0, 1], [1, 0]], [[1, -1], [0, -1]], [[-1, 0], [-1, 1]],
];
const STD_PRIO: [[[i32; 2]; 2]; 4] = [
    [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
];

/// Mirror or glide reflection class of a 2D periodic cell usable for non-orientable flat surfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct GlideAxis {
    pub m: [[i32; 2]; 2],
    pub t: [f64; 2],
    pub tau: [i32; 2],
    pub along: [i32; 2],
    pub across: [i32; 2],
    pub half: [f64; 2],
    pub length: f64,
    pub mirror: bool,
}

fn odd_int(x: f64) -> i32 {
    let n = (x.round_ties_even() as i32).max(1);
    if n % 2 != 0 { n } else { n + 1 }
}

fn wrap01(x: f64) -> f64 {
    let w = x - (x + 1e-7).floor();
    if w.abs() < 1e-7 { 0.0 } else { w }
}

/// Discover all orientation-reversing mirror/glide classes of `cell` (matching `flat.py::glides`).
pub fn glides(cell: &Cell2D) -> Vec<GlideAxis> {
    let Ok(geom) = CellGeom::from_cell(cell) else { return Vec::new() };
    let mut dart_set: HashSet<Dart2D> = HashSet::with_capacity(geom.bonds.len() * 2);
    for &(i, j, n1, n2) in &geom.bonds {
        dart_set.insert((i, j, n1, n2));
        dart_set.insert((j, i, -n1, -n2));
    }
    let (g00, g01, g11) = (vec2_dot(cell.a1, cell.a1), vec2_dot(cell.a1, cell.a2), vec2_dot(cell.a2, cell.a2));
    let tol_g = 1e-5 * (g00 + g11);
    let (l1, l2) = (vec2_norm(cell.a1), vec2_norm(cell.a2));
    let is_hex = (l1 - l2).abs() <= 1e-5 * l1.max(l2) && (g01 / (l1 * l2) + 0.5).abs() <= 1e-5;
    let prio: &[[[i32; 2]; 2]] = if is_hex { &HEX_PRIO } else { &STD_PRIO };

    let mut cands: Vec<(usize, [[i32; 2]; 2])> = Vec::new();
    for m00 in -2..=2 {
        for m01 in -2..=2 {
            for m10 in -2..=2 {
                for m11 in -2..=2 {
                    if m00 * m11 - m01 * m10 != -1 {
                        continue;
                    }
                    let mf = [[m00, m01], [m10, m11]];
                    let (c0, c1) = (mat2_vec(geom.a, [m00 as f64, m10 as f64]), mat2_vec(geom.a, [m01 as f64, m11 as f64]));
                    if (vec2_dot(c0, c0) - g00).abs() <= tol_g
                        && (vec2_dot(c0, c1) - g01).abs() <= tol_g
                        && (vec2_dot(c1, c1) - g11).abs() <= tol_g
                    {
                        cands.push((prio.iter().position(|&p| p == mf).unwrap_or(100), mf));
                    }
                }
            }
        }
    }
    cands.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut vecs: Vec<[i32; 2]> = (-4..=4)
        .flat_map(|a| (-4..=4).filter_map(move |b| (a != 0 || b != 0).then_some([a, b])))
        .collect();
    vecs.sort_by(|u, v| {
        vec2_norm(mat2_vec(geom.a, [u[0] as f64, u[1] as f64]))
            .total_cmp(&vec2_norm(mat2_vec(geom.a, [v[0] as f64, v[1] as f64])))
    });

    let m_sites = geom.frac_basis.len();
    let mut out = Vec::new();
    for (_, mf) in cands {
        let mf0 = mat2i_fvec(mf, geom.frac_basis[0]);
        let found_t = (0..m_sites).find_map(|j0| {
            let t = [wrap01(geom.frac_basis[j0][0] - mf0[0]), wrap01(geom.frac_basis[j0][1] - mf0[1])];
            let smap: Option<Vec<_>> = (0..m_sites)
                .map(|k| apply_to_site(mf, t, &geom.frac_basis, k))
                .collect();
            let smap = smap?;
            geom.bonds
                .iter()
                .all(|&(i, j, n1, n2)| {
                    let ((ji, ci), (jj, cj)) = (smap[i], smap[j]);
                    let mn = mat2i_vec(mf, [n1, n2]);
                    dart_set.contains(&(ji, jj, mn[0] + cj[0] - ci[0], mn[1] + cj[1] - ci[1]))
                })
                .then_some(t)
        });
        let Some(mut t) = found_t else { continue };
        let Some(&along) = vecs.iter().find(|&&v| mat2i_vec(mf, v) == v) else { continue };
        let Some(&across) = vecs.iter().find(|&&v| mat2i_vec(mf, v) == [-v[0], -v[1]]) else { continue };
        let mut tau_f = add2(mat2i_fvec(mf, t), t);
        let along_c = mat2_vec(geom.a, [along[0] as f64, along[1] as f64]);
        let alpha = vec2_dot(mat2_vec(geom.a, tau_f), along_c) / vec2_dot(along_c, along_c);
        if alpha.abs() >= 1.5 {
            let k_even = (0.5 * alpha).round_ties_even();
            t = sub2(t, [k_even * along[0] as f64, k_even * along[1] as f64]);
            tau_f = add2(mat2i_fvec(mf, t), t);
        }
        let mirror = tau_f[0].abs() < 1e-6 && tau_f[1].abs() < 1e-6;
        if mirror {
            t = add2(t, [along[0] as f64, along[1] as f64]);
            tau_f = add2(mat2i_fvec(mf, t), t);
        }
        let tau = [tau_f[0].round_ties_even() as i32, tau_f[1].round_ties_even() as i32];
        let tau_c = mat2_vec(geom.a, [tau[0] as f64, tau[1] as f64]);
        out.push(GlideAxis {
            m: mf, t, tau, along, across,
            half: scale2(0.5, tau_c), length: 0.5 * vec2_norm(tau_c), mirror,
        });
    }
    out
}

fn build_quotient(
    cell: &Cell2D,
    geom: &CellGeom,
    row0: [i32; 2],
    row1: [i32; 2],
    glide: Option<&GlideAxis>,
    keep: Option<&dyn Fn(usize, i32, i32) -> bool>,
    rep: Option<&dyn Fn(usize, i32, i32) -> bool>,
) -> Result<QuotientRaw, String> {
    let tor = SubTorus::new(row0, row1, cell.basis.len())?;
    let total = tor.labels.len();
    let mut orbit: Vec<usize> = (0..total).collect();
    if let Some(g) = glide {
        for (s, &(i, a, b)) in tor.labels.iter().enumerate() {
            let (j, c) = apply_to_site(g.m, g.t, &geom.frac_basis, i)
                .ok_or_else(|| "NOT FOUND: glide is not a symmetry of the cell".to_string())?;
            let ma = mat2i_vec(g.m, [a, b]);
            let s2 = tor.site(j, [ma[0] + c[0], ma[1] + c[1]]);
            if s2 == s {
                return Err("NOT FOUND: the glide fixes a site on this torus".to_string());
            }
            orbit[s] = s.min(s2);
        }
    }

    let kept: Vec<usize> = (0..total)
        .filter(|&s| keep.is_none_or(|kf| kf(tor.labels[s].0, tor.labels[s].1, tor.labels[s].2)))
        .collect();
    let kept_set: HashSet<usize> = kept.iter().copied().collect();
    let mut classes: Vec<usize> = kept.iter().map(|&s| orbit[s]).collect();
    classes.sort_unstable();
    classes.dedup();
    let cid: HashMap<usize, usize> = classes.iter().enumerate().map(|(k, &c)| (c, k)).collect();

    let mut reps: HashMap<usize, usize> = HashMap::with_capacity(classes.len());
    for &s in &kept {
        let c = orbit[s];
        let replace = reps.get(&c).is_none_or(|&prev| {
            rep.is_some_and(|rf| {
                let ((i, a, b), (ip, ap, bp)) = (tor.labels[s], tor.labels[prev]);
                rf(i, a, b) && !rf(ip, ap, bp)
            })
        });
        if replace {
            reps.insert(c, s);
        }
    }

    let mut sites = Vec::with_capacity(classes.len());
    let mut uv = Vec::with_capacity(classes.len());
    for &c in &classes {
        let (i, a, b) = tor.labels[reps[&c]];
        sites.push((i, a, b, 0));
        uv.push(add2(cell.basis[i], mat2_vec(geom.a, [a as f64, b as f64])));
    }

    let mc = glide.map(|g| {
        let mf_f = [[g.m[0][0] as f64, g.m[0][1] as f64], [g.m[1][0] as f64, g.m[1][1] as f64]];
        mat2_mul(geom.a, mat2_mul(mf_f, geom.a_inv))
    });

    let mut bonds = Vec::new();
    let mut seen_bonds: HashSet<(usize, usize)> = HashSet::new();
    for &s in &kept {
        let (i, a, b) = tor.labels[s];
        let n0 = [a, b];
        for &(p, q, n1, n2) in &geom.bonds {
            for (x, y, d) in [(p, q, [n1, n2]), (q, p, [-n1, -n2])] {
                if x != i || (x == y && d == [0, 0]) {
                    continue;
                }
                let n_end = [n0[0] + d[0], n0[1] + d[1]];
                if keep.is_some_and(|kf| !kf(y, n_end[0], n_end[1])) {
                    continue;
                }
                let s2 = tor.site(y, n_end);
                if !kept_set.contains(&s2) {
                    continue;
                }
                let mut key = if s <= s2 { (s, s2) } else { (s2, s) };
                if let Some(g) = glide {
                    let (gi, ci) = apply_to_site(g.m, g.t, &geom.frac_basis, i).unwrap();
                    let (gy, cy) = apply_to_site(g.m, g.t, &geom.frac_basis, y).unwrap();
                    let (m0, me) = (mat2i_vec(g.m, n0), mat2i_vec(g.m, n_end));
                    let gs1 = tor.site(gi, [m0[0] + ci[0], m0[1] + ci[1]]);
                    let gs2 = tor.site(gy, [me[0] + cy[0], me[1] + cy[1]]);
                    key = key.min(if gs1 <= gs2 { (gs1, gs2) } else { (gs2, gs1) });
                }
                if !seen_bonds.insert(key) {
                    continue;
                }
                let p_start = add2(cell.basis[i], mat2_vec(geom.a, [n0[0] as f64, n0[1] as f64]));
                let p_end = add2(cell.basis[y], mat2_vec(geom.a, [n_end[0] as f64, n_end[1] as f64]));
                let v = sub2(p_end, p_start);
                let (u_id, v_id) = (cid[&orbit[s]], cid[&orbit[s2]]);
                if s == reps[&orbit[s]] {
                    bonds.push((u_id, v_id, v));
                } else if s2 == reps[&orbit[s2]] {
                    bonds.push((v_id, u_id, [-v[0], -v[1]]));
                } else {
                    bonds.push((u_id, v_id, mat2_vec(mc.unwrap(), v)));
                }
            }
        }
    }

    let (pf, deg_basis) = plane_faces(cell, geom)?;
    let bond_set: HashSet<(usize, usize)> =
        bonds.iter().map(|&(a, b, _)| if a <= b { (a, b) } else { (b, a) }).collect();
    let mut faces = Vec::new();
    let mut seen_faces: HashSet<Vec<usize>> = HashSet::new();
    for &s in &kept {
        let (i, a, b) = tor.labels[s];
        for cyc in &pf {
            if cyc[0].0 != i {
                continue;
            }
            let (mut n, mut ids, mut ok) = ([a, b], Vec::with_capacity(cyc.len()), true);
            for &(x, _, n1, n2) in cyc {
                if keep.is_some_and(|kf| !kf(x, n[0], n[1])) {
                    ok = false;
                    break;
                }
                let sx = tor.site(x, n);
                if !kept_set.contains(&sx) {
                    ok = false;
                    break;
                }
                ids.push(cid[&orbit[sx]]);
                n = [n[0] + n1, n[1] + n2];
            }
            if ok
                && (0..ids.len()).all(|k| {
                    let (u, v) = (ids[k], ids[(k + 1) % ids.len()]);
                    bond_set.contains(&if u <= v { (u, v) } else { (v, u) })
                })
                && seen_faces.insert(canon_cycle(&ids))
            {
                faces.push(ids);
            }
        }
    }

    let plane_degree: Vec<usize> = sites.iter().map(|&(i, _, _, _)| deg_basis[i]).collect();
    let mut edge_sites = Vec::new();
    if keep.is_some() {
        let mut dkept = vec![0usize; classes.len()];
        for &(a, b, _) in &bonds {
            dkept[a] += 1;
            dkept[b] += 1;
        }
        edge_sites = (0..classes.len()).filter(|&k| dkept[k] != plane_degree[k]).collect();
    }
    Ok(QuotientRaw { sites, uv, bonds, faces, plane_degree, edge_sites })
}

/// Build a flat torus quotient `R^2 / (Z u_vec + Z v_vec)` with 3D doughnut embedding.
pub fn torus(cell: &Cell2D, u_vec: [i32; 2], v_vec: [i32; 2]) -> Result<SurfaceMesh, String> {
    let geom = CellGeom::from_cell(cell)?;
    let q = build_quotient(cell, &geom, u_vec, v_vec, None, None, None)?;
    let v1 = mat2_vec(geom.a, [u_vec[0] as f64, u_vec[1] as f64]);
    let v2 = mat2_vec(geom.a, [v_vec[0] as f64, v_vec[1] as f64]);
    let v12 = add2(v1, v2);
    let m_par = [[v1[0], v2[0]], [v1[1], v2[1]]];
    let m_inv = mat2_inv(m_par).ok_or_else(|| "singular torus periods".to_string())?;
    let uv: Vec<[f64; 2]> = q
        .uv
        .iter()
        .map(|&p| {
            let f = mat2_vec(m_inv, p);
            mat2_vec(m_par, [f[0] - (f[0] + 1e-7).floor(), f[1] - (f[1] + 1e-7).floor()])
        })
        .collect();
    let (n1, n2) = (vec2_norm(v1), vec2_norm(v2));
    let big = usize::from(n1 < n2);
    let r_min = n1.min(n2) / (2.0 * PI);
    let r_maj = (n1.max(n2) / (2.0 * PI)).max(1.8 * r_min);
    let edges = vec![
        dom_edge([0.0, 0.0], v1, "1", 0), dom_edge(v2, v12, "1", 0),
        dom_edge([0.0, 0.0], v2, "2", 1), dom_edge(v1, v12, "2", 1),
    ];
    Ok(q.into_mesh("torus", (true, true), uv, (vec![[0.0, 0.0], v1, v12, v2], edges), |p| {
        let f = mat2_vec(m_inv, p);
        let (u, v) = (2.0 * PI * f[big], 2.0 * PI * f[1 - big]);
        let rad = r_maj + r_min * v.cos();
        [rad * u.cos(), rad * u.sin(), r_min * v.sin()]
    }))
}

/// Build an open flat cylinder rolled along `circumference` with axial span `height`.
pub fn cylinder(
    cell: &Cell2D,
    circumference: [i32; 2],
    height: [i32; 2],
) -> Result<SurfaceMesh, String> {
    let geom = CellGeom::from_cell(cell)?;
    if circumference[0] * height[1] - circumference[1] * height[0] == 0 {
        return Err("circumference and height vectors must not be parallel".to_string());
    }
    let cc = mat2_vec(geom.a, [circumference[0] as f64, circumference[1] as f64]);
    let w = vec2_norm(cc);
    let e1 = scale2(1.0 / w, cc);
    let e2 = [-e1[1], e1[0]];
    let mut h_vec = height;
    if vec2_dot(mat2_vec(geom.a, [h_vec[0] as f64, h_vec[1] as f64]), e2) < 0.0 {
        h_vec = [-h_vec[0], -h_vec[1]];
    }
    let h_tot = vec2_dot(mat2_vec(geom.a, [h_vec[0] as f64, h_vec[1] as f64]), e2).abs();
    let g_h = gcd_i32(h_vec[0], h_vec[1]).max(1);
    let row1 = [(g_h + 2) * (h_vec[0] / g_h), (g_h + 2) * (h_vec[1] / g_h)];
    let keep = |i: usize, a: i32, b: i32| -> bool {
        let z = vec2_dot(add2(cell.basis[i], mat2_vec(geom.a, [a as f64, b as f64])), e2);
        (-1e-9..h_tot - 1e-9).contains(&z)
    };
    let q = build_quotient(cell, &geom, circumference, row1, None, Some(&keep), None)?;
    let uv: Vec<[f64; 2]> = q
        .uv
        .iter()
        .map(|&p| add2(scale2(vec2_dot(p, e1).rem_euclid(w), e1), scale2(vec2_dot(p, e2), e2)))
        .collect();
    let top = scale2(h_tot, e2);
    let top_c = add2(cc, top);
    let rad = w / (2.0 * PI);
    let edges = vec![dom_edge([0.0, 0.0], top, "1", 0), dom_edge(cc, top_c, "1", 0)];
    Ok(q.into_mesh("cylinder", (true, false), uv, (vec![[0.0, 0.0], cc, top_c, top], edges), |p| {
        let th = 2.0 * PI * vec2_dot(p, e1) / w;
        [rad * th.cos(), rad * th.sin(), vec2_dot(p, e2)]
    }))
}

struct GlideFrame {
    g: GlideAxis,
    row0: [i32; 2],
    row1: [i32; 2],
    p0: [f64; 2],
    col0: [f64; 2],
    col1: [f64; 2],
    u_mat: [[f64; 2]; 2],
    u_inv: [[f64; 2]; 2],
}

impl GlideFrame {
    fn new(cell: &Cell2D, geom: &CellGeom, glide_idx: usize, n_along: i32, n_across: i32) -> Result<Self, String> {
        let gl = glides(cell);
        if gl.is_empty() {
            return Err("IMPOSSIBLE: lattice has no mirror or glide symmetry".to_string());
        }
        if n_along < 1 || n_along % 2 == 0 {
            return Err("n_along must be a positive odd integer".to_string());
        }
        let mut g = gl[glide_idx % gl.len()].clone();
        let k_shift = (n_along - 1) / 2;
        g.t = [g.t[0] + (k_shift * g.tau[0]) as f64, g.t[1] + (k_shift * g.tau[1]) as f64];
        let row0 = [n_along * g.tau[0], n_along * g.tau[1]];
        let row1 = [n_across * g.across[0], n_across * g.across[1]];
        let col0 = mat2_vec(geom.a, [row0[0] as f64, row0[1] as f64]);
        let col1 = mat2_vec(geom.a, [row1[0] as f64, row1[1] as f64]);
        g.half = scale2(0.5, col0);
        let p0 = scale2(0.5, sub2(mat2_vec(geom.a, g.t), g.half));
        let u_mat = [[col0[0], col1[0]], [col0[1], col1[1]]];
        let u_inv = mat2_inv(u_mat).ok_or_else(|| "singular glide frame".to_string())?;
        Ok(Self { g, row0, row1, p0, col0, col1, u_mat, u_inv })
    }

    fn coords(&self, cell: &Cell2D, geom: &CellGeom, i: usize, a: i32, b: i32) -> [f64; 2] {
        mat2_vec(self.u_inv, sub2(add2(cell.basis[i], mat2_vec(geom.a, [a as f64, b as f64])), self.p0))
    }

    fn wrap_uv(&self, raw_uv: &[[f64; 2]]) -> Vec<[f64; 2]> {
        raw_uv
            .iter()
            .map(|&p| {
                let f = mat2_vec(self.u_inv, sub2(p, self.p0));
                let fw = [(f[0] + 1e-7).rem_euclid(1.0) - 1e-7, (f[1] + 0.5).rem_euclid(1.0) - 0.5];
                add2(self.p0, mat2_vec(self.u_mat, fw))
            })
            .collect()
    }
}

/// Build a closed non-orientable flat Klein bottle with figure-8 3D immersion.
pub fn klein_bottle(cell: &Cell2D, glide_idx: usize, n_along: i32, n_across: i32) -> Result<SurfaceMesh, String> {
    if n_across < 1 {
        return Err("n_across must be >= 1".to_string());
    }
    let geom = CellGeom::from_cell(cell)?;
    let gf = GlideFrame::new(cell, &geom, glide_idx, n_along, n_across)?;
    let rep = |i: usize, a: i32, b: i32| (gf.coords(cell, &geom, i, a, b)[0] + 1e-7).rem_euclid(1.0) < 0.5 - 1e-7;
    let q = build_quotient(cell, &geom, gf.row0, gf.row1, Some(&gf.g), None, Some(&rep))?;
    let uv = gf.wrap_uv(&q.uv);
    let a_half = scale2(0.5, gf.col0);
    let lo = sub2(gf.p0, scale2(0.5, gf.col1));
    let (lo_a, lo_b, lo_ab) = (add2(lo, a_half), add2(lo, gf.col1), add2(add2(lo, a_half), gf.col1));
    let a8 = vec2_norm(gf.col1) / arc8_perimeter();
    let r_maj = (vec2_norm(gf.col0) / (4.0 * PI)).max(K8_HOLE_RATIO * a8);
    let edges = vec![
        dom_edge(lo, lo_a, "1", 0), dom_edge(lo_b, lo_ab, "1", 0),
        dom_edge(lo, lo_b, "2", 1), dom_edge(lo_ab, lo_a, "2", 1),
    ];
    Ok(q.into_mesh("Klein bottle", (false, true), uv, (vec![lo, lo_a, lo_ab, lo_b], edges), |p| {
        let qc = mat2_vec(gf.u_inv, sub2(p, gf.p0));
        let (u, v) = (4.0 * PI * qc[0], arc8_eval(2.0 * PI * qc[1]));
        let (cu2, su2, sv, s2v) = ((0.5 * u).cos(), (0.5 * u).sin(), v.sin(), (2.0 * v).sin());
        let w = r_maj + a8 * (cu2 * sv - su2 * s2v);
        [w * u.cos(), w * u.sin(), a8 * (su2 * sv + cu2 * s2v)]
    }))
}

/// Build an open non-orientable flat Möbius band with 3D twisted-strip embedding.
pub fn mobius_band(cell: &Cell2D, glide_idx: usize, half_turns: i32, width: i32) -> Result<SurfaceMesh, String> {
    if width < 1 {
        return Err("width must be >= 1".to_string());
    }
    let geom = CellGeom::from_cell(cell)?;
    let across = 2 * width + 4;
    let gf = GlideFrame::new(cell, &geom, glide_idx, half_turns, across)?;
    let hcell = 1.0 / (across as f64);
    let keep = |i: usize, a: i32, b: i32| {
        ((gf.coords(cell, &geom, i, a, b)[1] + 0.5).rem_euclid(1.0) - 0.5).abs() <= (width as f64) * hcell + 1e-9
    };
    let rep = |i: usize, a: i32, b: i32| (gf.coords(cell, &geom, i, a, b)[0] + 1e-7).rem_euclid(1.0) < 0.5 - 1e-7;
    let q = build_quotient(cell, &geom, gf.row0, gf.row1, Some(&gf.g), Some(&keep), Some(&rep))?;
    let uv = gf.wrap_uv(&q.uv);
    let a_half = scale2(0.5, gf.col0);
    let w_scale = (width as f64) * hcell;
    let wv = scale2(w_scale, gf.col1);
    let (c_lo, c_hi) = (sub2(gf.p0, wv), add2(gf.p0, wv));
    let (c_lo_a, c_hi_a) = (add2(c_lo, a_half), add2(c_hi, a_half));
    let u1_norm = vec2_norm(gf.col1);
    let r_maj = (vec2_norm(gf.col0) / (4.0 * PI)).max(1.6 * w_scale * u1_norm);
    let edges = vec![dom_edge(c_lo, c_hi, "1", 0), dom_edge(c_hi_a, c_lo_a, "1", 0)];
    Ok(q.into_mesh("Moebius band", (false, false), uv, (vec![c_lo, c_lo_a, c_hi_a, c_hi], edges), |p| {
        let qc = mat2_vec(gf.u_inv, sub2(p, gf.p0));
        let t = 4.0 * PI * qc[0];
        let s = ((qc[1] + 0.5).rem_euclid(1.0) - 0.5) * u1_norm;
        let rad = r_maj + s * (0.5 * t).cos();
        [rad * t.cos(), rad * t.sin(), s * (0.5 * t).sin()]
    }))
}

/// Build a torus at `flat.py` / `report.py` default sizes (`sites = 220`, retry `400`).
pub fn default_torus(cell: &Cell2D) -> Result<SurfaceMesh, String> {
    let geom = CellGeom::from_cell(cell)?;
    let tc = sublattice_complement(geom.a, [1, 0]);
    let (a1, tc_len, m) = (vec2_norm(cell.a1), vec2_norm(mat2_vec(geom.a, [tc[0] as f64, tc[1] as f64])), cell.basis.len() as f64);
    for s in [DEFAULT_SITES, 400.0] {
        let n2 = ((s * a1 / (m * 2.2 * tc_len)).sqrt().round_ties_even() as i32).max(2);
        let n1 = ((2.2 * (n2 as f64) * tc_len / a1).round_ties_even() as i32).max(3);
        if let Ok(mesh) = torus(cell, [n1, 0], [n2 * tc[0], n2 * tc[1]]) && check_surface(&mesh).ok {
            return Ok(mesh);
        }
    }
    Err("NOT FOUND: torus checks failed at tried sizes".to_string())
}

/// Build a cylinder at `flat.py` / `report.py` default sizes (`sites = 220`, retry `400`).
pub fn default_cylinder(cell: &Cell2D) -> Result<SurfaceMesh, String> {
    let geom = CellGeom::from_cell(cell)?;
    let (a1, m) = (vec2_norm(cell.a1), cell.basis.len() as f64);
    let hc = mat2_det(geom.a).abs() / a1;
    for s in [DEFAULT_SITES, 400.0] {
        let c_vec = [((s * hc / (0.38 * m * a1)).sqrt().round_ties_even() as i32).max(3), 0];
        let tv = sublattice_complement(geom.a, c_vec);
        let per_row = (c_vec[0] * tv[1] - c_vec[1] * tv[0]).abs().max(1) as f64;
        let len = ((s / (m * per_row)).round_ties_even() as i32).max(2);
        if let Ok(mesh) = cylinder(cell, c_vec, [len * tv[0], len * tv[1]]) && check_surface(&mesh).ok {
            return Ok(mesh);
        }
    }
    Err("NOT FOUND: cylinder checks failed at tried sizes".to_string())
}

fn try_glide_surfaces(
    cell: &Cell2D,
    what: &str,
    build_one: impl Fn(&CellGeom, &GlideAxis, usize, f64, f64) -> Result<SurfaceMesh, String>,
) -> Result<SurfaceMesh, String> {
    let gl = glides(cell);
    if gl.is_empty() {
        return Err("IMPOSSIBLE: lattice has no mirror or glide symmetry".to_string());
    }
    let geom = CellGeom::from_cell(cell)?;
    let sa = mat2_det(geom.a).abs() / (cell.basis.len() as f64);
    for (w, s) in (0..gl.len()).map(|w| (w, DEFAULT_SITES)).chain(std::iter::once((0, 400.0))) {
        if let Ok(mesh) = build_one(&geom, &gl[w], w, s, sa) && check_surface(&mesh).ok {
            return Ok(mesh);
        }
    }
    Err(format!("NOT FOUND: {what} refused at every setting tried"))
}

/// Build a Klein bottle trying all glide classes at `sites = 220` then `which = 0` at `sites = 400`.
pub fn default_klein_bottle(cell: &Cell2D) -> Result<SurfaceMesh, String> {
    let kappa = 2.0 * PI * K8_HOLE_RATIO / arc8_perimeter();
    try_glide_surfaces(cell, "Klein bottle", |geom, g, w, s, sa| {
        let ac = vec2_norm(mat2_vec(geom.a, [g.across[0] as f64, g.across[1] as f64]));
        let tau = vec2_norm(mat2_vec(geom.a, [g.tau[0] as f64, g.tau[1] as f64]));
        let n_across = (((s * sa / kappa).sqrt() / ac).round_ties_even() as i32).max(2);
        klein_bottle(cell, w, odd_int(2.0 * kappa * (n_across as f64) * ac / tau), n_across)
    })
}

/// Build a Möbius band trying all glide classes at `sites = 220` then `which = 0` at `sites = 400`.
pub fn default_mobius_band(cell: &Cell2D) -> Result<SurfaceMesh, String> {
    try_glide_surfaces(cell, "Moebius band", |geom, g, w, s, sa| {
        let ac = vec2_norm(mat2_vec(geom.a, [g.across[0] as f64, g.across[1] as f64]));
        let tau = vec2_norm(mat2_vec(geom.a, [g.tau[0] as f64, g.tau[1] as f64]));
        let width = ((((s * sa / (4.0 * PI * 2.4)).sqrt() / ac) + 0.25).round_ties_even() as i32).max(1);
        let h = (width as f64) * ac;
        let core = (s * sa / (2.0 * h)).clamp(2.0 * PI * 1.6 * h, 2.0 * PI * 4.0 * h);
        mobius_band(cell, w, odd_int(2.0 * core / tau), width)
    })
}
