//! Shared 2D cell geometry, flat torus quotient, and figure-8 Klein bottle arclength helpers.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::types::{Cell2D, DomainEdge2D, SurfaceMesh};

pub(crate) type Dart2D = (usize, usize, i32, i32);

pub(crate) struct CellGeom {
    pub a: [[f64; 2]; 2],
    pub a_inv: [[f64; 2]; 2],
    pub frac_basis: Vec<[f64; 2]>,
    pub bonds: Vec<Dart2D>,
}

impl CellGeom {
    pub fn from_cell(cell: &Cell2D) -> Result<Self, String> {
        if cell.basis.is_empty() {
            return Err("cell has no basis sites".to_string());
        }
        let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
        let a_inv = mat2_inv(a).ok_or_else(|| "cell vectors are linearly dependent".to_string())?;
        let m = cell.basis.len();
        let frac_basis = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
        let mut bonds = Vec::with_capacity(cell.bonds.len());
        for b in &cell.bonds {
            if b.i >= m || b.j >= m {
                return Err("bond site index out of range".to_string());
            }
            bonds.push((b.i, b.j, b.n1, b.n2));
        }
        Ok(Self { a, a_inv, frac_basis, bonds })
    }
}

#[inline]
pub(crate) fn add2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
pub(crate) fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
pub(crate) fn scale2(s: f64, a: [f64; 2]) -> [f64; 2] {
    [s * a[0], s * a[1]]
}

#[inline]
pub(crate) fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

#[inline]
pub(crate) fn mat2_det(m: [[f64; 2]; 2]) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

#[inline]
pub(crate) fn mat2_inv(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = mat2_det(m);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([[m[1][1] * inv, -m[0][1] * inv], [-m[1][0] * inv, m[0][0] * inv]])
}

#[inline]
pub(crate) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

#[inline]
pub(crate) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

#[inline]
pub(crate) fn mat2i_vec(m: [[i32; 2]; 2], v: [i32; 2]) -> [i32; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

#[inline]
pub(crate) fn mat2i_fvec(m: [[i32; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] as f64 * v[0] + m[0][1] as f64 * v[1], m[1][0] as f64 * v[0] + m[1][1] as f64 * v[1]]
}

#[inline]
pub(crate) fn vec2_norm(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

#[inline]
pub(crate) fn vec2_dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub(crate) fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub(crate) fn dom_edge(p: [f64; 2], q: [f64; 2], label: &str, color_idx: usize) -> DomainEdge2D {
    DomainEdge2D { p, q, label: label.to_string(), color_idx }
}

pub(crate) fn median(mut vals: Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n % 2 == 1 { vals[n / 2] } else { 0.5 * (vals[n / 2 - 1] + vals[n / 2]) }
}

pub(crate) fn point_seg_dist(p: [f64; 2], s0: [f64; 2], s1: [f64; 2]) -> f64 {
    let d = sub2(s1, s0);
    let t = (vec2_dot(sub2(p, s0), d) / vec2_dot(d, d).max(1e-300)).clamp(0.0, 1.0);
    vec2_norm(sub2(p, add2(s0, scale2(t, d))))
}

pub(crate) fn seg_seg_dist(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> f64 {
    let da = sub2(a1, a0);
    let db = sub2(b1, b0);
    if cross2(da, sub2(b0, a0)) * cross2(da, sub2(b1, a0)) < 0.0
        && cross2(db, sub2(a0, b0)) * cross2(db, sub2(a1, b0)) < 0.0
    {
        return 0.0;
    }
    point_seg_dist(a0, b0, b1)
        .min(point_seg_dist(a1, b0, b1))
        .min(point_seg_dist(b0, a0, a1))
        .min(point_seg_dist(b1, a0, a1))
}

pub(crate) fn point_in_poly(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let (x, y) = (p[0], p[1]);
    let mut inside = false;
    let n = poly.len();
    for k in 0..n {
        let [x1, y1] = poly[k];
        let [x2, y2] = poly[(k + 1) % n];
        if (y1 - y2).abs() == 0.0 {
            continue;
        }
        if ((y1 > y) != (y2 > y)) && (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1)) {
            inside = !inside;
        }
    }
    inside
}

pub(crate) fn face_centre_2d(poly: &[[f64; 2]]) -> [f64; 2] {
    let n = poly.len();
    let (mut area2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    let (mut min_p, mut max_p) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut mean = [0.0, 0.0];
    for k in 0..n {
        let p = poly[k];
        let pn = poly[(k + 1) % n];
        let cr = p[0] * pn[1] - pn[0] * p[1];
        area2 += cr;
        cx += (p[0] + pn[0]) * cr;
        cy += (p[1] + pn[1]) * cr;
        mean = add2(mean, p);
        min_p = [min_p[0].min(p[0]), min_p[1].min(p[1])];
        max_p = [max_p[0].max(p[0]), max_p[1].max(p[1])];
    }
    let c = if (0.5 * area2).abs() > 1e-12 {
        [cx / (3.0 * area2), cy / (3.0 * area2)]
    } else {
        scale2(1.0 / (n as f64), mean)
    };
    let bdist = |pt: [f64; 2]| -> f64 {
        (0..n)
            .map(|k| point_seg_dist(pt, poly[k], poly[(k + 1) % n]))
            .fold(f64::INFINITY, f64::min)
    };
    let mut best = c;
    let mut dbest = -1.0_f64;
    for iy in 0..61 {
        let py = min_p[1] + (max_p[1] - min_p[1]) * (iy as f64) / 60.0;
        for ix in 0..61 {
            let pt = [min_p[0] + (max_p[0] - min_p[0]) * (ix as f64) / 60.0, py];
            if point_in_poly(pt, poly) {
                let d = bdist(pt);
                if d > dbest {
                    dbest = d;
                    best = pt;
                }
            }
        }
    }
    if dbest >= 0.0 && point_in_poly(c, poly) && bdist(c) >= 0.5 * dbest {
        c
    } else {
        best
    }
}

pub(crate) fn apply_to_site(
    mf: [[i32; 2]; 2],
    t: [f64; 2],
    frac_basis: &[[f64; 2]],
    k: usize,
) -> Option<(usize, [i32; 2])> {
    let g = add2(mat2i_fvec(mf, frac_basis[k]), t);
    for (j, &fj) in frac_basis.iter().enumerate() {
        let (d0, d1) = (g[0] - fj[0], g[1] - fj[1]);
        let (r0, r1) = (d0.round_ties_even(), d1.round_ties_even());
        if (d0 - r0).abs() < 1e-6 && (d1 - r1).abs() < 1e-6 {
            return Some((j, [r0 as i32, r1 as i32]));
        }
    }
    None
}

pub(crate) fn plane_faces(
    cell: &Cell2D,
    geom: &CellGeom,
) -> Result<(Vec<Vec<Dart2D>>, Vec<usize>), String> {
    let m = cell.basis.len();
    let mut darts = Vec::with_capacity(geom.bonds.len() * 2);
    let mut index = HashMap::with_capacity(geom.bonds.len() * 2);
    for &(i, j, n1, n2) in &geom.bonds {
        for d in [(i, j, n1, n2), (j, i, -n1, -n2)] {
            if index.insert(d, darts.len()).is_some() {
                return Err("duplicate dart in cell".to_string());
            }
            darts.push(d);
        }
    }
    let rev: Vec<usize> = darts.iter().map(|&(i, j, n1, n2)| index[&(j, i, -n1, -n2)]).collect();
    let mut by_site: Vec<Vec<(f64, usize)>> = vec![Vec::new(); m];
    for (k, &(i, j, n1, n2)) in darts.iter().enumerate() {
        let v = sub2(add2(cell.basis[j], mat2_vec(geom.a, [n1 as f64, n2 as f64])), cell.basis[i]);
        by_site[i].push((v[1].atan2(v[0]), k));
    }
    let mut prv = vec![0usize; darts.len()];
    let mut deg = vec![0usize; m];
    for (i, lst) in by_site.iter_mut().enumerate() {
        if lst.is_empty() {
            return Err(format!("site {i} has no incident bonds"));
        }
        lst.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        deg[i] = lst.len();
        let len = lst.len();
        for idx in 0..len {
            prv[lst[idx].1] = lst[(idx + len - 1) % len].1;
        }
    }
    let mut visited = vec![false; darts.len()];
    let mut faces = Vec::new();
    for d0 in 0..darts.len() {
        if visited[d0] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut d = d0;
        while !visited[d] {
            visited[d] = true;
            cyc.push(darts[d]);
            d = prv[rev[d]];
        }
        if d != d0 {
            return Err("face walk did not close on its start".to_string());
        }
        faces.push(cyc);
    }
    Ok((faces, deg))
}

pub(crate) fn canon_cycle(cyc: &[usize]) -> Vec<usize> {
    let k = cyc.len();
    if k == 0 {
        return Vec::new();
    }
    let rev: Vec<usize> = cyc.iter().rev().copied().collect();
    (0..k)
        .map(|s| cyc[s..].iter().chain(&cyc[..s]).copied().collect::<Vec<_>>())
        .chain((0..k).map(|s| rev[s..].iter().chain(&rev[..s]).copied().collect::<Vec<_>>()))
        .min()
        .unwrap_or_default()
}

pub(crate) fn sample_bond_curves(
    uv: &[[f64; 2]],
    xyz: &[[f64; 3]],
    bonds: &[(usize, usize, [f64; 2])],
    map_fn: impl Fn([f64; 2]) -> [f64; 3],
) -> Vec<Vec<[f64; 3]>> {
    let k = if bonds.len() <= 1500 { 6 } else if bonds.len() <= 6000 { 3 } else { 1 };
    bonds
        .iter()
        .map(|&(a, b, v)| {
            if k <= 1 {
                return vec![xyz[a], xyz[b]];
            }
            (0..=k)
                .map(|j| match j {
                    0 => xyz[a],
                    _ if j == k => xyz[b],
                    _ => map_fn(add2(uv[a], scale2(j as f64 / k as f64, v))),
                })
                .collect()
        })
        .collect()
}

pub(crate) struct SubTorus {
    n_col: [[f64; 2]; 2],
    n_col_inv: [[f64; 2]; 2],
    pub labels: Vec<(usize, i32, i32)>,
    index: HashMap<(usize, i32, i32), usize>,
}

impl SubTorus {
    pub fn new(row0: [i32; 2], row1: [i32; 2], m_sites: usize) -> Result<Self, String> {
        let det = row0[0] * row1[1] - row0[1] * row1[0];
        if det == 0 {
            return Err("the two vectors must not be parallel".to_string());
        }
        let n_col = [[row0[0] as f64, row1[0] as f64], [row0[1] as f64, row1[1] as f64]];
        let n_col_inv = mat2_inv(n_col).ok_or_else(|| "singular torus sublattice".to_string())?;
        let lim = row0[0].abs() + row0[1].abs() + row1[0].abs() + row1[1].abs() + 1;
        let k = det.unsigned_abs() as usize;
        let mut cells = Vec::with_capacity(k);
        for a in -lim..=lim {
            for b in -lim..=lim {
                let f = mat2_vec(n_col_inv, [a as f64, b as f64]);
                if f[0] >= -1e-9 && f[1] >= -1e-9 && f[0] < 1.0 - 1e-9 && f[1] < 1.0 - 1e-9 {
                    cells.push((a, b));
                }
            }
        }
        if cells.len() != k {
            return Err(format!("torus cell count mismatch: {} vs {k}", cells.len()));
        }
        let mut labels = Vec::with_capacity(k * m_sites);
        let mut index = HashMap::with_capacity(k * m_sites);
        for &(a, b) in &cells {
            for i in 0..m_sites {
                index.insert((i, a, b), labels.len());
                labels.push((i, a, b));
            }
        }
        Ok(Self { n_col, n_col_inv, labels, index })
    }

    pub fn site(&self, i: usize, n: [i32; 2]) -> usize {
        let f = mat2_vec(self.n_col_inv, [n[0] as f64, n[1] as f64]);
        let r = mat2_vec(self.n_col, [f[0] - (f[0] + 1e-9).floor(), f[1] - (f[1] + 1e-9).floor()]);
        self.index[&(i, r[0].round_ties_even() as i32, r[1].round_ties_even() as i32)]
    }
}

pub(crate) struct QuotientRaw {
    pub sites: Vec<(usize, i32, i32, usize)>,
    pub uv: Vec<[f64; 2]>,
    pub bonds: Vec<(usize, usize, [f64; 2])>,
    pub faces: Vec<Vec<usize>>,
    pub plane_degree: Vec<usize>,
    pub edge_sites: Vec<usize>,
}

impl QuotientRaw {
    pub fn into_mesh(
        self,
        name: &str,
        flags: (bool, bool),
        uv: Vec<[f64; 2]>,
        domain: (Vec<[f64; 2]>, Vec<DomainEdge2D>),
        xyz_map: impl Fn([f64; 2]) -> [f64; 3],
    ) -> SurfaceMesh {
        let (orientable, closed) = flags;
        let (poly, edges) = domain;
        let xyz: Vec<[f64; 3]> = uv.iter().map(|&p| xyz_map(p)).collect();
        let bond_curves_3d = sample_bond_curves(&uv, &xyz, &self.bonds, xyz_map);
        SurfaceMesh {
            name: name.to_string(),
            sites: self.sites,
            bonds: self.bonds,
            faces: self.faces,
            plane_degree: self.plane_degree,
            edge_sites: self.edge_sites,
            chi: 0,
            orientable,
            closed,
            cones: Vec::new(),
            uv,
            domain_polygons: vec![poly],
            domain_edges: edges,
            slits: Vec::new(),
            xyz,
            bond_curves_3d,
            cover: None,
        }
    }
}

pub(crate) fn sublattice_complement(a_mat: [[f64; 2]; 2], c: [i32; 2]) -> [i32; 2] {
    let g = gcd_i32(c[0], c[1]).max(1);
    let ac = mat2_vec(a_mat, [c[0] as f64, c[1] as f64]);
    let norm_ac = vec2_norm(ac);
    let mut best: Option<((i64, f64), [i32; 2])> = None;
    for a in -6..=6 {
        for b in -6..=6 {
            let det = a * c[1] - b * c[0];
            if det == 0 || det.abs() != g {
                continue;
            }
            let v = mat2_vec(a_mat, [a as f64, b as f64]);
            let nv = vec2_norm(v);
            let cosang = (vec2_dot(v, ac) / (nv * norm_ac)).abs();
            let key = ((cosang * 1e6).round_ties_even() as i64, nv);
            if best.as_ref().is_none_or(|(bk, _)| key.0 < bk.0 || (key.0 == bk.0 && key.1 < bk.1)) {
                best = Some((key, [a, b]));
            }
        }
    }
    best.map(|(_, v)| v).unwrap_or([0, 1])
}

struct Arc8Table {
    v8: [f64; 2001],
    s8: [f64; 2001],
    p8: f64,
}

fn arc8_table() -> &'static Arc8Table {
    static TABLE: OnceLock<Arc8Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        let (mut v8, mut s8) = ([0.0; 2001], [0.0; 2001]);
        for (k, vk) in v8.iter_mut().enumerate() {
            *vk = 2.0 * PI * (k as f64) / 2000.0;
        }
        for k in 1..2001 {
            s8[k] = s8[k - 1]
                + (v8[k].sin() - v8[k - 1].sin()).hypot((2.0 * v8[k]).sin() - (2.0 * v8[k - 1]).sin());
        }
        let p8 = s8[2000];
        for sk in &mut s8 {
            *sk = 2.0 * PI * (*sk) / p8;
        }
        Arc8Table { v8, s8, p8 }
    })
}

pub(crate) fn arc8_perimeter() -> f64 {
    arc8_table().p8
}

pub(crate) fn arc8_eval(x: f64) -> f64 {
    let tbl = arc8_table();
    let two_pi = 2.0 * PI;
    let k = (x / two_pi).floor();
    let x0 = (x - two_pi * k).clamp(0.0, two_pi);
    let idx = tbl.s8.partition_point(|&s| s < x0);
    let v0 = match idx {
        0 => 0.0,
        2001.. => two_pi,
        _ => {
            let (s_lo, s_hi) = (tbl.s8[idx - 1], tbl.s8[idx]);
            let frac = if (s_hi - s_lo).abs() > 1e-15 { (x0 - s_lo) / (s_hi - s_lo) } else { 0.0 };
            tbl.v8[idx - 1] + frac * (tbl.v8[idx] - tbl.v8[idx - 1])
        }
    };
    v0 + two_pi * k
}
