//! Internal Euclidean plane symmetry, primitive-cell reduction, and unimodular matrix helpers
//! for `DefectCatalogue` and wall classification.

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::PI;

use crate::types::{Bond2D, Cell2D, FracOp2D, Frame2D, Isometry2};

pub(crate) fn round_even(x: f64) -> f64 {
    let r = x.round_ties_even();
    if r == 0.0 { 0.0 } else { r }
}
pub(crate) fn round_dec(x: f64, digits: i32) -> f64 {
    let f = 10.0_f64.powi(digits);
    let r = (x * f).round_ties_even() / f;
    if r == 0.0 { 0.0 } else { r }
}
pub(crate) fn py_mod(x: f64, m: f64) -> f64 { x.rem_euclid(m) + 0.0 }
pub(crate) fn dot(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[0] + a[1] * b[1] }
pub(crate) fn norm(a: [f64; 2]) -> f64 { a[0].hypot(a[1]) }
pub(crate) fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[1] - a[1] * b[0] }
pub(crate) fn rot90(u: [f64; 2]) -> [f64; 2] { [-u[1], u[0]] }

pub(crate) fn canon_dir(u: [f64; 2]) -> [f64; 2] {
    let l = norm(u);
    let u = [u[0] / l, u[1] / l];
    if u[1] < -1e-7 || (u[1].abs() < 1e-7 && u[0] < 0.0) { [-u[0] + 0.0, -u[1] + 0.0] } else { [u[0] + 0.0, u[1] + 0.0] }
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

pub(crate) fn np_allclose_vec(a: [f64; 2], b: [f64; 2], rtol: f64, atol: f64) -> bool {
    (a[0] - b[0]).abs() <= atol + rtol * b[0].abs() && (a[1] - b[1]).abs() <= atol + rtol * b[1].abs()
}
pub(crate) fn np_allclose_mat(a: [[f64; 2]; 2], b: [[f64; 2]; 2], rtol: f64, atol: f64) -> bool {
    np_allclose_vec(a[0], b[0], rtol, atol) && np_allclose_vec(a[1], b[1], rtol, atol)
}
pub(crate) fn mat2_cols(a1: [f64; 2], a2: [f64; 2]) -> [[f64; 2]; 2] { [[a1[0], a2[0]], [a1[1], a2[1]]] }
pub(crate) fn mat2_det(m: [[f64; 2]; 2]) -> f64 { m[0][0] * m[1][1] - m[0][1] * m[1][0] }
pub(crate) fn imat2_det(m: [[i32; 2]; 2]) -> i32 { m[0][0] * m[1][1] - m[0][1] * m[1][0] }
pub(crate) fn mat2_transpose(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] { [[m[0][0], m[1][0]], [m[0][1], m[1][1]]] }
pub(crate) fn mat2_inv(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let d = mat2_det(m);
    [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]
}
pub(crate) fn imat2_inv(m: [[i32; 2]; 2]) -> [[i32; 2]; 2] {
    let d = imat2_det(m);
    debug_assert!(d.abs() == 1);
    [[m[1][1] * d, -m[0][1] * d], [-m[1][0] * d, m[0][0] * d]]
}
pub(crate) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
     [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
}
pub(crate) fn imat2_mul(a: [[i32; 2]; 2], b: [[i32; 2]; 2]) -> [[i32; 2]; 2] {
    [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
     [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
}
pub(crate) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] { [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]] }
pub(crate) fn imat2_vec(m: [[i32; 2]; 2], v: [i32; 2]) -> [i32; 2] { [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]] }
pub(crate) fn imat2_to_f64(m: [[i32; 2]; 2]) -> [[f64; 2]; 2] { [[m[0][0] as f64, m[0][1] as f64], [m[1][0] as f64, m[1][1] as f64]] }
pub(crate) fn solve2(m: [[f64; 2]; 2], rhs: [f64; 2]) -> [f64; 2] { mat2_vec(mat2_inv(m), rhs) }

pub(crate) fn eigvec_max_sym(m: [[f64; 2]; 2]) -> [f64; 2] {
    let (a, b, d) = (m[0][0], 0.5 * (m[0][1] + m[1][0]), m[1][1]);
    let lam_min = 0.5 * (a + d) - (0.5 * (a - d)).hypot(b);
    let (c0, c1) = ([a - lam_min, b], [b, d - lam_min]);
    let v = if norm(c0) >= norm(c1) { c0 } else { c1 };
    let lv = norm(v);
    if lv < 1e-15 { [1.0, 0.0] } else { [v[0] / lv, v[1] / lv] }
}

pub(crate) fn rotation_about(centre: [f64; 2], angle_rad: f64) -> Isometry2 {
    let (s, c) = angle_rad.sin_cos();
    let m = [[c, -s], [s, c]];
    let mc = mat2_vec(m, centre);
    Isometry2 { m, t: [centre[0] - mc[0], centre[1] - mc[1]] }
}

pub(crate) fn reflection_across(point: [f64; 2], direction: [f64; 2], glide: f64) -> Isometry2 {
    let l = norm(direction);
    let u = [direction[0] / l, direction[1] / l];
    let m = [[2.0 * u[0] * u[0] - 1.0, 2.0 * u[0] * u[1]], [2.0 * u[1] * u[0], 2.0 * u[1] * u[1] - 1.0]];
    let mp = mat2_vec(m, point);
    Isometry2 { m, t: [point[0] - mp[0] + glide * u[0], point[1] - mp[1] + glide * u[1]] }
}

pub(crate) fn describe_isometry(iso: Isometry2) -> String {
    let fmt_pt = |v: [f64; 2]| format!("[{}, {}]", round_dec(v[0], 4), round_dec(v[1], 4));
    if iso.det() > 0.0 {
        if np_allclose_mat(iso.m, Isometry2::IDENTITY.m, 1e-5, 1e-8) {
            return format!("translation by {}", fmt_pt(iso.t));
        }
        let ang = iso.m[1][0].atan2(iso.m[0][0]).to_degrees();
        let c = solve2([[1.0 - iso.m[0][0], -iso.m[0][1]], [-iso.m[1][0], 1.0 - iso.m[1][1]]], iso.t);
        return format!("rotation by {:.0} degrees about {}", ang, fmt_pt(c));
    }
    let u = canon_dir(eigvec_max_sym(iso.m));
    let n = rot90(u);
    let tn = 0.5 * dot(iso.t, n);
    let tu = dot(iso.t, u);
    let shift = if tu.abs() > 1e-6 { format!(", then a shift {:.4} along it", tu) } else { String::new() };
    format!("reflection across the line through {} along {}{}", fmt_pt([n[0] * tn, n[1] * tn]), fmt_pt(u), shift)
}

pub(crate) fn holohedry(a_mat: [[f64; 2]; 2]) -> Vec<[[i32; 2]; 2]> {
    let gm = mat2_mul(mat2_transpose(a_mat), a_mat);
    let max_gm = gm[0][0].abs().max(gm[0][1].abs()).max(gm[1][0].abs()).max(gm[1][1].abs());
    let mut out = Vec::new();
    for idx in 0..625 {
        let m = [[(idx / 125) % 5 - 2, (idx / 25) % 5 - 2], [(idx / 5) % 5 - 2, idx % 5 - 2]];
        if imat2_det(m).abs() == 1 {
            let mf = imat2_to_f64(m);
            if np_allclose_mat(mat2_mul(mat2_mul(mat2_transpose(mf), gm), mf), gm, 1e-5, 1e-7 * max_gm) {
                out.push(m);
            }
        }
    }
    out
}

pub(crate) fn kernel_and_complement(d: [[i32; 2]; 2]) -> ([f64; 2], [f64; 2]) {
    let r = if d[0] != [0, 0] { d[0] } else { d[1] };
    let g = gcd_i32(r[1], -r[0]);
    let k = [r[1] / g, -r[0] / g];
    for w0 in -3..=3 {
        for w1 in -3..=3 {
            if k[0] * w1 - k[1] * w0 == 1 {
                return ([k[0] as f64, k[1] as f64], [w0 as f64, w1 as f64]);
            }
        }
    }
    panic!("no unimodular complement for reflection kernel");
}

pub(crate) fn orbits<T, Op, Im, FImg, FSame>(items: &[T], ops: &[Op], mut image: FImg, mut same: FSame) -> Vec<Vec<usize>>
where
    FImg: FnMut(&T, &Op) -> Im,
    FSame: FnMut(&Im, &T) -> bool,
{
    let mut parent: Vec<usize> = (0..items.len()).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i { p[i] = p[p[i]]; i = p[i]; }
        i
    }
    for (i, it) in items.iter().enumerate() {
        for g in ops {
            let im = image(it, g);
            if let Some(j) = items.iter().position(|other| same(&im, other)) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b { parent[a.max(b)] = a.min(b); }
            }
        }
    }
    let (mut root_idx, mut out) = (HashMap::<usize, usize>::new(), Vec::<Vec<usize>>::new());
    for i in 0..items.len() {
        let r = find(&mut parent, i);
        if let Some(&pos) = root_idx.get(&r) { out[pos].push(i); } else { root_idx.insert(r, out.len()); out.push(vec![i]); }
    }
    out
}

pub(crate) type SiteKey = (i32, i32, usize);

#[derive(Clone, Debug)]
pub(crate) struct PerfectLattice {
    pub cell: Cell2D,
    pub a: [[f64; 2]; 2],
    pub ainv: [[f64; 2]; 2],
    pub rule: Vec<Vec<(i32, i32, usize, usize)>>,
}

impl PerfectLattice {
    pub fn new(cell: Cell2D) -> Self {
        let a = mat2_cols(cell.a1, cell.a2);
        let mut rule = vec![Vec::new(); cell.basis.len()];
        for b in &cell.bonds {
            rule[b.i].push((b.n1, b.n2, b.j, b.kind));
            rule[b.j].push((-b.n1, -b.n2, b.i, b.kind));
        }
        Self { cell, a, ainv: mat2_inv(a), rule }
    }
    pub fn pos(&self, key: SiteKey) -> [f64; 2] {
        let (sh, b) = (mat2_vec(self.a, [key.0 as f64, key.1 as f64]), self.cell.basis[key.2]);
        [sh[0] + b[0], sh[1] + b[1]]
    }
    pub fn locate(&self, p: [f64; 2]) -> Option<SiteKey> { locate_site(p, self.ainv, &self.cell.basis, 1e-6) }
    pub fn nbrs(&self, key: SiteKey) -> Vec<(SiteKey, usize)> {
        self.rule[key.2].iter().map(|&(d1, d2, s2, t)| ((key.0 + d1, key.1 + d2, s2), t)).collect()
    }
    pub fn sites_in_disk(&self, centre: [f64; 2], r: f64) -> Vec<SiteKey> {
        let fc = mat2_vec(self.ainv, centre);
        let n0 = [round_even(fc[0]) as i32, round_even(fc[1]) as i32];
        let k = (r * norm(self.cell.a1).max(norm(self.cell.a2)) / mat2_det(self.a).abs()).ceil() as i32 + 2;
        let mut out = Vec::new();
        for n1 in (n0[0] - k)..=(n0[0] + k) {
            for n2 in (n0[1] - k)..=(n0[1] + k) {
                for s in 0..self.cell.basis.len() {
                    let p = self.pos((n1, n2, s));
                    if (p[0] - centre[0]).hypot(p[1] - centre[1]) <= r + 1e-9 { out.push((n1, n2, s)); }
                }
            }
        }
        out
    }
    pub fn is_symmetry(&self, g: Isometry2) -> bool {
        for key in self.sites_in_disk([0.0, 0.0], 3.0 * norm(self.cell.a1)) {
            let Some(img) = self.locate(g.apply(self.pos(key))) else { return false };
            let mut mine = BTreeSet::new();
            for (k2, _) in self.nbrs(key) {
                let Some(lk2) = self.locate(g.apply(self.pos(k2))) else { return false };
                mine.insert(lk2);
            }
            if mine != self.nbrs(img).into_iter().map(|(k2, _)| k2).collect() { return false; }
        }
        true
    }
}

fn locate_site(p: [f64; 2], ainv: [[f64; 2]; 2], basis: &[[f64; 2]], tol: f64) -> Option<SiteKey> {
    basis.iter().enumerate().find_map(|(s, b)| {
        let f = mat2_vec(ainv, [p[0] - b[0], p[1] - b[1]]);
        let (r0, r1) = (round_even(f[0]), round_even(f[1]));
        ((f[0] - r0).abs() < tol && (f[1] - r1).abs() < tol).then_some((r0 as i32, r1 as i32, s))
    })
}

fn frac_mod(f: [f64; 2]) -> [f64; 2] {
    let mut g = [f[0] - (f[0] + 1e-6).floor(), f[1] - (f[1] + 1e-6).floor()];
    if (g[0] - 1.0).abs() < 1e-6 { g[0] = 0.0; }
    if (g[1] - 1.0).abs() < 1e-6 { g[1] = 0.0; }
    [g[0] + 0.0, g[1] + 0.0]
}

fn reduce_basis(mut b1: [f64; 2], mut b2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    for _ in 0..200 {
        if dot(b2, b2) < dot(b1, b1) - 1e-12 { std::mem::swap(&mut b1, &mut b2); }
        let k = round_even(dot(b1, b2) / dot(b1, b1));
        if k == 0.0 { break; }
        b2 = [b2[0] - k * b1[0], b2[1] - k * b1[1]];
    }
    if cross2(b1, b2) < 0.0 { b2 = [-b2[0], -b2[1]]; }
    (b1, b2)
}

fn rebase_cell(cell: &Cell2D, b1: [f64; 2], b2: [f64; 2]) -> Cell2D {
    let (a, b, binv) = (mat2_cols(cell.a1, cell.a2), mat2_cols(b1, b2), mat2_inv(mat2_cols(b1, b2)));
    let mut new_basis: Vec<[f64; 2]> = Vec::new();
    for &p in &cell.basis {
        let q = mat2_vec(b, frac_mod(mat2_vec(binv, p)));
        if !new_basis.iter().any(|&r| (q[0] - r[0]).hypot(q[1] - r[1]) < 1e-6) { new_basis.push(q); }
    }
    let mut bonds_set = BTreeSet::new();
    for bd in &cell.bonds {
        let sh = mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]);
        let (ni0, ni1, si) = locate_site(cell.basis[bd.i], binv, &new_basis, 1e-6).expect("rebase site i");
        let (nj0, nj1, sj) = locate_site([cell.basis[bd.j][0] + sh[0], cell.basis[bd.j][1] + sh[1]], binv, &new_basis, 1e-6).expect("rebase site j");
        let (dn0, dn1) = (nj0 - ni0, nj1 - ni1);
        bonds_set.insert((si, sj, dn0, dn1, bd.kind).min((sj, si, -dn0, -dn1, bd.kind)));
    }
    Cell2D { a1: b1, a2: b2, basis: new_basis, bonds: bonds_set.into_iter().map(|(i, j, n1, n2, kind)| Bond2D { i, j, n1, n2, kind }).collect() }
}

pub(crate) fn symmetry_ops(cell: &Cell2D, only_identity: bool, use_types: bool) -> Vec<FracOp2D> {
    let (a, ainv) = (mat2_cols(cell.a1, cell.a2), mat2_inv(mat2_cols(cell.a1, cell.a2)));
    let f_coords: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(ainv, p)).collect();
    if f_coords.is_empty() { return Vec::new(); }
    let mut bond_lookup = BTreeSet::new();
    for b in &cell.bonds {
        let t = if use_types { b.kind } else { 0 };
        bond_lookup.insert((b.i, b.j, b.n1, b.n2, t));
        bond_lookup.insert((b.j, b.i, -b.n1, -b.n2, t));
    }
    let mut mats = vec![[[1.0, 0.0], [0.0, 1.0]]];
    if !only_identity {
        mats = (0..12).map(|k| { let (s, c) = ((k as f64) * PI / 6.0).sin_cos(); [[c, -s], [s, c]] }).collect();
        let mut angles_i64 = BTreeSet::new();
        for q in 0..49 {
            let v = mat2_vec(a, [(q / 7 - 3) as f64, (q % 7 - 3) as f64]);
            if norm(v) > 1e-6 {
                let ang = py_mod(v[1].atan2(v[0]), PI);
                angles_i64.insert((ang * 1e9).round_ties_even() as i64);
                angles_i64.insert((py_mod(ang + 0.5 * PI, PI) * 1e9).round_ties_even() as i64);
            }
        }
        for ai in angles_i64 {
            let (s, c) = (2.0 * (ai as f64) * 1e-9).sin_cos();
            mats.push([[c, s], [s, -c]]);
        }
    }
    let (mut ops, n_sites) = (Vec::<FracOp2D>::new(), f_coords.len());
    for m in mats {
        let mf_f = mat2_mul(mat2_mul(ainv, m), a);
        let mf = [[round_even(mf_f[0][0]) as i32, round_even(mf_f[0][1]) as i32], [round_even(mf_f[1][0]) as i32, round_even(mf_f[1][1]) as i32]];
        if (mf_f[0][0] - mf[0][0] as f64).abs() > 1e-6 || (mf_f[0][1] - mf[0][1] as f64).abs() > 1e-6
            || (mf_f[1][0] - mf[1][0] as f64).abs() > 1e-6 || (mf_f[1][1] - mf[1][1] as f64).abs() > 1e-6 { continue; }
        let mf_pts: Vec<[f64; 2]> = f_coords.iter().map(|&p| mat2_vec(imat2_to_f64(mf), p)).collect();
        for fk0 in &f_coords {
            let t = frac_mod([fk0[0] - mf_pts[0][0], fk0[1] - mf_pts[0][1]]);
            let (mut perm, mut shift, mut ok) = (vec![0; n_sites], vec![[0i32; 2]; n_sites], true);
            for i in 0..n_sites {
                let tgt = [mf_pts[i][0] + t[0], mf_pts[i][1] + t[1]];
                if let Some((k, sh)) = f_coords.iter().enumerate().find_map(|(k, &fk)| {
                    let (d0, d1) = (tgt[0] - fk[0], tgt[1] - fk[1]);
                    let (r0, r1) = (round_even(d0), round_even(d1));
                    ((d0 - r0).abs() < 1e-6 && (d1 - r1).abs() < 1e-6).then_some((k, [r0 as i32, r1 as i32]))
                }) { perm[i] = k; shift[i] = sh; } else { ok = false; break; }
            }
            if ok && cell.bonds.iter().all(|b| {
                let mn = imat2_vec(mf, [b.n1, b.n2]);
                bond_lookup.contains(&(perm[b.i], perm[b.j], mn[0] + shift[b.j][0] - shift[b.i][0], mn[1] + shift[b.j][1] - shift[b.i][1], if use_types { b.kind } else { 0 }))
            }) && !ops.iter().any(|op| op.mf == mf && np_allclose_vec(op.t, t, 1e-5, 1e-8)) {
                ops.push(FracOp2D { mf, t });
            }
        }
    }
    ops
}

pub(crate) fn primitive_cell(cell: &Cell2D, use_types: bool) -> Cell2D {
    let ts: Vec<[f64; 2]> = symmetry_ops(cell, true, use_types).into_iter().map(|op| op.t).collect();
    if ts.len() == 1 {
        let (b1, b2) = reduce_basis(cell.a1, cell.a2);
        return if np_allclose_vec(b1, cell.a1, 1e-5, 1e-8) && np_allclose_vec(b2, cell.a2, 1e-5, 1e-8) { cell.clone() } else { rebase_cell(cell, b1, b2) };
    }
    let a = mat2_cols(cell.a1, cell.a2);
    let mut cands: Vec<[f64; 2]> = ts.iter().flat_map(|t| (0..25).map(move |q| mat2_vec(a, [t[0] + (q / 5 - 2) as f64, t[1] + (q % 5 - 2) as f64]))).filter(|&v| norm(v) > 1e-6).collect();
    cands.sort_by(|u, v| round_dec(norm(*u), 9).total_cmp(&round_dec(norm(*v), 9)).then_with(|| (-u[0]).total_cmp(&(-v[0]))).then_with(|| (-u[1]).total_cmp(&(-v[1]))));
    let (area, b1) = (mat2_det(a).abs() / (ts.len() as f64), cands[0]);
    let b2 = *cands.iter().find(|&&v| (cross2(b1, v).abs() - area).abs() < 1e-6 * area.max(1.0)).expect("primitive b2");
    let (b1, b2) = reduce_basis(b1, b2);
    rebase_cell(cell, b1, b2)
}

pub(crate) fn identify_wallpaper_group(cell: &Cell2D, use_types: bool) -> &'static str {
    let p = primitive_cell(cell, use_types);
    let (a, ainv) = (mat2_cols(p.a1, p.a2), mat2_inv(mat2_cols(p.a1, p.a2)));
    let (mut rot_orders, mut refl) = (Vec::new(), Vec::new());
    for op in symmetry_ops(&p, false, use_types) {
        let m = mat2_mul(mat2_mul(a, imat2_to_f64(op.mf)), ainv);
        if mat2_det(m) > 0.0 {
            let ang = m[1][0].atan2(m[0][0]);
            rot_orders.push((if ang.abs() < 1e-6 { 1 } else { round_even(2.0 * PI / ang.abs()) as usize }, op.mf, op.t));
        } else {
            refl.push((op.mf, op.t));
        }
    }
    let n_max = rot_orders.iter().map(|r| r.0).max().unwrap_or(1);
    let has_mirror = |mf: [[i32; 2]; 2], t: [f64; 2]| {
        let mp = [[(mf[0][0] + 1) as f64, mf[0][1] as f64], [mf[1][0] as f64, (mf[1][1] + 1) as f64]];
        (-3..=3).any(|l0| (-3..=3).any(|l1| norm(mat2_vec(mp, [t[0] + l0 as f64, t[1] + l1 as f64])) < 1e-6))
    };
    let has_glide = |mf: [[i32; 2]; 2], t: [f64; 2]| {
        let mp = [[(mf[0][0] + 1) as f64, mf[0][1] as f64], [mf[1][0] as f64, (mf[1][1] + 1) as f64]];
        (-3..=3).any(|l0| (-3..=3).any(|l1| {
            let g = mat2_vec(mp, [0.5 * (t[0] + l0 as f64), 0.5 * (t[1] + l1 as f64)]);
            (g[0] - round_even(g[0])).abs() > 1e-6 || (g[1] - round_even(g[1])).abs() > 1e-6
        }))
    };
    let centres_on_mirrors = |order: usize| {
        rot_orders.iter().filter(|r| r.0 == order).all(|&(_, mf, t)| {
            let im = [[(1 - mf[0][0]) as f64, -mf[0][1] as f64], [-mf[1][0] as f64, (1 - mf[1][1]) as f64]];
            (-3..=3).all(|l0| (-3..=3).all(|l1| {
                let c = solve2(im, [t[0] + l0 as f64, t[1] + l1 as f64]);
                refl.iter().any(|&(rf, s)| {
                    let rc = mat2_vec(imat2_to_f64(rf), c);
                    (rc[0] + s[0] - c[0] - round_even(rc[0] + s[0] - c[0])).abs() < 1e-6
                        && (rc[1] + s[1] - c[1] - round_even(rc[1] + s[1] - c[1])).abs() < 1e-6
                })
            }))
        })
    };
    let (any_m, all_m, any_g) = (refl.iter().any(|&(m, t)| has_mirror(m, t)), refl.iter().all(|&(m, t)| has_mirror(m, t)), refl.iter().any(|&(m, t)| has_glide(m, t)));
    match n_max {
        1 => if refl.is_empty() { "p1" } else if !any_m { "pg" } else if any_g { "cm" } else { "pm" },
        2 => if refl.is_empty() { "p2" } else if all_m { if any_g { "cmm" } else { "pmm" } } else if any_m { "pmg" } else { "pgg" },
        3 => if refl.is_empty() { "p3" } else if centres_on_mirrors(3) { "p3m1" } else { "p31m" },
        4 => if refl.is_empty() { "p4" } else if centres_on_mirrors(4) { "p4m" } else { "p4g" },
        6 => if refl.is_empty() { "p6" } else { "p6m" },
        _ => unreachable!("unexpected 2D crystallographic rotation order"),
    }
}

pub(crate) fn has_crossings(cell: &Cell2D) -> bool {
    let a = mat2_cols(cell.a1, cell.a2);
    let segs: Vec<([f64; 2], [f64; 2])> = cell.bonds.iter().map(|b| {
        let sh = mat2_vec(a, [b.n1 as f64, b.n2 as f64]);
        (cell.basis[b.i], [cell.basis[b.j][0] + sh[0], cell.basis[b.j][1] + sh[1]])
    }).collect();
    for n1 in -5..=5 {
        for n2 in -5..=5 {
            let sh = mat2_vec(a, [n1 as f64, n2 as f64]);
            for (k, &(p, q)) in segs.iter().enumerate() {
                for (l, &(r0, s0)) in segs.iter().enumerate() {
                    if (n1, n2) == (0, 0) && l <= k { continue; }
                    let (u_vec, v_vec) = ([q[0] - p[0], q[1] - p[1]], [s0[0] - r0[0], s0[1] - r0[1]]);
                    let den = cross2(u_vec, v_vec);
                    if den.abs() < 1e-9 { continue; }
                    let rp = [r0[0] + sh[0] - p[0], r0[1] + sh[1] - p[1]];
                    let (u, v) = (cross2(rp, v_vec) / den, cross2(rp, u_vec) / den);
                    if (1e-9..(1.0 - 1e-9)).contains(&u) && (1e-9..(1.0 - 1e-9)).contains(&v) { return true; }
                }
            }
        }
    }
    false
}

#[derive(Clone, Debug)]
pub(crate) struct LatticeGroup {
    pub cell: Cell2D,
    pub a: [[f64; 2]; 2],
    pub ainv: [[f64; 2]; 2],
    pub opsf: Vec<FracOp2D>,
    pub ops: Vec<Isometry2>,
    pub plus: Vec<Isometry2>,
    pub minus: Vec<Isometry2>,
    pub chiral: bool,
    pub name: &'static str,
    pub l: PerfectLattice,
    pub scale: f64,
    pub planar: bool,
    pub lmax: f64,
}

impl LatticeGroup {
    pub fn new(input_cell: &Cell2D) -> Self {
        let p = primitive_cell(input_cell, false);
        let a = mat2_cols(p.a1, p.a2);
        let ainv = mat2_inv(a);
        let opsf = symmetry_ops(&p, false, false);
        let ops: Vec<Isometry2> = opsf.iter().map(|op| Isometry2 { m: mat2_mul(mat2_mul(a, imat2_to_f64(op.mf)), ainv), t: mat2_vec(a, op.t) }).collect();
        let plus: Vec<Isometry2> = ops.iter().copied().filter(|g| g.det() > 0.0).collect();
        let minus: Vec<Isometry2> = ops.iter().copied().filter(|g| g.det() < 0.0).collect();
        let chiral = minus.is_empty();
        let name = identify_wallpaper_group(&p, false);
        let scale = norm(p.a1).max(norm(p.a2));
        let planar = !has_crossings(&p);
        let lmax = p.bonds.iter().map(|b| {
            let sh = mat2_vec(a, [b.n1 as f64, b.n2 as f64]);
            norm([p.basis[b.j][0] + sh[0] - p.basis[b.i][0], p.basis[b.j][1] + sh[1] - p.basis[b.i][1]])
        }).fold(0.0_f64, f64::max);
        let l = PerfectLattice::new(p.clone());
        Self { cell: p, a, ainv, opsf, ops, plus, minus, chiral, name, l, scale, planar, lmax }
    }

    pub fn is_lattice(&self, v: [f64; 2], tol: f64) -> bool {
        let f = mat2_vec(self.ainv, v);
        (f[0] - round_even(f[0])).abs() < tol && (f[1] - round_even(f[1])).abs() < tol
    }

    pub fn reduce(&self, p: [f64; 2]) -> [f64; 2] {
        let mut f = mat2_vec(self.ainv, p);
        f[0] -= (f[0] + 1e-6).floor();
        f[1] -= (f[1] + 1e-6).floor();
        if f[0].abs() < 1e-9 { f[0] = 0.0; }
        if f[1].abs() < 1e-9 { f[1] = 0.0; }
        mat2_vec(self.a, f)
    }

    pub fn same_point(&self, p: [f64; 2], q: [f64; 2]) -> bool {
        self.is_lattice([p[0] - q[0], p[1] - q[1]], 1e-4)
    }

    pub fn contains(&self, g: Isometry2, tol: f64) -> bool {
        self.ops.iter().find(|h| np_allclose_mat(h.m, g.m, 1e-5, 1e-6)).is_some_and(|h| self.is_lattice([g.t[0] - h.t[0], g.t[1] - h.t[1]], tol))
    }

    pub fn ita_frac(input_group: &str, input_cell: &Cell2D, frame: Frame2D, x: [f64; 2]) -> Option<[f64; 2]> {
        if input_group.is_empty() { return None; }
        let u_inv = if matches!(input_group, "cm" | "cmm") { [[1.0, 1.0], [-1.0, 1.0]] } else { [[1.0, 0.0], [0.0, 1.0]] };
        let c_inv = mat2_inv(mat2_mul(mat2_cols(input_cell.a1, input_cell.a2), u_inv));
        Some(mat2_vec(c_inv, [x[0] - frame.origin[0], x[1] - frame.origin[1]]))
    }

    pub fn solve_normaliser_cosets(&self) -> (usize, usize, usize, Vec<ScoredCoset>) {
        let pf: HashMap<[[i32; 2]; 2], [f64; 2]> = self.opsf.iter().map(|op| (op.mf, op.t)).collect();
        let hol = holohedry(self.a);
        let is_int2 = |v: [f64; 2]| (v[0] - round_even(v[0])).abs() < 1e-5 && (v[1] - round_even(v[1])).abs() < 1e-5;
        let id_i = [[1, 0], [0, 1]];
        let rot: Vec<_> = self.opsf.iter().copied().filter(|op| imat2_det(op.mf) == 1 && op.mf != id_i).collect();
        let refl: Vec<_> = self.opsf.iter().copied().filter(|op| imat2_det(op.mf) == -1).collect();
        let cont = if !rot.is_empty() { 0 } else if !refl.is_empty() { 1 } else { 2 };
        type FracElem = ([[i32; 2]; 2], [f64; 2]);
        type BestCand = ((u8, f64, f64), [[i32; 2]; 2], [f64; 2], Isometry2);

        let mut cands: Vec<FracElem> = Vec::new();
        for &m in &hol {
            let mi = imat2_inv(m);
            if !self.opsf.iter().all(|op| pf.contains_key(&imat2_mul(imat2_mul(m, op.mf), mi))) { continue; }
            let mf = imat2_to_f64(m);
            let eqs: Vec<FracElem> = self.opsf.iter().map(|op| {
                let rp = imat2_mul(imat2_mul(m, op.mf), mi);
                let (pft, mt) = (pf[&rp], mat2_vec(mf, op.t));
                (rp, [pft[0] - mt[0], pft[1] - mt[1]])
            }).collect();
            let sols: Vec<[f64; 2]> = if let Some(r0) = rot.first() {
                let rp = imat2_mul(imat2_mul(m, r0.mf), mi);
                let (pft, mt) = (pf[&rp], mat2_vec(mf, r0.t));
                let im = [[(1 - rp[0][0]) as f64, -rp[0][1] as f64], [-rp[1][0] as f64, (1 - rp[1][1]) as f64]];
                (0..16).map(|q| solve2(im, [pft[0] - mt[0] + (q / 4) as f64, pft[1] - mt[1] + (q % 4) as f64])).collect()
            } else if let Some(r0) = refl.first() {
                let rp = imat2_mul(imat2_mul(m, r0.mf), mi);
                let (pft, mt) = (pf[&rp], mat2_vec(mf, r0.t));
                let c = [pft[0] - mt[0], pft[1] - mt[1]];
                let i_rp = [[1 - rp[0][0], -rp[0][1]], [-rp[1][0], 1 - rp[1][1]]];
                let (kv, wv) = kernel_and_complement(i_rp);
                let d = mat2_vec(imat2_to_f64(i_rp), wv);
                let mut lams: Vec<f64> = Vec::new();
                for q in 0..169 {
                    let vv = [c[0] + (q / 13 - 6) as f64, c[1] + (q % 13 - 6) as f64];
                    if (vv[0] * d[1] - vv[1] * d[0]).abs() < 1e-7 {
                        let lam = py_mod(dot(vv, d) / dot(d, d), 1.0);
                        if !lams.iter().any(|&l2| (lam - l2).abs() < 1e-7 || ((lam - l2).abs() - 1.0).abs() < 1e-7) { lams.push(lam); }
                    }
                }
                lams.into_iter().flat_map(|lam| [0.0, 0.5].map(|mu| [lam * wv[0] + mu * kv[0], lam * wv[1] + mu * kv[1]])).collect()
            } else {
                vec![[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]
            };
            for s0 in sols {
                let s = [s0[0] - (s0[0] + 1e-7).floor(), s0[1] - (s0[1] + 1e-7).floor()];
                let ok = eqs.iter().all(|&(rp, c)| {
                    let lhs = mat2_vec([[(1 - rp[0][0]) as f64, -rp[0][1] as f64], [-rp[1][0] as f64, (1 - rp[1][1]) as f64]], s);
                    is_int2([lhs[0] - c[0], lhs[1] - c[1]])
                });
                if ok && !cands.iter().any(|&(m2, s2)| m == m2 && is_int2([s[0] - s2[0], s[1] - s2[1]])) { cands.push((m, s)); }
            }
        }
        let same_coset = |a: FracElem, b: FracElem| -> bool {
            let mai = imat2_inv(a.0);
            let Some(&pft) = pf.get(&imat2_mul(mai, b.0)) else { return false };
            let diff = mat2_vec(imat2_to_f64(mai), [b.1[0] - a.1[0], b.1[1] - a.1[1]]);
            is_int2([diff[0] - pft[0], diff[1] - pft[1]])
        };
        let mut cosets: Vec<Vec<FracElem>> = Vec::new();
        for c in cands {
            if let Some(cl) = cosets.iter_mut().find(|cl| same_coset(cl[0], c)) { cl.push(c); } else { cosets.push(vec![c]); }
        }
        let mut scored: Vec<ScoredCoset> = Vec::new();
        for cl in &cosets {
            if same_coset((id_i, [0.0, 0.0]), cl[0]) { continue; }
            let (m, s) = cl[0];
            let mut best: Option<BestCand> = None;
            for op in &self.opsf {
                let (mt, mf) = (mat2_vec(imat2_to_f64(m), op.t), imat2_mul(m, op.mf));
                let mc = mat2_mul(mat2_mul(self.a, imat2_to_f64(mf)), self.ainv);
                for q in 0..25 {
                    let sf = [mt[0] + s[0] + (q / 5 - 2) as f64, mt[1] + s[1] + (q % 5 - 2) as f64];
                    let tc = mat2_vec(self.a, sf);
                    let sc = if mf == id_i { (0u8, round_dec(norm(tc), 6), 0.0) }
                    else if imat2_det(mf) == 1 {
                        let cen = solve2([[1.0 - mc[0][0], -mc[0][1]], [-mc[1][0], 1.0 - mc[1][1]]], tc);
                        (1u8, round_dec(mc[1][0].atan2(mc[0][0]).abs(), 6), round_dec(norm(cen), 6))
                    } else {
                        let u = eigvec_max_sym(mc);
                        (2u8, round_dec(dot(tc, u).abs(), 6), round_dec(dot(tc, rot90(u)).abs(), 6))
                    };
                    if best.as_ref().is_none_or(|(b, _, _, _)| sc.0.cmp(&b.0).then_with(|| sc.1.total_cmp(&b.1)).then_with(|| sc.2.total_cmp(&b.2)).is_lt()) {
                        best = Some((sc, mf, sf, Isometry2 { m: mc, t: tc }));
                    }
                }
            }
            let (sc, mf, sf, sig) = best.expect("non-empty coset");
            scored.push((sc, match sc.0 { 0 => "translation", 1 => "rotation", _ => "mirror" }, sig, mf, sf));
        }
        scored.sort_by(|a, b| a.0.0.cmp(&b.0.0).then_with(|| a.0.1.total_cmp(&b.0.1)).then_with(|| a.0.2.total_cmp(&b.0.2)));
        (cosets.len(), cont, hol.len(), scored)
    }
}

pub(crate) type ScoredCoset = ((u8, f64, f64), &'static str, Isometry2, [[i32; 2]; 2], [f64; 2]);
