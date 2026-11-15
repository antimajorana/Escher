//! Euclidean 2D periodic unit cells (`Cell2D`): Lagrange-Gauss basis reduction,
//! primitive-cell reduction, Euclidean symmetry operations, wallpaper-group identification,
//! planarity diagnostics, and 2D Delaney-Dress symbol construction (`dsymbol_of_cell`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;

use crate::dsymbol::DSymbol;
use crate::plane::groups::frac_mod;
use crate::types::{Bond2D, Cell2D, FracOp2D, Isometry2};

const TOL: f64 = 1e-6;

pub(crate) fn mat2_det(m: [[f64; 2]; 2]) -> f64 { m[0][0] * m[1][1] - m[0][1] * m[1][0] }
pub(crate) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}
pub(crate) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}
pub(crate) fn mat2_inv(m: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], String> {
    let d = mat2_det(m);
    if d.abs() < 1e-15 { Err("singular 2x2 cell matrix".to_string()) } else { Ok([[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]) }
}
fn vec2_norm(v: [f64; 2]) -> f64 { v[0].hypot(v[1]) }
fn vec2_dot(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[0] + a[1] * b[1] }
fn allclose2(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() <= 1e-8 + 1e-5 * b[0].abs() && (a[1] - b[1]).abs() <= 1e-8 + 1e-5 * b[1].abs()
}
fn approx_int(x: f64, tol: f64) -> Option<i32> {
    let r = x.round_ties_even();
    ((x - r).abs() < tol).then_some(r as i32)
}
fn approx_int_vec2(v: [f64; 2], tol: f64) -> Option<[i32; 2]> {
    Some([approx_int(v[0], tol)?, approx_int(v[1], tol)?])
}
fn approx_int_mat2(m: [[f64; 2]; 2], tol: f64) -> Option<[[i32; 2]; 2]> {
    Some([[approx_int(m[0][0], tol)?, approx_int(m[0][1], tol)?], [approx_int(m[1][0], tol)?, approx_int(m[1][1], tol)?]])
}
fn i32_mat_to_f64(m: [[i32; 2]; 2]) -> [[f64; 2]; 2] {
    [[f64::from(m[0][0]), f64::from(m[0][1])], [f64::from(m[1][0]), f64::from(m[1][1])]]
}
pub(crate) fn canon_bond(i: usize, j: usize, n1: i32, n2: i32, kind: usize) -> Bond2D {
    if (i, j, n1, n2) <= (j, i, -n1, -n2) { Bond2D { i, j, n1, n2, kind } } else { Bond2D { i: j, j: i, n1: -n1, n2: -n2, kind } }
}

/// Lagrange-Gauss 2D lattice basis reduction (right-handed, `round_ties_even` matching `np.round`).
pub fn reduce_basis_vectors(mut b1: [f64; 2], mut b2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    for _ in 0..200 {
        if vec2_dot(b2, b2) < vec2_dot(b1, b1) - 1e-12 {
            std::mem::swap(&mut b1, &mut b2);
        }
        let denom = vec2_dot(b1, b1);
        if denom < 1e-30 {
            break;
        }
        let k = (vec2_dot(b1, b2) / denom).round_ties_even();
        if k == 0.0 {
            break;
        }
        b2 = [b2[0] - k * b1[0], b2[1] - k * b1[1]];
    }
    if b1[0] * b2[1] - b1[1] * b2[0] < 0.0 {
        b2 = [-b2[0], -b2[1]];
    }
    (b1, b2)
}

fn locate_site(p: [f64; 2], b_inv: [[f64; 2]; 2], basis: &[[f64; 2]]) -> Result<(usize, [i32; 2]), String> {
    for tol in [TOL, 1e-5] {
        for (s, &b) in basis.iter().enumerate() {
            if let Some(n) = approx_int_vec2(mat2_vec(b_inv, [p[0] - b[0], p[1] - b[1]]), tol) {
                return Ok((s, n));
            }
        }
    }
    Err(format!("point ({}, {}) is not a lattice site", p[0], p[1]))
}

fn candidate_matrices(a_mat: [[f64; 2]; 2]) -> Vec<[[f64; 2]; 2]> {
    let mut mats = Vec::with_capacity(64);
    for k in 0..12 {
        let (s, c) = (f64::from(k) * PI / 6.0).sin_cos();
        mats.push([[c, -s], [s, c]]);
    }
    let mut angle_keys: BTreeSet<i64> = BTreeSet::new();
    for nx in -3..=3 {
        for ny in -3..=3 {
            let v = mat2_vec(a_mat, [f64::from(nx), f64::from(ny)]);
            if vec2_norm(v) > TOL {
                let a0 = v[1].atan2(v[0]).rem_euclid(PI);
                for ang in [a0, (a0 + 0.5 * PI).rem_euclid(PI)] {
                    angle_keys.insert((ang * 1e9).round_ties_even() as i64);
                }
            }
        }
    }
    for key in angle_keys {
        let (s, c) = (2.0 * (key as f64) * 1e-9).sin_cos();
        mats.push([[c, s], [s, -c]]);
    }
    mats
}

fn apply_to_site(
    mf: [[i32; 2]; 2],
    t: [f64; 2],
    frac_sites: &[[f64; 2]],
    k: usize,
) -> Result<(usize, [i32; 2]), String> {
    let fk = frac_sites[k];
    let g = [
        f64::from(mf[0][0]) * fk[0] + f64::from(mf[0][1]) * fk[1] + t[0],
        f64::from(mf[1][0]) * fk[0] + f64::from(mf[1][1]) * fk[1] + t[1],
    ];
    for (j, &fj) in frac_sites.iter().enumerate() {
        if let Some(sh) = approx_int_vec2([g[0] - fj[0], g[1] - fj[1]], TOL) {
            return Ok((j, sh));
        }
    }
    Err("site orbit is not closed under the group".to_string())
}

fn segments_cross(p: [f64; 2], q: [f64; 2], r: [f64; 2], s: [f64; 2], tol: f64) -> bool {
    if p[0].max(q[0]) <= r[0].min(s[0]) + tol
        || r[0].max(s[0]) <= p[0].min(q[0]) + tol
        || p[1].max(q[1]) <= r[1].min(s[1]) + tol
        || r[1].max(s[1]) <= p[1].min(q[1]) + tol
    {
        return false;
    }
    let (a, b) = ([q[0] - p[0], q[1] - p[1]], [s[0] - r[0], s[1] - r[1]]);
    let den = a[0] * b[1] - a[1] * b[0];
    if den.abs() <= tol {
        return false;
    }
    let w = [r[0] - p[0], r[1] - p[1]];
    let u = (w[0] * b[1] - w[1] * b[0]) / den;
    let v = (w[0] * a[1] - w[1] * a[0]) / den;
    u > tol && u < 1.0 - tol && v > tol && v < 1.0 - tol
}

impl Cell2D {
    /// Column matrix `A = [a1 | a2]`.
    pub fn mat_a(&self) -> [[f64; 2]; 2] {
        [[self.a1[0], self.a2[0]], [self.a1[1], self.a2[1]]]
    }

    fn spacing(&self) -> f64 {
        (mat2_det(self.mat_a()).abs() / (self.basis.len().max(1) as f64)).sqrt()
    }

    fn rebase(&self, b1: [f64; 2], b2: [f64; 2], use_types: bool) -> Result<Self, String> {
        let a_mat = self.mat_a();
        let b_mat = [[b1[0], b2[0]], [b1[1], b2[1]]];
        let b_inv = mat2_inv(b_mat)?;
        let mut new_basis: Vec<[f64; 2]> = Vec::new();
        for &p in &self.basis {
            let q = mat2_vec(b_mat, frac_mod(mat2_vec(b_inv, p)));
            if !new_basis.iter().any(|&r| vec2_norm([q[0] - r[0], q[1] - r[1]]) < TOL) {
                new_basis.push(q);
            }
        }
        let mut seen: BTreeSet<(usize, usize, i32, i32, usize)> = BTreeSet::new();
        for bd in &self.bonds {
            let sh = mat2_vec(a_mat, [f64::from(bd.n1), f64::from(bd.n2)]);
            let q = [self.basis[bd.j][0] + sh[0], self.basis[bd.j][1] + sh[1]];
            let (si, ni) = locate_site(self.basis[bd.i], b_inv, &new_basis)?;
            let (sj, nj) = locate_site(q, b_inv, &new_basis)?;
            let kind = if use_types { bd.kind } else { 0 };
            let cb = canon_bond(si, sj, nj[0] - ni[0], nj[1] - ni[1], kind);
            seen.insert((cb.i, cb.j, cb.n1, cb.n2, cb.kind));
        }
        let bonds = seen
            .into_iter()
            .map(|(i, j, n1, n2, kind)| Bond2D { i, j, n1, n2, kind })
            .collect();
        Ok(Self { a1: b1, a2: b2, basis: new_basis, bonds })
    }

    /// Rebase this cell onto its Lagrange-Gauss reduced lattice vectors (`round_ties_even`).
    pub fn reduce_basis(&self) -> Self {
        let (b1, b2) = reduce_basis_vectors(self.a1, self.a2);
        if allclose2(b1, self.a1) && allclose2(b2, self.a2) {
            return self.clone();
        }
        self.rebase(b1, b2, true).unwrap_or_else(|_| self.clone())
    }

    fn symmetry_ops_internal(&self, only_identity: bool, use_types: bool) -> Vec<FracOp2D> {
        let Ok(a_inv) = mat2_inv(self.mat_a()) else {
            return Vec::new();
        };
        if self.basis.is_empty() {
            return Vec::new();
        }
        let a_mat = self.mat_a();
        let frac_sites: Vec<[f64; 2]> = self.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
        let mut bond_set: HashSet<(usize, usize, i32, i32, usize)> =
            HashSet::with_capacity(self.bonds.len() * 2);
        for b in &self.bonds {
            let k = if use_types { b.kind } else { 0 };
            bond_set.insert((b.i, b.j, b.n1, b.n2, k));
            bond_set.insert((b.j, b.i, -b.n1, -b.n2, k));
        }
        let mats = if only_identity {
            vec![[[1.0, 0.0], [0.0, 1.0]]]
        } else {
            candidate_matrices(a_mat)
        };
        let mut ops: Vec<FracOp2D> = Vec::new();
        let mut perm = vec![0usize; frac_sites.len()];
        let mut shift = vec![[0i32; 2]; frac_sites.len()];

        for m_cart in mats {
            let Some(mf) = approx_int_mat2(mat2_mul(mat2_mul(a_inv, m_cart), a_mat), TOL) else {
                continue;
            };
            let mf0 = [
                f64::from(mf[0][0]) * frac_sites[0][0] + f64::from(mf[0][1]) * frac_sites[0][1],
                f64::from(mf[1][0]) * frac_sites[0][0] + f64::from(mf[1][1]) * frac_sites[0][1],
            ];
            'trans: for fj in &frac_sites {
                let t = frac_mod([fj[0] - mf0[0], fj[1] - mf0[1]]);
                for i in 0..frac_sites.len() {
                    let Ok((p_idx, sh_vec)) = apply_to_site(mf, t, &frac_sites, i) else {
                        continue 'trans;
                    };
                    (perm[i], shift[i]) = (p_idx, sh_vec);
                }
                let ok = self.bonds.iter().all(|b| {
                    let dn0 = mf[0][0] * b.n1 + mf[0][1] * b.n2 + shift[b.j][0] - shift[b.i][0];
                    let dn1 = mf[1][0] * b.n1 + mf[1][1] * b.n2 + shift[b.j][1] - shift[b.i][1];
                    bond_set.contains(&(perm[b.i], perm[b.j], dn0, dn1, if use_types { b.kind } else { 0 }))
                });
                if ok && !ops.iter().any(|o| o.mf == mf && allclose2(o.t, t)) {
                    ops.push(FracOp2D { mf, t });
                }
            }
        }
        ops
    }

    /// Fractional symmetry operations `x_frac -> mf * x_frac + t` (`t` mod 1).
    pub fn symmetry_frac_ops(&self, use_types: bool) -> Vec<FracOp2D> {
        self.symmetry_ops_internal(false, use_types)
    }

    /// Cartesian Euclidean symmetry operations `x -> m * x + t` of this cell (`_symmetry_ops`).
    pub fn symmetry_ops(&self, use_types: bool) -> Vec<Isometry2> {
        let Ok(a_inv) = mat2_inv(self.mat_a()) else { return Vec::new() };
        let a_mat = self.mat_a();
        self.symmetry_frac_ops(use_types)
            .into_iter()
            .map(|op| Isometry2 { m: mat2_mul(mat2_mul(a_mat, i32_mat_to_f64(op.mf)), a_inv), t: mat2_vec(a_mat, op.t) })
            .collect()
    }

    /// Reduce a (possibly super-) cell to a primitive reduced cell (`primitive_cell`).
    pub fn try_primitive_cell(&self, use_types: bool) -> Result<Self, String> {
        let ts: Vec<[f64; 2]> = self.symmetry_ops_internal(true, use_types).into_iter().map(|op| op.t).collect();
        if ts.len() <= 1 {
            let (b1, b2) = reduce_basis_vectors(self.a1, self.a2);
            return if allclose2(b1, self.a1) && allclose2(b2, self.a2) { Ok(self.clone()) } else { self.rebase(b1, b2, use_types) };
        }
        let a_mat = self.mat_a();
        let mut cands: Vec<[f64; 2]> = Vec::new();
        for t in &ts {
            for nx in -2..=2 {
                for ny in -2..=2 {
                    let v = mat2_vec(a_mat, [t[0] + f64::from(nx), t[1] + f64::from(ny)]);
                    if vec2_norm(v) > TOL { cands.push(v); }
                }
            }
        }
        let r9 = |x: f64| (x * 1e9).round_ties_even() * 1e-9;
        cands.sort_by(|u, v| r9(vec2_norm(*u)).total_cmp(&r9(vec2_norm(*v))).then((-u[0]).total_cmp(&(-v[0]))).then((-u[1]).total_cmp(&(-v[1]))));
        let area = mat2_det(a_mat).abs() / (ts.len() as f64);
        let b1_raw = *cands.first().ok_or("no candidate primitive vector")?;
        let b2_raw = cands
            .into_iter()
            .find(|v| ((b1_raw[0] * v[1] - b1_raw[1] * v[0]).abs() - area).abs() < TOL * area.max(1.0))
            .ok_or("no second primitive vector with matching area")?;
        let (b1, b2) = reduce_basis_vectors(b1_raw, b2_raw);
        self.rebase(b1, b2, use_types)
    }

    pub fn primitive_cell(&self, use_types: bool) -> Self {
        self.try_primitive_cell(use_types).unwrap_or_else(|_| self.reduce_basis())
    }

    /// Identify the wallpaper group of the infinite graph (`identify_wallpaper_group`).
    pub fn try_identify_wallpaper_group(&self, use_types: bool) -> Result<String, String> {
        let cell = self.try_primitive_cell(use_types)?;
        let (a_mat, ops) = (cell.mat_a(), cell.symmetry_frac_ops(use_types));
        let a_inv = mat2_inv(a_mat)?;
        if ops.is_empty() { return Err("cell has no symmetry operations".to_string()); }
        let (mut rot_orders, mut refl) = (Vec::new(), Vec::new());
        for op in ops {
            let m_cart = mat2_mul(mat2_mul(a_mat, i32_mat_to_f64(op.mf)), a_inv);
            if mat2_det(m_cart) > 0.0 {
                let ang = m_cart[1][0].atan2(m_cart[0][0]);
                rot_orders.push((if ang.abs() < TOL { 1 } else { (2.0 * PI / ang.abs()).round() as i32 }, op.mf, op.t));
            } else {
                refl.push((op.mf, op.t));
            }
        }
        let max_n = rot_orders.iter().map(|r| r.0).max().unwrap_or(1);
        let has_mirror = |mf: [[i32; 2]; 2], t: [f64; 2]| -> bool {
            let p = i32_mat_to_f64([[mf[0][0] + 1, mf[0][1]], [mf[1][0], mf[1][1] + 1]]);
            (-3..=3).any(|lx| (-3..=3).any(|ly| {
                let pv = mat2_vec(p, [t[0] + f64::from(lx), t[1] + f64::from(ly)]);
                pv[0].abs() < TOL && pv[1].abs() < TOL
            }))
        };
        let has_glide = |mf: [[i32; 2]; 2], t: [f64; 2]| -> bool {
            let p = i32_mat_to_f64([[mf[0][0] + 1, mf[0][1]], [mf[1][0], mf[1][1] + 1]]);
            (-3..=3).any(|lx| (-3..=3).any(|ly| {
                let pv = mat2_vec(p, [t[0] + f64::from(lx), t[1] + f64::from(ly)]);
                approx_int_vec2([0.5 * pv[0], 0.5 * pv[1]], TOL).is_none()
            }))
        };
        let centres_on_mirrors = |order: i32| -> Result<bool, String> {
            for &(o, mf, t) in &rot_orders {
                if o != order { continue; }
                let im_inv = mat2_inv(i32_mat_to_f64([[1 - mf[0][0], -mf[0][1]], [-mf[1][0], 1 - mf[1][1]]]))?;
                for lx in -3..=3 {
                    for ly in -3..=3 {
                        let c = mat2_vec(im_inv, [t[0] + f64::from(lx), t[1] + f64::from(ly)]);
                        let on = refl.iter().any(|&(rf, s)| {
                            let rc = mat2_vec(i32_mat_to_f64(rf), c);
                            approx_int_vec2([rc[0] + s[0] - c[0], rc[1] + s[1] - c[1]], TOL).is_some()
                        });
                        if !on { return Ok(false); }
                    }
                }
            }
            Ok(true)
        };

        let mirrors: Vec<bool> = refl.iter().map(|&(m, t)| has_mirror(m, t)).collect();
        let glides: Vec<bool> = refl.iter().map(|&(m, t)| has_glide(m, t)).collect();
        let g = match max_n {
            1 if refl.is_empty() => "p1",
            1 if !mirrors.iter().any(|&x| x) => "pg",
            1 if glides.iter().any(|&x| x) => "cm",
            1 => "pm",
            2 if refl.is_empty() => "p2",
            2 if mirrors.iter().all(|&x| x) => {
                if glides.iter().any(|&x| x) { "cmm" } else { "pmm" }
            }
            2 if mirrors.iter().any(|&x| x) => "pmg",
            2 => "pgg",
            3 if refl.is_empty() => "p3",
            3 => if centres_on_mirrors(3)? { "p3m1" } else { "p31m" },
            4 if refl.is_empty() => "p4",
            4 => if centres_on_mirrors(4)? { "p4m" } else { "p4g" },
            6 => if refl.is_empty() { "p6" } else { "p6m" },
            other => return Err(format!("unexpected rotation order {other}")),
        };
        Ok(g.to_string())
    }

    pub fn identify_wallpaper_group(&self, use_types: bool) -> String {
        self.try_identify_wallpaper_group(use_types).unwrap_or_else(|_| "p1".to_string())
    }

    /// True if any two bonds properly cross under lattice translations (`[-5..=5]^2`).
    pub fn has_crossings(&self) -> bool {
        if self.bonds.is_empty() || self.basis.is_empty() {
            return false;
        }
        let a_mat = self.mat_a();
        let segs0: Vec<([f64; 2], [f64; 2])> = self
            .bonds
            .iter()
            .map(|b| {
                let sh = mat2_vec(a_mat, [f64::from(b.n1), f64::from(b.n2)]);
                (self.basis[b.i], [self.basis[b.j][0] + sh[0], self.basis[b.j][1] + sh[1]])
            })
            .collect();
        for nx in -5..=5 {
            for ny in -5..=5 {
                let sh = mat2_vec(a_mat, [f64::from(nx), f64::from(ny)]);
                for (k, &(p, q)) in segs0.iter().enumerate() {
                    let l_start = if nx == 0 && ny == 0 { k + 1 } else { 0 };
                    for &(r, s) in &segs0[l_start..] {
                        if segments_cross(p, q, [r[0] + sh[0], r[1] + sh[1]], [s[0] + sh[0], s[1] + sh[1]], 1e-9) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Smallest distance between distinct sites in units of average site spacing (`min_separation`).
    pub fn min_separation(&self) -> f64 {
        if self.basis.is_empty() {
            return f64::INFINITY;
        }
        let a_mat = self.mat_a();
        let mut best = f64::INFINITY;
        for nx in -3..=3 {
            for ny in -3..=3 {
                let sh = mat2_vec(a_mat, [f64::from(nx), f64::from(ny)]);
                for (i, &pi) in self.basis.iter().enumerate() {
                    for (j, &pj) in self.basis.iter().enumerate() {
                        if !(i == j && nx == 0 && ny == 0) {
                            best = best.min(vec2_norm([pi[0] - pj[0] - sh[0], pi[1] - pj[1] - sh[1]]));
                        }
                    }
                }
            }
        }
        best / self.spacing()
    }

    /// Smallest distance from a site to a bond it is not an endpoint of, in units of spacing (`clearance`).
    pub fn clearance(&self) -> f64 {
        if self.basis.is_empty() || self.bonds.is_empty() {
            return f64::INFINITY;
        }
        let a_mat = self.mat_a();
        let mut best = f64::INFINITY;
        for b in &self.bonds {
            let p = self.basis[b.i];
            let sh_end = mat2_vec(a_mat, [f64::from(b.n1), f64::from(b.n2)]);
            let q = [self.basis[b.j][0] + sh_end[0], self.basis[b.j][1] + sh_end[1]];
            let v = [q[0] - p[0], q[1] - p[1]];
            let l2 = vec2_dot(v, v);
            for sx in (b.n1.min(0) - 2)..=(b.n1.max(0) + 2) {
                for sy in (b.n2.min(0) - 2)..=(b.n2.max(0) + 2) {
                    let sh = mat2_vec(a_mat, [f64::from(sx), f64::from(sy)]);
                    for (k, &xk) in self.basis.iter().enumerate() {
                        if (k == b.i && sx == 0 && sy == 0) || (k == b.j && sx == b.n1 && sy == b.n2) {
                            continue;
                        }
                        let s_pos = [xk[0] + sh[0], xk[1] + sh[1]];
                        let dist = if l2 <= 1e-30 {
                            vec2_norm([s_pos[0] - p[0], s_pos[1] - p[1]])
                        } else {
                            let t = (vec2_dot([s_pos[0] - p[0], s_pos[1] - p[1]], v) / l2).clamp(0.0, 1.0);
                            vec2_norm([p[0] + t * v[0] - s_pos[0], p[1] + t * v[1] - s_pos[1]])
                        };
                        best = best.min(dist);
                    }
                }
            }
        }
        best / self.spacing()
    }

    /// Delaney-Dress symbol under this cell's own uncoloured symmetry group.
    pub fn dsymbol(&self) -> Result<DSymbol, String> {
        dsymbol_of_cell(self, &self.symmetry_ops(false))
    }
}

/// Construct the 2D Delaney-Dress symbol of `cell` under Euclidean operations `ops` (`papertilling.dsymbol.dsymbol`).
pub fn dsymbol_of_cell(cell: &Cell2D, ops: &[Isometry2]) -> Result<DSymbol, String> {
    if ops.is_empty() {
        return Err("ops must contain at least one symmetry operation".to_string());
    }
    if cell.basis.is_empty() || cell.bonds.is_empty() {
        return Err("cell must have sites and bonds".to_string());
    }
    let a_mat = cell.mat_a();
    let a_inv = mat2_inv(a_mat)?;
    let frac_sites: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();

    let mut darts: Vec<(usize, usize, i32, i32)> = Vec::with_capacity(cell.bonds.len() * 2);
    let mut index: HashMap<(usize, usize, i32, i32), usize> = HashMap::with_capacity(cell.bonds.len() * 2);
    for b in &cell.bonds {
        for d in [(b.i, b.j, b.n1, b.n2), (b.j, b.i, -b.n1, -b.n2)] {
            if index.insert(d, darts.len()).is_some() {
                return Err("a bond is listed twice, or joins a site to its own translate twice".to_string());
            }
            darts.push(d);
        }
    }
    let rev: Vec<usize> = darts.iter().map(|&(i, j, n1, n2)| index[&(j, i, -n1, -n2)]).collect();

    let mut at_site: Vec<Vec<(f64, usize)>> = vec![Vec::new(); frac_sites.len()];
    for (k, &(i, j, n1, n2)) in darts.iter().enumerate() {
        let df = [
            frac_sites[j][0] + f64::from(n1) - frac_sites[i][0],
            frac_sites[j][1] + f64::from(n2) - frac_sites[i][1],
        ];
        let v = mat2_vec(a_mat, df);
        at_site[i].push((v[1].atan2(v[0]), k));
    }
    if at_site.iter().any(Vec::is_empty) {
        return Err("a site has no bonds".to_string());
    }
    let (mut nxt, mut prv, mut deg) = (vec![0; darts.len()], vec![0; darts.len()], vec![0; frac_sites.len()]);
    for (i, lst) in at_site.iter_mut().enumerate() {
        lst.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let m = lst.len();
        deg[i] = m;
        for idx in 0..m {
            nxt[lst[idx].1] = lst[(idx + 1) % m].1;
            prv[lst[idx].1] = lst[(idx + m - 1) % m].1;
        }
    }

    let mut face_of = vec![usize::MAX; darts.len()];
    let mut face_sizes: Vec<u32> = Vec::new();
    for d0 in 0..darts.len() {
        if face_of[d0] != usize::MAX {
            continue;
        }
        let fid = face_sizes.len();
        let (mut d, mut len, mut w1, mut w2) = (d0, 0u32, 0i64, 0i64);
        while face_of[d] == usize::MAX {
            face_of[d] = fid;
            len += 1;
            w1 += i64::from(darts[d].2);
            w2 += i64::from(darts[d].3);
            d = prv[rev[d]];
        }
        if d != d0 {
            return Err("face walk did not close on its start (inconsistent rotation system)".to_string());
        }
        if w1 != 0 || w2 != 0 {
            return Err("a face winds around the torus (bonds cross, or the tiling is not planar)".to_string());
        }
        face_sizes.push(len);
    }

    let euler = frac_sites.len() as isize - (darts.len() / 2) as isize + face_sizes.len() as isize;
    if euler != 0 {
        return Err(format!("V - E + F = {euler} per cell, not 0: not a planar tiling"));
    }

    let mut gact: Vec<(Vec<usize>, bool)> = Vec::with_capacity(ops.len());
    for op in ops {
        let mf = approx_int_mat2(mat2_mul(mat2_mul(a_inv, op.m), a_mat), 1e-5)
            .ok_or("operation linear part is not an integer matrix in the cell basis")?;
        let t_frac = mat2_vec(a_inv, op.t);
        let site_img: Vec<(usize, [i32; 2])> = (0..frac_sites.len())
            .map(|k| apply_to_site(mf, t_frac, &frac_sites, k))
            .collect::<Result<_, _>>()?;
        let mut perm = Vec::with_capacity(darts.len());
        for &(i, j, n1, n2) in &darts {
            let ((pi, ci), (pj, cj)) = (site_img[i], site_img[j]);
            let dn0 = mf[0][0] * n1 + mf[0][1] * n2 + cj[0] - ci[0];
            let dn1 = mf[1][0] * n1 + mf[1][1] * n2 + cj[1] - ci[1];
            let &target = index
                .get(&(pi, pj, dn0, dn1))
                .ok_or("an operation does not map the bonds onto bonds")?;
            perm.push(target);
        }
        gact.push((perm, (mf[0][0] * mf[1][1] - mf[0][1] * mf[1][0]) < 0));
    }

    let nch = 2 * darts.len();
    let mut orbit = vec![usize::MAX; nch];
    let mut reps: Vec<usize> = Vec::new();
    let mut imgs: Vec<usize> = Vec::with_capacity(gact.len());
    for c in 0..nch {
        if orbit[c] != usize::MAX {
            continue;
        }
        let o = reps.len();
        reps.push(c);
        let (d, s) = (c / 2, c % 2);
        imgs.clear();
        for (perm, flip) in &gact {
            imgs.push(2 * perm[d] + if *flip { 1 - s } else { s });
        }
        imgs.sort_unstable();
        imgs.dedup();
        if imgs.len() != gact.len() {
            return Err("the group does not act freely on flags".to_string());
        }
        for &x in &imgs {
            orbit[x] = o;
        }
    }

    let s0: Vec<usize> = reps.iter().map(|&c| orbit[2 * rev[c / 2] + (1 - (c % 2))]).collect();
    let s1: Vec<usize> = reps.iter().map(|&c| orbit[if c % 2 == 0 { 2 * nxt[c / 2] + 1 } else { 2 * prv[c / 2] }]).collect();
    let s2: Vec<usize> = reps.iter().map(|&c| orbit[c ^ 1]).collect();
    let m01: Vec<u32> = reps.iter().map(|&c| face_sizes[face_of[if c % 2 == 0 { c / 2 } else { rev[c / 2] }]]).collect();
    let m12: Vec<u32> = reps.iter().map(|&c| deg[darts[c / 2].0] as u32).collect();
    let ds = DSymbol::new(s0, s1, s2, m01, m12).map_err(|e| e.to_string())?;
    if !ds.is_flat() { return Err("non-zero orbifold curvature on planar torus cell".to_string()); }
    Ok(ds)
}
