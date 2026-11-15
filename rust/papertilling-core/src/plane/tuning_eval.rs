//! Self-contained recipe evaluator and 17-group symmetry finder for Stage 2.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::f64::consts::PI;

use crate::types::{Bond2D, Cell2D, Recipe};

const S3: f64 = 1.732_050_807_568_877_2;
const TOL: f64 = 1e-6;
const MIRROR_DIRS: [[i32; 2]; 6] = [[0, 1], [1, 0], [1, 1], [1, -1], [1, 2], [2, 1]];

pub(crate) type PrimOp = ([[i32; 2]; 2], [f64; 2]);
pub(crate) type BareBond = (usize, usize, i32, i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CrystalSystem { Oblique, Rectangular, Square, Hexagonal }

#[derive(Clone, Debug)]
pub(crate) struct GroupSpec {
    pub symbol: &'static str,
    pub system: CrystalSystem,
    pub centred: bool,
    pub ops: Vec<PrimOp>,
}

fn parse_linear_coord(expr: &str) -> ([i32; 2], f64) {
    let s = expr.trim();
    let (mut m, mut t, mut start, b) = ([0i32; 2], 0.0, 0usize, s.as_bytes());
    for idx in 1..=b.len() {
        if idx == b.len() || b[idx] == b'+' || b[idx] == b'-' {
            match &s[start..idx] {
                "x" | "+x" => m[0] += 1,
                "-x" => m[0] -= 1,
                "y" | "+y" => m[1] += 1,
                "-y" => m[1] -= 1,
                "1/2" | "+1/2" => t += 0.5,
                "-1/2" => t -= 0.5,
                other => panic!("unexpected coordinate token: {other}"),
            }
            start = idx;
        }
    }
    (m, t)
}

pub(crate) fn frac_mod(f: [f64; 2], tol: f64) -> [f64; 2] {
    f.map(|x| { let y = x - (x + tol).floor(); if (y - 1.0).abs() < tol { 0.0 } else { y } })
}

pub(crate) fn group_spec(name: &str) -> Result<GroupSpec, String> {
    use CrystalSystem::{Hexagonal, Oblique, Rectangular, Square};
    let (symbol, system, centred, raw_ops): (&'static str, CrystalSystem, bool, &[&str]) =
        match name.trim().to_ascii_lowercase().as_str() {
            "p1" => ("p1", Oblique, false, &["x,y"]),
            "p2" => ("p2", Oblique, false, &["x,y", "-x,-y"]),
            "pm" | "p1m1" => ("pm", Rectangular, false, &["x,y", "-x,y"]),
            "pg" | "p1g1" => ("pg", Rectangular, false, &["x,y", "-x,y+1/2"]),
            "cm" | "c1m1" => ("cm", Rectangular, true, &["x,y", "-x,y"]),
            "pmm" | "p2mm" => ("pmm", Rectangular, false, &["x,y", "-x,-y", "-x,y", "x,-y"]),
            "pmg" | "p2mg" => ("pmg", Rectangular, false, &["x,y", "-x,-y", "-x+1/2,y", "x+1/2,-y"]),
            "pgg" | "p2gg" => ("pgg", Rectangular, false, &["x,y", "-x,-y", "-x+1/2,y+1/2", "x+1/2,-y+1/2"]),
            "cmm" | "c2mm" => ("cmm", Rectangular, true, &["x,y", "-x,-y", "-x,y", "x,-y"]),
            "p4" => ("p4", Square, false, &["x,y", "-x,-y", "-y,x", "y,-x"]),
            "p4m" | "p4mm" => ("p4m", Square, false, &["x,y", "-x,-y", "-y,x", "y,-x", "-x,y", "x,-y", "y,x", "-y,-x"]),
            "p4g" | "p4gm" => ("p4g", Square, false, &["x,y", "-x,-y", "-y,x", "y,-x", "-x+1/2,y+1/2", "x+1/2,-y+1/2", "y+1/2,x+1/2", "-y+1/2,-x+1/2"]),
            "p3" => ("p3", Hexagonal, false, &["x,y", "-y,x-y", "-x+y,-x"]),
            "p3m1" => ("p3m1", Hexagonal, false, &["x,y", "-y,x-y", "-x+y,-x", "-y,-x", "-x+y,y", "x,x-y"]),
            "p31m" => ("p31m", Hexagonal, false, &["x,y", "-y,x-y", "-x+y,-x", "y,x", "x-y,-y", "-x,-x+y"]),
            "p6" => ("p6", Hexagonal, false, &["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x"]),
            "p6m" | "p6mm" => ("p6m", Hexagonal, false, &["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x", "-y,-x", "-x+y,y", "x,x-y", "y,x", "x-y,-y", "-x,-x+y"]),
            _ => return Err(format!("unknown wallpaper group {name:?}")),
        };
    let mut ops: Vec<PrimOp> = raw_ops.iter().map(|s| {
        let (sx, sy) = s.split_once(',').unwrap();
        let ((r0, t0), (r1, t1)) = (parse_linear_coord(sx), parse_linear_coord(sy));
        ([r0, r1], frac_mod([t0, t1], TOL))
    }).collect();
    if centred {
        for idx in 0..ops.len() {
            let (m, t) = ops[idx];
            ops.push((m, frac_mod([t[0] + 0.5, t[1] + 0.5], TOL)));
        }
    }
    Ok(GroupSpec { symbol, system, centred, ops })
}

impl GroupSpec {
    pub fn lattice(&self, a: f64, b: Option<f64>, gamma: Option<f64>) -> ([f64; 2], [f64; 2]) {
        match self.system {
            CrystalSystem::Oblique => {
                let (bv, gr) = (b.unwrap_or(1.23 * a), gamma.unwrap_or(76.0).to_radians());
                ([a, 0.0], [bv * gr.cos(), bv * gr.sin()])
            }
            CrystalSystem::Rectangular => ([a, 0.0], [0.0, b.unwrap_or(if self.centred { 1.6 } else { 1.37 } * a)]),
            CrystalSystem::Square => ([a, 0.0], [0.0, a]),
            CrystalSystem::Hexagonal => ([a, 0.0], [-0.5 * a, 0.5 * S3 * a]),
        }
    }
}

pub(crate) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
     [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
}
pub(crate) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] { [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]] }
pub(crate) fn mat2_inv(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    (det.abs() >= 1e-14).then(|| [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]])
}
pub(crate) fn imat2_vec(m: [[i32; 2]; 2], v: [f64; 2]) -> [f64; 2] { [m[0][0] as f64 * v[0] + m[0][1] as f64 * v[1], m[1][0] as f64 * v[0] + m[1][1] as f64 * v[1]] }
pub(crate) fn imat2_ivec(m: [[i32; 2]; 2], v: [i32; 2]) -> [i32; 2] { [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]] }
pub(crate) fn is_int2(v: [f64; 2], tol: f64) -> bool { (v[0] - v[0].round()).abs() < tol && (v[1] - v[1].round()).abs() < tol }
pub(crate) fn canon_bond4(i: usize, j: usize, n1: i32, n2: i32) -> BareBond { (i, j, n1, n2).min((j, i, -n1, -n2)) }
pub(crate) fn cell_mat(cell: &Cell2D) -> [[f64; 2]; 2] { [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]] }

#[derive(Clone, Debug)]
pub(crate) struct SiteGen { pub base: [f64; 2], pub d_cols: Vec<[f64; 2]>, pub u: Vec<f64>, pub names: Vec<String>, pub offset: usize }

fn site_freedom(g: &GroupSpec, p: [f64; 2]) -> ([f64; 2], Vec<[f64; 2]>, Vec<f64>) {
    let stab: Vec<[[i32; 2]; 2]> = g.ops.iter().filter_map(|&(m, t)| {
        let img = imat2_vec(m, p);
        is_int2([img[0] + t[0] - p[0], img[1] + t[1] - p[1]], 1e-7).then_some(m)
    }).collect();
    if stab.iter().all(|&m| m == [[1, 0], [0, 1]]) { return ([0.0, 0.0], vec![[1.0, 0.0], [0.0, 1.0]], vec![p[0], p[1]]); }
    for dir in MIRROR_DIRS {
        if stab.iter().all(|&m| (m[0][0] - 1) * dir[0] + m[0][1] * dir[1] == 0 && m[1][0] * dir[0] + (m[1][1] - 1) * dir[1] == 0) {
            let df = [dir[0] as f64, dir[1] as f64];
            let t0 = (p[0] * df[0] + p[1] * df[1]) / (df[0] * df[0] + df[1] * df[1]);
            return ([p[0] - t0 * df[0], p[1] - t0 * df[1]], vec![df], vec![t0]);
        }
    }
    (p, Vec::new(), Vec::new())
}

#[derive(Clone, Debug)]
pub(crate) struct EvalFamily {
    pub g: GroupSpec,
    pub a: f64,
    pub cellpar: BTreeMap<String, f64>,
    pub site_names: Vec<String>,
    pub names: Vec<String>,
    pub u_mat: [[f64; 2]; 2],
    pub ui_mat: [[f64; 2]; 2],
    pub prim_ops: Vec<PrimOp>,
    pub gens: Vec<SiteGen>,
    pub site_orbit: Vec<usize>,
    pub site_op: Vec<usize>,
    pub site_fl: Vec<[f64; 2]>,
    pub perm: Vec<Vec<usize>>,
    pub shift: Vec<Vec<[i32; 2]>>,
    pub orbits: Vec<Vec<BareBond>>,
}

impl EvalFamily {
    pub fn from_recipe(recipe: &Recipe) -> Result<Self, String> {
        let (g, a) = (group_spec(&recipe.group)?, recipe.a);
        let (_a1, a2) = g.lattice(a, recipe.b, recipe.gamma);
        let (mut cellpar, mut cell_names) = (BTreeMap::new(), Vec::new());
        if matches!(g.system, CrystalSystem::Oblique | CrystalSystem::Rectangular) {
            cellpar.insert("b".to_string(), a2[0].hypot(a2[1]));
            cell_names.push("b".to_string());
        }
        if g.system == CrystalSystem::Oblique {
            cellpar.insert("gamma".to_string(), a2[1].atan2(a2[0]).to_degrees());
            cell_names.push("gamma".to_string());
        }
        let (u_mat, ui_mat) = if g.centred { ([[0.5, -0.5], [0.5, 0.5]], [[1.0, 1.0], [-1.0, 1.0]]) } else { ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]) };
        let reps = if g.centred { &g.ops[..g.ops.len() / 2] } else { &g.ops[..] };
        let prim_ops: Vec<PrimOp> = reps.iter().map(|&(m, t)| {
            let mp_f = mat2_mul(mat2_mul(ui_mat, [[m[0][0] as f64, m[0][1] as f64], [m[1][0] as f64, m[1][1] as f64]]), u_mat);
            ([[mp_f[0][0].round() as i32, mp_f[0][1].round() as i32], [mp_f[1][0].round() as i32, mp_f[1][1].round() as i32]], mat2_vec(ui_mat, t))
        }).collect();

        let mut fam = Self {
            g, a, cellpar, site_names: Vec::new(), names: Vec::new(), u_mat, ui_mat, prim_ops,
            gens: Vec::new(), site_orbit: Vec::new(), site_op: Vec::new(), site_fl: Vec::new(),
            perm: Vec::new(), shift: Vec::new(), orbits: Vec::new(),
        };
        let mut offset = 0usize;
        for &p in &recipe.sites {
            if fam.find_site_at(p, 1e-6).is_some() { return Err(format!("duplicate site orbit at ({}, {})", p[0], p[1])); }
            let (base, d_cols, u) = site_freedom(&fam.g, p);
            let o = fam.gens.len();
            let names = match d_cols.len() { 0 => Vec::new(), 1 => vec![format!("t{}", o + 1)], _ => vec![format!("x{}", o + 1), format!("y{}", o + 1)] };
            fam.site_names.extend(names.iter().cloned());
            let next_off = offset + d_cols.len();
            fam.gens.push(SiteGen { base, d_cols, u, names, offset });
            offset = next_off;
            let pp = mat2_vec(fam.ui_mat, p);
            let mut seen: Vec<[f64; 2]> = Vec::new();
            for (op_idx, &(mp, tp)) in fam.prim_ops.iter().enumerate() {
                let q_im = imat2_vec(mp, pp);
                let q = [q_im[0] + tp[0], q_im[1] + tp[1]];
                if !seen.iter().any(|&r| is_int2([q[0] - r[0], q[1] - r[1]], 1e-7)) {
                    seen.push(q);
                    fam.site_orbit.push(o);
                    fam.site_op.push(op_idx);
                    fam.site_fl.push([(q[0] + 1e-9).floor(), (q[1] + 1e-9).floor()]);
                }
            }
        }
        fam.names = cell_names.into_iter().chain(fam.site_names.iter().cloned()).collect();

        let f0 = fam.f_coords(&fam.values());
        for &(mp, tp) in &fam.prim_ops {
            let (mut pk, mut sk) = (Vec::with_capacity(f0.len()), Vec::with_capacity(f0.len()));
            for &fk in &f0 {
                let g_pos = imat2_vec(mp, fk);
                let target = [g_pos[0] + tp[0], g_pos[1] + tp[1]];
                let (j, sh) = f0.iter().enumerate().find_map(|(j, &fj)| {
                    let d = [target[0] - fj[0], target[1] - fj[1]];
                    is_int2(d, 1e-6).then_some((j, [d[0].round() as i32, d[1].round() as i32]))
                }).ok_or_else(|| "site orbit is not closed under group".to_string())?;
                pk.push(j);
                sk.push(sh);
            }
            fam.perm.push(pk);
            fam.shift.push(sk);
        }
        for &[p, q] in &recipe.bonds {
            let (i, ni) = fam.find_site_at(p, 1e-6).ok_or_else(|| format!("bond end {p:?} is not a site"))?;
            let (j, nj) = fam.find_site_at(q, 1e-6).ok_or_else(|| format!("bond end {q:?} is not a site"))?;
            let n = [nj[0] - ni[0], nj[1] - ni[1]];
            if i == j && n == [0, 0] { return Err("zero-length bond".to_string()); }
            let g_bond = canon_bond4(i, j, n[0], n[1]);
            if fam.orbits.iter().any(|orb| orb.contains(&g_bond)) { return Err("duplicate bond orbit".to_string()); }
            fam.orbits.push(fam.orbit_of(g_bond));
        }
        Ok(fam)
    }

    pub fn m(&self) -> usize { self.site_orbit.len() }

    pub fn orbit_of(&self, (i, j, n1, n2): BareBond) -> Vec<BareBond> {
        let mut out = BTreeSet::new();
        for (g_idx, &(mp, _)) in self.prim_ops.iter().enumerate() {
            let (mn, sj, si) = (imat2_ivec(mp, [n1, n2]), self.shift[g_idx][j], self.shift[g_idx][i]);
            out.insert(canon_bond4(self.perm[g_idx][i], self.perm[g_idx][j], mn[0] + sj[0] - si[0], mn[1] + sj[1] - si[1]));
        }
        out.into_iter().collect()
    }

    pub fn bonds(&self) -> Vec<BareBond> {
        let mut b: Vec<BareBond> = self.orbits.iter().flatten().copied().collect();
        b.sort_unstable();
        b
    }

    pub fn values(&self) -> BTreeMap<String, f64> {
        let mut v = self.cellpar.clone();
        for sg in &self.gens { v.extend(sg.names.iter().cloned().zip(sg.u.iter().copied())); }
        v
    }

    pub fn merged_values(&self, ov: Option<&BTreeMap<String, f64>>) -> BTreeMap<String, f64> {
        let mut v = self.values();
        if let Some(map) = ov { v.extend(map.iter().map(|(k, &val)| (k.clone(), val))); }
        v
    }

    pub fn site_vector(&self, vals: &BTreeMap<String, f64>) -> Vec<f64> { self.site_names.iter().map(|nm| vals[nm]).collect() }

    pub fn metric_vars(&self) -> &'static [&'static str] {
        match self.g.system {
            CrystalSystem::Oblique => &["b cos(gamma)", "b^2"],
            CrystalSystem::Rectangular => &["b^2"],
            CrystalSystem::Square | CrystalSystem::Hexagonal => &[],
        }
    }

    pub fn v_names(&self) -> Vec<String> { self.metric_vars().iter().map(|&s| s.to_string()).chain(self.site_names.iter().cloned()).collect() }

    pub fn v_vec(&self, vals: &BTreeMap<String, f64>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.metric_vars().len() + self.site_names.len());
        if let Some(&b_val) = vals.get("b") && self.cellpar.contains_key("b") {
            let r = b_val / self.a;
            if self.cellpar.contains_key("gamma") { out.push(r * vals["gamma"].to_radians().cos()); }
            out.push(r * r);
        }
        out.extend(self.site_names.iter().map(|nm| vals[nm]));
        out
    }

    pub fn values_from_v(&self, v: &[f64]) -> BTreeMap<String, f64> {
        let (mut out, mut k) = (BTreeMap::new(), 0usize);
        match self.g.system {
            CrystalSystem::Oblique => {
                let b_rel = v[1].max(1e-300).sqrt();
                out.insert("b".to_string(), b_rel * self.a);
                out.insert("gamma".to_string(), (v[0] / b_rel).clamp(-1.0, 1.0).acos().to_degrees());
                k = 2;
            }
            CrystalSystem::Rectangular => {
                out.insert("b".to_string(), v[0].max(1e-300).sqrt() * self.a);
                k = 1;
            }
            CrystalSystem::Square | CrystalSystem::Hexagonal => {}
        }
        out.extend(self.site_names.iter().cloned().zip(v[k..].iter().copied()));
        out
    }

    pub fn affine_sites(&self) -> (Vec<[Vec<f64>; 2]>, Vec<[f64; 2]>) {
        let (d, m) = (self.site_names.len(), self.m());
        let (mut p_mat, mut q_vec) = (vec![[vec![0.0; d], vec![0.0; d]]; m], vec![[0.0; 2]; m]);
        for k in 0..m {
            let (mp, tp) = self.prim_ops[self.site_op[k]];
            let mp_ui = mat2_mul([[mp[0][0] as f64, mp[0][1] as f64], [mp[1][0] as f64, mp[1][1] as f64]], self.ui_mat);
            let sg = &self.gens[self.site_orbit[k]];
            for (c_idx, &d_col) in sg.d_cols.iter().enumerate() {
                let v = mat2_vec(mp_ui, d_col);
                p_mat[k][0][sg.offset + c_idx] = v[0];
                p_mat[k][1][sg.offset + c_idx] = v[1];
            }
            let qb = mat2_vec(mp_ui, sg.base);
            q_vec[k] = [qb[0] + tp[0] - self.site_fl[k][0], qb[1] + tp[1] - self.site_fl[k][1]];
        }
        (p_mat, q_vec)
    }

    pub fn f_coords(&self, vals: &BTreeMap<String, f64>) -> Vec<[f64; 2]> {
        let pts: Vec<[f64; 2]> = self.gens.iter().map(|sg| {
            let gp = sg.d_cols.iter().zip(&sg.names).fold(sg.base, |acc, (dc, nm)| {
                let u_val = vals.get(nm).copied().unwrap_or(0.0);
                [acc[0] + dc[0] * u_val, acc[1] + dc[1] * u_val]
            });
            mat2_vec(self.ui_mat, gp)
        }).collect();
        (0..self.m()).map(|k| {
            let (mp, tp) = self.prim_ops[self.site_op[k]];
            let im = imat2_vec(mp, pts[self.site_orbit[k]]);
            [im[0] + tp[0] - self.site_fl[k][0], im[1] + tp[1] - self.site_fl[k][1]]
        }).collect()
    }

    fn find_site_at(&self, p: [f64; 2], tol: f64) -> Option<(usize, [i32; 2])> {
        if self.gens.is_empty() { return None; }
        let pp = mat2_vec(self.ui_mat, p);
        self.f_coords(&self.values()).into_iter().enumerate().find_map(|(k, fk)| {
            let d = [pp[0] - fk[0], pp[1] - fk[1]];
            is_int2(d, tol).then_some((k, [d[0].round() as i32, d[1].round() as i32]))
        })
    }

    pub fn cell_at(&self, vals: &BTreeMap<String, f64>) -> Cell2D {
        let a = vals.get("a").copied().unwrap_or(self.a);
        let b = vals.get("b").copied().or_else(|| self.cellpar.get("b").copied());
        let gamma = vals.get("gamma").copied().or_else(|| self.cellpar.get("gamma").copied());
        let (a1c, a2c) = self.g.lattice(a, b, gamma);
        let b_mat = mat2_mul([[a1c[0], a2c[0]], [a1c[1], a2c[1]]], self.u_mat);
        let basis = self.f_coords(vals).into_iter().map(|fk| mat2_vec(b_mat, fk)).collect();
        let mut bonds: Vec<Bond2D> = self.orbits.iter().enumerate().flat_map(|(kind, orb)| {
            orb.iter().map(move |&(i, j, n1, n2)| {
                let (ci, cj, cn1, cn2) = canon_bond4(i, j, n1, n2);
                Bond2D { i: ci, j: cj, n1: cn1, n2: cn2, kind }
            })
        }).collect();
        bonds.sort_by_key(|bd| (bd.i, bd.j, bd.n1, bd.n2, bd.kind));
        Cell2D { a1: [b_mat[0][0], b_mat[1][0]], a2: [b_mat[0][1], b_mat[1][1]], basis, bonds }
    }
}

fn reduce_basis(mut b1: [f64; 2], mut b2: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    for _ in 0..200 {
        if b2[0] * b2[0] + b2[1] * b2[1] < b1[0] * b1[0] + b1[1] * b1[1] - 1e-12 { std::mem::swap(&mut b1, &mut b2); }
        let k = ((b1[0] * b2[0] + b1[1] * b2[1]) / (b1[0] * b1[0] + b1[1] * b1[1])).round();
        if k == 0.0 { break; }
        b2 = [b2[0] - k * b1[0], b2[1] - k * b1[1]];
    }
    if b1[0] * b2[1] - b1[1] * b2[0] < 0.0 { b2 = [-b2[0], -b2[1]]; }
    (b1, b2)
}

fn rebase_cell(cell: &Cell2D, b1: [f64; 2], b2: [f64; 2]) -> Result<Cell2D, String> {
    let (a, b_mat) = (cell_mat(cell), [[b1[0], b2[0]], [b1[1], b2[1]]]);
    let binv = mat2_inv(b_mat).ok_or_else(|| "singular basis".to_string())?;
    let mut new_basis: Vec<[f64; 2]> = Vec::new();
    for &p in &cell.basis {
        let q = mat2_vec(b_mat, frac_mod(mat2_vec(binv, p), TOL));
        if !new_basis.iter().any(|&r| (q[0] - r[0]).hypot(q[1] - r[1]) < TOL) { new_basis.push(q); }
    }
    let locate = |p: [f64; 2]| -> Result<(usize, [i32; 2]), String> {
        new_basis.iter().enumerate().find_map(|(s, &b)| {
            let f = mat2_vec(binv, [p[0] - b[0], p[1] - b[1]]);
            is_int2(f, TOL).then_some((s, [f[0].round() as i32, f[1].round() as i32]))
        }).ok_or_else(|| "point is not a lattice site".to_string())
    };
    let mut bonds_set = BTreeSet::new();
    for bd in &cell.bonds {
        let sh = mat2_vec(a, [bd.n1 as f64, bd.n2 as f64]);
        let ((si, ni), (sj, nj)) = (locate(cell.basis[bd.i])?, locate([cell.basis[bd.j][0] + sh[0], cell.basis[bd.j][1] + sh[1]])?);
        bonds_set.insert(canon_bond4(si, sj, nj[0] - ni[0], nj[1] - ni[1]));
    }
    let bonds = bonds_set.into_iter().map(|(i, j, n1, n2)| Bond2D { i, j, n1, n2, kind: 0 }).collect();
    Ok(Cell2D { a1: b1, a2: b2, basis: new_basis, bonds })
}

fn symmetry_ops_bare(cell: &Cell2D, only_identity: bool) -> Result<Vec<PrimOp>, String> {
    let a = cell_mat(cell);
    let ainv = mat2_inv(a).ok_or_else(|| "singular cell".to_string())?;
    let f: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(ainv, p)).collect();
    if f.is_empty() { return Ok(Vec::new()); }
    let bond_lookup: HashSet<BareBond> = cell.bonds.iter().flat_map(|bd| [(bd.i, bd.j, bd.n1, bd.n2), (bd.j, bd.i, -bd.n1, -bd.n2)]).collect();
    let mut mfs = Vec::new();
    if only_identity {
        mfs.push([[1, 0], [0, 1]]);
    } else {
        let mut mats: Vec<[[f64; 2]; 2]> = (0..12).map(|k| {
            let ang = k as f64 * PI / 6.0;
            [[ang.cos(), -ang.sin()], [ang.sin(), ang.cos()]]
        }).collect();
        for n1 in -3..=3 {
            for n2 in -3..=3 {
                if n1 != 0 || n2 != 0 {
                    let v = mat2_vec(a, [n1 as f64, n2 as f64]);
                    let base = v[1].atan2(v[0]).rem_euclid(PI);
                    for x in [base, (base + 0.5 * PI).rem_euclid(PI)] {
                        let (c, s) = ((2.0 * x).cos(), (2.0 * x).sin());
                        mats.push([[c, s], [s, -c]]);
                    }
                }
            }
        }
        for m in mats {
            let mf_f = mat2_mul(mat2_mul(ainv, m), a);
            let r = [[mf_f[0][0].round(), mf_f[0][1].round()], [mf_f[1][0].round(), mf_f[1][1].round()]];
            if (mf_f[0][0] - r[0][0]).abs().max((mf_f[0][1] - r[0][1]).abs()) <= TOL && (mf_f[1][0] - r[1][0]).abs().max((mf_f[1][1] - r[1][1]).abs()) <= TOL {
                let mf = [[r[0][0] as i32, r[0][1] as i32], [r[1][0] as i32, r[1][1] as i32]];
                if !mfs.contains(&mf) { mfs.push(mf); }
            }
        }
    }
    let (mut ops, mut perm, mut shift): (Vec<PrimOp>, _, _) = (Vec::new(), vec![0usize; f.len()], vec![[0i32; 2]; f.len()]);
    for mf in mfs {
        let mf_pos: Vec<[f64; 2]> = f.iter().map(|&fk| imat2_vec(mf, fk)).collect();
        'trans: for fj in &f {
            let t = frac_mod([fj[0] - mf_pos[0][0], fj[1] - mf_pos[0][1]], TOL);
            for i in 0..f.len() {
                let target = [mf_pos[i][0] + t[0], mf_pos[i][1] + t[1]];
                let Some((p_idx, sh)) = f.iter().enumerate().find_map(|(k, &fk)| {
                    let d = [target[0] - fk[0], target[1] - fk[1]];
                    is_int2(d, TOL).then_some((k, [d[0].round() as i32, d[1].round() as i32]))
                }) else { continue 'trans };
                perm[i] = p_idx;
                shift[i] = sh;
            }
            let bonds_ok = cell.bonds.iter().all(|bd| {
                let mn = imat2_ivec(mf, [bd.n1, bd.n2]);
                bond_lookup.contains(&(perm[bd.i], perm[bd.j], mn[0] + shift[bd.j][0] - shift[bd.i][0], mn[1] + shift[bd.j][1] - shift[bd.i][1]))
            });
            if bonds_ok && !ops.iter().any(|&(m2, t2)| m2 == mf && (t[0] - t2[0]).abs() < 1e-5 && (t[1] - t2[1]).abs() < 1e-5) {
                ops.push((mf, t));
            }
        }
    }
    Ok(ops)
}

pub(crate) fn identify_wallpaper_group(cell: &Cell2D) -> Result<(String, usize), String> {
    let ts: Vec<[f64; 2]> = symmetry_ops_bare(cell, true)?.into_iter().map(|(_, t)| t).collect();
    let a = cell_mat(cell);
    let pc = if ts.len() == 1 {
        let (b1, b2) = reduce_basis(cell.a1, cell.a2);
        if (b1[0] - cell.a1[0]).hypot(b1[1] - cell.a1[1]) < 1e-8 && (b2[0] - cell.a2[0]).hypot(b2[1] - cell.a2[1]) < 1e-8 { cell.clone() } else { rebase_cell(cell, b1, b2)? }
    } else {
        let mut cands = Vec::new();
        for &t in &ts {
            for n1 in -2..=2 {
                for n2 in -2..=2 {
                    let v = mat2_vec(a, [t[0] + n1 as f64, t[1] + n2 as f64]);
                    if v[0].hypot(v[1]) > TOL { cands.push(v); }
                }
            }
        }
        cands.sort_by(|u, v| ((u[0].hypot(u[1]) * 1e9).round() as i64).cmp(&((v[0].hypot(v[1]) * 1e9).round() as i64)).then_with(|| (-u[0]).total_cmp(&(-v[0]))).then_with(|| (-u[1]).total_cmp(&(-v[1]))));
        let area = (a[0][0] * a[1][1] - a[0][1] * a[1][0]).abs() / (ts.len() as f64);
        let b1 = cands[0];
        let b2 = *cands.iter().find(|&&v| ((b1[0] * v[1] - b1[1] * v[0]).abs() - area).abs() < 1e-6 * area.max(1.0)).ok_or_else(|| "failed to find primitive basis".to_string())?;
        let (rb1, rb2) = reduce_basis(b1, b2);
        rebase_cell(cell, rb1, rb2)?
    };
    let pa = cell_mat(&pc);
    let painv = mat2_inv(pa).ok_or_else(|| "singular primitive cell".to_string())?;
    let (mut rots, mut refl) = (Vec::new(), Vec::new());
    for (mf, t) in symmetry_ops_bare(&pc, false)? {
        let mc = mat2_mul(mat2_mul(pa, [[mf[0][0] as f64, mf[0][1] as f64], [mf[1][0] as f64, mf[1][1] as f64]]), painv);
        if mc[0][0] * mc[1][1] - mc[0][1] * mc[1][0] > 0.0 {
            let ang = mc[1][0].atan2(mc[0][0]).abs();
            rots.push((if ang < TOL { 1 } else { (2.0 * PI / ang).round() as i32 }, mf, t));
        } else {
            refl.push((mf, t));
        }
    }
    let max_n = rots.iter().map(|&(o, _, _)| o).max().unwrap_or(1);
    let (mut any_mirror, mut all_mirror, mut any_glide) = (false, true, false);
    for &(mf, t) in &refl {
        let mp = [[mf[0][0] + 1, mf[0][1]], [mf[1][0], mf[1][1] + 1]];
        let (mut m_ok, mut g_ok) = (false, false);
        for l1 in -3..=3 {
            for l2 in -3..=3 {
                let v = imat2_vec(mp, [t[0] + l1 as f64, t[1] + l2 as f64]);
                m_ok |= v[0].abs() < TOL && v[1].abs() < TOL;
                g_ok |= (0.5 * v[0] - (0.5 * v[0]).round()).abs().max((0.5 * v[1] - (0.5 * v[1]).round()).abs()) > TOL;
            }
        }
        any_mirror |= m_ok;
        all_mirror &= m_ok;
        any_glide |= g_ok;
    }
    let centres_on_mirrors = |order: i32| -> bool {
        rots.iter().filter(|&&(o, _, _)| o == order).all(|&(_, mf, t)| {
            let Some(im_inv) = mat2_inv([[1.0 - mf[0][0] as f64, -mf[0][1] as f64], [-mf[1][0] as f64, 1.0 - mf[1][1] as f64]]) else { return false };
            (-3..=3).all(|l1| (-3..=3).all(|l2| {
                let c = mat2_vec(im_inv, [t[0] + l1 as f64, t[1] + l2 as f64]);
                refl.iter().any(|&(rf, s)| { let rc = imat2_vec(rf, c); is_int2([rc[0] + s[0] - c[0], rc[1] + s[1] - c[1]], TOL) })
            }))
        })
    };
    let sym = match max_n {
        1 if refl.is_empty() => "p1",
        1 if !any_mirror => "pg",
        1 if any_glide => "cm",
        1 => "pm",
        2 if refl.is_empty() => "p2",
        2 if all_mirror => if any_glide { "cmm" } else { "pmm" },
        2 if any_mirror => "pmg",
        2 => "pgg",
        3 => if refl.is_empty() { "p3" } else if centres_on_mirrors(3) { "p3m1" } else { "p31m" },
        4 => if refl.is_empty() { "p4" } else if centres_on_mirrors(4) { "p4m" } else { "p4g" },
        6 => if refl.is_empty() { "p6" } else { "p6m" },
        other => return Err(format!("unexpected rotation order {other}")),
    };
    Ok((sym.to_string(), pc.basis.len()))
}
