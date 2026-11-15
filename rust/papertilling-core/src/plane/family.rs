//! Parameterised periodic lattice families (`LatticeFamily`) of a wallpaper group `G`,
//! matching `papertilling.legacy.lattice_families.Family`.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::DSymbol;
pub use crate::plane::catalogue::CatalogueRule;
use crate::plane::cell::{
    canon_bond, dsymbol_of_cell, mat2_det, mat2_inv, mat2_mul, mat2_vec,
};
use crate::plane::groups::{
    CrystalSystem, canonical_symbol, conventional_cell, crystal_system, group_ops, prim_ops,
    prim_u, prim_u_inv,
};
use crate::types::{Bond2D, Cell2D, FracOp2D, Frame2D, Isometry2, Recipe};

fn is_int_vec2(v: [f64; 2], tol: f64) -> bool {
    (v[0] - v[0].round_ties_even()).abs() < tol && (v[1] - v[1].round_ties_even()).abs() < tol
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn canon_bare_bond(i: usize, j: usize, n1: i32, n2: i32) -> (usize, usize, i32, i32) {
    let a = (i, j, n1, n2);
    let b = (j, i, -n1, -n2);
    a.min(b)
}

fn site_symmetry_label(cart_mats: &[[[f64; 2]; 2]]) -> String {
    let mut rot = 1i32;
    let mut mir = 0usize;
    for &m in cart_mats {
        if mat2_det(m) > 0.0 {
            let ang = m[1][0].atan2(m[0][0]);
            if ang.abs() > 1e-6 {
                rot = rot.max((2.0 * PI / ang.abs()).round() as i32);
            }
        } else {
            mir += 1;
        }
    }
    if mir == 0 {
        return rot.to_string();
    }
    match rot {
        1 => "m".to_string(),
        2 => "2mm".to_string(),
        3 => "3m".to_string(),
        4 => "4mm".to_string(),
        6 => "6mm".to_string(),
        other => format!("{other}m"),
    }
}

type SiteFreedomTuple = (String, [f64; 2], [Vec<f64>; 2], Vec<f64>, String);
type SiteAction = (Vec<Vec<usize>>, Vec<Vec<[i32; 2]>>);

fn site_freedom(
    ops_conv: &[FracOp2D],
    a_conv: [[f64; 2]; 2],
    p: [f64; 2],
) -> Result<SiteFreedomTuple, String> {
    let a_inv = mat2_inv(a_conv)?;
    let mut rows: Vec<(i32, i32)> = Vec::new();
    let mut cart: Vec<[[f64; 2]; 2]> = Vec::new();
    for op in ops_conv {
        let img = [
            f64::from(op.mf[0][0]) * p[0] + f64::from(op.mf[0][1]) * p[1] + op.t[0] - p[0],
            f64::from(op.mf[1][0]) * p[0] + f64::from(op.mf[1][1]) * p[1] + op.t[1] - p[1],
        ];
        if is_int_vec2(img, 1e-7) {
            rows.push((op.mf[0][0] - 1, op.mf[0][1]));
            rows.push((op.mf[1][0], op.mf[1][1] - 1));
            let mf_f = [
                [f64::from(op.mf[0][0]), f64::from(op.mf[0][1])],
                [f64::from(op.mf[1][0]), f64::from(op.mf[1][1])],
            ];
            cart.push(mat2_mul(mat2_mul(a_conv, mf_f), a_inv));
        }
    }
    let sym = site_symmetry_label(&cart);
    let Some(&(r0, r1)) = rows.iter().find(|&&(x, y)| x != 0 || y != 0) else {
        return Ok((
            "general".to_string(),
            [0.0, 0.0],
            [vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![p[0], p[1]],
            sym,
        ));
    };
    if rows.iter().any(|&(s0, s1)| r0 * s1 - r1 * s0 != 0) {
        return Ok(("point".to_string(), p, [Vec::new(), Vec::new()], Vec::new(), sym));
    }
    // Rank 1 mirror line: orthogonal projection onto primitive integer direction d
    let (mut d0, mut d1) = (r1, -r0);
    let g = gcd_u32(d0.unsigned_abs(), d1.unsigned_abs()) as i32;
    d0 /= g;
    d1 /= g;
    if d0 < 0 || (d0 == 0 && d1 < 0) {
        d0 = -d0;
        d1 = -d1;
    }
    let min_nz = [d0.abs(), d1.abs()]
        .into_iter()
        .filter(|&x| x > 0)
        .min()
        .unwrap_or(1) as f64;
    let d = [f64::from(d0) / min_nz, f64::from(d1) / min_nz];
    let t0 = (p[0] * d[0] + p[1] * d[1]) / (d[0] * d[0] + d[1] * d[1]);
    let base = [p[0] - t0 * d[0], p[1] - t0 * d[1]];
    Ok((
        "line".to_string(),
        base,
        [vec![d[0]], vec![d[1]]],
        vec![t0],
        sym,
    ))
}

/// Generating site orbit descriptor inside a `LatticeFamily`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteGenerator {
    pub kind: String,
    pub base: [f64; 2],
    pub d_mat: [Vec<f64>; 2],
    pub u: Vec<f64>,
    pub sym: String,
    pub mult: usize,
    pub names: Vec<String>,
}

/// Parameterised lattice family of a wallpaper group `G` (`Family` in Python).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeFamily {
    pub group: String,
    pub system: CrystalSystem,
    pub a: f64,
    pub b: Option<f64>,
    pub gamma: Option<f64>,
    pub frame: Frame2D,
    pub gens: Vec<SiteGenerator>,
    pub site_orbit: Vec<usize>,
    site_op: Vec<usize>,
    site_fl: Vec<[f64; 2]>,
    slices: Vec<(usize, usize)>,
    pub cell_names: Vec<String>,
    pub site_names: Vec<String>,
    pub orbits: Vec<Vec<(usize, usize, i32, i32)>>,
    pub types: Vec<usize>,
    u_mat: [[f64; 2]; 2],
    u_inv: [[f64; 2]; 2],
    ops: Vec<FracOp2D>,
}

impl LatticeFamily {
    /// Construct a `LatticeFamily` from a `Recipe` and optional Cartesian `Frame2D`
    /// (`Family.from_recipe`).
    pub fn from_recipe(recipe: &Recipe, frame: Option<Frame2D>) -> Result<Self, String> {
        let sym = canonical_symbol(&recipe.group)?.to_string();
        let system = crystal_system(&sym)?;
        let a = recipe.a;
        let (a1_c, a2_c) = conventional_cell(&sym, a, recipe.b, recipe.gamma)?;
        let b = match system {
            CrystalSystem::Oblique | CrystalSystem::Rectangular => Some(a2_c[0].hypot(a2_c[1])),
            CrystalSystem::Square | CrystalSystem::Hexagonal => None,
        };
        let gamma = match system {
            CrystalSystem::Oblique => Some(a2_c[1].atan2(a2_c[0]).to_degrees()),
            _ => None,
        };
        let mut cell_names = Vec::new();
        if b.is_some() {
            cell_names.push("b".to_string());
        }
        if gamma.is_some() {
            cell_names.push("gamma".to_string());
        }

        let u_mat = prim_u(&sym)?;
        let u_inv = prim_u_inv(&sym)?;
        let ops_conv = group_ops(&sym)?;
        let ops = prim_ops(&sym)?;
        let a_conv = [[a1_c[0], a2_c[0]], [a1_c[1], a2_c[1]]];

        let mut fam = Self {
            group: sym,
            system,
            a,
            b,
            gamma,
            frame: frame.unwrap_or_default(),
            gens: Vec::new(),
            site_orbit: Vec::new(),
            site_op: Vec::new(),
            site_fl: Vec::new(),
            slices: Vec::new(),
            cell_names,
            site_names: Vec::new(),
            orbits: Vec::new(),
            types: Vec::new(),
            u_mat,
            u_inv,
            ops,
        };

        for &p in &recipe.sites {
            if fam.find_site_at(p, 1e-6).is_some() {
                return Err(format!(
                    "the point ({}, {}) lies on the orbit of a site that is already there",
                    p[0], p[1]
                ));
            }
            let (kind, base, d_mat, u, site_sym) = site_freedom(&ops_conv, a_conv, p)?;
            let o = fam.gens.len();
            let n_free = u.len();
            let names = match n_free {
                0 => Vec::new(),
                1 => vec![format!("t{}", o + 1)],
                2 => vec![format!("x{}", o + 1), format!("y{}", o + 1)],
                _ => unreachable!(),
            };
            let start = fam.site_names.len();
            fam.site_names.extend(names.iter().cloned());
            fam.slices.push((start, start + n_free));

            let pp = mat2_vec(fam.u_inv, p);
            let mut seen: Vec<[f64; 2]> = Vec::new();
            for (g_idx, op) in fam.ops.iter().enumerate() {
                let q = [
                    f64::from(op.mf[0][0]) * pp[0] + f64::from(op.mf[0][1]) * pp[1] + op.t[0],
                    f64::from(op.mf[1][0]) * pp[0] + f64::from(op.mf[1][1]) * pp[1] + op.t[1],
                ];
                if seen.iter().any(|&r| is_int_vec2([q[0] - r[0], q[1] - r[1]], 1e-7)) {
                    continue;
                }
                seen.push(q);
                fam.site_orbit.push(o);
                fam.site_op.push(g_idx);
                fam.site_fl.push([(q[0] + 1e-9).floor(), (q[1] + 1e-9).floor()]);
            }
            fam.gens.push(SiteGenerator {
                kind,
                base,
                d_mat,
                u,
                sym: site_sym,
                mult: seen.len(),
                names,
            });
        }

        if !recipe.bonds.is_empty() {
            let (perm, shift) = fam.site_action()?;
            for &[p, q] in &recipe.bonds {
                let (i, ni) = fam.find_site_at(p, 1e-6).ok_or_else(|| {
                    format!("bond end ({}, {}) is not a site", p[0], p[1])
                })?;
                let (j, nj) = fam.find_site_at(q, 1e-6).ok_or_else(|| {
                    format!("bond end ({}, {}) is not a site", q[0], q[1])
                })?;
                let n = [nj[0] - ni[0], nj[1] - ni[1]];
                if i == j && n == [0, 0] {
                    return Err("zero-length bond".to_string());
                }
                let g_bond = canon_bare_bond(i, j, n[0], n[1]);
                if fam.orbits.iter().any(|orb| orb.contains(&g_bond)) {
                    return Err("this bond is already in the lattice".to_string());
                }
                let mut orb_set: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
                for (g_idx, op) in fam.ops.iter().enumerate() {
                    let dn0 = op.mf[0][0] * g_bond.2
                        + op.mf[0][1] * g_bond.3
                        + shift[g_idx][g_bond.1][0]
                        - shift[g_idx][g_bond.0][0];
                    let dn1 = op.mf[1][0] * g_bond.2
                        + op.mf[1][1] * g_bond.3
                        + shift[g_idx][g_bond.1][1]
                        - shift[g_idx][g_bond.0][1];
                    orb_set.insert(canon_bare_bond(
                        perm[g_idx][g_bond.0],
                        perm[g_idx][g_bond.1],
                        dn0,
                        dn1,
                    ));
                }
                let ty = fam.orbits.len();
                fam.orbits.push(orb_set.into_iter().collect());
                fam.types.push(ty);
            }
        }

        Ok(fam)
    }

    pub fn num_sites(&self) -> usize {
        self.site_orbit.len()
    }

    pub fn num_bonds(&self) -> usize {
        self.orbits.iter().map(Vec::len).sum()
    }

    pub fn rank(&self) -> [usize; 4] {
        [self.gens.len(), self.orbits.len(), self.num_sites(), self.num_bonds()]
    }

    pub fn names(&self) -> Vec<String> {
        let mut out = self.cell_names.clone();
        out.extend(self.site_names.iter().cloned());
        out
    }

    pub fn parameter_list(&self) -> Vec<(String, f64)> {
        let mut out = Vec::with_capacity(self.cell_names.len() + self.site_names.len());
        if let Some(b_val) = self.b {
            out.push(("b".to_string(), b_val));
        }
        if let Some(g_val) = self.gamma {
            out.push(("gamma".to_string(), g_val));
        }
        for g in &self.gens {
            for (nm, &val) in g.names.iter().zip(&g.u) {
                out.push((nm.clone(), val));
            }
        }
        out
    }

    pub fn values(&self) -> BTreeMap<String, f64> {
        self.parameter_list().into_iter().collect()
    }

    pub fn set_param(&self, name: &str, value: f64) -> Result<Self, String> {
        let mut next = self.clone();
        next.apply_param_mut(name, value)?;
        Ok(next)
    }

    pub fn with_values<'a, I>(&self, values: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = (&'a str, f64)>,
    {
        let mut next = self.clone();
        for (name, val) in values {
            next.apply_param_mut(name, val)?;
        }
        Ok(next)
    }

    fn apply_param_mut(&mut self, name: &str, value: f64) -> Result<(), String> {
        if name == "a" {
            self.a = value;
            return Ok(());
        }
        if name == "b" && self.b.is_some() {
            self.b = Some(value);
            return Ok(());
        }
        if name == "gamma" && self.gamma.is_some() {
            self.gamma = Some(value);
            return Ok(());
        }
        for g in &mut self.gens {
            if let Some(pos) = g.names.iter().position(|n| n == name) {
                g.u[pos] = value;
                return Ok(());
            }
        }
        Err(format!(
            "{name:?} is not a free parameter of this family; free: {:?}",
            self.names()
        ))
    }

    /// Generating point `base + D * u` of site orbit `o` in conventional fractional coordinates.
    pub fn gen_point(&self, o: usize) -> [f64; 2] {
        let g = &self.gens[o];
        let mut pt = g.base;
        for (col, &u_val) in g.u.iter().enumerate() {
            pt[0] += g.d_mat[0][col] * u_val;
            pt[1] += g.d_mat[1][col] * u_val;
        }
        pt
    }

    fn conv_mat(&self) -> [[f64; 2]; 2] {
        let (a1, a2) = conventional_cell(&self.group, self.a, self.b, self.gamma)
            .expect("validated group in LatticeFamily");
        [[a1[0], a2[0]], [a1[1], a2[1]]]
    }

    /// Primitive fractional coordinates of the labelled sites (`Family._F`), keeping `_site_fl` frozen.
    pub fn prim_sites(&self) -> Vec<[f64; 2]> {
        let pts: Vec<[f64; 2]> = (0..self.gens.len())
            .map(|o| mat2_vec(self.u_inv, self.gen_point(o)))
            .collect();
        (0..self.site_orbit.len())
            .map(|k| {
                let op = &self.ops[self.site_op[k]];
                let p = pts[self.site_orbit[k]];
                let fl = self.site_fl[k];
                [
                    f64::from(op.mf[0][0]) * p[0] + f64::from(op.mf[0][1]) * p[1] + op.t[0] - fl[0],
                    f64::from(op.mf[1][0]) * p[0] + f64::from(op.mf[1][1]) * p[1] + op.t[1] - fl[1],
                ]
            })
            .collect()
    }

    /// Affine representation `F_k(u) = P_k * u + q_k` in primitive fractional coordinates (`_affine_sites`).
    pub fn affine_sites(&self) -> (Vec<[Vec<f64>; 2]>, Vec<[f64; 2]>) {
        let d = self.site_names.len();
        let m = self.site_orbit.len();
        let mut p_out = vec![[vec![0.0; d], vec![0.0; d]]; m];
        let mut q_out = vec![[0.0; 2]; m];
        for k in 0..m {
            let o = self.site_orbit[k];
            let op = &self.ops[self.site_op[k]];
            let g_info = &self.gens[o];
            let mf_f = [
                [f64::from(op.mf[0][0]), f64::from(op.mf[0][1])],
                [f64::from(op.mf[1][0]), f64::from(op.mf[1][1])],
            ];
            let mui = mat2_mul(mf_f, self.u_inv);
            let (s_lo, s_hi) = self.slices[o];
            for col in 0..(s_hi - s_lo) {
                let d_col = [g_info.d_mat[0][col], g_info.d_mat[1][col]];
                let v = mat2_vec(mui, d_col);
                p_out[k][0][s_lo + col] = v[0];
                p_out[k][1][s_lo + col] = v[1];
            }
            let mb = mat2_vec(mui, g_info.base);
            q_out[k] = [
                mb[0] + op.t[0] - self.site_fl[k][0],
                mb[1] + op.t[1] - self.site_fl[k][1],
            ];
        }
        (p_out, q_out)
    }

    fn find_site_at(&self, p_conv: [f64; 2], tol: f64) -> Option<(usize, [i32; 2])> {
        if self.gens.is_empty() {
            return None;
        }
        let f_sites = self.prim_sites();
        let pp = mat2_vec(self.u_inv, p_conv);
        for (k, &fk) in f_sites.iter().enumerate() {
            let d = [pp[0] - fk[0], pp[1] - fk[1]];
            if is_int_vec2(d, tol) {
                return Some((k, [d[0].round_ties_even() as i32, d[1].round_ties_even() as i32]));
            }
        }
        None
    }

    fn site_action(&self) -> Result<SiteAction, String> {
        let f_sites = self.prim_sites();
        let (mut perm, mut shift) = (Vec::with_capacity(self.ops.len()), Vec::with_capacity(self.ops.len()));
        for op in &self.ops {
            let (mut pk, mut sk) = (Vec::with_capacity(f_sites.len()), Vec::with_capacity(f_sites.len()));
            for &fk in &f_sites {
                let g = [
                    f64::from(op.mf[0][0]) * fk[0] + f64::from(op.mf[0][1]) * fk[1] + op.t[0],
                    f64::from(op.mf[1][0]) * fk[0] + f64::from(op.mf[1][1]) * fk[1] + op.t[1],
                ];
                let (j, c) = f_sites
                    .iter()
                    .enumerate()
                    .find_map(|(j, &fj)| {
                        let d = [g[0] - fj[0], g[1] - fj[1]];
                        is_int_vec2(d, 1e-6).then_some((j, [d[0].round_ties_even() as i32, d[1].round_ties_even() as i32]))
                    })
                    .ok_or("site orbit is not closed under the group")?;
                pk.push(j);
                sk.push(c);
            }
            perm.push(pk);
            shift.push(sk);
        }
        Ok((perm, shift))
    }

    /// Sorted list of all bare canonical bonds `(i, j, n1, n2)` across all bond orbits.
    pub fn bonds(&self) -> Vec<(usize, usize, i32, i32)> {
        let mut out: Vec<(usize, usize, i32, i32)> =
            self.orbits.iter().flat_map(|o| o.iter().copied()).collect();
        out.sort_unstable();
        out
    }

    /// Build the primitive `Cell2D` at the current parameter values, applying `self.frame` (`x -> Q*x + origin`).
    pub fn cell(&self) -> Cell2D {
        let b_prim = mat2_mul(self.conv_mat(), self.u_mat);
        let qb = mat2_mul(self.frame.q, b_prim);
        let a1 = [qb[0][0], qb[1][0]];
        let a2 = [qb[0][1], qb[1][1]];
        let basis: Vec<[f64; 2]> = self
            .prim_sites()
            .into_iter()
            .map(|f| {
                let bf = mat2_vec(b_prim, f);
                let qbf = mat2_vec(self.frame.q, bf);
                [qbf[0] + self.frame.origin[0], qbf[1] + self.frame.origin[1]]
            })
            .collect();
        let mut bonds: Vec<Bond2D> = Vec::with_capacity(self.num_bonds());
        for (&kind, orb) in self.types.iter().zip(&self.orbits) {
            for &(i, j, n1, n2) in orb {
                bonds.push(canon_bond(i, j, n1, n2, kind));
            }
        }
        bonds.sort_by_key(|b| (b.i, b.j, b.n1, b.n2, b.kind));
        Cell2D { a1, a2, basis, bonds }
    }

    /// Symmetry operations of `self.group` expressed as Cartesian `Isometry2` in `self.cell()`'s frame
    /// (`lattice.ops_in_cell`).
    pub fn ops_in_cell(&self) -> Result<Vec<Isometry2>, String> {
        let qb = mat2_mul(self.frame.q, mat2_mul(self.conv_mat(), self.u_mat));
        let qb_inv = mat2_inv(qb)?;
        let s = mat2_vec(qb_inv, self.frame.origin);
        let mut out = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let mf_f = [
                [f64::from(op.mf[0][0]), f64::from(op.mf[0][1])],
                [f64::from(op.mf[1][0]), f64::from(op.mf[1][1])],
            ];
            let m = mat2_mul(mat2_mul(qb, mf_f), qb_inv);
            let t_frac = [
                op.t[0] + (1.0 - mf_f[0][0]) * s[0] - mf_f[0][1] * s[1],
                op.t[1] - mf_f[1][0] * s[0] + (1.0 - mf_f[1][1]) * s[1],
            ];
            let t = mat2_vec(qb, t_frac);
            out.push(Isometry2 { m, t });
        }
        Ok(out)
    }

    /// Delaney-Dress symbol of `self.cell()` under `self.group`'s action (`family_key` / `Lattice.dsymbol`).
    pub fn dsymbol(&self) -> Result<DSymbol, String> {
        let c = self.cell();
        let ops = self.ops_in_cell()?;
        dsymbol_of_cell(&c, &ops)
    }

    /// Export a `Recipe` that rebuilds this family at its current parameter values (`Family.recipe`).
    pub fn to_recipe(&self) -> Recipe {
        let round12 = |x: f64| (x * 1e12).round_ties_even() * 1e-12;
        let f_sites = self.prim_sites();
        let sites: Vec<[f64; 2]> = (0..self.gens.len())
            .map(|o| {
                let p = self.gen_point(o);
                [round12(p[0]), round12(p[1])]
            })
            .collect();
        let bonds: Vec<[[f64; 2]; 2]> = self
            .orbits
            .iter()
            .map(|o| {
                let (i, j, n1, n2) = o[0];
                let p = mat2_vec(self.u_mat, f_sites[i]);
                let q = mat2_vec(
                    self.u_mat,
                    [f_sites[j][0] + f64::from(n1), f_sites[j][1] + f64::from(n2)],
                );
                let sh = [(p[0] + 1e-9).floor(), (p[1] + 1e-9).floor()];
                [
                    [round12(p[0] - sh[0]), round12(p[1] - sh[1])],
                    [round12(q[0] - sh[0]), round12(q[1] - sh[1])],
                ]
            })
            .collect();
        Recipe {
            group: self.group.clone(),
            sites,
            bonds,
            a: self.a,
            b: self.b,
            gamma: self.gamma,
        }
    }
}
