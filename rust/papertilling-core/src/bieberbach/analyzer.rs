//! Dimension-n crystallographic space-group extension analyzer:
//! primitive basis conversion, 2-cocycle `nu`, exact `H^2(P, Z^n)` & splitness,
//! and universal abelianizations `(G^+)^ab`, `G^ab`, and `chi_G`.

use std::collections::{HashMap, VecDeque};
use std::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

use super::rat::{
    imat_det, imat_eye, imat_mul, rmat_inv, rmat_mul, rmat_transpose, rmat_vec, IMat, RMat, Rat,
};
use super::snf::{hnf_compress_rows, invariant_factors_compressed, smith_normal_form, solve_int_linear, IntMat};

/// Conventional-basis Seitz generator `(R, t)` acting as `x -> R x + t`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvOp {
    pub r: IMat,
    pub t: Vec<Rat>,
}

/// Finitely generated abelian group `Z^{free_rank} (+) (+_i Z / torsion[i] Z)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbelianGroup {
    pub torsion: Vec<i128>,
    pub free_rank: usize,
}

impl AbelianGroup {
    /// Format invariant factors as e.g. `"0"`, `"Z^2"`, `"Z_2^3"`, `"Z x Z_2^2"`, `"Z_2 x Z_4"`.
    pub fn format(&self) -> String {
        let mut counts: Vec<(i128, usize)> = Vec::new();
        for &d in &self.torsion {
            if let Some(last) = counts.last_mut()
                && last.0 == d
            {
                last.1 += 1;
                continue;
            }
            counts.push((d, 1));
        }
        let mut parts: Vec<String> = counts
            .into_iter()
            .map(|(d, c)| if c == 1 { format!("Z_{d}") } else { format!("Z_{d}^{c}") })
            .collect();
        match self.free_rank {
            0 => {}
            1 => parts.push("Z".to_string()),
            r => parts.push(format!("Z^{r}")),
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" x ")
        }
    }

    pub fn is_trivial(&self) -> bool {
        self.torsion.is_empty() && self.free_rank == 0
    }
}

/// Exact `H^2(P, Z^n)` invariant factors and Seitz 2-cocycle splitness `[nu] == 0`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohomologyH2 {
    pub torsion: Vec<i128>,
    pub is_split: bool,
}

impl CohomologyH2 {
    pub fn format(&self) -> String {
        AbelianGroup { torsion: self.torsion.clone(), free_rank: 0 }.format()
    }
}

/// Abelianization `G^ab` together with the canonical homomorphism `chi_G: G -> G^ab`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbelianizationData {
    pub group: AbelianGroup,
    u_mat: IntMat,
    keep_slots: Vec<(usize, i128)>,
    non_id_pos: HashMap<usize, usize>,
    dim: usize,
}

impl AbelianizationData {
    /// Evaluate canonical abelianization invariant `chi_G(b | R_idx)` where `R_idx` is into `P`.
    pub fn chi(&self, b: &[i64], r_idx: usize) -> Vec<i128> {
        let mut v = vec![0_i128; self.u_mat.cols];
        for i in 0..self.dim {
            v[i] = i128::from(b[i]);
        }
        if r_idx != 0
            && let Some(&pos) = self.non_id_pos.get(&r_idx)
        {
            v[self.dim + pos] = 1;
        }
        let w = self.u_mat.mul_vec(&v);
        self.keep_slots
            .iter()
            .map(|&(row_i, mod_d)| if mod_d > 0 { w[row_i].rem_euclid(mod_d) } else { w[row_i] })
            .collect()
    }
}

/// Dimension-n crystallographic space-group extension analyzer (`1 -> Z^n -> G -> P -> 1`).
#[derive(Clone, Debug)]
pub struct SpaceGroupAnalyzer {
    pub dim: usize,
    pub number: i32,
    pub symbol: String,
    pub b_mat: RMat,
    pub b_inv: RMat,
    pub p_ops: Vec<IMat>,
    pub tau: Vec<Vec<Rat>>,
    pub(crate) nu: Vec<Vec<i64>>,
    pub(crate) p_lookup: HashMap<Vec<i64>, usize>,
    pub(crate) mul_table: Vec<usize>,
    pub(crate) inv_idx: Vec<usize>,
    pub(crate) det_sign: Vec<i64>,
}

pub(crate) fn imat_key(m: &IMat) -> Vec<i64> {
    let mut k = Vec::with_capacity(m.len() * m.len());
    for row in m {
        k.extend_from_slice(row);
    }
    k
}

pub(crate) fn imat_to_rmat(m: &IMat) -> RMat {
    m.iter().map(|row| row.iter().map(|&v| Rat::from_int(v)).collect()).collect()
}

pub(crate) fn rmat_to_imat(m: &RMat, context: &str) -> Result<IMat, String> {
    m.iter()
        .map(|row| {
            row.iter()
                .map(|r| r.as_i64().ok_or_else(|| format!("{context}: non-integer entry {r}")))
                .collect()
        })
        .collect()
}

fn rmat_identity(n: usize) -> RMat {
    (0..n).map(|r| (0..n).map(|c| if r == c { Rat::ONE } else { Rat::ZERO }).collect()).collect()
}

fn rvec_frac_unit(v: &[Rat]) -> Vec<Rat> {
    v.iter().map(|x| x.frac_unit()).collect()
}

fn rvec_add(a: &[Rat], b: &[Rat]) -> Vec<Rat> {
    a.iter().zip(b).map(|(x, y)| x.add(*y)).collect()
}

impl SpaceGroupAnalyzer {
    /// Construct from dimension `n`, IT number, HM symbol, conventional generators, and `TB` rows.
    /// `TB` rows give primitive basis vectors in conventional coordinates, so `B = TB^T`.
    pub fn new(
        dim: usize,
        number: i32,
        symbol: &str,
        gens_conv: &[ConvOp],
        tb_rows: &[Vec<Rat>],
    ) -> Result<Self, String> {
        if dim == 0 || dim > 3 {
            return Err(format!("SpaceGroupAnalyzer supports dim in 1..=3, got {dim}"));
        }
        if tb_rows.len() != dim || tb_rows.iter().any(|r| r.len() != dim) {
            return Err(format!("invalid TB shape for dim={dim}"));
        }
        let tb: RMat = tb_rows.to_vec();
        let b_mat = rmat_transpose(&tb);
        let b_inv = rmat_inv(&b_mat)?;

        let mut gens_prim: Vec<(IMat, Vec<Rat>)> = Vec::with_capacity(gens_conv.len());
        for (gi, g) in gens_conv.iter().enumerate() {
            if g.r.len() != dim || g.r.iter().any(|r| r.len() != dim) || g.t.len() != dim {
                return Err(format!("{symbol}: generator {gi} has mismatched dimension"));
            }
            let r_conv_rat = imat_to_rmat(&g.r);
            let r_prim_rat = rmat_mul(&b_inv, &rmat_mul(&r_conv_rat, &b_mat));
            let r_prim = rmat_to_imat(&r_prim_rat, &format!("{symbol} gen {gi}"))?;
            let det = imat_det(&r_prim);
            if det.abs() != 1 {
                return Err(format!("{symbol} gen {gi}: |det(R_prim)| = {} != 1", det.abs()));
            }
            let t_prim = rmat_vec(&b_inv, &g.t);
            gens_prim.push((r_prim, t_prim));
        }

        let id_mat = imat_eye(dim);
        let id_key = imat_key(&id_mat);
        let mut p_ops = vec![id_mat];
        let mut tau = vec![vec![Rat::ZERO; dim]];
        let mut p_lookup = HashMap::new();
        p_lookup.insert(id_key, 0_usize);
        let mut queue = VecDeque::from([0_usize]);

        while let Some(r1_idx) = queue.pop_front() {
            let r1 = p_ops[r1_idx].clone();
            let t1 = tau[r1_idx].clone();
            let r1_rat = imat_to_rmat(&r1);
            for (rg, tg) in &gens_prim {
                let r2 = imat_mul(&r1, rg);
                let r2_key = imat_key(&r2);
                let t2 = rvec_add(&t1, &rmat_vec(&r1_rat, tg));
                let t2_red = rvec_frac_unit(&t2);
                if let Some(&existing) = p_lookup.get(&r2_key) {
                    if tau[existing] != t2_red {
                        return Err(format!(
                            "{symbol}: inconsistent non-lattice translation remainder for point element"
                        ));
                    }
                } else {
                    if p_ops.len() >= 96 {
                        return Err(format!("{symbol}: point group exceeded finite bound (96)"));
                    }
                    let new_idx = p_ops.len();
                    p_ops.push(r2);
                    tau.push(t2_red);
                    p_lookup.insert(r2_key, new_idx);
                    queue.push_back(new_idx);
                }
            }
        }

        let p_len = p_ops.len();
        let mut mul_table = vec![0_usize; p_len * p_len];
        for i in 0..p_len {
            for j in 0..p_len {
                let prod = imat_mul(&p_ops[i], &p_ops[j]);
                let k = imat_key(&prod);
                let &ij = p_lookup
                    .get(&k)
                    .ok_or_else(|| format!("{symbol}: point group not closed under multiplication"))?;
                mul_table[i * p_len + j] = ij;
            }
        }
        let mut inv_idx = vec![0_usize; p_len];
        for i in 0..p_len {
            inv_idx[i] = (0..p_len)
                .find(|&j| mul_table[i * p_len + j] == 0)
                .ok_or_else(|| format!("{symbol}: element {i} missing inverse"))?;
        }
        let det_sign: Vec<i64> = p_ops.iter().map(imat_det).collect();

        let mut nu = vec![vec![0_i64; dim]; p_len * p_len];
        for i in 0..p_len {
            let r1_rat = imat_to_rmat(&p_ops[i]);
            for j in 0..p_len {
                let ij = mul_table[i * p_len + j];
                let r1_t2 = rmat_vec(&r1_rat, &tau[j]);
                let mut entry = Vec::with_capacity(dim);
                for c in 0..dim {
                    let val = tau[i][c].add(r1_t2[c]).sub(tau[ij][c]);
                    let v = val.as_i64().ok_or_else(|| {
                        format!("{symbol}: non-integer cocycle entry {val} at ({i},{j})[{c}]")
                    })?;
                    entry.push(v);
                }
                nu[i * p_len + j] = entry;
            }
        }

        let out = Self {
            dim,
            number,
            symbol: symbol.to_string(),
            b_mat,
            b_inv,
            p_ops,
            tau,
            nu,
            p_lookup,
            mul_table,
            inv_idx,
            det_sign,
        };
        if !out.verify_2_cocycle() {
            return Err(format!("{symbol}: 2-cocycle identity (d2 nu) == 0 failed"));
        }
        Ok(out)
    }

    #[inline]
    pub fn mul_idx(&self, i: usize, j: usize) -> usize {
        self.mul_table[i * self.p_ops.len() + j]
    }

    #[inline]
    pub fn nu_at(&self, i: usize, j: usize) -> &[i64] {
        &self.nu[i * self.p_ops.len() + j]
    }

    pub fn p_plus_indices(&self) -> Vec<usize> {
        (0..self.p_ops.len()).filter(|&i| self.det_sign[i] == 1).collect()
    }

    pub fn p_minus_indices(&self) -> Vec<usize> {
        (0..self.p_ops.len()).filter(|&i| self.det_sign[i] == -1).collect()
    }

    pub fn all_indices(&self) -> Vec<usize> {
        (0..self.p_ops.len()).collect()
    }

    pub fn mat_order(&self, r_idx: usize) -> usize {
        let mut cur = r_idx;
        let mut m = 1usize;
        while cur != 0 {
            cur = self.mul_idx(cur, r_idx);
            m += 1;
            debug_assert!(m <= 24);
        }
        m
    }

    pub fn r_conv_rat(&self, r_idx: usize) -> RMat {
        let r_prim = imat_to_rmat(&self.p_ops[r_idx]);
        rmat_mul(&self.b_mat, &rmat_mul(&r_prim, &self.b_inv))
    }

    pub fn to_conv_frac(&self, v_prim: &[Rat]) -> Vec<Rat> {
        rmat_vec(&self.b_mat, v_prim)
    }

    pub fn det_of(&self, r_idx: usize) -> i64 {
        self.det_sign[r_idx]
    }

    pub fn point_index(&self, r_prim: &IMat) -> Option<usize> {
        self.p_lookup.get(&imat_key(r_prim)).copied()
    }

    /// Check `(d2 nu)(R1, R2, R3) = R1*nu(R2,R3) - nu(R1*R2,R3) + nu(R1,R2*R3) - nu(R1,R2) == 0`.
    pub fn verify_2_cocycle(&self) -> bool {
        let p_len = self.p_ops.len();
        let n = self.dim;
        for i in 0..p_len {
            let r1 = &self.p_ops[i];
            for j in 0..p_len {
                let ij = self.mul_idx(i, j);
                let n12 = self.nu_at(i, j);
                for k in 0..p_len {
                    let jk = self.mul_idx(j, k);
                    let n23 = self.nu_at(j, k);
                    let n12_3 = self.nu_at(ij, k);
                    let n1_23 = self.nu_at(i, jk);
                    for r in 0..n {
                        let r1_n23: i64 = (0..n).map(|c| r1[r][c] * n23[c]).sum();
                        if r1_n23 - n12_3[r] + n1_23[r] - n12[r] != 0 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Seitz multiplication `(b1 | r1) * (b2 | r2) = (b1 + R1*b2 + nu(r1, r2) | r1*r2)`.
    pub fn seitz_mul(&self, b1: &[i64], r1: usize, b2: &[i64], r2: usize) -> (Vec<i64>, usize) {
        let n = self.dim;
        let nu12 = self.nu_at(r1, r2);
        let rm1 = &self.p_ops[r1];
        let mut out_b = Vec::with_capacity(n);
        for i in 0..n {
            let r1_b2: i64 = (0..n).map(|j| rm1[i][j] * b2[j]).sum();
            out_b.push(b1[i] + r1_b2 + nu12[i]);
        }
        (out_b, self.mul_idx(r1, r2))
    }

    /// Seitz inverse `(b | r)^(-1) = (-R^-1*b - R^-1*nu(r, r^-1) | r^-1)`.
    pub fn seitz_inv(&self, b: &[i64], r: usize) -> (Vec<i64>, usize) {
        let n = self.dim;
        let ri = self.inv_idx[r];
        let nur = self.nu_at(r, ri);
        let r_inv = &self.p_ops[ri];
        let mut out_b = Vec::with_capacity(n);
        for r_row in r_inv {
            let s: i64 = r_row.iter().zip(b).zip(nur).map(|((&ri, &bi), &ni)| ri * (bi + ni)).sum();
            out_b.push(-s);
        }
        (out_b, ri)
    }

    /// Seitz conjugation `h * g * h^(-1)`.
    pub fn seitz_conj(&self, bh: &[i64], rh: usize, bg: &[i64], rg: usize) -> (Vec<i64>, usize) {
        let (b_hg, r_hg) = self.seitz_mul(bh, rh, bg, rg);
        let (bh_inv, rh_inv) = self.seitz_inv(bh, rh);
        self.seitz_mul(&b_hg, r_hg, &bh_inv, rh_inv)
    }

    /// Exact `H^2(P, Z^n)` invariant factors and `[nu] == 0` splitness test via normalized `d1`.
    pub fn compute_cohomology_h2(&self) -> CohomologyH2 {
        let p_len = self.p_ops.len();
        if p_len <= 1 {
            return CohomologyH2 { torsion: Vec::new(), is_split: true };
        }
        let n = self.dim;
        let m1 = p_len - 1;
        let n1 = n * m1;
        let n2 = n * m1 * m1;
        let mut d1 = IntMat::zeros(n2, n1);
        let mut nu_vec = vec![0_i128; n2];
        let mut row_base = 0usize;
        for i in 1..p_len {
            let i0 = i - 1;
            let r1 = &self.p_ops[i];
            for j in 1..p_len {
                let j0 = j - 1;
                let ij = self.mul_idx(i, j);
                let val = self.nu_at(i, j);
                for p in 0..n {
                    nu_vec[row_base + p] = i128::from(val[p]);
                    d1.add_assign(row_base + p, i0 * n + p, 1);
                    for (q, &rq) in r1[p].iter().enumerate() {
                        d1.add_assign(row_base + p, j0 * n + q, i128::from(rq));
                    }
                    if ij != 0 {
                        d1.add_assign(row_base + p, (ij - 1) * n + p, -1);
                    }
                }
                row_base += n;
            }
        }
        let torsion: Vec<i128> =
            invariant_factors_compressed(&d1).into_iter().filter(|&d| d > 1).collect();
        let aug = d1.hstack_col(&nu_vec);
        let comp_aug = hnf_compress_rows(&aug);
        let mut comp_a = IntMat::zeros(comp_aug.rows, n1);
        let mut comp_b = Vec::with_capacity(comp_aug.rows);
        for r in 0..comp_aug.rows {
            for c in 0..n1 {
                comp_a.set(r, c, comp_aug.get(r, c));
            }
            comp_b.push(comp_aug.get(r, n1));
        }
        let sol = solve_int_linear(&comp_a, &comp_b);
        debug_assert!(sol.as_ref().is_none_or(|x| d1.mul_vec(x) == nu_vec));
        CohomologyH2 { torsion, is_split: sol.is_some() }
    }

    /// Universal abelianization `(Lambda (+) P_sub)^ab` on `Z^n (+) Z^{|P_sub|-1}`.
    pub fn compute_abelianization(&self, sub_indices: &[usize]) -> AbelianizationData {
        let n = self.dim;
        let non_id: Vec<usize> = sub_indices.iter().copied().filter(|&i| i != 0).collect();
        let mut non_id_pos = HashMap::with_capacity(non_id.len());
        for (pos, &r_idx) in non_id.iter().enumerate() {
            non_id_pos.insert(r_idx, pos);
        }
        let num_gens = n + non_id.len();
        let mut rel_rows: Vec<Vec<i128>> = Vec::new();
        for &r_idx in sub_indices {
            let r_mat = &self.p_ops[r_idx];
            for j in 0..n {
                let mut row = vec![0_i128; num_gens];
                let mut nonzero = false;
                for (i, r_row) in r_mat.iter().enumerate() {
                    let v = i128::from(if i == j { 1 } else { 0 } - r_row[j]);
                    row[i] = v;
                    if v != 0 {
                        nonzero = true;
                    }
                }
                if nonzero {
                    rel_rows.push(row);
                }
            }
        }
        for &i1 in &non_id {
            for &i2 in &non_id {
                let i12 = self.mul_idx(i1, i2);
                let nu12 = self.nu_at(i1, i2);
                let mut row = vec![0_i128; num_gens];
                for i in 0..n {
                    row[i] = -i128::from(nu12[i]);
                }
                row[n + non_id_pos[&i1]] += 1;
                row[n + non_id_pos[&i2]] += 1;
                if i12 != 0
                    && let Some(&pos12) = non_id_pos.get(&i12)
                {
                    row[n + pos12] -= 1;
                }
                if row.iter().any(|&v| v != 0) {
                    rel_rows.push(row);
                }
            }
        }
        if rel_rows.is_empty() {
            let keep_slots = (0..num_gens).map(|i| (i, 0_i128)).collect();
            return AbelianizationData {
                group: AbelianGroup { torsion: Vec::new(), free_rank: num_gens },
                u_mat: IntMat::eye(num_gens),
                keep_slots,
                non_id_pos,
                dim: n,
            };
        }
        let m_rels = IntMat::from_rows(&rel_rows);
        let h_row = hnf_compress_rows(&m_rels);
        let h_col = h_row.transpose();
        let snf = smith_normal_form(&h_col, true);
        let mut keep_slots = Vec::new();
        let mut torsion = Vec::new();
        let mut free_rank = 0usize;
        for (i, d_val) in (0..num_gens).map(|k| (k, snf.diag.get(k).copied().unwrap_or(0))) {
            if d_val != 1 {
                keep_slots.push((i, d_val));
                if d_val > 1 {
                    torsion.push(d_val);
                } else {
                    free_rank += 1;
                }
            }
        }
        AbelianizationData {
            group: AbelianGroup { torsion, free_rank },
            u_mat: snf.u,
            keep_slots,
            non_id_pos,
            dim: n,
        }
    }

    /// Construct `(1 -> Z^n -> G -> P -> 1)` projector `Pi_par = (1/|R|) sum_{k=0}^{ord-1} R^k`.
    pub(crate) fn projector_par(&self, r_idx: usize) -> RMat {
        let ord = self.mat_order(r_idx);
        let n = self.dim;
        let mut acc = vec![vec![Rat::ZERO; n]; n];
        let mut cur = 0usize;
        for _ in 0..ord {
            for i in 0..n {
                for j in 0..n {
                    acc[i][j] = acc[i][j].add(Rat::from_int(self.p_ops[cur][i][j]));
                }
            }
            cur = self.mul_idx(cur, r_idx);
        }
        let inv_ord = Rat::new(1, ord as i64);
        for row in &mut acc {
            for val in row {
                *val = val.mul(inv_ord);
            }
        }
        acc
    }

    pub fn rmat_identity_n(&self) -> RMat {
        rmat_identity(self.dim)
    }
}
