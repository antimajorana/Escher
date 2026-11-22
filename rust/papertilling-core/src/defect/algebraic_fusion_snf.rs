//! Exact integer Smith normal form, rational 2-vectors, canonical 2D sublattice cosets,
//! and finitely generated abelian group presentation helpers.

use std::collections::BTreeSet;

use crate::defect::sym_helper::mat2_vec;

pub(crate) const ID2: [[i64; 2]; 2] = [[1, 0], [0, 1]];

/// Result of exact integer Smith normal form `D = U * M * V` with unimodular `U`, `V`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SnfResult {
    pub diag: Vec<i64>,
    pub u: Vec<Vec<i64>>,
    pub v: Vec<Vec<i64>>,
}

pub(crate) fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 { let r = a % b; a = b; b = r; }
    a
}

pub(crate) fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 { 0 } else { (a / gcd_i64(a, b)).abs() * b.abs() }
}

pub(crate) fn mat_mul_i64(a: [[i64; 2]; 2], b: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

pub(crate) fn mat_vec_i64(m: [[i64; 2]; 2], v: [i64; 2]) -> [i64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

pub(crate) fn det_i64(m: [[i64; 2]; 2]) -> i64 { m[0][0] * m[1][1] - m[0][1] * m[1][0] }

pub(crate) fn inv_i64(m: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
    let d = det_i64(m);
    debug_assert!(d.abs() == 1);
    [[m[1][1] * d, -m[0][1] * d], [-m[1][0] * d, m[0][0] * d]]
}

pub(crate) fn i_minus(r: [[i64; 2]; 2]) -> [[i64; 2]; 2] { [[1 - r[0][0], -r[0][1]], [-r[1][0], 1 - r[1][1]]] }
pub(crate) fn i_plus(r: [[i64; 2]; 2]) -> [[i64; 2]; 2] { [[1 + r[0][0], r[0][1]], [r[1][0], 1 + r[1][1]]] }

pub(crate) fn rot_order(r: [[i64; 2]; 2]) -> usize {
    let mut cur = ID2;
    for k in 1..=6 {
        cur = mat_mul_i64(cur, r);
        if cur == ID2 { return k; }
    }
    unreachable!("2D crystallographic proper rotation order divides 4 or 6")
}

pub(crate) fn reflection_axis(r: [[i64; 2]; 2]) -> [i64; 2] {
    let im = i_minus(r);
    let row = if im[0] != [0, 0] { im[0] } else { im[1] };
    let g = gcd_i64(row[0], row[1]).max(1);
    let mut u = [-row[1] / g, row[0] / g];
    if u[0] < 0 || (u[0] == 0 && u[1] < 0) { u = [-u[0], -u[1]]; }
    u
}

fn div_round_i128(a: i128, b: i128) -> i128 {
    let (q, r) = (a / b, a % b);
    if 2 * r.abs() > b.abs() { if (a ^ b) >= 0 { q + 1 } else { q - 1 } } else { q }
}

fn identity_i128(n: usize) -> Vec<Vec<i128>> {
    (0..n).map(|i| { let mut r = vec![0i128; n]; r[i] = 1; r }).collect()
}

fn swap_cols(m: &mut [Vec<i128>], c1: usize, c2: usize) {
    if c1 != c2 { for row in m.iter_mut() { row.swap(c1, c2); } }
}

fn add_row_mul(m: &mut [Vec<i128>], dst: usize, src: usize, factor: i128) {
    if factor == 0 { return; }
    for j in 0..m[src].len() { let val = m[src][j]; m[dst][j] += factor * val; }
}

fn add_col_mul(m: &mut [Vec<i128>], dst: usize, src: usize, factor: i128) {
    if factor == 0 { return; }
    for row in m.iter_mut() { let val = row[src]; row[dst] += factor * val; }
}

/// Compute exact integer Smith normal form `D = U * M * V` using `i128` intermediates.
pub(crate) fn smith_normal_form(mat: &[Vec<i64>]) -> SnfResult {
    let nrows = mat.len();
    let ncols = mat.first().map_or(0, Vec::len);
    let mut a: Vec<Vec<i128>> = mat.iter().map(|r| r.iter().map(|&x| i128::from(x)).collect()).collect();
    let (mut u, mut v) = (identity_i128(nrows), identity_i128(ncols));
    let limit = nrows.min(ncols);
    for k in 0..limit {
        loop {
            let mut best: Option<(i128, usize, usize)> = None;
            for (i, row) in a.iter().enumerate().skip(k) {
                for (j, &val) in row.iter().enumerate().skip(k) {
                    if val != 0 && best.is_none_or(|(bm, _, _)| val.abs() < bm) {
                        best = Some((val.abs(), i, j));
                    }
                }
            }
            let Some((_, pi, pj)) = best else { break };
            a.swap(k, pi); u.swap(k, pi);
            swap_cols(&mut a, k, pj); swap_cols(&mut v, k, pj);
            loop {
                let mut changed = false;
                for i in (k + 1)..nrows {
                    if a[i][k] != 0 {
                        let q = div_round_i128(a[i][k], a[k][k]);
                        add_row_mul(&mut a, i, k, -q); add_row_mul(&mut u, i, k, -q);
                        if a[i][k] != 0 { a.swap(k, i); u.swap(k, i); changed = true; break; }
                    }
                }
                if changed { continue; }
                for j in (k + 1)..ncols {
                    if a[k][j] != 0 {
                        let q = div_round_i128(a[k][j], a[k][k]);
                        add_col_mul(&mut a, j, k, -q); add_col_mul(&mut v, j, k, -q);
                        if a[k][j] != 0 { swap_cols(&mut a, k, j); swap_cols(&mut v, k, j); changed = true; break; }
                    }
                }
                if !changed { break; }
            }
            if a[k][k] < 0 {
                for x in &mut a[k] { *x = -*x; }
                for x in &mut u[k] { *x = -*x; }
            }
            let piv = a[k][k];
            let bad_row = a.iter().enumerate().skip(k + 1)
                .find_map(|(i, row)| row.iter().skip(k + 1).any(|&val| val % piv != 0).then_some(i));
            if let Some(i) = bad_row {
                add_row_mul(&mut a, k, i, 1); add_row_mul(&mut u, k, i, 1);
            } else { break; }
        }
        if a[k][k] == 0 { break; }
    }
    let diag = (0..limit).map(|k| i64::try_from(a[k][k]).expect("SNF diag fits in i64")).collect();
    let conv = |m: Vec<Vec<i128>>| -> Vec<Vec<i64>> {
        m.into_iter().map(|r| r.into_iter().map(|x| i64::try_from(x).expect("SNF factor fits in i64")).collect()).collect()
    };
    SnfResult { diag, u: conv(u), v: conv(v) }
}

/// Test whether `M * x = rhs` has an exact integer solution `x in Z^ncols`.
pub(crate) fn solve_int_system_exists(mat: &[Vec<i64>], rhs: &[i64]) -> bool {
    let snf = smith_normal_form(mat);
    for (i, u_row) in snf.u.iter().enumerate() {
        let wi: i128 = u_row.iter().zip(rhs).map(|(&uij, &rj)| i128::from(uij) * i128::from(rj)).sum();
        let d = snf.diag.get(i).copied().unwrap_or(0);
        if (d == 0 && wi != 0) || (d != 0 && wi % i128::from(d) != 0) { return false; }
    }
    true
}

/// Extract `(free_rank, torsion_factors_gt_1)` for `Z^nrows / im(M)` from SNF diagonal `diag`.
pub(crate) fn invariant_factors_from_snf(nrows: usize, diag: &[i64]) -> (usize, Vec<i64>) {
    let rank = diag.iter().filter(|&&d| d != 0).count();
    (nrows.saturating_sub(rank), diag.iter().copied().filter(|&d| d > 1).collect())
}

/// Format `Z^{free_rank} (+) (+) Z/d_j Z` concisely (e.g. `"0"`, `"Z^2"`, `"(Z/2Z)^3"`, `"Z x Z/2Z"`).
pub(crate) fn format_abelian_group(free_rank: usize, torsion: &[i64]) -> String {
    let mut parts = Vec::new();
    match free_rank { 0 => {} 1 => parts.push("Z".to_string()), r => parts.push(format!("Z^{r}")) }
    let mut i = 0;
    while i < torsion.len() {
        let d = torsion[i];
        let mut j = i + 1;
        while j < torsion.len() && torsion[j] == d { j += 1; }
        let cnt = j - i;
        parts.push(if cnt == 1 { format!("Z/{d}Z") } else { format!("(Z/{d}Z)^{cnt}") });
        i = j;
    }
    if parts.is_empty() { "0".to_string() } else { parts.join(" x ") }
}

/// Exact GCD-reduced rational 2-vector `num / den` (`den > 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Frac2 {
    pub num: [i64; 2],
    pub den: i64,
}

impl Frac2 {
    pub(crate) fn new(num: [i64; 2], den: i64) -> Self {
        assert!(den != 0, "Frac2 denominator must be nonzero");
        let (mut n0, mut n1, mut d) = (num[0], num[1], den);
        if d < 0 { n0 = -n0; n1 = -n1; d = -d; }
        let g = gcd_i64(gcd_i64(n0, n1), d);
        Self { num: [n0 / g, n1 / g], den: d / g }
    }
    pub(crate) fn from_int(v: [i64; 2]) -> Self { Self { num: v, den: 1 } }
    pub(crate) fn add(self, other: Self) -> Self {
        let d = lcm_i64(self.den, other.den);
        let (s1, s2) = (d / self.den, d / other.den);
        Self::new([self.num[0] * s1 + other.num[0] * s2, self.num[1] * s1 + other.num[1] * s2], d)
    }
    pub(crate) fn sub(self, other: Self) -> Self {
        let d = lcm_i64(self.den, other.den);
        let (s1, s2) = (d / self.den, d / other.den);
        Self::new([self.num[0] * s1 - other.num[0] * s2, self.num[1] * s1 - other.num[1] * s2], d)
    }
    pub(crate) fn mat_mul(self, m: [[i64; 2]; 2]) -> Self { Self::new(mat_vec_i64(m, self.num), self.den) }
    pub(crate) fn div_scalar(self, k: i64) -> Self { Self::new(self.num, self.den * k) }
    pub(crate) fn mod_int(self) -> Self {
        Self::new([self.num[0].rem_euclid(self.den), self.num[1].rem_euclid(self.den)], self.den)
    }
    pub(crate) fn to_int_exact(self) -> Option<[i64; 2]> { (self.den == 1).then_some(self.num) }
    pub(crate) fn to_f64(self) -> [f64; 2] { [self.num[0] as f64 / self.den as f64, self.num[1] as f64 / self.den as f64] }
    pub(crate) fn from_f64_quantized(v: [f64; 2], tol: f64) -> Option<Self> {
        for den in [12i64, 24, 36] {
            let n0 = (v[0] * den as f64).round_ties_even() as i64;
            let n1 = (v[1] * den as f64).round_ties_even() as i64;
            if (v[0] - n0 as f64 / den as f64).abs() < tol && (v[1] - n1 as f64 / den as f64).abs() < tol {
                return Some(Self::new([n0, n1], den));
            }
        }
        None
    }
}

/// Canonical 2-row column HNF of `K = span_Z(cols) <= Z^2`.
/// Rank 2 returns `[[h00, 0], [h01, h11]]` with `h00 > 0, h11 > 0, 0 <= h01 < h00`.
pub(crate) fn hnf_2d_sublattice(cols: &[[i64; 2]]) -> Vec<[i64; 2]> {
    let mut row1_col: Option<[i64; 2]> = None;
    let mut row0_gcd = 0i64;
    for &raw in cols {
        if raw == [0, 0] { continue; }
        let mut v = if raw[1] < 0 { [-raw[0], -raw[1]] } else { raw };
        if v[1] == 0 { row0_gcd = gcd_i64(row0_gcd, v[0]); continue; }
        match row1_col {
            None => row1_col = Some(v),
            Some(mut cur) => {
                while v[1] != 0 {
                    let q = cur[1].div_euclid(v[1]);
                    let rem = [cur[0] - q * v[0], cur[1] - q * v[1]];
                    cur = v; v = rem;
                }
                row0_gcd = gcd_i64(row0_gcd, v[0]);
                row1_col = Some(cur);
            }
        }
    }
    match (row0_gcd > 0, row1_col) {
        (true, Some(mut c1)) => { c1[0] = c1[0].rem_euclid(row0_gcd); vec![[row0_gcd, 0], c1] }
        (true, None) => vec![[row0_gcd, 0]],
        (false, Some(c1)) => vec![c1],
        (false, None) => Vec::new(),
    }
}

/// Canonical coset `offset + span_Z(basis_cols)` inside `Z^2`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LatticeCoset {
    pub offset: [i64; 2],
    pub basis_cols: Vec<[i64; 2]>,
}

impl LatticeCoset {
    pub(crate) fn new(offset: [i64; 2], cols: &[[i64; 2]]) -> Self {
        let mut out = Self { offset, basis_cols: hnf_2d_sublattice(cols) };
        out.offset = out.reduce_vec(offset);
        out
    }
    pub(crate) fn index(&self) -> Option<i64> {
        (self.basis_cols.len() == 2).then(|| self.basis_cols[0][0] * self.basis_cols[1][1])
    }
    pub(crate) fn reduce_vec(&self, v: [i64; 2]) -> [i64; 2] {
        if self.basis_cols.len() == 2 {
            let (h00, h01, h11) = (self.basis_cols[0][0], self.basis_cols[1][0], self.basis_cols[1][1]);
            let q1 = v[1].div_euclid(h11);
            [(v[0] - q1 * h01).rem_euclid(h00), v[1] - q1 * h11]
        } else { v }
    }
    pub(crate) fn contains(&self, v: [i64; 2]) -> bool {
        let d = [v[0] - self.offset[0], v[1] - self.offset[1]];
        match self.basis_cols.len() {
            0 => d == [0, 0],
            1 => {
                let c = self.basis_cols[0];
                d[0] * c[1] == d[1] * c[0] && if c[0] != 0 { d[0] % c[0] == 0 } else { d[1] % c[1] == 0 }
            }
            _ => self.reduce_vec(d) == [0, 0],
        }
    }
    pub(crate) fn describe(&self) -> String {
        let Some(idx) = self.index() else {
            return format!("({}, {}) + rank-{} sublattice", self.offset[0], self.offset[1], self.basis_cols.len());
        };
        let (h00, h01, h11, z) = (self.basis_cols[0][0], self.basis_cols[1][0], self.basis_cols[1][1], self.offset == [0, 0]);
        if idx == 1 { return "all lattice vectors L".to_string(); }
        if idx == 2 {
            return match (h00, h01, h11, z) {
                (2, 1, 1, true) => "index-2 sublattice (b1+b2 = 0 mod 2)".to_string(),
                (2, 1, 1, false) => "index-2 coset (b1+b2 = 1 mod 2)".to_string(),
                (2, 0, 1, true) => "index-2 sublattice (b1 = 0 mod 2)".to_string(),
                (2, 0, 1, false) => "index-2 coset (b1 = 1 mod 2)".to_string(),
                (1, 0, 2, true) => "index-2 sublattice (b2 = 0 mod 2)".to_string(),
                (1, 0, 2, false) => "index-2 coset (b2 = 1 mod 2)".to_string(),
                _ => format!("index-2 coset ({}, {})", self.offset[0], self.offset[1]),
            };
        }
        if idx == 4 && h00 == 2 && h01 == 0 && h11 == 2 {
            return if z { "2L (even coordinates)".to_string() } else { format!("({}, {}) + 2L", self.offset[0], self.offset[1]) };
        }
        if z { format!("index-{idx} sublattice") } else { format!("({}, {}) + index-{idx} sublattice", self.offset[0], self.offset[1]) }
    }
}

/// Finite union of cosets of a canonical rank-2 sublattice, coalesced by its translation stabilizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LatticeCosetUnion {
    pub basis_cols: Vec<[i64; 2]>,
    pub offsets: Vec<[i64; 2]>,
}

impl LatticeCosetUnion {
    pub(crate) fn from_sublattice_and_offsets(cols: &[[i64; 2]], raw_offsets: &[[i64; 2]]) -> Self {
        let base = LatticeCoset::new([0, 0], cols);
        let mut set: BTreeSet<[i64; 2]> = raw_offsets.iter().map(|&v| base.reduce_vec(v)).collect();
        if base.basis_cols.len() == 2 && set.len() > 1 {
            let (h00, h11) = (base.basis_cols[0][0], base.basis_cols[1][1]);
            let mut stab_cols = base.basis_cols.clone();
            for r0 in 0..h00 {
                for r1 in 0..h11 {
                    if (r0, r1) == (0, 0) { continue; }
                    let shifted: BTreeSet<[i64; 2]> = set.iter().map(|&o| base.reduce_vec([o[0] + r0, o[1] + r1])).collect();
                    if shifted == set { stab_cols.push([r0, r1]); }
                }
            }
            let coalesced = LatticeCoset::new([0, 0], &stab_cols);
            set = set.into_iter().map(|v| coalesced.reduce_vec(v)).collect();
            return Self { basis_cols: coalesced.basis_cols, offsets: set.into_iter().collect() };
        }
        Self { basis_cols: base.basis_cols, offsets: set.into_iter().collect() }
    }
    pub(crate) fn index(&self) -> Option<i64> {
        (self.basis_cols.len() == 2).then(|| self.basis_cols[0][0] * self.basis_cols[1][1])
    }
    pub(crate) fn contains(&self, v: [i64; 2]) -> bool {
        if self.basis_cols.len() == 2 {
            let base = LatticeCoset { offset: [0, 0], basis_cols: self.basis_cols.clone() };
            self.offsets.contains(&base.reduce_vec(v))
        } else {
            self.offsets.iter().any(|&o| LatticeCoset { offset: o, basis_cols: self.basis_cols.clone() }.contains(v))
        }
    }
    pub(crate) fn contains_zero(&self) -> bool { self.contains([0, 0]) }
    pub(crate) fn describe(&self) -> String {
        if self.offsets.len() == 1 {
            return LatticeCoset { offset: self.offsets[0], basis_cols: self.basis_cols.clone() }.describe();
        }
        let Some(idx) = self.index() else { return format!("union of {} cosets", self.offsets.len()); };
        let (h00, h01, h11) = (self.basis_cols[0][0], self.basis_cols[1][0], self.basis_cols[1][1]);
        if !self.contains_zero() && self.offsets.len() as i64 == idx - 1 {
            if idx == 4 && h00 == 2 && h01 == 0 && h11 == 2 {
                return "L \\ 2L (odd parity mod 2; 3 of 4 index-4 cosets)".to_string();
            }
            return format!("L \\ index-{idx} sublattice ({} of {idx} nonzero cosets)", self.offsets.len());
        }
        format!("union of {} of {idx} cosets of index-{idx} sublattice", self.offsets.len())
    }
}

/// Exact element of a finitely generated abelian group in SNF coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AbelianElement {
    pub free: Vec<i64>,
    pub torsion: Vec<(i64, i64)>,
}

/// Exact homomorphism `Z^{2+m} -> G^ab` computed from Smith normal form of the presentation matrix.
#[derive(Clone, Debug)]
pub(crate) struct AbelianMap {
    pub free_rank: usize,
    pub torsion: Vec<i64>,
    pub label: String,
    u: Vec<Vec<i64>>,
    free_rows: Vec<usize>,
    torsion_rows: Vec<(usize, i64)>,
}

impl AbelianMap {
    pub(crate) fn from_presentation(nrows: usize, cols: &[Vec<i64>]) -> Self {
        let mut mat = vec![vec![0i64; cols.len()]; nrows];
        for (j, col) in cols.iter().enumerate() {
            for (i, &val) in col.iter().enumerate() { mat[i][j] = val; }
        }
        let snf = smith_normal_form(&mat);
        let (free_rank, torsion) = invariant_factors_from_snf(nrows, &snf.diag);
        let label = format_abelian_group(free_rank, &torsion);
        let (mut free_rows, mut torsion_rows) = (Vec::new(), Vec::new());
        for i in 0..nrows {
            let d = snf.diag.get(i).copied().unwrap_or(0);
            if d == 0 { free_rows.push(i); } else if d > 1 { torsion_rows.push((i, d)); }
        }
        Self { free_rank, torsion, label, u: snf.u, free_rows, torsion_rows }
    }
    pub(crate) fn from_space_group(
        n_ops: usize,
        matrices: &[[[i64; 2]; 2]],
        mult: &[Vec<usize>],
        nu: &[Vec<[i64; 2]>],
    ) -> Self {
        let nrows = 2 + n_ops;
        let mut cols = Vec::new();
        let mut c0 = vec![0i64; nrows];
        c0[2] = 1;
        cols.push(c0);
        for &r in matrices.iter().take(n_ops) {
            for c in 0..2 {
                let mut col = vec![0i64; nrows];
                col[0] = r[0][c] - i64::from(c == 0);
                col[1] = r[1][c] - i64::from(c == 1);
                if col[0] != 0 || col[1] != 0 { cols.push(col); }
            }
        }
        for i in 0..n_ops {
            for j in 0..n_ops {
                let mut col = vec![0i64; nrows];
                col[0] = -nu[i][j][0];
                col[1] = -nu[i][j][1];
                col[2 + i] += 1;
                col[2 + j] += 1;
                col[2 + mult[i][j]] -= 1;
                if col.iter().any(|&x| x != 0) { cols.push(col); }
            }
        }
        Self::from_presentation(nrows, &cols)
    }
    pub(crate) fn zero(&self) -> AbelianElement {
        AbelianElement { free: vec![0; self.free_rows.len()], torsion: self.torsion_rows.iter().map(|&(_, d)| (0, d)).collect() }
    }
    pub(crate) fn map_op(&self, l: [i64; 2], op_idx: usize) -> AbelianElement {
        let eval = |r: usize| -> i128 {
            let row = &self.u[r];
            i128::from(row[0]) * i128::from(l[0]) + i128::from(row[1]) * i128::from(l[1]) + i128::from(row[2 + op_idx])
        };
        AbelianElement {
            free: self.free_rows.iter().map(|&r| i64::try_from(eval(r)).expect("free coord fits in i64")).collect(),
            torsion: self.torsion_rows.iter().map(|&(r, d)| (eval(r).rem_euclid(i128::from(d)) as i64, d)).collect(),
        }
    }
    pub(crate) fn add(&self, a: &AbelianElement, b: &AbelianElement) -> AbelianElement {
        AbelianElement {
            free: a.free.iter().zip(&b.free).map(|(&x, &y)| x + y).collect(),
            torsion: a.torsion.iter().zip(&b.torsion).map(|(&(x, d), &(y, _))| ((x + y).rem_euclid(d), d)).collect(),
        }
    }
    pub(crate) fn order_of(&self, e: &AbelianElement) -> Option<i64> {
        if e.free.iter().any(|&x| x != 0) { return None; }
        Some(e.torsion.iter().fold(1i64, |acc, &(v, d)| lcm_i64(acc, d / gcd_i64(v, d))))
    }
    pub(crate) fn format_elem(&self, e: &AbelianElement) -> String {
        if self.label == "0" { return "0".to_string(); }
        if e.free.iter().all(|&x| x == 0) && e.torsion.iter().all(|&(v, _)| v == 0) {
            return format!("0 in {}", self.label);
        }
        let mut coords: Vec<String> = e.free.iter().map(ToString::to_string).collect();
        coords.extend(e.torsion.iter().map(|&(v, _)| v.to_string()));
        format!("({}) in {}", coords.join(", "), self.label)
    }
}

pub(crate) fn cart_norm_sq(a: [[f64; 2]; 2], m: [i64; 2]) -> f64 {
    let v = mat2_vec(a, [m[0] as f64, m[1] as f64]);
    v[0] * v[0] + v[1] * v[1]
}

pub(crate) fn shortest_nonzero_in_coset(coset: &LatticeCoset, a: [[f64; 2]; 2]) -> Option<[i64; 2]> {
    let lim = coset.index().unwrap_or(4).max(4) + 2;
    let mut best: Option<(f64, [i64; 2])> = None;
    for m0 in -lim..=lim {
        for m1 in -lim..=lim {
            if (m0, m1) != (0, 0) && coset.contains([m0, m1]) {
                let n2 = cart_norm_sq(a, [m0, m1]);
                if best.is_none_or(|(bn, _)| n2 < bn - 1e-9) { best = Some((n2, [m0, m1])); }
            }
        }
    }
    best.map(|(_, v)| v)
}

pub(crate) fn shortest_nonzero_in_union(union: &LatticeCosetUnion, a: [[f64; 2]; 2]) -> Vec<[i64; 2]> {
    let lim = union.index().unwrap_or(4).max(4) + 2;
    let (mut best_n2, mut cands) = (f64::INFINITY, Vec::new());
    for m0 in -lim..=lim {
        for m1 in -lim..=lim {
            if (m0, m1) != (0, 0) && union.contains([m0, m1]) {
                let n2 = cart_norm_sq(a, [m0, m1]);
                if n2 < best_n2 + 1e-7 * best_n2.max(1.0) {
                    if n2 < best_n2 { best_n2 = n2; }
                    cands.push((n2, [m0, m1]));
                }
            }
        }
    }
    let mut out: Vec<[i64; 2]> = cands.into_iter()
        .filter(|&(n2, _)| (n2 - best_n2).abs() <= 1e-7 * best_n2.max(1.0))
        .map(|(_, v)| v).collect();
    out.sort_by_key(|&v| (-v[0], -v[1]));
    out
}

pub(crate) fn shortest_nonzero_in_set(set: &BTreeSet<[i64; 2]>, a: [[f64; 2]; 2]) -> Vec<[i64; 2]> {
    let best_n2 = set.iter().copied().filter(|&v| v != [0, 0]).map(|v| cart_norm_sq(a, v)).fold(f64::INFINITY, f64::min);
    if !best_n2.is_finite() { return Vec::new(); }
    let mut out: Vec<[i64; 2]> = set.iter().copied()
        .filter(|&v| v != [0, 0] && (cart_norm_sq(a, v) - best_n2).abs() <= 1e-7 * best_n2.max(1.0))
        .collect();
    out.sort_by_key(|&v| (-v[0], -v[1]));
    out
}
