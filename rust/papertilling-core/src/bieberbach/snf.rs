//! Exact arbitrary-size rectangular integer Hermite Normal Form (HNF) and
//! Smith Normal Form (SNF) over `i128` for dimension-n Bieberbach cohomology,
//! abelianizations, and space-group conjugacy sector calculations.

use std::cmp::Ordering;

pub(crate) fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Extended Euclidean algorithm: returns `(g, x, y)` with `g >= 0` and `x*a + y*b == g`.
pub(crate) fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1_i128, 0_i128);
    let (mut old_t, mut t) = (0_i128, 1_i128);
    while r != 0 {
        let q = old_r.div_euclid(r);
        let nr = old_r - q * r;
        let ns = old_s - q * s;
        let nt = old_t - q * t;
        old_r = r;
        r = nr;
        old_s = s;
        s = ns;
        old_t = t;
        t = nt;
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// Dense rectangular integer matrix stored in row-major `Vec<i128>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i128>,
}

impl IntMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0; rows * cols] }
    }

    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1);
        }
        m
    }

    pub fn from_rows(rows_vec: &[Vec<i128>]) -> Self {
        if rows_vec.is_empty() {
            return Self::zeros(0, 0);
        }
        let rows = rows_vec.len();
        let cols = rows_vec[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for r in rows_vec {
            debug_assert_eq!(r.len(), cols);
            data.extend_from_slice(r);
        }
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> i128 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: i128) {
        self.data[r * self.cols + c] = val;
    }

    #[inline]
    pub fn add_assign(&mut self, r: usize, c: usize, delta: i128) {
        self.data[r * self.cols + c] += delta;
    }

    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    pub fn hstack_col(&self, col: &[i128]) -> Self {
        debug_assert_eq!(col.len(), self.rows);
        let mut out = Self::zeros(self.rows, self.cols + 1);
        for (r, &v) in col.iter().enumerate() {
            for c in 0..self.cols {
                out.set(r, c, self.get(r, c));
            }
            out.set(r, self.cols, v);
        }
        out
    }

    pub fn mul_mat(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.cols, rhs.rows);
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.get(i, k);
                if aik == 0 {
                    continue;
                }
                for j in 0..rhs.cols {
                    out.data[i * rhs.cols + j] += aik * rhs.get(k, j);
                }
            }
        }
        out
    }

    pub fn mul_vec(&self, v: &[i128]) -> Vec<i128> {
        debug_assert_eq!(self.cols, v.len());
        let mut out = vec![0_i128; self.rows];
        for (i, o) in out.iter_mut().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            *o = row.iter().zip(v).map(|(&a, &b)| a * b).sum();
        }
        out
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        for c in 0..self.cols {
            let i1 = r1 * self.cols + c;
            let i2 = r2 * self.cols + c;
            self.data.swap(i1, i2);
        }
    }

    fn swap_cols(&mut self, c1: usize, c2: usize) {
        if c1 == c2 {
            return;
        }
        for r in 0..self.rows {
            self.data.swap(r * self.cols + c1, r * self.cols + c2);
        }
    }

    fn negate_row(&mut self, r: usize) {
        let base = r * self.cols;
        for c in 0..self.cols {
            self.data[base + c] = -self.data[base + c];
        }
    }

    fn add_row_mul_from(&mut self, dst: usize, src: usize, factor: i128, start_c: usize) {
        if factor == 0 {
            return;
        }
        let d_base = dst * self.cols;
        let s_base = src * self.cols;
        for c in start_c..self.cols {
            let sv = self.data[s_base + c];
            if sv != 0 {
                self.data[d_base + c] += factor * sv;
            }
        }
    }

    fn add_row_mul(&mut self, dst: usize, src: usize, factor: i128) {
        self.add_row_mul_from(dst, src, factor, 0);
    }

    fn add_col_mul(&mut self, dst: usize, src: usize, factor: i128) {
        if factor == 0 {
            return;
        }
        for r in 0..self.rows {
            let sv = self.data[r * self.cols + src];
            if sv != 0 {
                self.data[r * self.cols + dst] += factor * sv;
            }
        }
    }

    fn combine_rows(&mut self, r1: usize, r2: usize, a: i128, b: i128, c: i128, d: i128) {
        let b1 = r1 * self.cols;
        let b2 = r2 * self.cols;
        for j in 0..self.cols {
            let x = self.data[b1 + j];
            let y = self.data[b2 + j];
            self.data[b1 + j] = a * x + b * y;
            self.data[b2 + j] = c * x + d * y;
        }
    }

    fn combine_cols(&mut self, c1: usize, c2: usize, a: i128, b: i128, c: i128, d: i128) {
        for r in 0..self.rows {
            let x = self.data[r * self.cols + c1];
            let y = self.data[r * self.cols + c2];
            self.data[r * self.cols + c1] = a * x + b * y;
            self.data[r * self.cols + c2] = c * x + d * y;
        }
    }
}

/// Unimodular row Hermite reduction compressing a tall matrix to its nonzero rows (`<= cols`).
/// Preserves exact column-lattice invariant factors (SNF of `M` == SNF of compressed rows).
pub fn hnf_compress_rows(m: &IntMat) -> IntMat {
    if m.rows == 0 || m.cols == 0 {
        return m.clone();
    }
    if m.rows <= 512 {
        return hnf_compress_rows_block(m);
    }
    let chunk = 256usize.max(m.cols);
    let mut acc: Vec<i128> = Vec::new();
    let mut acc_rows = 0usize;
    let mut row_start = 0usize;
    while row_start < m.rows {
        let take = (m.rows - row_start).min(chunk);
        acc.extend_from_slice(&m.data[row_start * m.cols..(row_start + take) * m.cols]);
        acc_rows += take;
        row_start += take;
        let block = IntMat { rows: acc_rows, cols: m.cols, data: std::mem::take(&mut acc) };
        let reduced = hnf_compress_rows_block(&block);
        acc_rows = reduced.rows;
        acc = reduced.data;
    }
    IntMat { rows: acc_rows, cols: m.cols, data: acc }
}

fn hnf_compress_rows_block(m: &IntMat) -> IntMat {
    let mut a = m.clone();
    let mut r_piv = 0usize;
    for c in 0..a.cols {
        if r_piv >= a.rows {
            break;
        }
        loop {
            let mut best_r = None;
            let mut best_abs = i128::MAX;
            for r in r_piv..a.rows {
                let v = a.get(r, c).abs();
                if v > 0 && v < best_abs {
                    best_abs = v;
                    best_r = Some(r);
                }
            }
            let Some(br) = best_r else { break };
            a.swap_rows(r_piv, br);
            if a.get(r_piv, c) < 0 {
                a.negate_row(r_piv);
            }
            let piv = a.get(r_piv, c);
            let mut changed = false;
            for r in (r_piv + 1)..a.rows {
                let val = a.get(r, c);
                if val != 0 {
                    let q = val.div_euclid(piv);
                    a.add_row_mul_from(r, r_piv, -q, c);
                    if a.get(r, c) != 0 {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if a.get(r_piv, c) != 0 {
            let piv = a.get(r_piv, c);
            for r in 0..r_piv {
                let val = a.get(r, c);
                if val != 0 {
                    let q = val.div_euclid(piv);
                    a.add_row_mul_from(r, r_piv, -q, c);
                }
            }
            r_piv += 1;
        }
    }
    let mut rows_out = Vec::with_capacity(r_piv);
    for r in 0..r_piv {
        let row = &a.data[r * a.cols..(r + 1) * a.cols];
        if row.iter().any(|&v| v != 0) {
            rows_out.push(row.to_vec());
        }
    }
    if rows_out.is_empty() {
        IntMat::zeros(0, m.cols)
    } else {
        IntMat::from_rows(&rows_out)
    }
}

/// Smith Normal Form decomposition `U * M * V = diag(diag_pos)` with `d_1 | d_2 | ... | d_r > 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnfDecomp {
    pub diag: Vec<i128>,
    pub u: IntMat,
    pub v: IntMat,
}

/// Compute exact Smith Normal Form, optionally tracking left/right unimodular matrices `U` and `V`.
pub fn smith_normal_form(m: &IntMat, track_uv: bool) -> SnfDecomp {
    let mut a = m.clone();
    let mut u = if track_uv { IntMat::eye(a.rows) } else { IntMat::zeros(0, 0) };
    let mut v = if track_uv { IntMat::eye(a.cols) } else { IntMat::zeros(0, 0) };
    let kmax = a.rows.min(a.cols);
    let mut t = 0usize;

    while t < kmax {
        // Choose smallest nonzero entry in submatrix [t.., t..]
        let mut best: Option<(usize, usize, i128)> = None;
        for r in t..a.rows {
            for c in t..a.cols {
                let val = a.get(r, c).abs();
                if val > 0 && best.is_none_or(|b| val < b.2) {
                    best = Some((r, c, val));
                }
            }
        }
        let Some((br, bc, _)) = best else { break };
        a.swap_rows(t, br);
        a.swap_cols(t, bc);
        if track_uv {
            u.swap_rows(t, br);
            v.swap_cols(t, bc);
        }
        if a.get(t, t) < 0 {
            a.negate_row(t);
            if track_uv {
                u.negate_row(t);
            }
        }

        loop {
            let mut touched = false;
            // Clear row t and col t via Bezout / Euclidean steps
            for r in (t + 1)..a.rows {
                let val = a.get(r, t);
                if val == 0 {
                    continue;
                }
                touched = true;
                let piv = a.get(t, t);
                if val % piv == 0 {
                    let q = val / piv;
                    a.add_row_mul(r, t, -q);
                    if track_uv {
                        u.add_row_mul(r, t, -q);
                    }
                } else {
                    let (g, p, q) = ext_gcd(piv, val);
                    let a_c = -val / g;
                    let d_c = piv / g;
                    a.combine_rows(t, r, p, q, a_c, d_c);
                    if track_uv {
                        u.combine_rows(t, r, p, q, a_c, d_c);
                    }
                }
            }
            for c in (t + 1)..a.cols {
                let val = a.get(t, c);
                if val == 0 {
                    continue;
                }
                touched = true;
                let piv = a.get(t, t);
                if val % piv == 0 {
                    let q = val / piv;
                    a.add_col_mul(c, t, -q);
                    if track_uv {
                        v.add_col_mul(c, t, -q);
                    }
                } else {
                    let (g, p, q) = ext_gcd(piv, val);
                    let a_c = -val / g;
                    let d_c = piv / g;
                    a.combine_cols(t, c, p, q, a_c, d_c);
                    if track_uv {
                        v.combine_cols(t, c, p, q, a_c, d_c);
                    }
                }
            }
            if !touched {
                // Ensure a(t,t) divides every remaining entry a(r, c)
                let piv = a.get(t, t);
                let mut bad_entry = None;
                'outer: for r in (t + 1)..a.rows {
                    for c in (t + 1)..a.cols {
                        let val = a.get(r, c);
                        if val % piv != 0 {
                            bad_entry = Some((r, c));
                            break 'outer;
                        }
                    }
                }
                if let Some((r, _c)) = bad_entry {
                    a.add_row_mul(t, r, 1);
                    if track_uv {
                        u.add_row_mul(t, r, 1);
                    }
                    touched = true;
                }
            }
            if a.get(t, t) < 0 {
                a.negate_row(t);
                if track_uv {
                    u.negate_row(t);
                }
            }
            if !touched {
                break;
            }
        }
        t += 1;
    }

    let mut diag = Vec::with_capacity(kmax);
    for i in 0..kmax {
        let mut d = a.get(i, i);
        if d < 0 {
            d = -d;
            if track_uv {
                u.negate_row(i);
            }
        }
        if d > 0 {
            diag.push(d);
        } else {
            break;
        }
    }
    debug_assert!(diag.windows(2).all(|w| w[1] % w[0] == 0));
    SnfDecomp { diag, u, v }
}

/// Nonzero SNF invariant factors after fast HNF row compression (for arbitrarily tall matrices).
pub fn invariant_factors_compressed(m: &IntMat) -> Vec<i128> {
    if m.rows == 0 || m.cols == 0 {
        return Vec::new();
    }
    let comp = hnf_compress_rows(m);
    if comp.rows == 0 {
        return Vec::new();
    }
    smith_normal_form(&comp, false).diag
}

/// Compute `invert_i128_unimodular(u)` for square unimodular `U` (`det = +-1`) via Gaussian elimination.
pub fn invert_unimodular(u: &IntMat) -> IntMat {
    let n = u.rows;
    debug_assert_eq!(n, u.cols);
    let mut aug = IntMat::zeros(n, 2 * n);
    for r in 0..n {
        for c in 0..n {
            aug.set(r, c, u.get(r, c));
        }
        aug.set(r, n + r, 1);
    }
    for piv_c in 0..n {
        let pr = (piv_c..n)
            .filter(|&r| aug.get(r, piv_c) != 0)
            .min_by_key(|&r| aug.get(r, piv_c).abs())
            .expect("unimodular matrix is invertible");
        aug.swap_rows(piv_c, pr);
        if aug.get(piv_c, piv_c) < 0 {
            aug.negate_row(piv_c);
        }
        loop {
            let pval = aug.get(piv_c, piv_c);
            let mut rem_found = false;
            for r in (piv_c + 1)..n {
                let rv = aug.get(r, piv_c);
                if rv != 0 {
                    let q = rv.div_euclid(pval);
                    aug.add_row_mul(r, piv_c, -q);
                    if aug.get(r, piv_c) != 0 {
                        rem_found = true;
                        let new_abs = aug.get(r, piv_c).abs();
                        if new_abs < pval {
                            aug.swap_rows(piv_c, r);
                            if aug.get(piv_c, piv_c) < 0 {
                                aug.negate_row(piv_c);
                            }
                            break;
                        }
                    }
                }
            }
            if !rem_found {
                break;
            }
        }
        debug_assert_eq!(aug.get(piv_c, piv_c), 1);
        for r in 0..piv_c {
            let rv = aug.get(r, piv_c);
            if rv != 0 {
                aug.add_row_mul(r, piv_c, -rv);
            }
        }
    }
    let mut inv = IntMat::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            inv.set(r, c, aug.get(r, n + c));
        }
    }
    inv
}

/// Solve `A * x = b` over `Z^cols`, returning `Some(x)` iff an integer solution exists.
pub fn solve_int_linear(a: &IntMat, b: &[i128]) -> Option<Vec<i128>> {
    debug_assert_eq!(a.rows, b.len());
    if a.cols == 0 {
        return b.iter().all(|&x| x == 0).then(Vec::new);
    }
    let snf = smith_normal_form(a, true);
    let y = snf.u.mul_vec(b);
    let r = snf.diag.len();
    for &yi in &y[r..] {
        if yi != 0 {
            return None;
        }
    }
    let mut z = vec![0_i128; a.cols];
    for i in 0..r {
        let di = snf.diag[i];
        match y[i].cmp(&0) {
            Ordering::Equal => z[i] = 0,
            _ if y[i] % di != 0 => return None,
            _ => z[i] = y[i] / di,
        }
    }
    Some(snf.v.mul_vec(&z))
}
