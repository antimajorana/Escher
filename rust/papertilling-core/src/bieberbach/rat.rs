//! Exact small rational (`i64 / i64`) arithmetic and `n <= 3` matrix/vector helpers.

use std::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

use super::snf::gcd_i128;

/// Exact reduced rational number `num / den` with `den > 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rat {
    pub num: i64,
    pub den: i64,
}

impl Rat {
    pub const ZERO: Self = Self { num: 0, den: 1 };
    pub const ONE: Self = Self { num: 1, den: 1 };

    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "rational denominator must be nonzero");
        if num == 0 {
            return Self::ZERO;
        }
        let g = gcd_i128(i128::from(num), i128::from(den)) as i64;
        let s = if den < 0 { -1 } else { 1 };
        Self { num: (num / g) * s, den: den.abs() / g }
    }

    pub fn from_int(v: i64) -> Self {
        Self { num: v, den: 1 }
    }

    pub fn from_f64_frac(x: f64) -> Result<Self, String> {
        for d in [1_i64, 2, 3, 4, 6, 8, 12, 24] {
            let n = (x * (d as f64)).round();
            if (x - n / (d as f64)).abs() < 1e-6 {
                return Ok(Self::new(n as i64, d));
            }
        }
        Err(format!("coordinate {x} is not a simple crystallographic rational"))
    }

    pub fn as_i64(self) -> Option<i64> {
        (self.den == 1).then_some(self.num)
    }

    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    pub fn floor_i64(self) -> i64 {
        self.num.div_euclid(self.den)
    }

    pub fn frac_unit(self) -> Self {
        Self::new(self.num.rem_euclid(self.den), self.den)
    }

    pub fn to_f64(self) -> f64 {
        (self.num as f64) / (self.den as f64)
    }
}

impl std::ops::Add for Rat {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        let num = i128::from(self.num) * i128::from(r.den) + i128::from(r.num) * i128::from(self.den);
        let den = i128::from(self.den) * i128::from(r.den);
        let g = gcd_i128(num, den);
        Self { num: (num / g) as i64, den: (den / g) as i64 }
    }
}

impl std::ops::Sub for Rat {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        self + Self { num: -r.num, den: r.den }
    }
}

impl std::ops::Mul for Rat {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        let num = i128::from(self.num) * i128::from(r.num);
        let den = i128::from(self.den) * i128::from(r.den);
        let g = gcd_i128(num, den);
        Self { num: (num / g) as i64, den: (den / g) as i64 }
    }
}

impl std::ops::Div for Rat {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        assert!(!r.is_zero(), "division by zero rational");
        let num = i128::from(self.num) * i128::from(r.den);
        let den = i128::from(self.den) * i128::from(r.num);
        let g = gcd_i128(num, den);
        let s = if den < 0 { -1 } else { 1 };
        Self { num: ((num / g) as i64) * s, den: (den.abs() / g) as i64 }
    }
}

impl std::fmt::Display for Rat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

pub(crate) type IMat = Vec<Vec<i64>>;
pub(crate) type RMat = Vec<Vec<Rat>>;

pub(crate) fn imat_eye(n: usize) -> IMat {
    (0..n).map(|r| (0..n).map(|c| i64::from(r == c)).collect()).collect()
}

pub(crate) fn imat_mul(a: &IMat, b: &IMat) -> IMat {
    let n = a.len();
    let mut out = vec![vec![0_i64; n]; n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

pub(crate) fn imat_vec(a: &IMat, v: &[i64]) -> Vec<i64> {
    a.iter().map(|r| r.iter().zip(v).map(|(&x, &y)| x * y).sum()).collect()
}

pub(crate) fn imat_det(a: &IMat) -> i64 {
    match a.len() {
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        3 => {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        }
        _ => panic!("dimension > 3 not supported in imat_det"),
    }
}

pub(crate) fn rmat_transpose(a: &RMat) -> RMat {
    let r = a.len();
    let c = a.first().map_or(0, Vec::len);
    (0..c).map(|j| (0..r).map(|i| a[i][j]).collect()).collect()
}

pub(crate) fn rmat_mul(a: &RMat, b: &RMat) -> RMat {
    let n = a.len();
    let m = b.first().map_or(0, Vec::len);
    let p = b.len();
    let mut out = vec![vec![Rat::ZERO; m]; n];
    for i in 0..n {
        for k in 0..p {
            if a[i][k].is_zero() {
                continue;
            }
            for j in 0..m {
                out[i][j] = out[i][j].add(a[i][k].mul(b[k][j]));
            }
        }
    }
    out
}

pub(crate) fn rmat_vec(a: &RMat, v: &[Rat]) -> Vec<Rat> {
    a.iter()
        .map(|row| row.iter().zip(v).fold(Rat::ZERO, |acc, (&x, &y)| acc.add(x.mul(y))))
        .collect()
}

pub(crate) fn rmat_inv(a: &RMat) -> Result<RMat, String> {
    let n = a.len();
    let mut aug = vec![vec![Rat::ZERO; 2 * n]; n];
    for r in 0..n {
        for c in 0..n {
            aug[r][c] = a[r][c];
        }
        aug[r][n + r] = Rat::ONE;
    }
    for col in 0..n {
        let Some(piv) = (col..n).find(|&r| !aug[r][col].is_zero()) else {
            return Err("singular rational matrix".to_string());
        };
        aug.swap(col, piv);
        let p = aug[col][col];
        for cell in &mut aug[col] {
            *cell = *cell / p;
        }
        for r in 0..n {
            if r != col && !aug[r][col].is_zero() {
                let fac = aug[r][col];
                for c in 0..2 * n {
                    let v = aug[col][c].mul(fac);
                    aug[r][c] = aug[r][c].sub(v);
                }
            }
        }
    }
    Ok(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}
