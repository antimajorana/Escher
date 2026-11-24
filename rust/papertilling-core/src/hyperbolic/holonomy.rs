//! Exact comparison of reflection words in the faithful geometric representation of [p,q] and
//! [p,q,r]. In the simple-root basis the generators have entries in Z[ζ], with ζ a primitive
//! root of unity. Evaluating modulo its cyclotomic polynomial avoids tolerance-based matches.

use std::f64::consts::PI;

use crate::honeycomb3d::coxeter_gram::{HoneycombRegime, classify_schlafli};

pub const MAX_WORD_LEN: usize = 12;
const MAX_PRODUCT_LEN: usize = 4 * MAX_WORD_LEN;

#[derive(Clone, Debug)]
pub struct ReflectionWord(Vec<usize>);

impl ReflectionWord {
    fn parse(text: &str, generators: usize) -> Result<Self, String> {
        let text = text.trim();
        if text == "e" {
            return Ok(Self(Vec::new()));
        }
        if text.is_empty() {
            return Err("Enter a mirror word or e for identity.".into());
        }
        let mut word = Vec::new();
        for c in text.chars().filter(|c| !c.is_whitespace()) {
            let Some(i) = c
                .to_digit(10)
                .map(|i| i as usize)
                .filter(|&i| i < generators)
            else {
                return Err(format!("Enter e or mirror indices 0–{}.", generators - 1));
            };
            word.push(i);
            if word.len() > MAX_WORD_LEN {
                return Err(format!("Use at most {MAX_WORD_LEN} mirrors per word."));
            }
        }
        Ok(Self(word))
    }

    pub fn is_proper(&self) -> bool {
        self.0.len().is_multiple_of(2)
    }

    /// `a.compose(b)` acts as `a(b(x))`.
    pub fn compose(&self, rhs: &Self) -> Self {
        Self(self.0.iter().chain(&rhs.0).copied().collect())
    }

    pub fn inverse(&self) -> Self {
        Self(self.0.iter().rev().copied().collect())
    }

    pub fn conjugate(&self, op: &Self) -> Self {
        self.compose(op).compose(&self.inverse())
    }
}

impl std::fmt::Display for ReflectionWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return f.write_str("e");
        }
        for i in &self.0 {
            write!(f, "{i}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ReflectionGroup {
    dimension: usize,
    order: usize,
    /// Coefficients 2 cos(π/m_ij) at adjacent generator pairs, stored as ζ exponents.
    exponents: Vec<usize>,
    modulus: Vec<i128>,
}

impl ReflectionGroup {
    pub fn h2(p: usize, q: usize) -> Result<Self, String> {
        if !is_h2(p, q) {
            return Err("Use a hyperbolic {p,q} with 3 ≤ p,q ≤ 12.".into());
        }
        Ok(Self::new(&[p, q]))
    }

    pub fn h3(p: u32, q: u32, r: u32) -> Result<Self, String> {
        if [p, q, r].iter().any(|&n| !(3..=6).contains(&n))
            || !matches!(
                classify_schlafli(p, q, r).regime,
                HoneycombRegime::HyperbolicCompactH3 | HoneycombRegime::HyperbolicParacompactH3
            )
        {
            return Err("Choose a compact or paracompact regular H³ honeycomb.".into());
        }
        Ok(Self::new(&[p as usize, q as usize, r as usize]))
    }

    fn new(edges: &[usize]) -> Self {
        let order = edges.iter().fold(1, |lcm, &m| {
            let n = 2 * m;
            lcm * n / gcd(lcm, n)
        });
        let exponents = edges.iter().map(|&m| order / (2 * m)).collect();
        Self {
            dimension: edges.len() + 1,
            order,
            exponents,
            modulus: cyclotomic(order),
        }
    }

    pub fn parse(&self, text: &str) -> Result<ReflectionWord, String> {
        ReflectionWord::parse(text, self.dimension)
    }

    pub fn same_isometry(&self, a: &ReflectionWord, b: &ReflectionWord) -> Result<bool, String> {
        Ok(self.matrix(a)? == self.matrix(b)?)
    }

    pub fn is_identity(&self, a: &ReflectionWord) -> Result<bool, String> {
        self.same_isometry(a, &ReflectionWord(Vec::new()))
    }

    pub fn kind(&self, a: &ReflectionWord) -> Result<&'static str, String> {
        if self.is_identity(a)? {
            return Ok("Identity");
        }
        if a.is_proper() {
            return Ok("Orientation-preserving isometry");
        }
        let involution = self.is_identity(&a.compose(a))?;
        Ok(match (self.dimension, involution) {
            (3, true) => "Reflection",
            (3, false) => "Glide reflection",
            (4, true) => {
                let matrix = self.matrix(a)?;
                let mut trace = vec![0; self.modulus.len() - 1];
                for i in 0..4 {
                    for (value, coefficient) in trace.iter_mut().zip(&matrix[i * 5]) {
                        *value += coefficient;
                    }
                }
                if trace[0] == 2 && trace[1..].iter().all(|&v| v == 0) {
                    "Reflection"
                } else {
                    "Point inversion"
                }
            }
            _ => "Orientation-reversing isometry",
        })
    }

    fn matrix(&self, a: &ReflectionWord) -> Result<Vec<Vec<i128>>, String> {
        if a.0.len() > MAX_PRODUCT_LEN || a.0.iter().any(|&i| i >= self.dimension) {
            return Err(format!(
                "The composite is limited to {MAX_PRODUCT_LEN} mirrors from this group."
            ));
        }
        let n = self.dimension;
        let mut result = Vec::with_capacity(n * n);
        // Right-multiply by each reflection. Its only nonidentity row is i:
        // column i changes sign, while adjacent columns acquire 2 cos(π/m_ij) times old column i.
        for j in 0..n {
            let mut row = vec![vec![0_i128; self.order]; n];
            row[j][0] = 1;
            for &i in &a.0 {
                let (left, rest) = row.split_at_mut(i);
                let (source, right) = rest.split_first_mut().unwrap();
                if let Some(target) = left.last_mut() {
                    add_cosine_multiple(target, source, self.exponents[i - 1]);
                }
                if let Some(target) = right.first_mut() {
                    add_cosine_multiple(target, source, self.exponents[i]);
                }
                for c in source {
                    *c = -*c;
                }
            }
            for mut p in row {
                remainder_monic(&mut p, &self.modulus);
                result.push(p);
            }
        }
        Ok(result)
    }
}

fn add_cosine_multiple(target: &mut [i128], source: &[i128], exponent: usize) {
    let n = source.len();
    for (k, &c) in source.iter().enumerate().filter(|&(_, &c)| c != 0) {
        target[(k + exponent) % n] += c;
        target[(k + n - exponent) % n] += c;
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Divide by the monic polynomial in place and leave only the remainder.
fn remainder_monic(p: &mut Vec<i128>, modulus: &[i128]) {
    let degree = modulus.len() - 1;
    for i in (degree..p.len()).rev() {
        let c = p[i];
        for (j, &b) in modulus[..degree].iter().enumerate() {
            p[i - degree + j] -= c * b;
        }
    }
    p.truncate(degree);
}

fn cyclotomic(n: usize) -> Vec<i128> {
    let mut factors: Vec<(usize, Vec<i128>)> = Vec::new();
    for d in (1..=n).filter(|&d| n.is_multiple_of(d)) {
        let mut p = vec![0; d + 1];
        p[0] = -1;
        p[d] = 1;
        for (_, factor) in factors.iter().filter(|(k, _)| d.is_multiple_of(*k)) {
            let degree = factor.len() - 1;
            let mut quotient = vec![0; p.len() - degree];
            for i in (degree..p.len()).rev() {
                let c = p[i];
                quotient[i - degree] = c;
                for (j, &b) in factor.iter().enumerate() {
                    p[i - degree + j] -= c * b;
                }
            }
            debug_assert!(p.iter().all(|&c| c == 0));
            p = quotient;
        }
        factors.push((d, p));
    }
    factors.pop().unwrap().1
}

fn is_h2(p: usize, q: usize) -> bool {
    (3..=12).contains(&p) && (3..=12).contains(&q) && (p - 2) * (q - 2) > 4
}

/// Angular mismatch at an attempted q-fold vertex made of copies of a regular core polygon,
/// keeping the side length of the regular {p,q} tiling. A negative result signals overlap.
pub fn wedge_angle_mismatch(p: usize, q: usize, core: usize) -> Option<f64> {
    if !is_h2(p, q) || !(3..=12).contains(&core) {
        return None;
    }
    let cosh_half_edge = (PI / p as f64).cos() / (PI / q as f64).sin();
    let angle = 2.0 * ((PI / core as f64).cos() / cosh_half_edge).asin();
    Some(2.0 * PI - q as f64 * angle)
}

#[cfg(test)]
mod tests;
