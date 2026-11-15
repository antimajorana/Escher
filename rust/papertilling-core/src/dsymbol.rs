//! Geometry-agnostic 2D Delaney-Dress symbols, canonical keys, and minimal network keys.

use std::{collections::HashMap, fmt};

use serde::{Deserialize, Serialize};

/// One breadth-first row `(m01, m12, s0, s1, s2)` of a canonical D-symbol key.
pub type KeyRow = (u32, u32, u32, u32, u32);
/// Canonical breadth-first key of a 2D Delaney-Dress symbol.
pub type DSymbolKey = Vec<KeyRow>;

/// Errors raised when constructing or canonicalising a Delaney-Dress symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DSymbolError {
    Empty,
    ShapeMismatch,
    IndexOutOfBounds { chamber: usize, target: usize, size: usize },
    NotInvolution { generator: usize, chamber: usize },
    ZeroDegree { chamber: usize },
    Disconnected,
}

impl fmt::Display for DSymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "D-set must contain at least one chamber"),
            Self::ShapeMismatch => write!(f, "sigma and degree arrays must have equal length"),
            Self::IndexOutOfBounds { chamber, target, size } => {
                write!(f, "chamber {chamber} points to {target}, outside 0..{size}")
            }
            Self::NotInvolution { generator, chamber } => {
                write!(f, "generator {generator} is not an involution at chamber {chamber}")
            }
            Self::ZeroDegree { chamber } => write!(f, "chamber {chamber} has zero degree"),
            Self::Disconnected => write!(f, "D-set not connected"),
        }
    }
}

impl std::error::Error for DSymbolError {}

/// A connected 2D Delaney-Dress symbol with chamber involutions `s0, s1, s2` and labels `m01, m12`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DSymbol {
    pub size: usize,
    pub s0: Vec<usize>,
    pub s1: Vec<usize>,
    pub s2: Vec<usize>,
    pub m01: Vec<u32>,
    pub m12: Vec<u32>,
}

impl DSymbol {
    /// Construct and validate a 2D Delaney-Dress symbol.
    pub fn new(s0: Vec<usize>, s1: Vec<usize>, s2: Vec<usize>, m01: Vec<u32>, m12: Vec<u32>) -> Result<Self, DSymbolError> {
        let size = m01.len();
        if size == 0 {
            return Err(DSymbolError::Empty);
        }
        if s0.len() != size || s1.len() != size || s2.len() != size || m12.len() != size {
            return Err(DSymbolError::ShapeMismatch);
        }
        if let Some(x) = (0..size).find(|&x| m01[x] == 0 || m12[x] == 0) {
            return Err(DSymbolError::ZeroDegree { chamber: x });
        }
        for (gen_idx, s) in [&s0, &s1, &s2].into_iter().enumerate() {
            for (x, &y) in s.iter().enumerate() {
                if y >= size {
                    return Err(DSymbolError::IndexOutOfBounds { chamber: x, target: y, size });
                }
                if s[y] != x {
                    return Err(DSymbolError::NotInvolution { generator: gen_idx, chamber: x });
                }
            }
        }
        if let Some(x) = (0..size).find(|&x| s0[s2[s0[s2[x]]]] != x) {
            return Err(DSymbolError::NotInvolution { generator: 20, chamber: x });
        }
        let _ = canonical_key(&[&s0, &s1, &s2], &m01, &m12)?;
        Ok(Self { size, s0, s1, s2, m01, m12 })
    }

    /// Reconstruct a validated `DSymbol` from a breadth-first key.
    pub fn from_key(key: &[KeyRow]) -> Result<Self, DSymbolError> {
        Self::new(
            key.iter().map(|r| r.2 as usize).collect(),
            key.iter().map(|r| r.3 as usize).collect(),
            key.iter().map(|r| r.4 as usize).collect(),
            key.iter().map(|r| r.0).collect(),
            key.iter().map(|r| r.1).collect(),
        )
    }

    pub fn canonical_key(&self) -> Result<DSymbolKey, DSymbolError> {
        canonical_key(&[&self.s0, &self.s1, &self.s2], &self.m01, &self.m12)
    }
    pub fn network_key(&self) -> Result<DSymbolKey, DSymbolError> {
        network_key(&self.canonical_key()?)
    }
    pub fn symmetry_index(&self) -> Result<usize, DSymbolError> {
        symmetry_index(&self.canonical_key()?)
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Exact rational check of flat orbifold curvature `sum (1/m01 + 1/m12 - 1/2) == 0`.
    pub fn is_flat(&self) -> bool {
        let (mut num, mut den): (i128, i128) = (0, 1);
        for (&a, &b) in self.m01.iter().zip(&self.m12) {
            let (ai, bi) = (i128::from(a), i128::from(b));
            let (t_num, t_den) = (2 * bi + 2 * ai - ai * bi, 2 * ai * bi);
            (num, den) = (num * t_den + t_num * den, den * t_den);
            let g = gcd(num.unsigned_abs(), den.unsigned_abs()) as i128;
            (num, den) = (num / g, den / g);
        }
        num == 0
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// Smallest breadth-first code over all start chambers, matching `papertilling.dsymbol.canonical_key`.
pub fn canonical_key<S: AsRef<[usize]>>(sig: &[S; 3], m01: &[u32], m12: &[u32]) -> Result<DSymbolKey, DSymbolError> {
    let n = m01.len();
    if n == 0 {
        return Err(DSymbolError::Empty);
    }
    let s = [sig[0].as_ref(), sig[1].as_ref(), sig[2].as_ref()];
    if m12.len() != n || s.iter().any(|arr| arr.len() != n) {
        return Err(DSymbolError::ShapeMismatch);
    }
    for arr in s {
        for (x, &y) in arr.iter().enumerate() {
            if y >= n {
                return Err(DSymbolError::IndexOutOfBounds { chamber: x, target: y, size: n });
            }
        }
    }

    let mut best: Option<DSymbolKey> = None;
    let mut num = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);

    for start in 0..n {
        num.fill(usize::MAX);
        order.clear();
        num[start] = 0;
        order.push(start);
        let (mut code, mut k, mut pruned) = (Vec::with_capacity(n), 0, false);

        while k < order.len() {
            let x = order[k];
            k += 1;
            let mut imgs = [0u32; 3];
            for (i, arr) in s.iter().enumerate() {
                let y = arr[x];
                if num[y] == usize::MAX {
                    num[y] = order.len();
                    order.push(y);
                }
                imgs[i] = num[y] as u32;
            }
            code.push((m01[x], m12[x], imgs[0], imgs[1], imgs[2]));
            if best.as_ref().is_some_and(|b| code.as_slice() > &b[..code.len()]) {
                pruned = true;
                break;
            }
        }
        if !pruned {
            if order.len() != n {
                return Err(DSymbolError::Disconnected);
            }
            if best.as_ref().is_none_or(|b| &code < b) {
                best = Some(code);
            }
        }
    }
    best.ok_or(DSymbolError::Disconnected)
}

/// Canonical key of the minimal D-symbol image (network with symmetry forgotten).
pub fn network_key(key: &[KeyRow]) -> Result<DSymbolKey, DSymbolError> {
    let n = key.len();
    if n == 0 {
        return Err(DSymbolError::Empty);
    }
    for (chamber, &(_, _, s0, s1, s2)) in key.iter().enumerate() {
        if let Some(target) = [s0 as usize, s1 as usize, s2 as usize].into_iter().find(|&t| t >= n) {
            return Err(DSymbolError::IndexOutOfBounds { chamber, target, size: n });
        }
    }
    let mut cls: HashMap<(u32, u32), usize> = HashMap::new();
    let mut cid: Vec<usize> = key
        .iter()
        .map(|&(m01, m12, _, _, _)| {
            let next = cls.len();
            *cls.entry((m01, m12)).or_insert(next)
        })
        .collect();
    let mut k = cls.len();

    loop {
        let mut next_cls: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
        let nid: Vec<usize> = (0..n)
            .map(|x| {
                let (_, _, s0, s1, s2) = key[x];
                let next = next_cls.len();
                *next_cls.entry((cid[x], cid[s0 as usize], cid[s1 as usize], cid[s2 as usize])).or_insert(next)
            })
            .collect();
        if next_cls.len() == k {
            break;
        }
        (k, cid) = (next_cls.len(), nid);
    }

    let mut rep = vec![usize::MAX; k];
    for (x, &c) in cid.iter().enumerate() {
        if rep[c] == usize::MAX {
            rep[c] = x;
        }
    }
    let s0: Vec<usize> = (0..k).map(|c| cid[key[rep[c]].2 as usize]).collect();
    let s1: Vec<usize> = (0..k).map(|c| cid[key[rep[c]].3 as usize]).collect();
    let s2: Vec<usize> = (0..k).map(|c| cid[key[rep[c]].4 as usize]).collect();
    let m01: Vec<u32> = (0..k).map(|c| key[rep[c]].0).collect();
    let m12: Vec<u32> = (0..k).map(|c| key[rep[c]].1).collect();
    canonical_key(&[s0, s1, s2], &m01, &m12)
}

/// Ratio `|key| / |network_key(key)|`: index of the chosen symmetry group in the full combinatorial group.
pub fn symmetry_index(key: &[KeyRow]) -> Result<usize, DSymbolError> {
    Ok(key.len() / network_key(key)?.len())
}
