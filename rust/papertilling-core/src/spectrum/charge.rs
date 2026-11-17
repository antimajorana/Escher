//! Multiplet clustering, chemical potential / filling resolution, LDOS, and
//! graph-geodesic Gaussian disclination/core charge diagnostics.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Contiguous near-degenerate energy multiplet `[start, end)` with mean energy and degeneracy `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultipletRange {
    pub energy: f64,
    pub start: usize,
    pub end: usize,
    pub dim: usize,
}

/// Real or complex eigenbasis matrix `(N, N)` indexed as `[site][state]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Eigenvectors {
    Real(Vec<Vec<f64>>),
    Complex(Vec<Vec<[f64; 2]>>),
}

impl Eigenvectors {
    pub fn len(&self) -> usize {
        match self {
            Self::Real(m) => m.len(),
            Self::Complex(m) => m.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn abs2(&self, site: usize, state: usize) -> f64 {
        match self {
            Self::Real(m) => {
                let v = m[site][state];
                v * v
            }
            Self::Complex(m) => {
                let [re, im] = m[site][state];
                re * re + im * im
            }
        }
    }
}

/// Group sorted eigenvalues into near-degenerate multiplets (matching `_cluster_multiplets` in `spectrum.py`).
pub fn cluster_multiplets(evals: &[f64], tol: f64) -> Vec<MultipletRange> {
    let n = evals.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (evals[j] - evals[j - 1]) < tol && (evals[j] - evals[i]) < 10.0 * tol {
            j += 1;
        }
        let mean = evals[i..j].iter().sum::<f64>() / ((j - i) as f64);
        out.push(MultipletRange { energy: mean, start: i, end: j, dim: j - i });
        i = j;
    }
    out
}

/// Resolve chemical potential from either explicit `mu` or fractional electron `filling` in `[0, 1]`.
pub fn resolve_mu(evals: &[f64], mu: f64, filling: Option<f64>) -> f64 {
    let Some(fill) = filling else { return mu; };
    let n = evals.len();
    if n == 0 {
        return mu;
    }
    let target_e = fill * (n as f64);
    if target_e <= 0.0 {
        return evals[0] - 1.0;
    }
    if target_e >= n as f64 {
        return evals[n - 1] + 1.0;
    }
    let k = target_e.round() as usize;
    if (target_e - (k as f64)).abs() < 1e-9 && k > 0 && k < n && (evals[k] - evals[k - 1]) > 1e-8 {
        return 0.5 * (evals[k - 1] + evals[k]);
    }
    let idx = (target_e.floor() as isize).clamp(0, (n as isize) - 1) as usize;
    evals[idx]
}

/// Exact unweighted BFS graph-geodesic distances from each seed site group (`-1` if unreachable).
pub fn geodesic_dists(n: usize, edges: &[(usize, usize)], seeds: &[Vec<usize>]) -> Vec<Vec<i32>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    let mut out = Vec::with_capacity(seeds.len());
    let mut dq = VecDeque::new();
    for sd in seeds {
        let mut dist = vec![-1_i32; n];
        dq.clear();
        for &s in sd {
            if s < n && dist[s] < 0 {
                dist[s] = 0;
                dq.push_back(s);
            }
        }
        while let Some(u) = dq.pop_front() {
            let next_d = dist[u] + 1;
            for &w in &adj[u] {
                if dist[w] < 0 {
                    dist[w] = next_d;
                    dq.push_back(w);
                }
            }
        }
        out.push(dist);
    }
    out
}

/// Local density of states for level `idx` (or its full near-degenerate multiplet when `multiplet == true`).
pub fn ldos_for_level(
    evecs: &Eigenvectors,
    multiplets: &[MultipletRange],
    idx: isize,
    multiplet: bool,
) -> Result<Vec<f64>, String> {
    let n = evecs.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let idx_norm = if idx < 0 { idx + (n as isize) } else { idx };
    if idx_norm < 0 || idx_norm >= (n as isize) {
        return Err(format!("level index {idx} out of range for N={n}"));
    }
    let k = idx_norm as usize;
    let (start, end) = if multiplet {
        multiplets
            .iter()
            .find(|m| m.start <= k && k < m.end)
            .map_or((k, k + 1), |m| (m.start, m.end))
    } else {
        (k, k + 1)
    };
    Ok((0..n)
        .map(|site| (start..end).map(|col| evecs.abs2(site, col)).sum())
        .collect())
}

/// Compute `(core_weight, delta_rho, delta_Q)` matching `papertilling/spectrum.py`.
#[allow(clippy::too_many_arguments)]
pub fn compute_charge_diagnostics(
    n: usize,
    is_closed: bool,
    edges: &[(usize, usize)],
    seeds: &[Vec<usize>],
    evals: &[f64],
    evecs: &Eigenvectors,
    mu_used: f64,
    has_filling: bool,
    core_radius: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let weights: Vec<f64> = evals
        .iter()
        .map(|&e| {
            if e < mu_used - 1e-8 {
                1.0
            } else if (e - mu_used).abs() <= 1e-8 {
                0.5
            } else {
                0.0
            }
        })
        .collect();
    let occ: Vec<f64> = (0..n)
        .map(|site| weights.iter().enumerate().map(|(col, &w)| if w == 0.0 { 0.0 } else { w * evecs.abs2(site, col) }).sum())
        .collect();
    let nu_ref = if is_closed || mu_used.abs() > 1e-12 || has_filling {
        occ.iter().sum::<f64>() / (n as f64)
    } else {
        0.5
    };
    let delta_rho: Vec<f64> = occ.iter().map(|&o| o - nu_ref).collect();

    let dists = geodesic_dists(n, edges, seeds);
    let r_ball = core_radius.unwrap_or(2.0);
    if dists.is_empty() {
        return (vec![0.0; n], delta_rho, Vec::new());
    }
    let inf_d = (n + 1000) as i32;
    let in_ball: Vec<bool> = (0..n)
        .map(|site| {
            let min_d = dists.iter().map(|d| if d[site] >= 0 { d[site] } else { inf_d }).min().unwrap_or(inf_d);
            (min_d as f64) <= r_ball + 1e-12
        })
        .collect();
    let core_weight: Vec<f64> = (0..n)
        .map(|col| (0..n).filter(|&site| in_ball[site]).map(|site| evecs.abs2(site, col)).sum())
        .collect();
    let sigma = r_ball.max(1e-6);
    let inv_2sig2 = -0.5 / (sigma * sigma);
    let delta_q: Vec<f64> = dists
        .iter()
        .map(|dc| {
            (0..n)
                .filter(|&site| dc[site] >= 0)
                .map(|site| {
                    let d = dc[site] as f64;
                    (d * d * inv_2sig2).exp() * delta_rho[site]
                })
                .sum()
        })
        .collect();
    (core_weight, delta_rho, delta_q)
}
