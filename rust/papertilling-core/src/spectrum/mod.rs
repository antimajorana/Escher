//! Stage 5 tight-binding nearest-neighbour spectrum, multiplet LDOS, zero modes,
//! and geodesic Gaussian disclination/core charge diagnostics.

pub mod charge;

use std::f64::consts::PI;

use faer::{Mat, Side, c64};
use serde::{Deserialize, Serialize};

use crate::defect::equivalence::inspect_crystal_cores;
use crate::types::{BuiltCrystal, SurfaceMesh};

pub use charge::{
    Eigenvectors, MultipletRange, cluster_multiplets, compute_charge_diagnostics, geodesic_dists,
    ldos_for_level, resolve_mu,
};

fn default_seam_sign() -> i32 { 1 }
fn default_true() -> bool { true }
fn default_t0() -> f64 { 1.0 }

/// Options controlling the tight-binding Hamiltonian, Fermi level, and charge diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpectrumOptions {
    pub beta: f64,
    pub flux: f64,
    pub mu: f64,
    pub filling: Option<f64>,
    #[serde(default = "default_seam_sign")]
    pub seam_sign: i32,
    #[serde(default = "default_true")]
    pub multiplet: bool,
    #[serde(default = "default_t0")]
    pub t0: f64,
    pub core_radius: Option<f64>,
    pub level_idx: Option<isize>,
}

impl Default for SpectrumOptions {
    fn default() -> Self {
        Self {
            beta: 0.0,
            flux: 0.0,
            mu: 0.0,
            filling: None,
            seam_sign: 1,
            multiplet: true,
            t0: 1.0,
            core_radius: None,
            level_idx: None,
        }
    }
}

/// Normalized tight-binding graph input shared across surfaces, crystals, and JSON bindings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TightBindingInput {
    pub n: usize,
    pub closed: bool,
    pub edges: Vec<(usize, usize)>,
    pub r_ij: Vec<f64>,
    pub p_i: Vec<[f64; 2]>,
    pub p_j: Vec<[f64; 2]>,
    pub seam: Vec<bool>,
    pub seeds: Vec<Vec<usize>>,
}

/// Tight-binding spectrum and charge diagnostics matching `papertilling.spectrum.SpectrumResult`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpectrumResult {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Eigenvectors,
    pub zero_mode_count: usize,
    pub core_weight: Vec<f64>,
    pub multiplets: Vec<MultipletRange>,
    pub ldos: Vec<f64>,
    pub mu: f64,
    pub delta_rho: Vec<f64>,
    pub cone_charges: Vec<f64>,
    #[serde(rename = "delta_Q")]
    pub delta_q: Vec<f64>,
}

impl SpectrumResult {
    /// Local density of states for eigenstate `idx` (or its full near-degenerate multiplet).
    pub fn ldos_for_level(&self, idx: isize, multiplet: bool) -> Result<Vec<f64>, String> {
        ldos_for_level(&self.eigenvectors, &self.multiplets, idx, multiplet)
    }
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Solve the tight-binding spectrum and charge diagnostics for a normalized [`TightBindingInput`].
pub fn solve_spectrum(input: &TightBindingInput, opts: &SpectrumOptions) -> Result<SpectrumResult, String> {
    if opts.seam_sign != 1 && opts.seam_sign != -1 {
        return Err("seam_sign must be +1 or -1".into());
    }
    if let Some(fill) = opts.filling && (!fill.is_finite() || !(0.0..=1.0).contains(&fill)) {
        return Err("filling must be in [0, 1]".into());
    }
    if let Some(cr) = opts.core_radius && (!cr.is_finite() || cr < 0.0) {
        return Err("core_radius must be non-negative".into());
    }
    if ![opts.beta, opts.flux, opts.mu, opts.t0].iter().all(|x| x.is_finite()) {
        return Err("beta, flux, mu, and t0 must be finite".into());
    }
    let m = input.edges.len();
    if input.r_ij.len() != m || input.p_i.len() != m || input.p_j.len() != m || input.seam.len() != m {
        return Err("edges, r_ij, p_i, p_j, and seam must have the same length".into());
    }
    let n = input.n;
    if let Some(&(u, v)) = input.edges.iter().find(|&&(u, v)| u >= n || v >= n) {
        return Err(format!("edge ({u}, {v}) out of bounds for n={n}"));
    }
    if n == 0 {
        return Ok(SpectrumResult {
            eigenvalues: Vec::new(),
            eigenvectors: Eigenvectors::Real(Vec::new()),
            zero_mode_count: 0,
            core_weight: Vec::new(),
            multiplets: Vec::new(),
            ldos: Vec::new(),
            mu: opts.mu,
            delta_rho: Vec::new(),
            cone_charges: Vec::new(),
            delta_q: Vec::new(),
        });
    }

    let l_mean = if m == 0 { 1.0 } else { input.r_ij.iter().sum::<f64>() / (m as f64) };
    let denom = l_mean.max(1e-12);
    let t_ij: Vec<f64> = (0..m)
        .map(|k| {
            let t = opts.t0 * (-opts.beta * (input.r_ij[k] / denom - 1.0)).exp();
            if opts.seam_sign == -1 && input.seam[k] { -t } else { t }
        })
        .collect();

    let (evals, evecs) = if opts.flux.abs() > 0.0 {
        let mut h = Mat::<c64>::zeros(n, n);
        let pre = PI * opts.flux; // 2 * pi * flux * 0.5
        for (k, &(u, v)) in input.edges.iter().enumerate() {
            let a_ij = pre * (input.p_i[k][0] * input.p_j[k][1] - input.p_i[k][1] * input.p_j[k][0]);
            let (s_a, c_a) = a_ij.sin_cos();
            let h_val = c64::new(-t_ij[k] * c_a, -t_ij[k] * s_a);
            h[(u, v)] += h_val;
            h[(v, u)] += h_val.conj();
        }
        let eig = h.self_adjoint_eigen(Side::Lower).map_err(|e| format!("self_adjoint_eigen failed: {e:?}"))?;
        let s = eig.S();
        let u_mat = eig.U();
        let evals: Vec<f64> = (0..n).map(|j| s[j].re).collect();
        let vecs: Vec<Vec<[f64; 2]>> = (0..n)
            .map(|i| (0..n).map(|j| { let z = u_mat[(i, j)]; [z.re, z.im] }).collect())
            .collect();
        (evals, Eigenvectors::Complex(vecs))
    } else {
        let mut h = Mat::<f64>::zeros(n, n);
        for (k, &(u, v)) in input.edges.iter().enumerate() {
            h[(u, v)] -= t_ij[k];
            h[(v, u)] -= t_ij[k];
        }
        let eig = h.self_adjoint_eigen(Side::Lower).map_err(|e| format!("self_adjoint_eigen failed: {e:?}"))?;
        let s = eig.S();
        let u_mat = eig.U();
        let evals: Vec<f64> = (0..n).map(|j| s[j]).collect();
        let vecs: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| u_mat[(i, j)]).collect()).collect();
        (evals, Eigenvectors::Real(vecs))
    };

    let zero_mode_count = evals.iter().filter(|&&e| e.abs() < 1e-6).count();
    let multiplets = cluster_multiplets(&evals, 1e-8);
    let mu_used = resolve_mu(&evals, opts.mu, opts.filling);
    let (core_weight, delta_rho, delta_q) = compute_charge_diagnostics(
        n,
        input.closed,
        &input.edges,
        &input.seeds,
        &evals,
        &evecs,
        mu_used,
        opts.filling.is_some(),
        opts.core_radius,
    );
    let idx_use = opts.level_idx.unwrap_or_else(|| {
        (0..n).min_by(|&a, &b| (evals[a] - mu_used).abs().total_cmp(&(evals[b] - mu_used).abs())).unwrap_or(0) as isize
    });
    let ldos = ldos_for_level(&evecs, &multiplets, idx_use, opts.multiplet)?;

    Ok(SpectrumResult {
        eigenvalues: evals,
        eigenvectors: evecs,
        zero_mode_count,
        core_weight,
        multiplets,
        ldos,
        mu: mu_used,
        delta_rho,
        cone_charges: delta_q.clone(),
        delta_q,
    })
}

/// Extract tight-binding graph from a [`SurfaceMesh`] and compute its spectrum.
pub fn surface_spectrum(mesh: &SurfaceMesh, opts: &SpectrumOptions) -> Result<SpectrumResult, String> {
    let n = mesh.sites.len();
    let is_proj = mesh.cover.is_some() && !mesh.orientable && mesh.chi == 1;
    let flat_kind = ["torus", "cylinder", "Klein", "Moebius", "genus"].iter().any(|t| mesh.name.contains(t));
    let use_3d = !mesh.xyz.is_empty() && !flat_kind;
    let cov = mesh.cover.as_deref();

    let (mut edges, mut r_ij, mut p_i, mut p_j, mut seam) = (
        Vec::with_capacity(mesh.bonds.len()),
        Vec::with_capacity(mesh.bonds.len()),
        Vec::with_capacity(mesh.bonds.len()),
        Vec::with_capacity(mesh.bonds.len()),
        Vec::with_capacity(mesh.bonds.len()),
    );
    let is_mobius = mesh.name.contains("Moebius");

    for (k, &(a, b, v)) in mesh.bonds.iter().enumerate() {
        edges.push((a, b));
        let s_flag = is_mobius
            && mesh.uv.len() == n
            && (mesh.uv[a][0] + v[0] - mesh.uv[b][0]).hypot(mesh.uv[a][1] + v[1] - mesh.uv[b][1]) > 1e-4;
        seam.push(s_flag);
        if is_proj && use_3d && let Some(c) = cov {
            let (ca, cb) = c.bonds.get(2 * k).copied().filter(|&(u, w)| {
                let (uu, uw, ua, ub) = (u % n, w % n, a % n, b % n);
                (uu == ua && uw == ub) || (uu == ub && uw == ua)
            }).unwrap_or_else(|| {
                c.bonds.iter().copied().find(|&(u, w)| {
                    let (uu, uw, ua, ub) = (u % n, w % n, a % n, b % n);
                    (uu == ua && uw == ub) || (uu == ub && uw == ua)
                }).unwrap_or((a, b))
            });
            r_ij.push(dist3(c.xyz[cb], c.xyz[ca]));
            p_i.push([c.xyz[ca][0], c.xyz[ca][1]]);
            p_j.push([c.xyz[cb][0], c.xyz[cb][1]]);
        } else if use_3d && mesh.xyz.len() == n {
            r_ij.push(dist3(mesh.xyz[b], mesh.xyz[a]));
            p_i.push([mesh.xyz[a][0], mesh.xyz[a][1]]);
            p_j.push([mesh.xyz[b][0], mesh.xyz[b][1]]);
        } else {
            r_ij.push(v[0].hypot(v[1]));
            let pa = mesh.uv.get(a).copied().unwrap_or([0.0, 0.0]);
            p_i.push(pa);
            p_j.push([pa[0] + v[0], pa[1] + v[1]]);
        }
    }

    let seeds: Vec<Vec<usize>> = mesh.cones.iter().filter_map(|c| {
        c.site.map(|s| vec![s]).or_else(|| c.face.and_then(|fi| mesh.faces.get(fi).cloned()))
    }).collect();

    solve_spectrum(&TightBindingInput { n, closed: mesh.closed, edges, r_ij, p_i, p_j, seam, seeds }, opts)
}

/// Extract tight-binding graph from a [`BuiltCrystal`] and compute its spectrum.
pub fn crystal_spectrum(crystal: &BuiltCrystal, opts: &SpectrumOptions) -> Result<SpectrumResult, String> {
    let n = crystal.pos.len();
    let m = crystal.edges.len();
    let (mut r_ij, mut p_i, mut p_j, mut seam) = (
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
        Vec::with_capacity(m),
    );
    for &(u, v) in &crystal.edges {
        let e = crystal.adj.get(u).and_then(|row| row.iter().find(|b| b.to == v));
        let is_seam = crystal.is_non_orientable && e.is_some_and(|b| b.across && b.seam.is_some());
        seam.push(is_seam);
        if is_seam && let Some(b) = e {
            r_ij.push(b.vec[0].hypot(b.vec[1]));
            p_i.push(crystal.pos[u]);
            p_j.push([crystal.pos[u][0] + b.vec[0], crystal.pos[u][1] + b.vec[1]]);
        } else {
            let (pu, pv) = (crystal.pos[u], crystal.pos[v]);
            r_ij.push((pv[0] - pu[0]).hypot(pv[1] - pu[1]));
            p_i.push(pu);
            p_j.push(pv);
        }
    }
    let mut seeds = Vec::new();
    if !crystal.is_non_orientable {
        seeds = inspect_crystal_cores(crystal)
            .into_iter()
            .filter(|c| !c.sites.is_empty())
            .map(|c| c.sites)
            .collect();
        if seeds.is_empty() && !crystal.cone_sites.is_empty() {
            let mut cs = crystal.cone_sites.clone();
            cs.sort_unstable();
            cs.dedup();
            seeds = cs.into_iter().map(|s| vec![s]).collect();
        }
    }
    solve_spectrum(
        &TightBindingInput { n, closed: false, edges: crystal.edges.clone(), r_ij, p_i, p_j, seam, seeds },
        opts,
    )
}
