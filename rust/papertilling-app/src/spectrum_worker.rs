//! Async tight-binding spectral worker (`<= 2,500` site guard) for `Defects`, `Combine`, and `Surfaces`.

use papertilling_core::defect::sheet3d::Sheet3DResult;
use papertilling_core::spectrum::{
    SpectrumOptions, SpectrumResult, crystal_spectrum, surface_spectrum,
};
use papertilling_core::types::{BuiltCrystal, SurfaceMesh};

use crate::combine_worker::LatestWorker;

/// Refuse O(N^3) dense diagonalization cleanly above this vertex count so the UI never stalls.
pub const SPECTRUM_MAX_SITES: usize = 2_500;

#[derive(Clone)]
pub enum SpectrumTarget {
    Surface(SurfaceMesh),
    Crystal { crystal: BuiltCrystal, sheet_xyz: Option<Vec<[f64; 3]>> },
}

impl SpectrumTarget {
    pub fn site_count(&self) -> usize {
        match self {
            Self::Surface(m) => m.sites.len(),
            Self::Crystal { crystal, .. } => crystal.pos.len(),
        }
    }
}

pub struct SpectrumJob {
    pub target: SpectrumTarget,
    pub opts: SpectrumOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpectrumColorOverlay {
    #[default]
    None,
    LdosMultiplet,
    SpectralCharge,
}

impl SpectrumColorOverlay {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "Off",
            Self::LdosMultiplet => "LDOS |ψ|² (multiplet)",
            Self::SpectralCharge => "Spectral charge δρ(μ)",
        }
    }
}

pub struct SpectrumPanelState {
    pub expanded: bool,
    pub opts: SpectrumOptions,
    pub color_overlay: SpectrumColorOverlay,
    pub selected_level: Option<usize>,
    pub result: Option<SpectrumResult>,
    pub last_error: Option<String>,
    pub stale: bool,
    worker: LatestWorker<SpectrumJob, Result<SpectrumResult, String>>,
}

impl Default for SpectrumPanelState {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectrumPanelState {
    pub fn new() -> Self {
        let worker = LatestWorker::spawn("tb-spectrum", (), |_, job: SpectrumJob| {
            run_spectrum_job(&job)
        });
        Self {
            expanded: false,
            opts: SpectrumOptions::default(),
            color_overlay: SpectrumColorOverlay::None,
            selected_level: None,
            result: None,
            last_error: None,
            stale: false,
            worker,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.worker.is_busy()
    }

    /// Clear previous spectral output immediately when the active lattice or target changes.
    pub fn clear_for_switch(&mut self) {
        self.worker.cancel();
        self.result = None;
        self.last_error = None;
        self.selected_level = None;
        self.stale = false;
    }

    pub fn submit_surface(&mut self, mesh: &SurfaceMesh, ctx: Option<&egui::Context>) {
        self.opts.flux = 0.0;
        self.submit_target(SpectrumTarget::Surface(mesh.clone()), ctx);
    }

    pub fn submit_crystal(
        &mut self,
        crystal: &BuiltCrystal,
        sheet: Option<&Sheet3DResult>,
        ctx: Option<&egui::Context>,
    ) {
        let sheet_xyz = sheet
            .filter(|s| s.xyz.len() == crystal.pos.len() && s.has_finite_geometry())
            .map(|s| s.xyz.clone());
        self.submit_target(
            SpectrumTarget::Crystal { crystal: crystal.clone(), sheet_xyz },
            ctx,
        );
    }

    fn submit_target(&mut self, target: SpectrumTarget, ctx: Option<&egui::Context>) {
        if let Some(c) = ctx {
            self.worker.set_repaint_ctx(c);
        }
        let n = target.site_count();
        if n == 0 {
            self.worker.cancel();
            self.result = None;
            self.last_error = Some(String::from("Spectrum unavailable: empty geometry."));
            self.stale = false;
            return;
        }
        if n > SPECTRUM_MAX_SITES {
            self.worker.cancel();
            self.result = None;
            self.last_error = Some(format!(
                "Spectrum guarded above {SPECTRUM_MAX_SITES} sites (N = {n}): reduce R or surface size to diagonalize."
            ));
            self.stale = false;
            return;
        }
        self.stale = self.result.is_some();
        self.last_error = None;
        let mut opts = self.opts.clone();
        opts.level_idx = self.selected_level.map(|i| i as isize);
        self.worker.submit(SpectrumJob { target, opts });
    }

    pub fn poll(&mut self, ctx: &egui::Context) -> bool {
        self.worker.set_repaint_ctx(ctx);
        if let Some(out) = self.worker.poll() {
            match out {
                Ok(Ok(res)) => {
                    self.result = Some(res);
                    self.last_error = None;
                }
                Ok(Err(err)) | Err(err) => {
                    self.last_error = Some(err);
                }
            }
            self.stale = false;
            return true;
        }
        false
    }

    /// Return `(per_site_values, is_diverging)` when an LDOS or excess-charge overlay is active.
    pub fn active_scalar_overlay(&self) -> Option<(Vec<f64>, bool)> {
        let res = self.result.as_ref()?;
        match self.color_overlay {
            SpectrumColorOverlay::None => None,
            SpectrumColorOverlay::LdosMultiplet => {
                let vals = match self.selected_level {
                    Some(idx) if idx < res.eigenvalues.len() => {
                        res.ldos_for_level(idx as isize, true).unwrap_or_else(|_| res.ldos.clone())
                    }
                    _ => res.ldos.clone(),
                };
                Some((vals, false))
            }
            SpectrumColorOverlay::SpectralCharge => Some((res.delta_rho.clone(), true)),
        }
    }
}

fn run_spectrum_job(job: &SpectrumJob) -> Result<SpectrumResult, String> {
    let n = job.target.site_count();
    if n > SPECTRUM_MAX_SITES {
        return Err(format!("N={n} exceeds SPECTRUM_MAX_SITES={SPECTRUM_MAX_SITES}"));
    }
    match &job.target {
        SpectrumTarget::Surface(mesh) => surface_spectrum(mesh, &job.opts),
        SpectrumTarget::Crystal { crystal, sheet_xyz } => {
            if let Some(xyz) = sheet_xyz && xyz.len() == crystal.pos.len() {
                let mut opts = job.opts.clone();
                if opts.beta.abs() > 1e-12 {
                    // Use relaxed 3D coordinates when strain-coupling beta > 0.
                    let mut crystal3 = crystal.clone();
                    for (u, adj_u) in crystal3.adj.iter_mut().enumerate() {
                        for e in adj_u {
                            let dx = xyz[e.to][0] - xyz[u][0];
                            let dy = xyz[e.to][1] - xyz[u][1];
                            let dz = xyz[e.to][2] - xyz[u][2];
                            let l3 = (dx * dx + dy * dy + dz * dz).sqrt();
                            let l2 = e.vec[0].hypot(e.vec[1]).max(1e-12);
                            let sc = l3 / l2;
                            e.vec = [e.vec[0] * sc, e.vec[1] * sc];
                        }
                    }
                    opts.t0 = job.opts.t0;
                    return crystal_spectrum(&crystal3, &opts);
                }
            }
            crystal_spectrum(crystal, &job.opts)
        }
    }
}
