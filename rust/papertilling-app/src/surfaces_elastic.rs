//! Async Föppl–von Kármán 3D surface relaxation state & UI controls for the `Surfaces` tab.

use papertilling_core::surface::elastic3d::{SurfaceElasticResult, relax_surface_3d};
use papertilling_core::types::SurfaceMesh;

use crate::combine_worker::LatestWorker;
use crate::mesh3d::{GpuMeshData, SurfaceColor3d, elastic_surface_gpu_mesh, mesh_from_surface};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurfaceShapeMode {
    #[default]
    AnalyticRadial,
    ElasticFvK,
}

impl SurfaceShapeMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AnalyticRadial => "Analytic / Radial",
            Self::ElasticFvK => "Elastic Relaxation (FvK)",
        }
    }
}

pub struct SurfaceElasticJob {
    pub mesh: SurfaceMesh,
    pub kappa_b: f64,
    pub max_iter: usize,
}

pub struct SurfaceElasticState {
    pub shape_mode: SurfaceShapeMode,
    pub kappa_b: f64,
    pub color_mode: SurfaceColor3d,
    pub last_result: Option<SurfaceElasticResult>,
    pub last_error: Option<String>,
    pub stale: bool,
    worker: LatestWorker<SurfaceElasticJob, Result<SurfaceElasticResult, String>>,
}

impl Default for SurfaceElasticState {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceElasticState {
    pub fn new() -> Self {
        let worker = LatestWorker::spawn("surface-elastic", (), |_, job: SurfaceElasticJob| {
            relax_surface_3d(&job.mesh, job.kappa_b, job.max_iter)
        });
        Self {
            shape_mode: SurfaceShapeMode::AnalyticRadial,
            kappa_b: 0.02,
            color_mode: SurfaceColor3d::FaceRingSize,
            last_result: None,
            last_error: None,
            stale: false,
            worker,
        }
    }

    pub fn is_busy(&self) -> bool {
        self.worker.is_busy()
    }

    /// Drop stale 3D FvK data cleanly when the target surface or active lattice switches.
    pub fn clear_for_switch(&mut self) {
        self.worker.cancel();
        self.last_result = None;
        self.last_error = None;
        self.stale = false;
    }

    pub fn request_relax(&mut self, mesh: SurfaceMesh, ctx: Option<&egui::Context>) {
        if let Some(c) = ctx {
            self.worker.set_repaint_ctx(c);
        }
        self.stale = self.last_result.is_some();
        self.last_error = None;
        self.worker.submit(SurfaceElasticJob { mesh, kappa_b: self.kappa_b, max_iter: 400 });
    }

    pub fn poll(&mut self, ctx: &egui::Context) -> bool {
        self.worker.set_repaint_ctx(ctx);
        if let Some(out) = self.worker.poll() {
            match out {
                Ok(Ok(res)) => {
                    self.last_result = Some(res);
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

    /// Build 3D GPU mesh respecting Analytic vs FvK mode and optional spectral overlay colors.
    pub fn build_gpu_mesh(
        &self,
        mesh: &SurfaceMesh,
        show_cover: bool,
        spectral_scalar: Option<(&[f64], bool)>,
    ) -> (GpuMeshData, String) {
        let use_cov = show_cover && mesh.cover.is_some();
        let wants_elastic = self.shape_mode == SurfaceShapeMode::ElasticFvK;
        let use_el = if wants_elastic { self.last_result.as_ref() } else { None };
        let gpu = if wants_elastic || self.color_mode != SurfaceColor3d::FaceRingSize || spectral_scalar.is_some() {
            elastic_surface_gpu_mesh(mesh, use_el, self.color_mode, show_cover, spectral_scalar)
        } else {
            mesh_from_surface(mesh, use_cov)
        };
        let suffix = match (wants_elastic, use_el, self.stale, self.is_busy()) {
            (true, Some(r), true, _) => format!(" | FvK (previous result — stale, γ={:.1})", r.fvk_gamma),
            (true, Some(r), false, false) => format!(" | FvK γ={:.1}, α={:.2e}", r.fvk_gamma, r.asphericity),
            (true, _, _, true) => String::from(" | FvK relaxing…"),
            _ => String::new(),
        };
        (gpu, suffix)
    }

    /// 3D colorbar of the colouring `build_gpu_mesh` applies (same relaxed result, mode and spectral values).
    pub fn color_legend(&self, spectral_scalar: Option<(&[f64], bool)>) -> Option<crate::view3d::ColorLegend3d> {
        let use_el = if self.shape_mode == SurfaceShapeMode::ElasticFvK { self.last_result.as_ref() } else { None };
        crate::mesh3d::color_scales::surface_color_legend(use_el, self.color_mode, spectral_scalar)
    }
}
