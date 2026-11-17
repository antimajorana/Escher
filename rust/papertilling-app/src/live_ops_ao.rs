//! Ariza–Ortiz 2D metric relaxation modes and gauge/GSV overlay builders for `Defects` and `Combine`.

use papertilling_core::defect::cone_builder::mark_long_bonds;
use papertilling_core::defect::gauge::plaquette_holonomies;
use papertilling_core::defect::gsv_elastic::{Elastic2DResult, gsv_diagnostics_2d, relax_elastic_2d};
use papertilling_core::plane::draw::relax_preview;
use papertilling_core::types::{BuiltCrystal, Cell2D};

use crate::canvas2d::gauge_overlay::{
    CrystalOverlayData, CrystalOverlayMode, PlaquetteBadge, color_sites_scalar,
};
use crate::mesh3d::sheet_colors::{rgba_u8, sequential_heat_rgba};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Defect2dMode {
    Raw,
    #[default]
    RelaxedDrawing,
    ElasticAoMetric,
}

impl Defect2dMode {
    pub const ALL: [Self; 3] = [Self::Raw, Self::RelaxedDrawing, Self::ElasticAoMetric];

    pub fn label(self) -> &'static str {
        match self {
            Self::Raw => "Raw surgery",
            Self::RelaxedDrawing => "Relaxed (drawing)",
            Self::ElasticAoMetric => "Elastic (AO metric)",
        }
    }
}

pub fn relax_crystal_ao_2d(crystal: &mut BuiltCrystal) -> Result<Elastic2DResult, String> {
    let res = relax_elastic_2d(crystal)?;
    if res.pos.len() == crystal.pos.len() {
        crystal.pos = res.pos.clone();
        mark_long_bonds(crystal);
    }
    Ok(res)
}

pub fn apply_combine_ao_mode(crystal: &mut BuiltCrystal, preview: bool, notes: &mut Vec<String>) {
    if preview {
        relax_preview(crystal, 8.0);
        return;
    }
    match relax_crystal_ao_2d(crystal) {
        Ok(res) => notes.push(format!("AO GSV E={:.4} (shear={:.4})", res.total_energy, res.shear_energy)),
        Err(err) => notes.push(format!("AO elastic fallback: {err}")),
    }
}

pub fn build_ao_crystal_overlay(
    crystal: &BuiltCrystal,
    mode: CrystalOverlayMode,
    group: &str,
    cell: &Cell2D,
) -> Option<CrystalOverlayData> {
    if crystal.pos.is_empty() {
        return None;
    }
    match mode {
        CrystalOverlayMode::GsvEnergyShear => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            let site_rgba = Some(color_sites_scalar(&diag.site_gsv_energy, false));
            let bmax = diag.bond_gsv_energy.iter().copied().fold(0.0_f64, f64::max).max(1e-6);
            let bond_rgba = Some(
                diag.bond_gsv_energy
                    .iter()
                    .map(|&e| {
                        let c = rgba_u8(sequential_heat_rgba(e.max(0.0), bmax, 1.0));
                        let w = (1.3 + 1.8 * (e.max(0.0) / bmax).min(1.0)) as f32;
                        (c, w)
                    })
                    .collect(),
            );
            Some(CrystalOverlayData { site_rgba, bond_rgba, ..Default::default() })
        }
        CrystalOverlayMode::PlaquetteFlux => {
            let hols = plaquette_holonomies(crystal, Some(group), Some(cell));
            let mut badges = Vec::new();
            for p in hols {
                if !(p.has_curvature || p.has_torsion || p.nonsymmorphic_shift) || p.vertices.is_empty() {
                    continue;
                }
                let mut sum = [0.0_f64; 2];
                let mut cnt = 0usize;
                for &u in &p.vertices {
                    if let Some(pt) = crystal.pos.get(u) {
                        sum[0] += pt[0];
                        sum[1] += pt[1];
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let center = [sum[0] / (cnt as f64), sum[1] / (cnt as f64)];
                let label = if p.det_sign < 0 {
                    String::from("det=-1")
                } else if p.has_curvature {
                    format!("Ω={:+.0}°", p.omega_deg)
                } else if p.has_torsion {
                    format!("α=({:+.2},{:+.2})", p.alpha_f[0], p.alpha_f[1])
                } else {
                    String::from("τ-shift")
                };
                badges.push(PlaquetteBadge {
                    center,
                    label,
                    is_rotation: p.has_curvature,
                    improper: p.det_sign < 0,
                });
            }
            Some(CrystalOverlayData { badges, ..Default::default() })
        }
        _ => None,
    }
}
