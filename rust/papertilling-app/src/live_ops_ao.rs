//! Ariza–Ortiz 2D metric relaxation modes and gauge/GSV overlay builders for `Defects` and `Combine`.

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::defect::cone_builder::mark_long_bonds;
use papertilling_core::defect::gauge::{plaquette_holonomies, wyckoff_rotocentres_with_frame};
use papertilling_core::defect::gsv_elastic::{Elastic2DResult, gsv_diagnostics_2d, relax_elastic_2d};
use papertilling_core::plane::draw::relax_preview;
use papertilling_core::types::{BuiltCrystal, Cell2D, Frame2D};

use crate::canvas2d::gauge_overlay::{CrystalOverlayData, CrystalOverlayMode, PlaquetteBadge, perceptual_heat_scale};
use crate::canvas2d::overlay_scales::{bond_strain_overlay, perceptual_colorbar};
use crate::mesh3d::sheet_colors::{rgba_u8, sequential_heat_rgba};

#[cfg(test)]
#[path = "live_ops_ao_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Defect2dMode {
    Raw,
    #[default]
    RelaxedDrawing,
    ElasticAoMetric,
}

impl Defect2dMode {
    /// The modes the UI offers. C1: the AO elastic view left the UI (it solves a Green-St Venant metric energy
    /// with geometric weights, not Ariza-Ortiz elasticity, and a glued 2D wall has no isometric immersion in the
    /// plane); its variant stays for session decoding, where it is migrated to `RelaxedDrawing` with a note.
    pub const ALL: [Self; 2] = [Self::Raw, Self::RelaxedDrawing];
    /// Every variant a saved session may name.
    pub const DECODE: [Self; 3] = [Self::Raw, Self::RelaxedDrawing, Self::ElasticAoMetric];

    pub fn label(self) -> &'static str {
        match self {
            Self::Raw => "Raw surgery",
            Self::RelaxedDrawing => "Relaxed (drawing)",
            Self::ElasticAoMetric => "Elastic (AO metric)",
        }
    }
}

#[allow(dead_code)] // C1 hidden: the AO elastic view left the UI; the core solver is unchanged
pub fn relax_crystal_ao_2d(crystal: &mut BuiltCrystal) -> Result<Elastic2DResult, String> {
    let res = relax_elastic_2d(crystal)?;
    if res.pos.len() == crystal.pos.len() {
        crystal.pos = res.pos.clone();
        mark_long_bonds(crystal);
    }
    Ok(res)
}

#[allow(dead_code)] // C1 hidden: the AO elastic view left the UI
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

fn heat_bonds(values: &[f64]) -> (Vec<([u8; 4], f32)>, f64) {
    let (scale, vmax) = perceptual_heat_scale(values);
    let cols = values
        .iter()
        .map(|&e| {
            let t = (e.max(0.0) / scale).clamp(0.0, 1.0).sqrt();
            let c = rgba_u8(sequential_heat_rgba(t, 1.0, 1.0));
            (c, (1.3 + 2.0 * t) as f32)
        })
        .collect();
    (cols, vmax.min(scale))
}

fn match_wyckoff_label(
    cat: &DefectCatalogue,
    wyck: &[papertilling_core::defect::gauge::WyckoffRotocentre],
    ainv: [[f64; 2]; 2],
    centre: [f64; 2],
) -> Option<String> {
    if let Some(cc) = cat.centre_class_of(centre)
        && !cc.label.starts_with("order-")
    {
        return Some(cc.label.clone());
    }
    let fc = [
        ainv[0][0] * centre[0] + ainv[0][1] * centre[1],
        ainv[1][0] * centre[0] + ainv[1][1] * centre[1],
    ];
    wyck.iter()
        .find(|w| {
            let df = [fc[0] - w.frac[0], fc[1] - w.frac[1]];
            (df[0] - df[0].round()).hypot(df[1] - df[1].round()) < 1e-3
        })
        .map(|w| w.label.clone())
}

pub fn build_ao_crystal_overlay(
    crystal: &BuiltCrystal,
    mode: CrystalOverlayMode,
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
) -> Option<CrystalOverlayData> {
    if crystal.pos.is_empty() {
        return None;
    }
    match mode {
        CrystalOverlayMode::GsvEnergyShear => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            let (site_cols, legend) = perceptual_colorbar(&diag.site_gsv_energy, "Site GSV E (√; bonds own scale)");
            let (bond_cols, _bmax) = heat_bonds(&diag.bond_gsv_energy);
            Some(CrystalOverlayData {
                site_rgba: Some(site_cols),
                bond_rgba: Some(bond_cols),
                legend: Some(legend),
                note_suffix: Some(format!("AO E_tot={:.4} (E_shear={:.4})", diag.total_energy, diag.shear_energy)),
                ..Default::default()
            })
        }
        CrystalOverlayMode::BondStretchStrain => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            Some(bond_strain_overlay(crystal.pos.len(), &crystal.edges, &diag.bond_strain))
        }
        CrystalOverlayMode::VertexAngleShear => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            let (site_cols, legend) = perceptual_colorbar(&diag.site_shear, "Site shear E (√; bonds: |γ|)");
            let (bond_cols, _bmax) = heat_bonds(&diag.bond_shear_strain);
            Some(CrystalOverlayData {
                site_rgba: Some(site_cols),
                bond_rgba: Some(bond_cols),
                legend: Some(legend),
                ..Default::default()
            })
        }
        CrystalOverlayMode::PlaquetteFlux => {
            let hols = plaquette_holonomies(crystal, Some(group), Some(cell));
            let cat = DefectCatalogue::with_bounds(group, cell, frame, 0.0, 0.0);
            let wyck = wyckoff_rotocentres_with_frame(group, cell, frame).unwrap_or_default();
            let ainv = papertilling_core::defect::gauge::SpaceGroupExtension::from_cell(group, cell)
                .map_or([[1.0, 0.0], [0.0, 1.0]], |e| e.a_inv);
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
                let wy_lbl = p.centre.and_then(|c| match_wyckoff_label(&cat, &wyck, ainv, c));
                let label = if p.det_sign < 0 {
                    if p.nonsymmorphic_shift { String::from("glide det=-1") } else { String::from("mirror det=-1") }
                } else if p.has_curvature {
                    let prefix = wy_lbl.map(|w| format!("{w} ")).unwrap_or_default();
                    if p.nonsymmorphic_shift || (p.alpha_f[0].hypot(p.alpha_f[1]) > 1e-4 && p.has_torsion) {
                        format!("{prefix}Ω={:+.0}°+τ", p.omega_deg)
                    } else {
                        format!("{prefix}Ω={:+.0}°", p.omega_deg)
                    }
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
            let chips = crate::canvas2d::overlay_legend::flux_legend_chips(&badges);
            Some(CrystalOverlayData { badges, chips, ..Default::default() })
        }
        _ => None,
    }
}
