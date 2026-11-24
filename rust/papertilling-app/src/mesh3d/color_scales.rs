//! Colour normalisations of the 3D sheet and surface meshes, shared by the mesh builders (`sheet.rs`,
//! `surface_elastic.rs`) and by the 3D colorbar legends, so a legend's range is the range the mesh is coloured with.

use papertilling_core::defect::sheet3d::Sheet3DResult;
use papertilling_core::surface::elastic3d::SurfaceElasticResult;
use papertilling_core::types::BuiltCrystal;

use super::sheet_modes::{Sheet3dColorMode, Sheet3dViewOpts};
use super::surface_elastic::SurfaceColor3d;
use crate::canvas2d::gauge_overlay::perceptual_heat_scale;
use crate::view3d::{ColorLegend3d, Legend3dMap};

/// Saturation of `strain_rgba` (|ε| = 8 %).
pub const STRAIN_SATURATION: f64 = 0.08;
/// Range of the face-strain fallback `sequential_heat_rgba(16 ε_f², 0.15)` used when no GSV site energies exist.
pub const FACE_STRAIN_ENERGY_VMAX: f64 = 0.15;

/// `max |v|` floored at 1e-6: the spectral (LDOS / δρ) colour normalisation of every mesh builder.
pub fn spectral_vmax(vals: &[f64]) -> f64 {
    vals.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1e-6)
}

/// Half-span of the radial-buckling `height_rgba` map of a relaxed surface (floored at 0.02; 0.05 without data).
pub fn buckle_span(el_res: Option<&SurfaceElasticResult>) -> f64 {
    el_res
        .map(|r| r.radial_buckling.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(0.02))
        .unwrap_or(0.05)
}

/// Per-site scalars and colour scales of a relaxed elastic sheet.
pub(crate) struct SheetColorScales {
    pub site_tot_e: Vec<f64>,
    pub bond_tot_e: Vec<f64>,
    pub site_shear: Vec<f64>,
    pub omega_span: f64,
    pub e_site_scale: f64,
    pub e_bond_scale: f64,
    pub sh_site_scale: f64,
    pub sh_bond_scale: f64,
    pub spec_vmax: f64,
    pub z_half_span: f64,
}

impl SheetColorScales {
    pub fn new(crystal: &BuiltCrystal, sheet: &Sheet3DResult, spectral: Option<(&[f64], bool)>) -> Self {
        let n = sheet.xyz.len();
        let site_tot_e: Vec<f64> = (0..n)
            .map(|u| {
                sheet.site_gsv_energy.get(u).copied().unwrap_or(0.0) + sheet.site_bend_energy.get(u).copied().unwrap_or(0.0)
            })
            .collect();
        let bond_tot_e: Vec<f64> = (0..crystal.edges.len())
            .map(|i| {
                sheet.bond_gsv_energy.get(i).copied().unwrap_or(0.0) + sheet.bond_bend_energy.get(i).copied().unwrap_or(0.0)
            })
            .collect();
        let mut site_shear = vec![0.0_f64; n];
        let mut site_sh_cnt = vec![0_usize; n];
        for (i, &(u, v)) in crystal.edges.iter().enumerate() {
            let sh = sheet.bond_shear_strain.get(i).copied().unwrap_or(0.0).abs();
            if u < n {
                site_shear[u] += sh;
                site_sh_cnt[u] += 1;
            }
            if v < n {
                site_shear[v] += sh;
                site_sh_cnt[v] += 1;
            }
        }
        for (s, c) in site_shear.iter_mut().zip(site_sh_cnt) {
            if c > 0 {
                *s /= c as f64;
            }
        }
        let omega_span = sheet.vertex_omega.iter().map(|w| w.abs()).fold(0.0_f64, f64::max).max(0.25);
        let (e_site_scale, _) = perceptual_heat_scale(&site_tot_e);
        let (e_bond_scale, _) = perceptual_heat_scale(&bond_tot_e);
        let (sh_site_scale, _) = perceptual_heat_scale(&site_shear);
        let (sh_bond_scale, _) = perceptual_heat_scale(&sheet.bond_shear_strain);
        let spec_vmax = spectral.map_or(1.0, |(v, _)| spectral_vmax(v));
        let z_half_span = sheet.xyz.iter().map(|p| p[2].abs()).fold(0.0_f64, f64::max).max(0.22 * sheet.mean_bond_len);
        Self {
            site_tot_e,
            bond_tot_e,
            site_shear,
            omega_span,
            e_site_scale,
            e_bond_scale,
            sh_site_scale,
            sh_bond_scale,
            spec_vmax,
            z_half_span,
        }
    }
}

fn legend(title: &str, min_val: f64, max_val: f64, map: Legend3dMap) -> Option<ColorLegend3d> {
    Some(ColorLegend3d { title: title.to_owned(), min_val, max_val, map })
}

fn spectral_legend(spectral: Option<(&[f64], bool)>) -> Option<ColorLegend3d> {
    let (vals, div) = spectral?;
    let vmax = spectral_vmax(vals);
    if div {
        legend("Spectral charge δρ(μ)", -vmax, vmax, Legend3dMap::Diverging)
    } else {
        legend("LDOS |ψ|²", 0.0, vmax, Legend3dMap::Sequential)
    }
}

/// Colorbar of a relaxed sheet in `opts.color_mode`, with the range `mesh_from_sheet3d_with_spectral` colours it
/// with; `None` for face ring size and whenever the builder falls back to ring / sublattice colours.
pub fn sheet_color_legend(
    crystal: &BuiltCrystal,
    sheet: &Sheet3DResult,
    opts: &Sheet3dViewOpts,
    spectral: Option<(&[f64], bool)>,
) -> Option<ColorLegend3d> {
    if sheet.xyz.is_empty() || !super::sheet::renderable(crystal, sheet, opts) {
        return None;
    }
    match opts.color_mode {
        Sheet3dColorMode::FaceRingSize => None,
        Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge => spectral_legend(spectral),
        Sheet3dColorMode::ElasticEnergyGsv if sheet.site_gsv_energy.is_empty() => {
            legend("Face strain energy 16ε²", 0.0, FACE_STRAIN_ENERGY_VMAX, Legend3dMap::Sequential)
        }
        Sheet3dColorMode::ElasticEnergyGsv => {
            let sc = SheetColorScales::new(crystal, sheet, None);
            legend("E_site (√; wires own scale)", 0.0, sc.e_site_scale, Legend3dMap::SequentialSqrt)
        }
        Sheet3dColorMode::BondShearStrain if sheet.bond_shear_strain.is_empty() => None,
        Sheet3dColorMode::BondShearStrain => {
            let sc = SheetColorScales::new(crystal, sheet, None);
            legend("Mean |γ| per site (wires own scale)", 0.0, sc.sh_site_scale, Legend3dMap::SequentialSqrt)
        }
        Sheet3dColorMode::BondStrain => {
            legend("Signed bond strain ε", -STRAIN_SATURATION, STRAIN_SATURATION, Legend3dMap::Strain)
        }
        Sheet3dColorMode::GaussianCurvature => {
            let sc = SheetColorScales::new(crystal, sheet, None);
            legend("Angle deficit Ω_v (rad)", -sc.omega_span, sc.omega_span, Legend3dMap::Diverging)
        }
        Sheet3dColorMode::HeightZ => {
            let sc = SheetColorScales::new(crystal, sheet, None);
            legend("Height z", -sc.z_half_span, sc.z_half_span, Legend3dMap::Height)
        }
    }
}

/// Colorbar of a surface mesh in `color_mode` with the range `elastic_surface_gpu_mesh` colours it with;
/// `None` for face ring size and when the mode's data (relaxed result or spectrum) is missing.
pub fn surface_color_legend(
    el_res: Option<&SurfaceElasticResult>,
    color_mode: SurfaceColor3d,
    spectral: Option<(&[f64], bool)>,
) -> Option<ColorLegend3d> {
    match color_mode {
        SurfaceColor3d::FaceRingSize => None,
        SurfaceColor3d::SpectralLdos | SurfaceColor3d::SpectralCharge => spectral_legend(spectral),
        SurfaceColor3d::BondStrain => el_res.and_then(|_| {
            legend("Site strain ε", -STRAIN_SATURATION, STRAIN_SATURATION, Legend3dMap::Strain)
        }),
        SurfaceColor3d::RadialBuckling => el_res.and_then(|r| {
            let s = buckle_span(Some(r));
            legend("Radial buckling r - R_mean", -s, s, Legend3dMap::Height)
        }),
    }
}
