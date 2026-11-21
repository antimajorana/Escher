//! Site colours and their colorbar from one normalisation, for the AO overlays whose maps are not plain linear:
//! the perceptual √-heat scale of GSV / vertex-shear energies and the saturating bond-stretch map.

use crate::mesh3d::sheet_colors::{rgba_u8, strain_rgba};

use super::gauge_overlay::{CrystalOverlayData, OverlayColorbar, color_diverging_strain, color_heat_perceptual, perceptual_heat_scale};

/// `strain_rgba` saturates where its argument reaches ±0.08.
pub const STRAIN_SATURATION: f64 = 0.08;
/// Gain of the bond-stretch map: a bond of strain `±vmax` is drawn as `strain_rgba(±0.18)` (`color_diverging_strain`).
pub const STRAIN_GAIN: f64 = 0.18;

/// Perceptually scaled heat colours of `values` and their √-scale colorbar over `[0, scale]`, where `scale` is the
/// value the sites are normalised by (`perceptual_heat_scale`): sites at or above it all take the bar's top colour.
pub fn perceptual_colorbar(values: &[f64], title: &str) -> (Vec<[u8; 4]>, OverlayColorbar) {
    let (cols, _) = color_heat_perceptual(values);
    let (scale, _) = perceptual_heat_scale(values);
    let legend = OverlayColorbar {
        title: title.to_owned(),
        min_val: 0.0,
        max_val: scale,
        diverging: false,
        sqrt_scale: true,
        strain_map: false,
    };
    (cols, legend)
}

/// Bond-stretch overlay: bonds by `color_diverging_strain(.., 1e-3)` (`strain_rgba(ε · 0.18 / vmax)`), sites by the
/// mean strain of their bonds under the same map, and a `strain_map` colorbar over the unsaturated range
/// `±0.08 / 0.18 · vmax` so its ends are the colours of the saturated bonds.
pub fn bond_strain_overlay(n_sites: usize, edges: &[(usize, usize)], bond_strain: &[f64]) -> CrystalOverlayData {
    let (bond_cols, vmax) = color_diverging_strain(bond_strain, 1e-3);
    let mut site_eps = vec![0.0_f64; n_sites];
    let mut site_cnt = vec![0_usize; n_sites];
    for (idx, &(u, v)) in edges.iter().enumerate() {
        let e = bond_strain.get(idx).copied().unwrap_or(0.0);
        if u < n_sites {
            site_eps[u] += e;
            site_cnt[u] += 1;
        }
        if v < n_sites {
            site_eps[v] += e;
            site_cnt[v] += 1;
        }
    }
    let gain = STRAIN_GAIN / vmax.max(1e-6);
    let site_cols = site_eps
        .iter()
        .zip(site_cnt)
        .map(|(&s, c)| rgba_u8(strain_rgba(if c > 0 { s / c as f64 } else { s } * gain, 1.0)))
        .collect();
    let half = STRAIN_SATURATION / gain;
    CrystalOverlayData {
        site_rgba: Some(site_cols),
        bond_rgba: Some(bond_cols),
        legend: Some(OverlayColorbar {
            title: String::from("Bond stretch ε=(l-l₀)/l₀"),
            min_val: -half,
            max_val: half,
            diverging: true,
            sqrt_scale: false,
            strain_map: true,
        }),
        ..Default::default()
    }
}

#[cfg(test)]
#[path = "overlay_scales_tests.rs"]
mod tests;
