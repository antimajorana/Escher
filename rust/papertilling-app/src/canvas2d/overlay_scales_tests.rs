//! Known answers for the AO colorbars: the perceptual scale `max(P95, 0.22 · max)` by hand, and colour-map
//! endpoints from the constants of `sheet_colors.rs` (x 255, rounded).

use super::*;
use crate::canvas2d::overlay_legend::colorbar_rgba;

const HEAT_TOP: [u8; 4] = [252, 217, 56, 255]; // c3 = (0.99, 0.85, 0.22)
const STRAIN_HI: [u8; 4] = [227, 56, 43, 255]; // hi = (0.89, 0.22, 0.17)
const STRAIN_LO: [u8; 4] = [36, 107, 222, 255]; // lo = (0.14, 0.42, 0.87)

#[test]
fn perceptual_bar_tops_out_at_the_saturation_scale_not_the_outlier() {
    let t0 = std::time::Instant::now();
    // 98 sites at 0.01, one at 0.22, one outlier at 1.0: P95 = 0.01, so scale = 0.22 · 1.0 = 0.22.
    let mut v = vec![0.01; 98];
    v.push(0.22);
    v.push(1.0);
    let (sites, leg) = perceptual_colorbar(&v, "GSV");
    assert!(leg.sqrt_scale && !leg.diverging && !leg.strain_map);
    assert_eq!(leg.min_val, 0.0);
    assert!((leg.max_val - 0.22).abs() < 1e-15, "bar max is the scale 0.22, not the outlier 1.0: {leg:?}");
    // The bar's top colour is the one both the 0.22 site and the 1.0 outlier are painted with.
    assert_eq!(colorbar_rgba(&leg, 1.0), HEAT_TOP);
    assert_eq!(sites[98], HEAT_TOP);
    assert_eq!(sites[99], HEAT_TOP);
    // Mid-bar (t = 0.5) on a √-scale is v = 0.25 · 0.22 = 0.055: sequential(0.5) = c1 + 0.375 (c2 - c1)
    // = (0.47875, 0.4775, 0.54625) -> (122, 122, 139).
    assert_eq!(colorbar_rgba(&leg, 0.5), [122, 122, 139, 255]);
    let (mid, _) = perceptual_colorbar(&[0.01, 0.055, 0.22, 1.0], "GSV");
    // Here P95 = 0.22 (index 2 of 4) and 0.22 · 1.0 = 0.22, so the scale is again 0.22.
    assert_eq!(mid[1], [122, 122, 139, 255]);

    // 19 sites at 0.1 and one at 10: P95 = 0.1, scale = 0.22 · 10 = 2.2.
    let mut w = vec![0.1; 19];
    w.push(10.0);
    let (_, leg2) = perceptual_colorbar(&w, "shear");
    assert!((leg2.max_val - 2.2).abs() < 1e-12, "{leg2:?}");
    println!("# time perceptual_bar {:?}", t0.elapsed());
}

#[test]
fn bond_stretch_bar_spans_the_unsaturated_range_and_its_ends_are_the_saturated_bonds() {
    // Path 0 - 1 - 2 with bond strains [-0.002, 0.004]: vmax = 0.004, bonds drawn as strain_rgba(ε · 0.18 / 0.004),
    // saturating at |ε| = 0.08 · 0.004 / 0.18 = 0.004 · 4/9.
    let ov = bond_strain_overlay(3, &[(0, 1), (1, 2)], &[-0.002, 0.004]);
    let leg = ov.legend.clone().expect("bond stretch colorbar");
    assert!(leg.strain_map && leg.diverging);
    let half = 0.004 * 4.0 / 9.0;
    assert!((leg.max_val - half).abs() < 1e-15 && (leg.min_val + half).abs() < 1e-15, "{leg:?}");
    let bonds = ov.bond_rgba.expect("bond colours");
    // ε = -0.002 is past -half: saturated blue = the bar's min end; ε = 0.004 is the bar's max end.
    assert_eq!(bonds[0].0, STRAIN_LO);
    assert_eq!(bonds[1].0, STRAIN_HI);
    assert_eq!(colorbar_rgba(&leg, 0.0), STRAIN_LO);
    assert_eq!(colorbar_rgba(&leg, 1.0), STRAIN_HI);
    // Sites use the same map: site 0 mean -0.002 (saturated), site 2 mean 0.004, site 1 mean 0.001 ->
    // strain_rgba(0.045) = mid + 0.5625 (hi - mid) = (0.8725, 0.50875, 0.498125) -> (222, 130, 127).
    let sites = ov.site_rgba.expect("site colours");
    assert_eq!(sites[0], STRAIN_LO);
    assert_eq!(sites[2], STRAIN_HI);
    assert_eq!(sites[1], [222, 130, 127, 255]);
    // Bar position of ε = 0.001 has the same colour.
    assert_eq!(colorbar_rgba(&leg, (0.001 + half) / (2.0 * half)), [222, 130, 127, 255]);
}
