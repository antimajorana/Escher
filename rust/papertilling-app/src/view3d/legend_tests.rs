//! 3D colorbar legend: None for categorical colouring, Some for every scalar mode, bar ends equal to the colours
//! the mesh builders paint the extreme sites with, hand ranges, and a headless egui_kittest layout check.

use papertilling_core::defect::sheet3d::{Sheet3DOptions, Sheet3DResult, relax_sheet_3d};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::catalogue::parse_named_str;
use papertilling_core::surface::elastic3d::SurfaceElasticResult;
use papertilling_core::types::BuiltCrystal;

use super::legend::{legend_rect_3d, zoom_hud_rect_3d};
use super::{ColorLegend3d, Legend3dMap, View3dState};
use crate::mesh3d::color_scales::{sheet_color_legend, surface_color_legend};
use crate::mesh3d::sheet_colors::rgba_u8;
use crate::mesh3d::{GpuMeshData, Sheet3dColorMode, Sheet3dViewOpts, SurfaceColor3d, Vertex3d, mesh_from_sheet3d_with_spectral};

fn square_sheet() -> (BuiltCrystal, Sheet3DResult) {
    let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
    let cell = named.named.into_iter().find(|n| n.key == "square").unwrap().cell;
    let crystal = build_point_defects(&cell, &[DefectSpec::disclination(90.0, [0.0, 0.0])], Some(3.5), None).unwrap();
    let sheet = relax_sheet_3d(&crystal, &cell, &Sheet3DOptions { max_iters: 5, ..Default::default() }, None);
    (crystal, sheet)
}

fn opaque_colors(mesh: &GpuMeshData) -> Vec<[u8; 4]> {
    mesh.faces[..mesh.opaque_vertex_count].iter().map(|v| rgba_u8(v.color)).collect()
}

#[test]
fn sheet_legend_is_none_for_ring_size_and_some_for_every_scalar_mode() {
    let t0 = std::time::Instant::now();
    let (crystal, sheet) = square_sheet();
    assert!(!sheet.site_gsv_energy.is_empty() && !sheet.bond_shear_strain.is_empty(), "{:?}", sheet.diagnostics);
    let n = crystal.pos.len();
    // Spectral data supplied for the spectral modes (without it the builder falls back to sublattice colours).
    let ldos: Vec<f64> = (0..n).map(|u| if u == 0 { 0.0 } else if u == 1 { 1.5 } else { 0.25 }).collect();
    let charge: Vec<f64> = (0..n).map(|u| if u == 0 { -0.3 } else { 0.1 }).collect();
    let none_opts = Sheet3dViewOpts { color_mode: Sheet3dColorMode::FaceRingSize, ..Default::default() };
    assert!(sheet_color_legend(&crystal, &sheet, &none_opts, None).is_none());
    for mode in Sheet3dColorMode::ALL.into_iter().filter(|m| *m != Sheet3dColorMode::FaceRingSize) {
        let opts = Sheet3dViewOpts { color_mode: mode, ..Default::default() };
        let spectral = match mode {
            Sheet3dColorMode::SpectralLdos => Some((ldos.as_slice(), false)),
            Sheet3dColorMode::SpectralCharge => Some((charge.as_slice(), true)),
            _ => None,
        };
        let leg = sheet_color_legend(&crystal, &sheet, &opts, spectral).unwrap_or_else(|| panic!("{mode:?}"));
        assert!(leg.min_val < leg.max_val && leg.max_val.is_finite(), "{mode:?}: {leg:?}");
    }
    // Hand ranges: spectral LDOS [0, 1.5] sequential, charge [-0.3, 0.3] diverging, strain +/-8 %.
    let leg_of = |mode, spectral| {
        sheet_color_legend(&crystal, &sheet, &Sheet3dViewOpts { color_mode: mode, ..Default::default() }, spectral).unwrap()
    };
    let l = leg_of(Sheet3dColorMode::SpectralLdos, Some((ldos.as_slice(), false)));
    assert_eq!((l.min_val, l.max_val, l.map), (0.0, 1.5, Legend3dMap::Sequential));
    let c = leg_of(Sheet3dColorMode::SpectralCharge, Some((charge.as_slice(), true)));
    assert!((c.min_val + 0.3).abs() < 1e-15 && (c.max_val - 0.3).abs() < 1e-15 && c.map == Legend3dMap::Diverging);
    let s = leg_of(Sheet3dColorMode::BondStrain, None);
    assert_eq!((s.min_val, s.max_val, s.map), (-0.08, 0.08, Legend3dMap::Strain));
    // Spectral mode without spectral data: the mesh keeps sublattice colours, so no colorbar.
    assert!(leg_of_opt(&crystal, &sheet, Sheet3dColorMode::SpectralLdos).is_none());
    println!("# time sheet_legend_modes {:?}", t0.elapsed());
}

fn leg_of_opt(c: &BuiltCrystal, s: &Sheet3DResult, mode: Sheet3dColorMode) -> Option<ColorLegend3d> {
    sheet_color_legend(c, s, &Sheet3dViewOpts { color_mode: mode, ..Default::default() }, None)
}

#[test]
fn sheet_bar_ends_equal_the_mesh_colours_of_the_extreme_sites() {
    let (crystal, sheet) = square_sheet();
    let n = crystal.pos.len();
    let ldos: Vec<f64> = (0..n).map(|u| if u == 0 { 0.0 } else if u == 1 { 1.5 } else { 0.25 }).collect();
    let opts = Sheet3dViewOpts { color_mode: Sheet3dColorMode::SpectralLdos, ..Default::default() };
    let mesh = mesh_from_sheet3d_with_spectral(&crystal, &sheet, &opts, Some((&ldos, false)));
    let leg = sheet_color_legend(&crystal, &sheet, &opts, Some((&ldos, false))).unwrap();
    let cols = opaque_colors(&mesh);
    assert!(cols.contains(&leg.sample(0.0)) && cols.contains(&leg.sample(1.0)));

    let charge: Vec<f64> = (0..n).map(|u| if u == 0 { -0.3 } else { 0.1 }).collect();
    let opts = Sheet3dViewOpts { color_mode: Sheet3dColorMode::SpectralCharge, ..Default::default() };
    let mesh = mesh_from_sheet3d_with_spectral(&crystal, &sheet, &opts, Some((&charge, true)));
    let leg = sheet_color_legend(&crystal, &sheet, &opts, Some((&charge, true))).unwrap();
    let cols = opaque_colors(&mesh);
    assert!(cols.contains(&leg.sample(0.0)), "site at -0.3 takes the min-end colour");
    assert!(cols.contains(&leg.rgba_of_value(0.1)));

    // Gaussian curvature: the site of largest |Omega_v| sits at an end of the bar when it exceeds the 0.25 floor.
    let opts = Sheet3dViewOpts { color_mode: Sheet3dColorMode::GaussianCurvature, ..Default::default() };
    let mesh = mesh_from_sheet3d_with_spectral(&crystal, &sheet, &opts, None);
    let leg = sheet_color_legend(&crystal, &sheet, &opts, None).unwrap();
    let w = sheet.vertex_omega.iter().copied().fold(0.0_f64, |a, b| if b.abs() > a.abs() { b } else { a });
    assert!(w.abs() > 0.25, "a 90 degree cone carries curvature {w}");
    let end = if w > 0.0 { leg.sample(1.0) } else { leg.sample(0.0) };
    assert!(opaque_colors(&mesh).contains(&end));
}

#[test]
fn surface_legend_needs_its_data_and_uses_the_buckling_span_floor() {
    let el = |buckle: Vec<f64>| SurfaceElasticResult {
        xyz: Vec::new(),
        cover_xyz: None,
        stretch_energy: 0.0,
        bend_energy: 0.0,
        total_energy: 0.0,
        r_mean: 1.0,
        fvk_gamma: 0.0,
        asphericity: 0.0,
        bond_strain: Vec::new(),
        site_strain: vec![0.0; buckle.len()],
        radial_buckling: buckle,
        converged: true,
        n_iter: 0,
    };
    let r = el(vec![-0.01, 0.03, 0.005]);
    assert!(surface_color_legend(Some(&r), SurfaceColor3d::FaceRingSize, None).is_none());
    let b = surface_color_legend(Some(&r), SurfaceColor3d::RadialBuckling, None).unwrap();
    assert_eq!((b.min_val, b.max_val, b.map), (-0.03, 0.03, Legend3dMap::Height));
    let small = surface_color_legend(Some(&el(vec![0.001])), SurfaceColor3d::RadialBuckling, None).unwrap();
    assert_eq!(small.max_val, 0.02, "span floor 0.02");
    let s = surface_color_legend(Some(&r), SurfaceColor3d::BondStrain, None).unwrap();
    assert_eq!((s.min_val, s.max_val, s.map), (-0.08, 0.08, Legend3dMap::Strain));
    assert!(surface_color_legend(None, SurfaceColor3d::BondStrain, None).is_none());
    assert!(surface_color_legend(None, SurfaceColor3d::RadialBuckling, None).is_none());
    let ldos = [0.0, 0.25, 1.5];
    let l = surface_color_legend(None, SurfaceColor3d::SpectralLdos, Some((&ldos, false))).unwrap();
    assert_eq!((l.min_val, l.max_val, l.map), (0.0, 1.5, Legend3dMap::Sequential));
    let q = surface_color_legend(None, SurfaceColor3d::SpectralCharge, Some((&[-0.3, 0.1], true))).unwrap();
    assert!((q.min_val + 0.3).abs() < 1e-15 && (q.max_val - 0.3).abs() < 1e-15 && q.map == Legend3dMap::Diverging);
    assert!(surface_color_legend(None, SurfaceColor3d::SpectralCharge, None).is_none());
}

#[test]
fn sqrt_bar_is_linear_in_colour_and_quadratic_in_value() {
    let leg = ColorLegend3d { title: String::from("E"), min_val: 0.0, max_val: 4.0, map: Legend3dMap::SequentialSqrt };
    assert_eq!(leg.value_at(0.5), 1.0);
    assert_eq!(leg.sample(0.5), leg.rgba_of_value(1.0));
    // sqrt(1/4) = 1/2: the bar midpoint is the colour of half the sequential map, sequential_heat(0.5, 1).
    let mid = ColorLegend3d { map: Legend3dMap::Sequential, max_val: 1.0, ..leg.clone() };
    assert_eq!(leg.sample(0.5), mid.rgba_of_value(0.5));
}

#[test]
fn set_mesh_clears_the_legend() {
    let mut v = View3dState::default();
    v.set_legend(Some(ColorLegend3d { title: String::from("h"), min_val: -1.0, max_val: 1.0, map: Legend3dMap::Height }));
    assert!(v.legend().is_some());
    v.set_mesh(GpuMeshData::default(), "other tab");
    assert!(v.legend().is_none());
}

#[test]
fn kittest_3d_legend_clears_zoom_hud_caption_and_hint_at_large_and_small_viewports() {
    let t0 = std::time::Instant::now();
    for size in [egui::vec2(1280.0, 800.0), egui::vec2(260.0, 180.0)] {
        let mut view = View3dState::default();
        let mesh = GpuMeshData { faces: vec![Vertex3d::ZERO; 3], radius: 1.0, ..Default::default() };
        view.set_mesh(mesh, "a long caption that runs across the top of the viewport | V=1 E=2 F=3");
        view.set_legend(Some(ColorLegend3d {
            title: String::from("Height z"),
            min_val: -0.5,
            max_val: 0.5,
            map: Legend3dMap::Height,
        }));
        let mut harness = egui_kittest::Harness::builder()
            .with_size(size)
            .with_accessibility_check(false)
            .build_ui_state(|ui, v: &mut View3dState| v.show(ui, &crate::theme::ThemeTokens::studio()), view);
        harness.run();
        let (rect, hud, leg) = harness.state().last_layout.expect("drawn");
        let hud = hud.expect("zoom HUD present");
        let leg = leg.expect("legend present");
        assert_eq!(Some(hud), zoom_hud_rect_3d(rect));
        assert_eq!(Some(leg), legend_rect_3d(rect));
        assert!(rect.contains_rect(leg), "{leg:?} outside {rect:?}");
        assert!(!leg.intersects(hud), "{size:?}: legend {leg:?} overlaps zoom HUD {hud:?}");
        let caption_band = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.right(), rect.top() + 24.0));
        let hint_band = egui::Rect::from_min_max(egui::pos2(rect.left(), rect.bottom() - 22.0), rect.right_bottom());
        assert!(!leg.intersects(caption_band) && !leg.intersects(hint_band), "{size:?}: {leg:?}");
    }
    // Too small for a legend (but the HUD remains): no legend rect.
    assert!(legend_rect_3d(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(195.0, 120.0))).is_none());
    println!("# time kittest_3d_legend {:?}", t0.elapsed());
}
