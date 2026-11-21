//! Known-answer tests of the 2D overlay legends: spectral colorbar ranges and end colours (colour-map formula
//! constants of `sheet_colors.rs`), categorical chips of hand-built badge / seam sets, and headless layout.

use std::collections::BTreeSet;

use super::*;
use crate::canvas2d::Canvas2D;
use crate::canvas2d::gauge_overlay::{LEGEND_RECT_TEST_ID, append_gauge_export_prims};
use crate::canvas2d::input::zoom_hud_rect;
use crate::scene2d::ScenePrimitives;

fn badge(is_rotation: bool, improper: bool) -> PlaquetteBadge {
    PlaquetteBadge { center: [0.0, 0.0], label: String::from("b"), is_rotation, improper }
}

fn seam(det_neg: bool) -> SeamConnectionMark {
    SeamConnectionMark { a: [0.0, 0.0], b: [1.0, 0.0], label: String::new(), det_neg, has_ge: true }
}

#[test]
fn spectral_charge_bar_is_symmetric_diverging_and_its_ends_match_the_extreme_sites() {
    let t0 = std::time::Instant::now();
    let ov = spectral_overlay_data(&[-0.3, 0.1], true);
    let leg = ov.legend.clone().expect("charge colorbar");
    assert!(leg.diverging && !leg.sqrt_scale);
    assert!((leg.min_val + 0.3).abs() < 1e-15 && (leg.max_val - 0.3).abs() < 1e-15, "{leg:?}");
    let sites = ov.site_rgba.expect("site colours");
    // Bar end at min is the colour of the site holding the minimum value -0.3.
    assert_eq!(colorbar_rgba(&leg, 0.0), sites[0]);
    // Hand values of `diverging_rgba` endpoints: lo = (0.13, 0.44, 0.82), hi = (0.84, 0.19, 0.17), x255 rounded.
    assert_eq!(colorbar_rgba(&leg, 0.0), [33, 112, 209, 255]);
    assert_eq!(colorbar_rgba(&leg, 1.0), [214, 48, 43, 255]);
    // A site at +0.3 under the same normalisation takes the max-end colour.
    let with_max = spectral_overlay_data(&[-0.3, 0.1, 0.3], true).site_rgba.unwrap();
    assert_eq!(colorbar_rgba(&leg, 1.0), with_max[2]);
    // Midpoint t = 0.5 is value 0: the colour of a site at 0 under the same normalisation.
    let zero = spectral_overlay_data(&[-0.3, 0.0], true).site_rgba.unwrap();
    assert_eq!(colorbar_rgba(&leg, 0.5), zero[1]);

    println!("# time spectral_charge {:?}", t0.elapsed());
}

#[test]
fn spectral_ldos_bar_is_sequential_from_zero_to_max_and_its_ends_match_the_sites() {
    let ov = spectral_overlay_data(&[0.0, 0.25, 1.5], false);
    let leg = ov.legend.clone().expect("ldos colorbar");
    assert!(!leg.diverging);
    assert_eq!((leg.min_val, leg.max_val), (0.0, 1.5));
    let sites = ov.site_rgba.expect("site colours");
    assert_eq!(colorbar_rgba(&leg, 0.0), sites[0]);
    assert_eq!(colorbar_rgba(&leg, 1.0), sites[2]);
    // Hand values of `sequential_heat_rgba` endpoints c0 = (0.89, 0.91, 0.94), c3 = (0.99, 0.85, 0.22).
    assert_eq!(sites[0], [227, 232, 240, 255]);
    assert_eq!(sites[2], [252, 217, 56, 255]);
}

#[test]
fn categorical_chips_are_one_per_distinct_painted_category() {
    // 2 rotations + 1 improper (an improper rotation counts as improper) -> 2 categories.
    let badges = [badge(true, false), badge(true, false), badge(true, true)];
    let chips = flux_legend_chips(&badges);
    assert_eq!(chips.len(), 2, "{chips:?}");
    assert_eq!(chips[0].category, LegendCategory::FluxImproper);
    assert_eq!(chips[1].category, LegendCategory::FluxRotation);
    // All three badge classes -> 3 chips; none -> none.
    assert_eq!(flux_legend_chips(&[badge(false, false), badge(true, false), badge(false, true)]).len(), 3);
    assert!(flux_legend_chips(&[]).is_empty());
    // Seams: three proper ones -> 1 chip; mixed -> 2.
    assert_eq!(seam_legend_chips(&[seam(false), seam(false), seam(false)]).len(), 1);
    assert_eq!(seam_legend_chips(&[seam(false), seam(true)]).len(), 2);

    // Chip export colours are exactly the colours the exported badges and seams are painted with.
    let ov = CrystalOverlayData {
        badges: vec![badge(true, false), badge(false, false), badge(false, true)],
        seams: vec![seam(true), seam(false)],
        ..Default::default()
    };
    let mut prims: ScenePrimitives = square_canvas().0.compiled.clone();
    let (c0, l0) = (prims.circles.len(), prims.lines.len());
    append_gauge_export_prims(&mut prims, &ov);
    let painted: BTreeSet<[u8; 4]> =
        prims.circles[c0..].iter().map(|c| c.stroke.0).chain(prims.lines[l0..].iter().map(|l| l.color)).collect();

    let chip_cols: BTreeSet<[u8; 4]> = flux_legend_chips(&ov.badges)
        .iter()
        .chain(seam_legend_chips(&ov.seams).iter())
        .map(|c| c.rgba)
        .collect();
    assert_eq!(painted, chip_cols);
    assert_eq!(chip_cols.len(), 5);
}

#[test]
fn categorical_legend_box_stays_inside_and_clear_of_the_zoom_hud() {
    for (w, h) in [(300.0_f32, 300.0_f32), (220.0, 200.0), (346.0, 220.0), (360.0, 200.0), (1280.0, 800.0)] {
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(w, h));
        for n in 1..=3 {
            let Some(leg) = categorical_legend_rect(rect, n) else { panic!("no legend at ({w}, {h}) n={n}") };
            assert!(rect.contains_rect(leg));
            let hud = zoom_hud_rect(rect).expect("hud");
            assert!(!leg.intersects(hud), "({w}, {h}) n={n}: {leg:?} vs {hud:?}");
        }
    }
    let tiny = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 300.0));
    assert!(categorical_legend_rect(tiny, 2).is_none());
    assert!(categorical_legend_rect(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)), 0).is_none());
}

fn square_canvas() -> (Canvas2D, papertilling_core::types::BuiltCrystal) {
    use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
    use papertilling_core::plane::catalogue::parse_named_str;
    let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
    let sq = named.named.into_iter().find(|n| n.key == "square").unwrap();
    let crystal = build_point_defects(&sq.cell, &[DefectSpec::disclination(90.0, [0.5, 0.5])], None, None).unwrap();
    let cv = Canvas2D::new(crate::scene2d::Scene2D::CrystalPatch {
        crystal: crystal.clone(),
        title: String::from("legend test"),
        note: String::new(),
    });
    (cv, crystal)
}

/// Headless egui_kittest frame of a canvas; returns the drawn `(canvas rect, legend rect)`.
fn drawn_legend(size: egui::Vec2, ov: CrystalOverlayData) -> Option<(egui::Rect, egui::Rect)> {
    let (mut cv, _) = square_canvas();
    cv.set_gauge_overlay(Some(ov));
    let tokens = crate::theme::ThemeTokens::paper();
    let mut harness = egui_kittest::Harness::builder()
        .with_size(size)
        .with_accessibility_check(false)
        .build_ui_state(move |ui, cv: &mut Canvas2D| cv.show(ui, &tokens), cv);
    harness.run();
    harness.ctx.data(|d| d.get_temp::<(egui::Rect, egui::Rect)>(egui::Id::unique(LEGEND_RECT_TEST_ID)))
}

#[test]
fn kittest_2d_legends_do_not_overlap_the_zoom_hud_at_large_and_small_viewports() {
    let t0 = std::time::Instant::now();
    let flux = CrystalOverlayData {
        badges: vec![badge(true, false), badge(false, true)],
        chips: flux_legend_chips(&[badge(true, false), badge(false, true)]),
        ..Default::default()
    };
    for size in [egui::vec2(1280.0, 800.0), egui::vec2(320.0, 240.0)] {
        for ov in [spectral_overlay_data(&[-0.3, 0.1], true), flux.clone()] {
            let (canvas, leg) = drawn_legend(size, ov).expect("legend drawn");
            let hud = zoom_hud_rect(canvas).expect("both legend and zoom HUD exist at this size");
            assert!(canvas.contains_rect(leg), "{leg:?} outside {canvas:?}");
            assert!(!leg.intersects(hud), "{size:?}: legend {leg:?} overlaps HUD {hud:?}");
        }
    }
    println!("# time kittest_2d_legends {:?}", t0.elapsed());
}

#[test]
fn exported_badges_and_seams_keep_their_five_hand_written_colours() {
    // The export colours of the flux badges and seams before legends existed (hand-written RGBA literals).
    let cases: [(CrystalOverlayData, [u8; 4]); 6] = [
        (CrystalOverlayData { badges: vec![badge(false, true)], ..Default::default() }, [142, 36, 170, 235]),
        // An improper rotation is painted as improper (improper takes precedence).
        (CrystalOverlayData { badges: vec![badge(true, true)], ..Default::default() }, [142, 36, 170, 235]),
        (CrystalOverlayData { badges: vec![badge(true, false)], ..Default::default() }, [198, 40, 40, 235]),
        (CrystalOverlayData { badges: vec![badge(false, false)], ..Default::default() }, [230, 81, 0, 235]),
        (CrystalOverlayData { seams: vec![seam(true)], ..Default::default() }, [186, 45, 118, 240]),
        (CrystalOverlayData { seams: vec![seam(false)], ..Default::default() }, [0, 131, 143, 235]),
    ];
    let base: ScenePrimitives = square_canvas().0.compiled.clone();
    for (ov, want) in cases {
        let mut prims = base.clone();
        let (c0, l0) = (prims.circles.len(), prims.lines.len());
        append_gauge_export_prims(&mut prims, &ov);
        let painted: Vec<[u8; 4]> =
            prims.circles[c0..].iter().map(|c| c.stroke.0).chain(prims.lines[l0..].iter().map(|l| l.color)).collect();
        assert_eq!(painted, vec![want], "{ov:?}");
        let chips: Vec<[u8; 4]> =
            flux_legend_chips(&ov.badges).iter().chain(seam_legend_chips(&ov.seams).iter()).map(|c| c.rgba).collect();
        assert_eq!(chips, vec![want], "chip of {ov:?}");
    }
}

#[test]
fn wall_bonds_without_ge_get_their_own_chip() {
    use super::{LegendCategory, seam_category, seam_legend_chips};
    let wall = SeamConnectionMark { a: [0.0, 0.0], b: [1.0, 0.0], label: String::from("wall"), det_neg: false, has_ge: false };
    assert_eq!(seam_category(&wall), LegendCategory::SeamWall);
    let chips = seam_legend_chips(&[wall, seam(false), seam(true)]);
    assert_eq!(chips.len(), 3, "wall, det=+1 and det=-1 are three categories");
    assert!(chips.iter().any(|c| c.label == "Wall bond (no g_e)"));
}

