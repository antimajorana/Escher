//! Quarantine and end-to-end verification tests for `3. Analysis` and the three-mode shell.

use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::space3d::nets3d::LineDefect3d;

use super::{AnalysisTarget, SpectrumColorOverlay};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::mesh3d::DefectViewMode;
use crate::theme::{StudioTheme, typography};

const FORBIDDEN_SPECTRAL: [&str; 6] = [
    "Tight-Binding",
    "Compute Spectrum",
    "LDOS",
    "Spectral charge",
    "δρ",
    "Peierls",
];

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, app| {
                let _ = app.frame_ui(ui);
            },
            app,
        )
}

fn click_combobox_value(h: &mut Harness<'static, PapertillingApp>, val: &str) {
    h.get_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|n| n.value().as_deref() == Some(val))
        .unwrap_or_else(|| panic!("no ComboBox with selected value '{val}'"))
        .click();
}

fn assert_no_forbidden_tokens(h: &Harness<'_, PapertillingApp>, where_ctx: &str) {
    for tok in FORBIDDEN_SPECTRAL {
        assert!(
            h.query_all_by_label_contains(tok).next().is_none(),
            "{where_ctx}: forbidden spectral token '{tok}' leaked into non-Analysis UI"
        );
        assert!(
            h.query_all_by_role(egui::accesskit::Role::ComboBox)
                .all(|n| !n.value().is_some_and(|v| v.contains(tok))),
            "{where_ctx}: forbidden spectral token '{tok}' leaked into ComboBox value"
        );
    }
    let app = h.state();
    if let Some(leg) = app.view3d.legend() {
        for tok in ["LDOS", "Spectral charge", "δρ"] {
            assert!(
                !leg.title.contains(tok),
                "{where_ctx}: spectral 3D legend '{}' leaked outside Analysis",
                leg.title
            );
        }
    }
    if let Some(ov) = app.canvas2d.gauge_overlay()
        && let Some(leg) = &ov.legend
    {
        for tok in ["LDOS", "Spectral charge", "δρ"] {
            assert!(
                !leg.title.contains(tok),
                "{where_ctx}: spectral 2D legend '{}' leaked outside Analysis",
                leg.title
            );
        }
    }
}

fn wait_until(h: &mut Harness<'static, PapertillingApp>, what: &str, pred: impl Fn(&PapertillingApp) -> bool) {
    let t0 = Instant::now();
    loop {
        h.step();
        if pred(h.state()) {
            h.run_steps(2);
            if pred(h.state()) {
                return;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(45), "{what}: timed out waiting for idle");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn spectral_ui_and_overlays_are_quarantined_to_analysis_mode_only() {
    let mut h = harness();
    h.run_steps(3);

    // 1. Lattices: visit all 4 sub-views including 0D/1D, 80-layer, 230-SG, H², and Manifolds + "3D Colour:" ComboBox.
    h.get_by_label("1. Lattices").click();
    h.run_steps(2);
    assert_eq!((h.state().active_mode, h.state().lattices_view), (AppMode::Lattices, LatticesView::Plane2D));
    assert_no_forbidden_tokens(&h, "1. Lattices -> 2D Plane");

    h.get_by_label("3D & Subperiodic").click();
    h.run_steps(2);
    assert!(h.state().atlas.pristine_only);
    for st_lbl in ["0D/1D Subperiodic", "80 Layer Groups", "230 3D Space Groups"] {
        h.get_by_label(st_lbl).click();
        h.run_steps(2);
        assert_no_forbidden_tokens(&h, st_lbl);
        assert!(
            h.query_all_by_label_contains("Line defect:").next().is_none()
                && h.query_all_by_label_contains("Frieze defect:").next().is_none(),
            "{st_lbl}: defect controls must be hidden in pristine Lattices mode"
        );
    }

    h.get_by_label("Non-Euclidean").click();
    h.run_steps(2);
    assert!(h.state().atlas.pristine_only);
    assert_no_forbidden_tokens(&h, "1. Lattices -> Non-Euclidean");
    assert!(
        h.query_all_by_label_contains("Central Wedge").next().is_none(),
        "Non-Euclidean pristine view must hide Central Wedge disclination controls"
    );

    h.get_by_label("Manifolds & Surfaces").click();
    wait_until(&mut h, "Manifolds surface build", |a| !a.surfaces.is_building());
    assert_no_forbidden_tokens(&h, "1. Lattices -> Manifolds");
    // Open "3D Colour:" ComboBox in inspector and verify ONLY non-spectral modes appear.
    click_combobox_value(&mut h, "Face ring size");
    h.run_steps(2);
    assert!(
        h.query_all_by_label_contains("Radial buckling").next().is_some(),
        "Manifolds 3D Colour ComboBox must open to check variants"
    );
    assert_no_forbidden_tokens(&h, "1. Lattices -> Manifolds (3D Colour popup open)");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    // 2. Defects: visit all 3 sub-views + open "2D Overlay:" and "Color:" ComboBoxes.
    h.get_by_label("2. Defects").click();
    wait_until(&mut h, "Defects Catalogue build", |a| !a.defects.is_busy() && !a.defects.initializing);
    assert_eq!((h.state().active_mode, h.state().defects_view), (AppMode::Defects, DefectsView::Catalogue2D));
    assert_no_forbidden_tokens(&h, "2. Defects -> Catalogue (2D)");

    // Open "Color:" (3D sheet) ComboBox.
    click_combobox_value(&mut h, "Face ring size");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Signed bond strain").next().is_some(), "Defects Color ComboBox opened");
    assert_no_forbidden_tokens(&h, "2. Defects -> Catalogue (2D) Color popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    // Open "2D Overlay:" ComboBox.
    click_combobox_value(&mut h, "Standard sublattice");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Plaquette flux").next().is_some(), "Defects 2D Overlay ComboBox opened");
    assert_no_forbidden_tokens(&h, "2. Defects -> Catalogue (2D) 2D Overlay popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    h.get_by_label("3D / H² / 0D–1D Defects").click();
    h.run_steps(2);
    assert!(!h.state().atlas.pristine_only);
    for st_lbl in ["3D Line Defects", "H² Wedge Disclinations", "0D/1D Subperiodic Defects"] {
        h.get_by_label(st_lbl).click();
        h.run_steps(2);
        assert_no_forbidden_tokens(&h, st_lbl);
    }

    h.get_by_label("Combine, Fuse & Move").click();
    wait_until(&mut h, "Combine build", |a| !a.combine.is_building());
    assert_no_forbidden_tokens(&h, "2. Defects -> Combine");

    click_combobox_value(&mut h, "Face ring size");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Signed bond strain").next().is_some(), "Combine Color ComboBox opened");
    assert_no_forbidden_tokens(&h, "2. Defects -> Combine Color popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    click_combobox_value(&mut h, "Standard sublattice");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Plaquette flux").next().is_some(), "Combine 2D Overlay ComboBox opened");
    assert_no_forbidden_tokens(&h, "2. Defects -> Combine 2D Overlay popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    // 3. Analysis mode: verify Tight-Binding panel appears, compute spectrum, verify LDOS + Peierls + 2D/3D legends.
    h.get_by_label("3. Analysis").click();
    wait_until(&mut h, "Analysis source idle", |a| !a.defects.is_busy() && !a.defects.initializing);
    assert_eq!((h.state().active_mode, h.state().active_tab), (AppMode::Analysis, NavTab::Analysis));
    assert!(h.query_all_by_label_contains("Tight-Binding").next().is_some(), "Tight-Binding panel visible in Analysis");
    assert!(h.query_all_by_label_contains("Peierls").next().is_some(), "Peierls flux slider visible for crystal target");

    h.get_by_label("Compute Spectrum").click();
    wait_until(&mut h, "Compute Spectrum", |a| !a.analysis.is_busy() && a.analysis.spectrum.result.is_some() && !a.analysis.view3d_dirty);

    assert!(h.query_all_by_label_contains("Tight-Binding").next().is_some(), "Tight-Binding header kept after compute");
    assert!(h.query_all_by_label_contains("LDOS").next().is_some(), "LDOS option visible in Analysis");
    assert!(h.query_all_by_label_contains("Peierls").next().is_some(), "Peierls slider visible in Analysis");
    let app = h.state();
    let leg3 = app.view3d.legend().expect("3D spectral legend present in Analysis");
    assert!(leg3.title.contains("LDOS"), "3D legend: {}", leg3.title);
    let leg2 = app
        .canvas2d
        .gauge_overlay()
        .and_then(|o| o.legend.as_ref())
        .expect("2D spectral legend present in Analysis");
    assert!(leg2.title.contains("LDOS"), "2D legend: {}", leg2.title);

    // 4. Switching back to Lattices and Defects clears spectral legends and overlays immediately.
    h.get_by_label("1. Lattices").click();
    h.run_steps(2);
    assert!(h.state().view3d.legend().is_none(), "3D spectral legend cleared on returning to Lattices");
    assert!(h.state().canvas2d.gauge_overlay().is_none(), "2D spectral overlay cleared on returning to Lattices");
    assert_no_forbidden_tokens(&h, "Return to 1. Lattices");

    h.get_by_label("2. Defects").click();
    wait_until(&mut h, "Return to 2. Defects", |a| !a.defects.is_busy());
    assert!(h.state().view3d.legend().is_none(), "3D spectral legend cleared on returning to Defects");
    assert!(h.state().canvas2d.gauge_overlay().is_none(), "2D spectral overlay cleared on returning to Defects");
    assert_no_forbidden_tokens(&h, "Return to 2. Defects");
}

#[test]
fn atlas_preserves_defect_selection_across_pristine_lattices_round_trip() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    app.select_defects_view(DefectsView::AtlasDefects);
    assert!(!app.atlas.pristine_only);
    assert_ne!(app.atlas.space3d.selected_defect, LineDefect3d::None);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, -1);

    app.atlas.space3d.selected_defect = LineDefect3d::PureDisclination90_123;
    app.atlas.hyperbolic.wedge_delta = -2;

    app.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
    assert!(app.atlas.pristine_only);
    assert_eq!(app.atlas.space3d.effective_defect(), LineDefect3d::None);

    app.select_lattices_view(LatticesView::NonEuclidean);
    assert!(app.atlas.pristine_only);
    assert_eq!(app.atlas.hyperbolic.effective_wedge(), 0);
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::HyperbolicH2);

    app.select_defects_view(DefectsView::AtlasDefects);
    assert!(!app.atlas.pristine_only);
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::SpaceGroups3D);
    assert_eq!(app.atlas.space3d.selected_defect, LineDefect3d::PureDisclination90_123);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, -2);

    // Explicit user choice of None / 0 in Defects mode survives round-tripping through Catalogue2D / Lattices.
    app.atlas.space3d.selected_defect = LineDefect3d::None;
    app.atlas.hyperbolic.wedge_delta = 0;
    app.select_defects_view(DefectsView::Catalogue2D);
    app.select_defects_view(DefectsView::AtlasDefects);
    assert_eq!(app.atlas.space3d.selected_defect, LineDefect3d::None);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, 0);
}

#[test]
fn analysis_supports_pristine_plane_surface_and_combined_targets() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let ctx = egui::Context::default();
    app.analysis.view_mode = DefectViewMode::Split2D3D;
    app.switch_mode(AppMode::Analysis);

    // Pristine 2D Lattice target
    app.analysis.select_target(AnalysisTarget::PlaneLattice);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(!app.analysis.plane_crystal.pos.is_empty());
    let item = app.gallery.active_item().clone();
    app.analysis.submit_active_source(&item, &app.surfaces, &app.defects, &app.combine, None);
    let res = app.analysis.spectrum.wait_idle();
    assert!(res.is_some());
    assert_eq!(app.analysis.spectrum.color_overlay, SpectrumColorOverlay::LdosMultiplet);
    app.analysis.sync_canvas(&app.surfaces, &app.defects, &app.combine, &mut app.canvas2d);
    assert!(app.canvas2d.gauge_overlay().and_then(|o| o.legend.as_ref()).is_some());

    // Surface target (honeycomb dodecahedron sphere = 20 sites) + stale invalidation on lattice switch
    let hc = app.backend.find_named("honeycomb").unwrap().clone();
    app.select_lattice_and_sync(hc.clone());
    app.surfaces.target = crate::backend::SurfaceTargetUi::Sphere;
    app.analysis.select_target(AnalysisTarget::Surface);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    app.analysis.submit_active_source(&hc, &app.surfaces, &app.defects, &app.combine, None);
    assert!(app.analysis.spectrum.wait_idle().is_some());
    let sq = app.backend.find_named("square").unwrap().clone();
    app.select_lattice_and_sync(sq.clone());
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(app.analysis.spectrum.result.is_none(), "changing lattice invalidates stale Surface spectrum");

    // CombinedDefects target + stale invalidation when a pin moves with identical site/edge counts
    app.select_defects_view(DefectsView::CombineAndFuse);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    while app.combine.is_building() {
        std::thread::sleep(Duration::from_millis(10));
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    }
    app.switch_mode(AppMode::Analysis);
    app.analysis.select_target(AnalysisTarget::CombinedDefects);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    let cur = app.gallery.active_item().clone();
    app.analysis.submit_active_source(&cur, &app.surfaces, &app.defects, &app.combine, None);
    assert!(app.analysis.spectrum.wait_idle().is_some());
    assert!(!app.combine.last_build.crystal.pos.is_empty());
    app.combine.last_build.crystal.pos[0][0] += 0.125;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(app.analysis.spectrum.result.is_none(), "moving sites invalidates stale CombinedDefects spectrum");
}
