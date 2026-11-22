//! End-to-end UI and equivalence tests for `Defects > Fuse & compare` (`NavTab::Fuse`).

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::FusePreset;
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, settled, wait_until};
use crate::app::{AppMode, DefectsView, NavTab, PapertillingApp};
use crate::theme::{StudioTheme, typography};

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1600.0, 1000.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(|ui, app| {
            let _ = app.frame_ui(ui);
        }, app)
}

fn click_top(h: &mut Harness<'static, PapertillingApp>, label: &str) {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label)
        .find(|n| top.contains_rect(n.rect()))
        .unwrap_or_else(|| panic!("missing '{label}' in top bar"))
        .click();
    h.run_steps(2);
}

#[test]
fn fuse_and_compare_default_hud_honeycomb_5_7_completion_equivalence() {
    let mut h = harness();
    h.run_steps(3);
    assert!(!h.state().advanced);
    assert_eq!(h.state().gallery.active_item().named_key.as_deref(), Some("honeycomb"));

    click_top(&mut h, "Defects");
    wait_until(&mut h, "Defects", settled);
    click_top(&mut h, "Fuse & compare");
    wait_until(&mut h, "Fuse & compare (honeycomb)", settled);

    {
        let app = h.state();
        assert_eq!(
            (app.active_mode, app.defects_view, app.active_tab),
            (AppMode::Defects, DefectsView::FuseAndCompare, NavTab::Fuse)
        );
        assert!(app.fuse.last_combined.refused.is_none(), "refused: {:?}", app.fuse.last_combined.refused);
        let bundle = app.fuse.last_bundle.as_ref().expect("bundle built").as_ref().expect("bundle ok");
        assert!(bundle.report.equivalent, "honeycomb +60/-60 dipole must match single dislocation: {:?}", bundle.report.verdict);
        let cand = app.fuse.active_single_candidate().expect("active candidate");
        assert!(
            cand.is_match && cand.kind == "completion",
            "expected healed 5-7 completion core match, got {} ({}, kind={})",
            cand.name,
            cand.description,
            cand.kind
        );
        assert!(!cand.crystal.pos.is_empty());

        let c_res = &app.fuse.combined_inspectors.circuit.overlay.as_ref().expect("combined Burgers loop").result;
        let s_res = &app.fuse.single_inspectors.circuit.overlay.as_ref().expect("single Burgers loop").result;
        assert!(c_res.burgers_norm > 0.5, "combined |b| = {}", c_res.burgers_norm);
        assert!((c_res.burgers_norm - s_res.burgers_norm).abs() < 1e-6, "{} vs {}", c_res.burgers_norm, s_res.burgers_norm);
        assert!(c_res.frank_angle_deg.unwrap_or(0.0).abs() < 1e-6 && s_res.frank_angle_deg.unwrap_or(0.0).abs() < 1e-6);
    }

    assert!(h.query_by_label("Rename").is_some(), "Fuse & compare is lattice-driven and keeps Rename");
    assert_no_forbidden_tokens(&h, "Fuse & compare honeycomb");
}

#[test]
fn square_p4m_fuse_stepper_wyckoff_shift_camera_sync_and_handoff() {
    let mut h = harness();
    h.run_steps(2);
    {
        let app = h.state_mut();
        let sq = app.backend.find_named("square").expect("square lattice").clone();
        app.select_lattice_and_sync(sq);
        app.select_defects_view(DefectsView::FuseAndCompare);
    }
    wait_until(&mut h, "Fuse & compare (square)", settled);

    {
        let app = h.state();
        assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("square"));
        if let Some(ref reason) = app.fuse.last_combined.refused {
            panic!("square default Fuse & compare preset was refused: {reason}");
        }
        let bundle = app.fuse.last_bundle.as_ref().expect("bundle").as_ref().expect("bundle ok");
        assert!(bundle.report.equivalent, "square default dipole should match Volterra single core: {:?}", bundle.report.verdict);
        let cand = app.fuse.active_single_candidate().expect("candidate");
        assert!(cand.is_match && cand.kind == "Volterra", "square edge dislocation matches Volterra core (got kind={})", cand.kind);
    }

    // Step B two radial shells farther apart: cores separate and no longer match the single elementary core,
    // then step 2 shells back closer to restore Volterra single-defect equivalence!
    for _ in 0..2 {
        h.get_by_label("Farther ▶").click();
        wait_until(&mut h, "Farther step", settled);
    }
    {
        let bundle = h.state().fuse.last_bundle.as_ref().unwrap().as_ref().unwrap();
        assert!(!bundle.report.equivalent, "well-separated dipole has distinct split core vs single defect");
    }
    for _ in 0..2 {
        h.get_by_label("◀ Closer").click();
        wait_until(&mut h, "Closer step", settled);
    }
    {
        let bundle = h.state().fuse.last_bundle.as_ref().unwrap().as_ref().unwrap();
        assert!(bundle.report.equivalent, "stepping back closer restores Volterra single-defect equivalence");
    }

    // Apply Disclination + Dislocation preset: D(1a, +90°) + b=[1,0] shifts Wyckoff class!
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        app.fuse.apply_preset(FusePreset::DisclinationPlusDislocation, &item, true);
    }
    wait_until(&mut h, "Disclination + Dislocation on square", settled);
    {
        let app = h.state();
        assert!(app.fuse.last_combined.refused.is_none(), "refused: {:?}", app.fuse.last_combined.refused);
        let bundle = app.fuse.last_bundle.as_ref().unwrap().as_ref().unwrap();
        assert!((bundle.report.charge.angle - 90.0).abs() < 1e-6);
        assert!(bundle.report.charge.label.contains("+90"));
    }

    // Camera sync check: panning left canvas propagates to right single_canvas on next frame.
    {
        let app = h.state_mut();
        app.canvas2d.pan = egui::vec2(1.25, -0.75);
        app.canvas2d.zoom = 42.0;
    }
    h.run_steps(2);
    assert_eq!(h.state().fuse.single_canvas.camera_tuple(), h.state().canvas2d.camera_tuple());

    // Handoff to Place & fuse transfers the exact 2 DefectSpecs.
    let expected_specs = h.state().fuse.current_specs().to_vec();
    h.get_by_label("Open pair in Place & fuse").click();
    wait_until(&mut h, "Open pair in Place & fuse", settled);
    assert_eq!(
        (h.state().active_mode, h.state().defects_view, h.state().active_tab),
        (AppMode::Defects, DefectsView::CombineAndFuse, NavTab::Combine)
    );
    assert_eq!(h.state().combine.specs, expected_specs);
}
