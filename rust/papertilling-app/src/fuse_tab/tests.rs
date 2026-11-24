//! End-to-end UI and equivalence tests for `Defects > Fuse & compare` (`NavTab::Fuse`).

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::{FuseDefectChoice, FusePreset, FuseTabState, compute_slots_for_b, spec_from_choice};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, settled, wait_until};
use crate::app::{AppMode, DefectsView, NavTab, PapertillingApp};
use crate::backend::AppBackend;
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_worker::empty_crystal;
use crate::live_combine::map_ref_to_drawn;
use crate::theme::{StudioTheme, typography};
use papertilling_core::defect::surgery::DefectSpec;

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
    h.get_by_label("Enclosing Burgers / holonomy loop").click();
    h.run_steps(2);
    assert!(!h.state().fuse.show_burgers_loop);
    assert_eq!(h.query_all_by_label("1.732").count(), 2, "both measured Burgers lengths remain visible");
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
        assert!(h.query_by_label("No core match in this scan").is_some());
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

#[test]
fn square_annihilation_preset_builds_vacuum_and_clicking_a_target_moves_b() {
    let mut h = harness();
    {
        let app = h.state_mut();
        let square = app.backend.find_named("square").unwrap().clone();
        app.select_lattice_and_sync(square);
        app.select_defects_view(DefectsView::FuseAndCompare);
    }
    wait_until(&mut h, "initial square dipole", settled);
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        app.fuse.apply_preset(FusePreset::OppositeAnnihilate, &item, true);
        assert_eq!(app.fuse.pos_a, app.fuse.pos_b);
        assert!(app.fuse.last_bundle.is_none(), "a new reaction clears the old verdict");
        assert_eq!(app.fuse.single_canvas.counts(), (0, 0));
    }
    wait_until(&mut h, "square annihilation", settled);
    {
        let app = h.state_mut();
        let fuse = &mut app.fuse;
        let bundle = fuse.last_bundle.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(bundle.report.kind, "vacuum");
        assert_eq!(bundle.delta_v, Some(0));
        assert!(fuse.active_single_candidate().unwrap().is_match);
        let item = app.gallery.active_item().clone();
        fuse.show_burgers_loop = false;
        fuse.sync_canvases(&item, &mut app.canvas2d, false);
        app.canvas2d.zoom = 45.0;
    }
    h.run_steps(2);

    let app = h.state();
    let [x, y] = app.fuse.pos_a;
    let [dx, dy] = app.fuse.info.cell.a1;
    let target = [x + dx, y + dy];
    let [sx, sy] = map_ref_to_drawn(&app.fuse.last_combined.crystal, target);
    let screen = app.canvas2d.world_to_screen_last([sx as f32, sy as f32]).unwrap();
    assert!(app.canvas2d.last_rect.contains(screen));
    h.hover_at(screen);
    h.step();
    h.drag_at(screen);
    h.step();
    h.drop_at(screen);
    wait_until(&mut h, "place B by clicking a target ring", settled);

    let [bx, by] = h.state().fuse.pos_b;
    assert!((bx - target[0]).hypot(by - target[1]) < 1e-6, "B stayed at [{bx}, {by}] instead of {target:?}");
}

#[test]
fn fuse_positions_follow_a_translated_anchor_and_swap_in_place() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut fuse = FuseTabState::new(square, &backend);
    let choice = fuse.choices[fuse.choice_b].clone();
    let initial = compute_slots_for_b(&fuse.info, &choice, fuse.pos_a);
    let shift = [12.0, -9.0];
    fuse.pos_a[0] += shift[0];
    fuse.pos_a[1] += shift[1];
    fuse.pos_b[0] += shift[0];
    fuse.pos_b[1] += shift[1];
    fuse.refresh_slots_b();
    assert!(!initial.is_empty());
    assert_eq!(initial.len(), fuse.slots_b.len());
    for (p, q) in initial.iter().zip(&fuse.slots_b) {
        assert!((q[0] - p[0] - shift[0]).hypot(q[1] - p[1] - shift[1]) < 1e-6);
    }

    let [a, b] = fuse.current_specs();
    fuse.swap_defects();
    assert_eq!(fuse.current_specs(), [b, a]);
}

#[test]
fn fuse_flips_burgers_vectors_across_distinct_p3_classes() {
    let backend = AppBackend::new();
    let triangular = backend.find_named("chiral_triangular").unwrap();
    let fuse = FuseTabState::new(triangular, &backend);
    let index = fuse.choices.iter().position(|choice| matches!(choice, FuseDefectChoice::Dislocation { .. })).unwrap();
    let flipped = fuse.flipped_choice_index(index).expect("catalogued inverse Burgers vector");
    assert_eq!(fuse.flipped_choice_index(flipped), Some(index));
    let get_b = |idx: usize| match spec_from_choice(&fuse.info, &fuse.choices[idx], [0.0, 0.0]) {
        DefectSpec::Dislocation { b, .. } => b,
        _ => panic!("expected a dislocation"),
    };
    let [x, y] = get_b(index);
    assert_eq!(get_b(flipped), [-x, -y]);
}

#[test]
fn camera_sync_keeps_translated_patches_centered_independently() {
    let patch = |point| {
        let mut crystal = empty_crystal();
        crystal.pos.push(point);
        crystal.ref_pos.push(point);
        crystal.basis_idx.push(0);
        crystal.side.push(None);
        crystal.adj.push(Vec::new());
        Canvas2D::new(Scene2D::CrystalPatch { crystal, title: String::new(), note: String::new() })
    };
    let mut left = patch([12.0, -9.0]);
    let mut right = patch([0.0, 0.0]);
    left.pan = egui::vec2(8.0, -10.0);
    left.zoom = 45.0;
    right.copy_pan_and_zoom_from(&left);
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0));
    assert_eq!(left.world_to_screen_at(rect, [12.0, -9.0]), right.world_to_screen_at(rect, [0.0, 0.0]));
    assert_eq!(left.zoom, right.zoom);
}
