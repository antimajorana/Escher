//! Stage 6 unit and headless UI tests for pin selection, cut-ray handles, popover controls, and 32-deep redo stack.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::*;
use crate::canvas2d::cut_ray_handle_screen_pos;
use crate::combine::history::CutDragResponse;
use crate::live_combine::cuts::{merge_coincident_pins, pin_explicit_cut_vec, set_pin_cut_angle};
use crate::mesh3d::DefectViewMode;
use crate::theme::{StudioTheme, ThemeTokens, typography};
use papertilling_core::types::Cell2D;

fn make_canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 2,
        show_unit_cell: false,
    })
}

#[test]
fn selection_persists_across_sync_and_exposes_selected_pin_cut_ray() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    assert_eq!(st.selected_pin, None);
    assert_eq!(canvas.selected_marker(), None);
    assert!(canvas.cut_rays().is_empty(), "branch cuts off and no pin selected -> empty");

    st.selected_pin = Some(1);
    st.popover_open = true;
    st.sync_selection_to_canvas(&mut canvas);
    assert_eq!(canvas.selected_marker(), Some(1));
    assert_eq!(canvas.cut_rays().len(), 1);
    assert_eq!(canvas.cut_rays()[0].spec_index, Some(1));

    // Rebuilding + full sync_to_canvas must preserve both selection halo and selected pin's ray.
    st.request_build(square, false, false, false);
    st.wait_idle(square, Some(&mut canvas));
    assert_eq!(canvas.selected_marker(), Some(1));
    assert_eq!(canvas.cut_rays().len(), 1);
    assert_eq!(canvas.cut_rays()[0].spec_index, Some(1));
}

#[test]
fn dislocation_cut_anchor_matches_at_plus_half_burgers_and_coincident_pins_clear_owner() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.apply_preset_dislocation(square, &backend, &mut canvas);
    st.wait_idle(square, Some(&mut canvas));
    assert_eq!(st.specs.len(), 1);
    assert!(matches!(st.specs[0], DefectSpec::Dislocation { .. }));

    st.show_branch_cuts = true;
    let rays = st.current_cut_rays();
    assert_eq!(rays.len(), 1);
    assert_eq!(rays[0].spec_index, Some(0), "dislocation cut anchored at at + b/2 must match spec 0");

    // Superposed coincident dislocations share origin -> owner must become None to prevent ambiguous handle drag.
    let dup = st.specs[0].clone();
    st.specs.push(dup);
    st.request_build(square, false, false, false);
    st.wait_idle(square, Some(&mut canvas));
    let rays_dup = st.current_cut_rays();
    assert!(rays_dup.iter().all(|r| r.spec_index.is_none()), "coincident pair must not bind single spec_index");
}

#[test]
fn redo_stack_caps_at_32_round_trips_and_clears_on_new_edit_or_changed_selection() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));

    for k in 1..=35 {
        st.push_undo();
        st.specs = vec![DefectSpec::dislocation([1.0, 0.0], Some([k as f64 * 0.25, 0.0]))];
    }
    assert_eq!(st.undo_stack.len(), 32, "undo history capped at 32");
    assert!(st.redo_stack.is_empty());

    let latest = st.specs.clone();
    st.selected_pin = Some(0);
    st.popover_open = true;

    st.undo_last(square);
    assert_eq!(st.redo_stack.len(), 1);
    assert_eq!(st.selected_pin, None, "undo changed specs[0], so stale selection was cleared");
    assert!(!st.popover_open);

    st.redo_last(square);
    assert_eq!(st.specs, latest);
    assert!(st.redo_stack.is_empty());

    // Undo twice, then make a fresh edit -> redo_stack clears.
    st.undo_last(square);
    st.undo_last(square);
    assert_eq!(st.redo_stack.len(), 2);
    st.push_undo();
    st.specs.push(DefectSpec::dislocation([0.0, 1.0], Some([2.0, 2.0])));
    assert!(st.redo_stack.is_empty(), "fresh edit must clear redo_stack");

    // Deleting pin 0 shifts selected_pin = Some(1) down to Some(0).
    st.selected_pin = Some(1);
    let kept = st.specs[1].clone();
    assert!(st.delete_pin(0, square, Some(&mut canvas)));
    assert_eq!(st.selected_pin, Some(0));
    assert_eq!(st.specs[0], kept);

    // Merging coincident pins clears selection safely.
    st.specs = vec![
        DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0])),
        DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0])),
    ];
    st.selected_pin = Some(1);
    st.push_undo();
    st.clear_selection();
    merge_coincident_pins(&st.info, &mut st.specs);
    assert_eq!(st.specs.len(), 1);
    assert_eq!(st.selected_pin, None);
}

#[test]
fn canvas_click_select_popover_actions_and_shift_cut_handle_drag_single_undo() {
    struct Env {
        backend: AppBackend,
        square: crate::backend::CatalogueItemUi,
        canvas: Canvas2D,
        st: CombineTabState,
    }
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap().clone();
    let mut canvas = make_canvas();
    let mut st = CombineTabState::new(&square, &backend);
    st.view_mode = DefectViewMode::Planar2D;
    st.wait_idle(&square, Some(&mut canvas));
    // Keep a single disclination at pin 0 so +30° (upper-right quadrant) builds without hitting a second pin's collar.
    st.specs.truncate(1);
    st.request_build(&square, false, false, true);
    st.wait_idle(&square, Some(&mut canvas));

    let mut h = Harness::builder()
        .with_size(egui::vec2(900.0, 640.0))
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, env: &mut Env| {
                let tokens = ThemeTokens::paper();
                env.st.show_2d_pin_editor(ui, &env.square, &env.backend, &mut env.canvas, &tokens);
            },
            Env { backend, square, canvas, st },
        );
    h.run_steps(3);

    // 1. Click pin 0 on the 2D canvas -> selects pin 0, opens floating popover inside canvas rect.
    let rect = h.state().canvas.last_rect;
    let p0_w = spec_world_pos(&h.state().st.info, 0, &h.state().st.specs[0]);
    let p0_s = h.state().canvas.world_to_screen_at(rect, [p0_w[0] as f32, p0_w[1] as f32]);
    let click_pos = |h: &mut Harness<'_, Env>, pos: egui::Pos2| {
        h.hover_at(pos);
        h.step();
        h.drag_at(pos);
        h.step();
        h.drop_at(pos);
        h.run_steps(2);
    };
    click_pos(&mut h, p0_s);

    assert_eq!(h.state().st.selected_pin, Some(0));
    assert!(h.state().st.popover_open);
    let pop_rect = h.state().st.last_popover_rect.expect("popover rendered");
    assert!(rect.contains_rect(pop_rect.shrink(1.0)), "popover {pop_rect:?} must stay inside {rect:?}");
    assert!(!pop_rect.contains(p0_s), "popover must not occlude selected pin at {p0_s:?}");

    // 2. Drag the selected pin's cut-ray handle with Shift held across multiple frames -> snaps to 15 deg & pushes 1 undo.
    let ray = h.state().canvas.cut_rays()[0];
    let hp = cut_ray_handle_screen_pos(&h.state().canvas, rect, &ray).expect("cut handle visible");
    let origin_w = ray.origin;
    let target_deg_raw = 37.0_f64; // nearest 15 deg multiple is 30.0 deg
    let target_w = [
        origin_w[0] + 2.2 * target_deg_raw.to_radians().cos(),
        origin_w[1] + 2.2 * target_deg_raw.to_radians().sin(),
    ];
    let target_s = h.state().canvas.world_to_screen_at(rect, [target_w[0] as f32, target_w[1] as f32]);
    let mid_s = egui::pos2(0.5 * (hp.x + target_s.x), 0.5 * (hp.y + target_s.y));

    let undo_before = h.state().st.undo_stack.len();
    h.hover_at(hp);
    h.step();
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
    h.event(egui::Event::PointerButton {
        pos: hp,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::SHIFT,
    });
    h.step();
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
    h.hover_at(mid_s);
    h.step();
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
    h.hover_at(target_s);
    h.step();
    h.event(egui::Event::PointerButton {
        pos: target_s,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::SHIFT,
    });
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(2);
    {
        let env = h.state_mut();
        env.st.wait_idle(&env.square, Some(&mut env.canvas));
    }
    assert_eq!(h.state().st.undo_stack.len(), undo_before + 1, "multi-frame cut drag pushes exactly 1 undo");
    let cvec = pin_explicit_cut_vec(&h.state().st.specs[0]).expect("explicit cut set");
    let snapped_deg = cvec[1].atan2(cvec[0]).to_degrees();
    assert!((snapped_deg - 30.0).abs() < 1e-6, "Shift snap expected 30°, got {snapped_deg}");

    // 3. Popover buttons: verify popover moved away from the 30° cut handle, then Flip 180°, Auto-Align, and click empty canvas.
    h.run_steps(2);
    let ray_30 = h.state().canvas.cut_rays()[0];
    let hp_30 = cut_ray_handle_screen_pos(&h.state().canvas, rect, &ray_30).expect("30° cut handle visible");
    let pop_rect_30 = h.state().st.last_popover_rect.expect("popover rendered after 30° snap");
    assert!(
        !pop_rect_30.expand(6.0).contains(hp_30),
        "popover {pop_rect_30:?} must not cover 30° cut-ray handle at {hp_30:?}"
    );
    assert!(!pop_rect_30.contains(p0_s), "popover {pop_rect_30:?} must not cover selected pin at {p0_s:?}");
    let ang_before = match h.state().st.specs[0] {
        DefectSpec::Disclination { angle_deg, .. } => angle_deg,
        ref other => panic!("expected disclination, got {other:?}"),
    };
    h.get_by_label("Flip 180°").click();
    h.run_steps(2);
    let ang_after = match h.state().st.specs[0] {
        DefectSpec::Disclination { angle_deg, .. } => angle_deg,
        ref other => panic!("expected disclination, got {other:?}"),
    };
    assert_eq!(ang_after, ang_before, "Flip 180° must preserve defect topological charge");
    let flipped_vec = pin_explicit_cut_vec(&h.state().st.specs[0]).expect("explicit cut kept after Flip 180°");
    let flipped_deg = flipped_vec[1].atan2(flipped_vec[0]).to_degrees();
    assert!(
        (flipped_deg - (-150.0)).abs() < 1e-6,
        "Flip 180° from +30° expected -150°, got {flipped_deg}"
    );

    h.get_by_label("Auto-Align").click();
    h.run_steps(2);
    assert!(pin_explicit_cut_vec(&h.state().st.specs[0]).is_none(), "Auto-Align resets explicit cut to None");

    let empty_pt = egui::pos2(rect.left() + 24.0, rect.top() + 56.0);
    click_pos(&mut h, empty_pt);
    assert_eq!(h.state().st.selected_pin, None);
    assert!(!h.state().st.popover_open);
}

#[test]
fn refused_cut_angle_retains_selected_handle_for_recovery_and_uncommitted_cut_undo_roundtrips() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    assert_eq!(st.specs.len(), 2);

    // Point pin 0's branch-cut ray straight across pin 1's core so surgery refuses (`applied_cuts` empty).
    let p0 = spec_world_pos(&st.info, 0, &st.specs[0]);
    let p1 = spec_world_pos(&st.info, 1, &st.specs[1]);
    let blocked_deg = (p1[1] - p0[1]).atan2(p1[0] - p0[0]).to_degrees();
    st.selected_pin = Some(0);
    st.popover_open = true;
    st.push_undo();
    set_pin_cut_angle(&mut st.specs[0], Some(blocked_deg));
    st.request_build(square, false, false, false);
    st.wait_idle(square, Some(&mut canvas));

    assert!(st.last_build.refused.is_some(), "expected surgical refusal when cut crosses partner collar");
    assert!(st.last_build.applied_cuts.is_empty());
    let rays = st.current_cut_rays();
    assert_eq!(rays.len(), 1, "selected pin must still synthesize its recovery cut ray on refusal");
    assert_eq!(rays[0].spec_index, Some(0));
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    assert!(
        cut_ray_handle_screen_pos(&canvas, rect, &rays[0]).is_some(),
        "selected pin handle must remain draggable to recover from surgical refusal"
    );

    // Flipping 180° away from partner recovers a valid build!
    assert!(st.flip_pin_cut_180(0, square, Some(&mut canvas)));
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_none(), "Flip 180° away from partner must recover valid crystal");

    // Uncommitted `cut_edit_anchor` (e.g. DragValue focused/active when Ctrl+Z fires) must undo back to pre-edit state.
    let pre_edit = st.specs.clone();
    let prev = st.apply_cut_drag_response(
        0,
        CutDragResponse { deg: 75.0, changed: true, started: true, finished: false, active: true },
    );
    assert_eq!(prev, Some(true));
    assert!(st.cut_edit_anchor.is_some());
    assert_ne!(st.specs, pre_edit);

    st.undo_last(square);
    assert_eq!(st.specs, pre_edit, "undo_last must commit pending cut_edit_anchor and restore pre-edit specs");
    assert_eq!(st.redo_stack.len(), 1);
    st.redo_last(square);
    let redone = pin_explicit_cut_vec(&st.specs[0]).unwrap();
    assert!((redone[1].atan2(redone[0]).to_degrees() - 75.0).abs() < 1e-6);
}
