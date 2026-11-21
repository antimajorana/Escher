//! Headless egui_kittest checks (1280 x 800) of the 3D camera input: pan drags, orbit vs Shift, drags that start on
//! the gizmo / zoom HUD / preset row, gizmo and preset-row clicks, and the overlay layout.

use egui::{Event, Modifiers, PointerButton};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::gizmo::{axis_ends, preset_area_3d};
use super::gizmo_tests::project;
use super::*;
use crate::app::{NavTab, PapertillingApp};
use crate::mesh3d::Vertex3d;
use crate::theme::{StudioTheme, typography};

const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// The app's bundled fonts and Paper theme, so button widths match the real app.
fn app_style(ctx: &egui::Context) {
    typography::register_fonts(ctx);
    StudioTheme::Paper.apply(ctx);
}

fn harness(view: View3dState, size: egui::Vec2) -> Harness<'static, View3dState> {
    let mut h = Harness::builder()
        .with_size(size)
        .with_step_dt(1.0 / 60.0) // real frame time, so two clicks 2 frames apart are a double-click
        .with_accessibility_check(false)
        .with_context_setup(app_style)
        .build_ui_state(|ui, v: &mut View3dState| v.show(ui, &crate::theme::ThemeTokens::paper()), View3dState::default());
    *h.state_mut() = view;
    h.run();
    h
}

fn viewport(h: &Harness<'static, View3dState>) -> egui::Rect {
    h.state().last_layout.expect("drawn").0
}

/// Press `button` at `from`, move to `to` in 20 frames and release; `shift` is held throughout.
fn drag(h: &mut Harness<'static, View3dState>, button: PointerButton, from: egui::Pos2, to: egui::Pos2, shift: bool) {
    let mods = if shift { Modifiers::SHIFT } else { Modifiers::NONE };
    h.event(Event::ModifiersChanged(mods));
    h.event(Event::PointerMoved(from));
    h.event(Event::PointerButton { pos: from, button, pressed: true, modifiers: mods });
    for step in 1..=20 {
        h.event(Event::PointerMoved(from + (to - from) * (step as f32 / 20.0)));
    }
    h.event(Event::PointerButton { pos: to, button, pressed: false, modifiers: mods });
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.step();
}

/// Primary drag from `from` to `to` in 20 frames with Shift held for the first 10 moves only (`shift_first`) or
/// pressed after them; the button is released after the modifier change.
fn drag_toggling_shift(h: &mut Harness<'static, View3dState>, from: egui::Pos2, to: egui::Pos2, shift_first: bool) {
    let (first, second) = if shift_first { (Modifiers::SHIFT, Modifiers::NONE) } else { (Modifiers::NONE, Modifiers::SHIFT) };
    h.event(Event::ModifiersChanged(first));
    h.event(Event::PointerMoved(from));
    h.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: first });
    for step in 1..=20 {
        if step == 11 {
            h.event(Event::ModifiersChanged(second));
        }
        h.event(Event::PointerMoved(from + (to - from) * (step as f32 / 20.0)));
    }
    h.event(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: second });
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.step();
}

#[test]
fn kittest_pan_or_orbit_is_chosen_at_drag_start_and_kept_when_shift_changes() {
    let size = egui::vec2(1280.0, 800.0);
    // Shift+left pan with Shift released before the button: still a pan to the end, never an orbit.
    let mut h = harness(View3dState::default(), size);
    let rect = viewport(&h);
    let from = project(h.state(), rect, [0.0; 3]);
    let to = from + egui::vec2(80.0, -60.0);
    drag_toggling_shift(&mut h, from, to, true);
    assert_eq!((h.state().yaw, h.state().pitch), (0.55, 0.35), "a Shift pan must not orbit");
    let after = project(h.state(), rect, [0.0; 3]);
    assert!(after.distance(to) < 1.0, "the pan continued after Shift was released: {after:?} vs {to:?}");
    // Plain orbit with Shift pressed halfway: still an orbit, 0.01 rad per px after egui's drag threshold.
    let mut h = harness(View3dState::default(), size);
    let c = rect.center();
    drag_toggling_shift(&mut h, c, c + egui::vec2(100.0, 0.0), false);
    let v = h.state();
    assert_eq!((v.camera_offset, v.pitch), (egui::Vec2::ZERO, 0.35), "Shift pressed mid-orbit must not pan");
    assert!(v.yaw - 0.55 > 0.9 && v.yaw - 0.55 <= 1.0 + 1e-4, "yaw {}", v.yaw);
}

#[test]
fn kittest_empty_viewport_next_to_the_preset_buttons_orbits_and_double_click_resets() {
    let size = egui::vec2(1280.0, 800.0);
    let h0 = harness(View3dState::default(), size);
    let (gizmo, row, _) = h0.state().last_overlays.expect("drawn");
    let p = row.expect("preset row").right_center() + egui::vec2(8.0, 0.0);
    assert!(!gizmo.expect("gizmo").contains(p));
    let mut h = harness(View3dState::default(), size);
    drag(&mut h, PointerButton::Primary, p, p + egui::vec2(0.0, 100.0), false);
    assert!(h.state().pitch - 0.35 > 0.9, "a left drag beside the buttons orbits: pitch {}", h.state().pitch);
    let mut h = harness(View3dState { distance: 7.0, yaw: 1.0, ..Default::default() }, size);
    h.event(Event::PointerMoved(p));
    for pressed in [true, false, true, false] {
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    }
    h.step();
    assert_eq!((h.state().yaw, h.state().distance), (0.55, DIST_DEFAULT), "double-click beside the buttons resets");
}

fn click_at(h: &mut Harness<'static, View3dState>, pos: egui::Pos2) {
    h.event(Event::PointerMoved(pos));
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    }
    h.step();
    h.run();
}

fn camera(v: &View3dState) -> (f32, f32, f32, egui::Vec2) {
    (v.yaw, v.pitch, v.distance, v.camera_offset)
}

#[test]
fn kittest_pan_drags_keep_the_orbit_centre_under_the_cursor_at_two_distances() {
    let t0 = std::time::Instant::now();
    let size = egui::vec2(1280.0, 800.0);
    let cases = [(PointerButton::Secondary, false), (PointerButton::Middle, false), (PointerButton::Primary, true)];
    for (button, shift) in cases {
        for distance in [DIST_DEFAULT, 9.0] {
            let mut h = harness(View3dState { distance, ..Default::default() }, size);
            let rect = viewport(&h);
            let from = project(h.state(), rect, [0.0; 3]);
            let to = from + egui::vec2(80.0, -60.0); // a 100 px drag
            drag(&mut h, button, from, to, shift);
            let v = h.state();
            let after = project(v, rect, [0.0; 3]);
            assert!(after.distance(to) < 1.0, "{button:?} shift={shift} d={distance}: {after:?} vs {to:?}");
            assert_eq!((v.yaw, v.pitch, v.distance), (0.55, 0.35, distance), "{button:?} must not orbit");
        }
    }
    println!("# time kittest_pan_drags {:?}", t0.elapsed());
}

#[test]
fn kittest_left_drag_orbits_but_shift_left_drag_only_pans() {
    let size = egui::vec2(1280.0, 800.0);
    let mut h = harness(View3dState::default(), size);
    let c = viewport(&h).center();
    drag(&mut h, PointerButton::Primary, c, c + egui::vec2(100.0, 0.0), false);
    assert!(h.state().yaw > 0.55 + 0.5, "left drag orbits: yaw {}", h.state().yaw);
    assert_eq!(h.state().camera_offset, egui::Vec2::ZERO);

    let mut h = harness(View3dState::default(), size);
    drag(&mut h, PointerButton::Primary, c, c + egui::vec2(100.0, 0.0), true);
    assert_eq!((h.state().yaw, h.state().pitch), (0.55, 0.35));
    assert!(h.state().camera_offset.x > 0.0);
}

#[test]
fn kittest_drags_starting_on_the_gizmo_hud_or_preset_row_neither_orbit_nor_pan() {
    let size = egui::vec2(1280.0, 800.0);
    let h0 = harness(View3dState::default(), size);
    let rect = viewport(&h0);
    let (gizmo, row, _) = h0.state().last_overlays.expect("drawn");
    let hud = h0.state().last_layout.expect("drawn").1.expect("zoom HUD");
    let starts = [
        gizmo.expect("gizmo").left_top() + egui::vec2(4.0, 4.0), // gizmo background, away from the axis ends
        hud.center() - egui::vec2(30.0, 0.0),                     // the zoom percentage label
        row.expect("preset row").right_center() + egui::vec2(-2.0, 0.0),
    ];
    let buttons = [(PointerButton::Primary, false), (PointerButton::Primary, true), (PointerButton::Secondary, false), (PointerButton::Middle, false)];
    for from in starts {
        for (button, shift) in buttons {
            let mut h = harness(View3dState::default(), size);
            let before = camera(h.state());
            drag(&mut h, button, from, rect.center(), shift);
            assert_eq!(camera(h.state()), before, "drag {button:?} shift={shift} from {from:?}");
        }
    }
}

#[test]
fn kittest_clicking_the_gizmo_x_end_applies_side_and_preset_buttons_apply_their_presets() {
    let size = egui::vec2(1280.0, 800.0);
    let mut h = harness(View3dState { camera_offset: egui::vec2(0.4, 0.1), ..Default::default() }, size);
    let gizmo = h.state().last_overlays.expect("drawn").0.expect("gizmo");
    let x_end = axis_ends(&h.state().rotation(), gizmo.center())[0].pos;
    click_at(&mut h, x_end);
    let v = h.state();
    assert_eq!((v.yaw, v.pitch, v.z_up), (-HALF_PI, 0.0, true), "Side (YZ)");
    assert_eq!(v.camera_offset, egui::Vec2::ZERO);

    // (button, yaw, pitch, z_up) written from the brief's layouts (derivation in CameraPreset::orientation).
    let cases = [("Top (XY)", 0.0, 0.0, false), ("Front (XZ)", 0.0, 0.0, true), ("Iso", 0.55, 0.35, false), ("Side (YZ)", -HALF_PI, 0.0, true)];
    for (label, yaw, pitch, z_up) in cases {
        h.state_mut().camera_offset = egui::vec2(-0.2, 0.3);
        h.state_mut().distance = 5.0;
        h.get_by_label(label).click();
        h.run();
        let v = h.state();
        assert_eq!((v.yaw, v.pitch, v.z_up), (yaw, pitch, z_up), "{label}");
        assert_eq!((v.camera_offset, v.distance), (egui::Vec2::ZERO, 5.0), "{label} clears pan, keeps zoom");
    }
    h.get_by_label("Reset").click();
    h.run();
    let v = h.state();
    assert_eq!((v.yaw, v.pitch, v.z_up, v.distance), (0.55, 0.35, false, DIST_DEFAULT));
}

const LABELS: [&str; 5] = ["Iso", "Top (XY)", "Front (XZ)", "Side (YZ)", "Reset"];

/// Every preset button is drawn where it was planned (the input blockers), inside the viewport, inside the preset
/// area, clear of `others` and of each other, with the five labels in order.
fn check_preset_buttons(v: &View3dState, others: &[egui::Rect], what: &str) {
    let rect = v.last_layout.expect("drawn").0;
    let (origin, max_w) = preset_area_3d(rect).expect("preset area");
    let labels: Vec<&str> = v.last_preset_buttons.iter().map(|b| b.0).collect();
    assert_eq!(labels, LABELS, "{what}");
    for (i, (label, planned, drawn)) in v.last_preset_buttons.iter().enumerate() {
        assert!(planned.min.distance(drawn.min) < 0.5 && planned.max.distance(drawn.max) < 0.5, "{what} {label}: {planned:?} drawn {drawn:?}");
        assert!(rect.contains_rect(*drawn) && drawn.right() <= origin.x + max_w + 0.5, "{what} {label}: {drawn:?}");
        for other in others {
            assert!(!drawn.intersects(*other), "{what} {label}: {drawn:?} overlaps {other:?}");
        }
        for (_, _, d2) in &v.last_preset_buttons[i + 1..] {
            assert!(!drawn.intersects(*d2), "{what} {label}: {drawn:?} overlaps {d2:?}");
        }
    }
}

#[test]
fn kittest_gizmo_and_preset_buttons_clear_caption_hint_legend_and_hud() {
    let t0 = std::time::Instant::now();
    // (size, preset buttons expected: Some(true/false) or None = either, checked for overlap only).
    let cases = [
        (egui::vec2(1280.0, 800.0), Some(true)),
        (egui::vec2(560.0, 420.0), Some(true)),
        (egui::vec2(300.0, 600.0), Some(true)),
        (egui::vec2(278.5, 700.0), Some(true)),
        (egui::vec2(220.0, 220.0), None),
        (egui::vec2(260.0, 180.0), Some(false)),
    ];
    for (size, want_row) in cases {
        let mut view = View3dState::default();
        let mesh = GpuMeshData { faces: vec![Vertex3d::ZERO; 3], radius: 1.0, ..Default::default() };
        view.set_mesh(mesh, "a long caption that runs across the top of the viewport | V=1 E=2 F=3");
        view.set_legend(Some(ColorLegend3d { title: "Height z".into(), min_val: -0.5, max_val: 0.5, map: Legend3dMap::Height }));
        let h = harness(view, size);
        let (rect, hud, legend) = h.state().last_layout.expect("drawn");
        let (gizmo, row, hint) = h.state().last_overlays.expect("drawn");
        let caption_band = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.right(), rect.top() + 24.0));
        let others: Vec<egui::Rect> = [hud, legend, hint, Some(caption_band)].into_iter().flatten().collect();
        if let Some(g) = gizmo {
            assert!(rect.contains_rect(g), "{size:?}: {g:?} outside {rect:?}");
            for other in &others {
                assert!(!g.intersects(*other), "{size:?}: gizmo {g:?} overlaps {other:?}");
            }
        }
        if row.is_some() {
            let mut with_gizmo = others.clone();
            with_gizmo.extend(gizmo);
            check_preset_buttons(h.state(), &with_gizmo, &format!("{size:?}"));
        }
        if let (Some(hint), Some(hud)) = (hint, hud) {
            assert!(!hint.intersects(hud), "{size:?}: hint {hint:?} overlaps zoom HUD {hud:?}");
        }
        if let Some(want) = want_row {
            assert_eq!(row.is_some(), want, "{size:?}");
        }
        assert_eq!(gizmo.is_some(), rect.width() >= 200.0 && rect.height() >= 200.0, "{size:?}: viewport {rect:?}");
    }
    println!("# time kittest_camera_overlays {:?}", t0.elapsed());
}

#[test]
fn kittest_narrow_pane_wraps_the_full_preset_labels_and_reset() {
    let mut h = harness(View3dState::default(), egui::vec2(300.0, 600.0));
    let rows = h.state().last_preset_buttons.iter().map(|b| b.2.top().round() as i32).collect::<std::collections::BTreeSet<_>>();
    assert!(rows.len() >= 2, "the 200 px wide area wraps the ~296 px row: {rows:?}");
    h.get_by_label("Front (XZ)").click();
    h.run();
    assert_eq!((h.state().yaw, h.state().pitch, h.state().z_up), (0.0, 0.0, true), "Front (XZ)");
    h.get_by_label("Side (YZ)").click();
    h.run();
    assert_eq!((h.state().yaw, h.state().pitch, h.state().z_up), (-HALF_PI, 0.0, true), "Side (YZ)");
    h.state_mut().distance = 7.0;
    h.get_by_label("Reset").click();
    h.run();
    let v = h.state();
    assert_eq!((v.yaw, v.pitch, v.z_up, v.distance), (0.55, 0.35, false, DIST_DEFAULT), "Reset");
}

#[test]
fn kittest_real_app_surfaces_3d_pane_shows_the_gizmo_and_all_preset_buttons_at_1280x800() {
    let t0 = std::time::Instant::now();
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    let mut h = Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .with_context_setup(app_style)
        .build_ui_state(|ui, app: &mut PapertillingApp| { let _ = app.frame_ui(ui); }, app);
    h.run();
    h.state_mut().switch_tab(NavTab::Surfaces);
    for _ in 0..4 {
        h.step();
    }
    let v = &h.state().view3d;
    let (rect, hud, _) = v.last_layout.expect("3D pane drawn on the Surfaces tab");
    let (gizmo, _, hint) = v.last_overlays.expect("3D pane drawn");
    println!("# 3D pane {rect:?} buttons {:?}", v.last_preset_buttons);
    let gizmo = gizmo.expect("gizmo");
    let caption_band = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.right(), rect.top() + 24.0));
    let others: Vec<egui::Rect> = [Some(gizmo), hud, hint, Some(caption_band)].into_iter().flatten().collect();
    check_preset_buttons(v, &others, "real app");
    h.get_by_label("Front (XZ)").click();
    for _ in 0..3 {
        h.step(); // the Surfaces tab keeps repainting (spinner), so `run` would not settle
    }
    assert!(h.state().view3d.z_up, "the Front (XZ) button works in the real pane");
    println!("# time kittest_real_app_3d_pane {:?}", t0.elapsed());
}
