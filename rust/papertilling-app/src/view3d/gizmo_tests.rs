//! Known-answer tests of the 3D camera presets, pan, pole-safe orbit and the axis gizmo geometry. The expected
//! layouts come from the brief (Top: x right, y up; Front: x right, z up; Side: y right, z up) and the expected pan
//! scale from the 45 deg vertical field of view: 1 / tan(22.5 deg) = 1 + sqrt(2).

use super::camera::{CameraPreset, OFFSET_MAX};
use super::gizmo::{AXIS_LEN_PX, PRESET_BTN_H, PRESET_GAP, axis_ends, gizmo_hit, preset_for_axis, wrap_buttons};
use super::*;

const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// Screen pixel position (y down) of world `point` in viewport `rect`, through the renderer's MVP.
pub(super) fn project(view: &View3dState, rect: egui::Rect, point: [f32; 3]) -> egui::Pos2 {
    let mvp = view.view_mvp(rect.width() / rect.height(), &view.rotation());
    let p = [point[0], point[1], point[2], 1.0];
    let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|col| mvp[col][row] * p[col]).sum());
    egui::pos2(
        rect.center().x + rect.width() * 0.5 * clip[0] / clip[3],
        rect.center().y - rect.height() * 0.5 * clip[1] / clip[3],
    )
}

/// Screen direction (x right, y UP) of the projected world axis `axis` from the origin, unnormalised (pixels).
fn axis_on_screen(view: &View3dState, rect: egui::Rect, axis: usize) -> egui::Vec2 {
    let mut tip = [0.0; 3];
    tip[axis] = 0.1;
    let d = project(view, rect, tip) - project(view, rect, [0.0; 3]);
    egui::vec2(d.x, -d.y)
}

fn close(a: egui::Vec2, b: egui::Vec2, tol: f32) -> bool {
    (a - b).length() <= tol
}

#[test]
fn yaw_and_pitch_alone_never_put_world_y_on_the_screen_x_axis() {
    // Column 1 of Rx(pitch) Ry(yaw) is (0, cos p, sin p): the reason Front and Side use the z-up turntable frame.
    for yaw in [-3.0_f32, -1.2, 0.0, 0.55, 2.0] {
        for pitch in [-HALF_PI, -0.7, 0.0, 0.35, 1.5] {
            assert_eq!(orbit_rotation(yaw, pitch)[1][0], 0.0, "yaw {yaw} pitch {pitch}");
        }
    }
}

#[test]
fn presets_project_the_world_axes_to_the_stated_screen_layout() {
    let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(900.0, 600.0));
    let mut view = View3dState { camera_offset: egui::vec2(0.3, -0.2), ..Default::default() };
    // (preset, [expected unit screen direction of x, y, z] (y up), index of the axis seen end-on, its sign towards
    // the camera: +1 = the +axis points at the camera).
    let (r, u, zero) = (egui::vec2(1.0, 0.0), egui::vec2(0.0, 1.0), egui::Vec2::ZERO);
    let cases = [
        (CameraPreset::Top, [r, u, zero], 2, 1.0),   // looks down -z
        (CameraPreset::Front, [r, zero, u], 1, -1.0), // looks along +y
        (CameraPreset::Side, [zero, r, u], 0, 1.0),  // looks along -x
    ];
    for (preset, dirs, end_on, towards) in cases {
        view.apply_preset(preset);
        assert_eq!(view.camera_offset, egui::Vec2::ZERO, "{preset:?} clears the pan");
        let scale = axis_on_screen(&view, rect, (end_on + 1) % 3).length();
        for (axis, want) in dirs.iter().enumerate() {
            let got = axis_on_screen(&view, rect, axis) / scale;
            assert!(close(got, *want, 1e-4), "{preset:?} axis {axis}: {got:?} != {want:?}");
        }
        let rot = view.rotation();
        assert!((rot[end_on][2] - towards).abs() < 1e-6, "{preset:?}: depth of axis {end_on} = {}", rot[end_on][2]);
    }
    // Iso is the reset orientation; hand formula R e_x = (cos y, sin p sin y, -cos p sin y) for Rx(p) Ry(y).
    view.apply_preset(CameraPreset::Iso);
    assert_eq!((view.yaw, view.pitch, view.z_up), (0.55, 0.35, false));
    let rot = view.rotation();
    let want = [0.55_f32.cos(), 0.35_f32.sin() * 0.55_f32.sin(), -0.35_f32.cos() * 0.55_f32.sin()];
    for k in 0..3 {
        assert!((rot[0][k] - want[k]).abs() < 1e-6);
    }
}

#[test]
fn pan_scale_is_two_distance_over_viewport_height_times_the_focal_factor() {
    let f = 1.0 + 2.0_f32.sqrt(); // 1 / tan(22.5 deg)
    for (size, d) in [(egui::vec2(1200.0, 800.0), DIST_DEFAULT), (egui::vec2(500.0, 900.0), 12.0)] {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut view = View3dState { distance: d, ..Default::default() };
        view.pan_by_pixels(rect, egui::vec2(100.0, 40.0));
        let per_px = 2.0 * d / (size.y * f);
        assert!(close(view.camera_offset, egui::vec2(100.0 * per_px, -40.0 * per_px), 1e-5), "{:?}", view.camera_offset);
        // The orbit centre (origin) followed the drag exactly.
        let moved = project(&view, rect, [0.0; 3]) - rect.center();
        assert!(close(moved, egui::vec2(100.0, 40.0), 1e-2), "{moved:?}");
    }
    // A very narrow viewport triggers the renderer's aspect clamp: x uses P00, not P11, and still tracks the cursor.
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(64.0, 1000.0));
    let mut view = View3dState::default();
    view.pan_by_pixels(rect, egui::vec2(10.0, -30.0));
    assert!(close(project(&view, rect, [0.0; 3]) - rect.center(), egui::vec2(10.0, -30.0), 1e-2));
}

#[test]
fn pan_offset_stays_finite_and_bounded() {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    let mut view = View3dState::default();
    view.pan_by_pixels(rect, egui::vec2(1e9, -1e9));
    assert!((view.camera_offset.length() - OFFSET_MAX).abs() < 1e-2, "{:?}", view.camera_offset);
    let before = view.camera_offset;
    for bad in [egui::vec2(f32::NAN, 0.0), egui::vec2(0.0, f32::INFINITY)] {
        view.pan_by_pixels(rect, bad);
        assert_eq!(view.camera_offset, before);
    }
    view.pan_by_pixels(egui::Rect::NOTHING, egui::vec2(5.0, 5.0));
    assert_eq!(view.camera_offset, before);
    // A cursor-anchored dolly out from the bound cannot push the offset past it either.
    view.zoom_at(rect, 0.01, Some(rect.right_top() + egui::vec2(-1.0, 1.0)));
    assert!(view.camera_offset.length() <= OFFSET_MAX * (1.0 + 1e-6), "{:?}", view.camera_offset);
}

#[test]
fn after_the_side_preset_a_rightward_drag_turns_about_the_vertical_z_axis() {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 800.0));
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Side);
    view.orbit_by_screen_drag(egui::vec2(20.0, 0.0));
    // z is the turntable axis (screen vertical) and stays straight up; world +x (towards the camera) swings right,
    // i.e. the front of the scene follows the cursor as in the default view.
    let z = axis_on_screen(&view, rect, 2);
    assert!(z.x.abs() < 1e-4 && z.y > 0.0, "{z:?}");
    assert!(axis_on_screen(&view, rect, 0).x > 0.0);
    assert!((view.yaw - (-HALF_PI + 0.2)).abs() < 1e-6 && view.pitch == 0.0 && view.z_up);
    // Same rule in the y frame: from Top, the near axis (+z) swings right on a rightward drag.
    view.apply_preset(CameraPreset::Top);
    view.orbit_by_screen_drag(egui::vec2(20.0, 0.0));
    assert!(axis_on_screen(&view, rect, 2).x > 0.0);
}

#[test]
fn preset_buttons_wrap_greedily_onto_at_most_three_rows() {
    let o = egui::pos2(10.0, 20.0);
    let row2 = PRESET_BTN_H + PRESET_GAP;
    // 40 + 3 + 50 = 93 <= 100 fits; 96 + 60 > 100 wraps; an exact fit (50 + 3 + 47 = 100) stays on the row.
    let rs = wrap_buttons(o, 100.0, &[40.0, 50.0, 60.0]).expect("fits");
    let mins: Vec<egui::Pos2> = rs.iter().map(|r| r.min).collect();
    assert_eq!(mins, vec![o, o + egui::vec2(43.0, 0.0), o + egui::vec2(0.0, row2)]);
    assert_eq!(rs[2].size(), egui::vec2(60.0, PRESET_BTN_H));
    assert_eq!(wrap_buttons(o, 100.0, &[50.0, 47.0]).expect("fits")[1].min, o + egui::vec2(53.0, 0.0));
    assert!(wrap_buttons(o, 100.0, &[120.0]).is_none(), "a button wider than the row");
    assert!(wrap_buttons(o, 100.0, &[90.0; 4]).is_none(), "four rows");
    assert_eq!(wrap_buttons(o, 100.0, &[90.0; 3]).expect("three rows")[2].min, o + egui::vec2(0.0, 2.0 * row2));
}

#[test]
fn gizmo_axis_ends_and_hits_match_the_view_rotation() {
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Top);
    let c = egui::pos2(100.0, 100.0);
    let ends = axis_ends(&view.rotation(), c);
    assert!(ends[0].pos.distance(c + egui::vec2(AXIS_LEN_PX, 0.0)) < 1e-4); // +x right
    assert!(ends[1].pos.distance(c + egui::vec2(0.0, -AXIS_LEN_PX)) < 1e-4); // +y up (screen y down)
    assert!(ends[2].pos.distance(c) < 1e-4 && (ends[2].depth - 1.0).abs() < 1e-6); // +z at the eye
    let gizmo = egui::Rect::from_center_size(c, egui::vec2(76.0, 76.0));
    let hit = |p: egui::Pos2| gizmo_hit(&view.rotation(), gizmo, p).map(|e| (e.axis, e.positive));
    assert_eq!(hit(c + egui::vec2(AXIS_LEN_PX + 6.0, 3.0)), Some((0, true)));
    assert_eq!(hit(c + egui::vec2(-AXIS_LEN_PX, 0.0)), Some((0, false)));
    assert_eq!(hit(c + egui::vec2(2.0, 1.0)), Some((2, true))); // +z in front of -z
    assert_eq!(hit(c + egui::vec2(AXIS_LEN_PX + 12.0, 0.0)), None);
    assert_eq!(preset_for_axis(0), CameraPreset::Side);
    assert_eq!(preset_for_axis(1), CameraPreset::Front);
    assert_eq!(preset_for_axis(2), CameraPreset::Top);
}
