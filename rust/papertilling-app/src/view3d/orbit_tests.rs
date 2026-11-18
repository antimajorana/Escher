//! Regression tests of orbiting after the axis presets. Known answers (textbook rotations, view space, x right,
//! y up, z towards the camera): a horizontal drag of `dx` px turns the whole view by `a = 0.01 dx` about the screen
//! vertical, so every view-space column becomes `Ry(a) v = (c x + s z, y, -s x + c z)`; a vertical drag of `dy` px
//! tilts it by `a = 0.01 dy` about the screen horizontal, `Rx(a) v = (x, c y - s z, s y + c z)`, up to the pitch limit.

use super::camera::{CameraPreset, PITCH_DRAG_LIMIT};
use super::*;

const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// View-space image of world axis `axis` (column of the view rotation).
fn col(view: &View3dState, axis: usize) -> [f32; 3] {
    let r = view.rotation();
    [r[axis][0], r[axis][1], r[axis][2]]
}

fn cols(view: &View3dState) -> [[f32; 3]; 3] {
    [col(view, 0), col(view, 1), col(view, 2)]
}

fn ry(a: f32, v: [f32; 3]) -> [f32; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]
}

fn rx(a: f32, v: [f32; 3]) -> [f32; 3] {
    let (s, c) = a.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

fn assert_close(got: [f32; 3], want: [f32; 3], what: &str) {
    let err = (0..3).map(|k| (got[k] - want[k]).abs()).fold(0.0, f32::max);
    assert!(err < 1e-5, "{what}: {got:?} != {want:?}");
}

#[test]
fn front_and_side_horizontal_drags_turn_about_the_vertical_z_axis_without_a_cap() {
    for preset in [CameraPreset::Front, CameraPreset::Side] {
        // 150 px is where the old roll-based Side hit the pitch clamp; 700 px is more than a full turn.
        for dx in [20.0, 150.0, 300.0, 700.0] {
            let mut view = View3dState::default();
            view.apply_preset(preset);
            let before = cols(&view);
            view.orbit_by_screen_drag(egui::vec2(dx, 0.0));
            for (k, b) in before.iter().enumerate() {
                assert_close(col(&view, k), ry(0.01 * dx, *b), &format!("{preset:?} dx {dx} axis {k}"));
            }
            // World z stays straight up on screen: no in-plane spin ("dial") and no tumble.
            assert_close(col(&view, 2), [0.0, 1.0, 0.0], &format!("{preset:?} dx {dx}: z up"));
        }
    }
    // The 300 px case by hand: Side has world y = view +x, so it ends at (cos 3, 0, -sin 3).
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Side);
    view.orbit_by_screen_drag(egui::vec2(300.0, 0.0));
    assert_close(col(&view, 1), [3.0_f32.cos(), 0.0, -(3.0_f32.sin())], "Side +300 px: world y");
}

#[test]
fn front_and_side_vertical_drags_tilt_both_ways_and_stop_at_the_pitch_limit() {
    for preset in [CameraPreset::Front, CameraPreset::Side] {
        for dy in [-50.0, 50.0] {
            let mut view = View3dState::default();
            view.apply_preset(preset);
            let before = cols(&view);
            view.orbit_by_screen_drag(egui::vec2(0.0, dy));
            for (k, b) in before.iter().enumerate() {
                assert_close(col(&view, k), rx(0.01 * dy, *b), &format!("{preset:?} dy {dy} axis {k}"));
            }
        }
        // A long vertical drag tilts to the drag limit and stops (no end-over-end tumble).
        let mut view = View3dState::default();
        view.apply_preset(preset);
        let before = cols(&view);
        view.orbit_by_screen_drag(egui::vec2(0.0, 1000.0));
        for (k, b) in before.iter().enumerate() {
            assert_close(col(&view, k), rx(PITCH_DRAG_LIMIT, *b), &format!("{preset:?} long tilt axis {k}"));
        }
    }
}

#[test]
fn orbiting_from_a_pitch_set_on_the_pole_neither_snaps_nor_leaves_the_pole_range() {
    let mut view = View3dState { pitch: -HALF_PI, ..Default::default() };
    view.orbit_by(0.3, 0.01); // inward: moves by exactly 0.01, no snap to -1.50
    assert!((view.pitch - (-HALF_PI + 0.01)).abs() < 1e-6, "{}", view.pitch);
    // Outward beyond the soft limit: the pitch holds (no snap to -1.50, never past -pi/2).
    view.orbit_by(0.0, -1.0);
    assert!((view.pitch - (-HALF_PI + 0.01)).abs() < 1e-6, "{}", view.pitch);
    view.pitch = -HALF_PI;
    view.orbit_by(0.0, -1.0);
    assert_eq!(view.pitch, -HALF_PI);
    view.orbit_by(0.0, 0.2);
    view.orbit_by(0.0, -1.0); // once inside the soft limit, drags clamp at it again
    assert_eq!(view.pitch, -PITCH_DRAG_LIMIT);
    view.pitch = -HALF_PI;
    for _ in 0..50 {
        view.orbit_by(0.13, -0.05);
        assert!(view.rotation().iter().flatten().all(|v| v.is_finite()));
    }
    view.orbit_by(f32::NAN, 0.1);
    assert!(view.yaw.is_finite() && view.pitch.is_finite());
}
