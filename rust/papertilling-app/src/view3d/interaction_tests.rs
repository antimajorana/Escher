use super::*;

fn project(view: &View3dState, rect: egui::Rect, point: [f32; 3]) -> egui::Pos2 {
    let mvp = view.view_mvp(rect.width() / rect.height(), &orbit_rotation(view.yaw, view.pitch));
    let p = [point[0], point[1], point[2], 1.0];
    let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|col| mvp[col][row] * p[col]).sum());
    egui::pos2(
        rect.center().x + rect.width() * 0.5 * clip[0] / clip[3],
        rect.center().y - rect.height() * 0.5 * clip[1] / clip[3],
    )
}

#[test]
fn cursor_ray_dolly_preserves_projected_points_at_nonzero_depth_and_at_distance_limits() {
    let rect = egui::Rect::from_min_size(egui::pos2(20.0, 50.0), egui::vec2(720.0, 480.0));
    let point = [0.18, -0.12, 0.15];
    let mut view = View3dState::default();
    let cursor = project(&view, rect, point);
    for factor in [1.25, 0.25, 100.0, 1e-4, 2.0, 20.0] {
        view.zoom_at(rect, factor, Some(cursor));
        let after = project(&view, rect, point);
        assert!(after.distance(cursor) < 1e-3, "{factor}: {cursor:?} -> {after:?}");
        assert!((DIST_MIN..=DIST_MAX).contains(&view.distance));
    }
    let offset = view.camera_offset;
    let distance = view.distance;
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        view.zoom_at(rect, bad, Some(cursor));
        assert_eq!((view.distance, view.camera_offset), (distance, offset));
    }
    view.reset();
    assert_eq!(view.camera_offset, egui::Vec2::ZERO);
    assert_eq!(view.distance, DIST_DEFAULT);
    assert!(project(&view, rect, point).distance(cursor) < 1e-3);
}

#[test]
fn cursor_ray_uses_the_renderers_aspect_clamp_in_a_very_narrow_viewport() {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(64.0, 1000.0));
    let point = [0.01, 0.02, 0.01];
    let mut view = View3dState::default();
    let cursor = project(&view, rect, point);
    assert!(rect.contains(cursor));
    view.zoom_at(rect, 2.0, Some(cursor));
    assert!(project(&view, rect, point).distance(cursor) < 1e-3);
}

struct Harness {
    ctx: egui::Context,
    view: View3dState,
    rect: egui::Rect,
    time: f64,
}

impl Harness {
    fn new() -> Self {
        let mut h = Self {
            ctx: egui::Context::default(), view: View3dState::default(),
            rect: egui::Rect::NOTHING, time: 0.0,
        };
        h.frame(Vec::new());
        h.frame(Vec::new());
        h
    }

    fn frame(&mut self, events: Vec<egui::Event>) {
        self.time += 1.0 / 60.0;
        let out = self.ctx.run_ui(egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))),
            time: Some(self.time), events, ..Default::default()
        }, |ui| {
            self.rect = ui.available_rect_before_wrap();
            self.view.show(ui);
        });
        out.drop_without_applying_deltas();
    }

    fn drag(&mut self, from: egui::Pos2, to: egui::Pos2) {
        self.frame(vec![egui::Event::PointerMoved(from)]);
        self.frame(vec![button(from, true)]);
        for step in 1..=20 {
            self.frame(vec![egui::Event::PointerMoved(from + (to - from) * (step as f32 / 20.0))]);
        }
        self.frame(vec![button(to, false)]);
    }
}

fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE }
}

#[test]
fn established_orbit_crosses_the_hud_but_a_hud_origin_does_not_orbit() {
    let mut plain = Harness::new();
    let mut crossing = Harness::new();
    let r = crossing.rect;
    let from = egui::pos2(r.right() - 230.0, r.bottom() - 16.0);
    let to = egui::pos2(r.right() - 2.0, from.y);
    crossing.drag(from, to);
    plain.drag(from - egui::vec2(0.0, 100.0), to - egui::vec2(0.0, 100.0));
    assert!((crossing.view.yaw - plain.view.yaw).abs() < 1e-5);
    assert!((crossing.view.pitch - plain.view.pitch).abs() < 1e-5);

    let mut hud = Harness::new();
    let yaw = hud.view.yaw;
    let from = hud.rect.right_bottom() - egui::vec2(130.0, 16.0);
    hud.drag(from, from - egui::vec2(0.0, 120.0));
    assert_eq!(hud.view.yaw, yaw);
    assert_eq!(hud.view.pitch, 0.35);
}

#[test]
fn real_pinch_and_unmodified_keys_are_cursor_anchored_but_command_keys_only_scale_gui() {
    let mut h = Harness::new();
    let point = [0.18, -0.12, 0.15];
    let cursor = project(&h.view, h.rect, point);
    h.frame(vec![egui::Event::PointerMoved(cursor), egui::Event::Zoom(1.5)]);
    assert!((h.view.distance - DIST_DEFAULT / 1.5).abs() < 1e-5);
    assert!(project(&h.view, h.rect, point).distance(cursor) < 1e-3);
    h.frame(vec![egui::Event::Key {
        key: egui::Key::Plus, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::SHIFT,
    }]);
    assert!((h.view.distance - DIST_DEFAULT / (1.5 * ZOOM_STEP_FACTOR)).abs() < 1e-5);
    let distance = h.view.distance;
    h.frame(vec![egui::Event::Key {
        key: egui::Key::Minus, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::COMMAND,
    }]);
    h.frame(Vec::new()); // Apply egui's GUI scale change on the next pass.
    assert_eq!(h.view.distance, distance);
    assert!(h.ctx.zoom_factor() < 1.0);
}
