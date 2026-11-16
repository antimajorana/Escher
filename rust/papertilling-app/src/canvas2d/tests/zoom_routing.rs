use super::*;

#[test]
fn established_pan_crosses_hud_without_losing_motion_and_hud_origin_does_not_pan() {
    let mut crossing = Harness::new();
    let mut plain = Harness::new();
    let from = crossing.rect.right_bottom() - egui::vec2(230.0, 16.0);
    let to = crossing.rect.right_bottom() - egui::vec2(2.0, 16.0);
    crossing.drag(from, to, 20, &[], false);
    plain.drag(from - egui::vec2(0.0, 100.0), to - egui::vec2(0.0, 100.0), 20, &[], false);
    assert!((crossing.canvas.pan - plain.canvas.pan).length() < 1e-4);
    assert!(crossing.canvas.pan.x > 200.0);

    let mut hud = Harness::new();
    let from = hud.rect.right_bottom() - egui::vec2(130.0, 16.0);
    let outs = hud.drag(from, from - egui::vec2(0.0, 100.0), 10, &[], true);
    assert_eq!(hud.canvas.pan, egui::Vec2::ZERO);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()));
}

#[test]
fn established_pin_drag_continues_over_hud_and_releases_normally() {
    let mut h = Harness::new();
    let from = h.rect.right_bottom() - egui::vec2(220.0, 16.0);
    let inside_hud = h.rect.right_bottom() - egui::vec2(130.0, 16.0);
    let markers = [pin(0, h.world(from))];
    h.frame(vec![egui::Event::PointerMoved(from)], &markers, true);
    h.frame(vec![press(from, true)], &markers, true);
    let first = h.frame(vec![egui::Event::PointerMoved(from + egui::vec2(20.0, 0.0))], &markers, true);
    assert!(first.dragged_marker.is_some());
    let over_hud = h.frame(vec![egui::Event::PointerMoved(inside_hud)], &markers, true);
    let (idx, world) = over_hud.dragged_marker.expect("HUD must not interrupt an established pin drag");
    assert_eq!(idx, 0);
    assert!(close(world, h.world(inside_hud), 1e-4));
    assert_eq!(h.canvas.pan, egui::Vec2::ZERO);
    let released = h.frame(vec![press(inside_hud, false)], &markers, true);
    assert!(released.drag_released);
    assert!(released.clicked_world.is_none());
}

#[test]
fn shifted_wheel_is_horizontal_and_never_replayed_as_zoom() {
    let mut h = Harness::new();
    let cursor = h.rect.center();
    let zoom = h.canvas.zoom;
    h.frame(vec![
        egui::Event::PointerMoved(cursor),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0.0, 120.0),
            phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel,
            modifiers: egui::Modifiers::SHIFT,
        },
    ], &[], false);
    for _ in 0..20 { h.frame(Vec::new(), &[], false); }
    assert_eq!(h.canvas.zoom, zoom);
}

#[test]
fn physical_plus_and_text_zoom_once_and_command_shortcuts_only_change_gui_scale() {
    let mut h = Harness::new();
    let cursor = h.rect.center();
    h.frame(vec![
        egui::Event::PointerMoved(cursor),
        egui::Event::Key {
            key: egui::Key::Plus, physical_key: None, pressed: true, repeat: false,
            modifiers: egui::Modifiers::SHIFT,
        },
        egui::Event::Text(String::from("+")),
    ], &[], false);
    assert!((h.canvas.zoom - 62.5).abs() < 1e-4, "one 1.25 step, not two");
    let zoom = h.canvas.zoom;
    h.frame(vec![egui::Event::Key {
        key: egui::Key::Equals, physical_key: None, pressed: true, repeat: false,
        modifiers: egui::Modifiers::COMMAND,
    }], &[], false);
    h.frame(Vec::new(), &[], false);
    assert_eq!(h.canvas.zoom, zoom);
    assert!(h.ctx.zoom_factor() > 1.0, "Cmd/Ctrl shortcut must remain available to egui");
}
