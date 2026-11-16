use super::*;

fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }
}

#[test]
fn keyboard_modifiers_are_checked_on_both_state_and_events_and_shift_plus_is_allowed() {
    for modifiers in [egui::Modifiers::CTRL, egui::Modifiers::COMMAND, egui::Modifiers::ALT, egui::Modifiers::MAC_CMD] {
        for k in [egui::Key::Plus, egui::Key::Equals, egui::Key::Minus, egui::Key::Num0] {
            let ctx = egui::Context::default();
            let raw = egui::RawInput {
                events: vec![key(k, modifiers), egui::Event::Text(String::from("+"))],
                ..Default::default()
            };
            let out = ctx.run_ui(raw, |ui| {
                assert_eq!(ui.input(keyboard_zoom_action), None, "{modifiers:?} {k:?}");
            });
            out.drop_without_applying_deltas();
        }
        let mut input = egui::InputState::default();
        input.modifiers = modifiers;
        input.events = vec![egui::Event::Text(String::from("0"))];
        assert_eq!(keyboard_zoom_action(&input), None);
    }
    let ctx = egui::Context::default();
    let out = ctx.run_ui(egui::RawInput {
        events: vec![key(egui::Key::Plus, egui::Modifiers::SHIFT), egui::Event::Text(String::from("+"))],
        ..Default::default()
    }, |ui| assert_eq!(ui.input(keyboard_zoom_action), Some(ZoomKeyAction::In)));
    out.drop_without_applying_deltas();
}

#[test]
fn processed_scroll_is_not_replayed_from_raw_events_or_double_counted_as_ctrl_zoom() {
    for modifiers in [egui::Modifiers::NONE, egui::Modifiers::SHIFT, egui::Modifiers::COMMAND] {
        let ctx = egui::Context::default();
        let mut measured = false;
        let out = ctx.run_ui(egui::RawInput {
            events: vec![egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, 5.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers,
            }],
            ..Default::default()
        }, |ui| {
            if measured { return; }
            measured = true;
            ui.input_mut(|i| {
                let factor = input_zoom_factor(i);
                if modifiers == egui::Modifiers::NONE {
                    assert!((factor - 0.01_f32.exp()).abs() < 1e-6);
                    i.smooth_scroll_delta = egui::Vec2::ZERO; // Simulate consumption by a ScrollArea.
                    assert_eq!(input_zoom_factor(i), 1.0);
                } else if modifiers == egui::Modifiers::SHIFT {
                    assert_ne!(i.smooth_scroll_delta.x, 0.0);
                    assert_eq!(factor, 1.0, "horizontal Shift-wheel must not zoom");
                } else {
                    assert_ne!(i.zoom_delta(), 1.0);
                    assert_eq!(factor, i.zoom_delta(), "Ctrl/Cmd wheel must be counted once");
                }
            });
        });
        out.drop_without_applying_deltas();
        assert!(measured);
    }
}

#[test]
fn a_key_and_its_text_event_are_consumed_together_once() {
    let ctx = egui::Context::default();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
    let mut actions = Vec::new();
    for frame in 0..3 {
        let mut events = vec![egui::Event::PointerMoved(rect.center())];
        if frame == 2 {
            events.extend([key(egui::Key::Plus, egui::Modifiers::NONE), egui::Event::Text(String::from("+"))]);
        }
        let out = ctx.run_ui(egui::RawInput {
            screen_rect: Some(rect), events, ..Default::default()
        }, |ui| {
            let (_, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            if let Some(action) = take_keyboard_zoom_action(ui, &response) {
                actions.push(action);
                assert_eq!(take_keyboard_zoom_action(ui, &response), None);
            }
        });
        out.drop_without_applying_deltas();
    }
    assert_eq!(actions, [ZoomKeyAction::In]);
}
