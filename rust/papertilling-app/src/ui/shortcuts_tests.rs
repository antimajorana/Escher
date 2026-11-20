//! Stage 6 global shortcut routing and priority-ordered Escape tests (`egui_kittest`).

use egui::{Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use crate::app::{NavTab, PapertillingApp};
use crate::combine::ToolMode;
use crate::presets::PRESETS;
use crate::theme::{StudioTheme, typography};

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1400.0, 900.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(|ui, app| { let _ = app.frame_ui(ui); }, app)
}

fn wait_combine(h: &mut Harness<'static, PapertillingApp>) {
    for _ in 0..400 {
        h.step();
        if !h.state().combine.is_building() && !h.state().combine.is_initializing() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn ctrl_b_and_ctrl_i_toggle_side_panels_unless_text_input_focused() {
    let mut h = harness();
    h.run_steps(2);
    assert!(h.state().left_rail_open && h.state().right_inspector_open);

    h.key_press_modifiers(Modifiers::COMMAND, Key::B);
    h.run_steps(2);
    assert!(!h.state().left_rail_open);
    assert_eq!(h.state().test_rects.left, egui::Rect::NOTHING);

    h.key_press_modifiers(Modifiers::COMMAND, Key::I);
    h.run_steps(2);
    assert!(!h.state().right_inspector_open);
    assert_eq!(h.state().test_rects.right, egui::Rect::NOTHING);

    h.key_press_modifiers(Modifiers::COMMAND, Key::B);
    h.key_press_modifiers(Modifiers::COMMAND, Key::I);
    h.run_steps(2);
    assert!(h.state().left_rail_open && h.state().right_inspector_open);

    // Focused text box suppresses Ctrl+B / Ctrl+I.
    h.get_by_role(egui::accesskit::Role::TextInput).focus();
    h.run_steps(2);
    h.key_press_modifiers(Modifiers::COMMAND, Key::B);
    h.key_press_modifiers(Modifiers::COMMAND, Key::I);
    h.run_steps(2);
    assert!(h.state().left_rail_open && h.state().right_inspector_open);
}

#[test]
fn combine_undo_redo_delete_and_priority_ordered_escape_chain() {
    let mut h = harness();
    h.run_steps(2);
    h.state_mut().switch_tab(NavTab::Combine);
    wait_combine(&mut h);
    assert_eq!(h.state().combine.specs.len(), 2);

    // Select pin 0, open popover, and switch tool to PlaceDislocation.
    {
        let app = h.state_mut();
        app.combine.selected_pin = Some(0);
        app.combine.popover_open = true;
        app.combine.tool_mode = ToolMode::PlaceDislocation;
        app.combine.sync_selection_to_canvas(&mut app.canvas2d);
    }
    h.run_steps(2);

    // Open Examples menu: Escape #0 must close the popup WITHOUT touching popover_open, tool_mode, or selected_pin.
    h.get_by_label("Examples").click();
    h.run_steps(3);
    assert!(h.query_by_label(PRESETS[0].name).is_some());
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.query_by_label(PRESETS[0].name).is_none(), "Esc closed open popup first");
    assert!(h.state().combine.popover_open);
    assert_eq!(h.state().combine.tool_mode, ToolMode::PlaceDislocation);
    assert_eq!(h.state().combine.selected_pin, Some(0));

    // Inline rename open: Escape yields to rename bar without closing popover.
    h.state_mut().gallery.begin_rename();
    h.run_steps(2);
    assert!(h.state().gallery.renaming);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(!h.state().gallery.renaming, "Esc cancelled inline rename");
    assert!(h.state().combine.popover_open, "popover stayed open while rename took Esc");

    // Escape #1 -> closes popover, keeps tool_mode and selected_pin!
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(!h.state().combine.popover_open);
    assert_eq!(h.state().combine.tool_mode, ToolMode::PlaceDislocation);
    assert_eq!(h.state().combine.selected_pin, Some(0));

    // Escape #2 -> resets tool_mode to PanAndDrag, keeps selected_pin!
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().combine.tool_mode, ToolMode::PanAndDrag);
    assert_eq!(h.state().combine.selected_pin, Some(0));

    // Escape #3 -> clears selected_pin!
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().combine.selected_pin, None);

    // Select pin 1 and press Delete -> 1 pin left; Ctrl+Z undoes -> 2 pins; Ctrl+Shift+Z redoes -> 1 pin;
    // Ctrl+Z then Ctrl+Y redoes -> 1 pin!
    h.state_mut().combine.selected_pin = Some(1);
    h.run_steps(2);
    h.key_press(Key::Delete);
    h.run_steps(2);
    assert_eq!(h.state().combine.specs.len(), 1);
    assert_eq!(h.state().combine.selected_pin, None);

    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run_steps(2);
    assert_eq!(h.state().combine.specs.len(), 2, "Ctrl+Z undid Delete");

    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z);
    h.run_steps(2);
    assert_eq!(h.state().combine.specs.len(), 1, "Ctrl+Shift+Z redid Delete");

    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run_steps(2);
    assert_eq!(h.state().combine.specs.len(), 2);

    h.key_press_modifiers(Modifiers::COMMAND, Key::Y);
    h.run_steps(2);
    assert_eq!(h.state().combine.specs.len(), 1, "Ctrl+Y redid Delete");

    // In Sheet3D view mode, invisible 2D popover does not swallow Escape, and Delete works on table selection.
    {
        let app = h.state_mut();
        app.combine.view_mode = crate::mesh3d::DefectViewMode::Sheet3D;
        app.combine.selected_pin = Some(0);
        app.combine.popover_open = true;
    }
    h.run_steps(2);
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().combine.selected_pin, None, "single Esc in Sheet3D clears selection without invisible popover trap");

    h.state_mut().combine.selected_pin = Some(0);
    h.run_steps(2);
    h.key_press(Key::Backspace);
    h.run_steps(2);
    assert!(h.state().combine.specs.is_empty(), "Backspace in Sheet3D deletes table-selected pin");
}
