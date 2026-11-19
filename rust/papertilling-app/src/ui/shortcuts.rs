//! Global keyboard shortcuts and priority-ordered Escape handling.

use crate::app::{NavTab, PapertillingApp};
use crate::combine::ToolMode;
use crate::mesh3d::DefectViewMode;

fn consume_unmodified_key(i: &mut egui::InputState, key: egui::Key) -> bool {
    if !i.modifiers.is_none() {
        return false;
    }
    let has_unmodified = i.events.iter().any(|ev| {
        matches!(
            ev,
            egui::Event::Key { key: k, pressed: true, modifiers, .. }
                if *k == key && modifiers.is_none()
        )
    });
    has_unmodified && i.consume_key(egui::Modifiers::NONE, key)
}

impl PapertillingApp {
    pub(crate) fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        if self.switcher.open {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                self.switcher.open = false;
                return;
            }
        } else if !egui::Popup::is_any_open(ctx) {
            if self.gallery.renaming
                || ctx.egui_wants_keyboard_input()
                || (self.active_tab == NavTab::Combine && self.combine.core_panel.has_open_modal())
            {
                // Yield without consuming Esc so inline rename, focused TextEdit/DragValue, or modal gets it.
            } else if self.active_tab == NavTab::Combine && !self.combine.dragging && !self.combine.cut_dragging {
                if self.combine.selected_pin.is_some()
                    && self.combine.popover_open
                    && self.combine.view_mode != DefectViewMode::Sheet3D
                    && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
                {
                    if self.combine.commit_cut_edit_anchor() {
                        let it = self.gallery.active_item().clone();
                        self.combine.request_build(&it, false, self.combine.auto_compare, false);
                    }
                    self.combine.popover_open = false;
                    return;
                }
                if self.combine.tool_mode != ToolMode::PanAndDrag
                    && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
                {
                    self.combine.tool_mode = ToolMode::PanAndDrag;
                    return;
                }
                if self.combine.selected_pin.is_some()
                    && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
                {
                    if self.combine.commit_cut_edit_anchor() {
                        let it = self.gallery.active_item().clone();
                        self.combine.request_build(&it, false, self.combine.auto_compare, false);
                    }
                    self.combine.clear_selection();
                    self.canvas2d.set_selected_marker(None);
                    self.canvas2d.set_cut_rays(&self.combine.current_cut_rays());
                    return;
                }
            }
        }

        if ctx.egui_wants_keyboard_input()
            || self.gallery.renaming
            || egui::Popup::is_any_open(ctx)
            || (self.active_tab == NavTab::Combine && self.combine.core_panel.has_open_modal())
        {
            return;
        }

        let k_switcher = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
        if ctx.input_mut(|i| i.consume_shortcut(&k_switcher)) {
            self.toggle_switcher();
            return;
        }
        if self.switcher.open {
            return;
        }

        let k_sidebar = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::B);
        if ctx.input_mut(|i| i.consume_shortcut(&k_sidebar)) {
            self.left_rail_open = !self.left_rail_open;
        }
        let k_inspector = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::I);
        if ctx.input_mut(|i| i.consume_shortcut(&k_inspector)) {
            self.right_inspector_open = !self.right_inspector_open;
        }

        if self.active_tab == NavTab::Combine && !self.combine.dragging && !self.combine.cut_dragging {
            let redo_shift = egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                egui::Key::Z,
            );
            let redo_y = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
            let undo_z = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
            if ctx.input_mut(|i| i.consume_shortcut(&redo_shift) || i.consume_shortcut(&redo_y)) {
                let it = self.gallery.active_item().clone();
                self.combine.redo_last(&it);
                self.combine.sync_selection_to_canvas(&mut self.canvas2d);
            } else if ctx.input_mut(|i| i.consume_shortcut(&undo_z)) {
                let it = self.gallery.active_item().clone();
                self.combine.undo_last(&it);
                self.combine.sync_selection_to_canvas(&mut self.canvas2d);
            } else if let Some(idx) = self.combine.selected_pin
                && ctx.input_mut(|i| {
                    consume_unmodified_key(i, egui::Key::Delete)
                        || consume_unmodified_key(i, egui::Key::Backspace)
                })
            {
                let it = self.gallery.active_item().clone();
                self.combine.delete_pin(idx, &it, Some(&mut self.canvas2d));
            }
        }
    }
}
