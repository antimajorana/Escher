//! Command Deck keyboard and focus rules (run first in `handle_global_shortcuts`).
//!
//! egui moves keyboard focus for Tab and the arrows in `Memory::begin_pass`, before the frame runs; consuming the
//! key alone does not stop that. So whenever a key below is acted on, `move_focus(None)` cancels the pending move.
//! Esc has already cleared the focus by the time this runs, which is why the rename test uses `gallery.renaming`.

use super::Realm;
use crate::app::{AppMode, NavTab, PapertillingApp};
use crate::ui::shortcuts::consume_unmodified_key;
use egui::{Event, FocusDirection, Key, KeyboardShortcut, Modifiers};

fn cancel_focus_move(ctx: &egui::Context) {
    ctx.memory_mut(|m| m.move_focus(FocusDirection::None));
}

/// Consume an unmodified press of any of `keys`.
fn take(ctx: &egui::Context, keys: &[Key]) -> bool {
    ctx.input_mut(|i| keys.iter().any(|&k| consume_unmodified_key(i, k)))
}

/// `/` as a key press (unmodified) or as typed text (some layouts need Shift); both events are removed so the
/// character never reaches the filter field. Returns `(hit, text_seen)`.
fn take_slash(ctx: &egui::Context) -> (bool, bool) {
    ctx.input_mut(|i| {
        let is_slash = |e: &Event| match e {
            Event::Text(t) => t == "/",
            Event::Key { key: Key::Slash, pressed: true, modifiers, .. } => modifiers.is_none(),
            _ => false,
        };
        let hit = i.events.iter().any(is_slash);
        let text_seen = i.events.iter().any(|e| matches!(e, Event::Text(t) if t == "/"));
        if hit {
            i.events.retain(|e| !is_slash(e) && !matches!(e, Event::Key { key: Key::Slash, .. }));
        }
        (hit, text_seen)
    })
}

/// The `/` text that belongs to a slash key press handled in an earlier frame (backends may deliver the key and
/// its text in different frames): drop it if it is the next typed input. Returns whether it is still awaited.
fn swallow_late_slash_text(ctx: &egui::Context) -> bool {
    ctx.input_mut(|i| {
        let next = i.events.iter().position(|e| {
            matches!(e, Event::Text(_) | Event::Paste(_)) || matches!(e, Event::Key { pressed: true, .. })
        });
        match next {
            None => true,
            Some(k) => {
                if matches!(&i.events[k], Event::Text(t) if t == "/") {
                    i.events.remove(k);
                }
                false
            }
        }
    })
}

const NUM_KEYS: [Key; 7] = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7];

impl PapertillingApp {
    /// Deck keys; true = the frame's remaining global shortcuts are blocked (the deck is open or just opened).
    pub(crate) fn handle_deck_keys(&mut self, ctx: &egui::Context) -> bool {
        if self.switcher.open {
            return false;
        }
        if self.deck.open {
            self.deck_keys_open(ctx);
            return true;
        }
        self.deck_keys_closed(ctx)
    }

    fn deck_keys_closed(&mut self, ctx: &egui::Context) -> bool {
        let in_combine = self.active_mode == AppMode::Defects && self.active_tab == NavTab::Combine;
        if ctx.text_edit_focused()
            || self.gallery.renaming
            || egui::Popup::is_any_open(ctx)
            || (in_combine
                && (self.combine.core_panel.has_open_modal() || self.combine.dragging || self.combine.cut_dragging))
        {
            return false;
        }
        if take(ctx, &[Key::Tab, Key::M]) {
            cancel_focus_move(ctx);
            self.deck_open(self.current_realm(), ctx);
            return true;
        }
        false
    }

    fn deck_keys_open(&mut self, ctx: &egui::Context) {
        if egui::Popup::is_any_open(ctx) {
            // a HUD menu or combo above the deck handles its own keys (Esc closes it, not the deck)
            return;
        }
        let focused = ctx.memory(|m| m.focused());
        let filter_focused = focused.is_some() && focused == self.deck.filter_id;
        if self.gallery.renaming || (ctx.text_edit_focused() && !filter_focused) {
            // the HUD rename field (or another text field) keeps its Enter / Esc
            return;
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            self.deck_close();
            return;
        }
        if ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::K))) {
            self.deck.focus_filter = true;
            return;
        }
        if self.deck.slash_text_pending {
            self.deck.slash_text_pending = swallow_late_slash_text(ctx);
        }
        let cols = self.deck.cols.max(1) as isize;
        if filter_focused {
            if take(ctx, &[Key::Enter]) {
                self.deck.equip_request = Some(self.deck.cur());
                if let Some(id) = self.deck.filter_id {
                    ctx.memory_mut(|m| m.surrender_focus(id));
                }
            } else if take(ctx, &[Key::ArrowUp]) {
                self.move_deck_cursor(-cols);
                cancel_focus_move(ctx);
            } else if take(ctx, &[Key::ArrowDown]) {
                self.move_deck_cursor(cols);
                cancel_focus_move(ctx);
            }
            return;
        }
        if let Some(n) = NUM_KEYS.iter().position(|&k| take(ctx, &[k])) {
            if let Some(r) = Realm::from_number(n + 1) {
                self.deck_select_realm(r);
            }
        } else if take(ctx, &[Key::Q]) {
            self.deck_select_realm(self.deck.realm.prev());
        } else if take(ctx, &[Key::E]) {
            self.deck_select_realm(self.deck.realm.next());
        } else if take(ctx, &[Key::ArrowLeft, Key::A]) {
            self.move_deck_cursor(-1);
        } else if take(ctx, &[Key::ArrowRight, Key::D]) {
            self.move_deck_cursor(1);
        } else if take(ctx, &[Key::ArrowUp, Key::W]) {
            self.move_deck_cursor(-cols);
        } else if take(ctx, &[Key::ArrowDown, Key::S]) {
            self.move_deck_cursor(cols);
        } else if take(ctx, &[Key::Enter, Key::Space]) {
            self.deck.equip_request = Some(self.deck.cur());
        } else if take(ctx, &[Key::Tab, Key::M]) {
            self.deck_close();
        } else {
            let (hit, text_seen) = take_slash(ctx);
            if hit {
                self.deck.focus_filter = true;
                self.deck.slash_text_pending = !text_seen;
            }
        }
        // arrows, Tab and Shift+Tab never walk focus through the panels hidden under the deck
        cancel_focus_move(ctx);
    }

    /// Move the cursor by `delta` cards, clamped to the cards drawn last frame.
    fn move_deck_cursor(&mut self, delta: isize) {
        let n = self.deck.n_cards;
        let new = if n == 0 {
            0
        } else {
            (self.deck.cur() as isize + delta).clamp(0, n as isize - 1) as usize
        };
        self.deck.set_cur(new);
    }
}
