//! Shared kittest helpers for the Command Deck tests (the harness is `analysis_mode::tests::harness()`, 1280x800).

use egui_kittest::Harness;

use super::Realm;
use super::realms::CardId;
use crate::app::{DefectsView, LatticesView, PapertillingApp};

pub(crate) type H = Harness<'static, PapertillingApp>;

/// Route to a Lattices sub-view through state (what the removed strip button did), then run two frames.
pub(crate) fn goto_lattices_view(h: &mut H, v: LatticesView) {
    h.state_mut().select_lattices_view(v);
    h.run_steps(2);
}

/// Route to a Defects sub-view through state (what the removed strip button did), then run two frames.
pub(crate) fn goto_defects_view(h: &mut H, v: DefectsView) {
    h.state_mut().select_defects_view(v);
    h.run_steps(2);
}

/// Press `key` (no modifiers) and run two frames.
pub(crate) fn press(h: &mut H, key: egui::Key) {
    h.key_press(key);
    h.run_steps(2);
}

pub(crate) fn press_mod(h: &mut H, mods: egui::Modifiers, key: egui::Key) {
    h.key_press_modifiers(mods, key);
    h.run_steps(2);
}

/// Press Tab and check that the deck opened.
pub(crate) fn open_deck(h: &mut H) {
    press(h, egui::Key::Tab);
    assert!(h.state().deck.open, "Tab opens the Command Deck");
}

/// Index of the card titled `title` in `realm` under the All pill with no filter.
pub(crate) fn deck_card_index(h: &mut H, realm: Realm, title: &str) -> usize {
    let cards = h.state_mut().deck_cards(realm, 0, "");
    cards
        .iter()
        .position(|c| c.title == title)
        .unwrap_or_else(|| panic!("realm {realm:?} has no card '{title}'"))
}

/// Equip `id` through state (as Enter on its card does), then run two frames.
pub(crate) fn equip_state(h: &mut H, id: CardId) {
    let ctx = h.ctx.clone();
    h.state_mut().deck_equip(id, &ctx);
    h.run_steps(2);
}
