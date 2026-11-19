//! Shared kittest helpers for the Command Deck tests (the harness is `analysis_mode::tests::harness()`, 1280x800).

use egui_kittest::Harness;

use super::Realm;
use super::realms::CardId;
use crate::analysis_mode::tests::wait_until;
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
#[allow(dead_code)] // M1 prewire: used by the navigation tests (WP-T).
pub(crate) fn press(h: &mut H, key: egui::Key) {
    h.key_press(key);
    h.run_steps(2);
}

#[allow(dead_code)] // M1 prewire: used by the navigation tests (WP-T).
pub(crate) fn press_mod(h: &mut H, mods: egui::Modifiers, key: egui::Key) {
    h.key_press_modifiers(mods, key);
    h.run_steps(2);
}

/// Press Tab and check that the deck opened.
#[allow(dead_code)] // M1 prewire: used by the navigation tests (WP-T).
pub(crate) fn open_deck(h: &mut H) {
    press(h, egui::Key::Tab);
    assert!(h.state().deck.open, "Tab opens the Command Deck");
}

/// Index of the card titled `title` in `realm` under the All pill with no filter.
#[allow(dead_code)] // M1 prewire: used by the navigation tests (WP-T).
pub(crate) fn deck_card_index(h: &mut H, realm: Realm, title: &str) -> usize {
    let cards = h.state_mut().deck_cards(realm, 0, "");
    cards
        .iter()
        .position(|c| c.title == title)
        .unwrap_or_else(|| panic!("realm {realm:?} has no card '{title}'"))
}

/// Equip `id` through state (as Enter on its card does), then run two frames.
#[allow(dead_code)] // M1 prewire: used by the state-level tests (WP-T).
pub(crate) fn equip_state(h: &mut H, id: CardId) {
    let ctx = h.ctx.clone();
    h.state_mut().deck_equip(id, &ctx);
    h.run_steps(2);
}

/// No build is running in the views a card can lead to.
#[allow(dead_code)] // M1 prewire: used by the navigation tests (WP-T).
pub(crate) fn idle(a: &PapertillingApp) -> bool {
    !a.workbench_busy()
        && !a.workbench.dirty
        && !a.surfaces.is_building()
        && !a.defects.is_busy()
        && !a.combine.is_building()
        && !a.analysis.is_busy()
}

/// Equip the card titled `title` in `realm` with real key presses: open, number key, D / S to the card, Enter;
/// then wait for the build.
#[allow(dead_code)] // M1 prewire: used by the navigation and quarantine tests (WP-T).
pub(crate) fn equip_by_keys(h: &mut H, realm: Realm, title: &str) {
    if !h.state().deck.open {
        open_deck(h);
    }
    let num = [
        egui::Key::Num1,
        egui::Key::Num2,
        egui::Key::Num3,
        egui::Key::Num4,
        egui::Key::Num5,
        egui::Key::Num6,
        egui::Key::Num7,
    ][realm.number() - 1];
    press(h, num);
    h.state_mut().deck.pill[realm.number() - 1] = 0;
    h.state_mut().deck.filter.clear();
    h.run_steps(2);
    let target = deck_card_index(h, realm, title);
    let cols = h.state().deck.cols.max(1);
    let cur = h.state().deck.cursor[realm.number() - 1];
    let (row, col) = (target / cols, target % cols);
    let (crow, ccol) = (cur / cols, cur % cols);
    let (key_r, key_c) = (if row >= crow { egui::Key::S } else { egui::Key::W }, if col >= ccol {
        egui::Key::D
    } else {
        egui::Key::A
    });
    for _ in 0..row.abs_diff(crow) {
        press(h, key_r);
    }
    for _ in 0..col.abs_diff(ccol) {
        press(h, key_c);
    }
    assert_eq!(h.state().deck.cursor[realm.number() - 1], target, "cursor reached '{title}'");
    press(h, egui::Key::Enter);
    assert!(!h.state().deck.open, "Enter on '{title}' closes the deck");
    wait_until(h, title, idle);
}
