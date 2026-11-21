//! kittest navigation tests N1-N6 (M1 plan section 6.2; written by WP-T). N7-N14 are in `tests_nav2.rs`.
//!
//! Keys go through the real egui input path (`key_press` + frames), so the focus rules of `keys.rs` are exercised.

use egui::{Event, Key, Modifiers};
use egui_kittest::kittest::{NodeT, Queryable};

use super::Realm;
use super::test_util::{H, open_deck, press, press_mod};
use crate::analysis_mode::tests::{equip_keys, harness, realm_key, settled, wait_until};
use crate::app::{LatticesView, NavTab};
use crate::backend::SurfaceTargetUi;

/// A fresh 1280x800 app (Paper theme, `Advanced` on for All Views tests) after its first frames.
pub(super) fn start() -> H {
    let mut h = harness();
    h.state_mut().set_advanced(true);
    h.run_steps(3);
    h
}

/// Click the node labelled `label` inside the HUD row. The Gallery rail may show the same lattice name or group
/// symbol underneath, so a plain `get_by_label` could find two nodes.
pub(super) fn click_in_hud(h: &mut H, label: &str) {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label)
        .find(|n| top.contains_rect(n.rect()))
        .unwrap_or_else(|| panic!("no '{label}' in the HUD"))
        .click();
    h.run_steps(2);
}

/// Whether the HUD node labelled `label` is disabled.
pub(super) fn hud_disabled(h: &H, label: &str) -> bool {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label)
        .find(|n| top.contains_rect(n.rect()))
        .unwrap_or_else(|| panic!("no '{label}' in the HUD"))
        .accesskit_node()
        .is_disabled()
}

/// Click the Gallery search box (the text field inside the left rail).
fn click_gallery_search(h: &mut H) {
    let left = h.state().test_rects.left.expand(1.0);
    h.get_all_by_role(egui::accesskit::Role::TextInput)
        .find(|n| left.contains_rect(n.rect()))
        .expect("the Gallery search box is in the left rail")
        .click();
    h.run_steps(2);
}

fn drop_focus(h: &mut H) {
    h.ctx.memory_mut(|m| {
        if let Some(id) = m.focused() {
            m.surrender_focus(id);
        }
    });
    h.run_steps(2);
}

#[test]
fn tab_and_m_toggle_deck() {
    let mut h = start();
    let s0 = h.state().deck.realm_switches;
    press(&mut h, Key::Tab);
    assert!(h.state().deck.open, "Tab opens the deck");
    assert_eq!(h.state().deck.realm, h.state().current_realm());
    assert_eq!(h.state().deck.realm, Realm::Tilings, "the start view (2D Plane) belongs to Tilings & Nets");
    press(&mut h, Key::Tab);
    assert!(!h.state().deck.open, "Tab closes the deck");
    press(&mut h, Key::M);
    assert!(h.state().deck.open, "M opens the deck");
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open, "Esc closes the deck");
    assert_eq!(h.state().deck.realm_switches, s0, "opening and closing never count as realm switches");
    assert!(h.ctx.memory(|m| m.focused()).is_none(), "no hidden widget kept keyboard focus");
}

#[test]
fn tab_is_not_stolen_from_text_fields() {
    let mut h = start();
    click_gallery_search(&mut h);
    assert!(h.ctx.text_edit_focused(), "the Gallery search box has focus");
    press(&mut h, Key::Tab);
    assert!(!h.state().deck.open, "Tab in a text field does not open the deck");

    click_gallery_search(&mut h);
    h.key_press(Key::M);
    h.event(Event::Text(String::from("m")));
    h.run_steps(2);
    assert!(!h.state().deck.open, "M typed in a text field does not open the deck");
    assert_eq!(h.state().gallery.filter.query, "m", "the M reached the search box");
    h.state_mut().gallery.filter.query.clear();
    drop_focus(&mut h);

    // the rename field: Tab and M stay with it
    h.state_mut().gallery.begin_rename();
    h.run_steps(2);
    assert!(h.state().gallery.renaming);
    press(&mut h, Key::Tab);
    press(&mut h, Key::M);
    assert!(!h.state().deck.open, "Tab / M during a rename do not open the deck");
    h.state_mut().gallery.cancel_rename();
    drop_focus(&mut h);

    // deck open, then the HUD rename: Enter commits the rename (no equip), Esc cancels it (the deck stays open)
    open_deck(&mut h);
    let tab0 = h.state().active_tab;
    h.state_mut().gallery.begin_rename();
    h.run_steps(2);
    assert!(h.ctx.text_edit_focused(), "the HUD rename field has focus");
    press(&mut h, Key::Enter);
    assert!(!h.state().gallery.renaming, "Enter committed the rename");
    assert!(h.state().deck.open, "Enter in the rename field does not equip a card");
    assert_eq!(h.state().active_tab, tab0);
    h.state_mut().gallery.begin_rename();
    h.run_steps(2);
    press(&mut h, Key::Escape);
    assert!(!h.state().gallery.renaming, "Esc cancelled the rename");
    assert!(h.state().deck.open, "Esc in the rename field leaves the deck open");
}

#[test]
fn number_keys_and_q_e_switch_realms() {
    let mut h = start();
    let s0 = h.state().deck.realm_switches;
    open_deck(&mut h);
    assert_eq!(h.state().deck.realm, Realm::Tilings);
    assert_eq!(h.state().deck.realm_switches, s0, "opening never counts");
    for r in Realm::SHOWN {
        press(&mut h, realm_key(r));
        assert_eq!(h.state().deck.realm.number(), r.number(), "number key {}", r.number());
    }
    // 3 -> 1 -> 2 -> 3 -> 4 -> 5 -> 7 (C1: realm 6 is hidden): every press changed the realm
    assert_eq!(h.state().deck.realm_switches, s0 + 6);
    press(&mut h, Key::Num6);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::PresetsIo, s0 + 6), "key 6 does nothing");
    press(&mut h, Key::Q);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::Forge, s0 + 7), "Q skips realm 6");
    press(&mut h, Key::E);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::PresetsIo, s0 + 8), "E skips realm 6");
    press(&mut h, Key::Num1);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::Geometries, s0 + 9));
    press(&mut h, Key::Q);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::PresetsIo, s0 + 10), "Q wraps");
    press(&mut h, Key::E);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::Geometries, s0 + 11), "E wraps");
    press(&mut h, Key::Num1);
    assert_eq!(h.state().deck.realm_switches, s0 + 11, "selecting the same realm is not a switch");
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open);
    press(&mut h, Key::Num3);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), (Realm::Geometries, s0 + 11), "closed deck");
}

#[test]
fn breadcrumb_click_opens_matching_realm() {
    let mut h = start();
    let bc = h.state().breadcrumb();
    let group = bc.group.expect("the 2D Plane shows the group segment").text;
    let lattice = bc.lattice.expect("the 2D Plane shows the lattice segment").text;
    for (label, realm) in [
        (String::from("E² Plane"), Realm::Geometries),
        (group, Realm::Groups),
        (lattice, Realm::Tilings),
        (String::from("Lattice: 2D Plane"), Realm::Tilings),
    ] {
        click_in_hud(&mut h, &label);
        assert!(h.state().deck.open, "'{label}' opens the deck");
        assert_eq!(h.state().deck.realm, realm, "'{label}'");
        press(&mut h, Key::Escape);
        assert!(!h.state().deck.open);
    }
    equip_keys(&mut h, Realm::Forge, "Place & fuse defects");
    click_in_hud(&mut h, "Defects: Combine, Fuse & Move");
    assert_eq!((h.state().deck.open, h.state().deck.realm), (true, Realm::Forge));
    // with the deck open, a segment only switches the realm
    click_in_hud(&mut h, "E² Plane");
    assert_eq!((h.state().deck.open, h.state().deck.realm), (true, Realm::Geometries));
}

#[test]
fn ctrl_k_focuses_search() {
    let mut h = start();
    press_mod(&mut h, Modifiers::COMMAND, Key::K);
    assert!(h.state().switcher.open, "deck closed: Ctrl+K opens the Quick Lattice Switcher");
    assert!(h.ctx.text_edit_focused(), "with its query field focused");
    press(&mut h, Key::Escape);
    assert!(!h.state().switcher.open);

    open_deck(&mut h);
    press_mod(&mut h, Modifiers::COMMAND, Key::K);
    assert!(!h.state().switcher.open, "deck open: Ctrl+K never opens the switcher");
    assert!(h.state().deck.open);
    let id = h.state().deck.filter_id.expect("the deck filter was drawn");
    assert!(h.ctx.memory(|m| m.has_focus(id)), "Ctrl+K focuses the deck filter");
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open, "Esc in the filter closes the deck");

    open_deck(&mut h);
    press(&mut h, realm_key(Realm::Manifolds));
    // "/" arrives as a key press and as typed text; neither lands in the field
    h.key_press(Key::Slash);
    h.event(Event::Text(String::from("/")));
    h.run_steps(2);
    let id = h.state().deck.filter_id.expect("the deck filter was drawn");
    assert!(h.ctx.memory(|m| m.has_focus(id)), "/ focuses the deck filter");
    assert!(h.state().deck.filter.is_empty(), "the '/' never reaches the filter");
    h.event(Event::Text(String::from("klein")));
    h.run_steps(2);
    assert_eq!(h.state().deck.filter, "klein");
    assert_eq!(h.state().deck.n_cards, 1, "realm 4 filtered by 'klein' leaves one card");
    press(&mut h, Key::Enter);
    wait_until(&mut h, "Klein bottle", settled);
    let a = h.state();
    assert!(!a.deck.open, "Enter in the filter equips the highlighted card");
    assert_eq!((a.lattices_view, a.surfaces.target), (LatticesView::Manifolds, SurfaceTargetUi::KleinBottle));
}

#[test]
fn arrows_wasd_move_cursor_and_enter_equips() {
    let mut h = start();
    open_deck(&mut h);
    press(&mut h, realm_key(Realm::Manifolds));
    let r = Realm::Manifolds.number() - 1;
    h.state_mut().deck.pill[r] = 0;
    h.state_mut().deck.cursor[r] = 0;
    h.run_steps(2);
    let cols = h.state().deck.cols.max(1);
    let n = h.state().deck.n_cards;
    assert!(n > 2 + cols, "realm 4 lists more than a row of cards ({n})");
    let cur = |h: &H| h.state().deck.cursor[r];
    for (key, want) in [
        (Key::D, 1),
        (Key::ArrowRight, 2),
        (Key::S, 2 + cols),
        (Key::W, 2),
        (Key::A, 1),
        (Key::ArrowLeft, 0),
        (Key::A, 0),
        (Key::ArrowUp, 0),
    ] {
        press(&mut h, key);
        assert_eq!(cur(&h), want, "{key:?} (cols {cols})");
    }
    for _ in 0..=n / cols {
        press(&mut h, Key::ArrowDown);
    }
    assert_eq!(cur(&h), n - 1, "the cursor stops on the last card");
    press(&mut h, Key::D);
    assert_eq!(cur(&h), n - 1, "never past the last card");
    equip_keys(&mut h, Realm::Manifolds, "Klein bottle");
    let a = h.state();
    assert!(!a.deck.open);
    assert_eq!((a.active_tab, a.surfaces.target), (NavTab::Surfaces, SurfaceTargetUi::KleinBottle));
}
