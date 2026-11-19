//! kittest navigation tests N7-N14 (M1 plan section 6.2; written by WP-T). Helpers live in `tests_nav.rs`.

use egui::Key;
use egui_kittest::kittest::Queryable;

use super::Realm;
use super::realms::{CardId, IoAction, card_realm};
use super::test_util::{H, equip_state, goto_defects_view, goto_lattices_view, open_deck, press};
use super::tests_nav::{click_in_hud, hud_disabled, start};
use crate::analysis_mode::AnalysisTarget;
use crate::analysis_mode::tests::{equip_keys, realm_key, settled, wait_until};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab};
use crate::mesh3d::DefectViewMode;
use crate::theme::StudioTheme;
use crate::workbench::WorkbenchView;

#[test]
fn every_workbench_card_equips_its_panel_and_sub_row() {
    let mut h = start();
    for view in WorkbenchView::ALL {
        let labels = h.state().workbench.sub_labels(view);
        for (i, title) in labels.iter().enumerate() {
            let realm = card_realm(CardId::Wb(view, u8::try_from(i).unwrap()));
            equip_keys(&mut h, realm, title);
            let a = h.state();
            assert_eq!(a.active_tab, NavTab::Workbench, "{view:?} row {i}");
            assert_eq!((a.workbench.view, a.workbench.sub_index(view)), (view, i), "{view:?} row {i}");
            assert!(
                h.query_all_by_label_contains("Workbench").next().is_none(),
                "{view:?} row {i}: no user-visible 'Workbench' text"
            );
        }
    }
}

#[test]
fn old_strips_are_gone() {
    let mut h = start();
    for mode in AppMode::ALL {
        h.state_mut().switch_mode(mode);
        h.run_steps(2);
        assert!(!h.state().deck.open);
        for gone in ["1. Lattices", "2. Defects", "3. Analysis", "Flat 3D: Slabs, Tubes, Flakes", "Detect & Fields"] {
            assert!(h.query_all_by_label(gone).next().is_none(), "{mode:?}: the old strip label '{gone}' is gone");
        }
        if mode == AppMode::Lattices {
            assert_eq!(h.query_all_by_label_contains("Lattices:").count(), 1, "only the HUD mode segment");
        }
    }
}

#[test]
fn esc_closes_deck_before_combine_chain() {
    let mut h = start();
    h.state_mut().switch_tab(NavTab::Combine);
    wait_until(&mut h, "Combine build", settled);
    {
        let app = h.state_mut();
        app.combine.selected_pin = Some(0);
        app.combine.sync_selection_to_canvas(&mut app.canvas2d);
    }
    h.run_steps(2);
    open_deck(&mut h);
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open, "Esc closes the deck");
    assert_eq!(h.state().combine.selected_pin, Some(0), "the same Esc did not reach the Combine chain");
}

#[test]
fn hud_view_toggle_and_undo_scope() {
    let mut h = start();
    goto_defects_view(&mut h, DefectsView::Catalogue2D);
    wait_until(&mut h, "Defects build", settled);
    click_in_hud(&mut h, "3D");
    assert_eq!(h.state().defects.view_mode, DefectViewMode::Sheet3D, "HUD 3D sets the Defects view mode");

    goto_lattices_view(&mut h, LatticesView::Plane2D);
    assert!(hud_disabled(&h, "3D"), "the 2D Plane has a fixed layout");
    assert!(hud_disabled(&h, "Undo"), "Undo is disabled outside Combine");

    h.state_mut().switch_tab(NavTab::Combine);
    wait_until(&mut h, "Combine build", settled);
    assert!(h.state().combine.specs.len() >= 2, "Combine starts with its pins");
    h.state_mut().combine.selected_pin = Some(1);
    h.run_steps(2);
    press(&mut h, Key::Delete);
    let n = h.state().combine.undo_stack.len();
    assert!(n > 0, "deleting a pin is undoable");
    assert!(!hud_disabled(&h, "Undo"), "Undo is enabled in Combine with history");
    click_in_hud(&mut h, "Undo");
    assert_eq!(h.state().combine.undo_stack.len(), n - 1, "HUD Undo pops one step");
}

#[test]
fn hud_is_one_row() {
    let mut h = start();
    for mode in AppMode::ALL {
        h.state_mut().switch_mode(mode);
        h.run_steps(2);
        let top = h.state().test_rects.top;
        assert!(top.height() <= 40.0, "{mode:?}: HUD height {}", top.height());
    }
    let check_host = |h: &H, what: &str| {
        let a = h.state();
        let (host, top, status) = (a.deck.host_rect, a.test_rects.top, a.test_rects.status);
        assert!((host.top() - top.bottom()).abs() < 0.5, "{what}: deck top {} vs HUD bottom {}", host.top(), top.bottom());
        assert!(
            (host.bottom() - status.top()).abs() < 0.5,
            "{what}: deck bottom {} vs status top {}",
            host.bottom(),
            status.top()
        );
    };
    check_host(&h, "1280x800");
    h.state_mut().switch_mode(AppMode::Lattices);
    h.set_size(egui::vec2(800.0, 400.0));
    h.run_steps(3);
    let top = h.state().test_rects.top;
    assert!(top.height() <= 40.0, "800x400: HUD height {}", top.height());
    let window = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 400.0)).expand(0.5);
    let theme = h
        .query_all_by_label("Theme: Paper")
        .find(|n| top.expand(1.0).contains_rect(n.rect()))
        .expect("the right cluster keeps the Theme button in the HUD row")
        .rect();
    assert!(window.contains_rect(theme), "Theme button {theme:?} inside the 800x400 window");
    check_host(&h, "800x400");
}

#[test]
fn deck_blocks_underlying_clicks_and_uses_tokens() {
    let mut h = start();
    open_deck(&mut h);
    let (alias0, tab0) = (h.state().gallery.base_item().alias_key.clone(), h.state().active_tab);
    let c = h.state().test_rects.left.center();
    h.hover_at(c);
    h.drag_at(c);
    h.drop_at(c);
    h.run_steps(2);
    assert!(h.state().gallery.base_item().alias_key == alias0, "the click did not reach the Gallery rail");
    assert_eq!(h.state().active_tab, tab0);
    assert!(h.state().deck.open, "a click on the deck keeps it open");

    click_in_hud(&mut h, "Theme: Paper");
    assert_eq!(h.state().theme, StudioTheme::Studio);
    assert!(h.state().deck.open, "switching the theme keeps the deck open");
    h.run_steps(2);
    assert!(h.query_all_by_label("⏎ Equip").next().is_some(), "the stat card renders under Studio");
}

/// Index of the kagome card in realm 3's Named pill.
fn kagome_named_index(h: &mut H) -> usize {
    let cards = h.state_mut().deck_cards(Realm::Tilings, 1, "");
    let a = h.state();
    cards
        .iter()
        .position(|c| match c.id {
            CardId::Lattice(r) => a.lattice_ref_item(r).and_then(|it| it.named_key.as_deref()) == Some("kagome"),
            _ => false,
        })
        .expect("kagome is in the Named pill")
}

/// Open the deck on realm 3's Named pill with the cursor on kagome.
fn deck_on_kagome(h: &mut H) {
    if !h.state().deck.open {
        open_deck(h);
    }
    press(h, realm_key(Realm::Tilings));
    let r = Realm::Tilings.number() - 1;
    h.state_mut().deck.pill[r] = 1;
    h.run_steps(2);
    let k = kagome_named_index(h);
    h.state_mut().deck.cursor[r] = k;
    h.run_steps(2);
}

#[test]
fn stat_card_buttons() {
    let mut h = start();
    let square = h.state().backend.find_named("square").expect("square").clone();
    h.state_mut().select_lattice_and_sync(square);
    h.run_steps(2);

    deck_on_kagome(&mut h);
    h.get_by_label("⚒ Open in Defects Forge").click();
    h.run_steps(2);
    wait_until(&mut h, "Open in Defects Forge", settled);
    let a = h.state();
    assert_eq!((a.active_mode, a.defects_view), (AppMode::Defects, DefectsView::Catalogue2D));
    assert_eq!(a.gallery.base_item().named_key.as_deref(), Some("kagome"));

    deck_on_kagome(&mut h);
    h.get_by_label("ψ Open in Analysis Lab").click();
    h.run_steps(2);
    let a = h.state();
    assert_eq!((a.active_mode, a.analysis.target), (AppMode::Analysis, AnalysisTarget::PlaneLattice));
}

#[test]
fn io_cards_never_open_dialogs_in_tests() {
    let mut h = start();
    equip_state(&mut h, CardId::Io(IoAction::ExportSvgPng));
    assert!(h.state().export_status.ends_with("(dialog skipped)"), "{}", h.state().export_status);
    assert!(!h.state().deck.open);
    // the same through the keys
    equip_keys(&mut h, Realm::PresetsIo, "Save session…");
    assert!(h.state().export_status.ends_with("(dialog skipped)"), "{}", h.state().export_status);
}
