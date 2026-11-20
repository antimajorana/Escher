//! C1 test (plan §4, T1): Analysis is postponed and unreachable from any user input. The expectations are the plan's
//! rules: no visible label names Analysis in either shown mode, with the deck open (every shown realm) or closed;
//! key 6 changes nothing; a v1, v2 or v3 session saved in Analysis opens the Lattices 2D Plane with one note.

use egui::Key;
use egui_kittest::kittest::Queryable;

use super::test_util::{open_deck, press};
use super::tests_nav::start;
use super::{ANALYSIS_LAB_POSTPONED_NOTE, ANALYSIS_POSTPONED_NOTE, Realm};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::session;

#[test]
fn analysis_hidden_everywhere() {
    let mut h = start();
    for mode in AppMode::SHOWN {
        h.state_mut().switch_mode(mode);
        h.run_steps(2);
        assert!(!h.state().deck.open);
        assert!(h.query_all_by_label_contains("Analysis").next().is_none(), "{mode:?}, deck closed");
        open_deck(&mut h);
        for r in Realm::SHOWN {
            h.state_mut().deck_select_realm(r);
            h.run_steps(2);
            let leaks = h.query_all_by_label_contains("Analysis").count();
            assert_eq!(leaks, 0, "{mode:?}, deck open at {r:?}: a visible label names Analysis");
        }
        press(&mut h, Key::Escape);
        assert!(!h.state().deck.open);
    }
    let named = h.state_mut().visible_analysis_labels();
    assert!(named.is_empty(), "label sources naming Analysis: {named:?}");

    // key 6 (the hidden Analysis Lab) leaves the deck where it was
    open_deck(&mut h);
    let before = (h.state().deck.realm, h.state().deck.realm_switches);
    press(&mut h, Key::Num6);
    assert_eq!((h.state().deck.realm, h.state().deck.realm_switches), before, "key 6 does nothing");
    assert!(h.state().deck.open);
    press(&mut h, Key::Escape);
}

#[test]
fn sessions_saved_in_analysis_restore_to_lattices_with_a_note() {
    let saved = [
        r#"{"schema_version": 1, "active_tab": "Analysis"}"#,
        r#"{"schema_version": 2, "active_tab": "Analysis", "active_mode": "Analysis"}"#,
        r#"{"schema_version": 3, "active_tab": "Analysis", "active_mode": "Analysis", "deck": {"realm": "AnalysisLab"}}"#,
    ];
    for raw in saved {
        let st = session::parse_session(raw).unwrap().0;
        let mut a = PapertillingApp::new(false, String::from("test"), String::from("test"));
        let notes = a.apply_session(&st, None);
        assert_eq!(notes, vec![String::from(ANALYSIS_POSTPONED_NOTE)], "{raw}");
        assert_eq!((a.active_mode, a.lattices_view, a.active_tab), (AppMode::Lattices, LatticesView::Plane2D, NavTab::Gallery), "{raw}");
        assert_eq!(a.deck.realm, Realm::Tilings, "{raw}");
        assert!(!a.deck.open && !a.deck.resume_saved_realm, "{raw}");
    }

    // Regression (WP-A0 gap): a v3 session in Defects with deck.realm = "AnalysisLab" stays in Defects and resets the deck to Forge.
    let def_raw = r#"{"schema_version": 3, "active_mode": "Defects", "defects_view": "Catalogue2D", "active_tab": "Defects", "deck": {"realm": "AnalysisLab"}}"#;
    let def_st = session::parse_session(def_raw).unwrap().0;
    let mut d = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let def_notes = d.apply_session(&def_st, None);
    assert_eq!(def_notes, vec![String::from(ANALYSIS_LAB_POSTPONED_NOTE)]);
    assert_eq!((d.active_mode, d.defects_view, d.active_tab), (AppMode::Defects, DefectsView::Catalogue2D, NavTab::Defects));
    assert_eq!(d.deck.realm, Realm::Forge);
    assert!(!d.deck.open && !d.deck.resume_saved_realm);
}

#[test]
fn hidden_realm_is_never_entered() {
    // Q / E and the number keys walk the six shown realms; opening or selecting the hidden one lands on Tilings
    assert_eq!(Realm::from_number(6), None);
    assert_eq!(Realm::Forge.next(), Realm::PresetsIo);
    assert_eq!(Realm::PresetsIo.prev(), Realm::Forge);
    assert_eq!((Realm::AnalysisLab.prev(), Realm::AnalysisLab.next()), (Realm::Forge, Realm::PresetsIo));
    let ctx = egui::Context::default();
    let mut a = PapertillingApp::new(false, String::from("test"), String::from("test"));
    a.deck_open(Realm::AnalysisLab, &ctx);
    assert_eq!(a.deck.realm, Realm::Tilings);
    a.deck_select_realm(Realm::AnalysisLab);
    assert_eq!(a.deck.realm, Realm::Tilings);
}
