//! Session tests S1-S7 for the deck field and schema 3 (M1 plan section 6.3 with the v3 decision; written by WP-D).
//!
//! State level only (no harness): sessions are parsed with `parse_session` and applied with `apply_session`, the
//! same path the app takes at start. Files go to a temp folder, never the OS config folder.

use super::Realm;
use crate::analysis_mode::AnalysisTarget;
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::backend::SurfaceTargetUi;
use crate::session::{self, SESSION_SCHEMA_VERSION, SessionError, SessionState};
use crate::theme::StudioTheme;
use crate::workbench::WorkbenchView;

const DECK_NOTE: &str = "unknown deck realm (kept current)";

fn app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn parse(raw: &str) -> SessionState {
    session::parse_session(raw).unwrap().0
}

fn has_deck_note(notes: &[String]) -> bool {
    notes.iter().any(|n| n.contains("deck"))
}

#[test]
fn s1_v1_without_deck_loads_closed_at_the_realm_of_its_view() {
    let st = parse(r#"{"schema_version": 1, "active_tab": "Surfaces"}"#);
    assert!(st.deck.is_none());
    let mut a = app();
    let notes = a.apply_session(&st, None);
    assert!(notes.is_empty(), "{notes:?}");
    assert!(!has_deck_note(&notes));
    assert_eq!((a.active_mode, a.lattices_view, a.active_tab), (AppMode::Lattices, LatticesView::Manifolds, NavTab::Surfaces));
    assert_eq!(a.current_realm(), Realm::Manifolds);
    assert_eq!(a.deck.realm, Realm::Manifolds);
    assert!(!a.deck.open);
}

#[test]
fn s2_v2_without_deck_keeps_todays_navigation() {
    // v2 shapes as written by the schema-2 app (cf. app/session_tests.rs): mode + views, no deck field
    let combine = r#"{"schema_version": 2, "active_tab": "Combine", "theme": "Studio",
        "active_mode": "Defects", "lattices_view": "Plane2D", "defects_view": "CombineAndFuse"}"#;
    let mut a = app();
    let notes = a.apply_session(&parse(combine), None);
    assert!(!has_deck_note(&notes), "{notes:?}");
    assert_eq!((a.active_mode, a.defects_view, a.active_tab), (AppMode::Defects, DefectsView::CombineAndFuse, NavTab::Combine));
    assert_eq!(a.theme, StudioTheme::Studio);
    assert_eq!(a.deck.realm, a.current_realm());
    assert_eq!(a.deck.realm, Realm::Forge);
    assert!(!a.deck.open);

    let analysis = r#"{"schema_version": 2, "active_tab": "Analysis", "active_mode": "Analysis",
        "analysis_target": "PlaneLattice", "analysis_view": "Sheet3D"}"#;
    let mut b = app();
    let notes = b.apply_session(&parse(analysis), None);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((b.active_mode, b.active_tab, b.analysis.target), (AppMode::Analysis, NavTab::Analysis, AnalysisTarget::PlaneLattice));
    assert_eq!(b.deck.realm, Realm::AnalysisLab);
    assert_eq!(b.deck.realm, b.current_realm());
}

#[test]
fn s3_deck_realm_round_trips_and_the_deck_loads_closed() {
    let mut a = app();
    a.deck.realm = Realm::Forge;
    a.deck.open = true;
    let st = a.capture_session();
    assert_eq!((st.schema_version, SESSION_SCHEMA_VERSION), (3, 3));
    let raw = st.to_json_pretty().unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["schema_version"], serde_json::json!(3));
    assert_eq!(v["deck"], serde_json::json!({ "realm": "Forge" }), "{raw}");
    assert!(v["deck"].get("open").is_none(), "the open flag is never saved");

    let mut b = app();
    let notes = b.apply_session(&parse(&raw), None);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(b.deck.realm, Realm::Forge, "the saved realm wins over the realm of the restored view");
    assert!(!b.deck.open);
    assert_eq!(b.capture_session().deck, Some(serde_json::json!({ "realm": "Forge" })));
}

#[test]
fn s4_bad_deck_value_notes_and_falls_back_without_losing_other_fields() {
    for deck in [r#"{"realm": "Bogus"}"#, "7", r#"{"realm": 3}"#] {
        let raw = format!(
            r#"{{"schema_version": 3, "theme": "Studio", "active_tab": "Surfaces", "active_mode": "Lattices",
                "lattices_view": "Manifolds", "surfaces": {{"target": "Torus"}}, "deck": {deck}}}"#
        );
        let (st, dropped) = session::parse_session(&raw).unwrap();
        assert_eq!(dropped, 0, "{deck}: any JSON value is kept for the deck field");
        let mut a = app();
        let notes = a.apply_session(&st, None);
        assert_eq!(notes, [DECK_NOTE], "{deck}");
        assert_eq!(a.deck.realm, a.current_realm(), "{deck}");
        assert_eq!(a.deck.realm, Realm::Manifolds, "{deck}");
        assert!(!a.deck.open);
        assert_eq!(a.theme, StudioTheme::Studio, "{deck}");
        assert_eq!((a.lattices_view, a.active_tab), (LatticesView::Manifolds, NavTab::Surfaces), "{deck}");
        assert_eq!(a.surfaces.target, SurfaceTargetUi::Torus, "{deck}");
    }
}

#[test]
fn s5_curved2d_orbifolds_restores_into_the_manifolds_realm() {
    let raw = r#"{"schema_version": 2, "active_mode": "Lattices", "lattices_view": "Curved2D",
        "workbench": {"view": "S5Curved2d", "s5": {"sub": "Orbifolds"}}}"#;
    let mut a = app();
    let notes = a.apply_session(&parse(raw), None);
    assert!(!has_deck_note(&notes), "{notes:?}");
    assert_eq!(a.active_tab, NavTab::Workbench);
    assert_eq!(a.workbench.view, WorkbenchView::S5Curved2d);
    assert_eq!(a.workbench.sub_index(WorkbenchView::S5Curved2d), 4);
    assert_eq!(a.current_realm(), Realm::Manifolds);
    assert_eq!(a.deck.realm, Realm::Manifolds);
    assert_eq!(a.breadcrumb().mode.text, "Lattices: {p,q} on S² / E² / H²");
}

#[test]
fn s6_v1_to_v3_load_and_v4_is_refused_and_never_overwritten() {
    for v in 1..=3 {
        let raw = format!(r#"{{"schema_version": {v}, "theme": "Studio"}}"#);
        assert!(session::parse_session(&raw).is_ok(), "schema {v} is read");
    }
    for v in [0_u32, 4, 99] {
        let raw = format!(r#"{{"schema_version": {v}}}"#);
        assert_eq!(session::parse_session(&raw).unwrap_err(), SessionError::Unsupported(v));
    }

    let dir = crate::session::tests::TempDir::new();
    let path = dir.0.join(session::SESSION_FILENAME);

    // a v3 file with a deck realm restores at start, and the exit save writes v3 back
    let mut src = app();
    src.deck.realm = Realm::PresetsIo;
    session::save_session_file(&path, &src.capture_session()).unwrap();
    let mut a = app();
    a.session.path = Some(path.clone());
    a.load_session_at_start(&egui::Context::default());
    assert!(a.session.exit_save_allowed);
    assert_eq!(a.deck.realm, Realm::PresetsIo);
    assert!(!a.deck.open);
    a.deck.realm = Realm::Groups;
    assert!(a.save_session_on_exit());
    let (saved, _) = session::load_session_file(&path).unwrap().unwrap();
    assert_eq!(saved.schema_version, 3);
    assert_eq!(saved.deck, Some(serde_json::json!({ "realm": "Groups" })));

    // a v4 file (a newer app) is refused, nothing is applied, and the exit save never replaces it
    let v4 = r#"{"schema_version": 4, "theme": "Studio", "deck": {"realm": "Forge"}}"#;
    std::fs::write(&path, v4).unwrap();
    let mut f = app();
    f.session.path = Some(path.clone());
    f.load_session_at_start(&egui::Context::default());
    assert!(f.export_status.starts_with("Session not restored"), "{}", f.export_status);
    assert!(!f.session.exit_save_allowed);
    assert_eq!((f.theme, f.deck.realm), (StudioTheme::Paper, Realm::default()));
    assert!(!f.save_session_on_exit());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), v4);
}

#[test]
fn s7_v1_workbench_tab_still_maps_to_the_2d_plane() {
    let mut a = app();
    let notes = a.apply_session(&parse(r#"{"schema_version": 1, "active_tab": "Workbench"}"#), None);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((a.active_mode, a.lattices_view, a.active_tab), (AppMode::Lattices, LatticesView::Plane2D, NavTab::Gallery));
    assert_eq!(a.deck.realm, Realm::Tilings);
    assert!(!a.deck.open);
}
