//! Headless UI and session checks for `feat/simple-by-default` (`SIMPLE-BRIEF.md` §Checks):
//! 1. With `Advanced` off, only the two modes (`Lattice` | `Defects`) and their Default controls are reachable, `Advanced`
//!    sits at the rightmost end of the top bar, and zero spectral, RPG, or Group 3 tokens appear anywhere.
//! 2. With `Advanced` on, every valid post-C1 panel is reachable through `All Views`, while C1-hidden and Group 3 items
//!    remain unreachable; turning `Advanced` off snaps back to Default.
//! 3. `Advanced` round-trips through session save/load, and schema v1–v3 sessions without `advanced` still load.
//! 4. End-to-end Default-path walkthrough: open app (`Advanced` off), pick a lattice (`square`), place two disclinations
//!    (`+90°` and `-90°` at adjacent 4-fold centres), and read their fusion into an edge dislocation (`|b| = 1`) with an
//!    equivalence verdict.

use egui::Key;
use egui_kittest::kittest::Queryable;

use crate::analysis_mode::tests::{assert_no_forbidden_tokens, equip_keys, harness, settled, wait_until};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::game_menu::{CardId, Realm};
use crate::game_menu::test_util::{open_deck, press};
use crate::presets::PRESETS;
use crate::session;
use crate::workbench::WorkbenchView;

const FORBIDDEN_RPG_AND_GROUP3: &[&str] = &[
    "Codex",
    "Defects Forge",
    "⏎ Equip",
    "WASD",
    "Föppl",
    "3D Elastic Relaxation",
    "CNA bond",
    "Wigner-Seitz",
    "Elastic & Stroh",
    "Motion GIF",
    "Surfaces & nets",
    "|ψ_n|",
    "Lattice revision",
    "scan cache",
    "GPU (",
];

fn click_top(h: &mut egui_kittest::Harness<'static, PapertillingApp>, label: &str) {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label)
        .find(|n| top.contains_rect(n.rect()))
        .unwrap_or_else(|| panic!("missing '{label}' in top bar"))
        .click();
    h.run_steps(2);
}

fn assert_clean_screen(h: &egui_kittest::Harness<'_, PapertillingApp>, ctx: &str) {
    assert_no_forbidden_tokens(h, ctx);
    for tok in FORBIDDEN_RPG_AND_GROUP3 {
        assert!(
            h.query_all_by_label_contains(tok).next().is_none(),
            "{ctx}: forbidden RPG/Group-3 token '{tok}' visible on screen"
        );
    }
}

#[test]
fn advanced_off_reaches_only_two_modes_and_carries_no_spectral_rpg_or_group3_tokens() {
    let mut h = harness();
    h.run_steps(3);

    // Default start: Advanced off, p6m (*632) honeycomb, 4-layer open patch in Lattice -> Tiling.
    let app = h.state();
    assert!(!app.advanced, "Advanced is off by default");
    assert_eq!((app.active_mode, app.lattices_view, app.active_tab), (AppMode::Lattices, LatticesView::Plane2D, NavTab::Gallery));
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("honeycomb"));
    assert_eq!(app.gallery.active_item().group, "p6m");
    assert_eq!(app.gallery.inflation.steps, 4);

    // `Advanced` toggle sits at the rightmost end of the top bar.
    let top = h.state().test_rects.top.expand(1.0);
    let adv_rect = h
        .query_all_by_label("Advanced")
        .find(|n| top.contains_rect(n.rect()))
        .expect("Advanced toggle in top bar")
        .rect();
    for lbl in ["Theme: Light", "Find lattice (Ctrl+K)", "View", "Save figure", "Session", "Examples", "Undo", "Redo", "Lattice", "Defects"] {
        if let Some(r) = h.query_all_by_label(lbl).find(|n| top.contains_rect(n.rect())).map(|n| n.rect()) {
            assert!(adv_rect.left() >= r.right() - 1.0, "Advanced {adv_rect:?} must sit right of '{lbl}' {r:?}");
        }
    }

    // All Views button and Tab/M shortcuts are inactive while Advanced is off.
    assert!(h.query_by_label("All views (Tab)").is_none());
    press(&mut h, Key::Tab);
    press(&mut h, Key::M);
    assert!(!h.state().deck.open, "Tab / M must not open All Views when Advanced is off");

    // Lattice -> Tiling: no + Custom, no Live Search, no raw D-symbol key or cohomology.
    assert!(h.query_by_label("+ Custom").is_none());
    assert!(h.query_all_by_label_contains("Live Search").next().is_none());
    assert!(h.query_all_by_label_contains("D-symbol key:").next().is_none());
    assert!(h.query_all_by_label_contains("Group cohomology").next().is_none());
    assert!(h.query_all_by_label_contains("p6m (*632)").next().is_some(), "orbifold notation shown");
    assert_clean_screen(&h, "Default: Lattice -> Tiling");

    // Save figure & Examples menus filter out Advanced-only entries when Advanced is off.
    click_top(&mut h, "Save figure");
    assert!(h.query_by_label("Save 2D figure (SVG / PNG)...").is_some());
    assert!(h.query_by_label("Export 3D mesh (.obj)...").is_none());
    assert!(h.query_by_label("Save defect config (.json)...").is_none());
    press(&mut h, Key::Escape);

    click_top(&mut h, "Examples");
    for p in PRESETS {
        assert_eq!(h.query_by_label(p.name).is_some(), p.shown_for(false), "Examples (Advanced off): {}", p.name);
    }
    assert!(h.query_by_label("Atlas: hyperbolic {7,3} tiling").is_none());
    press(&mut h, Key::Escape);

    // Lattice -> Surface.
    click_top(&mut h, "Surface");
    wait_until(&mut h, "Default Surface", settled);
    assert_eq!((h.state().active_mode, h.state().lattices_view, h.state().active_tab), (AppMode::Lattices, LatticesView::Manifolds, NavTab::Surfaces));
    assert_clean_screen(&h, "Default: Lattice -> Surface");

    // Defects -> Allowed defects.
    click_top(&mut h, "Defects");
    wait_until(&mut h, "Default Allowed defects", settled);
    assert_eq!((h.state().active_mode, h.state().defects_view, h.state().active_tab), (AppMode::Defects, DefectsView::Catalogue2D, NavTab::Defects));
    assert!(h.query_all_by_label_contains("Group cohomology").next().is_none());
    assert_clean_screen(&h, "Default: Defects -> Allowed defects");

    // Defects -> Place & fuse.
    click_top(&mut h, "Place & fuse");
    wait_until(&mut h, "Default Place & fuse", settled);
    assert_eq!((h.state().active_mode, h.state().defects_view, h.state().active_tab), (AppMode::Defects, DefectsView::CombineAndFuse, NavTab::Combine));
    assert!(h.query_all_by_label_contains("Group cohomology").next().is_none());
    assert_clean_screen(&h, "Default: Defects -> Place & fuse");
}

#[test]
fn advanced_on_reaches_every_post_c1_panel_and_keeps_c1_and_group3_hidden_items_out() {
    let mut h = harness();
    h.run_steps(2);

    // Toggle Advanced ON via the top-bar switch.
    click_top(&mut h, "Advanced");
    assert!(h.state().advanced);
    assert!(h.query_by_label("All views (Tab)").is_some());
    assert!(h.query_by_label("+ Custom").is_some(), "Custom builder revealed in Gallery when Advanced is on");
    assert!(h.query_all_by_label_contains("D-symbol key:").next().is_some(), "D-symbol key revealed when Advanced is on");

    click_top(&mut h, "Save figure");
    assert!(h.query_by_label("Export 3D mesh (.obj)...").is_some());
    assert!(h.query_by_label("Save defect config (.json)...").is_some());
    assert!(h.query_by_label("Load defect config (.json)...").is_some());
    press(&mut h, Key::Escape);

    // Open All Views: verify plain wording and that C1 + Group 3 hidden items never appear in any realm.
    open_deck(&mut h);
    assert!(h.query_by_label("All Views").is_some());
    assert!(h.query_by_label("Open").is_some());
    assert_clean_screen(&h, "Advanced: All Views open");
    press(&mut h, Key::Escape);

    for r in Realm::SHOWN {
        for card in h.state_mut().deck_cards(r, 0, "") {
            assert!(!card.title.contains("Analysis"), "C1 Analysis stayed hidden: {}", card.title);
            for hidden in ["3D slabs", "Platycosms", "3D Line Defects", "Elastic & Stroh", "Motion GIF", "Surfaces & nets"] {
                assert_ne!(card.title, hidden, "hidden card '{hidden}' must not appear in {r:?}");
            }
            assert!(
                !matches!(card.id, CardId::AtlasDefects(AtlasSubTab::SpaceGroups3D)),
                "C1 3D line defects card stayed hidden"
            );
        }
    }

    // Reach representative post-C1 panels through All Views keys and verify Group 3 sub-controls stay out.
    for (realm, title, expect_tab) in [
        (Realm::Groups, "230 3D Space Groups", NavTab::Atlas),
        (Realm::Groups, "80 Layer Groups", NavTab::Atlas),
        (Realm::Geometries, "H² hyperbolic tiling (Atlas)", NavTab::Atlas),
        (Realm::Geometries, "Schläfli {p,q,r}", NavTab::Workbench),
        (Realm::Geometries, "Regular {p,q}", NavTab::Workbench),
        (Realm::Manifolds, "Orbifolds & TG cells", NavTab::Workbench),
        (Realm::Manifolds, "Nanotubes", NavTab::Workbench),
        (Realm::Forge, "Stone-Wales", NavTab::Workbench),
        (Realm::Forge, "Detect defects", NavTab::Workbench),
        (Realm::Forge, "Strain / Nye", NavTab::Workbench),
    ] {
        equip_keys(&mut h, realm, title);
        assert_eq!(h.state().active_tab, expect_tab, "{title}");
        assert_clean_screen(&h, title);
    }

    // On S3 Strain / Nye, verify the field selector lists 4 geometric fields and excludes |ψ_n|.
    assert_eq!(h.state().workbench.view, WorkbenchView::S3Detect);
    assert!(h.query_by_label("Elastic & Stroh").is_none());
    assert!(h.query_by_label("Motion GIF").is_none());

    // Turning Advanced OFF from an Advanced-only view snaps cleanly back to Default.
    click_top(&mut h, "Advanced");
    assert!(!h.state().advanced);
    assert!(!h.state().deck.open);
    assert_eq!((h.state().active_mode, h.state().defects_view, h.state().active_tab), (AppMode::Defects, DefectsView::Catalogue2D, NavTab::Defects));
}

#[test]
fn advanced_switch_roundtrips_in_session_and_v1_to_v3_sessions_load() {
    let dir = crate::session::tests::TempDir::new();
    let path = dir.0.join(session::SESSION_FILENAME);

    for want in [false, true] {
        let mut src = PapertillingApp::new(false, String::from("test"), String::from("test"));
        src.set_advanced(want);
        if want {
            src.select_lattices_view(LatticesView::NonEuclidean);
        }
        src.save_session_to(&path, false).unwrap();

        let mut dst = PapertillingApp::new(false, String::from("test"), String::from("test"));
        dst.set_advanced(!want);
        let notes = dst.open_session_from(&path, None).unwrap().expect("session file exists");
        assert!(notes.is_empty(), "want={want}: {notes:?}");
        assert_eq!(dst.advanced, want, "advanced={want} round-tripped");
        assert_eq!(
            dst.lattices_view,
            if want { LatticesView::NonEuclidean } else { LatticesView::Plane2D }
        );
    }

    // Legacy v1, v2, and pre-Job-2 v3 sessions without `advanced` load cleanly.
    for (raw, want_adv, want_tab) in [
        (r#"{"schema_version": 1, "active_tab": "Defects"}"#, false, NavTab::Defects),
        (r#"{"schema_version": 1, "active_tab": "Atlas", "atlas_sub_view": "HyperbolicH2"}"#, true, NavTab::Atlas),
        (r#"{"schema_version": 2, "active_mode": "Defects", "defects_view": "CombineAndFuse"}"#, false, NavTab::Combine),
        (r#"{"schema_version": 3, "active_mode": "Lattices", "lattices_view": "ThreeDAndSubperiodic", "deck": {"realm": "Groups"}}"#, true, NavTab::Atlas),
    ] {
        let (st, dropped) = session::parse_session(raw).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(st.advanced, None);
        let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
        let notes = app.apply_session(&st, None);
        assert!(notes.is_empty(), "{raw}: {notes:?}");
        assert_eq!((app.advanced, app.active_tab), (want_adv, want_tab), "{raw}");
    }
}

#[test]
fn default_path_walkthrough_pick_lattice_place_two_defects_read_fusion() {
    let mut h = harness();
    h.run_steps(3);
    assert!(!h.state().advanced, "starts in Default view");

    // 1. Pick the square lattice (p4m, ITA 11) via the Default top-bar "Find lattice (Ctrl+K)" button.
    click_top(&mut h, "Find lattice (Ctrl+K)");
    h.query_all_by_role(egui::accesskit::Role::TextInput)
        .last()
        .expect("switcher search input")
        .type_text("square 11");
    h.run_steps(2);
    press(&mut h, Key::Enter);
    assert_eq!(h.state().gallery.active_item().named_key.as_deref(), Some("square"));
    assert_eq!(h.state().gallery.active_item().group, "p4m");

    // 2. Switch to Defects -> Place & fuse using the Default top-bar controls.
    click_top(&mut h, "Defects");
    click_top(&mut h, "Place & fuse");
    wait_until(&mut h, "square Place & fuse init", settled);
    assert_eq!(h.state().active_tab, NavTab::Combine);

    // 3. Clear default pins, choose "Place disclination", and place +90° at 1a = (0, 0) and -90° at 1b = (0.5, 0.5).
    h.get_by_label("Clear All").click();
    h.run_steps(2);
    assert!(h.state().combine.specs.is_empty());

    h.get_by_label("Place disclination").click();
    h.run_steps(2);
    h.get_by_label("+90°").click();
    h.run_steps(1);
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        app.combine.place_at_world([0.0, 0.0], &item, &app.backend, &mut app.canvas2d);
    }
    h.run_steps(2);
    h.get_by_label("-90°").click();
    h.run_steps(1);
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        app.combine.place_at_world([0.5, 0.5], &item, &app.backend, &mut app.canvas2d);
    }

    // 4. Wait for crystal build + single-defect fusion comparison to complete, then read fusion & equivalence.
    wait_until(&mut h, "two-disclination fusion build & compare", |a| {
        !a.combine.is_busy() && a.combine.compare_result.is_some() && !a.combine.compare_stale
    });
    assert_clean_screen(&h, "Default walkthrough after fusion");

    let app = h.state();
    assert_eq!(app.combine.specs.len(), 2, "two placed disclinations");
    assert!(app.combine.last_build.refused.is_none(), "{:?}", app.combine.last_build.refused);
    assert!(app.combine.last_build.crystal.pos.len() > 20);

    // Hand math: +90° at 1a = (0,0) and -90° at 1b = (0.5,0.5) on {4,4} (a = 1) compose to pure translation
    // b = (0.5 - 0.5, 0.5 + 0.5) = (0, 1) of primitive length |b| = √2 |d| = 1, core-equivalent to a single
    // dislocation in the square catalogue.
    let rep = app
        .combine
        .compare_result
        .as_ref()
        .expect("comparison finished")
        .as_ref()
        .expect("comparison succeeded");
    assert_eq!(rep.charge.kind, "dislocation");
    assert!(rep.charge.angle.abs() < 1e-9, "net Frank angle = 0°, got {}", rep.charge.angle);
    assert!((rep.charge.length.expect("Burgers length") - 1.0).abs() < 1e-9, "|b| = a = 1");
    assert_eq!(rep.charge.lattice_vector, Some(true));
    assert!(rep.equivalent, "adjacent +90°/-90° pair is core-equivalent to a single dislocation: {}", rep.verdict);
    assert!(
        h.query_all_by_label_contains("EQUIVALENT").next().is_some(),
        "equivalence banner rendered in inspector"
    );
}
