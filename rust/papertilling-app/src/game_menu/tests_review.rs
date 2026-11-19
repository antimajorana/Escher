//! Regression tests for the M1 review findings: HUD widths and density, deck keys with a pending rename or an
//! open menu, equip routing build counts, the HUD breadcrumb and thumbnail, cached verdict failures, the saved
//! deck realm and the realm-native session notes. Expectations are hand-checked (labels, counts, group symbols).

use egui::Key;
use egui_kittest::kittest::Queryable;

use super::hud_bar::HudDensity;
use super::realms::CardId;
use super::test_util::{H, equip_state, goto_defects_view, goto_lattices_view, open_deck, press};
use super::tests_nav::{click_in_hud, start};
use super::Realm;
use crate::app::{AppMode, DefectsView, LatticesView, NavTab};
use crate::backend::{CatalogueRuleUi, SurfaceTargetUi};
use crate::session;
use crate::theme::StudioTheme;

const CODEX: &str = "❖ Codex (Tab)";
const RENAME: &str = "✎ Rename";
/// Every control the right cluster can put on the row, at any density.
const RIGHT: [&str; 14] = [
    "Theme: Paper", "Theme: Studio", "Search (Ctrl+K)", "Search", "Menu", "View", "Export", "Session", "Presets", "Redo", "Undo",
    "Split", "3D", "2D",
];

/// Rect of the node labelled `label` inside the HUD row. AccessKit rects are the widget rects, not clipped: a widget
/// pushed under the right cluster still reports its rect there, so the tests compare right edges with the left edge
/// of the right cluster (a full width alone only shows the label was not truncated).
fn hud_rect(h: &H, label: &str) -> Option<egui::Rect> {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label).find(|n| top.contains_rect(n.rect())).map(|n| n.rect())
}

/// Left edge of the right cluster.
fn right_cluster_left(h: &H) -> f32 {
    RIGHT.iter().filter_map(|l| hud_rect(h, l)).map(|r| r.left()).fold(f32::INFINITY, f32::min)
}

fn set_width(h: &mut H, w: f32) {
    h.set_size(egui::vec2(w, 600.0));
    h.run_steps(4);
}

/// Findings 1 and 2: from 1280 px down to the window minimum, in the view with the widest right cluster
/// (Combine: Undo / Redo and the view toggles), the Codex button and the rename controls keep their whole width,
/// left of the right cluster, and the rename controls are reachable with a real pointer click.
#[test]
fn hud_keeps_codex_and_rename_whole() {
    let mut h = start();
    goto_lattices_view(&mut h, LatticesView::Plane2D);
    set_width(&mut h, 1280.0);
    let codex_w = hud_rect(&h, CODEX).expect("Codex at 1280").width();
    let rename_w = hud_rect(&h, RENAME).expect("Rename at 1280").width();
    assert!(codex_w > 60.0 && rename_w > 40.0, "natural widths {codex_w} / {rename_w}");
    let min_w = super::HUD_MIN_WINDOW_WIDTH;
    for (plane, view) in [(true, None), (false, Some(DefectsView::CombineAndFuse))] {
        match view {
            Some(v) => goto_defects_view(&mut h, v),
            None => goto_lattices_view(&mut h, LatticesView::Plane2D),
        }
        for w in [1280.0, 1240.0, 1100.0, 1000.0, 960.0, 900.0, 800.0, min_w] {
            set_width(&mut h, w);
            let what = format!("plane={plane} w={w}");
            let want = if w >= 1240.0 {
                HudDensity::Full
            } else if w >= 960.0 {
                HudDensity::Compact
            } else {
                HudDensity::Narrow
            };
            assert_eq!(h.state().deck.hud_density, want, "{what}");
            let right = right_cluster_left(&h);
            let codex = hud_rect(&h, CODEX).unwrap_or_else(|| panic!("{what}: no Codex"));
            let rename = hud_rect(&h, RENAME).unwrap_or_else(|| panic!("{what}: no Rename"));
            assert!((codex.width() - codex_w).abs() < 0.5, "{what}: Codex cut to {codex:?}");
            assert!((rename.width() - rename_w).abs() < 0.5, "{what}: Rename cut to {rename:?}");
            assert!(codex.left() >= 0.0 && codex.right() <= right, "{what}: Codex {codex:?}, right cluster {right}");
            assert!(rename.right() <= right + 0.5, "{what}: Rename {rename:?} under the right cluster at {right}");
            assert!(right >= rename.right() && right < w, "{what}: right cluster starts at {right}");
        }
    }
    // in Combine, at the narrowest width of each density: a pointer click starts the rename; Save and Cancel stay
    // whole, left of the right cluster
    for w in [1240.0, 960.0, min_w] {
        set_width(&mut h, w);
        click_in_hud(&mut h, RENAME);
        h.run_steps(2);
        assert!(h.state().gallery.renaming, "a real click on Rename at {w} px starts the rename");
        let right = right_cluster_left(&h);
        for l in ["Save", "Cancel"] {
            let r = hud_rect(&h, l).unwrap_or_else(|| panic!("w={w}: no {l} while renaming"));
            assert!(r.width() > 20.0 && r.right() <= right + 0.5, "w={w}: {l} {r:?} vs right cluster {right}");
        }
        press(&mut h, Key::Escape);
        assert!(!h.state().gallery.renaming);
    }
    // the folded controls are in Menu
    click_in_hud(&mut h, "Menu");
    for l in ["View", "Export", "Session", "Presets", "Undo", "Redo", "Layout"] {
        assert!(h.query_by_label(l).is_some(), "Menu holds {l}");
    }
}

/// Finding 3: opening the deck with the mouse while a rename is pending keeps the typed name, and the deck keys
/// work (the field lost focus); the first Esc closes the deck.
#[test]
fn deck_keys_work_after_mouse_open_during_rename() {
    let mut h = start();
    click_in_hud(&mut h, RENAME);
    assert!(h.state().gallery.renaming && h.state().gallery.rename_holds_keys());
    h.state_mut().gallery.rename_buf = String::from("Zed");
    click_in_hud(&mut h, CODEX);
    assert!(h.state().deck.open, "Codex opens the deck");
    assert!(!h.state().gallery.rename_holds_keys(), "the rename field gave up focus");
    let n = h.state().deck.n_cards;
    assert!(n >= 2, "{n} cards");
    h.state_mut().deck.set_cur(0);
    press(&mut h, Key::ArrowRight);
    assert_eq!(h.state().deck.cur(), 1, "ArrowRight moves the deck cursor");
    press(&mut h, Key::Num2);
    assert_eq!(h.state().deck.realm, Realm::Groups, "2 switches the realm");
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open, "the first Esc closes the deck");
    let g = &h.state().gallery;
    assert!(g.renaming && g.rename_buf == "Zed", "the typed name is still pending");
}

/// Finding 4: with the deck and a HUD menu open, Tab never moves focus (into the panels under the deck); Esc
/// closes the menu first, then the deck.
#[test]
fn tab_in_open_menu_stays_out_of_hidden_panels() {
    let mut h = start();
    open_deck(&mut h);
    click_in_hud(&mut h, "View");
    assert!(egui::Popup::is_any_open(&h.ctx), "View menu open");
    let focus0 = h.ctx.memory(|m| m.focused());
    for _ in 0..6 {
        press(&mut h, Key::Tab);
        assert_eq!(h.ctx.memory(|m| m.focused()), focus0, "Tab did not move focus");
        assert!(!h.ctx.text_edit_focused());
    }
    assert!(h.state().deck.open && egui::Popup::is_any_open(&h.ctx));
    press(&mut h, Key::Escape);
    assert!(!egui::Popup::is_any_open(&h.ctx), "Esc closes the menu");
    assert!(h.state().deck.open, "the deck stays open");
}

/// Findings 5 / 12 and 11: one synchronous surface build per Surface equip, none for leaving Manifolds.
#[test]
fn equip_routes_build_each_surface_once() {
    let mut h = start();
    goto_lattices_view(&mut h, LatticesView::Plane2D);
    let s0 = h.state().surfaces.syncs;
    equip_state(&mut h, CardId::Surface(SurfaceTargetUi::Torus));
    assert_eq!(h.state().active_tab, NavTab::Surfaces);
    assert_eq!(h.state().surfaces.syncs, s0 + 1, "from the 2D Plane: one build");
    equip_state(&mut h, CardId::Surface(SurfaceTargetUi::Cylinder));
    assert_eq!(h.state().surfaces.target, SurfaceTargetUi::Cylinder);
    assert_eq!(h.state().surfaces.syncs, s0 + 2, "already on Manifolds: one build of the new target");

    // another stored lattice (first Named card that is not the selected one)
    let cards = h.state_mut().deck_cards(Realm::Tilings, 1, "");
    let base = h.state().gallery.base_item().alias_key.clone();
    let other = |h: &H, skip: usize| {
        cards
            .iter()
            .filter_map(|c| match c.id {
                CardId::Lattice(r) => h.state().lattice_ref_item(r).filter(|it| it.alias_key != base).map(|_| r),
                _ => None,
            })
            .nth(skip)
            .expect("a second Named lattice")
    };
    // Lattice equip on Manifolds (lattice-driven, the view stays): exactly one rebuild for the new lattice
    let r1 = other(&h, 0);
    equip_state(&mut h, CardId::Lattice(r1));
    assert_eq!(h.state().active_tab, NavTab::Surfaces);
    assert_eq!(h.state().surfaces.syncs, s0 + 3, "Lattice equip on Manifolds rebuilds once");
    // Open in Defects Forge from Manifolds: no surface build for the view being left
    let r2 = other(&h, 1);
    let want = h.state().lattice_ref_item(r2).unwrap().alias_key.clone();
    h.state_mut().deck_open_in_forge(r2);
    h.run_steps(2);
    let a = h.state();
    assert_eq!(a.surfaces.syncs, s0 + 3, "leaving Manifolds does not rebuild the surface");
    assert_eq!((a.active_mode, a.active_tab), (AppMode::Defects, NavTab::Defects));
    assert_eq!(a.gallery.base_item().alias_key, want);
}

/// Finding 6: a chiral lattice shows `±` once in the HUD (the rename controls' one, with the detailed hover).
#[test]
fn chiral_lattice_shows_one_plus_minus() {
    let mut h = start();
    let i = crate::presets::PRESETS.iter().position(|p| p.name == "Chiral kagome (p6, no mirrors)").unwrap();
    h.state_mut().apply_preset(i).unwrap();
    goto_lattices_view(&mut h, LatticesView::Plane2D);
    assert!(h.state().gallery.readout.chiral, "chiral kagome is chiral");
    let bc = h.state().breadcrumb();
    let seg = bc.lattice.expect("lattice segment on the 2D Plane");
    assert!(!seg.text.contains('±') && seg.hover.contains("chiral (±)"), "{} | {}", seg.text, seg.hover);
    let top = h.state().test_rects.top.expand(1.0);
    let n = h.query_all_by_label("±").filter(|n| top.contains_rect(n.rect())).count();
    assert_eq!(n, 1, "one ± in the HUD");
}

/// Finding 7: views that do not show the Gallery lattice show no rename controls for it.
#[test]
fn no_rename_controls_in_views_without_a_lattice() {
    let mut h = start();
    goto_lattices_view(&mut h, LatticesView::NonEuclidean);
    assert!(h.state().breadcrumb().lattice.is_none());
    assert!(hud_rect(&h, RENAME).is_none(), "no Rename in the H² view");
    goto_lattices_view(&mut h, LatticesView::Plane2D);
    assert!(hud_rect(&h, RENAME).is_some());
}

/// Findings 8 and 10: p4g #1 of the bond rule at t1 = -1/2 is the square lattice, group p4m (the hand case of
/// `gallery_tab::tests`). The HUD thumbnail is built from the tuned lattice and the group segment names p4m.
#[test]
fn tuned_lattice_thumbnail_and_raised_group() {
    let mut h = start();
    {
        let a = h.state_mut();
        let r = CatalogueRuleUi::ALL.into_iter().find(|r| r.matches("bonds")).unwrap();
        a.gallery.select_rule(r, &a.backend, &mut a.canvas2d);
        a.gallery.select_group_by_symbol("p4g", &a.backend, &mut a.canvas2d);
        a.gallery.select_catalogue_entry(0, &a.backend, &mut a.canvas2d);
        assert_eq!(a.gallery.base_item().group, "p4g");
        assert_eq!(a.gallery.drag_param("t1", -0.5, &mut a.canvas2d), Some(-0.5));
    }
    h.run_steps(3);
    let a = h.state();
    assert_ne!(a.gallery.active_item().recipe, a.gallery.base_item().recipe);
    let (key, _) = a.deck.hud_thumb.as_ref().expect("a tuned thumbnail was built");
    assert!(key.1 == a.gallery.active_item().recipe, "built from the tuned recipe");
    let bc = a.breadcrumb();
    let g = bc.group.expect("group segment");
    assert_eq!(g.text, "p4g (4*2) → p4m");
    assert!(g.hover.contains("larger group p4m (special locus)"), "{}", g.hover);
}

/// Finding 9: a lattice whose verdicts cannot be built is cached as such (not rebuilt every frame).
#[test]
fn failed_verdicts_are_cached() {
    let mut a = crate::app::PapertillingApp::new(false, String::from("test"), String::from("test"));
    let mut bad = a.gallery.active_item().clone();
    bad.recipe.group = String::from("not-a-group");
    bad.recipe.sites.clear();
    assert!(a.surfaces.lattice_verdicts(&bad).is_none());
    assert!(a.surfaces.verdicts_cached_for(&bad), "the failure is cached");
    assert!(a.surfaces.lattice_verdicts(&bad).is_none());
}

/// Finding 13: the realm saved in a session is where the next Tab opens the deck (once); then the deck opens at
/// the realm of the current view again.
#[test]
fn saved_deck_realm_is_used_once() {
    let mut h = start();
    goto_lattices_view(&mut h, LatticesView::Plane2D);
    let raw = r#"{"schema_version": 3, "active_mode": "Lattices", "lattices_view": "Plane2D",
        "deck": {"realm": "PresetsIo"}}"#;
    let st = session::parse_session(raw).unwrap().0;
    let notes = h.state_mut().apply_session(&st, None);
    assert!(notes.is_empty(), "{notes:?}");
    h.run_steps(2);
    open_deck(&mut h);
    assert_eq!(h.state().deck.realm, Realm::PresetsIo, "the saved realm");
    press(&mut h, Key::Tab);
    assert!(!h.state().deck.open);
    open_deck(&mut h);
    assert_eq!(h.state().deck.realm, h.state().current_realm(), "then the realm of the view");
    assert_eq!(h.state().deck.realm, Realm::Tilings);
}

/// Finding 14: junk workbench data in a session gives notes without the internal "workbench" name.
#[test]
fn junk_stage_state_notes_are_realm_native() {
    for wb in ["5", r#"{"view": "Bogus"}"#] {
        let raw = format!(r#"{{"schema_version": 3, "workbench": {wb}}}"#);
        let st = session::parse_session(&raw).unwrap().0;
        let mut a = crate::app::PapertillingApp::new(false, String::from("test"), String::from("test"));
        let notes = a.apply_session(&st, None);
        assert_eq!(notes.len(), 1, "{wb}: {notes:?}");
        assert!(notes.iter().all(|n| !n.to_lowercase().contains("workbench")), "{wb}: {notes:?}");
    }
}

/// Finding 1, renamed lattice: with a long custom name and a long alias note, in both themes, in Combine, at the
/// narrowest width of each density, Rename and Reset name stay left of the right cluster.
#[test]
fn renamed_lattice_keeps_reset_name_reachable() {
    let mut h = start();
    goto_defects_view(&mut h, DefectsView::CombineAndFuse);
    {
        let g = &mut h.state_mut().gallery;
        g.set_label_in_memory("A very long custom name for this lattice, for the HUD width test");
        g.alias_note = Some(String::from("Saved for session only (the alias file could not be written: denied)"));
    }
    for theme in [StudioTheme::Paper, StudioTheme::Studio] {
        h.state_mut().theme = theme;
        theme.apply(&h.ctx);
        for w in [1240.0, 960.0, super::HUD_MIN_WINDOW_WIDTH] {
            set_width(&mut h, w);
            let right = right_cluster_left(&h);
            for l in [CODEX, RENAME, "Reset name"] {
                let r = hud_rect(&h, l).unwrap_or_else(|| panic!("{theme:?} w={w}: no {l}"));
                assert!(r.right() <= right + 0.5, "{theme:?} w={w}: {l} {r:?} under the right cluster at {right}");
            }
        }
    }
}

/// Findings 3 and 7 together: a rename left pending by a mouse-opened deck ends when the view no longer shows the
/// lattice, so Tab and Esc keep working there.
#[test]
fn pending_rename_ends_in_a_view_without_the_lattice() {
    let mut h = start();
    click_in_hud(&mut h, RENAME);
    click_in_hud(&mut h, CODEX);
    assert!(h.state().deck.open && h.state().gallery.renaming);
    equip_state(&mut h, CardId::HyperbolicAtlas);
    assert!(!h.state().deck.open, "equip closes the deck");
    assert_eq!(h.state().lattices_view, LatticesView::NonEuclidean);
    assert!(!h.state().gallery.renaming, "the rename ended with its controls");
    open_deck(&mut h);
    press(&mut h, Key::Escape);
    assert!(!h.state().deck.open, "Esc closes the deck");
}
