//! Headless UI and state tests for the Defects > Combinations sub-tab.

use egui_kittest::kittest::{NodeT, Queryable};

use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, settled, wait_until};
use crate::app::{AppMode, DefectsView, NavTab, PapertillingApp};
use crate::defects_tab::DefectSubTab;
use crate::game_menu::{CardId, Realm};

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
    "Group cohomology",
];

const EXTRA_SPECTRAL: &[&str] = &[
    "spectr",
    "eigen",
    "berry",
    "chern",
    "band structure",
    "density of states",
];

fn assert_clean_combos_screen(h: &egui_kittest::Harness<'_, PapertillingApp>, ctx: &str) {
    assert_no_forbidden_tokens(h, ctx);
    for tok in FORBIDDEN_RPG_AND_GROUP3 {
        assert!(
            h.query_all_by_label_contains(tok).next().is_none(),
            "{ctx}: forbidden token '{tok}' visible on screen"
        );
    }
    for node in h.query_all_by_role(egui::accesskit::Role::Label) {
        if let Some(txt) = node.value().or_else(|| node.accesskit_node().label()) {
            let low = txt.to_lowercase();
            for tok in EXTRA_SPECTRAL {
                assert!(
                    !low.contains(tok),
                    "{ctx}: spectral wording '{tok}' visible in label '{txt}'"
                );
            }
        }
    }
}

fn switch_to_square_combinations(h: &mut egui_kittest::Harness<'static, PapertillingApp>) {
    h.run_steps(2);
    {
        let app = h.state_mut();
        let sq = app.backend.find_named("square").expect("square").clone();
        app.select_lattice_and_sync(sq);
        app.select_defects_view(DefectsView::Catalogue2D);
        // A smaller table than the default bound of 1.5: these tests check the screen and the hand-over to
        // Place & fuse, not the table's contents, and must not depend on how fast the machine is.
        app.defects.combos.max_sep_scale = 1.0;
    }
    wait_until(h, "square Allowed defects init", settled);
    h.get_by_label("Combinations").click();
    h.run_steps(2);
    wait_until(h, "square Combinations scan", |a| {
        !a.defects.combos.is_busy() && a.defects.combos.table.is_some()
    });
    h.run_steps(2);
}

#[test]
fn combinations_default_view_square_renders_sections_cleanly_without_forbidden_tokens() {
    let mut h = harness();
    assert!(!h.state().advanced, "starts in default view (Advanced OFF)");
    switch_to_square_combinations(&mut h);

    let app = h.state();
    assert!(!app.advanced, "remains in default view (Advanced OFF)");
    assert_eq!(app.active_tab, NavTab::Defects);
    assert_eq!(app.defects.sub_tab, DefectSubTab::Combinations);
    assert!(!app.defects.is_busy(), "combinations does not hold single-defect busy state");

    let table = app.defects.combos.table.as_ref().expect("combinations table computed");
    assert_eq!(table.group, "p4m");
    assert!(!table.cancelled);
    assert!(table.built_count > 0, "square disclination pairs must build");
    assert!(
        table.charge_groups.iter().any(|g| g.has_single_defect),
        "square disclination pairs include single-defect/vacuum charges"
    );
    assert!(
        table.charge_groups.iter().any(|g| !g.has_single_defect),
        "square disclination pairs within 1.0a include charges no single defect carries (two +180 at distance 1)"
    );

    assert!(
        h.query_all_by_label_contains("Charges carried by a single defect (and vacuum)")
            .next()
            .is_some(),
        "single-defect section heading rendered"
    );
    assert!(
        h.query_all_by_label_contains("Charges no single defect carries")
            .next()
            .is_some(),
        "multi-defect section heading rendered"
    );
    assert!(
        h.query_all_by_label_contains("[COMPUTATIONAL EVIDENCE]")
            .next()
            .is_some(),
        "bounds banner honesty note rendered"
    );
    assert!(
        h.query_all_by_label_contains("Triple: D(1b,+90) + 2×D(1a,-90)")
            .next()
            .is_some(),
        "square lattice exposes the triple preset"
    );
    assert!(
        h.query_all_by_label_contains("Patch & Drawing").next().is_none(),
        "Patch & Drawing controls hidden on Combinations sub-tab"
    );
    assert_clean_combos_screen(&h, "Default: Defects -> Combinations (square)");
}

#[test]
fn combinations_open_in_place_and_fuse_transfers_exact_specs() {
    let mut h = harness();
    switch_to_square_combinations(&mut h);

    let expected_specs = {
        let tbl = h.state().defects.combos.table.as_ref().expect("table");
        let (cg_i, core_i, mem_i) = h.state().defects.combos.selected.expect("auto-selected member");
        tbl.charge_groups[cg_i].core_groups[core_i].members[mem_i].specs.clone()
    };
    assert!(!expected_specs.is_empty(), "selected combination has defect specs");

    // Click the first visible "Open in Place & fuse" button (from the table or right inspector).
    h.query_all_by_label("Open in Place & fuse")
        .next()
        .expect("Open in Place & fuse button visible")
        .click();
    h.run_steps(2);

    wait_until(&mut h, "Combine loaded transferred specs", |a| {
        a.active_tab == NavTab::Combine
            && a.preset_pending.is_none()
            && !a.combine.is_initializing()
    });
    let app = h.state();
    assert_eq!(app.active_mode, AppMode::Defects);
    assert_eq!(app.defects_view, DefectsView::CombineAndFuse);
    assert_eq!(app.active_tab, NavTab::Combine);
    assert_eq!(
        app.combine.specs, expected_specs,
        "Place & fuse received the exact DefectSpec list from Combinations"
    );
}

#[test]
fn combinations_game_menu_deck_and_stat_card_do_not_panic() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let ctx = egui::Context::default();
    let sq = app.backend.find_named("square").expect("square").clone();
    app.select_lattice_and_sync(sq);
    app.select_defects_view(DefectsView::Catalogue2D);
    app.defects.wait_idle(&mut app.canvas2d);

    let cards = app.deck_cards(Realm::Forge, 1, "");
    let combo_card = cards
        .iter()
        .find(|c| c.id == CardId::DefectCatalogue(DefectSubTab::Combinations))
        .expect("Combinations card listed in Realm::Forge");
    assert_eq!(combo_card.title, "Combinations");
    assert!(!combo_card.subtitle.is_empty());

    // Render stat card for Combinations without panicking (regression for n[s as usize]).
    let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.show_stat_card(ui, combo_card);
    });
    out.textures_delta.clear();

    // Equip Combinations card and confirm it switches sub-tab without triggering single-defect rebuild.
    app.deck_equip(CardId::DefectCatalogue(DefectSubTab::Combinations), &ctx);
    assert_eq!(app.active_tab, NavTab::Defects);
    assert_eq!(app.defects.sub_tab, DefectSubTab::Combinations);
    assert!(!app.defects.is_busy(), "equipping Combinations skips rebuild_crystal");
}

#[test]
fn combinations_reload_for_item_cancels_and_clears_table() {
    let mut h = harness();
    switch_to_square_combinations(&mut h);
    assert!(h.state().defects.combos.table.is_some());
    assert!(h.state().defects.combos.selected.is_some());

    {
        let app = h.state_mut();
        let kag = app.backend.find_named("kagome").expect("kagome").clone();
        app.defects.reload_for_item(&kag, &app.backend, &mut app.canvas2d, false);
    }

    let combos = &h.state().defects.combos;
    assert!(combos.table.is_none(), "reload_for_item cleared combinations table");
    assert!(combos.selected.is_none(), "reload_for_item cleared selection");
    assert!(!combos.is_busy(), "reload_for_item cancelled combinations worker");
    assert!(combos.error.is_none(), "reload_for_item cleared error");
}
