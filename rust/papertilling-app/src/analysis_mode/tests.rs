//! Quarantine and end-to-end verification tests for `3. Analysis` and the three-mode shell.

use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::space3d::nets3d::LineDefect3d;

use super::{AnalysisTarget, SpectrumColorOverlay};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::game_menu::Realm;
use crate::game_menu::test_util::{deck_card_index, open_deck, press};
use crate::mesh3d::DefectViewMode;
use crate::theme::{StudioTheme, typography};
use crate::workbench::WorkbenchView;

pub(crate) const FORBIDDEN_SPECTRAL: [&str; 6] = [
    "Tight-Binding",
    "Compute Spectrum",
    "LDOS",
    "Spectral charge",
    "δρ",
    "Peierls",
];

pub(crate) fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, app| {
                let _ = app.frame_ui(ui);
            },
            app,
        )
}

fn click_combobox_value(h: &mut Harness<'static, PapertillingApp>, val: &str) {
    h.get_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|n| n.value().as_deref() == Some(val))
        .unwrap_or_else(|| panic!("no ComboBox with selected value '{val}'"))
        .click();
}

pub(crate) fn assert_no_forbidden_tokens(h: &Harness<'_, PapertillingApp>, where_ctx: &str) {
    for tok in FORBIDDEN_SPECTRAL {
        assert!(
            h.query_all_by_label_contains(tok).next().is_none(),
            "{where_ctx}: forbidden spectral token '{tok}' leaked into non-Analysis UI"
        );
        assert!(
            h.query_all_by_role(egui::accesskit::Role::ComboBox)
                .all(|n| !n.value().is_some_and(|v| v.contains(tok))),
            "{where_ctx}: forbidden spectral token '{tok}' leaked into ComboBox value"
        );
    }
    let app = h.state();
    if let Some(leg) = app.view3d.legend() {
        for tok in ["LDOS", "Spectral charge", "δρ"] {
            assert!(
                !leg.title.contains(tok),
                "{where_ctx}: spectral 3D legend '{}' leaked outside Analysis",
                leg.title
            );
        }
    }
    if let Some(ov) = app.canvas2d.gauge_overlay()
        && let Some(leg) = &ov.legend
    {
        for tok in ["LDOS", "Spectral charge", "δρ"] {
            assert!(
                !leg.title.contains(tok),
                "{where_ctx}: spectral 2D legend '{}' leaked outside Analysis",
                leg.title
            );
        }
    }
}

pub(crate) fn wait_until(h: &mut Harness<'static, PapertillingApp>, what: &str, pred: impl Fn(&PapertillingApp) -> bool) {
    let t0 = Instant::now();
    loop {
        h.step();
        if pred(h.state()) {
            h.run_steps(2);
            if pred(h.state()) {
                return;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(45), "{what}: timed out waiting for idle");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The active tab's build has settled. Only the tab on screen is checked: the other tabs' builders may stay
/// pending until they are visited (e.g. `workbench.dirty` starts true and clears only on the Workbench tab).
pub(crate) fn settled(a: &PapertillingApp) -> bool {
    match a.active_tab {
        NavTab::Workbench => !a.workbench_busy() && !a.workbench.dirty,
        NavTab::Surfaces => !a.surfaces.is_building(),
        NavTab::Defects => !a.defects.is_busy() && !a.defects.initializing,
        NavTab::Combine => !a.combine.is_building() && !a.combine.is_initializing(),
        NavTab::Atlas => !a.atlas.dirty,
        NavTab::Analysis => !a.analysis.is_busy(),
        NavTab::Gallery => true,
    }
}

const NUM_KEYS: [egui::Key; 7] = [
    egui::Key::Num1,
    egui::Key::Num2,
    egui::Key::Num3,
    egui::Key::Num4,
    egui::Key::Num5,
    egui::Key::Num6,
    egui::Key::Num7,
];

/// The number key of `realm` (1-7).
pub(crate) fn realm_key(realm: Realm) -> egui::Key {
    NUM_KEYS[realm.number() - 1]
}

/// Equip the card titled `title` in `realm` with real key presses: Tab (if closed), the realm's number key, then
/// from card 0 `S` down the rows and `D` along the last one, Enter; then wait for the active tab's build.
pub(crate) fn equip_keys(h: &mut Harness<'static, PapertillingApp>, realm: Realm, title: &str) {
    if !h.state().deck.open {
        open_deck(h);
    }
    press(h, realm_key(realm));
    assert_eq!(h.state().deck.realm, realm, "number key selects {realm:?}");
    let r = realm.number() - 1;
    {
        let deck = &mut h.state_mut().deck;
        deck.pill[r] = 0;
        deck.filter.clear();
        deck.cursor[r] = 0;
    }
    h.run_steps(2);
    let target = deck_card_index(h, realm, title);
    let cols = h.state().deck.cols.max(1);
    for _ in 0..target / cols {
        press(h, egui::Key::S);
    }
    for _ in 0..target % cols {
        press(h, egui::Key::D);
    }
    assert_eq!(h.state().deck.cursor[r], target, "cursor reached '{title}'");
    press(h, egui::Key::Enter);
    assert!(!h.state().deck.open, "Enter on '{title}' closes the deck");
    wait_until(h, title, settled);
}

/// `label()` with its first letter uppercased (the surface card titles).
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

/// The realm and title of the card for the stage panel's current sub-row.
fn wb_card(a: &PapertillingApp, v: WorkbenchView) -> (Realm, String) {
    let i = a.workbench.sub_index(v);
    let realm = match v {
        WorkbenchView::S2Defects | WorkbenchView::S3Detect | WorkbenchView::S7Fusion => Realm::Forge,
        WorkbenchView::S4Flat3d => Realm::Manifolds,
        WorkbenchView::S5Curved2d if i <= 2 => Realm::Geometries,
        WorkbenchView::S5Curved2d => Realm::Manifolds,
        WorkbenchView::S6Honeycombs => Realm::Geometries,
    };
    (realm, a.workbench.sub_labels(v)[i].to_string())
}

/// The card that shows Lattices view `v` as it was last left (sub-tab, surface target, sub-row): what the
/// removed strip button restored.
fn lattices_card(a: &PapertillingApp, v: LatticesView) -> (Realm, String) {
    match v {
        LatticesView::Plane2D => (Realm::Geometries, String::from("E² Euclidean plane")),
        LatticesView::ThreeDAndSubperiodic => (Realm::Groups, a.atlas.last_pristine_3d_subtab.label().to_string()),
        LatticesView::NonEuclidean => (Realm::Geometries, String::from("H² hyperbolic tiling (Atlas)")),
        LatticesView::Manifolds => (Realm::Manifolds, capitalised(a.surfaces.target.label())),
        LatticesView::Flat3D | LatticesView::Curved2D | LatticesView::Honeycombs3D => {
            wb_card(a, v.workbench_view().expect("stage panel view"))
        }
    }
}

fn defects_card(a: &PapertillingApp, v: DefectsView) -> (Realm, String) {
    match v {
        DefectsView::Catalogue2D => (Realm::Forge, a.defects.sub_tab.label().to_string()),
        DefectsView::AtlasDefects => (Realm::Forge, a.atlas.last_defect_subtab.defect_label().to_string()),
        DefectsView::CombineAndFuse => (Realm::Forge, String::from("Place & fuse defects")),
        DefectsView::PointAndBoundary | DefectsView::DetectFields | DefectsView::FusionSteps => {
            wb_card(a, v.workbench_view().expect("stage panel view"))
        }
    }
}

fn equip_lattices_view(h: &mut Harness<'static, PapertillingApp>, v: LatticesView) {
    let (realm, title) = lattices_card(h.state(), v);
    equip_keys(h, realm, &title);
    assert_eq!((h.state().active_mode, h.state().lattices_view), (AppMode::Lattices, v), "'{title}'");
}

fn equip_defects_view(h: &mut Harness<'static, PapertillingApp>, v: DefectsView) {
    let (realm, title) = defects_card(h.state(), v);
    equip_keys(h, realm, &title);
    assert_eq!((h.state().active_mode, h.state().defects_view), (AppMode::Defects, v), "'{title}'");
}

/// Pill counts of realms 1-5: Geometries, Symmetry Groups, Tilings, Surfaces, Defects.
const PILLS_1_TO_5: [usize; 5] = [1, 1, 4, 3, 5];

/// Every pill and the first 24 cards of realms 1-5 (cards and stat card on screen), then one representative
/// equip per realm: no spectral token anywhere.
fn scan_deck_realms_1_to_5(h: &mut Harness<'static, PapertillingApp>) {
    let lattice = h.state().gallery.base_item().label.clone();
    let reps = [
        (Realm::Geometries, String::from("Regular {p,q}")),
        (Realm::Groups, String::from("80 Layer Groups")),
        // the current lattice's card: equipping it keeps the lattice the Analysis steps below were written for
        (Realm::Tilings, lattice),
        (Realm::Manifolds, String::from("Klein bottle")),
        (Realm::Forge, String::from("Stone-Wales")),
    ];
    for ((realm, rep), n_pills) in reps.into_iter().zip(PILLS_1_TO_5) {
        let (n, r) = (realm.number(), realm.number() - 1);
        if !h.state().deck.open {
            open_deck(h);
        }
        press(h, realm_key(realm));
        assert_eq!(h.state().deck.realm, realm);
        for pill in 1..n_pills {
            // set through state: pill labels ("Surfaces", ...) can also exist in the panels underneath
            h.state_mut().deck.pill[r] = pill;
            h.run_steps(2);
            assert_no_forbidden_tokens(h, &format!("deck realm {n} pill {pill}"));
        }
        h.state_mut().deck.pill[r] = 0;
        h.run_steps(2);
        let n_cards = h.state().deck.n_cards;
        assert!(n_cards > 0, "deck realm {n} lists cards");
        for k in 0..n_cards.min(24) {
            h.state_mut().deck.cursor[r] = k;
            h.run_steps(2);
            assert_no_forbidden_tokens(h, &format!("deck realm {n} card {k}"));
        }
        equip_keys(h, realm, &rep);
        assert_no_forbidden_tokens(h, &format!("realm {n} after equipping '{rep}'"));
        if h.state().deck.open {
            press(h, egui::Key::Escape);
        }
        assert_no_forbidden_tokens(h, &format!("HUD after realm {n}"));
    }
}

#[test]
fn spectral_ui_and_overlays_are_quarantined_to_analysis_mode_only() {
    let mut h = harness();
    h.state_mut().set_advanced(true);
    h.run_steps(3);

    // 1. Lattices: visit all sub-views including 0D/1D, 80-layer, 230-SG, H², and Manifolds.
    equip_lattices_view(&mut h, LatticesView::Plane2D);
    assert_no_forbidden_tokens(&h, "Lattice -> 2D Plane");

    equip_lattices_view(&mut h, LatticesView::ThreeDAndSubperiodic);
    h.run_steps(2);
    assert!(h.state().atlas.pristine_only);
    for st_lbl in ["0D/1D Subperiodic", "80 Layer Groups", "230 3D Space Groups"] {
        h.get_by_label(st_lbl).click();
        h.run_steps(2);
        assert_no_forbidden_tokens(&h, st_lbl);
        assert!(
            h.query_all_by_label_contains("Line defect:").next().is_none()
                && h.query_all_by_label_contains("Frieze defect:").next().is_none(),
            "{st_lbl}: defect controls must be hidden in pristine Lattices mode"
        );
    }

    equip_lattices_view(&mut h, LatticesView::NonEuclidean);
    h.run_steps(2);
    assert!(h.state().atlas.pristine_only);
    assert_no_forbidden_tokens(&h, "Lattice -> Non-Euclidean");
    assert!(
        h.query_all_by_label_contains("Central Wedge").next().is_none(),
        "Non-Euclidean pristine view must hide Central Wedge disclination controls"
    );

    equip_lattices_view(&mut h, LatticesView::Manifolds);
    wait_until(&mut h, "Manifolds surface build", |a| !a.surfaces.is_building());
    assert_no_forbidden_tokens(&h, "Lattice -> Manifolds");

    // Workbench Lattices views (stage S4..S6 panels).
    for (lbl, v) in [
        ("Flat 3D: Tubes, Ribbons, Flakes", LatticesView::Flat3D),
        ("{p,q} on S² / E² / H²", LatticesView::Curved2D),
        ("Honeycombs & TPMS", LatticesView::Honeycombs3D),
    ] {
        equip_lattices_view(&mut h, v);
        wait_until(&mut h, lbl, |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().active_tab, NavTab::Workbench, "{lbl}");
        assert_no_forbidden_tokens(&h, lbl);
        assert!(
            h.query_all_by_label_contains("Line defect:").next().is_none()
                && h.query_all_by_label_contains("Frieze defect:").next().is_none()
                && h.query_all_by_label_contains("Central Wedge").next().is_none(),
            "{lbl}: defect controls must be hidden in pristine Lattices mode"
        );
    }

    // 2. Defects: visit all sub-views + open "Color by:" 2D Overlay ComboBoxes.
    equip_defects_view(&mut h, DefectsView::Catalogue2D);
    wait_until(&mut h, "Defects Catalogue build", |a| !a.defects.is_busy() && !a.defects.initializing);
    assert_eq!((h.state().active_mode, h.state().defects_view), (AppMode::Defects, DefectsView::Catalogue2D));
    assert_no_forbidden_tokens(&h, "Defects -> Catalogue (2D)");

    // Open "Color by:" ComboBox.
    click_combobox_value(&mut h, "Standard sublattice");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Plaquette flux").next().is_some(), "Defects Color by ComboBox opened");
    assert_no_forbidden_tokens(&h, "Defects -> Catalogue (2D) Color by popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    equip_defects_view(&mut h, DefectsView::AtlasDefects);
    h.run_steps(2);
    assert!(!h.state().atlas.pristine_only);
    assert!(h.query_by_label("3D Line Defects").is_none(), "3D Line Defects sub-tab must stay hidden");
    for st_lbl in ["0D/1D Subperiodic Defects", "H² Wedge Disclinations"] {
        h.get_by_label(st_lbl).click();
        h.run_steps(2);
        assert_no_forbidden_tokens(&h, st_lbl);
    }

    // Workbench Defects views (stage S2, S3 panels).
    for (lbl, v) in [("Point, Stone-Wales & GB", DefectsView::PointAndBoundary), ("Detect & Fields", DefectsView::DetectFields)] {
        equip_defects_view(&mut h, v);
        wait_until(&mut h, lbl, |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().active_tab, NavTab::Workbench, "{lbl}");
        assert_no_forbidden_tokens(&h, lbl);
    }

    equip_defects_view(&mut h, DefectsView::CombineAndFuse);
    wait_until(&mut h, "Combine build", |a| !a.combine.is_building());
    assert_no_forbidden_tokens(&h, "Defects -> Combine");

    click_combobox_value(&mut h, "Standard sublattice");
    h.run_steps(2);
    assert!(h.query_all_by_label_contains("Plaquette flux").next().is_some(), "Combine Color by ComboBox opened");
    assert_no_forbidden_tokens(&h, "Defects -> Combine Color by popup open");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);

    // All Views: every pill and card of realms 1-5 (and their stat cards), one equip per realm, the HUD after.
    scan_deck_realms_1_to_5(&mut h);
    // back to Combine (built) so the Analysis steps start from the same state as before the deck scan
    equip_defects_view(&mut h, DefectsView::CombineAndFuse);
    assert_no_forbidden_tokens(&h, "Defects -> Combine (after the deck scan)");

    // 3. Analysis mode: verify Tight-Binding panel appears, compute spectrum, verify LDOS + Peierls + 2D/3D legends.
    // C1: Analysis is postponed and no user input reaches it, so the hidden mode is entered by the internal route
    h.state_mut().switch_mode(AppMode::Analysis);
    h.run_steps(2);
    wait_until(&mut h, "Analysis entered", settled);
    wait_until(&mut h, "Analysis source idle", |a| !a.defects.is_busy() && !a.defects.initializing);
    assert_eq!((h.state().active_mode, h.state().active_tab), (AppMode::Analysis, NavTab::Analysis));
    assert!(h.query_all_by_label_contains("Tight-Binding").next().is_some(), "Tight-Binding panel visible in Analysis");
    assert!(h.query_all_by_label_contains("Peierls").next().is_some(), "Peierls flux slider visible for crystal target");

    h.get_by_label("Compute Spectrum").click();
    wait_until(&mut h, "Compute Spectrum", |a| !a.analysis.is_busy() && a.analysis.spectrum.result.is_some() && !a.analysis.view3d_dirty);

    assert!(h.query_all_by_label_contains("Tight-Binding").next().is_some(), "Tight-Binding header kept after compute");
    assert!(h.query_all_by_label_contains("LDOS").next().is_some(), "LDOS option visible in Analysis");
    assert!(h.query_all_by_label_contains("Peierls").next().is_some(), "Peierls slider visible in Analysis");
    let app = h.state();
    let leg3 = app.view3d.legend().expect("3D spectral legend present in Analysis");
    assert!(leg3.title.contains("LDOS"), "3D legend: {}", leg3.title);
    let leg2 = app
        .canvas2d
        .gauge_overlay()
        .and_then(|o| o.legend.as_ref())
        .expect("2D spectral legend present in Analysis");
    assert!(leg2.title.contains("LDOS"), "2D legend: {}", leg2.title);

    // 4. Switching back to Lattices and Defects clears spectral legends and overlays immediately.
    // the view the removed "1. Lattices" strip restored here (Surfaces may show its own elastic colour legend)
    equip_lattices_view(&mut h, LatticesView::Honeycombs3D);
    assert!(h.state().view3d.legend().is_none(), "3D spectral legend cleared on returning to Lattices");
    assert!(h.state().canvas2d.gauge_overlay().is_none(), "2D spectral overlay cleared on returning to Lattices");
    assert_no_forbidden_tokens(&h, "Return to 1. Lattices");

    let v = h.state().defects_view;
    equip_defects_view(&mut h, v);
    wait_until(&mut h, "Return to 2. Defects", |a| !a.defects.is_busy());
    assert!(h.state().view3d.legend().is_none(), "3D spectral legend cleared on returning to Defects");
    assert!(h.state().canvas2d.gauge_overlay().is_none(), "2D spectral overlay cleared on returning to Defects");
    assert_no_forbidden_tokens(&h, "Return to 2. Defects");
}

#[test]
fn atlas_preserves_defect_selection_across_pristine_lattices_round_trip() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    app.select_defects_view(DefectsView::AtlasDefects);
    assert!(!app.atlas.pristine_only);
    assert_ne!(app.atlas.space3d.selected_defect, LineDefect3d::None);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, -1);

    app.atlas.space3d.selected_defect = LineDefect3d::PureDisclination90_123;
    app.atlas.hyperbolic.wedge_delta = -2;

    app.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
    assert!(app.atlas.pristine_only);
    assert_eq!(app.atlas.space3d.effective_defect(), LineDefect3d::None);

    app.select_lattices_view(LatticesView::NonEuclidean);
    assert!(app.atlas.pristine_only);
    assert_eq!(app.atlas.hyperbolic.effective_wedge(), 0);
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::HyperbolicH2);

    app.select_defects_view(DefectsView::AtlasDefects);
    assert!(!app.atlas.pristine_only);
    // C1 (user, round 5): the Defects atlas opens on the flat 2D defects (0D/1D rosettes and friezes)
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::Subperiodic);
    assert_eq!(app.atlas.space3d.selected_defect, LineDefect3d::PureDisclination90_123);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, -2);

    // Explicit user choice of None / 0 in Defects mode survives round-tripping through Catalogue2D / Lattices.
    app.atlas.space3d.selected_defect = LineDefect3d::None;
    app.atlas.hyperbolic.wedge_delta = 0;
    app.select_defects_view(DefectsView::Catalogue2D);
    app.select_defects_view(DefectsView::AtlasDefects);
    assert_eq!(app.atlas.space3d.selected_defect, LineDefect3d::None);
    assert_eq!(app.atlas.hyperbolic.wedge_delta, 0);
}

#[test]
fn analysis_supports_pristine_plane_surface_and_combined_targets() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let ctx = egui::Context::default();
    app.analysis.view_mode = DefectViewMode::Split2D3D;
    app.switch_mode(AppMode::Analysis);

    // Pristine 2D Lattice target
    app.analysis.select_target(AnalysisTarget::PlaneLattice);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(!app.analysis.plane_crystal.pos.is_empty());
    let item = app.gallery.active_item().clone();
    app.analysis.submit_active_source(&item, &app.surfaces, &app.defects, &app.combine, None);
    let res = app.analysis.spectrum.wait_idle();
    assert!(res.is_some());
    assert_eq!(app.analysis.spectrum.color_overlay, SpectrumColorOverlay::LdosMultiplet);
    app.analysis.sync_canvas(&app.surfaces, &app.defects, &app.combine, &mut app.canvas2d);
    assert!(app.canvas2d.gauge_overlay().and_then(|o| o.legend.as_ref()).is_some());

    // Surface target (honeycomb dodecahedron sphere = 20 sites) + stale invalidation on lattice switch
    let hc = app.backend.find_named("honeycomb").unwrap().clone();
    app.select_lattice_and_sync(hc.clone());
    app.surfaces.target = crate::backend::SurfaceTargetUi::Sphere;
    app.analysis.select_target(AnalysisTarget::Surface);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    app.analysis.submit_active_source(&hc, &app.surfaces, &app.defects, &app.combine, None);
    assert!(app.analysis.spectrum.wait_idle().is_some());
    let sq = app.backend.find_named("square").unwrap().clone();
    app.select_lattice_and_sync(sq.clone());
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(app.analysis.spectrum.result.is_none(), "changing lattice invalidates stale Surface spectrum");

    // CombinedDefects target + stale invalidation when a pin moves with identical site/edge counts
    app.select_defects_view(DefectsView::CombineAndFuse);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    while app.combine.is_building() {
        std::thread::sleep(Duration::from_millis(10));
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    }
    app.switch_mode(AppMode::Analysis);
    app.analysis.select_target(AnalysisTarget::CombinedDefects);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    let cur = app.gallery.active_item().clone();
    app.analysis.submit_active_source(&cur, &app.surfaces, &app.defects, &app.combine, None);
    assert!(app.analysis.spectrum.wait_idle().is_some());
    assert!(!app.combine.last_build.crystal.pos.is_empty());
    app.combine.last_build.crystal.pos[0][0] += 0.125;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| { let _ = app.frame_ui(ui); });
    assert!(app.analysis.spectrum.result.is_none(), "moving sites invalidates stale CombinedDefects spectrum");
}
