//! Pure and state-level Command Deck tests T1-T10 (M1 plan section 6.1; written by WP-T).
//!
//! Expected values are known answers written here (realm table, wallpaper-group orbifolds, card titles), never
//! values read back from the code under test.

use super::Realm;
use super::badges::Badge;
use super::realms::{Card, CardId, LatticeRef, card_realm, pills};
use super::search::filter_matches;
use crate::analysis_mode::AnalysisTarget;
use crate::analysis_mode::tests::FORBIDDEN_SPECTRAL;
use crate::app::{AppMode, DefectsView, LatticesView, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::backend::{SurfaceTargetUi, VerdictKindUi};
use crate::defects_tab::DefectSubTab;
use crate::gallery_tab::filter::query_matches;
use crate::theme::ThemeTokens;
use crate::workbench::WorkbenchView;

fn app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

/// One frame with no GPU backend: the texture deltas (new font glyphs) are consumed here, as a backend would,
/// because epaint asserts in debug builds that a dropped `TexturesDelta` is empty.
fn frame(a: &mut PapertillingApp, ctx: &egui::Context) {
    let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = a.frame_ui(ui);
    });
    out.textures_delta.clear();
}

/// Every card of `realm` under every pill.
fn all_pill_cards(a: &mut PapertillingApp, realm: Realm) -> Vec<Card> {
    (0..pills(realm).len()).flat_map(|p| a.deck_cards(realm, p, "")).collect()
}

/// Labels the HUD draws in Advanced mode.
const HUD_LABELS: [&str; 11] = [
    "Undo",
    "Redo",
    "Examples",
    "Session",
    "Save figure",
    "View",
    "Find lattice (Ctrl+K)",
    "Theme: Light",
    "Theme: Dark",
    "All views (Tab)",
    "Advanced",
];

/// Spectral wording beyond `FORBIDDEN_SPECTRAL`, checked case-insensitively on card text only. Bare "band" is not
/// listed: "Mobius band" is a surface card title.
const EXTRA_SPECTRAL: [&str; 6] = ["spectr", "eigen", "berry", "chern", "band structure", "density of states"];

fn assert_not_spectral(text: &str, what: &str) {
    for tok in FORBIDDEN_SPECTRAL {
        assert!(!text.contains(tok), "{what}: '{text}' contains forbidden token '{tok}'");
    }
    let low = text.to_lowercase();
    for tok in EXTRA_SPECTRAL {
        assert!(!low.contains(tok), "{what}: '{text}' contains spectral wording '{tok}'");
    }
}

#[test]
fn realm_table() {
    assert_eq!(Realm::ALL.len(), 7);
    let modes = [
        Some(AppMode::Lattices),
        Some(AppMode::Lattices),
        Some(AppMode::Lattices),
        Some(AppMode::Lattices),
        Some(AppMode::Defects),
        Some(AppMode::Analysis),
        None,
    ];
    for (i, r) in Realm::ALL.into_iter().enumerate() {
        assert_eq!(r.number(), i + 1, "{r:?}");
        assert_eq!(r.app_mode(), modes[i], "{r:?}");
        // C1: realm 6 (Analysis Lab) is hidden: key 6 maps to nothing, and Q / E walk the six shown realms
        if r.is_shown() {
            assert_eq!(Realm::from_number(i + 1), Some(r));
            assert_eq!(r.next().prev(), r);
        } else {
            assert_eq!(Realm::from_number(i + 1), None);
        }
    }
    assert_eq!(Realm::SHOWN.len(), 6);
    assert!(!Realm::SHOWN.contains(&Realm::AnalysisLab));
    assert_eq!(Realm::from_number(0), None);
    assert_eq!(Realm::from_number(8), None);
    assert_eq!(Realm::Geometries.prev(), Realm::PresetsIo);
    assert_eq!(Realm::PresetsIo.next(), Realm::Geometries);
    assert_eq!(Realm::Tilings.next(), Realm::Manifolds);
}

#[test]
fn every_workbench_sub_row_has_exactly_one_card_in_its_realm() {
    let mut a = app();
    let mut all = Vec::new();
    for r in Realm::ALL {
        all.extend(a.deck_cards(r, 0, ""));
    }
    for view in WorkbenchView::ALL {
        let labels = a.workbench.sub_labels(view);
        assert!(!labels.is_empty(), "{view:?} has sub-rows");
        for (i, label) in labels.iter().enumerate() {
            let id = CardId::Wb(view, u8::try_from(i).unwrap());
            let hits: Vec<&Card> = all.iter().filter(|c| c.id == id).collect();
            let hidden = (view == WorkbenchView::S4Flat3d && i <= 1)
                || (view == WorkbenchView::S3Detect && i >= 2)
                || (view == WorkbenchView::S5Curved2d && i == 3);
            if hidden {
                assert!(hits.is_empty(), "{view:?} row {i} ('{label}') is hidden");
                continue;
            }
            assert_eq!(hits.len(), 1, "{view:?} row {i} ('{label}'): exactly one card");
            let want = match view {
                WorkbenchView::S2Defects | WorkbenchView::S3Detect | WorkbenchView::S7Fusion => Realm::Forge,
                WorkbenchView::S4Flat3d => Realm::Manifolds,
                WorkbenchView::S5Curved2d if i <= 2 => Realm::Geometries,
                WorkbenchView::S5Curved2d => Realm::Manifolds,
                WorkbenchView::S6Honeycombs => Realm::Geometries,
            };
            assert_eq!(hits[0].realm, want, "{view:?} row {i}");
            assert_eq!(card_realm(id), want, "{view:?} row {i}");
            assert_eq!(hits[0].title, *label, "{view:?} row {i}");
        }
    }
    for c in &all {
        if let CardId::Wb(v, i) = c.id {
            assert!(usize::from(i) < a.workbench.sub_labels(v).len(), "{v:?} row {i} out of range");
        }
    }
}

/// A card that leads to each view (exhaustive matches: a new view needs a card here).
fn lattices_card(v: LatticesView) -> CardId {
    match v {
        LatticesView::Plane2D => CardId::Plane2D,
        LatticesView::ThreeDAndSubperiodic => CardId::AtlasPristine(AtlasSubTab::Layers80),
        LatticesView::NonEuclidean => CardId::HyperbolicAtlas,
        LatticesView::Manifolds => CardId::Surface(SurfaceTargetUi::Torus),
        LatticesView::Flat3D => CardId::Wb(WorkbenchView::S4Flat3d, 2), // C1: rows 0-1 are hidden
        LatticesView::Curved2D => CardId::Wb(WorkbenchView::S5Curved2d, 0),
        LatticesView::Honeycombs3D => CardId::Wb(WorkbenchView::S6Honeycombs, 0),
    }
}

fn defects_card(v: DefectsView) -> CardId {
    match v {
        DefectsView::Catalogue2D => CardId::DefectCatalogue(DefectSubTab::Disclinations),
        DefectsView::AtlasDefects => CardId::AtlasDefects(AtlasSubTab::HyperbolicH2), // C1: 3D line defects hidden
        DefectsView::CombineAndFuse => CardId::Combine,
        DefectsView::FuseAndCompare => CardId::FuseAndCompare,
        DefectsView::PointAndBoundary => CardId::Wb(WorkbenchView::S2Defects, 0),
        DefectsView::DetectFields => CardId::Wb(WorkbenchView::S3Detect, 0),
        DefectsView::FusionSteps => CardId::Wb(WorkbenchView::S7Fusion, 0),
    }
}

fn assert_listed(a: &mut PapertillingApp, id: CardId) {
    let realm = card_realm(id);
    assert!(a.deck_cards(realm, 0, "").iter().any(|c| c.id == id), "{id:?} is listed in {realm:?}");
}

#[test]
fn every_view_and_target_is_reachable() {
    let mut a = app();
    let ctx = egui::Context::default();
    frame(&mut a, &ctx);
    for v in LatticesView::ALL {
        let id = lattices_card(v);
        assert_listed(&mut a, id);
        a.deck_equip(id, &ctx);
        frame(&mut a, &ctx);
        assert_eq!((a.active_mode, a.lattices_view, a.active_tab), (AppMode::Lattices, v, v.nav_tab()), "{id:?}");
        assert!(!a.deck.open);
    }
    for v in DefectsView::ALL {
        let id = defects_card(v);
        assert_listed(&mut a, id);
        a.deck_equip(id, &ctx);
        frame(&mut a, &ctx);
        assert_eq!((a.active_mode, a.defects_view, a.active_tab), (AppMode::Defects, v, v.nav_tab()), "{id:?}");
    }
    // C1: Analysis is postponed: no shown realm lists its cards, and equipping one (internal only) is refused
    let before = (a.active_mode, a.active_tab);
    for t in AnalysisTarget::ALL {
        let id = CardId::Analysis(t);
        for r in Realm::SHOWN {
            assert!(a.deck_cards(r, 0, "").iter().all(|c| c.id != id), "{id:?} listed in {r:?}");
        }
        a.deck_equip(id, &ctx);
        frame(&mut a, &ctx);
        assert_eq!((a.active_mode, a.active_tab), before, "{id:?} is refused");
    }
}

#[test]
fn card_labels_unique() {
    let mut a = app();
    let mut view_labels: Vec<&str> = Vec::new();
    view_labels.extend(LatticesView::ALL.map(LatticesView::label));
    view_labels.extend(DefectsView::ALL.map(DefectsView::label));
    view_labels.extend(AppMode::ALL.map(AppMode::label));
    view_labels.extend(AtlasSubTab::ALL.map(AtlasSubTab::label));
    view_labels.extend(AtlasSubTab::ALL.map(AtlasSubTab::defect_label));
    view_labels.extend(AnalysisTarget::ALL.map(AnalysisTarget::label));
    view_labels.extend(WorkbenchView::ALL.map(WorkbenchView::heading));
    let tabs: Vec<&str> = Realm::ALL.iter().map(|r| r.tab_label()).collect();
    for (i, t) in tabs.iter().enumerate() {
        assert!(!tabs[..i].contains(t), "realm tab '{t}' repeated");
        assert!(!HUD_LABELS.contains(t), "realm tab '{t}' equals a HUD label");
    }
    let mut card_texts: Vec<String> = Vec::new();
    for r in Realm::ALL {
        let pl = pills(r);
        for (i, p) in pl.iter().enumerate() {
            assert!(!pl[..i].contains(p), "{r:?}: pill '{p}' repeated");
            assert!(!tabs.contains(p) && !HUD_LABELS.contains(p), "{r:?}: pill '{p}' equals a tab or HUD label");
        }
        for pill in 0..pl.len() {
            let cards = a.deck_cards(r, pill, "");
            let labels: Vec<String> = cards.iter().map(Card::a11y_label).collect();
            // "This group" lists the full catalogue (every rule), where one lattice may appear under several rules
            let this_group = r == Realm::Tilings && pill == 2;
            if !this_group {
                for (i, l) in labels.iter().enumerate() {
                    assert!(!labels[..i].contains(l), "{r:?} pill {pill}: card label '{l}' repeated");
                }
            }
            for (c, l) in cards.iter().zip(&labels) {
                assert!(l.starts_with(&format!("{} · ", r.number())), "{r:?}: a11y label '{l}'");
                assert!(!HUD_LABELS.contains(&c.title.as_str()), "{r:?}: card '{}' equals a HUD label", c.title);
                assert!(!HUD_LABELS.contains(&l.as_str()), "{r:?}: card '{l}' equals a HUD label");
                assert!(!view_labels.contains(&l.as_str()), "{r:?}: card '{l}' equals a view label or heading");
            }
            card_texts.extend(cards.iter().map(|c| c.title.clone()));
            card_texts.extend(labels);
        }
    }
    for t in tabs.iter().chain(Realm::ALL.iter().flat_map(|r| pills(*r).iter())) {
        assert!(!card_texts.iter().any(|c| c == t), "tab or pill '{t}' equals a card label");
    }
}

#[test]
fn realms_1_to_5_card_text_has_no_spectral_token() {
    let mut a = app();
    for r in &Realm::ALL[..5] {
        for c in all_pill_cards(&mut a, *r) {
            let what = format!("{r:?} card '{}'", c.title);
            assert_not_spectral(&c.title, &what);
            assert_not_spectral(&c.subtitle, &what);
            if let Some(b) = c.badge {
                assert_not_spectral(b.label(), &what);
            }
        }
    }
    let check = |a: &PapertillingApp, what: &str| {
        let bc = a.breadcrumb();
        for s in [Some(&bc.geometry), bc.group.as_ref(), bc.lattice.as_ref(), Some(&bc.mode)].into_iter().flatten() {
            assert_not_spectral(&s.text, what);
            assert_not_spectral(&s.hover, what);
        }
    };
    for v in LatticesView::ALL {
        a.select_lattices_view(v);
        check(&a, &format!("breadcrumb in Lattices / {v:?}"));
    }
    for v in DefectsView::ALL {
        a.select_defects_view(v);
        check(&a, &format!("breadcrumb in Defects / {v:?}"));
    }
}

#[test]
fn surface_badges_follow_verdicts() {
    for tokens in [ThemeTokens::paper(), ThemeTokens::studio()] {
        let ok = Badge::from_verdict(VerdictKindUi::Ok);
        let imp = Badge::from_verdict(VerdictKindUi::Impossible);
        let nf = Badge::from_verdict(VerdictKindUi::NotFound);
        assert_eq!((ok.label(), ok.colour(&tokens)), ("[PROVED OK]", tokens.status_ok));
        assert_eq!((imp.label(), imp.colour(&tokens)), ("[LOCKED / IMPOSSIBLE]", tokens.status_err));
        // a bounded search that found nothing is computational evidence, never a proof
        assert_eq!((nf.label(), nf.colour(&tokens)), ("[COMPUTATIONAL]", tokens.status_warn));
        assert!(!nf.label().contains("PROVED"));
    }
    let mut a = app();
    let item = a.gallery.active_item().clone();
    let kinds: Vec<VerdictKindUi> = a
        .surfaces
        .lattice_verdicts(&item)
        .expect("the start lattice has surface verdicts (built synchronously)")
        .iter()
        .map(|c| c.verdict)
        .collect();
    let cards = a.deck_cards(Realm::Manifolds, 1, "");
    for (k, t) in SurfaceTargetUi::ALL.into_iter().enumerate() {
        let card = cards.iter().find(|c| c.id == CardId::Surface(t)).unwrap_or_else(|| panic!("no {t:?} card"));
        assert_eq!(card.badge, Some(Badge::from_verdict(kinds[k])), "{t:?} badge follows verdict cell {k}");
    }
    for r in Realm::ALL {
        for c in a.deck_cards(r, 0, "") {
            if !matches!(c.id, CardId::Surface(_)) {
                assert_eq!(c.badge, None, "{r:?}: only surface cards carry a badge ('{}')", c.title);
            }
        }
    }
}

#[test]
fn breadcrumb_known_answers() {
    let mut a = app();
    let kagome = a.backend.find_named("kagome").expect("kagome").clone();
    a.select_lattice_and_sync(kagome);
    a.select_lattices_view(LatticesView::Plane2D);
    let bc = a.breadcrumb();
    assert_eq!((bc.geometry.text.as_str(), bc.geometry.realm), ("E² Plane", Realm::Geometries));
    let g = bc.group.expect("kagome on the 2D Plane shows its group");
    // kagome: wallpaper group p6m, Conway orbifold *632
    assert_eq!((g.text.as_str(), g.realm), ("p6m (*632)", Realm::Groups));
    let l = bc.lattice.expect("kagome on the 2D Plane shows the lattice");
    assert_eq!(l.realm, Realm::Tilings);
    assert!(!l.text.is_empty());
    assert_eq!((bc.mode.text.as_str(), bc.mode.realm), ("Lattice: 2D Plane", Realm::Tilings));

    let ss = a.backend.find_named("shastry_sutherland").expect("shastry_sutherland").clone();
    a.select_lattice_and_sync(ss);
    // Shastry-Sutherland: p4g, Conway orbifold 4*2
    assert_eq!(a.breadcrumb().group.expect("group segment").text, "p4g (4*2)");

    a.select_lattices_view(LatticesView::Honeycombs3D);
    let bc = a.breadcrumb();
    assert!(bc.group.is_none() && bc.lattice.is_none(), "honeycombs do not show the Gallery lattice");
    assert_eq!(bc.geometry.text, "S³ / E³ / H³ Honeycombs");

    a.select_defects_view(DefectsView::CombineAndFuse);
    let bc = a.breadcrumb();
    assert_eq!((bc.mode.text.as_str(), bc.mode.realm), ("Defects: Combine, Fuse & Move", Realm::Forge));
}

#[test]
fn current_realm_table() {
    let mut a = app();
    let lattices = [
        (LatticesView::Plane2D, Realm::Tilings),
        (LatticesView::ThreeDAndSubperiodic, Realm::Groups),
        (LatticesView::NonEuclidean, Realm::Geometries),
        (LatticesView::Manifolds, Realm::Manifolds),
        (LatticesView::Flat3D, Realm::Manifolds),
        (LatticesView::Honeycombs3D, Realm::Geometries),
    ];
    for (v, want) in lattices {
        a.select_lattices_view(v);
        assert_eq!(a.current_realm(), want, "Lattices / {v:?}");
    }
    a.enter_workbench(WorkbenchView::S5Curved2d, Some(0));
    assert_eq!((a.lattices_view, a.current_realm()), (LatticesView::Curved2D, Realm::Geometries), "S5 Regular");
    a.enter_workbench(WorkbenchView::S5Curved2d, Some(4));
    assert_eq!((a.lattices_view, a.current_realm()), (LatticesView::Curved2D, Realm::Manifolds), "S5 Orbifolds");
    for v in DefectsView::ALL {
        a.select_defects_view(v);
        assert_eq!(a.current_realm(), Realm::Forge, "Defects / {v:?}");
    }
    a.switch_mode(AppMode::Analysis);
    assert_eq!(a.current_realm(), Realm::AnalysisLab);
}

#[test]
fn filter_matches_titles_and_lattices() {
    let c = Card {
        id: CardId::Combine,
        realm: Realm::Forge,
        title: String::from("Place & fuse defects"),
        subtitle: String::from("only in the subtitle"),
        badge: None,
        thumb: None,
    };
    assert!(filter_matches(&c, None, "FUSE"));
    assert!(filter_matches(&c, None, "  place & fuse "));
    assert!(filter_matches(&c, None, ""));
    assert!(!filter_matches(&c, None, "subtitle"), "non-lattice cards match on the title only");

    let mut a = app();
    let idx = a.backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("kagome")).unwrap();
    let item = a.backend.named_lattices()[idx].clone();
    let r = LatticeRef::Named(idx);
    let lc = Card {
        id: CardId::Lattice(r),
        realm: Realm::Tilings,
        title: item.label.clone(),
        subtitle: String::new(),
        badge: None,
        thumb: Some(r),
    };
    for q in ["kagome", "p6m", "p4g", "zzz-no-such-lattice"] {
        assert_eq!(filter_matches(&lc, Some(&item), q), query_matches(q, &item), "lattice card, query '{q}'");
    }
    assert!(filter_matches(&lc, Some(&item), "p6m"), "kagome is p6m");

    let found = a.deck_cards(Realm::Tilings, 0, "p4g");
    assert!(!found.is_empty(), "realm 3 search finds p4g lattices");
    for card in &found {
        let CardId::Lattice(r) = card.id else { panic!("realm 3 search returned '{}'", card.title) };
        assert_eq!(a.lattice_ref_item(r).map(|it| it.group.as_str()), Some("p4g"), "'{}'", card.title);
    }
    let titles = |cards: Vec<Card>| cards.into_iter().map(|c| c.title).collect::<Vec<_>>();
    assert_eq!(titles(a.deck_cards(Realm::Forge, 0, "stone")), ["Stone-Wales"]);
    assert_eq!(titles(a.deck_cards(Realm::Manifolds, 0, "klein")), ["Klein bottle"]);
}

#[test]
fn deck_glyphs_exist_in_registered_fonts() {
    let ctx = egui::Context::default();
    crate::theme::typography::register_fonts(&ctx);
    for _ in 0..2 {
        // fonts set by register_fonts take effect at the start of the next pass
        ctx.run_ui(egui::RawInput::default(), |_| {}).textures_delta.clear();
    }
    let font = egui::FontId::proportional(13.0);
    for g in ["❖", "▸", "›", "⏎", "⚒", "ψ", "☀", "☾", "²", "³", "·", "…", "±"] {
        assert!(ctx.fonts_mut(|f| f.has_glyphs(&font, g)), "glyph '{g}' missing from the registered fonts");
    }
}
