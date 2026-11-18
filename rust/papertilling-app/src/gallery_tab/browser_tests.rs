//! Lattice browser: filter rules, ITA numbering, thumbnails and cross-list selection, checked against hand counts.

use std::collections::BTreeSet;

use super::filter::{BrowseFilter, ChiralFacet, SystemFacet, ita_number, query_matches, system_of};
use super::tests::canvas;
use super::thumb_cache::{BUILDS_PER_FRAME, LatticeThumbCache, ThumbGeom};
use super::{BrowseMode, GalleryTabState};
use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi};

fn named<'a>(backend: &'a AppBackend, key: &str) -> &'a CatalogueItemUi {
    backend.find_named(key).unwrap_or_else(|| panic!("named lattice {key}"))
}

#[test]
fn ita_numbers_follow_the_international_tables_order() {
    // ITA plane groups: 1 p1, 6 p2mm (pmm), 9 c2mm (cmm), 12 p4gm (p4g), 15 p31m, 17 p6mm (p6m).
    for (sym, n) in [("p1", 1), ("p2", 2), ("pmm", 6), ("cmm", 9), ("p4g", 12), ("p31m", 15), ("p6m", 17)] {
        assert_eq!(ita_number(sym), Some(n), "{sym}");
    }
    assert_eq!(ita_number("p7"), None);
}

#[test]
fn crystal_system_facet_keeps_the_hand_counted_groups() {
    let backend = AppBackend::new();
    let pick = |s: SystemFacet| -> BTreeSet<&str> {
        let f = BrowseFilter { system: Some(s), ..BrowseFilter::default() };
        backend.groups().iter().filter(|g| f.group_visible(g)).map(|g| g.symbol).collect()
    };
    let set = |v: &[&'static str]| v.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(pick(SystemFacet::Oblique), set(&["p1", "p2"]));
    assert_eq!(pick(SystemFacet::Rectangular), set(&["pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm"]));
    assert_eq!(pick(SystemFacet::Square), set(&["p4", "p4m", "p4g"]));
    assert_eq!(pick(SystemFacet::Hexagonal), set(&["p3", "p3m1", "p31m", "p6", "p6m"]));
    let no_mirror: BTreeSet<&str> =
        backend.groups().iter().filter(|g| !g.has_reflection_or_glide).map(|g| g.symbol).collect();
    assert_eq!(no_mirror, set(&["p1", "p2", "p3", "p4", "p6"]), "the five groups of proper motions only");
}

#[test]
fn query_matches_alias_name_symbol_and_ita_number() {
    let backend = AppBackend::new();
    let kag = named(&backend, "kagome");
    for q in ["kagome", "KAGOME", "kag", "p6m", "17", "ita17", "kag p6m"] {
        assert!(query_matches(q, kag), "{q:?} should match kagome (p6m)");
    }
    for q in ["16", "p6", "p4m", "kag p4m", "honeycomb"] {
        assert!(!query_matches(q, kag), "{q:?} should not match kagome (p6m)");
    }
    // "#n" is text: catalogue labels read "p4 #3 (1,2)", so "#3" must not mean ITA 3 (pm)
    let p4 = backend.catalogue_for_group("p4", CatalogueRuleUi::Bonds)[2].clone();
    assert!(p4.label.contains("#3"), "label format {:?}", p4.label);
    assert!(query_matches("p4 #3", &p4), "{:?} is found by its own label", p4.label);
    let br = named(&backend, "breathing_kagome");
    assert!(query_matches("kagome 14", br), "breathing kagome is p3m1 = ITA 14");
    let mut aliased = kag.clone();
    aliased.label = String::from("My Star Net");
    assert!(query_matches("star", &aliased), "custom alias is searched");
    assert!(query_matches("kagome", &aliased), "canonical name still found under an alias");
}

#[test]
fn chirality_facet_splits_named_lattices_by_their_flag() {
    let backend = AppBackend::new();
    let keep = |c: ChiralFacet| -> BTreeSet<String> {
        let f = BrowseFilter { chirality: c, ..BrowseFilter::default() };
        backend
            .named_lattices()
            .iter()
            .filter(|it| f.matches(it, system_of(backend.groups(), &it.group)))
            .filter_map(|it| it.named_key.clone())
            .collect()
    };
    let (chiral, achiral, any) = (keep(ChiralFacet::Chiral), keep(ChiralFacet::Achiral), keep(ChiralFacet::Any));
    assert_eq!(any.len(), 21);
    assert!(chiral.is_disjoint(&achiral));
    assert_eq!(chiral.len() + achiral.len(), 21);
    // groups p3, p4, p6 have no mirror line, so these three tilings are enantiomorphic;
    // p4m / p6m lattices carry mirrors and are achiral.
    for k in ["chiral_square", "chiral_kagome", "chiral_triangular"] {
        assert!(chiral.contains(k), "{k} is chiral");
    }
    for k in ["square", "triangular", "honeycomb", "kagome"] {
        assert!(achiral.contains(k), "{k} is achiral");
    }
}

#[test]
fn thumbnails_count_sites_and_bonds_per_cell_by_hand() {
    let backend = AppBackend::new();
    // bonds per cell = sites x coordination / 2: square 1x4/2, triangular 1x6/2, honeycomb 2x3/2, kagome 3x4/2
    for (key, sites, bonds) in [("square", 1, 2), ("triangular", 1, 3), ("honeycomb", 2, 3), ("kagome", 3, 6)] {
        let t0 = std::time::Instant::now();
        let g = ThumbGeom::build(named(&backend, key));
        println!("# time thumb {key}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        assert_eq!((g.sites_per_cell, g.bonds_per_cell), (sites, bonds), "{key}");
        assert!(!g.lines.is_empty() && !g.sites.is_empty(), "{key} preview has geometry");
    }
}

#[test]
fn thumb_cache_caps_builds_per_frame_and_reuses_entries() {
    let backend = AppBackend::new();
    let items = backend.named_lattices();
    assert!(items.len() > BUILDS_PER_FRAME);
    let mut cache = LatticeThumbCache::default();
    cache.begin_frame(1);
    for it in &items[..BUILDS_PER_FRAME] {
        assert!(cache.get_or_build(it).is_some());
    }
    assert!(cache.get_or_build(&items[BUILDS_PER_FRAME]).is_none(), "cap reached in this frame");
    assert!(cache.deferred);
    assert!(cache.get_or_build(&items[0]).is_some(), "cached entries are free");
    cache.begin_frame(2);
    assert!(!cache.deferred);
    assert!(cache.get_or_build(&items[BUILDS_PER_FRAME]).is_some());
    assert_eq!(cache.len(), BUILDS_PER_FRAME + 1);
}

#[test]
fn select_external_moves_mode_rule_group_and_selection_together() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let target = backend.catalogue_for_group("p4g", CatalogueRuleUi::Sheet)[2].clone();
    let rev0 = st.revision();
    st.select_external(target.clone(), &backend, &mut cv);
    assert_eq!(st.mode, BrowseMode::ByGroup);
    assert_eq!(st.rule, CatalogueRuleUi::Sheet);
    assert_eq!(backend.groups()[st.selected_group_idx].symbol, "p4g");
    assert_eq!(st.selected_cat_idx, 2);
    assert_eq!(st.base_item().alias_key, target.alias_key);
    assert!(st.revision() > rev0, "a new lattice bumps the revision");

    let honey = named(&backend, "honeycomb").clone();
    st.select_external(honey, &backend, &mut cv);
    assert_eq!(st.mode, BrowseMode::Named21);
    assert_eq!(st.active_item().named_key.as_deref(), Some("honeycomb"));
    assert_eq!(st.active_item().group, "p6m");
}

#[test]
fn rule_index_lists_every_group_once_per_rule() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    for rule in CatalogueRuleUi::ALL {
        let t0 = std::time::Instant::now();
        let n = st.rule_index(rule, &backend).len();
        println!("# time rule_index {}: {:.1} ms", rule.slug(), t0.elapsed().as_secs_f64() * 1e3);
        let expected: usize = backend.groups().iter().map(|g| g.counts[rule.index()]).sum();
        assert_eq!(n, expected, "{}", rule.slug());
        let groups: BTreeSet<&str> = st.rule_index_ready(rule).iter().map(|it| it.group.as_str()).collect();
        assert_eq!(groups.len(), 17, "{}", rule.slug());
    }
}

#[test]
fn card_grid_builds_previews_only_for_cards_on_screen() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    st.select_named(0, &backend, &mut cv);
    let ctx = egui::Context::default();
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 420.0))),
        ..Default::default()
    };
    for _ in 0..3 {
        let _ = ctx.run_ui(input(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let _ = st.show_left_panel(ui, &backend, &mut cv);
            });
        });
    }
    let built = st.thumbs.len();
    assert!(built >= 1, "visible cards get previews");
    assert!(built < 21, "only cards on screen are built, got {built} of 21");
}

// ---- background parameter moves (slow families) ----

fn straight_p1(backend: &AppBackend, cv: &mut crate::canvas2d::Canvas2D) -> GalleryTabState {
    let mut st = GalleryTabState::new(backend, cv);
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("straight_p1")).unwrap();
    st.select_named(idx, backend, cv);
    assert!(st.sliders().len() >= 10, "straight_p1 is the many-slider family");
    st
}

#[test]
fn background_move_gives_the_same_lattice_as_the_synchronous_move() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    let (mut cv_a, mut cv_s) = (canvas(), canvas());
    let mut asy = straight_p1(&backend, &mut cv_a);
    let mut syn = straight_p1(&backend, &mut cv_s);
    let s = asy.sliders()[0].clone();
    let raw = s.value + 2.0 * s.step;
    let rev0 = asy.revision();
    let t0 = std::time::Instant::now();
    asy.drag_param_async(&s.name, raw, &ctx);
    println!("# time submit background move: {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
    assert!(asy.retune_pending());
    assert_eq!(asy.pending_value(&s.name), Some(raw), "the slider shows the queued value");
    assert!(asy.wait_retune(&mut cv_a), "the move changes the lattice");
    syn.drag_param(&s.name, raw, &mut cv_s);
    assert!(!asy.retune_pending() && asy.pending_value(&s.name).is_none());
    assert_eq!(asy.active_item().recipe, syn.active_item().recipe);
    assert_eq!(asy.param_values(), syn.param_values());
    assert_eq!(asy.revision(), rev0 + 1);
}

#[test]
fn moves_queued_while_a_job_runs_are_all_applied_and_a_sync_move_cancels_them() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    let (mut cv_a, mut cv_s) = (canvas(), canvas());
    let mut asy = straight_p1(&backend, &mut cv_a);
    let mut syn = straight_p1(&backend, &mut cv_s);
    let (a, b) = (asy.sliders()[0].clone(), asy.sliders()[1].clone());
    // two sliders moved back to back, before the first result can land
    asy.drag_param_async(&a.name, a.value + 2.0 * a.step, &ctx);
    asy.drag_param_async(&b.name, b.value - 2.0 * b.step, &ctx);
    let _ = asy.wait_retune(&mut cv_a);
    syn.drag_param(&a.name, a.value + 2.0 * a.step, &mut cv_s);
    syn.drag_param(&b.name, b.value - 2.0 * b.step, &mut cv_s);
    assert_eq!(asy.param_values(), syn.param_values(), "neither move is lost");

    // a synchronous move (typed value) first applies a queued drag of another slider, then itself
    asy.drag_param_async(&a.name, a.value + 6.0 * a.step, &ctx);
    let _ = asy.set_param(&b.name, b.value, &mut cv_a).unwrap();
    assert!(!asy.retune_pending());
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!asy.poll_retune(Some(&mut cv_a)), "the superseded job is never applied");
    syn.drag_param(&a.name, a.value + 6.0 * a.step, &mut cv_s);
    let _ = syn.set_param(&b.name, b.value, &mut cv_s).unwrap();
    assert_eq!(asy.param_values(), syn.param_values(), "the queued drag of the other slider is kept");
}

#[test]
fn queued_moves_replay_in_order_across_sliders() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    let (mut cv_a, mut cv_s) = (canvas(), canvas());
    let mut asy = straight_p1(&backend, &mut cv_a);
    let mut syn = straight_p1(&backend, &mut cv_s);
    let (a, b) = (asy.sliders()[0].clone(), asy.sliders()[1].clone());
    let moves = [(a.name.clone(), a.value + 2.0 * a.step), (b.name.clone(), b.value + 2.0 * b.step), (a.name.clone(), a.value + 4.0 * a.step)];
    for (n, v) in &moves {
        asy.drag_param_async(n, *v, &ctx);
    }
    let _ = asy.wait_retune(&mut cv_a);
    for (n, v) in &moves {
        syn.drag_param(n, *v, &mut cv_s);
    }
    assert_eq!(asy.param_values(), syn.param_values(), "a, b, a replays as a, b, a");
    assert_eq!(asy.active_item().recipe, syn.active_item().recipe);
}

#[test]
fn rename_while_a_move_is_pending_keeps_the_new_name() {
    let mut backend = AppBackend::new();
    let ctx = egui::Context::default();
    let mut cv = canvas();
    let mut st = straight_p1(&backend, &mut cv);
    let s = st.sliders()[0].clone();
    st.drag_param_async(&s.name, s.value + 2.0 * s.step, &ctx);
    st.rename_active_lattice("My Ladder", &mut backend, &mut cv);
    let _ = st.wait_retune(&mut cv);
    assert_eq!(st.active_item().label, "My Ladder");
}

#[test]
fn switching_between_same_recipe_entries_bumps_the_revision() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let named_kag = named(&backend, "kagome").clone();
    let list = backend.catalogue_for_group("p6m", CatalogueRuleUi::Bonds);
    let Some(cat_kag) = list.iter().find(|it| it.named_key.as_deref() == Some("kagome")).cloned() else {
        return; // no stored bonds entry matched to kagome: nothing to compare
    };
    st.select_external(named_kag, &backend, &mut cv);
    let rev = st.revision();
    st.select_external(cat_kag.clone(), &backend, &mut cv);
    assert_eq!(st.active_item().alias_key, cat_kag.alias_key);
    assert!(st.revision() > rev, "another catalogue entry is another active item");
}

#[test]
fn numbers_outside_1_to_17_are_text_and_ita_may_be_spaced() {
    let backend = AppBackend::new();
    let kag = named(&backend, "kagome");
    assert!(query_matches("ITA 17", kag), "spaced form of ita17");
    assert!(!query_matches("ITA 16", kag));
    let mut run = kag.clone();
    run.label = String::from("Sample 42");
    assert!(query_matches("42", &run), "42 is not an ITA number: searched as text");
    assert!(!query_matches("+17", kag) || kag.label.contains("+17"), "'+17' is not an ITA number");
}

#[test]
fn the_named_lattice_highlights_its_catalogue_card_and_vice_versa() {
    use super::browser::is_selected;
    let backend = AppBackend::new();
    let kag = named(&backend, "kagome");
    let list = backend.catalogue_for_group("p6m", CatalogueRuleUi::Bonds);
    let cat = list.iter().find(|it| it.named_key.as_deref() == Some("kagome")).expect("p6m bonds list holds kagome");
    assert!(is_selected(kag, cat) && is_selected(cat, kag));
    let honey = named(&backend, "honeycomb");
    assert!(!is_selected(kag, honey));
    let other = list.iter().find(|it| it.named_key.is_none()).expect("an unnamed p6m entry");
    assert!(!is_selected(kag, other) && !is_selected(other, kag), "two unnamed entries are distinct by alias key only");
    // at startup the Gallery shows the p6m list with the named kagome selected: its card is highlighted
    let mut cv = canvas();
    let st = GalleryTabState::new(&backend, &mut cv);
    assert!(st.catalogue().iter().any(|it| is_selected(st.base_item(), it)));
}

#[test]
fn thumbnail_outline_is_the_cell_in_the_middle_of_the_patch() {
    let backend = AppBackend::new();
    for key in ["square", "triangular", "honeycomb", "kagome", "oblique", "brick"] {
        let g = ThumbGeom::build(named(&backend, key));
        let c = [0.25 * g.cell.iter().map(|p| p[0]).sum::<f32>(), 0.25 * g.cell.iter().map(|p| p[1]).sum::<f32>()];
        let (lo, hi) = g.bbox;
        assert!(lo[0] <= c[0] && c[0] <= hi[0] && lo[1] <= c[1] && c[1] <= hi[1], "{key}: outline centre {c:?} outside patch {lo:?}..{hi:?}");
    }
}
