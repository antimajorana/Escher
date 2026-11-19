//! Gallery tab: every group x rule x entry and every named lattice, the exported item and its revision, and the
//! side panel drawn headlessly.

use std::collections::BTreeSet;

use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::Cell2D;

use super::python_cases::{AT_SPECIAL, MOVES};
use super::{BrowseMode, GalleryTabState};
use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::live_plane::{format_dsymbol_key, shared_catalogues};

pub(super) fn canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 1,
        show_unit_cell: false,
    })
}

/// One headless frame of the side panel (no input).
pub(super) fn frame(ctx: &egui::Context, st: &mut GalleryTabState, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
    let mut changed = false;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        changed = st.show_side_panel(ui, backend, canvas);
    });
    changed
}

/// Select `rule` / `group` / entry `who` (an index) or the named lattice `who`.
pub(super) fn select(st: &mut GalleryTabState, backend: &AppBackend, canvas: &mut Canvas2D, rule: &str, group: &str, who: &str) {
    if rule == "named" {
        let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some(who)).unwrap();
        st.select_named(idx, backend, canvas);
    } else {
        let r = CatalogueRuleUi::ALL.into_iter().find(|r| r.matches(rule)).unwrap();
        st.select_rule(r, backend, canvas);
        st.select_group_by_symbol(group, backend, canvas);
        st.select_catalogue_entry(who.parse().unwrap(), backend, canvas);
    }
    assert_eq!(st.base_item().group, group, "{rule} {group} {who}");
}

fn letters_marked(st: &GalleryTabState, backend: &AppBackend) -> BTreeSet<char> {
    let it = st.active_item();
    backend.wyckoff_table(&it.group, &it.occupied_wyckoff).iter().filter(|w| w.occupied).map(|w| w.letter).collect()
}

fn check_default_readout(st: &GalleryTabState, item: &CatalogueItemUi, ctx: &str) {
    let ro = &st.readout;
    assert!(ro.status_ok, "{ctx}: {}", ro.status_message);
    assert_eq!(ro.active_group, item.group, "{ctx}");
    assert_eq!(ro.dsymbol_key, item.dsymbol_key, "{ctx}");
    assert_eq!(ro.network_key, item.network_key, "{ctx}");
    assert_eq!(ro.symmetry_index, item.symmetry_index, "{ctx}");
    assert_eq!(st.active_item().recipe, item.recipe, "{ctx}: default values export the stored recipe");
}

#[test]
fn every_group_rule_entry_and_named_lattice_switches_cleanly() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let cats = shared_catalogues();
    let ctx = egui::Context::default();
    let mut n_entries = 0;
    for rule in CatalogueRuleUi::ALL {
        st.select_rule(rule, &backend, &mut cv);
        for (g_idx, g) in backend.groups().iter().enumerate() {
            // start from the last entry of the previous list: a stale index must not survive the switch
            let last = st.catalogue().len().saturating_sub(1);
            st.select_catalogue_entry(last, &backend, &mut cv);
            st.select_group(g_idx, &backend, &mut cv);
            assert_eq!((st.selected_group_idx, st.selected_cat_idx, st.mode), (g_idx, 0, BrowseMode::ByGroup));
            assert_eq!(st.catalogue().len(), g.counts[rule.index()], "{} {}", rule.label(), g.symbol);
            let stored = match rule {
                CatalogueRuleUi::Points => &cats.points,
                CatalogueRuleUi::Sheet => &cats.sheet,
                CatalogueRuleUi::Bonds => &cats.bonds,
            };
            assert_eq!(stored[g.symbol].all.len(), st.catalogue().len());
            for (k, entry) in stored[g.symbol].all.iter().enumerate() {
                st.select_catalogue_entry(k, &backend, &mut cv);
                let item = st.catalogue()[k].clone();
                let tag = format!("{} {} #{}", rule.label(), g.symbol, k + 1);
                assert_eq!(st.base_item().label, item.label, "{tag}");
                check_default_readout(&st, &item, &tag);
                // the key stored in the data file by Python
                if let Some(key) = &entry.stored_key {
                    assert_eq!(st.readout.dsymbol_key, format_dsymbol_key(key), "{tag}");
                }
                let occupied: BTreeSet<char> = item.occupied_wyckoff.iter().copied().collect();
                assert!(!occupied.is_empty(), "{tag}");
                assert_eq!(letters_marked(&st, &backend), occupied, "{tag}");
                assert_eq!(st.sliders().len(), LatticeFamily::from_recipe(&item.recipe, None).unwrap().names().len(), "{tag}");
                n_entries += 1;
            }
            st.select_catalogue_entry(usize::MAX, &backend, &mut cv); // ignored, no panic
            frame(&ctx, &mut st, &backend, &mut cv);
        }
    }
    assert_eq!(n_entries, 533);
    for (k, nl) in cats.named.named.iter().enumerate() {
        st.select_named(k, &backend, &mut cv);
        let item = backend.named_lattices()[k].clone();
        check_default_readout(&st, &item, &nl.key);
        assert_eq!(st.readout.dsymbol_key, format_dsymbol_key(&nl.stored_dsymbol_key), "{}", nl.key);
        assert_eq!(st.readout.network_key, format_dsymbol_key(&nl.stored_network_key), "{}", nl.key);
        assert_eq!(st.readout.symmetry_index, nl.stored_symmetry_index, "{}", nl.key);
        assert_eq!(st.active_item().frame, nl.frame, "{}", nl.key);
        frame(&ctx, &mut st, &backend, &mut cv);
    }
    st.select_named(999, &backend, &mut cv); // ignored
    st.select_group_by_symbol("p7", &backend, &mut cv); // unknown group: ignored
    assert_eq!(st.mode, BrowseMode::Named21);
}

#[test]
fn drawing_the_panel_never_moves_a_parameter() {
    // egui's Slider with step_by / fixed_decimals / clamping rewrites the value on the first frame (b = 1.154 of
    // bonds p1 #1 became 1.15). Any frame without input must leave values, item and revision alone.
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let ctx = egui::Context::default();
    for (rule, group, who) in [("bonds", "p1", "0"), ("points", "p1", "0"), ("named", "p4", "chiral_square"), ("named", "pmg", "shastrys_ladder"), ("named", "p2", "oblique")] {
        select(&mut st, &backend, &mut cv, rule, group, who);
        let (vals, rev) = (st.param_values(), st.revision());
        for _ in 0..3 {
            assert!(!frame(&ctx, &mut st, &backend, &mut cv));
        }
        assert_eq!(st.param_values(), vals, "{rule} {group} {who}");
        assert_eq!(st.revision(), rev, "{rule} {group} {who}");
    }
    // shastrys_ladder has b = 4.0 outside the nominal [0.4, 3.0] (sheet.named("shastrys_ladder").values): kept,
    // the slider range is widened to it (ipywidgets would clamp it to 3.0 when the widget is built)
    select(&mut st, &backend, &mut cv, "named", "pmg", "shastrys_ladder");
    frame(&ctx, &mut st, &backend, &mut cv);
    let b = st.sliders().iter().find(|s| s.name == "b").unwrap();
    assert!((b.value - 4.0).abs() < 1e-12 && b.allowed_max >= b.value && b.nominal_max >= b.value, "{b:?}");
}

#[test]
fn slider_moves_reach_active_item_and_bump_revision_but_view_changes_do_not() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    assert_eq!(st.revision(), 0);
    select(&mut st, &backend, &mut cv, "bonds", "p4g", "0");
    let base = st.base_item().clone();
    let rev = st.revision();
    assert!(rev > 0);
    // view-only changes
    st.set_repeat(5, &mut cv);
    st.set_show_unit_cell(false, &mut cv);
    assert_eq!(st.revision(), rev);

    // p4g #1 (rotating squares): t1 = -1/2 is the square lattice, group p4m (tests/test_widgets.py)
    let t1 = st.drag_param("t1", -0.5, &mut cv).unwrap();
    assert_eq!(t1, -0.5);
    assert_eq!(st.revision(), rev + 1);
    let it = st.active_item().clone();
    assert_ne!(it.recipe, base.recipe);
    let fam = LatticeFamily::from_recipe(&it.recipe, Some(it.frame)).unwrap();
    assert!((fam.values()["t1"] + 0.5).abs() < 1e-12);
    assert_eq!((it.group.as_str(), it.label.as_str()), (base.group.as_str(), base.label.as_str()));
    assert!(!st.readout.status_ok && st.readout.active_group == "p4m");
    assert!(st.readout.status_message.contains("special value t1 = -0.5"), "{}", st.readout.status_message);
    // the same value again: nothing changes
    st.drag_param("t1", -0.5, &mut cv);
    assert_eq!(st.revision(), rev + 1);
    // beyond the upper limit: clamped to it (test_widget_limits_bar_and_snap)
    let hi = st.sliders()[0].allowed_max;
    assert_eq!(st.drag_param("t1", hi + 0.3, &mut cv), Some(hi));
    assert!(st.readout.status_ok, "{}", st.readout.status_message);
    assert_eq!(st.revision(), rev + 2);
    assert_eq!(st.drag_param("t1", f64::NAN, &mut cv), None);
    assert_eq!(st.drag_param("nope", 0.0, &mut cv), None);
    assert_eq!(st.revision(), rev + 2);
    // Reset: the stored recipe again
    st.reset_params(&mut cv);
    assert_eq!(st.active_item().recipe, base.recipe);
    assert_eq!(st.revision(), rev + 3);
    st.reset_params(&mut cv);
    assert_eq!(st.revision(), rev + 3);
}

#[test]
fn readout_group_and_status_match_python_at_special_values_and_after_moves() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let t0 = std::time::Instant::now();
    let mut n_moves = 0usize;
    for (table, name) in [(AT_SPECIAL, "special"), (MOVES, "move")] {
        for &(rule, group, who, param, value, py_group, py_ok) in table {
            select(&mut st, &backend, &mut cv, rule, group, who);
            let rev = st.revision();
            let v = st.set_param(param, value, &mut cv).unwrap_or_else(|| panic!("{rule} {group} {who} {param}"));
            assert!((v - value).abs() < 1e-8, "{name} {rule} {group} {who}: {param} = {v}, Python {value}");
            let ctx = format!("{name} {rule} {group} {who} {param} = {value}: {}", st.readout.status_message);
            assert_eq!(st.readout.active_group, py_group, "{ctx}");
            assert_eq!(st.readout.status_ok, py_ok, "{ctx}");
            if !py_ok {
                assert!(st.readout.status_message.starts_with(&format!("group is now {py_group}")), "{ctx}");
            }
            // the other tabs get this lattice: its recipe rebuilds the family at these values
            assert!(st.revision() > rev, "{ctx}");
            let it = st.active_item();
            let fam = LatticeFamily::from_recipe(&it.recipe, Some(it.frame)).unwrap();
            assert!((fam.values()[param] - v).abs() < 1e-9, "{ctx}");
            n_moves += 1;
        }
    }
    let per_move = t0.elapsed().as_secs_f64() / n_moves as f64;
    println!("# time Gallery select + one parameter move (sliders refresh + status): {per_move:.4} s on average");
}

#[test]
fn time_one_drag_on_the_largest_families() {
    // as tests/test_widgets.py::test_time_one_move (loose bound: only catches a blow-up)
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    for (rule, group, who) in [("named", "p1", "straight_p1"), ("points", "p1", "13"), ("bonds", "p4g", "0")] {
        select(&mut st, &backend, &mut cv, rule, group, who);
        let s = st.sliders().iter().find(|s| s.name.starts_with('x') || s.name.starts_with('t')).unwrap().clone();
        let t0 = std::time::Instant::now();
        for k in 1..=5 {
            st.drag_param(&s.name, s.value + 2.0 * f64::from(k) * s.step, &mut cv);
        }
        let dt = t0.elapsed().as_secs_f64() / 5.0;
        println!("# time one drag of {} on {rule} {group} {who} ({} sliders): {dt:.4} s", s.name, st.sliders().len());
        assert!(dt < 2.0);
    }
}

#[test]
fn tuned_named_lattice_keeps_its_frame_and_draws_what_it_exports() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    select(&mut st, &backend, &mut cv, "named", "p4g", "shastry_sutherland");
    let base = st.base_item().clone();
    assert_eq!(base.frame.origin, [0.5, -0.5]); // stored frame (sheet.named("shastry_sutherland").family.frame)
    let v = st.drag_param("t1", -0.1, &mut cv).unwrap();
    assert!((v + 0.1).abs() < 1e-9);
    let it = st.active_item();
    assert_eq!(it.frame, base.frame);
    let exported = LatticeFamily::from_recipe(&it.recipe, Some(it.frame)).unwrap().cell();
    let shown = st.cell();
    assert_eq!((exported.a1, exported.a2), (shown.a1, shown.a2));
    assert_eq!((exported.basis.len(), exported.bonds.len()), (shown.basis.len(), shown.bonds.len()));
    // the same sites up to lattice translations
    let a = [[shown.a1[0], shown.a2[0]], [shown.a1[1], shown.a2[1]]];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let frac = |p: [f64; 2]| [(a[1][1] * p[0] - a[0][1] * p[1]) / det, (-a[1][0] * p[0] + a[0][0] * p[1]) / det];
    for &p in &exported.basis {
        assert!(shown.basis.iter().any(|&q| {
            let d = frac([p[0] - q[0], p[1] - q[1]]);
            (d[0] - d[0].round()).abs() < 1e-9 && (d[1] - d[1].round()).abs() < 1e-9
        }));
    }
    assert!(st.readout.status_ok, "{}", st.readout.status_message);
}

#[test]
fn stage1_rule_defaults_naming_alias_slugs_and_chirality_badge_update_live() {
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    assert_eq!(st.rule, CatalogueRuleUi::Bonds);
    assert_eq!(CatalogueRuleUi::ALL, [CatalogueRuleUi::Bonds, CatalogueRuleUi::Sheet, CatalogueRuleUi::Points]);
    assert_eq!(CatalogueRuleUi::Bonds.label(), "Standard lattice");
    assert_eq!(CatalogueRuleUi::Sheet.label(), "Atomic crystals");
    assert_eq!(CatalogueRuleUi::Points.label(), "Delaunay lattice");
    assert_eq!(CatalogueRuleUi::Bonds.slug(), "bonds");
    assert_eq!(CatalogueRuleUi::Sheet.slug(), "sheet");
    assert_eq!(CatalogueRuleUi::Points.slug(), "points");

    let p4g_bonds = backend.catalogue_for_group("p4g", CatalogueRuleUi::Bonds);
    assert_eq!(p4g_bonds[0].alias_key.rule, "bonds");

    for key in ["chiral_square", "chiral_kagome", "chiral_triangular"] {
        let it = backend.find_named(key).unwrap();
        assert!(it.chiral, "{key} must have geometric chiral=true");
        assert!(!it.topologically_chiral, "{key} bare graph is achiral");
    }
    for key in ["square", "kagome", "honeycomb", "shastry_sutherland", "brick_pg"] {
        let it = backend.find_named(key).unwrap();
        assert!(!it.chiral, "{key} must be achiral");
        assert!(!it.topologically_chiral);
    }

    let p6_pts = backend.catalogue_for_group("p6", CatalogueRuleUi::Points);
    assert!(p6_pts[0].chiral);
    assert!(p6_pts[0].topologically_chiral, "p6 #1 (snub hexagonal 3.3.3.3.6) must be topologically chiral");

    // Moving oblique (p2, geometrically chiral) onto b=1, gamma=90 (p4m square special locus) flips chiral -> false.
    select(&mut st, &backend, &mut cv, "named", "p2", "oblique");
    assert!(st.readout.chiral);
    st.set_param("b", 1.0, &mut cv).unwrap();
    st.set_param("gamma", 90.0, &mut cv).unwrap();
    assert!(!st.readout.chiral, "square special locus must flip readout.chiral to false");
    assert_eq!(st.readout.active_group, "p4m");
}

#[test]
fn shell_inflation_toggle_kagome_produces_exact_shells_and_keeps_revision() {
    use papertilling_core::plane::inflation::{CentreSet, InflationRule, InflationSeed, PlaneTiling};

    use crate::scene2d::overlays::GalleryOverlayToggles;

    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let ctx = egui::Context::default();
    select(&mut st, &backend, &mut cv, "named", "p6m", "kagome");
    assert!(!st.inflation.enabled);
    assert!(matches!(cv.scene(), Scene2D::PeriodicCell { .. }));
    let rev = st.revision();

    // Hand counts for kagome cluster shells S_n (n = 0, 1, 2):
    // H_n = 3n(n+1)+1 = 1, 7, 19 hexagons; T_n = 6(n+1)^2 = 6, 24, 54 triangles => F_n = 7, 31, 73.
    // Every kagome edge belongs to exactly one of the T_2 = 54 edge-disjoint triangles => E_2 = 3 * 54 = 162,
    // and Euler's disk formula V - E + F = 1 gives V_2 = 162 + 1 - 73 = 90.
    st.set_show_unit_cell(false, &mut cv);
    st.inflation.rule = InflationRule::Auto;
    st.inflation.seed = InflationSeed::Dirichlet;
    st.inflation.steps = 2;
    st.set_inflation_enabled(true, &mut cv);
    assert_eq!(st.revision(), rev);
    assert!(st.inflation.last_error.is_none(), "{:?}", st.inflation.last_error);
    assert_eq!(st.inflation.last_shells.iter().map(|s| s.topology.faces).collect::<Vec<_>>(), vec![7, 31, 73]);
    match cv.scene() {
        Scene2D::VectorScene { primitives } => {
            assert_eq!(primitives.polygons.len(), 73);
            assert_eq!(primitives.site_count, 90);
            assert_eq!(primitives.bond_count, 162);
        }
        _ => panic!("expected VectorScene when shell inflation is enabled"),
    }

    // Explicit Cluster rule + hexagon Face seed + unit cell + ITA/Voronoi overlays.
    let hex = PlaneTiling::new(st.cell()).unwrap().face_sizes().iter().position(|&k| k == 6).unwrap();
    st.inflation.rule = InflationRule::Cluster(CentreSet::MaxPolygon);
    st.inflation.seed = InflationSeed::Face(hex);
    st.set_show_unit_cell(true, &mut cv);
    st.set_overlays(GalleryOverlayToggles { symmetry: true, voronoi: true }, &mut cv);
    assert_eq!(st.revision(), rev);
    assert_eq!(st.inflation.last_shells.iter().map(|s| s.topology.faces).collect::<Vec<_>>(), vec![7, 31, 73]);
    assert!(cv.gallery_overlay.is_some());
    match cv.scene() {
        Scene2D::VectorScene { primitives } => {
            assert_eq!(primitives.polygons.len(), 74); // 73 shell faces + 1 unit-cell parallelogram
            assert_eq!(primitives.site_count, 90);
            assert_eq!(primitives.bond_count, 162);
        }
        _ => panic!("expected VectorScene with unit cell and overlays"),
    }
    frame(&ctx, &mut st, &backend, &mut cv);

    // Switching back to Periodic grid restores PeriodicCell without bumping revision.
    st.set_inflation_enabled(false, &mut cv);
    assert!(matches!(cv.scene(), Scene2D::PeriodicCell { .. }));
    assert_eq!(st.revision(), rev);
}

#[test]
fn shell_inflation_error_falls_back_to_periodic_cell_without_panicking() {
    use papertilling_core::plane::inflation::InflationSeed;

    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    let ctx = egui::Context::default();
    select(&mut st, &backend, &mut cv, "named", "p6m", "kagome");
    let rev = st.revision();

    st.inflation.seed = InflationSeed::Face(999);
    st.set_inflation_enabled(true, &mut cv);
    assert!(st.inflation.last_error.is_some());
    assert!(st.inflation.last_shells.is_empty());
    assert!(matches!(cv.scene(), Scene2D::PeriodicCell { .. }));
    frame(&ctx, &mut st, &backend, &mut cv);

    st.inflation.seed = InflationSeed::Dirichlet;
    st.inflation.steps = 999;
    st.sync_scene(&mut cv, true);
    assert!(st.inflation.last_error.is_some());
    assert!(matches!(cv.scene(), Scene2D::PeriodicCell { .. }));
    assert_eq!(st.revision(), rev);
}
