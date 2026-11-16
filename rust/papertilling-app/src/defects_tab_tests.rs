//! Defects tab end-to-end: catalogue entries of the 17 first lattices and the 21 named lattices are built the
//! way the tab builds them, and the picture, the readout and the side panel are checked.
//!
//! The full sweep (every entry of all 38 lattices, ~1300 builds) is `#[ignore]`d (minutes); run it with
//! `cargo test --release -p papertilling-app every_entry -- --ignored --nocapture`.

use std::collections::HashSet;
use std::time::Instant;

use papertilling_core::types::BuiltCrystal;

use super::{DefectSubTab, DefectsTabState};
use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi, DefectSelectionUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::live_ops::defect_context;
use crate::scene2d::compile_scene;
use crate::scene_crystal::ghost_bonds;

pub(crate) fn app_lattices(backend: &AppBackend) -> Vec<CatalogueItemUi> {
    let mut out: Vec<CatalogueItemUi> = backend
        .groups()
        .iter()
        .filter_map(|g| backend.catalogue_for_group(g.symbol, CatalogueRuleUi::Points).into_iter().next())
        .collect();
    out.extend(backend.named_lattices().iter().cloned());
    out
}

pub(crate) fn empty_crystal() -> BuiltCrystal {
    BuiltCrystal {
        pos: Vec::new(),
        ref_pos: Vec::new(),
        basis_idx: Vec::new(),
        side: Vec::new(),
        cone_sites: Vec::new(),
        edges: Vec::new(),
        adj: Vec::new(),
        wall_sigma: None,
        is_non_orientable: false,
    }
}

fn finite(c: &BuiltCrystal) -> bool {
    c.pos.iter().all(|p| p[0].is_finite() && p[1].is_finite())
}

fn near(a: [f32; 2], b: [f64; 2]) -> bool {
    (f64::from(a[0]) - b[0]).hypot(f64::from(a[1]) - b[1]) < 1e-4
}

/// The picture: finite bbox, sites drawn, a title; ghost bonds drawn as two stubs of half their reference length
/// from their own end (`drawing_check.drawn`), never as a full segment.
fn check_scene(c: &BuiltCrystal, title: &str, note: &str, ctx: &str) {
    let ghosts = ghost_bonds(c);
    for (u, nbrs) in c.adj.iter().enumerate() {
        for d in nbrs {
            if let Some(g) = d.ghost_stub {
                let len = (g[0] - c.pos[u][0]).hypot(g[1] - c.pos[u][1]);
                let half = 0.5 * d.vec[0].hypot(d.vec[1]);
                assert!((len - half).abs() < 1e-6, "{ctx}: stub {u}->{} has length {len}, half bond {half}", d.to);
            }
        }
    }
    let scene = Scene2D::CrystalPatch { crystal: c.clone(), title: title.to_owned(), note: note.to_owned() };
    let prims = compile_scene(&scene);
    let (lo, hi) = prims.bbox;
    assert!(lo.iter().chain(&hi).all(|v| v.is_finite()) && lo[0] <= hi[0] && lo[1] <= hi[1], "{ctx}: bbox {lo:?} {hi:?}");
    assert!(prims.site_count > 0 && !prims.circles.is_empty(), "{ctx}: blank canvas");
    assert!(!prims.title.is_empty(), "{ctx}: no title");
    let solid = prims.lines.iter().filter(|l| !l.dashed).count();
    let dashed = prims.lines.iter().filter(|l| l.dashed).count();
    assert_eq!(solid, c.edges.len() - ghosts.len(), "{ctx}: plain bonds drawn");
    assert_eq!(dashed, 2 * ghosts.len(), "{ctx}: two stubs per ghost bond");
    for &(u, v) in &ghosts {
        let full = prims.lines.iter().any(|l| (near(l.a, c.pos[u]) && near(l.b, c.pos[v])) || (near(l.a, c.pos[v]) && near(l.b, c.pos[u])));
        assert!(!full, "{ctx}: ghost bond {u}-{v} drawn in full");
    }
}

/// Expected first words of the charge readout, from the catalogue (Python-checked, `defect_catalogue_fixture.rs`).
fn expected_charge(item: &CatalogueItemUi, sel: DefectSelectionUi) -> Option<(String, Option<f64>)> {
    let ctx = defect_context(item).ok()?;
    match sel {
        DefectSelectionUi::Dislocation(i) => {
            let d = ctx.cat.dislocations.get(i)?;
            Some((String::from("Charge: dislocation | Frank = +0.0°"), Some(d.length)))
        }
        DefectSelectionUi::Disclination(i) => {
            let d = ctx.cat.disclinations.get(i)?;
            Some((format!("Charge: disclination | Frank = {:+.1}°", d.angle), None))
        }
        DefectSelectionUi::Wall(_) => None,
    }
}

/// One build through the backend as the tab does it; returns seconds.
fn build_and_check(backend: &AppBackend, item: &CatalogueItemUi, sel: DefectSelectionUi, radius: f64, relaxed: bool) -> f64 {
    let t = Instant::now();
    let (c, r) = backend.build_defect(item, &sel, radius, relaxed);
    let secs = t.elapsed().as_secs_f64();
    let ctx = format!("{} {sel:?} R={radius} relaxed={relaxed}", item.label);
    assert!(!c.pos.is_empty() && finite(&c), "{ctx}: empty or non-finite crystal");
    assert_eq!(c.pos.len(), c.adj.len(), "{ctx}: adj length");
    assert!(c.edges.iter().all(|&(u, v)| u < v && v < c.pos.len()), "{ctx}: edge index");
    assert!(!r.title.is_empty() && !r.charge_summary.is_empty() && !r.core_rings.is_empty(), "{ctx}: empty readout");
    if let Some((want, b_len)) = expected_charge(item, sel) {
        assert!(r.charge_summary.starts_with(&want), "{ctx}: readout {:?}, expected {want:?} ({})", r.charge_summary, r.note);
        if let Some(len) = b_len {
            let b = papertilling_core::defect::charge::total_charge(&c).expect("charge").burgers.expect("b");
            assert!((b[0].hypot(b[1]) - len).abs() < 1e-6, "{ctx}: |b| {b:?} != catalogue {len}");
        }
        assert!(r.core_rings.starts_with("core"), "{ctx}: core readout {:?} says nothing of the core", r.core_rings);
    }
    check_scene(&c, &r.title, &r.note, &ctx);
    secs
}

fn all_selections(backend: &AppBackend, item: &CatalogueItemUi) -> Vec<DefectSelectionUi> {
    let cat = backend.defect_catalogue(item);
    let mut sels: Vec<DefectSelectionUi> = (0..cat.dislocations.len()).map(DefectSelectionUi::Dislocation).collect();
    sels.extend((0..cat.disclinations.len()).map(DefectSelectionUi::Disclination));
    sels.extend((0..cat.walls.len()).map(DefectSelectionUi::Wall));
    sels
}

fn sweep(keys: Option<&[&str]>) {
    let backend = AppBackend::new();
    let lats: Vec<CatalogueItemUi> = match keys {
        None => app_lattices(&backend),
        Some(k) => k.iter().map(|key| backend.find_named(key).unwrap().clone()).collect(),
    };
    let t0 = Instant::now();
    let (mut slowest, mut builds) = ((0.0, String::new()), 0usize);
    for item in &lats {
        let t = Instant::now();
        for sel in all_selections(&backend, item) {
            let s = build_and_check(&backend, item, sel, 3.8, true);
            builds += 1;
            if s > 2.0 {
                println!("#   slow: {:.2} s {} {sel:?}", s, item.label);
            }
            if s > slowest.0 {
                slowest = (s, format!("{} {sel:?}", item.label));
            }
        }
        println!("# time {} ({}): {builds} builds so far, {:.2} s", item.label, item.group, t.elapsed().as_secs_f64());
    }
    println!("# time {builds} builds: {:.1} s; slowest {:.2} s ({})", t0.elapsed().as_secs_f64(), slowest.0, slowest.1);
}

#[test]
#[ignore = "full sweep of ~1300 builds (minutes); run by hand"]
fn every_entry_of_38_lattices_builds_with_a_readable_picture() {
    assert_eq!(app_lattices(&AppBackend::new()).len(), 38);
    sweep(None);
}

#[test]
fn every_entry_of_six_named_lattices_builds_with_a_readable_picture() {
    sweep(Some(&["square", "kagome", "honeycomb", "shastry_sutherland", "chiral_kagome", "brick"]));
}

#[test]
fn every_mobius_seam_of_38_lattices_keeps_stubs_and_draws_no_bond_across() {
    let backend = AppBackend::new();
    let mut seams = 0;
    for item in app_lattices(&backend) {
        let cat = backend.defect_catalogue(&item);
        for (i, w) in cat.walls.iter().enumerate().filter(|(_, w)| w.kind == crate::backend::WallKindUi::MobiusWc) {
            let (c, r) = backend.build_defect(&item, &DefectSelectionUi::Wall(i), 3.8, true);
            let ctx = format!("{} {}", item.label, w.id);
            assert!(c.is_non_orientable, "{ctx}: orientable");
            assert!(c.side.iter().all(Option::is_none) && c.wall_sigma.is_none(), "{ctx}: A|B sides on a seam");
            let ghosts: HashSet<_> = ghost_bonds(&c);
            let seam: HashSet<_> = c.edges.iter().copied().filter(|&(u, v)| c.adj[u].iter().any(|d| d.to == v && d.across)).collect();
            assert!(!seam.is_empty() && seam == ghosts, "{ctx}: the seam bonds are exactly the stub bonds");
            assert!(r.holonomy_detail.contains("across the Möbius seam") && r.core_rings.contains("orientable=false"), "{ctx}");
            check_scene(&c, &r.title, &r.note, &ctx);
            seams += 1;
        }
    }
    // Python's counts (papertilling-core/tests/fixtures/defect_catalogues.json, one seam per mirror/glide class):
    // first lattices pm 2, pg 2, cm 2, pmm 4, pmg 3, pgg 2, cmm 4, p4m 4, p4g 3, p3m1 2, p31m 2, p6m 4 = 34;
    // named elongated_triangular, square, triangular, kagome, honeycomb, rectangular, brick 4 each, fractal 2,
    // breathing 2, ladder 3, shastry 3, zigzag 3, sutherlands 2, ladder_pm 2, brick_pg 2 = 47.
    assert_eq!(seams, 34 + 47, "Möbius seams over the 38 lattices");
}

#[test]
fn radius_extremes_and_raw_pictures_on_every_kind() {
    let backend = AppBackend::new();
    let t0 = Instant::now();
    for key in ["square", "kagome", "shastry_sutherland", "straight_p1", "brick_pg", "chiral_triangular"] {
        let item = backend.find_named(key).unwrap().clone();
        let cat = backend.defect_catalogue(&item);
        let mut sels = vec![DefectSelectionUi::Dislocation(0), DefectSelectionUi::Disclination(0)];
        sels.extend((0..cat.walls.len()).filter(|&i| i == 0 || cat.walls[i].kind != cat.walls[i - 1].kind).map(DefectSelectionUi::Wall));
        for sel in sels {
            for radius in [2.4, 6.2] {
                for relaxed in [false, true] {
                    build_and_check(&backend, &item, sel, radius, relaxed);
                }
            }
        }
    }
    println!("# time radius extremes: {:.1} s", t0.elapsed().as_secs_f64());
}

#[test]
fn side_panel_runs_headless_on_every_sub_tab() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    for key in ["square", "straight_p1", "kagome", "shastry_sutherland"] {
        let item = backend.find_named(key).unwrap().clone();
        let mut st = DefectsTabState::new(&item, &backend);
        let mut canvas = Canvas2D::new(Scene2D::CrystalPatch { crystal: empty_crystal(), title: String::new(), note: String::new() });
        for sub in DefectSubTab::ALL {
            st.sub_tab = sub;
            st.rebuild_crystal(&item, &backend, &mut canvas, true);
            st.wait_idle(&mut canvas);
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| st.show_side_panel(ui, &item, &backend, &mut canvas));
            assert!(canvas.counts().0 > 0, "{key} {sub:?}: blank canvas");
            assert!(!st.readout.charge_summary.is_empty(), "{key} {sub:?}");
        }
    }
    // The lattice changes on another tab (reload without canvas): the next draw of the side panel shows the new
    // lattice's catalogue and readout, not the old ones.
    let (sq, kag) = (backend.find_named("square").unwrap().clone(), backend.find_named("kagome").unwrap().clone());
    let mut st = DefectsTabState::new(&sq, &backend);
    let mut canvas = Canvas2D::new(Scene2D::CrystalPatch { crystal: empty_crystal(), title: String::new(), note: String::new() });
    st.sub_tab = DefectSubTab::Walls;
    st.selected_wall = 12;
    st.rebuild_crystal(&sq, &backend, &mut canvas, true);
    st.wait_idle(&mut canvas);
    assert!(st.readout.title.starts_with("Square — Wd4"), "{}", st.readout.title);
    st.reload_for_item(&kag, &backend, &mut canvas, false);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| st.show_side_panel(ui, &kag, &backend, &mut canvas));
    st.wait_idle(&mut canvas);
    assert!(st.readout.title.starts_with("Kagomé — Wb1") && st.catalogue.walls.len() == 16, "{}", st.readout.title);

    // p1 has no rotation centre: the Disclinations sub-tab shows the reference disk and says why.
    let p1 = backend.find_named("straight_p1").unwrap().clone();
    let mut st = DefectsTabState::new(&p1, &backend);
    let mut canvas = Canvas2D::new(Scene2D::CrystalPatch { crystal: empty_crystal(), title: String::new(), note: String::new() });
    st.sub_tab = DefectSubTab::Disclinations;
    st.rebuild_crystal(&p1, &backend, &mut canvas, true);
    st.wait_idle(&mut canvas);
    assert!(st.readout.charge_summary.contains("no rotational centres") && canvas.counts().0 > 0);
}

#[test]
fn slow_repairs_submit_without_blocking_and_newest_lattice_wins() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let zigzag = backend.find_named("zigzag").unwrap();
    let mut st = DefectsTabState::new(square, &backend);
    let mut canvas = Canvas2D::new(Scene2D::CrystalPatch {
        crystal: empty_crystal(), title: String::new(), note: String::new(),
    });
    st.sub_tab = DefectSubTab::Disclinations;
    st.reload_for_item(zigzag, &backend, &mut canvas, true);
    st.wait_idle(&mut canvas);
    st.selected_disclin = 5; // measured worst repair in the full 38-lattice sweep
    let t = Instant::now();
    st.rebuild_crystal(zigzag, &backend, &mut canvas, false);
    let submit = t.elapsed().as_secs_f64();
    println!("# time Defects zigzag repair submission: {:.3} ms", submit * 1000.0);
    assert!(submit < 0.5 && st.is_busy(), "repair must not run on the UI thread");
    let Scene2D::CrystalPatch { note, .. } = canvas.scene() else { panic!("previous picture"); };
    assert!(note.contains("stale"));
    st.wait_idle(&mut canvas);
    assert!(st.error.is_none());
    // The asynchronous result retains exactly the backend's graph, positions and readout.
    let (expected, readout) = backend.build_defect(zigzag, &DefectSelectionUi::Disclination(5), st.radius, st.relaxed);
    let Scene2D::CrystalPatch { crystal, .. } = canvas.scene() else { panic!("repaired picture"); };
    assert_eq!(crystal, &expected);
    assert_eq!(st.readout.charge_summary, readout.charge_summary);

    st.reload_for_item(zigzag, &backend, &mut canvas, true);
    st.reload_for_item(square, &backend, &mut canvas, true);
    assert_eq!(canvas.counts().0, 0, "no old lattice under a new label");
    st.wait_idle(&mut canvas);
    assert!(st.readout.title.starts_with("Square"));
    assert_eq!(st.catalogue.disclinations.len(), backend.defect_catalogue(square).disclinations.len());
}
