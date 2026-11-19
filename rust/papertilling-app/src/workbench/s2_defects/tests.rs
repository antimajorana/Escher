//! S2 panel tests. Expected values are hand counts or published formulas, never values read back from the panel:
//! - Stone-Wales on graphene: four hexagons become two pentagons and two heptagons (5-7-7-5), counts unchanged.
//! - A vacancy at a 3-coordinated honeycomb site removes 1 site and 3 bonds.
//! - Hexagonal CSL Σ7 is (m, n) = (2, 1): cos θ = (m² + 4mn + n²) / (2 (m² + mn + n²)) = 13/14, θ ≈ 21.79°.
//!   Square CSL Σ5 is (2, 1): θ = 2 atan(1/2) ≈ 53.13° (and 90° - θ).
//! - b = a1 on a hexagonal cell is 1/3<2 -1 -1 0> in Weber four-index notation.
//! - Honeycomb 2 x 2 torus: 2 x 2 x 2 = 8 sites, J1 = |a1| / √3 with coordination 3, p6m acts with 12 x 4 = 48 permutations.

use std::time::{Duration, Instant};

use super::point::OpKind;
use super::solver::{ExportKind, export_text, export_to};
use super::table::{DefectRow, RowKind, SortCol, sort_order};
use super::{S2Sub, cell_scale, make_host, point, solver, stone_wales};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, wait_until};
use crate::app::{DefectsView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;
use crate::workbench::{WbPanel, WorkbenchView};
use egui_kittest::kittest::Queryable;
use papertilling_core::defect::{PointDefectOp, burgers_index_info, list_csl_grain_boundaries};

fn new_app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn frame(app: &mut PapertillingApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.frame_ui(ui);
    });
}

fn settle(app: &mut PapertillingApp, ctx: &egui::Context, what: &str) {
    let t0 = Instant::now();
    for _ in 0..4 {
        frame(app, ctx);
    }
    while app.workbench_busy() || app.workbench.dirty {
        assert!(t0.elapsed() < Duration::from_secs(60), "{what}: still busy after 60 s");
        std::thread::sleep(Duration::from_millis(10));
        frame(app, ctx);
    }
}

/// App on the S2 view of the named lattice (`None` = the default lattice).
fn s2_app(lattice: Option<&str>, ctx: &egui::Context) -> PapertillingApp {
    let mut app = new_app();
    if let Some(k) = lattice {
        let item = app.backend.find_named(k).cloned().unwrap_or_else(|| panic!("named lattice {k}"));
        app.select_lattice_and_sync(item);
    }
    app.select_defects_view(DefectsView::PointAndBoundary);
    settle(&mut app, ctx, "enter S2");
    assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, WorkbenchView::S2Defects));
    app
}

fn enter_sub(app: &mut PapertillingApp, ctx: &egui::Context, s: S2Sub) {
    app.workbench.s2.sub = s;
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(app, ctx, s.label());
}

fn resubmit(app: &mut PapertillingApp, ctx: &egui::Context) {
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(app, ctx, "resubmit");
}

fn title(s: &Scene2D) -> Option<&str> {
    match s {
        Scene2D::VectorScene { primitives } => Some(&primitives.title),
        Scene2D::CrystalPatch { title, .. } => Some(title),
        _ => None,
    }
}

fn honeycomb_cell() -> (papertilling_core::types::Cell2D, String) {
    let ctx = egui::Context::default();
    let app = s2_app(Some("honeycomb"), &ctx);
    (app.gallery.cell().clone(), app.gallery.active_item().group.clone())
}

#[test]
fn every_s2_subrow_builds_without_error_on_three_lattices() {
    let ctx = egui::Context::default();
    for lat in [None, Some("honeycomb"), Some("square")] {
        let mut app = s2_app(lat, &ctx);
        for s in S2Sub::ALL.into_iter().chain([S2Sub::Point]) {
            enter_sub(&mut app, &ctx, s);
            let p = &app.workbench.s2;
            if lat.is_none() && s == S2Sub::StoneWales {
                eprintln!("default lattice {}: bond rotation -> {:?}", app.gallery.active_item().label, p.error().map_or("built", |e| e));
            }
            if lat.is_none() && s == S2Sub::StoneWales && p.error().is_some() {
                // some lattices (e.g. with triangles) admit no bond rotation: refused with a reason, never a panic
                assert!(p.error().is_some_and(|e| e.contains("admits")), "{:?}", p.error());
                continue;
            }
            assert!(p.error().is_none(), "{lat:?} {s:?}: {:?}", p.error());
            assert!(p.built[s.idx()].is_some(), "{lat:?} {s:?}: a result is stored");
            assert_eq!(title(app.canvas2d.scene()), title(&p.scenes().scene2d), "{lat:?} {s:?}");
        }
    }
}

#[test]
fn stone_wales_on_graphene_gives_5_7_7_5() {
    let (cell, _) = honeycomb_cell();
    let o = stone_wales::build(&cell, 3.0 * cell_scale(&cell), None, false).unwrap();
    let (mut before, mut after) = (o.result.before_rings, o.result.after_rings);
    before.sort_unstable();
    after.sort_unstable();
    assert_eq!(before, [6, 6, 6, 6]);
    assert_eq!(after, [5, 5, 7, 7]);
    assert_eq!(o.result.crystal.pos.len(), o.host.crystal.pos.len(), "a bond rotation keeps the sites");
    assert_eq!(o.result.crystal.edges.len(), o.host.crystal.edges.len(), "and the bond count");
    assert!(stone_wales::build(&cell, 3.0 * cell_scale(&cell), Some(usize::MAX), false).is_err(), "bad bond refused");
}

#[test]
fn vacancy_substitution_and_adatom_hand_counts() {
    let (cell, _) = honeycomb_cell();
    let r = 3.0 * cell_scale(&cell);
    let host = make_host(&cell, r).unwrap();
    let (n, e) = (host.crystal.pos.len(), host.crystal.edges.len());
    let site = (0..n).find(|i| !host.rim.contains(i) && host.crystal.adj[*i].len() == 3).unwrap();
    let vac = point::build(&cell, r, vec![PointDefectOp::Vacancy { site, reconstruct_ring: false }]).unwrap();
    assert_eq!((vac.result.crystal.pos.len(), vac.result.crystal.edges.len()), (n - 1, e - 3));
    let sub = point::build(&cell, r, vec![PointDefectOp::Substitutional { site, species: String::from("N") }]).unwrap();
    assert_eq!(sub.result.crystal.pos.len(), n);
    assert_eq!(sub.result.species[site], "N");
    let h = 1.25;
    let ad = point::build(&cell, r, vec![PointDefectOp::Adatom { host_site: site, height: h, species: String::from("H") }]).unwrap();
    assert_eq!((ad.result.crystal.pos.len(), ad.result.crystal.edges.len()), (n + 1, e + 1));
    assert!(ad.has_adatom() && ad.z().iter().filter(|z| (**z - h).abs() < 1e-12).count() == 1);
    let rim = *host.rim.iter().next().unwrap();
    assert!(point::build(&cell, r, vec![PointDefectOp::Vacancy { site: rim, reconstruct_ring: false }]).is_err(), "rim refused");
}

#[test]
fn csl_lists_contain_the_hand_derived_sigma7_and_sigma5() {
    let ctx = egui::Context::default();
    let app = s2_app(Some("honeycomb"), &ctx);
    let (g, cell) = (app.gallery.active_item().group.clone(), app.gallery.cell().clone());
    let theta7 = (13.0f64 / 14.0).acos().to_degrees();
    assert!((theta7 - 21.79).abs() < 0.005);
    let hex = list_csl_grain_boundaries(&g, &cell, 19);
    assert!(hex.iter().any(|e| e.sigma == 7 && (e.theta_deg - theta7).abs() < 1e-9), "{hex:?}");
    assert!(hex.iter().any(|e| e.sigma == 7 && (e.theta_deg - (60.0 - theta7)).abs() < 1e-9));
    // the panel's default boundary on the honeycomb is the first (smallest Σ) entry: Σ7
    assert_eq!(app.workbench.s2.gb.params().1, Some(7));

    let mut app = app;
    app.select_lattice_and_sync(app.backend.find_named("square").cloned().unwrap());
    let (g, cell) = (app.gallery.active_item().group.clone(), app.gallery.cell().clone());
    let theta5 = 2.0 * 0.5f64.atan().to_degrees();
    let sq = list_csl_grain_boundaries(&g, &cell, 17);
    for t in [theta5, 90.0 - theta5] {
        assert!(sq.iter().any(|e| e.sigma == 5 && (e.theta_deg - t).abs() < 1e-9), "{t} in {sq:?}");
    }
}

#[test]
fn weber_index_of_a1_on_the_honeycomb_and_burgers_basis() {
    let (cell, group) = honeycomb_cell();
    let info = burgers_index_info(&group, &cell, [1, 0]);
    let w = info.weber_hex.expect("hexagonal cell has a Weber index");
    assert!(w.third_scale && info.weber_str.as_deref().is_some_and(|s| s.starts_with("1/3[")));
    let mut abs: Vec<i32> = w.indices[..3].iter().map(|x| x.abs()).collect();
    abs.sort_unstable();
    assert_eq!((abs, w.indices[..3].iter().sum::<i32>(), w.indices[3]), (vec![1, 1, 2], 0, 0), "1/3<2 -1 -1 0>");

    // the catalogue's burgers_mn is in the (a1, a2) basis of gallery.cell(): |m a1 + n a2| / |a1| = norm
    let ctx = egui::Context::default();
    let app = s2_app(Some("honeycomb"), &ctx);
    let cat = app.backend.defect_catalogue(app.gallery.active_item());
    assert!(!cat.dislocations.is_empty());
    let mut shortest = f64::INFINITY;
    for d in &cat.dislocations {
        let (m, n) = (f64::from(d.burgers_mn.0), f64::from(d.burgers_mn.1));
        let b = [m * cell.a1[0] + n * cell.a2[0], m * cell.a1[1] + n * cell.a2[1]];
        assert!((b[0].hypot(b[1]) / cell_scale(&cell) - d.norm).abs() < 1e-9, "{}", d.id);
        shortest = shortest.min(d.norm);
    }
    assert!((shortest - 1.0).abs() < 1e-9, "shortest honeycomb translation is |a1|");
}

fn row(kind: RowKind, id: &str, sigma: Option<u32>) -> DefectRow {
    let mut r = super::table::build_row_for_tests(kind, id);
    r.sigma = sigma;
    r
}

#[test]
fn defect_table_sort_is_stable_with_empty_cells_last_and_a_permutation() {
    let rows = vec![
        row(RowKind::Boundary, "b", Some(7)),
        row(RowKind::Point, "a", None),
        row(RowKind::Boundary, "c", Some(3)),
        row(RowKind::Boundary, "d", Some(7)),
    ];
    let all = [true; 5];
    assert_eq!(sort_order(&rows, SortCol::Sigma, true, &all), vec![2, 0, 3, 1]);
    assert_eq!(sort_order(&rows, SortCol::Sigma, false, &all), vec![0, 3, 2, 1]);
    assert_eq!(sort_order(&rows, SortCol::Id, false, &all), vec![3, 2, 0, 1]);
    let mut only_points = [false; 5];
    only_points[RowKind::Point as usize] = true;
    assert_eq!(sort_order(&rows, SortCol::Id, true, &only_points), vec![1]);

    let ctx = egui::Context::default();
    let mut app = s2_app(Some("honeycomb"), &ctx);
    enter_sub(&mut app, &ctx, S2Sub::AllDefects);
    let t = &mut app.workbench.s2.table;
    let rows = &t.out.as_ref().unwrap().rows;
    assert_eq!(rows.iter().filter(|r| r.kind == RowKind::Point).count(), 5, "4 point kinds + the bond rotation");
    assert!(rows.iter().any(|r| r.kind == RowKind::Boundary && r.sigma == Some(7)));
    assert!(rows.iter().filter(|r| r.kind == RowKind::Dislocation).all(|r| r.weber.is_some()));
    for c in SortCol::ALL {
        for asc in [true, false] {
            let mut o = sort_order(rows, c, asc, &all);
            o.sort_unstable();
            assert_eq!(o, (0..rows.len()).collect::<Vec<_>>(), "{c:?} {asc}");
        }
    }
    t.header_clicked(SortCol::Sigma);
    assert_eq!((t.col, t.asc), (SortCol::Sigma, true));
    t.header_clicked(SortCol::Sigma);
    assert_eq!((t.col, t.asc), (SortCol::Sigma, false));
}

#[test]
fn solver_export_honeycomb_2x2_hand_counts() {
    let (cell, group) = honeycomb_cell();
    let o = solver::build(&group, &cell, 2, 2, 1).unwrap();
    assert_eq!(o.n_sites, 8);
    let j1 = &o.shells.shells[0];
    assert!((j1.distance - cell_scale(&cell) / 3f64.sqrt()).abs() < 1e-9);
    assert!((j1.mean_coordination - 3.0).abs() < 1e-12);
    assert_eq!((o.report.order, o.report.group_closed), (48, true));
    assert!(solver::build(&group, &cell, 7, 1, 1).is_err(), "supercell cap refused");
    let xyz = export_text(ExportKind::SupercellXyz, &cell, Some(&o), None).unwrap();
    assert_eq!(xyz.lines().next(), Some("8"));
    let path = std::env::temp_dir().join(format!("papertilling_s2_solver_{}.json", std::process::id()));
    export_to(&path, &export_text(ExportKind::SolverJson, &cell, Some(&o), None).unwrap()).unwrap();
    let back: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(back["n_sites"], 8);
    assert!(export_text(ExportKind::DefectXyz, &cell, Some(&o), None).is_err(), "no point-defect patch yet");
}

#[test]
fn canvas_picks_on_the_host_and_adatoms_add_a_3d_view() {
    let ctx = egui::Context::default();
    let mut app = s2_app(Some("honeycomb"), &ctx);
    let host = app.workbench.s2.point.out.as_ref().unwrap().host.clone();
    let n = host.crystal.pos.len();
    let rim = *host.rim.iter().next().unwrap();
    assert!(!app.workbench.s2.point.click(host.crystal.pos[rim]));
    assert!(app.workbench.s2.point.note.contains("rim"), "{}", app.workbench.s2.point.note);
    let inner = (0..n).find(|i| !host.rim.contains(i)).unwrap();
    assert!(app.workbench.s2.point.click(host.crystal.pos[inner]));
    assert_eq!(app.workbench.s2.point.ops, vec![PointDefectOp::Vacancy { site: inner, reconstruct_ring: false }]);
    assert!(!app.workbench.s2.point.click(host.crystal.pos[inner]), "already vacant");
    app.workbench.s2.point.kind = OpKind::Interstitial;
    let mid = app.workbench.s2.point.centre_target().unwrap();
    assert!(app.workbench.s2.point.click(mid));
    assert!(!app.workbench.s2.point.click(mid), "a second interstitial on the same spot overlaps");
    app.workbench.s2.point.ops.pop();
    app.workbench.s2.point.kind = OpKind::Adatom;
    let other = (0..n).find(|i| !host.rim.contains(i) && *i != inner).unwrap();
    assert!(app.workbench.s2.point.click(host.crystal.pos[other]));
    assert!(!app.workbench.s2.point.click(host.crystal.pos[other]), "a second adatom on the same site overlaps");
    resubmit(&mut app, &ctx);
    for _ in 0..4 {
        frame(&mut app, &ctx);
    }
    let o = app.workbench.s2.point.out.as_ref().unwrap();
    assert_eq!(o.result.crystal.pos.len(), n, "one vacancy and one adatom");
    assert!(app.workbench.s2.scenes().mesh.is_some() && app.view3d.has_geometry());
    let xyz = export_text(ExportKind::DefectXyz, app.gallery.cell(), None, Some(o)).unwrap();
    assert_eq!(xyz.lines().next(), Some(n.to_string().as_str()));

    // a lattice change clears the picks (they index the old host disk)
    app.select_lattice_and_sync(app.backend.find_named("square").cloned().unwrap());
    settle(&mut app, &ctx, "square");
    assert!(app.workbench.s2.point.ops.is_empty());
    assert!(app.workbench.s2.error().is_none());
}

#[test]
fn lattice_round_trip_rebuilds_non_point_subrows() {
    let ctx = egui::Context::default();
    let mut app = s2_app(Some("honeycomb"), &ctx);
    let hc = app.gallery.active_item().clone();
    let sq = app.backend.find_named("square").cloned().unwrap();
    for s in [S2Sub::GrainBoundary, S2Sub::AllDefects, S2Sub::Solver] {
        enter_sub(&mut app, &ctx, s);
        enter_sub(&mut app, &ctx, S2Sub::Point);
        app.select_lattice_and_sync(sq.clone());
        settle(&mut app, &ctx, "square");
        app.select_lattice_and_sync(hc.clone());
        settle(&mut app, &ctx, "honeycomb");
        enter_sub(&mut app, &ctx, s);
        let p = &app.workbench.s2;
        let built = match s {
            S2Sub::GrainBoundary => p.gb.out.is_some(),
            S2Sub::AllDefects => p.table.out.is_some(),
            _ => p.solver.out.is_some(),
        };
        assert!(built && p.error().is_none(), "{s:?} rebuilt after a lattice round trip");
        assert_eq!(title(app.canvas2d.scene()), title(&p.scenes().scene2d));
    }
}

#[test]
fn s2_panel_state_survives_json_roundtrip() {
    let mut p = super::Panel { sub: S2Sub::Solver, radius_cells: 4.5, ..Default::default() };
    p.solver.nx = 2;
    p.sw.relax = true;
    p.table.header_clicked(SortCol::Theta);
    p.table.header_clicked(SortCol::Theta);
    let v = p.save();
    let back: serde_json::Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    let mut q = super::Panel::default();
    let mut notes = Vec::new();
    q.restore(&back, &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(q.save(), v);
    q.restore(&serde_json::json!({"sub": "Nope", "radius_cells": "x", "solver": {"nx": 99}}), &mut notes);
    assert_eq!(notes.len(), 1);
    assert_eq!((q.sub, q.solver.nx), (S2Sub::Solver, 6), "junk ignored, sizes clamped");
}

#[test]
fn s2_subrows_quarantine_clean() {
    let mut h = harness();
    let hc = h.state().backend.find_named("honeycomb").cloned().unwrap();
    h.state_mut().select_lattice_and_sync(hc);
    h.run_steps(3);
    h.get_by_label("2. Defects").click();
    h.run_steps(2);
    h.get_by_label("Point, Stone-Wales & GB").click();
    wait_until(&mut h, "S2", |a| !a.workbench_busy() && !a.workbench.dirty);
    for s in S2Sub::ALL.into_iter().chain([S2Sub::Point]) {
        h.get_by_label(s.label()).click();
        wait_until(&mut h, s.label(), |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().workbench.s2.sub, s);
        assert!(h.state().workbench.s2.error().is_none(), "{s:?}: {:?}", h.state().workbench.s2.error());
        assert_no_forbidden_tokens(&h, s.label());
        if s == S2Sub::AllDefects {
            h.get_by_label("Σ").click();
            h.run_steps(2);
            assert_eq!(h.state().workbench.s2.table.col, SortCol::Sigma);
        }
    }
}

/// Pilot timing of the heaviest S2 builds (printed with `# time`; no time is asserted).
#[test]
fn pilot_timing_heaviest_s2_jobs() {
    let (cell, group) = honeycomb_cell();
    let r6 = 6.0 * cell_scale(&cell);
    let t = Instant::now();
    super::boundary::build(&group, &cell, (13.0f64 / 14.0).acos().to_degrees(), Some(7), r6).unwrap();
    eprintln!("# time GB radius 6 cells: {:.3}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    stone_wales::build(&cell, r6, None, true).unwrap();
    eprintln!("# time SW relax radius 6 cells: {:.3}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    solver::build(&group, &cell, 6, 6, 3).unwrap();
    eprintln!("# time solver 6x6x3: {:.3}s", t.elapsed().as_secs_f64());
}
