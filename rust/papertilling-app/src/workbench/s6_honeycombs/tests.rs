//! S6 panel tests. Expected values are hand counts or published values, never read back from the panel:
//! - Cubic honeycomb {4,3,4}: Vertex + Touching layer 1 is a 4×4×4 box of cubes: V = 5³ = 125, E = 3·4·5² = 300,
//!   F = 3·4²·5 = 240, C = 4³ = 64, χ = 1; Cell + FaceShells layer 1 is the central cube plus 6 face neighbours:
//!   (V, E, F, C) = (32, 60, 36, 7), χ = 1.
//! - 16-cell {3,3,4}: (8, 24, 32, 16), χ(S³) = 0 (Coxeter, Regular Polytopes, Table I(ii)).
//! - Wigner-Seitz cells: cube (8, 12, 6), rhombic dodecahedron (14, 24, 12) with 12 rhombi, truncated octahedron
//!   (24, 36, 14) with 6 squares and 8 hexagons.
//! - Seifert-Weber space: 1 cell, 6 faces, 6 edges, 1 vertex, H₁ = (Z/5)³ (Weber-Seifert 1933); the Poincaré
//!   homology sphere has H₁ = 0.
//! - Labyrinth nets per conventional cell: pcu 1 site, 6/2 = 3 bonds; dia 8 sites, 8·4/2 = 16 bonds, bond √3/4;
//!   srs 8 sites, 8·3/2 = 12 bonds, bond √2/4.
//! - P, D, G surfaces have genus 3 per primitive cell (Schoen 1970); the {6,4|4} complex has (V, E, F) = (12, 24, 8)
//!   on the 3-torus of the cubic cell.

use std::time::{Duration, Instant};

use egui_kittest::kittest::Queryable;
use papertilling_core::growth::SeedKind;
use papertilling_core::honeycomb3d::{
    EuclideanKind, GrowthRule, HoneycombRegime, TpmsKind, nodal_surface_mesh, petrie_coxeter_surface,
};

use super::present::CoreOut;
use super::tpms::{TpmsMode, build_net, net_drawing, unwrap_faces};
use super::{Panel, S6Job, S6Sub, run_job};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, wait_until};
use crate::app::{LatticesView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;
use crate::game_menu::test_util::goto_lattices_view;
use crate::workbench::{WbCtx, WbPanel, WorkbenchView};

pub(super) fn new_app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn frame(app: &mut PapertillingApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.frame_ui(ui);
    });
}

pub(super) fn settle(app: &mut PapertillingApp, ctx: &egui::Context, what: &str) {
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

pub(super) fn s6_app(ctx: &egui::Context) -> PapertillingApp {
    let mut app = new_app();
    app.select_lattices_view(LatticesView::Honeycombs3D);
    settle(&mut app, ctx, "enter S6");
    assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, WorkbenchView::S6Honeycombs));
    app
}

pub(super) fn resubmit(app: &mut PapertillingApp, ctx: &egui::Context) {
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(app, ctx, "resubmit");
}

pub(super) fn complex_counts(job: S6Job) -> ((usize, usize, usize, usize), i64) {
    let b = run_job(job).unwrap();
    let cx = b.core.complex().expect("not a cell complex");
    let (v, e, f, c) = cx.counts();
    match &b.core {
        CoreOut::Growth(_, meta) => {
            let cells = meta.stats.last().map_or(c, |s| s.cells);
            let chi = meta.stats.last().map_or(cx.euler(), |s| s.euler);
            ((v, e, f, cells), chi)
        }
        _ => ((v, e, f, c), cx.euler()),
    }
}

#[test]
fn default_view_is_cubic_vertex_touching_layer_one_with_hand_counts() {
    // Hand count: {4,3,4} with default Vertex + Touching at layers = 1 is a 4×4×4 box of unit cubes:
    // V = 5³ = 125, E = 3·4·5² = 300, F = 3·4²·5 = 240, C = 4³ = 64, χ = 125 − 300 + 240 − 64 = 1.
    let ctx = egui::Context::default();
    let app = s6_app(&ctx);
    let p = &app.workbench.s6;
    assert!(p.error().is_none(), "{:?}", p.error());
    let b = p.current().expect("default build stored");
    match &b.core {
        CoreOut::Growth(cx, meta) => {
            assert_eq!((cx.counts(), meta.stats.last().unwrap().euler), ((125, 300, 240, 64), 1));
            assert_eq!(meta.stats.iter().map(|s| s.cells).collect::<Vec<_>>(), vec![8, 64]);
        }
        _ => panic!("default is a growth cell complex"),
    }
    assert!(app.view3d.has_geometry());
    assert_eq!(app.view3d.caption(), b.caption);
    assert!(b.notes.iter().any(|n| n.starts_with("[PROVED")) && b.notes.iter().any(|n| n.starts_with("[COMPUTATIONAL EVIDENCE")));
}

#[test]
fn schlafli_known_counts_and_regimes() {
    assert_eq!(
        complex_counts(S6Job::SchlafliGrowth {
            p: 3,
            q: 3,
            r: 4,
            seed: SeedKind::Vertex,
            rule: GrowthRule::Touching,
            layers: 1,
        }),
        ((8, 24, 32, 16), 0)
    );
    assert_eq!(
        complex_counts(S6Job::SchlafliGrowth {
            p: 4,
            q: 3,
            r: 4,
            seed: SeedKind::Cell,
            rule: GrowthRule::Touching,
            layers: 0,
        }),
        ((8, 12, 6, 1), 1)
    );
    assert_eq!(
        complex_counts(S6Job::SchlafliGrowth {
            p: 4,
            q: 3,
            r: 4,
            seed: SeedKind::Cell,
            rule: GrowthRule::FaceShells,
            layers: 1,
        })
        .0,
        (32, 60, 36, 7)
    );
    let st = |p, q, r| super::honeycomb::SchlafliState {
        p,
        q,
        r,
        layers: 25,
        radius_centi: 999,
        ..Default::default()
    };
    assert_eq!(st(4, 3, 4).classify().regime, HoneycombRegime::EuclideanE3);
    assert_eq!(st(5, 3, 5).classify().regime, HoneycombRegime::HyperbolicCompactH3);
    assert_eq!(st(5, 3, 3).classify().regime, HoneycombRegime::SphericalS3);
    assert_eq!(
        st(4, 3, 4).job(),
        S6Job::SchlafliGrowth { p: 4, q: 3, r: 4, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 10 }
    );
    assert_eq!(
        st(4, 4, 3).job(),
        S6Job::ParacompactBall { p: 4, q: 4, r: 3, radius_centi: 300 },
        "paracompact dispatches to ball with radius_centi clamped to 30..=300"
    );
}

#[test]
fn refused_symbols_show_reason_and_placeholder_never_an_old_mesh() {
    let ctx = egui::Context::default();
    let mut app = s6_app(&ctx);
    for (p, q, r, want) in [(7, 3, 3, "hyperideal"), (3, 7, 3, "hyperideal"), (4, 4, 2, "reducible")] {
        app.workbench.s6.schlafli = super::honeycomb::SchlafliState { p, q, r, ..Default::default() };
        resubmit(&mut app, &ctx);
        let p6 = &app.workbench.s6;
        let e = p6.error().unwrap_or_else(|| panic!("{{{p},{q},{r}}} must be refused"));
        assert!(e.starts_with("[PROVED") && e.contains(want), "{e}");
        assert!(p6.current().is_none());
        let cap = app.view3d.caption();
        assert!(cap.contains("placeholder") && !cap.contains("unavailable") && !cap.contains("stale") && !cap.contains("Build failed"));
    }
    app.workbench.s6.schlafli = super::honeycomb::SchlafliState::default();
    resubmit(&mut app, &ctx);
    assert!(app.workbench.s6.error().is_none() && app.workbench.s6.current().is_some());
    // the gyroid carries no regular skew polyhedron: refused by the core with its reason
    app.workbench.s6.sub = S6Sub::Tpms;
    app.workbench.s6.tpms.kind = TpmsKind::G;
    app.workbench.s6.tpms.mode = TpmsMode::PetrieCoxeter;
    resubmit(&mut app, &ctx);
    assert!(app.workbench.s6.error().is_some_and(|e| e.contains("gyroid")), "{:?}", app.workbench.s6.error());
}

#[test]
fn euclidean_kinds_central_cells_and_balls() {
    let cases = [
        (EuclideanKind::Cubic, (8, 12, 6, 1)),
        (EuclideanKind::BitruncatedCubic, (24, 36, 14, 1)),
        (EuclideanKind::RhombicDodecahedral, (14, 24, 12, 1)),
    ];
    for (kind, counts) in cases {
        assert_eq!(
            complex_counts(S6Job::Euclid { kind, seed: SeedKind::Cell, rule: GrowthRule::Touching, layers: 0 }),
            (counts, 1),
            "{kind:?}"
        );
    }
    for kind in super::honeycomb::EUCLID_KINDS {
        assert_eq!(
            complex_counts(S6Job::Euclid { kind, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 1 }).1,
            1,
            "{kind:?}: a one-layer vertex patch is a 3-ball"
        );
    }
    assert!(
        run_job(S6Job::Euclid { kind: EuclideanKind::Cubic, seed: SeedKind::Cell, rule: GrowthRule::Touching, layers: 11 }).is_err(),
        "MAX_GROWTH_LAYERS respected"
    );
}

#[test]
fn voronoi_cells_and_dirichlet_quotients() {
    for (lat, counts, faces) in [("SC", (8, 12, 6), "6 × 4-gon"), ("FCC", (14, 24, 12), "12 × 4-gon"), ("BCC", (24, 36, 14), "6 × 4-gon, 8 × 6-gon")] {
        let b = run_job(S6Job::LatticeCell { lattice: lat }).unwrap();
        let CoreOut::Cell(c) = &b.core else { panic!("cell") };
        assert_eq!(c.counts(), counts, "{lat}");
        assert!(b.rows.iter().any(|(_, v)| v == faces), "{lat}: {:?}", b.rows);
    }
    let b = run_job(S6Job::Manifold { name: "seifert_weber" }).unwrap();
    let CoreOut::Manifold(_, q) = &b.core else { panic!("manifold") };
    assert_eq!((q.cells, q.faces, q.edges, q.vertices, q.euler), (1, 6, 6, 1, 0));
    assert_eq!(q.h1_label, "Z_5^3");
    let b = run_job(S6Job::Manifold { name: "poincare_homology_sphere" }).unwrap();
    let CoreOut::Manifold(_, q) = &b.core else { panic!("manifold") };
    assert_eq!((q.h1_label.as_str(), q.euler), ("0", 0));
    assert!(b.notes.iter().any(|n| n.starts_with("Combinatorial model only")), "the S³ drawing is labelled a model");
    let want = [12, 12, 6, 12, 14];
    for ((name, _), faces) in super::voronoi::MANIFOLDS.into_iter().zip(want) {
        let b = run_job(S6Job::Manifold { name }).unwrap();
        let CoreOut::Manifold(d, _) = &b.core else { panic!("manifold") };
        assert_eq!((d.faces.len(), d.pairings.len(), b.mesh.face_count), (faces, faces / 2, faces), "{name}");
    }
}

#[test]
fn labyrinth_nets_hand_counts_and_ghost_ends() {
    for (kind, sites, bonds, coord, len) in
        [(TpmsKind::P, 1, 3, 6, 1.0), (TpmsKind::D, 8, 16, 4, 3f64.sqrt() / 4.0), (TpmsKind::G, 8, 12, 3, 2f64.sqrt() / 4.0)]
    {
        let b = build_net(kind, 1, 1).unwrap();
        let CoreOut::Net(n) = &b.core else { panic!("net") };
        assert_eq!((n.positions.len(), n.edges.len()), (sites, bonds), "{kind:?}");
        assert!(n.coordination.iter().all(|&c| c == coord), "{kind:?}");
        assert!((n.bond_length - len).abs() < 1e-12, "{kind:?}");
        let two = build_net(kind, 2, 1).unwrap();
        let CoreOut::Net(n2) = &two.core else { panic!("net") };
        assert_eq!((n2.positions.len(), n2.edges.len()), (2 * sites, 2 * bonds), "{kind:?}: both labyrinths");
    }
    let net = papertilling_core::honeycomb3d::tpms_net(TpmsKind::P, 1).unwrap();
    let (pts, _, edges, real) = net_drawing(&net, 1);
    assert_eq!((real, pts.len() - real, edges.len()), (1, 3, 3));
    let (pts, _, edges, real) = net_drawing(&net, 2);
    assert_eq!((real, pts.len() - real, edges.len()), (8, 12, 24));
    assert!(edges.iter().all(|&(u, v, _)| u != v), "no self-loop is drawn");
}

#[test]
fn periodic_surfaces_genus_three_and_unwrapped_faces_are_short() {
    let pc = petrie_coxeter_surface(TpmsKind::P).unwrap();
    assert_eq!((pc.vertices.len(), pc.edges.len(), pc.faces.len(), pc.genus_primitive), (12, 24, 8, 3));
    for (kind, n) in [(TpmsKind::G, 16), (TpmsKind::D, 12)] {
        let s = nodal_surface_mesh(kind, n).unwrap();
        assert_eq!((s.genus_primitive, s.closed_manifold), (3, true), "{kind:?}");
        assert!(s.evidence.starts_with("[COMPUTATIONAL EVIDENCE"));
    }
    for s in [pc, petrie_coxeter_surface(TpmsKind::D).unwrap(), nodal_surface_mesh(TpmsKind::G, 16).unwrap()] {
        let (pts, faces) = unwrap_faces(&s, 2);
        assert_eq!(faces.len(), 8 * s.faces.len());
        for f in &faces {
            for m in 0..f.len() {
                let (a, b) = (pts[f[m]], pts[f[(m + 1) % f.len()]]);
                let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                assert!(d < 0.5, "{}: unwrapped edge of length {d} spans the cell", s.name);
            }
        }
    }
    let b = run_job(S6Job::Surface { kind: TpmsKind::P, mode: TpmsMode::Nodal, n: 12, reps: 1 }).unwrap();
    assert!(b.rows.iter().any(|(k, v)| k == "Genus per primitive cell" && v == "3"));
}

#[test]
fn s6_panel_state_survives_json_roundtrip() {
    let mut p = Panel { sub: S6Sub::Voronoi, ..Default::default() };
    p.schlafli = super::honeycomb::SchlafliState {
        p: 5,
        q: 3,
        r: 5,
        seed: SeedKind::Edge,
        rule: GrowthRule::FaceShells,
        layers: 2,
        radius_centi: 150,
    };
    p.euclid.kind = EuclideanKind::AlternatedCubic;
    p.euclid.seed = SeedKind::Face;
    p.euclid.rule = GrowthRule::FaceShells;
    p.euclid.layers = 4;
    p.tpms.kind = TpmsKind::G;
    p.tpms.mode = TpmsMode::Nodal;
    p.tpms.n = 24;
    p.tpms.reps = 2;
    assert_eq!(p.tpms.effective_reps(), 1, "a fine nodal mesh is drawn in one cell");
    p.vor.mode = super::voronoi::VorMode::Manifold;
    p.vor.manifold = 1;
    let v = p.save();
    let back: serde_json::Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    let mut q = Panel::default();
    let mut notes = Vec::new();
    q.restore(&back, &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(q.save(), v);
    let junk = serde_json::json!({
        "sub": "Nope",
        "schlafli": {"p": 99, "layers": 40, "radius_centi": 999},
        "tpms": {"kind": "Q", "n": 5000, "reps": 9},
        "euclid": {"layers": 77},
        "voronoi": {"manifold": "lens_space"}
    });
    q.restore(&junk, &mut notes);
    assert_eq!(notes.len(), 3, "{notes:?}");
    assert_eq!(
        (q.sub, q.schlafli.p, q.schlafli.layers, q.schlafli.radius_centi, q.tpms.n, q.tpms.reps, q.euclid.layers),
        (S6Sub::Voronoi, 8, 10, 300, 32, 2, 10),
        "clamped"
    );
    assert!(!notes.iter().any(|n| n.contains("workbench")));
}

#[test]
fn s6_subrows_quarantine_clean() {
    let mut h = harness();
    h.run_steps(3);
    goto_lattices_view(&mut h, LatticesView::Honeycombs3D);
    wait_until(&mut h, "S6", |a| !a.workbench_busy() && !a.workbench.dirty);
    for s in S6Sub::ALL.into_iter().chain([S6Sub::Schlafli]) {
        h.get_by_label(s.label()).click();
        wait_until(&mut h, s.label(), |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().workbench.s6.sub, s);
        assert!(h.state().workbench.s6.error().is_none(), "{s:?}: {:?}", h.state().workbench.s6.error());
        assert!(h.state().view3d.has_geometry(), "{s:?}");
        assert_no_forbidden_tokens(&h, s.label());
    }
}

/// Strict quarantine at the source level: no panel file (tests excluded) mentions spectral notions at all.
#[test]
fn s6_sources_carry_no_spectral_wording() {
    let sources = [
        include_str!("mod.rs"),
        include_str!("present.rs"),
        include_str!("honeycomb.rs"),
        include_str!("honeycomb_present.rs"),
        include_str!("tpms.rs"),
        include_str!("voronoi.rs"),
    ];
    for (k, src) in sources.iter().enumerate() {
        let low = src.to_lowercase();
        for bad in ["spectr", "eigen", "ldos", "peierls", " band", "tight-binding", "δρ"] {
            assert!(!low.contains(bad), "source #{k} contains {bad:?}");
        }
    }
}

/// Pilot timing of the heaviest builds the UI can reach (printed with `# time`; no time is asserted).
#[test]
fn pilot_timing_heaviest_s6_jobs() {
    let jobs = [
        S6Job::SchlafliGrowth { p: 4, q: 3, r: 4, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 2 },
        S6Job::SchlafliGrowth { p: 5, q: 3, r: 3, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 6 },
        S6Job::SchlafliGrowth { p: 3, q: 3, r: 5, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 4 },
        S6Job::SchlafliGrowth { p: 5, q: 3, r: 5, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 1 },
        S6Job::SchlafliGrowth { p: 3, q: 5, r: 3, seed: SeedKind::Cell, rule: GrowthRule::FaceShells, layers: 2 },
        S6Job::ParacompactBall { p: 4, q: 4, r: 4, radius_centi: 120 },
        S6Job::Euclid { kind: EuclideanKind::BitruncatedCubic, seed: SeedKind::Vertex, rule: GrowthRule::Touching, layers: 3 },
        S6Job::Surface { kind: TpmsKind::G, mode: TpmsMode::Nodal, n: 32, reps: 1 },
        S6Job::Surface { kind: TpmsKind::G, mode: TpmsMode::Nodal, n: 16, reps: 2 },
        S6Job::Net { kind: TpmsKind::G, labyrinths: 2, reps: 2 },
        S6Job::Manifold { name: "seifert_weber" },
    ];
    for job in jobs {
        let what = format!("{job:?}");
        let t = Instant::now();
        let b = run_job(job).unwrap_or_else(|e| panic!("{what}: {e}"));
        eprintln!("# time {what}: {:.3} s, {} mesh vertices", t.elapsed().as_secs_f64(), b.mesh.faces.len());
    }
}

/// Poll the panel's worker until a result lands (stored or dropped).
fn wait_poll(p: &mut Panel, what: &str) {
    let t0 = Instant::now();
    while !p.poll() {
        assert!(t0.elapsed() < Duration::from_secs(60), "{what}: no result after 60 s");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn shown_counts(p: &Panel) -> Option<(usize, usize, usize, usize)> {
    p.current()?.core.complex().map(|cx| cx.counts())
}

#[test]
fn stale_results_never_shown_under_current_inputs() {
    let app = new_app();
    let cx = WbCtx { item: app.gallery.active_item(), cell: app.gallery.cell(), smoke: false, backend: &app.backend, defect_crystal: None };
    let mut p = Panel {
        schlafli: super::honeycomb::SchlafliState {
            p: 4,
            q: 3,
            r: 5,
            seed: SeedKind::Cell,
            rule: GrowthRule::Touching,
            layers: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    p.submit(&cx, None);
    wait_poll(&mut p, "{4,3,5}");
    assert_eq!(shown_counts(&p), Some((8, 12, 6, 1)));
    // r -> 6 -> 5 before the {4,3,6} job is polled: the cache hit cancels it
    p.schlafli.r = 6;
    p.submit(&cx, None);
    assert!(p.pending.is_some() && p.is_busy());
    p.schlafli.r = 5;
    p.submit(&cx, None);
    assert!(p.pending.is_none() && !p.is_busy(), "the abandoned job is cancelled");
    std::thread::sleep(Duration::from_millis(300));
    for _ in 0..10 {
        let _ = p.poll();
    }
    assert_eq!(shown_counts(&p), Some((8, 12, 6, 1)), "still the {{4,3,5}} cube");
    assert_eq!(p.built[0].as_ref().map(|(k, _)| k.clone()), Some(format!("{:?}", p.job())));
    let stale = S6Job::ParacompactBall { p: 4, q: 3, r: 6, radius_centi: 100 };
    p.pending = Some((S6Sub::Schlafli, format!("{stale:?}")));
    p.accept(Ok(run_job(stale)));
    assert_eq!(p.built[0].as_ref().map(|(k, _)| k.clone()), Some(format!("{:?}", p.job())), "stale result dropped");
    p.schlafli.p = 5;
    assert!(p.current().is_none());
    p.schlafli.p = 4;
    assert!(p.current().is_some());
    p.sub = S6Sub::Voronoi;
    p.submit(&cx, None);
    wait_poll(&mut p, "SC cell");
    p.sub = S6Sub::Schlafli;
    p.schlafli.r = 6;
    p.submit(&cx, None);
    assert!(p.is_busy());
    p.sub = S6Sub::Voronoi;
    p.submit(&cx, None);
    assert!(p.pending.is_none() && !p.is_busy() && p.current().is_some(), "the cached sub-row is shown at once");
}

#[test]
fn paracompact_balls_carry_no_ball_euler_claim() {
    let b = run_job(S6Job::ParacompactBall { p: 6, q: 3, r: 3, radius_centi: 80 }).unwrap();
    assert!(!b.notes.iter().any(|n| n.contains("χ(B³)")), "{:?}", b.notes);
    assert!(b.notes.iter().any(|n| n.starts_with("No 3-ball test")), "{:?}", b.notes);
    let cube = run_job(S6Job::SchlafliGrowth {
        p: 4,
        q: 3,
        r: 4,
        seed: SeedKind::Cell,
        rule: GrowthRule::Touching,
        layers: 1,
    })
    .unwrap();
    assert!(cube.notes.iter().any(|n| n.contains("[PROVED: χ(B³) = 1]")), "compact cells keep the ball test");
}

#[test]
fn tpms_notes_describe_what_is_drawn() {
    let note = |mode, n| match run_job(S6Job::Surface { kind: TpmsKind::P, mode, n, reps: 1 }).unwrap().scene2d {
        Scene2D::VectorScene { primitives } => primitives.note,
        _ => panic!("vector scene"),
    };
    let exact = note(TpmsMode::PetrieCoxeter, 0);
    assert!(exact.starts_with("heat map of the nodal approximation only") && !exact.contains("the surface meets"), "{exact}");
    assert!(note(TpmsMode::Nodal, 12).contains("the surface meets this slice"));
    let g = build_net(TpmsKind::G, 2, 1).unwrap();
    assert!(g.notes.iter().any(|n| n.contains("G inverted through the origin, r ↦ −r")), "{:?}", g.notes);
    assert!(!g.notes.iter().any(|n| n.contains("x ↦ −x")));
}
