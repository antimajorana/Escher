//! S4 panel tests. Expected values are hand counts or published formulas, never values read back from the panel:
//! - Simple cubic on `T² × R`, repeats 3 × 3 × 3: 27 sites; a1, a2 periodic and a3 open, so 27 + 27 + 2·9 = 72 bonds
//!   and the two open layers hold 2·9 = 18 boundary sites.
//! - Simple cubic on the torocosm c1 with supercell 2 × 2 × 2: 8 vertices and 3·8 = 24 bonds (each site has 6 bond
//!   ends), χ₃ = 0 (every closed odd-dimensional manifold).
//! - Conway-Rossetti (2003): 10 platycosms, 6 orientable; 13 torsion-free ITA groups 1, 4, 7, 9, 19, 29, 33, 76, 78,
//!   144, 145, 169, 170.
//! - Saito, Dresselhaus & Dresselhaus (1998, Ch. 3), graphene in the 60° basis: |C_h| = a√(n²+nm+m²),
//!   cos θ = (2n+m) / (2√(n²+nm+m²)), 4(n²+nm+m²)/d_R atoms per tube cell with d_R = gcd(2n+m, 2m+n).
//! - Honeycomb (2 sites per cell) ribbon of 3 × 4 cells: 24 sites. Honeycomb flake about a hexagon centre: radius
//!   1.1 bonds holds one hexagon (6 sites, 6 bonds, point group 6mm of order 12); radius 2.7 bonds is coronene
//!   (24 sites, 30 bonds, 3 site orbits).

use std::time::{Duration, Instant};

use papertilling_core::flat3d::FlatSlabKind;
use papertilling_core::space3d::nets3d::CanonicalNet3d;
use papertilling_core::types::Cell2D;

use super::flakes::{Centre, FlakeParams, Shape};
use super::jobs::S4Out;
use super::tubes::{RibbonParams, TubeParams};
use super::{Panel, S4Sub, flakes, platycosms, slabs, torsion_free, tubes};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, wait_until};
use crate::app::{LatticesView, NavTab, PapertillingApp};
use crate::workbench::{WbPanel, WorkbenchView};
use egui_kittest::kittest::Queryable;

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

fn s4_app(lattice: Option<&str>, ctx: &egui::Context) -> PapertillingApp {
    let mut app = new_app();
    if let Some(k) = lattice {
        let item = app.backend.find_named(k).cloned().unwrap_or_else(|| panic!("named lattice {k}"));
        app.select_lattice_and_sync(item);
    }
    app.select_lattices_view(LatticesView::Flat3D);
    settle(&mut app, ctx, "enter S4");
    assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, WorkbenchView::S4Flat3d));
    app
}

fn enter_sub(app: &mut PapertillingApp, ctx: &egui::Context, s: S4Sub) {
    app.workbench.s4.sub = s;
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(app, ctx, s.label());
}

fn honeycomb() -> (Cell2D, String, [f64; 2]) {
    let ctx = egui::Context::default();
    let app = s4_app(Some("honeycomb"), &ctx);
    let it = app.gallery.active_item();
    (app.gallery.cell().clone(), it.group.clone(), it.frame.origin)
}

fn len2(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

#[test]
fn default_state_paints_the_torus_slab() {
    let ctx = egui::Context::default();
    let app = s4_app(None, &ctx);
    let p = &app.workbench.s4;
    assert_eq!(p.sub, S4Sub::Slabs);
    assert!(p.error().is_none(), "{:?}", p.error());
    let s = p.slab.out.as_ref().expect("default slab built");
    assert_eq!((s.positions.len(), s.bonds.len(), s.boundary_sites.len()), (27, 72, 18));
    assert!(app.view3d.has_geometry(), "default S4 state paints a non-empty mesh");
    assert!(app.view3d.caption().contains("T² × R"), "{}", app.view3d.caption());
}

#[test]
fn every_subrow_builds_on_the_default_and_honeycomb_lattices() {
    let ctx = egui::Context::default();
    for lat in [None, Some("honeycomb")] {
        let mut app = s4_app(lat, &ctx);
        for s in S4Sub::ALL {
            let t = Instant::now();
            enter_sub(&mut app, &ctx, s);
            let p = &app.workbench.s4;
            eprintln!("# time {lat:?} {s:?}: {:.3}s", t.elapsed().as_secs_f64());
            assert!(p.error().is_none(), "{lat:?} {s:?}: {:?}", p.error());
            assert!(p.built[s.idx()].is_some(), "{lat:?} {s:?}: a result is stored");
            let want = p.scenes();
            assert_eq!(want.mesh.is_some(), s != S4Sub::TorsionFree, "{s:?}: only the table is 2D only");
            if want.mesh.is_some() {
                assert!(app.view3d.has_geometry(), "{lat:?} {s:?}: mesh painted");
            }
        }
    }
}

#[test]
fn slab_refusal_is_proved_and_draws_nothing() {
    let e = slabs::build(CanonicalNet3d::BetaQuartz, FlatSlabKind::KleinSlab, [3, 3, 2]).unwrap_err();
    assert!(e.contains("IMPOSSIBLE [PROVED]"), "{e}");
    let k = slabs::build(CanonicalNet3d::SimpleCubic, FlatSlabKind::KleinSlab, [4, 4, 3]).unwrap();
    assert_eq!(k.positions.len(), 4 * 4 * 3);
    assert!(!k.orientable && k.bonds.iter().any(|b| b.2), "Klein slab: non-orientable with seam bonds");

    let ctx = egui::Context::default();
    let mut app = s4_app(None, &ctx);
    app.workbench.s4.slab.net = CanonicalNet3d::BetaQuartz;
    app.workbench.s4.slab.kind = FlatSlabKind::KleinSlab;
    enter_sub(&mut app, &ctx, S4Sub::Slabs);
    let p = &app.workbench.s4;
    assert!(p.error().is_some_and(|e| e.contains("[PROVED]")), "{:?}", p.error());
    assert!(p.slab.out.is_none() && p.scenes().mesh.is_none(), "no geometry for an impossible slab");
    assert!(!app.view3d.has_geometry(), "the old slab mesh is not left on screen");
}

#[test]
fn platycosm_table_quotient_counts_and_refusals() {
    let all = papertilling_core::flat3d::list_platycosms();
    assert_eq!((all.len(), all.iter().filter(|p| p.orientable).count()), (10, 6));
    let q = platycosms::build(CanonicalNet3d::SimpleCubic, "c1", Some([2, 2, 2]), false).unwrap();
    assert_eq!((q.vertices.len(), q.bonds.len(), q.euler_chi_3), (8, 24, 0));
    assert_eq!(q.h1_label, "Z^3");
    // hexagonal holohedry has no 4-fold axis; cubic has no 6-fold axis: both [PROVED] impossible
    let hcp_c4 = platycosms::build(CanonicalNet3d::HCP, "c4", None, false).unwrap_err();
    assert!(hcp_c4.contains("IMPOSSIBLE [PROVED]"), "{hcp_c4}");
    let sc_c6 = platycosms::build(CanonicalNet3d::SimpleCubic, "c6", None, false).unwrap_err();
    assert!(sc_c6.contains("IMPOSSIBLE [PROVED]"), "{sc_c6}");
    // simple cubic on c3 is only not found by the bounded search: never reported as impossible
    let sc_c3 = platycosms::build(CanonicalNet3d::SimpleCubic, "c3", None, false).unwrap_err();
    assert!(sc_c3.contains("NOT_FOUND") && sc_c3.contains("[COMPUTATIONAL EVIDENCE]") && !sc_c3.contains("IMPOSSIBLE"), "{sc_c3}");

    let ctx = egui::Context::default();
    let mut app = s4_app(None, &ctx);
    enter_sub(&mut app, &ctx, S4Sub::Platycosms);
    assert!(app.workbench.s4.platy.out.is_some());
    app.workbench.s4.platy.code = String::from("c6");
    enter_sub(&mut app, &ctx, S4Sub::Platycosms);
    let p = &app.workbench.s4;
    assert!(p.error().is_some_and(|e| e.contains("IMPOSSIBLE [PROVED]")), "{:?}", p.error());
    assert!(p.platy.out.is_none() && p.scenes().mesh.is_none());
}

#[test]
fn nanotubes_match_the_saito_formulas_and_ribbon_counts() {
    let (cell, _, _) = honeycomb();
    let a = len2(cell.a1);
    for (n, m) in [(5, 5), (6, 0), (4, 2)] {
        let p = TubeParams { n, m, periods: 1, periodic: true, std60: true };
        let t = tubes::build_tube(&cell, &p).unwrap();
        let q = n * n + n * m + m * m;
        let d_r = gcd(2 * n + m, 2 * m + n);
        assert_eq!(t.atoms_per_unit as i32, 4 * q / d_r, "({n},{m}) atoms per tube cell");
        let qf = f64::from(q);
        assert!((t.circumference - a * qf.sqrt()).abs() < 1e-9 * a, "({n},{m}) |C_h|");
        assert!((t.radius - t.circumference / std::f64::consts::TAU).abs() < 1e-12);
        let theta = (f64::from(2 * n + m) / (2.0 * qf.sqrt())).acos().to_degrees();
        assert!((t.chiral_angle_deg - theta).abs() < 1e-4, "({n},{m}) chiral angle");
        assert!(t.coordination_numbers.iter().all(|&c| c == 3), "closed tube: every site 3-coordinated");
    }
    let r = tubes::build_ribbon(&cell, &RibbonParams { dir: [1, 0], width: 3, length: 4, periodic: true }).unwrap();
    assert_eq!(r.positions_2d.len(), 2 * 3 * 4);
    assert!(!r.edge_sites.is_empty());
}

#[test]
fn honeycomb_flakes_are_benzene_and_coronene_and_orbits_partition() {
    let (cell, group, origin) = honeycomb();
    let a = len2(cell.a1);
    let bond = a / 3.0_f64.sqrt();
    for (r_bonds, sites, edges, orbits) in [(1.1, 6, 6, 1), (2.7, 24, 30, 3)] {
        for shape in [Shape::Disk, Shape::Polygon] {
            let p = FlakeParams { shape, n_gon: 6, size: r_bonds * bond / a, centre: Centre::FrameOrigin };
            let f = flakes::build(&group, &cell, origin, &p).unwrap();
            assert_eq!((f.crystal.pos.len(), f.crystal.edges.len(), f.site_orbits.len()), (sites, edges, orbits), "{shape:?} r={r_bonds}");
            assert_eq!(f.point_group_order, 12);
            let mut seen: Vec<usize> = f.site_orbits.iter().flatten().copied().collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..sites).collect::<Vec<_>>(), "orbits partition the sites");
        }
    }
    let ws = flakes::build(&group, &cell, origin, &FlakeParams { shape: Shape::WignerSeitz, size: 1.8, ..Default::default() }).unwrap();
    assert_eq!(ws.point_group_order, 12);
}

#[test]
fn torsion_free_table_is_the_published_bieberbach_list() {
    let rows = torsion_free::build().unwrap();
    let free: Vec<i32> = rows.iter().filter(|r| r.is_torsion_free).map(|r| r.number).collect();
    assert_eq!(free, torsion_free::BIEBERBACH_ITA);
    let mut st = torsion_free::TorsionState { out: Some(rows), ..Default::default() };
    assert_eq!(st.visible().len(), 13);
    st.only_free = false;
    assert_eq!(st.visible().len(), 230);
    st.query = String::from("19");
    assert_eq!(st.visible().len(), 1, "an ITA number matches exactly");
}

#[test]
fn invalid_inputs_are_refused_with_reasons() {
    let ctx = egui::Context::default();
    let mut app = s4_app(None, &ctx);
    app.workbench.s4.tube.p.n = 0;
    app.workbench.s4.tube.p.m = 0;
    enter_sub(&mut app, &ctx, S4Sub::Nanotubes);
    let p = &app.workbench.s4;
    assert!(p.error().is_some_and(|e| e.contains("non-zero")), "{:?}", p.error());
    assert!(p.scenes().mesh.is_none());
    app.workbench.s4.ribbon.p.dir = [0, 0];
    enter_sub(&mut app, &ctx, S4Sub::Ribbons);
    assert!(app.workbench.s4.error().is_some_and(|e| e.contains("non-zero")));
    // the drag widget clamps a typed size on the next frame, so check the refusal before any frame runs
    app.workbench.s4.sub = S4Sub::Flakes;
    app.workbench.s4.flake.p.size = 99.0;
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    assert!(app.workbench.s4.error().is_some_and(|e| e.contains("size")), "{:?}", app.workbench.s4.error());
    assert!(app.workbench.s4.scenes().mesh.is_none());
    app.workbench.s4.flake.p.size = 2.0;
    enter_sub(&mut app, &ctx, S4Sub::Flakes);
    assert!(app.workbench.s4.error().is_none(), "fixed input builds again: {:?}", app.workbench.s4.error());
}

#[test]
fn s4_panel_state_survives_json_roundtrip() {
    let mut p = Panel { sub: S4Sub::Platycosms, ..Default::default() };
    p.slab.kind = FlatSlabKind::MoebiusTube;
    p.slab.repeats = [4, 3, 3];
    p.platy.code = String::from("-a2");
    p.platy.auto_supercell = false;
    p.tube.p = TubeParams { n: 7, m: 2, periods: 3, periodic: true, std60: true };
    p.ribbon.p.dir = [1, 1];
    p.flake.p.shape = Shape::Polygon;
    p.torsion.only_free = false;
    let v = p.save();
    let back: serde_json::Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    let mut q = Panel::default();
    let mut notes = Vec::new();
    q.restore(&back, &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(q.save(), v);
    let junk = serde_json::json!({"sub": "Nope", "slab": {"repeats": [99, 0, 2], "kind": "Bad"}, "tube": {"n": 500}, "platy": {"code": "zz"}});
    q.restore(&junk, &mut notes);
    assert_eq!(notes.len(), 3, "{notes:?}");
    assert_eq!((q.sub, q.slab.repeats, q.tube.p.n, q.platy.code.as_str()), (S4Sub::Platycosms, [6, 1, 2], 12, "-a2"));
}

#[test]
fn s4_subrows_quarantine_clean() {
    let mut h = harness();
    h.get_by_label("Flat 3D: Slabs, Tubes, Flakes").click();
    wait_until(&mut h, "S4", |a| !a.workbench_busy() && !a.workbench.dirty);
    for s in S4Sub::ALL.into_iter().chain([S4Sub::Slabs]) {
        h.get_by_label(s.label()).click();
        wait_until(&mut h, s.label(), |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().workbench.s4.sub, s);
        assert!(h.state().workbench.s4.error().is_none(), "{s:?}: {:?}", h.state().workbench.s4.error());
        assert_no_forbidden_tokens(&h, s.label());
    }
}

/// Pilot timing of the heaviest S4 builds (printed with `# time`; no time is asserted).
#[test]
fn pilot_timing_heaviest_s4_jobs() {
    let t = Instant::now();
    slabs::build(CanonicalNet3d::Diamond, FlatSlabKind::TorusSlab, [6, 6, 6]).unwrap();
    eprintln!("# time slab Diamond 6x6x6: {:.3}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    torsion_free::build().unwrap();
    eprintln!("# time torsion-free 230 groups: {:.3}s", t.elapsed().as_secs_f64());
    for (net, code) in [(CanonicalNet3d::SimpleCubic, "c3"), (CanonicalNet3d::Diamond, "-a2"), (CanonicalNet3d::HCP, "c6")] {
        let t = Instant::now();
        let r = platycosms::build(net, code, None, false);
        eprintln!("# time platycosm auto search {net:?} on {code} ({}): {:.3}s", if r.is_ok() { "OK" } else { "refused" }, t.elapsed().as_secs_f64());
    }
}

// ---- review fixes (hand-checked expectations) ----

/// Inputs W built, drag to X (job in flight), drag back to W: the X job is forgotten and its late result never lands.
#[test]
fn returning_to_the_built_inputs_drops_the_job_in_flight() {
    let ctx = egui::Context::default();
    let mut app = s4_app(None, &ctx);
    let w = app.workbench.s4.built[S4Sub::Slabs.idx()].clone().expect("default slab built");
    app.workbench.s4.slab.repeats = [4, 3, 3];
    app.workbench_call(|wb, cx, _, _| wb.s4.submit(cx, None));
    assert!(app.workbench.s4.pending.as_ref().is_some_and(|(s, k)| *s == S4Sub::Slabs && *k != w), "X in flight");
    app.workbench.s4.slab.repeats = [3, 3, 3];
    app.workbench_call(|wb, cx, _, _| wb.s4.submit(cx, None));
    assert!(app.workbench.s4.pending.is_none(), "the X job is forgotten");
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(10));
        assert!(!app.workbench.s4.poll(), "no late result is accepted");
    }
    let p = &app.workbench.s4;
    assert_eq!(p.built[S4Sub::Slabs.idx()].as_deref(), Some(w.as_str()));
    assert_eq!(p.slab.out.as_ref().map(|s| s.positions.len()), Some(27), "still the 3 x 3 x 3 patch");
}

#[test]
fn accept_drops_a_result_for_inputs_no_longer_wanted() {
    let patch = || S4Out::Slab(Box::new(slabs::build(CanonicalNet3d::SimpleCubic, FlatSlabKind::TorusSlab, [2, 2, 2]).unwrap()), None);
    let mut p = Panel { pending: Some((S4Sub::Slabs, String::from("X"))), ..Default::default() };
    p.wanted[S4Sub::Slabs.idx()] = Some(String::from("W"));
    p.accept(Ok(Ok(patch())));
    assert!(p.slab.out.is_none() && p.built[S4Sub::Slabs.idx()].is_none(), "stale X dropped");
    p.pending = Some((S4Sub::Slabs, String::from("W")));
    p.accept(Ok(Ok(patch())));
    assert_eq!(p.slab.out.as_ref().map(|s| s.positions.len()), Some(8), "wanted W accepted: 2*2*2 sites");
    assert_eq!(p.built[S4Sub::Slabs.idx()].as_deref(), Some("W"));
}

#[test]
fn a_lattice_change_forgets_a_lattice_job_only() {
    let mut app = new_app();
    for (sub, forgotten) in [(S4Sub::Nanotubes, true), (S4Sub::Slabs, false)] {
        let mut p = Panel { lattice_key: String::from("old lattice"), pending: Some((sub, String::from("X"))), ..Default::default() };
        p.wanted[sub.idx()] = Some(String::from("X"));
        app.workbench_call(|_, cx, _, _| p.track_lattice(cx));
        assert_eq!(p.pending.is_none(), forgotten, "{sub:?}");
        assert_eq!(p.wanted[sub.idx()].is_none(), forgotten, "{sub:?}");
    }
}

/// The overlay lives in the sheet frame (s along C_h, z along T): C_h -> (|C_h|, 0), T -> (0, |T|), and the second
/// period of sites is the first shifted by exactly the drawn T.
#[test]
fn tube_overlay_is_drawn_in_the_unrolled_sheet_frame() {
    let (cell, _, _) = honeycomb();
    let t = tubes::build_tube(&cell, &TubeParams { n: 4, m: 2, periods: 2, periodic: false, std60: true }).unwrap();
    let (c, tt) = (tubes::to_sheet(&t, t.chiral_vec), tubes::to_sheet(&t, t.trans_vec));
    let tol = 1e-9 * t.circumference.max(t.period_length);
    assert!((c[0] - t.circumference).abs() < tol && c[1].abs() < tol, "C_h -> {c:?}");
    assert!(tt[0].abs() < tol && (tt[1] - t.period_length).abs() < tol, "T -> {tt:?}");
    let k = t.atoms_per_unit;
    for (i, (a, b)) in t.positions_2d.iter().zip(&t.positions_2d[k..]).enumerate() {
        assert!((b[0] - a[0] - tt[0]).abs() < tol && (b[1] - a[1] - tt[1]).abs() < tol, "site {i}");
    }
}

/// a1 = (1, 0), a2 = s(-1/2, sqrt3/2), (n, m) = (2, 1): T = (t1, t2) needs t2 / t1 = -(2 - s/2) / (s(s - 1)), so for
/// s = 1 + 1e-4 the search takes T = (-1, 14998) and d = 2*14998 + 1 = 29997 sites per cell (over the cap);
/// s = 1 + 1e-3 gives T = (-1, 1498), 2997 sites. Graphene (12, 11) over 4 periods: 4*397*4 = 6352 sites.
#[test]
fn a_huge_tube_cell_is_refused_before_it_is_built() {
    let near_hex = |s: f64| Cell2D { a1: [1.0, 0.0], a2: [-0.5 * s, 0.75_f64.sqrt() * s], basis: vec![[0.0, 0.0]], bonds: Vec::new() };
    let p = TubeParams { n: 2, m: 1, periods: 1, periodic: false, std60: true };
    let e = tubes::build_tube(&near_hex(1.0 + 1e-4), &p).unwrap_err();
    assert!(e.contains("REFUSED") && e.contains("(-1, 14998)") && e.contains("29997 sites"), "{e}");
    assert_eq!(tubes::tube_size(&near_hex(1.0 + 1e-3), &p).unwrap(), ([-1, 1498], 2997));
    let (cell, _, _) = honeycomb();
    let big = TubeParams { n: 12, m: 11, periods: 4, periodic: false, std60: true };
    assert_eq!(tubes::tube_size(&cell, &big).unwrap().1, 4 * 397 * 4);
}

#[test]
fn flake_bonds_are_checked_and_the_cut_is_named_by_its_shape() {
    let square = [(0, 1), (1, 2), (2, 3), (3, 0)];
    assert!(flakes::check_bonds(4, &square, &[vec![0, 1, 2, 3], vec![1, 2, 3, 0]]).is_ok(), "C4 maps the 4-cycle onto itself");
    let e = flakes::check_bonds(4, &square, &[vec![1, 0, 2, 3]]).unwrap_err();
    assert!(e.contains("(1, 2) to (0, 2)"), "swapping 0, 1 sends bond 1-2 to the non-bond 0-2: {e}");
    assert!(flakes::check_bonds(4, &square, &[vec![0, 0, 2, 3]]).unwrap_err().contains("does not permute"));
    assert!(flakes::cut_label(Shape::Polygon, Some(6), 2.0).contains("regular 6-gon cut, apothem"));
    let ws = flakes::cut_label(Shape::WignerSeitz, Some(6), 2.31);
    assert!(ws.contains("Wigner-Seitz") && ws.contains("circumradius") && !ws.contains("apothem"), "{ws}");
    assert!(flakes::cut_label(Shape::Disk, None, 1.0).starts_with("disk cut, radius"));
}

#[test]
fn platycosm_cell_count_is_evidence_and_only_the_theorem_is_proved() {
    let l = platycosms::cw_count_line(4, 12, 12, 4, 0);
    assert!(l.starts_with("[COMPUTATIONAL EVIDENCE]") && !l.contains("[PROVED]") && l.contains("not the crystal net"), "{l}");
    assert!(platycosms::CHI_THEOREM_LINE.starts_with("[PROVED]"));
}

/// Square net, mirror y -> -y: the core's glide axis is +-a1 and the across vector +-a2. HCP lattice: a1 + 2a2 =
/// (1, 0, 0) + 2(-1/2, sqrt3/2, 0) = (0, sqrt3, 0); (1/2, 0, 0) is no integer combination.
#[test]
fn glide_slabs_label_their_repeats_along_the_glide_axes() {
    assert!(slabs::repeats_label(FlatSlabKind::KleinSlab).contains("along glide"));
    assert!(slabs::repeats_label(FlatSlabKind::MoebiusTube).contains("along glide"));
    assert!(slabs::repeats_label(FlatSlabKind::TorusSlab).contains("(a1, a2, a3)"));
    let k = slabs::build(CanonicalNet3d::SimpleCubic, FlatSlabKind::KleinSlab, [4, 4, 3]).unwrap();
    let g = slabs::glide_axes(CanonicalNet3d::SimpleCubic, &k).expect("Klein slab glide axes");
    assert_eq!(g.along.map(|[x, y]| [x.abs(), y]), Some([1, 0]), "{g:?}");
    assert_eq!(g.across.map(|[x, y]| [x, y.abs()]), Some([0, 1]), "{g:?}");
    let t = slabs::build(CanonicalNet3d::SimpleCubic, FlatSlabKind::TorusSlab, [2, 2, 2]).unwrap();
    assert!(slabs::glide_axes(CanonicalNet3d::SimpleCubic, &t).is_none());
    let hex = papertilling_core::flat3d::Cell3D::from_canonical_net(CanonicalNet3d::HCP).lattice;
    assert_eq!(slabs::lattice_coords(&hex, [0.0, 3.0_f64.sqrt(), 0.0]), Some([1, 2]));
    assert_eq!(slabs::lattice_coords(&hex, [0.5, 0.0, 0.0]), None);
}
