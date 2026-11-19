//! S5 panel tests. Expected values are hand counts or published formulas, never values read back from the panel:
//! - δ = 4 − (p−2)(q−2): {5,3} δ = 1 (S²), {4,4} δ = 0 (E²), {7,3} δ = −1 (H²); 1/p + 1/q − 1/2 = δ/(2pq), so
//!   {5,3} gives 1/30 and {7,3} gives −1/42 (Coxeter, Regular Polytopes, §1.7, §4.5).
//! - Dodecahedron {5,3}: V, E, F = 20, 30, 12 (Coxeter, Table I); its fan mesh has 12 × 5 triangles × 3 vertices.
//! - {7,3} vertex-completion shells 1, 7, 21 (hand count: 7 edge neighbours, then 7 × 3), so 2 layers give 29 faces;
//!   a disk patch has V − E + F = 1.
//! - Truncated icosahedron 5.6.6: 60, 90, 32 with 12 pentagons and 20 hexagons (Cromwell, Polyhedra, ch. 2).
//! - *632 is p6m (Conway, Burgiel & Goodman-Strauss 2008); χ(*237) = 2 − 1 − (1/4 + 1/3 + 3/7) = −1/84
//!   (corner n costs (n−1)/(2n)).
//! - Klein quartic {7,3}: genus 3, |PSL(2,7)| = 168, so V, E, F = 168/3, 168/2, 168/7 = 56, 84, 24.

use std::time::{Duration, Instant};

use egui_kittest::kittest::Queryable;
use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::noneuclidean2d::build_unified_pq;

use super::draw::{CLIP_R, s2_picture};
use super::{S5Out, S5Sub, check_symbol, excess_fraction, large_h2, orbifolds, pq, surfaces, wythoff};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, wait_until};
use crate::app::{LatticesView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;
use crate::workbench::{WbPanel, WorkbenchView};

fn row<'a>(o: &'a S5Out, k: &str) -> &'a str {
    o.rows.iter().find(|(a, _)| a == k).map(|(_, b)| b.as_str()).unwrap_or_else(|| panic!("row {k:?} missing: {:?}", o.rows))
}

fn table_col(o: &S5Out, col: usize) -> Vec<String> {
    o.table.as_ref().expect("table").1.iter().map(|r| r[col].clone()).collect()
}

fn refused<T>(r: Result<T, String>) -> String {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected a refusal with a reason"),
    }
}

fn bbox_ok(s: &Scene2D) -> bool {
    let Scene2D::VectorScene { primitives: p } = s else { return false };
    let (lo, hi) = p.bbox;
    lo.iter().chain(hi.iter()).all(|v| v.is_finite() && v.abs() <= CLIP_R as f32 + 1e-3)
}

#[test]
fn delta_badges_and_curvature_fractions_by_hand() {
    assert_eq!(check_symbol(5, 3), Ok(GeometryKind::Spherical));
    assert_eq!(check_symbol(4, 4), Ok(GeometryKind::Euclidean));
    assert_eq!(check_symbol(7, 3), Ok(GeometryKind::Hyperbolic));
    assert!(check_symbol(13, 3).is_err(), "p above the panel range is refused");
    assert!(check_symbol(1, 3).is_err(), "p < 2 is refused by the core");
    assert_eq!(excess_fraction(5, 3).as_deref(), Some("1/30"));
    assert_eq!(excess_fraction(7, 3).as_deref(), Some("-1/42"));
    assert_eq!(excess_fraction(4, 4).as_deref(), Some("0"));
    assert!(super::delta_badge(7, 3).contains("= -1: H² (hyperbolic) [PROVED]"));
}

#[test]
fn dodecahedron_and_73_patch_hand_counts() {
    let o = pq::build(&pq::PqJob { p: 5, q: 3, layers: 2 }).unwrap();
    assert_eq!(row(&o, "V / E / F"), "20 / 30 / 12");
    assert_eq!(row(&o, "χ = V − E + F (sphere)"), "2");
    assert_eq!(row(&o, "1/p + 1/q − 1/2 = δ/(2pq)"), "1/30");
    assert_eq!(o.mesh.face_count, 12);
    assert_eq!(o.mesh.faces.len(), 12 * 5 * 3, "12 pentagons x 5 fan triangles x 3 vertices");
    assert!(bbox_ok(&o.scene2d));
    let h = pq::build(&pq::PqJob { p: 7, q: 3, layers: 2 }).unwrap();
    assert!(row(&h, "V / E / F").ends_with("/ 29"), "{:?}", h.rows);
    assert_eq!(row(&h, "χ = V − E + F (finite patch)"), "1");
    let e = pq::build(&pq::PqJob { p: 4, q: 4, layers: 1 }).unwrap();
    assert_eq!(row(&e, "Geometry"), "E² (Euclidean)");
    assert!(row(&e, "V / E / F").ends_with("/ 9"), "{{4,4}} 1 layer: 3 x 3 squares");
    assert!(refused(pq::build(&pq::PqJob { p: 4, q: 4, layers: 9 })).contains("allowed 1..=6"));
}

#[test]
fn stereographic_pictures_of_the_five_platonic_solids_stay_bounded() {
    for (p, q) in [(3, 3), (4, 3), (3, 4), (5, 3), (3, 5)] {
        let t = build_unified_pq(p, q, 1).unwrap();
        let (s, dropped) = s2_picture(&t.coords_3d, &t.faces, "t", "n").expect("vertices on the unit sphere");
        assert_eq!(dropped, 0, "{{{p},{q}}}");
        assert!(bbox_ok(&s), "{{{p},{q}}}");
        let Scene2D::VectorScene { primitives } = s else { unreachable!() };
        assert_eq!(primitives.polygons.len(), t.n_faces - 1, "every face but the outside one is filled");
    }
}

#[test]
fn truncated_icosahedron_is_566_with_12_pentagons_and_20_hexagons() {
    let digons = wythoff::build(&wythoff::WyJob { p: 2, q: 5, op: 0, layers: 1 });
    assert!(refused(digons).contains("digon"), "a hosohedron is refused here, never miscounted");
    let o = wythoff::build(&wythoff::WyJob { p: 3, q: 5, op: 3, layers: 2 }).unwrap();
    assert_eq!(row(&o, "Vertex symbol"), "5.6.6");
    assert_eq!(row(&o, "V / E / F"), "60 / 90 / 32");
    assert_eq!(table_col(&o, 0), ["5-gon", "6-gon"]);
    assert_eq!(table_col(&o, 1), ["12", "20"]);
    assert!(o.notes[0].contains("[PROVED]"));
    let snub = wythoff::build(&wythoff::WyJob { p: 4, q: 3, op: 7, layers: 1 }).unwrap();
    assert_eq!(row(&snub, "V / E / F"), "24 / 60 / 38", "snub cube");
    assert!(snub.notes[0].contains("[COMPUTATIONAL EVIDENCE]"));
}

#[test]
fn large_kernels_give_the_hand_shells_1_7_21() {
    for k in 0..large_h2::KERNELS.len() {
        let j = large_h2::LargeJob { p: 7, q: 3, target: 29, kernel: k, model: super::draw::DiskModel::Poincare };
        let o = large_h2::build(&j).unwrap();
        assert_eq!(table_col(&o, 1), ["1", "7", "21"], "kernel {k}");
        assert_eq!(row(&o, "V − E + F"), "1");
        assert!(bbox_ok(&o.scene2d));
    }
    for model in super::draw::DiskModel::ALL {
        let j = large_h2::LargeJob { p: 5, q: 4, target: 2000, kernel: 1, model };
        let o = large_h2::build(&j).unwrap();
        assert!(o.mesh.face_count >= 2000, "{model:?}: {}", o.mesh.face_count);
        assert!(bbox_ok(&o.scene2d), "{model:?}");
    }
    let bad = large_h2::LargeJob { p: 4, q: 4, target: 100, kernel: 0, model: super::draw::DiskModel::Poincare };
    assert!(refused(large_h2::build(&bad)).contains("not hyperbolic"));
}

#[test]
fn surfaces_refuse_non_hyperbolic_and_unfold_every_face() {
    let flat = surfaces::SurfJob { p: 4, q: 4, layers: 2, surface: 0, twist: 0.2 };
    assert!(refused(surfaces::build(&flat)).contains("only H² tilings"));
    for s in 0..surfaces::SURFACES.len() {
        let o = surfaces::build(&surfaces::SurfJob { p: 7, q: 3, layers: 2, surface: s, twist: 0.3 }).unwrap();
        assert_eq!(row(&o, "Faces placed"), "29");
        assert!(row(&o, "Net pieces / glued edges / cut edges").starts_with("29 / 28 /"), "a spanning tree of 29 faces has 28 edges");
        let svg = o.svg.as_deref().unwrap();
        assert!(svg.starts_with("<svg") && svg.matches("<polygon").count() == 29);
        assert_eq!(o.notes[0].starts_with("Isometric embedding"), s == 0, "only the hyperboloid bowl is called an isometry");
        assert!(o.notes.iter().all(|n| n.contains("[PROVED]") || n.contains("[COMPUTATIONAL EVIDENCE]")), "{:?}", o.notes);
    }
    let dir = std::env::temp_dir().join("wb_s5_net_test.svg");
    let msg = surfaces::export_to(&dir, "<svg/>").unwrap();
    assert!(msg.contains("6 bytes"));
    let _ = std::fs::remove_file(dir);
}

#[test]
fn orbifold_and_vertex_cards_and_klein_quartic() {
    let o = orbifolds::orbifold_card("*632").unwrap();
    assert_eq!(o.wallpaper_group.as_deref(), Some("p6m"));
    assert_eq!((o.chi_num, o.chi_den), (0, 1));
    let h = orbifolds::orbifold_card("*237").unwrap();
    assert_eq!((h.chi_num, h.chi_den, h.geometry_kind), (-1, 84, GeometryKind::Hyperbolic));
    assert!(orbifolds::orbifold_card("*23").is_err(), "*mn with m != n is a bad orbifold");
    assert!(orbifolds::orbifold_card(&"2".repeat(100)).unwrap_err().contains("longer than"));
    assert_eq!(orbifolds::vertex_card("3.4.6.4").unwrap().geometry_kind, GeometryKind::Euclidean);
    assert!(orbifolds::vertex_card("3.4.5").is_err(), "odd triangle between a square and a pentagon cannot close");
    let k = orbifolds::build_tg("klein_7_3").unwrap();
    assert_eq!(row(&k, "Genus / χ"), "3 / -4");
    assert_eq!(row(&k, "V / E / F"), "56 / 84 / 24");
    assert!(k.notes.iter().any(|n| n.contains("passed")));
    assert_eq!(k.table.as_ref().unwrap().1.len(), 84 - 24 + 1, "E − F + 1 side pairings");
    assert!(bbox_ok(&k.scene2d));
}

/// The Euler note agrees with the χ row; returns true when it reports an unexpected value (≠ 2 closed, ≠ 1 patch).
fn euler_consistent(o: &S5Out, what: &str) -> bool {
    let chi = o.rows.iter().find(|(k, _)| k.starts_with("χ")).map(|(_, v)| v.as_str()).unwrap_or_else(|| panic!("{what}: no χ row"));
    let note = o.notes.iter().find(|n| n.contains("V − E + F")).unwrap_or_else(|| panic!("{what}: no Euler note"));
    if note.contains("= 2 (Euler) [PROVED]") {
        assert_eq!(chi, "2", "{what}");
    } else if note.starts_with("V − E + F = 1,") {
        assert_eq!(chi, "1", "{what}");
    } else {
        assert!(note.starts_with(&format!("V − E + F = {chi} ≠")), "{what}: {note} vs χ {chi}");
        return true;
    }
    false
}

/// Every UI-reachable symbol gives a result or a refusal with a reason, never a panic or a non-finite picture.
#[test]
fn sweep_reachable_symbols_ok_or_reasoned_refusal() {
    let t0 = Instant::now();
    let (mut ok, mut refused, mut wy_refused) = (0, 0, 0);
    let mut odd_chi: Vec<String> = Vec::new();
    for p in 2..=super::MAX_PQ {
        for q in 2..=super::MAX_PQ {
            let layers = match check_symbol(p, q).unwrap() {
                GeometryKind::Hyperbolic => 2,
                _ => 1,
            };
            match pq::build(&pq::PqJob { p, q, layers }) {
                Ok(o) => {
                    assert!(bbox_ok(&o.scene2d), "{{{p},{q}}}");
                    assert!(!euler_consistent(&o, &format!("{{{p},{q}}}")), "regular {{{p},{q}}}: unexpected χ");
                    ok += 1;
                }
                Err(e) => {
                    assert!(!e.is_empty());
                    refused += 1;
                }
            }
            for op in 0..wythoff::OPS.len() {
                let what = format!("Wythoff {{{p},{q}}} op {op}");
                match wythoff::build(&wythoff::WyJob { p, q, op, layers: 1 }) {
                    Ok(o) => {
                        assert!(bbox_ok(&o.scene2d), "{what}");
                        if euler_consistent(&o, &what) {
                            odd_chi.push(what);
                        }
                    }
                    Err(e) => {
                        assert!(!e.is_empty(), "{what}");
                        wy_refused += 1;
                    }
                }
            }
        }
    }
    eprintln!("# time regular + Wythoff sweep (p, q <= 12): {:.3}s; regular ok {ok}, refused {refused}", t0.elapsed().as_secs_f64());
    eprintln!("Wythoff: {wy_refused} refused with a reason; {} results flag an unexpected V − E + F: {odd_chi:?}", odd_chi.len());
    let t1 = Instant::now();
    for p in 3..=super::MAX_PQ {
        for q in 3..=super::MAX_PQ {
            let j = large_h2::LargeJob { p, q, target: 100, kernel: (p + q) % 3, model: super::draw::DiskModel::Klein };
            match large_h2::build(&j) {
                Ok(o) => assert!(bbox_ok(&o.scene2d), "{{{p},{q}}}"),
                Err(e) => assert!(e.contains("not hyperbolic"), "{{{p},{q}}}: {e}"),
            }
        }
    }
    eprintln!("# time large-patch sweep: {:.3}s", t1.elapsed().as_secs_f64());
}

fn new_app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn settle(app: &mut PapertillingApp, ctx: &egui::Context, what: &str) {
    let t0 = Instant::now();
    let frame = |app: &mut PapertillingApp| {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _ = app.frame_ui(ui);
        });
    };
    for _ in 0..4 {
        frame(app);
    }
    while app.workbench_busy() || app.workbench.dirty {
        assert!(t0.elapsed() < Duration::from_secs(60), "{what}: still busy after 60 s");
        std::thread::sleep(Duration::from_millis(10));
        frame(app);
    }
}

#[test]
fn every_s5_subrow_default_builds_with_a_mesh() {
    let ctx = egui::Context::default();
    let mut app = new_app();
    app.select_lattices_view(LatticesView::Curved2D);
    settle(&mut app, &ctx, "enter S5");
    assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, WorkbenchView::S5Curved2d));
    for s in S5Sub::ALL {
        app.workbench.s5.sub = s;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
        settle(&mut app, &ctx, s.label());
        let p = &app.workbench.s5;
        assert!(p.error().is_none(), "{s:?}: {:?}", p.error());
        assert!(p.outs[s.idx()].is_some(), "{s:?}");
        assert!(app.view3d.has_geometry(), "{s:?}");
        assert!(!p.scenes().caption.contains("placeholder"), "{s:?}");
    }
    app.workbench.smoke_defaults();
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(&mut app, &ctx, "smoke defaults");
    assert_eq!(app.workbench.s5.sub, S5Sub::Regular);
    assert!(app.workbench.s5.error().is_none() && app.view3d.has_geometry());
}

#[test]
fn s5_panel_state_survives_json_roundtrip() {
    let mut p = super::Panel { sub: S5Sub::LargeH2, ..Default::default() };
    p.pq.set(4, 4, 3);
    p.wy.op = 8;
    p.large.target = 777;
    p.large.model = super::draw::DiskModel::UpperHalfPlane;
    p.surf.surface = 3;
    p.orb.orb_text = String::from("22x");
    let v = p.save();
    let back: serde_json::Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    let mut q = super::Panel::default();
    let mut notes = Vec::new();
    q.restore(&back, &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(q.save(), v);
    let junk = serde_json::json!({"sub": "Nope", "pq": {"p": 99, "layers": "x"}, "large": {"target": 10_000_000, "model": "?"}});
    q.restore(&junk, &mut notes);
    assert_eq!(notes.len(), 1);
    assert_eq!((q.sub, q.pq.p, q.large.target), (S5Sub::LargeH2, super::MAX_PQ, large_h2::MAX_TARGET), "junk ignored, sizes clamped");
    assert_eq!(q.large.model, super::draw::DiskModel::UpperHalfPlane);
}

#[test]
fn s5_subrows_quarantine_clean() {
    let mut h = harness();
    h.get_by_label("1. Lattices").click();
    h.run_steps(2);
    h.get_by_label("{p,q} on S² / E² / H²").click();
    wait_until(&mut h, "S5", |a| !a.workbench_busy() && !a.workbench.dirty);
    for s in S5Sub::ALL.into_iter().chain([S5Sub::Regular]) {
        h.get_by_label(s.label()).click();
        wait_until(&mut h, s.label(), |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().workbench.s5.sub, s);
        assert!(h.state().workbench.s5.error().is_none(), "{s:?}: {:?}", h.state().workbench.s5.error());
        assert_no_forbidden_tokens(&h, s.label());
    }
}

/// The deepest layer settings the UI allows, on corner symbols (E² 6, H² 4 for Regular; 3 for Wythoff).
#[test]
fn max_layer_corner_symbols_build_or_refuse_with_reason() {
    let t0 = Instant::now();
    for (p, q, layers) in [(4, 4, 6), (3, 6, 6), (6, 3, 6), (7, 3, 4), (3, 7, 4), (12, 12, 4), (3, 12, 4), (12, 3, 4)] {
        let o = pq::build(&pq::PqJob { p, q, layers }).unwrap_or_else(|e| panic!("{{{p},{q}}} x{layers}: {e}"));
        assert!(bbox_ok(&o.scene2d), "{{{p},{q}}} x{layers}");
        euler_consistent(&o, &format!("{{{p},{q}}} x{layers}"));
    }
    for (p, q) in [(4, 4), (3, 6), (7, 3), (12, 12), (3, 12)] {
        for op in 0..wythoff::OPS.len() {
            match wythoff::build(&wythoff::WyJob { p, q, op, layers: wythoff::MAX_LAYERS }) {
                Ok(o) => assert!(bbox_ok(&o.scene2d), "{{{p},{q}}} op {op}"),
                Err(e) => assert!(!e.is_empty()),
            }
        }
    }
    for s in 0..surfaces::SURFACES.len() {
        surfaces::build(&surfaces::SurfJob { p: 12, q: 12, layers: 4, surface: s, twist: 2.0 }).unwrap();
    }
    eprintln!("# time max-layer corner symbols: {:.3}s", t0.elapsed().as_secs_f64());
}

/// Pilot timing of the heaviest builds (printed with `# time`; no time is asserted).
#[test]
fn pilot_timing_heaviest_s5_jobs() {
    let d = large_h2::LargeState::default();
    assert!(d.target <= large_h2::MAX_DEFAULT_TARGET);
    for (target, kernel) in [(d.target, d.kernel), (large_h2::MAX_DEFAULT_TARGET, 2), (large_h2::MAX_TARGET, 0), (large_h2::MAX_TARGET, 2)] {
        let t = Instant::now();
        let o = large_h2::build(&large_h2::LargeJob { p: 7, q: 3, target, kernel, model: d.model }).unwrap();
        let built = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = o.scene2d.clone();
        eprintln!("# time large {{7,3}} target {target} kernel {kernel}: build + scenes {built:.3}s, scene clone {:.3}s", t.elapsed().as_secs_f64());
    }
    let t = Instant::now();
    surfaces::build(&surfaces::SurfJob { p: 3, q: 7, layers: 4, surface: 4, twist: 0.2 }).unwrap();
    eprintln!("# time surfaces {{3,7}} 4 layers on Kuen + net: {:.3}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    wythoff::build(&wythoff::WyJob { p: 7, q: 3, op: 6, layers: 3 }).unwrap();
    eprintln!("# time Wythoff omnitruncated {{7,3}} 3 layers: {:.3}s", t.elapsed().as_secs_f64());
}
