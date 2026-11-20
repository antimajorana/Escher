//! C1 WP-U2 tests for S5 curved 2D growth: 0-based layers, S² closure + hosohedra meridian edge keys,
//! E² `{4,4}/{3,6}/{6,3}` seeded growth across all 3 seeds, H² `{5,4}` seeded counts, stop banners at caps,
//! and kittest slider-to-cap checks.

use egui_kittest::kittest::Queryable;
use papertilling_core::growth::SeedKind;

use super::growth_table::GROWTH_HEADERS;
use super::tests::table_col;
use super::{S5Sub, large_h2, pq, surfaces, wythoff};
use crate::analysis_mode::tests::{harness, wait_until};
use crate::app::LatticesView;
use crate::game_menu::test_util::goto_lattices_view;

#[test]
fn s2_icosahedron_and_hosohedron_closure_and_meridian_edge_keys() {
    // c1-plan.md §2.4: icosahedron {3,5} Cell seed: S_0..=S_3 = [1, 10, 19, 20], χ = [1, 1, 1, 2], Closed at layer 3.
    let ico = pq::build(&pq::PqJob { p: 3, q: 5, layers: 6, seed: SeedKind::Cell }).unwrap();
    assert_eq!(ico.table.as_ref().unwrap().0.as_slice(), &GROWTH_HEADERS);
    assert_eq!(table_col(&ico, 1), ["1", "10", "19", "20"]);
    assert_eq!(table_col(&ico, 5), ["1", "1", "1", "2"]);
    assert!(ico.stopped.as_deref().is_some_and(|s| s.contains("Closed at layer 3") && s.contains("χ = 2")));

    // Hosohedron {2,4}: Vertex seed at pole 0 contains all 4 lunes at layer 0 -> immediate closure (χ = 2).
    let hos_v = pq::build(&pq::PqJob { p: 2, q: 4, layers: 4, seed: SeedKind::Vertex }).unwrap();
    assert_eq!(table_col(&hos_v, 1), ["4"]);
    assert_eq!(table_col(&hos_v, 5), ["2"]);
    assert!(hos_v.stopped.as_deref().is_some_and(|s| s.contains("Closed at layer 0")));

    // Hosohedron {2,4}: Edge seed on meridian 0 holds 2 lunes at layer 0 (∂V = 2, ∂E = 2, χ = 1)
    // and closes with all 4 lunes at layer 1 (∂V = 0, ∂E = 0, χ = 2).
    let hos_e = pq::build(&pq::PqJob { p: 2, q: 4, layers: 4, seed: SeedKind::Edge }).unwrap();
    assert_eq!(table_col(&hos_e, 1), ["2", "4"]);
    assert_eq!(table_col(&hos_e, 3), ["2", "0"], "∂V");
    assert_eq!(table_col(&hos_e, 4), ["2", "0"], "∂E");
    assert_eq!(table_col(&hos_e, 5), ["1", "2"], "χ");
    assert!(hos_e.stopped.as_deref().is_some_and(|s| s.contains("Closed at layer 1")));
}

#[test]
fn e2_regular_tilings_all_three_seeds_match_closed_forms_from_layer_zero() {
    // c1-plan.md §2.4 closed forms for n = 0..=6:
    // {4,4}: Cell (2n+1)², Vertex (2n+2)², Edge (2n+1)(2n+2)
    // {3,6}: Cell 6n²+6n+1, Vertex 6(n+1)², Edge 2(3n+1)(n+1)
    // {6,3}: Cell 3n²+3n+1, Vertex 3(n+1)², Edge (3n+2)(n+1)
    type F = fn(usize) -> usize;
    let cases: [(usize, usize, SeedKind, F); 9] = [
        (4, 4, SeedKind::Cell, |n| (2 * n + 1).pow(2)),
        (4, 4, SeedKind::Vertex, |n| (2 * n + 2).pow(2)),
        (4, 4, SeedKind::Edge, |n| (2 * n + 1) * (2 * n + 2)),
        (3, 6, SeedKind::Cell, |n| 6 * n * n + 6 * n + 1),
        (3, 6, SeedKind::Vertex, |n| 6 * (n + 1).pow(2)),
        (3, 6, SeedKind::Edge, |n| 2 * (3 * n + 1) * (n + 1)),
        (6, 3, SeedKind::Cell, |n| 3 * n * n + 3 * n + 1),
        (6, 3, SeedKind::Vertex, |n| 3 * (n + 1).pow(2)),
        (6, 3, SeedKind::Edge, |n| (3 * n + 2) * (n + 1)),
    ];
    for (p, q, seed, form) in cases {
        let o = pq::build(&pq::PqJob { p, q, layers: 6, seed }).unwrap();
        assert!(o.stopped.is_none());
        let want: Vec<String> = (0..=6).map(|n| form(n).to_string()).collect();
        assert_eq!(table_col(&o, 1), want, "{{{p},{q}}} {seed:?}");
        assert!(table_col(&o, 5).iter().all(|c| c == "1"), "{{{p},{q}}} {seed:?} χ = 1");
    }
}

#[test]
fn h2_54_seeded_counts_and_stop_banners_across_pq_wythoff_surfaces_and_large_h2() {
    // c1-plan.md §2.4: H² {5,4}:
    // Cell 1, 11, 51, 201, 761; Vertex 4, 24, 100, 384; Edge 2, 16, 70, 272; χ = 1 throughout.
    let c = pq::build(&pq::PqJob { p: 5, q: 4, layers: 4, seed: SeedKind::Cell }).unwrap();
    assert_eq!(table_col(&c, 1), ["1", "11", "51", "201", "761"]);
    assert!(c.stopped.is_none());

    let v = pq::build(&pq::PqJob { p: 5, q: 4, layers: 8, seed: SeedKind::Vertex }).unwrap();
    assert_eq!(table_col(&v, 1), ["4", "24", "100", "384"]);
    assert!(v.stopped.as_deref().is_some_and(|s| s.contains("Stopped at layer 3 (requested 8)")));

    let s = surfaces::build(&surfaces::SurfJob {
        p: 5,
        q: 4,
        layers: 8,
        seed: SeedKind::Edge,
        surface: 0,
        twist: 0.0,
    })
    .unwrap();
    assert_eq!(table_col(&s, 1), ["2", "16", "70", "272"]);
    assert!(s.stopped.as_deref().is_some_and(|m| m.contains("Stopped at layer 3 (requested 8)")));

    // Wythoff {7,3} truncated at slider max 6 hits WYTHOFF_TILE_BUDGET = 600 (|S_5| = 603 > 600).
    let w = wythoff::build(&wythoff::WyJob { p: 7, q: 3, op: 3, layers: 6 }).unwrap();
    assert!(w.stopped.as_deref().is_some_and(|m| m.contains("face budget 600")));
    assert!(w.extra_table.is_some());

    // Large H² {7,3} at target = 50 stops mid-shell in layer 3 (21 of 56 faces built; complete through layer 2).
    let lg = large_h2::build(&large_h2::LargeJob {
        p: 7,
        q: 3,
        target: 50,
        kernel: 0,
        model: super::draw::DiskModel::Poincare,
    })
    .unwrap();
    assert!(lg.stopped.as_deref().is_some_and(|m| m.contains("layer 3 is partial (21 of 56 faces built; complete through layer 2)")));
}

#[test]
fn kittest_s5_sliders_driven_to_caps_show_stop_banners() {
    let mut h = harness();
    goto_lattices_view(&mut h, LatticesView::Curved2D);
    wait_until(&mut h, "S5 init", |a| !a.workbench_busy() && !a.workbench.dirty);

    // Regular {5,4} at slider max layers = 8 -> Stopped banner appears.
    {
        let app = h.state_mut();
        app.workbench.s5.sub = S5Sub::Regular;
        app.workbench.s5.pq.p = 5;
        app.workbench.s5.pq.q = 4;
        app.workbench.s5.pq.layers = 8;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    }
    wait_until(&mut h, "S5 pq cap", |a| !a.workbench_busy() && !a.workbench.dirty);
    assert!(h.query_by_label_contains("Stopped at layer 3 (requested 8)").is_some());

    // Wythoff {7,3} at slider max layers = 6 -> face budget 600 banner appears.
    {
        let app = h.state_mut();
        app.workbench.s5.sub = S5Sub::Wythoff;
        app.workbench.s5.wy.p = 7;
        app.workbench.s5.wy.q = 3;
        app.workbench.s5.wy.layers = 6;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    }
    wait_until(&mut h, "S5 wythoff cap", |a| !a.workbench_busy() && !a.workbench.dirty);
    assert!(h.query_by_label_contains("face budget 600").is_some());

    // Surfaces {5,4} at slider max layers = 8 -> Stopped banner appears.
    {
        let app = h.state_mut();
        app.workbench.s5.sub = S5Sub::Surfaces;
        app.workbench.s5.surf.p = 5;
        app.workbench.s5.surf.q = 4;
        app.workbench.s5.surf.layers = 8;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    }
    wait_until(&mut h, "S5 surfaces cap", |a| !a.workbench_busy() && !a.workbench.dirty);
    assert!(h.query_by_label_contains("Stopped at layer 3 (requested 8)").is_some());
}
