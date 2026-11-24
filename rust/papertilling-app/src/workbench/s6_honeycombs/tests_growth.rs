//! C1 WP-U2 tests for S6 honeycomb growth: CellCap formatting (never printing partial count `120001`),
//! `boundary_only` readout above 10 000 cells, S³ `{3,3,5}` closure at layer 4, `{4,3,4}` all 4 seeds,
//! paracompact ball monotonicity + `[PROVED]` reason, and headless kittest slider/cap checks.

use egui_kittest::kittest::Queryable;
use papertilling_core::growth::h3_complex::{IDEAL_VERTEX_REASON, INFINITE_CELL_REASON};
use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::honeycomb3d::GrowthRule;
use papertilling_core::honeycomb3d::growth_build::{DRAW_ALL_FACES_MAX, MAX_GROWTH_CELLS};

use super::present::{Built, CoreOut};
use super::{S6Job, run_job};
use crate::analysis_mode::tests::{harness, wait_until};
use crate::app::LatticesView;
use crate::game_menu::test_util::goto_lattices_view;

fn all_text(b: &Built) -> String {
    let mut out = b.caption.clone();
    for (k, v) in &b.rows {
        out.push(' ');
        out.push_str(k);
        out.push(' ');
        out.push_str(v);
    }
    for n in &b.notes {
        out.push(' ');
        out.push_str(n);
    }
    if let CoreOut::Growth(_, meta) = &b.core {
        for bn in &meta.banners {
            out.push(' ');
            out.push_str(bn);
        }
    }
    out
}

#[test]
fn h3_cell_cap_notes_never_print_partial_lower_bound() {
    // c1-plan.md §2.2 & §2.4: {5,3,5} Vertex + Touching has S_0 = 20, S_1 = 3550, and S_2 > 120 000;
    // {4,3,5} Vertex + Touching has S_0 = 20, S_2 = 23 100, and S_3 > 120 000.
    // The partial counter (120 001) is an incomplete lower bound and must NEVER appear anywhere.
    assert_eq!(MAX_GROWTH_CELLS, 120_000);
    for ((p, q, r), want_stop_layer, want_last_cells) in [((5, 3, 5), 2, 3_550), ((4, 3, 5), 3, 23_100)] {
        let b = run_job(S6Job::SchlafliGrowth {
            p,
            q,
            r,
            seed: SeedKind::Vertex,
            rule: GrowthRule::Touching,
            layers: 10,
        })
        .unwrap();
        let CoreOut::Growth(_, meta) = &b.core else { panic!("expected Growth") };
        assert!(matches!(meta.stopped, Some(Stopped::CellCap { layer, .. }) if layer == want_stop_layer));
        assert_eq!(meta.stats.len(), want_stop_layer as usize);
        assert_eq!(meta.stats.last().unwrap().cells, want_last_cells);

        let text = all_text(&b);
        let expected = format!("layer {want_stop_layer} needs more than {MAX_GROWTH_CELLS} cells (cap {MAX_GROWTH_CELLS})");
        assert!(text.contains(&expected), "{{{p},{q},{r}}} missing cap message in: {text}");
        for forbidden in ["120001", "120,001", "120 001", "120_001"] {
            assert!(!text.contains(forbidden), "{{{p},{q},{r}}} leaked partial count {forbidden:?} in: {text}");
        }
    }
}

#[test]
fn boundary_only_at_435_vertex_s2_reads_23100_cells_and_chi_one_from_stats() {
    // c1-plan.md §2.4: {4,3,5} Vertex + Touching S_2 = 23 100 cells (> DRAW_ALL_FACES_MAX = 10 000) with χ = 1.
    // `cx.cells` is empty because only boundary faces are drawn; readout must take C = 23 100 and χ = 1 from `stats`.
    assert_eq!(DRAW_ALL_FACES_MAX, 10_000);
    let b = run_job(S6Job::SchlafliGrowth {
        p: 4,
        q: 3,
        r: 5,
        seed: SeedKind::Vertex,
        rule: GrowthRule::Touching,
        layers: 2,
    })
    .unwrap();
    let CoreOut::Growth(cx, meta) = &b.core else { panic!("expected Growth") };
    assert!(meta.boundary_only);
    assert!(cx.cells.is_empty(), "interior cells omitted above 10 000");
    assert_eq!(meta.stats.len(), 3);
    assert_eq!(meta.stats[0].cells, 20);
    assert_eq!(meta.stats[2].cells, 23_100);
    assert_eq!(meta.stats[2].euler, 1);

    // Drawn outer boundary of a 3-ball is a 2-sphere: V_∂ − E_∂ + F_∂ = 2.
    let (v, e, f, _) = cx.counts();
    assert_eq!(v as i64 - e as i64 + f as i64, 2);

    assert!(b.rows.iter().any(|(k, val)| k == "Cells in S_K" && val == "23100"), "{:?}", b.rows);
    assert!(b.rows.iter().any(|(k, val)| k == "V − E + F − C" && val == "1"), "{:?}", b.rows);
    assert!(b.caption.contains("23100 cells"), "{}", b.caption);
    assert!(meta.banners.iter().any(|m| m.contains("Boundary-only drawing")), "{:?}", meta.banners);
    assert!(!meta.banners.iter().any(|m| m.contains("not a 3-ball")), "{:?}", meta.banners);
}

#[test]
fn s3_600_cell_closes_at_layer_four_with_chi_zero() {
    // c1-plan.md §2.4: {3,3,5} (600-cell) Vertex + Touching layers 0..=4: [20, 130, 330, 530, 600],
    // χ = [1, 1, 1, 1, 0], Stopped::Closed { layer: 4 }.
    let b = run_job(S6Job::SchlafliGrowth {
        p: 3,
        q: 3,
        r: 5,
        seed: SeedKind::Vertex,
        rule: GrowthRule::Touching,
        layers: 10,
    })
    .unwrap();
    let CoreOut::Growth(cx, meta) = &b.core else { panic!("expected Growth") };
    let cells: Vec<usize> = meta.stats.iter().map(|s| s.cells).collect();
    let eulers: Vec<i64> = meta.stats.iter().map(|s| s.euler).collect();
    assert_eq!(cells, vec![20, 130, 330, 530, 600]);
    assert_eq!(eulers, vec![1, 1, 1, 1, 0]);
    assert_eq!(meta.stopped, Some(Stopped::Closed { layer: 4 }));
    assert_eq!(cx.counts(), (120, 720, 1200, 600));
    assert!(meta.banners.iter().any(|m| m.contains("Closed at layer 4") && m.contains("χ = 0")));
    assert!(!meta.banners.iter().any(|m| m.contains("not a 3-ball")));
}

#[test]
fn cubic_434_through_schlafli_matches_all_four_seed_box_formulas_and_reaches_layer_ten() {
    // Hand formulas (c1-plan.md §2.4): boxes of whole cubes in {4,3,4}:
    // Cell (2n+1)³, Vertex (2n+2)³, Face (2n+1)²(2n+2), Edge (2n+1)(2n+2)².
    type SeedFormula = (SeedKind, fn(usize) -> usize);
    let cases: [SeedFormula; 4] = [
        (SeedKind::Cell, |n| (2 * n + 1).pow(3)),
        (SeedKind::Vertex, |n| (2 * n + 2).pow(3)),
        (SeedKind::Face, |n| (2 * n + 1).pow(2) * (2 * n + 2)),
        (SeedKind::Edge, |n| (2 * n + 1) * (2 * n + 2).pow(2)),
    ];
    for (seed, f) in cases {
        let b = run_job(S6Job::SchlafliGrowth {
            p: 4,
            q: 3,
            r: 4,
            seed,
            rule: GrowthRule::Touching,
            layers: 4,
        })
        .unwrap();
        let CoreOut::Growth(_, meta) = &b.core else { panic!("expected Growth") };
        let got: Vec<usize> = meta.stats.iter().map(|s| s.cells).collect();
        let want: Vec<usize> = (0..=4).map(f).collect();
        assert_eq!(got, want, "{seed:?}");
        assert!(meta.stats.iter().all(|s| s.euler == 1), "{seed:?}");
    }
    // Layer 10 with Vertex seed: (2*10 + 2)³ = 22³ = 10 648 cells (> 10 000 -> boundary_only).
    let deep = run_job(S6Job::SchlafliGrowth {
        p: 4,
        q: 3,
        r: 4,
        seed: SeedKind::Vertex,
        rule: GrowthRule::Touching,
        layers: 10,
    })
    .unwrap();
    let CoreOut::Growth(_, meta) = &deep.core else { panic!("expected Growth") };
    assert_eq!(meta.stats.len(), 11);
    assert_eq!(meta.stats[10].cells, 10_648);
    assert!(meta.boundary_only);
}

#[test]
fn paracompact_ball_radius_monotonicity_and_proved_reasons() {
    // {6,3,3} has infinite {6,3} cells; {3,3,6} has finite tetrahedra with ideal {3,6} vertices.
    for ((p, q, r), want_reason) in [((6, 3, 3), INFINITE_CELL_REASON), ((3, 3, 6), IDEAL_VERTEX_REASON)] {
        let mut counts = Vec::new();
        for rc in [40, 80, 120, 160] {
            let b = run_job(S6Job::ParacompactBall { p, q, r, radius_centi: rc, seed_only: false }).unwrap();
            let CoreOut::Ball(ball) = &b.core else { panic!("expected Ball") };
            assert_eq!(ball.reason, want_reason);
            assert!(b.notes.iter().any(|n| n == want_reason));
            counts.push(ball.faces);
        }
        assert!(counts.windows(2).all(|w| w[1] >= w[0]), "{{{p},{q},{r}}} non-decreasing: {counts:?}");
        assert!(counts.last().unwrap() > counts.first().unwrap(), "{{{p},{q},{r}}} strictly grows: {counts:?}");
    }
}

#[test]
fn kittest_s6_h3_cap_banner_and_paracompact_ball_controls() {
    let mut h = harness();
    h.run_steps(3);
    goto_lattices_view(&mut h, LatticesView::Honeycombs3D);
    wait_until(&mut h, "S6 init", |a| !a.workbench_busy() && !a.workbench.dirty);

    // Drive {5,3,5} to the slider max (10 layers) and verify the stop banner renders without "120001".
    {
        let app = h.state_mut();
        app.workbench.s6.schlafli.p = 5;
        app.workbench.s6.schlafli.q = 3;
        app.workbench.s6.schlafli.r = 5;
        app.workbench.s6.schlafli.seed = SeedKind::Vertex;
        app.workbench.s6.schlafli.layers = 10;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    }
    wait_until(&mut h, "{5,3,5} cap", |a| !a.workbench_busy() && !a.workbench.dirty);
    assert!(h.query_all_by_label_contains("layer 2 needs more than 120000 cells (cap 120000)").next().is_some());
    assert!(h.query_all_by_label_contains("120001").next().is_none());

    // Switch to paracompact {3,3,6}: Ball radius R slider and [PROVED] ideal-vertex reason appear.
    {
        let app = h.state_mut();
        app.workbench.s6.schlafli.p = 3;
        app.workbench.s6.schlafli.q = 3;
        app.workbench.s6.schlafli.r = 6;
        app.workbench.s6.schlafli.surrounding_cells = true;
        app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    }
    wait_until(&mut h, "{3,3,6} ball", |a| !a.workbench_busy() && !a.workbench.dirty);
    assert!(h.query_all_by_label_contains("Ball radius R:").next().is_some());
    assert!(h.query_all_by_label_contains("[PROVED] an ideal vertex lies in infinitely many cells").next().is_some());
}
