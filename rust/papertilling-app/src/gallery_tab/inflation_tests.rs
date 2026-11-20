//! Known-answer and headless UI tests for 2D Gallery layer growth (C1 WP-U2, plan §2.4 & round 6).

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::plane::inflation::InflationRule;

use super::GalleryTabState;
use super::inflation_ui::{E2SeedChoice, MAX_E2_DRAW_FACES, shell_boundary_vertices};
use super::tests::{canvas, frame, select};
use crate::backend::AppBackend;
use crate::canvas2d::Scene2D;
use crate::theme::{StudioTheme, ThemeTokens, typography};

#[test]
fn honeycomb_default_four_layers_gives_75_hexagons_by_hand() {
    // Hand formula (c1-plan.md §2.4 & round 6): {6,3} vertex seed at layer n holds 3(n+1)^2 hexagons.
    // For n = 0..=4: [3, 12, 27, 48, 75], new cells [3, 9, 15, 21, 27], disk χ = 1 throughout.
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    assert!(st.inflation.enabled);
    assert_eq!(st.inflation.rule, InflationRule::Bdf);
    assert_eq!(st.inflation.seed_choice, E2SeedChoice::MostSymmetricVertex);
    assert_eq!(st.inflation.steps, 4);

    select(&mut st, &backend, &mut cv, "named", "p6m", "honeycomb");
    st.set_show_unit_cell(false, &mut cv);
    assert!(st.inflation.last_error.is_none(), "{:?}", st.inflation.last_error);
    assert!(st.inflation.last_stopped.is_none());

    let cells: Vec<usize> = st.inflation.last_shells.iter().map(|s| s.topology.faces).collect();
    let new_cells: Vec<usize> = st.inflation.last_shells.iter().map(|s| s.new_tiles).collect();
    assert_eq!(cells, vec![3, 12, 27, 48, 75]);
    assert_eq!(new_cells, vec![3, 9, 15, 21, 27]);
    assert!(st.inflation.last_shells.iter().all(|s| s.topology.chi == 1));
    match cv.scene() {
        Scene2D::VectorScene { primitives } => assert_eq!(primitives.polygons.len(), 75),
        other => panic!("expected VectorScene, got {other:?}"),
    }
}

#[test]
fn square_vertex_edge_tile_seeds_match_hand_box_formulas() {
    // Hand formulas (c1-plan.md §2.4): {4,4} touching-rule patches are a x b boxes of unit squares,
    // with F = a*b and perimeter ∂V = ∂E = 2(a + b), χ = 1:
    // - Vertex: (2n+2) x (2n+2) -> F = (2n+2)^2, ∂V = ∂E = 8(n+1).
    // - Edge:   (2n+1) x (2n+2) -> F = (2n+1)(2n+2), ∂V = ∂E = 8n + 6.
    // - Tile:   (2n+1) x (2n+1) -> F = (2n+1)^2, ∂V = ∂E = 8n + 4.
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    select(&mut st, &backend, &mut cv, "named", "p4m", "square");
    st.inflation.steps = 4;

    let cases: [(E2SeedChoice, Vec<usize>, Vec<usize>); 3] = [
        (E2SeedChoice::MostSymmetricVertex, vec![4, 16, 36, 64, 100], vec![8, 16, 24, 32, 40]),
        (E2SeedChoice::Edge(0), vec![2, 12, 30, 56, 90], vec![6, 14, 22, 30, 38]),
        (E2SeedChoice::Tile(0), vec![1, 9, 25, 49, 81], vec![4, 12, 20, 28, 36]),
    ];
    for (seed, want_cells, want_bnd) in cases {
        st.inflation.seed_choice = seed;
        st.sync_scene(&mut cv, true);
        assert!(st.inflation.last_error.is_none(), "{seed:?}: {:?}", st.inflation.last_error);
        let got_cells: Vec<usize> = st.inflation.last_shells.iter().map(|s| s.topology.faces).collect();
        let got_bv: Vec<usize> = st.inflation.last_shells.iter().map(shell_boundary_vertices).collect();
        let got_be: Vec<usize> = st.inflation.last_shells.iter().map(|s| s.topology.boundary_edges).collect();
        assert_eq!(got_cells, want_cells, "{seed:?} cells");
        assert_eq!(got_bv, want_bnd, "{seed:?} ∂V");
        assert_eq!(got_be, want_bnd, "{seed:?} ∂E");
        assert!(st.inflation.last_shells.iter().all(|s| s.topology.chi == 1), "{seed:?} χ");
    }
}

#[test]
fn cross_lattice_seed_clamping_never_errors_when_switching_to_smaller_cell() {
    // Kagome has 3 sites, 6 bonds, 3 face orbits; square has 1 site, 2 bonds, 1 face orbit.
    // High indices chosen on kagome clamp cleanly onto square's valid ranges (100, 90, 81 at n = 4).
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    for (choice, want_square_n4) in [
        (E2SeedChoice::Vertex(2), 100),
        (E2SeedChoice::Edge(5), 90),
        (E2SeedChoice::Tile(2), 81),
    ] {
        select(&mut st, &backend, &mut cv, "named", "p6m", "kagome");
        st.inflation.seed_choice = choice;
        st.inflation.steps = 4;
        st.sync_scene(&mut cv, true);
        assert!(st.inflation.last_error.is_none(), "kagome {choice:?}");

        select(&mut st, &backend, &mut cv, "named", "p4m", "square");
        assert!(st.inflation.last_error.is_none(), "square {choice:?}: {:?}", st.inflation.last_error);
        assert_eq!(st.inflation.last_shells.last().unwrap().topology.faces, want_square_n4, "{choice:?}");
    }
}

#[test]
fn face_drawing_cap_stops_at_last_complete_layer_and_sets_stop_note() {
    // Hand formula (c1-plan.md §2.4): {3,6} vertex seed has 6(n+1)^2 triangles:
    // layer 39 -> 6 * 40^2 = 9_600 <= 10_000; layer 40 -> 6 * 41^2 = 10_086 > 10_000.
    assert_eq!(MAX_E2_DRAW_FACES, 10_000);
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    select(&mut st, &backend, &mut cv, "named", "p6m", "triangular");
    st.set_show_unit_cell(false, &mut cv);
    st.inflation.steps = 40;
    st.sync_scene(&mut cv, true);

    assert!(st.inflation.last_error.is_none(), "{:?}", st.inflation.last_error);
    assert_eq!(st.inflation.last_shells.len(), 40, "complete layers 0..=39 kept");
    assert_eq!(st.inflation.last_shells.last().unwrap().n, 39);
    assert_eq!(st.inflation.last_shells.last().unwrap().topology.faces, 9_600);
    let stop = st.inflation.last_stopped.as_deref().expect("stop note set at layer 40");
    assert!(stop.contains("layer 40") && stop.contains("10086") && stop.contains("10000"), "{stop}");
    match cv.scene() {
        Scene2D::VectorScene { primitives } => assert_eq!(primitives.polygons.len(), 9_600),
        other => panic!("expected VectorScene, got {other:?}"),
    }

    let ctx = egui::Context::default();
    frame(&ctx, &mut st, &backend, &mut cv);
    assert_eq!(st.inflation.steps, 40, "drawing a frame without input keeps stored steps");
}

#[test]
fn kittest_more_drawer_reveals_seed_rule_and_periodic_grid_toggle() {
    struct Env {
        backend: AppBackend,
        cv: crate::canvas2d::Canvas2D,
        st: GalleryTabState,
    }
    let backend = AppBackend::new();
    let mut cv = canvas();
    let mut st = GalleryTabState::new(&backend, &mut cv);
    select(&mut st, &backend, &mut cv, "named", "p4m", "square");
    let env = Env { backend, cv, st };
    let tokens = ThemeTokens::paper();
    let mut h = Harness::builder()
        .with_size(egui::vec2(520.0, 760.0))
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            move |ui, e: &mut Env| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    e.st.show_inspector_panel(ui, &e.backend, &mut e.cv, &tokens);
                });
            },
            env,
        );
    h.run_steps(2);
    assert!(h.query_by_label_contains("Layers:").is_some());
    assert!(h.query_by_label_contains("Seed:").is_none(), "More starts collapsed");

    h.get_by_label("More").click();
    h.run_steps(2);
    assert!(h.query_by_label_contains("Seed:").is_some());
    assert!(h.query_by_label_contains("Rule:").is_some());

    h.get_by_label("Edge").click();
    h.run_steps(2);
    assert_eq!(h.state().st.inflation.seed_choice, E2SeedChoice::Edge(0));
    assert_eq!(h.state().st.inflation.last_shells.last().unwrap().topology.faces, 90);

    h.get_by_label("Periodic grid").click();
    h.run_steps(2);
    assert!(!h.state().st.inflation.enabled);
    assert!(h.query_by_label_contains("Patch repeat:").is_some());
    assert!(matches!(h.state().cv.scene(), Scene2D::PeriodicCell { .. }));
}
