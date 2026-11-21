//! C1 WP-U2 tests for Atlas H² 0-based seeded layer growth, Kagome/wedge Cell-seed guard,
//! boundary-letter seeded counts, drawn-patch topological table, and kittest cap stop banner.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::growth::SeedKind;
use papertilling_core::hyperbolic::HyperbolicStyle;

use super::HyperbolicPanelState;
use crate::app::{NavTab, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::atlas_tab::hyperbolic_shells::shell_table_seeded;
use crate::theme::{StudioTheme, typography};

#[test]
fn layer_zero_seeds_on_73_match_hand_boundary_and_euler_counts() {
    // Hand counts at layer 0 on {7,3}:
    // - Cell: 1 heptagon -> 7 V, 7 E, ∂V = 7, ∂E = 7, χ = 1.
    // - Edge: 2 heptagons sharing 1 edge -> 12 V, 13 E (1 interior, 12 boundary), ∂V = 12, ∂E = 12, χ = 1.
    // - Vertex: 3 heptagons meeting at 1 central vertex -> 16 V (1 interior, 15 boundary), 18 E (3 spoke, 15 outer),
    //   ∂V = 15, ∂E = 15, χ = 1.
    let cases = [
        (SeedKind::Cell, 1, 7, 7),
        (SeedKind::Edge, 2, 12, 12),
        (SeedKind::Vertex, 3, 15, 15),
    ];
    for (seed, want_cells, want_bv, want_be) in cases {
        let mut st = HyperbolicPanelState { p: 7, q: 3, layers: 0, seed, ..Default::default() };
        st.build_scenes();
        let cache = st.cache.as_ref().expect("built");
        assert_eq!(cache.base.faces.len(), want_cells, "{seed:?}");
        assert_eq!(cache.drawn_stats.len(), 1, "{seed:?}");
        let s0 = &cache.drawn_stats[0];
        assert_eq!(
            (s0.cells, s0.new_cells, s0.boundary_vertices, s0.boundary_edges, s0.euler),
            (want_cells, want_cells, want_bv, want_be, 1),
            "{seed:?}"
        );
    }
}

#[test]
fn h2_54_seeded_drawn_stats_agree_with_exact_boundary_letter_tables() {
    // c1-plan.md §2.4: {5,4} cumulative cells:
    // - Cell (0..=4):   [1, 11, 51, 201, 761]
    // - Vertex (0..=3): [4, 24, 100, 384]
    // - Edge (0..=3):   [2, 16, 70, 272]
    let cases: [(SeedKind, usize, &[u64]); 3] = [
        (SeedKind::Cell, 4, &[1, 11, 51, 201, 761]),
        (SeedKind::Vertex, 3, &[4, 24, 100, 384]),
        (SeedKind::Edge, 3, &[2, 16, 70, 272]),
    ];
    for (seed, lay, want) in cases {
        let tab = shell_table_seeded(5, 4, seed).unwrap();
        assert!(tab.exact);
        let tab_cum: Vec<u64> = tab.rows.iter().take(want.len()).map(|r| r.cumulative).collect();
        assert_eq!(tab_cum.as_slice(), want, "{seed:?} exact table");

        let mut st = HyperbolicPanelState { p: 5, q: 4, layers: lay, seed, ..Default::default() };
        st.build_scenes();
        let cache = st.cache.as_ref().unwrap();
        assert_eq!(cache.base.complete_through_shell, lay, "{seed:?}");
        let drawn_cum: Vec<u64> = cache.drawn_stats.iter().map(|s| s.cells as u64).collect();
        assert_eq!(drawn_cum.as_slice(), want, "{seed:?} drawn patch");
        assert!(cache.drawn_stats.iter().all(|s| s.euler == 1), "{seed:?} χ = 1");
    }
}

#[test]
fn kagome_and_nonzero_wedge_force_cell_seed_without_error() {
    // Even if `st.seed` was set to `Vertex` or `Edge`, switching to Kagome or Δp != 0 forces `Cell` cleanly.
    let mut kag = HyperbolicPanelState {
        p: 7,
        q: 3,
        style: HyperbolicStyle::RectifiedKagome,
        seed: SeedKind::Vertex,
        layers: 2,
        ..Default::default()
    };
    assert_eq!(kag.effective_seed(), SeedKind::Cell);
    kag.build_scenes();
    assert!(kag.cache.is_some(), "{}", kag.status);

    let mut wdg = HyperbolicPanelState {
        p: 7,
        q: 3,
        pristine_only: false,
        wedge_delta: -1,
        seed: SeedKind::Edge,
        layers: 2,
        ..Default::default()
    };
    assert_eq!(wdg.effective_seed(), SeedKind::Cell);
    wdg.build_scenes();
    assert!(wdg.cache.is_some(), "{}", wdg.status);
}

#[test]
fn kittest_atlas_h2_seed_toggle_and_slider_at_cap_shows_stop_banner() {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    app.atlas.set_pristine_only(true);
    app.atlas.sub_tab = AtlasSubTab::HyperbolicH2;
    app.atlas.dirty = true;
    app.switch_tab(NavTab::Atlas);

    let mut h = Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, app: &mut PapertillingApp| {
                let _ = app.frame_ui(ui);
            },
            app,
        );
    h.run_steps(4);

    // Click Vertex seed on {7,3} at layers = 2:
    // Hand (q = 3 spoke count): S_0 = 3 (∂E = 15, 3 concave + 12 convex); S_1 adds 12 heptagons sharing 12 spokes
    // -> |S_1| = 15 (∂E = 84 - 15 - 24 = 45, 12 concave + 33 convex); S_2 adds 45 - 12 = 33 -> |S_2| = 48.
    h.get_by_label("Vertex").click();
    h.run_steps(3);
    assert_eq!(h.state().atlas.hyperbolic.seed, SeedKind::Vertex);
    assert_eq!(h.state().atlas.hyperbolic.cache.as_ref().unwrap().base.faces.len(), 48);

    // Drive Layers to slider cap (8): {7,3} Vertex seed exceeds 1000 faces before layer 8, showing the stop banner.
    {
        let st = h.state_mut();
        st.atlas.hyperbolic.layers = 8;
        st.atlas.dirty = true;
    }
    h.run_steps(4);
    let complete = h.state().atlas.hyperbolic.cache.as_ref().unwrap().base.complete_through_shell;
    assert!(complete < 8);
    assert!(h.query_all_by_label_contains(&format!("Stopped at layer {complete} (requested 8)")).next().is_some());
    assert!(h.query_all_by_label_contains("Drawn patch layers").next().is_some());
}
