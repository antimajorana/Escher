#![allow(clippy::field_reassign_with_default)]
//! Layer-growth view tests (C1 WP-U4). Expected values are hand-derived (stated per test); the core's own
//! default drawing is only used as the "unchanged default" reference.

use std::collections::HashSet;

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::space3d::layer_slab::{build_layer, default_slab_patch};

use super::*;
use crate::app::{LatticesView, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::scene2d::Scene2D;
use crate::theme::{StudioTheme, typography};

#[test]
fn defaults_draw_the_default_slab() {
    let mut st = LayerGrowthState::default();
    for num in [1_u8, 61, 80] {
        let want = default_slab_patch(num).unwrap();
        let got = st.patch(num);
        assert_eq!(got.positions, want.positions, "#{num}");
        assert_eq!(got.bonds, want.bonds, "#{num}");
        assert!(st.refusal.is_none());
    }
}

#[test]
fn aa_stacks_multiply_the_sites_and_project_onto_the_first_copy() {
    // m AA copies of the same grown layer: m times its sites (hand), each copy straight above the first
    let mut st = LayerGrowthState { layers: 1, ..LayerGrowthState::default() };
    st.stack.spacing = 2.0;
    let one = st.patch(61);
    for m in 2..=4_u32 {
        st.stack.layers = m;
        let p = st.patch(61);
        assert!(st.refusal.is_none(), "{:?}", st.refusal);
        let n = one.positions.len();
        assert_eq!(p.positions.len(), m as usize * n);
        for k in 1..m as usize {
            for i in 0..n {
                let (a, b) = (p.positions[i], p.positions[k * n + i]);
                assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
                assert!((b[2] - a[2] - 2.0 * k as f64).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn an_offset_that_breaks_the_group_is_refused_with_one_layer_drawn() {
    // p11m (#4), s = (1/2, 0): with 2 copies the z-mirror between them must shift by 1/2, a glide, so the mirror
    // is lost (refused); with 3 copies the middle copy carries the mirror and the outer ones sit at 0 and 1 = 0
    // (kept). Hand derivation, as in the core's offset rule.
    let mut st = LayerGrowthState { layers: 1, oblique: true, ..LayerGrowthState::default() };
    st.stack.spacing = 2.0;
    let one = st.patch(4).positions.len();
    st.stack.layers = 2;
    st.stack.offset = [0.5, 0.0];
    let p = st.patch(4);
    assert!(st.refusal.as_deref().is_some_and(|r| r.contains("breaks")), "{:?}", st.refusal);
    assert!(p.info.starts_with("Refused:") && p.info.contains("(showing 1 layer)"), "{}", p.info);
    assert_eq!(p.positions.len(), one, "one layer is drawn meanwhile");
    st.stack.layers = 3;
    let p3 = st.patch(4);
    assert_eq!(p3.positions.len(), 3 * one);
    assert!(!p3.info.starts_with("Refused:"));
    assert!(st.refusal.is_none());
    // the same offset with the option off is plain AA
    st.oblique = false;
    st.stack.layers = 2;
    assert_eq!(st.patch(4).positions.len(), 2 * one);
    assert!(st.refusal.is_none());
}

#[test]
fn a_spacing_inside_the_slab_is_refused() {
    // copies closer than the slab thickness interpenetrate (hand); p1's three general sites are not coplanar
    let t = build_layer(1).unwrap().thickness;
    assert!(t > 0.0);
    let mut st = LayerGrowthState { layers: 1, ..LayerGrowthState::default() };
    st.stack.layers = 2;
    st.stack.spacing = t / 2.0;
    let p = st.patch(1);
    assert!(st.refusal.as_deref().is_some_and(|r| r.contains("interpenetrate")), "{:?}", st.refusal);
    assert!(p.info.starts_with("Refused:") && p.info.contains("(showing 1 layer)"), "{}", p.info);
}

#[test]
fn projection_scene_preserves_orbit_palette_and_interlayer_bond_colour() {
    // p1 (#1) has 3 general-position orbits (labels 0, 1, 2); 2 AA copies add interlayer bonds (kind 1).
    let mut lp = LayersPanelState::default();
    lp.selected_num = 1;
    lp.growth.layers = 1;
    lp.growth.stack.layers = 2;
    lp.growth.stack.spacing = 2.0;
    let (sc, _, _) = lp.build_scenes();
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    let fills: HashSet<[u8; 4]> = primitives.circles.iter().map(|c| c.fill).collect();
    assert_eq!(fills.len(), 3, "p1's 3 site orbits keep distinct 2D colours");
    let line_cols: HashSet<[u8; 4]> = primitives.lines.iter().map(|l| l.color).collect();
    assert_eq!(line_cols.len(), 2, "in-layer and interlayer bond colours are distinct in 2D");
}

#[test]
fn kittest_layers80_face_seed_and_refused_oblique_offset() {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    app.atlas.sub_tab = AtlasSubTab::Layers80;
    app.atlas.layers.selected_num = 4; // #4 p11m: horizontal mirror, so s = (1/2, 0) at m = 2 is [PROVED] refused
    app.atlas.dirty = true;
    app.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
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

    assert_eq!(h.state().atlas.sub_tab, AtlasSubTab::Layers80);
    assert!(h.query_all_by_label_contains("Grow and stack").next().is_some());
    assert!(h.query_all_by_label_contains("boundary").next().is_some(), "per-layer boundary counts shown");

    // Open "More" and switch seed to SeedKind::Face (2 cells on a Voronoi face, hand-derived).
    h.get_by_label("More").click();
    h.run_steps(2);
    h.get_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|n| n.value().as_deref() == Some("Vertex (cells at a Voronoi vertex)"))
        .expect("layer seed ComboBox")
        .click();
    h.run_steps(2);
    h.get_by_label("Face (2 cells on a Voronoi face)").click();
    h.run_steps(3);
    assert_eq!(h.state().atlas.layers.growth.seed, SeedKind::Face);
    assert!(h.query_all_by_label_contains("S0: 2 cells (+2)").next().is_some());
    assert!(h.state().view3d.caption().contains("Face seed"));

    // Enable oblique offset and set s = (0.5, 0.0) with 2 stacked copies on #4 (p11m): refused in panel & patch.info.
    h.get_by_label("Oblique offset per layer").click();
    h.run_steps(2);
    assert!(h.state().atlas.layers.growth.oblique);
    {
        let atlas = &mut h.state_mut().atlas;
        atlas.layers.growth.stack.layers = 2;
        atlas.layers.growth.stack.spacing = 2.0;
        atlas.layers.growth.stack.offset = [0.5, 0.0];
        atlas.dirty = true;
    }
    h.run_steps(3);

    assert!(h.query_all_by_label_contains("Refused:").next().is_some(), "refusal surfaced in panel");
    assert!(h.query_all_by_label_contains("One layer is drawn").next().is_some());
    let cap = h.state().view3d.caption().to_owned();
    assert!(cap.contains("Refused:") && cap.contains("(showing 1 layer)"), "3D caption: {cap}");
    let patch = h.state_mut().atlas.layers.growth.patch(4);
    assert!(patch.info.starts_with("Refused:") && patch.info.contains("(showing 1 layer)"), "{}", patch.info);
}
