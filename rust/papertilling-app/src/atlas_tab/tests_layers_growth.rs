//! Layer-growth view tests (C1 WP-U4). Expected values are hand-derived (stated per test); the core's own
//! default drawing is only used as the "unchanged default" reference.

use papertilling_core::space3d::layer_slab::{build_layer, default_slab_patch};

use super::*;

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
    assert_eq!(p.positions.len(), one, "one layer is drawn meanwhile");
    st.stack.layers = 3;
    assert_eq!(st.patch(4).positions.len(), 3 * one);
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
    st.patch(1);
    assert!(st.refusal.as_deref().is_some_and(|r| r.contains("interpenetrate")), "{:?}", st.refusal);
}
