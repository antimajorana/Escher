use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::AtlasSubTab;
use crate::app::{DefectsView, LatticesView, PapertillingApp};
use crate::game_menu::CardId;
use crate::theme::{StudioTheme, typography};

fn atlas_harness(sub: AtlasSubTab) -> Harness<'static, PapertillingApp> {
    let mut app = PapertillingApp::new(false, "kittest".into(), "kittest".into());
    app.set_advanced(true);
    app.deck_equip(CardId::AtlasDefects(sub), &egui::Context::default());
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1360.0, 840.0))
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
    harness.run_steps(4);
    harness
}

#[test]
fn crystal_atlas_selects_matching_nets_and_shows_real_rejoined_screw_bonds() {
    let mut h = atlas_harness(AtlasSubTab::SpaceGroups3D);
    h.get_by_label("Pc glide").click();
    h.run_steps(3);
    {
        let panel = &mut h.state_mut().atlas.space3d;
        assert_eq!(panel.selected_sg, 7);
        assert_eq!(panel.ensure_net().space_group_num, 7);
        assert!(panel.ensure_net().defect_axis.is_none());
    }
    assert!(h.query_by_label("Translation · shift (0, 0, 1)").is_some());
    h.get_by_label("+1 a₃").click();
    h.run_steps(3);
    let net = h.state_mut().atlas.space3d.ensure_net();
    assert!(net.defect_axis.is_some());
    assert!(net.bonds.iter().any(|&(_, _, kind)| kind == 1));

    h.get_by_label("Layer-group charges").click();
    h.run_steps(3);
    h.get_by_label("p11a glide").click();
    h.run_steps(3);
    assert_eq!(h.state().atlas.layers.selected_num, 5);
    assert!(h.query_by_label("Exact based match").is_some());
}

#[test]
fn hyperbolic_atlas_switches_between_word_comparison_and_h3_reference() {
    let mut h = atlas_harness(AtlasSubTab::HyperbolicH2);
    assert!(h.query_by_label("Exact based match").is_some());
    h.get_by_label("Glide probe").click();
    h.run_steps(3);
    assert!(h.query_by_label("Probe · Glide reflection").is_some());

    h.get_by_label("H³ Charges").click();
    h.run_steps(3);
    assert_eq!(h.state().atlas.sub_tab, AtlasSubTab::HyperbolicH3);
    assert!(h.state().view3d.mesh_data().vertex_count >= 20);
    assert!(h.query_by_label("Klein ball").is_some());

    let app = h.state_mut();
    app.atlas.last_hyperbolic_subtab = AtlasSubTab::HyperbolicH3;
    app.select_lattices_view(LatticesView::NonEuclidean);
    assert!(app.atlas.pristine_only);
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::HyperbolicH3);
    app.select_defects_view(DefectsView::AtlasDefects);
    assert!(!app.atlas.pristine_only);
    assert_eq!(app.atlas.sub_tab, AtlasSubTab::HyperbolicH3);
}
