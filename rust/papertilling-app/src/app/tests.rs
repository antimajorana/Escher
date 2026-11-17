use super::*;
use crate::combine_tab::{MotionMode, ToolMode};
use crate::live_combine::spec_world_pos;
use papertilling_core::defect::combine_types::{
    ClassifiedCharge, EquivalentSingleReport, NotebookConfigJson,
};
use papertilling_core::types::Isometry2;

#[test]
fn combine_pin_drag_snap_json_roundtrip_and_app_tabs() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    app.switch_tab(NavTab::Combine);
    assert_eq!(app.active_tab, NavTab::Combine);
    assert!(app.canvas2d.counts().0 > 0);

    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let oblique = backend.find_named("oblique").unwrap();
    let mut canvas = Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 2,
        show_unit_cell: false,
    });

    let mut st = CombineTabState::new(square, &backend);
    st.motion_mode = MotionMode::FreeSetup;
    st.clear_defects(square, &backend, &mut canvas);

    // Place +90 disclination near 1b (0.5, 0.5), then drag/snap to (1.0, 0.0)
    st.tool_mode = ToolMode::PlaceDisclination;
    st.selected_angle_deg = 90;
    st.place_at_world([0.48, 0.53], square, &backend, &mut canvas);
    assert_eq!(st.specs.len(), 1);
    assert!((spec_world_pos(&st.info, 0, &st.specs[0])[0] - 0.5).abs() < 1e-6);
    assert!(!st.move_pin_to(0, [0.49, 0.51], square, &backend, &mut canvas));
    assert!(st.move_pin_to(0, [1.03, -0.04], square, &backend, &mut canvas));
    assert!((spec_world_pos(&st.info, 0, &st.specs[0])[0] - 1.0).abs() < 1e-6);
    assert!(!st.move_pin_to(99, [0.0, 0.0], square, &backend, &mut canvas));

    // Filter to 2c (2-fold bond midpoint) auto-snaps angle to 180 and point to (0.5, 0)
    st.centre_filter_idx = 3;
    st.place_at_world([0.47, 0.02], square, &backend, &mut canvas);
    assert_eq!(st.specs.len(), 2);
    assert!((spec_world_pos(&st.info, 1, &st.specs[1])[0] - 0.5).abs() < 1e-6);

    // Dislocation placement and continuous drag
    st.tool_mode = ToolMode::PlaceDislocation;
    st.custom_mn = (1, 0);
    st.place_at_world([0.25, -0.35], square, &backend, &mut canvas);
    assert!(st.move_pin_to(2, [0.65, -0.15], square, &backend, &mut canvas));
    st.finish_drag(square);
    st.wait_idle(square, Some(&mut canvas));
    assert!(!st.is_busy() && !st.last_build.preview);

    // NotebookConfigJson round-trip handling both Err and Ok compare_result gracefully
    st.compare_result = Some(Err(String::from("pending")));
    let raw_err = st.to_notebook_json(square).to_json_pretty().unwrap();
    let cfg_err = NotebookConfigJson::from_json_str(&raw_err).unwrap();
    assert!(cfg_err.summary.is_none() && cfg_err.format == "papertilling.config.v1");

    st.compare_result = Some(Ok(EquivalentSingleReport {
        charge: ClassifiedCharge {
            kind: String::from("dislocation"), angle: 0.0, holonomy: Isometry2::IDENTITY,
            b: Some([1.0, 0.0]), lattice_vector: Some(true), b_rep: Some([1.0, 0.0]),
            length: Some(1.0), centre: None, centre_class: None, label: String::from("T1 (1,0)"),
            single: None, loop_sum: [1.0, 0.0],
        },
        kind: String::from("dislocation"), single: None, candidates: Vec::new(),
        matches: vec![String::from("V1")],
        volterra_matches: vec![String::from("V1")],
        completion_matches: Vec::new(),
        match_tier: String::from("volterra"),
        equivalent: true, verdict: String::from("matches V1"),
        scan: None, core_nodes: 4, core_edges: 4, config_core: None, gallery: Vec::new(),
    }));
    let raw_ok = st.to_notebook_json(square).to_json_pretty().unwrap();
    let cfg_ok = NotebookConfigJson::from_json_str(&raw_ok).unwrap();
    assert!(cfg_ok.summary.as_ref().is_some_and(|s| s.equivalent && s.matches == ["V1"]));

    let mut st2 = CombineTabState::new(oblique, &backend);
    assert!(st2.load_notebook_json(&cfg_ok, &backend, &mut canvas).is_ok());
    assert_eq!(st2.effective_item(oblique).group, "p4m");
    assert_eq!(st2.specs.len(), 3);

    let mut bad_cfg = cfg_ok;
    bad_cfg.recipe.group = String::from("invalid_group");
    assert!(st2.load_notebook_json(&bad_cfg, &backend, &mut canvas).is_err());
    assert!(NotebookConfigJson::from_json_str("{not valid json").is_err());
}

#[test]
fn tabs_reload_lazily_for_the_current_gallery_lattice() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    app.gallery.select_group_by_symbol("p4m", &app.backend, &mut app.canvas2d);
    let p4m = app.gallery.active_item().clone();
    assert_ne!(item_key(&p4m), app.combine_seen, "the start lattice is p6m kagome");
    app.switch_tab(NavTab::Combine);
    assert_eq!(app.combine_seen, item_key(&p4m));
    app.combine.wait_idle(&p4m, Some(&mut app.canvas2d));
    assert_eq!(app.combine.info.cell.basis.len(), p4m.sites_per_cell);
    assert!(app.combine.specs.iter().all(|s| matches!(s, papertilling_core::defect::surgery::DefectSpec::Disclination { angle_deg, .. } if (angle_deg.abs() - 90.0).abs() < 1e-9)),
        "p4m pair preset uses its 4-fold centres (±90°)");
    app.switch_tab(NavTab::Defects);
    assert_eq!(app.defects_seen, item_key(&p4m));
}

#[test]
fn tuned_gallery_geometry_reaches_surfaces_defects_and_combine() {
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let idx = app.backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("oblique")).unwrap();
    app.gallery.select_named(idx, &app.backend, &mut app.canvas2d);
    app.gallery.set_param("b", 0.8, &mut app.canvas2d).unwrap();
    app.gallery.set_param("gamma", 60.0, &mut app.canvas2d).unwrap();
    let item = app.gallery.active_item().clone();
    assert_eq!((item.recipe.b, item.recipe.gamma), (Some(0.8), Some(60.0)));

    app.switch_tab(NavTab::Combine);
    app.combine.wait_idle(&item, Some(&mut app.canvas2d));
    let a2 = app.combine.info.cell.a2;
    assert!((a2[0] - 0.4).abs() < 1e-10 && (a2[1] - 0.4 * 3.0_f64.sqrt()).abs() < 1e-10);

    app.switch_tab(NavTab::Defects);
    app.defects.wait_idle(&mut app.canvas2d);
    let Scene2D::CrystalPatch { crystal, .. } = app.canvas2d.scene() else { panic!("crystal"); };
    assert!(crystal.adj.iter().flatten().any(|e| (e.vec[0].hypot(e.vec[1]) - 0.8).abs() < 1e-9));

    app.surfaces.target = SurfaceTargetUi::Torus;
    app.surfaces.auto_size = false;
    (app.surfaces.params.h, app.surfaces.params.k) = (3, 4);
    app.switch_tab(NavTab::Surfaces);
    let Scene2D::SurfaceNet { mesh } = app.canvas2d.scene() else { panic!("surface net"); };
    assert_eq!(mesh.sites.len(), 12, "one square-network site per cell on a 3×4 torus");
    assert!(mesh.bonds.iter().any(|(_, _, v)| (v[0].hypot(v[1]) - 0.8).abs() < 1e-9));
}
