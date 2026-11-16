use super::*;
use crate::backend::CatalogueRuleUi;
use papertilling_core::types::Cell2D;

fn canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 2,
        show_unit_cell: false,
    })
}

#[test]
fn combine_tab_state_init_and_presets() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let p1_first = &backend.catalogue_for_group("p1", CatalogueRuleUi::Points)[0];
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    assert_eq!(st.specs.len(), 2);
    st.wait_idle(square, Some(&mut canvas));
    assert!(!st.last_build.crystal.pos.is_empty());
    st.ensure_compared(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.compare_result.is_some());
    if let Some(Ok(rep)) = &st.compare_result { assert!(rep.equivalent); }
    assert_eq!(CombineTabState::new(p1_first, &backend).specs.len(), 1);
    st.apply_preset_dislocation(square, &backend, &mut canvas);
    assert_eq!(st.specs.len(), 1);
    st.clear_defects(square, &backend, &mut canvas);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.specs.is_empty() && st.compare_result.is_none());
}

#[test]
fn json_import_keeps_python_patch_radius_through_a_ui_frame() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    let mut cfg = st.to_notebook_json(square);
    (cfg.options.r_patch, cfg.options.grid, cfg.options.radius) = (Some(17.25), 5, 2);
    let msg = st.load_notebook_json(&cfg, &backend, &mut canvas).unwrap();
    assert!(msg.contains("grid 5 -> 3") && !msg.contains("r_patch"), "{msg}");
    let ctx = egui::Context::default();
    for _ in 0..2 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| st.show_side_panel(ui, square, &backend, &mut canvas));
    }
    assert_eq!(st.opts.manual_radius, 17.25);
    assert_eq!(st.to_notebook_json(square).options.r_patch, Some(17.25));
}

#[test]
fn refused_surgery_is_not_reported_as_equivalent() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    // Four +90 disclinations: total Frank angle +360, an open patch cannot close (refused).
    st.specs = [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]].map(|p| DefectSpec::disclination(90.0, p)).to_vec();
    st.request_build(square, false, true, false);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_some(), "surgery must refuse a +360 total");
    match &st.compare_result {
        Some(Err(msg)) => assert!(msg.contains("refused"), "{msg}"),
        other => panic!("expected an error, got {:?}", other.as_ref().map(|r| r.as_ref().map(|x| &x.verdict))),
    }
    assert!(st.to_notebook_json(square).summary.is_none());
}

#[test]
fn drag_previews_then_compares_the_released_configuration_once() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.run_compare(square);
    st.wait_idle(square, Some(&mut canvas));
    let before = st.cache_stats;
    // +90 at 1a, -90 at 1b on the square lattice: a single dislocation.
    st.specs = vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
    for k in 1..=6 {
        assert!(st.move_pin_to(1, [0.5 + k as f64, 0.5], square, &backend, &mut canvas));
        assert!(!st.is_comparing(), "no comparison while dragging");
    }
    assert!(st.preview_dirty, "mouse moves coalesce while a preview is in flight");
    let first = st.builder.wait(Duration::from_secs(20)).expect("in-flight preview was not discarded").unwrap();
    assert!(first.0.preview);
    st.take_build(Ok(first));
    st.poll(None, square, Some(&mut canvas));
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.preview && st.cache_stats == before);
    st.finish_drag(square);
    st.wait_idle(square, Some(&mut canvas));
    assert!(!st.last_build.preview && !st.compare_stale);
    let rep = st.compare_result.as_ref().unwrap().as_ref().unwrap();
    // A ±90 dipole has |b| = √2 |d|: 1 before the drag, √2 |(6.5, 0.5)| ≈ 9.2 after.
    assert_eq!(rep.kind, "dislocation");
    assert!(rep.charge.length.unwrap() > 5.0, "compared the released configuration");
    assert!(st.to_notebook_json(square).summary.is_some());
}

#[test]
fn lattice_reload_drops_obsolete_builds_and_keeps_the_ui_responsive() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let p1 = backend.find_named("straight_p1").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    let t = std::time::Instant::now();
    st.reload_for_item(p1, &backend, &mut canvas, true);
    println!("# time submit lattice switch: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);
    assert!(st.is_initializing());
    assert!(st.specs.is_empty() && canvas.counts().0 == 0, "no stale lattice under the new label");
    st.reload_for_item(square, &backend, &mut canvas, true);
    st.wait_idle(square, Some(&mut canvas));
    assert!(!st.is_initializing());
    assert_eq!(st.specs.len(), 2, "the newest lattice is the square: its ±90 pair");
    assert!(canvas.counts().0 > 0);
    assert!(st.specs.iter().all(|s| matches!(s, DefectSpec::Disclination { angle_deg, .. } if angle_deg.abs() == 90.0)));
}

#[test]
fn extreme_pin_separation_is_a_view_budget_refusal_not_a_false_charge() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.specs = vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [1e8 + 0.5, 0.5])];
    st.request_build(square, false, true, false);
    st.wait_idle(square, Some(&mut canvas));
    assert_eq!(st.last_build.charge_label, "Live view too large");
    assert!(st.last_build.refused.as_ref().unwrap().contains("budget"));
    assert!(st.to_notebook_json(square).summary.is_none());
    assert!(matches!(st.compare_result, Some(Err(_))));
}

#[test]
fn recipe_controls_the_imported_group_and_unknown_formats_do_not_mutate_state() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut st = CombineTabState::new(square, &backend);
    let mut canvas = canvas();
    let mut cfg = st.to_notebook_json(square);
    cfg.group = String::from("p1"); // metadata ignored by Python; the recipe is p4m
    let imported = backend.item_from_notebook_json(&cfg).unwrap();
    assert_eq!(imported.group, "p4m");
    cfg.format = String::from("papertilling.config.v999");
    let before = st.specs.clone();
    assert!(st.load_notebook_json(&cfg, &backend, &mut canvas).is_err());
    assert_eq!(st.specs, before);
    cfg.format = String::from("papertilling.config.v1");
    cfg.frame.q = [[0.0; 2]; 2];
    assert!(st.load_notebook_json(&cfg, &backend, &mut canvas).is_err());
    assert_eq!(st.specs, before, "a degenerate frame cannot partially replace the current state");
}

#[test]
fn failed_build_cannot_compare_the_previous_crystal_as_the_current_configuration() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut state = CombineTabState::new(square, &backend);
    state.wait_idle(square, None);
    state.take_build(Err(String::from("test worker failure")));
    state.run_compare(square);
    assert!(matches!(state.compare_result, Some(Err(_))));
    assert!(state.last_build.refused.is_some());
    assert!(state.to_notebook_json(square).summary.is_none());
}

#[test]
fn dislocation_star_members_include_full_orbit() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let sq_info = backend.combine_lattice_info(square);
    let members = &sq_info.dislocations[0].members;
    assert_eq!(members.len(), 4);
    for d in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        assert!(members.contains(&d), "missing {d:?} from {members:?}");
    }
    let hc = backend.find_named("honeycomb").unwrap();
    let hc_info = backend.combine_lattice_info(hc);
    assert_eq!(hc_info.dislocations[0].members.len(), 6);
}

#[test]
fn branch_cut_overlay_populates_and_clears_on_new_scene() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    assert!(!st.show_branch_cuts);
    assert!(canvas.cut_rays().is_empty());

    st.show_branch_cuts = true;
    st.sync_to_canvas(square, &mut canvas, false);
    assert_eq!(canvas.cut_rays().len(), 2);

    canvas.set_scene(
        Scene2D::PeriodicCell {
            cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
            repeat: 2,
            show_unit_cell: false,
        },
        true,
    );
    assert!(canvas.cut_rays().is_empty());
}

#[test]
fn pin_cut_angle_override_and_coincident_superposition() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));

    crate::live_combine::cuts::set_pin_cut_angle(&mut st.specs[0], Some(90.0));
    match st.specs[0] {
        DefectSpec::Disclination { cut: Some(c), .. } => {
            assert!(c[0].abs() < 1e-12 && (c[1] - 1.0).abs() < 1e-12, "{c:?}");
        }
        ref other => panic!("unexpected spec: {other:?}"),
    }
    crate::live_combine::cuts::set_pin_cut_angle(&mut st.specs[0], None);
    assert!(matches!(st.specs[0], DefectSpec::Disclination { cut: None, .. }));

    st.specs = vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.0, 0.0])];
    st.request_build(square, false, true, false);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_none());
    assert_eq!(st.last_build.coalesced_count, 2);
    assert_eq!(st.last_build.charge_label, "I (no defect)");
    assert!(st.last_sheet3d.sheet.xyz.iter().all(|p| p[2].abs() < 1e-6));

    crate::live_combine::cuts::merge_coincident_pins(&st.info, &mut st.specs);
    assert!(st.specs.is_empty());
}
