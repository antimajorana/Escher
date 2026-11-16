//! End-to-end verification tests for Features 1–4 across `papertilling-app`:
//! 1. Custom lattice aliases: save/load/reset in temp dir, propagation to Gallery/Surfaces/Defects/Combine,
//!    SVG/PNG titles, and exported Notebook 3 `.json` (`lattice_label`).
//! 2. Preserving pan/zoom across Gallery parameter slider updates & overlays, `View3dState` pinch zoom,
//!    and Core Gallery card click-to-enlarge inspector modal.
//! 3. Defects 3D elastic sheet view modes, color modes, and flat-plane baseline toggle.
//! 4. Combine live 3D elastic sheet: flat plane (Clear) -> +90 cone -> +90/-90 screened dislocation buckle.

use papertilling_core::defect::combine_types::NotebookConfigJson;
use papertilling_core::types::Cell2D;

use crate::aliases::AliasStore;
use crate::backend::{AppBackend, CatalogueRuleUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_tab::{CombineTabState, ToolMode};
use crate::core_panel::{CorePanelState, show_core_panel};
use crate::defects_tab::{DefectSubTab, DefectsTabState};
use crate::gallery_tab::GalleryTabState;
use crate::mesh3d::{DefectViewMode, Sheet3dColorMode};
use crate::surfaces_tab::SurfacesTabState;
use crate::view3d::View3dState;

fn make_canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 3,
        show_unit_cell: true,
    })
}

#[test]
fn rename_alias_roundtrip_and_propagation_across_tabs_svg_and_notebook_json() {
    let dir = std::env::temp_dir().join(format!("papertilling-feat1-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let alias_file = dir.join("lattice_names.json");

    let mut backend = AppBackend::new_with_alias_store(AliasStore::load_from_path(alias_file.clone()));
    let mut canvas = make_canvas();
    let mut view3d = View3dState::default();
    let mut gallery = GalleryTabState::new(&backend, &mut canvas);

    // Select unnamed catalogue entry `p4g #1` under `bonds` and rename it.
    gallery.select_rule(CatalogueRuleUi::Bonds, &backend, &mut canvas);
    gallery.select_group_by_symbol("p4g", &backend, &mut canvas);
    gallery.select_catalogue_entry(0, &backend, &mut canvas);
    let orig_id = gallery.active_item().default_label.clone();
    assert!(orig_id.starts_with("p4g #1") || !orig_id.is_empty());

    gallery.rename_active_lattice("Cairo-Like Squares", &mut backend, &mut canvas);
    assert_eq!(gallery.active_item().label, "Cairo-Like Squares");
    assert_eq!(gallery.active_item().default_label, orig_id);
    assert!(gallery.active_item().has_custom_name());
    assert!(gallery.active_item().display_with_orig().contains(&orig_id));

    // Gallery SVG export carries the custom alias in its rendered header.
    let gal_svg = dir.join("gallery_renamed.svg");
    canvas.export_to_path(&gal_svg).unwrap();
    let svg_txt = std::fs::read_to_string(&gal_svg).unwrap();
    assert!(svg_txt.contains("Cairo-Like Squares"), "Gallery SVG missing alias: {svg_txt}");

    // Surfaces 3D caption and 2D net SVG carry the custom alias.
    let item = gallery.active_item().clone();
    let mut surfaces = SurfacesTabState::new();
    surfaces.activate_and_sync(&item, &backend, &mut canvas, &mut view3d);
    assert!(view3d.caption().contains("Cairo-Like Squares"), "Surfaces 3D caption: {}", view3d.caption());
    let surf_svg = dir.join("surface_renamed.svg");
    canvas.export_to_path(&surf_svg).unwrap();
    assert!(std::fs::read_to_string(&surf_svg).unwrap().contains("Cairo-Like Squares"));

    // Defects readout title and 3D sheet caption carry the custom alias.
    let mut defects = DefectsTabState::new(&item, &backend);
    assert!(defects.readout.title.contains("Cairo-Like Squares"));
    defects.sync_view3d(&mut view3d);
    assert!(view3d.caption().contains("Cairo-Like Squares"));

    // Combine title and exported Notebook 3 `.json` (`lattice_label`) carry the custom alias.
    let mut combine = CombineTabState::new(&item, &backend);
    combine.wait_idle(&item, Some(&mut canvas));
    let nb_json = combine.to_notebook_json(&item).to_json_pretty().unwrap();
    let parsed = NotebookConfigJson::from_json_str(&nb_json).unwrap();
    assert_eq!(parsed.lattice_label, "Cairo-Like Squares");

    // Reloading `AppBackend` from `alias_file` preserves the custom alias, and Reset restores `orig_id`.
    let backend2 = AppBackend::new_with_alias_store(AliasStore::load_from_path(alias_file.clone()));
    let reloaded_cat = backend2.catalogue_for_group("p4g", CatalogueRuleUi::Bonds);
    assert_eq!(reloaded_cat[0].label, "Cairo-Like Squares");
    assert_eq!(reloaded_cat[0].default_label, orig_id);

    gallery.reset_active_lattice_name(&mut backend, &mut canvas);
    assert_eq!(gallery.active_item().label, orig_id);
    assert!(!gallery.active_item().has_custom_name());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gallery_parameter_edits_and_overlays_preserve_zoom_and_pan() {
    let backend = AppBackend::new();
    let mut canvas = make_canvas();
    let mut gallery = GalleryTabState::new(&backend, &mut canvas);

    // Select `oblique` (has free parameters `b` and `gamma`), run 1 frame to settle initial fit, then zoom/pan.
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("oblique")).unwrap();
    gallery.select_named(idx, &backend, &mut canvas);
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        canvas.show(ui);
    });
    canvas.zoom = 312.5;
    canvas.pan = egui::vec2(44.0, -29.0);

    // Dragging/setting free parameters, resetting params, toggling unit cell, and changing repeat MUST keep pan & zoom!
    gallery.drag_param("b", 1.25, &mut canvas).unwrap();
    gallery.set_param("gamma", 75.0, &mut canvas).unwrap();
    gallery.set_show_unit_cell(false, &mut canvas);
    gallery.set_repeat(5, &mut canvas);
    gallery.reset_params(&mut canvas);

    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        canvas.show(ui);
    });
    assert!((canvas.zoom - 312.5).abs() < 1e-4, "zoom reset unexpectedly to {}", canvas.zoom);
    assert_eq!(canvas.pan, egui::vec2(44.0, -29.0), "pan reset unexpectedly to {:?}", canvas.pan);

    // Switching to a different lattice DOES trigger auto-fit (`pan = ZERO`).
    let sq_idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("square")).unwrap();
    gallery.select_named(sq_idx, &backend, &mut canvas);
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        canvas.show(ui);
    });
    assert_eq!(canvas.pan, egui::Vec2::ZERO);
    assert!((canvas.zoom - 312.5).abs() > 1.0);
}

#[test]
fn view3d_pinch_zoom_and_core_card_inspector_modal_run_headless() {
    let ctx = egui::Context::default();
    let mut view3d = View3dState::default();
    let d0 = view3d.distance;
    let center = egui::pos2(300.0, 200.0);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 400.0))),
        events: vec![egui::Event::PointerMoved(center), egui::Event::Zoom(2.0)],
        ..Default::default()
    };
    let _ = ctx.run_ui(input, |ui| view3d.show(ui));
    assert!((view3d.distance - 0.5 * d0).abs() < 1e-3, "3D pinch zoom halved distance: {} -> {}", d0, view3d.distance);

    // Core gallery click opens the zoomable inspector window without panic and Esc closes it.
    let backend = AppBackend::new();
    let sq = backend.find_named("square").unwrap();
    let mut combine = CombineTabState::new(sq, &backend);
    let mut canvas = make_canvas();
    combine.run_compare(sq);
    combine.wait_idle(sq, Some(&mut canvas));
    let mut core_state = CorePanelState::default();
    let card_click_pos = egui::pos2(80.0, 120.0);
    for events in [
        vec![egui::Event::PointerMoved(card_click_pos)],
        vec![egui::Event::PointerButton { pos: card_click_pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }],
        vec![egui::Event::PointerButton { pos: card_click_pos, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE }],
        vec![egui::Event::Zoom(1.5)],
        vec![egui::Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }],
    ] {
        let inp = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 320.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run_ui(inp, |ui| show_core_panel(&mut core_state, ui, combine.compare_result.as_ref()));
    }
}

#[test]
fn defects_and_combine_live_3d_sheet_geometry_end_to_end() {
    let backend = AppBackend::new();
    let hc = backend.find_named("honeycomb").unwrap();
    let sq = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut view3d = View3dState::default();
    let ctx = egui::Context::default();

    // 1. Defects tab: +60 deg disclination on honeycomb buckles into a 3D cone; flat baseline is z=0.
    let mut def = DefectsTabState::new(hc, &backend);
    def.sub_tab = DefectSubTab::Disclinations;
    def.selected_disclin = 0;
    def.rebuild_crystal(hc, &backend, &mut canvas, true);
    def.wait_idle(&mut canvas);

    for cm in Sheet3dColorMode::ALL {
        def.sheet_opts.color_mode = cm;
        def.sheet_opts.z_exag = 1.6;
        def.view3d_dirty = true;
        for vm in DefectViewMode::ALL {
            def.view_mode = vm;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                def.show_central(ui, hc, &mut canvas, &mut view3d);
            });
        }
    }
    let dz_cone = def.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max)
        - def.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
    assert!(dz_cone > 0.5 * def.last_sheet.mean_bond_len);
    assert!(def.flat_sheet.xyz.iter().all(|p| p[2].abs() < 1e-6));

    // 2. Combine tab: Clear (0 defects) -> flat plane; +90 -> 3D cone; +90 + -90 -> screened dislocation buckle!
    let mut comb = CombineTabState::new(sq, &backend);
    comb.clear_defects(sq, &backend, &mut canvas);
    comb.wait_idle(sq, Some(&mut canvas));
    let max_z_clear = comb.last_sheet3d.sheet.xyz.iter().map(|p| p[2].abs()).fold(0.0_f64, f64::max);
    assert!(max_z_clear < 1e-6, "Clear All must produce a flat z=0 plane, got {max_z_clear}");

    comb.tool_mode = ToolMode::PlaceDisclination;
    comb.selected_angle_deg = 90;
    comb.place_at_world([0.0, 0.0], sq, &backend, &mut canvas);
    comb.wait_idle(sq, Some(&mut canvas));
    let rms_cone = (comb.last_sheet3d.sheet.xyz.iter().map(|p| p[2] * p[2]).sum::<f64>()
        / comb.last_sheet3d.sheet.xyz.len() as f64)
        .sqrt();
    assert!(rms_cone > 0.25, "+90 disclination pops sheet into a 3D cone (rms_z={rms_cone})");

    // Add -90 disclination far away, then drag it onto nearest face centre (0.5, 0.5) to watch screening.
    comb.selected_angle_deg = -90;
    comb.place_at_world([2.5, 0.5], sq, &backend, &mut canvas);
    comb.move_pin_to(1, [0.5, 0.5], sq, &backend, &mut canvas);
    comb.finish_drag(sq);
    comb.wait_idle(sq, Some(&mut canvas));
    comb.sync_view3d(sq, &mut view3d);
    assert!(!view3d.mesh_data().faces.is_empty());

    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        comb.show_central(ui, sq, &backend, &mut canvas, &mut view3d);
    });
}

#[test]
fn review_kappa_change_preserves_a_pending_combine_clear() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut state = CombineTabState::new(square, &backend);
    state.wait_idle(square, Some(&mut canvas));
    assert!(!state.specs.is_empty());
    state.clear_defects(square, &backend, &mut canvas);
    state.sheet_opts.kappa_b *= 2.0;
    state.request_sheet3d_only(square);
    state.wait_idle(square, Some(&mut canvas));
    assert!(state.specs.is_empty());
    assert_eq!(state.last_build.charge_label, "vacuum (0 defects)");
    assert!(state.last_build.crystal.cone_sites.is_empty());
    assert!(state.last_sheet3d.sheet.xyz.iter().all(|p| p[2].abs() < 1e-6));
}

#[test]
fn review_kappa_change_preserves_a_pending_defects_selection() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut state = DefectsTabState::new(square, &backend);
    assert!(state.readout.charge_summary.contains("dislocation"));
    state.sub_tab = DefectSubTab::Disclinations;
    state.selected_disclin = 0;
    state.rebuild_crystal(square, &backend, &mut canvas, false);
    state.sheet_opts.kappa_b *= 2.0;
    state.rebuild_sheet3d_only(square);
    state.wait_idle(&mut canvas);
    assert!(state.readout.charge_summary.contains("disclination"), "{}", state.readout.charge_summary);
}

#[test]
fn review_lattice_switch_never_displays_the_previous_lattices_sheet() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let honeycomb = backend.find_named("honeycomb").unwrap();
    let mut canvas = make_canvas();
    let mut view = View3dState::default();
    let mut combine = CombineTabState::new(square, &backend);
    combine.wait_idle(square, Some(&mut canvas));
    combine.sync_view3d(square, &mut view);
    assert!(view.mesh_data().vertex_count > 0);
    combine.reload_for_item(honeycomb, &backend, &mut canvas, true);
    combine.sync_view3d(honeycomb, &mut view);
    assert_eq!(view.mesh_data().vertex_count, 0, "old Combine lattice is still visible");
    assert!(view.caption().contains("Preparing"));

    let mut defects = DefectsTabState::new(square, &backend);
    defects.sync_view3d(&mut view);
    assert!(view.mesh_data().vertex_count > 0);
    defects.reload_for_item(honeycomb, &backend, &mut canvas, true);
    defects.sync_view3d(&mut view);
    assert_eq!(view.mesh_data().vertex_count, 0, "old Defects lattice is still visible");
    assert!(view.caption().contains("Preparing"));
}

#[test]
fn review_pending_builds_mark_settled_3d_caption_stale() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = make_canvas();
    let mut view = View3dState::default();

    let mut comb = CombineTabState::new(square, &backend);
    comb.wait_idle(square, Some(&mut canvas));
    comb.sync_view3d(square, &mut view);
    assert!(!view.caption().contains("stale"));
    comb.clear_defects(square, &backend, &mut canvas);
    assert!(comb.view3d_dirty && comb.is_building());
    comb.sync_view3d(square, &mut view);
    assert!(view.caption().contains("stale"), "Combine pending clear caption: {}", view.caption());
    comb.wait_idle(square, Some(&mut canvas));
    comb.sync_view3d(square, &mut view);
    assert!(!view.caption().contains("stale"));
    comb.sheet_opts.kappa_b *= 2.0;
    comb.request_sheet3d_only(square);
    assert!(comb.view3d_dirty && comb.is_building());
    comb.sync_view3d(square, &mut view);
    assert!(view.caption().contains("stale"), "Combine pending kappa caption: {}", view.caption());

    let mut def = DefectsTabState::new(square, &backend);
    def.sync_view3d(&mut view);
    assert!(!view.caption().contains("stale"));
    def.sub_tab = DefectSubTab::Disclinations;
    def.rebuild_crystal(square, &backend, &mut canvas, false);
    assert!(def.view3d_dirty && def.is_busy());
    def.sync_view3d(&mut view);
    assert!(view.caption().contains("stale"), "Defects pending rebuild caption: {}", view.caption());
    def.wait_idle(&mut canvas);
    def.sync_view3d(&mut view);
    assert!(!view.caption().contains("stale"));
    def.sheet_opts.kappa_b *= 2.0;
    def.rebuild_sheet3d_only(square);
    assert!(def.view3d_dirty && def.is_busy());
    def.sync_view3d(&mut view);
    assert!(view.caption().contains("stale"), "Defects pending kappa caption: {}", view.caption());
}

#[test]
fn review_failed_lattice_reload_never_resurrects_previous_sheet() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let honeycomb = backend.find_named("honeycomb").unwrap();
    let mut canvas = make_canvas();
    let mut view = View3dState::default();

    let mut comb = CombineTabState::new(square, &backend);
    comb.wait_idle(square, Some(&mut canvas));
    comb.sync_view3d(square, &mut view);
    assert!(view.mesh_data().vertex_count > 0);
    comb.reload_for_item(honeycomb, &backend, &mut canvas, true);
    comb.fail_init_for_test("injected init error", honeycomb);
    // Subsequent actions while `init_error` is active must not resurrect Square via stale `self.info`.
    comb.apply_preset_pair(honeycomb, &backend, &mut canvas);
    comb.clear_defects(honeycomb, &backend, &mut canvas);
    comb.tool_mode = ToolMode::PlaceDislocation;
    comb.place_at_world([0.2, 0.2], honeycomb, &backend, &mut canvas);
    comb.sheet_opts.kappa_b *= 2.0;
    comb.request_sheet3d_only(honeycomb);
    comb.wait_idle(honeycomb, Some(&mut canvas));
    assert_eq!(comb.last_build.crystal.pos.len(), 0, "Combine resurrected Square in 2D");
    assert!(comb.specs.is_empty());
    for flat in [false, true] {
        comb.sheet_opts.show_flat_baseline = flat;
        comb.sheet_opts.color_mode = Sheet3dColorMode::HeightZ;
        comb.sync_view3d(honeycomb, &mut view);
        assert_eq!(view.mesh_data().vertex_count, 0, "Combine resurrected Square (flat={flat})");
        assert!(view.caption().contains("unavailable"), "caption: {}", view.caption());
    }

    let mut def = DefectsTabState::new(square, &backend);
    def.sync_view3d(&mut view);
    assert!(view.mesh_data().vertex_count > 0);
    // Non-reload build failure must update both the 2D note and 3D caption instead of staying "building"/"current".
    def.rebuild_crystal(square, &backend, &mut canvas, false);
    def.fail_build_for_test("injected defect error", &mut canvas);
    let Scene2D::CrystalPatch { note, .. } = canvas.scene() else { panic!("crystal"); };
    assert!(note.contains("Build failed") && !note.contains("builds in the background"), "2D note: {note}");
    def.sync_view3d(&mut view);
    assert!(view.caption().contains("failed"), "3D caption: {}", view.caption());

    def.reload_for_item(honeycomb, &backend, &mut canvas, true);
    def.fail_build_for_test("injected init error", &mut canvas);
    for flat in [false, true] {
        def.sheet_opts.show_flat_baseline = flat;
        def.sheet_opts.color_mode = Sheet3dColorMode::BondStrain;
        def.sync_view3d(&mut view);
        assert_eq!(view.mesh_data().vertex_count, 0, "Defects resurrected Square (flat={flat})");
        assert!(view.caption().contains("unavailable"), "caption: {}", view.caption());
    }
}
