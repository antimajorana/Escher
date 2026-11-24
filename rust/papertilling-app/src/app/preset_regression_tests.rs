//! Presets do not depend on what was on screen before (review regressions): earlier Surfaces corner-centre mode and
//! spectrum panel, earlier Combine radius options and spectral overlay; and the Combine undo checkpoint holds only
//! the user's own pins (none when Combine reloads its lattice). Known answers: dodecahedron V/E/F = 20/30/12,
//! chi = 2; a dislocation dipole has two Burgers vectors summing to zero; the 5-7 pair is +-360/6.

use papertilling_core::defect::surgery::DefectSpec;

use super::super::item_key;
use super::{apply, harness, wait_settled};
use crate::analysis_mode::{AnalysisTarget, SpectrumColorOverlay};
use crate::app::{AppMode, NavTab};
use crate::backend::CentreModeUi;
use crate::canvas2d::Scene2D;
use crate::canvas2d::gauge_overlay::CrystalOverlayMode;
use crate::combine::{MANUAL_R_MIN, MotionMode};
use crate::mesh3d::{Sheet3dColorMode, Sheet3dViewOpts, SurfaceColor3d};

const NO_UNDO: &str = "earlier pins have no undo";

#[test]
fn surfaces_preset_resets_corner_centre_mode_and_elastic_colour() {
    let mut h = harness();
    h.run_steps(2);
    h.state_mut().switch_tab(NavTab::Surfaces);
    wait_settled(&mut h, "surfaces");
    {
        let s = &mut h.state_mut().surfaces;
        s.params.centre_mode = CentreModeUi::Site;
        s.elastic.color_mode = SurfaceColor3d::RadialBuckling;
    }
    apply(&mut h, "Honeycomb sphere (dodecahedron)");
    let app = h.state();
    let s = &app.surfaces;
    assert_eq!(s.params.centre_mode, CentreModeUi::Auto);
    assert!(matches!(s.elastic.color_mode, SurfaceColor3d::FaceRingSize));
    let Scene2D::SurfaceNet { mesh } = app.canvas2d.scene() else { panic!("no surface net") };
    // regular dodecahedron: 20 vertices, 30 edges, 12 pentagons, chi = 2, orientable
    assert_eq!((mesh.sites.len(), mesh.bonds.len(), mesh.faces.len(), mesh.chi, mesh.orientable), (20, 30, 12, 2, true));
}

#[test]
fn combine_preset_resets_radius_options_and_overlays() {
    let mut h = harness();
    h.run_steps(2);
    let name = "Combine: square dislocation dipole";
    apply(&mut h, name);
    let r_fresh = h.state().combine.last_build.applied_radius;
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        let c = &mut app.combine;
        (c.opts.auto_radius, c.opts.manual_radius) = (false, MANUAL_R_MIN);
        c.sheet_opts.show_flat_baseline = true;
        c.sheet_opts.z_exag = 2.5;
        c.sheet_opts.kappa_b = 0.5;
        c.show_branch_cuts = true;
        c.show_glide_lines = false;
        c.motion_mode = MotionMode::FreeSetup;
        c.inspectors.circuit.enabled = true;
        c.sheet_opts.color_mode = Sheet3dColorMode::BondStrain;
        c.inspectors.overlay_mode = CrystalOverlayMode::BondStretchStrain;
        c.request_build(&item, false, false, false);
    }
    wait_settled(&mut h, "manual radius");
    apply(&mut h, name);
    let c = &h.state().combine;
    assert!(c.opts.auto_radius, "auto radius restored");
    assert!((c.last_build.applied_radius - r_fresh).abs() < 1e-9, "{} vs fresh {r_fresh}", c.last_build.applied_radius);
    assert_eq!(c.sheet_opts, Sheet3dViewOpts::default());
    assert!(!c.show_branch_cuts && c.show_glide_lines && c.motion_mode == MotionMode::Conserved);
    assert!(!c.inspectors.circuit.enabled);
    assert!(matches!(c.inspectors.overlay_mode, CrystalOverlayMode::Standard));
    let bs: Vec<[f64; 2]> = c.specs.iter().map(|s| match s {
        DefectSpec::Dislocation { b, .. } => *b,
        _ => panic!("dislocations only"),
    }).collect();
    assert!(bs.len() == 2 && (bs[0][0] + bs[1][0]).abs() < 1e-12 && (bs[0][1] + bs[1][1]).abs() < 1e-12);
}

/// The user's single pin on the current Combine lattice (one dislocation or disclination kept).
fn user_pins(h: &mut egui_kittest::Harness<'static, super::PapertillingApp>) {
    let app = h.state_mut();
    let item = app.gallery.active_item().clone();
    app.combine.specs.truncate(1);
    app.combine.undo_stack.clear();
    app.combine.request_build(&item, false, false, false);
    wait_settled(h, "user pins");
}

#[test]
fn combine_preset_on_another_lattice_checkpoints_no_seed_pins() {
    let mut h = harness(); // starts on kagome
    h.run_steps(2);
    h.state_mut().switch_tab(NavTab::Combine);
    wait_settled(&mut h, "kagome combine");
    user_pins(&mut h);
    apply(&mut h, "Combine: square dislocation dipole");
    let app = h.state();
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("square"));
    // the kagome pin cannot be restored on the square lattice, and the seed pair was never the user's: 0 checkpoints
    assert_eq!(app.combine.undo_stack.len(), 0);
    assert!(app.export_status.contains(NO_UNDO), "{}", app.export_status);
    assert_eq!(app.combine.specs.len(), 2);
}

#[test]
fn combine_preset_after_imported_json_reloads_without_checkpoint() {
    let mut h = harness();
    h.run_steps(2);
    apply(&mut h, "Combine: square dislocation dipole");
    user_pins(&mut h);
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        let cfg = app.combine.to_notebook_json(&item);
        app.combine.load_notebook_json(&cfg, &app.backend, &mut app.canvas2d).unwrap();
        assert!(app.combine.override_item.is_some());
    }
    wait_settled(&mut h, "imported override");
    apply(&mut h, "Combine: square dislocation dipole");
    let app = h.state();
    assert!(app.combine.override_item.is_none());
    assert_eq!(app.combine.undo_stack.len(), 0, "override reload must not checkpoint seed pins");
    assert!(app.export_status.contains(NO_UNDO), "{}", app.export_status);
    assert_eq!(app.combine.specs.len(), 2);
    assert!(app.combine.json_status.is_empty());
}

#[test]
fn combine_preset_on_a_retuned_lattice_reloads_without_checkpoint() {
    let mut h = harness();
    h.run_steps(2);
    apply(&mut h, "Combine: square dislocation dipole");
    user_pins(&mut h);
    // Simulate a Combine session built from a differently tuned recipe on the same alias key.
    h.state_mut().combine_seen.2.a += 0.25;
    apply(&mut h, "Combine: square dislocation dipole");
    let app = h.state();
    assert_eq!(app.combine_seen, item_key(app.gallery.active_item()));
    assert_eq!(app.combine.undo_stack.len(), 0, "retuned reload must not checkpoint seed pins");
    assert!(app.export_status.contains(NO_UNDO), "{}", app.export_status);
    assert_eq!(app.combine.specs.len(), 2);
}

#[test]
fn gallery_preset_resets_tuned_sliders_on_same_lattice() {
    let mut h = harness();
    h.run_steps(2);
    let name = "Chiral kagome (p6, no mirrors)";
    apply(&mut h, name);
    let orig_recipe = h.state().gallery.active_item().recipe.clone();
    let s = h.state().gallery.sliders().first().expect("chiral_kagome has sliders").clone();
    let target = if (s.allowed_max - s.default_value).abs() >= (s.default_value - s.allowed_min).abs() {
        0.5 * (s.default_value + s.allowed_max)
    } else {
        0.5 * (s.default_value + s.allowed_min)
    };
    {
        let app = h.state_mut();
        let v = app.gallery.set_param(&s.name, target, &mut app.canvas2d).unwrap();
        assert!((v - s.default_value).abs() > 1e-9, "slider {} moved", s.name);
        assert_ne!(app.gallery.active_item().recipe, orig_recipe);
    }
    apply(&mut h, name);
    let app = h.state();
    assert!((app.gallery.sliders()[0].value - s.default_value).abs() < 1e-9);
    assert_eq!(app.gallery.active_item().recipe, orig_recipe);
}

#[test]
fn defects_and_analysis_presets_reset_sheet_opts_circuit_and_spectrum() {
    let mut h = harness();
    h.run_steps(2);
    apply(&mut h, "Analysis: -60 deg honeycomb eigenmodes");
    {
        let app = h.state_mut();
        app.analysis.spectrum.selected_level = Some(0);
        let d = &mut app.defects;
        d.sheet_opts.show_flat_baseline = true;
        d.sheet_opts.z_exag = 2.5;
        d.sheet_opts.kappa_b = 0.5;
        d.inspectors.circuit.enabled = true;
    }
    apply(&mut h, "Honeycomb +60 deg disclination");
    {
        let app = h.state();
        assert_eq!(app.active_mode, AppMode::Defects);
        let d = &app.defects;
        assert_eq!(d.sheet_opts, Sheet3dViewOpts::default());
        assert!(!d.inspectors.circuit.enabled);
        assert!(matches!(d.inspectors.overlay_mode, CrystalOverlayMode::Standard));
    }
    {
        let d = &mut h.state_mut().defects;
        d.sheet_opts.show_flat_baseline = true;
        d.sheet_opts.z_exag = 2.5;
        d.sheet_opts.kappa_b = 0.5;
        d.inspectors.circuit.enabled = true;
    }
    apply(&mut h, "Analysis: -60 deg honeycomb eigenmodes");
    let app = h.state();
    assert_eq!(app.active_mode, AppMode::Analysis);
    assert_eq!(app.analysis.target, AnalysisTarget::SingleDefect);
    let d = &app.defects;
    assert!(!d.sheet_opts.show_flat_baseline);
    assert!((d.sheet_opts.z_exag - 1.0).abs() < 1e-6);
    assert!((d.sheet_opts.kappa_b - 0.02).abs() < 1e-9);
    assert!(!d.inspectors.circuit.enabled);
    assert!(app.analysis.spectrum.selected_level.is_none());
    let res = app.analysis.spectrum.result.as_ref().expect("fresh LDOS computed");
    assert_eq!(res.ldos.len(), d.last_crystal.pos.len());
    assert_eq!(app.analysis.spectrum.color_overlay, SpectrumColorOverlay::LdosMultiplet);
}

#[test]
fn combine_preset_clears_pending_combine_from_session_restore() {
    let mut h = harness();
    h.run_steps(2);
    // Build a 1-pin Combine configuration for `square`, then simulate a session restore while on Gallery
    // with non-default options so an un-discarded `pending_combine` would overwrite opts, grid, and undo.
    apply(&mut h, "Combine: square dislocation dipole");
    user_pins(&mut h);
    let mut cfg = {
        let app = h.state();
        let item = app.gallery.active_item();
        app.combine.to_notebook_json(item)
    };
    assert_eq!(cfg.defects.len(), 1);
    cfg.options.r_patch = Some(10.0);
    cfg.options.radius = 3;
    cfg.options.grid = 3;
    h.state_mut().combine.grid = 3;
    h.state_mut().switch_tab(NavTab::Gallery);
    h.state_mut().session.pending_combine = Some(cfg);

    apply(&mut h, "Combine: square dislocation dipole");
    let app = h.state();
    assert!(app.session.pending_combine.is_none(), "preset must discard deferred session Combine restore");
    assert!(app.combine.opts.auto_radius, "pending_combine must not overwrite auto_radius");
    assert_eq!(app.combine.opts.core_radius, 2, "pending_combine must not overwrite core_radius");
    assert_eq!(app.combine.grid, 1, "preset must reset compare grid");
    assert!(app.combine.json_status.is_empty());
    assert_eq!(app.combine.undo_stack.len(), 1, "user pin on same lattice checkpointed once");
    let bs: Vec<[f64; 2]> = app.combine.specs.iter().map(|s| match s {
        DefectSpec::Dislocation { b, .. } => *b,
        _ => panic!("dislocations only"),
    }).collect();
    assert_eq!(bs.len(), 2, "preset dipole kept instead of 1-pin session restore");
    assert!((bs[0][0] + bs[1][0]).abs() < 1e-12 && (bs[0][1] + bs[1][1]).abs() < 1e-12);
}

#[test]
fn gallery_preset_switches_to_periodic_grid_independently_of_open_patch_state() {
    let name = "Kagome: ITA symmetry + Voronoi";

    // Case 1: start from Open patch (inflation.enabled == true, steps == 4).
    let mut h_open = harness();
    h_open.run_steps(2);
    {
        let app = h_open.state_mut();
        app.gallery.inflation.steps = 4;
        app.gallery.set_inflation_enabled(true, &mut app.canvas2d);
        assert!(app.gallery.inflation.enabled && app.gallery.inflation.steps == 4);
    }
    apply(&mut h_open, name);
    let snap_from_open = super::snapshot(h_open.state());
    {
        let app = h_open.state();
        assert!(!app.gallery.inflation.enabled);
        assert!(app.gallery.inflation.last_shells.is_empty());
        assert!(matches!(app.canvas2d.scene(), Scene2D::PeriodicCell { repeat: 4, show_unit_cell: true, .. }));
        // 4 * 4 * 6 = 96 Kagome bonds + 4 unit-cell border lines = 100 lines.
        assert_eq!(app.canvas2d.counts().1, 100);
    }

    // Case 2: start from Periodic grid (inflation.enabled == false, repeat == 1).
    let mut h_grid = harness();
    h_grid.run_steps(2);
    {
        let app = h_grid.state_mut();
        app.gallery.set_repeat(1, &mut app.canvas2d);
        app.gallery.set_inflation_enabled(false, &mut app.canvas2d);
        assert!(!app.gallery.inflation.enabled && app.gallery.repeat == 1);
    }
    apply(&mut h_grid, name);
    let app = h_grid.state();
    assert!(!app.gallery.inflation.enabled);
    assert!(matches!(app.canvas2d.scene(), Scene2D::PeriodicCell { repeat: 4, show_unit_cell: true, .. }));
    assert_eq!(app.canvas2d.counts().1, 100);
    assert_eq!(super::snapshot(app), snap_from_open, "Gallery preset result must be independent of prior patch mode");
}
