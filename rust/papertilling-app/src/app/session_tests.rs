//! Session restore on headless apps (no disk unless a temp path is given), the exit-save gating, and the menu.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::*;
use crate::aliases::LatticeAliasKey;
use crate::backend::CatalogueRuleUi;
use crate::scene2d::overlays::GalleryOverlayToggles;
use crate::session::{self, SessionState};
use crate::theme::typography;

fn app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn session_from(raw: &str) -> SessionState {
    session::parse_session(raw).unwrap().0
}

#[test]
fn restore_on_a_headless_app_reproduces_tab_theme_lattice_view_and_camera() {
    let mut a = app();
    let oblique = a.backend.find_named("oblique").unwrap().clone();
    a.select_lattice_and_sync(oblique);
    a.gallery.set_param("b", 1.25, &mut a.canvas2d).unwrap();
    a.gallery.set_repeat(5, &mut a.canvas2d);
    a.gallery.set_overlays(GalleryOverlayToggles { symmetry: true, voronoi: false }, &mut a.canvas2d);
    a.theme = StudioTheme::Studio;
    a.surfaces.target = SurfaceTargetUi::Torus;
    a.surfaces.auto_size = false;
    a.surfaces.params.h = 2;
    a.surfaces.params.k = 3;
    a.switch_tab(NavTab::Surfaces);
    a.view3d.restore_camera(Some(-1.0), Some(0.5), Some(6.0), Some([0.5, -0.25]), Some(true));
    let st = a.capture_session();
    let st = session::parse_session(&st.to_json_pretty().unwrap()).unwrap().0;

    let mut b = app();
    let ctx = egui::Context::default();
    let notes = b.apply_session(&st, Some(&ctx));
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(b.active_tab, NavTab::Surfaces);
    assert_eq!(b.theme, StudioTheme::Studio);
    assert!(ctx.style_of(ctx.theme()).visuals.dark_mode, "the restored theme is applied to the context");
    assert_eq!(b.gallery.base_item().named_key.as_deref(), Some("oblique"));
    let b_val = b.gallery.param_values().into_iter().find(|(n, _)| n == "b").unwrap().1;
    assert_eq!(b_val, 1.25);
    assert_eq!(b.gallery.active_item().recipe, a.gallery.active_item().recipe);
    assert_eq!(b.gallery.repeat, 5);
    assert_eq!(b.gallery.overlays, GalleryOverlayToggles { symmetry: true, voronoi: false });
    assert_eq!((b.surfaces.target, b.surfaces.auto_size, b.surfaces.params.h, b.surfaces.params.k), (SurfaceTargetUi::Torus, false, 2, 3));
    assert_eq!((b.view3d.yaw, b.view3d.pitch, b.view3d.distance, b.view3d.z_up), (-1.0, 0.5, 6.0, true));
    assert_eq!(b.view3d.camera_offset, egui::vec2(0.5, -0.25));
    assert_eq!(b.capture_session(), st, "a restored app saves the same session again");
}

#[test]
fn slider_value_far_out_of_range_is_clamped_to_the_sliders_allowed_max() {
    let backend = AppBackend::new();
    let p4g = backend.catalogue_for_group("p4g", CatalogueRuleUi::Bonds)[0].alias_key.clone();
    let raw = format!(
        r#"{{"schema_version": 1, "lattice": {{"alias_key": {{"group": "{}", "rule": "{}", "dsymbol_key_or_index": "{}"}}, "group": "p4g"}},
            "gallery": {{"params": {{"t1": 1e6, "no_such_param": 2.0}}}}}}"#,
        p4g.group, p4g.rule, p4g.dsymbol_key_or_index
    );
    let mut b = app();
    let notes = b.apply_session(&session_from(&raw), None);
    assert_eq!(b.gallery.base_item().alias_key, p4g);
    let t1 = b.gallery.sliders().iter().find(|s| s.name == "t1").unwrap().clone();
    assert_eq!(t1.value, t1.allowed_max, "1e6 lands on the slider's own upper limit");
    assert!(t1.allowed_max < 1e6);
    assert_eq!(notes, ["1 unknown parameter(s) ignored"]);

    // Far below allowed_min clamps cleanly to allowed_min
    let raw_low = format!(
        r#"{{"schema_version": 1, "lattice": {{"alias_key": {{"group": "{}", "rule": "{}", "dsymbol_key_or_index": "{}"}}, "group": "p4g"}},
            "gallery": {{"params": {{"t1": -1e6}}}}}}"#,
        p4g.group, p4g.rule, p4g.dsymbol_key_or_index
    );
    assert!(b.apply_session(&session_from(&raw_low), None).is_empty());
    let t1_low = b.gallery.sliders().iter().find(|s| s.name == "t1").unwrap().clone();
    assert_eq!(t1_low.value, t1_low.allowed_min, "-1e6 lands on the slider's own lower limit");
}

#[test]
fn unknown_lattice_and_out_of_range_values_fall_back_with_notes_and_no_panic() {
    let raw = r##"{"schema_version": 1, "active_tab": "Nowhere", "png_scale": 3,
        "lattice": {"alias_key": {"group": "p6m", "rule": "bonds", "dsymbol_key_or_index": "#999"}, "group": "p6m"},
        "recent": [{"group": "zz", "rule": "named", "dsymbol_key_or_index": "#0"}],
        "layout": {"left_rail_width": 10000.0, "split_ratio": -3.0},
        "defects": {"kind": "Walls", "wall_index": 9999, "radius": 100.0, "draw_mode": "Sideways"},
        "camera": {"yaw": 7.0, "pitch": 10.0, "distance": 1e30}}"##;
    let mut b = app();
    let notes = b.apply_session(&session_from(raw), None);
    let has = |s: &str| notes.iter().any(|n| n.contains(s));
    assert!(has("lattice not found") && has("PNG scale 3") && has("recent lattice") && has("out of range") && has("unknown tab"), "{notes:?}");
    assert_eq!(b.gallery.base_item().named_key.as_deref(), Some("honeycomb"), "default lattice kept (C1: the app opens on the p6m honeycomb)");
    assert_eq!(b.active_tab, NavTab::Gallery);
    assert_eq!(b.png_scale, 2);
    assert_eq!((b.left_rail_width, b.viewport_split_ratio), (420.0, 0.2), "clamped to the panel limits (260..=420, 0.2..=0.8)");
    assert_eq!((b.defects.sub_tab, b.defects.selected_wall), (crate::defects_tab::DefectSubTab::Walls, 0));
    assert_eq!(b.defects.radius, 8.0, "radius slider range 2.4..=8.0");
    assert!((b.view3d.yaw - (7.0 - std::f32::consts::TAU)).abs() < 1e-6, "yaw 7 wraps to 7 - 2 pi");
    assert_eq!(b.view3d.pitch, std::f32::consts::FRAC_PI_2, "pitch stops at the pole");
    assert!(b.view3d.distance.is_finite() && b.view3d.distance < 100.0);
}

#[test]
fn nan_and_null_numbers_keep_the_current_values() {
    let fresh = app();
    let mut st = fresh.capture_session();
    st.camera.yaw = Some(f32::NAN);
    st.camera.distance = Some(f32::INFINITY);
    st.layout.left_rail_width = Some(f32::NAN);
    st.defects.radius = Some(f64::NAN);
    let back = session::parse_session(&st.to_json_pretty().unwrap()).unwrap().0;
    for s in [&st, &back] {
        let mut b = app();
        let notes = b.apply_session(s, None);
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!((b.view3d.yaw, b.view3d.distance), (fresh.view3d.yaw, fresh.view3d.distance));
        assert_eq!((b.left_rail_width, b.defects.radius), (fresh.left_rail_width, fresh.defects.radius));
    }

    // NaN inside gallery params or combine options never propagates into SliderState or Combine
    let mut c = app();
    let oblique = c.backend.find_named("oblique").unwrap().clone();
    c.select_lattice_and_sync(oblique);
    let default_b = c.gallery.param_values().into_iter().find(|(n, _)| n == "b").unwrap().1;
    let mut bad_nan = c.capture_session();
    bad_nan.gallery.params.insert(String::from("b"), Some(f64::NAN));
    let mut bad_comb = c.combine.to_notebook_json(c.gallery.active_item());
    bad_comb.options.r_patch = Some(f64::NAN);
    bad_nan.combine = Some(bad_comb);
    let notes = c.apply_session(&bad_nan, None);
    assert!(notes.iter().any(|n| n.contains("not applied")) && notes.iter().any(|n| n.contains("non-finite")), "{notes:?}");
    let restored_b = c.gallery.param_values().into_iter().find(|(n, _)| n == "b").unwrap().1;
    assert_eq!(restored_b, default_b);
    assert!(c.session.pending_combine.is_none());
}

#[test]
fn combine_setup_is_kept_only_for_its_own_lattice() {
    let mut b = app();
    let kagome_cfg = b.combine.to_notebook_json(b.gallery.active_item());
    let mut st = b.capture_session();
    st.combine = Some(kagome_cfg.clone());
    assert!(b.apply_session(&st, None).is_empty());
    assert_eq!(b.session.pending_combine.as_ref(), Some(&kagome_cfg));
    let oblique = b.backend.find_named("oblique").unwrap();
    st.lattice = Some(session::LatticeSession { alias_key: oblique.alias_key.clone(), group: oblique.group.clone(), named_key: None });
    st.gallery.params.clear();
    let notes = b.apply_session(&st, None);
    assert!(notes.iter().any(|n| n.contains("another lattice")), "{notes:?}");
    assert!(b.session.pending_combine.is_none());
}

#[test]
fn files_corrupt_empty_foreign_or_missing_restore_nothing_and_exit_save_respects_them() {
    let dir = crate::session::tests::TempDir::new();
    let path = dir.0.join(session::SESSION_FILENAME);
    let mut b = app();
    assert!(b.session.path.is_none(), "tests never use the OS config folder");
    assert_eq!(b.open_session_from(&path, None).unwrap(), None);

    for (raw, overwrite) in [("{oops", true), ("", true), (r#"{"schema_version": 99, "theme": "Studio"}"#, false)] {
        std::fs::write(&path, raw).unwrap();
        assert!(b.open_session_from(&path, None).is_err());
        assert_eq!((b.theme, b.active_tab), (StudioTheme::Paper, NavTab::Gallery), "nothing applied");
        let mut c = app();
        c.session.path = Some(path.clone());
        c.load_session_at_start(&egui::Context::default());
        assert!(c.export_status.starts_with("Session not restored"), "{}", c.export_status);
        assert_eq!(c.session.exit_save_allowed, overwrite, "{raw:?}");
        assert_eq!(c.save_session_on_exit(), overwrite);
        if overwrite {
            assert!(matches!(session::load_session_file(&path), Ok(Some(_))), "a corrupt file is replaced by a valid session");
        } else {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), raw, "a foreign schema file is never replaced");
        }
    }

    // a good file restores at start, and the exit save writes the current state back to it
    let mut good = crate::session::tests::sample_state();
    good.lattice = None;
    good.gallery.params.clear();
    good.recent.clear();
    good.combine = None;
    good.active_mode = Some(String::from("Lattices"));
    good.lattices_view = Some(String::from("NonEuclidean"));
    good.active_tab = Some(String::from("Atlas"));
    session::save_session_file(&path, &good).unwrap();
    let mut c = app();
    c.session.path = Some(path.clone());
    c.load_session_at_start(&egui::Context::default());
    assert_eq!((c.active_tab, c.theme, c.gallery.repeat), (NavTab::Atlas, StudioTheme::Studio, 5));
    c.gallery.set_repeat(2, &mut c.canvas2d);
    assert!(c.save_session_on_exit());
    assert_eq!(session::load_session_file(&path).unwrap().unwrap().0.gallery.repeat, Some(2));
}

#[test]
fn in_memory_modes_never_save() {
    assert!(session_hooks::SessionHost::for_mode(true).path.is_none(), "smoke-test mode has no session file");
    assert!(session_hooks::SessionHost::for_mode(false).path.is_none(), "neither do tests");
    let real = session::default_session_path();
    let stamp = |p: &Option<std::path::PathBuf>| p.as_ref().and_then(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    let before = stamp(&real);
    let mut b = app();
    assert!(!b.save_session_on_exit());
    let mut smoke = PapertillingApp::new(true, String::from("mock"), String::from("mock"));
    smoke.load_session_at_start(&egui::Context::default());
    assert_eq!(smoke.active_tab, NavTab::Surfaces, "smoke-test mode ignores any OS session file");
    assert!(!smoke.save_session_on_exit());
    assert_eq!(stamp(&real), before, "the real session file is untouched");
}

#[test]
fn restoring_builds_only_the_active_tab_and_restores_defects_and_combine() {
    // 1) Restoring onto Gallery with another lattice leaves Defects stale (not built) and Combine idle.
    let mut src = app();
    let square = src.backend.find_named("square").unwrap().clone();
    src.select_lattice_and_sync(square);
    src.defects.sub_tab = crate::defects_tab::DefectSubTab::Disclinations;
    src.defects.selected_disclin = 1;
    src.defects.radius = 4.6;
    src.defects.draw_mode = crate::live_ops_ao::Defect2dMode::Raw;
    src.defects.view_mode = crate::mesh3d::DefectViewMode::Sheet3D;
    src.defects.sheet_opts.color_mode = crate::mesh3d::Sheet3dColorMode::HeightZ;
    let mut st_gallery = src.capture_session();
    st_gallery.active_tab = Some(String::from("Gallery"));

    let mut b = app();
    let init_item = b.gallery.active_item().clone();
    b.combine.wait_idle(&init_item, None);
    assert!(b.apply_session(&st_gallery, None).is_empty());
    assert_eq!(b.active_tab, NavTab::Gallery);
    assert!(b.defects.stale && b.defects.last_crystal.pos.is_empty(), "Defects must not build while Gallery is active");
    assert_ne!(b.combine_seen, item_key(b.gallery.active_item()), "Combine must not reload while Gallery is active");
    assert!(!b.combine.is_busy());

    // 2) Restoring onto Defects runs its background build and applies all defect settings.
    let mut st_defects = st_gallery;
    st_defects.active_mode = Some(String::from("Defects"));
    st_defects.defects_view = Some(String::from("Catalogue2D"));
    st_defects.active_tab = Some(String::from("Defects"));
    let mut d_app = app();
    let d_init_item = d_app.gallery.active_item().clone();
    d_app.combine.wait_idle(&d_init_item, None);
    assert!(d_app.apply_session(&st_defects, None).is_empty());
    assert_eq!(d_app.active_tab, NavTab::Defects);
    d_app.defects.wait_idle(&mut d_app.canvas2d);
    d_app.defects.sync_view3d(&mut d_app.view3d);
    assert_eq!(d_app.defects.sub_tab, crate::defects_tab::DefectSubTab::Disclinations);
    assert_eq!(d_app.defects.selected_disclin, 1);
    assert_eq!(d_app.defects.radius, 4.6);
    assert_eq!(d_app.defects.draw_mode, crate::live_ops_ao::Defect2dMode::Raw);
    assert!(!d_app.defects.relaxed);
    assert_eq!(d_app.defects.view_mode, crate::mesh3d::DefectViewMode::Sheet3D);
    assert_eq!(d_app.defects.sheet_opts.color_mode, crate::mesh3d::Sheet3dColorMode::HeightZ);
    assert!(!d_app.defects.last_crystal.pos.is_empty() && d_app.view3d.has_geometry());
    assert_ne!(d_app.combine_seen, item_key(d_app.gallery.active_item()));
    assert!(!d_app.combine.is_busy());

    // 3) Restoring onto Combine applies pending Combine specs, clears override_item, and completes async build.
    let mut c_src = app();
    assert!(c_src.combine.specs.len() >= 2);
    c_src.combine.specs.truncate(1);
    c_src.combine.opts.auto_radius = false;
    c_src.combine.opts.manual_radius = 10.0;
    c_src.combine.opts.core_radius = 3;
    c_src.combine.grid = 2;
    c_src.switch_tab(NavTab::Combine);
    let mut st_comb = c_src.capture_session();
    assert!(st_comb.combine.is_some());

    let mut c_app = app();
    assert!(c_app.apply_session(&st_comb, None).is_empty());
    assert_eq!(c_app.active_tab, NavTab::Combine);
    assert!(c_app.session.pending_combine.is_none());
    assert!(c_app.combine.override_item.is_none());
    assert!(c_app.combine.json_status.is_empty(), "{}", c_app.combine.json_status);
    assert_eq!(c_app.combine.specs, c_src.combine.specs);
    assert_eq!(c_app.combine.opts.manual_radius, 10.0);
    assert_eq!(c_app.combine.opts.core_radius, 3);
    assert_eq!(c_app.combine.grid, 2);
    let item = c_app.gallery.active_item().clone();
    c_app.combine.wait_idle(&item, Some(&mut c_app.canvas2d));
    c_app.combine.sync_view3d(&item, &mut c_app.view3d);
    assert!(c_app.combine.last_build.refused.is_none() && c_app.view3d.has_geometry());

    // 4) Switching lattice while pending_combine was deferred drops the stale Combine setup on capture.
    st_comb.active_mode = Some(String::from("Lattices"));
    st_comb.lattices_view = Some(String::from("Plane2D"));
    st_comb.active_tab = Some(String::from("Gallery"));
    let mut p_app = app();
    assert!(p_app.apply_session(&st_comb, None).is_empty());
    assert!(p_app.session.pending_combine.is_some());
    let oblique = p_app.backend.find_named("oblique").unwrap().clone();
    p_app.select_lattice_and_sync(oblique);
    assert!(p_app.capture_session().combine.is_none());

    // 5) Spectral 3D colour modes on Defects reset to FaceRingSize with cleared overlays and a note.
    for mode in ["SpectralLdos", "SpectralCharge"] {
        let mut s_app = app();
        s_app.defects.inspectors.overlay_mode = crate::canvas2d::gauge_overlay::CrystalOverlayMode::BondStretchStrain;
        let mut st_spec = s_app.capture_session();
        st_spec.defects.color_mode = Some(String::from(mode));
        let notes = s_app.apply_session(&st_spec, None);
        assert!(
            notes.iter().any(|n| n.contains("3D spectral colouring reset to ring size (spectral views are postponed)")),
            "{mode}: {notes:?}"
        );
        assert_eq!(s_app.defects.sheet_opts.color_mode, crate::mesh3d::Sheet3dColorMode::FaceRingSize);
        assert_eq!(s_app.defects.inspectors.overlay_mode, crate::canvas2d::gauge_overlay::CrystalOverlayMode::Standard);
    }

    // 6) Unlocked export ("Save Session As...") leaves no stray .lock sidecar.
    let dir = crate::session::tests::TempDir::new();
    let exported = dir.0.join("user_save.json");
    c_app.save_session_to(&exported, false).unwrap();
    assert!(exported.is_file());
    assert!(!dir.0.join("user_save.json.lock").exists());
}

#[test]
fn recent_lattices_are_restored_most_recent_first() {
    let backend = AppBackend::new();
    let keys: Vec<LatticeAliasKey> =
        ["square", "oblique", "honeycomb"].iter().map(|k| backend.find_named(k).unwrap().alias_key.clone()).collect();
    let mut st = app().capture_session();
    st.recent = keys.clone();
    let mut b = app();
    assert!(b.apply_session(&st, None).is_empty());
    let got: Vec<LatticeAliasKey> = b.switcher.recent.iter().map(|r| r.alias_key.clone()).collect();
    // honeycomb (the restored lattice) was already selected, so the stored order stays as written
    assert_eq!(got, keys);
}

#[test]
fn top_bar_offers_save_and_open_session() {
    let mut harness = Harness::builder()
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
            app(),
        );
    harness.run();
    harness.get_by_label("Session").click();
    harness.run();
    // present, but never clicked here: they open native file dialogs
    assert_eq!(harness.get_all_by_label("Save Session As...").count(), 1);
    assert_eq!(harness.get_all_by_label("Open Session...").count(), 1);
}

#[test]
fn v1_session_migrates_to_v2_on_load_saves_v2_on_exit_and_leaves_v99_untouched() {
    use super::modes::{AppMode, DefectsView, LatticesView};
    use crate::analysis_mode::AnalysisTarget;
    use crate::atlas_tab::AtlasSubTab;

    let dir = crate::session::tests::TempDir::new();
    let path = dir.0.join(session::SESSION_FILENAME);

    // 1) v1 Defects + SpectralLdos migrates to AppMode::Defects with FaceRingSize and explicit note, then writes v2 on exit.
    let v1_def = r#"{
        "schema_version": 1,
        "active_tab": "Defects",
        "defects": { "kind": "Dislocations", "color_mode": "SpectralLdos" }
    }"#;
    std::fs::write(&path, v1_def).unwrap();
    let mut a = app();
    a.session.path = Some(path.clone());
    a.load_session_at_start(&egui::Context::default());
    assert_eq!((a.active_mode, a.defects_view, a.active_tab), (AppMode::Defects, DefectsView::Catalogue2D, NavTab::Defects));
    assert_eq!(a.defects.sheet_opts.color_mode, crate::mesh3d::Sheet3dColorMode::FaceRingSize);
    assert!(a.export_status.contains("spectral views are postponed"), "{}", a.export_status);
    assert!(a.save_session_on_exit());
    let (saved_v2, _) = session::load_session_file(&path).unwrap().unwrap();
    assert_eq!(saved_v2.schema_version, 3);
    assert_eq!(saved_v2.active_mode.as_deref(), Some("Defects"));
    assert_eq!(saved_v2.defects_view.as_deref(), Some("Catalogue2D"));

    // 2) v1 Atlas + HyperbolicH2 vs SpaceGroups3D maps into the right pristine Lattices sub-views.
    let v1_hyp = r#"{"schema_version": 1, "active_tab": "Atlas", "atlas_sub_view": "HyperbolicH2"}"#;
    let st_hyp = session::parse_session(v1_hyp).unwrap().0;
    let mut h = app();
    assert!(h.apply_session(&st_hyp, None).is_empty());
    assert_eq!((h.active_mode, h.lattices_view), (AppMode::Lattices, LatticesView::NonEuclidean));
    assert!(h.atlas.pristine_only);
    assert_eq!(h.atlas.sub_tab, AtlasSubTab::HyperbolicH2);

    let v1_3d = r#"{"schema_version": 1, "active_tab": "Atlas", "atlas_sub_view": "SpaceGroups3D"}"#;
    let st_3d = session::parse_session(v1_3d).unwrap().0;
    assert!(h.apply_session(&st_3d, None).is_empty());
    assert_eq!((h.active_mode, h.lattices_view), (AppMode::Lattices, LatticesView::ThreeDAndSubperiodic));
    assert!(h.atlas.pristine_only);
    assert_eq!(h.atlas.sub_tab, AtlasSubTab::SpaceGroups3D);

    // 3) A session saved in Analysis (internal route only) keeps its target but opens Lattices with a note (C1:
    //    Analysis is postponed), without stale spectral data.
    let mut an_src = app();
    an_src.analysis.select_target(AnalysisTarget::PlaneLattice);
    an_src.analysis.view_mode = crate::mesh3d::DefectViewMode::Sheet3D;
    an_src.switch_mode(AppMode::Analysis);
    let st_an = an_src.capture_session();
    assert_eq!(st_an.schema_version, 3);
    let mut an_dst = app();
    assert_eq!(an_dst.apply_session(&st_an, None), vec![String::from(crate::game_menu::ANALYSIS_POSTPONED_NOTE)]);
    assert_eq!((an_dst.active_mode, an_dst.active_tab), (AppMode::Lattices, NavTab::Gallery));
    assert_eq!(an_dst.analysis.target, AnalysisTarget::PlaneLattice);
    assert_eq!(an_dst.analysis.view_mode, crate::mesh3d::DefectViewMode::Sheet3D);
    assert!(an_dst.analysis.spectrum.result.is_none() && !an_dst.analysis.spectrum.is_busy());

    // 4) Future schema v99 is refused and never overwritten on exit.
    let v99 = r#"{"schema_version": 99, "theme": "Studio"}"#;
    std::fs::write(&path, v99).unwrap();
    let mut f = app();
    f.session.path = Some(path.clone());
    f.load_session_at_start(&egui::Context::default());
    assert!(!f.session.exit_save_allowed);
    assert!(!f.save_session_on_exit());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), v99);
}

#[test]
fn restored_surface_size_wins_over_boundary_options_edited_before() {
    // WP-U3: the session stores (h, k); boundary options edited before the restore must not hide the restored size
    let mut a = app();
    a.surfaces.target = SurfaceTargetUi::Torus;
    a.surfaces.auto_size = false;
    (a.surfaces.params.h, a.surfaces.params.k) = (2, 3);
    let st = a.capture_session();
    let mut b = app();
    b.surfaces.choose_target(SurfaceTargetUi::Torus);
    b.surfaces.bc.t1 = [5, 0];
    let notes = b.apply_session(&st, None);
    assert!(notes.is_empty(), "{notes:?}");
    assert!(!b.surfaces.bc.explicit, "the options map from the restored (h, k) again");
    assert_eq!((b.surfaces.params.h, b.surfaces.params.k), (2, 3));
}
