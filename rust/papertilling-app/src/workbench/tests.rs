//! Generic Workbench host tests over `WorkbenchView::ALL` (stage-independent: they compare with what the active
//! panel itself reports, so they keep holding when a stage replaces its stub), plus hand-count checks of the 2D
//! scene helpers.

use std::time::{Duration, Instant};

use super::scene_helpers::{NAN_RGBA, heat_rgba, polyline_scene, unit_circle_lines, vector_scene, with_lines};
use super::{ModeView, WorkbenchView};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;

fn new_app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn frame(app: &mut PapertillingApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.frame_ui(ui);
    });
}

fn route(app: &mut PapertillingApp, wv: WorkbenchView) {
    match wv.mode_view() {
        ModeView::Lattices(v) => app.select_lattices_view(v),
        ModeView::Defects(v) => app.select_defects_view(v),
    }
}

/// Run frames until the workbench (and, on S3, its Defects source) is idle and synced; at most 20 s.
fn settle(app: &mut PapertillingApp, ctx: &egui::Context, what: &str) {
    let t0 = Instant::now();
    for _ in 0..6 {
        frame(app, ctx);
    }
    while app.workbench_busy() || app.workbench.dirty {
        assert!(t0.elapsed() < Duration::from_secs(20), "{what}: workbench still busy after 20 s");
        std::thread::sleep(Duration::from_millis(10));
        frame(app, ctx);
    }
}

fn scene_title(s: &Scene2D) -> Option<&str> {
    match s {
        Scene2D::VectorScene { primitives } => Some(&primitives.title),
        Scene2D::CrystalPatch { title, .. } => Some(title),
        _ => None,
    }
}

fn mode_and_view(app: &PapertillingApp) -> ModeView {
    match app.active_mode {
        AppMode::Defects => ModeView::Defects(app.defects_view),
        _ => ModeView::Lattices(app.lattices_view),
    }
}

#[test]
fn routing_tables_are_consistent_and_labels_unique() {
    let wb_lat: Vec<_> = LatticesView::ALL.into_iter().filter(|v| v.nav_tab() == NavTab::Workbench).collect();
    let wb_def: Vec<_> = DefectsView::ALL.into_iter().filter(|v| v.nav_tab() == NavTab::Workbench).collect();
    assert_eq!((wb_lat.len(), wb_def.len()), (3, 2), "S4..S6 are Lattices views, S2 and S3 Defects views");
    for v in LatticesView::ALL {
        assert_eq!(v.workbench_view().is_some(), v.nav_tab() == NavTab::Workbench, "{v:?}");
    }
    for v in DefectsView::ALL {
        assert_eq!(v.workbench_view().is_some(), v.nav_tab() == NavTab::Workbench, "{v:?}");
    }
    for wv in WorkbenchView::ALL {
        let back = match wv.mode_view() {
            ModeView::Lattices(v) => v.workbench_view(),
            ModeView::Defects(v) => v.workbench_view(),
        };
        assert_eq!(back, Some(wv));
    }
    let mut labels: Vec<&str> = LatticesView::ALL.iter().map(|v| v.label()).collect();
    labels.extend(DefectsView::ALL.iter().map(|v| v.label()));
    labels.extend(AppMode::ALL.iter().map(|m| m.label()));
    let n = labels.len();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels.len(), n, "sub-nav and mode labels must be unique (kittest get_by_label)");

    // the internal route never shows: headings are realm-native and never equal a Command Deck card title
    let mut app = new_app();
    let titles: Vec<String> = crate::game_menu::Realm::ALL
        .into_iter()
        .flat_map(|r| app.deck_cards(r, 0, "").into_iter().map(|c| c.title))
        .collect();
    for wv in WorkbenchView::ALL {
        let h = wv.heading();
        assert!(!h.contains("Workbench"), "{wv:?}: heading '{h}' names the internal route");
        assert!(!titles.iter().any(|t| t == h), "{wv:?}: heading '{h}' equals a card title");
    }
}

#[test]
fn every_workbench_view_builds_from_default_state_without_panic() {
    let ctx = egui::Context::default();
    let mut app = new_app();
    for wv in WorkbenchView::ALL {
        route(&mut app, wv);
        settle(&mut app, &ctx, &format!("{wv:?}"));
        assert_eq!(app.active_tab, NavTab::Workbench, "{wv:?}");
        assert_eq!(app.workbench.view, wv);
        assert_eq!(mode_and_view(&app), wv.mode_view());
        assert!(app.workbench.error().is_none(), "{wv:?}: {:?}", app.workbench.error());
        assert!(!app.workbench.dirty);
        let want = app.workbench.active().scenes();
        assert_eq!(scene_title(app.canvas2d.scene()), scene_title(&want.scene2d), "{wv:?}: canvas shows the panel scene");
        if wv.uses_3d() {
            assert!(app.view3d.has_geometry(), "{wv:?}: a 3D view never shows an empty mesh");
            let cap = app.view3d.caption();
            assert!(!cap.contains("unavailable") && !cap.contains("stale") && !cap.contains("Build failed"), "{cap}");
        }
        if want.mesh.is_none() {
            assert_eq!(app.view3d.caption(), super::CAPTION_2D_ONLY, "{wv:?}");
        }
    }
}

#[test]
fn workbench_view_survives_session_roundtrip() {
    let ctx = egui::Context::default();
    let mut app = new_app();
    app.set_advanced(true);
    for wv in WorkbenchView::ALL {
        route(&mut app, wv);
        settle(&mut app, &ctx, &format!("{wv:?}"));
        let st = app.capture_session();
        assert_eq!(st.workbench.as_ref().and_then(|w| w.get("view")).and_then(|v| v.as_str()), Some(format!("{wv:?}").as_str()));
        let json = st.to_json_pretty().unwrap();
        let (back, dropped) = crate::session::parse_session(&json).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(back.workbench, st.workbench);
        let mut fresh = new_app();
        let notes = fresh.apply_session(&back, None);
        assert!(notes.iter().all(|n| !n.contains("workbench")), "{wv:?}: {notes:?}");
        assert_eq!(fresh.active_tab, NavTab::Workbench, "{wv:?}");
        assert_eq!(fresh.workbench.view, wv);
        assert_eq!(mode_and_view(&fresh), wv.mode_view());
    }
}

#[test]
fn workbench_restore_is_lenient() {
    let mut app = new_app();
    let before = app.workbench.view;
    let mut notes = Vec::new();
    app.workbench.restore(Some(&serde_json::json!(5)), &mut notes);
    assert_eq!(notes.len(), 1, "a non-object is ignored with one note");
    app.workbench.restore(Some(&serde_json::json!({"view": "NoSuchView"})), &mut notes);
    assert_eq!(notes.len(), 2, "an unknown view keeps the current one with one note");
    assert_eq!(app.workbench.view, before);
    app.workbench.restore(None, &mut notes);
    assert_eq!(notes.len(), 2, "a session without the field (schema 2 before P0) is silent");
}

#[test]
fn workbench_reresyncs_after_lattice_switch() {
    let ctx = egui::Context::default();
    let mut app = new_app();
    for wv in [WorkbenchView::S2Defects, WorkbenchView::S5Curved2d] {
        route(&mut app, wv);
        settle(&mut app, &ctx, "before switch");
        let current = app.gallery.base_item().alias_key.clone();
        let other = ["square", "honeycomb"]
            .into_iter()
            .filter_map(|k| app.backend.find_named(k).cloned())
            .find(|it| it.alias_key != current)
            .expect("square or honeycomb differs from the current lattice");
        app.select_lattice_and_sync(other);
        settle(&mut app, &ctx, "after switch");
        assert_eq!(app.active_tab, NavTab::Workbench);
        assert!(!matches!(app.canvas2d.scene(), Scene2D::PeriodicCell { .. }), "{wv:?}: Gallery cell left on the canvas");
        let want = app.workbench.active().scenes();
        assert_eq!(scene_title(app.canvas2d.scene()), scene_title(&want.scene2d), "{wv:?}");
    }
}

#[test]
fn switch_tab_workbench_maps_back_to_its_mode() {
    let mut app = new_app();
    for wv in WorkbenchView::ALL {
        route(&mut app, wv);
        app.switch_tab(NavTab::Gallery);
        assert_eq!(app.active_tab, NavTab::Gallery);
        app.switch_tab(NavTab::Workbench);
        assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, wv));
        assert_eq!(mode_and_view(&app), wv.mode_view());
    }
}

// ---- scene helpers: known answers by hand count ----

#[test]
fn vector_scene_unit_square_hand_counts() {
    let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [f64::NAN, 0.0]];
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 9), (1, 1), (0, 4)];
    let faces = vec![vec![0, 1, 2, 3], vec![0, 1], vec![0, 1, 7], vec![0, 1, 4]];
    let Scene2D::VectorScene { primitives: p } = vector_scene(&pts, &edges, &faces, |_| [1, 2, 3, 4], "t", "n") else {
        panic!("vector scene expected")
    };
    assert_eq!(p.polygons.len(), 1, "only the square is a valid face");
    assert_eq!(p.lines.len(), 4, "4 sides; out-of-range, self-loop and NaN edges skipped");
    assert_eq!(p.circles.len(), 4, "the NaN site is skipped");
    assert_eq!((p.site_count, p.bond_count), (4, 4));
    assert_eq!(p.bbox, ([0.0, 0.0], [1.0, 1.0]));
    assert_eq!(p.edge_line_indices, vec![Some(0), Some(1), Some(2), Some(3), None, None, None]);
    assert_eq!(p.site_circle_indices, vec![Some(0), Some(1), Some(2), Some(3), None]);
    assert!((p.circles[0].radius_world - 0.12).abs() < 1e-6, "0.12 x mean edge length 1");
}

#[test]
fn polylines_and_unit_circle_hand_counts() {
    let Scene2D::VectorScene { primitives: p } =
        polyline_scene(&[vec![[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]], vec![[0.0, 0.0], [f64::NAN, 1.0]]], [0, 0, 0, 255], "t", "n")
    else {
        panic!("vector scene expected")
    };
    assert_eq!(p.lines.len(), 2);
    assert_eq!(p.bbox, ([0.0, 0.0], [1.0, 2.0]));

    let n = 8;
    let ring = unit_circle_lines(n, [0, 0, 0, 255]);
    assert_eq!(ring.len(), n);
    let chord = 2.0 * (std::f64::consts::PI / n as f64).sin(); // chord of a regular n-gon inscribed in the unit circle
    for l in &ring {
        for q in [l.a, l.b] {
            assert!((f64::from(q[0]).hypot(f64::from(q[1])) - 1.0).abs() < 1e-6);
        }
        let len = f64::from(l.a[0] - l.b[0]).hypot(f64::from(l.a[1] - l.b[1]));
        assert!((len - chord).abs() < 1e-6, "{len} vs {chord}");
    }
    assert_eq!(unit_circle_lines(1, [0; 4]).len(), 3, "at least a triangle");

    let empty = vector_scene(&[], &[], &[], |_| [0; 4], "t", "n");
    let Scene2D::VectorScene { primitives: p } = with_lines(empty, ring) else { panic!("vector scene expected") };
    let (lo, hi) = p.bbox;
    assert!((lo[0] + 1.0).abs() < 1e-6 && (hi[0] - 1.0).abs() < 1e-6, "{:?}", p.bbox);
}

#[test]
fn heat_ramp_endpoints_midpoint_and_clamping() {
    let lo = heat_rgba(-2.0, -2.0, 2.0);
    let mid = heat_rgba(0.0, -2.0, 2.0);
    let hi = heat_rgba(2.0, -2.0, 2.0);
    assert!(lo[2] > lo[0], "low end is blue");
    assert!(hi[0] > hi[2], "high end is red");
    assert!(mid[0] == mid[1] && mid[1] == mid[2], "midpoint is neutral grey-white");
    assert_eq!(heat_rgba(-9.0, -2.0, 2.0), lo, "clamped below");
    assert_eq!(heat_rgba(9.0, -2.0, 2.0), hi, "clamped above");
    assert_eq!(heat_rgba(f64::NAN, -2.0, 2.0), NAN_RGBA);
    assert_eq!(heat_rgba(1.0, 3.0, 3.0), mid, "an empty range maps to the midpoint");
    // a diverging ramp is not monotone per channel, but red minus blue rises from end to end
    let warmth: Vec<i32> = (0..=8)
        .map(|k| {
            let c = heat_rgba(f64::from(k), 0.0, 8.0);
            i32::from(c[0]) - i32::from(c[2])
        })
        .collect();
    assert!(warmth.windows(2).all(|w| w[0] < w[1]), "red minus blue rises strictly: {warmth:?}");
}
