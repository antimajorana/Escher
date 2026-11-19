//! Curated presets: every preset reaches its tab and lattice and builds without refusal (known answers below are
//! hand counts, ITA plane groups, and the pasted Python `surface_table` of `surfaces_tab/tests.rs`), applying one
//! twice is idempotent, and a Combine preset pushes exactly one undo checkpoint.

use std::time::{Duration, Instant};

use egui_kittest::Harness;
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::hyperbolic::HyperbolicStyle;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};

use super::{NavTab, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::atlas_tab::hyperbolic_ui::{DiskProjection, HypPick};
use crate::atlas_tab::hyperbolic_view::DiskIsometry;
use crate::backend::WallKindUi;
use crate::canvas2d::Scene2D;
use crate::defects_tab::DefectSubTab;
use crate::presets::PRESETS;
use crate::theme::{StudioTheme, typography};

#[path = "preset_regression_tests.rs"]
mod regression;

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(|ui, app| { let _ = app.frame_ui(ui); }, app)
}

fn preset_idx(name: &str) -> usize {
    PRESETS.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("no preset {name}"))
}

/// Every background job of the active tab has landed (and its 3D view is synced where the tab has one).
fn settled(app: &PapertillingApp) -> bool {
    match app.active_tab {
        NavTab::Gallery | NavTab::Atlas => true,
        NavTab::Surfaces => !app.surfaces.is_building() && !app.surfaces.elastic.is_busy(),
        NavTab::Defects => {
            let d = &app.defects;
            app.preset_pending.is_none() && !d.is_busy() && !d.initializing && !d.view3d_dirty
        }
        NavTab::Combine => {
            let c = &app.combine;
            app.preset_pending.is_none() && !c.is_initializing() && !c.is_building() && !c.view3d_dirty
        }
        NavTab::Analysis => {
            let a = &app.analysis;
            app.preset_pending.is_none()
                && !app.defects.is_busy()
                && !app.defects.initializing
                && !a.is_busy()
                && !a.view3d_dirty
                && (a.spectrum.result.is_some() || a.spectrum.last_error.is_some())
        }
    }
}

fn wait_settled(h: &mut Harness<'static, PapertillingApp>, what: &str) {
    let t0 = Instant::now();
    loop {
        h.step();
        if settled(h.state()) {
            h.step();
            if settled(h.state()) {
                return;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "{what}: builds did not finish within 60 s");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Apply preset `name` and wait for its builds; prints the time taken.
fn apply(h: &mut Harness<'static, PapertillingApp>, name: &str) {
    let t0 = Instant::now();
    h.state_mut().apply_preset(preset_idx(name)).unwrap();
    wait_settled(h, name);
    println!("# preset '{name}': {:.2} s", t0.elapsed().as_secs_f64());
}

/// Everything a preset sets, as text (for the idempotence check).
fn snapshot(app: &PapertillingApp) -> String {
    let it = app.gallery.active_item();
    let tab = match app.active_tab {
        NavTab::Gallery => format!("{:?} {} {}", app.gallery.overlays, app.gallery.repeat, app.gallery.show_unit_cell),
        NavTab::Surfaces => {
            let s = &app.surfaces;
            format!("{:?} {} {} {} {:?} {:?}", s.target, s.auto_size, s.params.h, s.params.k, s.elastic.shape_mode, app.canvas2d.counts())
        }
        NavTab::Defects => {
            let d = &app.defects;
            format!("{:?} {:?} {} {:?} {:?} {}", d.current_selection(), d.sub_tab, d.radius, d.view_mode, d.sheet_opts.color_mode, d.readout.title)
        }
        NavTab::Combine => format!("{:?} {:?} {}", app.combine.specs, app.combine.view_mode, app.combine.undo_stack.len()),
        NavTab::Atlas => {
            let a = &app.atlas;
            format!(
                "{} {:?} {} {:?} {:?} {:?} {} {} {:?} {} {} {:?} {:?} {} {:?} {:?}",
                a.pristine_only, a.sub_tab, a.space3d.selected_sg, a.space3d.selected_net, a.space3d.selected_defect,
                (a.space3d.view.show_cell, a.space3d.view.show_axes, a.space3d.view.z_slice_range, a.space3d.view.depth_cue, a.space3d.view.picked),
                a.hyperbolic.p, a.hyperbolic.q, a.hyperbolic.style, a.hyperbolic.layers,
                a.hyperbolic.wedge_delta, a.hyperbolic.disk_proj, a.hyperbolic.pan, a.hyperbolic.disk_zoom,
                a.hyperbolic.picked, a.hyperbolic.hovered
            )
        }
        NavTab::Analysis => {
            let a = &app.analysis;
            format!(
                "{:?} {:?} {:?} {} {}",
                a.target, a.view_mode, a.spectrum.color_overlay,
                a.spectrum.result.as_ref().map_or(0, |r| r.eigenvalues.len()),
                app.defects.readout.title
            )
        }
    };
    format!("{:?} {:?} {:?} {:?} {:?} {} rev={} | {tab}", app.active_mode, app.lattices_view, app.defects_view, app.active_tab, it.alias_key, it.group, app.gallery.revision())
}

/// Hand-written expectations: tab, named key (`None`: not a named lattice / lattice kept) and ITA plane group.
const EXPECT: [(&str, NavTab, Option<&str>, &str); 15] = [
    ("Kagome: ITA symmetry + Voronoi", NavTab::Gallery, Some("kagome"), "p6m"),
    ("Chiral kagome (p6, no mirrors)", NavTab::Gallery, Some("chiral_kagome"), "p6"),
    ("p4g bond net #1 + Voronoi", NavTab::Gallery, None, "p4g"),
    ("Honeycomb sphere (dodecahedron)", NavTab::Surfaces, Some("honeycomb"), "p6m"),
    ("Square-lattice torus 6x6", NavTab::Surfaces, Some("square"), "p4m"),
    ("Kagome Mobius band", NavTab::Surfaces, Some("kagome"), "p6m"),
    ("Shastry-Sutherland projective plane", NavTab::Surfaces, Some("shastry_sutherland"), "p4g"),
    ("Edge dislocation, triangular lattice", NavTab::Defects, Some("triangular"), "p6m"),
    ("Honeycomb +60 deg disclination", NavTab::Defects, Some("honeycomb"), "p6m"),
    ("Kagome twin wall Wb1", NavTab::Defects, Some("kagome"), "p6m"),
    ("Analysis: -60 deg honeycomb eigenmodes", NavTab::Analysis, Some("honeycomb"), "p6m"),
    ("Combine: square dislocation dipole", NavTab::Combine, Some("square"), "p4m"),
    ("Combine: honeycomb 5-7 pair", NavTab::Combine, Some("honeycomb"), "p6m"),
    ("Atlas: rutile #136 screw dispiration", NavTab::Atlas, None, ""),
    ("Atlas: hyperbolic {7,3} tiling", NavTab::Atlas, None, ""),
];

/// The tab-specific known answers of preset `name` (the app is settled on it).
fn check_known_answers(name: &str, app: &PapertillingApp) {
    let disclin = |i: usize| {
        let d = &app.defects.catalogue.disclinations[i];
        let order = app.defects.catalogue.centres.iter().find(|c| c.id == d.centre_id).map(|c| c.order);
        (d.frank_angle_deg, order)
    };
    let cat_dump = || format!("{:#?}", app.defects.catalogue);
    let net = || match app.canvas2d.scene() {
        Scene2D::SurfaceNet { mesh } => (mesh.sites.len(), mesh.bonds.len(), mesh.faces.len(), mesh.chi, mesh.orientable),
        _ => panic!("{name}: no surface net"),
    };
    match name {
        "Kagome: ITA symmetry + Voronoi" => {
            let o = app.gallery.overlays;
            assert!(o.symmetry && o.voronoi && app.gallery.repeat == 4, "{name}");
        }
        "Chiral kagome (p6, no mirrors)" => assert!(app.gallery.overlays.symmetry && !app.gallery.overlays.voronoi),
        "p4g bond net #1 + Voronoi" => {
            let k = &app.gallery.active_item().alias_key;
            assert_eq!((k.group.as_str(), k.rule.as_str()), ("p4g", "bonds"));
            assert!(app.gallery.overlays.voronoi && !app.gallery.overlays.symmetry);
        }
        // dodecahedron: V = 20, E = 30, F = 12, chi = 2 (Python surface_table: 20 sites)
        "Honeycomb sphere (dodecahedron)" => assert_eq!(net(), (20, 30, 12, 2, true), "{name}"),
        // one site, two bonds, one square per cell: 36 / 72 / 36 on a 6x6 torus, chi = 0
        "Square-lattice torus 6x6" => assert_eq!(net(), (36, 72, 36, 0, true), "{name}"),
        // Python surface_table: kagome Moebius band OK with 325 sites; non-orientable
        "Kagome Mobius band" => {
            let (v, _, _, _, orientable) = net();
            assert_eq!((v, orientable), (325, false), "{name}");
        }
        // hemi-icosahedron: V = 6, E = 15, F = 10, chi = 1, non-orientable
        "Shastry-Sutherland projective plane" => assert_eq!(net(), (6, 15, 10, 1, false), "{name}"),
        "Edge dislocation, triangular lattice" => {
            let d = &app.defects.catalogue.dislocations[0];
            assert!((d.norm - 1.0).abs() < 1e-9, "{name}: shortest |b| = a = 1\n{}", cat_dump());
            assert_eq!(app.defects.current_selection(), crate::backend::DefectSelectionUi::Dislocation(0));
        }
        "Honeycomb +60 deg disclination" => {
            assert_eq!(disclin(3), (60, Some(6)), "{name}\n{}", cat_dump());
            assert_eq!(app.defects.sub_tab, DefectSubTab::Disclinations);
        }
        "Kagome twin wall Wb1" => {
            let w = &app.defects.catalogue.walls[0];
            assert!(w.kind == WallKindUi::TwinWb && w.id == "Wb1", "{name}\n{}", cat_dump());
        }
        "Analysis: -60 deg honeycomb eigenmodes" => {
            assert_eq!(app.active_mode, super::AppMode::Analysis);
            assert_eq!(app.analysis.target, crate::analysis_mode::AnalysisTarget::SingleDefect);
            assert_eq!(disclin(2), (-60, Some(6)), "{name}\n{}", cat_dump());
            let res = app.analysis.spectrum.result.as_ref().expect("spectrum computed");
            assert_eq!(res.ldos.len(), app.defects.last_crystal.pos.len(), "{name}: one LDOS value per site");
            assert_eq!(app.analysis.spectrum.color_overlay, crate::analysis_mode::SpectrumColorOverlay::LdosMultiplet);
            assert!(app.view3d.legend().is_some(), "{name}: 3D LDOS colorbar");
            assert!(app.canvas2d.gauge_overlay().and_then(|o| o.legend.as_ref()).is_some(), "{name}: 2D LDOS colorbar");
        }
        "Combine: square dislocation dipole" => {
            let bs: Vec<[f64; 2]> = app.combine.specs.iter().map(|s| match s {
                DefectSpec::Dislocation { b, .. } => *b,
                _ => panic!("{name}: dislocations only"),
            }).collect();
            assert_eq!(bs.len(), 2);
            assert!((bs[0][0].hypot(bs[0][1]) - 1.0).abs() < 1e-9, "{name}: |b| = a = 1");
            assert!((bs[0][0] + bs[1][0]).abs() < 1e-12 && (bs[0][1] + bs[1][1]).abs() < 1e-12, "{name}: b1 + b2 = 0");
        }
        "Combine: honeycomb 5-7 pair" => {
            let angles: Vec<f64> = app.combine.specs.iter().map(|s| match s {
                DefectSpec::Disclination { angle_deg, .. } => *angle_deg,
                _ => panic!("{name}: disclinations only"),
            }).collect();
            assert_eq!(angles, [60.0, -60.0], "{name}: +-360/6");
        }
        "Atlas: rutile #136 screw dispiration" => {
            assert_eq!((app.active_mode, app.defects_view), (super::AppMode::Defects, super::DefectsView::AtlasDefects));
            assert!(!app.atlas.pristine_only);
            let sp = &app.atlas.space3d;
            assert_eq!(
                (app.atlas.sub_tab, sp.selected_sg, sp.selected_net, sp.selected_defect),
                (AtlasSubTab::SpaceGroups3D, 136, CanonicalNet3d::Rutile136, LineDefect3d::ScrewDispiration42_136),
                "{name}"
            );
            assert!(
                !sp.view.show_cell && !sp.view.show_axes && !sp.view.depth_cue
                    && sp.view.z_slice_range == [0.0, 1.0]
                    && sp.view.picked.is_none() && sp.view.hovered.is_none(),
                "{name}: Space3dView reset"
            );
            let s = sp.summary.as_ref().expect("#136 analysed");
            // P4_2/mnm has a 4_2 screw axis and an n glide: nonsymmorphic
            assert!(format!("{}", s.number) == "136" && !s.is_split, "{name}");
        }
        "Atlas: hyperbolic {7,3} tiling" => {
            assert_eq!((app.active_mode, app.lattices_view), (super::AppMode::Lattices, super::LatticesView::NonEuclidean));
            assert!(app.atlas.pristine_only);
            let hy = &app.atlas.hyperbolic;
            assert_eq!(
                (app.atlas.sub_tab, hy.p, hy.q, hy.style, hy.layers, hy.wedge_delta, hy.disk_proj),
                (AtlasSubTab::HyperbolicH2, 7, 3, HyperbolicStyle::RegularPQ, 2, 0, DiskProjection::Poincare),
                "{name}"
            );
            assert_eq!(hy.pan, DiskIsometry::IDENTITY, "{name}: pan reset");
            assert_eq!(hy.disk_zoom, 1.0, "{name}: zoom reset");
            assert!(hy.picked.is_none() && hy.hovered.is_none(), "{name}: picks reset");
            assert!(app.atlas.hyperbolic.status.contains("κ(D) = -"), "{name}: {}", app.atlas.hyperbolic.status);
        }
        _ => panic!("no known answer for {name}"),
    }
}

/// Apply each preset of `tab` twice: stated tab and lattice, no refusal, 3D geometry, idempotent second apply.
fn check_tab(tab: NavTab) {
    let mut h = harness();
    h.run_steps(2);
    let mut cases: Vec<_> = EXPECT.iter().copied().filter(|e| e.1 == tab).collect();
    if tab == NavTab::Atlas {
        // rutile #136 and {7,3} are close to the Atlas defaults: move the Atlas away from them first, and apply
        // {7,3} before rutile so the rutile preset has to switch the sub-view back to space groups
        let a = &mut h.state_mut().atlas;
        (a.space3d.selected_sg, a.space3d.selected_net, a.space3d.selected_defect) =
            (221, CanonicalNet3d::SimpleCubic, LineDefect3d::None);
        a.space3d.view.show_cell = true;
        a.space3d.view.show_axes = true;
        a.space3d.view.depth_cue = true;
        a.space3d.view.z_slice_range = [0.2, 0.8];
        a.space3d.view.picked = Some(0);
        a.space3d.view.hovered = Some(1);
        (a.hyperbolic.p, a.hyperbolic.q, a.hyperbolic.layers, a.hyperbolic.wedge_delta) = (5, 4, 3, 1);
        a.hyperbolic.style = HyperbolicStyle::DualQP;
        a.hyperbolic.disk_proj = DiskProjection::Klein;
        a.hyperbolic.pan = DiskIsometry::from_origin([0.3, 0.0]);
        a.hyperbolic.disk_zoom = 2.5;
        a.hyperbolic.picked = Some(HypPick::Face(0));
        a.hyperbolic.hovered = Some(HypPick::Vertex(0));
        cases.reverse();
    }
    for (name, want_tab, key, group) in cases {
        let before = h.state().gallery.active_item().alias_key.clone();
        apply(&mut h, name);
        let app = h.state();
        assert_eq!(app.active_tab, want_tab, "{name}: tab");
        let it = app.gallery.active_item();
        if want_tab == NavTab::Atlas {
            assert_eq!(it.alias_key, before, "{name}: an Atlas scene keeps the lattice");
        } else {
            assert_eq!(it.group, group, "{name}: plane group");
            if key.is_some() {
                assert_eq!(it.named_key.as_deref(), key, "{name}: named lattice");
            }
        }
        match want_tab {
            NavTab::Surfaces => assert!(!app.surfaces.has_refusal(), "{name}: surface refused"),
            NavTab::Defects => assert!(app.defects.error.is_none(), "{name}: {:?}", app.defects.error),
            NavTab::Combine => {
                assert!(app.combine.init_error.is_none() && app.combine.last_build.refused.is_none(), "{name}: {:?}", app.combine.last_build.refused);
                assert!(app.combine.last_build.crystal.pos.len() > 10, "{name}: crystal built");
            }
            _ => {}
        }
        if want_tab != NavTab::Gallery {
            assert!(app.view3d.mesh_data().vertex_count > 0, "{name}: empty 3D view");
        }
        assert!(app.canvas2d.counts().0 > 0, "{name}: empty 2D view");
        check_known_answers(name, app);
        let first = snapshot(app);
        apply(&mut h, name);
        assert_eq!(snapshot(h.state()), first, "{name}: applying twice changes the state");
    }
}

#[test]
fn gallery_presets_reach_their_lattice_and_overlays() {
    check_tab(NavTab::Gallery);
}

#[test]
fn surfaces_presets_build_ok_surfaces_with_known_counts() {
    check_tab(NavTab::Surfaces);
}

#[test]
fn defects_presets_select_the_expected_catalogue_classes() {
    check_tab(NavTab::Defects);
}

#[test]
fn combine_presets_place_their_pins() {
    check_tab(NavTab::Combine);
}

#[test]
fn atlas_presets_build_their_scenes() {
    check_tab(NavTab::Atlas);
}

#[test]
fn analysis_presets_compute_spectral_ldos() {
    check_tab(NavTab::Analysis);
}

#[test]
fn preset_list_has_unique_names_descriptions_and_valid_lattices() {
    let app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    assert!((12..=15).contains(&PRESETS.len()));
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PRESETS.len(), "unique names");
    for p in PRESETS {
        assert!(!p.description.trim().is_empty(), "{}: description", p.name);
        if let Some(l) = p.lattice {
            assert!(app.preset_lattice_item(l).is_some(), "{}: lattice {l:?}", p.name);
        }
    }
    for tab in NavTab::ALL {
        assert!(PRESETS.iter().any(|p| p.tab() == tab), "{tab:?} has a preset");
    }
    for mode in super::AppMode::ALL {
        assert!(PRESETS.iter().any(|p| p.mode() == mode), "{mode:?} has a preset");
    }
    assert_eq!(EXPECT.map(|e| e.0), PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().as_slice(), "one expectation per preset");
}

#[test]
fn combine_preset_pushes_one_undo_checkpoint_and_undo_restores_the_pins() {
    let mut h = harness();
    h.run_steps(2);
    apply(&mut h, "Combine: square dislocation dipole");
    let preset_specs = h.state().combine.specs.clone();
    // the user replaces the pins by a single dislocation of their own
    {
        let app = h.state_mut();
        let item = app.gallery.active_item().clone();
        app.combine.specs.truncate(1);
        app.combine.undo_stack.clear();
        app.combine.request_build(&item, false, false, false);
    }
    wait_settled(&mut h, "user pins");
    let user_specs = h.state().combine.specs.clone();
    assert_eq!(user_specs.len(), 1);

    apply(&mut h, "Combine: square dislocation dipole");
    assert_eq!(h.state().combine.undo_stack.len(), 1, "exactly one checkpoint");
    assert_eq!(h.state().combine.specs, preset_specs);
    apply(&mut h, "Combine: square dislocation dipole");
    assert_eq!(h.state().combine.undo_stack.len(), 1, "re-applying the same pins pushes nothing");

    let app = h.state_mut();
    let item = app.gallery.active_item().clone();
    app.combine.undo_last(&item);
    assert_eq!(app.combine.specs, user_specs, "undo restores the user's pins");
    wait_settled(&mut h, "undo");
    assert!(h.state().combine.last_build.refused.is_none());
}
