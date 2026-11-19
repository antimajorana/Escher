#![allow(clippy::field_reassign_with_default)] // tests set a few fields of default states
//! Known-answer and headless (egui_kittest, 1280 x 800) tests of the interactive Atlas: unit-cell box and axes wires,
//! the z-slice slab in the mesh, 3D picking through the camera under two presets, the H² highlight in both scenes,
//! the Mobius pan readout, and the dirty flag after view toggles.

use egui::{Event, Modifiers, PointerButton};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};

use super::AtlasSubTab;
use super::hyperbolic_scene::{HYP_PICK_RGBA, HYP_PICK_RGBA8, sheet_points};
use super::hyperbolic_ui::{DiskProjection, HypPick, HyperbolicPanelState};
use super::hyperbolic_view::{DiskIsometry, dist_from_origin, lorentz_centroid, poincare_to_klein};
use super::space3d_geom::net_frame;
use super::space3d_ui::Space3dPanelState;
use super::space3d_view::pick_xy;
use crate::app::{NavTab, PapertillingApp};
use crate::mesh3d::atlas_net_view::{DEPTH_CUE_MIN_ALPHA, net_mesh_points};
use crate::scene2d::Scene2D;
use crate::theme::{StudioTheme, typography};
use crate::view3d::{CameraPreset, View3dState, pick_nearest, project_point};

const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn simple_cubic() -> Space3dPanelState {
    let mut s = Space3dPanelState::default();
    s.selected_net = CanonicalNet3d::SimpleCubic;
    s.selected_defect = LineDefect3d::None;
    s
}

#[test]
fn unit_cell_box_adds_twelve_wires_and_axes_three() {
    let mut s = simple_cubic();
    let base = s.build_scenes(&IDENTITY).1.wires.len();
    s.view.show_cell = true;
    let with_cell = s.build_scenes(&IDENTITY).1.wires.len();
    s.view.show_axes = true;
    let with_both = s.build_scenes(&IDENTITY).1.wires.len();
    // a wire is a vertex pair (LineList)
    assert_eq!(with_cell - base, 2 * 12);
    assert_eq!(with_both - with_cell, 2 * 3);
}

#[test]
fn slab_filters_sites_and_bonds_of_the_simple_cubic_mesh() {
    // 2x2x2 simple-cubic block: 8 sites, 12 unit bonds (4 in each z layer, 4 vertical).
    let mut s = simple_cubic();
    let (sc, mesh, _) = s.build_scenes(&IDENTITY);
    assert_eq!((mesh.vertex_count, mesh.edge_count), (8, 12));
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    assert_eq!((primitives.site_count, primitives.bond_count), (8, 12));
    // lower half: one layer, 4 sites and its 4 in-layer bonds
    s.view.z_slice_range = [0.0, 0.5];
    let (sc, mesh, _) = s.build_scenes(&IDENTITY);
    assert_eq!((mesh.vertex_count, mesh.edge_count), (4, 4));
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    assert_eq!((primitives.site_count, primitives.bond_count), (4, 4));
    // the projection's frame does not move with the slab
    let full_bbox = { s.view.z_slice_range = [0.0, 1.0]; s.build_scenes(&IDENTITY).0 };
    let Scene2D::VectorScene { primitives: full } = full_bbox else { panic!() };
    assert_eq!(full.bbox, primitives.bbox);
}

#[test]
fn xy_pick_takes_the_top_site_and_never_a_hidden_one() {
    // two sites stacked on the z axis and one aside
    let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    assert_eq!(pick_xy(&pos, &[true; 3], [0.01, 0.0], 0.1), Some(1));
    assert_eq!(pick_xy(&pos, &[true, false, true], [0.01, 0.0], 0.1), Some(0));
    assert_eq!(pick_xy(&pos, &[false, false, true], [0.01, 0.0], 0.1), None);
    assert_eq!(pick_xy(&pos, &[true; 3], [0.5, 0.0], 0.1), None);
}

#[test]
fn projected_point_is_picked_under_two_camera_presets() {
    use crate::view3d::CameraPreset;
    // 400 x 300 viewport at the origin; default distance 3.1; vertical FOV 45 deg: f = 1 / tan(22.5 deg) = 1 + sqrt(2).
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
    let (f, aspect, dist) = (1.0 + 2.0_f32.sqrt(), 400.0 / 300.0, 3.1_f32);
    let pts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]];
    let all = [true; 3];
    let centre = egui::pos2(200.0, 150.0);
    // (0.5, 0, 0): x stays the screen x under Top and Front, at depth 3.1: x_ndc = f / aspect * 0.5 / 3.1
    let x_screen = egui::pos2(200.0 + 200.0 * f / aspect * 0.5 / dist, 150.0);
    // (0, 0, 0.5) under Front (z up): y_ndc = f * 0.5 / 3.1 above the centre
    let z_screen = egui::pos2(200.0, 150.0 - 150.0 * f * 0.5 / dist);

    let mut v = View3dState::default();
    v.apply_preset(CameraPreset::Top);
    let mvp = v.mvp_for_rect(rect);
    let (s1, _) = project_point(&mvp, rect, pts[1]).unwrap();
    assert!(s1.distance(x_screen) < 0.01, "{s1:?} vs {x_screen:?}");
    assert_eq!(pick_nearest(&pts, &all, &mvp, rect, x_screen, 4.0), Some(1));
    // Top looks down -z: (0, 0, 0.5) covers the origin and is nearer, so it wins the centre
    assert_eq!(pick_nearest(&pts, &all, &mvp, rect, centre, 4.0), Some(2));
    // outside the slab it is never picked
    assert_eq!(pick_nearest(&pts, &[true, true, false], &mvp, rect, centre, 4.0), Some(0));
    assert_eq!(pick_nearest(&pts, &[false, true, false], &mvp, rect, centre, 4.0), None);

    v.apply_preset(CameraPreset::Front);
    let mvp = v.mvp_for_rect(rect);
    let (s2, _) = project_point(&mvp, rect, pts[2]).unwrap();
    assert!(s2.distance(z_screen) < 0.01, "{s2:?} vs {z_screen:?}");
    assert_eq!(pick_nearest(&pts, &all, &mvp, rect, z_screen, 4.0), Some(2));
    assert_eq!(pick_nearest(&pts, &all, &mvp, rect, x_screen, 4.0), Some(1));
    assert_eq!(pick_nearest(&pts, &all, &mvp, rect, centre, 4.0), Some(0));
    assert_eq!(pick_nearest(&pts, &[false, true, true], &mvp, rect, centre, 4.0), None);
}

fn count_hyp_highlight(state: &mut HyperbolicPanelState) -> (usize, usize, usize) {
    let (sc, mesh, _) = state.build_scenes();
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    let polys = primitives.polygons.iter().filter(|p| p.fill == HYP_PICK_RGBA8).count();
    let rings = primitives.circles.iter().filter(|c| c.fill == HYP_PICK_RGBA8).count();
    let verts = mesh.faces.iter().filter(|v| v.color[..3] == HYP_PICK_RGBA[..3]).count();
    (polys, rings, verts)
}

#[test]
fn picked_polygon_is_highlighted_in_the_disk_and_on_the_sheet() {
    let mut h = HyperbolicPanelState::default(); // {7,3}, 2 layers, Poincare
    assert_eq!(count_hyp_highlight(&mut h), (0, 0, 0));
    // the disk centre lies in the central heptagon, away from its vertices
    assert_eq!(h.pick_2d([0.0, 0.0], 100.0, 9.0), Some(HypPick::Face(0)));
    h.picked = Some(HypPick::Face(0));
    // one highlighted polygon in 2D; a fan of 7 triangles (21 vertices) on the sheet
    assert_eq!(count_hyp_highlight(&mut h), (1, 0, 21));
    h.picked = Some(HypPick::Vertex(h.cache.as_ref().unwrap().base.faces[0][0]));
    // one ring in 2D; an octahedron marker (8 triangles, 24 vertices) on the sheet
    assert_eq!(count_hyp_highlight(&mut h), (0, 1, 24));
    // Klein disk: the same pick at the centre
    h.disk_proj = super::hyperbolic_ui::DiskProjection::Klein;
    h.build_scenes();
    assert_eq!(h.pick_2d([0.0, 0.0], 100.0, 9.0), Some(HypPick::Face(0)));
}

#[test]
fn mobius_pan_moves_the_central_face_to_a_known_distance() {
    let mut h = HyperbolicPanelState::default();
    h.build_scenes(); // builds the tiling (a new tiling clears the pick)
    h.picked = Some(HypPick::Face(0));
    h.pan = DiskIsometry::from_origin([0.3, 0.0]);
    h.build_scenes();
    // the central heptagon's centre (the origin) is sent to 0.3: d = 2 artanh 0.3
    let z = h.picked_view_point().unwrap();
    assert!((z[0] - 0.3).abs() < 1e-9 && z[1].abs() < 1e-9, "{z:?}");
    assert!((dist_from_origin(z) - 2.0 * 0.3_f64.atanh()).abs() < 1e-9);
    // the 2D pick follows the content
    assert_eq!(h.pick_2d([0.3, 0.0], 100.0, 9.0), Some(HypPick::Face(0)));
}

fn app_harness() -> Harness<'static, PapertillingApp> {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    app.atlas.space3d.selected_net = CanonicalNet3d::SimpleCubic;
    app.atlas.space3d.selected_defect = LineDefect3d::None;
    app.atlas.sub_tab = AtlasSubTab::SpaceGroups3D;
    app.atlas.dirty = true;
    app.switch_tab(NavTab::Atlas);
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
    h
}

fn click(h: &mut Harness<'static, PapertillingApp>, pos: egui::Pos2) {
    h.event(Event::PointerMoved(pos));
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.step();
    h.run_steps(12); // > 0.5 s: the next click is not a double-click
}

#[test]
fn kittest_unit_cell_toggle_refreshes_the_scene_and_clears_dirty() {
    let mut h = app_harness();
    assert!(!h.state().atlas.dirty);
    let wires0 = h.state().view3d.mesh_data().wires.len();
    let gen0 = h.state().view3d.mesh_generation();
    h.get_by_label("Unit cell").click();
    h.run_steps(3);
    let st = h.state();
    assert!(st.atlas.space3d.view.show_cell);
    assert!(!st.atlas.dirty, "the smoke/export dirty flag clears in the same frame");
    assert!(st.view3d.mesh_generation() > gen0);
    assert_eq!(st.view3d.mesh_data().wires.len(), wires0 + 24);
    h.get_by_label("Axes").click();
    h.run_steps(3);
    assert_eq!(h.state().view3d.mesh_data().wires.len(), wires0 + 30);
    assert!(!h.state().atlas.dirty);
}

#[test]
fn kittest_click_in_the_3d_view_picks_the_site_under_two_presets() {
    let mut h = app_harness();
    // Narrow 3D split pane (~320x720, aspect ~0.45): zoom out slightly so every cube corner sits inside `ptr.rect`.
    h.state_mut().view3d.distance = 4.2;
    let net = h.state_mut().atlas.space3d.ensure_net().clone();
    let pts = net_mesh_points(&net);
    let p = &net.positions;
    // Site 0 and its partner share a column along the view axis of the preset (z for Top, y for Front); the click
    // goes to the projection of the one in front (perspective separates the two images by more than the pick
    // radius, so this checks the projection, not the depth tie-break, which the pure test above covers).
    let partner = |i: usize, axis: usize| {
        (0..p.len()).find(|&j| j != i && (0..3).all(|k| k == axis || (p[j][k] - p[i][k]).abs() < 1e-9)).unwrap()
    };
    for (preset, axis) in [(CameraPreset::Top, 2), (CameraPreset::Front, 1)] {
        h.state_mut().view3d.apply_preset(preset);
        h.run_steps(2);
        let j = partner(0, axis);
        // Top: larger z is nearer; Front (camera on -y): smaller y is nearer
        let front = if axis == 2 { if p[0][2] > p[j][2] { 0 } else { j } } else if p[0][1] < p[j][1] { 0 } else { j };
        let ptr = h.state().view3d.pointer().expect("3D view drawn");
        let mvp = h.state().view3d.mvp_for_rect(ptr.rect);
        let (pos, _) = project_point(&mvp, ptr.rect, pts[front]).unwrap();
        assert!(ptr.rect.contains(pos));
        click(&mut h, pos);
        assert_eq!(h.state().atlas.space3d.view.picked, Some(front), "{preset:?}");
        assert!(!h.state().atlas.dirty);
    }
    // Front, slab keeping only the lower z layer: a click on an upper-layer site picks nothing
    let upper = (0..p.len()).max_by(|&a, &b| p[a][2].total_cmp(&p[b][2])).unwrap();
    h.state_mut().atlas.space3d.view.z_slice_range = [0.0, 0.4];
    h.state_mut().atlas.space3d.view.changed = true;
    h.run_steps(2);
    let ptr = h.state().view3d.pointer().unwrap();
    let mvp = h.state().view3d.mvp_for_rect(ptr.rect);
    let (pos, _) = project_point(&mvp, ptr.rect, pts[upper]).unwrap();
    assert!(ptr.rect.contains(pos));
    click(&mut h, pos);
    assert_eq!(h.state().atlas.space3d.view.picked, None);
}

#[test]
fn kittest_h2_pan_button_moves_the_tiling_and_clears_dirty() {
    let mut h = app_harness();
    h.state_mut().atlas.sub_tab = AtlasSubTab::HyperbolicH2;
    h.state_mut().atlas.dirty = true;
    h.run_steps(3);
    assert!(!h.state().atlas.dirty);
    let gen0 = h.state().view3d.mesh_generation();
    h.get_by_label("→").click();
    h.run_steps(3);
    let st = h.state();
    assert_ne!(st.atlas.hyperbolic.pan, DiskIsometry::IDENTITY);
    assert!(st.view3d.mesh_generation() > gen0);
    assert!(!st.atlas.dirty);
}

#[test]
fn depth_cue_fades_far_sites_and_bonds_and_refreshes_on_rotation() {
    let mut s = simple_cubic();
    assert!(!s.needs_depth_refresh(&IDENTITY));
    s.view.depth_cue = true;
    assert!(s.needs_depth_refresh(&IDENTITY));
    let (sc, mesh, _) = s.build_scenes(&IDENTITY);
    assert!(!s.needs_depth_refresh(&IDENTITY));
    let mut front = View3dState::default();
    front.apply_preset(CameraPreset::Front);
    assert!(s.needs_depth_refresh(&front.rotation()));

    // 2x2x2 simple cubic block: mesh z is +-1/sqrt(3), so every site has alpha < 1 and sits in faded faces.
    assert_eq!(mesh.opaque_vertex_count, 0);
    let z_mesh = 1.0 / 3.0_f32.sqrt();
    let a_hi = DEPTH_CUE_MIN_ALPHA + (1.0 - DEPTH_CUE_MIN_ALPHA) * 0.5 * (1.0 + z_mesh);
    let a_lo = DEPTH_CUE_MIN_ALPHA + (1.0 - DEPTH_CUE_MIN_ALPHA) * 0.5 * (1.0 - z_mesh);
    assert!(mesh.faces.iter().all(|v| (v.color[3] - a_hi).abs() < 1e-4 || (v.color[3] - a_lo).abs() < 1e-4));
    assert!(mesh.faces.iter().any(|v| (v.color[3] - a_hi).abs() < 1e-4));
    assert!(mesh.faces.iter().any(|v| (v.color[3] - a_lo).abs() < 1e-4));
    assert!(mesh.wires.iter().any(|v| v.color[3] > 0.70) && mesh.wires.iter().any(|v| v.color[3] < 0.45));
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    assert_eq!(primitives.circles.iter().filter(|c| c.fill[3] == 255).count(), 4);
    assert_eq!(primitives.circles.iter().filter(|c| c.fill[3] == 64).count(), 4);
}

#[test]
fn h2_disk_zoom_scales_coordinates_and_preserves_2d_and_3d_picking() {
    let mut h = HyperbolicPanelState::default();
    h.disk_zoom = 2.5;
    let (sc, mesh, _) = h.build_scenes();
    let Scene2D::VectorScene { primitives } = sc else { panic!("vector scene") };
    assert_eq!(primitives.bbox, ([-1.05, -1.05], [1.05, 1.05]));
    let hyp = h.view.as_ref().unwrap().clone();
    assert!((primitives.circles[0].center[0] - (hyp.poincare_pts[0][0] * 2.5) as f32).abs() < 1e-5);
    assert_eq!(h.pick_2d([hyp.poincare_pts[0][0] * 2.5, hyp.poincare_pts[0][1] * 2.5], 100.0, 9.0), Some(HypPick::Vertex(0)));

    let f1 = &hyp.faces[1];
    let c1 = lorentz_centroid(&f1.iter().map(|&u| hyp.poincare_pts[u]).collect::<Vec<_>>());
    assert_eq!(h.pick_2d([c1[0] * 2.5, c1[1] * 2.5], 100.0, 9.0), Some(HypPick::Face(1)));
    h.disk_proj = DiskProjection::Klein;
    h.build_scenes();
    let k1 = poincare_to_klein(c1);
    assert_eq!(h.pick_2d([k1[0] * 2.5, k1[1] * 2.5], 100.0, 9.0), Some(HypPick::Face(1)));

    // 3D sheet picking: off-centroid point inside face 1 (away from centroid) and vertex 0.
    let mut v = View3dState::default();
    v.apply_preset(CameraPreset::Top);
    v.distance = 1.2;
    v.set_mesh(mesh, "h2");
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 600.0));
    let mvp = v.mvp_for_rect(rect);
    let pts = sheet_points(&hyp, v.mesh_data().center, v.mesh_data().radius);
    let c_mesh: [f32; 3] = core::array::from_fn(|k| f1.iter().map(|&u| pts[u][k]).sum::<f32>() / f1.len() as f32);
    let mid_edge: [f32; 3] = core::array::from_fn(|k| 0.5 * (pts[f1[0]][k] + pts[f1[1]][k]));
    let off_center: [f32; 3] = core::array::from_fn(|k| 0.3 * c_mesh[k] + 0.7 * mid_edge[k]);
    let (screen_pt, _) = project_point(&mvp, rect, off_center).unwrap();
    assert_eq!(h.pick_3d(&v, rect, screen_pt), Some(HypPick::Face(1)));
    let (sv, _) = project_point(&mvp, rect, pts[0]).unwrap();
    assert_eq!(h.pick_3d(&v, rect, sv), Some(HypPick::Vertex(0)));
}

#[test]
fn kittest_depth_cue_panned_3d_pick_2d_pick_and_inspector_readout() {
    let mut h = app_harness();
    h.get_by_label("Depth cue").click();
    h.run_steps(3);
    assert!(h.state().atlas.space3d.view.depth_cue);
    assert_eq!(h.state().view3d.mesh_data().opaque_vertex_count, 0);
    assert!(!h.state().atlas.dirty);

    // Changing camera preset refreshes depth cue on the next frame.
    let gen0 = h.state().view3d.mesh_generation();
    h.get_by_label("Front (XZ)").click();
    h.run_steps(3);
    assert!(h.state().view3d.mesh_generation() > gen0);

    // Panned + zoomed z-up Side preset: picking respects camera offset and updates Right Inspector readout.
    h.state_mut().view3d.apply_preset(CameraPreset::Side);
    h.state_mut().view3d.camera_offset = egui::vec2(0.08, -0.12);
    h.state_mut().view3d.distance = 3.6;
    h.run_steps(3);
    let net = h.state_mut().atlas.space3d.ensure_net().clone();
    let pts = net_mesh_points(&net);
    let front_x = net
        .positions
        .iter()
        .position(|p| p[0] > -0.5 && p[1] < -0.5 && p[2] > -0.5)
        .expect("front-left-top SC corner");
    let ptr = h.state().view3d.pointer().unwrap();
    let mvp = h.state().view3d.mvp_for_rect(ptr.rect);
    let (pos3, _) = project_point(&mvp, ptr.rect, pts[front_x]).unwrap();
    assert!(ptr.rect.contains(pos3));
    click(&mut h, pos3);
    assert_eq!(h.state().atlas.space3d.view.picked, Some(front_x));
    assert!(h.query_by_label_contains(&format!("Site #{front_x}")).is_some());
    assert!(h.query_by_label_contains("Coordination: 3 bonds").is_some());
    assert!(h.query_by_label_contains("Fractional:").is_some());

    // Clicking the same xy column in the 2D projection picks the topmost (+z) site there.
    let top_z = (0..net.positions.len())
        .filter(|&j| (net.positions[j][0] - net.positions[0][0]).abs() < 1e-9 && (net.positions[j][1] - net.positions[0][1]).abs() < 1e-9)
        .max_by(|&a, &b| net.positions[a][2].total_cmp(&net.positions[b][2]))
        .unwrap();
    let rect2 = h.state().canvas2d.last_rect;
    let pos2 = h.state().canvas2d.world_to_screen_at(rect2, [net.positions[top_z][0] as f32, net.positions[top_z][1] as f32]);
    click(&mut h, pos2);
    assert_eq!(h.state().atlas.space3d.view.picked, Some(top_z));

    // Full slab button restores all 8 sites after a narrower slice.
    h.state_mut().atlas.space3d.view.z_slice_range = [0.0, 0.4];
    h.state_mut().atlas.space3d.view.changed = true;
    h.run_steps(2);
    assert_eq!(h.state().view3d.mesh_data().vertex_count, 4);
    h.get_by_label("Full slab").click();
    h.run_steps(2);
    assert_eq!(h.state().atlas.space3d.view.z_slice_range, [0.0, 1.0]);
    assert_eq!(h.state().view3d.mesh_data().vertex_count, 8);
}

#[test]
fn kittest_h2_picking_klein_toggle_drag_pan_and_inspector_readout() {
    let mut h = app_harness();
    h.state_mut().atlas.sub_tab = AtlasSubTab::HyperbolicH2;
    h.state_mut().atlas.dirty = true;
    h.run_steps(3);

    // Click central face in 2D disk -> synced 2D/3D highlight + readout (including origin angle guard).
    let rect2 = h.state().canvas2d.last_rect;
    let origin_px = h.state().canvas2d.world_to_screen_at(rect2, [0.0, 0.0]);
    click(&mut h, origin_px);
    assert_eq!(h.state().atlas.hyperbolic.picked, Some(HypPick::Face(0)));
    assert!(!h.state().atlas.dirty);
    assert!(h.query_by_label_contains("Face #0 (7-gon)").is_some());
    assert!(h.query_by_label_contains("Schläfli {7,3}").is_some());
    assert!(h.query_by_label_contains("Layer (generation): 0").is_some());
    assert!(h.query_by_label_contains("d(0,z) = 2 artanh|z| = 0.0000").is_some());
    assert!(h.query_by_label_contains("Δφ = — (at view origin)").is_some());

    // Toggle Klein Chords: preserves pick and clears dirty.
    h.get_by_label("Klein Chords").click();
    h.run_steps(3);
    assert_eq!(h.state().atlas.hyperbolic.disk_proj, DiskProjection::Klein);
    assert_eq!(h.state().atlas.hyperbolic.picked, Some(HypPick::Face(0)));
    assert!(!h.state().atlas.dirty);

    // Click vertex 1 on the 3D hyperboloid sheet -> non-zero d(0,z) & Δφ readout, and Center on pick moves it to origin.
    h.state_mut().view3d.apply_preset(CameraPreset::Top);
    h.run_steps(2);
    let hyp = h.state().atlas.hyperbolic.view.as_ref().unwrap().clone();
    let ptr = h.state().view3d.pointer().unwrap();
    let mvp = h.state().view3d.mvp_for_rect(ptr.rect);
    let pts3 = sheet_points(&hyp, h.state().view3d.mesh_data().center, h.state().view3d.mesh_data().radius);
    let (sv, _) = project_point(&mvp, ptr.rect, pts3[1]).unwrap();
    assert!(ptr.rect.contains(sv));
    click(&mut h, sv);
    assert_eq!(h.state().atlas.hyperbolic.picked, Some(HypPick::Vertex(1)));
    assert!(h.query_by_label_contains("Vertex #1").is_some());
    let pi = core::f64::consts::PI;
    let r0 = ((pi / 7.0 + pi / 3.0).cos() / (pi / 7.0 - pi / 3.0).cos()).sqrt();
    let want_d = format!("d(0,z) = 2 artanh|z| = {:.4}", 2.0 * r0.atanh());
    let want_phi = format!("Δφ = arg z = {:+.2}°", (2.0 * pi / 7.0).to_degrees());
    assert!(h.query_by_label_contains(&want_d).is_some(), "missing {want_d}");
    assert!(h.query_by_label_contains(&want_phi).is_some(), "missing {want_phi}");
    h.get_by_label("Center on pick").click();
    h.run_steps(3);
    let centered = h.state().atlas.hyperbolic.picked_view_point().unwrap();
    assert!(centered[0].hypot(centered[1]) < 1e-6, "{centered:?}");

    // Drag across the 2D disk -> Mobius pan updates `hyperbolic.pan`, keeps `canvas2d.pan == ZERO`.
    let pan_before = h.state().atlas.hyperbolic.pan;
    let from = rect2.center();
    h.event(Event::PointerMoved(from));
    h.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    for k in 1..=6 {
        h.event(Event::PointerMoved(from + egui::vec2(8.0 * k as f32, 0.0)));
        h.step();
    }
    h.event(Event::PointerButton { pos: from + egui::vec2(48.0, 0.0), button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    assert_ne!(h.state().atlas.hyperbolic.pan, pan_before);
    assert_eq!(h.state().canvas2d.pan, egui::Vec2::ZERO);
    assert!(!h.state().atlas.dirty);
}

#[test]
fn hyperbolic_kagome_cluster_faces_shells_and_picking() {
    use papertilling_core::hyperbolic::HyperbolicStyle;

    let mut h = HyperbolicPanelState { p: 7, q: 3, style: HyperbolicStyle::RectifiedKagome, ..Default::default() };
    let (_sc, mesh, _) = h.build_scenes();
    let cache = h.cache.as_ref().unwrap();
    assert!(cache.base.faces.iter().any(|f| f.len() == 7), "missing 7-gons in (7.3.7.3) kagome");
    assert!(cache.base.faces.iter().any(|f| f.len() == 3), "missing 3-gons in (7.3.7.3) kagome");
    assert_eq!(cache.face_gen, cache.base.face_shell);
    assert_eq!(h.shell_key, Some((7, 3)));
    assert_eq!(h.drawn_patch_note, cache.base.growth_rule);

    // 2D picking: central 7-gon at origin, adjacent 3-gon at face 1, and vertex 0.
    let hyp = h.view.as_ref().unwrap().clone();
    let tri_idx = hyp.faces.iter().position(|f| f.len() == 3).unwrap();
    let tri_c = lorentz_centroid(&hyp.faces[tri_idx].iter().map(|&u| hyp.poincare_pts[u]).collect::<Vec<_>>());
    assert_eq!(h.pick_2d([0.0, 0.0], 100.0, 9.0), Some(HypPick::Face(0)));
    assert_eq!(h.pick_2d(tri_c, 100.0, 4.0), Some(HypPick::Face(tri_idx)));
    assert_eq!(h.pick_2d(hyp.poincare_pts[0], 100.0, 9.0), Some(HypPick::Vertex(0)));
    h.disk_proj = DiskProjection::Klein;
    h.build_scenes();
    assert_eq!(h.pick_2d(poincare_to_klein(tri_c), 100.0, 4.0), Some(HypPick::Face(tri_idx)));

    // 3D sheet picking: central 7-gon, 3-gon, and vertex 0.
    let mut v = View3dState::default();
    v.apply_preset(CameraPreset::Top);
    v.distance = 1.2;
    v.set_mesh(mesh, "kagome");
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 600.0));
    let mvp = v.mvp_for_rect(rect);
    let pts = sheet_points(&hyp, v.mesh_data().center, v.mesh_data().radius);
    let f0 = &hyp.faces[0];
    let c0: [f32; 3] = core::array::from_fn(|k| f0.iter().map(|&u| pts[u][k]).sum::<f32>() / f0.len() as f32);
    let tri_f = &hyp.faces[tri_idx];
    let c3: [f32; 3] = core::array::from_fn(|k| tri_f.iter().map(|&u| pts[u][k]).sum::<f32>() / tri_f.len() as f32);
    let (sc0, _) = project_point(&mvp, rect, c0).unwrap();
    let (sct, _) = project_point(&mvp, rect, c3).unwrap();
    let (sv0, _) = project_point(&mvp, rect, pts[0]).unwrap();
    assert_eq!(h.pick_3d(&v, rect, sc0), Some(HypPick::Face(0)));
    assert_eq!(h.pick_3d(&v, rect, sct), Some(HypPick::Face(tri_idx)));
    assert_eq!(h.pick_3d(&v, rect, sv0), Some(HypPick::Vertex(0)));
}

#[test]
fn net_frame_matches_all_canonical_nets_and_defect_block_origin() {
    use papertilling_core::space3d::nets3d::build_net_3d;
    let mod1_dist = |a: f64, b: f64| {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d)
    };
    let check_motif = |net: CanonicalNet3d, motif: &[[f64; 3]]| {
        let patch = build_net_3d(net, LineDefect3d::None);
        let fr = net_frame(net, LineDefect3d::None);
        assert_eq!(patch.positions.len(), 8 * motif.len(), "{net:?}");
        for p in &patch.positions {
            let f = fr.fractional(*p);
            assert!(
                motif.iter().any(|m| (0..3).all(|k| mod1_dist(f[k], m[k]) < 1e-6)),
                "{net:?}: unexpected fractional {f:?}"
            );
        }
    };
    for net in CanonicalNet3d::ALL {
        match net {
            CanonicalNet3d::SimpleCubic => check_motif(net, &[[0.0, 0.0, 0.0]]),
            CanonicalNet3d::BCC => check_motif(net, &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            CanonicalNet3d::FCC => check_motif(net, &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]),
            CanonicalNet3d::Diamond => check_motif(net, &[
                [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
                [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
            ]),
            CanonicalNet3d::HCP => check_motif(net, &[[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]),
            CanonicalNet3d::Tetragonal123 => {
                assert!((net_frame(net, LineDefect3d::None).lat[2][2] - 1.15).abs() < 1e-12);
                check_motif(net, &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]);
            }
            CanonicalNet3d::Rutile136 => {
                assert!((net_frame(net, LineDefect3d::None).lat[2][2] - 0.72).abs() < 1e-12);
                check_motif(net, &[
                    [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.3, 0.3, 0.0],
                    [0.7, 0.7, 0.0], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5],
                ]);
            }
            CanonicalNet3d::Perovskite => check_motif(net, &[
                [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
            ]),
            CanonicalNet3d::Pyrochlore => check_motif(net, &[
                [0.0, 0.0, 0.0], [0.25, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.25], [0.5, 0.5, 0.5],
            ]),
            CanonicalNet3d::BetaQuartz => {
                assert!((net_frame(net, LineDefect3d::None).lat[2][2] - 1.35).abs() < 1e-12);
                check_motif(net, &[[0.5, 0.0, 0.0], [0.0, 0.5, 1.0 / 3.0], [0.5, 0.5, 2.0 / 3.0]]);
            }
        }
    }
    // Defect blocks use nx = ny = 3 (mid = 1.5), leaving xy coordinates intact for a screw dislocation along z.
    let fr = net_frame(CanonicalNet3d::Rutile136, LineDefect3d::ScrewDislocationZ);
    assert_eq!(fr.origin, [-1.5, -1.5, -0.72]);
    let screw = build_net_3d(CanonicalNet3d::Rutile136, LineDefect3d::ScrewDislocationZ);
    let xy_motif = [[0.0, 0.0], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [0.8, 0.2], [0.2, 0.8]];
    for p in &screw.positions {
        let f = fr.fractional(*p);
        assert!(xy_motif.iter().any(|m| mod1_dist(f[0], m[0]) < 1e-6 && mod1_dist(f[1], m[1]) < 1e-6), "{f:?}");
    }
}
