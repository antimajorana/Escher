//! Fusion & core-connections view: hand-count checks of the scene builder, geometry-first default UI,
//! 4 cosmetic drawing-fix regressions, and session backward-compatibility.

use egui_kittest::kittest::Queryable;
use papertilling_core::fusion::{gate_route, hamiltonian_route, wang_core_connections};

use super::scene::{convex_partition, core_scene, face_outline, step_scene, triangulate};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, settled, wait_until};
use crate::app::{AppMode, DefectsView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;
use crate::workbench::WorkbenchView;

fn tri_area(t: &[[f64; 2]; 3]) -> f64 {
    0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])).abs()
}

fn poly_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    0.5 * (0..n)
        .map(|i| pts[i][0] * pts[(i + 1) % n][1] - pts[i][1] * pts[(i + 1) % n][0])
        .sum::<f64>()
        .abs()
}

fn filled_area(scene: &Scene2D) -> f64 {
    let Scene2D::VectorScene { primitives } = scene else { panic!("vector scene expected") };
    primitives
        .polygons
        .iter()
        .map(|p| {
            let q: Vec<[f64; 2]> = p.points.iter().map(|x| [f64::from(x[0]), f64::from(x[1])]).collect();
            poly_area(&q)
        })
        .sum()
}

#[test]
fn triangulation_hand_counts() {
    // Unit square: 2 triangles, area 1; convex partition emits 1 single polygon (no interior seam).
    let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let t = triangulate(&sq);
    assert_eq!(t.len(), 2);
    assert!((t.iter().map(tri_area).sum::<f64>() - 1.0).abs() < 1e-12);
    let cp = convex_partition(&sq);
    assert_eq!(cp.len(), 1);
    assert!((poly_area(&cp[0]) - 1.0).abs() < 1e-12);

    // An L of three unit squares, listed clockwise: 6 corners give 4 triangles (or 2 convex rectangles), area 3.
    let l = [[0.0, 0.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]];
    let t = triangulate(&l);
    assert_eq!(t.len(), 4);
    assert!((t.iter().map(tri_area).sum::<f64>() - 3.0).abs() < 1e-12);
    for tri in &t {
        let c = [(tri[0][0] + tri[1][0] + tri[2][0]) / 3.0, (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0];
        assert!(!(c[0] > 1.0 && c[1] > 1.0), "triangle centre {c:?} lies in the missing square");
    }
    let cp = convex_partition(&l);
    assert_eq!(cp.len(), 2);
    assert!((cp.iter().map(|p| poly_area(p)).sum::<f64>() - 3.0).abs() < 1e-12);
}

#[test]
fn faces_fill_the_same_region_before_and_after_fusion() {
    let steps = gate_route().unwrap();
    // Shoelace by hand on Fig. 8(a): 7-sided face has area 2.5 and 5-sided face has area 1.5; total 4.0.
    for k in [0, 4, 5] {
        let sc = step_scene(&steps[k], k, steps.len(), true, true, false);
        let a = filled_area(&sc);
        assert!((a - 4.0).abs() < 1e-6, "step {k}: filled area {a}");
    }
    // Step 5 has 3 convex squares: each must emit 1 PolyPrim (zero diagonal seams).
    let Scene2D::VectorScene { primitives } = step_scene(&steps[5], 5, steps.len(), true, true, false) else {
        unreachable!()
    };
    assert_eq!(primitives.polygons.len(), 3, "three convex squares emit 3 polygons, not 6 triangles");

    // All 5 enumerated lattice cores tile the same cavity of total area 4.0.
    let en = wang_core_connections();
    for core in &en.cores {
        let a = filled_area(&core_scene(core, en.cores.len(), true, true, false));
        assert!((a - 4.0).abs() < 1e-5, "core {}: filled area {a}", core.id);
    }

    let seven = steps[0].map.bounded_faces().into_iter().find(|f| f.sides() == 7).unwrap();
    let outline = face_outline(&steps[0].map, &seven);
    assert_eq!(outline.len(), 7);
    assert!((triangulate(&outline).iter().map(tri_area).sum::<f64>() - 2.5).abs() < 1e-9);
    // Step 1: the dangling edge 11 is walked out and back; its leaf is not a corner of the outline.
    let eight = steps[1].map.bounded_faces().into_iter().find(|f| f.sides() == 8).unwrap();
    let leaf = steps[1].map.verts[papertilling_core::fusion::wang::S].pos;
    assert!(face_outline(&steps[1].map, &eight).iter().all(|p| (p[0] - leaf[0]).hypot(p[1] - leaf[1]) > 1e-6));
}

#[test]
fn step_2_face_outline_has_no_degenerate_sliver_past_leftmost_vertex() {
    let steps = gate_route().unwrap();
    let Scene2D::VectorScene { primitives } = step_scene(&steps[2], 2, steps.len(), true, true, false) else {
        unreachable!()
    };
    let min_x = primitives
        .polygons
        .iter()
        .flat_map(|p| p.points.iter().map(|q| f64::from(q[0])))
        .fold(f64::INFINITY, f64::min);
    assert!(min_x >= -1e-6, "step 2 polygon spilled past leftmost vertex a=(0,0): min_x = {min_x}");

    // The 7-sided face at Step 2 must partition into 2 clean convex pieces (not a razor-thin sliver at a=(0,0)).
    let seven = steps[2].map.bounded_faces().into_iter().find(|f| f.sides() == 7).unwrap();
    let outline = face_outline(&steps[2].map, &seven);
    let pieces = convex_partition(&outline);
    assert!(pieces.len() <= 2, "step 2 7-sided face should partition into <= 2 convex pieces, got {}", pieces.len());
    // Every corner at vertex a=(0,0) must have interior angle >= 20 deg (prevents egui AA miter spikes).
    for poly in &pieces {
        let n = poly.len();
        for i in 0..n {
            if poly[i][0].hypot(poly[i][1]) < 1e-6 {
                let (prev, curr, next) = (poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]);
                let (ux, uy) = (prev[0] - curr[0], prev[1] - curr[1]);
                let (vx, vy) = (next[0] - curr[0], next[1] - curr[1]);
                let ang = ((ux * vx + uy * vy) / (ux.hypot(uy) * vx.hypot(vy))).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(ang >= 20.0, "acute corner {ang:.2} deg at leftmost vertex a");
            }
        }
    }
}

fn relative_luminance(rgba: [u8; 4]) -> f64 {
    let chan = |c: u8| {
        let s = f64::from(c) / 255.0;
        if s <= 0.03928 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * chan(rgba[0]) + 0.7152 * chan(rgba[1]) + 0.0722 * chan(rgba[2])
}

fn composite_over_white(rgba: [u8; 4]) -> [u8; 4] {
    let a = f64::from(rgba[3]) / 255.0;
    let mix = |c: u8| (f64::from(c) * a + 255.0 * (1.0 - a)).round() as u8;
    [mix(rgba[0]), mix(rgba[1]), mix(rgba[2]), 255]
}

#[test]
fn step_3_face_2_and_edge_11_labels_are_separated_and_high_contrast() {
    let steps = gate_route().unwrap();
    let Scene2D::VectorScene { primitives } = step_scene(&steps[3], 3, steps.len(), true, false, false) else {
        unreachable!()
    };
    let face2 = primitives.labels.iter().find(|t| t.text == "2").expect("2-sided face label");
    let edge11 = primitives.labels.iter().find(|t| t.text == "11").expect("edge 11 label");
    let edge12 = primitives.labels.iter().find(|t| t.text == "12").expect("edge 12 label");
    let d11 = f64::from(face2.pos[0] - edge11.pos[0]).hypot(f64::from(face2.pos[1] - edge11.pos[1]));
    let d12 = f64::from(face2.pos[0] - edge12.pos[0]).hypot(f64::from(face2.pos[1] - edge12.pos[1]));
    assert!(d11 >= 0.18, "face '2' and edge '11' labels overlap: dist = {d11:.3}");
    assert!(d12 >= 0.18, "face '2' and edge '12' labels overlap: dist = {d12:.3}");
    // Highlighted edge numbers must use high-contrast dark ink (relative luminance < 0.15) against tinted faces.
    let face2_bg = composite_over_white(crate::scene2d::ring_tint_rgba(2));
    let l_face = relative_luminance(face2_bg);
    for (name, col) in [("edge 11", edge11.color), ("edge 12", edge12.color)] {
        let l_ink = relative_luminance(col);
        assert!(l_ink < 0.15, "{name} label luminance {l_ink:.4} is not dark ink (< 0.15): {col:?}");
        let contrast = (l_face.max(l_ink) + 0.05) / (l_face.min(l_ink) + 0.05);
        assert!(contrast >= 4.5, "{name} WCAG contrast {contrast:.2} against 2-gon tint < 4.5");
    }
    // Step 3's 6-gon face label "6" must clear every Complete vertex disc (radius 0.13) with room to spare.
    let face6 = primitives.labels.iter().find(|t| t.text == "6" && t.size_px > 14.0).expect("6-sided face label");
    for v in steps[3].map.verts.iter().filter(|v| v.kind == papertilling_core::fusion::VertexKind::Complete) {
        let dv = (f64::from(face6.pos[0]) - v.pos[0]).hypot(f64::from(face6.pos[1]) - v.pos[1]);
        assert!(dv >= 0.25, "face '6' label at {:?} collides with vertex {} at {:?}: dist = {dv:.3}", face6.pos, v.name, v.pos);
    }
}

#[test]
fn scene_marks_odd_vertices_and_faces() {
    let steps = gate_route().unwrap();
    let labels = |k: usize| -> Vec<String> {
        let Scene2D::VectorScene { primitives } = step_scene(&steps[k], k, steps.len(), false, false, false) else {
            unreachable!()
        };
        let mut l: Vec<String> = primitives.labels.into_iter().map(|t| t.text).collect();
        l.sort();
        l
    };
    // Fig. 8(a): vertices with 2, 3, 3 edges and faces with 5 and 7 sides are numbered; nothing else.
    assert_eq!(labels(0), ["2", "3", "3", "5", "7"]);
    // After leaf -> loop: vertices 3, 3, 6 and faces 1, 7.
    assert_eq!(labels(2), ["1", "3", "3", "6", "7"]);
    // Fused lattice: all squares, all four-valent: no number at all.
    assert!(labels(5).is_empty());
    // With edge numbers on, each of the 14 edges is labelled once.
    let Scene2D::VectorScene { primitives } = step_scene(&steps[5], 5, steps.len(), true, false, false) else {
        unreachable!()
    };
    assert_eq!(primitives.labels.len(), 14);
}

fn click_top(h: &mut egui_kittest::Harness<'static, PapertillingApp>, label: &str) {
    let top = h.state().test_rects.top.expand(1.0);
    h.query_all_by_label(label)
        .find(|n| top.contains_rect(n.rect()))
        .unwrap_or_else(|| panic!("missing '{label}' in top bar"))
        .click();
    h.run_steps(2);
}

fn canvas_title(app: &PapertillingApp) -> String {
    match app.canvas2d.scene() {
        Scene2D::VectorScene { primitives } => primitives.title.clone(),
        _ => String::new(),
    }
}

#[test]
fn fusion_view_defaults_to_geometry_first_core_connections_and_optional_gate_labels() {
    let mut h = harness();
    h.run_steps(3);
    assert!(!h.state().advanced);
    click_top(&mut h, "Defects");
    wait_until(&mut h, "Defects", settled);
    click_top(&mut h, "Fusion steps");
    wait_until(&mut h, "Fusion steps", settled);
    {
        let app = h.state();
        assert_eq!(
            (app.active_mode, app.defects_view, app.active_tab),
            (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench)
        );
        assert_eq!(app.workbench.view, WorkbenchView::S7Fusion);
        assert!(app.workbench.error().is_none());
        assert!(canvas_title(app).starts_with("Core 1 of 5"), "{}", canvas_title(app));
        assert!(!app.advanced, "the view does not need Advanced");
    }
    assert_no_forbidden_tokens(&h, "Fusion core connections");
    assert!(h.query_all_by_label_contains("Workbench").next().is_none(), "internal route name never shown");
    assert!(h.query_all_by_label_contains("Group cohomology").next().is_none());
    // Default has show_gate_labels == false: no CX, H(, X/Z or X3 tokens anywhere.
    for tok in ["CX", "X/Z", "H(", "X3"] {
        assert!(h.query_all_by_label_contains(tok).next().is_none(), "'{tok}' must be hidden by default");
    }
    assert!(h.query_by_label("Rename").is_none(), "no Rename in a view without a Gallery lattice");
    // Explicit labelled vs D_2 symmetry-orbit explanation and slide-reachability evidence must be visible.
    assert!(h.query_all_by_label_contains("2 symmetry orbits").next().is_some());
    assert!(h.query_all_by_label_contains("orbit #1 (size 1)").next().is_some());
    assert!(h.query_all_by_label_contains("orbit #2 (size 4)").next().is_some());
    assert!(h.query_all_by_label_contains("Slide-move BFS verified").next().is_some());
    assert!(h.query_all_by_label_contains("E = 13").next().is_some());

    // Select all 5 enumerated lattice cores in the UI.
    for (idx, snippet) in ["2. c–g, d–g", "3. c–g, c–h", "4. c–h, d–h", "5. d–g, d–h", "1. c–h, d–g"].iter().enumerate() {
        h.query_all_by_label_contains(snippet)
            .next()
            .unwrap_or_else(|| panic!("missing core button containing '{snippet}'"))
            .click();
        h.run_steps(3);
        let want_num = if idx == 4 { 1 } else { idx + 2 };
        assert!(canvas_title(h.state()).starts_with(&format!("Core {want_num} of 5")), "{}", canvas_title(h.state()));
        assert_no_forbidden_tokens(&h, snippet);
    }

    // Switch to Route 1 (edge slides): plain geometric labels by default, NO CX / X/Z / X3.
    h.get_by_label("Route 1: edge slides (5 steps)").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 0 of 5"), "{}", canvas_title(h.state()));
    assert!(h.query_by_label("1. Slide edge 12 along 11").is_some());
    for tok in ["CX", "X/Z", "H(", "X3"] {
        assert!(h.query_all_by_label_contains(tok).next().is_none(), "'{tok}' must be hidden by default on Route 1");
    }
    h.get_by_label("Next ▶").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 1 of 5"), "{}", canvas_title(h.state()));
    assert!(!canvas_title(h.state()).contains("CX("), "{}", canvas_title(h.state()));

    // Toggle optional toric-code / gate annotations ON: CX( and X3 now appear.
    h.get_by_label("Show toric-code / gate annotations").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).contains("CX(11 -> 12)"), "{}", canvas_title(h.state()));
    assert!(h.query_all_by_label_contains("X3").next().is_some(), "X3 should appear when gate annotations are enabled");
    h.get_by_label("2. H(11)").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 2 of 5"));
    assert!(h.query_by_label("Inner vertices").is_some() && h.query_by_label("Inner faces").is_some());

    // Switch to Route 2 (merge & add): three lattices (step 0 of 2).
    h.get_by_label("Route 2: merge & add (2 steps)").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 0 of 2"), "{}", canvas_title(h.state()));
}

#[test]
fn session_restore_clamps_and_reports_and_handles_legacy_json() {
    use crate::workbench::WbPanel;
    let mut p = super::Panel::default();
    assert!(!p.show_gate_labels, "show_gate_labels must default to false");
    let mut notes = Vec::new();
    // Legacy 2-element step JSON without `gate_labels` remains backward-compatible.
    p.restore(&serde_json::json!({"route": 1, "step": [3, 2], "numbers": false}), &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    // The old route 1 was the Hamiltonian route, now view 2, shown at its saved step 2.
    assert_eq!((p.route, p.step, p.show_numbers, p.show_gate_labels), (2, [0, 3, 2], false, false));

    // New 3-element step JSON with `gate_labels`.
    p.restore(&serde_json::json!({"route": 2, "step": [2, 4, 1], "gate_labels": true}), &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((p.route, p.step, p.show_gate_labels), (2, [2, 4, 1], true));
    assert_eq!(p.save()["step"], serde_json::json!([2, 4, 1]));
    assert_eq!(p.save()["gate_labels"], serde_json::json!(true));

    // Out-of-range values are clamped/kept and reported.
    p.restore(&serde_json::json!({"route": 7, "step": [99, 0, 0]}), &mut notes);
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert_eq!((p.route, p.step), (2, [2, 0, 0]));
}

#[test]
fn every_step_and_core_exports_as_a_figure() {
    let dir = std::env::var_os("PAPERTILLING_FUSION_PNG_DIR").map(std::path::PathBuf::from);
    for (name, steps) in [("gates", gate_route().unwrap()), ("hamiltonian", hamiltonian_route().unwrap())] {
        for (k, st) in steps.iter().enumerate() {
            let png = crate::export::encode_scene_png(&step_scene(st, k, steps.len(), true, true, false), 900, 760).unwrap();
            assert_eq!(&png[1..4], b"PNG", "{name} step {k}");
            if let Some(d) = &dir {
                std::fs::write(d.join(format!("fusion_app_{name}_{k}.png")), &png).unwrap();
            }
        }
    }
    let en = wang_core_connections();
    for core in &en.cores {
        let png = crate::export::encode_scene_png(&core_scene(core, en.cores.len(), true, true, false), 900, 760).unwrap();
        assert_eq!(&png[1..4], b"PNG", "core {}", core.id);
    }
}

#[test]
fn counts_table_reports_the_change_per_row() {
    let steps = gate_route().unwrap();
    let rows = |k: usize, gates: bool| {
        super::Panel::count_rows(&steps[k].map, &steps[0].map, gates).map(|(name, n, change)| (name, n, change))
    };
    let none = String::from("no change");
    assert_eq!(rows(1, false), [("Inner vertices", 5, none.clone()), ("Edges", 14, none.clone()), ("Inner faces", 2, none.clone())]);
    assert_eq!(
        rows(1, true),
        [("Inner vertices", 5, none.clone()), ("Edges (qubits)", 14, none.clone()), ("Inner faces", 2, none.clone())]
    );
    for k in 2..=5 {
        assert_eq!(
            rows(k, false),
            [("Inner vertices", 4, String::from("-1")), ("Edges", 14, none.clone()), ("Inner faces", 3, String::from("+1"))],
            "step {k}"
        );
    }
}

#[test]
fn fusion_view_reachable_and_kept_for_advanced_off_and_on() {
    use crate::analysis_mode::tests::equip_keys;
    use crate::game_menu::Realm;

    for on in [false, true] {
        let mut h = harness();
        h.run_steps(2);
        h.state_mut().set_advanced(on);
        h.run_steps(2);
        if on {
            equip_keys(&mut h, Realm::Forge, "Core connections");
            wait_until(&mut h, "Fusion steps (Advanced=on)", settled);
        } else {
            click_top(&mut h, "Defects");
            wait_until(&mut h, "Defects", settled);
            click_top(&mut h, "Fusion steps");
            wait_until(&mut h, "Fusion steps", settled);
        }
        let app = h.state();
        assert_eq!(app.advanced, on);
        assert_eq!(
            (app.active_mode, app.defects_view, app.active_tab),
            (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench),
            "Advanced = {on}"
        );
        assert_eq!(app.workbench.view, WorkbenchView::S7Fusion);
        assert!(canvas_title(app).starts_with("Core 1 of 5"), "Advanced = {on}: {}", canvas_title(app));
        assert_no_forbidden_tokens(&h, if on { "Fusion Advanced=on" } else { "Fusion Advanced=off" });

        // Toggling Advanced back and forth must preserve FusionSteps without snapping away.
        h.state_mut().set_advanced(!on);
        h.run_steps(3);
        let app = h.state();
        assert_eq!(
            (app.active_mode, app.defects_view, app.active_tab),
            (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench),
            "after toggling Advanced from {on} to {}",
            !on
        );
        assert!(canvas_title(app).starts_with("Core 1 of 5"), "{}", canvas_title(app));
    }
}

#[test]
fn live_adjacent_corner_f90_v2m90_cavity_loads_enumerates_and_switches_back() {
    use papertilling_core::defect::surgery::DefectSpec;
    use papertilling_core::types::{Bond2D, Cell2D, Frame2D};

    let cell = Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![
            Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
        ],
    };
    let req = super::CavityRequest {
        group: String::from("p4m"),
        cell,
        frame: Frame2D::default(),
        lattice_name: String::from("Square"),
        specs: vec![
            DefectSpec::disclination(90.0, [0.5, 0.5]),
            DefectSpec::disclination(-180.0, [0.0, 0.0]),
        ],
        existing_crystal: None,
    };

    let mut h = harness();
    h.run_steps(2);
    {
        let app = h.state_mut();
        app.workbench.load_fusion_cavity(req);
        app.select_defects_view(DefectsView::FusionSteps);
    }
    h.run_steps(4);

    {
        let app = h.state();
        assert_eq!(
            (app.active_mode, app.defects_view, app.active_tab),
            (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench)
        );
        assert!(app.workbench.error().is_none(), "error: {:?}", app.workbench.error());
        assert!(canvas_title(app).starts_with("Reference cavity:"), "{}", canvas_title(app));
        let live = app.workbench.live_fusion_cavity().expect("live cavity must be loaded");
        let ext = live.extraction.as_ref().expect("extraction must succeed");
        let en = live.enumeration.as_ref().expect("enumeration must succeed");
        assert!(ext.algebraic_charge.contains("1b") && ext.algebraic_charge.contains("-90"), "{}", ext.algebraic_charge);
        assert!(ext.far_field_source.contains("far-field single defect"), "{}", ext.far_field_source);
        assert_eq!(ext.cavity_cycle.len(), 15);
        assert_eq!(en.cavity_symmetry_order, 10);
        assert!(!en.cores.is_empty());
        assert!(en.symmetry_orbit_count >= 1);
        for core in &en.cores {
            assert_eq!(core.map.euler(), 2);
            assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
            assert!(core.census.face_sides.iter().all(|&s| s >= 3));
        }
    }

    assert_no_forbidden_tokens(&h, "Live F+90 V2-90 cavity");
    assert!(h.query_all_by_label_contains("Group cohomology").next().is_none());
    assert!(h.query_all_by_label_contains("labelled core(s)").next().is_some());
    assert!(h.query_all_by_label_contains("symmetry orbit(s)").next().is_some());
    assert!(h.query_all_by_label_contains("orbit #1").next().is_some());

    // Switch back to the Wang Fig. 8(a) worked example via the button.
    h.get_by_label("Switch to Wang Fig. 8(a) example").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Core 1 of 5"), "{}", canvas_title(h.state()));
}

#[test]
fn combine_inspect_core_reconnections_button_opens_live_fusion_view() {
    let mut h = harness();
    h.run_steps(2);
    click_top(&mut h, "Defects");
    wait_until(&mut h, "Defects", settled);
    click_top(&mut h, "Place & fuse");
    wait_until(&mut h, "Place & fuse", settled);
    assert!(!h.state().combine.specs.is_empty(), "Combine opens with a non-empty preset");
    h.get_by_label("Inspect core reconnections").click();
    h.run_steps(4);
    let app = h.state();
    assert_eq!(
        (app.active_mode, app.defects_view, app.active_tab),
        (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench)
    );
    assert!(app.workbench.live_fusion_cavity().is_some(), "button must load live cavity into s7_fusion");
    assert!(canvas_title(app).starts_with("Reference cavity:"), "{}", canvas_title(app));
    assert_no_forbidden_tokens(&h, "Combine -> Inspect core reconnections");
}
