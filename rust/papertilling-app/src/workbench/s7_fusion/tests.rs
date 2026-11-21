//! Fusion steps view: hand-count checks of the scene builder, and the default (Advanced off) path in the UI.

use egui_kittest::kittest::Queryable;
use papertilling_core::fusion::gate_route;

use super::scene::{face_outline, step_scene, triangulate};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, settled, wait_until};
use crate::app::{AppMode, DefectsView, NavTab, PapertillingApp};
use crate::canvas2d::Scene2D;
use crate::workbench::WorkbenchView;

fn tri_area(t: &[[f64; 2]; 3]) -> f64 {
    0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])).abs()
}

fn filled_area(scene: &Scene2D) -> f64 {
    let Scene2D::VectorScene { primitives } = scene else { panic!("vector scene expected") };
    primitives
        .polygons
        .iter()
        .map(|p| {
            let q: Vec<[f64; 2]> = p.points.iter().map(|x| [f64::from(x[0]), f64::from(x[1])]).collect();
            tri_area(&[q[0], q[1], q[2]])
        })
        .sum()
}

#[test]
fn triangulation_hand_counts() {
    // Unit square: 2 triangles, area 1.
    let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let t = triangulate(&sq);
    assert_eq!(t.len(), 2);
    assert!((t.iter().map(tri_area).sum::<f64>() - 1.0).abs() < 1e-12);
    // An L of three unit squares, listed clockwise: 6 corners give 4 triangles, area 3. A fan from the
    // centroid would cover area outside the L; ear clipping must not.
    let l = [[0.0, 0.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]];
    let t = triangulate(&l);
    assert_eq!(t.len(), 4);
    assert!((t.iter().map(tri_area).sum::<f64>() - 3.0).abs() < 1e-12);
    for tri in &t {
        let c = [(tri[0][0] + tri[1][0] + tri[2][0]) / 3.0, (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0];
        assert!(!(c[0] > 1.0 && c[1] > 1.0), "triangle centre {c:?} lies in the missing square");
    }
}

#[test]
fn faces_fill_the_same_region_before_and_after_fusion() {
    let steps = gate_route().unwrap();
    // Shoelace by hand on the drawing of Fig. 8(a): the seven-sided face a b c d s g h has area 2.5 and the
    // five-sided face d e f g s has area 1.5. The fused lattice has squares of area 1, 1 and 2 on the same
    // region. Steps 0, 4 and 5 have straight edges only.
    for k in [0, 4, 5] {
        let a = filled_area(&step_scene(&steps[k], k, steps.len(), true, true));
        assert!((a - 4.0).abs() < 1e-6, "step {k}: filled area {a}");
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
fn scene_marks_odd_vertices_and_faces() {
    let steps = gate_route().unwrap();
    let labels = |k: usize| -> Vec<String> {
        let Scene2D::VectorScene { primitives } = step_scene(&steps[k], k, steps.len(), false, false) else { unreachable!() };
        let mut l: Vec<String> = primitives.labels.into_iter().map(|t| t.text).collect();
        l.sort();
        l
    };
    // Fig. 8(a): vertices with 2, 3, 3 edges and faces with 5 and 7 sides are numbered; nothing else.
    assert_eq!(labels(0), ["2", "3", "3", "5", "7"]);
    // After H(11): vertices 3, 3, 6 and faces 1, 7 (hand count in the core tests).
    assert_eq!(labels(2), ["1", "3", "3", "6", "7"]);
    // Fused lattice: all squares, all four-valent: no number at all.
    assert!(labels(5).is_empty());
    // With edge numbers on, each of the 14 edges is labelled once.
    let Scene2D::VectorScene { primitives } = step_scene(&steps[5], 5, steps.len(), true, false) else { unreachable!() };
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
fn fusion_steps_is_reachable_with_advanced_off_and_steps_through() {
    let mut h = harness();
    h.run_steps(3);
    assert!(!h.state().advanced);
    click_top(&mut h, "Defects");
    wait_until(&mut h, "Defects", settled);
    click_top(&mut h, "Fusion steps");
    wait_until(&mut h, "Fusion steps", settled);
    {
        let app = h.state();
        assert_eq!((app.active_mode, app.defects_view, app.active_tab), (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench));
        assert_eq!(app.workbench.view, WorkbenchView::S7Fusion);
        assert!(app.workbench.error().is_none());
        assert!(canvas_title(app).starts_with("Step 0 of 5"), "{}", canvas_title(app));
        assert!(!app.advanced, "the view does not need Advanced");
    }
    assert_no_forbidden_tokens(&h, "Fusion steps");
    assert!(h.query_all_by_label_contains("Workbench").next().is_none(), "the internal route name is never shown");
    // Next: the first gate.
    h.get_by_label("Next ▶").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).contains("CX(11 -> 12)"), "{}", canvas_title(h.state()));
    // Jump to the Hadamard step from the list; the inspector reports one vertex fewer and one face more.
    h.get_by_label("2. H(11)").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 2 of 5"));
    assert!(h.query_by_label("Inner vertices").is_some() && h.query_by_label("Inner faces").is_some());
    // The lattice chip and Rename belong to views that show the Gallery lattice.
    assert!(h.query_by_label("Rename").is_none(), "no Rename in a view without a Gallery lattice");
    // The other route has three lattices.
    h.get_by_label("Fusion by Hamiltonian changes").click();
    h.run_steps(3);
    assert!(canvas_title(h.state()).starts_with("Step 0 of 2"), "{}", canvas_title(h.state()));
}

#[test]
fn session_restore_clamps_and_reports() {
    use crate::workbench::WbPanel;
    let mut p = super::Panel::default();
    let mut notes = Vec::new();
    p.restore(&serde_json::json!({"route": 1, "step": [3, 2], "numbers": false}), &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((p.route, p.step, p.show_numbers), (1, [3, 2], false));
    assert_eq!(p.save()["step"], serde_json::json!([3, 2]));
    p.restore(&serde_json::json!({"route": 7, "step": [99, 0]}), &mut notes);
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert_eq!((p.route, p.step), (1, [3, 0]));
}

#[test]
fn every_step_exports_as_a_figure() {
    // "Save figure" path: each step must encode as a PNG. Set PAPERTILLING_FUSION_PNG_DIR to keep the files.
    let dir = std::env::var_os("PAPERTILLING_FUSION_PNG_DIR").map(std::path::PathBuf::from);
    for (name, steps) in [("gates", gate_route().unwrap()), ("hamiltonian", papertilling_core::fusion::hamiltonian_route().unwrap())] {
        for (k, st) in steps.iter().enumerate() {
            let png = crate::export::encode_scene_png(&step_scene(st, k, steps.len(), true, true), 900, 760).unwrap();
            assert_eq!(&png[1..4], b"PNG", "{name} step {k}");
            if let Some(d) = &dir {
                std::fs::write(d.join(format!("fusion_app_{name}_{k}.png")), &png).unwrap();
            }
        }
    }
}

#[test]
fn counts_table_reports_the_change_per_row() {
    // The note, Sec. 5: one vertex fewer, the same edges, one face more; the change happens at H(11).
    let steps = gate_route().unwrap();
    let rows = |k: usize| super::Panel::count_rows(&steps[k].map, &steps[0].map).map(|(name, n, change)| (name, n, change));
    let none = String::from("no change");
    assert_eq!(rows(1), [("Inner vertices", 5, none.clone()), ("Edges (qubits)", 14, none.clone()), ("Inner faces", 2, none.clone())]);
    for k in 2..=5 {
        assert_eq!(
            rows(k),
            [("Inner vertices", 4, String::from("-1")), ("Edges (qubits)", 14, none.clone()), ("Inner faces", 3, String::from("+1"))],
            "step {k}"
        );
    }
}

#[test]
fn advanced_switch_keeps_the_view() {
    let ctx = egui::Context::default();
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    let frame = |app: &mut PapertillingApp| {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _ = app.frame_ui(ui);
        });
    };
    app.select_defects_view(DefectsView::FusionSteps);
    for on in [true, false, true] {
        app.set_advanced(on);
        for _ in 0..4 {
            frame(&mut app);
        }
        assert_eq!((app.active_mode, app.defects_view, app.active_tab), (AppMode::Defects, DefectsView::FusionSteps, NavTab::Workbench), "Advanced = {on}");
        assert!(canvas_title(&app).starts_with("Step 0 of 5"), "Advanced = {on}: {}", canvas_title(&app));
    }
}
