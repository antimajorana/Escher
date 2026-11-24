//! Headless egui tests of `Canvas2D::show_interactive`: real pointer events (press, move, release, wheel)
//! through `egui::Context::run_ui`, no window.

use super::*;
use papertilling_core::types::Cell2D;

const SCREEN: egui::Vec2 = egui::vec2(800.0, 600.0);

struct Harness {
    ctx: egui::Context,
    time: f64,
    canvas: Canvas2D,
    rect: egui::Rect,
    fit_zoom: f32,
}

fn square_scene() -> Scene2D {
    Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 3,
        show_unit_cell: false,
    }
}

fn pin(index: usize, pos: [f64; 2]) -> CanvasMarker {
    CanvasMarker { index, pos, kind: MarkerKind::DisclinationPlus, label: String::from("D(+90°)") }
}

fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE }
}

fn wheel(dy: f32) -> egui::Event {
    egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, dy),
        phase: egui::TouchPhase::Move,
        source: egui::MouseWheelSource::Wheel,
        modifiers: egui::Modifiers::NONE,
    }
}

impl Harness {
    /// After the first frame (auto-fit), the view is set to 50 points per world unit, so world points within
    /// about +-6 of the scene centre are on the 800 x 600 canvas.
    fn new() -> Self {
        let mut h = Self { ctx: egui::Context::default(), time: 0.0, canvas: Canvas2D::new(square_scene()), rect: egui::Rect::NOTHING, fit_zoom: 0.0 };
        h.frame(Vec::new(), &[], false);
        h.fit_zoom = h.canvas.zoom;
        h.canvas.zoom = 50.0;
        h
    }

    fn frame(&mut self, events: Vec<egui::Event>, markers: &[CanvasMarker], click_to_place: bool) -> CanvasInteraction {
        self.time += 1.0 / 60.0;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
            time: Some(self.time),
            events,
            ..Default::default()
        };
        let mut out = CanvasInteraction::default();
        let (canvas, rect) = (&mut self.canvas, &mut self.rect);
        let _ = self.ctx.run_ui(input, |ui| {
            *rect = ui.available_rect_before_wrap();
            out = canvas.show_interactive(ui, markers, click_to_place, None, &ThemeTokens::paper());
        });
        out
    }

    fn screen(&self, w: [f64; 2]) -> egui::Pos2 {
        self.canvas.world_to_screen_at(self.rect, [w[0] as f32, w[1] as f32])
    }

    fn world(&self, s: egui::Pos2) -> [f64; 2] {
        self.canvas.screen_to_world_at(self.rect, s)
    }

    /// After a 1 s pause (so consecutive gestures never count as a double click): press at `from`, move in
    /// `steps` frames to `to`, release; returns every frame's output.
    fn drag(&mut self, from: egui::Pos2, to: egui::Pos2, steps: usize, markers: &[CanvasMarker], place: bool) -> Vec<CanvasInteraction> {
        self.time += 1.0;
        let mut outs = vec![self.frame(vec![egui::Event::PointerMoved(from)], markers, place)];
        outs.push(self.frame(vec![press(from, true)], markers, place));
        for k in 1..=steps {
            let p = from + (to - from) * (k as f32 / steps as f32);
            outs.push(self.frame(vec![egui::Event::PointerMoved(p)], markers, place));
        }
        outs.push(self.frame(vec![press(to, false)], markers, place));
        outs
    }

    fn click(&mut self, at: egui::Pos2, markers: &[CanvasMarker], place: bool) -> Vec<CanvasInteraction> {
        self.drag(at, at, 0, markers, place)
    }
}

fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) < tol
}

#[test]
fn click_on_empty_places_click_on_pin_selects() {
    let mut h = Harness::new();
    let markers = [pin(0, [1.0, 1.0])];
    let empty = h.screen([2.0, 0.0]);
    let outs = h.click(empty, &markers, true);
    let placed: Vec<[f64; 2]> = outs.iter().filter_map(|o| o.clicked_world).collect();
    assert_eq!(placed.len(), 1, "one click on empty canvas places exactly once");
    assert!(close(placed[0], [2.0, 0.0], 1e-3));
    assert!(outs.iter().all(|o| o.clicked_marker.is_none() && o.dragged_marker.is_none()));

    // a click 5 pt from the pin centre (inside the 14 pt pick radius) selects it and never places a stacked defect
    let on_pin = h.screen([1.0, 1.0]) + egui::vec2(3.0, -4.0);
    let outs = h.click(on_pin, &markers, true);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()));
    assert_eq!(outs.iter().filter_map(|o| o.clicked_marker).collect::<Vec<_>>(), vec![0]);

    // pan mode: a click on empty canvas places nothing
    let outs = h.click(h.screen([2.5, 0.5]), &markers, false);
    assert!(outs.iter().all(|o| o.clicked_world.is_none() && o.clicked_marker.is_none()));
}

#[test]
fn drag_from_pin_moves_it_and_does_not_pan() {
    let mut h = Harness::new();
    let markers = [pin(0, [0.0, 0.0]), pin(1, [2.0, 2.0])];
    let pan0 = h.canvas.pan;
    let (from, to) = (h.screen([2.0, 2.0]), h.screen([2.0, 2.0]) + egui::vec2(60.0, 30.0));
    let target = h.world(to);
    let outs = h.drag(from, to, 6, &markers, true);
    let moves: Vec<(usize, [f64; 2])> = outs.iter().filter_map(|o| o.dragged_marker).collect();
    assert!(!moves.is_empty(), "a drag that starts on a pin reports pin moves");
    assert!(moves.iter().all(|&(i, _)| i == 1));
    assert!(close(moves.last().unwrap().1, target, 1e-3));
    assert_eq!(h.canvas.pan, pan0, "dragging a pin must not pan the view");
    assert!(outs.last().unwrap().drag_released);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()), "a drag is not a click");
    assert_eq!(h.canvas.active_drag_pin, None);
}

#[test]
fn drag_on_empty_canvas_pans() {
    let mut h = Harness::new();
    let markers = [pin(0, [0.0, 0.0])];
    let pan0 = h.canvas.pan;
    let from = h.screen([2.5, 2.5]);
    let outs = h.drag(from, from + egui::vec2(-50.0, 20.0), 5, &markers, true);
    assert!(outs.iter().all(|o| o.dragged_marker.is_none() && !o.drag_released && o.clicked_world.is_none()));
    let d = h.canvas.pan - pan0;
    assert!((d.x + 50.0).abs() < 1.0 && (d.y - 20.0).abs() < 1.0, "pan moved by {d:?}");
}

#[test]
fn tiny_jiggle_on_pin_is_a_click_not_a_drag() {
    let mut h = Harness::new();
    let markers = [pin(0, [1.0, 0.0])];
    let p = h.screen([1.0, 0.0]);
    let outs = h.drag(p, p + egui::vec2(2.0, 1.0), 2, &markers, true);
    assert!(outs.iter().all(|o| o.dragged_marker.is_none()));
    assert_eq!(outs.iter().filter_map(|o| o.clicked_marker).count(), 1);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()));
}

#[test]
fn scroll_zoom_during_pin_drag_keeps_pin_under_pointer() {
    let mut h = Harness::new();
    let markers = [pin(0, [1.0, 1.0])];
    let from = h.screen([1.0, 1.0]);
    let to = from + egui::vec2(40.0, 0.0);
    h.frame(vec![egui::Event::PointerMoved(from)], &markers, false);
    h.frame(vec![press(from, true)], &markers, false);
    let mut last = None;
    for k in 1..=4 {
        let o = h.frame(vec![egui::Event::PointerMoved(from + (to - from) * (k as f32 / 4.0))], &markers, false);
        last = o.dragged_marker.or(last);
    }
    let (idx, before) = last.expect("pin drag started");
    assert_eq!(idx, 0);
    let zoom0 = h.canvas.zoom;
    let mut seen = Vec::new();
    for k in 0..12 {
        let ev = if k == 0 { vec![wheel(240.0)] } else { Vec::new() };
        if let Some((_, w)) = h.frame(ev, &markers, false).dragged_marker {
            seen.push(w);
        }
    }
    assert!(h.canvas.zoom > zoom0 * 1.2, "wheel zoomed in ({zoom0} -> {})", h.canvas.zoom);
    assert!(!seen.is_empty());
    for w in &seen {
        assert!(close(*w, before, 2e-3), "pin under the pointer moved from {before:?} to {w:?} while zooming");
    }
    let end = h.frame(vec![press(to, false)], &markers, false);
    assert!(end.drag_released);
}

#[test]
fn double_click_fits_the_view() {
    let mut h = Harness::new();
    let p = h.screen([2.5, 2.5]);
    h.drag(p, p + egui::vec2(80.0, 0.0), 4, &[], false);
    assert_ne!(h.canvas.pan, egui::Vec2::ZERO);
    assert_ne!(h.canvas.zoom, h.fit_zoom);
    let q = h.screen([2.0, 2.0]);
    let mut outs = h.click(q, &[], false);
    for pressed in [true, false] {
        outs.push(h.frame(vec![press(q, pressed)], &[], false)); // second click within 0.3 s
    }
    h.frame(Vec::new(), &[], false); // the fit is applied on the next frame
    assert_eq!(h.canvas.pan, egui::Vec2::ZERO);
    assert!((h.canvas.zoom - h.fit_zoom).abs() < 1e-3);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()));
}

#[test]
fn cursor_anchored_pinch_and_zoom_at_invariance_and_keyboard_shortcuts() {
    let mut h = Harness::new();
    let cursor = egui::pos2(610.0, 145.0);
    let world_before = h.canvas.screen_to_world_at(h.rect, cursor);

    // Direct zoom_at invariance across wide zoom factors (up to 4000.0 and down to 2.0).
    for factor in [1.8_f32, 4.5, 0.35, 12.0, 0.08] {
        h.canvas.zoom_at(h.rect, factor, Some(cursor));
        let world_after = h.canvas.screen_to_world_at(h.rect, cursor);
        assert!(close(world_before, world_after, 1e-3), "factor {factor}: {world_before:?} vs {world_after:?}");
    }

    // Trackpad pinch / Ctrl+wheel (`egui::Event::Zoom`) zooms around the pointer position.
    h.canvas.zoom = 60.0;
    h.canvas.pan = egui::vec2(-18.0, 12.0);
    let w0 = h.canvas.screen_to_world_at(h.rect, cursor);
    h.frame(vec![egui::Event::PointerMoved(cursor), egui::Event::Zoom(2.0)], &[], false);
    assert!((h.canvas.zoom - 120.0).abs() < 1e-3, "pinch zoom doubled: {}", h.canvas.zoom);
    let w1 = h.canvas.screen_to_world_at(h.rect, cursor);
    assert!(close(w0, w1, 1e-3), "pinch zoom kept cursor world point fixed: {w0:?} vs {w1:?}");

    // Keyboard `+`, `-`, and `0` shortcuts while pointer is inside canvas.
    let z_before = h.canvas.zoom;
    h.frame(vec![egui::Event::PointerMoved(cursor), egui::Event::Text(String::from("+"))], &[], false);
    assert!(h.canvas.zoom > z_before);
    let z_plus = h.canvas.zoom;
    h.frame(vec![egui::Event::PointerMoved(cursor), egui::Event::Text(String::from("-"))], &[], false);
    assert!(h.canvas.zoom < z_plus);
    h.frame(vec![egui::Event::PointerMoved(cursor), egui::Event::Text(String::from("0"))], &[], false);
    h.frame(Vec::new(), &[], false);
    assert_eq!(h.canvas.pan, egui::Vec2::ZERO);
    assert!((h.canvas.zoom - h.fit_zoom).abs() < 1e-3);

    // Clicking inside the bottom-right Zoom HUD never places a defect pin even when click_to_place = true.
    let hud_pt = h.rect.right_bottom() - egui::vec2(24.0, 16.0);
    let outs = h.click(hud_pt, &[], true);
    assert!(outs.iter().all(|o| o.clicked_world.is_none()), "HUD click must not place a defect pin");
}

#[test]
fn pin_badge_text_and_core_ring_labels_keep_contrast_in_paper_and_studio() {
    use crate::scene2d::{PolyPrim, TextPrim};
    use crate::theme::StudioTheme;

    for theme in StudioTheme::ALL {
        let tokens = theme.tokens();
        for kind in [MarkerKind::DisclinationPlus, MarkerKind::DisclinationMinus, MarkerKind::Dislocation] {
            let (fill, _) = marker_colors(kind, &tokens);
            let ratio = ThemeTokens::contrast_wcag_ratio(tokens.text_on_accent, fill);
            assert!(
                ratio >= 4.5,
                "{theme:?} {kind:?} pin text_on_accent contrast {ratio:.2} < 4.5"
            );
        }
    }

    let studio = ThemeTokens::studio();
    let poly_scene = Scene2D::VectorScene {
        primitives: crate::scene2d::ScenePrimitives {
            polygons: vec![PolyPrim {
                points: vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]],
                fill: [244, 196, 68, 145],
                stroke: None,
            }],
            lines: Vec::new(),
            circles: Vec::new(),
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: String::new(),
            note: String::new(),
            bbox: ([0.0, 0.0], [2.0, 2.0]),
            site_count: 0,
            bond_count: 0,
        },
    };
    let inside_lbl = TextPrim {
        pos: [1.0, 0.8],
        text: String::from("3"),
        color: [42, 48, 58, 225],
        size_px: 10.5,
    };
    let outside_lbl = TextPrim {
        pos: [3.0, 0.8],
        text: String::from("a1"),
        color: [24, 72, 145, 245],
        size_px: 12.0,
    };
    let polys = match &poly_scene {
        Scene2D::VectorScene { primitives } => &primitives.polygons,
        _ => unreachable!(),
    };
    // a label over a filled polygon of a non-crystal scene (e.g. a surface-net gluing label) keeps its colour code
    assert_eq!(
        render::scene_label_color(&poly_scene, polys, &inside_lbl, &studio),
        to_color32(inside_lbl.color),
        "label inside a filled polygon keeps its compiled colour outside CrystalPatch"
    );
    let paper = ThemeTokens::paper();
    assert_eq!(
        render::scene_label_color(&poly_scene, polys, &inside_lbl, &paper),
        to_color32(inside_lbl.color),
        "Paper keeps the compiled ink (as exported)"
    );
    assert_eq!(
        render::scene_label_color(&poly_scene, polys, &outside_lbl, &studio),
        studio.remap_scene_ink(outside_lbl.color),
        "label outside filled polygon must still lift on dark Studio canvas"
    );
}

mod zoom_routing;

#[test]
fn studio_ring_numbers_meet_wcag_aa_on_every_ring_tint() {
    use crate::scene2d::{PolyPrim, TextPrim};
    use crate::theme::ThemeTokens;
    use crate::theme::contrast::{contrast_ratio, over};
    let studio = ThemeTokens::studio();
    for sides in 3..=12 {
        let fill = crate::scene2d::ring_tint_rgba(sides);
        let ring = PolyPrim { points: vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], fill, stroke: None };
        let lbl = TextPrim { pos: [1.0, 0.8], text: sides.to_string(), color: [42, 48, 58, 225], size_px: 10.5 };
        let scene = crystal_scene();
        // a translucent outer face drawn first, then the ring face: the label sits on both
        let outer = PolyPrim { points: vec![[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0]], fill: [235, 240, 246, 85], stroke: None };
        let ink = render::scene_label_color(&scene, &[outer.clone(), ring.clone()], &lbl, &studio);
        let under_outer = over(outer.fill, studio.bg_canvas_2d);
        let bg = over(fill, egui::Color32::from_rgb(under_outer[0], under_outer[1], under_outer[2]));
        let r = contrast_ratio([ink.r(), ink.g(), ink.b()], bg);
        // the previous dark ink [42, 48, 58] reached only about 2.0-2.8:1 here
        assert!(r >= 4.5, "{sides}-ring label contrast {r:.2} < 4.5 (WCAG AA, small text)");
    }
}

fn crystal_scene() -> Scene2D {
    Scene2D::CrystalPatch { crystal: crate::combine_worker::empty_crystal(), title: String::new(), note: String::new() }
}

#[test]
fn studio_ring_ink_applies_to_crystal_patches_only_and_paper_is_unchanged() {
    use crate::scene2d::{PolyPrim, TextPrim};
    use crate::theme::ThemeTokens;
    let ring = PolyPrim { points: vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], fill: crate::scene2d::ring_tint_rgba(4), stroke: None };
    let lbl = TextPrim { pos: [1.0, 0.8], text: String::from("4"), color: [42, 48, 58, 225], size_px: 10.5 };
    let polys = std::slice::from_ref(&ring);
    // blue 4-ring on the dark canvas: white ink (dark ink would be about 3.4:1)
    assert_eq!(render::scene_label_color(&crystal_scene(), polys, &lbl, &ThemeTokens::studio()), crate::theme::contrast::INK_LIGHT);
    assert_eq!(render::scene_label_color(&crystal_scene(), polys, &lbl, &ThemeTokens::paper()), to_color32(lbl.color));
    let vec_scene = Scene2D::VectorScene { primitives: crate::scene2d::ScenePrimitives { polygons: vec![ring.clone()], ..empty_prims() } };
    assert_eq!(render::scene_label_color(&vec_scene, polys, &lbl, &ThemeTokens::studio()), to_color32(lbl.color));
}

#[test]
fn concave_core_fill_stays_inside_its_outline() {
    use crate::scene2d::polygon::point_in_polygon;

    // Two 3 × 1 arms overlap in one square. The missing quadrant used to be painted by the fan.
    let mut outline = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]];
    for _ in 0..2 {
        let points: Vec<_> = outline.iter().map(|&[x, y]| egui::pos2(x, y)).collect();
        let mesh = render::concave_fill_mesh(&points, egui::Color32::WHITE);
        let mut area = 0.0;
        for t in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (mesh.vertices[t[0] as usize].pos, mesh.vertices[t[1] as usize].pos, mesh.vertices[t[2] as usize].pos);
            let center = [(a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0];
            assert!(point_in_polygon(center, &outline), "triangle center {center:?} is outside the ring");
            area += ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() / 2.0;
        }
        assert!((area - 5.0).abs() < 1e-6, "expected the area of both arms, got {area}");
        outline.reverse();
    }
}

fn empty_prims() -> crate::scene2d::ScenePrimitives {
    crate::scene2d::ScenePrimitives {
        polygons: Vec::new(),
        lines: Vec::new(),
        circles: Vec::new(),
        labels: Vec::new(),
        site_circle_indices: Vec::new(),
        edge_line_indices: Vec::new(),
        title: String::new(),
        note: String::new(),
        bbox: ([0.0, 0.0], [2.0, 2.0]),
        site_count: 0,
        bond_count: 0,
    }
}

#[test]
fn unit_cell_labels_sit_outside_the_cell_on_outward_normals() {
    use crate::scene2d::periodic::edge_label_pos;
    // square cell: a1 label below the bottom edge, a2 label left of the left edge
    let (a1, a2) = ([1.0_f32, 0.0], [0.0_f32, 1.0]);
    let p1 = edge_label_pos(a1, a2);
    let p2 = edge_label_pos(a2, a1);
    assert!((p1[0] - 0.5).abs() < 1e-6 && p1[1] < 0.0, "{p1:?}");
    assert!((p2[1] - 0.5).abs() < 1e-6 && p2[0] < 0.0, "{p2:?}");
    // skew hexagonal cell a1 = (1, 0), a2 = (-1/2, sqrt3/2): both labels outside the parallelogram
    let (h1, h2) = ([1.0_f32, 0.0], [-0.5_f32, 0.866_025_4]);
    let cell = [[0.0, 0.0], h1, [h1[0] + h2[0], h1[1] + h2[1]], h2];
    for p in [edge_label_pos(h1, h2), edge_label_pos(h2, h1)] {
        assert!(!crate::scene2d::polygon::point_in_polygon(p, &cell), "{p:?} must lie outside the unit cell");
    }
}
