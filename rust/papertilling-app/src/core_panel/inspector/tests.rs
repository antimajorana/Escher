use super::*;
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::scene2d::compile_scene;
use papertilling_core::types::Cell2D;

fn scene() -> Scene2D {
    Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 3, show_unit_cell: false,
    }
}

struct Harness {
    ctx: egui::Context,
    canvas: Canvas2D,
    inspector: CoreInspectorModal,
    inspector_rect: egui::Rect,
    cover: bool,
    editing: bool,
    text: String,
    time: f64,
}

impl Harness {
    fn new() -> Self {
        let mut h = Self {
            ctx: egui::Context::default(), canvas: Canvas2D::new(scene()),
            inspector: CoreInspectorModal {
                title: String::from("test"), group_tag: String::new(), is_match: false,
                description: String::new(), rings_text: String::new(), also_text: String::new(),
                prims: compile_scene(&scene()), pan: egui::Vec2::ZERO, zoom: 80.0, fit_zoom: 80.0,
                world_center: [0.0, 0.0], auto_fit: false,
            },
            inspector_rect: egui::Rect::NOTHING, cover: false, editing: false,
            text: String::new(), time: 0.0,
        };
        for _ in 0..3 { h.frame(Vec::new()); }
        h.canvas.zoom = 50.0;
        h
    }

    fn frame(&mut self, events: Vec<egui::Event>) {
        self.time += 1.0 / 60.0;
        let ctx = self.ctx.clone();
        let out = ctx.run_ui(egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 650.0))),
            time: Some(self.time), events, ..Default::default()
        }, |ui| {
            if self.editing {
                ui.add(egui::TextEdit::singleline(&mut self.text)).request_focus();
            }
            self.canvas.show(ui, &crate::theme::ThemeTokens::paper());
            egui::Window::new("Inspector input test")
                .fixed_pos([200.0, 100.0]).fixed_size([400.0, 350.0])
                .resizable(false).movable(false).collapsible(false).show(ui.ctx(), |ui| {
                    let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                    self.inspector_rect = rect;
                    let hud = egui::Rect::from_min_max(
                        rect.right_bottom() - egui::vec2(164.0, 30.0),
                        rect.right_bottom() - egui::vec2(6.0, 6.0),
                    );
                    self.inspector.handle_input(ui, rect, &response, hud);
                });
            if self.cover {
                egui::Area::new(egui::Id::unique("cover inspector"))
                    .order(egui::Order::Foreground).fixed_pos(self.inspector_rect.min)
                    .show(ui.ctx(), |ui| {
                        ui.allocate_exact_size(self.inspector_rect.size(), egui::Sense::click_and_drag());
                    });
            }
        });
        out.drop_without_applying_deltas();
    }
}

fn plus() -> egui::Event {
    egui::Event::Key {
        key: egui::Key::Plus, physical_key: None, pressed: true, repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE }
}

#[test]
fn shortcuts_and_pinch_belong_only_to_the_viewport_under_the_pointer() {
    let mut h = Harness::new();
    h.frame(vec![egui::Event::PointerMoved(egui::pos2(50.0, 550.0)), plus()]);
    assert!((h.canvas.zoom - 62.5).abs() < 1e-4);
    assert_eq!(h.inspector.zoom, 80.0, "outside shortcut must not also zoom inspector");

    let cursor = h.inspector_rect.center();
    h.frame(vec![egui::Event::PointerMoved(cursor), plus(), egui::Event::Text(String::from("+"))]);
    assert!((h.inspector.zoom - 100.0).abs() < 1e-4);
    assert!((h.canvas.zoom - 62.5).abs() < 1e-4, "covered background must not zoom");
    h.frame(vec![egui::Event::Zoom(1.5)]);
    assert!((h.inspector.zoom - 150.0).abs() < 1e-4);
    assert!((h.canvas.zoom - 62.5).abs() < 1e-4);
}

#[test]
fn covered_inspector_and_text_edits_do_not_receive_zoom_shortcuts() {
    let mut h = Harness::new();
    h.cover = true;
    h.frame(Vec::new());
    h.frame(Vec::new());
    h.frame(vec![egui::Event::PointerMoved(h.inspector_rect.center()), plus(), egui::Event::Zoom(1.5)]);
    assert_eq!(h.inspector.zoom, 80.0);
    assert_eq!(h.canvas.zoom, 50.0);

    h.cover = false;
    h.editing = true;
    h.frame(Vec::new());
    h.frame(Vec::new());
    h.frame(vec![egui::Event::PointerMoved(h.inspector_rect.center()), plus(), egui::Event::Text(String::from("+"))]);
    assert_eq!(h.inspector.zoom, 80.0);
    assert_eq!(h.canvas.zoom, 50.0);
    assert_eq!(h.text, "+", "text input must still get the character");
}

#[test]
fn established_inspector_pan_crosses_hud_but_hud_origin_does_not_pan() {
    let mut h = Harness::new();
    let from = h.inspector_rect.right_bottom() - egui::vec2(230.0, 16.0);
    let to = h.inspector_rect.right_bottom() - egui::vec2(2.0, 16.0);
    h.frame(vec![egui::Event::PointerMoved(from)]);
    h.frame(vec![button(from, true)]);
    for n in 1..=20 {
        h.frame(vec![egui::Event::PointerMoved(from + (to - from) * (n as f32 / 20.0))]);
    }
    h.frame(vec![button(to, false)]);
    assert!((h.inspector.pan.x - 228.0).abs() < 1.0);
    assert!(h.inspector.pan.y.abs() < 1e-4);

    h.time += 1.0;
    let pan = h.inspector.pan;
    let from = h.inspector_rect.right_bottom() - egui::vec2(130.0, 16.0);
    let to = from - egui::vec2(0.0, 100.0);
    h.frame(vec![egui::Event::PointerMoved(from)]);
    h.frame(vec![button(from, true)]);
    h.frame(vec![egui::Event::PointerMoved(to)]);
    h.frame(vec![button(to, false)]);
    assert_eq!(h.inspector.pan, pan);
}
