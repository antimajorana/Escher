//! S5 workbench panel (Lattices -> "{p,q} on S² / E² / H²"): P0 stub. References no S5 core item (that core is not on
//! this branch); the real panel ({p,q}, Wythoff, large H², embedded surfaces and nets, orbifolds) replaces this folder.
//! It pushes a small non-empty placeholder mesh so the smoke stage of this view can paint and advance.

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

pub(crate) const STUB_TITLE: &str = "S5 curved 2D panel (stub)";

#[derive(Default)]
pub(crate) struct Panel;

impl WbPanel for Panel {
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        ui.label("S5 panel (stub): needs the S5 core merged.");
        false
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, _cx: &WbCtx, _tokens: &ThemeTokens) {
        ui.label("3D: placeholder tetrahedron (4 sites, 6 edges).");
    }

    fn submit(&mut self, _cx: &WbCtx, _repaint: Option<&egui::Context>) {}

    fn poll(&mut self) -> bool {
        false
    }

    fn scenes(&self) -> WbScenes {
        WbScenes {
            scene2d: empty_scene(STUB_TITLE, "core not merged yet"),
            mesh: Some(placeholder_mesh()),
            caption: String::from("S5 placeholder tetrahedron (core not merged yet)"),
        }
    }

    fn is_busy(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }
}
