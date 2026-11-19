//! S4 workbench panel (Lattices -> "Flat 3D: Slabs, Tubes, Flakes"): P0 stub. References no S4 core item (that core is not on
//! this branch); the real panel (slabs, platycosms, nanotubes and ribbons, flakes, torsion-free table) replaces this folder.
//! It pushes a small non-empty placeholder mesh so the smoke stage of this view can paint and advance.

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

pub(crate) const STUB_TITLE: &str = "S4 flat 3D panel (stub)";

#[derive(Default)]
pub(crate) struct Panel;

impl WbPanel for Panel {
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        ui.label("S4 panel (stub): needs the S4 core merged.");
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
            caption: String::from("S4 placeholder tetrahedron (core not merged yet)"),
        }
    }

    fn is_busy(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }
}
