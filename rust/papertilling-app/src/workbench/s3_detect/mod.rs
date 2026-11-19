//! S3 workbench panel (Defects -> "Detect & Fields"): P0 stub. References no S3 core item (that core is not on this
//! branch); the real panel (detection, strain / Nye fields, elastic & Stroh, motion GIF) replaces this folder.

use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::theme::ThemeTokens;
use crate::workbench::{WbCtx, WbPanel, WbScenes};

pub(crate) const STUB_TITLE: &str = "S3 detection panel (stub)";

#[derive(Default)]
pub(crate) struct Panel {
    source_sites: usize,
}

impl WbPanel for Panel {
    fn show_controls(&mut self, ui: &mut egui::Ui, _cx: &WbCtx) -> bool {
        ui.label("S3 panel (stub): needs the S3 core merged.");
        false
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui, _cx: &WbCtx, _tokens: &ThemeTokens) {
        ui.label(format!("Source crystal (Defects tab): {} sites", self.source_sites));
    }

    fn submit(&mut self, cx: &WbCtx, _repaint: Option<&egui::Context>) {
        self.source_sites = cx.defect_crystal.map_or(0, |c| c.pos.len());
    }

    fn poll(&mut self) -> bool {
        false
    }

    fn scenes(&self) -> WbScenes {
        WbScenes { scene2d: empty_scene(STUB_TITLE, "core not merged yet"), mesh: None, caption: String::new() }
    }

    fn is_busy(&self) -> bool {
        false
    }

    fn error(&self) -> Option<&str> {
        None
    }

    fn lattice_dependent(&self) -> bool {
        true
    }
}
