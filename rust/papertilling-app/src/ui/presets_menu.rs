//! "Presets" menu of the Top Bar (shown on every tab): the curated showcase presets, grouped by mode.

use crate::app::{AppMode, PapertillingApp};
use crate::presets::PRESETS;

impl PapertillingApp {
    pub(crate) fn show_presets_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Presets", |ui| {
            let mut picked = None;
            let mut first_section = true;
            for mode in AppMode::ALL {
                let mut rows = PRESETS.iter().enumerate().filter(|(_, p)| p.mode() == mode).peekable();
                if rows.peek().is_none() {
                    continue;
                }
                if !first_section {
                    ui.separator();
                }
                first_section = false;
                ui.label(egui::RichText::new(mode.label()).strong());
                for (idx, p) in rows {
                    if ui.button(p.name).on_hover_text(p.description).clicked() {
                        picked = Some(idx);
                    }
                }
            }
            if let Some(idx) = picked {
                ui.close();
                if let Err(err) = self.apply_preset(idx) {
                    self.export_status = err;
                }
            }
        })
        .response
        .on_hover_text("Curated showcase configurations: lattice, mode and settings in one click");
    }
}
