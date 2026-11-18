//! "Presets" menu of the Top Bar (shown on every tab): the curated showcase presets, grouped by target tab.

use crate::app::{NavTab, PapertillingApp};
use crate::presets::PRESETS;

/// Short tab names for the menu's section headings.
fn tab_heading(tab: NavTab) -> &'static str {
    match tab {
        NavTab::Gallery => "Gallery",
        NavTab::Surfaces => "Surfaces",
        NavTab::Defects => "Defects",
        NavTab::Combine => "Combine",
        NavTab::Atlas => "Atlas",
    }
}

impl PapertillingApp {
    pub(crate) fn show_presets_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Presets", |ui| {
            let mut picked = None;
            let mut first_section = true;
            for tab in NavTab::ALL {
                let mut rows = PRESETS.iter().enumerate().filter(|(_, p)| p.tab() == tab).peekable();
                if rows.peek().is_none() {
                    continue;
                }
                if !first_section {
                    ui.separator();
                }
                first_section = false;
                ui.label(egui::RichText::new(tab_heading(tab)).strong());
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
        .on_hover_text("Curated showcase configurations: lattice, tab and settings in one click");
    }
}
