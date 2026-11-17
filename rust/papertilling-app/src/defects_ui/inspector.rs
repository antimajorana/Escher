use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::defects_tab::DefectsTabState;
use crate::theme::ThemeTokens;

impl DefectsTabState {
    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) {
        if self.initializing {
            return;
        }
        ui.strong("Core Charge & 3D Buckling Readout");
        if self.is_busy() || self.error.is_some() {
            ui.horizontal(|ui| {
                if self.is_busy() {
                    ui.spinner();
                }
                ui.label("Previous readout — stale");
            });
        }
        ui.label(&self.readout.charge_summary);
        ui.small(&self.readout.holonomy_detail);
        ui.small(format!("Core structure: {}", self.readout.core_rings));
        if !self.last_sheet.xyz.is_empty() {
            let z_min = self.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
            let z_max = self.last_sheet.xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
            let dz_a = (z_max - z_min) / self.last_sheet.mean_bond_len.max(1e-9);
            ui.small(format!(
                "3D Sheet: Δz/a = {dz_a:.2} | E_stretch = {:.4}, E_bend = {:.4}",
                self.last_sheet.stretch_energy, self.last_sheet.bend_energy
            ));
        }
        if !self.readout.note.is_empty() {
            ui.colored_label(
                tokens.status_warn,
                format!("Note: {}", self.readout.note),
            );
        }

        let (circ_dirty, ov_dirty, spec_recomp, spec_col) =
            self.inspectors.show_controls_ui(ui, "def_insp", 0, true, tokens);
        if spec_col {
            self.inspectors.sync_3d_from_spectrum(&mut self.sheet_opts);
            self.view3d_dirty = true;
        }
        if circ_dirty || ov_dirty {
            self.refresh_and_sync_inspectors(active_item, canvas);
        }
        if spec_recomp && !self.last_crystal.pos.is_empty() {
            self.inspectors
                .spectrum
                .submit_crystal(&self.last_crystal, Some(&self.last_sheet), Some(ui.ctx()));
        }
        crate::bieberbach_ui::show_bieberbach_inspector_section(
            ui,
            &active_item.group,
            &active_item.recipe,
            &self.inspectors,
            "def_bieberbach",
            tokens,
        );
    }
}
