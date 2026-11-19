//! Combine-tab helpers for terminated Half-Wall selection, presets, `.obj` export, and inspectors.

use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi, WallEntryUi};
use crate::canvas2d::Canvas2D;
use crate::combine::CombineTabState;
use crate::live_ops::{collect_walls_ui, defect_context};
use crate::theme::ThemeTokens;

/// Metadata of a terminable (`Wa`/`Wb`/`Wd`) wall for the `Combine` studio.
#[derive(Clone, Debug)]
pub struct CombineHalfWallChoice {
    pub id: String,
    pub label: String,
    pub ending_summary: String,
}

#[derive(Clone, Debug, Default)]
pub struct CombineHalfWallState {
    pub choices: Vec<CombineHalfWallChoice>,
    pub selected_idx: usize,
}

impl CombineHalfWallState {
    pub fn from_item(item: &CatalogueItemUi) -> Self {
        let mut st = Self::default();
        st.refresh(item);
        st
    }

    pub fn refresh(&mut self, item: &CatalogueItemUi) {
        self.choices = defect_context(item)
            .map(|ctx| collect_terminable_walls(&collect_walls_ui(&ctx.cat)))
            .unwrap_or_default();
        self.selected_idx = self.selected_idx.min(self.choices.len().saturating_sub(1));
    }

    pub fn active_choice(&self) -> Option<&CombineHalfWallChoice> {
        self.choices.get(self.selected_idx).or_else(|| self.choices.first())
    }

    pub fn show_tool_selector(&mut self, ui: &mut egui::Ui) {
        if self.choices.is_empty() {
            ui.small("No orientably terminable walls (Wa/Wb/Wd) on this lattice.");
            return;
        }
        ui.horizontal(|ui| {
            ui.small("Half-Wall:");
            let sel = self.active_choice().map_or("", |c| c.label.as_str());
            egui::ComboBox::from_id_salt("comb_hw_sel")
                .selected_text(sel)
                .show_ui(ui, |ui| {
                    for (i, ch) in self.choices.iter().enumerate() {
                        ui.selectable_value(&mut self.selected_idx, i, &ch.label)
                            .on_hover_text(&ch.ending_summary);
                    }
                });
        });
        if let Some(ch) = self.active_choice() {
            ui.small(&ch.ending_summary);
        }
    }
}

pub fn collect_terminable_walls(walls: &[WallEntryUi]) -> Vec<CombineHalfWallChoice> {
    walls
        .iter()
        .filter(|w| w.can_end_orientable)
        .map(|w| CombineHalfWallChoice {
            id: w.id.clone(),
            label: format!("{} ({})", w.id, w.kind.label()),
            ending_summary: w.ending_summary.clone(),
        })
        .collect()
}

pub fn build_half_wall_spec(choice: &CombineHalfWallChoice, at: [f64; 2]) -> DefectSpec {
    DefectSpec::HalfWall {
        wall_id: Some(choice.id.clone()),
        at,
        direction: None,
        sigma: None,
    }
}

impl CombineTabState {
    pub fn apply_preset_half_wall(
        &mut self,
        gallery_item: &CatalogueItemUi,
        _backend: &AppBackend,
        _canvas: &mut Canvas2D,
    ) {
        let Some(choice) = self.half_wall.active_choice().cloned() else {
            return;
        };
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.clear_selection();
        self.specs = vec![build_half_wall_spec(&choice, [0.0, 0.0])];
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub(crate) fn place_half_wall_at(&mut self, world: [f64; 2]) {
        let Some(choice) = self.half_wall.active_choice().cloned() else {
            return;
        };
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.specs.push(build_half_wall_spec(&choice, world));
    }

    pub(crate) fn on_combine_build_landed(&mut self, gallery_item: &CatalogueItemUi) {
        if self.last_build.preview || self.last_build.crystal.pos.is_empty() {
            return;
        }
        let (group, frame) = {
            let it = self.effective_item(gallery_item);
            (it.group.clone(), it.frame)
        };
        let pins: Vec<[f64; 2]> = self.specs.iter().enumerate()
            .map(|(i, sp)| crate::live_combine::spec_world_pos(&self.info, i, sp))
            .collect();
        self.inspectors.refresh_circuit(&group, &self.info.cell, frame, &self.last_build.crystal, &pins);
        if self.inspectors.spectrum.expanded {
            self.inspectors.spectrum.submit_crystal(&self.last_build.crystal, Some(&self.last_sheet3d.sheet), None);
        }
    }

    pub(crate) fn poll_spectrum_inspector(
        &mut self,
        ctx: Option<&egui::Context>,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) {
        let Some(c) = ctx else { return; };
        if self.inspectors.spectrum.poll(c) {
            if let Some(cv) = canvas {
                self.sync_inspectors_to_canvas(gallery_item, cv);
            }
            self.view3d_dirty = true;
        }
    }

    pub(crate) fn sync_inspectors_to_canvas(
        &self,
        gallery_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
    ) {
        let item = self.effective_item(gallery_item);
        let ao = crate::live_ops_ao::build_ao_crystal_overlay(
            &self.last_build.crystal,
            self.inspectors.overlay_mode,
            &item.group,
            &self.info.cell,
            item.frame,
        );
        self.inspectors.sync_to_canvas(canvas, &self.last_build.crystal, ao);
    }

    pub(crate) fn on_combine_circuit_changed(&mut self, gallery_item: &CatalogueItemUi, canvas: &mut Canvas2D) {
        let (group, frame) = {
            let it = self.effective_item(gallery_item);
            (it.group.clone(), it.frame)
        };
        let pins: Vec<[f64; 2]> = self.specs.iter().enumerate()
            .map(|(i, sp)| crate::live_combine::spec_world_pos(&self.info, i, sp))
            .collect();
        self.inspectors.refresh_circuit(&group, &self.info.cell, frame, &self.last_build.crystal, &pins);
        self.sync_inspectors_to_canvas(gallery_item, canvas);
    }

    pub(crate) fn on_combine_sheet_color_changed(&mut self, ctx: &egui::Context) {
        if self.inspectors.on_sheet_color_changed(self.sheet_opts.color_mode) && !self.last_build.crystal.pos.is_empty() {
            self.inspectors.spectrum.submit_crystal(&self.last_build.crystal, Some(&self.last_sheet3d.sheet), Some(ctx));
        }
        self.view3d_dirty = true;
    }

    pub(crate) fn show_inspectors_ui(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) {
        let (circ_dirty, ov_dirty, spec_recomp, spec_col) = self.inspectors.show_controls_ui(
            ui, "comb_insp", self.specs.len(), true, tokens,
        );
        crate::bieberbach_ui::show_bieberbach_inspector_section(
            ui,
            &gallery_item.group,
            &gallery_item.recipe,
            &self.inspectors,
            "comb_bieberbach",
            tokens,
        );
        if spec_col {
            self.inspectors.sync_3d_from_spectrum(&mut self.sheet_opts);
            self.view3d_dirty = true;
        }
        if circ_dirty {
            self.on_combine_circuit_changed(gallery_item, canvas);
        } else if ov_dirty {
            self.sync_inspectors_to_canvas(gallery_item, canvas);
        }
        if spec_recomp && !self.last_build.crystal.pos.is_empty() {
            self.inspectors.spectrum.submit_crystal(&self.last_build.crystal, Some(&self.last_sheet3d.sheet), Some(ui.ctx()));
        }
    }

    pub fn export_active_obj(
        &self,
        gallery_item: &CatalogueItemUi,
        path: &std::path::Path,
    ) -> Result<String, String> {
        let sheet = if self.sheet_opts.show_flat_baseline {
            &self.last_sheet3d.flat_sheet
        } else {
            &self.last_sheet3d.sheet
        };
        let title = format!(
            "{} — {} defect(s)",
            self.effective_item(gallery_item).display_with_orig(),
            self.specs.len()
        );
        crate::export_obj::export_sheet_obj(&title, sheet, path)
    }
}
