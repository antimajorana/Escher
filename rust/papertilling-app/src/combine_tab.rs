use egui_extras::{Column, Size, StripBuilder, TableBuilder};
use papertilling_core::defect::combine_types::{
    ConfigOptionsJson, ConfigSummaryJson, EquivalentSingleReport, NotebookConfigJson,
};
use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasMarker, MarkerKind, Scene2D};
use crate::core_panel::{CorePanelState, show_core_panel};
use crate::live_combine::{
    CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo, burgers_from_mn,
    preset_neighbour_pair, preset_shortest_dislocation, spec_short_label, spec_world_pos,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    PanAndDrag,
    PlaceDisclination,
    PlaceDislocation,
}

pub struct CombineTabState {
    pub tool_mode: ToolMode,
    override_item: Option<CatalogueItemUi>,
    pub(crate) info: CombineLatticeInfo,
    pub specs: Vec<DefectSpec>,
    pub(crate) selected_angle_deg: i32,
    pub(crate) centre_filter_idx: usize,
    selected_disloc_idx: usize,
    pub custom_mn: (i32, i32),
    pub opts: CombineBuildOptions,
    pub auto_compare: bool,
    pub grid: usize,
    pub(crate) last_build: CombineBuildOutput,
    pub compare_result: Option<Result<EquivalentSingleReport, String>>,
    compare_pending_after_drag: bool,
    last_hover_world: Option<[f64; 2]>,
    pub json_status: String,
    pub core_panel: CorePanelState,
}

impl CombineTabState {
    pub fn new(gallery_item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let info = backend.combine_lattice_info(gallery_item);
        let specs = preset_neighbour_pair(&info);
        let selected_angle_deg = info.allowed_angles.first().copied().unwrap_or(90);
        let custom_mn = info.dislocations.first().map_or((1, 0), |d| d.burgers_mn);
        let opts = CombineBuildOptions {
            draw_mode: CombineDrawMode::FastRelaxed,
            full_planar_repair: false,
            auto_radius: true,
            manual_radius: info.r_auto_base,
            core_radius: 2,
        };
        let last_build = backend.build_combined(gallery_item, &info, &specs, &opts);
        Self {
            tool_mode: ToolMode::PanAndDrag,
            override_item: None,
            info,
            specs,
            selected_angle_deg,
            centre_filter_idx: 0,
            selected_disloc_idx: 0,
            custom_mn,
            opts,
            auto_compare: true,
            grid: 1,
            last_build,
            compare_result: None,
            compare_pending_after_drag: false,
            last_hover_world: None,
            json_status: String::new(),
            core_panel: CorePanelState::default(),
        }
    }

    pub fn effective_item<'a>(&'a self, gallery_item: &'a CatalogueItemUi) -> &'a CatalogueItemUi {
        self.override_item.as_ref().unwrap_or(gallery_item)
    }

    fn reset_lattice_defaults(&mut self, info: CombineLatticeInfo) {
        self.selected_angle_deg = info.allowed_angles.first().copied().unwrap_or(90);
        self.centre_filter_idx = 0;
        self.selected_disloc_idx = 0;
        self.custom_mn = info.dislocations.first().map_or((1, 0), |d| d.burgers_mn);
        self.opts.manual_radius = info.r_auto_base;
        self.info = info;
    }

    pub fn reload_for_item(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D, sync_canvas: bool) {
        self.override_item = None;
        self.reset_lattice_defaults(backend.combine_lattice_info(gallery_item));
        self.specs = preset_neighbour_pair(&self.info);
        self.rebuild_internal(gallery_item, backend, sync_canvas && self.auto_compare);
        if sync_canvas {
            self.sync_to_canvas(gallery_item, canvas, true);
        }
    }

    pub fn ensure_compared(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend) {
        if self.auto_compare && self.compare_result.is_none() && !self.specs.is_empty() {
            self.run_compare(gallery_item, backend);
        }
    }

    pub fn sync_to_canvas(&self, gallery_item: &CatalogueItemUi, canvas: &mut Canvas2D, reset_view: bool) {
        let item = self.effective_item(gallery_item);
        let title = format!("{} ({}) — {} defect(s) | {}", item.label, item.group, self.specs.len(), self.last_build.charge_label);
        canvas.set_scene(Scene2D::CrystalPatch { crystal: self.last_build.crystal.clone(), title, note: self.last_build.note.clone() }, reset_view);
    }

    fn rebuild_internal(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend, cmp: bool) {
        let item = self.override_item.as_ref().unwrap_or(gallery_item);
        self.last_build = backend.build_combined(item, &self.info, &self.specs, &self.opts);
        self.compare_result = (!self.specs.is_empty() && cmp).then(|| {
            backend.equivalent_single(item, &self.info, &self.specs, &self.last_build.crystal, self.opts.core_radius, self.grid)
        });
    }

    pub fn run_compare(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend) {
        let item = self.override_item.as_ref().unwrap_or(gallery_item);
        self.compare_result = (!self.specs.is_empty()).then(|| {
            backend.equivalent_single(item, &self.info, &self.specs, &self.last_build.crystal, self.opts.core_radius, self.grid)
        });
    }

    pub fn apply_preset_pair(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        self.specs = preset_neighbour_pair(&self.info);
        self.rebuild_internal(gallery_item, backend, self.auto_compare);
        self.sync_to_canvas(gallery_item, canvas, true);
    }

    pub fn apply_preset_dislocation(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        self.specs = preset_shortest_dislocation(&self.info);
        self.rebuild_internal(gallery_item, backend, self.auto_compare);
        self.sync_to_canvas(gallery_item, canvas, true);
    }

    pub fn clear_defects(&mut self, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        self.specs.clear();
        self.rebuild_internal(gallery_item, backend, self.auto_compare);
        self.sync_to_canvas(gallery_item, canvas, true);
    }

    fn active_centre_and_angle(&self) -> (Option<&str>, f64) {
        if let Some(c) = self.centre_filter_idx.checked_sub(1).and_then(|k| self.info.centres.get(k)) {
            let want = self.selected_angle_deg as f64;
            let ang = if c.angles.iter().any(|&a| (a - want).abs() <= 0.5) {
                want
            } else {
                c.angles.first().copied().unwrap_or(want)
            };
            (Some(c.id.as_str()), ang)
        } else {
            (None, self.selected_angle_deg as f64)
        }
    }

    pub fn place_at_world(&mut self, world: [f64; 2], gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        let item = self.override_item.as_ref().unwrap_or(gallery_item);
        match self.tool_mode {
            ToolMode::PanAndDrag => return,
            ToolMode::PlaceDisclination => {
                if self.info.centres.is_empty() {
                    return;
                }
                let (cid, ang) = self.active_centre_and_angle();
                let (snapped, _) = backend.snap_disclination(item, &self.info, ang, cid, world);
                self.specs.push(DefectSpec::disclination(ang, snapped));
            }
            ToolMode::PlaceDislocation => {
                if self.custom_mn == (0, 0) {
                    self.custom_mn = (1, 0);
                }
                let b = burgers_from_mn(&self.info, self.custom_mn.0, self.custom_mn.1);
                self.specs.push(DefectSpec::dislocation(b, Some(world)));
            }
        }
        self.rebuild_internal(gallery_item, backend, self.auto_compare);
        self.sync_to_canvas(gallery_item, canvas, false);
    }

    pub fn move_pin_to(&mut self, idx: usize, world: [f64; 2], gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let item = self.override_item.as_ref().unwrap_or(gallery_item);
        let Some(old_pos) = self.specs.get(idx).map(|s| spec_world_pos(&self.info, idx, s)) else {
            return false;
        };
        let new_spec = match self.specs[idx] {
            DefectSpec::Disclination { angle_deg, cut, .. } => {
                let (snapped, _) = backend.snap_disclination(item, &self.info, angle_deg, None, world);
                if (snapped[0] - old_pos[0]).hypot(snapped[1] - old_pos[1]) < 1e-6 {
                    return false;
                }
                DefectSpec::Disclination { angle_deg, at: snapped, cut }
            }
            DefectSpec::Dislocation { b, cut, .. } => {
                if (world[0] - old_pos[0]).hypot(world[1] - old_pos[1]) < 1e-4 {
                    return false;
                }
                DefectSpec::Dislocation { b, at: Some(world), cut }
            }
        };
        self.specs[idx] = new_spec;
        self.rebuild_internal(gallery_item, backend, false);
        self.compare_pending_after_drag = self.auto_compare;
        self.sync_to_canvas(gallery_item, canvas, false);
        true
    }

    pub fn to_notebook_json(&self, gallery_item: &CatalogueItemUi) -> NotebookConfigJson {
        let item = self.effective_item(gallery_item);
        let summary = self.compare_result.as_ref().and_then(|r| r.as_ref().ok()).map(|rep| ConfigSummaryJson {
            charge: rep.charge.label.clone(),
            kind: rep.kind.clone(),
            equivalent: rep.equivalent,
            matches: rep.matches.clone(),
            verdict: rep.verdict.clone(),
        });
        NotebookConfigJson {
            format: String::from("papertilling.config.v1"),
            lattice_label: item.label.clone(),
            group: item.group.clone(),
            recipe: item.recipe.clone(),
            frame: item.frame,
            defects: self.specs.clone(),
            options: ConfigOptionsJson {
                r_patch: (!self.opts.auto_radius).then_some(self.opts.manual_radius),
                picture: if self.opts.draw_mode == CombineDrawMode::Raw { "raw" } else { "relaxed" }.into(),
                grid: self.grid,
                radius: self.opts.core_radius,
            },
            summary,
        }
    }

    pub fn load_notebook_json(&mut self, cfg: &NotebookConfigJson, backend: &AppBackend, canvas: &mut Canvas2D) -> Result<String, String> {
        let imported = backend.item_from_notebook_json(cfg)?;
        self.reset_lattice_defaults(backend.combine_lattice_info(&imported));
        self.opts.auto_radius = cfg.options.r_patch.is_none();
        self.opts.manual_radius = cfg.options.r_patch.unwrap_or(self.info.r_auto_base).clamp(2.4, 16.0);
        self.opts.draw_mode = if cfg.options.picture == "raw" { CombineDrawMode::Raw } else { CombineDrawMode::FastRelaxed };
        self.grid = cfg.options.grid.clamp(1, 3);
        self.opts.core_radius = cfg.options.radius.clamp(1, 4);
        self.specs = cfg.defects.clone();
        let msg = format!("Imported {} ({}) with {} defect(s)", imported.label, imported.group, self.specs.len());
        self.override_item = Some(imported.clone());
        self.rebuild_internal(&imported, backend, self.auto_compare || cfg.summary.is_some());
        self.sync_to_canvas(&imported, canvas, true);
        self.json_status = msg.clone();
        Ok(msg)
    }

    pub fn export_json_dialog(&mut self, gallery_item: &CatalogueItemUi) {
        let dlg = rfd::FileDialog::new().set_title("Export Notebook 3 defect configuration (.json)").set_file_name("defect_config.json").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.save_file() {
            let res = self.to_notebook_json(gallery_item).to_json_pretty().and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()));
            self.json_status = res.map_or_else(|e| format!("JSON export error: {e}"), |()| format!("Exported {}", path.file_name().and_then(|s| s.to_str()).unwrap_or("config.json")));
        }
    }

    pub fn import_json_dialog(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        let dlg = rfd::FileDialog::new().set_title("Import Notebook 3 defect configuration (.json)").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.pick_file()
            && let Err(err) = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|raw| NotebookConfigJson::from_json_str(&raw)).and_then(|cfg| self.load_notebook_json(&cfg, backend, canvas))
        {
            self.json_status = format!("JSON import error: {err}");
        }
    }

    fn build_markers(&self) -> Vec<CanvasMarker> {
        self.specs.iter().enumerate().map(|(idx, s)| CanvasMarker {
            index: idx,
            pos: spec_world_pos(&self.info, idx, s),
            kind: match *s {
                DefectSpec::Disclination { angle_deg, .. } if angle_deg >= 0.0 => MarkerKind::DisclinationPlus,
                DefectSpec::Disclination { .. } => MarkerKind::DisclinationMinus,
                DefectSpec::Dislocation { .. } => MarkerKind::Dislocation,
            },
            label: spec_short_label(&self.info, s),
        }).collect()
    }

    pub fn show_side_panel(&mut self, ui: &mut egui::Ui, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        let (mut dirty, mut reset_view, mut trigger_compare) = (false, false, false);
        ui.horizontal_wrapped(|ui| {
            let item = self.effective_item(gallery_item);
            ui.strong(format!("Lattice: {} ({})", item.label, item.group));
            if self.override_item.is_some() && ui.small_button("Use Gallery").clicked() {
                self.reload_for_item(gallery_item, backend, canvas, true);
            }
        });
        ui.separator();
        ui.strong("Tool Palette (Click / Drag)");
        ui.radio_value(&mut self.tool_mode, ToolMode::PanAndDrag, "Pan / Select & Drag Defects");
        let has_centres = !self.info.centres.is_empty();
        ui.add_enabled_ui(has_centres, |ui| {
            ui.radio_value(&mut self.tool_mode, ToolMode::PlaceDisclination, "Click to Place Disclination");
        });
        ui.radio_value(&mut self.tool_mode, ToolMode::PlaceDislocation, "Click to Place Dislocation");

        if self.tool_mode == ToolMode::PlaceDisclination && has_centres {
            let filt_c = self.centre_filter_idx.checked_sub(1).and_then(|k| self.info.centres.get(k));
            ui.horizontal_wrapped(|ui| {
                ui.small("α:");
                for &ang in &self.info.allowed_angles {
                    let ok = filt_c.is_none_or(|c| c.angles.iter().any(|&a| (a - ang as f64).abs() <= 0.5));
                    if ui.add_enabled(ok, egui::Button::selectable(self.selected_angle_deg == ang, format!("{ang:+}°"))).clicked() {
                        self.selected_angle_deg = ang;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.small("Centre:");
                let sel_txt = self.centre_filter_idx.checked_sub(1).and_then(|k| self.info.centres.get(k))
                    .map_or_else(|| String::from("Nearest allowed"), |c| format!("{} ({} {})", c.id, c.label, c.location));
                let prev = self.centre_filter_idx;
                egui::ComboBox::from_id_salt("comb_centre_cls").selected_text(sel_txt).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.centre_filter_idx, 0, "Nearest allowed centre");
                    for (i, c) in self.info.centres.iter().enumerate() {
                        ui.selectable_value(&mut self.centre_filter_idx, i + 1, format!("{} ({} {}, {}-fold)", c.id, c.label, c.location, c.order));
                    }
                });
                if self.centre_filter_idx != prev {
                    self.selected_angle_deg = self.active_centre_and_angle().1.round() as i32;
                }
            });
        } else if self.tool_mode == ToolMode::PlaceDislocation {
            ui.horizontal_wrapped(|ui| {
                ui.small("Class:");
                for (i, d) in self.info.dislocations.iter().enumerate() {
                    let on = self.selected_disloc_idx == i && self.custom_mn == d.burgers_mn;
                    if ui.selectable_label(on, format!("{}({},{})", d.id, d.burgers_mn.0, d.burgers_mn.1)).clicked() {
                        self.selected_disloc_idx = i;
                        self.custom_mn = d.burgers_mn;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.small("b = m·a1 + n·a2:");
                ui.add(egui::DragValue::new(&mut self.custom_mn.0).range(-4..=4).prefix("m="));
                ui.add(egui::DragValue::new(&mut self.custom_mn.1).range(-4..=4).prefix("n="));
            });
        }

        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(has_centres, egui::Button::new("+α / -α Pair")).clicked() {
                self.apply_preset_pair(gallery_item, backend, canvas);
            }
            if ui.button("Shortest Disloc").clicked() {
                self.apply_preset_dislocation(gallery_item, backend, canvas);
            }
            if ui.button("Clear All").clicked() {
                self.clear_defects(gallery_item, backend, canvas);
            }
        });

        ui.separator();
        ui.strong(format!("Placed Defects ({})", self.specs.len()));
        let (mut flip_idx, mut remove_idx) = (None, None);
        if self.specs.is_empty() {
            ui.small("Empty — use a preset or Click-to-Place.");
        } else {
            ui.push_id("placed_specs_tbl", |ui| {
                TableBuilder::new(ui)
                    .striped(true)
                    .max_scroll_height(108.0)
                    .column(Column::exact(24.0))
                    .column(Column::exact(72.0))
                    .column(Column::remainder())
                    .column(Column::exact(32.0))
                    .column(Column::exact(24.0))
                    .header(17.0, |mut h| {
                        for t in ["#", "Kind", "(x, y)", "±", ""] {
                            h.col(|ui| { ui.strong(t); });
                        }
                    })
                    .body(|mut body| {
                        for (idx, sp) in self.specs.iter().enumerate() {
                            let pos = spec_world_pos(&self.info, idx, sp);
                            let kstr = spec_short_label(&self.info, sp);
                            body.row(18.0, |mut row| {
                                row.col(|ui| { ui.strong(format!("#{}", idx + 1)); });
                                row.col(|ui| { ui.label(kstr); });
                                row.col(|ui| { ui.small(format!("({:+.2}, {:+.2})", pos[0], pos[1])); });
                                row.col(|ui| { if ui.small_button("±").clicked() { flip_idx = Some(idx); } });
                                row.col(|ui| { if ui.small_button("×").clicked() { remove_idx = Some(idx); } });
                            });
                        }
                    });
            });
        }
        if let Some(i) = flip_idx {
            match &mut self.specs[i] {
                DefectSpec::Disclination { angle_deg, .. } => *angle_deg = -*angle_deg,
                DefectSpec::Dislocation { b, .. } => *b = [-b[0], -b[1]],
            }
            dirty = true;
        }
        if let Some(i) = remove_idx {
            self.specs.remove(i);
            dirty = true;
        }

        ui.separator();
        ui.strong("Patch & Scan Controls");
        ui.horizontal(|ui| {
            if ui.selectable_label(self.opts.draw_mode == CombineDrawMode::FastRelaxed, "Fast Relaxed (<100ms)").clicked() {
                self.opts.draw_mode = CombineDrawMode::FastRelaxed;
                dirty = true;
            }
            if ui.selectable_label(self.opts.draw_mode == CombineDrawMode::Raw, "Raw Surgery").clicked() {
                self.opts.draw_mode = CombineDrawMode::Raw;
                dirty = true;
            }
        });
        if self.opts.draw_mode == CombineDrawMode::FastRelaxed && ui.checkbox(&mut self.opts.full_planar_repair, "Full Planar Repair").changed() {
            dirty = true;
        }
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.opts.auto_radius, "Auto (~500 sites)").changed() {
                (dirty, reset_view) = (true, true);
            }
            if !self.opts.auto_radius && ui.add(egui::Slider::new(&mut self.opts.manual_radius, 2.8..=13.0).step_by(0.2)).changed() {
                (dirty, reset_view) = (true, true);
            }
        });
        ui.small(format!(
            "Updated in {:.1} ms ({} sites, R={:.1})",
            self.last_build.elapsed_ms, self.last_build.crystal.pos.len(), self.last_build.applied_radius
        ));

        ui.horizontal(|ui| {
            trigger_compare |= ui.button("Compare Single").clicked();
            trigger_compare |= ui.checkbox(&mut self.auto_compare, "Auto").changed() && self.auto_compare;
            ui.small("grid:");
            for g in [1_usize, 2, 3] {
                if ui.selectable_label(self.grid == g, g.to_string()).clicked() {
                    self.grid = g;
                    trigger_compare |= self.auto_compare;
                }
            }
        });

        if dirty {
            self.rebuild_internal(gallery_item, backend, self.auto_compare);
            self.sync_to_canvas(gallery_item, canvas, reset_view);
        } else if trigger_compare {
            self.run_compare(gallery_item, backend);
        }

        ui.separator();
        ui.strong("Total Charge & Equivalence");
        ui.label(&self.last_build.charge_label);
        ui.small(&self.last_build.charge_detail);
        ui.small(format!("Core: {} nodes, {} edges | {}", self.last_build.core_nodes, self.last_build.core_edges, self.last_build.core_summary));
        if !self.last_build.local_cores.is_empty() {
            let lc: Vec<&str> = self.last_build.local_cores.iter().map(|c| c.label.as_str()).collect();
            ui.small(format!("Cores: {}", lc.join(", ")));
        }
        match &self.compare_result {
            Some(Ok(rep)) => {
                let (bg, fg, tag) = if rep.equivalent {
                    (egui::Color32::from_rgb(225, 244, 228), egui::Color32::from_rgb(34, 108, 42), "EQUIVALENT TO SINGLE")
                } else {
                    (egui::Color32::from_rgb(255, 241, 214), egui::Color32::from_rgb(168, 92, 10), "DISTINCT FROM SINGLE")
                };
                egui::Frame::NONE.fill(bg).inner_margin(4.0).corner_radius(3.0).show(ui, |ui| {
                    ui.colored_label(fg, egui::RichText::new(tag).strong().size(11.0));
                    ui.small(&rep.verdict);
                });
            }
            Some(Err(err)) => { ui.small(format!("Single-defect check: {err}")); }
            None => {}
        }

        ui.separator();
        ui.strong("Notebook 3 .json");
        ui.horizontal(|ui| {
            if ui.button("Export .json (Notebook 3)...").clicked() {
                self.export_json_dialog(gallery_item);
            }
            if ui.button("Import .json...").clicked() {
                self.import_json_dialog(backend, canvas);
            }
        });
        if !self.json_status.is_empty() {
            ui.small(&self.json_status);
        }
    }

    pub fn show_central(&mut self, ui: &mut egui::Ui, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        let panel_h = if self.core_panel.collapsed { 28.0 } else { 246.0 };
        StripBuilder::new(ui)
            .clip(true)
            .size(Size::remainder().at_least(150.0))
            .size(Size::exact(panel_h))
            .vertical(|mut strip| {
                strip.cell(|ui| {
                    ui.strong("2D Live Combined Crystal [drag pin: move/snap | drag empty: pan | scroll: zoom | dbl-click: fit]");
                    ui.add_space(2.0);
                    let click_to_place = self.tool_mode != ToolMode::PanAndDrag;
                    let preview_world = match (self.tool_mode, self.last_hover_world) {
                        (ToolMode::PlaceDisclination, Some(hw)) if !self.info.centres.is_empty() => {
                            let (cid, ang) = self.active_centre_and_angle();
                            Some(backend.snap_disclination(self.effective_item(gallery_item), &self.info, ang, cid, hw).0)
                        }
                        (ToolMode::PlaceDislocation, Some(hw)) => Some(hw),
                        _ => None,
                    };
                    let markers = self.build_markers();
                    let inter = canvas.show_interactive(ui, &markers, click_to_place, preview_world);
                    self.last_hover_world = inter.hovered_world;
                    if let Some((pin_idx, world)) = inter.dragged_marker {
                        self.move_pin_to(pin_idx, world, gallery_item, backend, canvas);
                    }
                    if inter.drag_released && self.compare_pending_after_drag {
                        self.compare_pending_after_drag = false;
                        self.run_compare(gallery_item, backend);
                    }
                    if let Some(world) = inter.clicked_world {
                        self.place_at_world(world, gallery_item, backend, canvas);
                    }
                });
                strip.cell(|ui| {
                    ui.separator();
                    show_core_panel(&mut self.core_panel, ui, self.compare_result.as_ref());
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CatalogueRuleUi;
    use papertilling_core::types::Cell2D;

    #[test]
    fn combine_tab_state_init_and_presets() {
        let backend = AppBackend::new();
        let square = backend.find_named("square").unwrap();
        let p1_first = &backend.catalogue_for_group("p1", CatalogueRuleUi::Points)[0];
        let mut canvas = Canvas2D::new(Scene2D::PeriodicCell {
            cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
            repeat: 2,
            show_unit_cell: false,
        });
        let mut st = CombineTabState::new(square, &backend);
        assert_eq!(st.specs.len(), 2);
        assert!(!st.last_build.crystal.pos.is_empty());
        st.ensure_compared(square, &backend);
        assert!(st.compare_result.is_some());
        if let Some(Ok(rep)) = &st.compare_result {
            assert!(rep.equivalent);
        }
        assert_eq!(CombineTabState::new(p1_first, &backend).specs.len(), 1);
        st.apply_preset_dislocation(square, &backend, &mut canvas);
        assert_eq!(st.specs.len(), 1);
        st.clear_defects(square, &backend, &mut canvas);
        assert!(st.specs.is_empty() && st.compare_result.is_none());
    }
}
