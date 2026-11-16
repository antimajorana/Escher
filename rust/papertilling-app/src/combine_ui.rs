//! Combine tab panels (side controls and the live canvas + core gallery strip).

use egui_extras::{Column, Size, StripBuilder, TableBuilder};
use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::combine_tab::{CombineTabState, MANUAL_R_MAX, MANUAL_R_MIN, ToolMode};
use crate::core_panel::show_core_panel;
use crate::live_combine::{CombineDrawMode, spec_short_label, spec_world_pos};

impl CombineTabState {
    pub fn show_side_panel(&mut self, ui: &mut egui::Ui, gallery_item: &CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        self.poll(Some(ui.ctx()), gallery_item, Some(canvas));
        if self.is_initializing() {
            ui.spinner();
            ui.label("Preparing lattice…");
            return;
        }
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
            self.disclination_tool_ui(ui);
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
                ui.small("b = m·a1 + n·a2 (primitive):")
                    .on_hover_text(format!("Catalogue basis: a1={:?}, a2={:?}", self.info.burgers_basis[0], self.info.burgers_basis[1]));
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
        if self.specs.is_empty() {
            ui.small("Empty — use a preset or Click-to-Place.");
        } else if self.placed_table_ui(ui) {
            dirty = true;
        }

        ui.separator();
        ui.strong("Patch & Scan Controls");
        ui.horizontal(|ui| {
            if ui.selectable_label(self.opts.draw_mode == CombineDrawMode::FastRelaxed, "Fast Relaxed").clicked() {
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
            if !self.opts.auto_radius && ui.add(egui::Slider::new(&mut self.opts.manual_radius, MANUAL_R_MIN..=MANUAL_R_MAX).logarithmic(true)).changed() {
                (dirty, reset_view) = (true, true);
            }
        });
        ui.horizontal(|ui| {
            if self.is_building() {
                ui.spinner();
                ui.small("Building…");
            }
            let kind = if self.last_build.preview { "preview" } else { "updated" };
            ui.small(format!(
                "{kind} in {:.1} ms ({} sites, R={:.1})",
                self.last_build.elapsed_ms, self.last_build.crystal.pos.len(), self.last_build.applied_radius
            ));
        });

        ui.horizontal(|ui| {
            trigger_compare |= ui.button("Compare Single").clicked();
            trigger_compare |= ui.checkbox(&mut self.auto_compare, "Auto").changed() && self.auto_compare;
            ui.small("grid:");
            for g in [1_usize, 2, 3] {
                if ui.selectable_label(self.grid == g, g.to_string()).clicked() && self.grid != g {
                    self.grid = g;
                    trigger_compare |= self.auto_compare;
                }
            }
        });

        if dirty {
            self.request_build(gallery_item, false, self.auto_compare, reset_view);
        } else if trigger_compare {
            self.run_compare(gallery_item);
        }

        ui.separator();
        self.equivalence_ui(ui);

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

    fn disclination_tool_ui(&mut self, ui: &mut egui::Ui) {
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
    }

    /// The placed-defects table; returns true when a defect was flipped or removed.
    fn placed_table_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let (mut flip_idx, mut remove_idx) = (None, None);
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
        if let Some(i) = flip_idx {
            match &mut self.specs[i] {
                DefectSpec::Disclination { angle_deg, .. } => *angle_deg = -*angle_deg,
                DefectSpec::Dislocation { b, .. } => *b = [-b[0], -b[1]],
            }
        }
        if let Some(i) = remove_idx {
            self.specs.remove(i);
        }
        flip_idx.is_some() || remove_idx.is_some()
    }

    fn equivalence_ui(&self, ui: &mut egui::Ui) {
        ui.strong("Total Charge & Equivalence");
        ui.label(&self.last_build.charge_label);
        ui.small(&self.last_build.charge_detail);
        if self.last_build.preview {
            ui.small("Core: (read out when the drag ends)");
        } else {
            ui.small(format!("Core: {} nodes, {} edges | {}", self.last_build.core_nodes, self.last_build.core_edges, self.last_build.core_summary));
        }
        if !self.last_build.local_cores.is_empty() {
            let lc: Vec<&str> = self.last_build.local_cores.iter().map(|c| c.label.as_str()).collect();
            ui.small(format!("Cores: {}", lc.join(", ")));
        }
        if self.is_comparing() || self.dragging {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small(if self.dragging { "Comparison runs when the drag ends" } else { "Comparing cores…" });
            });
        }
        let stale = self.compare_stale || self.is_comparing() || self.dragging;
        match &self.compare_result {
            Some(Ok(rep)) => {
                let (bg, fg, tag) = if rep.equivalent {
                    (egui::Color32::from_rgb(225, 244, 228), egui::Color32::from_rgb(34, 108, 42), "EQUIVALENT TO SINGLE")
                } else {
                    (egui::Color32::from_rgb(255, 241, 214), egui::Color32::from_rgb(168, 92, 10), "DISTINCT FROM SINGLE")
                };
                let (bg, fg) = if stale { (egui::Color32::from_gray(236), egui::Color32::from_gray(110)) } else { (bg, fg) };
                egui::Frame::NONE.fill(bg).inner_margin(4.0).corner_radius(3.0).show(ui, |ui| {
                    let tag = if stale { format!("{tag} (previous configuration)") } else { tag.to_string() };
                    ui.colored_label(fg, egui::RichText::new(tag).strong().size(11.0));
                    ui.small(&rep.verdict);
                });
                if !stale {
                    ui.small(format!("compared in {:.1} ms (scan cache {} hit(s), {} miss(es))", self.compare_ms, self.cache_stats.0, self.cache_stats.1));
                }
            }
            Some(Err(err)) => { ui.small(format!("Single-defect check: {err}")); }
            None => {}
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
                    ui.horizontal(|ui| {
                        ui.strong("2D Live Combined Crystal [drag pin: move/snap | drag empty: pan | scroll: zoom | dbl-click: fit]");
                        if self.is_building() {
                            ui.spinner();
                        }
                    });
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
                    let released_elsewhere = self.dragging && !ui.input(|i| i.pointer.any_down());
                    if inter.drag_released || released_elsewhere {
                        self.finish_drag(gallery_item);
                    }
                    if let Some(world) = inter.clicked_world {
                        self.place_at_world(world, gallery_item, backend, canvas);
                    }
                    self.poll(Some(ui.ctx()), gallery_item, Some(canvas));
                });
                strip.cell(|ui| {
                    ui.separator();
                    if self.compare_stale && self.compare_result.is_some() {
                        ui.small("Showing the previous comparison while the new one runs.");
                    }
                    show_core_panel(&mut self.core_panel, ui, self.compare_result.as_ref());
                });
            });
    }
}
