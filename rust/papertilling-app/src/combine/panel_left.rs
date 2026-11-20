use super::{CombineTabState, MANUAL_R_MAX, MANUAL_R_MIN, MotionMode, ToolMode};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::live_combine::CombineDrawMode;
use crate::theme::ThemeTokens;

impl CombineTabState {
    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
        #[allow(unused_variables)] advanced: bool,
    ) -> bool {
        self.poll(Some(ui.ctx()), gallery_item, Some(canvas));
        if self.is_initializing() {
            ui.spinner();
            ui.label("Preparing lattice…");
            return false;
        }
        if let Some(err) = &self.init_error {
            ui.colored_label(tokens.status_err, err);
            if ui.button("Retry loading lattice").clicked() {
                self.reload_for_item(gallery_item, backend, canvas, true);
            }
            ui.small("You can also import a Notebook 3 configuration via Export -> Import .json... in the top bar.");
            return false;
        }
        let (mut dirty, mut reset_view, mut trigger_compare) = (false, false, false);
        ui.horizontal_wrapped(|ui| {
            let item = self.effective_item(gallery_item);
            ui.strong(format!("Lattice: {} ({})", item.display_with_orig(), item.group));
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
        let has_half_walls = !self.half_wall.choices.is_empty();
        ui.add_enabled_ui(has_half_walls, |ui| {
            ui.radio_value(&mut self.tool_mode, ToolMode::PlaceHalfWall, "Click to Place Half-Wall (terminated)");
        });

        if self.tool_mode == ToolMode::PlaceDisclination && has_centres {
            self.disclination_tool_ui(ui);
        } else if self.tool_mode == ToolMode::PlaceHalfWall && has_half_walls {
            self.half_wall.show_tool_selector(ui);
        } else if self.tool_mode == ToolMode::PlaceDislocation {
            self.dislocation_tool_ui(ui);
        }

        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(has_centres, egui::Button::new("+α / -α Pair")).clicked() {
                self.apply_preset_pair(gallery_item, backend, canvas);
            }
            if ui.button("Shortest Disloc").clicked() {
                self.apply_preset_dislocation(gallery_item, backend, canvas);
            }
            if ui.add_enabled(has_half_walls, egui::Button::new("Terminated Half-Wall")).clicked() {
                self.apply_preset_half_wall(gallery_item, backend, canvas);
            }
            if ui.button("Clear All").clicked() {
                self.clear_defects(gallery_item, backend, canvas);
            }
        });

        ui.separator();
        ui.strong("Motion Physics (ΔN = 0)");
        let prev_mode = self.motion_mode;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.motion_mode, MotionMode::Conserved, "Conserved (ΔN=0)")
                .on_hover_text("Dislocations glide only along b; disclinations are locked unless Shift+dragged");
            ui.selectable_value(&mut self.motion_mode, MotionMode::StepDisclinationEmit, "Step D (emit b)")
                .on_hover_text("Dragging a disclination steps it by emitting a ΔN = 0 dislocation");
            ui.selectable_value(&mut self.motion_mode, MotionMode::FreeSetup, "Free Setup")
                .on_hover_text("Unconstrained repositioning / climb for preparing initial defects");
        });
        if self.motion_mode != prev_mode {
            self.sync_to_canvas(gallery_item, canvas, false);
        }
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.show_glide_lines, "Show glide rails").changed() {
                self.sync_to_canvas(gallery_item, canvas, false);
            }
            let busy_drag = self.dragging || self.cut_dragging;
            if ui
                .add_enabled(!self.undo_stack.is_empty() && !busy_drag, egui::Button::new("↶ Undo").small())
                .clicked()
            {
                self.undo_last(gallery_item);
                self.sync_selection_to_canvas(canvas);
            }
            if ui
                .add_enabled(!self.redo_stack.is_empty() && !busy_drag, egui::Button::new("↷ Redo").small())
                .clicked()
            {
                self.redo_last(gallery_item);
                self.sync_selection_to_canvas(canvas);
            }
        });
        if !self.kinematic_status.is_empty() {
            egui::Frame::NONE
                .fill(tokens.accent_soft)
                .stroke(egui::Stroke::new(1.0, tokens.border_focus))
                .corner_radius(3.0)
                .inner_margin(4.0)
                .show(ui, |ui| {
                    ui.colored_label(
                        tokens.text_primary,
                        egui::RichText::new(&self.kinematic_status).size(11.0),
                    );
                });
        }

        ui.separator();
        ui.strong("Patch & Scan Controls");
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(self.opts.draw_mode == CombineDrawMode::FastRelaxed, "Fast Relaxed").clicked() {
                self.opts.draw_mode = CombineDrawMode::FastRelaxed;
                dirty = true;
            }
            if ui.selectable_label(self.opts.draw_mode == CombineDrawMode::Raw, "Raw Surgery").clicked() {
                self.opts.draw_mode = CombineDrawMode::Raw;
                dirty = true;
            }
            // C1: the "Elastic (AO metric)" choice left the UI (see `live_ops_ao::Defect2dMode::ALL`)
        });
        if self.opts.draw_mode == CombineDrawMode::FastRelaxed
            && ui.checkbox(&mut self.opts.full_planar_repair, "Full Planar Repair").changed()
        {
            dirty = true;
        }
        if ui.checkbox(&mut self.show_branch_cuts, "Show Volterra branch cuts").changed() {
            self.sync_to_canvas(gallery_item, canvas, false);
        }
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.opts.auto_radius, "Auto (~500 sites)").changed() {
                (dirty, reset_view) = (true, true);
            }
            if !self.opts.auto_radius
                && ui.add(egui::Slider::new(&mut self.opts.manual_radius, MANUAL_R_MIN..=MANUAL_R_MAX).logarithmic(true)).changed()
            {
                (dirty, reset_view) = (true, true);
            }
        });
        ui.horizontal(|ui| {
            if self.is_building() {
                ui.spinner();
                ui.small("Building…");
            }
            let kind = if self.last_build.preview { "preview" } else { "updated" };
            let coal = if self.last_build.coalesced_count > 0 {
                format!(", {} superposed", self.last_build.coalesced_count)
            } else {
                String::new()
            };
            ui.small(format!(
                "{kind} in {:.1} ms ({} sites{coal}, R={:.1})",
                self.last_build.elapsed_ms,
                self.last_build.crystal.pos.len(),
                self.last_build.applied_radius
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

        if !self.json_status.is_empty() {
            ui.separator();
            ui.small(&self.json_status);
        }
        true
    }

    fn dislocation_tool_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.small("Class:");
            for (i, d) in self.info.dislocations.iter().enumerate() {
                let on = self.selected_disloc_idx == i && d.members.contains(&self.custom_mn);
                if ui.selectable_label(on, format!("{}({},{})", d.id, d.burgers_mn.0, d.burgers_mn.1)).clicked() {
                    self.selected_disloc_idx = i;
                    if !d.members.contains(&self.custom_mn) {
                        self.custom_mn = d.burgers_mn;
                    }
                }
            }
        });
        if let Some(d_cls) = self.info.dislocations.get(self.selected_disloc_idx)
            && !d_cls.members.is_empty()
        {
            ui.horizontal_wrapped(|ui| {
                ui.small("Star directions:");
                for &(m, n) in &d_cls.members {
                    if ui.selectable_label(self.custom_mn == (m, n), format!("({m},{n})")).clicked() {
                        self.custom_mn = (m, n);
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            ui.small("b = m·a1 + n·a2 (primitive):")
                .on_hover_text(format!("Catalogue basis: a1={:?}, a2={:?}", self.info.burgers_basis[0], self.info.burgers_basis[1]));
            ui.add(egui::DragValue::new(&mut self.custom_mn.0).range(-4..=4).prefix("m="));
            ui.add(egui::DragValue::new(&mut self.custom_mn.1).range(-4..=4).prefix("n="));
        });
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
}
