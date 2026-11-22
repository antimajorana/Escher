//! Left control rail for `Fuse & compare`: reaction presets, Defect A & B catalogue pickers,
//! discrete shell approach stepper (`◀ Closer` / `Farther ▶`), and view/circuit toggles.

use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::gallery_tab::filter::group_with_orbifold;
use crate::theme::ThemeTokens;

use super::{FuseDefectChoice, FuseTabState, choice_label, choice_short_label};

impl FuseTabState {
    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        left_canvas: &mut Canvas2D,
        _tokens: &ThemeTokens,
    ) {
        ui.strong(format!(
            "Lattice: {} · {}",
            item.label,
            group_with_orbifold(&item.group)
        ));
        ui.small("Pick two catalogue defects, vary their separation Δ, and compare the fused crystal side-by-side against its equivalent single defect.");

        if self.choices.is_empty() {
            ui.separator();
            ui.label("No elementary defects available for this lattice.");
            return;
        }

        // 1. Curated reaction presets for the active lattice
        let presets = self.available_presets();
        if !presets.is_empty() {
            ui.separator();
            ui.strong("Quick Fusion Reactions");
            ui.horizontal_wrapped(|ui| {
                for p in presets {
                    if ui.button(p.label()).on_hover_text(p.tooltip()).clicked() {
                        self.apply_preset(p, item, true);
                    }
                }
            });
        }

        // 2. Defect A & Defect B pickers
        ui.separator();
        let mut rebuild = false;
        let mut fit_needed = false;

        ui.horizontal(|ui| {
            ui.strong("Defect A (anchor)");
            if ui.small_button("± Flip").on_hover_text("Invert topological charge of Defect A").clicked()
                && let Some(flipped) = self.flipped_choice_index(self.choice_a)
            {
                self.choice_a = flipped;
                rebuild = true;
            }
        });
        let cur_a_lbl = choice_label(&self.info, &self.choices[self.choice_a]);
        egui::ComboBox::from_id_salt("fuse_pick_a")
            .width(ui.available_width() - 4.0)
            .selected_text(cur_a_lbl)
            .show_ui(ui, |ui| {
                for i in 0..self.choices.len() {
                    let lbl = choice_label(&self.info, &self.choices[i]);
                    if ui.selectable_value(&mut self.choice_a, i, lbl).changed() {
                        self.pos_a = self.default_pos_for_choice(self.choice_a);
                        self.refresh_slots_b();
                        rebuild = true;
                        fit_needed = true;
                    }
                }
            });
        ui.small(format!("Anchor position: ({:+.3}, {:+.3})", self.pos_a[0], self.pos_a[1]));

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("Defect B (movable)");
            if ui.small_button("± Flip").on_hover_text("Invert topological charge of Defect B").clicked()
                && let Some(flipped) = self.flipped_choice_index(self.choice_b)
            {
                self.choice_b = flipped;
                self.refresh_slots_b();
                rebuild = true;
            }
            if ui.small_button("Swap A ↔ B").on_hover_text("Exchange Defect A and Defect B").clicked() {
                std::mem::swap(&mut self.choice_a, &mut self.choice_b);
                self.pos_a = self.default_pos_for_choice(self.choice_a);
                self.refresh_slots_b();
                rebuild = true;
            }
        });
        let cur_b_lbl = choice_label(&self.info, &self.choices[self.choice_b]);
        egui::ComboBox::from_id_salt("fuse_pick_b")
            .width(ui.available_width() - 4.0)
            .selected_text(cur_b_lbl)
            .show_ui(ui, |ui| {
                for i in 0..self.choices.len() {
                    let lbl = choice_label(&self.info, &self.choices[i]);
                    if ui.selectable_value(&mut self.choice_b, i, lbl).changed() {
                        self.refresh_slots_b();
                        rebuild = true;
                    }
                }
            });

        // 3. Separation & Approach Stepper
        ui.separator();
        ui.strong("Separation & Approach (Δ = r_B − r_A)");
        let a_scale = self.info.cell.a1[0].hypot(self.info.cell.a1[1]).max(1e-6);
        let dx = self.pos_b[0] - self.pos_a[0];
        let dy = self.pos_b[1] - self.pos_a[1];
        let dist = dx.hypot(dy);
        ui.label(format!(
            "|Δ| = {:.3} ({:.2} a)  ·  Δ = ({:+.2}, {:+.2})",
            dist,
            dist / a_scale,
            dx,
            dy
        ));

        if !self.slots_b.is_empty() {
            let cur_slot = self.current_slot_index().unwrap_or(0);
            let max_slot = self.slots_b.len() - 1;
            let slot_dist = |p: [f64; 2]| (p[0] - self.pos_a[0]).hypot(p[1] - self.pos_a[1]);
            let best_in_shell = |target_d: f64| -> Option<usize> {
                self.slots_b
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| (slot_dist(**p) - target_d).abs() <= 1e-3)
                    .max_by(|(_, u), (_, v)| {
                        let (ux, uy) = (u[0] - self.pos_a[0], u[1] - self.pos_a[1]);
                        let (vx, vy) = (v[0] - self.pos_a[0], v[1] - self.pos_a[1]);
                        let du = ux.hypot(uy).max(1e-9);
                        let dv = vx.hypot(vy).max(1e-9);
                        let dot_u = (ux * dx + uy * dy) / du;
                        let dot_v = (vx * dx + vy * dy) / dv;
                        dot_u.total_cmp(&dot_v)
                    })
                    .map(|(i, _)| i)
            };
            let closer_idx = (0..cur_slot)
                .rev()
                .map(|i| slot_dist(self.slots_b[i]))
                .find(|&d| d + 1e-3 < dist)
                .and_then(best_in_shell);
            let farther_idx = ((cur_slot + 1)..self.slots_b.len())
                .map(|i| slot_dist(self.slots_b[i]))
                .find(|&d| d > dist + 1e-3)
                .and_then(best_in_shell);

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(closer_idx.is_some(), egui::Button::new("◀ Closer"))
                    .on_hover_text("Step Defect B one radial lattice shell closer to Defect A along the current ray")
                    .clicked()
                    && let Some(idx) = closer_idx
                {
                    self.pos_b = self.slots_b[idx];
                    rebuild = true;
                }
                if ui
                    .add_enabled(farther_idx.is_some(), egui::Button::new("Farther ▶"))
                    .on_hover_text("Step Defect B one radial lattice shell farther from Defect A along the current ray")
                    .clicked()
                    && let Some(idx) = farther_idx
                {
                    self.pos_b = self.slots_b[idx];
                    rebuild = true;
                }
                ui.small(format!("site {} / {}", cur_slot + 1, self.slots_b.len()));
            });
            let mut slot_slider = cur_slot;
            if max_slot > 0
                && ui
                    .add(egui::Slider::new(&mut slot_slider, 0..=max_slot).text("Site index"))
                    .changed()
            {
                self.pos_b = self.slots_b[slot_slider];
                rebuild = true;
            }
        }
        ui.small(format!(
            "Tip: drag pin 2 ({}) or click any hollow target ring on the left crystal to place B.",
            choice_short_label(&self.info, &self.choices[self.choice_b])
        ));

        // 4. Display, Sync & Burgers Loop
        ui.separator();
        ui.strong("Comparison & Overlays");
        ui.checkbox(&mut self.sync_views, "Sync zoom & pan across both views");
        if ui
            .checkbox(&mut self.show_burgers_loop, "Enclosing Burgers / holonomy loop")
            .changed()
        {
            self.sync_canvases(item, left_canvas, false);
        }
        ui.checkbox(&mut self.show_target_sites, "Show allowed target sites for B");

        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.label("Drawing:");
            if ui.selectable_label(self.relaxed, "Relaxed").clicked() && !self.relaxed {
                self.relaxed = true;
                rebuild = true;
            }
            if ui.selectable_label(!self.relaxed, "Unrelaxed cut").clicked() && self.relaxed {
                self.relaxed = false;
                rebuild = true;
            }
        });

        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.auto_radius, "Auto patch size").changed() {
                rebuild = true;
                fit_needed = true;
            }
            if !self.auto_radius
                && ui
                    .add(egui::Slider::new(&mut self.manual_radius, 3.5..=14.0).text("R"))
                    .changed()
            {
                rebuild = true;
            }
        });

        if rebuild {
            self.submit_build(item, fit_needed);
        }
    }

    fn flipped_choice_index(&self, idx: usize) -> Option<usize> {
        match *self.choices.get(idx)? {
            FuseDefectChoice::Disclination { class_idx, angle_deg } => {
                self.choices.iter().position(|c| {
                    matches!(*c, FuseDefectChoice::Disclination { class_idx: ci, angle_deg: a } if ci == class_idx && a == -angle_deg)
                })
            }
            FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign } => {
                if let Some(pos) = self.choices.iter().position(|c| {
                    matches!(*c, FuseDefectChoice::Dislocation { disloc_idx: di, member_idx: mi, sign: s } if di == disloc_idx && mi == member_idx && s == -sign)
                }) {
                    return Some(pos);
                }
                let (m0, n0) = self.info.dislocations[disloc_idx].members[member_idx];
                let want = (-m0 * i32::from(sign), -n0 * i32::from(sign));
                self.choices.iter().position(|c| match *c {
                    FuseDefectChoice::Dislocation { disloc_idx: di, member_idx: mi, sign: 1 } if di == disloc_idx => {
                        self.info.dislocations[di].members[mi] == want
                    }
                    _ => false,
                })
            }
        }
    }
}
