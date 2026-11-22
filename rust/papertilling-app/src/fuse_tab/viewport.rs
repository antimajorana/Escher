//! Side-by-side 2D split viewport for `Fuse & compare`:
//! `[Combined Crystal (A + B)]` on the left, `[Equivalent Single Defect C]` on the right,
//! with bidirectional zoom/pan synchronization, interactive target-site rings, and Burgers loops.

use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::live_combine::{map_drawn_to_ref, map_ref_to_drawn, snap_disclination_centre};
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::{show_defect_viewport_header, show_split_2d_3d};

use super::{FuseDefectChoice, FuseTabState, choice_short_label};

impl FuseTabState {
    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        item: &CatalogueItemUi,
        left_canvas: &mut Canvas2D,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        self.poll(ui.ctx(), item, left_canvas);
        show_defect_viewport_header(ui, self.is_busy());

        let left_cam_before = left_canvas.camera_tuple();
        let right_cam_before = self.single_canvas.camera_tuple();

        let markers = self.build_markers();
        let a_lbl = self
            .choices
            .get(self.choice_a)
            .map_or_else(|| String::from("A"), |c| choice_short_label(&self.info, c));
        let b_lbl = self
            .choices
            .get(self.choice_b)
            .map_or_else(|| String::from("B"), |c| choice_short_label(&self.info, c));

        let mut cand_clicked: Option<usize> = None;
        let mut inter_out = None;

        show_split_2d_3d(
            ui,
            split_ratio,
            "fuse_compare_split",
            tokens,
            |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(22.0);
                    ui.strong(format!("Combined Crystal: {a_lbl} + {b_lbl}"));
                });
                ui.add_space(2.0);
                let inter = left_canvas.show_interactive(ui, &markers, false, None, tokens);
                if self.show_target_sites
                    && !self.last_combined.crystal.pos.is_empty()
                    && left_canvas.last_rect.width() > 1.0
                {
                    let painter = ui.painter_at(left_canvas.last_rect);
                    let ring_col = egui::Color32::from_rgba_unmultiplied(
                        tokens.accent_primary.r(),
                        tokens.accent_primary.g(),
                        tokens.accent_primary.b(),
                        120,
                    );
                    for &slot in &self.slots_b {
                        let is_cur = (slot[0] - self.pos_b[0]).hypot(slot[1] - self.pos_b[1]) < 1e-3;
                        if is_cur {
                            continue;
                        }
                        let drawn = map_ref_to_drawn(&self.last_combined.crystal, slot);
                        if let Some(sp) = left_canvas.world_to_screen_last([drawn[0] as f32, drawn[1] as f32])
                            && left_canvas.last_rect.contains(sp)
                        {
                            painter.circle_stroke(sp, 4.5, egui::Stroke::new(1.25, ring_col));
                        }
                    }
                }
                inter_out = Some(inter);
            },
            |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(22.0);
                    ui.strong("Equivalent Single Defect");
                    if let Some(Ok(bundle)) = &self.last_bundle
                        && let Some(cur) = bundle.candidates.get(self.selected_candidate).or_else(|| bundle.candidates.first())
                    {
                        let star = if cur.is_match { " ★" } else { "" };
                        let selected_lbl = format!("{}{star}", cur.name);
                        if bundle.candidates.len() > 1 {
                            let mut sel = self.selected_candidate;
                            egui::ComboBox::from_id_salt("fuse_single_candidate_top")
                                .width(185.0)
                                .selected_text(selected_lbl)
                                .show_ui(ui, |ui| {
                                    for (idx, c) in bundle.candidates.iter().enumerate() {
                                        let s = if c.is_match { " ★" } else { "" };
                                        if ui
                                            .selectable_value(&mut sel, idx, format!("{}{s}", c.name))
                                            .on_hover_text(&c.description)
                                            .changed()
                                        {
                                            cand_clicked = Some(idx);
                                        }
                                    }
                                });
                        } else {
                            ui.small(selected_lbl);
                        }
                    }
                });
                ui.add_space(2.0);
                self.single_canvas.show(ui, tokens);
            },
        );

        if let Some(new_idx) = cand_clicked {
            self.selected_candidate = new_idx;
            self.sync_single_canvas_only(item, false);
            if self.sync_views {
                self.single_canvas.copy_camera_from(left_canvas);
            }
        }

        // Bidirectional camera synchronization
        if self.sync_views {
            let left_cam_after = left_canvas.camera_tuple();
            let right_cam_after = self.single_canvas.camera_tuple();
            let left_moved = left_cam_after != left_cam_before;
            let right_moved = right_cam_after != right_cam_before;
            if left_moved {
                self.single_canvas.copy_camera_from(left_canvas);
            } else if right_moved {
                left_canvas.copy_camera_from(&self.single_canvas);
            } else if (left_canvas.zoom - self.single_canvas.zoom).abs() > 1e-3
                || left_canvas.pan != self.single_canvas.pan
            {
                self.single_canvas.copy_camera_from(left_canvas);
            }
        }

        // Handle interactive pin drag or target-ring click on the left canvas
        if let Some(inter) = inter_out {
            if let Some((pin_idx, drawn_world)) = inter.dragged_marker {
                self.dragging_pin = Some(pin_idx);
                let ref_pt = if self.last_combined.crystal.pos.is_empty() {
                    drawn_world
                } else {
                    map_drawn_to_ref(&self.last_combined.crystal, drawn_world)
                };
                if pin_idx == 1 {
                    if let Some(snapped) = self.nearest_slot_b(ref_pt)
                        && (snapped[0] - self.pos_b[0]).hypot(snapped[1] - self.pos_b[1]) > 1e-4
                    {
                        self.pos_b = snapped;
                        self.submit_build(item, false);
                    }
                } else if pin_idx == 0 {
                    let snapped = self.snap_anchor_a(item, ref_pt);
                    if (snapped[0] - self.pos_a[0]).hypot(snapped[1] - self.pos_a[1]) > 1e-4 {
                        let delta = [self.pos_b[0] - self.pos_a[0], self.pos_b[1] - self.pos_a[1]];
                        self.pos_a = snapped;
                        self.pos_b = [snapped[0] + delta[0], snapped[1] + delta[1]];
                        self.refresh_slots_b();
                        self.submit_build(item, false);
                    }
                }
            }
            if inter.drag_released {
                self.dragging_pin = None;
            }
            if inter.clicked_empty
                && let Some(drawn_click) = inter.clicked_world
                && !self.last_combined.crystal.pos.is_empty()
            {
                let ref_click = map_drawn_to_ref(&self.last_combined.crystal, drawn_click);
                let tol = 0.45 * self.info.lat_info.scale.max(0.5);
                if let Some(snapped) = self.nearest_slot_b(ref_click)
                    && (snapped[0] - ref_click[0]).hypot(snapped[1] - ref_click[1]) <= tol
                    && (snapped[0] - self.pos_b[0]).hypot(snapped[1] - self.pos_b[1]) > 1e-4
                {
                    self.pos_b = snapped;
                    self.submit_build(item, false);
                }
            }
        }
    }

    fn nearest_slot_b(&self, ref_pt: [f64; 2]) -> Option<[f64; 2]> {
        self.slots_b
            .iter()
            .copied()
            .min_by(|a, b| {
                let da = (a[0] - ref_pt[0]).hypot(a[1] - ref_pt[1]);
                let db = (b[0] - ref_pt[0]).hypot(b[1] - ref_pt[1]);
                da.total_cmp(&db)
            })
    }

    fn snap_anchor_a(&self, item: &CatalogueItemUi, ref_pt: [f64; 2]) -> [f64; 2] {
        match self.choices.get(self.choice_a) {
            Some(FuseDefectChoice::Disclination { class_idx, angle_deg }) => {
                let cid = &self.info.centres[*class_idx].id;
                snap_disclination_centre(item, &self.info, f64::from(*angle_deg), Some(cid), ref_pt).0
            }
            _ => ref_pt,
        }
    }
}
