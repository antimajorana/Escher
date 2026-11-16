//! Side panel of the Gallery tab.

use egui_extras::{Column, TableBuilder};

use super::{BrowseMode, GalleryTabState};
use crate::backend::{AppBackend, CatalogueRuleUi, SliderParamUi};
use crate::canvas2d::Canvas2D;

/// A parameter move cheaper than this is applied live while the slider is dragged.
const LIVE_MOVE_SECS: f64 = 0.05;

impl GalleryTabState {
    /// Inline rename control bar (Enter to save, Esc to cancel, Reset name when custom alias is active).
    pub fn show_rename_bar(&mut self, ui: &mut egui::Ui, backend: &mut AppBackend, canvas: &mut Canvas2D, id_salt: &str) {
        ui.push_id(id_salt, |ui| {
            ui.horizontal_wrapped(|ui| {
                if self.renaming {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.rename_buf)
                            .desired_width(165.0)
                            .hint_text(&self.active_item.default_label),
                    );
                    if self.rename_focus_pending {
                        resp.request_focus();
                        self.rename_focus_pending = false;
                    }
                    let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if esc {
                        self.cancel_rename();
                    } else if (enter && (resp.lost_focus() || resp.has_focus())) || ui.small_button("Save").clicked() {
                        self.commit_rename(backend, canvas);
                    } else if ui.small_button("Cancel").clicked() {
                        self.cancel_rename();
                    }
                    ui.small(format!("(id: {})", self.active_item.default_label));
                } else {
                    let title_resp = ui
                        .add(
                            egui::Label::new(egui::RichText::new(&self.active_item.label).strong())
                                .sense(egui::Sense::click()),
                        )
                        .on_hover_text("Double-click to give this lattice a custom name");
                    if title_resp.double_clicked() {
                        self.begin_rename();
                    }
                    if self.active_item.has_custom_name() {
                        ui.small(format!("[{}]", self.active_item.default_label));
                    }
                    if ui.small_button("✎ Rename").on_hover_text("Give this lattice a persistent custom name").clicked() {
                        self.begin_rename();
                    }
                    if self.active_item.has_custom_name()
                        && ui.small_button("Reset name").on_hover_text("Restore default catalogue name").clicked()
                    {
                        self.reset_active_lattice_name(backend, canvas);
                    }
                }
            });
            if let Some(note) = &self.alias_note {
                ui.small(note);
            }
        });
    }

    /// Draw the side panel. Returns true when another lattice was selected (group, rule, list entry, named
    /// lattice); slider moves are reported by `revision()` only.
    pub fn show_side_panel(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let mut lattice_changed = false;
        ui.horizontal(|ui| {
            ui.strong("Browse:");
            if ui.selectable_label(self.mode == BrowseMode::ByGroup, "17 Groups").clicked() {
                self.select_group(self.selected_group_idx, backend, canvas);
                lattice_changed = true;
            }
            if ui.selectable_label(self.mode == BrowseMode::Named21, "Named (21)").clicked() {
                self.select_named(self.selected_named_idx, backend, canvas);
                lattice_changed = true;
            }
        });
        ui.add_space(4.0);

        if self.mode == BrowseMode::ByGroup {
            lattice_changed |= self.group_tables(ui, backend, canvas);
        } else {
            lattice_changed |= self.named_table(ui, backend, canvas);
        }

        ui.separator();
        self.wyckoff_table(ui, backend);
        ui.separator();
        self.parameter_sliders(ui, canvas);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Patch repeat:");
            let mut r = self.repeat;
            if ui.add(egui::Slider::new(&mut r, 1..=6)).changed() {
                self.set_repeat(r, canvas);
            }
        });
        let mut show = self.show_unit_cell;
        if ui.checkbox(&mut show, "Show unit cell (a1, a2)").changed() {
            self.set_show_unit_cell(show, canvas);
        }

        ui.separator();
        self.readout_lines(ui);
        lattice_changed
    }

    fn group_tables(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Rule:");
            for r in CatalogueRuleUi::ALL {
                if ui.selectable_label(self.rule == r, r.label()).clicked() {
                    self.select_rule(r, backend, canvas);
                    changed = true;
                }
            }
        });
        ui.add_space(3.0);

        let groups = backend.groups();
        let mut picked_grp: Option<usize> = None;
        ui.push_id("gallery_groups_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(135.0)
                .column(Column::exact(46.0))
                .column(Column::exact(84.0))
                .column(Column::exact(34.0))
                .column(Column::remainder())
                .header(18.0, |mut h| {
                    for t in ["Grp", "System", "|P|", "List N"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, g) in groups.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let tip = format!(
                                    "{} ({}) — max {}-fold, {}",
                                    g.symbol,
                                    g.crystal_system,
                                    g.max_rotation,
                                    if g.has_reflection_or_glide { "achiral" } else { "chiral" }
                                );
                                if ui.selectable_label(self.selected_group_idx == idx, g.symbol).on_hover_text(tip).clicked() {
                                    picked_grp = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(g.crystal_system);
                            });
                            row.col(|ui| {
                                ui.label(g.point_order.to_string());
                            });
                            row.col(|ui| {
                                ui.label(g.counts[self.rule.index()].to_string());
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_grp {
            self.select_group(idx, backend, canvas);
            changed = true;
        }

        ui.separator();
        let grp = groups.get(self.selected_group_idx).map_or("?", |g| g.symbol);
        ui.strong(format!("Lattices in {grp} ({}, {} entries)", self.rule.label(), self.catalogue().len()));
        if self.catalogue().is_empty() {
            ui.small("No stored lattice for this group under this rule.");
        }
        let mut picked_cat: Option<usize> = None;
        ui.push_id("gallery_cat_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(96.0)
                .column(Column::exact(26.0))
                .column(Column::remainder())
                .column(Column::exact(38.0))
                .column(Column::exact(44.0))
                .header(18.0, |mut h| {
                    for t in ["#", "Lattice", "Sites", "Coord"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, it) in self.catalogue().iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                ui.label(it.index.to_string());
                            });
                            row.col(|ui| {
                                let txt = it.display_with_orig();
                                if ui.selectable_label(self.selected_cat_idx == idx, txt).clicked() {
                                    picked_cat = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(it.sites_per_cell.to_string());
                            });
                            row.col(|ui| {
                                ui.label(&it.coordination);
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_cat {
            self.select_catalogue_entry(idx, backend, canvas);
            changed = true;
        }
        changed
    }

    fn named_table(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        ui.strong("21 Named Lattices (18 sheet + 3 variants)");
        let named = backend.named_lattices();
        let mut picked_named: Option<usize> = None;
        ui.push_id("named_21_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(185.0)
                .column(Column::remainder())
                .column(Column::exact(44.0))
                .column(Column::exact(44.0))
                .column(Column::exact(40.0))
                .header(18.0, |mut h| {
                    for t in ["Name", "Grp", "Sheet", "Sites"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for (idx, it) in named.iter().enumerate() {
                        body.row(18.0, |mut row| {
                            row.col(|ui| {
                                let txt = it.display_with_orig();
                                if ui.selectable_label(self.selected_named_idx == idx, txt).clicked() {
                                    picked_named = Some(idx);
                                }
                            });
                            row.col(|ui| {
                                ui.label(&it.group);
                            });
                            row.col(|ui| {
                                ui.label(it.sheet_group.as_deref().unwrap_or("-"));
                            });
                            row.col(|ui| {
                                ui.label(it.sites_per_cell.to_string());
                            });
                        });
                    }
                });
        });
        if let Some(idx) = picked_named {
            self.select_named(idx, backend, canvas);
            return true;
        }
        false
    }

    fn wyckoff_table(&self, ui: &mut egui::Ui, backend: &AppBackend) {
        let wyck = backend.wyckoff_table(&self.active_item.group, &self.active_item.occupied_wyckoff);
        ui.strong(format!("ITA Wyckoff Table ({})", self.active_item.group));
        ui.push_id("wyckoff_tbl", |ui| {
            TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(110.0)
                .column(Column::exact(38.0))
                .column(Column::exact(48.0))
                .column(Column::exact(76.0))
                .column(Column::remainder())
                .header(17.0, |mut h| {
                    for t in ["Pos", "Sym", "Coords", "State"] {
                        h.col(|ui| {
                            ui.strong(t);
                        });
                    }
                })
                .body(|mut body| {
                    for w in &wyck {
                        body.row(17.0, |mut row| {
                            let tag = format!("{}{}", w.multiplicity, w.letter);
                            row.col(|ui| {
                                if w.occupied {
                                    ui.colored_label(egui::Color32::from_rgb(22, 108, 192), egui::RichText::new(tag).strong());
                                } else {
                                    ui.label(tag);
                                }
                            });
                            row.col(|ui| {
                                ui.label(&w.site_symmetry);
                            });
                            row.col(|ui| {
                                ui.label(&w.coords_expr);
                            });
                            row.col(|ui| {
                                if w.occupied {
                                    ui.colored_label(egui::Color32::from_rgb(24, 134, 75), "occupied");
                                } else {
                                    ui.small("-");
                                }
                            });
                        });
                    }
                });
        });
    }

    fn parameter_sliders(&mut self, ui: &mut egui::Ui, canvas: &mut Canvas2D) {
        ui.horizontal(|ui| {
            ui.strong("Free Parameters");
            if !self.sliders().is_empty() && ui.small_button("Reset").clicked() {
                self.reset_params(canvas);
            }
        });
        if self.sliders().is_empty() {
            ui.small("No free parameters: lattice is rigid inside its group (scale a only).");
            return;
        }
        // Live update while dragging when a move is cheap; otherwise only on release (ipywidgets'
        // continuous_update=False), so a 20-parameter lattice does not drag at a few frames per second.
        let live = self.last_move_secs < LIVE_MOVE_SECS;
        // (name, value, from a pointer drag): drags go on the step grid, typed / keyboard values are set as given
        let mut moved: Option<(String, f64, bool)> = None;
        let mut pending = self.pending_drag.clone();
        for s in self.sliders() {
            let shown = pending.as_ref().filter(|(n, _)| *n == s.name).map_or(s.value, |p| p.1);
            let mut v = shown;
            ui.horizontal(|ui| {
                ui.label(&s.name);
                // No egui clamping, step or rounding: those would rewrite the stored value on the first frame (a
                // default such as b = 1.154 would become 1.15 without any input). Grid, limits and snapping are
                // applied by SliderState when the user moves the slider.
                let resp = ui.add_enabled(
                    s.allowed_min < s.allowed_max,
                    egui::Slider::new(&mut v, s.allowed_min..=s.allowed_max)
                        .clamping(egui::SliderClamping::Never)
                        .custom_formatter(|x, _| format!("{x:.3}")),
                );
                let by_pointer = resp.dragged() || resp.drag_stopped();
                if resp.changed() && v != shown {
                    if resp.dragged() && !live {
                        pending = Some((s.name.clone(), v));
                    } else {
                        moved = Some((s.name.clone(), v, by_pointer));
                    }
                }
                if !resp.dragged() && pending.as_ref().is_some_and(|(n, _)| *n == s.name) {
                    moved = pending.take().map(|(n, x)| (n, x, true));
                }
            });
            draw_range_bar(ui, s, shown);
        }
        ui.small(
            "Bars: red = forbidden (bonds cross, a bond through a site, sites coincide or another tiling); \
             black = current value; grey = default; blue tick = special value where the group is larger.",
        );
        self.pending_drag = pending;
        if let Some((name, v, by_pointer)) = moved {
            self.pending_drag = None;
            if by_pointer {
                self.drag_param(&name, v, canvas);
            } else {
                self.set_param(&name, v, canvas);
            }
        }
    }

    fn readout_lines(&self, ui: &mut egui::Ui) {
        ui.strong("Delaney-Dress Symbol & Symmetry");
        let status_col = if self.readout.status_ok {
            egui::Color32::from_rgb(24, 128, 56)
        } else {
            egui::Color32::from_rgb(184, 38, 32)
        };
        ui.colored_label(status_col, &self.readout.status_message);
        ui.small(format!(
            "Group: {} | Orbits: {} site, {} bond | Origin: ({:.2}, {:.2})",
            self.readout.active_group,
            self.active_item.site_orbits,
            self.active_item.bond_orbits,
            self.active_item.frame.origin[0],
            self.active_item.frame.origin[1]
        ));
        ui.small(format!("D-symbol key: {}", self.readout.dsymbol_key));
        ui.small(format!("Network key: {}", self.readout.network_key));
        ui.small(format!("Symmetry index [Aut(net) : G] = {}", self.readout.symmetry_index));
        let moved = self.active_item.recipe != self.base_item().recipe;
        ui.small(format!(
            "Lattice revision {}: the other tabs use this lattice{}",
            self.revision(),
            if moved { " at the slider values (Reset: the stored one)" } else { " as stored" }
        ));
    }
}

fn draw_range_bar(ui: &mut egui::Ui, s: &SliderParamUi, current: f64) {
    let w = (ui.available_width() - 8.0).max(80.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 22.0), egui::Sense::hover());
    {
        let mut tip = vec![format!(
            "{}: limits [{:.4}, {:.4}], step {}, default {:.4} (grey mark)",
            s.name, s.allowed_min, s.allowed_max, s.step, s.default_value
        )];
        for sp in &s.special_ticks {
            tip.push(format!("Special {} at {:.4}: {}", sp.supergroup, sp.value, sp.equation));
        }
        for fb in &s.forbidden {
            tip.push(format!("[{:.3}, {:.3}]: {}", fb.lo, fb.hi, fb.reason));
        }
        resp.on_hover_text(tip.join("\n"));
    }
    let painter = ui.painter_at(rect);
    let span = (s.nominal_max - s.nominal_min).max(1e-6);
    let x_of = |val: f64| -> f32 {
        let t = ((val - s.nominal_min) / span).clamp(0.0, 1.0) as f32;
        rect.left() + t * rect.width()
    };
    let bar = egui::Rect::from_min_max(
        egui::pos2(rect.left(), rect.top() + 11.0),
        egui::pos2(rect.right(), rect.top() + 19.0),
    );
    painter.rect_filled(bar, 2.0, egui::Color32::from_rgb(216, 238, 198));
    for f in &s.forbidden {
        let fb = egui::Rect::from_min_max(egui::pos2(x_of(f.lo), bar.top()), egui::pos2(x_of(f.hi), bar.bottom()));
        painter.rect_filled(fb, 1.0, egui::Color32::from_rgb(238, 168, 168));
    }
    painter.rect_stroke(bar, 2.0, egui::Stroke::new(1.0, egui::Color32::from_gray(145)), egui::StrokeKind::Inside);
    let dx = x_of(s.default_value);
    painter.line_segment(
        [egui::pos2(dx, bar.top()), egui::pos2(dx, bar.bottom())],
        egui::Stroke::new(1.5, egui::Color32::from_gray(120)),
    );
    for sp in &s.special_ticks {
        let sx = x_of(sp.value);
        painter.line_segment(
            [egui::pos2(sx, bar.top() - 2.0), egui::pos2(sx, bar.bottom() + 2.0)],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(21, 101, 192)),
        );
        painter.text(
            egui::pos2(sx, rect.top() + 5.0),
            egui::Align2::CENTER_CENTER,
            &sp.supergroup,
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(21, 101, 192),
        );
    }
    let cx = x_of(current);
    painter.line_segment(
        [egui::pos2(cx, bar.top() - 2.0), egui::pos2(cx, bar.bottom() + 2.0)],
        egui::Stroke::new(2.4, egui::Color32::BLACK),
    );
}
