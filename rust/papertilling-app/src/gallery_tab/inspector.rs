//! Right-inspector analytical controls of the Gallery tab (Wyckoff table, sliders, D-symbol & Bieberbach).

use egui_extras::{Column, TableBuilder};

use super::GalleryTabState;
use crate::backend::{AppBackend, SliderParamUi};
use crate::canvas2d::Canvas2D;
use crate::theme::ThemeTokens;

/// A parameter move cheaper than this is applied live while the slider is dragged.
const LIVE_MOVE_SECS: f64 = 0.05;

impl GalleryTabState {
    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
        advanced: bool,
    ) {
        self.parameter_sliders(ui, canvas, tokens);

        ui.separator();
        if self.inflation.show_default_controls(ui, tokens) {
            self.sync_scene(canvas, true);
        }
        let mut show = self.show_unit_cell;
        if ui.checkbox(&mut show, "Unit cell (a₁, a₂)").changed() {
            self.set_show_unit_cell(show, canvas);
        }
        let mut ov = self.overlays;
        let sym = ui
            .checkbox(&mut ov.symmetry, "Symmetry elements")
            .on_hover_text("Mirrors (solid red), glides (dashed blue, chevrons along the glide), 2/3/4/6-fold centres \
                            (lens / triangle / square / hexagon); hover a centre for its Wyckoff position");
        let vor = ui.checkbox(&mut ov.voronoi, "Voronoi cells").on_hover_text("Voronoi cell of every site (the Delaunay dual)");
        if sym.changed() || vor.changed() {
            self.set_overlays(ov, canvas);
        }
        if let Some(note) = canvas.gallery_overlay_note() {
            ui.colored_label(tokens.status_warn, note);
        }
        let (infl_changed, mode_toggle, new_repeat) =
            self.inflation.show_more_controls(ui, &self.cell, self.repeat);
        if let Some(en) = mode_toggle {
            self.set_inflation_enabled(en, canvas);
        } else if infl_changed {
            self.sync_scene(canvas, true);
        }
        if let Some(r) = new_repeat {
            self.set_repeat(r, canvas);
        }

        ui.separator();
        self.wyckoff_table(ui, backend, tokens);

        ui.separator();
        self.readout_lines(ui, tokens, advanced);
    }

    fn wyckoff_table(&self, ui: &mut egui::Ui, backend: &AppBackend, tokens: &ThemeTokens) {
        let wyck = backend.wyckoff_table(&self.active_item.group, &self.active_item.occupied_wyckoff);
        egui::CollapsingHeader::new(format!(
            "Wyckoff positions ({})",
            super::filter::group_with_orbifold(&self.active_item.group)
        ))
        .id_salt("gal_wyckoff")
        .default_open(false)
        .show(ui, |ui| {
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
                                        ui.colored_label(
                                            tokens.accent_primary,
                                            egui::RichText::new(tag).strong(),
                                        );
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
                                        ui.colored_label(tokens.status_ok, "occupied");
                                    } else {
                                        ui.small("-");
                                    }
                                });
                            });
                        }
                    });
            });
        });
    }

    fn parameter_sliders(&mut self, ui: &mut egui::Ui, canvas: &mut Canvas2D, tokens: &ThemeTokens) {
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
        // A cheap move is applied on the UI thread as the slider moves; a slow family (last move above
        // LIVE_MOVE_SECS) sends each pointer move to the background worker, newest wins, and the slider shows the
        // queued value until the result lands.
        let live = self.last_move_secs < LIVE_MOVE_SECS;
        // (name, value, from a pointer drag): drags go on the step grid, typed / keyboard values are set as given
        let mut moved: Option<(String, f64, bool)> = None;
        for s in self.sliders() {
            let shown = self.pending_value(&s.name).unwrap_or(s.value);
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
                    moved = Some((s.name.clone(), v, by_pointer));
                }
            });
            draw_range_bar(ui, s, shown, tokens);
        }
        if self.retune_pending() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small(format!("Retuning in the background (last move {:.2} s)…", self.last_move_secs));
            });
        }
        ui.small(
            "Bars: red = forbidden (bonds cross, a bond through a site, sites coincide or another tiling); \
             ink = current value; grey = default; blue tick = special value where the group is larger.",
        );
        if let Some((name, v, by_pointer)) = moved {
            if by_pointer && !live {
                self.drag_param_async(&name, v, ui.ctx());
            } else if by_pointer {
                self.drag_param(&name, v, canvas);
            } else {
                self.set_param(&name, v, canvas);
            }
        }
    }

    fn readout_lines(&self, ui: &mut egui::Ui, tokens: &ThemeTokens, advanced: bool) {
        ui.strong("Symmetry & Chirality");
        let status_col = if self.readout.status_ok {
            tokens.status_ok
        } else {
            tokens.status_err
        };
        ui.colored_label(status_col, &self.readout.status_message);
        ui.small(format!(
            "Group: {} · Orbits: {} site, {} bond",
            super::filter::group_with_orbifold(&self.readout.active_group),
            self.active_item.site_orbits,
            self.active_item.bond_orbits,
        ));
        let hand = match (self.readout.chiral, self.readout.topologically_chiral) {
            (true, true) => "± chiral enantiomorph pair (chiral bare graph)",
            (true, false) => "± chiral enantiomorph pair (achiral bare graph)",
            (false, _) => "achiral (mirror/glide symmetry)",
        };
        ui.small(format!("Chirality: {hand}"));
        if self.active_item.recipe != self.base_item().recipe {
            ui.small("Parameters modified from catalogue default");
        }
        if advanced {
            ui.small(format!("D-symbol key: {}", self.readout.dsymbol_key));
            ui.small(format!("Network key: {}", self.readout.network_key));
            ui.small(format!(
                "Symmetry index [Aut(net) : G] = {} · Origin: ({:.2}, {:.2})",
                self.readout.symmetry_index,
                self.active_item.frame.origin[0],
                self.active_item.frame.origin[1]
            ));
            crate::bieberbach_ui::show_wallpaper_bieberbach_collapsible(
                ui,
                &self.readout.active_group,
                "gal_bieberbach",
                tokens,
            );
        }
    }
}

fn draw_range_bar(ui: &mut egui::Ui, s: &SliderParamUi, current: f64, tokens: &ThemeTokens) {
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
    painter.rect_filled(bar, 2.0, tokens.band_valid);
    for f in &s.forbidden {
        let fb = egui::Rect::from_min_max(egui::pos2(x_of(f.lo), bar.top()), egui::pos2(x_of(f.hi), bar.bottom()));
        painter.rect_filled(fb, 1.0, tokens.band_forbidden);
    }
    painter.rect_stroke(bar, 2.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
    let dx = x_of(s.default_value);
    painter.line_segment(
        [egui::pos2(dx, bar.top()), egui::pos2(dx, bar.bottom())],
        egui::Stroke::new(1.5, tokens.text_muted),
    );
    for sp in &s.special_ticks {
        let sx = x_of(sp.value);
        painter.line_segment(
            [egui::pos2(sx, bar.top() - 2.0), egui::pos2(sx, bar.bottom() + 2.0)],
            egui::Stroke::new(2.0, tokens.accent_primary),
        );
        painter.text(
            egui::pos2(sx, rect.top() + 5.0),
            egui::Align2::CENTER_CENTER,
            &sp.supergroup,
            egui::FontId::proportional(10.0),
            tokens.accent_primary,
        );
    }
    let cx = x_of(current);
    painter.line_segment(
        [egui::pos2(cx, bar.top() - 2.0), egui::pos2(cx, bar.bottom() + 2.0)],
        egui::Stroke::new(2.4, tokens.text_primary),
    );
}
