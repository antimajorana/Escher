use egui_extras::{Column, TableBuilder};

use crate::backend::{
    AppBackend, CatalogueItemUi, CatalogueRuleUi, DSymbolReadoutUi, SliderParamUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowseMode {
    ByGroup,
    Named21,
}

pub struct GalleryTabState {
    mode: BrowseMode,
    pub selected_group_idx: usize,
    pub rule: CatalogueRuleUi,
    catalogue: Vec<CatalogueItemUi>,
    selected_cat_idx: usize,
    selected_named_idx: usize,
    param_values: Vec<(String, f64)>,
    sliders: Vec<SliderParamUi>,
    pub repeat: i32,
    pub show_unit_cell: bool,
    pub readout: DSymbolReadoutUi,
    active_item: CatalogueItemUi,
}

impl GalleryTabState {
    pub fn new(backend: &AppBackend, canvas: &mut Canvas2D) -> Self {
        let selected_group_idx = 16; // p6m (Kagomé)
        let grp = backend.groups()[selected_group_idx].symbol;
        let rule = CatalogueRuleUi::Sheet;
        let catalogue = backend.catalogue_for_group(grp, rule);
        let selected_cat_idx = catalogue
            .iter()
            .position(|it| it.named_key.as_deref() == Some("kagome"))
            .unwrap_or(0);
        let active_item = backend
            .find_named("kagome")
            .cloned()
            .unwrap_or_else(|| catalogue[selected_cat_idx].clone());
        let sliders = backend.sliders_for_item(&active_item, &[]);
        let param_values: Vec<(String, f64)> = sliders
            .iter()
            .map(|s| (s.name.clone(), s.value))
            .collect();
        let (cell, readout) = backend.build_cell(&active_item, &param_values);
        let repeat = 3;
        let show_unit_cell = true;
        canvas.set_scene(
            Scene2D::PeriodicCell {
                cell,
                repeat,
                show_unit_cell,
            },
            true,
        );
        let selected_named_idx = backend
            .named_lattices()
            .iter()
            .position(|it| it.named_key.as_deref() == Some("kagome"))
            .unwrap_or(0);
        Self {
            mode: BrowseMode::ByGroup,
            selected_group_idx,
            rule,
            catalogue,
            selected_cat_idx,
            selected_named_idx,
            param_values,
            sliders,
            repeat,
            show_unit_cell,
            readout,
            active_item,
        }
    }

    pub fn active_item(&self) -> &CatalogueItemUi {
        &self.active_item
    }

    pub fn select_group_by_symbol(
        &mut self,
        symbol: &str,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) {
        if let Some(idx) = backend.groups().iter().position(|g| g.symbol == symbol) {
            self.selected_group_idx = idx;
            self.mode = BrowseMode::ByGroup;
            self.reload_group_catalogue(backend, canvas);
        }
    }

    fn reload_group_catalogue(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        let grp = backend.groups()[self.selected_group_idx].symbol;
        self.catalogue = backend.catalogue_for_group(grp, self.rule);
        self.selected_cat_idx = 0;
        if let Some(first) = self.catalogue.first().cloned() {
            self.activate_item(first, backend, canvas, true);
        }
    }

    fn activate_item(
        &mut self,
        item: CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        reset_view: bool,
    ) {
        if let Some(g_idx) = backend.groups().iter().position(|g| g.symbol == item.group) {
            self.selected_group_idx = g_idx;
        }
        self.sliders = backend.sliders_for_item(&item, &[]);
        self.param_values = self
            .sliders
            .iter()
            .map(|s| (s.name.clone(), s.default_value))
            .collect();
        self.active_item = item;
        self.rebuild_scene(backend, canvas, reset_view);
    }

    pub fn sync_canvas(&self, backend: &AppBackend, canvas: &mut Canvas2D, reset_view: bool) {
        let (cell, _) = backend.build_cell(&self.active_item, &self.param_values);
        canvas.set_scene(
            Scene2D::PeriodicCell {
                cell,
                repeat: self.repeat,
                show_unit_cell: self.show_unit_cell,
            },
            reset_view,
        );
    }

    fn rebuild_scene(&mut self, backend: &AppBackend, canvas: &mut Canvas2D, reset_view: bool) {
        self.sliders = backend.sliders_for_item(&self.active_item, &self.param_values);
        let (cell, readout) = backend.build_cell(&self.active_item, &self.param_values);
        self.readout = readout;
        canvas.set_scene(
            Scene2D::PeriodicCell {
                cell,
                repeat: self.repeat,
                show_unit_cell: self.show_unit_cell,
            },
            reset_view,
        );
    }

    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) -> bool {
        let mut lattice_changed = false;
        ui.horizontal(|ui| {
            ui.strong("Browse:");
            if ui
                .selectable_label(self.mode == BrowseMode::ByGroup, "17 Groups")
                .clicked()
            {
                self.mode = BrowseMode::ByGroup;
                self.reload_group_catalogue(backend, canvas);
                lattice_changed = true;
            }
            if ui
                .selectable_label(self.mode == BrowseMode::Named21, "Named (21)")
                .clicked()
            {
                self.mode = BrowseMode::Named21;
                let item = backend.named_lattices()[self.selected_named_idx].clone();
                self.activate_item(item, backend, canvas, true);
                lattice_changed = true;
            }
        });
        ui.add_space(4.0);

        if self.mode == BrowseMode::ByGroup {
            ui.horizontal(|ui| {
                ui.label("Rule:");
                for r in CatalogueRuleUi::ALL {
                    if ui.selectable_label(self.rule == r, r.label()).clicked() {
                        self.rule = r;
                        self.reload_group_catalogue(backend, canvas);
                        lattice_changed = true;
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
                                    if ui
                                        .selectable_label(self.selected_group_idx == idx, g.symbol)
                                        .on_hover_text(tip)
                                        .clicked()
                                    {
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
                self.selected_group_idx = idx;
                self.reload_group_catalogue(backend, canvas);
                lattice_changed = true;
            }

            ui.separator();
            ui.strong(format!(
                "Lattices in {} ({}, {} entries)",
                backend.groups()[self.selected_group_idx].symbol,
                self.rule.label(),
                self.catalogue.len()
            ));
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
                        for (idx, it) in self.catalogue.iter().enumerate() {
                            body.row(18.0, |mut row| {
                                row.col(|ui| {
                                    ui.label(it.index.to_string());
                                });
                                row.col(|ui| {
                                    if ui
                                        .selectable_label(self.selected_cat_idx == idx, &it.label)
                                        .clicked()
                                    {
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
                self.selected_cat_idx = idx;
                let item = self.catalogue[idx].clone();
                self.activate_item(item, backend, canvas, true);
                lattice_changed = true;
            }
        } else {
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
                                    if ui
                                        .selectable_label(self.selected_named_idx == idx, &it.label)
                                        .clicked()
                                    {
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
                self.selected_named_idx = idx;
                let item = named[idx].clone();
                self.activate_item(item, backend, canvas, true);
                lattice_changed = true;
            }
        }

        ui.separator();
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

        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Free Parameters");
            if !self.sliders.is_empty() && ui.small_button("Reset").clicked() {
                for s in &self.sliders {
                    if let Some(pair) = self.param_values.iter_mut().find(|(n, _)| *n == s.name) {
                        pair.1 = s.default_value;
                    }
                }
                self.rebuild_scene(backend, canvas, false);
            }
        });

        if self.sliders.is_empty() {
            ui.small("No free parameters: lattice is rigid inside its group (scale a only).");
        } else {
            let mut moved = false;
            for s in &self.sliders.clone() {
                let mut v = s.value;
                ui.horizontal(|ui| {
                    ui.label(&s.name);
                    let resp = ui.add(
                        egui::Slider::new(&mut v, s.allowed_min..=s.allowed_max)
                            .step_by(s.step)
                            .fixed_decimals(3),
                    );
                    if resp.changed() {
                        for sp in &s.special_ticks {
                            if (v - sp.value).abs() <= 0.5 * s.step + 1e-9 {
                                v = sp.value;
                                break;
                            }
                        }
                        if let Some(pair) = self.param_values.iter_mut().find(|(n, _)| *n == s.name) {
                            pair.1 = v;
                        } else {
                            self.param_values.push((s.name.clone(), v));
                        }
                        moved = true;
                    }
                });
                draw_range_bar(ui, s, v);
            }
            if moved {
                self.rebuild_scene(backend, canvas, false);
            }
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Patch repeat:");
            if ui.add(egui::Slider::new(&mut self.repeat, 1..=6)).changed() {
                self.rebuild_scene(backend, canvas, true);
            }
        });
        if ui
            .checkbox(&mut self.show_unit_cell, "Show unit cell (a1, a2)")
            .changed()
        {
            self.rebuild_scene(backend, canvas, false);
        }

        ui.separator();
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
        ui.small(format!(
            "Symmetry index [Aut(net) : G] = {}",
            self.readout.symmetry_index
        ));

        lattice_changed
    }
}

fn draw_range_bar(ui: &mut egui::Ui, s: &SliderParamUi, current: f64) {
    let w = (ui.available_width() - 8.0).max(80.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 22.0), egui::Sense::hover());
    if !s.special_ticks.is_empty() || !s.forbidden.is_empty() {
        let mut tip = Vec::new();
        for sp in &s.special_ticks {
            tip.push(format!("Special {}: {}", sp.supergroup, sp.equation));
        }
        for fb in &s.forbidden {
            tip.push(format!("[{:.2}, {:.2}]: {}", fb.lo, fb.hi, fb.reason));
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
        let fb = egui::Rect::from_min_max(
            egui::pos2(x_of(f.lo), bar.top()),
            egui::pos2(x_of(f.hi), bar.bottom()),
        );
        painter.rect_filled(fb, 1.0, egui::Color32::from_rgb(238, 168, 168));
    }
    painter.rect_stroke(
        bar,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(145)),
        egui::StrokeKind::Inside,
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
