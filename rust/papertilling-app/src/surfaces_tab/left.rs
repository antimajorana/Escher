use egui_extras::{Column, TableBuilder};
use papertilling_core::surface::symmetry_facts;

use super::{SurfacesTabState, size_controls, verdict_colour};
use crate::backend::{
    AppBackend, CatalogueItemUi, CentreModeUi, SurfaceCellVerdictUi, SurfaceTargetUi,
};
use crate::canvas2d::Canvas2D;
use crate::theme::ThemeTokens;
use crate::view3d::View3dState;

impl SurfacesTabState {
    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        tokens: &ThemeTokens,
    ) -> Option<String> {
        self.poll_background(ui.ctx(), canvas, view3d);
        ui.strong(format!("Active lattice: {} ({})", active_item.display_with_orig(), active_item.group));
        ui.add_space(3.0);
        if self.show_controls(ui, active_item) {
            self.activate_and_sync(active_item, backend, canvas, view3d);
        }

        ui.separator();
        ui.strong("This lattice: surface_table verdicts (click)");
        let mut clicked_target: Option<SurfaceTargetUi> = None;
        if let Some(cells) = self.lattice_verdicts(active_item).cloned() {
            ui.horizontal_wrapped(|ui| {
                for (c_idx, cell) in cells.iter().enumerate() {
                    let t = SurfaceTargetUi::ALL[c_idx];
                    let txt = egui::RichText::new(format!("{} {}", t.short_header(), cell.verdict.short()))
                        .size(10.5)
                        .color(verdict_colour(cell.verdict, tokens));
                    let resp = ui.selectable_label(self.target == t, txt).on_hover_text(cell_tooltip(t, cell));
                    if resp.clicked() {
                        clicked_target = Some(t);
                    }
                }
            });
        }

        ui.separator();
        ui.strong("17 x 8 verdicts, first `points` lattice of each group (click)");
        let matrix = backend.surface_verdict_matrix_ready().unwrap_or_else(|| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small("Computing the reference table…");
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            &[]
        });
        let mut clicked_cell: Option<(String, SurfaceTargetUi)> = None;
        ui.push_id("verdict_17x8_tbl", |ui| {
            let mut tb = TableBuilder::new(ui)
                .striped(true)
                .max_scroll_height(195.0)
                .column(Column::exact(36.0));
            for _ in 0..8 {
                tb = tb.column(Column::exact(31.0));
            }
            tb.header(17.0, |mut h| {
                h.col(|ui| {
                    ui.small("Grp");
                });
                for t in SurfaceTargetUi::ALL {
                    h.col(|ui| {
                        ui.small(t.short_header());
                    });
                }
            })
            .body(|mut body| {
                for row_data in matrix {
                    body.row(17.0, |mut row| {
                        let is_cur_grp = row_data.group == active_item.group;
                        row.col(|ui| {
                            if is_cur_grp {
                                ui.strong(&row_data.group);
                            } else {
                                ui.small(&row_data.group);
                            }
                        });
                        for (c_idx, cell) in row_data.cells.iter().enumerate() {
                            let surf_t = SurfaceTargetUi::ALL[c_idx];
                            let selected = is_cur_grp && self.target == surf_t;
                            row.col(|ui| {
                                let txt = egui::RichText::new(cell.verdict.short())
                                    .size(10.0)
                                    .color(verdict_colour(cell.verdict, tokens));
                                let tip = format!(
                                    "{} ({})\n{}",
                                    row_data.lattice_name,
                                    row_data.group,
                                    cell_tooltip(surf_t, cell)
                                );
                                if ui.selectable_label(selected, txt).on_hover_text(tip).clicked() {
                                    clicked_cell = Some((row_data.group.clone(), surf_t));
                                }
                            });
                        }
                    });
                }
            });
        });

        let mut requested_group: Option<String> = None;
        if let Some((grp, surf_t)) = clicked_cell {
            self.target = surf_t;
            self.auto_size = true;
            // Every matrix cell refers to the first points-rule lattice, including a click in the
            // current group while a named or tuned lattice of that group is active.
            requested_group = Some(grp);
        }
        if let Some(t) = clicked_target {
            self.target = t;
            self.auto_size = true;
            self.activate_and_sync(active_item, backend, canvas, view3d);
        }
        requested_group
    }

    fn show_controls(&mut self, ui: &mut egui::Ui, item: &CatalogueItemUi) -> bool {
        let mut dirty = false;
        ui.label("Target surface (8):");
        egui::Grid::new("surf_target_grid").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
            for (i, t) in SurfaceTargetUi::ALL.iter().copied().enumerate() {
                if ui.selectable_label(self.target == t, t.label()).clicked() && self.target != t {
                    self.target = t;
                    dirty = true;
                }
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        ui.separator();
        ui.strong("Construction Controls");
        ui.horizontal(|ui| {
            ui.label("Size:");
            dirty |= ui.radio_value(&mut self.auto_size, true, "as in the table").changed();
            dirty |= ui.radio_value(&mut self.auto_size, false, "custom (h, k)").changed();
        });
        let sc = size_controls(self.target, symmetry_facts(&item.group).max_rotation_order());
        ui.add_enabled_ui(!self.auto_size, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("h: {}", sc.h_label));
                dirty |= ui.add(egui::Slider::new(&mut self.params.h, sc.h_range.clone())).changed();
            });
            ui.horizontal(|ui| {
                ui.label(format!("k: {}", sc.k_label));
                dirty |= ui.add(egui::Slider::new(&mut self.params.k, sc.k_range.clone())).changed();
            });
        });
        if matches!(self.target, SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane) {
            ui.horizontal(|ui| {
                ui.label("Corner centre:");
                for cm in CentreModeUi::ALL {
                    if ui.selectable_label(self.params.centre_mode == cm, cm.label()).clicked() {
                        self.params.centre_mode = cm;
                        dirty = true;
                    }
                }
            });
            ui.small("auto: a face-centre class first, then a site class");
        }
        let has_cover = self.target == SurfaceTargetUi::ProjectivePlane;
        ui.add_enabled_ui(has_cover, |ui| {
            dirty |= ui.checkbox(&mut self.show_cover, "RP² antipodal double cover (S²)").changed();
        });
        dirty
    }
}

fn cell_tooltip(t: SurfaceTargetUi, cell: &SurfaceCellVerdictUi) -> String {
    format!(
        "{} — {}\nSites: {} | Cones: {}\n{}",
        t.label(),
        cell.scoped_badge(),
        cell.sites.map_or(String::from("-"), |s| s.to_string()),
        cell.cones_summary,
        cell.reason
    )
}
