use egui_extras::{Column, TableBuilder};
use papertilling_core::types::{SurfaceCheckReport, SurfaceMesh};

use crate::backend::{
    AppBackend, CatalogueItemUi, CentreModeUi, SurfaceBuildParamsUi, SurfaceTargetUi, VerdictKindUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::mesh3d::{GpuMeshData, mesh_from_surface};
use crate::view3d::View3dState;

pub struct SurfacesTabState {
    pub target: SurfaceTargetUi,
    pub params: SurfaceBuildParamsUi,
    pub show_cover: bool,
    current_mesh: Option<SurfaceMesh>,
    current_report: Option<SurfaceCheckReport>,
    refusal_msg: Option<String>,
}

impl SurfacesTabState {
    pub fn new() -> Self {
        Self {
            target: SurfaceTargetUi::Sphere,
            params: SurfaceBuildParamsUi::default(),
            show_cover: false,
            current_mesh: None,
            current_report: None,
            refusal_msg: None,
        }
    }

    pub fn activate_and_sync(
        &mut self,
        item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        match backend.build_surface(item, self.target, &self.params) {
            Ok((mesh, report)) => {
                let use_cov = self.show_cover && mesh.cover.is_some();
                let gpu = mesh_from_surface(&mesh, use_cov);
                let caption = format!(
                    "{}{} | V={} E={} F={} χ={} (R={:.2}, c=[{:.1},{:.1},{:.1}])",
                    mesh.name,
                    if use_cov { " [S² double cover]" } else { "" },
                    gpu.vertex_count,
                    gpu.edge_count,
                    gpu.face_count,
                    mesh.chi,
                    gpu.radius,
                    gpu.center[0],
                    gpu.center[1],
                    gpu.center[2]
                );
                view3d.set_mesh(gpu, caption);
                canvas.set_scene(Scene2D::SurfaceNet { mesh: mesh.clone() }, true);
                self.current_mesh = Some(mesh);
                self.current_report = Some(report);
                self.refusal_msg = None;
            }
            Err(reason) => {
                view3d.set_mesh(GpuMeshData::default(), format!("{} — {reason}", item.label));
                self.current_mesh = None;
                self.current_report = None;
                self.refusal_msg = Some(reason);
            }
        }
    }

    pub fn show_side_panel(
        &mut self,
        ui: &mut egui::Ui,
        active_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) -> Option<String> {
        let mut requested_group: Option<String> = None;
        let mut dirty = false;

        ui.strong(format!("Active lattice: {} ({})", active_item.label, active_item.group));
        ui.add_space(3.0);
        ui.label("Target surface (8):");
        egui::Grid::new("surf_target_grid")
            .num_columns(2)
            .spacing([6.0, 4.0])
            .show(ui, |ui| {
                for (i, t) in SurfaceTargetUi::ALL.iter().copied().enumerate() {
                    if ui.selectable_label(self.target == t, t.label()).clicked() {
                        self.target = t;
                        if t == SurfaceTargetUi::Genus2 {
                            self.params.genus_g = 2;
                        } else if t == SurfaceTargetUi::Genus3 {
                            self.params.genus_g = 3;
                        }
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
            ui.label("Size h:");
            dirty |= ui.add(egui::Slider::new(&mut self.params.h, 1..=8)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Size k:");
            dirty |= ui.add(egui::Slider::new(&mut self.params.k, 0..=8)).changed();
        });
        if matches!(self.target, SurfaceTargetUi::Genus2 | SurfaceTargetUi::Genus3) {
            ui.horizontal(|ui| {
                ui.label("Genus g:");
                for g in [2_u32, 3_u32] {
                    if ui
                        .selectable_label(self.params.genus_g == g, format!("g = {g}"))
                        .clicked()
                    {
                        self.params.genus_g = g;
                        self.target = if g == 2 {
                            SurfaceTargetUi::Genus2
                        } else {
                            SurfaceTargetUi::Genus3
                        };
                        dirty = true;
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Corner centre:");
            for cm in CentreModeUi::ALL {
                if ui
                    .selectable_label(self.params.centre_mode == cm, cm.label())
                    .clicked()
                {
                    self.params.centre_mode = cm;
                    dirty = true;
                }
            }
        });

        let has_cover = self.target == SurfaceTargetUi::ProjectivePlane
            || self.current_mesh.as_ref().is_some_and(|m| m.cover.is_some());
        ui.add_enabled_ui(has_cover, |ui| {
            if ui
                .checkbox(&mut self.show_cover, "RP² antipodal double cover (S²)")
                .changed()
            {
                dirty = true;
            }
        });

        if dirty {
            self.activate_and_sync(active_item, backend, canvas, view3d);
        }

        ui.separator();
        ui.strong("Topological Invariants Check");
        if let (Some(mesh), Some(rep)) = (&self.current_mesh, &self.current_report) {
            let badge_col = if rep.ok {
                egui::Color32::from_rgb(24, 134, 68)
            } else {
                egui::Color32::from_rgb(196, 38, 32)
            };
            ui.horizontal(|ui| {
                ui.colored_label(
                    badge_col,
                    egui::RichText::new(if rep.ok { "CHECKS OK" } else { "CHECK FAILED" }).strong(),
                );
                ui.label(format!(
                    "| {} | {}",
                    if mesh.orientable { "orientable" } else { "non-orientable" },
                    if mesh.closed { "closed" } else { "open rim" }
                ));
            });
            let (v, e, f) = (mesh.sites.len(), mesh.bonds.len(), mesh.faces.len());
            ui.small(format!(
                "V = {v}, E = {e}, F = {f}  =>  V - E + F = {} (χ = {})",
                v as i32 - e as i32 + f as i32,
                mesh.chi
            ));
            ui.small(format!("Cones: {}", summarize_cones(mesh)));
            ui.small(format!(
                "loops:{}  double:{}  coord:{}  euler:{}  faces>=3:{}",
                ok_sym(rep.no_loops),
                ok_sym(rep.no_double_bonds),
                ok_sym(rep.plane_coordination),
                ok_sym(rep.euler),
                ok_sym(rep.faces_at_least_3)
            ));
        } else if let Some(msg) = &self.refusal_msg {
            ui.colored_label(egui::Color32::from_rgb(184, 42, 34), egui::RichText::new(msg).strong());
        }

        ui.separator();
        ui.strong("17 x 8 Surface Verdict Matrix (click cell)");
        let matrix = backend.surface_verdict_matrix();
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
                            let col = match cell.verdict {
                                VerdictKindUi::Ok => egui::Color32::from_rgb(24, 134, 68),
                                VerdictKindUi::Impossible => egui::Color32::from_rgb(158, 52, 52),
                                VerdictKindUi::NotFound => egui::Color32::from_rgb(186, 118, 16),
                            };
                            row.col(|ui| {
                                let txt = egui::RichText::new(cell.verdict.short())
                                    .size(10.0)
                                    .color(col);
                                let resp = ui.selectable_label(selected, txt).on_hover_text(format!(
                                    "{} ({}) on {} — {}\nSites: {} | Cones: {}\n{}",
                                    row_data.lattice_name,
                                    row_data.group,
                                    surf_t.label(),
                                    cell.verdict.badge(),
                                    cell.sites.map_or(String::from("-"), |s| s.to_string()),
                                    cell.cones_summary,
                                    cell.reason
                                ));
                                if resp.clicked() {
                                    clicked_cell = Some((row_data.group.clone(), surf_t));
                                }
                            });
                        }
                    });
                }
            });
        });

        if let Some((grp, surf_t)) = clicked_cell {
            self.target = surf_t;
            if grp != active_item.group {
                requested_group = Some(grp);
            } else {
                self.activate_and_sync(active_item, backend, canvas, view3d);
            }
        }

        requested_group
    }
}

fn ok_sym(ok: bool) -> &'static str {
    if ok { "ok" } else { "FAIL" }
}

fn summarize_cones(mesh: &SurfaceMesh) -> String {
    if mesh.cones.is_empty() {
        return String::from("none (flat)");
    }
    let mut counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for c in &mesh.cones {
        *counts.entry(c.angle_deg.round() as i32).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .rev()
        .map(|(ang, cnt)| format!("{cnt} x {ang:+}°"))
        .collect::<Vec<_>>()
        .join(", ")
}
