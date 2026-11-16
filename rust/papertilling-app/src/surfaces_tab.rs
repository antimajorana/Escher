use std::ops::RangeInclusive;

use egui_extras::{Column, TableBuilder};
use papertilling_core::surface::symmetry_facts;
use papertilling_core::types::{Frame2D, Recipe, SurfaceCheckReport, SurfaceMesh};

use crate::backend::{
    AppBackend, CatalogueItemUi, CentreModeUi, SurfaceBuildParamsUi, SurfaceCellVerdictUi, SurfaceTargetUi,
    VerdictKindUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::live_ops;
use crate::mesh3d::{GpuMeshData, mesh_from_surface};
use crate::view3d::View3dState;

#[cfg(test)]
mod tests;

/// What the two size controls mean for a target, and the values offered (every value is used as shown).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeControls {
    pub h_label: &'static str,
    pub k_label: &'static str,
    pub h_range: RangeInclusive<i32>,
    pub k_range: RangeInclusive<i32>,
}

/// Size controls of `target` for a lattice whose highest rotation order is `max_rot` (2: the pillow needs h, k >= 1).
/// The largest values keep a rebuild well under a second on the heaviest catalogue lattices (see the timing test).
pub fn size_controls(target: SurfaceTargetUi, max_rot: u32) -> SizeControls {
    let c = |h_label, k_label, h_range, k_range| SizeControls { h_label, k_label, h_range, k_range };
    match target {
        SurfaceTargetUi::Torus => c("T1 = (h, 0)", "T2 = (0, k)", 1..=24, 1..=24),
        SurfaceTargetUi::Cylinder => c("rows h", "wrap (0, k)", 1..=24, 1..=24),
        SurfaceTargetUi::MobiusBand => c("width h", "along 2k+1", 1..=12, 1..=12),
        SurfaceTargetUi::KleinBottle => c("across h", "along 2k+1", 1..=12, 1..=12),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            c("size h", "size k", 1..=8, if max_rot == 2 { 1..=8 } else { 0..=8 })
        }
        SurfaceTargetUi::Genus2 | SurfaceTargetUi::Genus3 => c("torus (h, 0)", "torus (0, k)", 1..=12, 1..=12),
    }
}

/// An empty net: shown when the surface is refused, so the canvas never keeps the previous surface's net.
pub fn empty_net(name: &str) -> SurfaceMesh {
    SurfaceMesh {
        name: name.to_string(),
        sites: Vec::new(),
        bonds: Vec::new(),
        faces: Vec::new(),
        plane_degree: Vec::new(),
        edge_sites: Vec::new(),
        chi: 0,
        orientable: true,
        closed: true,
        cones: Vec::new(),
        uv: Vec::new(),
        domain_polygons: Vec::new(),
        domain_edges: Vec::new(),
        slits: Vec::new(),
        xyz: Vec::new(),
        bond_curves_3d: Vec::new(),
        cover: None,
    }
}

fn verdict_colour(v: VerdictKindUi) -> egui::Color32 {
    match v {
        VerdictKindUi::Ok => egui::Color32::from_rgb(24, 134, 68),
        VerdictKindUi::Impossible => egui::Color32::from_rgb(158, 52, 52),
        VerdictKindUi::NotFound => egui::Color32::from_rgb(186, 118, 16),
    }
}

pub struct SurfacesTabState {
    pub target: SurfaceTargetUi,
    pub params: SurfaceBuildParamsUi,
    pub show_cover: bool,
    /// Sizes as in `surface_table` (the surface behind the verdict cell) instead of the (h, k) controls.
    pub auto_size: bool,
    current_mesh: Option<SurfaceMesh>,
    current_report: Option<SurfaceCheckReport>,
    refusal_msg: Option<String>,
    lattice_verdicts: Option<(VerdictKey, [SurfaceCellVerdictUi; 8])>,
}

type VerdictKey = (String, Recipe, Frame2D);

impl SurfacesTabState {
    pub fn new() -> Self {
        Self {
            target: SurfaceTargetUi::Sphere,
            params: SurfaceBuildParamsUi::default(),
            show_cover: false,
            auto_size: true,
            current_mesh: None,
            current_report: None,
            refusal_msg: None,
            lattice_verdicts: None,
        }
    }

    /// Clamp `(h, k)` into the values the controls offer for the current target and lattice.
    pub fn clamp_params(&mut self, item: &CatalogueItemUi) {
        let sc = size_controls(self.target, symmetry_facts(&item.group).max_rotation_order());
        self.params.h = self.params.h.clamp(*sc.h_range.start(), *sc.h_range.end());
        self.params.k = self.params.k.clamp(*sc.k_range.start(), *sc.k_range.end());
        match self.target {
            SurfaceTargetUi::Genus2 => self.params.genus_g = 2,
            SurfaceTargetUi::Genus3 => self.params.genus_g = 3,
            _ => {}
        }
    }

    pub fn activate_and_sync(
        &mut self,
        item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        self.clamp_params(item);
        let built = if self.auto_size {
            live_ops::build_surface_mesh_auto(item, self.target, self.params.centre_mode, self.params.genus_g)
        } else {
            backend.build_surface(item, self.target, &self.params)
        };
        match built {
            Ok((mesh, report)) => {
                let use_cov = self.show_cover && mesh.cover.is_some();
                let gpu = mesh_from_surface(&mesh, use_cov);
                let caption = format!(
                    "{} on {}{} | V={} E={} F={} χ={} (R={:.2}, c=[{:.1},{:.1},{:.1}])",
                    item.label,
                    mesh.name,
                    if use_cov { " [S² double cover]" } else { "" },
                    gpu.vertex_count,
                    gpu.edge_count,
                    gpu.face_count,
                    if use_cov { 2 } else { mesh.chi },
                    gpu.radius,
                    gpu.center[0],
                    gpu.center[1],
                    gpu.center[2],
                );
                view3d.set_mesh(gpu, caption);
                canvas.set_scene(Scene2D::SurfaceNet { mesh: mesh.clone() }, true);
                self.current_mesh = Some(mesh);
                self.current_report = Some(report);
                self.refusal_msg = None;
            }
            Err(reason) => {
                view3d.set_mesh(GpuMeshData::default(), format!("{} on {}: {reason}", item.label, self.target.label()));
                canvas.set_scene(Scene2D::SurfaceNet { mesh: empty_net(self.target.label()) }, true);
                self.current_mesh = None;
                self.current_report = None;
                self.refusal_msg = Some(reason);
            }
        }
    }

    /// The 8 verdicts of `surface_table` on the active lattice itself (cached per lattice).
    fn lattice_verdicts(&mut self, item: &CatalogueItemUi) -> Option<&[SurfaceCellVerdictUi; 8]> {
        let key = (item.group.clone(), item.recipe.clone(), item.frame);
        if self.lattice_verdicts.as_ref().is_none_or(|(k, _)| *k != key) {
            let cells = live_ops::build_verdict_cells(item).ok()?;
            self.lattice_verdicts = Some((key, cells));
        }
        self.lattice_verdicts.as_ref().map(|(_, c)| c)
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

    fn show_check_readout(&self, ui: &mut egui::Ui) {
        ui.strong("Topological Invariants Check");
        if let (Some(mesh), Some(rep)) = (&self.current_mesh, &self.current_report) {
            let badge_col = if rep.ok { verdict_colour(VerdictKindUi::Ok) } else { egui::Color32::from_rgb(196, 38, 32) };
            ui.horizontal(|ui| {
                ui.colored_label(badge_col, egui::RichText::new(if rep.ok { "CHECKS OK" } else { "CHECK FAILED" }).strong());
                ui.label(format!(
                    "| {} | {}",
                    if mesh.orientable { "orientable" } else { "non-orientable" },
                    if mesh.closed { "closed" } else { "open rim" }
                ));
            });
            let (v, e, f) = (mesh.sites.len(), mesh.bonds.len(), mesh.faces.len());
            ui.small(format!(
                "V = {v}, E = {e}, F = {f}  =>  V - E + F = {} (χ = {})",
                v as i64 - e as i64 + f as i64,
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
            let col = if msg.starts_with("IMPOSSIBLE") {
                verdict_colour(VerdictKindUi::Impossible)
            } else {
                verdict_colour(VerdictKindUi::NotFound)
            };
            ui.colored_label(col, egui::RichText::new(msg).strong());
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
        ui.strong(format!("Active lattice: {} ({})", active_item.label, active_item.group));
        ui.add_space(3.0);
        if self.show_controls(ui, active_item) {
            self.activate_and_sync(active_item, backend, canvas, view3d);
        }
        ui.separator();
        self.show_check_readout(ui);

        ui.separator();
        ui.strong("This lattice: surface_table verdicts (click)");
        let mut clicked_target: Option<SurfaceTargetUi> = None;
        if let Some(cells) = self.lattice_verdicts(active_item).cloned() {
            ui.horizontal_wrapped(|ui| {
                for (c_idx, cell) in cells.iter().enumerate() {
                    let t = SurfaceTargetUi::ALL[c_idx];
                    let txt = egui::RichText::new(format!("{} {}", t.short_header(), cell.verdict.short()))
                        .size(10.5)
                        .color(verdict_colour(cell.verdict));
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
            let mut tb = TableBuilder::new(ui).striped(true).max_scroll_height(195.0).column(Column::exact(36.0));
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
                                let txt = egui::RichText::new(cell.verdict.short()).size(10.0).color(verdict_colour(cell.verdict));
                                let tip = format!("{} ({})\n{}", row_data.lattice_name, row_data.group, cell_tooltip(surf_t, cell));
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
}

fn cell_tooltip(t: SurfaceTargetUi, cell: &SurfaceCellVerdictUi) -> String {
    format!(
        "{} — {}\nSites: {} | Cones: {}\n{}",
        t.label(),
        cell.verdict.badge(),
        cell.sites.map_or(String::from("-"), |s| s.to_string()),
        cell.cones_summary,
        cell.reason
    )
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
    counts.into_iter().rev().map(|(ang, cnt)| format!("{cnt} x {ang:+}°")).collect::<Vec<_>>().join(", ")
}
