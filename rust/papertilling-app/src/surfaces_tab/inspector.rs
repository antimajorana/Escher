use papertilling_core::types::SurfaceMesh;

use super::{SurfacesTabState, verdict_colour};
use crate::backend::VerdictKindUi;
use crate::canvas2d::Canvas2D;
use crate::mesh3d::SurfaceColor3d;
use crate::spectrum_ui::show_spectrum_panel;
use crate::spectrum_worker::SpectrumColorOverlay;
use crate::surfaces_elastic::SurfaceShapeMode;
use crate::theme::ThemeTokens;
use crate::view3d::View3dState;

impl SurfacesTabState {
    pub fn show_inspector_panel(
        &mut self,
        ui: &mut egui::Ui,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        tokens: &ThemeTokens,
    ) {
        let (need_fvk, need_resync) = self.elastic.show_controls(ui, tokens);
        if need_fvk && let Some(mesh) = &self.current_mesh {
            self.elastic.request_relax(mesh.clone(), Some(ui.ctx()));
        }
        let sp_ev = show_spectrum_panel(ui, &mut self.spectrum, "surf_spec", false, false, tokens);
        if sp_ev.recompute && let Some(mesh) = &self.current_mesh {
            let mut m = mesh.clone();
            if self.elastic.shape_mode == SurfaceShapeMode::ElasticFvK
                && let Some(res) = &self.elastic.last_result
            {
                m.xyz = res.xyz.clone();
                if let (Some(c), Some(cx)) = (m.cover.as_deref_mut(), res.cover_xyz.as_ref()) {
                    c.xyz = cx.clone();
                }
            }
            self.spectrum.submit_surface(&m, Some(ui.ctx()));
        }
        let mut resync_3d = need_resync || sp_ev.redraw_colors;
        if sp_ev.redraw_colors {
            self.elastic.color_mode = match self.spectrum.color_overlay {
                SpectrumColorOverlay::LdosMultiplet => SurfaceColor3d::SpectralLdos,
                SpectrumColorOverlay::SpectralCharge => SurfaceColor3d::SpectralCharge,
                SpectrumColorOverlay::None => SurfaceColor3d::FaceRingSize,
            };
            resync_3d = true;
        }
        if resync_3d {
            self.sync_view3d_and_colors(canvas, view3d);
        }
        ui.separator();
        self.show_check_readout(ui, tokens);
    }

    fn show_check_readout(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("Topological Invariants Check");
        if let (Some(mesh), Some(rep)) = (&self.current_mesh, &self.current_report) {
            let badge_col = if rep.ok {
                verdict_colour(VerdictKindUi::Ok, tokens)
            } else {
                tokens.status_err
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
                verdict_colour(VerdictKindUi::Impossible, tokens)
            } else {
                verdict_colour(VerdictKindUi::NotFound, tokens)
            };
            ui.colored_label(col, egui::RichText::new(msg).strong());
        }
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
