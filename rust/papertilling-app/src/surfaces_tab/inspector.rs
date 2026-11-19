use papertilling_core::types::SurfaceMesh;

use super::{SurfacesTabState, verdict_colour};
use crate::backend::VerdictKindUi;
use crate::canvas2d::Canvas2D;
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
        if need_resync {
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
