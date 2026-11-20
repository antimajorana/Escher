//! The "Boundary" part of the Surfaces left panel (C1 WP-U3): the selector with a verdict colour and reason on each
//! entry, and the options of the chosen boundary (layers, circumference, supercell, glide class...). Default view:
//! plain words only; the other surfaces and the verdict tables sit under "More" (see `left.rs`).

use super::bc::{BC_TARGETS, BcSummary, BcControls, plain_label};
use super::{SurfacesTabState, verdict_colour};
use crate::backend::{CatalogueItemUi, SurfaceTargetUi};
use crate::theme::ThemeTokens;

/// The selector order: the six boundary conditions, then the two cone metrics.
const SELECTOR: [SurfaceTargetUi; 8] = [
    BC_TARGETS[0],
    BC_TARGETS[1],
    BC_TARGETS[2],
    BC_TARGETS[3],
    BC_TARGETS[4],
    BC_TARGETS[5],
    SurfaceTargetUi::Sphere,
    SurfaceTargetUi::ProjectivePlane,
];

fn vec2(ui: &mut egui::Ui, label: &str, v: &mut [i32; 2]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for c in v.iter_mut() {
            changed |= ui.add(egui::DragValue::new(c).range(-12..=12)).changed();
        }
    });
    changed
}

fn int(ui: &mut egui::Ui, label: &str, v: &mut i32, range: std::ops::RangeInclusive<i32>) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(v, range)).changed()
    })
    .inner
}

impl SurfacesTabState {
    /// Selector and options. Returns true when a build is needed.
    pub(super) fn show_boundary(&mut self, ui: &mut egui::Ui, item: &CatalogueItemUi, tokens: &ThemeTokens) -> bool {
        let mut dirty = false;
        let summary = self.bc_summary(item).cloned();
        let table = self.lattice_verdicts(item).cloned();
        ui.strong("Boundary");
        egui::Grid::new("surf_boundary_grid").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
            for (i, t) in SELECTOR.into_iter().enumerate() {
                let (kind, reason) = match (BC_TARGETS.contains(&t), &summary, &table) {
                    (true, Some(s), _) => s.verdict(t).map(|v| (v.kind, v.text.clone())).unzip(),
                    (false, _, Some(cells)) => {
                        let c = &cells[SurfaceTargetUi::ALL.iter().position(|&x| x == t).unwrap_or(0)];
                        (Some(c.verdict), Some(c.reason.clone()))
                    }
                    _ => (None, None),
                };
                let mut text = egui::RichText::new(plain_label(t));
                if let Some(k) = kind {
                    text = text.color(verdict_colour(k, tokens));
                }
                let mut resp = ui.selectable_label(self.target == t, text);
                if let (Some(k), Some(r)) = (kind, reason) {
                    resp = resp.on_hover_text(format!("{}: {r}", k.short()));
                }
                if resp.clicked() && self.target != t {
                    self.choose_target(t);
                    dirty = true;
                }
                if i % 2 == 1 {
                    ui.end_row();
                }
            }
        });
        dirty |= self.show_bc_options(ui, summary.as_ref());
        dirty
    }

    fn show_bc_options(&mut self, ui: &mut egui::Ui, summary: Option<&BcSummary>) -> bool {
        let legacy_sized = !self.bc.explicit
            && !self.auto_size
            && matches!(
                self.target,
                SurfaceTargetUi::Torus | SurfaceTargetUi::Cylinder | SurfaceTargetUi::MobiusBand | SurfaceTargetUi::KleinBottle
            );
        if legacy_sized {
            // sizes set elsewhere (presets, session): show them as the options they stand for
            self.bc = BcControls::from_hk(self.params.h, self.params.k);
        }
        let table_sized = self.auto_size && BC_TARGETS.contains(&self.target);
        if table_sized {
            ui.small("Sized as in the verdict table; change an option below to size it here.");
        }
        let mut edited = false;
        let bc = &mut self.bc;
        match self.target {
            SurfaceTargetUi::Open => {
                edited |= int(ui, "Layers:", &mut bc.layers, 1..=BcControls::MAX_LAYERS);
                ui.small("Tiles grown around the most symmetric vertex; the edge stays open.");
            }
            SurfaceTargetUi::Cylinder => {
                edited |= vec2(ui, "Circumference (n, m):", &mut bc.wrap);
                edited |= int(ui, "Rows:", &mut bc.rows, 1..=24);
            }
            SurfaceTargetUi::Torus | SurfaceTargetUi::TorusHex => {
                edited |= vec2(ui, "Supercell vector 1:", &mut bc.t1);
                edited |= vec2(ui, "Supercell vector 2:", &mut bc.t2);
                if self.target == SurfaceTargetUi::TorusHex {
                    ui.small("The same torus drawn on a hexagon; a rectangular supercell gives a rectangle.");
                }
            }
            SurfaceTargetUi::MobiusBand | SurfaceTargetUi::KleinBottle => {
                if let Some(s) = summary
                    && !s.glides.is_empty()
                {
                    let n = s.glides.len();
                    bc.glide = bc.glide.min(n - 1);
                    egui::ComboBox::from_id_salt("bc_glide_class")
                        .selected_text(format!("Twist: {}", s.glides[bc.glide]))
                        .show_ui(ui, |ui| {
                            for (i, g) in s.glides.iter().enumerate() {
                                edited |= ui.selectable_value(&mut bc.glide, i, g).changed();
                            }
                        });
                } else if let Some(v) = summary.and_then(|s| s.verdict(self.target)) {
                    // the core's own verdict: "IMPOSSIBLE" only when the complete glide search proves it
                    ui.small(format!("{}: {}", v.kind.short(), v.text));
                }
                edited |= int(ui, "Steps along:", &mut bc.steps, 1..=25);
                if self.target == SurfaceTargetUi::MobiusBand {
                    edited |= int(ui, "Width (rows):", &mut bc.width, 1..=12);
                } else {
                    edited |= int(ui, "Cells across:", &mut bc.across, 1..=12);
                }
            }
            _ => {}
        }
        if edited {
            self.bc.explicit = true;
            self.auto_size = false;
        }
        edited
    }
}
