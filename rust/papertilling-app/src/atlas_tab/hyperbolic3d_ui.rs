//! Regular hyperbolic honeycombs with the same charge comparison as the H² atlas.

use papertilling_core::honeycomb3d::coxeter_gram::{HoneycombRegime, classify_schlafli};
use papertilling_core::hyperbolic::holonomy::ReflectionGroup;

use super::hyperbolic_charge_ui::HyperbolicChargeUi;
use super::hyperbolic_scene::empty_scene;
use crate::combine_worker::LatestWorker;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;
use crate::workbench::s6_honeycombs::{AtlasHoneycomb, build_atlas_h3};

#[derive(Clone, Copy, PartialEq, Eq)]
struct PatchChoice {
    symbol: [u32; 3],
    layers: u32,
    radius_centi: u32,
    ball_preview: bool,
}

pub(super) struct Hyperbolic3dPanel {
    choice: PatchChoice,
    cache: Option<(PatchChoice, Result<AtlasHoneycomb, String>)>,
    worker: Option<LatestWorker<PatchChoice, Result<AtlasHoneycomb, String>>>,
    pending: Option<PatchChoice>,
    charges: HyperbolicChargeUi,
}

impl Default for Hyperbolic3dPanel {
    fn default() -> Self {
        Self {
            choice: PatchChoice {
                symbol: [5, 3, 5],
                layers: 0,
                radius_centi: 140,
                ball_preview: false,
            },
            cache: None,
            worker: None,
            pending: None,
            charges: HyperbolicChargeUi::default(),
        }
    }
}

impl Hyperbolic3dPanel {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let old = self.choice;
        ui.heading("Regular H³ honeycombs");
        egui::ComboBox::new("h3_atlas_symbol", "Honeycomb")
            .selected_text(symbol_text(self.choice.symbol))
            .show_ui(ui, |ui| {
                for regime in [
                    HoneycombRegime::HyperbolicCompactH3,
                    HoneycombRegime::HyperbolicParacompactH3,
                ] {
                    ui.small(if regime == HoneycombRegime::HyperbolicCompactH3 {
                        "Compact"
                    } else {
                        "Paracompact"
                    });
                    for p in 3..=6 {
                        for q in 3..=6 {
                            for r in 3..=6 {
                                if classify_schlafli(p, q, r).regime == regime {
                                    ui.selectable_value(
                                        &mut self.choice.symbol,
                                        [p, q, r],
                                        symbol_text([p, q, r]),
                                    );
                                }
                            }
                        }
                    }
                }
            });
        if self.choice.symbol != old.symbol {
            self.choice.layers = 0;
            self.choice.ball_preview = false;
        }
        let [p, q, r] = self.choice.symbol;
        let cls = classify_schlafli(p, q, r);
        if cls.regime == HoneycombRegime::HyperbolicCompactH3 {
            ui.add(egui::Slider::new(&mut self.choice.layers, 0..=10).text("Layers"));
            ui.small("Layer 0 is one cell. Each step adds every cell sharing a face, edge or vertex with the previous layer.");
        } else {
            ui.checkbox(
                &mut self.choice.ball_preview,
                "Include surrounding cells (radius preview)",
            );
            if cls.has_infinite_euclidean_cells {
                ui.small("Each cell has infinitely many faces. Layer 0 shows the seed cell’s faces within the chosen radius.");
            } else {
                ui.small("Layer 0 is one cell. Infinitely many cells meet at each ideal vertex, so layer 1 is infinite.");
            }
            if cls.has_infinite_euclidean_cells || self.choice.ball_preview {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Radius");
                    for (r, label) in [(140, "1.4"), (180, "1.8"), (220, "2.2")] {
                        ui.selectable_value(&mut self.choice.radius_centi, r, label);
                    }
                });
            }
        }
        ui.small("Klein ball · curvature K = −1");
        if self.pending.is_some() {
            ui.spinner();
        }
        self.choice != old
    }

    pub(super) fn poll(&mut self, ctx: &egui::Context) -> bool {
        let Some(worker) = &mut self.worker else {
            return false;
        };
        worker.set_repaint_ctx(ctx);
        if let Some(result) = worker.poll()
            && let Some(key) = self.pending.take()
            && key == self.choice
        {
            self.cache = Some((key, result.and_then(|r| r)));
            return true;
        }
        if worker.is_busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        false
    }

    pub(super) fn show_inspector(
        &mut self,
        ui: &mut egui::Ui,
        tokens: &ThemeTokens,
        pristine_only: bool,
    ) {
        let [p, q, r] = self.choice.symbol;
        if !pristine_only {
            self.charges.show(ui, ReflectionGroup::h3(p, q, r), true);
            ui.separator();
        }
        ui.small(format!(
            "Dihedral angle of a regular cell: 2π/{r} = {:.1}°",
            360.0 / f64::from(r)
        ));
        ui.small(format!("{r} cells around an interior edge give zero closure deficit in the reference honeycomb."));
        match &self.cache {
            Some((key, Ok(built))) if *key == self.choice => built.show_inspector(ui, tokens),
            Some((key, Err(err))) if *key == self.choice => {
                ui.label(err.as_str());
            }
            _ => {}
        }
    }

    pub(super) fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        let key = self.choice;
        if self.cache.as_ref().is_none_or(|(k, _)| *k != key) {
            if self.pending != Some(key) {
                self.cache = None;
                // A single finite cell is immediate and is also used by headless exports.
                let [p, q, r] = key.symbol;
                let cls = classify_schlafli(p, q, r);
                if !cls.has_infinite_euclidean_cells && !key.ball_preview && key.layers == 0 {
                    if let Some(worker) = &mut self.worker {
                        worker.cancel();
                    }
                    self.pending = None;
                    self.cache = Some((key, build_atlas_h3(p, q, r, 0, false, key.radius_centi)));
                } else {
                    let worker = self.worker.get_or_insert_with(|| {
                        LatestWorker::spawn("H3 atlas", (), |_, job: PatchChoice| {
                            let [p, q, r] = job.symbol;
                            build_atlas_h3(p, q, r, job.layers, job.ball_preview, job.radius_centi)
                        })
                    });
                    worker.submit(key);
                    self.pending = Some(key);
                }
            }
            if self.pending.is_some() {
                return (
                    empty_scene("Building honeycomb", ""),
                    GpuMeshData {
                        radius: 1.0,
                        ..Default::default()
                    },
                    "Building honeycomb…".into(),
                );
            }
        }
        match &self.cache.as_ref().expect("completed build").1 {
            Ok(built) => built.scenes(),
            Err(err) => (
                empty_scene("H³ patch unavailable", err),
                GpuMeshData {
                    radius: 1.0,
                    ..Default::default()
                },
                err.clone(),
            ),
        }
    }
}

fn symbol_text([p, q, r]: [u32; 3]) -> String {
    format!("{{{p},{q},{r}}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_honeycomb_grows_whole_contact_layers() {
        // {4,3,5}: 6 face neighbours, 12·(5−3) edge-only, and 8·10 vertex-only.
        let h = build_atlas_h3(4, 3, 5, 1, false, 140).unwrap();
        assert_eq!(h.layer_counts(), [1, 111]);
        // Four ideal tetrahedron vertices have infinite stars. The seed itself is finite.
        assert_eq!(
            build_atlas_h3(3, 3, 6, 1, false, 140)
                .unwrap()
                .layer_counts(),
            [1]
        );
    }

    #[test]
    fn all_regular_h3_atlas_choices_build_a_visible_patch() {
        let mut count = 0;
        for p in 3..=6 {
            for q in 3..=6 {
                for r in 3..=6 {
                    if ReflectionGroup::h3(p, q, r).is_err() {
                        continue;
                    }
                    for (layers, radius) in [(0, 140), (1, 180), (2, 220)] {
                        let built = build_atlas_h3(p, q, r, layers, layers > 0, radius).unwrap();
                        let (_, mesh, _) = built.scenes();
                        assert!(
                            built.has_cell_edges() && mesh.vertex_count > 0 && mesh.edge_count > 0,
                            "empty H³ patch for [{p},{q},{r}] with {layers} shells or radius {radius}"
                        );
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 45);
    }
}
