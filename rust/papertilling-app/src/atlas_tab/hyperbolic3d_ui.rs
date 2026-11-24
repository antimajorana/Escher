//! Regular hyperbolic honeycombs with the same charge comparison as the H² atlas.

use papertilling_core::honeycomb3d::coxeter_gram::{HoneycombRegime, classify_schlafli};
use papertilling_core::hyperbolic::holonomy::ReflectionGroup;

use super::hyperbolic_charge_ui::HyperbolicChargeUi;
use super::hyperbolic_scene::empty_scene;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::wb_mesh::placeholder_mesh;
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;
use crate::workbench::s6_honeycombs::{AtlasHoneycomb, build_atlas_h3};

#[derive(Clone, Copy, PartialEq, Eq)]
struct PatchChoice {
    symbol: [u32; 3],
    layers: u32,
    radius_centi: u32,
}

pub(super) struct Hyperbolic3dPanel {
    choice: PatchChoice,
    cache: Option<(PatchChoice, Result<AtlasHoneycomb, String>)>,
    charges: HyperbolicChargeUi,
}

impl Default for Hyperbolic3dPanel {
    fn default() -> Self {
        Self {
            choice: PatchChoice {
                symbol: [5, 3, 5],
                layers: 1,
                radius_centi: 140,
            },
            cache: None,
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
        let [p, q, r] = self.choice.symbol;
        let cls = classify_schlafli(p, q, r);
        if cls.regime == HoneycombRegime::HyperbolicCompactH3 {
            ui.add(egui::Slider::new(&mut self.choice.layers, 0..=2).text("Cell shells"));
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.small("Ball radius");
                for (r, label) in [(140, "1.4"), (180, "1.8"), (220, "2.2")] {
                    ui.selectable_value(&mut self.choice.radius_centi, r, label);
                }
            });
            if cls.has_ideal_vertices {
                ui.small(
                    "Ideal vertices lie at infinity. The ball clips cells at the chosen distance.",
                );
            }
        }
        ui.small("Reference geometry drawn in the Poincaré ball. Sectional curvature K = −1.");
        self.choice != old
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
        if self
            .cache
            .as_ref()
            .is_none_or(|(key, _)| *key != self.choice)
        {
            let [p, q, r] = self.choice.symbol;
            let built = build_atlas_h3(p, q, r, self.choice.layers, self.choice.radius_centi);
            self.cache = Some((self.choice, built));
        }
        match &self.cache.as_ref().unwrap().1 {
            Ok(built) => built.scenes(),
            Err(err) => (
                empty_scene("H³ patch unavailable", err),
                placeholder_mesh(),
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
    fn all_regular_h3_atlas_choices_build_a_visible_patch() {
        let mut panel = Hyperbolic3dPanel::default();
        let mut count = 0;
        for p in 3..=6 {
            for q in 3..=6 {
                for r in 3..=6 {
                    if ReflectionGroup::h3(p, q, r).is_err() {
                        continue;
                    }
                    panel.choice.symbol = [p, q, r];
                    for (layers, radius) in [(0, 140), (1, 180), (2, 220)] {
                        panel.choice.layers = layers;
                        panel.choice.radius_centi = radius;
                        let (_, mesh, _) = panel.build_scenes();
                        let built = panel.cache.as_ref().unwrap().1.as_ref().unwrap();
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
