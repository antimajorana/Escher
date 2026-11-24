//! All 80 ITE Vol. E Layer groups panel with live Bieberbach cohomology & 3D slab preview.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::space3d::{analyze_layer_group, list_layer_groups};

use super::crystal_charge_ui::{CrystalChargeUi, CrystalGroup};
use crate::mesh3d::atlas_mesh::mesh_from_net3d;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{CirclePrim, LinePrim, Scene2D, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;

pub struct LayersPanelState {
    pub filter: String,
    pub selected_num: u8,
    pub summary: Option<BieberbachSummary>,
    /// In-plane growth and stacking (C1 WP-U4, `layers_growth`).
    pub growth: super::layers_growth::LayerGrowthState,
    cached_num: u8,
    charges: CrystalChargeUi,
}

impl Default for LayersPanelState {
    fn default() -> Self {
        Self {
            filter: String::new(),
            selected_num: 1,
            summary: analyze_layer_group(1).ok(),
            growth: super::layers_growth::LayerGrowthState::default(),
            cached_num: 1,
            charges: CrystalChargeUi::default(),
        }
    }
}

impl LayersPanelState {
    pub fn ensure_cached(&mut self) {
        if self.cached_num != self.selected_num || self.summary.is_none() {
            self.cached_num = self.selected_num;
            self.summary = analyze_layer_group(self.selected_num).ok();
        }
    }

    pub fn show_controls(&mut self, ui: &mut egui::Ui, pristine_only: bool) -> bool {
        let mut dirty = false;
        self.ensure_cached();
        if !pristine_only {
            ui.horizontal_wrapped(|ui| {
                for (number, label) in [(4, "p11m mirror"), (5, "p11a glide")] {
                    if ui.small_button(label).clicked() {
                        self.selected_num = number;
                        dirty = true;
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            ui.small("Filter (80 Layer Groups):");
            ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(130.0));
        });
        let q = self.filter.trim().to_lowercase();
        egui::ScrollArea::vertical().id_salt("atlas_layers_scroll").max_height(190.0).show(ui, |ui| {
            for lg in list_layer_groups() {
                if !q.is_empty()
                    && !lg.symbol.to_lowercase().contains(&q)
                    && !lg.pointgroup.to_lowercase().contains(&q)
                    && !lg.number.to_string().contains(&q)
                {
                    continue;
                }
                let lbl = format!("#{:>2} {} ({}, {})", lg.number, lg.symbol, lg.pointgroup, lg.system);
                if ui.selectable_label(self.selected_num == lg.number, lbl).clicked() {
                    self.selected_num = lg.number;
                    dirty = true;
                }
            }
        });
        dirty |= super::layers_growth::show(ui, self);
        self.ensure_cached();
        dirty
    }

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens, pristine_only: bool) {
        self.ensure_cached();
        if !pristine_only {
            self.charges.show(ui, CrystalGroup::Layer(self.selected_num));
            ui.separator();
        }
        ui.strong("Layer Group Cohomology (ITE Vol. E)");
        if let Some(s) = &self.summary {
            let (st, col) = if s.is_split {
                ("split ([ν]=0)", tokens.status_ok)
            } else {
                ("nonsymmorphic ([ν]≠0)", tokens.status_warn)
            };
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!("Layer #{} {} | |P|={} | ", s.number, s.symbol, s.point_order));
                ui.colored_label(col, egui::RichText::new(st).strong());
            });
            ui.small(format!(
                "H²(P, Λ) = {} | (G⁺)ᵃᵇ = {} | Gᵃᵇ = {}",
                s.h2_label, s.ab_plus_label, s.ab_full_label
            ));
            if !s.torus_counts.is_empty() {
                let tor: Vec<String> = s.torus_counts.iter().map(|(n, c)| format!("N={n}: {c}")).collect();
                ui.small(format!("#Conj(G / NΛ): {}", tor.join(", ")));
            }
        }
    }

    pub fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        self.ensure_cached();
        let patch = self.growth.patch(self.selected_num);
        let mesh = mesh_from_net3d(&patch);
        let site_fill = |kind: u8| match kind % 4 {
            0 => [56, 112, 199, 255],
            1 => [217, 92, 51, 255],
            2 => [46, 163, 112, 255],
            _ => [148, 92, 184, 255],
        };
        let circles: Vec<CirclePrim> = patch
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.055,
                min_px: 3.0,
                max_px: 9.0,
                fill: site_fill(patch.site_kinds.get(i).copied().unwrap_or(0)),
                stroke: ([255, 255, 255, 255], 1.0),
            })
            .collect();
        let lines: Vec<LinePrim> = patch
            .bonds
            .iter()
            .map(|&(u, v, b_kind)| LinePrim {
                a: [patch.positions[u][0] as f32, patch.positions[u][1] as f32],
                b: [patch.positions[v][0] as f32, patch.positions[v][1] as f32],
                color: if b_kind == 1 { [242, 82, 36, 255] } else { [44, 62, 88, 200] },
                width: if b_kind == 1 { 2.2 } else { 1.4 },
                dashed: false,
                arrow_mid: false,
            })
            .collect();
        let pts: Vec<[f32; 2]> = circles.iter().map(|c| c.center).collect();
        let sum_note = self
            .summary
            .as_ref()
            .map(|s| format!("H²={}, Gᵃᵇ={}", s.h2_label, s.ab_full_label))
            .unwrap_or_default();
        let prim = ScenePrimitives {
            polygons: Vec::new(),
            lines,
            circles,
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: format!("{} (2D z-projection)", patch.name),
            note: sum_note,
            bbox: compute_bbox(pts.into_iter()),
            site_count: patch.positions.len(),
            bond_count: patch.bonds.len(),
        };
        (Scene2D::VectorScene { primitives: prim }, mesh, patch.info)
    }
}
