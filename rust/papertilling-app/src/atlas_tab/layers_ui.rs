//! All 80 ITE Vol. E Layer groups panel with live Bieberbach cohomology & 3D slab preview.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::space3d::nets3d::build_layer_slab_patch;
use papertilling_core::space3d::{analyze_layer_group, list_layer_groups};

use crate::mesh3d::atlas_mesh::mesh_from_net3d;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{CirclePrim, LinePrim, Scene2D, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;

pub struct LayersPanelState {
    pub filter: String,
    pub selected_num: u8,
    pub summary: Option<BieberbachSummary>,
    cached_num: u8,
}

impl Default for LayersPanelState {
    fn default() -> Self {
        Self {
            filter: String::new(),
            selected_num: 1,
            summary: analyze_layer_group(1).ok(),
            cached_num: 1,
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

    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        self.ensure_cached();
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

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.ensure_cached();
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
        let patch = build_layer_slab_patch(self.selected_num).unwrap();
        let mesh = mesh_from_net3d(&patch);
        let circles: Vec<CirclePrim> = patch
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.055,
                min_px: 3.0,
                max_px: 9.0,
                fill: if patch.site_kinds[i] == 0 { [42, 92, 168, 255] } else { [214, 92, 52, 255] },
                stroke: ([255, 255, 255, 255], 1.0),
            })
            .collect();
        let lines: Vec<LinePrim> = patch
            .bonds
            .iter()
            .map(|&(u, v, _)| LinePrim {
                a: [patch.positions[u][0] as f32, patch.positions[u][1] as f32],
                b: [patch.positions[v][0] as f32, patch.positions[v][1] as f32],
                color: [44, 62, 88, 200],
                width: 1.4,
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
