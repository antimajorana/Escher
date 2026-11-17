//! All 230 3D Space Groups (`73 symmorphic + 157 nonsymmorphic`), canonical 3D crystal nets,
//! and `#123 P4/mmm` vs `#136 P4_2/mnm` forced `4_2` screw-dispiration visualizer.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::space3d::nets3d::{build_net_3d, CanonicalNet3d, LineDefect3d};
use papertilling_core::space3d::{analyze_space_group_3d, list_space_groups_3d};

use crate::mesh3d::atlas_mesh::mesh_from_net3d;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{CirclePrim, LinePrim, Scene2D, ScenePrimitives, compute_bbox};
use crate::theme::ThemeTokens;

pub struct Space3dPanelState {
    pub filter: String,
    pub selected_sg: u16,
    pub selected_net: CanonicalNet3d,
    pub selected_defect: LineDefect3d,
    pub summary: Option<BieberbachSummary>,
    cached_sg: u16,
}

impl Default for Space3dPanelState {
    fn default() -> Self {
        Self {
            filter: String::new(),
            selected_sg: 136,
            selected_net: CanonicalNet3d::Rutile136,
            selected_defect: LineDefect3d::ScrewDispiration42_136,
            summary: analyze_space_group_3d(136).ok(),
            cached_sg: 136,
        }
    }
}

impl Space3dPanelState {
    pub fn ensure_cached(&mut self) {
        if self.cached_sg != self.selected_sg || self.summary.is_none() {
            self.cached_sg = self.selected_sg;
            self.summary = analyze_space_group_3d(self.selected_sg).ok();
        }
    }

    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        self.ensure_cached();
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("#123 P4/mmm (Pure 90°)").clicked() {
                self.selected_sg = 123;
                self.selected_net = CanonicalNet3d::Tetragonal123;
                self.selected_defect = LineDefect3d::PureDisclination90_123;
                dirty = true;
            }
            if ui.small_button("#136 P4_2/mnm (4_2 Screw-Disp.)").clicked() {
                self.selected_sg = 136;
                self.selected_net = CanonicalNet3d::Rutile136;
                self.selected_defect = LineDefect3d::ScrewDispiration42_136;
                dirty = true;
            }
        });
        ui.horizontal(|ui| {
            ui.small("Filter (230 Space Groups):");
            ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(120.0));
        });
        let q = self.filter.trim().to_lowercase();
        egui::ScrollArea::vertical().id_salt("atlas_sg230_scroll").max_height(155.0).show(ui, |ui| {
            for sg in list_space_groups_3d() {
                if !q.is_empty()
                    && !sg.symbol.to_lowercase().contains(&q)
                    && !sg.schoenflies.to_lowercase().contains(&q)
                    && !sg.number.to_string().contains(&q)
                {
                    continue;
                }
                let tag = if sg.is_split { "split" } else { "nonsym" };
                let lbl = format!("#{:>3} {} [{}] ({tag}, {})", sg.number, sg.symbol, sg.schoenflies, sg.ab_full_label);
                if ui.selectable_label(self.selected_sg == sg.number, lbl).clicked() {
                    self.selected_sg = sg.number;
                    dirty = true;
                }
            }
        });

        ui.separator();
        ui.strong("3D Crystal Net & Line Defect");
        egui::ComboBox::from_id_salt("atlas_3d_net")
            .selected_text(self.selected_net.label())
            .show_ui(ui, |ui| {
                for n in CanonicalNet3d::ALL {
                    if ui.selectable_value(&mut self.selected_net, n, n.label()).changed() {
                        dirty = true;
                    }
                }
            });
        egui::ComboBox::from_id_salt("atlas_3d_def")
            .selected_text(self.selected_defect.label())
            .show_ui(ui, |ui| {
                for d in LineDefect3d::ALL {
                    if ui.selectable_value(&mut self.selected_defect, d, d.label()).changed() {
                        dirty = true;
                    }
                }
            });
        dirty
    }

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.ensure_cached();
        ui.strong("3D Space Group Cohomology");
        if let Some(s) = &self.summary {
            let (st, col) = if s.is_split {
                ("split ([ν]=0)", tokens.status_ok)
            } else {
                ("nonsymmorphic ([ν]≠0)", tokens.status_warn)
            };
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!("#{} {} | |P|={} | ", s.number, s.symbol, s.point_order));
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
        ui.separator();
        ui.strong("Active 3D Net & Defect");
        ui.small(format!("Net: {}", self.selected_net.label()));
        ui.small(format!("Line defect: {}", self.selected_defect.label()));
    }

    pub fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        self.ensure_cached();
        let net = build_net_3d(self.selected_net, self.selected_defect);
        let mesh = mesh_from_net3d(&net);
        let circles: Vec<CirclePrim> = net
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.055,
                min_px: 2.5,
                max_px: 8.5,
                fill: match net.site_kinds[i] % 3 {
                    0 => [42, 92, 168, 255],
                    1 => [214, 92, 52, 255],
                    _ => [42, 148, 96, 255],
                },
                stroke: ([255, 255, 255, 255], 0.8),
            })
            .collect();
        let lines: Vec<LinePrim> = net
            .bonds
            .iter()
            .map(|&(u, v, bk)| LinePrim {
                a: [net.positions[u][0] as f32, net.positions[u][1] as f32],
                b: [net.positions[v][0] as f32, net.positions[v][1] as f32],
                color: if bk == 1 { [224, 58, 32, 255] } else { [44, 62, 88, 190] },
                width: if bk == 1 { 2.4 } else { 1.3 },
                dashed: false,
                arrow_mid: false,
            })
            .collect();
        let pts: Vec<[f32; 2]> = circles.iter().map(|c| c.center).collect();
        let prim = ScenePrimitives {
            polygons: Vec::new(),
            lines,
            circles,
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: format!("3D Net (xy-projection): {}", net.name),
            note: net.defect_label.clone(),
            bbox: compute_bbox(pts.into_iter()),
            site_count: net.positions.len(),
            bond_count: net.bonds.len(),
        };
        (Scene2D::VectorScene { primitives: prim }, mesh, net.info)
    }
}
