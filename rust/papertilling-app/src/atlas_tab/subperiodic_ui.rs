//! 0D Rosettes (`C_n, D_n`) and 1D Frieze groups (`p1..p2mg`) panel & 2D/3D scene builder.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::subperiodic::{
    analyze_frieze_group, build_frieze_patch, build_rosette, FriezeDefect, FriezePatch,
    FriezeTopology, FRIEZE_SYMBOLS,
};

use crate::mesh3d::atlas_mesh::mesh_from_frieze3d;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{compute_bbox, CirclePrim, LinePrim, Scene2D, ScenePrimitives};

pub struct SubperiodicPanelState {
    pub is_frieze: bool,
    pub dihedral: bool,
    pub rosette_n: usize,
    pub wedge_delta: i32,
    pub mirror_wall: bool,
    pub frieze_idx: usize,
    pub cells: usize,
    pub topology: FriezeTopology,
    pub defect: FriezeDefect,
    pub summary: Option<BieberbachSummary>,
    pub summary_idx: usize,
}

impl Default for SubperiodicPanelState {
    fn default() -> Self {
        let sym = FRIEZE_SYMBOLS[6]; // default p2mg (nonsymmorphic)
        let summary = analyze_frieze_group(sym).ok().map(|(s, _)| s);
        Self {
            is_frieze: true,
            dihedral: true,
            rosette_n: 5,
            wedge_delta: 0,
            mirror_wall: false,
            frieze_idx: 6,
            cells: 10,
            topology: FriezeTopology::Mobius,
            defect: FriezeDefect::GlideDispiration,
            summary,
            summary_idx: 6,
        }
    }
}

impl SubperiodicPanelState {
    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal(|ui| {
            dirty |= ui.selectable_value(&mut self.is_frieze, false, "0D Rosettes (Cₙ, Dₙ)").changed();
            dirty |= ui.selectable_value(&mut self.is_frieze, true, "1D Friezes (7 groups)").changed();
        });
        ui.separator();

        if !self.is_frieze {
            ui.horizontal(|ui| {
                dirty |= ui.selectable_value(&mut self.dihedral, false, "Cyclic Cₙ").changed();
                dirty |= ui.selectable_value(&mut self.dihedral, true, "Dihedral Dₙ").changed();
            });
            ui.horizontal(|ui| {
                ui.small("Order n:");
                dirty |= ui.add(egui::Slider::new(&mut self.rosette_n, 2..=12)).changed();
            });
            ui.horizontal(|ui| {
                ui.small("Wedge Δn:");
                for d in [-1, 0, 1] {
                    dirty |= ui.selectable_value(&mut self.wedge_delta, d, format!("{d:+}")).changed();
                }
            });
            if self.dihedral {
                dirty |= ui.checkbox(&mut self.mirror_wall, "Mirror domain wall").changed();
            }
            let r = build_rosette(self.dihedral, self.rosette_n, self.wedge_delta, self.mirror_wall);
            ui.small(format!("Point group {} | |P|={} | Gᵃᵇ = {}", r.symbol, r.point_order, r.ab_label));
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.small("Group:");
                for (i, &sym) in FRIEZE_SYMBOLS.iter().enumerate() {
                    if ui.selectable_label(self.frieze_idx == i, sym).clicked() {
                        self.frieze_idx = i;
                        dirty = true;
                    }
                }
            });
            if self.summary_idx != self.frieze_idx || self.summary.is_none() {
                self.summary_idx = self.frieze_idx;
                self.summary = analyze_frieze_group(FRIEZE_SYMBOLS[self.frieze_idx]).ok().map(|(s, _)| s);
            }
            ui.horizontal_wrapped(|ui| {
                ui.small("Quotient:");
                dirty |= ui.selectable_value(&mut self.topology, FriezeTopology::OpenStrip, "Open").changed();
                dirty |= ui.selectable_value(&mut self.topology, FriezeTopology::Cylinder, "Cylinder").changed();
                dirty |= ui.selectable_value(&mut self.topology, FriezeTopology::Mobius, "Möbius").changed();
            });
            ui.horizontal_wrapped(|ui| {
                ui.small("Defect:");
                dirty |= ui.selectable_value(&mut self.defect, FriezeDefect::None, "Pristine").changed();
                dirty |= ui.selectable_value(&mut self.defect, FriezeDefect::HalfTurn180, "180° Twist").changed();
                dirty |= ui.selectable_value(&mut self.defect, FriezeDefect::GlideDispiration, "Glide Disp.").changed();
            });
            ui.horizontal(|ui| {
                ui.small("Cells:");
                dirty |= ui.add(egui::Slider::new(&mut self.cells, 4..=20)).changed();
            });
            if let Some(s) = &self.summary {
                let st = if s.is_split { "split ([ν]=0)" } else { "nonsymmorphic ([ν]≠0)" };
                ui.small(format!("#{} {} | |P|={} | {st}", s.number, s.symbol, s.point_order));
                ui.small(format!("H²(P, Z) = {} | (G⁺)ᵃᵇ = {} | Gᵃᵇ = {}", s.h2_label, s.ab_plus_label, s.ab_full_label));
            }
        }
        dirty
    }

    pub fn build_scenes(&self) -> (Scene2D, GpuMeshData, String) {
        if !self.is_frieze {
            let r = build_rosette(self.dihedral, self.rosette_n, self.wedge_delta, self.mirror_wall);
            let lines: Vec<LinePrim> = r
                .edges
                .iter()
                .map(|&(u, v, ek)| LinePrim {
                    a: [r.positions[u][0] as f32, r.positions[u][1] as f32],
                    b: [r.positions[v][0] as f32, r.positions[v][1] as f32],
                    color: match ek {
                        1 => [216, 48, 36, 255],
                        2 => [34, 152, 72, 255],
                        _ => [36, 56, 88, 220],
                    },
                    width: if ek == 0 { 1.6 } else { 2.8 },
                    dashed: ek == 2,
                    arrow_mid: false,
                })
                .collect();
            let circles: Vec<CirclePrim> = r
                .positions
                .iter()
                .enumerate()
                .map(|(i, &p)| CirclePrim {
                    center: [p[0] as f32, p[1] as f32],
                    radius_world: if i == 0 { 0.08 } else { 0.06 },
                    min_px: 3.5,
                    max_px: 10.0,
                    fill: if i == 0 { [216, 62, 42, 255] } else { [42, 92, 168, 255] },
                    stroke: ([255, 255, 255, 255], 1.2),
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
                title: format!("0D Rosette {} (Δn={:+})", r.symbol, r.wedge_delta),
                note: format!("Gᵃᵇ = {}", r.ab_label),
                bbox: compute_bbox(pts.into_iter()),
                site_count: r.positions.len(),
                bond_count: r.edges.len(),
            };
            let frieze_dummy: FriezePatch = build_frieze_patch("p1", 6, FriezeTopology::OpenStrip, FriezeDefect::None).unwrap();
            let _ = frieze_dummy;
            let pts3: Vec<[f64; 3]> = r.positions.iter().map(|p| [p[0], p[1], 0.0]).collect();
            let fake_net = papertilling_core::space3d::nets3d::Net3dPatch {
                name: r.symbol.clone(),
                space_group_num: 1,
                defect_label: format!("G^ab={}", r.ab_label),
                positions: pts3,
                bonds: r.edges.clone(),
                site_kinds: (0..r.positions.len()).map(|i| u8::from(i == 0)).collect(),
                defect_axis: None,
                info: format!("0D Rosette {}", r.symbol),
            };
            let mesh = crate::mesh3d::atlas_mesh::mesh_from_net3d(&fake_net);
            (Scene2D::VectorScene { primitives: prim }, mesh, format!("0D Rosette {}", r.symbol))
        } else {
            let sym = FRIEZE_SYMBOLS[self.frieze_idx];
            let patch = build_frieze_patch(sym, self.cells, self.topology, self.defect).unwrap();
            let lines: Vec<LinePrim> = patch
                .edges
                .iter()
                .map(|&(u, v, ek)| LinePrim {
                    a: [patch.pos2d[u][0] as f32, patch.pos2d[u][1] as f32],
                    b: [patch.pos2d[v][0] as f32, patch.pos2d[v][1] as f32],
                    color: match ek {
                        1 => [216, 48, 36, 255],
                        2 => [34, 152, 72, 220],
                        _ => [36, 56, 88, 220],
                    },
                    width: if ek == 0 { 1.4 } else { 2.4 },
                    dashed: ek == 2,
                    arrow_mid: false,
                })
                .collect();
            let circles: Vec<CirclePrim> = patch
                .pos2d
                .iter()
                .map(|&p| CirclePrim {
                    center: [p[0] as f32, p[1] as f32],
                    radius_world: 0.05,
                    min_px: 2.5,
                    max_px: 7.0,
                    fill: [42, 92, 168, 255],
                    stroke: ([255, 255, 255, 255], 0.8),
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
                title: format!("1D Frieze {} — {:?}", patch.iucr_name, patch.topology),
                note: format!("H²={}, Gᵃᵇ={}", patch.summary.h2_label, patch.summary.ab_full_label),
                bbox: compute_bbox(pts.into_iter()),
                site_count: patch.pos2d.len(),
                bond_count: patch.edges.len(),
            };
            let mesh = mesh_from_frieze3d(&patch);
            (
                Scene2D::VectorScene { primitives: prim },
                mesh,
                format!("1D Frieze {} ({:?})", patch.symbol, patch.topology),
            )
        }
    }
}
