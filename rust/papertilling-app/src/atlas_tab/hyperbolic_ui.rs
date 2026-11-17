//! Hyperbolic `H^2` `{p, q}` tilings, `(p.q.p.q)` kagome, Poincare geodesics, Klein chords,
//! 3D Minkowski hyperboloid bowl, wedge disclinations, and `kappa(D) < 0` inspector.

use std::f64::consts::PI;

use papertilling_core::hyperbolic::{build_hyperbolic_tiling, HyperbolicStyle};

use crate::mesh3d::atlas_mesh::mesh_from_hyperboloid;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{compute_bbox, ring_tint_rgba, CirclePrim, LinePrim, PolyPrim, Scene2D, ScenePrimitives};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiskProjection {
    Poincare,
    Klein,
}

pub struct HyperbolicPanelState {
    pub p: usize,
    pub q: usize,
    pub style: HyperbolicStyle,
    pub layers: usize,
    pub wedge_delta: i32,
    pub disk_proj: DiskProjection,
    pub status: String,
}

impl Default for HyperbolicPanelState {
    fn default() -> Self {
        Self {
            p: 7,
            q: 3,
            style: HyperbolicStyle::RegularPQ,
            layers: 2,
            wedge_delta: 0,
            disk_proj: DiskProjection::Poincare,
            status: String::new(),
        }
    }
}

impl HyperbolicPanelState {
    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal_wrapped(|ui| {
            ui.small("Presets:");
            for (lp, lq, st, lbl) in [
                (7, 3, HyperbolicStyle::RegularPQ, "{7,3}"),
                (3, 7, HyperbolicStyle::RegularPQ, "{3,7}"),
                (7, 3, HyperbolicStyle::RectifiedKagome, "(3.7.3.7) Kagome"),
                (5, 4, HyperbolicStyle::RegularPQ, "{5,4}"),
                (8, 3, HyperbolicStyle::RegularPQ, "{8,3}"),
            ] {
                if ui.small_button(lbl).clicked() {
                    self.p = lp;
                    self.q = lq;
                    self.style = st;
                    dirty = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.small("p:");
            dirty |= ui.add(egui::DragValue::new(&mut self.p).range(3..=12)).changed();
            ui.small("q:");
            dirty |= ui.add(egui::DragValue::new(&mut self.q).range(3..=12)).changed();
            ui.small("Layers:");
            dirty |= ui.add(egui::DragValue::new(&mut self.layers).range(1..=4)).changed();
        });
        ui.horizontal_wrapped(|ui| {
            ui.small("Tiling:");
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::RegularPQ, "Regular {p,q}").changed();
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::RectifiedKagome, "Kagome").changed();
            dirty |= ui.selectable_value(&mut self.style, HyperbolicStyle::DualQP, "Dual {q,p}").changed();
        });
        ui.horizontal_wrapped(|ui| {
            ui.small("2D Disk:");
            dirty |= ui.selectable_value(&mut self.disk_proj, DiskProjection::Poincare, "Poincare Geodesics").changed();
            dirty |= ui.selectable_value(&mut self.disk_proj, DiskProjection::Klein, "Klein Chords").changed();
        });
        ui.horizontal(|ui| {
            ui.small("Central Wedge Δp:");
            for d in [-1, 0, 1] {
                dirty |= ui.selectable_value(&mut self.wedge_delta, d, format!("{d:+}")).changed();
            }
        });
        if !self.status.is_empty() {
            ui.small(&self.status);
        }
        dirty
    }

    pub fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        let hyp = match build_hyperbolic_tiling(self.p, self.q, self.style, self.layers, self.wedge_delta) {
            Ok(h) => h,
            Err(e) => {
                self.status = e;
                return (
                    Scene2D::VectorScene {
                        primitives: ScenePrimitives {
                            polygons: Vec::new(),
                            lines: Vec::new(),
                            circles: Vec::new(),
                            labels: Vec::new(),
                            site_circle_indices: Vec::new(),
                            edge_line_indices: Vec::new(),
                            title: "Non-hyperbolic (p, q)".to_string(),
                            note: self.status.clone(),
                            bbox: ([-1.0, -1.0], [1.0, 1.0]),
                            site_count: 0,
                            bond_count: 0,
                        },
                    },
                    GpuMeshData { radius: 1.0, ..Default::default() },
                    self.status.clone(),
                );
            }
        };
        self.status = format!(
            "{} | κ(D) = {:.4} ({:?}) | PSU(1,1) |Tr/2|={:.3}",
            hyp.symbol_label, hyp.curvature_kappa, hyp.geometry_kind, hyp.psu11_half_trace
        );

        let mut lines = Vec::new();
        // Unit boundary circle S^1_infinity:
        let rim = 72usize;
        for k in 0..rim {
            let t0 = 2.0 * PI * (k as f64) / (rim as f64);
            let t1 = 2.0 * PI * ((k + 1) as f64) / (rim as f64);
            lines.push(LinePrim {
                a: [t0.cos() as f32, t0.sin() as f32],
                b: [t1.cos() as f32, t1.sin() as f32],
                color: [110, 130, 160, 180],
                width: 1.2,
                dashed: true,
                arrow_mid: false,
            });
        }

        let pts_2d = match self.disk_proj {
            DiskProjection::Poincare => &hyp.poincare_pts,
            DiskProjection::Klein => &hyp.klein_pts,
        };
        let polygons: Vec<PolyPrim> = hyp
            .faces
            .iter()
            .enumerate()
            .filter_map(|(fi, f)| {
                let p_f: Vec<[f32; 2]> = f.iter().filter_map(|&u| pts_2d.get(u).map(|p| [p[0] as f32, p[1] as f32])).collect();
                (p_f.len() >= 3).then(|| PolyPrim {
                    fill: if fi == 0 && hyp.wedge_delta != 0 {
                        [245, 130, 96, 150]
                    } else {
                        ring_tint_rgba(f.len())
                    },
                    stroke: None,
                    points: p_f,
                })
            })
            .collect();

        match self.disk_proj {
            DiskProjection::Poincare => {
                for arc in &hyp.edge_arcs {
                    for w in arc.windows(2) {
                        lines.push(LinePrim {
                            a: [w[0][0] as f32, w[0][1] as f32],
                            b: [w[1][0] as f32, w[1][1] as f32],
                            color: [32, 48, 76, 230],
                            width: 1.4,
                            dashed: false,
                            arrow_mid: false,
                        });
                    }
                }
            }
            DiskProjection::Klein => {
                for &(u, v) in &hyp.edges {
                    lines.push(LinePrim {
                        a: [pts_2d[u][0] as f32, pts_2d[u][1] as f32],
                        b: [pts_2d[v][0] as f32, pts_2d[v][1] as f32],
                        color: [32, 48, 76, 230],
                        width: 1.4,
                        dashed: false,
                        arrow_mid: false,
                    });
                }
            }
        }
        let circles: Vec<CirclePrim> = pts_2d
            .iter()
            .map(|p| CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.018,
                min_px: 1.5,
                max_px: 5.5,
                fill: [28, 78, 152, 230],
                stroke: ([255, 255, 255, 220], 0.5),
            })
            .collect();

        let prim = ScenePrimitives {
            polygons,
            lines,
            circles,
            labels: Vec::new(),
            site_circle_indices: Vec::new(),
            edge_line_indices: Vec::new(),
            title: format!("Hyperbolic {} ({:?})", hyp.symbol_label, self.disk_proj),
            note: format!("κ(D) = {:.4}, faces = {}, vertices = {}", hyp.curvature_kappa, hyp.faces.len(), hyp.poincare_pts.len()),
            bbox: compute_bbox([(-1.05, -1.05), (1.05, 1.05)].into_iter().map(|(x, y)| [x, y])),
            site_count: hyp.poincare_pts.len(),
            bond_count: hyp.edges.len(),
        };
        let mesh = mesh_from_hyperboloid(&hyp);
        let cap = format!("Hyperboloid H² {} ({} faces)", hyp.symbol_label, hyp.faces.len());
        (Scene2D::VectorScene { primitives: prim }, mesh, cap)
    }
}
