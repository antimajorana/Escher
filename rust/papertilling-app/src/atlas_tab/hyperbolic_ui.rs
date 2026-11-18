//! Hyperbolic `H^2` `{p, q}` tilings, `(p.q.p.q)` kagome, Poincare geodesics, Klein chords,
//! 3D Minkowski hyperboloid bowl, wedge disclinations, and `kappa(D) < 0` inspector.

use std::f64::consts::PI;

use papertilling_core::hyperbolic::inflation::{vertex_completion_counts_budget, ShellCounts};
use papertilling_core::hyperbolic::{build_hyperbolic_tiling, HyperbolicStyle};

use crate::mesh3d::atlas_mesh::mesh_from_hyperboloid;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::{compute_bbox, ring_tint_rgba, CirclePrim, LinePrim, PolyPrim, Scene2D, ScenePrimitives};
use crate::theme::ThemeTokens;

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
    /// Effective `{p, q}` whose shell table the inspector shows.
    pub shell_key: Option<(usize, usize)>,
    /// Uncapped BDF vertex-completion shell counts, at most [`SHELL_CACHE_MAX`] `{p, q}` keys
    /// (oldest dropped first), so Regular <-> Dual toggles and revisits do not recompute.
    pub shell_cache: Vec<((usize, usize), Result<ShellCounts, String>)>,
    /// Which shells the drawn patch contains (from the tiling's `growth_rule`).
    pub drawn_patch_note: String,
}

/// Shells and face budget of the inspector's shell table (interactive, so kept small).
const SHELL_TABLE_MAX_N: usize = 6;
const SHELL_TABLE_FACE_BUDGET: usize = 20_000;
/// Number of `{p, q}` shell tables kept in [`HyperbolicPanelState::shell_cache`].
const SHELL_CACHE_MAX: usize = 16;

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
            shell_key: None,
            shell_cache: Vec::new(),
            drawn_patch_note: String::new(),
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
        dirty
    }

    pub fn show_inspector(&self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        ui.strong("Hyperbolic H² Metric & Status");
        let is_hyperbolic = (self.p.saturating_sub(2)) * (self.q.saturating_sub(2)) > 4;
        if !self.status.is_empty() {
            let col = if is_hyperbolic { tokens.status_ok } else { tokens.status_err };
            ui.colored_label(col, egui::RichText::new(&self.status).small());
        }
        ui.small(format!(
            "Schläfli {{{}, {}}} | Layers: {} | Wedge Δp: {:+}",
            self.p, self.q, self.layers, self.wedge_delta
        ));
        ui.small(format!("Projection: {:?} | Style: {:?}", self.disk_proj, self.style));
        self.show_shell_table(ui);
    }

    /// BDF vertex completion `S_{n+1} = S_n + U(S_n)` from one p-gon (arXiv:1805.02665, Sec. IV-V):
    /// exact counts from the uncapped counter, not from the drawn (capped) patch.
    fn show_shell_table(&self, ui: &mut egui::Ui) {
        let Some(((p, q), res)) = self.shell_key.and_then(|k| self.shell_cache.iter().find(|(c, _)| *c == k)) else {
            return;
        };
        ui.separator();
        ui.strong(format!("Vertex completion shells, {{{p},{q}}}, one {p}-gon seed"));
        let counts = match res {
            Ok(c) => c,
            Err(e) => {
                ui.small(e);
                return;
            }
        };
        egui::Grid::new("hyp_shell_table").num_columns(4).spacing([10.0, 2.0]).striped(true).show(ui, |ui| {
            for h in ["n", "|S_n|", "a_n", "a_n/a_{n-1}"] {
                ui.small(egui::RichText::new(h).strong());
            }
            ui.end_row();
            for s in &counts.shells {
                ui.small(s.n.to_string());
                ui.small(s.cumulative.to_string());
                ui.small(s.shell.to_string());
                ui.small(s.ratio.map_or("-".to_string(), |r| format!("{r:.5}")));
                ui.end_row();
            }
        });
        ui.small(format!(
            "λ+ = γ + √(γ²−1) = {:.5}, γ = ((p−2)(q−2)−2)/2: BDF Eq. 10–11 give it as the growth eigenvalue of the \
             boundary under vertex completion. That the shell ratio a_n/a_{{n-1}} approaches it is observed in this \
             table, not a formula from the paper; λ+ is not a count formula.",
            counts.lambda_plus
        ));
        if let Some(stop) = &counts.stopped {
            ui.small(format!("Table stops early: {stop}"));
        }
        if self.style == HyperbolicStyle::RectifiedKagome {
            ui.small("Kagome: counts are of the underlying {p,q} p-gons.");
        }
        if self.wedge_delta != 0 {
            ui.small("Wedge core Δp ≠ 0: the drawn patch is edge-BFS; this table does not describe it.");
        }
        ui.small(format!("Drawn patch: {}", self.drawn_patch_note));
    }

    pub fn build_scenes(&mut self) -> (Scene2D, GpuMeshData, String) {
        let (pe, qe) = if self.style == HyperbolicStyle::DualQP { (self.q, self.p) } else { (self.p, self.q) };
        if !self.shell_cache.iter().any(|(k, _)| *k == (pe, qe)) {
            let res = vertex_completion_counts_budget(pe, qe, SHELL_TABLE_MAX_N, SHELL_TABLE_FACE_BUDGET);
            if self.shell_cache.len() >= SHELL_CACHE_MAX {
                self.shell_cache.drain(..1); // drop the oldest table
            }
            self.shell_cache.push(((pe, qe), res));
        }
        self.shell_key = Some((pe, qe));
        let hyp = match build_hyperbolic_tiling(self.p, self.q, self.style, self.layers, self.wedge_delta) {
            Ok(h) => h,
            Err(e) => {
                self.status = e;
                self.drawn_patch_note.clear();
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
        self.drawn_patch_note = hyp.growth_rule.clone();
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
