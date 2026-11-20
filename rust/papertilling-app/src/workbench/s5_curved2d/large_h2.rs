//! Large H² patches via `build_hyperbolic_large` with a choice of kernel (graph combinatorial, spatial hash in the
//! Poincaré disk, Dunham hyperboloid), drawn in the Poincaré, Klein or upper-half-plane model and, in 3D, on the
//! hemisphere model. Faces are tinted by shell (vertex-completion layer).

use papertilling_core::hyperbolic::boundary_counts::vertex_completion_counts_combinatorial;
use papertilling_core::noneuclidean2d::{H2Kernel, H2SurfaceKind, build_hyperbolic_large, map_poincare_to_surface};
use serde_json::{Value, json};

use super::draw::{DiskModel, h2_picture};
use super::growth_table::{GROWTH_HEADERS, non_disk_flags, patch_layer_stats, stats_to_rows};
use super::{MAX_PQ, S5Out, delta_badge, get_usize, pq_controls};
use crate::mesh3d::wb_mesh::{mesh_from_polygons, wb_face_tint};
use crate::workbench::scene_helpers::heat_rgba;

/// Default and upper bound of the target face count (the default stays small; the slider reaches 100k).
pub(crate) const DEFAULT_TARGET: usize = 5_000;
pub(crate) const MAX_DEFAULT_TARGET: usize = 20_000;
pub(crate) const MAX_TARGET: usize = 100_000;
/// At most this many faces (the innermost) go into the 3D hemisphere mesh.
pub(crate) const MAX_3D_FACES: usize = 30_000;

pub(crate) const KERNELS: [(H2Kernel, &str); 3] = [
    (H2Kernel::GraphCombinatorial, "graph combinatorial"),
    (H2Kernel::SpatialHashPoincare, "spatial hash (Poincaré disk)"),
    (H2Kernel::DunhamHyperboloid, "Dunham hyperboloid"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LargeJob {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) target: usize,
    pub(crate) kernel: usize,
    pub(crate) model: DiskModel,
}

pub(crate) struct LargeState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) target: usize,
    pub(crate) kernel: usize,
    pub(crate) model: DiskModel,
}

impl Default for LargeState {
    fn default() -> Self {
        Self { p: 7, q: 3, target: DEFAULT_TARGET, kernel: 0, model: DiskModel::Poincare }
    }
}

impl LargeState {
    pub(crate) fn job(&self) -> LargeJob {
        LargeJob { p: self.p, q: self.q, target: self.target, kernel: self.kernel, model: self.model }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = pq_controls(ui, &mut self.p, &mut self.q, 3);
        ui.label(delta_badge(self.p, self.q));
        ui.horizontal(|ui| {
            ui.label("Kernel:");
            egui::ComboBox::from_id_salt("wb_s5_kernel").selected_text(KERNELS[self.kernel.min(2)].1).show_ui(ui, |ui| {
                for (i, (_, label)) in KERNELS.iter().enumerate() {
                    rebuild |= ui.selectable_value(&mut self.kernel, i, *label).changed();
                }
            });
        });
        ui.label("Target faces:");
        let slider = egui::Slider::new(&mut self.target, 1..=MAX_TARGET).logarithmic(true);
        let r = ui.add(slider);
        // rebuild when a drag ends or on a click / typed value, not on every drag step
        rebuild |= r.drag_stopped() || (r.changed() && !r.dragged());
        if self.target > MAX_DEFAULT_TARGET {
            ui.small("Above 20 000 faces the 2D canvas redraws slowly.");
        }
        ui.horizontal(|ui| {
            ui.label("2D model:");
            egui::ComboBox::from_id_salt("wb_s5_model").selected_text(self.model.label()).show_ui(ui, |ui| {
                for m in DiskModel::ALL {
                    rebuild |= ui.selectable_value(&mut self.model, m, m.label()).changed();
                }
            });
        });
        rebuild
    }

    pub(super) fn save(&self) -> Value {
        json!({ "p": self.p, "q": self.q, "target": self.target, "kernel": self.kernel, "model": format!("{:?}", self.model) })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 3, MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 3, MAX_PQ) {
            self.q = x;
        }
        if let Some(x) = get_usize(v, "target", 1, MAX_TARGET) {
            self.target = x;
        }
        if let Some(x) = get_usize(v, "kernel", 0, KERNELS.len() - 1) {
            self.kernel = x;
        }
        if let Some(m) = v.get("model").and_then(Value::as_str).and_then(|s| DiskModel::ALL.into_iter().find(|m| format!("{m:?}") == s)) {
            self.model = m;
        }
    }
}

pub(crate) fn build(j: &LargeJob) -> Result<S5Out, String> {
    if j.p > MAX_PQ || j.q > MAX_PQ {
        return Err(format!("{{{},{}}} refused: this panel offers p, q <= {MAX_PQ}", j.p, j.q));
    }
    if !(1..=MAX_TARGET).contains(&j.target) {
        return Err(format!("target of {} faces refused: allowed 1..={MAX_TARGET}", j.target));
    }
    let (kernel, kname) = *KERNELS.get(j.kernel).ok_or_else(|| format!("unknown kernel #{}", j.kernel))?;
    let patch = build_hyperbolic_large(j.p, j.q, j.target, kernel)?;
    let s = patch.summary();
    let faces: Vec<Vec<usize>> = patch.faces.iter().zip(&patch.active).filter(|(_, a)| **a).map(|(f, _)| f.clone()).collect();
    let shells: Vec<usize> = patch.face_shell.iter().zip(&patch.active).filter(|(_, a)| **a).map(|(s, _)| *s).collect();
    let max_shell = s.max_shell.max(1) as f64;
    let tint = |fi: usize| {
        let mut c = heat_rgba(shells.get(fi).copied().unwrap_or(0) as f64, 0.0, max_shell);
        c[3] = 190;
        c
    };
    let title = format!("S5 large {{{},{}}} patch", j.p, j.q);
    let note = if j.model.chords_are_geodesics() {
        format!("{}; faces drawn through their vertices (chords are geodesics in this model)", j.model.label())
    } else {
        format!("{}; faces drawn through their vertices (straight chords, not geodesic arcs)", j.model.label())
    };
    let (scene2d, dropped) = h2_picture(&patch.poincare_pts, &faces, tint, j.model, false, &title, &note);

    let pts3: Vec<[f64; 3]> =
        patch.poincare_pts.iter().map(|&z| map_poincare_to_surface(z, H2SurfaceKind::Hemisphere).unwrap_or([f64::NAN; 3])).collect();
    let faces3 = &faces[..faces.len().min(MAX_3D_FACES)];
    let mesh = mesh_from_polygons(&pts3, faces3, &[], wb_face_tint);

    let stats = patch_layer_stats(&faces, &shells, s.max_shell);
    let stop_msg = match vertex_completion_counts_combinatorial(j.p, j.q, s.max_shell) {
        Some(exact) if stats.len() == s.max_shell + 1 && (stats[s.max_shell].new_cells as u64) < exact[s.max_shell] => {
            format!(
                "Stopped at target {} faces: layer {} is partial ({} of {} faces built; complete through layer {}) [COMPUTATIONAL EVIDENCE].",
                j.target,
                s.max_shell,
                stats[s.max_shell].new_cells,
                exact[s.max_shell],
                s.max_shell.saturating_sub(1)
            )
        }
        _ => format!(
            "Stopped at target {} faces after completing layer {} ({} faces) [COMPUTATIONAL EVIDENCE].",
            j.target, s.max_shell, s.total_faces
        ),
    };

    let rows = vec![
        (String::from("Kernel"), kname.to_string()),
        (String::from("Target / built faces"), format!("{} / {}", j.target, s.total_faces)),
        (String::from("Active faces"), s.n_active_faces.to_string()),
        (String::from("V / E"), format!("{} / {}", s.n_vertices, s.n_edges)),
        (String::from("V − E + F"), (s.n_vertices as i64 - s.n_edges as i64 + s.total_faces as i64).to_string()),
        (String::from("Outermost shell"), s.max_shell.to_string()),
    ];
    let mut notes = vec![
        stop_msg.clone(),
        String::from("Counts of the built patch [COMPUTATIONAL EVIDENCE]; a disk patch has V − E + F = 1."),
        String::from("3D: hemisphere model of H² (conformal, not an isometric embedding) [PROVED]."),
    ];
    notes.extend(non_disk_flags(&stats, None));
    if faces.len() > MAX_3D_FACES {
        notes.push(format!("3D shows the innermost {MAX_3D_FACES} of {} faces.", faces.len()));
    }
    if dropped > 0 {
        notes.push(format!("{dropped} face(s) outside the 2D window (|w| <= {}) were not drawn.", super::draw::CLIP_R));
    }
    let table = (!stats.is_empty()).then(|| (GROWTH_HEADERS.to_vec(), stats_to_rows(&stats)));
    let caption = format!("{{{},{}}} patch on the hemisphere model: {} faces", j.p, j.q, faces3.len());
    Ok(S5Out {
        scene2d,
        mesh,
        caption,
        rows,
        notes,
        stopped: Some(stop_msg),
        table,
        extra_table: None,
        svg: None,
    })
}
