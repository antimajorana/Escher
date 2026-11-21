//! Solver export: the periodic `nx x ny` supercell with its neighbour shells J1..J3 (each split into space-group bond
//! orbits) and the site permutations of the space group on the torus; written as extended XYZ and JSON.
//! File dialogs never open under `--smoke-test` or in tests (`export_to` is the dialog-free core).

use std::path::Path;

use papertilling_core::export::{SupercellSolverExport, export_cell_extxyz, export_crystal_extxyz, periodic_supercell_permutations};
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use super::point::PointOut;
use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;
use crate::workbench::WbCtx;
use crate::workbench::scene_helpers::vector_scene;

pub(super) const SCENE_TITLE: &str = "S2 solver supercell";
const MAX_N: usize = 6;
const MAX_SHELLS: usize = 3;

pub(crate) type SolverOut = SupercellSolverExport;

pub(crate) fn build(group: &str, cell: &Cell2D, nx: usize, ny: usize, shells: usize) -> Result<SolverOut, String> {
    if !(1..=MAX_N).contains(&nx) || !(1..=MAX_N).contains(&ny) || !(1..=MAX_SHELLS).contains(&shells) {
        return Err(format!("supercell {nx} x {ny} with {shells} shells refused: need 1..={MAX_N} cells and 1..={MAX_SHELLS} shells"));
    }
    periodic_supercell_permutations(group, cell, nx, ny, shells)
}

/// The three export payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportKind {
    SupercellXyz,
    DefectXyz,
    SolverJson,
}

impl ExportKind {
    fn button(self) -> &'static str {
        match self {
            Self::SupercellXyz => "Save supercell .extxyz",
            Self::DefectXyz => "Save point-defect patch .extxyz",
            Self::SolverJson => "Save shells + permutations .json",
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::SupercellXyz => "supercell.extxyz",
            Self::DefectXyz => "point_defects.extxyz",
            Self::SolverJson => "solver_export.json",
        }
    }
}

/// The text of one export (no file system access).
pub(crate) fn export_text(kind: ExportKind, cell: &Cell2D, out: Option<&SolverOut>, point: Option<&PointOut>) -> Result<String, String> {
    match kind {
        ExportKind::SupercellXyz => {
            let o = out.ok_or("build the supercell first")?;
            export_cell_extxyz(cell, o.nx, o.ny, None)
        }
        ExportKind::DefectXyz => {
            let p = point.ok_or("build a point-defect patch first (Point defects sub-view)")?;
            export_crystal_extxyz(&p.result.crystal, Some(&p.z()), Some(&p.result.species))
        }
        ExportKind::SolverJson => {
            let o = out.ok_or("build the supercell first")?;
            serde_json::to_string_pretty(o).map_err(|e| format!("JSON encoding failed: {e}"))
        }
    }
}

/// Write `text` to `path`; returns a status line.
pub(crate) fn export_to(path: &Path, text: &str) -> Result<String, String> {
    std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(format!("wrote {} ({} bytes)", path.display(), text.len()))
}

pub(crate) struct SolverState {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) shells: usize,
    pub(crate) out: Option<Box<SolverOut>>,
    pub(crate) status: String,
}

impl Default for SolverState {
    fn default() -> Self {
        Self { nx: 3, ny: 3, shells: 2, out: None, status: String::new() }
    }
}

impl SolverState {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = false;
        ui.horizontal(|ui| {
            ui.label("Supercell nx:");
            rebuild |= ui.add(egui::DragValue::new(&mut self.nx).range(1..=MAX_N)).changed();
            ui.label("ny:");
            rebuild |= ui.add(egui::DragValue::new(&mut self.ny).range(1..=MAX_N)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Neighbour shells (J1..J3):");
            rebuild |= ui.add(egui::DragValue::new(&mut self.shells).range(1..=MAX_SHELLS)).changed();
        });
        ui.small("Periodic torus; the 2D canvas shows the sites and the shell-1 bonds inside the box.");
        rebuild
    }

    pub(super) fn show_inspector(&mut self, ui: &mut egui::Ui, cx: &WbCtx, point: Option<&PointOut>) {
        if let Some(o) = &self.out {
            ui.label(format!("{} x {} supercell, {} sites, group {}", o.nx, o.ny, o.n_sites, o.group));
            egui::Grid::new("wb_s2_shell_grid").striped(true).show(ui, |ui| {
                for h in ["Shell", "Distance", "Bonds / cell", "Mean coord.", "Orbits"] {
                    ui.strong(h);
                }
                ui.end_row();
                for s in &o.shells.shells {
                    let per_cell: usize = s.orbits.iter().map(|b| b.bonds_per_cell).sum();
                    ui.label(format!("J{}", s.shell));
                    ui.label(format!("{:.4}", s.distance));
                    ui.label(per_cell.to_string());
                    ui.label(format!("{:.3}", s.mean_coordination));
                    ui.label(s.orbits.len().to_string());
                    ui.end_row();
                }
            });
            ui.label(format!("Shell classification {}", o.shells.provenance));
            let r = &o.report;
            ui.label(format!("Site permutation group: order {}, closed under composition: {}", r.order, r.group_closed));
            ui.label(format!("Permutations {}", r.provenance));
        } else {
            ui.label("Building the supercell ...");
        }
        ui.separator();
        for kind in [ExportKind::SupercellXyz, ExportKind::DefectXyz, ExportKind::SolverJson] {
            if ui.button(kind.button()).clicked() {
                self.status = self.export(kind, cx, point);
            }
        }
        if !self.status.is_empty() {
            ui.small(self.status.as_str());
        }
    }

    fn export(&self, kind: ExportKind, cx: &WbCtx, point: Option<&PointOut>) -> String {
        let text = match export_text(kind, cx.cell, self.out.as_deref(), point) {
            Ok(t) => t,
            Err(e) => return format!("refused: {e}"),
        };
        if cx.smoke || cfg!(test) {
            return format!("{} ready ({} bytes); file dialog skipped", kind.file_name(), text.len());
        }
        let dialog = rfd::FileDialog::new().set_title(kind.button()).set_file_name(kind.file_name());
        match dialog.save_file() {
            Some(path) => export_to(&path, &text).unwrap_or_else(|e| e),
            None => String::from("export cancelled"),
        }
    }

    pub(super) fn scene(&self) -> Scene2D {
        let Some(o) = &self.out else { return empty_scene(SCENE_TITLE, "building ...") };
        let pts: Vec<[f64; 2]> = o.positions.iter().map(|p| [p[0], p[1]]).collect();
        let d1 = o.shell_distances.first().copied().unwrap_or(0.0);
        let inside = |&(i, j): &(usize, usize)| match (pts.get(i), pts.get(j)) {
            (Some(p), Some(q)) => (p[0] - q[0]).hypot(p[1] - q[1]) <= d1 * (1.0 + 1e-6),
            _ => false,
        };
        let edges: Vec<(usize, usize)> =
            o.shell_edges.first().map_or_else(Vec::new, |e| e.iter().map(|&(i, j, _)| (i, j)).filter(inside).collect());
        let note = format!("{} sites, {} shell-1 bonds drawn (wrap-around bonds hidden)", pts.len(), edges.len());
        vector_scene(&pts, &edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &note)
    }

    pub(super) fn save(&self) -> Value {
        json!({ "nx": self.nx, "ny": self.ny, "shells": self.shells })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        let get = |k: &str, hi: usize| v.get(k).and_then(Value::as_u64).map(|x| usize::try_from(x).unwrap_or(1).clamp(1, hi));
        if let Some(x) = get("nx", MAX_N) {
            self.nx = x;
        }
        if let Some(x) = get("ny", MAX_N) {
            self.ny = x;
        }
        if let Some(x) = get("shells", MAX_SHELLS) {
            self.shells = x;
        }
    }
}
