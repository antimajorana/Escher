//! Chiral `(n, m)` nanotubes and nanoribbons rolled or cut from the Gallery's current 2D lattice.
//!
//! The tube period `T` comes from the core's metric search (`find_orthogonal_period`); when the search is refused
//! (e.g. an irrational metric ratio) the core error is shown as the refusal, and nothing is drawn. Before the build,
//! [`tube_size`] repeats that search and refuses a tube cell too large to build and draw (a nearly degenerate metric
//! can give a period T with a huge coefficient).

use papertilling_core::flat3d::{NanoribbonPatch, NanotubeMesh, build_nanoribbon, build_nanotube_opts, build_supercell_2d, find_orthogonal_period};
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use super::{Scenes, int_row, points_scene};
use crate::mesh3d::wb_mesh::mesh_from_points_edges;
use crate::workbench::scene_helpers::{line_prim, with_lines};

pub(super) const TUBE_TITLE: &str = "S4 nanotube (unrolled sheet)";
pub(super) const RIBBON_TITLE: &str = "S4 nanoribbon";
pub(super) const MAX_NM: i32 = 12;
pub(super) const MAX_PERIODS: usize = 4;
pub(super) const MAX_RIBBON: usize = 8;
/// Largest tube built and drawn (graphene at the (n, m) caps needs 6352: 4·397 sites per cell, 4 periods).
pub(super) const MAX_TUBE_SITES: usize = 20_000;
/// Largest lattice-cell box the core scans to pick the tube-cell cosets.
pub(super) const MAX_COSET_SCAN: i64 = 1_000_000;

/// The core's `standard_hex_60` test: equal lengths at 120° (to 1e-5) switch to the 60° basis `(a1 + a2, a2)`.
fn is_hex_120(cell: &Cell2D) -> bool {
    let (l1, l2) = (cell.a1[0].hypot(cell.a1[1]), cell.a2[0].hypot(cell.a2[1]));
    let cos_g = (cell.a1[0] * cell.a2[0] + cell.a1[1] * cell.a2[1]) / (l1 * l2).max(1e-30);
    (l1 - l2).abs() <= 1e-5 * l1.max(l2) && (cos_g + 0.5).abs() <= 1e-5
}

/// The period `T` (in the basis the core rolls in) and the number of sites the core would build, found with the
/// core's own period search before any site is built; a tube over [`MAX_TUBE_SITES`] sites, or one whose coset scan
/// exceeds [`MAX_COSET_SCAN`] cells, is refused with its size.
pub(super) fn tube_size(cell: &Cell2D, t: &TubeParams) -> Result<([i32; 2], usize), String> {
    let work = if t.std60 && is_hex_120(cell) { build_supercell_2d(cell, [[1, 1], [0, 1]])? } else { cell.clone() };
    let tv = find_orthogonal_period(&work, t.n, t.m)?;
    let (n, m) = (i64::from(t.n), i64::from(t.m));
    let (t1, t2) = (i64::from(tv[0]), i64::from(tv[1]));
    let d_det = n * t2 - m * t1;
    if d_det <= 0 {
        return Err(format!("REFUSED: the period T = ({t1}, {t2}) does not span a tube cell with C_h = ({n}, {m}) (nothing drawn)"));
    }
    let span = |a: [i64; 4]| a.iter().max().copied().unwrap_or(0) - a.iter().min().copied().unwrap_or(0) + 5;
    let scan = span([0, n, t1, n + t1]).saturating_mul(span([0, m, t2, m + t2]));
    let per_cell = u64::try_from(d_det).unwrap_or(u64::MAX).saturating_mul(work.basis.len() as u64);
    let sites = per_cell.saturating_mul(t.periods as u64);
    if sites > MAX_TUBE_SITES as u64 || scan > MAX_COSET_SCAN {
        return Err(format!(
            "REFUSED: the period search gives T = ({t1}, {t2}), so one tube cell holds {per_cell} sites ({sites} over {} periods; the cap is {MAX_TUBE_SITES}; coset scan {scan} cells, cap {MAX_COSET_SCAN}). A nearly degenerate metric makes T this long; nothing is built or drawn",
            t.periods
        ));
    }
    Ok((tv, sites as usize))
}

/// A vector of the rolled plane in the unrolled-sheet frame of `positions_2d`: `(v · ĉ, v · t̂)`, with ĉ along `C_h`
/// and t̂ along `T`.
pub(super) fn to_sheet(t: &NanotubeMesh, v: [f64; 2]) -> [f64; 2] {
    let c = t.circumference.max(1e-300);
    let p = t.period_length.max(1e-300);
    let along = |w: [f64; 2], l: f64| (v[0] * w[0] + v[1] * w[1]) / l;
    [along(t.chiral_vec, c), along(t.trans_vec, p)]
}

pub(super) fn build_tube(cell: &Cell2D, t: &TubeParams) -> Result<NanotubeMesh, String> {
    tube_size(cell, t)?;
    let out = build_nanotube_opts(cell, t.n, t.m, t.periods, t.periodic, t.std60)?;
    if out.positions_3d.is_empty() {
        return Err(String::from("the tube has no sites (refused, nothing drawn)"));
    }
    Ok(out)
}

pub(super) fn build_ribbon(cell: &Cell2D, r: &RibbonParams) -> Result<NanoribbonPatch, String> {
    let out = build_nanoribbon(cell, r.dir, r.width, r.length, r.periodic)?;
    if out.positions_2d.is_empty() {
        return Err(String::from("the ribbon has no sites (refused, nothing drawn)"));
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TubeParams {
    pub(super) n: i32,
    pub(super) m: i32,
    pub(super) periods: usize,
    pub(super) periodic: bool,
    pub(super) std60: bool,
}

impl Default for TubeParams {
    fn default() -> Self {
        Self { n: 4, m: 4, periods: 2, periodic: false, std60: false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RibbonParams {
    pub(super) dir: [i32; 2],
    pub(super) width: usize,
    pub(super) length: usize,
    pub(super) periodic: bool,
}

impl Default for RibbonParams {
    fn default() -> Self {
        Self { dir: [1, 0], width: 3, length: 4, periodic: false }
    }
}

#[derive(Default)]
pub(super) struct TubeState {
    pub(super) p: TubeParams,
    pub(super) out: Option<NanotubeMesh>,
}

#[derive(Default)]
pub(super) struct RibbonState {
    pub(super) p: RibbonParams,
    pub(super) out: Option<NanoribbonPatch>,
}

impl TubeState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let p = &self.p;
        if p.n == 0 && p.m == 0 {
            return Err(String::from("the chiral vector (n, m) must be non-zero"));
        }
        if p.n.abs() > MAX_NM || p.m.abs() > MAX_NM {
            return Err(format!("|n| and |m| must be at most {MAX_NM}"));
        }
        if p.periods == 0 || p.periods > MAX_PERIODS {
            return Err(format!("tube periods along the axis must lie in 1..={MAX_PERIODS}"));
        }
        Ok(())
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let p = &mut self.p;
        let mut ch = false;
        ui.horizontal(|ui| {
            ui.label("Chiral vector (n, m):");
            ch |= ui.add(egui::DragValue::new(&mut p.n).range(-MAX_NM..=MAX_NM)).changed();
            ch |= ui.add(egui::DragValue::new(&mut p.m).range(-MAX_NM..=MAX_NM)).changed();
        });
        ch |= int_row(ui, "Periods along the axis:", &mut p.periods, 1, MAX_PERIODS);
        ch |= ui.checkbox(&mut p.periodic, "Close the axis periodically").changed();
        ch |= ui.checkbox(&mut p.std60, "Use the standard 60° hexagonal basis").changed();
        ch
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(t) = &self.out else {
            ui.label("No tube built for these settings.");
            return;
        };
        ui.strong(format!("({}, {}) tube, T = ({}, {})", t.n, t.m, t.t_vec[0], t.t_vec[1]));
        ui.label(format!(
            "[COMPUTATIONAL EVIDENCE] |C_h| = {:.6}, radius = |C_h| / 2π = {:.6}, period |T| = {:.6}",
            t.circumference, t.radius, t.period_length
        ));
        ui.label(format!("[COMPUTATIONAL EVIDENCE] chiral angle {:.4}° (from a1), {} sites per tube cell", t.chiral_angle_deg, t.atoms_per_unit));
        let cos = to_sheet(t, t.trans_vec)[0] / t.period_length.max(1e-300);
        ui.label(format!("[COMPUTATIONAL EVIDENCE] |cos(C_h, T)| = {:.1e} (the core accepts T up to 1e-6 in floating point)", cos.abs()));
        ui.small("Graphene in the 60° basis: |C_h| = a√(n²+nm+m²), 4(n²+nm+m²)/d_R atoms per cell (Saito et al. 1998, Ch. 3).");
        let seams = t.bonds.iter().filter(|b| b.2).count();
        ui.label(format!("Built: {} sites over {} periods, {} bonds ({} across the seam, red)", t.positions_3d.len(), t.periods_z, t.bonds.len(), seams));
        let (lo, hi) = t.coordination_numbers.iter().fold((usize::MAX, 0), |(a, b), &c| (a.min(c), b.max(c)));
        if lo <= hi {
            ui.label(format!("Coordination numbers {lo}..{hi}{}", if t.periodic_z { " (axis closed)" } else { " (open ends)" }));
        }
    }

    pub(super) fn scenes(&self) -> Option<Scenes> {
        let t = self.out.as_ref()?;
        let flat: Vec<(usize, usize)> = t.bonds.iter().filter(|b| !b.2).map(|b| (b.0, b.1)).collect();
        let scene = points_scene(&t.positions_2d, &flat, &[], TUBE_TITLE, "red: chiral vector C_h; blue: tube period T");
        let scene = with_lines(
            scene,
            vec![
                line_prim([0.0, 0.0], to_sheet(t, t.chiral_vec), [192, 57, 43, 255], 2.4),
                line_prim([0.0, 0.0], to_sheet(t, t.trans_vec), [41, 98, 255, 255], 2.4),
            ],
        );
        let edges: Vec<(usize, usize, u8)> = t.bonds.iter().map(|b| (b.0, b.1, u8::from(b.2))).collect();
        let kinds: Vec<u8> = t.site_basis_idx.iter().map(|&b| (b % 4) as u8).collect();
        let mesh = mesh_from_points_edges(&t.positions_3d, &edges, &kinds);
        Some((scene, Some(mesh), format!("({}, {}) tube: {} sites", t.n, t.m, t.positions_3d.len())))
    }

    pub(super) fn save(&self) -> Value {
        let p = &self.p;
        json!({"n": p.n, "m": p.m, "periods": p.periods, "periodic": p.periodic, "std60": p.std60})
    }

    pub(super) fn restore(&mut self, v: &Value) {
        let p = &mut self.p;
        let int = |k: &str| v.get(k).and_then(Value::as_i64);
        if let Some(n) = int("n") {
            p.n = n.clamp(-i64::from(MAX_NM), i64::from(MAX_NM)) as i32;
        }
        if let Some(m) = int("m") {
            p.m = m.clamp(-i64::from(MAX_NM), i64::from(MAX_NM)) as i32;
        }
        if let Some(k) = int("periods") {
            p.periods = k.clamp(1, MAX_PERIODS as i64) as usize;
        }
        if let Some(b) = v.get("periodic").and_then(Value::as_bool) {
            p.periodic = b;
        }
        if let Some(b) = v.get("std60").and_then(Value::as_bool) {
            p.std60 = b;
        }
    }
}

impl RibbonState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let p = &self.p;
        if p.dir == [0, 0] {
            return Err(String::from("the ribbon direction must be non-zero"));
        }
        if p.dir.iter().any(|d| d.abs() > MAX_NM) {
            return Err(format!("ribbon direction entries must be at most {MAX_NM} in size"));
        }
        if p.width == 0 || p.width > MAX_RIBBON || p.length == 0 || p.length > MAX_RIBBON {
            return Err(format!("ribbon width and length must lie in 1..={MAX_RIBBON} cells"));
        }
        Ok(())
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let p = &mut self.p;
        let mut ch = false;
        ui.horizontal(|ui| {
            ui.label("Direction (a1, a2 steps):");
            ch |= ui.add(egui::DragValue::new(&mut p.dir[0]).range(-MAX_NM..=MAX_NM)).changed();
            ch |= ui.add(egui::DragValue::new(&mut p.dir[1]).range(-MAX_NM..=MAX_NM)).changed();
        });
        ch |= int_row(ui, "Width (cells):", &mut p.width, 1, MAX_RIBBON);
        ch |= int_row(ui, "Length (cells):", &mut p.length, 1, MAX_RIBBON);
        ch |= ui.checkbox(&mut p.periodic, "Periodic along the ribbon").changed();
        ch
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(r) = &self.out else {
            ui.label("No ribbon built for these settings.");
            return;
        };
        ui.strong(format!("Ribbon along ({}, {}), width direction ({}, {})", r.chiral_dir[0], r.chiral_dir[1], r.width_dir[0], r.width_dir[1]));
        let seams = r.bonds.iter().filter(|b| b.2).count();
        ui.label(format!(
            "[COMPUTATIONAL EVIDENCE] {} × {} cells: {} sites, {} bonds ({} periodic, red), {} edge sites (ringed)",
            r.width_cells,
            r.length_cells,
            r.positions_2d.len(),
            r.bonds.len(),
            seams,
            r.edge_sites.len()
        ));
        ui.label(format!("Period vector ({:.4}, {:.4}){}", r.period_vec[0], r.period_vec[1], if r.periodic_along { ", periodic" } else { ", open" }));
    }

    pub(super) fn scenes(&self) -> Option<Scenes> {
        let r = self.out.as_ref()?;
        let flat: Vec<(usize, usize)> = r.bonds.iter().filter(|b| !b.2).map(|b| (b.0, b.1)).collect();
        let scene = points_scene(&r.positions_2d, &flat, &r.edge_sites, RIBBON_TITLE, "ringed: edge sites");
        let edges: Vec<(usize, usize, u8)> = r.bonds.iter().map(|b| (b.0, b.1, u8::from(b.2))).collect();
        let mut kinds = vec![0_u8; r.positions_3d.len()];
        for &e in &r.edge_sites {
            if let Some(k) = kinds.get_mut(e) {
                *k = 1;
            }
        }
        let mesh = mesh_from_points_edges(&r.positions_3d, &edges, &kinds);
        Some((scene, Some(mesh), format!("Nanoribbon: {} sites, {} on an edge", r.positions_3d.len(), r.edge_sites.len())))
    }

    pub(super) fn save(&self) -> Value {
        let p = &self.p;
        json!({"dir": p.dir, "width": p.width, "length": p.length, "periodic": p.periodic})
    }

    pub(super) fn restore(&mut self, v: &Value) {
        let p = &mut self.p;
        if let Some(d) = v.get("dir").and_then(Value::as_array).filter(|a| a.len() == 2) {
            let c = |x: &Value| x.as_i64().map(|k| k.clamp(-i64::from(MAX_NM), i64::from(MAX_NM)) as i32);
            if let (Some(a), Some(b)) = (c(&d[0]), c(&d[1])) {
                p.dir = [a, b];
            }
        }
        let size = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| (x as usize).clamp(1, MAX_RIBBON));
        if let Some(w) = size("width") {
            p.width = w;
        }
        if let Some(l) = size("length") {
            p.length = l;
        }
        if let Some(b) = v.get("periodic").and_then(Value::as_bool) {
            p.periodic = b;
        }
    }
}
