//! Regular `{p,q,r}` (S³ polychora, the cubic E³ honeycomb, compact and paracompact H³ patches) and the four
//! Euclidean honeycombs: controls, the live `classify_schlafli` badge, and the cell-complex presentation.

use std::collections::HashMap;

use papertilling_core::honeycomb3d::honeycombs_e3::MAX_E3_SHELLS;
use papertilling_core::honeycomb3d::honeycombs_h3::{MAX_H3_SHELLS, MAX_PATCH_LAYERS};
use papertilling_core::honeycomb3d::{
    CellComplex3d, Classification, EuclideanKind, HoneycombRegime, build_euclidean_honeycomb, build_honeycomb,
    classify_schlafli,
};
use serde_json::{Value, json};

use super::present::{Built, COUNTS_EVIDENCE, CoreOut, palette_rgba, projection_scene};
use crate::mesh3d::wb_mesh::mesh_from_polygons;
use crate::workbench::scene_helpers::unit_circle_lines;

/// Range offered for each Schläfli entry (wide enough that degenerate and hyperideal symbols are reachable).
pub(crate) const PQR_RANGE: std::ops::RangeInclusive<u32> = 2..=8;

pub(crate) const EUCLID_KINDS: [EuclideanKind; 4] =
    [EuclideanKind::Cubic, EuclideanKind::BitruncatedCubic, EuclideanKind::RhombicDodecahedral, EuclideanKind::AlternatedCubic];

/// Short badge of a regime (never equal to a sub-row label).
pub(crate) fn regime_badge(r: HoneycombRegime) -> &'static str {
    match r {
        HoneycombRegime::SphericalS3 => "regime: spherical (S³ polychoron)",
        HoneycombRegime::EuclideanE3 => "regime: flat (E³ honeycomb)",
        HoneycombRegime::HyperbolicCompactH3 => "regime: compact hyperbolic (H³)",
        HoneycombRegime::HyperbolicParacompactH3 => "regime: paracompact hyperbolic (H³)",
        HoneycombRegime::HyperidealRefused => "refused: hyperideal",
        HoneycombRegime::DegenerateRefused => "refused: degenerate",
    }
}

/// Shell cap of a regime: `None` when the whole object is built (S³) or nothing is (refused).
pub(crate) fn shell_cap(r: HoneycombRegime) -> Option<u32> {
    match r {
        HoneycombRegime::EuclideanE3 => Some(MAX_E3_SHELLS),
        HoneycombRegime::HyperbolicCompactH3 | HoneycombRegime::HyperbolicParacompactH3 => Some(MAX_H3_SHELLS),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SchlafliState {
    pub(crate) p: u32,
    pub(crate) q: u32,
    pub(crate) r: u32,
    pub(crate) shells: u32,
    pub(crate) patch_layers: u32,
}

impl Default for SchlafliState {
    fn default() -> Self {
        Self { p: 4, q: 3, r: 4, shells: 1, patch_layers: 1 }
    }
}

impl SchlafliState {
    pub(crate) fn classify(&self) -> Classification {
        classify_schlafli(self.p, self.q, self.r)
    }

    /// Job inputs with irrelevant entries zeroed (so equal builds share a cache key).
    pub(crate) fn params(&self) -> (u32, u32, u32, u32, u32) {
        let c = self.classify();
        let shells = shell_cap(c.regime).map_or(0, |cap| self.shells.min(cap));
        let layers = if c.has_infinite_euclidean_cells { self.patch_layers.clamp(1, MAX_PATCH_LAYERS) } else { 1 };
        (self.p, self.q, self.r, shells, layers)
    }

    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.horizontal(|ui| {
            for (name, v) in [("p", &mut self.p), ("q", &mut self.q), ("r", &mut self.r)] {
                ui.label(name);
                ch |= ui.add(egui::DragValue::new(v).range(PQR_RANGE).speed(0.05)).changed();
            }
        });
        let c = self.classify();
        let col = if c.is_buildable() { egui::Color32::from_rgb(39, 174, 96) } else { egui::Color32::from_rgb(192, 57, 43) };
        ui.colored_label(col, regime_badge(c.regime));
        ui.add(egui::Label::new(egui::RichText::new(&c.reason).small()).wrap());
        match shell_cap(c.regime) {
            Some(cap) => {
                self.shells = self.shells.min(cap);
                ui.horizontal(|ui| {
                    ui.label("Shells of cells:");
                    ch |= ui.add(egui::Slider::new(&mut self.shells, 0..=cap)).changed();
                });
            }
            None if c.regime == HoneycombRegime::SphericalS3 => {
                ui.label("Whole polychoron (all cells), stereographic projection.");
            }
            None => {}
        }
        if c.has_infinite_euclidean_cells {
            ui.horizontal(|ui| {
                ui.label("Patch rings of infinite cells:");
                ch |= ui.add(egui::Slider::new(&mut self.patch_layers, 1..=MAX_PATCH_LAYERS)).changed();
            });
        }
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({"p": self.p, "q": self.q, "r": self.r, "shells": self.shells, "patch_layers": self.patch_layers})
    }

    pub(crate) fn restore(&mut self, v: &Value) {
        let get = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x.min(u64::from(u32::MAX)) as u32);
        let clamp_pqr = |x: u32| x.clamp(*PQR_RANGE.start(), *PQR_RANGE.end());
        if let Some(x) = get("p") {
            self.p = clamp_pqr(x);
        }
        if let Some(x) = get("q") {
            self.q = clamp_pqr(x);
        }
        if let Some(x) = get("r") {
            self.r = clamp_pqr(x);
        }
        if let Some(x) = get("shells") {
            self.shells = x.min(MAX_E3_SHELLS.max(MAX_H3_SHELLS));
        }
        if let Some(x) = get("patch_layers") {
            self.patch_layers = x.clamp(1, MAX_PATCH_LAYERS);
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EuclidState {
    pub(crate) kind: EuclideanKind,
    pub(crate) shells: u32,
}

impl Default for EuclidState {
    fn default() -> Self {
        Self { kind: EuclideanKind::Cubic, shells: 1 }
    }
}

impl EuclidState {
    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        egui::ComboBox::from_id_salt("wb_s6_euclid_kind").selected_text(self.kind.label()).show_ui(ui, |ui| {
            for k in EUCLID_KINDS {
                ch |= ui.selectable_value(&mut self.kind, k, k.label()).changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Shells of cells:");
            ch |= ui.add(egui::Slider::new(&mut self.shells, 0..=MAX_E3_SHELLS)).changed();
        });
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({"kind": format!("{:?}", self.kind), "shells": self.shells})
    }

    pub(crate) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        if let Some(s) = v.get("kind").and_then(Value::as_str) {
            match EUCLID_KINDS.into_iter().find(|k| format!("{k:?}") == s) {
                Some(k) => self.kind = k,
                None => notes.push(format!("S6 panel: unknown Euclidean honeycomb {s:?} (kept current)")),
            }
        }
        if let Some(x) = v.get("shells").and_then(Value::as_u64) {
            self.shells = x.min(u64::from(MAX_E3_SHELLS)) as u32;
        }
    }
}

/// Build `{p,q,r}`; a refused symbol returns the core's `[PROVED]` reason.
pub(crate) fn build_schlafli(p: u32, q: u32, r: u32, shells: u32, layers: u32) -> Result<Built, String> {
    let c = classify_schlafli(p, q, r);
    if !c.is_buildable() {
        return Err(c.reason);
    }
    if let Some(cap) = shell_cap(c.regime).filter(|&cap| shells > cap) {
        return Err(format!("shells = {shells} exceeds the maximum {cap} for this regime"));
    }
    let cx = build_honeycomb(p, q, r, shells, layers)?;
    Ok(present_complex(cx, Some(&c), shells, layers))
}

pub(crate) fn build_euclid(kind: EuclideanKind, shells: u32) -> Result<Built, String> {
    let cx = build_euclidean_honeycomb(kind, shells)?;
    Ok(present_complex(cx, None, shells, 0))
}

/// Model note of a regime string (`CellComplex3d::regime`).
fn model_note(regime: &str) -> &'static str {
    match regime {
        "SphericalS3" => "Drawn by stereographic projection of S³ from a pole that is not a vertex: conformal, not isometric (cells near the pole look large).",
        "EuclideanE3" => "Drawn in Euclidean coordinates (isometric).",
        _ => "Drawn in the Poincaré ball model of H³: conformal, not isometric; ideal vertices lie on the unit sphere.",
    }
}

/// Mesh (faces tinted by the shell of their innermost cell), xy projection, inspector rows and stamps.
pub(crate) fn present_complex(cx: CellComplex3d, cls: Option<&Classification>, shells: u32, layers: u32) -> Built {
    let mut face_shell = vec![u32::MAX; cx.faces.len()];
    for (ci, cell) in cx.cells.iter().enumerate() {
        let s = cx.cell_shell.get(ci).copied().unwrap_or(0);
        for &f in cell {
            if let Some(fs) = face_shell.get_mut(f) {
                *fs = (*fs).min(s);
            }
        }
    }
    let shell_of: HashMap<&[usize], u32> = cx.faces.iter().zip(&face_shell).map(|(f, &s)| (f.as_slice(), s)).collect();
    let spherical = cx.regime == "SphericalS3";
    let tint = |face: &[usize]| {
        let s = shell_of.get(face).copied().unwrap_or(0);
        if spherical {
            palette_rgba(s as usize, if s == 0 { 0.6 } else { 0.22 })
        } else {
            let alpha = [0.85, 0.42, 0.26, 0.18][s.min(3) as usize];
            palette_rgba(s as usize, alpha)
        }
    };
    let edges: Vec<(usize, usize)> = cx.edges.iter().map(|e| (e[0], e[1])).collect();
    let mesh = mesh_from_polygons(&cx.points, &cx.faces, &edges, tint);
    let hyperbolic = cx.regime.starts_with("Hyperbolic");
    let extra = if hyperbolic { unit_circle_lines(96, [120, 120, 120, 255]) } else { Vec::new() };
    let title = format!("{}: xy projection of the edges", cx.name);
    let scene2d = projection_scene(&cx.points, edges.iter().copied(), extra, &title, model_note(&cx.regime));
    let (v, e, f, c) = cx.counts();
    let mut rows = vec![
        (String::from("Complex"), cx.name.clone()),
        (String::from("Regime"), cx.regime.clone()),
        (String::from("V, E, F, C"), format!("{v}, {e}, {f}, {c}")),
        (String::from("V − E + F − C"), cx.euler().to_string()),
    ];
    if !spherical {
        rows.push((String::from("Shells requested"), shells.to_string()));
        let max_shell = cx.cell_shell.iter().copied().max().unwrap_or(0);
        let per: Vec<String> = (0..=max_shell).map(|s| cx.cell_shell.iter().filter(|&&x| x == s).count().to_string()).collect();
        rows.push((String::from("Cells per shell"), per.join(", ")));
    }
    let ideal = cx.is_ideal.iter().filter(|&&b| b).count();
    let truncated = cx.cell_truncated.iter().filter(|&&b| b).count();
    if hyperbolic {
        rows.push((String::from("Ideal vertices"), ideal.to_string()));
        rows.push((String::from("Truncated infinite cells"), truncated.to_string()));
        if truncated > 0 {
            rows.push((String::from("Patch rings per infinite cell"), layers.to_string()));
        }
    }
    let mut notes = Vec::new();
    if let Some(c) = cls {
        notes.push(c.reason.clone());
    } else {
        notes.push(String::from("[PROVED: the four Euclidean honeycombs here are the SC, BCC and FCC Voronoi tilings and the octet tiling, all face-to-face tilings of E³]"));
    }
    notes.push(String::from(COUNTS_EVIDENCE));
    notes.push(String::from(if spherical {
        "Expected V − E + F − C = 0 = χ(S³) for a cell decomposition of S³ [PROVED: Euler characteristic of S³]."
    } else if truncated > 0 {
        "No 3-ball test here: a truncated infinite cell is an open patch of its faces, not a closed 3-cell, so V − E + F − C counts a complex that is not a cell decomposition of B³ and need not equal 1."
    } else {
        "Expected V − E + F − C = 1 for a patch that is a 3-ball [PROVED: χ(B³) = 1]; any other value means the patch is not a ball."
    }));
    if truncated > 0 {
        notes.push(String::from(
            "Infinite horospherical cells are cut to finite patches: the drawn cells are pieces, not whole cells (no geometry is approximated).",
        ));
    }
    notes.push(String::from(model_note(&cx.regime)));
    let caption = format!("{} — {v} V, {e} E, {f} F, {c} cells", cx.name);
    Built { core: CoreOut::Complex(Box::new(cx)), scene2d, mesh, caption, rows, notes }
}
