//! C1 WP-U7: growing a defected lattice by Dirichlet-seeded inflation (plan §5 T7, Q8), drawn at the end of the
//! Defects catalogue's left panel.
//!
//! - **Default view:** one "Layers" control (off until ticked, 4 layers) and the readout table. The seed choice sits in
//!   a collapsed "More" section.
//! - **Scope (Q8):** point defects (dislocations and disclinations here; Stone–Wales in the S2 bond-rotation panel,
//!   through `workbench::s2_defects::stone_wales::build_grown`). Walls and half-walls are refused with the reason.
//! - **Picture:** the crystal [`grow_point_defects`] built (rebuilt at a larger radius until the asked layers are
//!   complete, at most 3 times), at its relaxed positions when the drawing is relaxed. Sites and bonds are coloured
//!   by the first layer whose faces contain them; the rest of the crystal is grey. The layer count shown is the core's
//!   count of complete layers (layers that do not touch the crystal's rim), never the count asked for.

use std::collections::{HashMap, HashSet};

#[cfg(test)]
use papertilling_core::defect::charge::infer_patch_disk;
use papertilling_core::defect::charge::total_charge;
#[cfg(test)]
use papertilling_core::defect::core_graph::LatticeCoreInfo;
#[cfg(test)]
use papertilling_core::defect::defect_growth::grow_crystal;
use papertilling_core::defect::defect_growth::{
    DEFECT_MAX_CELLS, defect_core_in_cell, grow_point_defects,
};
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::growth::face_graph::CrystalComplex;
use papertilling_core::growth::{CellComplex, Growth, GrowthCaps, LayerStats, Stopped, grow};
use papertilling_core::types::BuiltCrystal;
#[cfg(test)]
use papertilling_core::types::Cell2D;

use crate::backend::{CatalogueItemUi, DefectReadoutUi, DefectSelectionUi};
use crate::canvas2d::gauge_overlay::{CrystalOverlayData, OverlayColorbar};
use crate::canvas2d::overlay_legend::colorbar_rgba;
use crate::defects_tab::{DefectSubTab, DefectsTabState};
use crate::live_ops::{defect_context, summarize_rings_and_core};
use crate::live_ops_ao::Defect2dMode;

/// Layers asked for by default (S_0..S_4).
pub(crate) const DEFAULT_LAYERS: u32 = 4;
/// Largest number of layers the control offers.
pub(crate) const MAX_LAYERS: u32 = 8;
/// Why walls are not grown (Q8).
pub(crate) const WALLS_REFUSED: &str = "Growth of walls and half-walls comes later: C1 grows point defects first (Q8).";

/// The seed of the growth ("More" section).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SeedChoice {
    /// Q8 default: the perfect lattice's Dirichlet cell about the core, carried through the surgery.
    #[default]
    DirichletCore,
    /// The faces at the crystal sites nearest the core (ties included), in reference coordinates.
    NearestSites,
}

impl SeedChoice {
    pub(crate) const ALL: [Self; 2] = [Self::DirichletCore, Self::NearestSites];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::DirichletCore => "Dirichlet cell about the core",
            Self::NearestSites => "Faces at the sites nearest the core",
        }
    }
}

/// What the worker is asked to grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GrowthRequest {
    pub(crate) layers: u32,
    pub(crate) seed: SeedChoice,
}

/// One grown crystal, ready to draw and to read out.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GrowthView {
    pub(crate) asked: u32,
    /// Complete layers: S_0..S_{complete-1} are shown.
    pub(crate) complete: usize,
    /// Host crystal radius that the layers live on.
    pub(crate) radius: f64,
    pub(crate) stats: Vec<LayerStats>,
    pub(crate) seed_line: String,
    pub(crate) note: String,
    /// First layer containing each site / each bond (`crystal.edges` order), `None` outside S_n.
    pub(crate) site_layer: Vec<Option<u32>>,
    pub(crate) bond_layer: Vec<Option<u32>>,
}

/// Growth result carried with a defect build.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum GrowthOutcome {
    #[default]
    Off,
    Shown(GrowthView),
    Refused(String),
}

/// Panel state for the Layers control.
#[derive(Clone, Debug)]
pub(crate) struct GrowthUi {
    pub(crate) on: bool,
    pub(crate) layers: u32,
    pub(crate) seed: SeedChoice,
    pub(crate) outcome: GrowthOutcome,
}

impl Default for GrowthUi {
    fn default() -> Self {
        Self { on: false, layers: DEFAULT_LAYERS, seed: SeedChoice::default(), outcome: GrowthOutcome::Off }
    }
}

impl GrowthUi {
    /// The request for the current sub-tab (walls are never grown).
    pub(crate) fn request(&self, sub_tab: DefectSubTab) -> Option<GrowthRequest> {
        (self.on && sub_tab != DefectSubTab::Walls).then_some(GrowthRequest { layers: self.layers, seed: self.seed })
    }

    /// The shell colouring, when a grown crystal is shown.
    pub(crate) fn overlay(&self) -> Option<CrystalOverlayData> {
        match &self.outcome {
            GrowthOutcome::Shown(v) if self.on => Some(v.overlay()),
            _ => None,
        }
    }
}

/// The readout's headline: the core's count of complete layers.
pub(crate) fn complete_label(v: &GrowthView) -> String {
    match v.complete {
        0 => format!("No complete layer (asked 0..{})", v.asked),
        k => format!("Complete layers: 0..{} ({k} of {} asked)", k - 1, v.asked + 1),
    }
}

impl GrowthView {
    pub(crate) fn from_growth(
        crystal: &BuiltCrystal,
        cc: &CrystalComplex,
        g: &Growth<u32>,
        asked: u32,
        radius: f64,
        seed_line: String,
        note: String,
    ) -> Self {
        let edge_id: HashMap<(usize, usize), usize> =
            crystal.edges.iter().enumerate().map(|(i, &(a, b))| ((a.min(b), a.max(b)), i)).collect();
        let mut site_layer = vec![None; crystal.pos.len()];
        let mut bond_layer = vec![None; crystal.edges.len()];
        for (k, layer) in g.layers.iter().enumerate() {
            for &f in layer {
                let w = &cc.faces[f as usize];
                for (i, &s) in w.iter().enumerate() {
                    site_layer[s].get_or_insert(k as u32);
                    let t = w[(i + 1) % w.len()];
                    if let Some(&e) = edge_id.get(&(s.min(t), s.max(t))) {
                        bond_layer[e].get_or_insert(k as u32);
                    }
                }
            }
        }
        Self { asked, complete: g.layers.len(), radius, stats: g.stats.clone(), seed_line, note, site_layer, bond_layer }
    }

    fn legend(&self) -> OverlayColorbar {
        OverlayColorbar {
            title: String::from("Growth layer (0 = seed)"),
            min_val: 0.0,
            max_val: self.complete.saturating_sub(1) as f64,
            diverging: false,
            sqrt_scale: false,
            strain_map: false,
        }
    }

    /// Sites and bonds coloured by layer on the legend's scale; grey and thin outside S_n.
    pub(crate) fn overlay(&self) -> CrystalOverlayData {
        let legend = self.legend();
        let top = legend.max_val.max(1.0);
        let col = |k: Option<u32>| k.map(|k| colorbar_rgba(&legend, f64::from(k) / top));
        let grey = [150, 150, 150, 70];
        CrystalOverlayData {
            site_rgba: Some(self.site_layer.iter().map(|&k| col(k).unwrap_or(grey)).collect()),
            bond_rgba: Some(self.bond_layer.iter().map(|&k| col(k).map_or((grey, 0.8), |c| (c, 2.4))).collect()),
            note_suffix: Some(complete_label(self)),
            legend: Some(legend),
            ..Default::default()
        }
    }
}

/// The point-defect specs and title of a catalogue selection; walls are refused.
fn specs_for(item: &CatalogueItemUi, sel: &DefectSelectionUi) -> Result<(Vec<DefectSpec>, String), String> {
    let ctx = defect_context(item)?;
    let cat = &ctx.cat;
    match *sel {
        DefectSelectionUi::Dislocation(i) => {
            let e = cat.dislocations.get(i).ok_or_else(|| format!("{} has no dislocation class {i}", item.label))?;
            let t = format!("{} — Dislocation {} b=({}, {})", item.label, e.id, e.b_frac[0], e.b_frac[1]);
            Ok((vec![DefectSpec::Dislocation { b: e.b, at: Some(e.at), cut: None }], t))
        }
        DefectSelectionUi::Disclination(i) => {
            let e = cat
                .disclinations
                .get(i)
                .ok_or_else(|| format!("{} has no disclination class {i} (|P+| = 1 has none)", item.group))?;
            let t = format!("{} — Disclination {} ({:+.0}° at {})", item.label, e.id, e.angle, e.wyckoff);
            Ok((vec![DefectSpec::Disclination { angle_deg: e.angle, at: e.point, cut: None }], t))
        }
        DefectSelectionUi::Wall(_) => Err(WALLS_REFUSED.into()),
    }
}

/// Faces containing the sites nearest `core` (reference coordinates, ties within 1e-6 included).
fn nearest_site_seed(crystal: &BuiltCrystal, cc: &CrystalComplex, core: [f64; 2]) -> Vec<u32> {
    let d = |s: usize| (crystal.ref_pos[s][0] - core[0]).hypot(crystal.ref_pos[s][1] - core[1]);
    let used: HashSet<usize> = cc.faces.iter().flatten().copied().collect();
    let dmin = used.iter().map(|&s| d(s)).fold(f64::INFINITY, f64::min);
    let near: HashSet<usize> = used.into_iter().filter(|&s| d(s) <= dmin + 1e-6).collect();
    (0..cc.faces.len() as u32).filter(|&f| cc.faces[f as usize].iter().any(|s| near.contains(s))).collect()
}

/// Grow `seed` on `cc`, keeping complete layers only (none of their faces has a rim vertex), as `grow_crystal` does.
fn grow_complete(cc: &mut CrystalComplex, seed: Vec<u32>, layers: u32) -> Growth<u32> {
    let mut g = grow(&mut cc.complex, seed, GrowthCaps { max_layers: layers, max_cells: DEFECT_MAX_CELLS });
    let rim: HashSet<u32> = cc.rim.iter().map(|&s| s as u32).collect();
    let first_bad =
        g.layers.iter().position(|l| l.iter().any(|&c| cc.complex.vertices_of(c).iter().any(|v| rim.contains(v))));
    if let Some(k) = first_bad {
        g.layers.truncate(k);
        g.stats.truncate(k);
        g.stopped = Some(Stopped::Rim { layer: k as u32 });
    }
    g
}

fn charge_lines(crystal: &BuiltCrystal) -> (String, String) {
    let z = |x: f64| if x.abs() < 5e-4 { 0.0 } else { x };
    match total_charge(crystal) {
        Ok(ch) => {
            let s = match ch.burgers {
                Some(b) => format!("Charge: {} | Frank = {:+.1}°, b = ({:+.3}, {:+.3})", ch.kind, z(ch.frank_deg), z(b[0]), z(b[1])),
                None => format!("Charge: {} | Frank = {:+.1}°", ch.kind, z(ch.frank_deg)),
            };
            (s, format!("Outer loop S = ({:+.3}, {:+.3})", z(ch.loop_sum[0]), z(ch.loop_sum[1])))
        }
        Err(e) => (String::from("Boundary charge unavailable"), format!("total_charge: {e}")),
    }
}

/// Build the grown defect for the Defects catalogue (runs on the defect worker).
pub(crate) fn build_grown(
    item: &CatalogueItemUi,
    sel: &DefectSelectionUi,
    req: GrowthRequest,
    draw_mode: Defect2dMode,
) -> Result<(BuiltCrystal, DefectReadoutUi, GrowthView), String> {
    let (specs, title) = specs_for(item, sel)?;
    let ctx = defect_context(item)?;
    let relax = draw_mode != Defect2dMode::Raw;
    let sizing_layers = match req.seed {
        SeedChoice::DirichletCore => req.layers,
        SeedChoice::NearestSites => req.layers.saturating_add(1),
    };
    let out = grow_point_defects(&ctx.cell, &specs, sizing_layers, None, relax, DEFECT_MAX_CELLS)?;
    let r = &out.result;
    let view = match req.seed {
        SeedChoice::DirichletCore => {
            let s = &r.seed;
            let mut line = format!(
                "Seed: {} face(s) = Dirichlet cell about the core ({} perfect face(s)",
                s.cells.len(),
                s.perfect_faces
            );
            if s.missing > 0 {
                line.push_str(&format!(", {} cut by the surgery and replaced by {} core face(s)", s.missing, s.core_faces));
            }
            line.push(')');
            GrowthView::from_growth(&out.crystal, &r.complex, &r.growth, req.layers, out.radius, line, out.note.clone())
        }
        SeedChoice::NearestSites => {
            let core = defect_core_in_cell(&ctx.cell, &specs)?;
            let mut cc = r.complex.clone();
            let seed = nearest_site_seed(&out.crystal, &cc, core);
            let line = format!("Seed: {} face(s) at the site(s) nearest the core", seed.len());
            let g = grow_complete(&mut cc, seed, req.layers);
            let stop_msg = match g.stopped {
                Some(Stopped::Rim { layer }) if layer <= req.layers => {
                    let shown = if layer == 0 {
                        String::from("no complete layer")
                    } else {
                        format!("showing layers 0..{}", layer - 1)
                    };
                    format!(
                        "layer {layer} reaches the crystal's rim at radius {:.2} after {} rebuild(s); {shown}",
                        out.radius, out.rebuilds
                    )
                }
                Some(Stopped::CellCap { layer, .. }) if layer <= req.layers => {
                    format!("layer {layer} needs more than {DEFECT_MAX_CELLS} faces (cap {DEFECT_MAX_CELLS})")
                }
                _ => String::new(),
            };
            let sizing_msg = format!("crystal sized at radius {:.2}", out.radius);
            let mut parts = vec![stop_msg.as_str(), sizing_msg.as_str()];
            parts.extend(out.note.split("; ").filter(|p| !p.is_empty() && !p.starts_with("layer ")));
            let note = parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("; ");
            GrowthView::from_growth(&out.crystal, &cc, &g, req.layers, out.radius, line, note)
        }
    };
    let (charge_summary, holonomy_detail) = charge_lines(&out.crystal);
    let readout = DefectReadoutUi {
        title: format!("{title} — grown ({})", complete_label(&view)),
        charge_summary,
        holonomy_detail,
        core_rings: summarize_rings_and_core(&out.crystal, &ctx.lat_info),
        note: view.note.clone(),
    };
    Ok((out.crystal, readout, view))
}

pub(crate) fn stone_wales_seed_line(n_seed: usize) -> String {
    format!("Seed: {n_seed} face(s) about the rotated bond (the 5-7-7-5 core on the honeycomb)")
}

/// Stone–Wales growth on an already-sized host disk (used by unit tests; the S2 panel sizes via `grow_with_rebuild`).
#[cfg(test)]
pub(crate) fn grow_stone_wales(
    host: &BuiltCrystal,
    flipped: &BuiltCrystal,
    edge: usize,
    cell: &Cell2D,
    layers: u32,
) -> Result<GrowthView, String> {
    let &(u, v) = host.edges.get(edge).ok_or("the rotated bond is not in the host disk")?;
    let core = [0.5 * (host.ref_pos[u][0] + host.ref_pos[v][0]), 0.5 * (host.ref_pos[u][1] + host.ref_pos[v][1])];
    let info = LatticeCoreInfo::new(cell);
    let cg = grow_crystal(flipped, &info.lat, core, layers, DEFECT_MAX_CELLS)?;
    let line = stone_wales_seed_line(cg.seed.cells.len());
    let (_, radius) = infer_patch_disk(host);
    let note = match cg.growth.stopped {
        Some(Stopped::Rim { layer }) if layer <= layers => {
            format!("layer {layer} reaches the rim of the disk: raise the radius")
        }
        _ => String::new(),
    };
    Ok(GrowthView::from_growth(flipped, &cg.complex, &cg.growth, layers, radius, line, note))
}

/// The per-layer readout table.
pub(crate) fn show_readout(ui: &mut egui::Ui, v: &GrowthView, id: &str) {
    ui.label(complete_label(v));
    ui.small(&v.seed_line);
    egui::Grid::new(id).striped(true).show(ui, |ui| {
        for h in ["Layer", "Faces", "New", "∂V", "∂E", "χ"] {
            ui.strong(h);
        }
        ui.end_row();
        for (k, s) in v.stats.iter().enumerate() {
            ui.label(k.to_string());
            ui.label(s.cells.to_string());
            ui.label(s.new_cells.to_string());
            ui.label(s.boundary_vertices.to_string());
            ui.label(s.boundary_edges.to_string());
            ui.label(s.euler.to_string());
            ui.end_row();
        }
    });
    for (k, s) in v.stats.iter().enumerate().filter(|(_, s)| s.euler != 1) {
        ui.small(format!("layer {k} is not a disk (χ = {})", s.euler));
    }
    if !v.note.is_empty() {
        ui.small(&v.note);
    }
}

/// Draw the growth controls; true when the defect picture must be rebuilt. (`PapertillingApp` is not in scope at
/// this slot, which is a `DefectsTabState` method, so it takes the tab state and the Gallery lattice.)
pub(crate) fn show(ui: &mut egui::Ui, defects: &mut DefectsTabState, _item: &CatalogueItemUi) -> bool {
    let walls = defects.sub_tab == DefectSubTab::Walls;
    let g = &mut defects.growth;
    let mut dirty = false;
    ui.separator();
    ui.horizontal(|ui| {
        let tick = ui.add_enabled(!walls, egui::Checkbox::new(&mut g.on, "Layers"));
        dirty |= tick.on_disabled_hover_text(WALLS_REFUSED).changed();
        let slider = egui::Slider::new(&mut g.layers, 0..=MAX_LAYERS);
        dirty |= ui.add_enabled(g.on && !walls, slider).changed() && g.on;
    });
    if walls {
        ui.small(WALLS_REFUSED);
        return false;
    }
    egui::CollapsingHeader::new("More").id_salt("defect_growth_more").default_open(false).show(ui, |ui| {
        for s in SeedChoice::ALL {
            if ui.radio_value(&mut g.seed, s, s.label()).changed() && g.on {
                dirty = true;
            }
        }
        ui.small("Layers follow the touching rule: each adds every face touching the previous boundary.");
    });
    if g.on {
        match &g.outcome {
            GrowthOutcome::Shown(v) => show_readout(ui, v, "defect_growth_tbl"),
            GrowthOutcome::Refused(why) => {
                ui.small(format!("Not grown: {why}"));
            }
            GrowthOutcome::Off => {
                ui.small("Growing ...");
            }
        }
    }
    dirty
}

#[cfg(test)]
#[path = "growth_tests.rs"]
mod tests;
