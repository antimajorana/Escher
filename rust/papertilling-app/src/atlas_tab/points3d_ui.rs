//! The 230-group points catalogue in the Atlas 3D space-group panel (C1 WP-U4, plan §5 T4): the selected group's
//! lattices by the 2D "points" rule (the first 7 shown, the rest as extra), and the chosen one grown by inflation
//! on its Dirichlet cells (never wrapped on a torus), layer by layer, with the seed under "Growth seed".
//!
//! Advanced only: this panel lives inside the existing "230 3D Space Groups" atlas view, reached from its Command
//! Deck card; it adds no mode, tab, strip or menu entry.

use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::space3d::points3d::{
    PointsEntry, PointsGrowth, coverage_line, grow_points3d, points_group, points_note,
};

use super::space3d_ui::Space3dPanelState;

/// Deepest layer offered by the slider.
pub const MAX_LAYERS: u32 = 4;
/// Cell cap of one growth (a layer that would pass it is not built, and the stop is shown).
pub const MAX_CELLS: usize = 12_000;
/// Seeds offered under "Growth seed" (for a net, the Dirichlet seed is the cell seed).
const SEEDS: [(SeedKind, &str); 4] = [
    (SeedKind::Vertex, "Vertex (cells at a Voronoi vertex)"),
    (SeedKind::Edge, "Edge (cells on a Voronoi edge)"),
    (SeedKind::Face, "Face (2 cells on a Voronoi face)"),
    (SeedKind::Cell, "Cell (one Dirichlet cell)"),
];

/// Which entry is grown and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogueChoice {
    pub sg: u16,
    /// Index among all the group's entries (shown first, then extra).
    pub idx: usize,
    pub seed: SeedKind,
    pub layers: u32,
}

impl Default for CatalogueChoice {
    fn default() -> Self {
        Self { sg: 1, idx: 0, seed: SeedKind::Cell, layers: 1 }
    }
}

/// A grown entry and the choice that made it.
#[derive(Clone, Debug)]
pub struct CatalogueView {
    pub choice: CatalogueChoice,
    pub growth: PointsGrowth,
}

impl CatalogueView {
    /// Species caption of site kind `kind` (the orbit index of the entry).
    pub fn orbit_label(&self, kind: u8) -> String {
        let w = points_group(self.choice.sg)
            .ok()
            .and_then(|g| g.entries.get(self.choice.idx))
            .and_then(|e| e.wyckoff.get(usize::from(kind)).cloned())
            .unwrap_or_else(|| "?".into());
        format!("orbit {} (Wyckoff {w})", usize::from(kind) + 1)
    }
}

/// Grow `choice`; the refusal reason when the entry cannot be grown.
pub fn grow_choice(choice: CatalogueChoice) -> Result<CatalogueView, String> {
    grow_points3d(choice.sg, choice.idx, choice.seed, choice.layers, MAX_CELLS).map(|growth| CatalogueView { choice, growth })
}

/// One-line description of an entry.
pub fn entry_label(k: usize, e: &PointsEntry) -> String {
    let deg: Vec<String> = e.degrees.iter().map(ToString::to_string).collect();
    let s = if e.sites == 1 { "site" } else { "sites" };
    format!("{}. {} | {} {s}/cell | degree {}", k + 1, e.wyckoff.join(" + "), e.sites, deg.join(", "))
}

/// Readout of a growth stop under the cell cap `cap`, if any.
pub fn stop_text_cap(stopped: Option<&Stopped>, cap: usize) -> Option<String> {
    stopped.map(|s| match s {
        Stopped::CellCap { layer, .. } => format!("Stopped: layer {layer} would pass {cap} cells and was not built."),
        Stopped::Closed { layer } => format!("Closed at layer {layer}: the whole complex is grown."),
        Stopped::Infinite { layer, reason } => format!("Stopped: layer {layer} would be infinite ({reason})."),
        Stopped::Precision { layer, gap } => {
            format!("Stopped: layer {layer} lost merge precision (smallest key gap {gap:.2e}).")
        }
        Stopped::Rim { layer } => {
            format!("Stopped: layer {layer} reached the outer boundary of the finite patch and was not built.")
        }
    })
}

impl Space3dPanelState {
    /// Draw `c` instead of the canonical net (None: back to it); clears the pick.
    pub(crate) fn set_catalogue(&mut self, c: Option<CatalogueView>) {
        self.view.catalogue = c;
        self.view.picked = None;
        self.view.hovered = None;
        self.view.changed = true;
    }

    /// Grow `choice` and draw it; on a refusal keep the current drawing and show the reason. True when redrawn.
    pub(crate) fn choose_catalogue(&mut self, choice: CatalogueChoice) -> bool {
        match grow_choice(choice) {
            Ok(v) => {
                self.view.catalogue_error = None;
                self.set_catalogue(Some(v));
                true
            }
            Err(e) => {
                self.view.catalogue_error = Some(format!("Refused: group #{}, lattice {}. {e}", choice.sg, choice.idx + 1));
                false
            }
        }
    }

    /// The "Net:" line of the inspector: the catalogue entry when one is drawn.
    pub(crate) fn show_net_line(&self, ui: &mut egui::Ui) {
        if let Some(name) = self.active_screw_name() {
            ui.small(name);
            return;
        }
        match &self.view.catalogue {
            Some(c) => {
                ui.small(format!("Net: {}", c.growth.patch.name));
            }
            None => {
                ui.small(format!("Net: {}", self.selected_net.label()));
            }
        }
    }
}

/// Draw the catalogue entries of the selected space group; true when the scenes must be rebuilt.
pub(crate) fn show(ui: &mut egui::Ui, sp: &mut Space3dPanelState) -> bool {
    let mut dirty = false;
    if sp.view.catalogue.as_ref().is_some_and(|c| c.choice.sg != sp.selected_sg) {
        sp.set_catalogue(None);
        sp.view.catalogue_error = None;
        dirty = true;
    }
    ui.separator();
    egui::CollapsingHeader::new(egui::RichText::new("Ranked lattices").strong())
        .id_salt("atlas_points3d")
        .default_open(true)
        .show(ui, |ui| dirty |= show_body(ui, sp));
    dirty
}

fn show_body(ui: &mut egui::Ui, sp: &mut Space3dPanelState) -> bool {
    let mut dirty = false;
    let num = sp.selected_sg;
    let Ok(group) = points_group(num) else {
        ui.small(format!("No catalogue data for #{num}."));
        return false;
    };
    ui.small("Fewest independent site orbits first, then fewest sites per primitive cell.")
        .on_hover_text(format!("An orbit contains all sites related by this group's symmetries.\n{}", points_note()));
    ui.small(format!("#{num} {} · {} sampled lattices", group.symbol, group.entries.len()))
        .on_hover_text(coverage_line(num));
    let active = sp.view.catalogue.as_ref().map(|c| c.choice);
    let mut pick: Option<usize> = None;
    let shown = group.shown().len();
    for (k, e) in group.entries.iter().enumerate().take(shown) {
        let r = ui.selectable_label(active.is_some_and(|c| c.idx == k), entry_label(k, e));
        if r.on_hover_text(entry_hover(e)).clicked() {
            pick = Some(k);
        }
    }
    if group.entries.len() > shown {
        egui::CollapsingHeader::new(format!("Extra ({})", group.entries.len() - shown))
            .id_salt("atlas_points3d_extra")
            .show(ui, |ui| {
                for (k, e) in group.entries.iter().enumerate().skip(shown) {
                    let r = ui.selectable_label(active.is_some_and(|c| c.idx == k), entry_label(k, e));
                    if r.on_hover_text(entry_hover(e)).clicked() {
                        pick = Some(k);
                    }
                }
            });
    }
    let base = active.unwrap_or(CatalogueChoice { sg: num, ..CatalogueChoice::default() });
    if let Some(k) = pick {
        dirty |= sp.choose_catalogue(CatalogueChoice { sg: num, idx: k, ..base });
    }
    if !sp.pristine_only && sp.screw_periods != 0 {
        ui.small("The screw preview uses a fixed block of cells. Select Reference to adjust the growth layers.");
    } else if let Some(c) = active {
        let mut layers = c.layers;
        if ui.add(egui::Slider::new(&mut layers, 0..=MAX_LAYERS).text("Layers")).changed() {
            dirty |= sp.choose_catalogue(CatalogueChoice { layers, ..c });
        }
        egui::CollapsingHeader::new("Growth seed").id_salt("atlas_points3d_more").show(ui, |ui| {
            let mut seed = c.seed;
            egui::ComboBox::from_id_salt("atlas_points3d_seed")
                .selected_text(SEEDS.iter().find(|s| s.0 == seed).map_or("?", |s| s.1))
                .show_ui(ui, |ui| {
                    for (k, label) in SEEDS {
                        ui.selectable_value(&mut seed, k, label);
                    }
                });
            if seed != c.seed {
                dirty |= sp.choose_catalogue(CatalogueChoice { seed, ..c });
            }
        });
        if let Some(v) = &sp.view.catalogue {
            show_readout(ui, &v.growth);
        }
    } else {
        ui.small("Click a lattice to grow it.");
    }
    if let Some(e) = &sp.view.catalogue_error {
        ui.colored_label(egui::Color32::from_rgb(200, 70, 40), e.as_str());
    }
    dirty
}

fn entry_hover(e: &PointsEntry) -> String {
    let also: Vec<String> = e.also.iter().map(|w| w.join(" + ")).collect();
    let also = if also.is_empty() { String::new() } else { format!("\nSame lattice from: {}", also.join("; ")) };
    format!("Wyckoff labels use the catalogue setting. This network occurred in {:.0}% of samples with these Wyckoff positions and this space group.{also}", 100.0 * e.share)
}

fn show_readout(ui: &mut egui::Ui, g: &PointsGrowth) {
    for (k, s) in g.stats.iter().enumerate() {
        let closed = matches!(g.stopped, Some(Stopped::Closed { layer }) if layer as usize == k);
        let flag = if !closed && s.euler != 1 { " (layer is not a ball)" } else { "" };
        ui.small(format!(
            "S{k}: {} cells (+{}) | boundary {} vertices, {} edges | χ = {}{flag}",
            s.cells, s.new_cells, s.boundary_vertices, s.boundary_edges, s.euler
        ));
    }
    if let Some(t) = stop_text_cap(g.stopped.as_ref(), MAX_CELLS) {
        ui.small(t);
    }
}

#[cfg(test)]
#[path = "tests_points3d.rs"]
mod tests;
