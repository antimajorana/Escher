//! Layer-group growth and stacking in the Atlas "80 Layer Groups" view (C1 WP-U4, plan §5 T5 with Q9, Q12): the
//! exact-group layer grown in-plane by the touching rule on its Dirichlet cells, then stacked AA along the normal
//! with a user spacing. The seed and an optional oblique offset sit under "More"; an offset that breaks the group,
//! or a spacing that makes layers interpenetrate, is refused with the reason (one layer is drawn meanwhile).
//!
//! Advanced only: this panel lives inside the existing layer-group atlas view, reached from its Command Deck card.

use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::space3d::layer_slab::{SlabGrowth, StackSpec, grow_layer_slab};
use papertilling_core::space3d::nets3d::Net3dPatch;

use super::layers_ui::LayersPanelState;
use super::points3d_ui::stop_text_cap;

/// Deepest in-plane layer offered by the slider.
pub const MAX_INPLANE: u32 = 6;
/// Most stacked copies offered.
pub const MAX_STACK: u32 = 8;
/// Cell cap of the in-plane growth (as `default_slab_patch`).
pub const MAX_CELLS: usize = 20_000;
const SEEDS: [(SeedKind, &str); 4] = [
    (SeedKind::Vertex, "Vertex (cells at a Voronoi vertex)"),
    (SeedKind::Edge, "Edge (cells on a Voronoi edge)"),
    (SeedKind::Face, "Face (2 cells on a Voronoi face)"),
    (SeedKind::Cell, "Cell (one Dirichlet cell)"),
];

type Key = (u8, SeedKind, u32, u32, [u64; 3]);

/// Growth and stacking settings of the layer view, with the last build.
pub struct LayerGrowthState {
    pub seed: SeedKind,
    /// In-plane touching-rule layers.
    pub layers: u32,
    pub stack: StackSpec,
    /// Whether the oblique offset is used (else AA, offset 0).
    pub oblique: bool,
    built: Option<(Key, SlabGrowth)>,
    /// The refusal of the requested stack, if any (one layer is drawn instead).
    pub refusal: Option<String>,
}

impl Default for LayerGrowthState {
    /// The defaults of `layer_slab::default_slab_patch`: vertex seed, 2 layers, one copy.
    fn default() -> Self {
        Self { seed: SeedKind::Vertex, layers: 2, stack: StackSpec::default(), oblique: false, built: None, refusal: None }
    }
}

impl LayerGrowthState {
    /// The requested stack (offset 0 unless the oblique option is on).
    pub fn spec(&self) -> StackSpec {
        StackSpec { offset: if self.oblique { self.stack.offset } else { [0.0, 0.0] }, ..self.stack }
    }

    fn key(&self, num: u8) -> Key {
        let s = self.spec();
        (num, self.seed, self.layers, s.layers, [s.spacing.to_bits(), s.offset[0].to_bits(), s.offset[1].to_bits()])
    }

    /// The grown (and stacked) slab of group `num`, rebuilt only when a setting changed. A refused stack falls
    /// back to one layer, records the reason in `refusal`, and prepends it to `patch.info` for the 3D caption.
    pub fn growth(&mut self, num: u8) -> Result<&SlabGrowth, String> {
        let key = self.key(num);
        if self.built.as_ref().is_none_or(|(k, _)| *k != key) {
            self.built = None;
            self.refusal = None;
            let g = match grow_layer_slab(num, self.seed, self.layers, &self.spec(), MAX_CELLS) {
                Ok(g) => g,
                Err(e) => {
                    let one = StackSpec { layers: 1, ..StackSpec::default() };
                    let mut g = grow_layer_slab(num, self.seed, self.layers, &one, MAX_CELLS)?;
                    g.patch.info = format!("Refused: {e} (showing 1 layer) | {}", g.patch.info);
                    self.refusal = Some(format!("Refused: {e}. One layer is drawn."));
                    g
                }
            };
            self.built = Some((key, g));
        }
        Ok(&self.built.as_ref().expect("built above").1)
    }

    /// The drawing of group `num` (an empty patch carrying the reason if even one layer is refused).
    pub fn patch(&mut self, num: u8) -> Net3dPatch {
        match self.growth(num) {
            Ok(g) => g.patch.clone(),
            Err(e) => Net3dPatch {
                name: format!("Layer #{num}"),
                space_group_num: u16::from(num),
                defect_label: e.clone(),
                positions: Vec::new(),
                bonds: Vec::new(),
                site_kinds: Vec::new(),
                defect_axis: None,
                info: format!("Refused: {e}"),
            },
        }
    }
}

/// Draw the layer-growth controls; true when the scenes must be rebuilt.
pub(crate) fn show(ui: &mut egui::Ui, layers: &mut LayersPanelState) -> bool {
    let mut dirty = false;
    ui.separator();
    egui::CollapsingHeader::new(egui::RichText::new("Grow and stack").strong())
        .id_salt("atlas_layers_growth")
        .default_open(true)
        .show(ui, |ui| dirty |= show_body(ui, layers));
    dirty
}

fn show_body(ui: &mut egui::Ui, lp: &mut LayersPanelState) -> bool {
    let num = lp.selected_num;
    let st = &mut lp.growth;
    let mut dirty = false;
    dirty |= ui.add(egui::Slider::new(&mut st.layers, 0..=MAX_INPLANE).text("In-plane layers")).changed();
    dirty |= ui.add(egui::Slider::new(&mut st.stack.layers, 1..=MAX_STACK).text("Stacked layers")).changed();
    dirty |= ui
        .add(egui::Slider::new(&mut st.stack.spacing, 0.2..=4.0).text("Spacing"))
        .on_hover_text("Distance between stacked copies along the normal (AA stacking)")
        .changed();
    egui::CollapsingHeader::new("More").id_salt("atlas_layers_more").show(ui, |ui| {
        egui::ComboBox::from_id_salt("atlas_layers_seed")
            .selected_text(SEEDS.iter().find(|s| s.0 == st.seed).map_or("?", |s| s.1))
            .show_ui(ui, |ui| {
                for (k, label) in SEEDS {
                    dirty |= ui.selectable_value(&mut st.seed, k, label).changed();
                }
            });
        dirty |= ui
            .checkbox(&mut st.oblique, "Oblique offset per layer")
            .on_hover_text("Shift each copy in-plane by (s1, s2) in lattice units; refused if it breaks the group")
            .changed();
        if st.oblique {
            ui.horizontal(|ui| {
                for (k, name) in ["s1", "s2"].into_iter().enumerate() {
                    ui.small(name);
                    dirty |= ui.add(egui::DragValue::new(&mut st.stack.offset[k]).speed(0.01).range(-1.0..=1.0).fixed_decimals(3)).changed();
                }
            });
        }
    });
    match st.growth(num) {
        Ok(g) => {
            ui.small(g.symmetry_note.as_str());
            for (k, s) in g.stats.iter().enumerate() {
                let closed = matches!(g.stopped, Some(Stopped::Closed { layer }) if layer as usize == k);
                let chi = if g.euler_defined {
                    let flag = if !closed && s.euler != 1 { " (layer is not a disk)" } else { "" };
                    format!(" | χ = {}{flag}", s.euler)
                } else {
                    String::new()
                };
                ui.small(format!(
                    "S{k}: {} cells (+{}) | boundary {} vertices, {} edges{chi}",
                    s.cells, s.new_cells, s.boundary_vertices, s.boundary_edges
                ));
            }
            if !g.euler_defined {
                ui.small("χ is not shown: the Dirichlet cells of a non-planar layer are unbounded.");
            }
            if let Some(t) = stop_text_cap(g.stopped.as_ref(), MAX_CELLS) {
                ui.small(t);
            }
        }
        Err(e) => {
            ui.colored_label(egui::Color32::from_rgb(200, 70, 40), format!("Refused: {e}"));
        }
    }
    if let Some(r) = &st.refusal {
        ui.colored_label(egui::Color32::from_rgb(200, 70, 40), r.as_str());
    }
    dirty
}

#[cfg(test)]
#[path = "tests_layers_growth.rs"]
mod tests;
