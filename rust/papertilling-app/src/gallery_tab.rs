//! Gallery tab state: browse the 17 groups (three list rules) or the 21 named lattices, move the free parameters.
//!
//! Contract with the other tabs: `active_item()` is the lattice at the current slider values (its `recipe` is
//! `Family.recipe()` at those values, i.e. Python's `box.lattice()`; the catalogue item itself at the default
//! values), and `revision()` increases whenever that item changes (another lattice, a slider, Reset), never on
//! view-only changes (patch repeat, unit cell).

use papertilling_core::plane::tuning::SliderState;
use papertilling_core::types::Cell2D;

use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi, DSymbolReadoutUi, SliderParamUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::live_plane::{slider_params_ui, tune_item};

pub(crate) mod browser;
mod builder_ui;
pub(crate) mod filter;
pub(crate) mod inflation_ui;
mod inspector;
mod retune_async;
mod search_worker;
pub(crate) mod thumb_cache;
mod ui;

#[cfg(test)]
mod python_cases;
#[cfg(test)]
mod browser_tests;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BrowseMode {
    ByGroup,
    Named21,
    CustomBuilder,
}

pub struct GalleryTabState {
    mode: BrowseMode,
    pub selected_group_idx: usize,
    pub rule: CatalogueRuleUi,
    catalogue: Vec<CatalogueItemUi>,
    selected_cat_idx: usize,
    selected_named_idx: usize,
    /// The catalogue / named item as stored (default parameter values).
    base_item: CatalogueItemUi,
    /// Coupled sliders of `base_item` (`None`: no free parameter).
    slider_state: Option<SliderState>,
    sliders: Vec<SliderParamUi>,
    /// Cost of the last parameter move; above `LIVE_MOVE_SECS` pointer moves run on `retune` (newest wins), with
    /// the moves not yet applied queued in `pending_moves`.
    last_move_secs: f64,
    retune: Option<retune_async::RetuneWorker>,
    pending_moves: Vec<(String, f64)>,
    pub repeat: i32,
    pub show_unit_cell: bool,
    /// View only: ITA symmetry elements / Voronoi cells over the drawn patch.
    pub overlays: crate::scene2d::overlays::GalleryOverlayToggles,
    /// View only: BDF / cluster shell inflation patch overlay mode.
    pub inflation: inflation_ui::GalleryInflationState,
    pub readout: DSymbolReadoutUi,
    /// `base_item` at the slider values (what the other tabs use).
    active_item: CatalogueItemUi,
    cell: Cell2D,
    revision: u64,
    pub renaming: bool,
    pub rename_buf: String,
    rename_focus_pending: bool,
    /// The rename field had keyboard focus when it was last drawn.
    rename_has_focus: bool,
    pub alias_note: Option<String>,
    pub custom_builder: builder_ui::CustomBuilderState,
    /// Lattice browser: card / table view, search and facets, previews, and per-rule lists of all 17 groups.
    pub(crate) view: browser::BrowserView,
    pub(crate) filter: filter::BrowseFilter,
    pub(crate) thumbs: thumb_cache::LatticeThumbCache,
    rule_index: [Option<Vec<CatalogueItemUi>>; 3],
}

fn same_lattice(a: &CatalogueItemUi, b: &CatalogueItemUi) -> bool {
    a.alias_key == b.alias_key
        && a.recipe == b.recipe && a.frame == b.frame && a.group == b.group && a.label == b.label && a.named_key == b.named_key
}

impl GalleryTabState {
    pub fn new(backend: &AppBackend, canvas: &mut Canvas2D) -> Self {
        let selected_group_idx = backend.groups().iter().position(|g| g.symbol == "p6m").unwrap_or(0);
        let rule = CatalogueRuleUi::Bonds;
        let catalogue = backend.catalogue_for_group(backend.groups()[selected_group_idx].symbol, rule);
        let selected_cat_idx = catalogue
            .iter()
            .position(|it| it.named_key.as_deref() == Some("kagome"))
            .unwrap_or(0);
        let selected_named_idx = backend
            .named_lattices()
            .iter()
            .position(|it| it.named_key.as_deref() == Some("kagome"))
            .unwrap_or(0);
        let item = backend
            .find_named("kagome")
            .or_else(|| catalogue.get(selected_cat_idx))
            .or_else(|| backend.named_lattices().first())
            .cloned()
            .expect("the embedded catalogues are not empty");
        let t = tune_item(&item, None);
        let sliders = backend.sliders_for_item(&item, &[]);
        let mut st = Self {
            mode: BrowseMode::ByGroup,
            selected_group_idx,
            rule,
            catalogue,
            selected_cat_idx,
            selected_named_idx,
            base_item: item.clone(),
            slider_state: None,
            sliders,
            last_move_secs: 0.0,
            retune: None,
            pending_moves: Vec::new(),
            repeat: 3,
            show_unit_cell: true,
            overlays: Default::default(),
            inflation: inflation_ui::GalleryInflationState::default(),
            readout: t.readout,
            active_item: item.clone(),
            cell: t.cell,
            revision: 0,
            renaming: false,
            rename_buf: String::new(),
            rename_focus_pending: false,
            rename_has_focus: false,
            alias_note: None,
            custom_builder: builder_ui::CustomBuilderState::new(),
            view: browser::BrowserView::default(),
            filter: filter::BrowseFilter::default(),
            thumbs: thumb_cache::LatticeThumbCache::default(),
            rule_index: [None, None, None],
        };
        st.activate_item(item, backend, canvas, true);
        st.revision = 0;
        st
    }

    /// The lattice at the current slider values (see the module documentation).
    pub fn active_item(&self) -> &CatalogueItemUi {
        &self.active_item
    }

    /// The selected catalogue / named item at its default values.
    pub fn base_item(&self) -> &CatalogueItemUi {
        &self.base_item
    }

    /// Increases on every change of `active_item()` (lattice or parameter values), not on view changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The cell drawn in the Gallery (the base family at the slider values, in the item's frame).
    pub fn cell(&self) -> &Cell2D {
        &self.cell
    }

    pub fn sliders(&self) -> &[SliderParamUi] {
        &self.sliders
    }

    /// Current values of the free parameters, in slider order.
    pub fn param_values(&self) -> Vec<(String, f64)> {
        self.sliders.iter().map(|s| (s.name.clone(), s.value)).collect()
    }

    #[cfg(test)]
    pub fn catalogue(&self) -> &[CatalogueItemUi] {
        &self.catalogue
    }

    /// The all-groups list of `rule` if already built by [`Self::rule_index`] (empty otherwise).
    pub(crate) fn rule_index_ready(&self, rule: CatalogueRuleUi) -> &[CatalogueItemUi] {
        self.rule_index[rule.index()].as_deref().unwrap_or_default()
    }

    pub fn select_group_by_symbol(&mut self, symbol: &str, backend: &AppBackend, canvas: &mut Canvas2D) {
        if let Some(idx) = backend.groups().iter().position(|g| g.symbol == symbol) {
            self.select_group(idx, backend, canvas);
        }
    }

    /// Show the list of group `idx` (rule unchanged) and select its first lattice.
    pub fn select_group(&mut self, idx: usize, backend: &AppBackend, canvas: &mut Canvas2D) {
        if idx < backend.groups().len() {
            self.selected_group_idx = idx;
            self.mode = BrowseMode::ByGroup;
            self.reload_group_catalogue(backend, canvas);
        }
    }

    pub fn select_rule(&mut self, rule: CatalogueRuleUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        self.rule = rule;
        self.mode = BrowseMode::ByGroup;
        self.reload_group_catalogue(backend, canvas);
    }

    /// Select entry `idx` of the current group list (ignored if out of range).
    pub fn select_catalogue_entry(&mut self, idx: usize, backend: &AppBackend, canvas: &mut Canvas2D) {
        if let Some(item) = self.catalogue.get(idx).cloned() {
            self.selected_cat_idx = idx;
            self.activate_item(item, backend, canvas, true);
        }
    }

    /// Browse the named lattices and select number `idx` (ignored if out of range).
    pub fn select_named(&mut self, idx: usize, backend: &AppBackend, canvas: &mut Canvas2D) {
        if let Some(item) = backend.named_lattices().get(idx).cloned() {
            self.mode = BrowseMode::Named21;
            self.selected_named_idx = idx;
            self.activate_item(item, backend, canvas, true);
        }
    }

    fn reload_group_catalogue(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        let Some(grp) = backend.groups().get(self.selected_group_idx).map(|g| g.symbol) else {
            return;
        };
        self.catalogue = backend.catalogue_for_group(grp, self.rule);
        self.selected_cat_idx = 0;
        if let Some(first) = self.catalogue.first().cloned() {
            self.activate_item(first, backend, canvas, true);
        }
    }

    fn activate_item(&mut self, item: CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D, reset_view: bool) {
        if let Some(g_idx) = backend.groups().iter().position(|g| g.symbol == item.group) {
            self.selected_group_idx = g_idx;
        }
        self.slider_state = SliderState::new(item.recipe.clone()).ok().filter(|s| !s.sliders.is_empty());
        self.base_item = item;
        self.cancel_retune();
        self.last_move_secs = 0.0;
        self.renaming = false;
        self.rename_focus_pending = false;
        self.alias_note = None;
        self.retune(canvas, reset_view);
    }

    pub fn begin_rename(&mut self) {
        self.renaming = true;
        self.rename_buf = self.active_item.label.clone();
        self.rename_focus_pending = true;
    }

    pub fn cancel_rename(&mut self) {
        self.renaming = false;
        self.rename_focus_pending = false;
    }

    /// A rename is in progress and its field has (or is about to get) the keyboard. A pending rename whose field
    /// lost focus does not hold Enter / Esc / arrows.
    pub(crate) fn rename_holds_keys(&self) -> bool {
        self.renaming && (self.rename_has_focus || self.rename_focus_pending)
    }

    pub fn commit_rename(&mut self, backend: &mut AppBackend, canvas: &mut Canvas2D) {
        let name = self.rename_buf.clone();
        self.renaming = false;
        self.rename_focus_pending = false;
        self.rename_active_lattice(&name, backend, canvas);
    }

    pub fn rename_active_lattice(&mut self, new_name: &str, backend: &mut AppBackend, canvas: &mut Canvas2D) {
        self.alias_note = backend
            .set_lattice_alias(&self.base_item, new_name)
            .err()
            .map(|e| format!("Saved for session only ({e})"));
        self.refresh_aliases(backend, canvas);
    }

    pub fn reset_active_lattice_name(&mut self, backend: &mut AppBackend, canvas: &mut Canvas2D) {
        self.renaming = false;
        self.rename_focus_pending = false;
        self.alias_note = backend
            .reset_lattice_alias(&self.base_item)
            .err()
            .map(|e| format!("Reset for session only ({e})"));
        self.refresh_aliases(backend, canvas);
    }

    fn refresh_aliases(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        for it in self.catalogue.iter_mut().chain(self.rule_index.iter_mut().flatten().flatten()) {
            it.label = backend.resolve_alias(it);
        }
        let next_lbl = backend.resolve_alias(&self.base_item);
        if self.active_item.label != next_lbl || self.base_item.label != next_lbl {
            self.base_item.label = next_lbl.clone();
            self.active_item.label = next_lbl;
            self.revision += 1;
            self.sync_scene(canvas, false);
        }
    }

    /// Move parameter `name` as the slider dragged to `raw` does (step grid, limits, snap onto a special value
    /// within half a step); returns the stored value, `None` for an unknown name or NaN.
    pub fn drag_param(&mut self, name: &str, raw: f64, canvas: &mut Canvas2D) -> Option<f64> {
        let t0 = std::time::Instant::now();
        self.flush_pending_moves();
        let v = self.slider_state.as_mut()?.drag_to(name, raw).ok()?;
        self.retune(canvas, false);
        self.last_move_secs = t0.elapsed().as_secs_f64();
        Some(v)
    }

    /// Set parameter `name` to `value` (clamped to its limits, snapped within half a step onto a special value),
    /// as `box.sliders[name].value = value` in Python; returns the stored value.
    pub fn set_param(&mut self, name: &str, value: f64, canvas: &mut Canvas2D) -> Option<f64> {
        self.flush_pending_moves();
        let v = self.slider_state.as_mut()?.set_value(name, value).ok()?;
        self.retune(canvas, false);
        Some(v)
    }

    /// Every parameter back to its default value (the Reset button).
    pub fn reset_params(&mut self, canvas: &mut Canvas2D) {
        self.flush_pending_moves();
        if let Some(st) = self.slider_state.as_mut()
            && st.reset().is_ok()
        {
            self.retune(canvas, false);
        }
    }

    /// Recompute sliders, cell, status and the exported item after a change of lattice or values.
    fn retune(&mut self, canvas: &mut Canvas2D, reset_view: bool) {
        let sliders = self.slider_state.as_ref().map(slider_params_ui).unwrap_or_default();
        let t = tune_item(&self.base_item, self.slider_state.as_ref());
        self.apply_tuned(sliders, t);
        self.sync_scene(canvas, reset_view);
    }

    /// Store a retuned lattice (synchronous or background move); bumps the revision when the item changed.
    fn apply_tuned(&mut self, sliders: Vec<SliderParamUi>, t: crate::live_plane::TunedLattice) {
        self.sliders = sliders;
        let next = match t.item {
            Some(item) => item,
            // not writable as a recipe here: keep the last exportable item of this lattice
            None if same_lattice_base(&self.active_item, &self.base_item) => self.active_item.clone(),
            None => self.base_item.clone(),
        };
        if !same_lattice(&next, &self.active_item) {
            self.revision += 1;
        }
        self.active_item = next;
        self.cell = t.cell;
        self.readout = t.readout;
    }

    pub(super) fn sync_scene(&mut self, canvas: &mut Canvas2D, reset_view: bool) {
        self.draw_cell(self.cell().clone(), canvas, reset_view);
    }

    fn draw_cell(&mut self, cell: Cell2D, canvas: &mut Canvas2D, reset_view: bool) {
        let title = format!("{} ({})", self.active_item.display_with_orig(), self.active_item.group);
        let inflated = if self.inflation.enabled {
            self.inflation.build_scene(&cell, self.show_unit_cell).ok()
        } else {
            self.inflation.clear_status();
            None
        };
        let overlay_repeat = inflated.as_ref().map_or(self.repeat, |&(_, rep)| rep);
        let overlay = self.overlays.any().then(|| {
            let it = &self.active_item;
            crate::scene2d::overlays::build_gallery_overlay(&cell, &it.group, it.frame, overlay_repeat, self.overlays)
        });
        let scene = match inflated {
            Some((scene, _)) => scene,
            None => Scene2D::PeriodicCell { cell, repeat: self.repeat, show_unit_cell: self.show_unit_cell },
        };
        canvas.set_scene_with_title(scene, Some(title), reset_view);
        if overlay.is_some() {
            canvas.set_gallery_overlay(overlay);
        }
    }

    /// View only: which scientific overlays are drawn (symmetry elements, Voronoi cells).
    pub fn set_overlays(&mut self, overlays: crate::scene2d::overlays::GalleryOverlayToggles, canvas: &mut Canvas2D) {
        self.overlays = overlays;
        self.sync_scene(canvas, false);
    }

    /// View only: toggle between the periodic grid and finite shell inflation patch.
    pub fn set_inflation_enabled(&mut self, enabled: bool, canvas: &mut Canvas2D) {
        self.inflation.enabled = enabled;
        self.sync_scene(canvas, true);
    }

    /// Draw the current lattice again from the stored item and the slider values (e.g. back on the Gallery tab).
    pub fn sync_canvas(&mut self, backend: &AppBackend, canvas: &mut Canvas2D, reset_view: bool) {
        let (cell, _) = backend.build_cell(&self.base_item, &self.param_values());
        self.draw_cell(cell, canvas, reset_view);
    }

    /// View only: patch size (1..=6 cells each way), preserving current pan/zoom.
    pub fn set_repeat(&mut self, repeat: i32, canvas: &mut Canvas2D) {
        self.repeat = repeat.clamp(1, 6);
        self.sync_scene(canvas, false);
    }

    /// View only: draw the unit cell vectors.
    pub fn set_show_unit_cell(&mut self, show: bool, canvas: &mut Canvas2D) {
        self.show_unit_cell = show;
        self.sync_scene(canvas, false);
    }
}

/// `a` was derived from `base` (same catalogue entry or named lattice; the recipe may differ).
fn same_lattice_base(a: &CatalogueItemUi, base: &CatalogueItemUi) -> bool {
    a.frame == base.frame && a.group == base.group && a.label == base.label && a.named_key == base.named_key && a.index == base.index
}
