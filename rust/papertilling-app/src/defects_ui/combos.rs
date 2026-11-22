//! Background worker and UI state for the Defects > Combinations sub-tab.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use papertilling_core::defect::algebraic_fusion::{
    SpaceGroupAlgebraReport, analyze_space_group_fusion,
};
use papertilling_core::defect::combinations::build_combinations_table_cached;
use papertilling_core::defect::combinations_types::{
    CombinationMode, CombinationsOptions, CombinationsTable,
};
use papertilling_core::defect::single_cache::SingleDefectCache;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Frame2D, Recipe};

use crate::aliases::LatticeAliasKey;
use crate::backend::{CatalogueItemUi, DefectCatalogueUi};
use crate::combine_worker::LatestWorker;
use crate::core_panel::CoreInspectorModal;
use crate::scene2d::ScenePrimitives;

const TRIPLE_ID_A: &str = "D(1b, +90)";
const TRIPLE_ID_B: &str = "D(1a, -90)";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ComboModeKind {
    #[default]
    DisclinationPairs,
    DislocationPairs,
    MixedPairs,
    TriplePreset,
}

impl ComboModeKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::DisclinationPairs => "Disclination pairs",
            Self::DislocationPairs => "Dislocation pairs",
            Self::MixedPairs => "Mixed pairs",
            Self::TriplePreset => "Triple: D(1b,+90) + 2×D(1a,-90)",
        }
    }

    pub(crate) fn has_triple_preset(cat: &DefectCatalogueUi) -> bool {
        let has_a = cat.disclinations.iter().any(|d| d.id == TRIPLE_ID_A);
        let has_b = cat.disclinations.iter().any(|d| d.id == TRIPLE_ID_B);
        has_a && has_b
    }

    pub(crate) fn available_for(cat: &DefectCatalogueUi) -> Vec<Self> {
        let mut out = vec![
            Self::DisclinationPairs,
            Self::DislocationPairs,
            Self::MixedPairs,
        ];
        if Self::has_triple_preset(cat) {
            out.push(Self::TriplePreset);
        }
        out
    }

    pub(crate) fn to_core_mode(self) -> CombinationMode {
        match self {
            Self::DisclinationPairs => CombinationMode::DisclinationPairs,
            Self::DislocationPairs => CombinationMode::DislocationPairs,
            Self::MixedPairs => CombinationMode::MixedPairs,
            Self::TriplePreset => CombinationMode::Triple {
                ids: [
                    TRIPLE_ID_A.to_string(),
                    TRIPLE_ID_B.to_string(),
                    TRIPLE_ID_B.to_string(),
                ],
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CombosKey {
    alias_key: LatticeAliasKey,
    group: String,
    recipe: Recipe,
    frame: Frame2D,
    mode: ComboModeKind,
    max_sep_q: i32,
    core_radius: usize,
    scan_grid: usize,
}

struct CombosJob {
    item: CatalogueItemUi,
    mode: CombinationMode,
    max_sep_scale: f64,
    core_radius: usize,
    scan_grid: usize,
    bmax: f64,
    cancel: Arc<AtomicBool>,
}

pub(crate) struct CombosOutput {
    alias_key: LatticeAliasKey,
    a_scale: f64,
    table: CombinationsTable,
}

pub(crate) struct CombosState {
    worker: LatestWorker<CombosJob, Result<CombosOutput, String>>,
    cancel: Option<Arc<AtomicBool>>,
    pub(crate) mode_kind: ComboModeKind,
    pub(crate) max_sep_scale: f64,
    pub(crate) core_radius: usize,
    pub(crate) scan_grid: usize,
    pub(crate) bmax: f64,
    pub(crate) table: Option<CombinationsTable>,
    table_alias: Option<LatticeAliasKey>,
    pub(crate) a_scale: f64,
    pub(crate) error: Option<String>,
    computed_for: Option<CombosKey>,
    pub(crate) algebra: Option<Result<SpaceGroupAlgebraReport, String>>,
    algebra_for: Option<String>,
    pub(crate) algebra_filter: String,
    pub(crate) selected: Option<(usize, usize, usize)>,
    pub(crate) thumb_cache: HashMap<String, ScenePrimitives>,
    pub(crate) inspector_modal: Option<CoreInspectorModal>,
    pub(crate) cancelled_by_user: bool,
}

impl Drop for CombosState {
    fn drop(&mut self) {
        if let Some(flag) = self.cancel.take() {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

impl CombosState {
    pub(crate) fn new() -> Self {
        let worker = LatestWorker::spawn(
            "defects-combos",
            SingleDefectCache::default(),
            |cache: &mut SingleDefectCache, job: CombosJob| -> Result<CombosOutput, String> {
                let ctx = crate::live_ops::defect_context(&job.item)?;
                let a_scale = ctx.cell.a1[0].hypot(ctx.cell.a1[1]).max(1e-9);
                let opts = CombinationsOptions {
                    max_sep: job.max_sep_scale * a_scale,
                    radius: job.core_radius,
                    grid: job.scan_grid,
                    draw_thumbnails: true,
                    include_single_gallery: false,
                    bmax: job.bmax,
                };
                let table = build_combinations_table_cached(
                    &job.item.group,
                    &ctx.cell,
                    job.item.frame,
                    &job.mode,
                    &opts,
                    cache,
                    Some(&job.cancel),
                )?;
                Ok(CombosOutput {
                    alias_key: job.item.alias_key,
                    a_scale,
                    table,
                })
            },
        );
        Self {
            worker,
            cancel: None,
            mode_kind: ComboModeKind::DisclinationPairs,
            max_sep_scale: 1.5,
            core_radius: 2,
            scan_grid: 1,
            bmax: 2.0,
            table: None,
            table_alias: None,
            a_scale: 1.0,
            error: None,
            computed_for: None,
            algebra: None,
            algebra_for: None,
            algebra_filter: String::new(),
            selected: None,
            thumb_cache: HashMap::new(),
            inspector_modal: None,
            cancelled_by_user: false,
        }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.worker.is_busy()
    }

    pub(crate) fn table_for_item(&self, item: &CatalogueItemUi) -> Option<&CombinationsTable> {
        self.table
            .as_ref()
            .filter(|_| self.table_alias.as_ref() == Some(&item.alias_key))
    }

    fn make_key(&self, item: &CatalogueItemUi) -> CombosKey {
        CombosKey {
            alias_key: item.alias_key.clone(),
            group: item.group.clone(),
            recipe: item.recipe.clone(),
            frame: item.frame,
            mode: self.mode_kind,
            max_sep_q: (self.max_sep_scale * 100.0).round() as i32,
            core_radius: self.core_radius,
            scan_grid: self.scan_grid,
        }
    }

    fn flip_cancel(&mut self) {
        if let Some(flag) = self.cancel.take() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Cancel any running scan and clear all cached results (used on lattice reload).
    pub(crate) fn clear_and_cancel(&mut self) {
        self.flip_cancel();
        self.worker.cancel();
        self.table = None;
        self.table_alias = None;
        self.error = None;
        self.computed_for = None;
        self.algebra = None;
        self.algebra_for = None;
        self.algebra_filter.clear();
        self.selected = None;
        self.thumb_cache.clear();
        self.inspector_modal = None;
        self.cancelled_by_user = false;
    }

    /// Synchronously ensure the exact `[PROVED]` space-group algebraic fusion report is cached
    /// for `item` (runs in milliseconds without waiting for Volterra surgery).
    pub(crate) fn ensure_algebra(&mut self, item: &CatalogueItemUi) {
        let key = format!(
            "{:?}|{}|{:?}|{:?}|{:.3}",
            item.alias_key, item.group, item.recipe, item.frame, self.bmax
        );
        if self.algebra_for.as_deref() == Some(key.as_str()) && self.algebra.is_some() {
            return;
        }
        let res = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).and_then(|fam| {
            let cell = fam.cell();
            if cell.basis.is_empty() {
                Err(format!("{}: the recipe gives no sites", item.label))
            } else {
                analyze_space_group_fusion(&item.group, &cell, item.frame, self.bmax)
            }
        });
        self.algebra = Some(res);
        self.algebra_for = Some(key);
    }

    /// Cancel the in-flight computation without auto-restarting on the next frame.
    pub(crate) fn cancel_current(&mut self, item: &CatalogueItemUi) {
        self.flip_cancel();
        self.worker.cancel();
        self.computed_for = Some(self.make_key(item));
        self.cancelled_by_user = true;
    }

    pub(crate) fn submit(&mut self, item: &CatalogueItemUi) {
        self.flip_cancel();
        let flag = Arc::new(AtomicBool::new(false));
        self.cancel = Some(Arc::clone(&flag));
        self.cancelled_by_user = false;
        self.error = None;
        self.computed_for = Some(self.make_key(item));
        self.worker.submit(CombosJob {
            item: item.clone(),
            mode: self.mode_kind.to_core_mode(),
            max_sep_scale: self.max_sep_scale,
            core_radius: self.core_radius,
            scan_grid: self.scan_grid,
            bmax: self.bmax,
            cancel: flag,
        });
    }

    /// Start computation lazily once when entering Combinations or when item/controls change.
    pub(crate) fn ensure_started(&mut self, item: &CatalogueItemUi, cat: &DefectCatalogueUi) {
        if self.mode_kind == ComboModeKind::TriplePreset && !ComboModeKind::has_triple_preset(cat) {
            self.mode_kind = ComboModeKind::DisclinationPairs;
        }
        let key = self.make_key(item);
        if self.computed_for.as_ref() != Some(&key) {
            self.submit(item);
        }
    }

    pub(crate) fn poll(&mut self, ctx: &egui::Context) {
        self.worker.set_repaint_ctx(ctx);
        if let Some(outer) = self.worker.poll() {
            match outer.and_then(|inner| inner) {
                Ok(out) => {
                    self.selected = first_member_index(&out.table);
                    self.table = Some(out.table);
                    self.table_alias = Some(out.alias_key);
                    self.a_scale = out.a_scale;
                    self.error = None;
                    self.thumb_cache.clear();
                    self.inspector_modal = None;
                }
                Err(err) => {
                    self.error = Some(err);
                }
            }
        }
    }
}

fn first_member_index(table: &CombinationsTable) -> Option<(usize, usize, usize)> {
    for (cg_i, cg) in table.charge_groups.iter().enumerate() {
        for (core_i, core) in cg.core_groups.iter().enumerate() {
            if !core.members.is_empty() {
                return Some((cg_i, core_i, 0));
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "combos_tests.rs"]
mod tests;
