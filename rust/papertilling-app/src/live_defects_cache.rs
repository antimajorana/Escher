//! The Defects tab's per-lattice data (cell, `DefectCatalogue`, `LatticeCoreInfo`), computed once per lattice:
//! building the catalogue takes up to ~2 s (p1: twin-wall network tests) and was redone at every click.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Cell2D, Frame2D, Recipe};

use crate::backend::CatalogueItemUi;

pub(crate) struct DefectContext {
    pub cell: Cell2D,
    pub cat: DefectCatalogue,
    pub lat_info: LatticeCoreInfo,
}

type Key = (String, Recipe, Frame2D);
type Slots = Vec<(Key, Arc<DefectContext>)>;

const CAPACITY: usize = 4;

fn slots() -> &'static Mutex<Slots> {
    static SLOTS: OnceLock<Mutex<Slots>> = OnceLock::new();
    SLOTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// The cached context of `item` (same group, recipe and frame), built on first use; `Err` for a recipe that
/// does not make a lattice (no panic).
pub(crate) fn defect_context(item: &CatalogueItemUi) -> Result<Arc<DefectContext>, String> {
    let key: Key = (item.group.clone(), item.recipe.clone(), item.frame);
    {
        let guard = slots().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, hit)) = guard.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(hit));
        }
    }
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    if cell.basis.is_empty() {
        return Err(format!("{}: the recipe gives no sites", item.label));
    }
    let ctx = Arc::new(DefectContext {
        cat: DefectCatalogue::new(&item.group, &cell, item.frame),
        lat_info: LatticeCoreInfo::new(&cell),
        cell,
    });
    let mut guard = slots().lock().unwrap_or_else(PoisonError::into_inner);
    guard.retain(|(k, _)| *k != key);
    guard.insert(0, (key, Arc::clone(&ctx)));
    guard.truncate(CAPACITY);
    Ok(ctx)
}
