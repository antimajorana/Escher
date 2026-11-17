use std::sync::{Once, OnceLock};

use papertilling_core::defect::combine_types::NotebookConfigJson;
use papertilling_core::types::{BuiltCrystal, Cell2D, SurfaceCheckReport, SurfaceMesh};

use crate::aliases::AliasStore;
use crate::live_combine::{self, CombineLatticeInfo};
use crate::live_ops;
use crate::live_plane::{self, PlaneCatalogues};

pub mod types;
pub use types::*;

pub struct AppBackend {
    cats: &'static PlaneCatalogues,
    groups: Vec<GroupInfo>,
    named: Vec<CatalogueItemUi>,
    aliases: AliasStore,
}

/// The 17 x 8 surface verdict matrix depends only on the stored catalogues: one per process.
fn verdict_matrix_cell() -> &'static OnceLock<Vec<SurfaceVerdictRowUi>> {
    static CELL: OnceLock<Vec<SurfaceVerdictRowUi>> = OnceLock::new();
    &CELL
}

impl Default for AppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBackend {
    pub fn new() -> Self {
        Self::new_with_alias_store(AliasStore::in_memory())
    }

    pub fn new_with_os_config(smoke_test: bool) -> Self {
        let aliases = if smoke_test || cfg!(test) {
            AliasStore::in_memory()
        } else {
            AliasStore::load_default()
        };
        Self::new_with_alias_store(aliases)
    }

    pub fn new_with_alias_store(aliases: AliasStore) -> Self {
        let cats = live_plane::shared_catalogues();
        let groups = live_plane::build_groups_info(cats);
        let named = live_plane::build_named_lattices(cats, &aliases);
        Self { cats, groups, named, aliases }
    }

    pub fn resolve_alias(&self, item: &CatalogueItemUi) -> String {
        self.aliases
            .get(&item.alias_key)
            .map(str::to_owned)
            .unwrap_or_else(|| item.default_label.clone())
    }

    pub fn set_lattice_alias(&mut self, item: &CatalogueItemUi, new_name: &str) -> Result<(), String> {
        let res = self.aliases.set(item.alias_key.clone(), new_name, &item.default_label);
        self.named = live_plane::build_named_lattices(self.cats, &self.aliases);
        res
    }

    pub fn reset_lattice_alias(&mut self, item: &CatalogueItemUi) -> Result<(), String> {
        let res = self.aliases.reset(&item.alias_key);
        self.named = live_plane::build_named_lattices(self.cats, &self.aliases);
        res
    }

    /// Start computing the surface verdict matrix (about 1.3 s) off the UI thread.
    pub fn prefetch_surface_verdicts(&self) {
        static START: Once = Once::new();
        START.call_once(|| {
            let cats = self.cats;
            let _ = std::thread::Builder::new().name(String::from("verdict-matrix")).spawn(move || {
                verdict_matrix_cell().get_or_init(|| live_ops::build_verdict_matrix(cats));
            });
        });
    }

    pub fn groups(&self) -> &[GroupInfo] {
        &self.groups
    }

    pub fn named_lattices(&self) -> &[CatalogueItemUi] {
        &self.named
    }

    pub fn find_named(&self, key: &str) -> Option<&CatalogueItemUi> {
        self.named
            .iter()
            .find(|it| it.named_key.as_deref() == Some(key))
    }

    pub fn catalogue_for_group(
        &self,
        group: &str,
        rule: CatalogueRuleUi,
    ) -> Vec<CatalogueItemUi> {
        live_plane::catalogue_for_group(self.cats, group, rule, &self.aliases)
    }

    pub fn wyckoff_table(&self, group: &str, occupied: &[char]) -> Vec<WyckoffRowUi> {
        live_plane::wyckoff_rows_for_group(group, occupied)
    }

    pub fn sliders_for_item(
        &self,
        item: &CatalogueItemUi,
        current_vals: &[(String, f64)],
    ) -> Vec<SliderParamUi> {
        live_plane::sliders_for_item(item, current_vals)
    }

    pub fn build_cell(
        &self,
        item: &CatalogueItemUi,
        params: &[(String, f64)],
    ) -> (Cell2D, DSymbolReadoutUi) {
        live_plane::build_cell_for_item(item, params)
    }

    #[cfg(test)]
    pub fn surface_verdict_matrix(&self) -> &[SurfaceVerdictRowUi] {
        verdict_matrix_cell().get_or_init(|| live_ops::build_verdict_matrix(self.cats))
    }

    pub fn surface_verdict_matrix_ready(&self) -> Option<&[SurfaceVerdictRowUi]> {
        self.prefetch_surface_verdicts();
        verdict_matrix_cell().get().map(Vec::as_slice)
    }

    pub fn build_surface(
        &self,
        item: &CatalogueItemUi,
        target: SurfaceTargetUi,
        params: &SurfaceBuildParamsUi,
    ) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
        live_ops::build_surface_mesh(item, target, params)
    }

    pub fn defect_catalogue(&self, item: &CatalogueItemUi) -> DefectCatalogueUi {
        live_ops::catalogue_for_lattice(item)
    }

    pub fn build_defect(
        &self,
        item: &CatalogueItemUi,
        selection: &DefectSelectionUi,
        radius: f64,
        relaxed: bool,
    ) -> (BuiltCrystal, DefectReadoutUi) {
        live_ops::build_defect_patch(item, selection, radius, relaxed)
    }

    pub fn combine_lattice_info(&self, item: &CatalogueItemUi) -> CombineLatticeInfo {
        live_combine::combine_lattice_info(item)
    }

    pub fn snap_disclination(
        &self,
        item: &CatalogueItemUi,
        info: &CombineLatticeInfo,
        angle_deg: f64,
        centre_id_filter: Option<&str>,
        near: [f64; 2],
    ) -> ([f64; 2], String) {
        live_combine::snap_disclination_centre(item, info, angle_deg, centre_id_filter, near)
    }

    pub fn item_from_notebook_json(
        &self,
        cfg: &NotebookConfigJson,
    ) -> Result<CatalogueItemUi, String> {
        live_combine::item_from_notebook_json(cfg, &self.named)
    }
}
