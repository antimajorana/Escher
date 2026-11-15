use std::sync::OnceLock;

use papertilling_core::defect::combine_types::{EquivalentSingleReport, NotebookConfigJson};
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::types::{
    BuiltCrystal, Cell2D, Frame2D, Recipe, SurfaceCheckReport, SurfaceMesh,
};

use crate::live_combine::{
    self, CombineBuildOptions, CombineBuildOutput, CombineLatticeInfo,
};
use crate::live_ops;
use crate::live_plane::{self, PlaneCatalogues};

#[derive(Clone, Debug)]
pub struct GroupInfo {
    pub symbol: &'static str,
    pub crystal_system: &'static str,
    pub point_order: u8,
    pub max_rotation: u8,
    pub has_reflection_or_glide: bool,
    pub counts: [usize; 3], // [points, sheet, bonds]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogueRuleUi {
    Points,
    Sheet,
    Bonds,
}

impl CatalogueRuleUi {
    pub const ALL: [Self; 3] = [Self::Points, Self::Sheet, Self::Bonds];

    pub fn label(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::Sheet => "sheet",
            Self::Bonds => "bonds",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Points => 0,
            Self::Sheet => 1,
            Self::Bonds => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WyckoffRowUi {
    pub letter: char,
    pub multiplicity: u8,
    pub site_symmetry: String,
    pub coords_expr: String,
    pub occupied: bool,
}

#[derive(Clone, Debug)]
pub struct CatalogueItemUi {
    pub index: usize,
    pub label: String,
    pub named_key: Option<String>,
    pub group: String,
    pub sheet_group: Option<String>,
    pub sites_per_cell: usize,
    pub site_orbits: usize,
    pub bond_orbits: usize,
    pub coordination: String,
    pub dsymbol_key: String,
    pub network_key: String,
    pub symmetry_index: usize,
    pub occupied_wyckoff: Vec<char>,
    pub recipe: Recipe,
    pub frame: Frame2D,
}

#[derive(Clone, Debug)]
pub struct SpecialTickUi {
    pub value: f64,
    pub supergroup: String,
    pub equation: String,
}

#[derive(Clone, Debug)]
pub struct ForbiddenSpanUi {
    pub lo: f64,
    pub hi: f64,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct SliderParamUi {
    pub name: String,
    pub value: f64,
    pub default_value: f64,
    pub nominal_min: f64,
    pub nominal_max: f64,
    pub allowed_min: f64,
    pub allowed_max: f64,
    pub step: f64,
    pub special_ticks: Vec<SpecialTickUi>,
    pub forbidden: Vec<ForbiddenSpanUi>,
}

#[derive(Clone, Debug)]
pub struct DSymbolReadoutUi {
    pub dsymbol_key: String,
    pub network_key: String,
    pub symmetry_index: usize,
    pub active_group: String,
    pub status_ok: bool,
    pub status_message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceTargetUi {
    Torus,
    Cylinder,
    MobiusBand,
    KleinBottle,
    Sphere,
    ProjectivePlane,
    Genus2,
    Genus3,
}

impl SurfaceTargetUi {
    pub const ALL: [Self; 8] = [
        Self::Torus,
        Self::Cylinder,
        Self::MobiusBand,
        Self::KleinBottle,
        Self::Sphere,
        Self::ProjectivePlane,
        Self::Genus2,
        Self::Genus3,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Torus => "torus",
            Self::Cylinder => "cylinder",
            Self::MobiusBand => "Mobius band",
            Self::KleinBottle => "Klein bottle",
            Self::Sphere => "sphere",
            Self::ProjectivePlane => "projective plane",
            Self::Genus2 => "genus 2",
            Self::Genus3 => "genus 3",
        }
    }

    pub fn short_header(self) -> &'static str {
        match self {
            Self::Torus => "torus",
            Self::Cylinder => "cyl",
            Self::MobiusBand => "Mobius",
            Self::KleinBottle => "Klein",
            Self::Sphere => "sphere",
            Self::ProjectivePlane => "RP²",
            Self::Genus2 => "g=2",
            Self::Genus3 => "g=3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreModeUi {
    Auto,
    Face,
    Site,
}

impl CentreModeUi {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Face, Self::Site];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Face => "face",
            Self::Site => "site",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceBuildParamsUi {
    pub h: i32,
    pub k: i32,
    pub genus_g: u32,
    pub centre_mode: CentreModeUi,
}

impl Default for SurfaceBuildParamsUi {
    fn default() -> Self {
        Self {
            h: 4,
            k: 4,
            genus_g: 2,
            centre_mode: CentreModeUi::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerdictKindUi {
    Ok,
    Impossible,
    NotFound,
}

impl VerdictKindUi {
    pub fn badge(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Impossible => "IMPOSSIBLE",
            Self::NotFound => "NOT FOUND",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Impossible => "IMP",
            Self::NotFound => "N/F",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceCellVerdictUi {
    pub verdict: VerdictKindUi,
    pub sites: Option<usize>,
    pub cones_summary: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct SurfaceVerdictRowUi {
    pub group: String,
    pub lattice_name: String,
    pub cells: [SurfaceCellVerdictUi; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WallKindUi {
    DomainWa,
    TwinWb,
    MobiusWc,
    GlideWd,
}

impl WallKindUi {
    pub fn label(self) -> &'static str {
        match self {
            Self::DomainWa => "Domain Wa",
            Self::TwinWb => "Twin Wb",
            Self::MobiusWc => "Möbius Wc",
            Self::GlideWd => "Glide-breaking Wd",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DislocationEntryUi {
    pub id: String,
    pub burgers_mn: (i32, i32),
    pub norm: f64,
    pub orbit_size: usize,
    pub mirror_partner: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CentreClassUi {
    pub id: String,
    pub wyckoff: String,
    pub order: u8,
    pub location: String,
}

#[derive(Clone, Debug)]
pub struct DisclinationEntryUi {
    pub id: String,
    pub centre_id: String,
    pub frank_angle_deg: i32,
    pub location: String,
    pub wyckoff: String,
}

#[derive(Clone, Debug)]
pub struct WallEntryUi {
    pub id: String,
    pub kind: WallKindUi,
    pub direction_mn: (i32, i32),
    pub description: String,
    pub default_note: String,
}

#[derive(Clone, Debug)]
pub struct DefectCatalogueUi {
    pub dislocations: Vec<DislocationEntryUi>,
    pub centres: Vec<CentreClassUi>,
    pub disclinations: Vec<DisclinationEntryUi>,
    pub walls: Vec<WallEntryUi>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefectSelectionUi {
    Dislocation(usize),
    Disclination(usize),
    Wall(usize),
}

#[derive(Clone, Debug)]
pub struct DefectReadoutUi {
    pub title: String,
    pub charge_summary: String,
    pub holonomy_detail: String,
    pub core_rings: String,
    pub note: String,
}

pub struct AppBackend {
    cats: &'static PlaneCatalogues,
    groups: Vec<GroupInfo>,
    named: Vec<CatalogueItemUi>,
    verdict_matrix: OnceLock<Vec<SurfaceVerdictRowUi>>,
}

impl Default for AppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBackend {
    pub fn new() -> Self {
        let cats = live_plane::shared_catalogues();
        let groups = live_plane::build_groups_info(cats);
        let named = live_plane::build_named_lattices(cats);
        Self {
            cats,
            groups,
            named,
            verdict_matrix: OnceLock::new(),
        }
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
        live_plane::catalogue_for_group(self.cats, group, rule)
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

    pub fn surface_verdict_matrix(&self) -> &[SurfaceVerdictRowUi] {
        self.verdict_matrix
            .get_or_init(|| live_ops::build_verdict_matrix(self.cats))
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

    pub fn build_combined(
        &self,
        item: &CatalogueItemUi,
        info: &CombineLatticeInfo,
        specs: &[DefectSpec],
        opts: &CombineBuildOptions,
    ) -> CombineBuildOutput {
        live_combine::build_combined_crystal(item, info, specs, opts)
    }

    pub fn equivalent_single(
        &self,
        item: &CatalogueItemUi,
        info: &CombineLatticeInfo,
        specs: &[DefectSpec],
        crystal: &BuiltCrystal,
        core_radius: usize,
        grid: usize,
    ) -> Result<EquivalentSingleReport, String> {
        live_combine::run_equivalent_single(item, info, specs, crystal, core_radius, grid)
    }

    pub fn item_from_notebook_json(
        &self,
        cfg: &NotebookConfigJson,
    ) -> Result<CatalogueItemUi, String> {
        live_combine::item_from_notebook_json(cfg, &self.named)
    }
}
