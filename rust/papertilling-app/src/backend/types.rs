use papertilling_core::types::{Frame2D, Recipe};

use crate::aliases::LatticeAliasKey;

#[derive(Clone, Debug)]
pub struct GroupInfo {
    pub symbol: &'static str,
    pub crystal_system: &'static str,
    pub point_order: u8,
    pub max_rotation: u8,
    pub has_reflection_or_glide: bool,
    pub counts: [usize; 3], // [points, sheet, bonds]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CatalogueRuleUi {
    #[default]
    Bonds,
    Sheet,
    Points,
}

impl CatalogueRuleUi {
    pub const ALL: [Self; 3] = [Self::Bonds, Self::Sheet, Self::Points];

    pub fn slug(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::Sheet => "sheet",
            Self::Bonds => "bonds",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Points => "Delaunay",
            Self::Sheet => "Atomic",
            Self::Bonds => "Standard",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Points => "Delaunay lattice",
            Self::Sheet => "Atomic crystals",
            Self::Bonds => "Standard lattice",
        }
    }

    #[cfg(test)]
    pub fn matches(self, s: &str) -> bool {
        let q = s.trim();
        q.eq_ignore_ascii_case(self.slug())
            || q.eq_ignore_ascii_case(self.short_label())
            || q.eq_ignore_ascii_case(self.label())
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
    pub default_label: String,
    pub alias_key: LatticeAliasKey,
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
    pub chiral: bool,
    pub topologically_chiral: bool,
    pub occupied_wyckoff: Vec<char>,
    pub recipe: Recipe,
    pub frame: Frame2D,
}

impl CatalogueItemUi {
    pub fn has_custom_name(&self) -> bool {
        self.label != self.default_label
    }

    pub fn display_with_orig(&self) -> String {
        if self.has_custom_name() {
            format!("{} [{}]", self.label, self.default_label)
        } else {
            self.label.clone()
        }
    }
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
    pub chiral: bool,
    pub topologically_chiral: bool,
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
    /// C1 (WP-U3): an open patch of the lattice (no gluing).
    Open,
    /// C1 (WP-U3): a torus glued from a hexagonal domain (3 pairs of sides).
    TorusHex,
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

    /// Every target: [`Self::ALL`] (the 8 columns of the verdict matrix) and the two C1 boundary conditions that are
    /// not columns of the Python `surface_table` (session decoding, deck cards).
    pub const EVERY: [Self; 10] = [
        Self::Torus,
        Self::Cylinder,
        Self::MobiusBand,
        Self::KleinBottle,
        Self::Sphere,
        Self::ProjectivePlane,
        Self::Genus2,
        Self::Genus3,
        Self::Open,
        Self::TorusHex,
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
            Self::Open => "open patch",
            Self::TorusHex => "torus (hexagonal domain)",
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
            Self::Open => "open",
            Self::TorusHex => "torus6",
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
    pub scope: Option<String>,
    pub sites: Option<usize>,
    pub cones_summary: String,
    pub reason: String,
}

impl SurfaceCellVerdictUi {
    pub fn scoped_badge(&self) -> String {
        match (self.verdict, self.scope.as_deref()) {
            (VerdictKindUi::Impossible, Some(sc)) => format!("IMPOSSIBLE ({sc})"),
            _ => self.verdict.badge().to_string(),
        }
    }
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
    pub can_end_orientable: bool,
    pub ending_summary: String,
}

#[derive(Clone, Debug, Default)]
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

#[derive(Clone, Debug, Default)]
pub struct DefectReadoutUi {
    pub title: String,
    pub charge_summary: String,
    pub holonomy_detail: String,
    pub core_rings: String,
    pub note: String,
}
