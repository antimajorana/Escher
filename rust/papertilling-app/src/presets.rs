//! Curated showcase presets: declarative data only (applied by `PapertillingApp::apply_preset`, `app/preset_apply.rs`).
//!
//! Each preset names its tab, its lattice (a named key or a `(group, rule, index)` catalogue entry; `None` keeps the
//! current lattice, for Atlas scenes that do not use it) and every setting its tab owns, so applying it does not
//! depend on what was on screen before. Defect selections are plain catalogue indices (the order of
//! `AppBackend::defect_catalogue`); the tests check each one against the expected class.

use papertilling_core::hyperbolic::HyperbolicStyle;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};

use crate::analysis_mode::{AnalysisTarget, SpectrumColorOverlay};
#[cfg(test)]
use crate::app::NavTab;
use crate::app::AppMode;
use crate::backend::{CatalogueRuleUi, SurfaceTargetUi};
use crate::defects_tab::DefectSubTab;
use crate::live_ops_ao::Defect2dMode;
use crate::mesh3d::{DefectViewMode, Sheet3dColorMode};
use crate::surfaces_elastic::SurfaceShapeMode;

/// Which lattice a preset selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetLattice {
    /// One of the named lattices (`AppBackend::find_named`).
    Named(&'static str),
    /// Entry `index` of the catalogue list of `group` under `rule` (as in the Gallery's group browser).
    Catalogue { group: &'static str, rule: CatalogueRuleUi, index: usize },
}

/// Surface size: the `surface_table` size (auto) or explicit `(h, k)` controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceSize {
    Auto,
    Manual { h: i32, k: i32 },
}

/// The pins a Combine preset places, computed from the lattice's Combine catalogue once it is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombinePins {
    /// The shortest dislocation `b` at its catalogue core and `-b` at the equivalent core `cells` lattice steps
    /// away along the cell vector least parallel to `b` (zero net Burgers vector).
    DislocationDipole { cells: i32 },
    /// `+α` / `-α` disclinations of the highest rotation order at neighbouring centres (the tab's default pair).
    NeighbourDisclinationPair,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasScene {
    SpaceGroup { number: u16, net: CanonicalNet3d, defect: LineDefect3d },
    Hyperbolic { p: usize, q: usize, style: HyperbolicStyle, layers: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PresetSettings {
    Gallery { repeat: i32, symmetry: bool, voronoi: bool, unit_cell: bool },
    Surfaces { target: SurfaceTargetUi, size: SurfaceSize, shape: SurfaceShapeMode },
    Defects {
        sub_tab: DefectSubTab,
        /// Index into the dislocation, disclination or wall list of the lattice's defect catalogue.
        index: usize,
        radius: f64,
        draw: Defect2dMode,
        view: DefectViewMode,
        colour: Sheet3dColorMode,
    },
    Combine { pins: CombinePins, view: DefectViewMode },
    Analysis {
        target: AnalysisTarget,
        defect_sub_tab: DefectSubTab,
        defect_index: usize,
        radius: f64,
        draw: Defect2dMode,
        view: DefectViewMode,
        overlay: SpectrumColorOverlay,
    },
    Atlas(AtlasScene),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    pub lattice: Option<PresetLattice>,
    pub settings: PresetSettings,
}

impl Preset {
    pub fn mode(&self) -> AppMode {
        match self.settings {
            PresetSettings::Gallery { .. } | PresetSettings::Surfaces { .. } => AppMode::Lattices,
            PresetSettings::Defects { .. } | PresetSettings::Combine { .. } => AppMode::Defects,
            PresetSettings::Analysis { .. } => AppMode::Analysis,
            PresetSettings::Atlas(AtlasScene::SpaceGroup { .. }) => AppMode::Defects,
            PresetSettings::Atlas(AtlasScene::Hyperbolic { .. }) => AppMode::Lattices,
        }
    }

    #[cfg(test)]
    pub fn tab(&self) -> NavTab {
        match self.settings {
            PresetSettings::Gallery { .. } => NavTab::Gallery,
            PresetSettings::Surfaces { .. } => NavTab::Surfaces,
            PresetSettings::Defects { .. } => NavTab::Defects,
            PresetSettings::Combine { .. } => NavTab::Combine,
            PresetSettings::Analysis { .. } => NavTab::Analysis,
            PresetSettings::Atlas(_) => NavTab::Atlas,
        }
    }
}

const fn named(key: &'static str) -> Option<PresetLattice> {
    Some(PresetLattice::Named(key))
}

const fn surface(target: SurfaceTargetUi, size: SurfaceSize) -> PresetSettings {
    PresetSettings::Surfaces { target, size, shape: SurfaceShapeMode::AnalyticRadial }
}

const fn defect(sub_tab: DefectSubTab, index: usize, view: DefectViewMode, colour: Sheet3dColorMode) -> PresetSettings {
    PresetSettings::Defects { sub_tab, index, radius: 4.4, draw: Defect2dMode::RelaxedDrawing, view, colour }
}

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Kagome: ITA symmetry + Voronoi",
        description: "Kagome net (p6m) with its 6-, 3- and 2-fold centres, mirrors and Voronoi cells drawn over a 4x4 patch.",
        lattice: named("kagome"),
        settings: PresetSettings::Gallery { repeat: 4, symmetry: true, voronoi: true, unit_cell: true },
    },
    Preset {
        name: "Chiral kagome (p6, no mirrors)",
        description: "Twisted kagome: rotations only (p6), so the symmetry overlay shows no mirror or glide lines.",
        lattice: named("chiral_kagome"),
        settings: PresetSettings::Gallery { repeat: 3, symmetry: true, voronoi: false, unit_cell: true },
    },
    Preset {
        name: "p4g bond net #1 + Voronoi",
        description: "First bond-rule lattice of plane group p4g from the group browser, with Voronoi cells.",
        lattice: Some(PresetLattice::Catalogue { group: "p4g", rule: CatalogueRuleUi::Bonds, index: 0 }),
        settings: PresetSettings::Gallery { repeat: 3, symmetry: false, voronoi: true, unit_cell: true },
    },
    Preset {
        name: "Honeycomb sphere (dodecahedron)",
        description: "Honeycomb wrapped on the sphere at the surface-table size: 12 pentagons, 20 sites (chi = 2).",
        lattice: named("honeycomb"),
        settings: surface(SurfaceTargetUi::Sphere, SurfaceSize::Auto),
    },
    Preset {
        name: "Square-lattice torus 6x6",
        description: "Square lattice on a 6x6 torus: 36 sites, 72 bonds, 36 squares (chi = 0).",
        lattice: named("square"),
        settings: surface(SurfaceTargetUi::Torus, SurfaceSize::Manual { h: 6, k: 6 }),
    },
    Preset {
        name: "Kagome Mobius band",
        description: "Kagome on a Mobius band (a mirror direction makes the twist possible) at the surface-table size.",
        lattice: named("kagome"),
        settings: surface(SurfaceTargetUi::MobiusBand, SurfaceSize::Auto),
    },
    Preset {
        name: "Shastry-Sutherland projective plane",
        description: "Shastry-Sutherland lattice (p4g) folded into RP2: the hemi-icosahedron, 6 sites, 15 bonds, 10 faces.",
        lattice: named("shastry_sutherland"),
        settings: surface(SurfaceTargetUi::ProjectivePlane, SurfaceSize::Auto),
    },
    Preset {
        name: "Edge dislocation, triangular lattice",
        description: "Shortest Burgers vector (|b| = a) dislocation in the triangular lattice, relaxed, 2D + 3D sheet.",
        lattice: named("triangular"),
        settings: defect(DefectSubTab::Dislocations, 0, DefectViewMode::Split2D3D, Sheet3dColorMode::FaceRingSize),
    },
    Preset {
        name: "Honeycomb +60 deg disclination",
        description: "A single +60 deg wedge disclination at a hexagon centre (6-fold) of the honeycomb, 2D + 3D sheet.",
        lattice: named("honeycomb"),
        settings: defect(DefectSubTab::Disclinations, 3, DefectViewMode::Split2D3D, Sheet3dColorMode::FaceRingSize),
    },
    Preset {
        name: "Kagome twin wall Wb1",
        description: "The first twin domain wall (Wb1) of the kagome catalogue across the whole patch.",
        lattice: named("kagome"),
        settings: defect(DefectSubTab::Walls, 0, DefectViewMode::Split2D3D, Sheet3dColorMode::FaceRingSize),
    },
    Preset {
        name: "Analysis: -60 deg honeycomb eigenmodes",
        description: "A -60 deg disclination at a hexagon centre in 3. Analysis mode with multiplet eigenmode coloring.",
        lattice: named("honeycomb"),
        settings: PresetSettings::Analysis {
            target: AnalysisTarget::SingleDefect,
            defect_sub_tab: DefectSubTab::Disclinations,
            defect_index: 2,
            radius: 4.4,
            draw: Defect2dMode::RelaxedDrawing,
            view: DefectViewMode::Sheet3D,
            overlay: SpectrumColorOverlay::LdosMultiplet,
        },
    },
    Preset {
        name: "Combine: square dislocation dipole",
        description: "Two edge dislocations +b and -b three cells apart in the square lattice (net Burgers vector 0).",
        lattice: named("square"),
        settings: PresetSettings::Combine { pins: CombinePins::DislocationDipole { cells: 3 }, view: DefectViewMode::Split2D3D },
    },
    Preset {
        name: "Combine: honeycomb 5-7 pair",
        description: "+60 deg and -60 deg disclinations at neighbouring hexagon centres: a 5-7 pair, i.e. a dislocation.",
        lattice: named("honeycomb"),
        settings: PresetSettings::Combine { pins: CombinePins::NeighbourDisclinationPair, view: DefectViewMode::Split2D3D },
    },
    Preset {
        name: "Atlas: rutile #136 screw dispiration",
        description: "Space group #136 P4_2/mnm, rutile net with its forced 4_2 screw dispiration line.",
        lattice: None,
        settings: PresetSettings::Atlas(AtlasScene::SpaceGroup {
            number: 136,
            net: CanonicalNet3d::Rutile136,
            defect: LineDefect3d::ScrewDispiration42_136,
        }),
    },
    Preset {
        name: "Atlas: hyperbolic {7,3} tiling",
        description: "Heptagons, three at each vertex ((p-2)(q-2) = 5 > 4: hyperbolic), Poincare disk and hyperboloid.",
        lattice: None,
        settings: PresetSettings::Atlas(AtlasScene::Hyperbolic { p: 7, q: 3, style: HyperbolicStyle::RegularPQ, layers: 2 }),
    },
];
