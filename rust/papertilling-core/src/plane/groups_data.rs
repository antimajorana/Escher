//! Static International Tables (ITA Vol. A, Part 6) symmetry operations and 72 Wyckoff rows
//! for the 17 Euclidean plane (wallpaper) groups in standard settings.

use serde::{Deserialize, Serialize};

/// Crystal system of a 2D Euclidean wallpaper group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrystalSystem {
    Oblique,
    Rectangular,
    Square,
    Hexagonal,
}

impl CrystalSystem {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Oblique => "oblique",
            Self::Rectangular => "rectangular",
            Self::Square => "square",
            Self::Hexagonal => "hexagonal",
        }
    }
}

/// Raw static ITA Wyckoff row descriptor.
#[derive(Debug, Clone, Copy)]
pub struct RawWyckoffRow {
    pub letter: char,
    pub multiplicity: u8,
    pub site_symmetry: &'static str,
    pub coords: &'static [&'static str],
}

/// Raw static group table entry in ITA standard setting.
#[derive(Debug, Clone, Copy)]
pub struct RawGroupSpec {
    pub symbol: &'static str,
    pub full_symbol: &'static str,
    pub ita_number: u8,
    pub system: CrystalSystem,
    pub centred: bool,
    pub chiral: bool,
    pub has_rotations: bool,
    pub general_ops: &'static [&'static str],
    pub wyckoff: &'static [RawWyckoffRow],
}

const OPS_P1: &[&str] = &["x,y"];
const OPS_P2: &[&str] = &["x,y", "-x,-y"];
const OPS_PM: &[&str] = &["x,y", "-x,y"];
const OPS_PG: &[&str] = &["x,y", "-x,y+1/2"];
const OPS_PMM: &[&str] = &["x,y", "-x,-y", "-x,y", "x,-y"];
const OPS_PMG: &[&str] = &["x,y", "-x,-y", "-x+1/2,y", "x+1/2,-y"];
const OPS_PGG: &[&str] = &["x,y", "-x,-y", "-x+1/2,y+1/2", "x+1/2,-y+1/2"];
const OPS_P4: &[&str] = &["x,y", "-x,-y", "-y,x", "y,-x"];
const OPS_P4M: &[&str] = &["x,y", "-x,-y", "-y,x", "y,-x", "-x,y", "x,-y", "y,x", "-y,-x"];
const OPS_P4G: &[&str] = &[
    "x,y",
    "-x,-y",
    "-y,x",
    "y,-x",
    "-x+1/2,y+1/2",
    "x+1/2,-y+1/2",
    "y+1/2,x+1/2",
    "-y+1/2,-x+1/2",
];
const OPS_P3: &[&str] = &["x,y", "-y,x-y", "-x+y,-x"];
const OPS_P3M1: &[&str] = &["x,y", "-y,x-y", "-x+y,-x", "-y,-x", "-x+y,y", "x,x-y"];
const OPS_P31M: &[&str] = &["x,y", "-y,x-y", "-x+y,-x", "y,x", "x-y,-y", "-x,-x+y"];
const OPS_P6: &[&str] = &["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x"];
const OPS_P6M: &[&str] = &[
    "x,y",
    "-y,x-y",
    "-x+y,-x",
    "-x,-y",
    "y,-x+y",
    "x-y,x",
    "-y,-x",
    "-x+y,y",
    "x,x-y",
    "y,x",
    "x-y,-y",
    "-x,-x+y",
];

const W_P1: &[RawWyckoffRow] = &[RawWyckoffRow {
    letter: 'a',
    multiplicity: 1,
    site_symmetry: "1",
    coords: OPS_P1,
}];

const W_P2: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "2", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "2", coords: &["0,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 1, site_symmetry: "2", coords: &["1/2,0"] },
    RawWyckoffRow { letter: 'd', multiplicity: 1, site_symmetry: "2", coords: &["1/2,1/2"] },
    RawWyckoffRow { letter: 'e', multiplicity: 2, site_symmetry: "1", coords: OPS_P2 },
];

const W_PM: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: ".m.", coords: &["0,y"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: ".m.", coords: &["1/2,y"] },
    RawWyckoffRow { letter: 'c', multiplicity: 2, site_symmetry: "1", coords: OPS_PM },
];

const W_PG: &[RawWyckoffRow] = &[RawWyckoffRow {
    letter: 'a',
    multiplicity: 2,
    site_symmetry: "1",
    coords: OPS_PG,
}];

const W_CM: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 2, site_symmetry: ".m.", coords: &["0,y"] },
    RawWyckoffRow { letter: 'b', multiplicity: 4, site_symmetry: "1", coords: OPS_PM },
];

const W_PMM: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "2mm", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "2mm", coords: &["0,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 1, site_symmetry: "2mm", coords: &["1/2,0"] },
    RawWyckoffRow { letter: 'd', multiplicity: 1, site_symmetry: "2mm", coords: &["1/2,1/2"] },
    RawWyckoffRow { letter: 'e', multiplicity: 2, site_symmetry: "..m", coords: &["x,0", "-x,0"] },
    RawWyckoffRow { letter: 'f', multiplicity: 2, site_symmetry: "..m", coords: &["x,1/2", "-x,1/2"] },
    RawWyckoffRow { letter: 'g', multiplicity: 2, site_symmetry: ".m.", coords: &["0,y", "0,-y"] },
    RawWyckoffRow { letter: 'h', multiplicity: 2, site_symmetry: ".m.", coords: &["1/2,y", "1/2,-y"] },
    RawWyckoffRow { letter: 'i', multiplicity: 4, site_symmetry: "1", coords: OPS_PMM },
];

const W_PMG: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 2, site_symmetry: "2..", coords: &["0,0", "1/2,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "2..", coords: &["0,1/2", "1/2,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 2, site_symmetry: ".m.", coords: &["1/4,y", "3/4,-y"] },
    RawWyckoffRow { letter: 'd', multiplicity: 4, site_symmetry: "1", coords: OPS_PMG },
];

const W_PGG: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 2, site_symmetry: "2..", coords: &["0,0", "1/2,1/2"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "2..", coords: &["0,1/2", "1/2,0"] },
    RawWyckoffRow { letter: 'c', multiplicity: 4, site_symmetry: "1", coords: OPS_PGG },
];

const W_CMM: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 2, site_symmetry: "2mm", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "2mm", coords: &["0,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 4, site_symmetry: "2..", coords: &["1/4,1/4", "3/4,1/4"] },
    RawWyckoffRow { letter: 'd', multiplicity: 4, site_symmetry: "..m", coords: &["x,0", "-x,0"] },
    RawWyckoffRow { letter: 'e', multiplicity: 4, site_symmetry: ".m.", coords: &["0,y", "0,-y"] },
    RawWyckoffRow { letter: 'f', multiplicity: 8, site_symmetry: "1", coords: OPS_PMM },
];

const W_P4: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "4..", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "4..", coords: &["1/2,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 2, site_symmetry: "2..", coords: &["1/2,0", "0,1/2"] },
    RawWyckoffRow { letter: 'd', multiplicity: 4, site_symmetry: "1", coords: OPS_P4 },
];

const W_P4M: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "4mm", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "4mm", coords: &["1/2,1/2"] },
    RawWyckoffRow { letter: 'c', multiplicity: 2, site_symmetry: "2mm.", coords: &["1/2,0", "0,1/2"] },
    RawWyckoffRow {
        letter: 'd',
        multiplicity: 4,
        site_symmetry: "..m",
        coords: &["x,x", "-x,-x", "-x,x", "x,-x"],
    },
    RawWyckoffRow {
        letter: 'e',
        multiplicity: 4,
        site_symmetry: ".m.",
        coords: &["x,0", "-x,0", "0,x", "0,-x"],
    },
    RawWyckoffRow {
        letter: 'f',
        multiplicity: 4,
        site_symmetry: ".m.",
        coords: &["x,1/2", "-x,1/2", "1/2,x", "1/2,-x"],
    },
    RawWyckoffRow { letter: 'g', multiplicity: 8, site_symmetry: "1", coords: OPS_P4M },
];

const W_P4G: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 2, site_symmetry: "4..", coords: &["0,0", "1/2,1/2"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "2.mm", coords: &["1/2,0", "0,1/2"] },
    RawWyckoffRow {
        letter: 'c',
        multiplicity: 4,
        site_symmetry: "..m",
        coords: &["x,x+1/2", "-x,-x+1/2", "-x+1/2,x", "x+1/2,-x"],
    },
    RawWyckoffRow { letter: 'd', multiplicity: 8, site_symmetry: "1", coords: OPS_P4G },
];

const W_P3: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "3..", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "3..", coords: &["1/3,2/3"] },
    RawWyckoffRow { letter: 'c', multiplicity: 1, site_symmetry: "3..", coords: &["2/3,1/3"] },
    RawWyckoffRow { letter: 'd', multiplicity: 3, site_symmetry: "1", coords: OPS_P3 },
];

const W_P3M1: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "3m.", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 1, site_symmetry: "3m.", coords: &["1/3,2/3"] },
    RawWyckoffRow { letter: 'c', multiplicity: 1, site_symmetry: "3m.", coords: &["2/3,1/3"] },
    RawWyckoffRow {
        letter: 'd',
        multiplicity: 3,
        site_symmetry: ".m.",
        coords: &["x,-x", "x,2x", "-2x,-x"],
    },
    RawWyckoffRow { letter: 'e', multiplicity: 6, site_symmetry: "1", coords: OPS_P3M1 },
];

const W_P31M: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "3.m", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "3..", coords: &["1/3,2/3", "2/3,1/3"] },
    RawWyckoffRow {
        letter: 'c',
        multiplicity: 3,
        site_symmetry: "..m",
        coords: &["x,0", "0,x", "-x,-x"],
    },
    RawWyckoffRow { letter: 'd', multiplicity: 6, site_symmetry: "1", coords: OPS_P31M },
];

const W_P6: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "6..", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "3..", coords: &["1/3,2/3", "2/3,1/3"] },
    RawWyckoffRow {
        letter: 'c',
        multiplicity: 3,
        site_symmetry: "2..",
        coords: &["1/2,0", "0,1/2", "1/2,1/2"],
    },
    RawWyckoffRow { letter: 'd', multiplicity: 6, site_symmetry: "1", coords: OPS_P6 },
];

const W_P6M: &[RawWyckoffRow] = &[
    RawWyckoffRow { letter: 'a', multiplicity: 1, site_symmetry: "6mm", coords: &["0,0"] },
    RawWyckoffRow { letter: 'b', multiplicity: 2, site_symmetry: "3m.", coords: &["1/3,2/3", "2/3,1/3"] },
    RawWyckoffRow {
        letter: 'c',
        multiplicity: 3,
        site_symmetry: "2mm",
        coords: &["1/2,0", "0,1/2", "1/2,1/2"],
    },
    RawWyckoffRow {
        letter: 'd',
        multiplicity: 6,
        site_symmetry: "..m",
        coords: &["x,0", "0,x", "-x,-x", "-x,0", "0,-x", "x,x"],
    },
    RawWyckoffRow {
        letter: 'e',
        multiplicity: 6,
        site_symmetry: ".m.",
        coords: &["x,-x", "x,2x", "-2x,-x", "-x,x", "-x,-2x", "2x,x"],
    },
    RawWyckoffRow { letter: 'f', multiplicity: 12, site_symmetry: "1", coords: OPS_P6M },
];

/// Complete static table for the 17 plane groups.
pub const GROUP_SPECS: [RawGroupSpec; 17] = [
    RawGroupSpec { symbol: "p1", full_symbol: "p1", ita_number: 1, system: CrystalSystem::Oblique, centred: false, chiral: true, has_rotations: false, general_ops: OPS_P1, wyckoff: W_P1 },
    RawGroupSpec { symbol: "p2", full_symbol: "p2", ita_number: 2, system: CrystalSystem::Oblique, centred: false, chiral: true, has_rotations: true, general_ops: OPS_P2, wyckoff: W_P2 },
    RawGroupSpec { symbol: "pm", full_symbol: "p1m1", ita_number: 3, system: CrystalSystem::Rectangular, centred: false, chiral: false, has_rotations: false, general_ops: OPS_PM, wyckoff: W_PM },
    RawGroupSpec { symbol: "pg", full_symbol: "p1g1", ita_number: 4, system: CrystalSystem::Rectangular, centred: false, chiral: false, has_rotations: false, general_ops: OPS_PG, wyckoff: W_PG },
    RawGroupSpec { symbol: "cm", full_symbol: "c1m1", ita_number: 5, system: CrystalSystem::Rectangular, centred: true, chiral: false, has_rotations: false, general_ops: OPS_PM, wyckoff: W_CM },
    RawGroupSpec { symbol: "pmm", full_symbol: "p2mm", ita_number: 6, system: CrystalSystem::Rectangular, centred: false, chiral: false, has_rotations: true, general_ops: OPS_PMM, wyckoff: W_PMM },
    RawGroupSpec { symbol: "pmg", full_symbol: "p2mg", ita_number: 7, system: CrystalSystem::Rectangular, centred: false, chiral: false, has_rotations: true, general_ops: OPS_PMG, wyckoff: W_PMG },
    RawGroupSpec { symbol: "pgg", full_symbol: "p2gg", ita_number: 8, system: CrystalSystem::Rectangular, centred: false, chiral: false, has_rotations: true, general_ops: OPS_PGG, wyckoff: W_PGG },
    RawGroupSpec { symbol: "cmm", full_symbol: "c2mm", ita_number: 9, system: CrystalSystem::Rectangular, centred: true, chiral: false, has_rotations: true, general_ops: OPS_PMM, wyckoff: W_CMM },
    RawGroupSpec { symbol: "p4", full_symbol: "p4", ita_number: 10, system: CrystalSystem::Square, centred: false, chiral: true, has_rotations: true, general_ops: OPS_P4, wyckoff: W_P4 },
    RawGroupSpec { symbol: "p4m", full_symbol: "p4mm", ita_number: 11, system: CrystalSystem::Square, centred: false, chiral: false, has_rotations: true, general_ops: OPS_P4M, wyckoff: W_P4M },
    RawGroupSpec { symbol: "p4g", full_symbol: "p4gm", ita_number: 12, system: CrystalSystem::Square, centred: false, chiral: false, has_rotations: true, general_ops: OPS_P4G, wyckoff: W_P4G },
    RawGroupSpec { symbol: "p3", full_symbol: "p3", ita_number: 13, system: CrystalSystem::Hexagonal, centred: false, chiral: true, has_rotations: true, general_ops: OPS_P3, wyckoff: W_P3 },
    RawGroupSpec { symbol: "p3m1", full_symbol: "p3m1", ita_number: 14, system: CrystalSystem::Hexagonal, centred: false, chiral: false, has_rotations: true, general_ops: OPS_P3M1, wyckoff: W_P3M1 },
    RawGroupSpec { symbol: "p31m", full_symbol: "p31m", ita_number: 15, system: CrystalSystem::Hexagonal, centred: false, chiral: false, has_rotations: true, general_ops: OPS_P31M, wyckoff: W_P31M },
    RawGroupSpec { symbol: "p6", full_symbol: "p6", ita_number: 16, system: CrystalSystem::Hexagonal, centred: false, chiral: true, has_rotations: true, general_ops: OPS_P6, wyckoff: W_P6 },
    RawGroupSpec { symbol: "p6m", full_symbol: "p6mm", ita_number: 17, system: CrystalSystem::Hexagonal, centred: false, chiral: false, has_rotations: true, general_ops: OPS_P6M, wyckoff: W_P6M },
];

/// Group symbol aliases matching `wallpaper_lattices._GROUP_ALIASES`.
pub const GROUP_ALIASES: &[(&str, &str)] = &[
    ("p2mm", "pmm"),
    ("p2mg", "pmg"),
    ("p2gg", "pgg"),
    ("c2mm", "cmm"),
    ("c1m1", "cm"),
    ("p1m1", "pm"),
    ("p1g1", "pg"),
    ("p4mm", "p4m"),
    ("p4gm", "p4g"),
    ("p6mm", "p6m"),
];
