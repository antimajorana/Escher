//! Test-only list of every stored lattice: all drawings of the three embedded catalogues
//! (`points.json`, `sheet.json`, `minimal_b.json`) plus the named lattices with their own frames.

use std::sync::OnceLock;

use crate::plane::catalogue::{parse_catalogue_str, parse_named_str};
use crate::plane::family::LatticeFamily;
use crate::types::{Cell2D, Frame2D, Recipe};

const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");
const SHEET_JSON: &str = include_str!("../../../../papertilling/data/sheet.json");
const MINIMAL_B_JSON: &str = include_str!("../../../../papertilling/data/minimal_b.json");
const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

pub(crate) struct StoredLattice {
    pub(crate) label: String,
    pub(crate) group: String,
    pub(crate) recipe: Recipe,
    pub(crate) frame: Frame2D,
    /// `LatticeFamily::from_recipe(recipe, Some(frame)).cell()`.
    pub(crate) cell: Cell2D,
}

fn build() -> Vec<StoredLattice> {
    let t0 = std::time::Instant::now();
    let mut out = Vec::new();
    for (rule, text) in [("points", POINTS_JSON), ("sheet", SHEET_JSON), ("bonds", MINIMAL_B_JSON)] {
        let cats = parse_catalogue_str(text).expect("embedded catalogue");
        for (grp, cat) in &cats {
            for (idx, ent) in cat.drawings.iter().enumerate() {
                let frame = Frame2D::default();
                let fam = LatticeFamily::from_recipe(&ent.recipe, Some(frame)).expect("catalogue recipe");
                out.push(StoredLattice {
                    label: format!("{rule}/{grp}#{idx}"),
                    group: fam.group.clone(),
                    recipe: ent.recipe.clone(),
                    frame,
                    cell: fam.cell(),
                });
            }
        }
    }
    for nl in parse_named_str(NAMED_JSON).expect("embedded named.json").named {
        let fam = LatticeFamily::from_recipe(&nl.recipe, Some(nl.frame)).expect("named recipe");
        out.push(StoredLattice {
            label: format!("named/{}", nl.key),
            group: fam.group.clone(),
            recipe: nl.recipe.clone(),
            frame: nl.frame,
            cell: fam.cell(),
        });
    }
    println!("# time stored lattice load ({} lattices): {:?}", out.len(), t0.elapsed());
    out
}

/// Every stored lattice, parsed once per test binary.
pub(crate) fn stored_lattices() -> &'static [StoredLattice] {
    static CELLS: OnceLock<Vec<StoredLattice>> = OnceLock::new();
    CELLS.get_or_init(build)
}
