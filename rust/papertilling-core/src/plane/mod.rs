//! Euclidean 2D plane module: 17 wallpaper groups, ITA Wyckoff tables, cells,
//! families, catalogues, parameter tuning, supergroup loci, and planar drawing.

pub mod basis_reduce;
pub mod catalogue;
pub mod cell;
pub mod custom_builder;
pub mod draw;
pub mod family;
pub mod groups;
pub mod groups_data;
pub mod inflation;
#[cfg(test)]
mod inflation_cluster_tests;
#[cfg(test)]
mod inflation_dirichlet_tests;
#[cfg(test)]
mod inflation_tests;
pub mod live_search;
pub mod loci;
pub mod planarity;
pub mod relax;
mod relax_normal;
#[cfg(test)]
mod relax_parity_tests;
pub mod relax_preview;
#[cfg(test)]
mod relax_reference;
#[cfg(test)]
mod test_fixtures;
pub mod tuning;
mod tuning_eval;
pub mod tutte;
pub mod voronoi;

#[cfg(test)]
mod tuning_tests;

/// International Tables (ITA) 72-row Wyckoff position descriptors (Stage 1).
pub mod wyckoff {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WyckoffPosition {
        pub group: String,
        pub letter: char,
        pub multiplicity: u8,
        pub site_symmetry: String,
    }
}
