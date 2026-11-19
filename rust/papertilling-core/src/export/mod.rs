//! Solver-facing exports: extended XYZ files, neighbour shells split into space-group bond orbits, and
//! site-permutation groups of periodic tori and finite clusters (Section D of the Stage 2 plan).

pub mod permutations;
pub mod solver;
#[cfg(test)]
mod solver_tests;

pub use permutations::{
    CLUSTER_TOL, PermutationGroupReport, SupercellSolverExport, cluster_site_permutations, group_closed,
    periodic_supercell_permutations, torus_index,
};
pub use solver::{
    PeriodicBond, ShellClassification, ShellInfo, ShellOrbit, cell_coordination, classify_cell_shells,
    export_cell_extxyz, export_crystal_extxyz, export_surface_extxyz, fmt_f8,
};
