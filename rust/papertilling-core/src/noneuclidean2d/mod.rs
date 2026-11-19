//! Stage 5 Core: Unified `S^2/E^2/H^2` `{p, q}` realiser, Conway orbifolds, Wythoff &
//! vertex configs, UHP/3D `H^2` surfaces & papercraft nets, `O(N)` `10^5+` kernels, and `TGCell` quotients.

pub mod fast_h2_kernel;
pub mod h2_builders;
pub mod h2_comb_builder;
pub mod h2_spatial_hash;
pub mod models_surfaces;
pub mod orbifold_conway;
#[cfg(test)]
mod tests;
pub mod tgcell_quotients;
pub mod unified_pq;
pub mod unified_s2;
pub mod vertex_config;
pub mod wythoff_archimedean;
pub mod wythoff_h2;
pub mod wythoff_planar;

pub use fast_h2_kernel::{H2LargePatch, build_hyperbolic_large};
pub use h2_spatial_hash::{H2Kernel, H2LargeSummary, RegionFilter};
pub use models_surfaces::{
    H2SurfaceKind, H2SurfaceMesh, PapercraftNet, build_h2_surface_mesh, embed_tiling_on_surface,
    geodesic_arc_poincare, geodesic_arc_uhp, hyp_distance_poincare, hyp_distance_uhp,
    hyperboloid_to_poincare_exact, klein_to_poincare, map_poincare_to_surface,
    poincare_to_hyperboloid_exact, poincare_to_klein, poincare_to_uhp, surface_point,
    uhp_to_poincare, unfold_papercraft_net,
};
pub use orbifold_conway::{
    ConwayOrbifold, WALLPAPER_ORBIFOLD_TABLE, orbifold_to_wallpaper, parse_orbifold,
    wallpaper_orbifold_table, wallpaper_to_orbifold,
};
pub use tgcell_quotients::{
    SidePairing, TGCELL_NAMES, TgCellQuotient, list_tgcell_quotients, tgcell_quotient,
    verify_tgcell_invariants,
};
pub use unified_pq::{UnifiedPqTiling, build_unified_pq, delta_pq, geometry_kind_pq};
pub use vertex_config::{VertexConfigInfo, canonical_vertex_symbol, parse_vertex_config};
pub use wythoff_archimedean::{SnubHandedness, WythoffOp, WythoffTiling, build_wythoff};
