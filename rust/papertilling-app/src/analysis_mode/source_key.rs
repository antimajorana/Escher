//! Exact geometry digest & `SourceKey` for invalidating stale `3. Analysis` spectra.

use papertilling_core::surface::elastic3d::SurfaceElasticResult;
use papertilling_core::types::{BuiltCrystal, Frame2D, Recipe, SurfaceMesh};

use super::{AnalysisModeState, AnalysisTarget};
use crate::backend::CatalogueItemUi;
use crate::combine::CombineTabState;
use crate::defects_tab::DefectsTabState;
use crate::live_ops_ao::Defect2dMode;
use crate::surfaces_elastic::SurfaceShapeMode;
use crate::surfaces_tab::SurfacesTabState;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv_u64(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

pub(super) fn crystal_digest(c: &BuiltCrystal) -> u64 {
    let mut h = fnv_u64(FNV_OFFSET, c.pos.len() as u64);
    for &[x, y] in &c.pos {
        h = fnv_u64(h, x.to_bits());
        h = fnv_u64(h, y.to_bits());
    }
    h = fnv_u64(h, c.edges.len() as u64);
    for &(i, j) in &c.edges {
        h = fnv_u64(h, (i as u64) ^ ((j as u64).rotate_left(32)));
    }
    h = fnv_u64(h, u64::from(c.wall_sigma.is_some()));
    fnv_u64(h, u64::from(c.is_non_orientable))
}

pub(super) fn surface_digest(mesh: Option<&SurfaceMesh>, fvk: Option<&SurfaceElasticResult>) -> u64 {
    let Some(m) = mesh else {
        return 0;
    };
    let xyz = fvk.map_or(m.xyz.as_slice(), |el| el.xyz.as_slice());
    let mut h = fnv_u64(FNV_OFFSET, xyz.len() as u64);
    for &[x, y, z] in xyz {
        h = fnv_u64(h, x.to_bits());
        h = fnv_u64(h, y.to_bits());
        h = fnv_u64(h, z.to_bits());
    }
    h = fnv_u64(h, m.bonds.len() as u64);
    for &(i, j, _) in &m.bonds {
        h = fnv_u64(h, (i as u64) ^ ((j as u64).rotate_left(32)));
    }
    h
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SourceKey {
    Plane {
        item: (String, Recipe, Frame2D),
        radius_q: i64,
        digest: u64,
        kappa_q: i64,
    },
    Surface {
        item: (String, Recipe, Frame2D),
        name: String,
        digest: u64,
        fvk_active: bool,
        cover: bool,
        refused: bool,
    },
    SingleDefect {
        item: (String, Recipe, Frame2D),
        title: String,
        draw_mode: Defect2dMode,
        digest: u64,
        kappa_q: i64,
        flat: bool,
    },
    CombinedDefects {
        item: (String, Recipe, Frame2D),
        digest: u64,
        charge: String,
        kappa_q: i64,
        flat: bool,
    },
}

pub(super) fn build_source_key(
    st: &AnalysisModeState,
    item: &CatalogueItemUi,
    surfaces: &SurfacesTabState,
    defects: &DefectsTabState,
    combine: &CombineTabState,
) -> SourceKey {
    let item_id = (item.group.clone(), item.recipe.clone(), item.frame);
    match st.target {
        AnalysisTarget::PlaneLattice => SourceKey::Plane {
            item: item_id,
            radius_q: (st.plane_radius * 100.0).round() as i64,
            digest: crystal_digest(&st.plane_crystal),
            kappa_q: (st.sheet_opts.kappa_b * 100_000.0).round() as i64,
        },
        AnalysisTarget::Surface => {
            let fvk_active = surfaces.elastic.shape_mode == SurfaceShapeMode::ElasticFvK;
            let fvk_res = fvk_active.then_some(surfaces.elastic.last_result.as_ref()).flatten();
            SourceKey::Surface {
                item: item_id,
                name: surfaces
                    .current_mesh
                    .as_ref()
                    .map_or_else(|| surfaces.target.label().to_owned(), |m| m.name.clone()),
                digest: surface_digest(surfaces.current_mesh.as_ref(), fvk_res),
                fvk_active,
                cover: surfaces.show_cover,
                refused: surfaces.has_refusal(),
            }
        }
        AnalysisTarget::SingleDefect => {
            let c = if defects.sheet_opts.show_flat_baseline {
                &defects.flat_crystal
            } else {
                &defects.last_crystal
            };
            SourceKey::SingleDefect {
                item: item_id,
                title: defects.readout.title.clone(),
                draw_mode: defects.draw_mode,
                digest: crystal_digest(c),
                kappa_q: (defects.sheet_opts.kappa_b * 100_000.0).round() as i64,
                flat: defects.sheet_opts.show_flat_baseline,
            }
        }
        AnalysisTarget::CombinedDefects => {
            let c = if combine.sheet_opts.show_flat_baseline {
                &combine.last_sheet3d.flat_crystal
            } else {
                &combine.last_build.crystal
            };
            SourceKey::CombinedDefects {
                item: item_id,
                digest: crystal_digest(c),
                charge: combine.last_build.charge_label.clone(),
                kappa_q: (combine.sheet_opts.kappa_b * 100_000.0).round() as i64,
                flat: combine.sheet_opts.show_flat_baseline,
            }
        }
    }
}
