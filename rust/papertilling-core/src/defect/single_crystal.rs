//! Full-patch 2D crystal builder for single-defect equivalence comparison (`Fuse & compare`).
//!
//! Given a multi-defect crystal patch, computes its [`EquivalentSingleReport`] and constructs full-patch
//! (`r_patch`) 2D [`BuiltCrystal`]s for each candidate single-defect core (both Volterra and healed /
//! completion cores for dislocations, single disclination cones, or the defect-free vacuum disk).

use crate::defect::algebraic_fusion::specs_abelian_charge;
use crate::defect::combinations::vacuum_delta_v;
use crate::defect::combine_types::{CoreCandidateInfo, CoreGalleryEntry, EquivalentSingleReport};
use crate::defect::cone_builder::ConeBuilder;
use crate::defect::core_gallery::add_chords;
use crate::defect::core_graph::{LatticeCoreInfo, core_graph};
use crate::defect::equivalence::inspect_crystal_cores;
use crate::defect::single_cache::{SingleDefectCache, equivalent_single_cached};
use crate::defect::surgery::{DefectSpec, build_point_defects_with_info_detailed};
use crate::defect::volterra_patch::{build_volterra_dislocation, drawn_charged_faces};
use crate::plane::draw::{ensure_embedded, relax_picture};
use crate::types::{BuiltCrystal, Cell2D, Frame2D};

/// One full-patch 2D crystal representing an elementary single defect (or vacuum) to compare side-by-side.
#[derive(Clone, Debug)]
pub struct SingleEquivalentCrystal {
    pub name: String,
    pub kind: String,
    pub is_match: bool,
    pub crystal: BuiltCrystal,
    pub core_nodes: usize,
    pub core_edges: usize,
    pub rings: Vec<usize>,
    pub under_count: usize,
    pub added_bonds: usize,
    pub description: String,
}

/// Complete equivalence result + full-patch single-defect crystals for side-by-side rendering.
#[derive(Clone, Debug)]
pub struct SingleEquivalenceBundle {
    pub report: EquivalentSingleReport,
    pub delta_v: Option<i32>,
    pub abelian_summary: Option<String>,
    pub candidates: Vec<SingleEquivalentCrystal>,
    pub default_index: usize,
}

fn relax_if(crystal: &mut BuiltCrystal, relaxed: bool) {
    if relaxed {
        relax_picture(crystal);
        let _ = ensure_embedded(crystal, false);
    }
}

fn measure_core(crystal: &BuiltCrystal, lat_info: &LatticeCoreInfo, radius: usize) -> (usize, usize) {
    core_graph(crystal, lat_info, radius)
        .or_else(|_| core_graph(crystal, lat_info, 1))
        .map_or((0, 0), |g| (g.node_count(), g.edge_count()))
}

fn restore_completion_chords(
    full: &mut BuiltCrystal,
    entry: &CoreGalleryEntry,
    lat_info: &LatticeCoreInfo,
) -> Option<()> {
    if entry.chords.is_empty() {
        return None;
    }
    let (ring, chords) = drawn_charged_faces(full).into_iter().find_map(|(ring, _)| {
        let find_vertex = |index: usize| -> Option<usize> {
            let &[x, y] = entry.crystal.pos.get(index)?;
            ring.iter().copied().find(|&i| (full.pos[i][0] - x).hypot(full.pos[i][1] - y) < 1e-4)
        };
        let chords = entry.chords.iter().map(|&(u, v)| {
            let (u, v) = (find_vertex(u)?, find_vertex(v)?);
            (u != v).then_some((u, v))
        }).collect::<Option<Vec<_>>>()?;
        Some((ring, chords))
    })?;
    add_chords(full, &ring, &chords, lat_info).0.then_some(())
}

#[allow(clippy::too_many_arguments)]
fn build_full_dislocation_candidate(
    lat_info: &LatticeCoreInfo,
    b: [f64; 2],
    cand: &CoreCandidateInfo,
    gal: Option<&CoreGalleryEntry>,
    r_patch: f64,
    radius: usize,
    relaxed: bool,
    is_match: bool,
) -> Option<SingleEquivalentCrystal> {
    let core_pt = cand.core_point?;
    let r_eff = r_patch.max(4.5 * lat_info.scale + b[0].hypot(b[1]));
    let mut full = build_volterra_dislocation(lat_info, b, core_pt, cand.cut, r_eff).ok()?;
    let is_volterra = cand.volterra == Some(true);
    if !is_volterra {
        restore_completion_chords(&mut full, gal?, lat_info)?;
    }

    relax_if(&mut full, relaxed);
    let (core_nodes, core_edges) = measure_core(&full, lat_info, radius);
    let kind = if is_volterra {
        String::from("Volterra")
    } else {
        String::from("completion")
    };
    let description = gal
        .map(|g| g.description.clone())
        .unwrap_or_else(|| format!("rings {:?}", cand.rings));
    Some(SingleEquivalentCrystal {
        name: cand.name.clone(),
        kind,
        is_match,
        crystal: full,
        core_nodes,
        core_edges,
        rings: cand.rings.clone(),
        under_count: cand.under.len(),
        added_bonds: cand.added,
        description,
    })
}

/// Compute single-defect equivalence for `combined` and build full-patch (`r_patch`) 2D crystals for
/// every candidate single-defect core (Volterra & completion), disclination cone, or vacuum disk.
#[allow(clippy::too_many_arguments)]
pub fn build_single_equivalence_bundle(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    combined: &BuiltCrystal,
    r_patch: f64,
    radius: usize,
    grid: usize,
    relaxed: bool,
    cache: &mut SingleDefectCache,
) -> Result<SingleEquivalenceBundle, String> {
    let lat_info = LatticeCoreInfo::new(cell);
    // `draw_gallery = false` keeps unrelaxed coordinates on `report.gallery` so completion chords map 1-to-1 onto full `r_patch`.
    let mut eff_radius = radius;
    let report = match equivalent_single_cached(group, cell, frame, specs, combined, radius, grid, false, cache) {
        Ok(r) => r,
        Err(_) if radius > 1 => {
            eff_radius = 1;
            equivalent_single_cached(group, cell, frame, specs, combined, 1, grid, false, cache)?
        }
        Err(e) => return Err(e),
    };
    let abelian_summary = specs_abelian_charge(group, cell, frame, specs)
        .ok()
        .map(|(p, f)| format!("(G+)^ab = {p}  ·  G^ab = {f}  [PROVED]"));
    let delta_v = (report.kind == "vacuum").then(|| vacuum_delta_v(&lat_info, combined)).flatten();

    let mut candidates: Vec<SingleEquivalentCrystal> = Vec::new();
    match report.kind.as_str() {
        "vacuum" => {
            let mut disk = ConeBuilder::new(&lat_info, [0.0, 0.0], r_patch.max(3.5 * lat_info.scale)).finish()?;
            relax_if(&mut disk, relaxed);
            candidates.push(SingleEquivalentCrystal {
                name: String::from("Perfect lattice (vacuum)"),
                kind: String::from("vacuum"),
                is_match: report.equivalent && delta_v == Some(0),
                crystal: disk,
                core_nodes: 0,
                core_edges: 0,
                rings: Vec::new(),
                under_count: 0,
                added_bonds: 0,
                description: String::from("Defect-free periodic crystal"),
            });
        }
        "disclination" => {
            if let Some(DefectSpec::Disclination { angle_deg, at, .. }) = report.single {
                let single_spec = [DefectSpec::disclination(angle_deg, at)];
                if let Ok((mut cone, _, _, _)) =
                    build_point_defects_with_info_detailed(&lat_info, &single_spec, Some(r_patch), Some(at))
                        .or_else(|_| build_point_defects_with_info_detailed(&lat_info, &single_spec, None, Some(at)))
                {
                    relax_if(&mut cone, relaxed);
                    let (cn, ce) = measure_core(&cone, &lat_info, eff_radius);
                    let mut rings: Vec<usize> = inspect_crystal_cores(&cone).into_iter().filter_map(|c| c.ring).collect();
                    rings.sort_unstable();
                    let cls = report.charge.centre_class.as_deref().unwrap_or("?");
                    candidates.push(SingleEquivalentCrystal {
                        name: format!("Single D({cls}, {angle_deg:+.0}°)"),
                        kind: String::from("disclination"),
                        is_match: report.equivalent,
                        crystal: cone,
                        core_nodes: cn,
                        core_edges: ce,
                        rings,
                        under_count: 0,
                        added_bonds: 0,
                        description: format!("Elementary disclination {angle_deg:+.0}° at Wyckoff {cls}"),
                    });
                }
            }
        }
        "dislocation" => {
            let b = report.charge.b.unwrap_or([0.0, 0.0]);
            for (idx, cand) in report.candidates.iter().enumerate() {
                let is_match = report.matches.iter().any(|m| m == &cand.name);
                let gal = report.gallery.get(idx).filter(|g| g.name == cand.name)
                    .or_else(|| report.gallery.iter().find(|g| g.name == cand.name));
                if let Some(built) = build_full_dislocation_candidate(
                    &lat_info, b, cand, gal, r_patch, eff_radius, relaxed, is_match,
                ) {
                    candidates.push(built);
                }
            }
        }
        _ => {}
    }

    let default_index = candidates.iter().position(|c| c.is_match).unwrap_or(0);
    Ok(SingleEquivalenceBundle {
        report,
        delta_v,
        abelian_summary,
        candidates,
        default_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defect::surgery::build_point_defects;
    use crate::defect::volterra_patch::drawn_faces;
    use crate::types::Bond2D;

    fn square_cell() -> Cell2D {
        Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
            ],
        }
    }

    fn honeycomb_cell() -> Cell2D {
        let s3 = 3.0_f64.sqrt() / 2.0;
        Cell2D {
            a1: [1.0, 0.0],
            a2: [0.5, s3],
            basis: vec![[0.5, s3 / 3.0], [1.0, 2.0 * s3 / 3.0]],
            bonds: vec![
                Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 1, n1: -1, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 1, n1: 0, n2: -1, kind: 0 },
            ],
        }
    }

    #[test]
    fn square_dipole_builds_matching_full_patch_volterra_dislocation() {
        let cell = square_cell();
        let specs = [
            DefectSpec::disclination(90.0, [0.0, 0.0]),
            DefectSpec::disclination(-90.0, [0.5, 0.5]),
        ];
        let combined = build_point_defects(&cell, &specs, None, None).unwrap();
        let mut cache = SingleDefectCache::default();
        let bundle = build_single_equivalence_bundle(
            "p4m", &cell, Frame2D::default(), &specs, &combined, 8.5, 2, 1, true, &mut cache,
        )
        .unwrap();
        assert!(bundle.report.equivalent);
        assert_eq!(bundle.report.match_tier, "volterra");
        assert!(!bundle.candidates.is_empty());
        let best = &bundle.candidates[bundle.default_index];
        assert!(best.is_match);
        assert_eq!(best.kind, "Volterra");
        assert!(best.crystal.pos.len() > 80, "expected full-patch crystal, got {} sites", best.crystal.pos.len());
    }

    #[test]
    fn honeycomb_57_dipole_builds_matching_full_patch_completion_dislocation() {
        let cell = honeycomb_cell();
        // Adjacent hexagon centres (0, 0) +60 deg (pentagon) + (1, 0) -60 deg (heptagon) -> healed 5-7 core!
        let specs = [
            DefectSpec::disclination(60.0, [0.0, 0.0]),
            DefectSpec::disclination(-60.0, [1.0, 0.0]),
        ];
        let combined = build_point_defects(&cell, &specs, None, None).unwrap();
        let mut cache = SingleDefectCache::default();
        let bundle = build_single_equivalence_bundle(
            "p6m", &cell, Frame2D::default(), &specs, &combined, 8.5, 2, 1, true, &mut cache,
        )
        .unwrap();
        assert!(bundle.report.equivalent);
        assert_eq!(bundle.report.match_tier, "completion");
        let best = &bundle.candidates[bundle.default_index];
        assert!(best.is_match);
        assert_eq!(best.kind, "completion");
        assert!(best.rings.contains(&5) && best.rings.contains(&7), "rings={:?}", best.rings);
        for candidate in &bundle.candidates {
            let charge = crate::defect::charge::total_charge(&candidate.crystal).unwrap();
            assert_eq!(charge.kind, "dislocation", "{}", candidate.name);
            assert_eq!(charge.frank_deg, 0.0);
            let [x, y] = charge.burgers.unwrap();
            assert!((x.hypot(y) - 1.0).abs() < 1e-8, "{} has Burgers vector {x}, {y}", candidate.name);
        }
        let rendered_faces: Vec<usize> = drawn_faces(&best.crystal).iter().map(Vec::len).collect();
        assert!(rendered_faces.contains(&5) && rendered_faces.contains(&7));
        assert!(best.crystal.pos.len() > 120, "expected full-patch healed 5-7 crystal, got {}", best.crystal.pos.len());
    }
}
