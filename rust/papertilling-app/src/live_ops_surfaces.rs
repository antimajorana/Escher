use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::WALLPAPER_GROUPS;
use papertilling_core::surface::{
    RefusalKind, SurfaceError, check_surface, cylinder, default_cylinder, default_genus_surface,
    default_klein_bottle, default_mobius_band, default_torus, genus_surface_with_options,
    klein_bottle, mobius_band, polyhedron, surface_table, symmetry_facts_of_cell, torus,
    try_poly_for_table_with_centre,
};
use papertilling_core::types::{Cell2D, SurfaceCheckReport, SurfaceMesh};

use crate::backend::{
    CatalogueItemUi, CentreModeUi, SurfaceBuildParamsUi, SurfaceCellVerdictUi, SurfaceTargetUi,
    SurfaceVerdictRowUi, VerdictKindUi,
};
use crate::live_plane::PlaneCatalogues;

pub fn build_verdict_matrix(cats: &PlaneCatalogues) -> Vec<SurfaceVerdictRowUi> {
    WALLPAPER_GROUPS
        .iter()
        .filter_map(|&grp| {
            let group_cat = cats.points.get(grp)?;
            let entry = group_cat.lattices.first().or_else(|| group_cat.all.first())?;
            let cell = entry.family.cell();
            let lattice_name = cats
                .named
                .named
                .iter()
                .find(|nl| nl.group == grp && nl.network_key == entry.network_key)
                .map_or_else(|| format!("{grp} #1"), |nl| nl.name.clone());
            Some(SurfaceVerdictRowUi {
                group: grp.to_string(),
                lattice_name,
                cells: verdict_cells(grp, &cell),
            })
        })
        .collect()
}

fn verdict_cells(group: &str, cell: &Cell2D) -> [SurfaceCellVerdictUi; 8] {
    let mapped: Vec<SurfaceCellVerdictUi> = surface_table(group, cell)
        .into_iter()
        .map(|r| {
            let verdict = match r.status.as_str() {
                "OK" => VerdictKindUi::Ok,
                "IMPOSSIBLE" => VerdictKindUi::Impossible,
                _ => VerdictKindUi::NotFound,
            };
            let reason = match r.hint.as_deref() {
                Some(h) if !h.is_empty() => format!("{} ({h})", r.reason),
                _ => r.reason,
            };
            SurfaceCellVerdictUi {
                verdict,
                scope: r.scope,
                sites: r.sites,
                cones_summary: r.cones,
                reason,
            }
        })
        .collect();
    mapped.try_into().expect("8 SURFACES rows")
}

/// The 8 `surface_table` verdicts of one lattice (the row of the verdict matrix for the lattice on screen).
pub fn build_verdict_cells(item: &CatalogueItemUi) -> Result<[SurfaceCellVerdictUi; 8], String> {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    Ok(verdict_cells(&item.group, &cell))
}

/// Python's `SURFACES` name of a target (the key of `surface_table` rows and of the IMPOSSIBLE proofs).
pub fn surface_table_name(target: SurfaceTargetUi) -> &'static str {
    match target {
        SurfaceTargetUi::Torus => "torus",
        SurfaceTargetUi::Cylinder => "cylinder",
        SurfaceTargetUi::MobiusBand => "Moebius band",
        SurfaceTargetUi::KleinBottle => "Klein bottle",
        SurfaceTargetUi::Sphere => "sphere",
        SurfaceTargetUi::ProjectivePlane => "projective plane",
        SurfaceTargetUi::Genus2 => "genus 2",
        SurfaceTargetUi::Genus3 => "genus 3",
        // not in Python's SURFACES (C1 boundary conditions of the Surfaces tab; they never ask for this name)
        SurfaceTargetUi::Open => "open patch",
        SurfaceTargetUi::TorusHex => "torus (hexagonal domain)",
    }
}

fn centre_filter(mode: CentreModeUi) -> Option<&'static str> {
    match mode {
        CentreModeUi::Auto => None,
        CentreModeUi::Face => Some("face centre"),
        CentreModeUi::Site => Some("site"),
    }
}

/// A refusal message that starts with its verdict (`IMPOSSIBLE (group/lattice): `, `NOT FOUND: ` or `invalid size: `).
fn verdict_message(msg: String) -> String {
    if msg.starts_with("IMPOSSIBLE") || msg.starts_with("NOT FOUND") || msg.starts_with("invalid size") {
        msg
    } else if msg.contains("must be")
        || msg.contains("must not")
        || msg.starts_with("invalid")
        || msg.contains("size (0, 0)")
    {
        format!("invalid size: {msg}")
    } else {
        format!("NOT FOUND: {msg}")
    }
}

pub(crate) fn surface_error_message(e: SurfaceError) -> String {
    match e.kind {
        RefusalKind::Impossible => match e.scope {
            Some(sc) => format!("IMPOSSIBLE ({sc}): {e}"),
            None => format!("IMPOSSIBLE: {e}"),
        },
        RefusalKind::NotFound { .. } => {
            let s = e.to_string();
            if s.starts_with("NOT FOUND") { s } else { format!("NOT FOUND: {s}") }
        }
        RefusalKind::InvalidArg => format!("invalid size: {e}"),
    }
}

/// The lattice on `target` at exactly the size `(h, k)` of the controls (no silent fallback to another size).
pub fn build_surface_mesh(
    item: &CatalogueItemUi,
    target: SurfaceTargetUi,
    params: &SurfaceBuildParamsUi,
) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
    if matches!(target, SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex) {
        return open_or_hex(item, target);
    }
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    let facts = symmetry_facts_of_cell(&item.group, &cell);
    if let Some(reason) = facts.impossible_reason(surface_table_name(target)) {
        return Err(format!("IMPOSSIBLE (group): {reason}"));
    }
    let (h, k) = (params.h, params.k);
    let mesh = match target {
        SurfaceTargetUi::Torus => torus(&cell, [h.max(1), 0], [0, k.max(1)]),
        SurfaceTargetUi::Cylinder => cylinder(&cell, [0, k.max(1)], [h.max(1), 0]),
        SurfaceTargetUi::MobiusBand => mobius_band(&cell, 0, 2 * k.max(1) + 1, h.max(1)),
        SurfaceTargetUi::KleinBottle => klein_bottle(&cell, 0, 2 * k.max(1) + 1, h.max(1)),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            let proj = target == SurfaceTargetUi::ProjectivePlane;
            return polyhedron(
                &facts.group,
                &cell,
                Some((h.max(0), k.max(0))),
                centre_filter(params.centre_mode),
                proj,
            )
            .map(|m| {
                let rep = check_surface(&m);
                (m, rep)
            })
            .map_err(surface_error_message);
        }
        SurfaceTargetUi::Genus2 | SurfaceTargetUi::Genus3 => {
            let g = if target == SurfaceTargetUi::Genus2 { 2 } else { params.genus_g.max(3) as usize };
            genus_surface_with_options(&cell, g, [h.max(1), 0], [0, k.max(1)], None)
        }
        SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex => unreachable!("routed to open_or_hex above"),
    }
    .map_err(verdict_message)?;
    let report = check_surface(&mesh);
    Ok((mesh, report))
}

/// The lattice on `target` at the sizes `surface_table` uses (its default, then its retries): the surface behind
/// the verdict-matrix cell. `centre_mode` restricts the sphere / projective-plane corners to one kind of centre.
pub fn build_surface_mesh_auto(
    item: &CatalogueItemUi,
    target: SurfaceTargetUi,
    centre_mode: CentreModeUi,
    genus_g: u32,
) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
    if matches!(target, SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex) {
        return open_or_hex(item, target);
    }
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    let facts = symmetry_facts_of_cell(&item.group, &cell);
    if let Some(reason) = facts.impossible_reason(surface_table_name(target)) {
        return Err(format!("IMPOSSIBLE (group): {reason}"));
    }
    let mesh = match target {
        SurfaceTargetUi::Torus => default_torus(&cell),
        SurfaceTargetUi::Cylinder => default_cylinder(&cell),
        SurfaceTargetUi::MobiusBand => default_mobius_band(&cell),
        SurfaceTargetUi::KleinBottle => default_klein_bottle(&cell),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            let proj = target == SurfaceTargetUi::ProjectivePlane;
            let row = try_poly_for_table_with_centre(
                &facts.group,
                &cell,
                proj,
                centre_filter(centre_mode),
            );
            match row.mesh {
                Some(m) => Ok(m),
                None => {
                    let prefix = match (row.status.as_str(), row.scope.as_deref()) {
                        ("IMPOSSIBLE", Some(sc)) => format!("IMPOSSIBLE ({sc})"),
                        _ => row.status,
                    };
                    Err(format!("{prefix}: {}", row.reason))
                }
            }
        }
        SurfaceTargetUi::Genus2 => default_genus_surface(&cell, 2),
        SurfaceTargetUi::Genus3 => default_genus_surface(&cell, genus_g.max(3) as usize),
        SurfaceTargetUi::Open | SurfaceTargetUi::TorusHex => unreachable!("routed to open_or_hex above"),
    }
    .map_err(verdict_message)?;
    let report = check_surface(&mesh);
    Ok((mesh, report))
}

/// The open patch or the hexagonal torus at the default options of the Surfaces tab (the tab itself builds them
/// from its own options, see `surfaces_tab::bc`).
fn open_or_hex(item: &CatalogueItemUi, target: SurfaceTargetUi) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
    crate::surfaces_tab::bc::build_bc(item, target, &Default::default()).map(|(m, r, _)| (m, r))
}
