//! Polyhedral sphere and projective-plane constructions and rotation-centre catalogue.

use serde::{Deserialize, Serialize};

use crate::defect::catalogue::{CentreClassEntry, DefectCatalogue};
use crate::types::{Cell2D, Frame2D, SurfaceMesh};

use super::alt_polyhedra::{alternating_matches_centre, build_alternating_rp2, find_split_order4_pair};
use super::obstruction::check_local_cone_obstruction;
use super::pieces::{ConeSurface, LatticeData, glue};
use super::report::symmetry_facts;
pub use super::solids::{SolidNet, solid_net};
use super::solids::{populate_3d, solid_name};
use super::{RefusalKind, SurfaceError};

/// Rotation-centre equivalence class on the reduced primitive cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CentreClass {
    pub id: String,
    pub label: String,
    pub gclass: String,
    pub order: u32,
    pub point: [f64; 2],
    pub location: String,
    pub members: Vec<[f64; 2]>,
}

fn ita_catalogue(group: &str, cell: &Cell2D) -> DefectCatalogue {
    let cat = DefectCatalogue::with_bounds(group, cell, Frame2D::default(), 0.0, 0.0);
    if !group.is_empty() && cat.centre_classes.iter().any(|r| r.label.starts_with("order-")) {
        let max_ord = cat.centre_classes.iter().map(|r| r.order).max().unwrap_or(1);
        for c in cat.centre_classes.iter().filter(|r| r.order == max_ord) {
            let trial = DefectCatalogue::with_bounds(
                group,
                cell,
                Frame2D { q: [[1.0, 0.0], [0.0, 1.0]], origin: c.point },
                0.0,
                0.0,
            );
            if !trial.centre_classes.iter().any(|r| r.label.starts_with("order-")) {
                return trial;
            }
        }
    }
    cat
}

fn to_centre_class(r: &CentreClassEntry) -> CentreClass {
    CentreClass {
        id: r.id.clone(),
        label: r.label.clone(),
        gclass: r.gclass.clone(),
        order: r.order as u32,
        point: r.point,
        location: r.location.clone(),
        members: r.members.clone(),
    }
}

/// Proper rotation-centre classes of `cell`, sorted descending by order and `"face centre"` before `"site"`.
pub fn candidate_centres(group: &str, cell: &Cell2D) -> Vec<CentreClass> {
    if LatticeData::new(group, cell).is_err() {
        return Vec::new();
    }
    let cat = ita_catalogue(group, cell);
    let loc_rank = |loc: &str| match loc {
        "face centre" => 0,
        "site" => 1,
        "bond midpoint" => 2,
        "on a bond" => 3,
        _ => 4,
    };
    let mut indexed: Vec<(usize, CentreClass)> = cat
        .centre_classes
        .iter()
        .enumerate()
        .map(|(i, r)| (i, to_centre_class(r)))
        .collect();
    indexed.sort_by_key(|(idx, r)| (-(r.order as i32), loc_rank(&r.location), *idx));
    indexed.into_iter().map(|(_, r)| r).collect()
}

pub(crate) fn build_polyhedron(
    group: &str,
    cell: &Cell2D,
    size: Option<(i32, i32)>,
    centre_filter: Option<&str>,
    projective: bool,
) -> Result<ConeSurface, SurfaceError> {
    let what = if projective { "projective plane" } else { "sphere" };
    let facts = symmetry_facts(group);
    if let Some(reason) = facts.impossible_reason(what) {
        return Err(SurfaceError::impossible(reason));
    }
    let lat = LatticeData::new(group, cell).map_err(SurfaceError::invalid)?;
    let cat = ita_catalogue(group, cell);
    let n = cat.centre_classes.iter().map(|r| r.order as u32).max().unwrap_or(1);
    if n == 1 {
        return Err(SurfaceError::not_found(format!("no rotation centre found for {group}"), String::new(), false));
    }
    if centre_filter.is_none() {
        check_local_cone_obstruction(&lat, &cat, what)?;
    }
    let solid = solid_name(n)?;
    let sz = size.unwrap_or(if projective || n == 2 { (1, 1) } else { (1, 0) });
    let recs: Vec<&CentreClassEntry> = cat.centre_classes.iter().filter(|r| r.order as u32 == n).collect();
    let can_alt = projective
        && solid == "cube"
        && alternating_matches_centre(find_split_order4_pair(&cat), centre_filter);
    let (cands, skipped): (Vec<&CentreClassEntry>, Vec<&CentreClassEntry>) = if let Some(cf) = centre_filter {
        let hit: Vec<&CentreClassEntry> = recs
            .iter()
            .copied()
            .filter(|r| cf == r.id || cf == r.label || cf == r.location || cf == r.gclass)
            .collect();
        if hit.is_empty() && !can_alt {
            return Err(SurfaceError::not_found(
                format!("no order-{n} rotation centre class {cf:?} in {group}"),
                String::new(),
                true,
            ));
        }
        (hit, Vec::new())
    } else {
        let mut usable: Vec<&CentreClassEntry> = recs
            .iter()
            .copied()
            .filter(|r| r.location == "face centre" || r.location == "site")
            .collect();
        usable.sort_by_key(|r| usize::from(r.location != "face centre"));
        let skip = recs
            .iter()
            .copied()
            .filter(|r| r.location != "face centre" && r.location != "site")
            .collect();
        (usable, skip)
    };
    if cands.is_empty() && !can_alt {
        let mut err = SurfaceError::not_found(
            format!("{what} not found for {group}: every order-{n} centre is at a bond midpoint"),
            String::new(),
            false,
        );
        err.tried.push(format!(
            "centres {} (bond midpoints)",
            skipped.iter().map(|r| r.label.as_str()).collect::<Vec<_>>().join(", ")
        ));
        return Err(err);
    }
    let mut errors: Vec<(String, SurfaceError)> = Vec::new();
    let kind_name = if projective {
        if solid == "pillow" {
            "hemi-pillow (projective plane)".to_string()
        } else {
            format!("hemi-{solid} (projective plane)")
        }
    } else {
        format!("{solid} (sphere)")
    };
    for rec in cands {
        let net_res = solid_net(&lat, solid, sz, rec.point, projective);
        let (pieces, pairs, f3, v3) = match net_res {
            Ok(v) => v,
            Err(e) => {
                if matches!(e.kind, RefusalKind::InvalidArg) {
                    return Err(e);
                }
                errors.push((rec.label.clone(), e));
                continue;
            }
        };
        let lookup = |pt: [f64; 2]| cat.centre_class_of(pt).map(|r| (r.label.clone(), r.location.clone()));
        match glue(&lat, &pieces, &pairs, &kind_name, Some(&lookup), true) {
            Ok(mut surf) => {
                let ch = surf.checks();
                if (projective && surf.mesh.orientable) || !ch.ok || surf.mesh.chi != if projective { 1 } else { 2 } {
                    errors.push((
                        rec.label.clone(),
                        SurfaceError::not_found(format!("{kind_name} checks failed at centre {}", rec.label), String::new(), true),
                    ));
                    continue;
                }
                populate_3d(&mut surf, &lat, &pieces, f3.as_deref(), v3.as_deref(), projective && solid == "pillow", projective)?;
                return Ok(surf);
            }
            Err(e) => errors.push((rec.label.clone(), e)),
        }
    }
    if can_alt && let Some(surf) = build_alternating_rp2(&lat, &cat, sz, centre_filter, &mut errors) {
        return Ok(surf);
    }
    let retry = errors.iter().any(|(_, e)| matches!(e.kind, RefusalKind::NotFound { retry: true }));
    let ladder = if solid == "pillow" { [(1, 1), (1, 2), (2, 1), (2, 2)] } else { [(1, 0), (1, 1), (2, 0), (2, 1)] };
    let other: Vec<_> = ladder.into_iter().filter(|&z| z != sz).take(2).map(|z| format!("size={z:?}")).collect();
    let mut err = SurfaceError::not_found(
        format!("{what} of {group} not found: {solid} at size {sz:?} refused at every order-{n} centre class tried"),
        String::new(),
        retry,
    )
    .with_hint(format!("try {}", other.join(" or ")));
    err.tried = errors.iter().map(|(lab, e)| format!("size {sz:?}, centre {lab}: {}", e.reason)).collect();
    err.centres = errors.iter().map(|(lab, _)| lab.clone()).collect();
    err.causes = errors.into_iter().map(|(_, e)| e.to_string()).collect();
    Err(err)
}

/// `sphere` (`projective == false`) or `projective_plane`, keeping the refusal's kind (IMPOSSIBLE / NOT FOUND /
/// invalid argument) instead of flattening it to a string.
pub fn polyhedron(
    group: &str,
    cell: &Cell2D,
    size: Option<(i32, i32)>,
    centre_filter: Option<&str>,
    projective: bool,
) -> Result<SurfaceMesh, SurfaceError> {
    build_polyhedron(group, cell, size, centre_filter, projective).map(|s| s.mesh)
}

pub fn sphere(group: &str, cell: &Cell2D, size: Option<(i32, i32)>, centre_filter: Option<&str>) -> Result<SurfaceMesh, String> {
    build_polyhedron(group, cell, size, centre_filter, false).map(|s| s.mesh).map_err(|e| e.to_string())
}

pub fn projective_plane(group: &str, cell: &Cell2D, size: Option<(i32, i32)>, centre_filter: Option<&str>) -> Result<SurfaceMesh, String> {
    build_polyhedron(group, cell, size, centre_filter, true).map(|s| s.mesh).map_err(|e| e.to_string())
}
