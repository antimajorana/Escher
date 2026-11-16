use std::collections::{BTreeMap, HashMap};

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::charge::{frame_faces, total_charge};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph};
use papertilling_core::defect::sheet3d::{Sheet3DOptions, Sheet3DResult, relax_sheet_3d};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::draw::{ensure_embedded, relax_picture};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::WALLPAPER_GROUPS;
use papertilling_core::surface::{
    RefusalKind, SurfaceError, check_surface, cylinder, default_cylinder, default_genus_surface,
    default_klein_bottle, default_mobius_band, default_torus, genus_surface_with_options,
    klein_bottle, mobius_band, polyhedron, surface_table, symmetry_facts, torus,
    try_poly_for_table_with_centre,
};
use papertilling_core::types::{
    BuiltCrystal, Cell2D, DirectedBondData, SurfaceCheckReport, SurfaceMesh,
};

use crate::backend::{
    CatalogueItemUi, CentreClassUi, CentreModeUi, DefectCatalogueUi, DefectReadoutUi,
    DefectSelectionUi, DisclinationEntryUi, DislocationEntryUi, SurfaceBuildParamsUi,
    SurfaceCellVerdictUi, SurfaceTargetUi, SurfaceVerdictRowUi, VerdictKindUi, WallEntryUi,
    WallKindUi,
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
            Some(SurfaceVerdictRowUi { group: grp.to_string(), lattice_name, cells: verdict_cells(grp, &cell) })
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
            SurfaceCellVerdictUi { verdict, sites: r.sites, cones_summary: r.cones, reason }
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
    }
}

fn centre_filter(mode: CentreModeUi) -> Option<&'static str> {
    match mode {
        CentreModeUi::Auto => None,
        CentreModeUi::Face => Some("face centre"),
        CentreModeUi::Site => Some("site"),
    }
}

/// A refusal message that starts with its verdict (`IMPOSSIBLE: `, `NOT FOUND: ` or `invalid size: `).
fn verdict_message(msg: String) -> String {
    if msg.starts_with("IMPOSSIBLE") || msg.starts_with("NOT FOUND") || msg.starts_with("invalid size") {
        msg
    } else if msg.contains("must be") || msg.contains("must not") || msg.starts_with("invalid") || msg.contains("size (0, 0)") {
        format!("invalid size: {msg}")
    } else {
        format!("NOT FOUND: {msg}")
    }
}

fn surface_error_message(e: SurfaceError) -> String {
    match e.kind {
        RefusalKind::Impossible => format!("IMPOSSIBLE: {e}"),
        RefusalKind::NotFound { .. } => format!("NOT FOUND: {e}"),
        RefusalKind::InvalidArg => format!("invalid size: {e}"),
    }
}

/// The lattice on `target` at exactly the size `(h, k)` of the controls (no silent fallback to another size):
/// torus T1 = (h, 0), T2 = (0, k); cylinder wrap (0, k), h rows; Moebius band width h, along 2 k + 1 glide steps;
/// Klein bottle across h, along 2 k + 1; sphere / projective plane size (h, k); genus g over the torus (h, 0) x (0, k).
/// A refusal is returned as a message starting with its verdict (IMPOSSIBLE is proved from the symmetry only).
pub fn build_surface_mesh(
    item: &CatalogueItemUi,
    target: SurfaceTargetUi,
    params: &SurfaceBuildParamsUi,
) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    if let Some(reason) = symmetry_facts(&item.group).impossible_reason(surface_table_name(target)) {
        return Err(format!("IMPOSSIBLE: {reason}"));
    }
    let (h, k) = (params.h, params.k);
    let mesh = match target {
        SurfaceTargetUi::Torus => torus(&cell, [h.max(1), 0], [0, k.max(1)]),
        SurfaceTargetUi::Cylinder => cylinder(&cell, [0, k.max(1)], [h.max(1), 0]),
        SurfaceTargetUi::MobiusBand => mobius_band(&cell, 0, 2 * k.max(1) + 1, h.max(1)),
        SurfaceTargetUi::KleinBottle => klein_bottle(&cell, 0, 2 * k.max(1) + 1, h.max(1)),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            let proj = target == SurfaceTargetUi::ProjectivePlane;
            return polyhedron(&item.group, &cell, Some((h.max(0), k.max(0))), centre_filter(params.centre_mode), proj)
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
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    if let Some(reason) = symmetry_facts(&item.group).impossible_reason(surface_table_name(target)) {
        return Err(format!("IMPOSSIBLE: {reason}"));
    }
    let mesh = match target {
        SurfaceTargetUi::Torus => default_torus(&cell),
        SurfaceTargetUi::Cylinder => default_cylinder(&cell),
        SurfaceTargetUi::MobiusBand => default_mobius_band(&cell),
        SurfaceTargetUi::KleinBottle => default_klein_bottle(&cell),
        SurfaceTargetUi::Sphere | SurfaceTargetUi::ProjectivePlane => {
            let proj = target == SurfaceTargetUi::ProjectivePlane;
            let row = try_poly_for_table_with_centre(&item.group, &cell, proj, centre_filter(centre_mode));
            match row.mesh {
                Some(m) => Ok(m),
                None => Err(format!("{}: {}", row.status, row.reason)),
            }
        }
        SurfaceTargetUi::Genus2 => default_genus_surface(&cell, 2),
        SurfaceTargetUi::Genus3 => default_genus_surface(&cell, genus_g.max(3) as usize),
    }
    .map_err(verdict_message)?;
    let report = check_surface(&mesh);
    Ok((mesh, report))
}

#[path = "live_defects_cache.rs"]
mod defect_cache;
pub(crate) use defect_cache::defect_context;

/// Lattice vector (in the catalogue's primitive cell) along the unit direction `u` of a mirror line of period `per`.
fn line_direction_frac(cat: &DefectCatalogue, u: [f64; 2], line_id: &str) -> (i32, i32) {
    let Some(per) = cat.mirrors.iter().find(|m| m.id == line_id).map(|m| m.period) else {
        return (1, 0);
    };
    let (a1, a2) = (cat.cell.a1, cat.cell.a2);
    let det = a1[0] * a2[1] - a1[1] * a2[0];
    let v = [per * u[0], per * u[1]];
    let f = [(v[0] * a2[1] - v[1] * a2[0]) / det, (a1[0] * v[1] - a1[1] * v[0]) / det];
    if f.iter().all(|x| x.is_finite() && (x - x.round()).abs() < 1e-6) {
        (f[0].round() as i32, f[1].round() as i32)
    } else {
        (1, 0)
    }
}

fn collect_walls_ui(cat: &DefectCatalogue) -> Vec<WallEntryUi> {
    let mut walls = Vec::new();
    for w in &cat.domain_walls {
        walls.push(WallEntryUi {
            id: w.id.clone(),
            kind: WallKindUi::DomainWa,
            direction_mn: (1, 0),
            description: format!("{} ({}; {})", w.coset, w.what, w.centre_map),
            default_note: if w.complete { String::new() } else { String::from("continuous normaliser sample") },
        });
    }
    for w in &cat.twin_walls {
        let pos_tag = if w.through_site { "through site" } else { "between sites" };
        let vis_tag = if w.sigma_in_g {
            "mirror in G"
        } else if w.network_invisible {
            "network-invisible"
        } else {
            "twin"
        };
        walls.push(WallEntryUi {
            id: w.id.clone(),
            kind: WallKindUi::TwinWb,
            direction_mn: (w.direction_frac[0], w.direction_frac[1]),
            description: format!("dir [{},{}] {pos_tag} ({vis_tag})", w.direction_frac[0], w.direction_frac[1]),
            default_note: if w.n_irregular > 0 { format!("{} irregular wall sites/period", w.n_irregular) } else { String::new() },
        });
    }
    for w in &cat.mobius_seams {
        walls.push(WallEntryUi {
            id: w.id.clone(),
            kind: WallKindUi::MobiusWc,
            direction_mn: line_direction_frac(cat, w.direction, &w.line),
            description: format!("{} ({}), strip len={}", w.line, w.line_kind, w.length),
            default_note: String::from("non-orientable Möbius seam (stubs across seam)"),
        });
    }
    for w in &cat.glide_breaking {
        walls.push(WallEntryUi {
            id: w.id.clone(),
            kind: WallKindUi::GlideWd,
            direction_mn: line_direction_frac(cat, w.direction, &w.line),
            description: format!("{} ({}), slip/T={:.2}", w.line, w.line_kind, w.slip_over_period),
            default_note: w.same_as_twin.as_deref().map_or_else(String::new, |tw| format!("same geometry as {tw}")),
        });
    }
    walls
}

pub fn catalogue_for_lattice(item: &CatalogueItemUi) -> DefectCatalogueUi {
    let Ok(ctx) = defect_context(item) else {
        return DefectCatalogueUi { dislocations: Vec::new(), centres: Vec::new(), disclinations: Vec::new(), walls: Vec::new() };
    };
    let (cell, cat) = (&ctx.cell, &ctx.cat);
    let scale = cell.a1[0].hypot(cell.a1[1]).max(1e-9);
    let walls = collect_walls_ui(cat);
    let dislocations = cat
        .dislocations
        .iter()
        .map(|d| DislocationEntryUi {
            id: d.id.clone(),
            burgers_mn: (d.b_frac[0], d.b_frac[1]),
            norm: d.length / scale,
            orbit_size: d.size,
            mirror_partner: d.mirror_id.clone(),
        })
        .collect();
    let centres = cat
        .centre_classes
        .iter()
        .map(|c| CentreClassUi { id: c.id.clone(), wyckoff: c.label.clone(), order: c.order as u8, location: c.location.clone() })
        .collect();
    let disclinations = cat
        .disclinations
        .iter()
        .map(|d| DisclinationEntryUi {
            id: d.id.clone(),
            centre_id: d.centre_class.clone(),
            frank_angle_deg: d.angle.round() as i32,
            location: d.location.clone(),
            wyckoff: d.wyckoff.clone(),
        })
        .collect();
    DefectCatalogueUi { dislocations, centres, disclinations, walls }
}

pub(crate) fn summarize_rings_and_core(crystal: &BuiltCrystal, lat_info: &LatticeCoreInfo) -> String {
    let faces = frame_faces(crystal);
    let ring_str = if faces.len() <= 1 {
        format!("V={}, E={}", crystal.pos.len(), crystal.edges.len())
    } else {
        let k_out = faces.iter().enumerate().max_by_key(|(_, f)| f.len()).map_or(0, |(i, _)| i);
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for (_, f) in faces.iter().enumerate().filter(|&(i, _)| i != k_out) {
            *hist.entry(f.len()).or_default() += 1;
        }
        hist.into_iter().map(|(sz, cnt)| format!("{sz}-gon×{cnt}")).collect::<Vec<_>>().join(", ")
    };
    let mut why = String::new();
    for r in [2_usize, 1_usize] {
        match core_graph(crystal, lat_info, r) {
            Ok(g) => return format!("core(r={r}): {} sites, {} bonds | {ring_str}", g.node_count(), g.edge_count()),
            Err(err) => why = err,
        }
    }
    format!("core: none ({why}) | {ring_str}")
}

pub(crate) fn build_pristine_disk(info: &LatticeCoreInfo, radius: f64) -> BuiltCrystal {
    let keys = info.lat.sites_in_disk([0.0, 0.0], radius.max(2.5));
    let n = keys.len();
    let idx_map: HashMap<_, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| info.lat.pos(k)).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|&k| k.2).collect();
    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        for (k2, kind) in info.lat.nbrs(k) {
            if let Some(&j) = idx_map.get(&k2) {
                if i < j {
                    edges.push((i, j));
                }
                adj[i].push(DirectedBondData {
                    to: j,
                    vec: [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]],
                    kind,
                    seam: None,
                    ang: None,
                    across: false,
                    ghost_stub: None,
                });
            }
        }
    }
    edges.sort_unstable();
    BuiltCrystal {
        ref_pos: pos.clone(),
        pos,
        basis_idx,
        side: vec![None; n],
        cone_sites: Vec::new(),
        edges,
        adj,
        wall_sigma: None,
        is_non_orientable: false,
    }
}

fn fallback_patch(info: &LatticeCoreInfo, radius: f64, title: String, summary: &str, note: String) -> (BuiltCrystal, DefectReadoutUi) {
    (
        build_pristine_disk(info, radius),
        DefectReadoutUi {
            title,
            charge_summary: summary.to_string(),
            holonomy_detail: String::from("Reference disk"),
            core_rings: String::from("defect-free reference disk"),
            note,
        },
    )
}

/// A recipe that makes no lattice: one site and the reason, instead of a panic.
fn no_lattice_patch(item: &CatalogueItemUi, err: String) -> (BuiltCrystal, DefectReadoutUi) {
    let crystal = BuiltCrystal {
        pos: vec![[0.0, 0.0]],
        ref_pos: vec![[0.0, 0.0]],
        basis_idx: vec![0],
        side: vec![None],
        cone_sites: Vec::new(),
        edges: Vec::new(),
        adj: vec![Vec::new()],
        wall_sigma: None,
        is_non_orientable: false,
    };
    let readout = DefectReadoutUi {
        title: format!("{} — no lattice", item.label),
        charge_summary: String::from("No lattice from this recipe"),
        holonomy_detail: String::new(),
        core_rings: String::from("-"),
        note: err,
    };
    (crystal, readout)
}

pub fn build_defect_patch(
    item: &CatalogueItemUi,
    selection: &DefectSelectionUi,
    radius: f64,
    relaxed: bool,
) -> (BuiltCrystal, DefectReadoutUi) {
    let ctx = match defect_context(item) {
        Ok(ctx) => ctx,
        Err(err) => return no_lattice_patch(item, err),
    };
    let (cell, cat, lat_info) = (&ctx.cell, &ctx.cat, &ctx.lat_info);

    match *selection {
        DefectSelectionUi::Dislocation(idx) => {
            let Some(entry) = cat.dislocations.get(idx).or_else(|| cat.dislocations.first()) else {
                return fallback_patch(lat_info, radius, format!("{} — vacuum", item.label), "No dislocations", String::new());
            };
            let spec = [DefectSpec::Dislocation { b: entry.b, at: Some(entry.at), cut: None }];
            let title = format!("{} — Dislocation {} b=({}, {})", item.label, entry.id, entry.b_frac[0], entry.b_frac[1]);
            build_and_finish_point_defect(cell, lat_info, &spec, radius, relaxed, title)
        }
        DefectSelectionUi::Disclination(idx) => {
            let Some(entry) = cat.disclinations.get(idx).or_else(|| cat.disclinations.first()) else {
                let msg = format!("{} has |P+| = 1 (no rotational centres)", item.group);
                return fallback_patch(lat_info, radius, format!("{} — no disclinations", item.label), &msg, String::new());
            };
            let spec = [DefectSpec::Disclination { angle_deg: entry.angle, at: entry.point, cut: None }];
            let title = format!("{} — Disclination {} ({:+.0}° at {})", item.label, entry.id, entry.angle, entry.wyckoff);
            build_and_finish_point_defect(cell, lat_info, &spec, radius, relaxed, title)
        }
        DefectSelectionUi::Wall(idx) => {
            let walls = collect_walls_ui(cat);
            let Some(w_ui) = walls.get(idx).or_else(|| walls.first()) else {
                return fallback_patch(lat_info, radius, format!("{} — no walls", item.label), "No Wa-Wd walls", String::new());
            };
            let built = if relaxed {
                cat.wall_picture(&w_ui.id, Some(radius)).map(|(c, inf)| {
                    let draw_note = inf.map(|i| i.note).unwrap_or_default();
                    let n = [w_ui.default_note.as_str(), draw_note.as_str()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("; ");
                    (c, n)
                })
            } else {
                cat.build_wall(&w_ui.id, Some(radius)).map(|c| (c, w_ui.default_note.clone()))
            };
            let Some((crystal, note)) = built else {
                return fallback_patch(lat_info, radius, format!("{} — {}", item.label, w_ui.id), "Wall unavailable", String::new());
            };
            let across_cnt = crystal.edges.iter().filter(|&&(u, v)| crystal.adj[u].iter().any(|e| e.to == v && e.across)).count();
            let across_what = if crystal.is_non_orientable { "across the Möbius seam" } else { "across the A|B wall" };
            let readout = DefectReadoutUi {
                title: format!("{} — {} ({})", item.label, w_ui.id, w_ui.kind.label()),
                charge_summary: format!("{}: {}", w_ui.kind.label(), w_ui.description),
                holonomy_detail: format!("Patch: {} sites, {} bonds ({across_cnt} {across_what})", crystal.pos.len(), crystal.edges.len()),
                core_rings: format!("orientable={} | cones={}", !crystal.is_non_orientable, crystal.cone_sites.len()),
                note,
            };
            (crystal, readout)
        }
    }
}

fn build_and_finish_point_defect(
    cell: &Cell2D,
    lat_info: &LatticeCoreInfo,
    specs: &[DefectSpec],
    radius: f64,
    relaxed: bool,
    title: String,
) -> (BuiltCrystal, DefectReadoutUi) {
    let mut notes = Vec::new();
    let mut crystal = match build_point_defects(cell, specs, Some(radius), None) {
        Ok(c) => c,
        Err(_) => match build_point_defects(cell, specs, None, None) {
            Ok(c) => {
                notes.push(String::from("expanded to minimum safe surgical radius"));
                c
            }
            Err(err) => return fallback_patch(lat_info, radius, title, "Surgery refused", err),
        },
    };
    if relaxed {
        relax_picture(&mut crystal);
        let info = ensure_embedded(&mut crystal, false);
        if !info.note.is_empty() {
            notes.push(info.note);
        }
    }
    let (charge_summary, holonomy_detail) = match total_charge(&crystal) {
        Ok(ch) => {
            let z = |x: f64| if x.abs() < 5e-4 { 0.0 } else { x }; // no "-0.0" / "-0.000" in the readout
            let s = match ch.burgers {
                Some(b) => format!("Charge: {} | Frank = {:+.1}°, b = ({:+.3}, {:+.3})", ch.kind, z(ch.frank_deg), z(b[0]), z(b[1])),
                None => format!("Charge: {} | Frank = {:+.1}°", ch.kind, z(ch.frank_deg)),
            };
            let h = format!("Outer loop S = ({:+.3}, {:+.3}), H.t = ({:+.3}, {:+.3})", z(ch.loop_sum[0]), z(ch.loop_sum[1]), z(ch.holonomy.t[0]), z(ch.holonomy.t[1]));
            (s, h)
        }
        Err(err) => (String::from("Boundary charge unavailable"), format!("total_charge: {err}")),
    };
    let readout = DefectReadoutUi {
        title,
        charge_summary,
        holonomy_detail,
        core_rings: summarize_rings_and_core(&crystal, lat_info),
        note: notes.join("; "),
    };
    (crystal, readout)
}

/// Relax `crystal` into 3D together with a defect-free flat baseline disk of the same lattice and radius.
pub fn build_defect_sheets_3d(
    item: &CatalogueItemUi,
    crystal: &BuiltCrystal,
    radius: f64,
    kappa_b: f64,
) -> (Sheet3DResult, BuiltCrystal, Sheet3DResult) {
    let opts = Sheet3DOptions { kappa_b, max_iters: 240, grad_tol: 1e-6 };
    let Ok(ctx) = defect_context(item) else {
        let empty = BuiltCrystal {
            pos: Vec::new(),
            ref_pos: Vec::new(),
            basis_idx: Vec::new(),
            side: Vec::new(),
            cone_sites: Vec::new(),
            edges: Vec::new(),
            adj: Vec::new(),
            wall_sigma: None,
            is_non_orientable: false,
        };
        let empty_s = relax_sheet_3d(&empty, &Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: Vec::new(), bonds: Vec::new() }, &opts, None);
        return (empty_s.clone(), empty, empty_s);
    };
    let sheet = relax_sheet_3d(crystal, &ctx.cell, &opts, None);
    let flat_crystal = build_pristine_disk(&ctx.lat_info, radius);
    let flat_sheet = relax_sheet_3d(&flat_crystal, &ctx.cell, &opts, None);
    (sheet, flat_crystal, flat_sheet)
}

#[cfg(test)]
mod tests {
    use crate::backend::{
        AppBackend, DefectSelectionUi, SurfaceBuildParamsUi, SurfaceTargetUi, VerdictKindUi,
    };

    #[test]
    fn live_verdict_matrix_surfaces_and_defects_on_kagome() {
        let backend = AppBackend::new();
        let mat = backend.surface_verdict_matrix();
        assert_eq!(mat.len(), 17);
        let p1_row = mat.iter().find(|r| r.group == "p1").unwrap();
        assert_eq!(p1_row.cells[0].verdict, VerdictKindUi::Ok); // torus
        assert_eq!(p1_row.cells[4].verdict, VerdictKindUi::Impossible); // sphere
        let p6m_row = mat.iter().find(|r| r.group == "p6m").unwrap();
        assert_eq!(p6m_row.cells[4].verdict, VerdictKindUi::Ok); // sphere
        assert_eq!(p6m_row.cells[5].verdict, VerdictKindUi::Ok); // projective plane

        let kagome = backend.find_named("kagome").unwrap();
        let params = SurfaceBuildParamsUi::default();
        let (sph, rep_s) = backend.build_surface(kagome, SurfaceTargetUi::Sphere, &params).unwrap();
        assert!(rep_s.ok && sph.chi == 2 && sph.cones.len() == 12);

        let dcat = backend.defect_catalogue(kagome);
        assert!(!dcat.dislocations.is_empty() && !dcat.disclinations.is_empty() && !dcat.walls.is_empty());
        let (c_dis, r_dis) = backend.build_defect(kagome, &DefectSelectionUi::Dislocation(0), 5.5, true);
        assert!(!c_dis.pos.is_empty() && r_dis.charge_summary.contains("dislocation"));
        let (c_dsc, r_dsc) = backend.build_defect(kagome, &DefectSelectionUi::Disclination(0), 5.5, true);
        assert!(!c_dsc.pos.is_empty() && r_dsc.charge_summary.contains("disclination"));
    }
}
