use std::collections::{BTreeMap, HashMap};

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::charge::{frame_faces, total_charge};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::draw::{ensure_embedded, relax_picture};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::WALLPAPER_GROUPS;
use papertilling_core::surface::{
    check_surface, cylinder, default_cylinder, default_genus_surface, default_klein_bottle,
    default_mobius_band, default_torus, genus_surface, genus_surface_with_options, klein_bottle,
    mobius_band, projective_plane, sphere, surface_table, torus, try_rp2_for_table,
    try_sphere_for_table,
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
            let mapped: Vec<SurfaceCellVerdictUi> = surface_table(grp, &cell)
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
            let cells: [SurfaceCellVerdictUi; 8] = mapped.try_into().expect("8 SURFACES rows");
            Some(SurfaceVerdictRowUi { group: grp.to_string(), lattice_name, cells })
        })
        .collect()
}

pub fn build_surface_mesh(
    item: &CatalogueItemUi,
    target: SurfaceTargetUi,
    params: &SurfaceBuildParamsUi,
) -> Result<(SurfaceMesh, SurfaceCheckReport), String> {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    let (h, k) = (params.h.max(1), params.k.max(0));
    let is_def = params.h == 4 && params.k == 4;
    let cf = match params.centre_mode {
        CentreModeUi::Auto => None,
        CentreModeUi::Face => Some("face centre"),
        CentreModeUi::Site => Some("site"),
    };

    let mesh = match target {
        SurfaceTargetUi::Torus => match torus(&cell, [h, 0], [0, k.max(1)]) {
            Ok(m) if check_surface(&m).ok => m,
            _ if is_def => default_torus(&cell)?,
            res => res?,
        },
        SurfaceTargetUi::Cylinder => match cylinder(&cell, [0, k.max(3)], [h, 0]) {
            Ok(m) if check_surface(&m).ok => m,
            _ if is_def => default_cylinder(&cell)?,
            res => res?,
        },
        SurfaceTargetUi::MobiusBand => {
            if is_def {
                default_mobius_band(&cell)?
            } else {
                mobius_band(&cell, 0, 2 * k.max(2) + 1, h).or_else(|_| default_mobius_band(&cell))?
            }
        }
        SurfaceTargetUi::KleinBottle => {
            if is_def {
                default_klein_bottle(&cell)?
            } else {
                klein_bottle(&cell, 0, 2 * k.max(2) + 1, h.max(2))
                    .or_else(|_| default_klein_bottle(&cell))?
            }
        }
        SurfaceTargetUi::Sphere => match sphere(&item.group, &cell, Some((h, k)), cf) {
            Ok(m) => m,
            Err(err) if is_def && cf.is_none() => {
                try_sphere_for_table(&item.group, &cell).mesh.ok_or(err)?
            }
            Err(err) => return Err(err),
        },
        SurfaceTargetUi::ProjectivePlane => {
            match projective_plane(&item.group, &cell, Some((h, k)), cf) {
                Ok(m) => m,
                Err(err) if is_def && cf.is_none() => {
                    try_rp2_for_table(&item.group, &cell).mesh.ok_or(err)?
                }
                Err(err) => return Err(err),
            }
        }
        SurfaceTargetUi::Genus2 | SurfaceTargetUi::Genus3 => {
            let g = if target == SurfaceTargetUi::Genus2 { 2 } else { params.genus_g.max(3) as usize };
            if h > 6 || k > 6 {
                genus_surface_with_options(&cell, g, [h.max(6), 0], [0, k.max(6)], None)
                    .or_else(|_| default_genus_surface(&cell, g))?
            } else {
                genus_surface(&cell, g).or_else(|_| default_genus_surface(&cell, g))?
            }
        }
    };
    let report = check_surface(&mesh);
    Ok((mesh, report))
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
            direction_mn: (1, 0),
            description: format!("{} ({}), strip len={}", w.line, w.line_kind, w.length),
            default_note: String::from("non-orientable Möbius seam (stubs across seam)"),
        });
    }
    for w in &cat.glide_breaking {
        walls.push(WallEntryUi {
            id: w.id.clone(),
            kind: WallKindUi::GlideWd,
            direction_mn: (1, 0),
            description: format!("{} ({}), slip/T={:.2}", w.line, w.line_kind, w.slip_over_period),
            default_note: w.same_as_twin.as_deref().map_or_else(String::new, |tw| format!("same geometry as {tw}")),
        });
    }
    walls
}

pub fn catalogue_for_lattice(item: &CatalogueItemUi) -> DefectCatalogueUi {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).expect("valid recipe").cell();
    let cat = DefectCatalogue::new(&item.group, &cell, item.frame);
    let scale = cell.a1[0].hypot(cell.a1[1]).max(1e-9);
    let walls = collect_walls_ui(&cat);
    let dislocations = cat
        .dislocations
        .into_iter()
        .map(|d| DislocationEntryUi {
            id: d.id,
            burgers_mn: (d.b_frac[0], d.b_frac[1]),
            norm: d.length / scale,
            orbit_size: d.size,
            mirror_partner: d.mirror_id,
        })
        .collect();
    let centres = cat
        .centre_classes
        .into_iter()
        .map(|c| CentreClassUi { id: c.id, wyckoff: c.label, order: c.order as u8, location: c.location })
        .collect();
    let disclinations = cat
        .disclinations
        .into_iter()
        .map(|d| DisclinationEntryUi {
            id: d.id,
            centre_id: d.centre_class,
            frank_angle_deg: d.angle.round() as i32,
            location: d.location,
            wyckoff: d.wyckoff,
        })
        .collect();
    DefectCatalogueUi { dislocations, centres, disclinations, walls }
}

fn summarize_rings_and_core(crystal: &BuiltCrystal, lat_info: &LatticeCoreInfo) -> String {
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
    for r in [2_usize, 1_usize] {
        if let Ok(g) = core_graph(crystal, lat_info, r) {
            return format!("core(r={r}): {} sites, {} bonds | {ring_str}", g.node_count(), g.edge_count());
        }
    }
    ring_str
}

fn build_pristine_disk(info: &LatticeCoreInfo, radius: f64) -> BuiltCrystal {
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

pub fn build_defect_patch(
    item: &CatalogueItemUi,
    selection: &DefectSelectionUi,
    radius: f64,
    relaxed: bool,
) -> (BuiltCrystal, DefectReadoutUi) {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).expect("valid recipe").cell();
    let cat = DefectCatalogue::new(&item.group, &cell, item.frame);
    let lat_info = LatticeCoreInfo::new(&cell);

    match *selection {
        DefectSelectionUi::Dislocation(idx) => {
            let Some(entry) = cat.dislocations.get(idx).or_else(|| cat.dislocations.first()) else {
                return fallback_patch(&lat_info, radius, format!("{} — vacuum", item.label), "No dislocations", String::new());
            };
            let spec = [DefectSpec::Dislocation { b: entry.b, at: Some(entry.at), cut: None }];
            let title = format!("{} — Dislocation {} b=({}, {})", item.label, entry.id, entry.b_frac[0], entry.b_frac[1]);
            build_and_finish_point_defect(&cell, &lat_info, &spec, radius, relaxed, title)
        }
        DefectSelectionUi::Disclination(idx) => {
            let Some(entry) = cat.disclinations.get(idx).or_else(|| cat.disclinations.first()) else {
                let msg = format!("{} has |P+| = 1 (no rotational centres)", item.group);
                return fallback_patch(&lat_info, radius, format!("{} — no disclinations", item.label), &msg, String::new());
            };
            let spec = [DefectSpec::Disclination { angle_deg: entry.angle, at: entry.point, cut: None }];
            let title = format!("{} — Disclination {} ({:+.0}° at {})", item.label, entry.id, entry.angle, entry.wyckoff);
            build_and_finish_point_defect(&cell, &lat_info, &spec, radius, relaxed, title)
        }
        DefectSelectionUi::Wall(idx) => {
            let walls = collect_walls_ui(&cat);
            let Some(w_ui) = walls.get(idx).or_else(|| walls.first()) else {
                return fallback_patch(&lat_info, radius, format!("{} — no walls", item.label), "No Wa-Wd walls", String::new());
            };
            let built = if relaxed {
                cat.wall_picture(&w_ui.id, Some(radius)).map(|(c, inf)| {
                    let n = inf.filter(|i| !i.note.is_empty()).map_or_else(|| w_ui.default_note.clone(), |i| i.note);
                    (c, n)
                })
            } else {
                cat.build_wall(&w_ui.id, Some(radius)).map(|c| (c, w_ui.default_note.clone()))
            };
            let Some((crystal, note)) = built else {
                return fallback_patch(&lat_info, radius, format!("{} — {}", item.label, w_ui.id), "Wall unavailable", String::new());
            };
            let across_cnt = crystal.edges.iter().filter(|&&(u, v)| crystal.adj[u].iter().any(|e| e.to == v && e.across)).count();
            let readout = DefectReadoutUi {
                title: format!("{} — {} ({})", item.label, w_ui.id, w_ui.kind.label()),
                charge_summary: format!("{}: {}", w_ui.kind.label(), w_ui.description),
                holonomy_detail: format!("Patch: {} sites, {} bonds ({across_cnt} across A|B seam)", crystal.pos.len(), crystal.edges.len()),
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
            let s = match ch.burgers {
                Some(b) => format!("Charge: {} | Frank = {:+.1}°, b = ({:+.3}, {:+.3})", ch.kind, ch.frank_deg, b[0], b[1]),
                None => format!("Charge: {} | Frank = {:+.1}°", ch.kind, ch.frank_deg),
            };
            let h = format!("Outer loop S = ({:+.3}, {:+.3}), H.t = ({:+.3}, {:+.3})", ch.loop_sum[0], ch.loop_sum[1], ch.holonomy.t[0], ch.holonomy.t[1]);
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
