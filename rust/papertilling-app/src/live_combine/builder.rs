use std::time::Instant;

use papertilling_core::defect::algebraic_fusion::specs_abelian_charge;
use papertilling_core::defect::charge::{total_charge, total_frank};
use papertilling_core::defect::combinations::algebraic_specs_charge_label;
use papertilling_core::defect::combine_api::{classify_crystal_charge, inspect_crystal_cores};
use papertilling_core::defect::combine_types::NotebookConfigJson;
use papertilling_core::defect::core_graph::core_graph;
use papertilling_core::defect::surgery::{
    DefectSpec, build_point_defects_detailed, patch_centre_and_radius,
};
use papertilling_core::dsymbol;
use papertilling_core::plane::basis_reduce::is_geometrically_chiral;
use papertilling_core::plane::draw::{ensure_embedded, relax_picture, relax_preview};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::planarity;
use papertilling_core::types::BuiltCrystal;

use super::cuts;
use super::snap::spec_world_pos;
use super::{CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo};
use crate::aliases::LatticeAliasKey;
use crate::backend::CatalogueItemUi;
use crate::combine_worker::empty_crystal;
use crate::live_ops::{PRISTINE_DISK_MIN_RADIUS, build_pristine_disk, summarize_rings};
use crate::live_plane::{coordination_string, format_dsymbol_key, occupied_wyckoff_letters};

fn append_abelian_charge(detail: &mut String, ab_line: Option<&str>) {
    if let Some(line) = ab_line {
        if !detail.is_empty() {
            detail.push('\n');
        }
        detail.push_str(line);
    }
}

const LIVE_VIEW_LIMIT: &str = "Desktop live-view budget exceeded (~20,000 sites). Move pins closer or choose a smaller patch.";

fn fits_live_view_budget(info: &CombineLatticeInfo, radius: f64, inserted_turns: f64) -> bool {
    let area = (info.cell.a1[0] * info.cell.a2[1] - info.cell.a1[1] * info.cell.a2[0]).abs();
    let estimated_sites = std::f64::consts::PI * radius.powi(2) * info.cell.basis.len() as f64
        * (1.0 + inserted_turns) / area;
    estimated_sites.is_finite() && estimated_sites <= 20_000.0
}

/// Build, draw and read out the combined crystal. `preview` (live pin drag): a cheaper picture,
/// no planarity check and no core readouts; the final build on drag release does the full work.
pub fn build_combined_crystal(
    item: &CatalogueItemUi,
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    opts: &CombineBuildOptions,
    preview: bool,
) -> CombineBuildOutput {
    let t0 = Instant::now();
    let lattice_key = (item.group.clone(), info.cell.clone());
    let base_r = if opts.auto_radius { info.r_auto_base } else { opts.manual_radius };
    let reference_r = base_r.max(PRISTINE_DISK_MIN_RADIUS);

    let refused = |reason: String, charge_label: &str, charge_detail, patch_centre| {
        let standin_r = reference_r.min(info.r_auto_base.max(PRISTINE_DISK_MIN_RADIUS));
        let (crystal, applied_radius) = if fits_live_view_budget(info, standin_r, 0.0) {
            (build_pristine_disk(&info.lat_info, standin_r), standin_r)
        } else {
            (empty_crystal(), 0.0)
        };
        CombineBuildOutput {
            lattice_key: lattice_key.clone(),
            crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius,
            charge_label: charge_label.to_owned(), charge_detail,
            core_summary: String::from("Core: not computed"),
            local_cores: Vec::new(), patch_centre, applied_cuts: Vec::new(), coalesced_count: 0,
            note: cuts::format_refusal_note(&reason), preview, refused: Some(reason),
        }
    };

    if specs.is_empty() {
        if !fits_live_view_budget(info, reference_r, 0.0) {
            return refused(LIVE_VIEW_LIMIT.into(), "Live view too large", LIVE_VIEW_LIMIT.into(), [0.0, 0.0]);
        }
        let crystal = build_pristine_disk(&info.lat_info, reference_r);
        return CombineBuildOutput {
            lattice_key, crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: reference_r,
            charge_label: String::from("vacuum (0 defects)"),
            charge_detail: String::from("Frank = +0.0°, b = (0.000, 0.000)"),
            core_summary: String::from("Core: defect-free reference disk"),
            local_cores: Vec::new(), patch_centre: [0.0, 0.0], applied_cuts: Vec::new(), coalesced_count: 0,
            note: String::new(), preview, refused: None,
        };
    }

    let abelian_line = specs_abelian_charge(&item.group, &info.cell, item.frame, specs)
        .ok()
        .map(|(p, f)| format!("Abelian charge: (G+)^ab = {p}, G^ab = {f} [PROVED]"));

    let pts: Vec<[f64; 2]> = specs.iter().enumerate().map(|(i, s)| spec_world_pos(info, i, s)).collect();
    let n_pts = pts.len() as f64;
    let ctr = [pts.iter().map(|p| p[0]).sum::<f64>() / n_pts, pts.iter().map(|p| p[1]).sum::<f64>() / n_pts];
    let max_d = pts.iter().map(|p| (p[0] - ctr[0]).hypot(p[1] - ctr[1])).fold(0.0_f64, f64::max);
    let max_b = specs
        .iter()
        .filter_map(|s| match *s {
            DefectSpec::Dislocation { b, .. } => Some(b[0].hypot(b[1])),
            DefectSpec::Disclination { .. } | DefectSpec::HalfWall { .. } => None,
        })
        .fold(0.0_f64, f64::max);

    let r_target = if opts.auto_radius {
        let core_margin = ((2 * info.lat_info.r_det + opts.core_radius + 4) as f64 * info.lat_info.lmax + max_b)
            .max(5.0 * info.lat_info.scale);
        base_r.max(max_d + core_margin)
    } else {
        opts.manual_radius
    };

    let inserted_turns: f64 = specs.iter().filter_map(|s| match s {
        DefectSpec::Disclination { angle_deg, .. } if *angle_deg < 0.0 => Some(-angle_deg / 360.0),
        _ => None,
    }).sum();
    let budget_refusal = || {
        let mut charge_detail = String::from(LIVE_VIEW_LIMIT);
        append_abelian_charge(&mut charge_detail, abelian_line.as_deref());
        refused(LIVE_VIEW_LIMIT.into(), "Live view too large", charge_detail, ctr)
    };
    if !fits_live_view_budget(info, r_target.max(PRISTINE_DISK_MIN_RADIUS), inserted_turns) {
        return budget_refusal();
    }

    let mut notes = Vec::new();
    let mut applied_r = r_target;
    let mut result = build_point_defects_detailed(&info.cell, specs, Some(r_target), None);
    if result.is_err()
        && let Ok((_, r_safe)) = patch_centre_and_radius(&info.lat_info, specs, None, None)
        && (r_safe - r_target).abs() > 1e-9
    {
        if !fits_live_view_budget(info, r_safe.max(PRISTINE_DISK_MIN_RADIUS), inserted_turns) {
            return budget_refusal();
        }
        result = build_point_defects_detailed(&info.cell, specs, Some(r_safe), None);
        applied_r = r_safe;
        notes.push(format!("retried at surgical radius R={r_safe:.1}"));
    }
    let (mut crystal, applied_cuts, coalesced_count, build_meta) = match result {
        Ok(res) => res,
        Err(err) => {
            let mut charge_detail = match algebraic_specs_charge_label(&item.group, &info.cell, item.frame, specs) {
                Some(lbl) => format!("Algebraic total charge: {lbl}\n{err}"),
                None => err.clone(),
            };
            append_abelian_charge(&mut charge_detail, abelian_line.as_deref());
            return refused(err, "Surgery refused", charge_detail, ctr);
        }
    };
    match (build_meta.used_fallback_order, build_meta.used_backtracking) {
        (true, true) => notes.push(String::from("(built via fallback order + cut search)")),
        (true, false) => notes.push(String::from("(built via fallback order)")),
        (false, true) => notes.push(String::from("(built via backtracked cut search)")),
        (false, false) => {}
    }
    if coalesced_count > 0 {
        notes.push(if applied_cuts.is_empty() {
            format!("{coalesced_count} coincident pin(s) cancelled to vacuum")
        } else {
            format!("superposed {coalesced_count} coincident same-apex pin(s)")
        });
    }

    if opts.draw_mode == CombineDrawMode::FastRelaxed && preview {
        relax_preview_picture(&mut crystal);
        align_crystal_orientation(&mut crystal);
    } else if opts.draw_mode == CombineDrawMode::FastRelaxed {
        relax_picture(&mut crystal);
        if opts.full_planar_repair {
            let d_info = ensure_embedded(&mut crystal, false);
            if !d_info.note.is_empty() {
                notes.push(d_info.note);
            }
        } else {
            let chk = planarity::check(&crystal);
            let bad = chk.crossings + chk.folded.unwrap_or(0);
            if bad > 0 {
                notes.push(format!("fast relax has {bad} crossing/fold(s) — toggle Full Planar Repair"));
            }
        }
        align_crystal_orientation(&mut crystal);
    }
    // C1: the AO elastic branch (`CombineDrawMode::ElasticAoMetric`, `live_ops_ao::apply_combine_ao_mode`) left
    // the UI; nothing selects that mode any more, so it draws as Raw.

    let (charge_label, mut charge_detail) = match classify_crystal_charge(&item.group, &info.cell, item.frame, &crystal) {
        Ok(ch) => {
            let mut det = format!("kind={} | Frank={:+.1}° | loop S=({:+.3}, {:+.3})", ch.kind, ch.angle, ch.loop_sum[0], ch.loop_sum[1]);
            if !preview
                && let Some(ordered_charge) =
                    algebraic_specs_charge_label(&item.group, &info.cell, item.frame, specs)
                && ordered_charge != ch.label
            {
                use std::fmt::Write as _;
                let _ = write!(
                    det,
                    "\nPin-order algebraic charge: {ordered_charge}\nMeasured boundary charge: {}\nEquivalence uses the measured charge and core graph.",
                    ch.label
                );
            }
            (ch.label, det)
        }
        Err(_) => match total_charge(&crystal) {
            Ok(tc) => {
                let lbl = match tc.burgers {
                    Some(b) => format!("{} (Frank {:+.1}°, b = [{:+.3}, {:+.3}])", tc.kind, tc.frank_deg, b[0], b[1]),
                    None => format!("{} (Frank {:+.1}°)", tc.kind, tc.frank_deg),
                };
                (lbl, format!("loop S = ({:+.3}, {:+.3})", tc.loop_sum[0], tc.loop_sum[1]))
            }
            Err(e) => (String::from("Charge unavailable"), e),
        },
    };
    append_abelian_charge(&mut charge_detail, abelian_line.as_deref());

    if preview {
        return CombineBuildOutput {
            lattice_key, crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: applied_r,
            charge_label, charge_detail,
            core_summary: String::from("(core readout on release)"), local_cores: Vec::new(),
            patch_centre: ctr, applied_cuts, coalesced_count, note: notes.join("; "),
            preview, refused: None,
        };
    }
    let radius = opts.core_radius;
    let rings = summarize_rings(&crystal);
    let core_summary = match core_graph(&crystal, &info.lat_info, radius) {
        Ok(graph) => format!(
            "Core (r={radius}): {} sites, {} bonds | {rings}",
            graph.node_count(),
            graph.edge_count()
        ),
        Err(err) => format!("Core (r={radius}) unavailable: {err} | {rings}"),
    };

    CombineBuildOutput {
        lattice_key,
        core_summary,
        local_cores: inspect_crystal_cores(&crystal),
        crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: applied_r,
        charge_label, charge_detail,
        patch_centre: ctr, applied_cuts, coalesced_count, note: notes.join("; "),
        preview, refused: None,
    }
}

/// Rigidly rotate (and, for zero total Frank angle, translate) a relaxed 2D crystal so its bulk
/// bond orientation matches the reference lattice without distorting any bond length or angle.
pub(crate) fn align_crystal_orientation(crystal: &mut BuiltCrystal) {
    let n = crystal.pos.len();
    if n == 0 || crystal.adj.len() != n {
        return;
    }
    let (mut c_sum, mut s_sum) = (0.0, 0.0);
    for (u, adj_u) in crystal.adj.iter().enumerate() {
        let pu = crystal.pos[u];
        for e in adj_u {
            if u >= e.to || e.across {
                continue;
            }
            let pure_trans = e.seam.is_none_or(|g| {
                (g.det() - 1.0).abs() < 1e-6 && ((g.m[0][0] + g.m[1][1]) - 2.0).abs() < 1e-6
            });
            if !pure_trans {
                continue;
            }
            let ev = e.vec;
            let le = ev[0].hypot(ev[1]);
            if le < 1e-9 {
                continue;
            }
            let d = [crystal.pos[e.to][0] - pu[0], crystal.pos[e.to][1] - pu[1]];
            let ld = d[0].hypot(d[1]);
            if ld < 1e-9 || ld > 3.0 * le {
                continue;
            }
            c_sum += ev[0] * d[0] + ev[1] * d[1];
            s_sum += ev[0] * d[1] - ev[1] * d[0];
        }
    }
    if c_sum.hypot(s_sum) < 1e-9 {
        return;
    }
    let theta = s_sum.atan2(c_sum);
    let inv_n = 1.0 / (n as f64);
    let cx = crystal.pos.iter().map(|p| p[0]).sum::<f64>() * inv_n;
    let cy = crystal.pos.iter().map(|p| p[1]).sum::<f64>() * inv_n;
    let (sin_t, cos_t) = (-theta).sin_cos();
    let rot = |p: [f64; 2]| -> [f64; 2] {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        [cx + cos_t * dx - sin_t * dy, cy + sin_t * dx + cos_t * dy]
    };
    for p in &mut crystal.pos {
        *p = rot(*p);
    }
    for adj_u in &mut crystal.adj {
        for e in adj_u {
            if let Some(g) = e.ghost_stub {
                e.ghost_stub = Some(rot(g));
            }
        }
    }
    if crystal.ref_pos.len() == n && !crystal.is_non_orientable && total_frank(crystal).abs() < 1e-2 {
        let tx = crystal.ref_pos.iter().zip(&crystal.pos).map(|(r, p)| r[0] - p[0]).sum::<f64>() * inv_n;
        let ty = crystal.ref_pos.iter().zip(&crystal.pos).map(|(r, p)| r[1] - p[1]).sum::<f64>() * inv_n;
        for p in &mut crystal.pos {
            p[0] += tx;
            p[1] += ty;
        }
        for adj_u in &mut crystal.adj {
            for e in adj_u {
                if let Some(g) = &mut e.ghost_stub {
                    g[0] += tx;
                    g[1] += ty;
                }
            }
        }
    }
}

/// In-drag picture (the full `relax_picture` runs on release).
fn relax_preview_picture(crystal: &mut BuiltCrystal) {
    relax_preview(crystal, 8.0);
}

pub fn item_from_notebook_json(
    cfg: &NotebookConfigJson,
    named_list: &[CatalogueItemUi],
) -> Result<CatalogueItemUi, String> {
    let fam = LatticeFamily::from_recipe(&cfg.recipe, Some(cfg.frame))?;
    let cell = fam.cell();
    let det = cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0];
    if cell.basis.is_empty() || cell.bonds.is_empty() || !det.is_finite() || det == 0.0
        || cell.a1.iter().chain(&cell.a2).chain(cell.basis.iter().flatten()).any(|x| !x.is_finite()) {
        return Err(String::from("configuration recipe/frame does not define a finite, non-degenerate lattice"));
    }
    // Python load_config rebuilds from the recipe; `group` is redundant descriptive metadata,
    // not permission to classify the same cell using a different wallpaper group.
    let group = cfg.recipe.group.clone();
    let (dsymbol_key, network_key, symmetry_index, topologically_chiral) =
        match fam.dsymbol().and_then(|ds| ds.canonical_key().map_err(|e| e.to_string())) {
            Ok(ck) => {
                let nk = dsymbol::network_key(&ck).unwrap_or_else(|_| ck.clone());
                let si = dsymbol::symmetry_index(&ck).unwrap_or(1);
                let tc = dsymbol::is_topologically_chiral(&nk).unwrap_or(false);
                (format_dsymbol_key(&ck), format_dsymbol_key(&nk), si, tc)
            }
            Err(_) => (String::from("-"), String::from("-"), 1, false),
        };
    let chiral = is_geometrically_chiral(&cell).unwrap_or(false);
    // Display names are user-controlled and need not be unique. They never establish identity.
    let same_action = |it: &&CatalogueItemUi| it.group == group && dsymbol_key != "-" && it.dsymbol_key == dsymbol_key;
    let matched_named = named_list.iter()
        .find(|it| it.group == group && it.recipe == cfg.recipe && it.frame == cfg.frame)
        .or_else(|| named_list.iter().filter(same_action)
            .find(|it| it.label == cfg.lattice_label || it.default_label == cfg.lattice_label))
        .or_else(|| named_list.iter().find(same_action));
    let label = if !cfg.lattice_label.is_empty() {
        cfg.lattice_label.clone()
    } else if let Some(nm) = matched_named {
        nm.label.clone()
    } else {
        format!("{group} (imported)")
    };

    let default_label = matched_named.map_or_else(|| label.clone(), |nm| nm.default_label.clone());
    let alias_key = matched_named.map_or_else(
        || LatticeAliasKey::new(&group, "imported", 1, &dsymbol_key),
        |nm| nm.alias_key.clone(),
    );

    Ok(CatalogueItemUi {
        index: 1,
        label,
        default_label,
        alias_key,
        named_key: matched_named.and_then(|n| n.named_key.clone()),
        group,
        sheet_group: matched_named.and_then(|n| n.sheet_group.clone()),
        sites_per_cell: cell.basis.len(),
        site_orbits: cfg.recipe.sites.len(),
        bond_orbits: cfg.recipe.bonds.len().max(1),
        coordination: coordination_string(&cell),
        dsymbol_key,
        network_key,
        symmetry_index,
        chiral,
        topologically_chiral,
        occupied_wyckoff: occupied_wyckoff_letters(&cfg.recipe),
        recipe: cfg.recipe.clone(),
        frame: cfg.frame,
    })
}
