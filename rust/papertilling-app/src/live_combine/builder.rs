use std::time::Instant;

use papertilling_core::defect::charge::total_charge;
use papertilling_core::defect::combine_api::{classify_crystal_charge, inspect_crystal_cores};
use papertilling_core::defect::combine_types::NotebookConfigJson;
use papertilling_core::defect::core_graph::core_graph;
use papertilling_core::defect::surgery::{
    DefectSpec, build_point_defects_with_cuts, patch_centre_and_radius,
};
use papertilling_core::dsymbol;
use papertilling_core::plane::draw::{ensure_embedded, relax_picture, relax_preview};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::planarity;
use papertilling_core::types::BuiltCrystal;

use super::cuts;
use super::snap::spec_world_pos;
use super::{CombineBuildOptions, CombineBuildOutput, CombineDrawMode, CombineLatticeInfo};
use crate::aliases::LatticeAliasKey;
use crate::backend::CatalogueItemUi;
use crate::live_ops::{build_pristine_disk, summarize_rings_and_core};
use crate::live_plane::{coordination_string, format_dsymbol_key, occupied_wyckoff_letters};

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
    let base_r = if opts.auto_radius { info.r_auto_base } else { opts.manual_radius };

    if specs.is_empty() {
        let crystal = build_pristine_disk(&info.lat_info, base_r);
        return CombineBuildOutput {
            crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: base_r,
            charge_label: String::from("vacuum (0 defects)"),
            charge_detail: String::from("Frank = +0.0°, b = (0.000, 0.000)"),
            core_nodes: 0, core_edges: 0, core_summary: String::from("defect-free reference disk"),
            local_cores: Vec::new(), patch_centre: [0.0, 0.0], applied_cuts: Vec::new(), coalesced_count: 0,
            note: String::new(), preview, refused: None,
        };
    }

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
        base_r.max(max_d + (5.8 * info.lat_info.lmax + max_b).max(3.6 * info.lat_info.scale))
    } else {
        opts.manual_radius
    };

    let area = (info.cell.a1[0] * info.cell.a2[1] - info.cell.a1[1] * info.cell.a2[0]).abs();
    let inserted_turns: f64 = specs.iter().filter_map(|s| match s {
        DefectSpec::Disclination { angle_deg, .. } if *angle_deg < 0.0 => Some(-angle_deg / 360.0),
        _ => None,
    }).sum();
    let estimated_sites = std::f64::consts::PI * r_target * r_target * info.cell.basis.len() as f64
        * (1.0 + inserted_turns) / area;
    if !estimated_sites.is_finite() || estimated_sites > 20_000.0 {
        let reason = String::from("desktop live-view budget exceeded (~20,000 sites); move pins closer or use a smaller manual patch. No equivalence was computed.");
        return CombineBuildOutput {
            crystal: build_pristine_disk(&info.lat_info, base_r.min(info.r_auto_base)),
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
            applied_radius: base_r.min(info.r_auto_base),
            charge_label: String::from("Live view too large"),
            charge_detail: reason.clone(),
            core_nodes: 0, core_edges: 0, core_summary: String::from("not computed"),
            local_cores: Vec::new(), patch_centre: ctr, applied_cuts: Vec::new(), coalesced_count: 0,
            note: reason.clone(), preview, refused: Some(reason),
        };
    }

    let mut notes = Vec::new();
    let mut applied_r = r_target;
    let (mut crystal, applied_cuts, coalesced_count) = match build_point_defects_with_cuts(&info.cell, specs, Some(r_target), None) {
        Ok(res) => res,
        Err(_) => match build_point_defects_with_cuts(&info.cell, specs, None, None) {
            Ok(res) => {
                if let Ok((_, r_safe)) = patch_centre_and_radius(&info.lat_info, specs, None, None) {
                    applied_r = r_safe;
                }
                notes.push(String::from("expanded to safe surgical radius"));
                res
            }
            Err(err) => {
                return CombineBuildOutput {
                    crystal: build_pristine_disk(&info.lat_info, base_r),
                    elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    applied_radius: base_r,
                    charge_label: String::from("Surgery refused"),
                    charge_detail: err.clone(),
                    core_nodes: 0, core_edges: 0, core_summary: String::from("refused"),
                    local_cores: Vec::new(), patch_centre: ctr, applied_cuts: Vec::new(), coalesced_count: 0,
                    note: cuts::format_refusal_note(&err),
                    preview,
                    refused: Some(err),
                };
            }
        },
    };
    if coalesced_count > 0 {
        notes.push(if applied_cuts.is_empty() {
            format!("{coalesced_count} coincident pin(s) cancelled to vacuum")
        } else {
            format!("superposed {coalesced_count} coincident same-apex pin(s)")
        });
    }

    if opts.draw_mode == CombineDrawMode::FastRelaxed && preview {
        relax_preview_picture(&mut crystal);
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
    } else if opts.draw_mode == CombineDrawMode::ElasticAoMetric {
        crate::live_ops_ao::apply_combine_ao_mode(&mut crystal, preview, &mut notes);
    }

    let (charge_label, charge_detail) = match classify_crystal_charge(&item.group, &info.cell, item.frame, &crystal) {
        Ok(ch) => {
            let det = format!("kind={} | Frank={:+.1}° | loop S=({:+.3}, {:+.3})", ch.kind, ch.angle, ch.loop_sum[0], ch.loop_sum[1]);
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

    if preview {
        return CombineBuildOutput {
            crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: applied_r,
            charge_label, charge_detail, core_nodes: 0, core_edges: 0,
            core_summary: String::from("(core readout on release)"), local_cores: Vec::new(),
            patch_centre: ctr, applied_cuts, coalesced_count, note: notes.join("; "),
            preview, refused: None,
        };
    }
    let (core_nodes, core_edges) = core_graph(&crystal, &info.lat_info, opts.core_radius)
        .or_else(|_| core_graph(&crystal, &info.lat_info, 1))
        .map_or((0, 0), |g| (g.node_count(), g.edge_count()));

    CombineBuildOutput {
        core_summary: summarize_rings_and_core(&crystal, &info.lat_info),
        local_cores: inspect_crystal_cores(&crystal),
        crystal, elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0, applied_radius: applied_r,
        charge_label, charge_detail, core_nodes, core_edges,
        patch_centre: ctr, applied_cuts, coalesced_count, note: notes.join("; "),
        preview, refused: None,
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
    let (dsymbol_key, network_key, symmetry_index) =
        match fam.dsymbol().and_then(|ds| ds.canonical_key().map_err(|e| e.to_string())) {
            Ok(ck) => {
                let nk = dsymbol::network_key(&ck).unwrap_or_else(|_| ck.clone());
                let si = dsymbol::symmetry_index(&ck).unwrap_or(1);
                (format_dsymbol_key(&ck), format_dsymbol_key(&nk), si)
            }
            Err(_) => (String::from("-"), String::from("-"), 1),
        };
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
        occupied_wyckoff: occupied_wyckoff_letters(&cfg.recipe),
        recipe: cfg.recipe.clone(),
        frame: cfg.frame,
    })
}
