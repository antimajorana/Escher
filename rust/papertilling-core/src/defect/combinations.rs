//! General defect Combinations ("Equivalence") table engine.
//!
//! Enumerates symmetry-inequivalent multi-defect placements, builds each crystal with
//! build-order fallback and rim-margin retry, extracts degree-labelled core graphs
//! unconditionally, computes exact vacuum site-count excess `delta_v`, groups by total
//! `G~+` charge and distinct core graph, and compares each distinct core once against
//! the cached single-defect gallery.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use petgraph::graph::UnGraph;

use crate::defect::charge::{frame_faces, outer_face_index};
use crate::defect::combinations_gen::generate_candidates;
use crate::defect::combinations_types::{
    ChargeGroup, CombinationCandidate, CombinationEntry, CombinationMode, CombinationsOptions,
    CombinationsTable, CoreGroup, RefusedCombination, SingleMatchStatus,
};
use crate::defect::combine_types::ClassifiedCharge;
use crate::defect::cone_geometry::{add2, cross2, dot2, mat2_vec, norm2, same_iso, sub2};
use crate::defect::core_gallery::{config_core_entry, resolve_patch_rim};
use crate::defect::core_graph::{
    LatticeCoreInfo, SiteKey, core_graph_with_rim, isomorphic_cores, patch_rim,
};
use crate::defect::equivalence::{
    catalogue_for_combinations, classify_crystal_charge_with_cat, fmt_signed_g, same_charge_in_grp,
};
use crate::defect::single_cache::SingleDefectCache;
use crate::defect::surgery::{DefectSpec, build_point_defects_with_info, patch_centre_and_radius};
use crate::defect::sym_helper::round_dec;
use crate::defect::volterra_patch::wl_hash_labelled;
use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

#[inline]
fn clean0(x: f64) -> f64 {
    if x.abs() < 1e-9 { 0.0 } else { x }
}

fn build_orders(n: usize) -> Vec<Vec<usize>> {
    match n {
        0 => Vec::new(),
        1 => vec![vec![0]],
        2 => vec![vec![0, 1], vec![1, 0]],
        3 => vec![
            vec![0, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ],
        _ => vec![(0..n).collect(), (0..n).rev().collect()],
    }
}

fn canonical_charge_label(charge: &ClassifiedCharge) -> String {
    match charge.kind.as_str() {
        "vacuum" => "vacuum (total charge 0)".to_string(),
        "dislocation" => {
            let b = charge.b_rep.or(charge.b).unwrap_or([0.0, 0.0]);
            let len = charge.length.unwrap_or_else(|| norm2(b));
            format!("dislocation b = ({:.3}, {:.3}), |b| = {:.3}", clean0(b[0]), clean0(b[1]), len)
        }
        "disclination" => {
            let cls = charge.centre_class.as_deref().unwrap_or("?");
            format!("disclination {}° at {cls}", fmt_signed_g(charge.angle))
        }
        "whole_turn" => format!("whole turn ({}°)", fmt_signed_g(charge.angle)),
        "partial_dislocation" => {
            let b = charge.b.unwrap_or([0.0, 0.0]);
            let len = charge.length.unwrap_or_else(|| norm2(b));
            format!(
                "partial dislocation b = ({:.3}, {:.3}), |b| = {:.3}",
                clean0(b[0]),
                clean0(b[1]),
                len
            )
        }
        _ => charge.label.clone(),
    }
}

fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    let ab = sub2(b, a);
    let len2 = dot2(ab, ab);
    if len2 < 1e-18 {
        return norm2(sub2(p, a)) < 1e-6;
    }
    let t = dot2(sub2(p, a), ab) / len2;
    if !(-1e-6..=(1.0 + 1e-6)).contains(&t) {
        return false;
    }
    let proj = add2(a, [t * ab[0], t * ab[1]]);
    norm2(sub2(p, proj)) < 1e-6
}

/// Half-open Dan Sunday winding number for a point strictly off the polygon boundary.
fn dan_sunday_winding(p: [f64; 2], poly: &[[f64; 2]]) -> i32 {
    let mut wn = 0i32;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if a[1] <= p[1] {
            if b[1] > p[1] && cross2(sub2(b, a), sub2(p, a)) > 0.0 {
                wn += 1;
            }
        } else if b[1] <= p[1] && cross2(sub2(b, a), sub2(p, a)) < 0.0 {
            wn -= 1;
        }
    }
    wn
}

/// Exact site-count excess `V_crystal - V_perfect` for a vacuum (`H = I`) crystal patch.
fn vacuum_delta_v(lat_info: &LatticeCoreInfo, crystal: &BuiltCrystal) -> Option<i32> {
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return None;
    }
    let k_out = outer_face_index(&faces);
    let ccw_outer: Vec<usize> = faces[k_out].iter().copied().rev().collect();
    if ccw_outer.len() < 3 {
        return None;
    }

    let lat = &lat_info.lat;
    let mut dm = Isometry2::IDENTITY;
    let mut x = crystal.ref_pos[ccw_outer[0]];
    let mut poly: Vec<[f64; 2]> = Vec::with_capacity(ccw_outer.len());
    let mut boundary_keys: HashSet<SiteKey> = HashSet::with_capacity(ccw_outer.len());

    for k in 0..ccw_outer.len() {
        let key = lat.locate(x)?;
        let snapped = lat.pos(key);
        poly.push(snapped);
        boundary_keys.insert(key);
        let (u, v) = (ccw_outer[k], ccw_outer[(k + 1) % ccw_outer.len()]);
        let e = crystal.adj[u].iter().find(|b| b.to == v)?;
        x = add2(x, dm.apply_linear(e.vec));
        if let Some(g) = e.seam {
            dm = dm.compose(&g.inverse());
        }
    }
    // Verify loop closure, trivial holonomy, and positive CCW area.
    if norm2(sub2(x, poly[0])) > 1e-5 || !same_iso(&dm, &Isometry2::IDENTITY, 1e-5) {
        return None;
    }
    let area2: f64 = (0..poly.len()).map(|i| cross2(poly[i], poly[(i + 1) % poly.len()])).sum();
    if area2 <= 1e-6 {
        return None;
    }

    let mut min_f = [f64::INFINITY; 2];
    let mut max_f = [f64::NEG_INFINITY; 2];
    for &p in &poly {
        let f = mat2_vec(lat.a_inv, p);
        min_f[0] = min_f[0].min(f[0]);
        min_f[1] = min_f[1].min(f[1]);
        max_f[0] = max_f[0].max(f[0]);
        max_f[1] = max_f[1].max(f[1]);
    }
    let n0_lo = min_f[0].floor() as i32 - 2;
    let n0_hi = max_f[0].ceil() as i32 + 2;
    let n1_lo = min_f[1].floor() as i32 - 2;
    let n1_hi = max_f[1].ceil() as i32 + 2;

    let mut v_perfect = 0usize;
    for n0 in n0_lo..=n0_hi {
        for n1 in n1_lo..=n1_hi {
            for s in 0..lat.cell.basis.len() {
                let key = (n0, n1, s);
                if boundary_keys.contains(&key) {
                    v_perfect += 1;
                    continue;
                }
                let pt = lat.pos(key);
                let on_edge = (0..poly.len())
                    .any(|i| point_on_segment(pt, poly[i], poly[(i + 1) % poly.len()]));
                if on_edge || dan_sunday_winding(pt, &poly) != 0 {
                    v_perfect += 1;
                }
            }
        }
    }
    Some(crystal.pos.len() as i32 - v_perfect as i32)
}

struct BuiltRecord {
    entry: CombinationEntry,
    crystal: BuiltCrystal,
    rim: HashSet<usize>,
    charge: ClassifiedCharge,
    h_cfg: UnGraph<u16, ()>,
    wl_hash: u64,
}

fn try_build_candidate(
    lat_info: &LatticeCoreInfo,
    cat: &crate::defect::catalogue::DefectCatalogue,
    cand: &CombinationCandidate,
    radius: usize,
) -> Result<BuiltRecord, String> {
    let frank_sum: f64 = cand
        .specs
        .iter()
        .filter_map(|s| match s {
            DefectSpec::Disclination { angle_deg, .. } => Some(*angle_deg),
            _ => None,
        })
        .sum();
    if frank_sum >= 360.0 - 1e-6 {
        return Err(format!(
            "total Frank angle {frank_sum:+} deg >= 360 deg: an open patch cannot close into a cone"
        ));
    }
    let orders = build_orders(cand.specs.len());
    if orders.is_empty() {
        return Err("candidate has no defect specs".to_string());
    }

    let mut last_err = "candidate build failed".to_string();
    for order in orders {
        let ordered_specs: Vec<DefectSpec> = order.iter().map(|&i| cand.specs[i].clone()).collect();
        let (mut crystal, _, _) = match build_point_defects_with_info(lat_info, &ordered_specs, None, None) {
            Ok(built) => built,
            Err(e) => {
                last_err = e;
                continue;
            }
        };
        let mut rim = resolve_patch_rim(&crystal, lat_info, &ordered_specs);
        let h_cfg = match core_graph_with_rim(&crystal, &rim, lat_info, radius) {
            Ok(h) => h,
            Err(e) if e.contains("rim") => {
                let retry = patch_centre_and_radius(lat_info, &ordered_specs, None, None).and_then(
                    |(ctr, r_def)| {
                        let r_big = r_def * 1.4;
                        let (c2, _, _) =
                            build_point_defects_with_info(lat_info, &ordered_specs, Some(r_big), None)?;
                        let rim2 = patch_rim(&c2, lat_info, ctr, r_big);
                        let h2 = core_graph_with_rim(&c2, &rim2, lat_info, radius)?;
                        Ok((c2, rim2, h2))
                    },
                );
                match retry {
                    Ok((c2, rim2, h2)) => {
                        crystal = c2;
                        rim = rim2;
                        h2
                    }
                    Err(e2) => {
                        last_err = e2;
                        continue;
                    }
                }
            }
            Err(e) => {
                last_err = e;
                continue;
            }
        };

        let charge = match classify_crystal_charge_with_cat(cat, &crystal) {
            Ok(q) => q,
            Err(e) => {
                last_err = e;
                continue;
            }
        };
        let wl_hash = wl_hash_labelled(&h_cfg);
        let delta_v = (charge.kind == "vacuum").then(|| vacuum_delta_v(lat_info, &crystal)).flatten();
        let entry = CombinationEntry {
            label: cand.label.clone(),
            defect_ids: cand.defect_ids.clone(),
            separation: cand.separation,
            displacements: cand.displacements.clone(),
            specs: ordered_specs,
            build_order: order,
            delta_v,
        };
        return Ok(BuiltRecord { entry, crystal, rim, charge, h_cfg, wl_hash });
    }
    Err(last_err)
}

fn match_tier_rank(tier: &str) -> u8 {
    match tier {
        "exact" => 0,
        "volterra" => 1,
        "completion" => 2,
        _ => 3,
    }
}

fn charge_sort_key(cg: &ChargeGroup) -> (u8, i64, i64, i64, String) {
    let q = &cg.charge;
    let sc = |x: f64| (round_dec(x, 5) * 1e5).round_ties_even() as i64;
    match q.kind.as_str() {
        "vacuum" => (0, 0, 0, 0, cg.charge_label.clone()),
        "dislocation" => {
            let b = q.b_rep.or(q.b).unwrap_or([0.0, 0.0]);
            let len = q.length.unwrap_or_else(|| norm2(b));
            (1, sc(len), sc(b[0]), sc(b[1]), cg.charge_label.clone())
        }
        "disclination" => (
            2,
            sc(q.angle.abs()),
            if q.angle >= 0.0 { 0 } else { 1 },
            0,
            q.centre_class.clone().unwrap_or_else(|| cg.charge_label.clone()),
        ),
        "whole_turn" => (3, sc(q.angle.abs()), if q.angle >= 0.0 { 0 } else { 1 }, 0, cg.charge_label.clone()),
        "partial_dislocation" => (4, sc(q.length.unwrap_or(0.0)), 0, 0, cg.charge_label.clone()),
        _ => (5, 0, 0, 0, cg.charge_label.clone()),
    }
}

/// Build the defect Combinations ("Equivalence") table using a fresh [`SingleDefectCache`].
pub fn build_combinations_table(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    mode: &CombinationMode,
    opts: &CombinationsOptions,
    cancel: Option<&AtomicBool>,
) -> Result<CombinationsTable, String> {
    let mut cache = SingleDefectCache::default();
    build_combinations_table_cached(group, cell, frame, mode, opts, &mut cache, cancel)
}

/// Build the defect Combinations ("Equivalence") table, reusing (and populating) `cache`.
#[allow(clippy::too_many_arguments)]
pub fn build_combinations_table_cached(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    mode: &CombinationMode,
    opts: &CombinationsOptions,
    cache: &mut SingleDefectCache,
    cancel: Option<&AtomicBool>,
) -> Result<CombinationsTable, String> {
    let t0 = Instant::now();
    if !opts.max_sep.is_finite() || opts.max_sep < 0.0 {
        return Err("max_sep must be finite and non-negative".to_string());
    }
    if !opts.bmax.is_finite() || opts.bmax < 0.0 {
        return Err("bmax must be finite and non-negative".to_string());
    }
    let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

    let lat_info = LatticeCoreInfo::new(cell);
    let cat = catalogue_for_combinations(group, cell, frame, opts.bmax);
    if is_cancelled() {
        return Ok(CombinationsTable {
            group: cat.group,
            max_sep: opts.max_sep,
            radius: opts.radius,
            grid: opts.grid,
            total_candidates: 0,
            built_count: 0,
            refused_count: 0,
            cancelled: true,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
            charge_groups: Vec::new(),
            refused: Vec::new(),
        });
    }

    let candidates = generate_candidates(&cat, mode, opts.max_sep)?;
    let total_candidates = candidates.len();
    let mut built_records: Vec<BuiltRecord> = Vec::with_capacity(total_candidates);
    let mut refused: Vec<RefusedCombination> = Vec::new();
    let mut cancelled = false;

    for cand in &candidates {
        if is_cancelled() {
            cancelled = true;
            break;
        }
        match try_build_candidate(&lat_info, &cat, cand, opts.radius) {
            Ok(rec) => built_records.push(rec),
            Err(reason) => refused.push(RefusedCombination {
                label: cand.label.clone(),
                defect_ids: cand.defect_ids.clone(),
                separation: cand.separation,
                displacements: cand.displacements.clone(),
                specs: cand.specs.clone(),
                reason,
            }),
        }
    }

    let built_count = built_records.len();
    let refused_count = refused.len();

    // 1. Group built records by total topological charge in G~+.
    let mut raw_charge_groups: Vec<(ClassifiedCharge, Vec<BuiltRecord>)> = Vec::new();
    for rec in built_records {
        if let Some((_, list)) = raw_charge_groups
            .iter_mut()
            .find(|(q, _)| same_charge_in_grp(&cat.grp, &rec.charge, q, false))
        {
            list.push(rec);
        } else {
            let q = rec.charge.clone();
            raw_charge_groups.push((q, vec![rec]));
        }
    }

    // 2. Inside each charge group, partition by distinct core graph, match once, and draw thumbnail once.
    let mut charge_groups: Vec<ChargeGroup> = Vec::with_capacity(raw_charge_groups.len());
    for (charge, recs) in raw_charge_groups {
        if is_cancelled() {
            cancelled = true;
            break;
        }
        let mut raw_cores: Vec<Vec<BuiltRecord>> = Vec::new();
        for rec in recs {
            if let Some(grp_list) = raw_cores.iter_mut().find(|g| {
                g[0].wl_hash == rec.wl_hash
                    && g[0].entry.delta_v == rec.entry.delta_v
                    && isomorphic_cores(&g[0].h_cfg, &rec.h_cfg)
            }) {
                grp_list.push(rec);
            } else {
                raw_cores.push(vec![rec]);
            }
        }

        let has_single_defect = charge.kind == "vacuum" || charge.single.is_some();
        let mut single_scan = None;
        let mut single_gallery = Vec::new();
        let mut core_groups: Vec<CoreGroup> = Vec::with_capacity(raw_cores.len());

        for mut bucket in raw_cores {
            if is_cancelled() {
                cancelled = true;
                break;
            }
            bucket.sort_by(|a, b| {
                round_dec(a.entry.separation, 6)
                    .total_cmp(&round_dec(b.entry.separation, 6))
                    .then_with(|| a.entry.label.cmp(&b.entry.label))
            });
            let rep = &bucket[0];
            let core_nodes = rep.h_cfg.node_count();
            let core_edges = rep.h_cfg.edge_count();
            let delta_v = rep.entry.delta_v;

            let (status, match_tier, single_matches, volterra_matches, completion_matches) =
                if has_single_defect {
                    match cache.match_core_against_single(
                        &lat_info,
                        &cat,
                        cell,
                        // The representative's own charge: a group holds every orientation of a Burgers vector,
                        // and the single-defect scan depends on the orientation.
                        &rep.charge,
                        &rep.h_cfg,
                        rep.wl_hash,
                        opts.radius,
                        opts.grid,
                        opts.include_single_gallery,
                        cancel,
                    )? {
                        None => {
                            cancelled = true;
                            break;
                        }
                        Some(m) => {
                            if single_scan.is_none() && m.scan.is_some() {
                                single_scan = m.scan;
                            }
                            if opts.include_single_gallery && single_gallery.is_empty() && !m.gallery.is_empty() {
                                single_gallery = m.gallery;
                            }
                            let st = if !m.matches.is_empty() {
                                SingleMatchStatus::MatchesSingle
                            } else {
                                SingleMatchStatus::CoreDiffers
                            };
                            (st, m.match_tier, m.matches, m.volterra_matches, m.completion_matches)
                        }
                    }
                } else {
                    (SingleMatchStatus::NoSingleDefect, "none".to_string(), Vec::new(), Vec::new(), Vec::new())
                };

            let thumbnail = if core_nodes == 0 {
                None
            } else {
                config_core_entry(&rep.crystal, &rep.rim, &lat_info, opts.radius, opts.draw_thumbnails)
            };
            let members: Vec<CombinationEntry> = bucket.into_iter().map(|r| r.entry).collect();
            core_groups.push(CoreGroup {
                core_nodes,
                core_edges,
                status,
                match_tier,
                single_matches,
                volterra_matches,
                completion_matches,
                delta_v,
                thumbnail,
                members,
            });
        }

        core_groups.sort_by(|a, b| {
            a.status
                .cmp(&b.status)
                .then_with(|| match_tier_rank(&a.match_tier).cmp(&match_tier_rank(&b.match_tier)))
                .then_with(|| a.core_nodes.cmp(&b.core_nodes))
                .then_with(|| a.core_edges.cmp(&b.core_edges))
                .then_with(|| {
                    let as0 = a.members.first().map_or(0.0, |m| round_dec(m.separation, 6));
                    let bs0 = b.members.first().map_or(0.0, |m| round_dec(m.separation, 6));
                    as0.total_cmp(&bs0)
                })
                .then_with(|| {
                    let al = a.members.first().map_or("", |m| m.label.as_str());
                    let bl = b.members.first().map_or("", |m| m.label.as_str());
                    al.cmp(bl)
                })
        });

        let charge_label = canonical_charge_label(&charge);
        charge_groups.push(ChargeGroup {
            charge,
            charge_label,
            has_single_defect,
            single_scan,
            single_gallery,
            core_groups,
        });
    }

    charge_groups.sort_by_key(charge_sort_key);
    refused.sort_by(|a, b| {
        round_dec(a.separation, 6)
            .total_cmp(&round_dec(b.separation, 6))
            .then_with(|| a.label.cmp(&b.label))
    });

    Ok(CombinationsTable {
        group: cat.group,
        max_sep: opts.max_sep,
        radius: opts.radius,
        grid: opts.grid,
        total_candidates,
        built_count,
        refused_count,
        cancelled,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        charge_groups,
        refused,
    })
}

#[cfg(test)]
#[path = "combinations_tests.rs"]
mod tests;
