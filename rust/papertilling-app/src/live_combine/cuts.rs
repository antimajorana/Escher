//! Helpers for Volterra branch-cut overlays, explicit per-pin cut angles, refusal explanations,
//! and coincident same-kind pin checks/merges in the Combine studio.

use papertilling_core::defect::surgery::{AppliedCut, DefectSpec};
use papertilling_core::types::BuiltCrystal;

use crate::canvas2d::CanvasCutRay;
use crate::live_combine::{CombineLatticeInfo, map_ref_to_drawn, spec_core_ref_pos, spec_world_pos};

/// Explicit branch-cut or half-wall direction vector stored on `spec`, if any.
pub fn pin_explicit_cut_vec(spec: &DefectSpec) -> Option<[f64; 2]> {
    match *spec {
        DefectSpec::Disclination { cut, .. } | DefectSpec::Dislocation { cut, .. } => cut,
        DefectSpec::HalfWall { direction, .. } => direction,
    }
}

fn spec_cut_anchor(info: &CombineLatticeInfo, idx: usize, spec: &DefectSpec) -> [f64; 2] {
    spec_core_ref_pos(info, idx, spec)
}

fn spec_matches_cut_kind(spec: &DefectSpec, ac: &AppliedCut) -> bool {
    ac.is_half_wall == matches!(spec, DefectSpec::HalfWall { .. })
        && ac.strip_end.is_some() == matches!(spec, DefectSpec::Dislocation { .. })
}

fn cut_match_tolerance(info: &CombineLatticeInfo) -> f64 {
    0.75 * info.lat_info.lmax + 0.25
}

fn find_matching_cut_index(
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    idx: usize,
    applied_cuts: &[AppliedCut],
) -> Option<usize> {
    let spec = specs.get(idx)?;
    let anchor = spec_cut_anchor(info, idx, spec);
    let tol = cut_match_tolerance(info);
    applied_cuts
        .iter()
        .enumerate()
        .filter(|(_, ac)| spec_matches_cut_kind(spec, ac))
        .map(|(ci, ac)| (ci, (ac.origin[0] - anchor[0]).hypot(ac.origin[1] - anchor[1])))
        .filter(|&(_, d)| d < tol)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(ci, _)| ci)
}

/// True when placed pin `idx` coincides (`<= 1e-6`) with another placed pin of the same variant.
pub fn is_pin_coincident(info: &CombineLatticeInfo, specs: &[DefectSpec], idx: usize) -> bool {
    let Some(sp) = specs.get(idx) else {
        return false;
    };
    let pi = spec_world_pos(info, idx, sp);
    let ki = std::mem::discriminant(sp);
    specs.iter().enumerate().any(|(j, sj)| {
        j != idx
            && std::mem::discriminant(sj) == ki
            && (pi[0] - spec_world_pos(info, j, sj)[0])
                .hypot(pi[1] - spec_world_pos(info, j, sj)[1])
                <= 1e-6
    })
}

fn match_cut_spec_indices(
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    applied_cuts: &[AppliedCut],
) -> Vec<Option<usize>> {
    let coincident: Vec<bool> = (0..specs.len())
        .map(|i| is_pin_coincident(info, specs, i))
        .collect();
    let tol = cut_match_tolerance(info);
    let mut owners = vec![None; applied_cuts.len()];
    let mut used_specs = vec![false; specs.len()];
    for (ci, ac) in applied_cuts.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        let mut tied = false;
        for (idx, sp) in specs.iter().enumerate() {
            if !spec_matches_cut_kind(sp, ac) {
                continue;
            }
            let anchor = spec_cut_anchor(info, idx, sp);
            let d = (ac.origin[0] - anchor[0]).hypot(ac.origin[1] - anchor[1]);
            if d >= tol {
                continue;
            }
            match best {
                None => {
                    best = Some((idx, d));
                    tied = false;
                }
                Some((_, bd)) if d < bd - 1e-6 => {
                    best = Some((idx, d));
                    tied = false;
                }
                Some((_, bd)) if (d - bd).abs() <= 1e-6 => {
                    tied = true;
                }
                _ => {}
            }
        }
        if let Some((idx, _)) = best
            && !tied
            && !coincident[idx]
            && !used_specs[idx]
            && find_matching_cut_index(info, specs, idx, applied_cuts) == Some(ci)
        {
            used_specs[idx] = true;
            owners[ci] = Some(idx);
        }
    }
    owners
}

/// Convert one `AppliedCut` into a world-space `CanvasCutRay` extending out to the patch rim.
pub fn build_cut_ray(
    ac: &AppliedCut,
    patch_centre: [f64; 2],
    applied_radius: f64,
    spec_index: Option<usize>,
) -> CanvasCutRay {
    let reach = applied_radius.max(3.0);
    let len_d = ac.dir[0].hypot(ac.dir[1]).max(1e-12);
    let ux = ac.dir[0] / len_d;
    let uy = ac.dir[1] / len_d;
    let ox = ac.origin[0] - patch_centre[0];
    let oy = ac.origin[1] - patch_centre[1];
    let b_dot = ox * ux + oy * uy;
    let c_val = ox * ox + oy * oy - reach * reach;
    let disc = (b_dot * b_dot - c_val).max(0.0);
    let s_hit = (-b_dot + disc.sqrt()).max(1.5);
    let end = [ac.origin[0] + s_hit * ux, ac.origin[1] + s_hit * uy];
    let strip = ac.strip_end.map(|sb| {
        let sa = [2.0 * ac.origin[0] - sb[0], 2.0 * ac.origin[1] - sb[1]];
        (sa, sb)
    });
    CanvasCutRay {
        origin: ac.origin,
        end,
        strip,
        is_insertion: ac.is_insertion,
        is_half_wall: ac.is_half_wall,
        spec_index,
    }
}

/// Build world-space branch-cut rays with each unambiguous ray matched to its `specs` index.
pub fn build_cut_rays_for_specs(
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    applied_cuts: &[AppliedCut],
    patch_centre: [f64; 2],
    applied_radius: f64,
) -> Vec<CanvasCutRay> {
    let owners = match_cut_spec_indices(info, specs, applied_cuts);
    applied_cuts
        .iter()
        .enumerate()
        .map(|(ci, ac)| build_cut_ray(ac, patch_centre, applied_radius, owners[ci]))
        .collect()
}

/// Synthesize a `CanvasCutRay` directly from `specs[idx]` (for live cut-handle dragging or recovery
/// when a surgical refusal leaves `applied_cuts` empty), returning `None` if `idx` is coincident/invalid.
pub fn build_fallback_cut_ray_for_spec(
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    idx: usize,
    applied_cuts: &[AppliedCut],
    patch_centre: [f64; 2],
    applied_radius: f64,
) -> Option<CanvasCutRay> {
    let spec = specs.get(idx)?;
    if is_pin_coincident(info, specs, idx) {
        return None;
    }
    let origin = spec_cut_anchor(info, idx, spec);
    let dir = match pin_explicit_cut_vec(spec) {
        Some(c) => {
            let l = c[0].hypot(c[1]).max(1e-12);
            [c[0] / l, c[1] / l]
        }
        None => {
            let rad = default_cut_angle_deg(info, specs, idx, applied_cuts).to_radians();
            [rad.cos(), rad.sin()]
        }
    };
    let (strip_end, is_insertion, is_half_wall) = match *spec {
        DefectSpec::Dislocation { b, .. } => {
            let at = spec_world_pos(info, idx, spec);
            (Some([at[0] + b[0], at[1] + b[1]]), dir[0] * b[1] - dir[1] * b[0] > 0.0, false)
        }
        DefectSpec::Disclination { .. } => (None, false, false),
        DefectSpec::HalfWall { .. } => (None, false, true),
    };
    let ac = AppliedCut { origin, dir, strip_end, is_insertion, is_half_wall };
    Some(build_cut_ray(&ac, patch_centre, applied_radius, Some(idx)))
}

/// Shift a reference-frame `CanvasCutRay` so its origin sits at the deformed drawn defect core.
pub fn map_cut_ray_to_drawn(crystal: &BuiltCrystal, mut ray: CanvasCutRay) -> CanvasCutRay {
    let drawn_origin = map_ref_to_drawn(crystal, ray.origin);
    let dx = drawn_origin[0] - ray.origin[0];
    let dy = drawn_origin[1] - ray.origin[1];
    ray.origin = drawn_origin;
    ray.end = [ray.end[0] + dx, ray.end[1] + dy];
    if let Some((sa, sb)) = ray.strip {
        ray.strip = Some(([sa[0] + dx, sa[1] + dy], [sb[0] + dx, sb[1] + dy]));
    }
    ray
}

/// Translate low-level surgical refusal messages into physical/geometric explanations.
pub fn format_refusal_note(err: &str) -> String {
    if err.contains("two walks disagree") || err.contains("cut runs into") || err.contains("overlaps itself") {
        format!("Atomic-scale core/collar overlap (or blocked branch cut): move defects >= 1 cell apart or adjust cut angle ({err})")
    } else if err.contains("close into a cone") {
        format!("Gauss-Bonnet limit: {err}")
    } else {
        format!("surgery refused: {err}")
    }
}

/// Return a sensible initial cut angle (degrees in `[-180, 180]`) for placed pin `idx`.
pub fn default_cut_angle_deg(
    info: &CombineLatticeInfo,
    specs: &[DefectSpec],
    idx: usize,
    applied_cuts: &[AppliedCut],
) -> f64 {
    let Some(spec) = specs.get(idx) else {
        return 7.0;
    };
    if let Some(c) = pin_explicit_cut_vec(spec) {
        return c[1].atan2(c[0]).to_degrees();
    }
    if let Some(ci) = find_matching_cut_index(info, specs, idx, applied_cuts) {
        let best = &applied_cuts[ci];
        return best.dir[1].atan2(best.dir[0]).to_degrees();
    }
    match *spec {
        DefectSpec::Dislocation { b, .. } => {
            let lb = b[0].hypot(b[1]).max(1e-12);
            (b[0] / lb).atan2(-b[1] / lb).to_degrees()
        }
        DefectSpec::Disclination { .. } | DefectSpec::HalfWall { .. } => 0.123_f64.to_degrees(),
    }
}

pub fn set_pin_cut_angle(spec: &mut DefectSpec, angle_deg: Option<f64>) {
    let vec = angle_deg.map(|deg| {
        let r = deg.to_radians();
        [r.cos(), r.sin()]
    });
    match spec {
        DefectSpec::Disclination { cut, .. } | DefectSpec::Dislocation { cut, .. } => *cut = vec,
        DefectSpec::HalfWall { direction, .. } => *direction = vec,
    }
}

/// True when two placed pins of the same kind share the same point (`<= 1e-6`).
pub fn has_coincident_same_kind_pins(info: &CombineLatticeInfo, specs: &[DefectSpec]) -> bool {
    for i in 0..specs.len() {
        let pi = spec_world_pos(info, i, &specs[i]);
        let di = matches!(specs[i], DefectSpec::Dislocation { .. });
        for (j, sj) in specs.iter().enumerate().skip(i + 1) {
            if matches!(sj, DefectSpec::Dislocation { .. }) == di {
                let pj = spec_world_pos(info, j, sj);
                if (pi[0] - pj[0]).hypot(pi[1] - pj[1]) <= 1e-6 {
                    return true;
                }
            }
        }
    }
    false
}

/// Permanently combine coincident same-kind pins in the Combine list (dropping exact 0-sum pairs).
pub fn merge_coincident_pins(info: &CombineLatticeInfo, specs: &mut Vec<DefectSpec>) {
    let mut groups: Vec<(bool, [f64; 2], Vec<DefectSpec>)> = Vec::new();
    for (idx, s) in specs.iter().enumerate() {
        let pos = spec_world_pos(info, idx, s);
        let is_d = matches!(s, DefectSpec::Dislocation { .. });
        if let Some(grp) = groups
            .iter_mut()
            .find(|(gd, gpos, _)| *gd == is_d && (gpos[0] - pos[0]).hypot(gpos[1] - pos[1]) <= 1e-6)
        {
            grp.2.push(s.clone());
        } else {
            groups.push((is_d, pos, vec![s.clone()]));
        }
    }
    let mut merged = Vec::with_capacity(groups.len());
    for (_, gpos, grp) in groups {
        if grp.len() == 1 {
            merged.push(grp.into_iter().next().expect("len 1"));
            continue;
        }
        match grp[0] {
            DefectSpec::Disclination { at, cut, .. } => {
                let sum: f64 = grp
                    .iter()
                    .filter_map(|s| if let DefectSpec::Disclination { angle_deg, .. } = s { Some(*angle_deg) } else { None })
                    .sum();
                if sum.abs() > 1e-9 {
                    let cut0 = grp
                        .iter()
                        .find_map(|s| if let DefectSpec::Disclination { cut: c, .. } = s { *c } else { None })
                        .or(cut);
                    merged.push(DefectSpec::Disclination { angle_deg: sum, at, cut: cut0 });
                }
            }
            DefectSpec::Dislocation { cut, .. } => {
                let mut sum = [0.0, 0.0];
                for s in &grp {
                    if let DefectSpec::Dislocation { b, .. } = s {
                        sum[0] += b[0];
                        sum[1] += b[1];
                    }
                }
                if sum[0].hypot(sum[1]) > 1e-9 {
                    let cut0 = grp
                        .iter()
                        .find_map(|s| if let DefectSpec::Dislocation { cut: c, .. } = s { *c } else { None })
                        .or(cut);
                    merged.push(DefectSpec::Dislocation { b: sum, at: Some(gpos), cut: cut0 });
                }
            }
            DefectSpec::HalfWall { .. } => merged.extend(grp),
        }
    }
    *specs = merged;
}
