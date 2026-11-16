//! Helpers for Volterra branch-cut overlays, explicit per-pin cut angles, refusal explanations,
//! and coincident same-kind pin checks/merges in the Combine studio.

use papertilling_core::defect::surgery::{AppliedCut, DefectSpec};

use crate::canvas2d::CanvasCutRay;
use crate::live_combine::{CombineLatticeInfo, spec_world_pos};

/// Build world-space branch-cut rays extending outward from each defect core to the patch rim.
pub fn build_cut_rays(applied_cuts: &[AppliedCut], patch_centre: [f64; 2], applied_radius: f64) -> Vec<CanvasCutRay> {
    let reach = applied_radius.max(3.0);
    applied_cuts
        .iter()
        .map(|ac| {
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
            CanvasCutRay { origin: ac.origin, end, strip, is_insertion: ac.is_insertion }
        })
        .collect()
}

/// Translate low-level surgical refusal messages into physical/geometric explanations.
pub fn format_refusal_note(err: &str) -> String {
    if err.contains("two walks disagree") || err.contains("cut runs into") || err.contains("overlaps itself") {
        format!("Atomic-scale core/collar overlap (or blocked branch cut): move defects >= 1 cell apart or adjust cut angle ({err})")
    } else if err.contains("close into a cone") {
        format!("Gauss-Bonnet limit (defect-free disk shown): {err}")
    } else {
        format!("surgery refused (defect-free disk shown): {err}")
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
    if let Some(c) = match *spec {
        DefectSpec::Disclination { cut, .. } | DefectSpec::Dislocation { cut, .. } => cut,
    } {
        return c[1].atan2(c[0]).to_degrees();
    }
    let anchor = match *spec {
        DefectSpec::Disclination { at, .. } => at,
        DefectSpec::Dislocation { b, .. } => {
            let at = spec_world_pos(info, idx, spec);
            [at[0] + 0.5 * b[0], at[1] + 0.5 * b[1]]
        }
    };
    if let Some(best) = applied_cuts
        .iter()
        .filter(|ac| ac.strip_end.is_some() == matches!(spec, DefectSpec::Dislocation { .. }))
        .min_by(|a, b| {
            let da = (a.origin[0] - anchor[0]).hypot(a.origin[1] - anchor[1]);
            let db = (b.origin[0] - anchor[0]).hypot(b.origin[1] - anchor[1]);
            da.total_cmp(&db)
        })
        && (best.origin[0] - anchor[0]).hypot(best.origin[1] - anchor[1]) < 0.75 * info.lat_info.lmax + 0.25
    {
        return best.dir[1].atan2(best.dir[0]).to_degrees();
    }
    match *spec {
        DefectSpec::Dislocation { b, .. } => {
            let lb = b[0].hypot(b[1]).max(1e-12);
            (b[0] / lb).atan2(-b[1] / lb).to_degrees()
        }
        DefectSpec::Disclination { .. } => 0.123_f64.to_degrees(),
    }
}

pub fn set_pin_cut_angle(spec: &mut DefectSpec, angle_deg: Option<f64>) {
    let vec = angle_deg.map(|deg| {
        let r = deg.to_radians();
        [r.cos(), r.sin()]
    });
    match spec {
        DefectSpec::Disclination { cut, .. } | DefectSpec::Dislocation { cut, .. } => *cut = vec,
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
        }
    }
    *specs = merged;
}
