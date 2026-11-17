//! Particle-conserving defect kinematics on 2D crystals:
//! 1D dislocation glide (`Delta_N = 0`, zero climb), disclination stepping with
//! dislocation emission, dislocation absorption by disclinations, and parallel transport.

use crate::defect::combine_api::{classify_crystal_charge, same_charge};
use crate::defect::cone_geometry::{
    cross2, dot2, mat2_inv, mat2_vec, norm2, rot_mat, rotation_about, sub2,
};
use crate::defect::core_graph::LatticeCoreInfo;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::defect::sym_helper::{LatticeGroup, gcd_i32};
use crate::types::{Cell2D, Frame2D};

#[derive(Clone, Debug, PartialEq)]
pub struct DisclinationStepResult {
    pub specs: Vec<DefectSpec>,
    pub emitted: Option<DefectSpec>,
    pub target_at: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbsorptionPreview {
    pub disc_idx: usize,
    pub old_at: [f64; 2],
    pub new_at: [f64; 2],
}

fn spec_pos(lat: &LatticeCoreInfo, idx: usize, spec: &DefectSpec) -> [f64; 2] {
    let (a1, a2) = (lat.lat.cell.a1, lat.lat.cell.a2);
    match *spec {
        DefectSpec::Disclination { at, .. } | DefectSpec::HalfWall { at, .. } => at,
        DefectSpec::Dislocation { at: Some(p), .. } => p,
        DefectSpec::Dislocation { at: None, .. } => [
            (0.371 + 3.0 * idx as f64) * a1[0] + 0.413 * a2[0],
            (0.371 + 3.0 * idx as f64) * a1[1] + 0.413 * a2[1],
        ],
    }
}

fn common_check_patch(
    lat: &LatticeCoreInfo,
    a: &[DefectSpec],
    b: &[DefectSpec],
) -> ([f64; 2], f64) {
    let mut pts: Vec<[f64; 2]> = a.iter().enumerate().map(|(i, s)| spec_pos(lat, i, s)).collect();
    pts.extend(b.iter().enumerate().map(|(i, s)| spec_pos(lat, i, s)));
    let n = pts.len().max(1) as f64;
    let ctr = [
        pts.iter().map(|p| p[0]).sum::<f64>() / n,
        pts.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let max_d = pts.iter().map(|&p| norm2(sub2(p, ctr))).fold(0.0_f64, f64::max);
    let max_b = a
        .iter()
        .chain(b.iter())
        .filter_map(|s| match *s {
            DefectSpec::Dislocation { b, .. } => Some(norm2(b)),
            DefectSpec::Disclination { .. } | DefectSpec::HalfWall { .. } => None,
        })
        .fold(0.0_f64, f64::max);
    let r = max_d + (14.5 * lat.lmax + max_b).max(6.0 * lat.scale);
    (ctr, r)
}

fn distinct_orbit_vectors(grp: &LatticeGroup, base: [f64; 2], seed: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for v in seed {
        if norm2(v) > 1e-6 && !out.iter().any(|&p| norm2(sub2(v, p)) < 1e-6) {
            out.push(v);
        }
    }
    for op in &grp.plus {
        let bm = mat2_vec(op.m, base);
        if norm2(bm) > 1e-6 && !out.iter().any(|&p| norm2(sub2(bm, p)) < 1e-6) {
            out.push(bm);
        }
    }
    out
}

/// Exact perpendicular spacing between adjacent `Delta_N = const` atomic glide planes for `b`.
pub fn glide_line_spacing(cell: &Cell2D, b: [f64; 2]) -> f64 {
    let lat = LatticeCoreInfo::new(cell);
    let area = cross2(lat.lat.cell.a1, lat.lat.cell.a2).abs();
    let f = mat2_vec(lat.lat.a_inv, b);
    let g = gcd_i32(f[0].round_ties_even() as i32, f[1].round_ties_even() as i32).max(1) as f64;
    let b_prim = [b[0] / g, b[1] / g];
    area / norm2(b_prim).max(1e-12)
}

/// Project `target_world` strictly onto the 1D glide line of `b` passing through `anchor_at`.
pub fn project_glide_at(b: [f64; 2], anchor_at: [f64; 2], target_world: [f64; 2]) -> [f64; 2] {
    let lb = norm2(b);
    if lb < 1e-12 {
        return anchor_at;
    }
    let ub = [b[0] / lb, b[1] / lb];
    let s = dot2(sub2(target_world, anchor_at), ub);
    [anchor_at[0] + s * ub[0], anchor_at[1] + s * ub[1]]
}

/// Local preview of the Burgers vector emitted when a disclination steps `c_old -> c_new`.
pub fn preview_emitted_burgers(angle_deg: f64, c_old: [f64; 2], c_new: [f64; 2]) -> [f64; 2] {
    let ra = rot_mat(angle_deg.to_radians());
    let dc = sub2(c_new, c_old);
    let r_dc = mat2_vec(ra, dc);
    sub2(dc, r_dc)
}

/// Step disclination `disc_idx` from its current centre to `target_at` while conserving
/// outer holonomy (`G~+`) and particle number (`Delta_N = 0`) by emitting one dislocation.
pub fn step_disclination_emit(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    disc_idx: usize,
    target_at: [f64; 2],
) -> Result<DisclinationStepResult, String> {
    let Some(&DefectSpec::Disclination { angle_deg, at: c_old, .. }) = specs.get(disc_idx) else {
        return Err(format!("spec {} is not a disclination", disc_idx + 1));
    };
    let lat = LatticeCoreInfo::new(cell);
    if !lat.lat.is_symmetry(&rotation_about(target_at, angle_deg.to_radians()), 2.2) {
        return Err(format!("target {target_at:?} is not a {angle_deg}-deg rotation centre"));
    }
    if norm2(sub2(target_at, c_old)) <= 1e-6 {
        return Ok(DisclinationStepResult {
            specs: specs.to_vec(),
            emitted: None,
            target_at,
        });
    }

    let c_before = build_point_defects(cell, specs, None, None)?;
    let q_before = classify_crystal_charge(group, cell, frame, &c_before)?;

    let grp = LatticeGroup::new(cell);
    let ra = rot_mat(angle_deg.to_radians());
    let dc = sub2(target_at, c_old);
    let mid = [0.5 * (c_old[0] + target_at[0]), 0.5 * (c_old[1] + target_at[1])];
    let b_base = sub2(dc, mat2_vec(ra, dc));
    let b_alt = sub2(mat2_vec([[ra[0][0], ra[1][0]], [ra[0][1], ra[1][1]]], dc), dc);
    let cand_bs = distinct_orbit_vectors(&grp, b_base, vec![b_base, b_alt]);

    for b_em in cand_bs {
        let lb = norm2(b_em);
        let ub = [b_em[0] / lb, b_em[1] / lb];
        for s_mul in [0.0, -1.35, 1.35, -2.4, 2.4, -3.5, 3.5] {
            let off = s_mul * lat.lmax.max(lb) + 0.037 * lat.lmax;
            let core_d = [mid[0] + off * ub[0], mid[1] + off * ub[1]];
            let at_d = [core_d[0] - 0.5 * b_em[0], core_d[1] - 0.5 * b_em[1]];
            let mut new_specs = specs.to_vec();
            new_specs[disc_idx] = DefectSpec::Disclination { angle_deg, at: target_at, cut: None };
            let emitted = DefectSpec::Dislocation { b: b_em, at: Some(at_d), cut: None };
            new_specs.push(emitted.clone());
            let (ctr, r_check) = common_check_patch(&lat, specs, &new_specs);
            if let Ok(c_after) = build_point_defects(cell, &new_specs, Some(r_check), Some(ctr))
                && let Ok(q_after) = classify_crystal_charge(group, cell, frame, &c_after)
                && same_charge(group, cell, &q_before, &q_after, false)
            {
                return Ok(DisclinationStepResult {
                    specs: new_specs,
                    emitted: Some(emitted),
                    target_at,
                });
            }
        }
    }
    Err("no particle-conserving dislocation emission found for this disclination step".to_string())
}

/// Absorb dislocation `disl_idx` into disclination `disc_idx` if it sits on a valid `Delta_N = 0` glide line.
pub fn absorb_dislocation(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    disl_idx: usize,
    disc_idx: usize,
) -> Result<(Vec<DefectSpec>, [f64; 2]), String> {
    let Some(&DefectSpec::Dislocation { b, .. }) = specs.get(disl_idx) else {
        return Err(format!("spec {} is not a dislocation", disl_idx + 1));
    };
    let Some(&DefectSpec::Disclination { angle_deg, at: c_old, .. }) = specs.get(disc_idx) else {
        return Err(format!("spec {} is not a disclination", disc_idx + 1));
    };
    let lat = LatticeCoreInfo::new(cell);
    let lb = norm2(b);
    if lb < 1e-12 {
        return Err("dislocation Burgers vector must be non-zero".to_string());
    }
    let ub = [b[0] / lb, b[1] / lb];
    let nb = [-b[1] / lb, b[0] / lb];
    let at_disl = spec_pos(&lat, disl_idx, &specs[disl_idx]);
    let x_disl = [at_disl[0] + 0.5 * b[0], at_disl[1] + 0.5 * b[1]];
    let dp = glide_line_spacing(cell, b);

    let c_before = build_point_defects(cell, specs, None, None)?;
    let q_before = classify_crystal_charge(group, cell, frame, &c_before)?;

    let grp = LatticeGroup::new(cell);
    let ra = rot_mat(angle_deg.to_radians());
    let inv_i_ra = mat2_inv([[1.0 - ra[0][0], -ra[0][1]], [-ra[1][0], 1.0 - ra[1][1]]]);
    let cand_bs = distinct_orbit_vectors(&grp, b, vec![b, mat2_vec(ra, b)]);

    let mut valid_cands: Vec<(f64, [f64; 2])> = Vec::new();
    for b_eff in cand_bs {
        let c_new = sub2(c_old, mat2_vec(inv_i_ra, b_eff));
        if !lat.lat.is_symmetry(&rotation_about(c_new, angle_deg.to_radians()), 2.2) {
            continue;
        }
        let mid = [0.5 * (c_old[0] + c_new[0]), 0.5 * (c_old[1] + c_new[1])];
        let diff = sub2(mid, x_disl);
        if dot2(diff, nb).abs() <= 0.45 * dp {
            valid_cands.push((dot2(diff, ub).abs(), c_new));
        }
    }
    if valid_cands.is_empty() {
        return Err(
            "cannot absorb: dislocation glide line misses the Delta_N = 0 channel (would require climb)"
                .to_string(),
        );
    }
    valid_cands.sort_by(|x, y| x.0.total_cmp(&y.0));

    let target_k = if disc_idx < disl_idx { disc_idx } else { disc_idx - 1 };
    for (_, c_new) in valid_cands {
        let mut new_specs: Vec<DefectSpec> = specs
            .iter()
            .enumerate()
            .filter_map(|(k, s)| (k != disl_idx).then_some(s.clone()))
            .collect();
        new_specs[target_k] = DefectSpec::Disclination { angle_deg, at: c_new, cut: None };
        let (ctr, r_check) = common_check_patch(&lat, specs, &new_specs);
        if let Ok(c_after) = build_point_defects(cell, &new_specs, Some(r_check), Some(ctr))
            && let Ok(q_after) = classify_crystal_charge(group, cell, frame, &c_after)
            && same_charge(group, cell, &q_before, &q_after, false)
        {
            return Ok((new_specs, c_new));
        }
    }
    Err("cannot absorb: outer holonomy conservation failed".to_string())
}

/// Fast drag-time preview (no crystal rebuild): detects if a gliding dislocation is close to a disclination
/// on a valid `Delta_N = 0` channel where release can absorb it.
pub fn find_absorption_preview(
    cell: &Cell2D,
    specs: &[DefectSpec],
    disl_idx: usize,
    current_at: [f64; 2],
) -> Option<AbsorptionPreview> {
    let &DefectSpec::Dislocation { b, .. } = specs.get(disl_idx)? else {
        return None;
    };
    let lb = norm2(b);
    if lb < 1e-12 {
        return None;
    }
    let nb = [-b[1] / lb, b[0] / lb];
    let x_disl = [current_at[0] + 0.5 * b[0], current_at[1] + 0.5 * b[1]];
    let lat = LatticeCoreInfo::new(cell);
    let at_ref = spec_pos(&lat, disl_idx, &specs[disl_idx]);
    let x_ref = [at_ref[0] + 0.5 * b[0], at_ref[1] + 0.5 * b[1]];
    let capture_r = 1.35 * lat.lmax + 0.5 * lb;
    let dp = glide_line_spacing(cell, b);
    let grp = LatticeGroup::new(cell);

    let mut best: Option<((f64, f64), AbsorptionPreview)> = None;
    for (disc_idx, sp) in specs.iter().enumerate() {
        let DefectSpec::Disclination { angle_deg, at: c_old, .. } = *sp else {
            continue;
        };
        let d_core = norm2(sub2(c_old, x_disl));
        if d_core > capture_r {
            continue;
        }
        let ra = rot_mat(angle_deg.to_radians());
        let inv_i_ra = mat2_inv([[1.0 - ra[0][0], -ra[0][1]], [-ra[1][0], 1.0 - ra[1][1]]]);
        let cand_bs = distinct_orbit_vectors(&grp, b, vec![b, mat2_vec(ra, b)]);
        for b_eff in cand_bs {
            let c_new = sub2(c_old, mat2_vec(inv_i_ra, b_eff));
            if !lat.lat.is_symmetry(&rotation_about(c_new, angle_deg.to_radians()), 2.2) {
                continue;
            }
            let mid = [0.5 * (c_old[0] + c_new[0]), 0.5 * (c_old[1] + c_new[1])];
            if dot2(sub2(mid, x_disl), nb).abs() <= 0.45 * dp {
                let score = (d_core, norm2(sub2(mid, x_ref)));
                if best.is_none_or(|(b_score, _)| score < b_score) {
                    best = Some((score, AbsorptionPreview { disc_idx, old_at: c_old, new_at: c_new }));
                }
            }
        }
    }
    best.map(|(_, prev)| prev)
}

/// On release of a dislocation glide, ensure `G~+` holonomy was preserved (parallel-transporting `b`
/// by an element of `P+` if the dislocation crossed a disclination branch cut).
pub fn parallel_transport_glided_dislocation(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    before_specs: &[DefectSpec],
    after_specs: &[DefectSpec],
    disl_idx: usize,
) -> Result<(Vec<DefectSpec>, bool), String> {
    if !before_specs.iter().any(|s| matches!(s, DefectSpec::Disclination { .. })) {
        build_point_defects(cell, after_specs, None, None)?;
        return Ok((after_specs.to_vec(), false));
    }
    let Some(&DefectSpec::Dislocation { b, at, cut }) = after_specs.get(disl_idx) else {
        return Err(format!("spec {} is not a dislocation", disl_idx + 1));
    };
    let c_before = build_point_defects(cell, before_specs, None, None)?;
    let q_before = classify_crystal_charge(group, cell, frame, &c_before)?;

    if let Ok(c_after) = build_point_defects(cell, after_specs, None, None)
        && let Ok(q_after) = classify_crystal_charge(group, cell, frame, &c_after)
        && same_charge(group, cell, &q_before, &q_after, false)
    {
        return Ok((after_specs.to_vec(), false));
    }

    let grp = LatticeGroup::new(cell);
    let cand_bs = distinct_orbit_vectors(&grp, b, vec![]);
    for bm in cand_bs {
        if norm2(sub2(bm, b)) < 1e-6 {
            continue;
        }
        let mut trial = after_specs.to_vec();
        trial[disl_idx] = DefectSpec::Dislocation { b: bm, at, cut };
        if let Ok(c_after) = build_point_defects(cell, &trial, None, None)
            && let Ok(q_after) = classify_crystal_charge(group, cell, frame, &c_after)
            && same_charge(group, cell, &q_before, &q_after, false)
        {
            return Ok((trial, true));
        }
    }
    Err("glide path blocked by defect core or branch cut".to_string())
}
