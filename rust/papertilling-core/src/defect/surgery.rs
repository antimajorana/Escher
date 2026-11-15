//! Cone-safe single and multi-defect surgical builder on 2D lattices.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

pub use crate::defect::cone_builder::mark_long_bonds;
use crate::defect::cone_builder::{ConeBuilder, ray_distance};
use crate::defect::core_graph::{
    LatticeCoreInfo, TWO_PI, check_frames, frame_faces, mat2_vec, norm2, rot_mat, rotation_about,
    sub2, total_frank,
};
use crate::types::{BuiltCrystal, Cell2D};

/// Specification of an elementary dislocation or disclination on an open patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DefectSpec {
    Dislocation {
        b: [f64; 2],
        #[serde(default)]
        at: Option<[f64; 2]>,
        #[serde(default)]
        cut: Option<[f64; 2]>,
    },
    Disclination {
        #[serde(alias = "angle")]
        angle_deg: f64,
        at: [f64; 2],
        #[serde(default)]
        cut: Option<[f64; 2]>,
    },
}

impl DefectSpec {
    pub fn dislocation(b: [f64; 2], at: Option<[f64; 2]>) -> Self {
        Self::Dislocation { b, at, cut: None }
    }

    pub fn disclination(angle_deg: f64, at: [f64; 2]) -> Self {
        Self::Disclination { angle_deg, at, cut: None }
    }
}

#[derive(Clone, Debug)]
struct NormSpec {
    is_disloc: bool,
    b: [f64; 2],
    angle_deg: f64,
    at: [f64; 2],
    cut: Option<[f64; 2]>,
}

fn validate_specs(lat_info: &LatticeCoreInfo, specs: &[DefectSpec]) -> Result<Vec<NormSpec>, String> {
    if specs.is_empty() {
        return Err("give at least one defect spec".to_string());
    }
    let lat = &lat_info.lat;
    let mut out = Vec::with_capacity(specs.len());
    let mut total_frank_req = 0.0;
    for (idx, s) in specs.iter().enumerate() {
        match *s {
            DefectSpec::Dislocation { b, at, cut } => {
                let f = mat2_vec(lat.a_inv, b);
                if norm2(b) < 1e-9
                    || (f[0] - f[0].round_ties_even()).abs() > 1e-6
                    || (f[1] - f[1].round_ties_even()).abs() > 1e-6
                {
                    return Err(format!("spec {}: b = {:?} is not a non-zero lattice translation", idx + 1, b));
                }
                let def_at = [
                    (0.371 + 3.0 * idx as f64) * lat.cell.a1[0] + 0.413 * lat.cell.a2[0],
                    (0.371 + 3.0 * idx as f64) * lat.cell.a1[1] + 0.413 * lat.cell.a2[1],
                ];
                out.push(NormSpec { is_disloc: true, b, angle_deg: 0.0, at: at.unwrap_or(def_at), cut });
            }
            DefectSpec::Disclination { angle_deg, at, cut } => {
                if angle_deg.abs() < 1e-9 || angle_deg.abs() >= 360.0 - 1e-9 {
                    return Err(format!("spec {}: disclination needs 0 < |angle| < 360", idx + 1));
                }
                if !lat.is_symmetry(&rotation_about(at, angle_deg.to_radians()), 2.2) {
                    return Err(format!(
                        "spec {}: rotation by {} deg about {:?} is not a lattice symmetry",
                        idx + 1,
                        angle_deg,
                        at
                    ));
                }
                total_frank_req += angle_deg;
                out.push(NormSpec { is_disloc: false, b: [0.0, 0.0], angle_deg, at, cut });
            }
        }
    }
    if total_frank_req >= 360.0 - 1e-6 {
        return Err(format!(
            "total Frank angle {total_frank_req:+} deg >= 360 deg: an open patch cannot close into a cone"
        ));
    }
    Ok(out)
}

/// Compute the patch `(centre, R)` chosen by `build_point_defects`.
pub fn patch_centre_and_radius(
    lat_info: &LatticeCoreInfo,
    specs: &[DefectSpec],
    r_opt: Option<f64>,
    centre_opt: Option<[f64; 2]>,
) -> Result<([f64; 2], f64), String> {
    let norm = validate_specs(lat_info, specs)?;
    let n = norm.len() as f64;
    let ctr = centre_opt.unwrap_or_else(|| {
        [
            norm.iter().map(|s| s.at[0]).sum::<f64>() / n,
            norm.iter().map(|s| s.at[1]).sum::<f64>() / n,
        ]
    });
    let r = r_opt.unwrap_or_else(|| {
        let max_b = norm.iter().filter(|s| s.is_disloc).map(|s| norm2(s.b)).fold(0.0_f64, f64::max);
        let max_d = norm.iter().map(|s| norm2(sub2(s.at, ctr))).fold(0.0_f64, f64::max);
        max_d + (12.5 * lat_info.lmax + max_b).max(5.0 * lat_info.scale)
    });
    Ok((ctr, r))
}

fn candidate_cuts(idx: usize, specs: &[NormSpec]) -> Vec<[f64; 2]> {
    let d = &specs[idx];
    let other_pts: Vec<[f64; 2]> = specs
        .iter()
        .enumerate()
        .filter(|&(j, o)| j != idx && norm2(sub2(o.at, d.at)) > 1e-6)
        .map(|(_, o)| o.at)
        .collect();
    if d.is_disloc {
        let lb = norm2(d.b).max(1e-30);
        let c0 = [-d.b[1] / lb, d.b[0] / lb];
        let mut cand: Vec<[f64; 2]> = [0.0_f64, 25.0, -25.0, 50.0, -50.0, 72.0, -72.0]
            .into_iter()
            .map(|deg| mat2_vec(rot_mat(deg.to_radians()), c0))
            .collect();
        if !other_pts.is_empty() {
            cand.sort_by(|a, b| {
                let ca = other_pts.iter().map(|&pt| ray_distance(pt, d.at, *a)).fold(f64::INFINITY, f64::min);
                let cb = other_pts.iter().map(|&pt| ray_distance(pt, d.at, *b)).fold(f64::INFINITY, f64::min);
                (-ca).total_cmp(&(-cb))
            });
        }
        return cand;
    }
    let alpha = d.angle_deg.to_radians();
    let mut cand: Vec<[f64; 2]> = (0..12)
        .map(|k| {
            let th = 0.117 + (30.0 * k as f64).to_radians();
            [th.cos(), th.sin()]
        })
        .collect();
    if other_pts.is_empty() {
        return cand;
    }
    let other_angs: Vec<f64> = other_pts.iter().map(|&pt| (pt[1] - d.at[1]).atan2(pt[0] - d.at[0])).collect();
    let wedge_clear = |u: [f64; 2]| -> f64 {
        let th0 = u[1].atan2(u[0]);
        let mut worst = PI;
        for &phi in &other_angs {
            let gap = if alpha > 0.0 {
                let cw = (th0 - phi).rem_euclid(TWO_PI);
                if cw <= alpha {
                    return -1.0;
                }
                (cw - alpha).min(TWO_PI - cw)
            } else {
                ((phi - th0 + PI).rem_euclid(TWO_PI) - PI).abs()
            };
            worst = worst.min(gap);
        }
        worst
    };
    cand.sort_by(|a, b| (-wedge_clear(*a)).total_cmp(&(-wedge_clear(*b))));
    cand
}

/// Build single or multi-defect configurations on an open disk patch using cone-safe Volterra surgery.
pub fn build_point_defects(
    cell: &Cell2D,
    specs: &[DefectSpec],
    r_opt: Option<f64>,
    centre_opt: Option<[f64; 2]>,
) -> Result<BuiltCrystal, String> {
    let lat_info = LatticeCoreInfo::new(cell);
    let norm = validate_specs(&lat_info, specs)?;
    let (ctr, r_patch) = patch_centre_and_radius(&lat_info, specs, r_opt, centre_opt)?;
    let mut builder = ConeBuilder::new(&lat_info, ctr, r_patch);
    for (idx, d) in norm.iter().enumerate() {
        let cuts = match d.cut {
            Some(c) => vec![c],
            None => candidate_cuts(idx, &norm),
        };
        let mut last_err = None;
        for cut in cuts {
            let snap = builder.snapshot();
            let res = if d.is_disloc {
                builder.add_dislocation(d.b, d.at, Some(cut))
            } else {
                builder.add_disclination(d.angle_deg, d.at, Some(cut))
            };
            match res {
                Ok(()) => {
                    last_err = None;
                    break;
                }
                Err(e) => {
                    builder.restore(snap);
                    last_err = Some(e);
                }
            }
        }
        if let Some(e) = last_err {
            return Err(format!("defect {}: {e}", idx + 1));
        }
    }
    let crystal = builder.finish()?;
    let expected_frank: f64 = norm.iter().filter(|s| !s.is_disloc).map(|s| s.angle_deg).sum();
    if lat_info.planar {
        let frames_bad = check_frames(&crystal);
        let faces = frame_faces(&crystal);
        let euler = (crystal.pos.len() as isize) - (crystal.edges.len() as isize) + (faces.len() as isize);
        let frank = total_frank(&crystal);
        let min_deg = crystal.adj.iter().map(Vec::len).min().unwrap_or(0);
        if frames_bad != 0 || euler != 2 || (frank - expected_frank).abs() > 1e-2 || min_deg < 1 {
            return Err(format!(
                "combinatorial check failed (frames_bad={frames_bad}, euler={euler}, frank={frank:.3} vs {expected_frank:.3}, min_deg={min_deg})"
            ));
        }
    }
    Ok(crystal)
}
