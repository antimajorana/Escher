//! Surgical half-wall builder on a planar disk patch with exact fractional holonomy,
//! matching `papertilling/defects/_half_wall.py`.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::defect::catalogue::DefectCatalogue;
use crate::defect::charge::{check_frames, frame_faces, total_frank};
use crate::defect::cone_builder::mark_long_bonds;
use crate::defect::core_graph::{
    LatticeCoreInfo, SiteKey, TWO_PI, add2, dot2, mat2_vec, norm2, sub2,
};
use crate::defect::half_wall_cuts::{
    CutGeom, build_crystal_from_raw, cut_pure_slip, cut_rot, cut_strip, keep_main_component,
    zip_half_wall,
};
use crate::defect::wall_endings::{WallEndingInfo, nearest_ending_holonomy_in_grp};
use crate::plane::draw::{ensure_embedded, relax_picture};
use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedHalfWall {
    pub wall_id: Option<String>,
    pub wall_kind: String,
    pub at: [f64; 2],
    pub direction: [f64; 2],
    pub sigma: Isometry2,
    pub ending: WallEndingInfo,
    pub holonomy: Isometry2,
}

pub(crate) struct HalfWallBuild {
    pub(crate) crystal: BuiltCrystal,
    pub(crate) sites: Vec<(SiteKey, usize)>,
}

fn lookup_wall_entry(cat: &DefectCatalogue, wall_id: &str) -> Result<(String, Isometry2, [f64; 2]), String> {
    let find_dw = || cat.domain_walls.iter().find(|w| w.id == wall_id).map(|w| (w.kind.clone(), w.sigma, w.direction));
    let find_gb = || cat.glide_breaking.iter().find(|w| w.id == wall_id).map(|w| ("glide_breaking".to_string(), w.sigma, w.direction));
    let find_ms = || cat.mobius_seams.iter().find(|w| w.id == wall_id).map(|w| ("mobius_seam".to_string(), w.sigma, w.direction));
    let find_tw = || cat.twin_walls.iter().find(|w| w.id == wall_id).map(|w| ("twin_wall".to_string(), w.sigma, w.direction));
    let hit = match wall_id.get(..2) {
        Some("Wa") => find_dw(),
        Some("Wc") => find_ms(),
        Some("Wd") => find_gb(),
        Some("Wb") => find_tw(),
        _ => find_dw().or_else(find_gb).or_else(find_ms).or_else(find_tw),
    };
    hit.ok_or_else(|| format!("unknown wall id {wall_id:?} for {}", cat.group))
}

pub fn resolve_half_wall_spec(
    cell: &Cell2D,
    wall_id: Option<&str>,
    sigma: Option<Isometry2>,
    at: [f64; 2],
    direction: Option<[f64; 2]>,
) -> Result<ResolvedHalfWall, String> {
    if !at[0].is_finite() || !at[1].is_finite() {
        return Err("half_wall `at` must be finite".to_string());
    }
    let cat = DefectCatalogue::new("", cell, Frame2D::default());
    let (wall_kind, sigma_eff, default_dir) = if let Some(wid) = wall_id {
        let (wk, s_cat, d_cat) = lookup_wall_entry(&cat, wid)?;
        (wk, sigma.unwrap_or(s_cat), Some(d_cat))
    } else if let Some(s) = sigma {
        ("wall".to_string(), s, None)
    } else {
        return Err("half_wall spec must provide either wall_id or sigma".to_string());
    };
    let u = direction.or(default_dir).unwrap_or([1.0, 0.0]);
    let nu = norm2(u);
    if !nu.is_finite() || nu < 1e-9 {
        return Err("half_wall direction must be non-zero".to_string());
    }
    let u_hat = [u[0] / nu, u[1] / nu];
    let (ending, holonomy) = nearest_ending_holonomy_in_grp(&cat.grp, sigma_eff, at, &wall_kind)?;
    Ok(ResolvedHalfWall {
        wall_id: wall_id.map(str::to_string),
        wall_kind,
        at,
        direction: u_hat,
        sigma: sigma_eff,
        ending,
        holonomy,
    })
}

pub(crate) fn build_half_wall_raw(
    lat_info: &LatticeCoreInfo,
    resolved: &ResolvedHalfWall,
    r_opt: Option<f64>,
    centre_opt: Option<[f64; 2]>,
    check: bool,
) -> Result<HalfWallBuild, String> {
    let (at, u_hat, hol) = (resolved.at, resolved.direction, resolved.holonomy);
    let n_hat = [-u_hat[1], u_hat[0]];
    let (rh, th) = (hol.m, hol.t);
    let mut alpha = rh[1][0].atan2(rh[0][0]);
    if alpha <= -PI + 1e-9 { alpha = PI; }
    let theta = TWO_PI - alpha;
    let t_rel = add2(sub2(mat2_vec(rh, at), at), th);
    let r_patch = r_opt.unwrap_or_else(|| (8.5 * lat_info.lmax + norm2(t_rel)).max(4.0 * lat_info.scale));
    let ctr = centre_opt.unwrap_or(at);
    let allk = lat_info.lat.sites_in_disk(ctr, r_patch + norm2(sub2(ctr, at)) + norm2(t_rel) + 2.5 * lat_info.lmax);
    let pos_all: HashMap<SiteKey, [f64; 2]> = allk.iter().map(|&k| (k, lat_info.lat.pos(k))).collect();
    let geom = CutGeom { lat: &lat_info.lat, allk: &allk, pos_all: &pos_all, at, ctr, u_hat, n_hat, t_rel, r_patch };
    let raw = if alpha.abs() <= 1e-7 {
        let tn = dot2(th, n_hat);
        if tn.abs() < 1e-7 { cut_pure_slip(&geom) } else { cut_strip(&geom, tn) }
    } else {
        cut_rot(&geom, alpha)
    };
    let mut raw = keep_main_component(raw);
    if alpha.abs() > 1e-7 || norm2(th) > 1e-7 {
        zip_half_wall(lat_info, &mut raw, hol, &geom, theta);
    }
    let expected_frank = if alpha.abs() > 1e-7 { alpha.to_degrees() } else { 0.0 };
    let mut crystal = build_crystal_from_raw(&raw);
    mark_long_bonds(&mut crystal);
    if check && lat_info.planar {
        let (frames_bad, faces, frank) = (check_frames(&crystal), frame_faces(&crystal), total_frank(&crystal));
        let euler = (crystal.pos.len() as isize) - (crystal.edges.len() as isize) + (faces.len() as isize);
        let min_deg = crystal.adj.iter().map(Vec::len).min().unwrap_or(0);
        if frames_bad != 0 || euler != 2 || (frank - expected_frank).abs() > 1e-2 || min_deg < 1 {
            return Err(format!("half-wall combinatorial check failed (frames_bad={frames_bad}, euler={euler}, frank={frank:.3} vs {expected_frank:.3}, min_deg={min_deg})"));
        }
    }
    Ok(HalfWallBuild { crystal, sites: raw.sites })
}

pub fn build_half_wall(
    cell: &Cell2D,
    wall_id: Option<&str>,
    sigma: Option<Isometry2>,
    at: [f64; 2],
    direction: Option<[f64; 2]>,
    r_opt: Option<f64>,
    relax: bool,
) -> Result<(ResolvedHalfWall, BuiltCrystal), String> {
    let resolved = resolve_half_wall_spec(cell, wall_id, sigma, at, direction)?;
    let lat_info = LatticeCoreInfo::new(cell);
    let mut crystal = build_half_wall_raw(&lat_info, &resolved, r_opt, None, true)?.crystal;
    if relax {
        relax_picture(&mut crystal);
        if lat_info.planar { ensure_embedded(&mut crystal, false); }
    }
    Ok((resolved, crystal))
}
