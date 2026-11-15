//! Planar Tutte embedding, local disk repair, and wall-opening drawing layer.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use svg::Document;
use svg::node::element::Line;

use crate::defect::core_graph::{dot2, norm2, sub2};
use crate::plane::planarity::{
    PlanarityReport, bond_unit, check_positions, component_roots, has_frames, median_slice,
    signed_area, split_frame_faces,
};
pub use crate::plane::relax::relax_picture;
use crate::plane::relax::{
    RelaxSetup, gauss_newton, pos_to_z, relax_guarded, relax_pinned, work_copy_crystal, z_to_pos,
};
pub use crate::plane::tutte::{solve_tutte_spd, tutte_solve};
use crate::plane::tutte::{connectivity_note, drawn_boundary_loop, global_tutte, local_tutte};
use crate::types::BuiltCrystal;

const LOCAL_RADII: [usize; 5] = [1, 2, 4, 8, 16];
const OPEN_GAPS: [f64; 3] = [0.5, 0.75, 1.0];
const OPEN_SLIDES: [f64; 1] = [0.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelaxationMode {
    GaussNewton,
    Tutte,
    Rigid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PlanarityCheck {
    pub crossings: usize,
    pub overlaps: usize,
    pub crowded: usize,
    pub folds: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrawingInfo {
    pub stage: String,
    pub check: PlanarityCheck,
    pub folded: Option<usize>,
    pub note: String,
    pub three_connected: Option<String>,
    pub radius: Option<usize>,
    pub free: Option<usize>,
    pub steps: Option<usize>,
    pub gap: Option<f64>,
    pub slide: Option<f64>,
    pub strain: f64,
}

/// Render 2D line segments `[(x1, y1, x2, y2)]` to a minimal standalone SVG string.
pub fn segments_to_svg(segments: &[(f64, f64, f64, f64)], view_box: (f64, f64, f64, f64)) -> String {
    let mut doc = Document::new().set("viewBox", view_box);
    for &(x1, y1, x2, y2) in segments {
        let line = Line::new()
            .set("x1", x1)
            .set("y1", y1)
            .set("x2", x2)
            .set("y2", y2)
            .set("stroke", "#222")
            .set("stroke-width", 0.04);
        doc = doc.add(line);
    }
    doc.to_string()
}

fn to_planarity_check(rep: &PlanarityReport) -> PlanarityCheck {
    PlanarityCheck {
        crossings: rep.crossings,
        overlaps: rep.overlaps,
        crowded: rep.crowded,
        folds: rep.folded.unwrap_or(0),
    }
}

/// Format one-line diagnostic note (`""` when clean; `"rigid picture (...)"` on rigid fallback).
pub fn drawing_note(stage: &str, check: &PlanarityCheck, folded: Option<usize>) -> String {
    let mut parts = Vec::new();
    if check.crossings > 0 {
        parts.push(format!("{} crossing{}", check.crossings, if check.crossings == 1 { "" } else { "s" }));
    }
    if check.overlaps > 0 {
        parts.push(format!("{} overlap{}", check.overlaps, if check.overlaps == 1 { "" } else { "s" }));
    }
    if check.crowded > 0 {
        parts.push(format!("{} crowded site{}", check.crowded, if check.crowded == 1 { "" } else { "s" }));
    }
    if let Some(f) = folded
        && f > 0
    {
        parts.push(format!("{} fold{}", f, if f == 1 { "" } else { "s" }));
    }
    if parts.is_empty() {
        return String::new();
    }
    if stage == "rigid" {
        format!("rigid picture: {}", parts.join(", "))
    } else {
        format!("picture keeps {}", parts.join(", "))
    }
}

fn is_combinatorially_planar(crystal: &BuiltCrystal) -> bool {
    if crystal.edges.is_empty() {
        return true;
    }
    let roots = component_roots(crystal.pos.len(), &crystal.edges);
    let mut v_cnt: HashMap<usize, usize> = HashMap::new();
    let mut e_cnt: HashMap<usize, usize> = HashMap::new();
    for (u, &r) in roots.iter().enumerate() {
        if !crystal.adj[u].is_empty() {
            *v_cnt.entry(r).or_default() += 1;
        }
    }
    for &(u, _) in &crystal.edges {
        *e_cnt.entry(roots[u]).or_default() += 1;
    }
    for (&r, &vc) in &v_cnt {
        let ec = e_cnt.get(&r).copied().unwrap_or(0);
        if vc >= 3 && ec > 3 * vc - 6 {
            return false;
        }
    }
    if has_frames(crystal) {
        let (inner, outer) = split_frame_faces(crystal);
        let v_tot: usize = v_cnt.values().sum();
        let f_tot = inner.len() + outer.len();
        return (v_tot as isize) - (crystal.edges.len() as isize) + (f_tot as isize) == 2 * (v_cnt.len() as isize);
    }
    true
}

fn wall_sides(crystal: &BuiltCrystal) -> Option<([f64; 2], Vec<bool>)> {
    let sig = crystal.wall_sigma?;
    if sig.det() > 0.0 || crystal.side.is_empty() {
        return None;
    }
    let (mut has_a, mut has_b) = (false, false);
    let mut is_b = Vec::with_capacity(crystal.side.len());
    for &s in &crystal.side {
        match s {
            Some('A') => {
                has_a = true;
                is_b.push(false);
            }
            Some('B') => {
                has_b = true;
                is_b.push(true);
            }
            _ => return None,
        }
    }
    if !has_a || !has_b {
        return None;
    }
    let s00 = sig.m[0][0];
    let s01 = 0.5 * (sig.m[0][1] + sig.m[1][0]);
    let s11 = sig.m[1][1];
    let lam_min = 0.5 * (s00 + s11 - ((s00 - s11).powi(2) + 4.0 * s01 * s01).sqrt());
    let (c1, c2) = ([s01, lam_min - s00], [lam_min - s11, s01]);
    let raw = if norm2(c1) >= norm2(c2) { if norm2(c1) > 1e-12 { c1 } else { [1.0, 0.0] } } else { c2 };
    let l = norm2(raw).max(1e-12);
    let mut nu = [raw[0] / l, raw[1] / l];
    let (mut sa, mut ca, mut sb, mut cb) = (0.0, 0usize, 0.0, 0usize);
    for (i, &b_flag) in is_b.iter().enumerate() {
        if b_flag {
            sb += dot2(crystal.pos[i], nu);
            cb += 1;
        } else {
            sa += dot2(crystal.pos[i], nu);
            ca += 1;
        }
    }
    if sa / (ca as f64) < sb / (cb as f64) {
        nu = [-nu[0], -nu[1]];
    }
    Some((nu, is_b))
}

fn across_slide(crystal: &BuiltCrystal, x: &[[f64; 2]], nu: [f64; 2], is_b: &[bool]) -> f64 {
    let tau = [-nu[1], nu[0]];
    let mut comps = Vec::new();
    for &(u, v) in &crystal.edges {
        if let Some(e) = crystal.adj[u].iter().find(|b| b.to == v)
            && e.across
        {
            if !is_b[u] && is_b[v] {
                comps.push(dot2(sub2(x[v], x[u]), tau));
            } else if is_b[u] && !is_b[v] {
                comps.push(dot2(sub2(x[u], x[v]), tau));
            }
        }
    }
    if comps.is_empty() { 0.0 } else { median_slice(&mut comps) }
}

fn open_wall(
    crystal: &BuiltCrystal,
    x: &[[f64; 2]],
    gap: f64,
    slide: f64,
    nu: [f64; 2],
    is_b: &[bool],
) -> Vec<[f64; 2]> {
    let mut lens = Vec::new();
    let mut kinds = Vec::new();
    for &(u, v) in &crystal.edges {
        if let Some(e) = crystal.adj[u].iter().find(|b| b.to == v)
            && !e.across
        {
            lens.push(norm2(sub2(x[v], x[u])));
            kinds.push(e.kind);
        }
    }
    let unit = bond_unit(&lens, &kinds);
    let sl = if slide.abs() > 0.0 { slide * across_slide(crystal, x, nu, is_b) } else { 0.0 };
    let tau = [-nu[1], nu[0]];
    let shift = [gap * unit * nu[0] + sl * tau[0], gap * unit * nu[1] + sl * tau[1]];
    x.iter().zip(is_b).map(|(&p, &b)| if b { [p[0] - shift[0], p[1] - shift[1]] } else { p }).collect()
}

fn apply_positions(crystal: &mut BuiltCrystal, pos: Vec<[f64; 2]>) -> f64 {
    crystal.pos = pos;
    let mut worst = 0.0_f64;
    for u in 0..crystal.adj.len() {
        let pu = crystal.pos[u];
        for e in &mut crystal.adj[u] {
            e.ghost_stub = None;
            let ell = norm2(e.vec);
            if ell > 1e-12 {
                worst = worst.max((norm2(sub2(crystal.pos[e.to], pu)) - ell).abs() / ell);
            }
        }
    }
    worst
}

struct Candidate {
    key: usize,
    order: usize,
    stage: String,
    pos: Vec<[f64; 2]>,
    rep: PlanarityReport,
    radius: Option<usize>,
    free: Option<usize>,
    steps: Option<usize>,
    gap: Option<f64>,
    slide: Option<f64>,
}

fn embed_core(
    crystal: &BuiltCrystal,
    relax_given: bool,
    check_overlaps: bool,
    allow_wall_local: bool,
) -> (Vec<[f64; 2]>, DrawingInfo, PlanarityReport) {
    let mut x0 = crystal.pos.clone();
    let n = x0.len();
    let frames = has_frames(crystal);
    let (inner_faces, outer_faces) = if frames { split_frame_faces(crystal) } else { (Vec::new(), Vec::new()) };
    let inner_opt = frames.then_some(inner_faces.as_slice());

    let bad_key = |c: &PlanarityReport| -> usize {
        1_000_000 * (c.crossings + c.folded.unwrap_or(0)) + c.overlaps + if check_overlaps { c.crowded } else { 0 }
    };
    let is_ok = |c: &PlanarityReport| -> bool { bad_key(c) == 0 && c.folded.unwrap_or(0) == 0 };
    let full = |z_pos: &[[f64; 2]]| -> PlanarityReport { check_positions(crystal, z_pos, false, inner_opt) };

    let c_given = check_positions(crystal, &x0, true, inner_opt);
    let mut cands: Vec<Candidate> = vec![Candidate {
        key: bad_key(&c_given),
        order: 0,
        stage: "given".to_string(),
        pos: x0.clone(),
        rep: c_given.clone(),
        radius: None,
        free: None,
        steps: None,
        gap: None,
        slide: None,
    }];

    let finish = |cands: Vec<Candidate>,
                  force_idx: Option<usize>,
                  three_conn: Option<String>|
     -> (Vec<[f64; 2]>, DrawingInfo, PlanarityReport) {
        let chosen = match force_idx {
            Some(i) => &cands[i],
            None => cands.iter().min_by_key(|c| (c.key, std::cmp::Reverse(c.order))).expect("non-empty cands"),
        };
        let pchk = to_planarity_check(&chosen.rep);
        let note = drawing_note(&chosen.stage, &pchk, chosen.rep.folded);
        let info = DrawingInfo {
            stage: chosen.stage.clone(),
            check: pchk,
            folded: chosen.rep.folded,
            note,
            three_connected: three_conn,
            radius: chosen.radius,
            free: chosen.free,
            steps: chosen.steps,
            gap: chosen.gap,
            slide: chosen.slide,
            strain: 0.0,
        };
        (chosen.pos.clone(), info, c_given.clone())
    };

    if c_given.crossings == 0
        && c_given.folded.unwrap_or(0) == 0
        && (!check_overlaps || (c_given.overlaps == 0 && c_given.crowded == 0))
    {
        return finish(cands, Some(0), None);
    }
    if !is_combinatorially_planar(crystal) {
        return finish(cands, Some(0), Some("no: not planar".to_string()));
    }

    let sides = wall_sides(crystal);
    let is_wall = sides.is_some() || crystal.wall_sigma.is_some();
    if let Some((nu, ref is_b)) = sides {
        let mut opened_best: Option<(usize, Vec<[f64; 2]>)> = None;
        for slide in OPEN_SLIDES {
            for gap in OPEN_GAPS {
                let y = open_wall(crystal, &x0, gap, slide, nu, is_b);
                let cy = full(&y);
                let ky = bad_key(&cy);
                let ok = is_ok(&cy);
                if opened_best.as_ref().is_none_or(|(bk, _)| ky < *bk) {
                    opened_best = Some((ky, y.clone()));
                }
                cands.push(Candidate {
                    key: ky,
                    order: cands.len(),
                    stage: "opened".to_string(),
                    pos: y,
                    rep: cy,
                    radius: None,
                    free: None,
                    steps: None,
                    gap: Some(gap),
                    slide: Some(slide),
                });
                if ok {
                    return finish(cands, None, None);
                }
            }
        }
        if let Some((_, best_y)) = opened_best {
            x0 = best_y;
        }
    }

    let w = work_copy_crystal(crystal, &x0);
    if relax_given && (!is_wall || allow_wall_local) {
        let setup = RelaxSetup::from_crystal(&w, &x0);
        let (z, _, _, _) = gauss_newton(&setup, &pos_to_z(&x0), 200, 1e-10, 1e-3);
        let p_rel = z_to_pos(&z);
        let c_rel = full(&p_rel);
        let ok = is_ok(&c_rel);
        cands.push(Candidate {
            key: bad_key(&c_rel),
            order: cands.len(),
            stage: "relaxed".to_string(),
            pos: p_rel,
            rep: c_rel,
            radius: None,
            free: None,
            steps: None,
            gap: None,
            slide: None,
        });
        if ok {
            return finish(cands, None, None);
        }
    }
    if is_wall && !allow_wall_local {
        return finish(cands, None, None);
    }

    let c0 = full(&x0);
    let mut seeds = c0.bad_sites.clone();
    if check_overlaps {
        seeds.extend(c0.crowded_sites.iter().copied());
    }
    let rim: HashSet<usize> = if frames {
        outer_faces.iter().flatten().copied().collect()
    } else {
        drawn_boundary_loop(crystal, &x0).into_iter().collect()
    };

    for k in LOCAL_RADII {
        let (mut p_loc, nfree) = local_tutte(crystal, &x0, &seeds, k, &rim);
        if (nfree as f64) > (if !rim.is_empty() { 0.9 } else { 0.5 }) * (n as f64) {
            break;
        }
        let mut c_loc = full(&p_loc);
        if !is_ok(&c_loc) && !rim.is_empty() {
            let (p_loc2, _) = local_tutte(crystal, &x0, &seeds, k, &HashSet::new());
            let c_loc2 = full(&p_loc2);
            if bad_key(&c_loc2) < bad_key(&c_loc) {
                p_loc = p_loc2;
                c_loc = c_loc2;
            }
        }
        if c_loc.crossings > 0 || c_loc.folded.unwrap_or(0) > 0 {
            cands.push(Candidate {
                key: bad_key(&c_loc),
                order: cands.len(),
                stage: format!("local k={k}"),
                pos: p_loc,
                rep: c_loc,
                radius: Some(k),
                free: Some(nfree),
                steps: None,
                gap: None,
                slide: None,
            });
            continue;
        }
        let setup_loc = RelaxSetup::from_crystal(&w, &p_loc);
        let (z_loc, steps) = relax_guarded(
            &setup_loc,
            &pos_to_z(&p_loc),
            |z| bad_key(&full(&z_to_pos(z))),
            150,
            1e-8,
            16,
            120,
        );
        let p_fin = z_to_pos(&z_loc);
        let c_fin = full(&p_fin);
        let ok = is_ok(&c_fin);
        cands.push(Candidate {
            key: bad_key(&c_fin),
            order: cands.len(),
            stage: "local".to_string(),
            pos: p_fin,
            rep: c_fin,
            radius: Some(k),
            free: Some(nfree),
            steps: Some(steps),
            gap: None,
            slide: None,
        });
        if ok {
            return finish(cands, None, None);
        }
    }

    if is_wall {
        return finish(cands, None, None);
    }

    let outer_rims = if frames { outer_faces.clone() } else { vec![drawn_boundary_loop(crystal, &x0)] };
    let (mut p_tut, mut pinned) = global_tutte(crystal, &outer_rims, &x0, 1.0);
    if !inner_faces.is_empty() {
        let sign_sum = |pts: &[[f64; 2]]| -> isize {
            inner_faces
                .iter()
                .map(|f| {
                    let a = signed_area(pts, f);
                    if a > 0.0 { 1 } else if a < 0.0 { -1 } else { 0 }
                })
                .sum()
        };
        let s0 = sign_sum(&x0).signum();
        if s0 != 0 && sign_sum(&p_tut).signum() == -s0 {
            (p_tut, pinned) = global_tutte(crystal, &outer_rims, &x0, -1.0);
        }
    }
    let three_conn = Some(connectivity_note(crystal, &pinned));
    let c_tut = full(&p_tut);
    cands.push(Candidate {
        key: bad_key(&c_tut),
        order: cands.len(),
        stage: "tutte".to_string(),
        pos: p_tut.clone(),
        rep: c_tut,
        radius: None,
        free: None,
        steps: None,
        gap: None,
        slide: None,
    });

    let setup_t = RelaxSetup::from_crystal(&w, &p_tut);
    let p_tut_z = pos_to_z(&p_tut);
    let (z_tr, _, _, _) = gauss_newton(&setup_t, &p_tut_z, 200, 1e-10, 1e-3);
    let p_tr = z_to_pos(&z_tr);
    let c_tr = full(&p_tr);
    let ok_tr = is_ok(&c_tr);
    cands.push(Candidate {
        key: bad_key(&c_tr),
        order: cands.len(),
        stage: "tutte+relax".to_string(),
        pos: p_tr,
        rep: c_tr,
        radius: None,
        free: None,
        steps: None,
        gap: None,
        slide: None,
    });
    if ok_tr {
        return finish(cands, None, three_conn);
    }

    let pinned_set: HashSet<usize> = pinned.into_iter().collect();
    let free_coords: Vec<usize> =
        (0..n).filter(|v| !pinned_set.contains(v)).flat_map(|v| [2 * v, 2 * v + 1]).collect();
    let (z_pin, _) = relax_pinned(&setup_t, &p_tut_z, &free_coords, 200, 1e-9);
    let p_pin = z_to_pos(&z_pin);
    let c_pin = full(&p_pin);
    let ok_pin = is_ok(&c_pin);
    cands.push(Candidate {
        key: bad_key(&c_pin),
        order: cands.len(),
        stage: "pinned".to_string(),
        pos: p_pin,
        rep: c_pin,
        radius: None,
        free: None,
        steps: None,
        gap: None,
        slide: None,
    });
    if ok_pin {
        return finish(cands, None, three_conn);
    }

    let (z_g, steps_g) =
        relax_guarded(&setup_t, &p_tut_z, |z| bad_key(&full(&z_to_pos(z))), 150, 1e-8, 16, 120);
    let p_g = z_to_pos(&z_g);
    let c_g = full(&p_g);
    cands.push(Candidate {
        key: bad_key(&c_g),
        order: cands.len(),
        stage: "guarded".to_string(),
        pos: p_g,
        rep: c_g,
        radius: None,
        free: None,
        steps: Some(steps_g),
        gap: None,
        slide: None,
    });
    finish(cands, None, three_conn)
}

/// Repair or re-embed `crystal` in place if it has crossings or folded faces (or overlaps/crowding when `check_overlaps`).
pub fn ensure_embedded(crystal: &mut BuiltCrystal, check_overlaps: bool) -> DrawingInfo {
    let (pos, mut info, _) = embed_core(crystal, false, check_overlaps, false);
    if info.stage != "given" {
        info.strain = apply_positions(crystal, pos);
    }
    info
}

/// Produce a planar wall picture with default `wall_local = false` (`embedded_copy(d)` in Python).
pub fn draw_wall_picture(crystal: &BuiltCrystal) -> (BuiltCrystal, Option<DrawingInfo>) {
    draw_wall_picture_with_options(crystal, false)
}

/// Produce a planar wall picture (`None` for non-orientable Möbius seams; rigid side-B opening, optional
/// local repair when `wall_local == true` as used for glide-breaking walls, and `"rigid"` fallback with note).
pub fn draw_wall_picture_with_options(crystal: &BuiltCrystal, wall_local: bool) -> (BuiltCrystal, Option<DrawingInfo>) {
    if crystal.is_non_orientable {
        return (crystal.clone(), None);
    }
    let (pos, mut info, given_rep) = embed_core(crystal, true, true, wall_local);
    let mut out = crystal.clone();
    let is_clean =
        info.check.crossings == 0 && info.check.overlaps == 0 && info.check.crowded == 0 && info.check.folds == 0;
    let given_worse = given_rep.crossings + given_rep.folded.unwrap_or(0);
    let best_worse = info.check.crossings + info.check.folds;
    if info.stage == "given" {
        // Keep exact given drawing (including any stubs if present).
    } else if !is_clean && given_worse <= best_worse {
        let gchk = to_planarity_check(&given_rep);
        info.stage = "rigid".to_string();
        info.check = gchk;
        info.folded = given_rep.folded;
        info.note = drawing_note("rigid", &gchk, given_rep.folded);
    } else {
        info.strain = apply_positions(&mut out, pos);
        info.note = drawing_note(&info.stage, &info.check, info.folded);
    }
    (out, Some(info))
}
