use std::time::Instant;

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::charge::total_charge;
use papertilling_core::defect::combine_api::{
    classify_crystal_charge, inspect_crystal_cores,
};
use papertilling_core::defect::combine_types::{CoreFaceSummary, NotebookConfigJson};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph};
use papertilling_core::defect::equivalence::CentreSnapper;
use papertilling_core::defect::surgery::{
    AppliedCut, DefectSpec, build_point_defects_with_cuts, patch_centre_and_radius,
};
use papertilling_core::dsymbol;

pub mod cuts;
use papertilling_core::plane::draw::{ensure_embedded, relax_picture, relax_preview};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::planarity;
use papertilling_core::types::{BuiltCrystal, Cell2D, Isometry2};

use crate::aliases::LatticeAliasKey;
use crate::backend::CatalogueItemUi;
use crate::live_ops::{build_pristine_disk, summarize_rings_and_core};
use crate::live_plane::{coordination_string, format_dsymbol_key, occupied_wyckoff_letters};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineDrawMode {
    FastRelaxed,
    Raw,
    ElasticAoMetric,
}

#[derive(Clone, Debug)]
pub struct CombineCentreOption {
    pub id: String,
    pub label: String,
    pub order: usize,
    pub location: String,
    pub point: [f64; 2],
    pub angles: Vec<f64>,
    pub unit_seeds: Vec<[f64; 2]>,
}

#[derive(Clone, Debug)]
pub struct CombineDislocOption {
    pub id: String,
    pub burgers_mn: (i32, i32),
    pub b: [f64; 2],
    pub at: [f64; 2],
    pub members: Vec<(i32, i32)>,
}

#[derive(Clone, Debug)]
pub struct CombineLatticeInfo {
    pub cell: Cell2D,
    pub lat_info: LatticeCoreInfo,
    pub centres: Vec<CombineCentreOption>,
    pub allowed_angles: Vec<i32>,
    pub dislocations: Vec<CombineDislocOption>,
    pub r_auto_base: f64,
    /// The catalogue writes Burgers coefficients in its reduced primitive basis, not the drawn cell.
    pub burgers_basis: [[f64; 2]; 2],
    pub snapper: CentreSnapper,
}

#[derive(Clone, Debug)]
pub struct CombineBuildOptions {
    pub draw_mode: CombineDrawMode,
    pub full_planar_repair: bool,
    pub auto_radius: bool,
    pub manual_radius: f64,
    pub core_radius: usize,
}

#[derive(Clone, Debug)]
pub struct CombineBuildOutput {
    pub crystal: BuiltCrystal,
    pub elapsed_ms: f64,
    pub applied_radius: f64,
    pub charge_label: String,
    pub charge_detail: String,
    pub core_nodes: usize,
    pub core_edges: usize,
    pub core_summary: String,
    pub local_cores: Vec<CoreFaceSummary>,
    pub patch_centre: [f64; 2],
    pub applied_cuts: Vec<AppliedCut>,
    pub coalesced_count: usize,
    pub note: String,
    /// Cheap in-drag picture: no planarity check and no core readouts (they stay at zero / empty).
    pub preview: bool,
    /// The surgery refused the defects (the reason); `crystal` is then a defect-free stand-in disk.
    pub refused: Option<String>,
}

fn iso_rotation_about(at: [f64; 2], angle_deg: f64) -> Isometry2 {
    let (s, c) = angle_deg.to_radians().sin_cos();
    let t = [at[0] - (c * at[0] - s * at[1]), at[1] - (s * at[0] + c * at[1])];
    Isometry2 { m: [[c, -s], [s, c]], t }
}

fn reduce_in_cell(lat_info: &LatticeCoreInfo, p: [f64; 2]) -> [f64; 2] {
    let (a, inv) = (lat_info.lat.a, lat_info.lat.a_inv);
    let fx = inv[0][0] * p[0] + inv[0][1] * p[1];
    let fy = inv[1][0] * p[0] + inv[1][1] * p[1];
    let (ux, uy) = (fx - (fx + 1e-7).floor(), fy - (fy + 1e-7).floor());
    [a[0][0] * ux + a[0][1] * uy, a[1][0] * ux + a[1][1] * uy]
}

fn same_mod_cell(lat_info: &LatticeCoreInfo, p: [f64; 2], q: [f64; 2]) -> bool {
    let inv = lat_info.lat.a_inv;
    let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
    let fx = inv[0][0] * dx + inv[0][1] * dy;
    let fy = inv[1][0] * dx + inv[1][1] * dy;
    (fx - fx.round_ties_even()).abs() < 1e-4 && (fy - fy.round_ties_even()).abs() < 1e-4
}

fn collect_unit_seeds(
    lat_info: &LatticeCoreInfo,
    p0: [f64; 2],
    p0_order: usize,
    rep: [f64; 2],
    ang: f64,
    target_size: usize,
) -> Vec<[f64; 2]> {
    let mut seeds = vec![reduce_in_cell(lat_info, rep)];
    if target_size <= 1 || p0_order <= 1 {
        return seeds;
    }
    let step = 360.0 / (p0_order as f64);
    for k in 1..p0_order {
        if seeds.len() >= target_size {
            break;
        }
        let img = reduce_in_cell(lat_info, iso_rotation_about(p0, (k as f64) * step).apply(rep));
        if lat_info.lat.is_symmetry(&iso_rotation_about(img, ang), 2.2)
            && !seeds.iter().any(|&s| same_mod_cell(lat_info, s, img))
        {
            seeds.push(img);
        }
    }
    seeds
}

pub fn combine_lattice_info(item: &CatalogueItemUi) -> CombineLatticeInfo {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).expect("valid recipe").cell();
    let cat = DefectCatalogue::new(&item.group, &cell, item.frame);
    let lat_info = LatticeCoreInfo::new(&cell);
    let p0 = cat.centre_classes.first().map_or([0.0, 0.0], |c| c.point);
    let p0_order = cat.centre_classes.first().map_or(1, |c| c.order);

    let mut angle_set: Vec<i32> = Vec::new();
    let centres: Vec<CombineCentreOption> = cat
        .centre_classes
        .iter()
        .map(|c| {
            for &ang in &c.angles {
                let deg = ang.round() as i32;
                if !angle_set.contains(&deg) {
                    angle_set.push(deg);
                }
            }
            let test_ang = c.angles.first().copied().unwrap_or(180.0);
            CombineCentreOption {
                id: c.id.clone(),
                label: c.label.clone(),
                order: c.order,
                location: c.location.clone(),
                point: c.point,
                angles: c.angles.clone(),
                unit_seeds: collect_unit_seeds(&lat_info, p0, p0_order, c.point, test_ang, c.size),
            }
        })
        .collect();
    angle_set.sort_by_key(|&d| (d.abs(), -d));

    let dislocations = cat
        .dislocations
        .iter()
        .map(|d| CombineDislocOption {
            id: d.id.clone(),
            burgers_mn: (d.b_frac[0], d.b_frac[1]),
            b: d.b,
            at: d.at,
            members: d.members.iter().map(|&[m, n]| (m, n)).collect(),
        })
        .collect();

    let cell_area = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs().max(1e-6);
    let density = (cell.basis.len().max(1) as f64) / cell_area;
    let r_500 = (500.0 / (std::f64::consts::PI * density)).sqrt();
    let r_auto_base = r_500.max((6.2 * lat_info.lmax).max(3.6 * lat_info.scale)).clamp(3.0, 14.0);

    let burgers_basis = [cat.cell.a1, cat.cell.a2];
    let snapper = CentreSnapper::new(&item.group, &cell, item.frame);
    CombineLatticeInfo { cell, lat_info, centres, allowed_angles: angle_set, dislocations, r_auto_base, burgers_basis, snapper }
}

pub fn snap_disclination_centre(
    _item: &CatalogueItemUi,
    info: &CombineLatticeInfo,
    angle_deg: f64,
    centre_id_filter: Option<&str>,
    near: [f64; 2],
) -> ([f64; 2], String) {
    if let Some(cid) = centre_id_filter
        && let Some(hit) = snap_to_class(info, angle_deg, cid, near)
    {
        return hit;
    }
    info.snapper.nearest(angle_deg, near).unwrap_or_else(|_| (near, String::from("unconstrained")))
}

fn snap_to_class(info: &CombineLatticeInfo, angle_deg: f64, centre_id: &str, near: [f64; 2]) -> Option<([f64; 2], String)> {
    let (inv, a) = (info.lat_info.lat.a_inv, info.lat_info.lat.a);
    let n0_0 = (inv[0][0] * near[0] + inv[0][1] * near[1]).floor() as i32;
    let n0_1 = (inv[1][0] * near[0] + inv[1][1] * near[1]).floor() as i32;

    let mut best: Option<(f64, [f64; 2], String)> = None;
    for c in &info.centres {
        if c.id != centre_id || !c.angles.iter().any(|&a_deg| (a_deg - angle_deg).abs() <= 0.5) {
            continue;
        }
        for &seed in &c.unit_seeds {
            for d1 in -2..=2 {
                for d2 in -2..=2 {
                    let (m1, m2) = ((n0_0 + d1) as f64, (n0_1 + d2) as f64);
                    let cand = [seed[0] + a[0][0] * m1 + a[0][1] * m2, seed[1] + a[1][0] * m1 + a[1][1] * m2];
                    let dist2 = (cand[0] - near[0]).powi(2) + (cand[1] - near[1]).powi(2);
                    if best.as_ref().is_none_or(|b| dist2 < b.0 - 1e-12) {
                        best = Some((dist2, cand, c.label.clone()));
                    }
                }
            }
        }
    }
    best.map(|(_, p, lbl)| (p, lbl))
}

pub fn preset_neighbour_pair(info: &CombineLatticeInfo) -> Vec<DefectSpec> {
    let Some(top_order) = info.centres.iter().map(|c| c.order).max() else {
        return preset_shortest_dislocation(info);
    };
    let alpha = 360.0 / (top_order as f64);
    let top: Vec<&CombineCentreOption> = info.centres.iter().filter(|c| c.order == top_order).collect();
    let p_plus = top[0].point;
    let p_minus = if top.len() >= 2 {
        top.last().expect("non-empty").point
    } else {
        [p_plus[0] + info.cell.a1[0], p_plus[1] + info.cell.a1[1]]
    };
    vec![DefectSpec::disclination(alpha, p_plus), DefectSpec::disclination(-alpha, p_minus)]
}

pub fn preset_shortest_dislocation(info: &CombineLatticeInfo) -> Vec<DefectSpec> {
    if let Some(d0) = info.dislocations.first() {
        vec![DefectSpec::dislocation(d0.b, Some(d0.at))]
    } else {
        let at = [0.371 * info.cell.a1[0] + 0.413 * info.cell.a2[0], 0.371 * info.cell.a1[1] + 0.413 * info.cell.a2[1]];
        vec![DefectSpec::dislocation(info.cell.a1, Some(at))]
    }
}

pub fn burgers_from_mn(info: &CombineLatticeInfo, m: i32, n: i32) -> [f64; 2] {
    let [a1, a2] = info.burgers_basis;
    [(m as f64) * a1[0] + (n as f64) * a2[0], (m as f64) * a1[1] + (n as f64) * a2[1]]
}

pub fn mn_from_burgers(info: &CombineLatticeInfo, b: [f64; 2]) -> Option<(i32, i32)> {
    let [a1, a2] = info.burgers_basis;
    let a = [[a1[0], a2[0]], [a1[1], a2[1]]];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-12 {
        return None;
    }
    let fm = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
    let fn_ = (-a[1][0] * b[0] + a[0][0] * b[1]) / det;
    let (rm, rn) = (fm.round_ties_even(), fn_.round_ties_even());
    ((fm - rm).abs() < 1e-4 && (fn_ - rn).abs() < 1e-4).then_some((rm as i32, rn as i32))
}

pub fn spec_world_pos(info: &CombineLatticeInfo, idx: usize, spec: &DefectSpec) -> [f64; 2] {
    let (a1, a2) = (info.lat_info.lat.cell.a1, info.lat_info.lat.cell.a2);
    match *spec {
        DefectSpec::Disclination { at, .. } | DefectSpec::HalfWall { at, .. } => at,
        DefectSpec::Dislocation { at: Some(p), .. } => p,
        DefectSpec::Dislocation { at: None, .. } => [
            (0.371 + 3.0 * idx as f64) * a1[0] + 0.413 * a2[0],
            (0.371 + 3.0 * idx as f64) * a1[1] + 0.413 * a2[1],
        ],
    }
}

pub fn centre_class_at(info: &CombineLatticeInfo, p: [f64; 2]) -> Option<&CombineCentreOption> {
    info.centres.iter().find(|c| c.unit_seeds.iter().any(|&s| same_mod_cell(&info.lat_info, p, s)))
}

pub fn spec_short_label(info: &CombineLatticeInfo, spec: &DefectSpec) -> String {
    match spec {
        DefectSpec::Disclination { angle_deg, at, .. } => match centre_class_at(info, *at) {
            Some(c) => format!("D({}, {angle_deg:+.0}°)", c.label),
            None => format!("D({angle_deg:+.0}°)"),
        },
        DefectSpec::Dislocation { b, .. } => mn_from_burgers(info, *b)
            .map_or_else(|| format!("T({:.1},{:.1})", b[0], b[1]), |(m, n)| format!("T({m},{n})")),
        DefectSpec::HalfWall { wall_id, .. } => format!("HW({})", wall_id.as_deref().unwrap_or("wall")),
    }
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

#[cfg(test)]
mod json_tests;
#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AppBackend, CatalogueRuleUi};

    #[test]
    fn snap_centres_burgers_and_refusal_fallback() {
        let backend = AppBackend::new();
        let kagome = backend.find_named("kagome").unwrap();
        let info = combine_lattice_info(kagome);
        assert_eq!(info.centres.len(), 3); // 1a (size 1), 2b (size 2), 3c (size 3)
        assert_eq!(info.centres[0].unit_seeds.len(), 1);
        assert_eq!(info.centres[1].unit_seeds.len(), 2);
        assert_eq!(info.centres[2].unit_seeds.len(), 3);

        let b21 = burgers_from_mn(&info, 2, -1);
        assert_eq!(mn_from_burgers(&info, b21), Some((2, -1)));

        // p3 split 3-fold classes C2 (1b) and C3 (1c) stay distinct under class filter
        let p3_first = &backend.catalogue_for_group("p3", CatalogueRuleUi::Points)[0];
        let p3_info = combine_lattice_info(p3_first);
        assert_eq!(p3_info.centres.len(), 3);
        let (p_c2, _) = snap_disclination_centre(p3_first, &p3_info, 120.0, Some("C2"), p3_info.centres[2].point);
        let (p_c3, _) = snap_disclination_centre(p3_first, &p3_info, 120.0, Some("C3"), p3_info.centres[2].point);
        assert!((p_c2[0] - p_c3[0]).hypot(p_c2[1] - p_c3[1]) > 0.1);

        // Invalid rotation angle at generic point falls back cleanly without panicking
        let opts = CombineBuildOptions {
            draw_mode: CombineDrawMode::FastRelaxed,
            full_planar_repair: false,
            auto_radius: true,
            manual_radius: info.r_auto_base,
            core_radius: 2,
        };
        let bad = [DefectSpec::disclination(73.0, [0.123, 0.456])];
        let out_bad = build_combined_crystal(kagome, &info, &bad, &opts, false);
        assert!(!out_bad.crystal.pos.is_empty() && !out_bad.note.is_empty());
    }
}
