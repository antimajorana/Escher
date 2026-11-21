use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::equivalence::CentreSnapper;
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{BuiltCrystal, Isometry2};

use super::{CombineCentreOption, CombineDislocOption, CombineLatticeInfo};
use crate::backend::CatalogueItemUi;

pub(super) fn iso_rotation_about(at: [f64; 2], angle_deg: f64) -> Isometry2 {
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

pub(super) fn same_mod_cell(lat_info: &LatticeCoreInfo, p: [f64; 2], q: [f64; 2]) -> bool {
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

pub fn spec_core_ref_pos(info: &CombineLatticeInfo, idx: usize, spec: &DefectSpec) -> [f64; 2] {
    let at = spec_world_pos(info, idx, spec);
    match *spec {
        DefectSpec::Disclination { .. } | DefectSpec::HalfWall { .. } => at,
        DefectSpec::Dislocation { b, .. } => [at[0] + 0.5 * b[0], at[1] + 0.5 * b[1]],
    }
}

fn idw_map(src: &[[f64; 2]], dst: &[[f64; 2]], pt: [f64; 2]) -> [f64; 2] {
    if src.is_empty() || src.len() != dst.len() {
        return pt;
    }
    const K: usize = 6;
    let mut best: [(f64, usize); K] = [(f64::INFINITY, 0); K];
    let mut count = 0usize;
    for (i, &p) in src.iter().enumerate() {
        let d2 = (p[0] - pt[0]).powi(2) + (p[1] - pt[1]).powi(2);
        if count < K {
            best[count] = (d2, i);
            count += 1;
            let mut j = count - 1;
            while j > 0 && best[j].0 < best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        } else if d2 < best[K - 1].0 {
            best[K - 1] = (d2, i);
            let mut j = K - 1;
            while j > 0 && best[j].0 < best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    let (mut ex, mut ey, mut en) = (0.0, 0.0, 0usize);
    for &(d2, i) in &best[..count] {
        if d2 < 1e-12 {
            ex += dst[i][0];
            ey += dst[i][1];
            en += 1;
        }
    }
    if en > 0 {
        return [ex / en as f64, ey / en as f64];
    }
    let (mut wx, mut wy, mut ws) = (0.0, 0.0, 0.0);
    for &(d2, i) in &best[..count] {
        let w = 1.0 / (d2 + 1e-6);
        wx += w * (dst[i][0] - src[i][0]);
        wy += w * (dst[i][1] - src[i][1]);
        ws += w;
    }
    if ws > 0.0 { [pt[0] + wx / ws, pt[1] + wy / ws] } else { pt }
}

/// Map a reference-lattice point (`crystal.ref_pos`) into the deformed drawn crystal (`crystal.pos`).
pub fn map_ref_to_drawn(crystal: &BuiltCrystal, ref_pt: [f64; 2]) -> [f64; 2] {
    idw_map(&crystal.ref_pos, &crystal.pos, ref_pt)
}

/// Approximate inverse of `map_ref_to_drawn`: map a drawn canvas point back to `crystal.ref_pos`.
pub fn map_drawn_to_ref(crystal: &BuiltCrystal, drawn_pt: [f64; 2]) -> [f64; 2] {
    idw_map(&crystal.pos, &crystal.ref_pos, drawn_pt)
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
