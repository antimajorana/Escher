//! Topological charge classification in `G~+`, mirror conjugation, nearest allowed
//! rotation centres, local core face inspection, and configuration equivalence.

use petgraph::graph::UnGraph;
use std::collections::VecDeque;

use crate::defect::catalogue::DefectCatalogue;
use crate::defect::charge::{
    frame_faces, frank_angles, loop_charge, outer_face_index, total_charge,
};
use crate::defect::combine_types::{ClassifiedCharge, ConfigComparison, CoreFaceSummary};
use crate::defect::cone_geometry::{add2, mat2_inv, mat2_vec, norm2, sub2};
use crate::defect::core_graph::{LatticeCoreInfo, core_graph, isomorphic_cores};
use crate::defect::surgery::DefectSpec;
use crate::defect::sym_helper::{
    LatticeGroup, np_allclose_mat, round_dec, round_even, solve2,
};
use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

pub(crate) fn py_float4(x: f64) -> String {
    let r = round_dec(x, 4);
    format!("{r:?}")
}

pub(crate) fn fmt_vec2_round4(v: [f64; 2]) -> String {
    format!("[{}, {}]", py_float4(v[0]), py_float4(v[1]))
}

pub(crate) fn fmt_signed_g(x: f64) -> String {
    let r = round_dec(x, 6);
    if (r - r.round()).abs() < 1e-9 {
        format!("{:+.0}", r)
    } else {
        format!("{:+}", round_dec(r, 4))
    }
}

/// Check whether Euclidean holonomies `h1` and `h2` are conjugate in the proper space group `G+`.
pub(crate) fn same_proper_holonomy(
    h1: Isometry2,
    h2: Isometry2,
    ops_plus: &[Isometry2],
    _a: [[f64; 2]; 2],
    ainv: [[f64; 2]; 2],
    tol: f64,
) -> bool {
    for g in ops_plus {
        let hc = g.compose(&h1.compose(&g.inverse()));
        if !np_allclose_mat(hc.m, h2.m, 1e-5, tol) {
            continue;
        }
        let dlt = sub2(h2.t, hc.t);
        if np_allclose_mat(hc.m, Isometry2::IDENTITY.m, 1e-5, tol) {
            if norm2(dlt) < tol {
                return true;
            }
            continue;
        }
        let im = [[1.0 - hc.m[0][0], -hc.m[0][1]], [-hc.m[1][0], 1.0 - hc.m[1][1]]];
        let f = mat2_vec(ainv, solve2(im, dlt));
        if (f[0] - round_even(f[0])).abs() < tol && (f[1] - round_even(f[1])).abs() < tol {
            return true;
        }
    }
    false
}

/// Standard 2-colouring BFS bipartiteness test on an undirected graph.
pub(crate) fn is_bipartite<N, E>(g: &UnGraph<N, E>) -> bool {
    let mut color: Vec<u8> = vec![0; g.node_count()];
    let mut dq = VecDeque::new();
    for start in g.node_indices() {
        if color[start.index()] != 0 {
            continue;
        }
        color[start.index()] = 1;
        dq.push_back(start);
        while let Some(u) = dq.pop_front() {
            let cu = color[u.index()];
            let need = if cu == 1 { 2 } else { 1 };
            for v in g.neighbors(u) {
                if v == u {
                    return false;
                }
                let cv = color[v.index()];
                if cv == 0 {
                    color[v.index()] = need;
                    dq.push_back(v);
                } else if cv != need {
                    return false;
                }
            }
        }
    }
    true
}

fn catalogue_for_centres(group: &str, cell: &Cell2D, frame: Frame2D) -> DefectCatalogue {
    let eff_group = if group.is_empty() {
        LatticeGroup::new(cell).name
    } else {
        group
    };
    let cat = DefectCatalogue::with_bounds(eff_group, cell, frame, 0.0, 0.0);
    if frame == Frame2D::default()
        && let Some(first) = cat.centre_classes.first()
        && cat.centre_classes.iter().any(|c| c.label.starts_with("order-"))
    {
        let shifted = Frame2D { q: frame.q, origin: first.point };
        let retry = DefectCatalogue::with_bounds(cat.group.as_str(), &cat.cell, shifted, 0.0, 0.0);
        if retry.centre_classes.iter().all(|c| !c.label.starts_with("order-")) {
            return retry;
        }
    }
    cat
}

/// The rotation-centre classes of one lattice (order, label, members mod the lattice), built once, for repeated
/// snapping (`nearest_allowed_centre` without rebuilding the catalogue on every call; the Combine tab snaps on
/// every hover and drag frame).
#[derive(Clone, Debug)]
pub struct CentreSnapper {
    a: [[f64; 2]; 2],
    ainv: [[f64; 2]; 2],
    classes: Vec<(usize, String, Vec<[f64; 2]>)>,
}

impl CentreSnapper {
    pub fn new(group: &str, cell: &Cell2D, frame: Frame2D) -> Self {
        let cat = catalogue_for_centres(group, cell, frame);
        let classes = cat.centre_classes.iter().map(|r| (r.order, r.label.clone(), r.members.clone())).collect();
        Self { a: cat.grp.a, ainv: cat.grp.ainv, classes }
    }

    /// Nearest point of a class whose order allows `angle_deg` (`angle * order / 360` an integer), with its label;
    /// the rule of Python `LatticeData.nearest_centre`.
    pub fn nearest(&self, angle_deg: f64, near: [f64; 2]) -> Result<([f64; 2], String), String> {
        if !angle_deg.is_finite() || near.iter().any(|x| !x.is_finite()) {
            return Err("angle and position must be finite".to_string());
        }
        if angle_deg.abs() < 1e-9 {
            return Err("disclination angle must be non-zero".to_string());
        }
        let f_near = mat2_vec(self.ainv, near);
        let n0 = [f_near[0].floor(), f_near[1].floor()];
        let mut best: Option<(f64, [f64; 2], &str)> = None;
        for (order, label, members) in &self.classes {
            let k = angle_deg * (*order as f64) / 360.0;
            if (k - round_even(k)).abs() > 1e-6 {
                continue;
            }
            for &q in members {
                for i in -1..=2 {
                    for j in -1..=2 {
                        let sh = mat2_vec(self.a, [n0[0] + i as f64, n0[1] + j as f64]);
                        let p = add2(q, sh);
                        let dist = norm2(sub2(p, near));
                        if best.as_ref().is_none_or(|(bd, _, _)| dist < *bd - 1e-9) {
                            best = Some((dist, p, label.as_str()));
                        }
                    }
                }
            }
        }
        best.map(|(_, p, lbl)| (p, lbl.to_string()))
            .ok_or_else(|| format!("no rotation centre of this lattice allows {angle_deg} degrees"))
    }
}

/// Snap `near` to the nearest allowed Wyckoff rotation centre of `G` permitting `angle_deg`,
/// returning `(snapped_cartesian_point, centre_label)`. For repeated calls build a `CentreSnapper` once.
pub fn nearest_allowed_centre(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    angle_deg: f64,
    near: [f64; 2],
) -> Result<([f64; 2], String), String> {
    if angle_deg.abs() < 1e-9 {
        return Err("disclination angle must be non-zero".to_string());
    }
    CentreSnapper::new(group, cell, frame).nearest(angle_deg, near)
}

fn orbit_rep(grp: &LatticeGroup, b: [f64; 2]) -> [f64; 2] {
    grp.plus
        .iter()
        .map(|g| {
            let mb = g.apply_linear(b);
            [round_dec(mb[0], 12), round_dec(mb[1], 12)]
        })
        .max_by(|u, v| {
            round_dec(u[0], 5)
                .total_cmp(&round_dec(v[0], 5))
                .then_with(|| round_dec(u[1], 5).total_cmp(&round_dec(v[1], 5)))
        })
        .unwrap_or(b)
}

/// Classify the outer-boundary holonomy and unreduced Frank angle of `crystal` in `G~+`.
pub fn classify_crystal_charge(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    crystal: &BuiltCrystal,
) -> Result<ClassifiedCharge, String> {
    let rep = total_charge(crystal)?;
    let angle = if rep.frank_deg == 0.0 { 0.0 } else { rep.frank_deg };
    if rep.kind != "disclination" {
        let grp = LatticeGroup::new(cell);
        let b = rep.burgers.unwrap_or([0.0, 0.0]);
        let f = mat2_vec(grp.ainv, b);
        let lattice_vector =
            Some((f[0] - round_even(f[0])).abs() < 1e-5 && (f[1] - round_even(f[1])).abs() < 1e-5);
        if rep.kind == "vacuum" {
            return Ok(ClassifiedCharge {
                kind: "vacuum".to_string(),
                angle,
                holonomy: rep.holonomy,
                b: Some(b),
                lattice_vector,
                b_rep: None,
                length: None,
                centre: None,
                centre_class: None,
                label: "I (no defect)".to_string(),
                single: None,
                loop_sum: rep.loop_sum,
            });
        }
        if rep.kind == "dislocation" {
            let brep = orbit_rep(&grp, b);
            let len = norm2(b);
            let label = format!(
                "dislocation [b] = P+ orbit of {} (|b| = {:.4})",
                fmt_vec2_round4(brep),
                len
            );
            return Ok(ClassifiedCharge {
                kind: "dislocation".to_string(),
                angle,
                holonomy: rep.holonomy,
                b: Some(b),
                lattice_vector,
                b_rep: Some(brep),
                length: Some(len),
                centre: None,
                centre_class: None,
                label,
                single: Some(DefectSpec::Dislocation { b, at: None, cut: None }),
                loop_sum: rep.loop_sum,
            });
        }
        let label = format!("whole-turn {:+.0} deg with b = {}", angle, fmt_vec2_round4(b));
        return Ok(ClassifiedCharge {
            kind: "whole_turn".to_string(),
            angle,
            holonomy: rep.holonomy,
            b: Some(b),
            lattice_vector,
            b_rep: None,
            length: None,
            centre: None,
            centre_class: None,
            label,
            single: None,
            loop_sum: rep.loop_sum,
        });
    }

    let c = rep.centre.expect("disclination report must carry centre");
    let cat = catalogue_for_centres(group, cell, frame);
    let w = cat.centre_class_of(c).map_or("?", |r| r.label.as_str()).to_string();
    let label = format!("disclination D({w}, {:+.0} deg) about {}", angle, fmt_vec2_round4(c));
    let single = (angle.abs() < 360.0 - 1e-6)
        .then_some(DefectSpec::Disclination { angle_deg: angle, at: c, cut: None });
    Ok(ClassifiedCharge {
        kind: "disclination".to_string(),
        angle,
        holonomy: rep.holonomy,
        b: None,
        lattice_vector: None,
        b_rep: None,
        length: None,
        centre: Some(c),
        centre_class: Some(w),
        label,
        single,
        loop_sum: rep.loop_sum,
    })
}

pub(crate) fn same_charge_in_grp(
    grp: &LatticeGroup,
    q1: &ClassifiedCharge,
    q2: &ClassifiedCharge,
    mirror: bool,
) -> bool {
    if (q1.angle - q2.angle).abs() > 1e-3 {
        return false;
    }
    if same_proper_holonomy(q1.holonomy, q2.holonomy, &grp.plus, grp.a, grp.ainv, 1e-4) {
        return true;
    }
    if mirror
        && let Some(&m0) = grp.minus.first()
    {
        let hm = m0.compose(&q1.holonomy.inverse().compose(&m0.inverse()));
        return same_proper_holonomy(hm, q2.holonomy, &grp.plus, grp.a, grp.ainv, 1e-4);
    }
    false
}

/// Check whether `q1` and `q2` lie in the same conjugacy class of `G~+` (or `G~` when `mirror = true`).
pub fn same_charge(
    _group: &str,
    cell: &Cell2D,
    q1: &ClassifiedCharge,
    q2: &ClassifiedCharge,
    mirror: bool,
) -> bool {
    let grp = LatticeGroup::new(cell);
    same_charge_in_grp(&grp, q1, q2, mirror)
}

/// Extract per-core local face/site summaries (`Configuration.cores()`).
pub fn inspect_crystal_cores(crystal: &BuiltCrystal) -> Vec<CoreFaceSummary> {
    let faces = frame_faces(crystal);
    if faces.is_empty() {
        return Vec::new();
    }
    let k_out = outer_face_index(&faces);
    let franks = frank_angles(crystal, &faces);
    let mut out = Vec::new();
    for (idx, f) in faces.iter().enumerate() {
        if idx == k_out {
            continue;
        }
        let (h, s) = loop_charge(crystal, f);
        let is_trans = (h.m[0][0] - 1.0).abs() <= 1e-6
            && h.m[0][1].abs() <= 1e-6
            && h.m[1][0].abs() <= 1e-6
            && (h.m[1][1] - 1.0).abs() <= 1e-6;
        if is_trans {
            if norm2(s) > 1e-6 {
                out.push(CoreFaceSummary {
                    kind: "dislocation".to_string(),
                    sites: f.clone(),
                    ring: Some(f.len()),
                    coordination: f.iter().map(|&v| crystal.adj[v].len()).collect(),
                    b: Some(s),
                    angle: None,
                    site: None,
                    label: format!("dislocation b = {} on a {}-ring", fmt_vec2_round4(s), f.len()),
                });
            }
            continue;
        }
        let ang = round_dec(franks[idx], 6);
        let c = mat2_vec(
            mat2_inv([[1.0 - h.m[0][0], -h.m[0][1]], [-h.m[1][0], 1.0 - h.m[1][1]]]),
            h.t,
        );
        let mut dm = Isometry2::IDENTITY;
        let mut x = crystal.ref_pos[f[0]];
        let mut site = None;
        for k in 0..f.len() {
            if norm2(sub2(x, c)) < 1e-5 {
                site = Some(f[k]);
                break;
            }
            let (u, v) = (f[k], f[(k + 1) % f.len()]);
            let e = crystal.adj[u].iter().find(|b| b.to == v).expect("face bond must exist");
            x = add2(x, dm.apply_linear(e.vec));
            if let Some(g) = e.seam {
                dm = dm.compose(&g.inverse());
            }
        }
        if let Some(sv) = site {
            let deg_s = crystal.adj[sv].len();
            out.push(CoreFaceSummary {
                kind: "disclination".to_string(),
                sites: vec![sv],
                ring: None,
                coordination: vec![deg_s],
                b: None,
                angle: Some(ang),
                site: Some(sv),
                label: format!("disclination {:+.0} deg on a site with {} bonds", ang, deg_s),
            });
        } else {
            out.push(CoreFaceSummary {
                kind: "disclination".to_string(),
                sites: f.clone(),
                ring: Some(f.len()),
                coordination: f.iter().map(|&v| crystal.adj[v].len()).collect(),
                b: None,
                angle: Some(ang),
                site: None,
                label: format!("disclination {:+.0} deg on a {}-ring", ang, f.len()),
            });
        }
    }
    out
}

/// Compare two built configurations on the same lattice (`same_charge`, `same_charge_up_to_mirror`, `same_core`).
pub fn compare_configurations(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    c1: &BuiltCrystal,
    c2: &BuiltCrystal,
    radius: usize,
) -> Result<ConfigComparison, String> {
    let lat_info = LatticeCoreInfo::new(cell);
    let grp = LatticeGroup::new(cell);
    let q1 = classify_crystal_charge(group, cell, frame, c1)?;
    let q2 = classify_crystal_charge(group, cell, frame, c2)?;
    let h1 = core_graph(c1, &lat_info, radius)?;
    let h2 = core_graph(c2, &lat_info, radius)?;
    Ok(ConfigComparison {
        same_charge: same_charge_in_grp(&grp, &q1, &q2, false),
        same_charge_up_to_mirror: same_charge_in_grp(&grp, &q1, &q2, true),
        same_core: isomorphic_cores(&h1, &h2),
        charge1: q1.label,
        charge2: q2.label,
        core1: (h1.node_count(), h1.edge_count()),
        core2: (h2.node_count(), h2.edge_count()),
    })
}

/// Group built configurations on the same lattice into equivalence classes (`0, 1, ...`).
pub fn classify_configurations(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    crystals: &[BuiltCrystal],
    radius: usize,
) -> Result<Vec<usize>, String> {
    if crystals.is_empty() {
        return Ok(Vec::new());
    }
    let lat_info = LatticeCoreInfo::new(cell);
    let grp = LatticeGroup::new(cell);
    let mut charges = Vec::with_capacity(crystals.len());
    let mut graphs = Vec::with_capacity(crystals.len());
    for c in crystals {
        charges.push(classify_crystal_charge(group, cell, frame, c)?);
        graphs.push(core_graph(c, &lat_info, radius)?);
    }
    let mut reps: Vec<usize> = Vec::new();
    let mut labels = Vec::with_capacity(crystals.len());
    for i in 0..crystals.len() {
        if let Some((k, _)) = reps.iter().enumerate().find(|&(_, &j)| {
            same_charge_in_grp(&grp, &charges[i], &charges[j], false)
                && isomorphic_cores(&graphs[i], &graphs[j])
        }) {
            labels.push(k);
        } else {
            reps.push(i);
            labels.push(reps.len() - 1);
        }
    }
    Ok(labels)
}
