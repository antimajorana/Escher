//! Layer 1 holonomy & fusion observables: conjugate defect fusion channels,
//! signed ribbon-dart face walks, Cartan plaquette holonomies, twisted coboundary,
//! and vertex G-gauge transformations.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::charge::{corners_order, loop_charge, outer_face_index};
use crate::defect::cone_geometry::{mat2_inv, mat2_vec, norm2, rotation_about, sub2};
use crate::defect::gauge::{SpaceGroupElement, SpaceGroupExtension, in_image_of_int_mat};
use crate::defect::surgery::DefectSpec;
use crate::defect::sym_helper::{imat2_det, imat2_vec, mat2_mul, np_allclose_mat, round_dec};
use crate::types::{BuiltCrystal, Cell2D, Isometry2};

const ID2: [[i32; 2]; 2] = [[1, 0], [0, 1]];

/// Conjugate defect fusion channel `[g1 * h * g2 * h^-1]` for `h in G`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FusionChannel {
    pub kind: String,
    pub angle_deg: f64,
    pub det_sign: i32,
    pub burgers_coset_rep: Option<[i32; 2]>,
    pub shortest_nonzero_burgers_frac: Vec<[i32; 2]>,
    pub parity_mod2: Option<i32>,
    pub reachable_proper: bool,
    pub reachable_improper: bool,
    pub rep: SpaceGroupElement,
}

fn spec_to_element(ext: &SpaceGroupExtension, spec: &DefectSpec) -> Result<SpaceGroupElement, String> {
    let iso = match *spec {
        DefectSpec::Dislocation { b, .. } => {
            Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [-b[0], -b[1]] }
        }
        DefectSpec::Disclination { angle_deg, at, .. } => {
            rotation_about(at, angle_deg.to_radians())
        }
        DefectSpec::HalfWall { .. } => {
            return Err("half_wall cannot be converted to a point group element".into());
        }
    };
    ext.from_isometry(&iso)
        .ok_or_else(|| format!("defect spec {spec:?} does not lie in space group {}", ext.group))
}

pub fn fusion_channels_for_elements(
    ext: &SpaceGroupExtension,
    g1: SpaceGroupElement,
    g2: SpaceGroupElement,
) -> Vec<FusionChannel> {
    let mut channels: Vec<FusionChannel> = Vec::new();
    let p_plus_reps: Vec<SpaceGroupElement> = (0..ext.point_group.len())
        .filter(|&i| imat2_det(ext.point_group[i]) == 1)
        .map(|i| ext.coset_rep(i))
        .collect();

    for r_h_idx in 0..ext.point_group.len() {
        let h0 = ext.coset_rep(r_h_idx);
        let is_proper_h = imat2_det(h0.r) > 0;
        let g2c = ext.conjugate(&g2, &h0);
        let prod0 = ext.compose(&g1, &g2c);
        if prod0.r == ID2 {
            let b0 = [-prod0.l_vec[0], -prod0.l_vec[1]];
            let i_minus_r1 =
                [[1 - g1.r[0][0], -g1.r[0][1]], [-g1.r[1][0], 1 - g1.r[1][1]]];
            let mut cands = Vec::new();
            for lx in -5..=5 {
                for ly in -5..=5 {
                    let sh = imat2_vec(i_minus_r1, [lx, ly]);
                    cands.push([b0[0] + sh[0], b0[1] + sh[1]]);
                }
            }
            cands.sort_by_key(|&b| (b[0] * b[0] + b[1] * b[1], b[0], b[1]));
            cands.dedup();
            let has_zero = cands.first() == Some(&[0, 0]);
            let nz_min = cands
                .iter()
                .copied()
                .filter(|&b| b != [0, 0])
                .min_by_key(|b| b[0] * b[0] + b[1] * b[1]);
            let shortest_nz: Vec<[i32; 2]> = match nz_min {
                Some(m) => {
                    let m2 = m[0] * m[0] + m[1] * m[1];
                    cands
                        .iter()
                        .copied()
                        .filter(|&b| b != [0, 0] && b[0] * b[0] + b[1] * b[1] == m2)
                        .collect()
                }
                None => Vec::new(),
            };
            let min_all = cands[0];
            let orb: Vec<[i32; 2]> = p_plus_reps.iter().map(|h| imat2_vec(h.r, min_all)).collect();
            let rep_b = orb.into_iter().max_by_key(|&b| (b[0], b[1])).unwrap_or(min_all);
            let parity = Some((rep_b[0] + rep_b[1]).rem_euclid(2));
            let kind = if has_zero { "vacuum" } else { "dislocation" }.to_string();
            if let Some(ch) = channels.iter_mut().find(|c| {
                c.det_sign == 1 && c.angle_deg == 0.0 && c.burgers_coset_rep == Some(rep_b)
            }) {
                ch.reachable_proper |= is_proper_h;
                ch.reachable_improper |= !is_proper_h;
            } else {
                let rep_elem = ext.element([-rep_b[0], -rep_b[1]], ID2).unwrap();
                channels.push(FusionChannel {
                    kind,
                    angle_deg: 0.0,
                    det_sign: 1,
                    burgers_coset_rep: Some(rep_b),
                    shortest_nonzero_burgers_frac: shortest_nz,
                    parity_mod2: parity,
                    reachable_proper: is_proper_h,
                    reachable_improper: !is_proper_h,
                    rep: rep_elem,
                });
            }
            continue;
        }
        let det_f = imat2_det(prod0.r);
        let rc = ext.point_group_cart[prod0.r_idx];
        let ang = if det_f > 0 { round_dec(rc[1][0].atan2(rc[0][0]).to_degrees(), 6) } else { 0.0 };
        let i_minus_rf =
            [[1 - prod0.r[0][0], -prod0.r[0][1]], [-prod0.r[1][0], 1 - prod0.r[1][1]]];
        for l0 in -2..=2 {
            for l1 in -2..=2 {
                let h = ext.element([l0, l1], h0.r).unwrap();
                let prod = ext.compose(&g1, &ext.conjugate(&g2, &h));
                let exists = channels.iter_mut().find(|ch| {
                    if ch.det_sign != det_f || (ch.angle_deg - ang).abs() > 1e-4 {
                        return false;
                    }
                    p_plus_reps.iter().any(|q| {
                        let c = ext.conjugate(&prod, q);
                        c.r == ch.rep.r
                            && in_image_of_int_mat(
                                i_minus_rf,
                                [ch.rep.l_vec[0] - c.l_vec[0], ch.rep.l_vec[1] - c.l_vec[1]],
                            )
                    })
                });
                if let Some(ch) = exists {
                    ch.reachable_proper |= is_proper_h;
                    ch.reachable_improper |= !is_proper_h;
                } else {
                    channels.push(FusionChannel {
                        kind: if det_f > 0 { "disclination" } else { "improper" }.to_string(),
                        angle_deg: ang,
                        det_sign: det_f,
                        burgers_coset_rep: None,
                        shortest_nonzero_burgers_frac: Vec::new(),
                        parity_mod2: None,
                        reachable_proper: is_proper_h,
                        reachable_improper: !is_proper_h,
                        rep: prod,
                    });
                }
            }
        }
    }
    channels
}

pub fn fusion_channels(
    group: &str,
    cell: &Cell2D,
    specs: &[DefectSpec],
) -> Result<Vec<FusionChannel>, String> {
    if specs.len() != 2 {
        return Err(format!("fusion_channels requires exactly 2 defect specs, got {}", specs.len()));
    }
    let ext = SpaceGroupExtension::from_cell(group, cell)?;
    let g1 = spec_to_element(&ext, &specs[0])?;
    let g2 = spec_to_element(&ext, &specs[1])?;
    Ok(fusion_channels_for_elements(&ext, g1, g2))
}

/// Trace bounded polygon faces on orientable OR non-orientable (`Wc`) crystals via signed darts `(u, v, sign)`.
pub fn ribbon_faces(crystal: &BuiltCrystal) -> Vec<Vec<usize>> {
    let corners = corners_order(crystal);
    let nbr: Vec<Vec<usize>> =
        corners.iter().map(|lst| lst.iter().map(|&(_, v)| v).collect()).collect();
    let idx_map: Vec<HashMap<usize, usize>> = nbr
        .iter()
        .map(|lst| lst.iter().copied().enumerate().map(|(k, v)| (v, k)).collect())
        .collect();
    let det_map: HashMap<(usize, usize), i32> = crystal
        .adj
        .iter()
        .enumerate()
        .flat_map(|(u, row)| {
            row.iter().map(move |e| {
                ((u, e.to), if e.seam.is_none_or(|g| g.det() > 0.0) { 1 } else { -1 })
            })
        })
        .collect();
    let mut seen: HashSet<(usize, usize, i32)> = HashSet::new();
    let mut faces = Vec::new();
    for s0 in [1i32, -1] {
        for (u, lst) in nbr.iter().enumerate() {
            for &v in lst {
                if seen.contains(&(u, v, s0)) {
                    continue;
                }
                let mut f = Vec::new();
                let mut walk_darts = Vec::new();
                let (mut a, mut b, mut s) = (u, v, s0);
                while seen.insert((a, b, s)) {
                    walk_darts.push((a, b, s));
                    f.push(a);
                    let ds = det_map.get(&(a, b)).copied().unwrap_or(1);
                    let s_next = s * ds;
                    let blst = &nbr[b];
                    let Some(&k) = idx_map[b].get(&a) else { break };
                    let next_v = if s_next > 0 {
                        blst[if k == 0 { blst.len() - 1 } else { k - 1 }]
                    } else {
                        blst[(k + 1) % blst.len()]
                    };
                    a = b;
                    b = next_v;
                    s = s_next;
                }
                for (da, db, ds_start) in walk_darts {
                    let ds = det_map.get(&(da, db)).copied().unwrap_or(1);
                    seen.insert((db, da, -ds_start * ds));
                }
                faces.push(f);
            }
        }
    }
    faces
}

/// Gauge-covariant plaquette curvature & torsion observable on one interior face.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaquetteHolonomy {
    pub face_idx: usize,
    pub vertices: Vec<usize>,
    pub w: Isometry2,
    pub omega_deg: f64,
    pub det_sign: i32,
    pub alpha_f: [f64; 2],
    pub centre: Option<[f64; 2]>,
    pub nonsymmorphic_shift: bool,
    pub has_curvature: bool,
    pub has_torsion: bool,
}

pub fn plaquette_holonomies(
    crystal: &BuiltCrystal,
    group: Option<&str>,
    cell: Option<&Cell2D>,
) -> Vec<PlaquetteHolonomy> {
    let walks = ribbon_faces(crystal);
    if walks.is_empty() {
        return Vec::new();
    }
    let k_out = outer_face_index(&walks);
    let ext = cell
        .and_then(|c| SpaceGroupExtension::from_cell(group.unwrap_or(""), c).ok())
        .or_else(|| group.and_then(|g| SpaceGroupExtension::from_group(g).ok()));

    let mut out = Vec::new();
    for (idx, f) in walks.into_iter().enumerate() {
        if idx == k_out {
            continue;
        }
        let (h, alpha_f) = loop_charge(crystal, &f);
        let det_sign = if h.det() > 0.0 { 1 } else { -1 };
        let rot_deg = if det_sign > 0 {
            round_dec(h.m[1][0].atan2(h.m[0][0]).to_degrees(), 6)
        } else {
            0.0
        };
        let is_id_rot = det_sign > 0 && np_allclose_mat(h.m, Isometry2::IDENTITY.m, 1e-5, 1e-6);
        let has_curvature = det_sign < 0 || !is_id_rot;
        let has_torsion = !has_curvature && norm2(alpha_f) > 1e-6;
        let centre = if has_curvature && det_sign > 0 {
            let im = [[1.0 - h.m[0][0], -h.m[0][1]], [-h.m[1][0], 1.0 - h.m[1][1]]];
            Some(mat2_vec(mat2_inv(im), h.t))
        } else {
            None
        };
        let nonsymmorphic_shift = ext.as_ref().is_some_and(|e| {
            e.is_nonsymmorphic
                && e.from_isometry(&h).is_some_and(|el| {
                    e.tau[el.r_idx][0].abs() > 1e-5 || e.tau[el.r_idx][1].abs() > 1e-5
                })
        });
        out.push(PlaquetteHolonomy {
            face_idx: idx,
            vertices: f,
            w: h,
            omega_deg: rot_deg,
            det_sign,
            alpha_f,
            centre,
            nonsymmorphic_shift,
            has_curvature,
            has_torsion,
        });
    }
    out
}

/// Twisted 1-cochain `(d_R m)(u -> v) = R_e m_v - m_u` where `R_e = g_{u->v}^{-1}.m` maps `v`'s frame into `u`'s.
pub fn twisted_coboundary(
    crystal: &BuiltCrystal,
    m_cochain: &[[f64; 2]],
) -> BTreeMap<(usize, usize), [f64; 2]> {
    let mut out = BTreeMap::new();
    for (u, row) in crystal.adj.iter().enumerate() {
        for e in row {
            let r_e = e.seam.map_or([[1.0, 0.0], [0.0, 1.0]], |g| g.inverse().m);
            out.insert((u, e.to), sub2(mat2_vec(r_e, m_cochain[e.to]), m_cochain[u]));
        }
    }
    out
}

/// Evaluate `(d_R^2 m)_f = sum_{e in df} F_e (d_R m)(e)` per interior face (`== 0` whenever `R_f == I`).
pub fn verify_d_r_squared(
    crystal: &BuiltCrystal,
    m_cochain: &[[f64; 2]],
) -> Vec<(usize, bool, [f64; 2])> {
    let omega = twisted_coboundary(crystal, m_cochain);
    let walks = ribbon_faces(crystal);
    if walks.is_empty() {
        return Vec::new();
    }
    let k_out = outer_face_index(&walks);
    let mut out = Vec::new();
    for (fi, f) in walks.iter().enumerate() {
        if fi == k_out {
            continue;
        }
        let mut f_mat = [[1.0, 0.0], [0.0, 1.0]];
        let mut res = [0.0, 0.0];
        for m in 0..f.len() {
            let (u, v) = (f[m], f[(m + 1) % f.len()]);
            let w_e = omega[&(u, v)];
            let step = mat2_vec(f_mat, w_e);
            res[0] += step[0];
            res[1] += step[1];
            let e = crystal.adj[u].iter().find(|b| b.to == v).expect("face bond");
            if let Some(g) = e.seam {
                f_mat = mat2_mul(f_mat, g.inverse().m);
            }
        }
        let is_flat = np_allclose_mat(f_mat, [[1.0, 0.0], [0.0, 1.0]], 1e-5, 1e-6);
        out.push((fi, is_flat, res));
    }
    out
}

/// Apply a local vertex `G`-gauge transformation `h_u` (`e.vec -> R_u e.vec`, `g_{uv} -> h_v g_{uv} h_u^-1`).
pub fn apply_gauge_transform(crystal: &BuiltCrystal, vertex_gauges: &[Isometry2]) -> BuiltCrystal {
    let mut out = crystal.clone();
    for (u, row) in out.adj.iter_mut().enumerate() {
        let hu = vertex_gauges[u];
        let hu_inv = hu.inverse();
        for e in row {
            let hv = vertex_gauges[e.to];
            e.vec = hu.apply_linear(e.vec);
            let g_old = e.seam.unwrap_or(Isometry2::IDENTITY);
            let g_new = hv.compose(&g_old.compose(&hu_inv));
            let is_id = np_allclose_mat(g_new.m, [[1.0, 0.0], [0.0, 1.0]], 1e-6, 1e-7)
                && norm2(g_new.t) < 1e-7;
            e.seam = (!is_id).then_some(g_new);
            if !out.cone_sites.contains(&u) {
                e.ang = None;
            } else if let Some(a) = e.ang {
                let v = hu.apply_linear([a.cos(), a.sin()]);
                e.ang = Some(v[1].atan2(v[0]).rem_euclid(std::f64::consts::TAU));
            }
        }
    }
    out.is_non_orientable = crystal.is_non_orientable;
    out
}
