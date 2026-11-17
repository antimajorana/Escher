//! Homotopy defect catalogue (`DefectCatalogue`): dislocations, rotation-centre `G_+`-classes,
//! disclinations, and symmetry-breaking walls, matching `papertilling/defects/catalogue.py`.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

use super::sym_helper::{
    LatticeGroup, dot, gcd_i32, imat2_det, imat2_vec, mat2_det, mat2_vec, norm, np_allclose_mat,
    orbits, py_mod, rotation_about, round_dec, round_even, solve2,
};
use super::wall_build::{build_interface, build_mobius_seam, build_wall};
use super::walls::{
    DomainWallEntry, GlideBreakingEntry, MirrorLineEntry, MobiusSeamEntry, NormaliserInfo,
    TwinWallEntry, compute_all_lines, compute_domain_walls, compute_glide_breaking,
    compute_mirrors, compute_mobius_seams, compute_normaliser, compute_twin_walls,
};

const GENERIC_POINT: [f64; 2] = [0.371, 0.413];
const T13: f64 = 1.0 / 3.0;
const T23: f64 = 2.0 / 3.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DislocationEntry {
    pub id: String,
    pub b: [f64; 2],
    pub b_frac: [i32; 2],
    pub length: f64,
    pub angle: f64,
    pub size: usize,
    pub self_inverse: bool,
    pub inverse_id: String,
    pub mirror_id: Option<String>,
    pub at: [f64; 2],
    pub holonomy: Isometry2,
    pub members: Vec<[i32; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CentreClassEntry {
    pub id: String,
    pub order: usize,
    pub gclass: String,
    pub label: String,
    pub location: String,
    pub mirror_of: Option<String>,
    pub point: [f64; 2],
    pub frac: [f64; 2],
    pub size: usize,
    pub split: bool,
    pub angles: Vec<f64>,
    pub(crate) members: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisclinationEntry {
    pub id: String,
    pub centre_class: String,
    pub wyckoff: String,
    pub gclass: String,
    pub order: usize,
    pub angle: f64,
    pub location: String,
    pub point: [f64; 2],
    pub inverse_id: String,
    pub mirror_id: Option<String>,
    pub holonomy: Isometry2,
}

#[derive(Clone, Debug)]
pub struct DefectCatalogue {
    pub group: String,
    pub chiral: bool,
    pub cell: Cell2D,
    pub r0: f64,
    pub b_min: f64,
    pub dislocations: Vec<DislocationEntry>,
    pub centre_classes: Vec<CentreClassEntry>,
    pub disclinations: Vec<DisclinationEntry>,
    pub mirrors: Vec<MirrorLineEntry>,
    pub normaliser: NormaliserInfo,
    pub domain_walls: Vec<DomainWallEntry>,
    pub twin_walls: Vec<TwinWallEntry>,
    pub mobius_seams: Vec<MobiusSeamEntry>,
    pub glide_breaking: Vec<GlideBreakingEntry>,
    pub(crate) grp: LatticeGroup,
    all_centres: Vec<([f64; 2], usize)>,
}

struct ItaCentreSpec {
    name: &'static str,
    gclass: &'static str,
    order: usize,
    points: &'static [[f64; 2]],
}

fn proper_centre_classes(group: &str) -> &'static [ItaCentreSpec] {
    const P2_PMM: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "1a", gclass: "1a", order: 2, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "1b", gclass: "1b", order: 2, points: &[[0.0, 0.5]] },
        ItaCentreSpec { name: "1c", gclass: "1c", order: 2, points: &[[0.5, 0.0]] },
        ItaCentreSpec { name: "1d", gclass: "1d", order: 2, points: &[[0.5, 0.5]] },
    ];
    const PMG: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "2a(i)", gclass: "2a", order: 2, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "2a(ii)", gclass: "2a", order: 2, points: &[[0.5, 0.0]] },
        ItaCentreSpec { name: "2b(i)", gclass: "2b", order: 2, points: &[[0.0, 0.5]] },
        ItaCentreSpec { name: "2b(ii)", gclass: "2b", order: 2, points: &[[0.5, 0.5]] },
    ];
    const PGG: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "2a(i)", gclass: "2a", order: 2, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "2a(ii)", gclass: "2a", order: 2, points: &[[0.5, 0.5]] },
        ItaCentreSpec { name: "2b(i)", gclass: "2b", order: 2, points: &[[0.0, 0.5]] },
        ItaCentreSpec { name: "2b(ii)", gclass: "2b", order: 2, points: &[[0.5, 0.0]] },
    ];
    const CMM: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "2a", gclass: "2a", order: 2, points: &[[0.0, 0.0], [0.5, 0.5]] },
        ItaCentreSpec { name: "2b", gclass: "2b", order: 2, points: &[[0.0, 0.5], [0.5, 0.0]] },
        ItaCentreSpec { name: "4c(i)", gclass: "4c", order: 2, points: &[[0.25, 0.25], [0.75, 0.75]] },
        ItaCentreSpec { name: "4c(ii)", gclass: "4c", order: 2, points: &[[0.75, 0.25], [0.25, 0.75]] },
    ];
    const P4_P4M: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "1a", gclass: "1a", order: 4, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "1b", gclass: "1b", order: 4, points: &[[0.5, 0.5]] },
        ItaCentreSpec { name: "2c", gclass: "2c", order: 2, points: &[[0.5, 0.0], [0.0, 0.5]] },
    ];
    const P4G: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "2a(i)", gclass: "2a", order: 4, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "2a(ii)", gclass: "2a", order: 4, points: &[[0.5, 0.5]] },
        ItaCentreSpec { name: "2b", gclass: "2b", order: 2, points: &[[0.5, 0.0], [0.0, 0.5]] },
    ];
    const P3_P3M1: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "1a", gclass: "1a", order: 3, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "1b", gclass: "1b", order: 3, points: &[[T13, T23]] },
        ItaCentreSpec { name: "1c", gclass: "1c", order: 3, points: &[[T23, T13]] },
    ];
    const P31M: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "1a", gclass: "1a", order: 3, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "2b(i)", gclass: "2b", order: 3, points: &[[T13, T23]] },
        ItaCentreSpec { name: "2b(ii)", gclass: "2b", order: 3, points: &[[T23, T13]] },
    ];
    const P6_P6M: &[ItaCentreSpec] = &[
        ItaCentreSpec { name: "1a", gclass: "1a", order: 6, points: &[[0.0, 0.0]] },
        ItaCentreSpec { name: "2b", gclass: "2b", order: 3, points: &[[T13, T23], [T23, T13]] },
        ItaCentreSpec { name: "3c", gclass: "3c", order: 2, points: &[[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]] },
    ];
    match group {
        "p2" | "pmm" => P2_PMM,
        "pmg" => PMG,
        "pgg" => PGG,
        "cmm" => CMM,
        "p4" | "p4m" => P4_P4M,
        "p4g" => P4G,
        "p3" | "p3m1" => P3_P3M1,
        "p31m" => P31M,
        "p6" | "p6m" => P6_P6M,
        _ => &[],
    }
}

fn location_in_group(grp: &LatticeGroup, all_centres: &[([f64; 2], usize)], c: [f64; 2]) -> &'static str {
    if grp.l.locate(c).is_some() {
        return "site";
    }
    let fc = mat2_vec(grp.ainv, c);
    let n0 = [round_even(fc[0]) as i32, round_even(fc[1]) as i32];
    let k = (grp.lmax / norm(grp.cell.a1).min(norm(grp.cell.a2))).ceil() as i32 + 2;
    for d1 in -k..=k {
        for d2 in -k..=k {
            let o = mat2_vec(grp.a, [(n0[0] + d1) as f64, (n0[1] + d2) as f64]);
            for bd in &grp.cell.bonds {
                let p = [o[0] + grp.cell.basis[bd.i][0], o[1] + grp.cell.basis[bd.i][1]];
                let sh = mat2_vec(grp.a, [bd.n1 as f64, bd.n2 as f64]);
                let q = [o[0] + grp.cell.basis[bd.j][0] + sh[0], o[1] + grp.cell.basis[bd.j][1] + sh[1]];
                let dv = [q[0] - p[0], q[1] - p[1]];
                let s = dot([c[0] - p[0], c[1] - p[1]], dv) / dot(dv, dv);
                if (-1e-9..(1.0 + 1e-9)).contains(&s) && norm([p[0] + s * dv[0] - c[0], p[1] + s * dv[1] - c[1]]) < 1e-6 {
                    return if (s - 0.5).abs() < 1e-6 { "bond midpoint" } else { "on a bond" };
                }
            }
        }
    }
    if !grp.planar {
        "elsewhere"
    } else if all_centres.iter().any(|&(q, _)| grp.same_point(c, q)) {
        "face centre"
    } else {
        "inside a face"
    }
}

fn primitive_vectors(grp: &LatticeGroup, b_min: f64, factor: f64) -> Vec<[i32; 2]> {
    let lim = factor * b_min + 1e-6;
    let k = (lim * grp.scale / mat2_det(grp.a).abs()).floor() as i32 + 1;
    let mut out = Vec::new();
    for m0 in -k..=k {
        for m1 in -k..=k {
            if (m0, m1) != (0, 0) && gcd_i32(m0, m1) == 1 && norm(mat2_vec(grp.a, [m0 as f64, m1 as f64])) <= lim {
                out.push([m0, m1]);
            }
        }
    }
    out
}

fn compute_dislocations(grp: &LatticeGroup, prim_vecs: &[[i32; 2]]) -> Vec<DislocationEntry> {
    let p_plus: Vec<[[i32; 2]; 2]> = grp.opsf.iter().filter(|op| imat2_det(op.mf) == 1).map(|op| op.mf).collect();
    let orbs = orbits(prim_vecs, &p_plus, |&m, &mf| imat2_vec(mf, m), |a, b| a == b);
    let ang = |m: [i32; 2]| -> f64 {
        let v = mat2_vec(grp.a, [m[0] as f64, m[1] as f64]);
        py_mod(round_dec(py_mod(v[1].atan2(v[0]).to_degrees(), 360.0), 6), 360.0)
    };
    type DislocRec = ([i32; 2], [f64; 2], f64, f64, Vec<[i32; 2]>);
    let mut recs: Vec<DislocRec> = orbs
        .into_iter()
        .map(|o| {
            let mut mem: Vec<[i32; 2]> = o.into_iter().map(|i| prim_vecs[i]).collect();
            mem.sort_by(|a, b| ang(*a).total_cmp(&ang(*b)));
            let rep = mem[0];
            let b = mat2_vec(grp.a, [rep[0] as f64, rep[1] as f64]);
            (rep, b, norm(b), ang(rep), mem)
        })
        .collect();
    recs.sort_by(|a, b| round_dec(a.2, 6).total_cmp(&round_dec(b.2, 6)).then_with(|| a.3.total_cmp(&b.3)));

    let mut where_map: HashMap<[i32; 2], usize> = HashMap::new();
    for (k, r) in recs.iter().enumerate() {
        for &m in &r.4 {
            where_map.insert(m, k);
        }
    }
    let m0 = grp.opsf.iter().find(|op| imat2_det(op.mf) == -1).map(|op| op.mf);
    let at = mat2_vec(grp.a, GENERIC_POINT);
    recs.into_iter()
        .enumerate()
        .map(|(k, (rep, b, length, angle, mem))| {
            let inv_k = where_map[&[-rep[0], -rep[1]]];
            let mirror_id = m0.and_then(|mf| {
                let mv = imat2_vec(mf, rep);
                where_map.get(&[-mv[0], -mv[1]]).map(|&mk| format!("T{}", mk + 1))
            });
            DislocationEntry {
                id: format!("T{}", k + 1),
                b,
                b_frac: rep,
                length,
                angle,
                size: mem.len(),
                self_inverse: inv_k == k,
                inverse_id: format!("T{}", inv_k + 1),
                mirror_id,
                at,
                holonomy: Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [-b[0], -b[1]] },
                members: mem,
            }
        })
        .collect()
}

fn compute_all_centres(grp: &LatticeGroup) -> Vec<([f64; 2], usize)> {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for g in &grp.plus {
        if np_allclose_mat(g.m, Isometry2::IDENTITY.m, 1e-5, 1e-8) {
            continue;
        }
        let im = [[1.0 - g.m[0][0], -g.m[0][1]], [-g.m[1][0], 1.0 - g.m[1][1]]];
        for m0 in -2..=2 {
            for m1 in -2..=2 {
                let sh = mat2_vec(grp.a, [m0 as f64, m1 as f64]);
                let c = grp.reduce(solve2(im, [g.t[0] + sh[0], g.t[1] + sh[1]]));
                if !pts.iter().any(|&q| grp.same_point(c, q)) {
                    pts.push(c);
                }
            }
        }
    }
    pts.into_iter()
        .map(|c| (c, grp.plus.iter().filter(|g| grp.same_point(g.apply(c), c)).count()))
        .collect()
}

fn compute_centre_classes(
    grp: &LatticeGroup,
    all_centres: &[([f64; 2], usize)],
    input_group: &str,
    input_cell: &Cell2D,
    frame: Frame2D,
) -> Vec<CentreClassEntry> {
    if all_centres.is_empty() {
        return Vec::new();
    }
    let pts: Vec<[f64; 2]> = all_centres.iter().map(|&(c, _)| c).collect();
    let n_of: Vec<usize> = all_centres.iter().map(|&(_, n)| n).collect();
    let plus_orbs = orbits(&pts, &grp.plus, |&c, h| grp.reduce(h.apply(c)), |&im, &q| grp.same_point(im, q));
    let full_orbs = orbits(&pts, &grp.ops, |&c, h| grp.reduce(h.apply(c)), |&im, &q| grp.same_point(im, q));
    let mut gid = vec![0usize; pts.len()];
    for (k, o) in full_orbs.iter().enumerate() {
        for &i in o {
            gid[i] = k;
        }
    }
    let frac_of = |c: [f64; 2]| -> [f64; 2] {
        let f = mat2_vec(grp.ainv, c);
        [py_mod(round_dec(f[0], 4), 1.0), py_mod(round_dec(f[1], 4), 1.0)]
    };

    struct RawRec {
        point: [f64; 2],
        frac: [f64; 2],
        order: usize,
        members: Vec<[f64; 2]>,
        g_index: usize,
        ita_xy: Option<[f64; 2]>,
    }

    let mut recs: Vec<RawRec> = plus_orbs
        .into_iter()
        .map(|o| {
            let rep = o
                .iter()
                .map(|&i| pts[i])
                .min_by(|&a, &b| {
                    let (fa, fb) = (frac_of(a), frac_of(b));
                    fa[0].total_cmp(&fb[0]).then_with(|| fa[1].total_cmp(&fb[1]))
                })
                .expect("non-empty centre orbit");
            RawRec {
                point: rep,
                frac: frac_of(rep),
                order: n_of[o[0]],
                members: o.iter().map(|&i| pts[i]).collect(),
                g_index: gid[o[0]],
                ita_xy: LatticeGroup::ita_frac(input_group, input_cell, frame, rep),
            }
        })
        .collect();
    recs.sort_by(|a, b| (b.order as i32).cmp(&(a.order as i32)).then_with(|| a.frac[0].total_cmp(&b.frac[0])).then_with(|| a.frac[1].total_cmp(&b.frac[1])));

    let table = proper_centre_classes(grp.name);
    let match_rec = |xy: [f64; 2], want_ord: Option<usize>, tol: f64| -> Option<&'static ItaCentreSpec> {
        let hits: Vec<&ItaCentreSpec> = table
            .iter()
            .filter(|rec| want_ord.is_none_or(|o| rec.order == o))
            .filter(|rec| rec.points.iter().any(|&q| (xy[0] - q[0] - round_even(xy[0] - q[0])).abs() < tol && (xy[1] - q[1] - round_even(xy[1] - q[1])).abs() < tol))
            .collect();
        if want_ord.is_some() {
            hits.first().copied()
        } else {
            let g0 = hits.first()?.gclass;
            hits.iter().all(|h| h.gclass == g0).then_some(hits[0])
        }
    };

    let (mut glabel, mut letters, mut count) = (HashMap::new(), HashMap::new(), HashMap::new());
    for r in &recs {
        *count.entry(r.g_index).or_insert(0usize) += 1;
        glabel.entry(r.g_index).or_insert_with(|| {
            r.ita_xy
                .and_then(|xy| match_rec(xy, None, 1e-6).map(|rec| rec.gclass.to_string()))
                .unwrap_or_else(|| {
                    let idx = *letters.entry(r.order).and_modify(|v| *v += 1).or_insert(0usize);
                    format!("order-{} class {}", r.order, b"ABCDEFGH"[idx] as char)
                })
        });
    }
    let names: Vec<Option<&'static str>> = recs.iter().map(|r| r.ita_xy.and_then(|xy| match_rec(xy, Some(r.order), 1e-5).map(|rec| rec.name))).collect();
    let use_ita = names.iter().all(Option::is_some) && {
        let mut s = BTreeSet::new();
        names.iter().all(|n| s.insert(n.unwrap_or("")))
    };

    let mut seen: HashMap<usize, usize> = HashMap::new();
    let mut entries: Vec<CentreClassEntry> = recs
        .into_iter()
        .enumerate()
        .map(|(k, r)| {
            let gclass = glabel[&r.g_index].clone();
            let sn = *seen.entry(r.g_index).and_modify(|v| *v += 1).or_insert(1);
            let cnt = count[&r.g_index];
            let label = if use_ita { names[k].unwrap().to_string() } else if cnt == 1 { gclass.clone() } else { format!("{gclass}/{sn}") };
            let ord_i = r.order as i32;
            let angles: Vec<f64> = (-ord_i..=ord_i)
                .filter(|&k2| k2 != 0 && ((k2 as f64) * 360.0 / (r.order as f64)).abs() <= 180.0 + 1e-9)
                .map(|k2| (k2 as f64) * 360.0 / (r.order as f64))
                .collect();
            CentreClassEntry {
                id: format!("C{}", k + 1),
                order: r.order,
                gclass,
                label,
                location: location_in_group(grp, all_centres, r.point).to_string(),
                mirror_of: None,
                point: r.point,
                frac: r.frac,
                size: r.members.len(),
                split: cnt > 1,
                angles,
                members: r.members,
            }
        })
        .collect();

    if let Some(m0) = grp.minus.first() {
        let mids: Vec<Option<String>> = entries
            .iter()
            .map(|r| {
                let img = m0.apply(r.point);
                entries.iter().find(|c| c.members.iter().any(|&q| grp.same_point(img, q))).map(|c| c.id.clone())
            })
            .collect();
        for (e, mid) in entries.iter_mut().zip(mids) {
            e.mirror_of = mid;
        }
    }
    entries
}

fn compute_disclinations(centres: &[CentreClassEntry]) -> Vec<DisclinationEntry> {
    let by_id: HashMap<&str, &CentreClassEntry> = centres.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut out = Vec::new();
    for r in centres {
        for &a in &r.angles {
            out.push(DisclinationEntry {
                id: format!("D({}, {:+.0})", r.label, a),
                centre_class: r.id.clone(),
                wyckoff: r.label.clone(),
                gclass: r.gclass.clone(),
                order: r.order,
                angle: a,
                location: r.location.clone(),
                point: r.point,
                inverse_id: format!("D({}, {:+.0})", r.label, -a),
                mirror_id: r.mirror_of.as_deref().map(|mid| format!("D({}, {:+.0})", by_id[mid].label, a)),
                holonomy: rotation_about(r.point, a.to_radians()),
            });
        }
    }
    out
}

impl DefectCatalogue {
    pub fn new(group: &str, cell: &Cell2D, frame: Frame2D) -> Self {
        Self::with_bounds(group, cell, frame, 2.0, 2.0)
    }

    pub fn with_bounds(group: &str, cell: &Cell2D, frame: Frame2D, bmax: f64, wall_bmax: f64) -> Self {
        let grp = LatticeGroup::new(cell);
        let r0 = 3.5 * grp.scale;
        let mut b_min = f64::INFINITY;
        for m0 in -3..=3 {
            for m1 in -3..=3 {
                if (m0, m1) != (0, 0) {
                    b_min = b_min.min(norm(mat2_vec(grp.a, [m0 as f64, m1 as f64])));
                }
            }
        }
        let dislocations = compute_dislocations(&grp, &primitive_vectors(&grp, b_min, bmax));
        let all_centres = compute_all_centres(&grp);
        let centre_classes = compute_centre_classes(&grp, &all_centres, group, cell, frame);
        let disclinations = compute_disclinations(&centre_classes);
        let all_lines = compute_all_lines(&grp);
        let mirrors = compute_mirrors(&grp, &all_lines);
        let normaliser = compute_normaliser(&grp, &centre_classes, |pt| location_in_group(&grp, &all_centres, pt));
        let domain_walls = compute_domain_walls(&grp, &normaliser);
        let (twin_dirs, twin_walls) = compute_twin_walls(&grp, &all_lines, &normaliser, &primitive_vectors(&grp, b_min, wall_bmax), 4);
        let mobius_seams = compute_mobius_seams(&grp, &mirrors, 4);
        let glide_breaking = compute_glide_breaking(&grp, &mirrors, &normaliser, &twin_dirs, &twin_walls);

        Self {
            group: grp.name.to_string(),
            chiral: grp.chiral,
            cell: grp.cell.clone(),
            r0,
            b_min,
            dislocations,
            centre_classes,
            disclinations,
            mirrors,
            normaliser,
            domain_walls,
            twin_walls,
            mobius_seams,
            glide_breaking,
            grp,
            all_centres,
        }
    }

    pub fn location(&self, c: [f64; 2]) -> &'static str {
        location_in_group(&self.grp, &self.all_centres, c)
    }

    pub fn centre_class_of(&self, c: [f64; 2]) -> Option<&CentreClassEntry> {
        self.centre_classes.iter().find(|r| r.members.iter().any(|&q| self.grp.same_point(c, q)))
    }

    pub fn check_disclination(&self, angle_deg: f64, at: [f64; 2]) -> Result<&CentreClassEntry, String> {
        let Some(r) = self.centre_class_of(at) else {
            return Err(format!("[{}, {}] is not a rotation centre of {}", round_dec(at[0], 4), round_dec(at[1], 4), self.group));
        };
        let k = angle_deg * (r.order as f64) / 360.0;
        if angle_deg.abs() < 1e-9 || (k - round_even(k)).abs() > 1e-6 {
            return Err(format!("angle {angle_deg} is not allowed at an order-{} centre", r.order));
        }
        if !self.grp.contains(rotation_about(at, angle_deg.to_radians()), 1e-5) {
            return Err("rotation not in G".to_string());
        }
        Ok(r)
    }

    pub fn build_wall(&self, id: &str, r: Option<f64>) -> Option<BuiltCrystal> {
        let rad = r.unwrap_or(self.r0);
        if let Some(idx) = id.strip_prefix("Wa").and_then(|s| s.parse::<usize>().ok()) {
            let w = self.domain_walls.get(idx.checked_sub(1)?)?;
            return Some(build_interface(&self.cell, w.sigma, w.point, w.direction, rad));
        }
        if let Some(idx) = id.strip_prefix("Wb").and_then(|s| s.parse::<usize>().ok()) {
            let w = self.twin_walls.get(idx.checked_sub(1)?)?;
            return Some(build_wall(&self.cell, w.point, w.direction, 0.0, rad));
        }
        if let Some(idx) = id.strip_prefix("Wc").or_else(|| id.strip_prefix('M')).and_then(|s| s.parse::<usize>().ok()) {
            let w = self.mobius_seams.get(idx.checked_sub(1)?)?;
            return Some(build_mobius_seam(&self.cell, w.sigma, w.length, w.width));
        }
        if let Some(idx) = id.strip_prefix("Wd").and_then(|s| s.parse::<usize>().ok()) {
            let w = self.glide_breaking.get(idx.checked_sub(1)?)?;
            return Some(build_wall(&self.cell, w.point, w.direction, w.glide, rad));
        }
        None
    }

    /// Interactive wall builder that also scales Möbius `Wc` ribbon length/width with `radius`.
    pub fn build_wall_scaled(&self, id: &str, radius: f64) -> Option<BuiltCrystal> {
        if let Some(idx) = id.strip_prefix("Wc").or_else(|| id.strip_prefix('M')).and_then(|s| s.parse::<usize>().ok()) {
            let w = self.mobius_seams.get(idx.checked_sub(1)?)?;
            let period = norm(w.sigma.t).max(0.5 * self.grp.scale).max(1e-6);
            let len = ((2.2 * radius / period).round() as usize).clamp(w.length, 24);
            let wid = (0.85 * radius).max(w.width);
            return Some(build_mobius_seam(&self.cell, w.sigma, len, wid));
        }
        self.build_wall(id, Some(radius))
    }

    /// Build a wall crystal and produce its planar picture (`DefectEntry.picture(R)` in Python:
    /// Möbius seams keep stubs with `None` info; glide-breaking `Wd` walls allow local repair;
    /// `Wa`/`Wb` walls open rigidly or fall back to `"rigid"` with an explicit diagnostic note).
    pub fn wall_picture(
        &self,
        id: &str,
        r: Option<f64>,
    ) -> Option<(BuiltCrystal, Option<crate::plane::draw::DrawingInfo>)> {
        let crystal = self.build_wall(id, r)?;
        if crystal.is_non_orientable {
            return Some((crystal, None));
        }
        let wall_local = id.starts_with("Wd");
        Some(crate::plane::draw::draw_wall_picture_with_options(&crystal, wall_local))
    }

    pub fn wall_picture_scaled(
        &self,
        id: &str,
        radius: f64,
    ) -> Option<(BuiltCrystal, Option<crate::plane::draw::DrawingInfo>)> {
        let crystal = self.build_wall_scaled(id, radius)?;
        if crystal.is_non_orientable {
            return Some((crystal, None));
        }
        let wall_local = id.starts_with("Wd");
        Some(crate::plane::draw::draw_wall_picture_with_options(&crystal, wall_local))
    }
}
