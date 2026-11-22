//! Enumeration of defect pairs and triples, one representative per proper symmetry (`G_+`) orbit.
//!
//! Every mode applies exact `G_+` symmetry reduction (plus slot-swap for identical species):
//! - Disclination pairs: exact `G_+` + swap reduction (every core sits on an exact rotation centre).
//! - Dislocation pairs: exact proper point-group `P_+` + swap reduction; each kept representative sits at
//!   `core_a = ta.at` (generic point `A * (0.371, 0.413)`), while dropped `g in G_+` rotated copies sit at
//!   the co-rotated anchor `g(core_a)`, so every merge is an exact symmetry of the lattice.
//! - Mixed pairs: exact `Stab^+(p_a)` reduction around each disclination centre `p_a`; dropped `g` copies
//!   carry the co-rotated dislocation core nudge `g(nudge)`.
//! - Triples: exact `G_+ x S_3` reduction with the same co-rotated generic offsets for dislocation cores.

use std::collections::{HashMap, HashSet};

use crate::defect::catalogue::{CentreClassEntry, DefectCatalogue, DisclinationEntry, DislocationEntry};
use crate::defect::combinations_types::{CombinationCandidate, CombinationMode};
use crate::defect::cone_geometry::{add2, norm2, sub2};
use crate::defect::surgery::DefectSpec;
use crate::defect::sym_helper::{
    LatticeGroup, imat2_det, imat2_vec, mat2_det, mat2_vec, norm, round_dec,
};

/// Small generic fractional offset used to place a dislocation core off sites/bonds relative to a disclination.
const MIXED_CORE_NUDGE_FRAC: [f64; 2] = [0.113, 0.037];

const PERMS_2: [[usize; 2]; 2] = [[0, 1], [1, 0]];
const PERMS_3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[inline]
fn clean0(x: f64) -> f64 {
    if x.abs() < 1e-9 { 0.0 } else { x }
}

#[inline]
fn q2(v: [f64; 2]) -> [i64; 2] {
    [(v[0] * 1e5).round_ties_even() as i64, (v[1] * 1e5).round_ties_even() as i64]
}

fn lattice_span(grp: &LatticeGroup, radius: f64) -> i32 {
    let h = mat2_det(grp.a).abs() / norm(grp.cell.a1).max(norm(grp.cell.a2)).max(1e-12);
    (radius / h.max(1e-12)).ceil() as i32 + 2
}

/// Sort key preferring shorter separation, then positive-y / positive-x quadrant representatives.
fn disp_sort_key(d: [f64; 2], sep: f64) -> (i64, bool, bool, i64, i64) {
    let r_sep = (round_dec(sep, 6) * 1e6).round_ties_even() as i64;
    let q = q2(d);
    (r_sep, d[1] < -1e-6, d[0] < -1e-6, q[0], q[1])
}

/// Enumerate all non-zero lattice vectors `A * [m0, m1]` with length `<= max_sep`.
fn lattice_displacements(grp: &LatticeGroup, max_sep: f64) -> Vec<([i32; 2], [f64; 2], f64)> {
    let span = lattice_span(grp, max_sep);
    let mut out = Vec::new();
    for m0 in -span..=span {
        for m1 in -span..=span {
            if (m0, m1) == (0, 0) {
                continue;
            }
            let d = mat2_vec(grp.a, [m0 as f64, m1 as f64]);
            let sep = norm2(d);
            if sep > 1e-6 && sep <= max_sep + 1e-6 {
                out.push(([m0, m1], d, sep));
            }
        }
    }
    out.sort_by_key(|&(_, d, sep)| disp_sort_key(d, sep));
    out
}

/// Enumerate all points in `centre_class` within `(0, max_sep]` of `anchor`.
fn centre_points_near(
    grp: &LatticeGroup,
    cls: &CentreClassEntry,
    anchor: [f64; 2],
    max_sep: f64,
) -> Vec<([f64; 2], [f64; 2], f64)> {
    let span = lattice_span(grp, max_sep);
    let f_anc = mat2_vec(grp.ainv, anchor);
    let base = [f_anc[0].floor() as i32, f_anc[1].floor() as i32];
    let mut out = Vec::new();
    for &q in &cls.members {
        for m0 in (base[0] - span)..=(base[0] + span) {
            for m1 in (base[1] - span)..=(base[1] + span) {
                let sh = mat2_vec(grp.a, [m0 as f64, m1 as f64]);
                let p = add2(q, sh);
                let d = sub2(p, anchor);
                let sep = norm2(d);
                if sep > 1e-6 && sep <= max_sep + 1e-6 {
                    out.push((p, d, sep));
                }
            }
        }
    }
    out.sort_by_key(|&(_, d, sep)| disp_sort_key(d, sep));
    out
}

/// Proper point-group matrices `M_f` of elements in `G_+` stabilising `point` modulo lattice translations.
fn stabiliser_mats(grp: &LatticeGroup, point: [f64; 2]) -> Vec<[[i32; 2]; 2]> {
    grp.opsf
        .iter()
        .zip(&grp.ops)
        .filter(|(op, iso)| imat2_det(op.mf) == 1 && grp.same_point(iso.apply(point), point))
        .map(|(op, _)| op.mf)
        .collect()
}

/// Guard against `surgery.rs` coalescing two distinct dislocation cores whose cut origins `at = core - 0.5*b` coincide.
fn anti_coalesce_at(mut at: [f64; 2], sep_dir: [f64; 2], earlier_ats: &[[f64; 2]]) -> [f64; 2] {
    let s = norm2(sep_dir);
    if s <= 1e-6 {
        return at;
    }
    let u = [sep_dir[0] / s, sep_dir[1] / s];
    let mut step = 1.0;
    while earlier_ats.iter().any(|&prev| norm2(sub2(at, prev)) <= 1e-5) {
        at = add2(at, [1e-4 * step * u[0], 1e-4 * step * u[1]]);
        step += 1.0;
    }
    at
}

type DisclPairKey = ((usize, [i64; 2]), (usize, [i64; 2]));

fn canon_discl_pair(
    grp: &LatticeGroup,
    ids: [usize; 2],
    pts: [[f64; 2]; 2],
) -> DisclPairKey {
    let mut best: Option<DisclPairKey> = None;
    for g in &grp.plus {
        for pi in PERMS_2 {
            let p0 = grp.reduce(g.apply(pts[pi[0]]));
            let f0 = q2(mat2_vec(grp.ainv, p0));
            let d1 = g.apply_linear(sub2(pts[pi[1]], pts[pi[0]]));
            let fd1 = q2(mat2_vec(grp.ainv, d1));
            let key = ((ids[pi[0]], f0), (ids[pi[1]], fd1));
            if best.is_none_or(|b| key < b) {
                best = Some(key);
            }
        }
    }
    best.expect("G_+ is non-empty")
}

fn gen_disclination_pairs(cat: &DefectCatalogue, max_sep: f64) -> Vec<CombinationCandidate> {
    let cls_by_id: HashMap<&str, &CentreClassEntry> =
        cat.centre_classes.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut out = Vec::new();
    let mut seen: HashSet<DisclPairKey> = HashSet::new();
    for i in 0..cat.disclinations.len() {
        let da = &cat.disclinations[i];
        let p0 = da.point;
        for j in i..cat.disclinations.len() {
            let db = &cat.disclinations[j];
            let Some(&cj) = cls_by_id.get(db.centre_class.as_str()) else { continue };
            for (p1, d, sep) in centre_points_near(&cat.grp, cj, p0, max_sep) {
                let key = canon_discl_pair(&cat.grp, [i, j], [p0, p1]);
                if !seen.insert(key) {
                    continue;
                }
                let label = format!("{} + {} @ ({:.2}, {:.2})", da.id, db.id, clean0(d[0]), clean0(d[1]));
                out.push(CombinationCandidate {
                    label,
                    defect_ids: vec![da.id.clone(), db.id.clone()],
                    separation: sep,
                    displacements: vec![d],
                    specs: vec![
                        DefectSpec::disclination(da.angle, p0),
                        DefectSpec::disclination(db.angle, p1),
                    ],
                });
            }
        }
    }
    out
}

type DislocPairKey = ([i32; 2], [i32; 2], [i32; 2]);

/// Proper point-group matrices `M_f in P_+` (`det(M_f) = +1`).
fn proper_point_mats(grp: &LatticeGroup) -> Vec<[[i32; 2]; 2]> {
    grp.opsf.iter().filter(|op| imat2_det(op.mf) == 1).map(|op| op.mf).collect()
}

/// Exact `P_+` (+ swap when `allow_swap`) canonical key for a dislocation pair `(ba, bb, df)`.
///
/// Each kept representative sits at `core_a = ta.at`; its dropped `g in G_+` rotated copies sit at the
/// co-rotated anchor `g(core_a)`, so every merge is an exact proper symmetry of the lattice.
fn canon_disloc_pair(
    p_plus: &[[[i32; 2]; 2]],
    ba: [i32; 2],
    bb: [i32; 2],
    df: [i32; 2],
    allow_swap: bool,
) -> DislocPairKey {
    let mut best: Option<DislocPairKey> = None;
    for &mf in p_plus {
        let k0 = (imat2_vec(mf, ba), imat2_vec(mf, bb), imat2_vec(mf, df));
        let k = if allow_swap {
            let k1 = (imat2_vec(mf, bb), imat2_vec(mf, ba), imat2_vec(mf, [-df[0], -df[1]]));
            k0.min(k1)
        } else {
            k0
        };
        if best.is_none_or(|b| k < b) {
            best = Some(k);
        }
    }
    best.expect("P_+ is non-empty")
}

/// Enumerate dislocation pairs modulo proper point-group rotations `P_+` and swap (`i == j`).
///
/// Because `P_+` acts transitively (and freely) on `ta.members`, fixing `b_a = ta.b` at `core_a = ta.at`
/// hits every `P_+`-orbit; dropped rotated copies sit at the co-rotated anchor `g(core_a)`.
fn gen_dislocation_pairs(cat: &DefectCatalogue, max_sep: f64) -> Vec<CombinationCandidate> {
    let disps = lattice_displacements(&cat.grp, max_sep);
    let p_plus = proper_point_mats(&cat.grp);
    let mut out = Vec::new();
    let mut seen: HashSet<DislocPairKey> = HashSet::new();
    for i in 0..cat.dislocations.len() {
        let ta = &cat.dislocations[i];
        let ba_f = ta.b_frac;
        let ba = ta.b;
        let core_a = ta.at;
        let at_a = sub2(core_a, [0.5 * ba[0], 0.5 * ba[1]]);
        for j in i..cat.dislocations.len() {
            let tb = &cat.dislocations[j];
            for &bb_f in &tb.members {
                let bb = mat2_vec(cat.grp.a, [bb_f[0] as f64, bb_f[1] as f64]);
                for &(df, d, sep) in &disps {
                    let key = canon_disloc_pair(&p_plus, ba_f, bb_f, df, i == j);
                    if !seen.insert(key) {
                        continue;
                    }
                    let core_b = add2(core_a, d);
                    let raw_at_b = sub2(core_b, [0.5 * bb[0], 0.5 * bb[1]]);
                    let at_b = anti_coalesce_at(raw_at_b, d, &[at_a]);
                    let label = format!(
                        "{}({},{}) + {}({},{}) @ ({:.2}, {:.2})",
                        ta.id, ba_f[0], ba_f[1], tb.id, bb_f[0], bb_f[1], clean0(d[0]), clean0(d[1])
                    );
                    out.push(CombinationCandidate {
                        label,
                        defect_ids: vec![ta.id.clone(), tb.id.clone()],
                        separation: sep,
                        displacements: vec![d],
                        specs: vec![
                            DefectSpec::dislocation(ba, Some(at_a)),
                            DefectSpec::dislocation(bb, Some(at_b)),
                        ],
                    });
                }
            }
        }
    }
    out
}

/// Enumerate `(disclination, dislocation)` pairs modulo the proper site stabiliser `Stab^+(p_a)`.
///
/// Each kept representative places the dislocation core at `p_a + d + nudge`; its dropped `g in Stab^+(p_a)`
/// rotated copies sit at `p_a + g(d) + g(nudge)` (co-rotated nudge), so every merge is an exact `G_+` symmetry.
fn gen_mixed_pairs(cat: &DefectCatalogue, max_sep: f64) -> Vec<CombinationCandidate> {
    let disps = lattice_displacements(&cat.grp, max_sep);
    let nudge = mat2_vec(cat.grp.a, MIXED_CORE_NUDGE_FRAC);
    let mut out = Vec::new();
    for da in &cat.disclinations {
        let p_a = da.point;
        let stab = stabiliser_mats(&cat.grp, p_a);
        for tb in &cat.dislocations {
            let mut seen: HashSet<([i32; 2], [i32; 2])> = HashSet::new();
            for &bb_f in &tb.members {
                let bb = mat2_vec(cat.grp.a, [bb_f[0] as f64, bb_f[1] as f64]);
                for &(df, d, sep) in &disps {
                    let key = stab
                        .iter()
                        .map(|&mf| (imat2_vec(mf, bb_f), imat2_vec(mf, df)))
                        .min()
                        .expect("stabiliser is non-empty");
                    if !seen.insert(key) {
                        continue;
                    }
                    let core_b = add2(add2(p_a, d), nudge);
                    let at_b = sub2(core_b, [0.5 * bb[0], 0.5 * bb[1]]);
                    let label = format!(
                        "{} + {}({},{}) @ ({:.2}, {:.2})",
                        da.id, tb.id, bb_f[0], bb_f[1], clean0(d[0]), clean0(d[1])
                    );
                    out.push(CombinationCandidate {
                        label,
                        defect_ids: vec![da.id.clone(), tb.id.clone()],
                        separation: sep,
                        displacements: vec![d],
                        specs: vec![
                            DefectSpec::disclination(da.angle, p_a),
                            DefectSpec::dislocation(bb, Some(at_b)),
                        ],
                    });
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
enum ResolvedItem<'a> {
    Discl(&'a DisclinationEntry, &'a CentreClassEntry, usize),
    Disloc(&'a DislocationEntry, usize),
}

fn resolve_item<'a>(
    cat: &'a DefectCatalogue,
    cls_by_id: &HashMap<&str, &'a CentreClassEntry>,
    id: &str,
) -> Result<ResolvedItem<'a>, String> {
    if let Some((idx, d)) = cat.disclinations.iter().enumerate().find(|(_, d)| d.id == id) {
        let cls = cls_by_id
            .get(d.centre_class.as_str())
            .ok_or_else(|| format!("missing centre class {} for {}", d.centre_class, d.id))?;
        return Ok(ResolvedItem::Discl(d, cls, idx));
    }
    if let Some((idx, t)) = cat.dislocations.iter().enumerate().find(|(_, t)| t.id == id) {
        return Ok(ResolvedItem::Disloc(t, cat.disclinations.len() + idx));
    }
    Err(format!("unknown defect id {id:?} in catalogue for {}", cat.group))
}

type TripleKey = [(usize, [i64; 2], [i64; 2]); 3];

/// Exact `G_+ x S_3` canonical key for 3 defects `(tag, pos, b_vec)` (with co-rotated offsets for dropped
/// `G_+` copies whenever dislocations are present).
fn canon_triple_general(
    grp: &LatticeGroup,
    tags: [usize; 3],
    pts: [[f64; 2]; 3],
    bvecs: [[f64; 2]; 3],
    is_discl: [bool; 3],
) -> TripleKey {
    let mut best: Option<TripleKey> = None;
    for g in &grp.plus {
        for pi in PERMS_3 {
            let p0_orig = pts[pi[0]];
            let f0 = if is_discl[pi[0]] {
                q2(mat2_vec(grp.ainv, grp.reduce(g.apply(p0_orig))))
            } else {
                [0, 0]
            };
            let b0 = q2(mat2_vec(grp.ainv, g.apply_linear(bvecs[pi[0]])));
            let fd1 = q2(mat2_vec(grp.ainv, g.apply_linear(sub2(pts[pi[1]], p0_orig))));
            let b1 = q2(mat2_vec(grp.ainv, g.apply_linear(bvecs[pi[1]])));
            let fd2 = q2(mat2_vec(grp.ainv, g.apply_linear(sub2(pts[pi[2]], p0_orig))));
            let b2 = q2(mat2_vec(grp.ainv, g.apply_linear(bvecs[pi[2]])));
            let key = [(tags[pi[0]], f0, b0), (tags[pi[1]], fd1, b1), (tags[pi[2]], fd2, b2)];
            if best.is_none_or(|b| key < b) {
                best = Some(key);
            }
        }
    }
    best.expect("G_+ is non-empty")
}

fn gen_triple(
    cat: &DefectCatalogue,
    ids: &[String; 3],
    max_sep: f64,
) -> Result<Vec<CombinationCandidate>, String> {
    let cls_by_id: HashMap<&str, &CentreClassEntry> =
        cat.centre_classes.iter().map(|c| (c.id.as_str(), c)).collect();
    let items = [
        resolve_item(cat, &cls_by_id, &ids[0])?,
        resolve_item(cat, &cls_by_id, &ids[1])?,
        resolve_item(cat, &cls_by_id, &ids[2])?,
    ];
    let tags = items.map(|it| match it {
        ResolvedItem::Discl(_, _, tag) | ResolvedItem::Disloc(_, tag) => tag,
    });
    let is_discl = items.map(|it| matches!(it, ResolvedItem::Discl(..)));
    let has_discl = is_discl.iter().any(|&x| x);

    // Choose anchor slot: first disclination if any, otherwise slot 0 (dislocation).
    let anchor_slot = is_discl.iter().position(|&x| x).unwrap_or(0);
    let others: [usize; 2] = match anchor_slot {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let disps = lattice_displacements(&cat.grp, max_sep);
    let nudge = mat2_vec(cat.grp.a, MIXED_CORE_NUDGE_FRAC);

    // Anchor placement(s) for slot `anchor_slot`: `(logical_pos, b_vec, spec)`.
    let anchor_pos = match items[anchor_slot] {
        ResolvedItem::Discl(d, _, _) => d.point,
        ResolvedItem::Disloc(t, _) => t.at,
    };
    let anchor_b = match items[anchor_slot] {
        ResolvedItem::Discl(..) => [0.0, 0.0],
        ResolvedItem::Disloc(t, _) => t.b,
    };

    // Options for each non-anchor slot: `(logical_pos, b_vec, b_frac_opt)`.
    let slot_options = |slot: usize| -> Vec<([f64; 2], [f64; 2])> {
        match items[slot] {
            ResolvedItem::Discl(_, cls, _) => centre_points_near(&cat.grp, cls, anchor_pos, max_sep)
                .into_iter()
                .map(|(p, _, _)| (p, [0.0, 0.0]))
                .collect(),
            ResolvedItem::Disloc(t, _) => {
                let mut opts = Vec::new();
                for &bf in &t.members {
                    let bv = mat2_vec(cat.grp.a, [bf[0] as f64, bf[1] as f64]);
                    for &(_, d, _) in &disps {
                        opts.push((add2(anchor_pos, d), bv));
                    }
                }
                opts
            }
        }
    };

    let opts1 = slot_options(others[0]);
    let opts2 = slot_options(others[1]);
    let mut seen: HashSet<TripleKey> = HashSet::new();
    let mut out = Vec::new();

    for &(p1, b1) in &opts1 {
        for &(p2, b2) in &opts2 {
            let d12 = norm2(sub2(p2, p1));
            if d12 <= 1e-6 || d12 > max_sep + 1e-6 {
                continue;
            }
            let mut pts = [[0.0; 2]; 3];
            let mut bvecs = [[0.0; 2]; 3];
            pts[anchor_slot] = anchor_pos;
            bvecs[anchor_slot] = anchor_b;
            pts[others[0]] = p1;
            bvecs[others[0]] = b1;
            pts[others[1]] = p2;
            bvecs[others[1]] = b2;

            let key = canon_triple_general(&cat.grp, tags, pts, bvecs, is_discl);
            if !seen.insert(key) {
                continue;
            }

            let d01 = sub2(pts[1], pts[0]);
            let d02 = sub2(pts[2], pts[0]);
            let sep = norm2(d01).max(norm2(d02)).max(norm2(sub2(pts[2], pts[1])));
            let mut specs = Vec::with_capacity(3);
            let mut disloc_ats: Vec<[f64; 2]> = Vec::new();
            for s in 0..3 {
                match items[s] {
                    ResolvedItem::Discl(d, _, _) => specs.push(DefectSpec::disclination(d.angle, pts[s])),
                    ResolvedItem::Disloc(..) => {
                        let core = if has_discl { add2(pts[s], nudge) } else { pts[s] };
                        let raw_at = sub2(core, [0.5 * bvecs[s][0], 0.5 * bvecs[s][1]]);
                        let dir = sub2(pts[s], anchor_pos);
                        let at = anti_coalesce_at(raw_at, dir, &disloc_ats);
                        disloc_ats.push(at);
                        specs.push(DefectSpec::dislocation(bvecs[s], Some(at)));
                    }
                }
            }
            let label = format!(
                "{} + {} + {} @ ({:.2}, {:.2}), ({:.2}, {:.2})",
                ids[0], ids[1], ids[2], clean0(d01[0]), clean0(d01[1]), clean0(d02[0]), clean0(d02[1])
            );
            out.push(CombinationCandidate {
                label,
                defect_ids: ids.to_vec(),
                separation: sep,
                displacements: vec![d01, d02],
                specs,
            });
        }
    }
    Ok(out)
}

/// Plain-language description of the exact `G_+` symmetry reduction applied for `mode`.
pub(crate) fn symmetry_note_for_mode(mode: &CombinationMode) -> String {
    match mode {
        CombinationMode::DisclinationPairs => {
            "Exact G+ reduction (+ swap for identical species); all cores sit on rotation centres.".to_string()
        }
        CombinationMode::DislocationPairs => {
            "Exact P+ rotation reduction (+ swap for identical classes); each kept pair starts at core_a = ta.at and dropped G+ copies sit at the co-rotated anchor g(core_a).".to_string()
        }
        CombinationMode::MixedPairs => {
            "Exact Stab+(p_a) reduction around each disclination centre; dropped G+ copies carry the co-rotated dislocation core offset g(nudge).".to_string()
        }
        CombinationMode::Triple { .. } => {
            "Exact G+ x S_3 reduction; dropped G+ copies carry co-rotated dislocation core offsets.".to_string()
        }
        CombinationMode::Custom { .. } => {
            "Explicit caller-supplied defect placements (no automatic symmetry reduction).".to_string()
        }
    }
}

/// Generate all symmetry-inequivalent defect combination candidates for `mode` within `max_sep`.
pub(crate) fn generate_candidates(
    cat: &DefectCatalogue,
    mode: &CombinationMode,
    max_sep: f64,
) -> Result<Vec<CombinationCandidate>, String> {
    if !max_sep.is_finite() || max_sep < 0.0 {
        return Err("max_sep must be finite and non-negative".to_string());
    }
    match mode {
        CombinationMode::DisclinationPairs => Ok(gen_disclination_pairs(cat, max_sep)),
        CombinationMode::DislocationPairs => Ok(gen_dislocation_pairs(cat, max_sep)),
        CombinationMode::MixedPairs => Ok(gen_mixed_pairs(cat, max_sep)),
        CombinationMode::Triple { ids } => gen_triple(cat, ids, max_sep),
        CombinationMode::Custom { candidates } => Ok(candidates.clone()),
    }
}
