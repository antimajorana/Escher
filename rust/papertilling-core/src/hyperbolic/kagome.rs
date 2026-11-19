//! Hyperbolic rectified `(p.q.p.q)` kagome grown by the cluster rule (`wedge_delta == 0`).
//!
//! * For `p > q`: centres are the `p`-gons (`P_{n+1}` = edge-BFS on `{p, q}`, since two `p`-gons share
//!   a rectified edge-midpoint vertex iff they share a `{p, q}` edge), decorated by the `q`-gons at
//!   distinct vertices of `P_n`.
//! * For `p < q`: centres are the `q`-gons on dual `{q, p}` mapped into `RegularPQ` orientation by
//!   `T = B_x(R) * R_z(pi)` (`cosh R = cot(pi/p) cot(pi/q)`, so `T` sends dual vertex `0` to `(0, 0, 1)`
//!   and dual face `(0,0,1)` to primal vertex `0`). Seed `F_0` is not a centre, so `P_0` consists of the
//!   `p` `q`-gons around `F_0`, `P_{n+1}` is edge-BFS on `{q, p}`, and decoration puts `p`-gons at
//!   distinct dual vertices (with `F_0` emitted at `faces[0]`).
//! * For `p == q`: every face is a `p`-gon and `(p.p.p.p)` is `{p, 4}` rotated by `pi / p` about
//!   `(0, 0, 1)`, so the cluster rule equals vertex completion on `{p, 4}` rotated by `pi / p`.

use std::f64::consts::PI;

use super::inflation::{
    BaseFace, HypPointSet, IDENTITY, Mat3, T_PRECISION_LIMIT, Vec3, grow_vertex_completion, hyp_dist,
    hyperboloid_to_poincare, mat_mul, mat_vec,
};
use super::{DrawCaps, Patch, c_norm2, keep_whole_shells};

fn bx_rz_pi(r: f64) -> Mat3 {
    let (ch, sh) = (r.cosh(), r.sinh());
    [[-ch, 0.0, sh], [0.0, -1.0, 0.0], [-sh, 0.0, ch]]
}

/// Cyclic rectified vertices of the corner polygon around vertex `j` of `base` (in CCW order when `g = IDENTITY`).
fn qgon_midpoints_at_vertex(base: &BaseFace, j: usize) -> Vec<Vec3> {
    let (p, q) = (base.p, base.q);
    let (a, b) = (j, (j + p - 1) % p);
    let mut pts = Vec::with_capacity(q);
    pts.push(base.edge_midpoints[b]);
    pts.push(base.edge_midpoints[a]);
    for k in 2..q {
        let side = if k.is_multiple_of(2) { b } else { a };
        pts.push(mat_vec(&base.star[j][k - 2], base.edge_midpoints[side]));
    }
    pts.reverse();
    pts
}

type RawFace = (Vec3, Vec<Vec3>);

fn equal_order_kagome(p: usize, lay: usize, caps: DrawCaps) -> Result<Patch, String> {
    let DrawCaps { max_faces, max_centre_radius } = caps;
    let vc4 = grow_vertex_completion(p, 4, lay, 2 * max_faces)?;
    let (keep, mut notes) = keep_whole_shells(&vc4.shell_sizes, &vc4.stopped, max_faces);
    let angle = PI / p as f64;
    let (ca, sa) = (angle.cos(), angle.sin());
    let rot = |v: Vec3| [ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]];
    let mut set = HypPointSet::new(hyp_dist(vc4.base.vertices[0], vc4.base.vertices[1]));
    let r2_max = max_centre_radius * max_centre_radius;
    let mut complete = keep - 1;
    let (mut faces, mut face_shell) = (Vec::new(), Vec::new());
    for (i, &s) in vc4.shell.iter().enumerate().take_while(|&(_, &s)| s < keep) {
        if i > 0 && c_norm2(hyperboloid_to_poincare(vc4.centre(i))) > r2_max {
            complete = complete.min(s - 1);
            continue;
        }
        faces.push(vc4.face_vertices(i).into_iter().map(|v| set.insert(rot(v)).0).collect());
        face_shell.push(s);
    }
    if set.suspect > 0 {
        return Err(format!("{{{p},{p}}}: f64 precision: {} drawn points have an ambiguous dedup distance", set.suspect));
    }
    if complete + 1 < keep {
        notes.push(format!("shell {} truncated by the centre radius cap {max_centre_radius}", complete + 1));
    }
    let growth_rule = format!(
        "cluster rule on ({p}.{p}.{p}.{p}) kagome (all {p}-gons are centres): S_0..S_{complete} complete, {} faces drawn (requested S_{lay}){}",
        faces.len(),
        notes.iter().map(|n| format!("; {n}")).collect::<String>()
    );
    let pts = set.pts.iter().map(|&x| hyperboloid_to_poincare(x)).collect();
    Ok(Patch { pts, faces, face_shell, complete_through_shell: complete, growth_rule })
}

pub(super) fn kagome_cluster_patch(p: usize, q: usize, lay: usize, caps: DrawCaps) -> Result<Patch, String> {
    if p == q {
        return equal_order_kagome(p, lay, caps);
    }
    let DrawCaps { max_faces, max_centre_radius } = caps;
    let dual_mode = p < q;
    let base = if dual_mode { BaseFace::new(q, p)? } else { BaseFace::new(p, q)? };
    let bp = base.p;
    let t_mat = if dual_mode { bx_rz_pi(base.circumradius) } else { IDENTITY };
    let l_k = hyp_dist(base.edge_midpoints[0], base.edge_midpoints[1 % bp]);
    let origin = [0.0, 0.0, 1.0];
    let mut centres = HypPointSet::new(2.0 * base.inradius);
    let mut verts = HypPointSet::new(base.edge_length());
    let mut p_mats = vec![IDENTITY];
    if dual_mode {
        p_mats.extend_from_slice(&base.star[0]);
    }
    for &g in &p_mats {
        centres.insert(mat_vec(&g, origin));
    }
    let face_pair = |g: &Mat3| -> RawFace {
        let wg = mat_mul(&t_mat, g);
        (mat_vec(&wg, origin), base.edge_midpoints.iter().map(|&x| mat_vec(&wg, x)).collect())
    };
    let dec_pair = |g: &Mat3, j: usize| -> RawFace {
        let wg = mat_mul(&t_mat, g);
        let pts = qgon_midpoints_at_vertex(&base, j).into_iter().map(|x| mat_vec(&wg, x)).collect();
        (mat_vec(&wg, base.vertices[j]), pts)
    };
    let mut dec0: Vec<RawFace> = Vec::new();
    for &g in &p_mats {
        for (j, &bv) in base.vertices.iter().enumerate() {
            if verts.insert(mat_vec(&g, bv)).1 {
                dec0.push(dec_pair(&g, j));
            }
        }
    }
    let mut p0_faces: Vec<RawFace> = p_mats.iter().map(face_pair).collect();
    let s0 = if dual_mode {
        let mut s = Vec::with_capacity(p0_faces.len() + dec0.len());
        s.push(dec0.remove(0));
        s.append(&mut p0_faces);
        s.extend(dec0);
        s
    } else {
        p0_faces.extend(dec0);
        p0_faces
    };
    let mut total = s0.len();
    let mut raw_shells = vec![s0];
    let mut notes = Vec::new();
    let mut p_start = 0;
    for s in 0..lay {
        let p_end = p_mats.len();
        let (mut next_p, mut next_dec) = (Vec::new(), Vec::new());
        let mut aborted: Option<String> = None;
        'grow: for fi in p_start..p_end {
            let g = p_mats[fi];
            for k in 0..bp {
                let h = mat_mul(&g, &base.reflections[k]);
                let c = mat_vec(&h, origin);
                if !c.iter().all(|x| x.is_finite()) || c[2] > T_PRECISION_LIMIT {
                    aborted = Some(format!("f64 precision limit: shell {} not built", s + 1));
                    break 'grow;
                }
                if centres.insert(c).1 {
                    p_mats.push(h);
                    next_p.push(face_pair(&h));
                    for (j, &bv) in base.vertices.iter().enumerate() {
                        if verts.insert(mat_vec(&h, bv)).1 {
                            next_dec.push(dec_pair(&h, j));
                        }
                    }
                    if total + next_p.len() + next_dec.len() > 2 * max_faces {
                        aborted = Some(format!("shell {} omitted by the {max_faces}-face cap (estimated larger)", s + 1));
                        break 'grow;
                    }
                }
            }
        }
        if let Some(note) = aborted {
            notes.push(note);
            break;
        }
        if centres.suspect + verts.suspect > 0 {
            notes.push(format!("f64 precision: ambiguous dedup: shell {} not built", s + 1));
            break;
        }
        let a = next_p.len() + next_dec.len();
        if total + a > max_faces {
            notes.push(format!("shell {} ({a} faces) omitted by the {max_faces}-face cap", s + 1));
            break;
        }
        next_p.extend(next_dec);
        total += next_p.len();
        raw_shells.push(next_p);
        p_start = p_end;
    }
    let keep = raw_shells.len();
    let mut complete = keep - 1;
    let r2_max = max_centre_radius * max_centre_radius;
    let mut kag_set = HypPointSet::new(l_k);
    let (mut faces, mut face_shell) = (Vec::new(), Vec::new());
    for (s, sh) in raw_shells.into_iter().enumerate() {
        for (cen, vpts) in sh {
            if s > 0 && c_norm2(hyperboloid_to_poincare(cen)) > r2_max {
                complete = complete.min(s - 1);
                continue;
            }
            faces.push(vpts.into_iter().map(|x| kag_set.insert(x).0).collect());
            face_shell.push(s);
        }
    }
    if kag_set.suspect > 0 {
        return Err(format!("{{{p},{q}}}: f64 precision: {} drawn points have an ambiguous dedup distance", kag_set.suspect));
    }
    if complete + 1 < keep {
        notes.push(format!("shell {} truncated by the centre radius cap {max_centre_radius}", complete + 1));
    }
    let c_sides = p.max(q);
    let growth_rule = format!(
        "cluster rule on ({p}.{q}.{p}.{q}) kagome (centres: {c_sides}-gons; {p}-gons and {q}-gons): S_0..S_{complete} complete, {} faces drawn (requested S_{lay}){}",
        faces.len(),
        notes.iter().map(|n| format!("; {n}")).collect::<String>()
    );
    let pts = kag_set.pts.iter().map(|&x| hyperboloid_to_poincare(x)).collect();
    Ok(Patch { pts, faces, face_shell, complete_through_shell: complete, growth_rule })
}
