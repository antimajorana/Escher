//! Exact `O(N)` combinatorial boundary-word builder for `H2LargePatch`.

use std::f64::consts::PI;

use crate::hyperbolic::inflation::BaseFace;

use super::fast_h2_kernel::H2LargePatch;
use super::h2_builders::{su_rot_p, su_step_gen};
use super::h2_spatial_hash::{H2Kernel, PoincarePolarHash, SU_ID, Su11, poincare_to_su_trans, su_mul, su_origin};

#[derive(Clone, Copy)]
struct BSlot {
    vid: usize,
    deg: usize,
    owner_face: usize,
    owner_side: usize,
}

fn link_pair(fa: usize, sa: usize, fb: usize, sb: usize, nbrs: &mut [Vec<Option<usize>>]) {
    nbrs[fa][sa] = Some(fb);
    nbrs[fb][sb] = Some(fa);
}

fn push_vert(
    f_su: Su11,
    k: usize,
    vert_su0: &[Su11],
    pts: &mut Vec<[f64; 2]>,
    vf: &mut Vec<Vec<usize>>,
) -> usize {
    let id = pts.len();
    pts.push(su_origin(su_mul(f_su, vert_su0[k])));
    vf.push(Vec::new());
    id
}

pub(crate) fn build_graph_combinatorial(p: usize, q: usize, target: usize) -> Result<H2LargePatch, String> {
    let base = BaseFace::new(p, q)?;
    let h_gen = su_step_gen(p, base.inradius);
    let rho_v = (0.5 * base.circumradius).tanh();
    let vert_su0: Vec<Su11> = (0..p)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p as f64);
            poincare_to_su_trans([rho_v * th.cos(), rho_v * th.sin()]).unwrap()
        })
        .collect();
    let mut poincare_pts: Vec<[f64; 2]> = vert_su0.iter().map(|&s| su_origin(s)).collect();
    let mut faces = vec![(0..p).collect::<Vec<_>>()];
    let mut face_shell = vec![0usize];
    let mut face_neighbors = vec![vec![None; p]];
    let mut vertex_faces = vec![vec![0usize]; p];
    let mut face_su = vec![SU_ID];
    let mut face_centres_poincare = vec![[0.0, 0.0]];
    let mut face_hash = PoincarePolarHash::new(2.0 * base.inradius);
    face_hash.push_unchecked(SU_ID);
    let mut n_edges = p;

    let mut boundary: Vec<BSlot> = (0..p)
        .map(|k| BSlot { vid: k, deg: 1, owner_face: 0, owner_side: k })
        .collect();

    let mut shell = 1usize;
    while faces.len() < target {
        let len_b = boundary.len();
        let mut next_b: Vec<BSlot> = Vec::with_capacity(len_b * 2);
        let mut shell_complete = true;
        if q == 3 {
            let i0 = (0..len_b).find(|&i| boundary[i].deg == 1).ok_or("non-convex q=3 boundary")?;
            let (mut off, mut prev_face) = (0usize, None::<(usize, usize)>);
            let mut first_face = None::<(usize, usize, usize)>;
            while off < len_b && faces.len() < target {
                let i = (i0 + off) % len_b;
                let (in1, in2) = ((i + 1) % len_b, (i + 2) % len_b);
                let m_old = if boundary[in1].deg == 2 { 2 } else { 1 };
                off += m_old;
                let c = p - m_old - 1;
                let f_id = faces.len();
                // The new face lies across the anchor's outer side s (anchor -> next slot); H_s Rot(s) sends
                // labels 0, 1 to the owner's s + 1, s, i.e. cyc[0] = next slot, cyc[1] = anchor.
                let anc = if m_old == 2 { in1 } else { i };
                let s_anc = boundary[anc].owner_side;
                let step = su_mul(h_gen[s_anc], su_rot_p(p, s_anc as isize));
                let f_su = su_mul(face_su[boundary[anc].owner_face], step);
                face_su.push(f_su);
                face_centres_poincare.push(su_origin(f_su));
                face_hash.push_unchecked(f_su);
                face_shell.push(shell);
                face_neighbors.push(vec![None; p]);

                let mut cyc = vec![0usize; p];
                if m_old == 1 {
                    cyc[0] = boundary[in1].vid;
                    cyc[1] = boundary[i].vid;
                    link_pair(f_id, 0, boundary[i].owner_face, boundary[i].owner_side, &mut face_neighbors);
                } else {
                    cyc[0] = boundary[in2].vid;
                    cyc[1] = boundary[in1].vid;
                    cyc[2] = boundary[i].vid;
                    link_pair(f_id, 0, boundary[in1].owner_face, boundary[in1].owner_side, &mut face_neighbors);
                    link_pair(f_id, 1, boundary[i].owner_face, boundary[i].owner_side, &mut face_neighbors);
                }
                let is_first_of_shell = prev_face.is_none();
                let shared_prev = if let Some((pf, sv)) = prev_face {
                    link_pair(f_id, m_old, pf, p - 1, &mut face_neighbors);
                    sv
                } else {
                    push_vert(f_su, m_old + 1, &vert_su0, &mut poincare_pts, &mut vertex_faces)
                };
                cyc[m_old + 1] = shared_prev;
                let mut wrapped = false;
                for (offset_k, item) in cyc.iter_mut().enumerate().skip(m_old + 2) {
                    let is_last = offset_k == p - 1;
                    *item = if is_last
                        && off >= len_b
                        && let Some((ff, fv, f_side)) = first_face
                    {
                        link_pair(f_id, p - 1, ff, f_side, &mut face_neighbors);
                        wrapped = true;
                        fv
                    } else {
                        push_vert(f_su, offset_k, &vert_su0, &mut poincare_pts, &mut vertex_faces)
                    };
                }
                if first_face.is_none() {
                    first_face = Some((f_id, shared_prev, m_old));
                }
                prev_face = Some((f_id, cyc[p - 1]));
                n_edges += c + usize::from(is_first_of_shell) - usize::from(wrapped);
                for &v in &cyc {
                    vertex_faces[v].push(f_id);
                }
                faces.push(cyc);
                // Outer sides are m_old + 1 ..= p - 2; side p - 1 (into cyc[p - 1]) is shared with the next
                // face, whose cyc[m_old + 1] is that same vertex, so it is not pushed here (no duplicates).
                for s_off in 0..c - 1 {
                    let s_idx = m_old + 1 + s_off;
                    next_b.push(BSlot {
                        vid: faces[f_id][s_idx],
                        deg: if s_off == 0 { 2 } else { 1 },
                        owner_face: f_id,
                        owner_side: s_idx,
                    });
                }
            }
            if faces.len() >= target && off < len_b {
                shell_complete = false;
            }
        } else {
            let mut prev_face = None::<(usize, usize)>;
            let mut first_face = None::<(usize, usize, usize)>;
            'outer: for i in 0..len_b {
                let n_corner = (q - boundary[i].deg) - 2;
                let in1 = (i + 1) % len_b;
                for cf in 0..=n_corner {
                    if faces.len() >= target {
                        shell_complete = false;
                        break 'outer;
                    }
                    let is_edge_face = cf == n_corner;
                    let f_id = faces.len();
                    let f_su = if is_edge_face {
                        su_mul(face_su[boundary[i].owner_face], su_mul(h_gen[boundary[i].owner_side], su_rot_p(p, boundary[i].owner_side as isize)))
                    } else if let Some((pf, _)) = prev_face {
                        su_mul(face_su[pf], su_mul(h_gen[p - 1], su_rot_p(p, (p - 1) as isize)))
                    } else {
                        // First corner of the shell: from the edge face E across side s (cyc[1] = vertex i),
                        // C_last = E H_1 Rot(2) (labels 0, p - 1 -> E's 1, 2) and each earlier corner is
                        // C H_0 Rot(1) (labels 0, p - 1 -> C's 0, 1).
                        let s = boundary[i].owner_side;
                        let mut g = su_mul(face_su[boundary[i].owner_face], su_mul(h_gen[s], su_rot_p(p, s as isize)));
                        g = su_mul(g, su_mul(h_gen[1], su_rot_p(p, 2)));
                        for _ in 1..n_corner {
                            g = su_mul(g, su_mul(h_gen[0], su_rot_p(p, 1)));
                        }
                        g
                    };
                    face_su.push(f_su);
                    face_centres_poincare.push(su_origin(f_su));
                    face_hash.push_unchecked(f_su);
                    face_shell.push(shell);
                    face_neighbors.push(vec![None; p]);

                    let mut cyc = vec![0usize; p];
                    let (old_count, prev_side) = if is_edge_face {
                        cyc[0] = boundary[in1].vid;
                        cyc[1] = boundary[i].vid;
                        link_pair(f_id, 0, boundary[i].owner_face, boundary[i].owner_side, &mut face_neighbors);
                        (2, 1)
                    } else {
                        cyc[0] = boundary[i].vid;
                        (1, 0)
                    };
                    if p == 3 && is_edge_face {
                        let shared = if let Some((pf, sv)) = prev_face {
                            link_pair(f_id, 1, pf, 2, &mut face_neighbors);
                            sv
                        } else {
                            push_vert(f_su, 2, &vert_su0, &mut poincare_pts, &mut vertex_faces)
                        };
                        cyc[2] = shared;
                        if i + 1 == len_b && let Some((ff, _, f_side)) = first_face {
                            link_pair(f_id, 2, ff, f_side, &mut face_neighbors);
                        } else {
                            n_edges += 1;
                        }
                        if first_face.is_none() {
                            first_face = Some((f_id, shared, 1));
                        }
                        prev_face = Some((f_id, shared));
                    } else {
                        let is_first_of_shell = prev_face.is_none();
                        let shared_prev = if let Some((pf, sv)) = prev_face {
                            link_pair(f_id, prev_side, pf, p - 1, &mut face_neighbors);
                            sv
                        } else {
                            push_vert(f_su, old_count, &vert_su0, &mut poincare_pts, &mut vertex_faces)
                        };
                        cyc[old_count] = shared_prev;
                        let wrap_p3_corner = p == 3 && !is_edge_face && i + 1 == len_b && cf + 1 == n_corner;
                        let mut wrapped = false;
                        for (k, slot) in cyc.iter_mut().enumerate().skip(old_count + 1) {
                            let is_last = k == p - 1;
                            let at_wrap = is_last && i + 1 == len_b && is_edge_face;
                            *slot = if (at_wrap || (is_last && wrap_p3_corner))
                                && let Some((ff, fv, f_side)) = first_face
                            {
                                if at_wrap {
                                    link_pair(f_id, p - 1, ff, f_side, &mut face_neighbors);
                                    wrapped = true;
                                }
                                fv
                            } else {
                                push_vert(f_su, k, &vert_su0, &mut poincare_pts, &mut vertex_faces)
                            };
                        }
                        if first_face.is_none() {
                            first_face = Some((f_id, shared_prev, prev_side));
                        }
                        prev_face = Some((f_id, cyc[p - 1]));
                        // New edges: sides old_count ..= p - 1 (side old_count - 1 is shared with the previous face).
                        // Outer sides: old_count ..= p - 2; cyc[p - 1] is the next face's first vertex.
                        let c_new = p - old_count;
                        n_edges += c_new + usize::from(is_first_of_shell) - usize::from(wrapped);
                        for s_off in 0..c_new - 1 {
                            let s_idx = old_count + s_off;
                            next_b.push(BSlot {
                                vid: cyc[s_idx],
                                deg: if s_off == 0 { 2 } else { 1 },
                                owner_face: f_id,
                                owner_side: s_idx,
                            });
                        }
                    }
                    for &v in &cyc {
                        vertex_faces[v].push(f_id);
                    }
                    faces.push(cyc);
                }
            }
            if p == 3 {
                for slot in &mut next_b {
                    if vertex_faces[slot.vid].len() >= 3 {
                        slot.deg = 3;
                    }
                }
            }
        }
        if !shell_complete {
            break;
        }
        boundary = next_b;
        shell += 1;
    }
    Ok(H2LargePatch::from_parts(
        p, q, H2Kernel::GraphCombinatorial, faces, face_shell, face_centres_poincare, poincare_pts, face_neighbors, vertex_faces, n_edges, face_hash,
    ))
}
