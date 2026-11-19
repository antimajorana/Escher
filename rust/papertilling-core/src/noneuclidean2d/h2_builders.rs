//! `DunhamHyperboloid` and `SpatialHashPoincare` `O(N)` builders for `H2LargePatch`.

use std::f64::consts::PI;

use crate::hyperbolic::inflation::{
    BaseFace, HypPointSet, IDENTITY, Mat3, Vec3, hyperboloid_to_poincare, mat_mul, mat_vec, mink,
};

use super::fast_h2_kernel::H2LargePatch;
use super::h2_spatial_hash::{
    FastMap, H2Kernel, PoincarePolarHash, SU_ID, Su11, poincare_to_su_trans, su_mul,
};

/// Half-turns about the edge midpoints of the base `p`-gon (edge `k` at polar angle
/// `(k + 1/2) 2 pi / p`, distance `r` = inradius): `T_{2r} R_pi = (i cosh r, -i sinh r e^{i phi})`,
/// mapping the base face onto its neighbour across edge `k`.
pub(crate) fn su_step_gen(p: usize, inradius: f64) -> Vec<Su11> {
    let (ch, sh) = (inradius.cosh(), inradius.sinh());
    (0..p)
        .map(|k| {
            let ph = 2.0 * PI * (k as f64 + 0.5) / (p as f64);
            [0.0, ch, sh * ph.sin(), -sh * ph.cos()]
        })
        .collect()
}

pub(crate) fn su_rot_p(p: usize, steps: isize) -> Su11 {
    let half_ang = PI * (steps as f64) / (p as f64);
    [half_ang.cos(), half_ang.sin(), 0.0, 0.0]
}

fn lorentz_reortho(g: &mut Mat3) {
    let col = |m: &Mat3, c: usize| [m[0][c], m[1][c], m[2][c]];
    let set_col = |m: &mut Mat3, c: usize, v: Vec3| {
        m[0][c] = v[0];
        m[1][c] = v[1];
        m[2][c] = v[2];
    };
    let mut e2 = col(g, 2);
    let n2 = (-mink(e2, e2)).max(1e-18).sqrt();
    e2 = [e2[0] / n2, e2[1] / n2, e2[2] / n2];
    let mut e0 = col(g, 0);
    let d02 = mink(e0, e2);
    e0 = [e0[0] + d02 * e2[0], e0[1] + d02 * e2[1], e0[2] + d02 * e2[2]];
    let n0 = mink(e0, e0).max(1e-18).sqrt();
    e0 = [e0[0] / n0, e0[1] / n0, e0[2] / n0];
    let mut e1 = col(g, 1);
    let (d12, d10) = (mink(e1, e2), mink(e1, e0));
    e1 = [
        e1[0] + d12 * e2[0] - d10 * e0[0],
        e1[1] + d12 * e2[1] - d10 * e0[1],
        e1[2] + d12 * e2[2] - d10 * e0[2],
    ];
    let n1 = mink(e1, e1).max(1e-18).sqrt();
    e1 = [e1[0] / n1, e1[1] / n1, e1[2] / n1];
    set_col(g, 0, e0);
    set_col(g, 1, e1);
    set_col(g, 2, e2);
}

#[inline]
fn lorentz_pt_to_su(pt: Vec3) -> Su11 {
    let t1 = (pt[2] + 1.0).max(1e-12);
    let a = (0.5 * t1).sqrt();
    let inv = 1.0 / (2.0 * t1).sqrt();
    [a, 0.0, pt[0] * inv, pt[1] * inv]
}

pub(crate) fn build_dunham_hyperboloid(p: usize, q: usize, target: usize) -> Result<H2LargePatch, String> {
    let base = BaseFace::new(p, q)?;
    let mut centres = HypPointSet::new(2.0 * base.inradius);
    let mut verts = HypPointSet::new(base.edge_length());
    let mut mats = vec![IDENTITY];
    let mut face_shell = vec![0usize];
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(target);
    let mut vertex_faces: Vec<Vec<usize>> = Vec::new();
    let mut completed: Vec<bool> = Vec::new();
    let mut face_hash = PoincarePolarHash::new(2.0 * base.inradius);

    centres.insert([0.0, 0.0, 1.0]);
    face_hash.push_unchecked(SU_ID);
    let mut cyc0 = Vec::with_capacity(p);
    for &bv in &base.vertices {
        let (vi, new) = verts.insert(bv);
        if new {
            completed.push(false);
            vertex_faces.push(Vec::new());
        }
        vertex_faces[vi].push(0);
        cyc0.push(vi);
    }
    faces.push(cyc0);

    let mut cursor = 0usize;
    let mut shell = 1usize;
    let mut shell_end = 1usize;
    while faces.len() < target && cursor < mats.len() {
        if cursor >= shell_end {
            shell += 1;
            shell_end = mats.len();
        }
        let fi = cursor;
        cursor += 1;
        let g = mats[fi];
        for j in 0..p {
            let vi = faces[fi][j];
            if completed[vi] {
                continue;
            }
            completed[vi] = true;
            for star_m in &base.star[j] {
                if faces.len() >= target {
                    break;
                }
                let mut h = mat_mul(&g, star_m);
                lorentz_reortho(&mut h);
                let c = [h[0][2], h[1][2], h[2][2]];
                if centres.insert(c).1 {
                    let nf = faces.len();
                    mats.push(h);
                    face_shell.push(shell);
                    face_hash.push_unchecked(lorentz_pt_to_su(c));
                    let mut cyc = Vec::with_capacity(p);
                    for &bv in &base.vertices {
                        let (nvi, new) = verts.insert(mat_vec(&h, bv));
                        if new {
                            completed.push(false);
                            vertex_faces.push(Vec::new());
                        }
                        vertex_faces[nvi].push(nf);
                        cyc.push(nvi);
                    }
                    faces.push(cyc);
                }
            }
            if faces.len() >= target {
                break;
            }
        }
    }
    let poincare_pts = verts.pts.iter().map(|&x| hyperboloid_to_poincare(x)).collect();
    let face_centres_poincare = centres.pts.iter().map(|&x| hyperboloid_to_poincare(x)).collect();
    finalize_patch(p, q, H2Kernel::DunhamHyperboloid, faces, face_shell, face_centres_poincare, poincare_pts, vertex_faces, face_hash)
}

pub(crate) fn build_spatial_hash_poincare(p: usize, q: usize, target: usize) -> Result<H2LargePatch, String> {
    let base = BaseFace::new(p, q)?;
    let h_gen = su_step_gen(p, base.inradius);
    let rot_p1 = su_rot_p(p, 1);
    // step_j = H_j * Rot(2 pi / p) fixes vertex j (Rot sends j -> j+1, the half-turn H_j swaps j and
    // j+1) and preserves orientation, so it is the rotation by 2 pi / q about vertex j.
    let mut star_su = Vec::with_capacity(p);
    for &hg in &h_gen {
        let step_j = su_mul(hg, rot_p1);
        let mut ring = Vec::with_capacity(q - 1);
        let mut cur = SU_ID;
        for _ in 1..q {
            cur = su_mul(cur, step_j);
            ring.push(cur);
        }
        star_su.push(ring);
    }
    let vert_su0: Vec<Su11> = (0..p)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p as f64);
            let r_v = (0.5 * base.circumradius).tanh();
            poincare_to_su_trans([r_v * th.cos(), r_v * th.sin()]).unwrap()
        })
        .collect();

    let v_est = (target * p / q.max(2)).max(64) + 64;
    let mut face_hash = PoincarePolarHash::with_capacity(2.0 * base.inradius, target + 16);
    let mut vert_hash = PoincarePolarHash::with_capacity(base.edge_length(), v_est);
    face_hash.insert(SU_ID);
    let mut face_shell = Vec::with_capacity(target);
    face_shell.push(0usize);
    let mut faces = Vec::with_capacity(target);
    let mut vertex_faces: Vec<Vec<usize>> = Vec::with_capacity(v_est);
    let mut completed: Vec<bool> = Vec::with_capacity(v_est);

    let mut cyc0 = Vec::with_capacity(p);
    for &vsu in &vert_su0 {
        let (vi, new) = vert_hash.insert(vsu);
        if new {
            completed.push(false);
            vertex_faces.push(Vec::new());
        }
        vertex_faces[vi].push(0);
        cyc0.push(vi);
    }
    faces.push(cyc0);

    let mut cursor = 0usize;
    let mut shell = 1usize;
    let mut shell_end = 1usize;
    while faces.len() < target && cursor < face_hash.items.len() {
        if cursor >= shell_end {
            shell += 1;
            shell_end = face_hash.items.len();
        }
        let fi = cursor;
        cursor += 1;
        let g = face_hash.items[fi];
        for (j, ring) in star_su.iter().enumerate() {
            let vi = faces[fi][j];
            if completed[vi] {
                continue;
            }
            completed[vi] = true;
            for &sm in ring {
                if faces.len() >= target {
                    break;
                }
                let h = su_mul(g, sm);
                let (nf, inserted) = face_hash.insert(h);
                if inserted {
                    face_shell.push(shell);
                    let mut cyc = Vec::with_capacity(p);
                    for &vsu in &vert_su0 {
                        let (nvi, new) = vert_hash.insert(su_mul(h, vsu));
                        if new {
                            completed.push(false);
                            vertex_faces.push(Vec::new());
                        }
                        vertex_faces[nvi].push(nf);
                        cyc.push(nvi);
                    }
                    faces.push(cyc);
                }
            }
            if faces.len() >= target {
                break;
            }
        }
    }
    let face_centres_poincare = face_hash.z_pts.clone();
    let poincare_pts = vert_hash.z_pts;
    finalize_patch(p, q, H2Kernel::SpatialHashPoincare, faces, face_shell, face_centres_poincare, poincare_pts, vertex_faces, face_hash)
}

#[allow(clippy::too_many_arguments)]
fn finalize_patch(
    p: usize,
    q: usize,
    kernel: H2Kernel,
    faces: Vec<Vec<usize>>,
    face_shell: Vec<usize>,
    face_centres_poincare: Vec<[f64; 2]>,
    poincare_pts: Vec<[f64; 2]>,
    vertex_faces: Vec<Vec<usize>>,
    face_hash: PoincarePolarHash,
) -> Result<H2LargePatch, String> {
    let nf = faces.len();
    let mut face_neighbors = vec![vec![None; p]; nf];
    let mut edge_map: FastMap<(usize, usize), (usize, usize)> = FastMap::default();
    edge_map.reserve(nf * p / 2 + 16);
    let mut n_edges = 0usize;
    for (fi, f) in faces.iter().enumerate() {
        for s in 0..p {
            let (u, v) = (f[s], f[(s + 1) % p]);
            let key = (u.min(v), u.max(v));
            if let Some(&(of, os)) = edge_map.get(&key) {
                face_neighbors[fi][s] = Some(of);
                face_neighbors[of][os] = Some(fi);
            } else {
                edge_map.insert(key, (fi, s));
                n_edges += 1;
            }
        }
    }
    Ok(H2LargePatch::from_parts(
        p,
        q,
        kernel,
        faces,
        face_shell,
        face_centres_poincare,
        poincare_pts,
        face_neighbors,
        vertex_faces,
        n_edges,
        face_hash,
    ))
}
