//! Generic cell-by-cell orbit builder for the reflection group `[p,q,r]` (shared by S³ and H³).
//!
//! The central cell is the orbit of the seed face `F0 = {(R0 R1)^k v0}` under `<R0,R1,R2>`; its
//! faces are found by BFS across edges (`g ↦ g (R0R1)^k R2`). The neighbour of cell `M` across
//! its face `M g F0` is `M g R3 C0` (the mirror `R3` contains `F0`). Vertices, faces and cells are
//! deduplicated by position (Klein coordinates in H³, unit vectors in S³).

use std::collections::VecDeque;

use super::coxeter_gram::{form, mat_mul, mat_vec, CoxeterRealisation, Mat4, Vec4};
use super::{CellComplex3d, ComplexBuilder, PointIndex};

const MAX_CELLS: usize = 50_000;
const MAX_PATCH_FACES: usize = 2_000;

pub struct OrbitOptions {
    /// Cell shells around the central cell (`u32::MAX` = until closure, used for S³).
    pub shells: u32,
    /// For infinite cells: edge-adjacency rings of faces kept around the seed face.
    pub patch_layers: u32,
    /// Cells are infinite (`(p-2)(q-2) == 4`).
    pub infinite_cells: bool,
    /// Vertices are ideal (`(q-2)(r-2) == 4`).
    pub ideal_vertices: bool,
}

fn key(real: &CoxeterRealisation, x: &Vec4) -> [f64; 4] {
    if real.eta[0] < 0.0 {
        [x[1] / x[0], x[2] / x[0], x[3] / x[0], 0.0]
    } else {
        let n = form(&real.eta, x, x).sqrt();
        x.map(|c| c / n)
    }
}

/// Faces of the (possibly truncated) central cell, as group elements `g` with `g F0` the face.
fn cell_patch(real: &CoxeterRealisation, p: u32, opts: &OrbitOptions) -> Result<Vec<Mat4>, String> {
    let r = &real.reflections;
    let rot = mat_mul(&r[0], &r[1]);
    let mut rot_pows = vec![super::coxeter_gram::identity4()];
    for k in 1..p as usize {
        rot_pows.push(mat_mul(&rot_pows[k - 1], &rot));
    }
    let steps: Vec<Mat4> = rot_pows.iter().map(|rk| mat_mul(rk, &r[2])).collect();
    let mut index = PointIndex::<4>::new(1e-7);
    let id = super::coxeter_gram::identity4();
    index.insert(key(real, &mat_vec(&id, &real.w[2])));
    let mut out = vec![id];
    let mut queue = VecDeque::from([(id, 0u32)]);
    while let Some((g, depth)) = queue.pop_front() {
        if opts.infinite_cells && depth >= opts.patch_layers {
            continue;
        }
        for s in &steps {
            let h = mat_mul(&g, s);
            if index.insert(key(real, &mat_vec(&h, &real.w[2]))).1 {
                out.push(h);
                queue.push_back((h, depth + 1));
                if out.len() > MAX_PATCH_FACES {
                    return Err("cell patch did not close (more than 2000 faces)".into());
                }
            }
        }
    }
    Ok(out)
}

/// Build the complex; render coordinates of S³ vertices are left as zero (set by the caller).
pub fn build_orbit_complex(
    real: &CoxeterRealisation,
    p: u32,
    opts: &OrbitOptions,
    name: &str,
    regime: &str,
) -> Result<CellComplex3d, String> {
    let lorentz = real.eta[0] < 0.0;
    let r = &real.reflections;
    let rot = mat_mul(&r[0], &r[1]);
    let mut face0 = vec![real.w[0]];
    for k in 1..p as usize {
        face0.push(mat_vec(&rot, &face0[k - 1]));
    }
    let patch = cell_patch(real, p, opts)?;
    let mut b = ComplexBuilder::new(name, regime);
    let mut vindex = PointIndex::<4>::new(1e-7);
    let mut cindex = PointIndex::<4>::new(1e-7);
    let id = super::coxeter_gram::identity4();
    cindex.insert(key(real, &real.w[3]));
    let mut queue = VecDeque::from([(id, 0u32)]);
    while let Some((m, shell)) = queue.pop_front() {
        let mut faces = Vec::with_capacity(patch.len());
        for g in &patch {
            let mg = mat_mul(&m, g);
            let mut cycle = Vec::with_capacity(face0.len());
            for v in &face0 {
                let x = mat_vec(&mg, v);
                let (vid, new) = vindex.insert(key(real, &x));
                if new {
                    push_vertex(&mut b.out, real, &x, lorentz, opts.ideal_vertices);
                }
                cycle.push(vid);
            }
            faces.push(b.add_face(cycle));
            if shell < opts.shells {
                let nm = mat_mul(&mg, &r[3]);
                if cindex.insert(key(real, &mat_vec(&nm, &real.w[3]))).1 {
                    queue.push_back((nm, shell + 1));
                    if cindex.len() > MAX_CELLS {
                        return Err(format!("more than {MAX_CELLS} cells; reduce shells"));
                    }
                }
            }
        }
        b.add_cell(faces, shell, opts.infinite_cells);
    }
    Ok(b.finish())
}

fn push_vertex(out: &mut CellComplex3d, real: &CoxeterRealisation, x: &Vec4, lorentz: bool, ideal: bool) {
    if !lorentz {
        let n = form(&real.eta, x, x).sqrt();
        out.coords4.push(x.map(|c| c / n));
        out.points.push([0.0; 3]);
        out.is_ideal.push(false);
        return;
    }
    if ideal {
        let y = x.map(|c| c / x[0]);
        let s = [y[1], y[2], y[3]];
        let n = super::norm3(&s);
        let s = s.map(|c| c / n); // exact unit sphere (removes round-off of the light-cone condition)
        out.coords4.push([1.0, s[0], s[1], s[2]]);
        out.points.push(s);
        out.klein.push(s);
        out.is_ideal.push(true);
    } else {
        let n = (-form(&real.eta, x, x)).sqrt();
        let y = x.map(|c| c / n);
        out.coords4.push(y);
        out.points.push([y[1] / (1.0 + y[0]), y[2] / (1.0 + y[0]), y[3] / (1.0 + y[0])]);
        out.klein.push([y[1] / y[0], y[2] / y[0], y[3] / y[0]]);
        out.is_ideal.push(false);
    }
}
