//! Twisted Z_2-equivariant Mobius seam (`Wc`) face extraction, 3D half-twist seeding,
//! and seam normal-alignment diagnostics for `sheet3d`.

use std::collections::{HashMap, HashSet};

use crate::defect::core_graph::TWO_PI;
use crate::defect::gauge::ribbon_faces;
use crate::defect::sheet3d_energy::{cross3, dot3, norm3, scale3, sub3};
use crate::defect::sym_helper::{dot, eigvec_max_sym, rot90};
use crate::plane::planarity::component_roots;
use crate::types::BuiltCrystal;

/// Extract interior bounded faces and boundary flags on a non-orientable (`Wc`) crystal.
/// Uses signed ribbon-dart walks (`ribbon_faces`) and drops the longest boundary walk(s) per component.
pub(crate) fn extract_mobius_faces(
    crystal: &BuiltCrystal,
) -> (Vec<Vec<usize>>, Vec<bool>, Vec<String>) {
    let n = crystal.pos.len();
    let walks = ribbon_faces(crystal);
    if walks.is_empty() {
        return (Vec::new(), vec![true; n], vec!["Mobius crystal has no ribbon faces".into()]);
    }
    let roots = component_roots(n, &crystal.edges);
    let mut by_comp: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, f) in walks.iter().enumerate() {
        if let Some(&u) = f.first() {
            by_comp.entry(roots[u]).or_default().push(i);
        }
    }

    let mut exterior = HashSet::new();
    let mut diagnostics = Vec::new();
    for indices in by_comp.values() {
        if indices.len() <= 1 {
            diagnostics.push("Mobius component has only one walk; no interior faces".into());
            exterior.extend(indices.iter().copied());
            continue;
        }
        let bnd_idx = *indices
            .iter()
            .max_by_key(|&&i| {
                let unique = walks[i].iter().copied().collect::<HashSet<_>>().len();
                (walks[i].len(), usize::from(unique < walks[i].len()))
            })
            .unwrap();
        exterior.insert(bnd_idx);
        for &i in indices {
            if i != bnd_idx
                && walks[i].iter().copied().collect::<HashSet<_>>().len() != walks[i].len()
            {
                diagnostics.push("Non-simple interior face walk on Mobius strip".into());
            }
        }
    }

    let mut boundary = vec![false; n];
    for &i in &exterior {
        for &u in &walks[i] {
            boundary[u] = true;
        }
    }
    let faces: Vec<Vec<usize>> = walks
        .into_iter()
        .enumerate()
        .filter(|(i, f)| !exterior.contains(i) && f.len() >= 3)
        .map(|(_, f)| f)
        .collect();

    let mut incidence = HashMap::<(usize, usize), usize>::new();
    let mut face_count = vec![0usize; n];
    for f in &faces {
        for m in 0..f.len() {
            let (u, v) = (f[m], f[(m + 1) % f.len()]);
            *incidence.entry((u.min(v), u.max(v))).or_default() += 1;
            face_count[u] += 1;
        }
    }
    for &(u, v) in &crystal.edges {
        if incidence.get(&(u.min(v), u.max(v))).copied().unwrap_or(0) < 2 {
            boundary[u] = true;
            boundary[v] = true;
        }
    }
    for u in 0..n {
        if face_count[u] == 0 {
            boundary[u] = true;
        }
    }
    (faces, boundary, diagnostics)
}

/// Smooth Z_2-twisted 3D Mobius strip seed derived directly from the seam isometry `(u, n)` frame.
pub(crate) fn seed_mobius_3d(crystal: &BuiltCrystal) -> Vec<[f64; 3]> {
    let n = crystal.pos.len();
    let seam_iso = crystal
        .adj
        .iter()
        .flatten()
        .filter_map(|e| e.seam.filter(|g| g.det() < 0.0))
        .next();
    let Some(g) = seam_iso else {
        return crystal.pos.iter().map(|&p| [p[0], p[1], 0.0]).collect();
    };

    let u_hat = eigvec_max_sym(g.m);
    let n_hat = rot90(u_hat);
    let tn = 0.5 * dot(g.t, n_hat);
    let x0 = [n_hat[0] * tn, n_hat[1] * tn];
    let ell = dot(g.t, u_hat).abs().max(1e-6);

    let a_vals_pos: Vec<f64> = crystal
        .pos
        .iter()
        .map(|&p| dot([p[0] - x0[0], p[1] - x0[1]], u_hat))
        .collect();
    let min_a = a_vals_pos.iter().copied().fold(f64::INFINITY, f64::min);
    let max_a = a_vals_pos.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let a_vals = if min_a < -0.1 * ell && max_a <= 0.1 * ell {
        a_vals_pos.into_iter().map(|a| -a).collect()
    } else {
        a_vals_pos
    };

    let w_vals: Vec<f64> = crystal
        .pos
        .iter()
        .map(|&p| dot([p[0] - x0[0], p[1] - x0[1]], n_hat))
        .collect();
    let w_max = w_vals.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
    let r_ring = (ell / TWO_PI).max(0.8 * w_max).max(1e-3);

    let mut xyz = Vec::with_capacity(n);
    for (&a, &w) in a_vals.iter().zip(&w_vals) {
        let theta = TWO_PI * (a / ell);
        let half = 0.5 * theta;
        let rad = r_ring + w * half.cos();
        xyz.push([rad * theta.cos(), rad * theta.sin(), w * half.sin()]);
    }
    xyz
}

/// Newell cyclic unit normal for one face polygon in 3D.
pub fn face_cyclic_normal(xyz: &[[f64; 3]], face: &[usize]) -> [f64; 3] {
    let k = face.len();
    if k < 3 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / (k as f64);
    let mut c = [0.0; 3];
    for &u in face {
        c[0] += xyz[u][0] * inv;
        c[1] += xyz[u][1] * inv;
        c[2] += xyz[u][2] * inv;
    }
    let mut n = [0.0; 3];
    for m in 0..k {
        let a = sub3(xyz[face[m]], c);
        let b = sub3(xyz[face[(m + 1) % k]], c);
        let cr = cross3(a, b);
        n[0] += cr[0];
        n[1] += cr[1];
        n[2] += cr[2];
    }
    let len = norm3(n).max(1e-15);
    scale3(n, 1.0 / len)
}

/// Minimum `tau * (n1 . n2)` across all shared interior edges, where `tau = -(dir1 * dir2)`.
/// Values near `+1.0` confirm crease-free Z_2-twisted bending across the Mobius seam.
pub fn min_twisted_hinge_alignment(xyz: &[[f64; 3]], faces: &[Vec<usize>]) -> Option<f64> {
    let normals: Vec<[f64; 3]> = faces.iter().map(|f| face_cyclic_normal(xyz, f)).collect();
    let mut edge_inc: HashMap<(usize, usize), Vec<(usize, i32)>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        let k = f.len();
        for m in 0..k {
            let (u, v) = (f[m], f[(m + 1) % k]);
            let key = (u.min(v), u.max(v));
            let dir = if u < v { 1 } else { -1 };
            edge_inc.entry(key).or_default().push((fi, dir));
        }
    }
    let mut worst: Option<f64> = None;
    for inc in edge_inc.values() {
        if inc.len() == 2 && inc[0].0 != inc[1].0 {
            let tau = -(inc[0].1 * inc[1].1) as f64;
            let align = tau * dot3(normals[inc[0].0], normals[inc[1].0]);
            worst = Some(worst.map_or(align, |w| w.min(align)));
        }
    }
    worst
}
