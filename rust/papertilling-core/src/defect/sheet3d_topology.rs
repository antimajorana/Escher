//! Validation and face extraction for the elastic-sheet view, without changing the input graph.

use std::collections::{HashMap, HashSet};

use crate::defect::sheet3d::Sheet3DOptions;
use crate::defect::charge::{frame_faces, frank_angles};
use crate::plane::draw::relax_preview;
use crate::plane::planarity::{self, check_positions, component_roots, signed_area, split_frame_faces};
use crate::plane::tutte::{drawn_boundary_loop, global_tutte, local_tutte};
use crate::types::{BuiltCrystal, Cell2D, Isometry2};

pub(super) fn validate_inputs(
    crystal: &BuiltCrystal,
    cell: &Cell2D,
    opts: &Sheet3DOptions,
    warm_z: Option<&[f64]>,
) -> Result<(), String> {
    validate_crystal(crystal)?;
    if !opts.kappa_b.is_finite()
        || opts.kappa_b < 0.0
        || !opts.grad_tol.is_finite()
        || opts.grad_tol < 0.0
    {
        return Err("bending modulus and gradient tolerance must be finite and nonnegative".into());
    }
    if warm_z.is_some_and(|w| w.len() != crystal.pos.len() || w.iter().any(|z| !z.is_finite())) {
        return Err("warm_z must contain one finite height per vertex".into());
    }
    if crystal.pos.is_empty() {
        return Ok(());
    }
    let det = cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0];
    if !det.is_finite()
        || det == 0.0
        || cell.basis.is_empty()
        || cell
            .a1
            .iter()
            .chain(&cell.a2)
            .chain(cell.basis.iter().flatten())
            .any(|x| !x.is_finite())
        || cell
            .bonds
            .iter()
            .any(|b| b.i >= cell.basis.len() || b.j >= cell.basis.len())
    {
        return Err("invalid finite lattice cell or bond indices".into());
    }
    for b in &cell.bonds {
        let pi = cell.basis[b.i];
        let pj = cell.basis[b.j];
        let dx = f64::from(b.n1) * cell.a1[0] + f64::from(b.n2) * cell.a2[0] + pj[0] - pi[0];
        let dy = f64::from(b.n1) * cell.a1[1] + f64::from(b.n2) * cell.a2[1] + pj[1] - pi[1];
        let length = dx.hypot(dy);
        if !length.is_finite() || length <= 0.0 {
            return Err("lattice bonds must have finite positive rest lengths".into());
        }
    }
    Ok(())
}

/// Validate the indexed crystal structure before frame walks, numerical work or GPU conversion.
/// Drawing-only ghost marks are deliberately not used to decide which bonds exist.
pub fn validate_crystal(crystal: &BuiltCrystal) -> Result<(), String> {
    let n = crystal.pos.len();
    if crystal.ref_pos.len() != n
        || crystal.adj.len() != n
        || crystal.side.len() != n
        || crystal.basis_idx.len() != n
    {
        return Err("inconsistent per-vertex crystal arrays".into());
    }
    if crystal
        .pos
        .iter()
        .chain(&crystal.ref_pos)
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err("nonfinite crystal coordinates".into());
    }
    if crystal.cone_sites.iter().any(|&u| u >= n) {
        return Err("invalid cone-site index".into());
    }
    if crystal.wall_sigma.is_some_and(|g| !valid_isometry(g)) {
        return Err("invalid wall isometry".into());
    }
    let mut edges = HashSet::with_capacity(crystal.edges.len());
    for &(u, v) in &crystal.edges {
        if u >= n || v >= n || u == v || !edges.insert((u.min(v), u.max(v))) {
            return Err("invalid or duplicate crystal edge".into());
        }
    }
    let mut directed = HashSet::with_capacity(crystal.edges.len().saturating_mul(2));
    for (u, row) in crystal.adj.iter().enumerate() {
        for e in row {
            if e.to >= n
                || e.to == u
                || !directed.insert((u, e.to))
                || !edges.contains(&(u.min(e.to), u.max(e.to)))
            {
                return Err("invalid, duplicate or unlisted adjacency bond".into());
            }
            if e.vec.iter().any(|x| !x.is_finite())
                || e.ang.is_some_and(|a| !a.is_finite())
                || e.seam.is_some_and(|g| !valid_isometry(g))
            {
                return Err("nonfinite or invalid bond reference frame".into());
            }
        }
    }
    if edges
        .iter()
        .any(|&(u, v)| !directed.contains(&(u, v)) || !directed.contains(&(v, u)))
    {
        return Err("missing reciprocal adjacency bond".into());
    }
    Ok(())
}

fn valid_isometry(g: Isometry2) -> bool {
    g.m.iter().flatten().chain(&g.t).all(|x| x.is_finite())
        && (g.m[0][0].hypot(g.m[0][1]) - 1.0).abs() < 1e-6
        && (g.m[1][0].hypot(g.m[1][1]) - 1.0).abs() < 1e-6
        && (g.m[0][0] * g.m[1][0] + g.m[0][1] * g.m[1][1]).abs() < 1e-6
}

/// All bounded face walks, with no face-size or walk-length cutoff.
/// The trusted frame walk supplies the rotation system. An open point-defect disk has an exterior
/// Frank angle greater than a full turn (720 degrees minus the unreduced interior charge).
/// Signed chart area is a second way to identify the exterior, including frameless planar walls.
/// Ambiguous components are diagnosed rather than deleting their longest, potentially core, ring.
pub(super) fn extract_interior_faces(
    crystal: &BuiltCrystal,
    chart: &[[f64; 2]],
    use_frames: bool,
) -> (Vec<Vec<usize>>, Vec<bool>, Vec<String>) {
    let n = crystal.pos.len();
    let mut drawn;
    let topology = if use_frames {
        crystal
    } else {
        drawn = crystal.clone();
        for (u, row) in drawn.adj.iter_mut().enumerate() {
            for e in row {
                e.vec = [chart[e.to][0] - chart[u][0], chart[e.to][1] - chart[u][1]];
                e.ang = None;
            }
        }
        &drawn
    };
    let walks = frame_faces(topology);
    let roots = component_roots(n, &crystal.edges);
    let mut euler = HashMap::<usize, isize>::new();
    for &root in &roots {
        *euler.entry(root).or_default() += 1;
    }
    for &(u, _) in &crystal.edges {
        *euler.entry(roots[u]).or_default() -= 1;
    }
    let mut by_component: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, f) in walks.iter().enumerate() {
        if let Some(&u) = f.first() {
            by_component.entry(roots[u]).or_default().push(i);
            *euler.entry(roots[u]).or_default() += 1;
        }
    }
    let areas: Vec<f64> = walks.iter().map(|f| signed_area(chart, f)).collect();
    let frank = use_frames.then(|| frank_angles(topology, &walks));
    let mut exterior = HashSet::new();
    let mut diagnostics = Vec::new();
    for indices in by_component.values() {
        if euler[&roots[walks[indices[0]][0]]] != 2 {
            diagnostics.push("Non-disk face rotation system: spring-only preview for this component; surface topology unavailable".into());
            exterior.extend(indices.iter().copied());
            continue;
        }
        let charge_candidates: Vec<_> = indices
            .iter()
            .copied()
            .filter(|&i| {
                frank
                    .as_ref()
                    .is_some_and(|angles| angles[i] > 360.0 + 1e-6)
            })
            .collect();
        let largest_area = *indices
            .iter()
            .max_by(|&&a, &&b| areas[a].abs().total_cmp(&areas[b].abs()))
            .unwrap();
        let area = areas[largest_area];
        let geometric_exterior = area.is_finite()
            && area != 0.0
            && indices.iter().filter(|&&i| i != largest_area).all(|&i| {
                areas[i].is_finite() && (areas[i] == 0.0 || areas[i].signum() != area.signum())
            });
        let outer = if charge_candidates.len() == 1 {
            charge_candidates[0]
        } else if geometric_exterior {
            largest_area
        } else {
            diagnostics.push("Ambiguous exterior face: spring-only preview for this component; surface topology unavailable".into());
            exterior.extend(indices.iter().copied());
            continue;
        };
        if indices.iter().any(|&i| {
            i != outer && walks[i].iter().copied().collect::<HashSet<_>>().len() != walks[i].len()
        }) {
            diagnostics.push("Non-simple face walk: spring-only preview for this component; polygon surface unavailable".into());
            exterior.extend(indices.iter().copied());
            continue;
        }
        exterior.insert(outer);
    }
    let mut boundary = vec![false; n];
    for &i in &exterior {
        for &u in &walks[i] {
            boundary[u] = true;
        }
    }
    let faces: Vec<_> = walks
        .into_iter()
        .enumerate()
        .filter(|(i, f)| !exterior.contains(i) && f.len() >= 3)
        .map(|(_, f)| f)
        .collect();
    let mut incidence = HashMap::<(usize, usize), usize>::new();
    let mut face_count = vec![0_usize; n];
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
    diagnostics.sort();
    diagnostics.dedup();
    (faces, boundary, diagnostics)
}

/// Deterministic unfolded planar chart for initializing 2D reference and 3D sheet optimization.
pub(super) fn prepare_planar_chart(crystal: &BuiltCrystal, has_valid_frames: bool) -> Vec<[f64; 2]> {
    let mut tmp = crystal.clone();
    for e in tmp.adj.iter_mut().flatten() {
        e.ghost_stub = None;
    }
    let chk = planarity::check(&tmp);
    if chk.crossings == 0 && chk.overlaps == 0 && chk.folded.unwrap_or(0) == 0 {
        return tmp.pos;
    }
    relax_preview(&mut tmp, f64::INFINITY);
    for e in tmp.adj.iter_mut().flatten() {
        e.ghost_stub = None;
    }
    let chk2 = planarity::check(&tmp);
    if chk2.crossings == 0 && chk2.folded.unwrap_or(0) == 0 {
        return tmp.pos;
    }

    let (inner_faces, outer_rims) = if has_valid_frames {
        split_frame_faces(&tmp)
    } else {
        (Vec::new(), vec![drawn_boundary_loop(&tmp, &tmp.pos)])
    };
    let inner_opt = has_valid_frames.then_some(inner_faces.as_slice());
    let rim_set: HashSet<usize> = outer_rims.iter().flatten().copied().collect();
    if !chk2.bad_sites.is_empty() {
        let (p_loc, nfree) = local_tutte(&tmp, &tmp.pos, &chk2.bad_sites, 4, &rim_set);
        let max_frac = if !rim_set.is_empty() { 0.9 } else { 0.5 };
        if (nfree as f64) <= max_frac * (tmp.pos.len() as f64) {
            let c_loc = check_positions(&tmp, &p_loc, false, inner_opt);
            if c_loc.crossings == 0 && c_loc.folded.unwrap_or(0) == 0 {
                let mut relaxed = tmp.clone();
                relaxed.pos = p_loc.clone();
                relax_preview(&mut relaxed, f64::INFINITY);
                let c_rel = check_positions(&relaxed, &relaxed.pos, false, inner_opt);
                return if c_rel.crossings == 0 && c_rel.folded.unwrap_or(0) == 0 {
                    relaxed.pos
                } else {
                    p_loc
                };
            }
        }
    }

    let (mut p_tut, _) = global_tutte(&tmp, &outer_rims, &crystal.pos, 1.0);
    if !inner_faces.is_empty() {
        let sign_sum = |pts: &[[f64; 2]]| -> isize {
            inner_faces
                .iter()
                .map(|f| {
                    let a = signed_area(pts, f);
                    if a > 0.0 { 1 } else if a < 0.0 { -1 } else { 0 }
                })
                .sum()
        };
        let s0 = sign_sum(&crystal.pos).signum();
        if s0 != 0 && sign_sum(&p_tut).signum() == -s0 {
            (p_tut, _) = global_tutte(&tmp, &outer_rims, &crystal.pos, -1.0);
        }
    }
    let c_tut = check_positions(&tmp, &p_tut, false, inner_opt);
    tmp.pos = p_tut.clone();
    relax_preview(&mut tmp, f64::INFINITY);
    let c_rel = check_positions(&tmp, &tmp.pos, false, inner_opt);
    if (c_rel.crossings, c_rel.folded.unwrap_or(0)) <= (c_tut.crossings, c_tut.folded.unwrap_or(0)) {
        tmp.pos
    } else {
        p_tut
    }
}
