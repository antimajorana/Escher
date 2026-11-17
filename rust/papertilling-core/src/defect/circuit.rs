//! Exact Burgers-circuit parallel-transport engine and basepoint rotation,
//! matching `papertilling/defects/circuit.py`.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::defect::combine_types::ClassifiedCharge;
use crate::defect::core_graph::{mat2_vec, norm2, sub2};
use crate::defect::equivalence::classify_holonomy_charge;
use crate::defect::sym_helper::round_dec;
use crate::types::{BuiltCrystal, Cell2D, Frame2D, Isometry2};

pub use crate::defect::circuit_auto::auto_burgers_circuit;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CircuitResult {
    pub path: Vec<usize>,
    pub ref_path: Vec<[f64; 2]>,
    pub frames: Vec<[[f64; 2]; 2]>,
    pub cum_angle_deg: Vec<f64>,
    pub det_signs: Vec<i32>,
    pub orientation_preserving: bool,
    pub h: Isometry2,
    pub closure_vector: [f64; 2],
    pub burgers_norm: f64,
    pub frank_angle_deg: Option<f64>,
    pub charge: Option<ClassifiedCharge>,
}

pub fn shift_circuit_basepoint(path: &[usize], new_v0: usize) -> Result<Vec<usize>, String> {
    let core = if path.len() >= 4 && path.first() == path.last() { &path[..path.len() - 1] } else { path };
    if core.len() < 3 {
        return Err("circuit path must have at least 3 distinct edges".to_string());
    }
    let idx = core.iter().position(|&v| v == new_v0).ok_or_else(|| format!("vertex {new_v0} does not lie on the circuit path"))?;
    let mut rot = Vec::with_capacity(core.len() + 1);
    rot.extend_from_slice(&core[idx..]);
    rot.extend_from_slice(&core[..idx]);
    rot.push(rot[0]);
    Ok(rot)
}

fn frank_from_walk(crystal: &BuiltCrystal, verts: &[usize], vecs: &[[f64; 2]], gs: &[Option<Isometry2>], h_rot: [[f64; 2]; 2]) -> f64 {
    let m = vecs.len();
    let mut turns = 0.0;
    for k in 0..m {
        let prev = (k + m - 1) % m;
        let vi = gs[prev].map_or(vecs[prev], |g| mat2_vec(g.m, vecs[prev]));
        let vo = vecs[k];
        turns += (vi[0] * vo[1] - vi[1] * vo[0]).atan2(vi[0] * vo[0] + vi[1] * vo[1]).to_degrees();
    }
    let area: f64 = (0..m).map(|k| {
        let (a, b) = (crystal.pos[verts[k]], crystal.pos[verts[k + 1]]);
        a[0] * b[1] - a[1] * b[0]
    }).sum();
    let omega_gb = 360.0 * if area >= 0.0 { 1.0 } else { -1.0 } - turns;
    let rot_deg = h_rot[1][0].atan2(h_rot[0][0]).to_degrees();
    round_dec(rot_deg + 360.0 * ((omega_gb - rot_deg) / 360.0).round(), 6)
}

pub fn burgers_circuit(
    crystal: &BuiltCrystal,
    path: &[usize],
    cell_info: Option<(&str, &Cell2D, Frame2D)>,
) -> Result<CircuitResult, String> {
    if path.len() < 4 || path.first() != path.last() {
        return Err("path must be a closed walk [v0, v1, ..., v0] with at least 3 bonds".to_string());
    }
    let (mut f, mut h) = (Isometry2::IDENTITY, Isometry2::IDENTITY);
    let (mut ref_path, mut frames, mut cum_angle_deg, mut det_signs) = (vec![[0.0, 0.0]], vec![f.m], vec![0.0], vec![1i32]);
    let (mut vecs, mut gs) = (Vec::with_capacity(path.len() - 1), Vec::with_capacity(path.len() - 1));
    for w in path.windows(2) {
        let (u, v) = (w[0], w[1]);
        let e = crystal.adj.get(u).and_then(|lst| lst.iter().find(|b| b.to == v)).ok_or_else(|| format!("circuit traverses missing directed bond ({u}, {v})"))?;
        vecs.push(e.vec);
        gs.push(e.seam);
        let step = mat2_vec(f.m, e.vec);
        let prev = *ref_path.last().unwrap();
        ref_path.push([prev[0] + step[0], prev[1] + step[1]]);
        let mut d_ang = 0.0;
        if let Some(g) = e.seam {
            f = f.compose(&g.inverse());
            h = g.compose(&h);
            if g.det() > 0.0 && *det_signs.last().unwrap() > 0 {
                d_ang = g.m[1][0].atan2(g.m[0][0]).to_degrees();
            }
        }
        frames.push(f.m);
        let det_s = if f.det() > 0.0 { 1 } else { -1 };
        det_signs.push(det_s);
        cum_angle_deg.push(*cum_angle_deg.last().unwrap() + if det_s > 0 { d_ang } else { 0.0 });
    }
    let orientation_preserving = *det_signs.last().unwrap() > 0;
    let frank_angle_deg = if orientation_preserving {
        let fd = frank_from_walk(crystal, path, &vecs, &gs, h.m);
        let shift = fd - *cum_angle_deg.last().unwrap();
        if shift.abs() > 1.0 {
            let last_seam = gs.iter().rposition(Option::is_some).unwrap_or(gs.len() - 1);
            for val in &mut cum_angle_deg[(last_seam + 1)..] {
                *val += shift;
            }
        }
        Some(fd)
    } else {
        None
    };
    let mut closure_vector = sub2(*ref_path.last().unwrap(), ref_path[0]);
    for c in &mut closure_vector {
        if c.abs() < 1e-11 { *c = 0.0; }
    }
    let burgers_norm = norm2(closure_vector);
    let charge = match cell_info {
        Some((group, cell, frame)) => Some(classify_holonomy_charge(group, cell, frame, frank_angle_deg, h, closure_vector)?),
        None => None,
    };
    Ok(CircuitResult {
        path: path.to_vec(), ref_path, frames, cum_angle_deg, det_signs,
        orientation_preserving, h, closure_vector, burgers_norm, frank_angle_deg, charge,
    })
}

pub(super) fn mobius_seam_circuit(crystal: &BuiltCrystal, cell_info: Option<(&str, &Cell2D, Frame2D)>) -> Result<CircuitResult, String> {
    let mut adj = vec![Vec::new(); crystal.adj.len()];
    let mut seeds = Vec::new();
    for (u, lst) in crystal.adj.iter().enumerate() {
        for e in lst {
            let sgn = if e.seam.is_none_or(|g| g.det() > 0.0) { 1i32 } else { -1 };
            adj[u].push((e.to, sgn));
            if sgn == -1 { seeds.push(u); }
        }
        adj[u].sort_unstable();
    }
    let &start = seeds.first().ok_or("non-orientable defect has no orientation-reversing seam edge")?;
    let mut parent: BTreeMap<(usize, i32), Option<(usize, i32)>> = BTreeMap::from([((start, 1), None)]);
    let mut dq = VecDeque::from([(start, 1i32)]);
    let goal = (start, -1i32);
    while !parent.contains_key(&goal) && let Some((u, s)) = dq.pop_front() {
        for &(v, es) in &adj[u] {
            let nxt = (v, s * es);
            if let std::collections::btree_map::Entry::Vacant(e) = parent.entry(nxt) {
                e.insert(Some((u, s)));
                dq.push_back(nxt);
            }
        }
    }
    if !parent.contains_key(&goal) {
        return Err("could not find an orientation-reversing closed loop".to_string());
    }
    let (mut walk, mut cur) = (Vec::new(), Some(goal));
    while let Some(st) = cur { walk.push(st.0); cur = parent[&st]; }
    walk.reverse();
    burgers_circuit(crystal, &walk, cell_info)
}
