//! G-gauge-invariant GSV metric elasticity model construction, evaluation, and diagnostics.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::defect::charge::check_frames;
use crate::defect::cone_geometry::{dot2, mat2_vec, norm2, sub2};
use crate::defect::gsv_mobius::extract_mobius_faces;
use crate::defect::sheet3d_topology::{extract_interior_faces, prepare_planar_chart, validate_crystal};
use crate::defect::sym_helper::mat2_mul;
use crate::types::{BuiltCrystal, Cell2D};

#[derive(Clone, Copy, Debug)]
pub(crate) struct StretchTerm {
    pub edge_idx: usize,
    pub u: usize,
    pub v: usize,
    pub l0: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShearTerm {
    pub u: usize,
    pub v: usize,
    pub w: usize,
    pub c_bar: f64,
    pub alpha: f64,
    pub edge_uv: usize,
    pub edge_uw: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct GsvModel {
    pub n: usize,
    pub edge_rest: Vec<f64>,
    pub stretches: Vec<StretchTerm>,
    pub shears_2d: Vec<ShearTerm>,
    pub shears_3d: Vec<ShearTerm>,
    pub faces: Vec<Vec<usize>>,
    pub is_boundary: Vec<bool>,
    pub mean_bond_len: f64,
    pub chart_2d: Vec<[f64; 2]>,
    pub has_valid_frames: bool,
    pub diagnostics: Vec<String>,
}

/// Options for 2D GSV metric elastic relaxation [`crate::defect::gsv_elastic::relax_elastic_2d_with_opts`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Elastic2DOptions {
    pub max_gn_iters: usize,
    pub max_newton_iters: usize,
    pub grad_tol: f64,
}

impl Default for Elastic2DOptions {
    fn default() -> Self {
        Self { max_gn_iters: 80, max_newton_iters: 50, grad_tol: 1e-12 }
    }
}

/// Relaxed 2D G-gauge-invariant GSV elastic equilibrium and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Elastic2DResult {
    pub pos: Vec<[f64; 2]>,
    pub total_energy: f64,
    pub stretch_energy: f64,
    pub shear_energy: f64,
    pub bond_strain: Vec<f64>,
    pub bond_shear_strain: Vec<f64>,
    pub bond_gsv_energy: Vec<f64>,
    pub site_gsv_energy: Vec<f64>,
    pub site_shear: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

pub(crate) fn rest_lengths_from_cell(cell: &Cell2D) -> HashMap<usize, f64> {
    let mut sums: HashMap<usize, (f64, usize)> = HashMap::new();
    for b in &cell.bonds {
        if let (Some(&pi), Some(&pj)) = (cell.basis.get(b.i), cell.basis.get(b.j)) {
            let dx = f64::from(b.n1) * cell.a1[0] + f64::from(b.n2) * cell.a2[0] + pj[0] - pi[0];
            let dy = f64::from(b.n1) * cell.a1[1] + f64::from(b.n2) * cell.a2[1] + pj[1] - pi[1];
            let len = dx.hypot(dy);
            if len > 1e-9 {
                let ent = sums.entry(b.kind).or_insert((0.0, 0));
                ent.0 += len;
                ent.1 += 1;
            }
        }
    }
    sums.into_iter().map(|(k, (s, c))| (k, s / (c as f64))).collect()
}

/// Develop corner `(w_{m-1}, u, w_{m+1})` around `face \ {u}` when the Cartan loop closes at `u`.
pub(crate) fn develop_cone_corner_c_bar(
    crystal: &BuiltCrystal,
    face: &[usize],
    m: usize,
) -> Option<f64> {
    let k = face.len();
    if k < 3 {
        return None;
    }
    let u = face[m];
    let w_next = face[(m + 1) % k];
    let w_prev = face[(m + k - 1) % k];
    let d_plus = crystal.adj[u].iter().find(|b| b.to == w_next)?.vec;
    let mut f_mat = [[1.0, 0.0], [0.0, 1.0]];
    let mut cur = [0.0_f64, 0.0];
    let mut v_back = [0.0_f64, 0.0];
    for step in 0..k {
        let a = face[(m + step) % k];
        let b = face[(m + step + 1) % k];
        let e = crystal.adj[a].iter().find(|x| x.to == b)?;
        let dv = mat2_vec(f_mat, e.vec);
        cur[0] += dv[0];
        cur[1] += dv[1];
        if step + 1 == k {
            v_back = dv;
        } else if let Some(g) = e.seam {
            f_mat = mat2_mul(f_mat, g.inverse().m);
        }
    }
    let _ = w_prev;
    if norm2(cur) > 1e-4 {
        return None;
    }
    Some(dot2(d_plus, [-v_back[0], -v_back[1]]))
}

impl GsvModel {
    pub(crate) fn from_crystal(
        crystal: &BuiltCrystal,
        cell: Option<&Cell2D>,
    ) -> Result<Self, String> {
        validate_crystal(crystal)?;
        let n = crystal.pos.len();
        let kind_len = cell.map(rest_lengths_from_cell).unwrap_or_default();
        let mut edge_rest = vec![0.0_f64; crystal.edges.len()];
        let mut edge_map = HashMap::with_capacity(crystal.edges.len());
        let mut stretches = Vec::with_capacity(crystal.edges.len());
        let mut len_sum = 0.0_f64;

        for (e_idx, &(u, v)) in crystal.edges.iter().enumerate() {
            edge_map.insert((u.min(v), u.max(v)), e_idx);
            let d_opt = crystal.adj[u].iter().find(|d| d.to == v);
            let l0 = match d_opt {
                Some(d) if norm2(d.vec) > 1e-6 => norm2(d.vec),
                Some(d) => kind_len.get(&d.kind).copied().unwrap_or(1.0).max(1e-6),
                None => kind_len.values().next().copied().unwrap_or(1.0).max(1e-6),
            };
            edge_rest[e_idx] = l0;
            len_sum += l0;
            stretches.push(StretchTerm { edge_idx: e_idx, u, v, l0 });
        }
        let mean_bond_len =
            if edge_rest.is_empty() { 1.0 } else { len_sum / (edge_rest.len() as f64) };
        if !mean_bond_len.is_finite() || edge_rest.iter().any(|l| !l.is_finite() || *l <= 0.0) {
            return Err("nonfinite physical bond rest lengths".into());
        }

        let max_init_rel_strain = stretches
            .iter()
            .map(|s| (norm2(sub2(crystal.pos[s.u], crystal.pos[s.v])) - s.l0).abs() / s.l0)
            .fold(0.0_f64, f64::max);
        let has_flip_seam =
            crystal.adj.iter().flatten().any(|e| e.seam.is_some_and(|g| g.det() < 0.0));
        let has_valid_frames = !crystal.is_non_orientable
            && !has_flip_seam
            && crystal.wall_sigma.is_none()
            && check_frames(crystal) == 0;
        let safe_chart_repair = crystal.pos.iter().flatten().all(|x| x.abs() < 1e12);
        let chart_2d =
            if max_init_rel_strain < 1e-7 || crystal.is_non_orientable || !safe_chart_repair {
                crystal.pos.clone()
            } else {
                prepare_planar_chart(crystal, has_valid_frames)
            };
        if chart_2d.iter().flatten().any(|x| !x.is_finite()) {
            return Err("nonfinite planar chart".into());
        }

        let mut diagnostics = Vec::new();
        let (faces, is_boundary) = if crystal.is_non_orientable || has_flip_seam {
            let (f, b, notes) = extract_mobius_faces(crystal);
            diagnostics.extend(notes);
            (f, b)
        } else {
            let (f, b, notes) = extract_interior_faces(crystal, &chart_2d, has_valid_frames);
            diagnostics.extend(notes);
            (f, b)
        };

        let dx_local = |u: usize, v: usize, e_idx: usize| -> [f64; 2] {
            let d = crystal.adj[u].iter().find(|b| b.to == v).expect("bond exists");
            if norm2(d.vec) > 1e-6 {
                d.vec
            } else {
                let dv = sub2(chart_2d[v], chart_2d[u]);
                let nrm = norm2(dv).max(1e-12);
                let l0 = edge_rest[e_idx];
                [dv[0] * l0 / nrm, dv[1] * l0 / nrm]
            }
        };

        let cone_set: HashSet<usize> = crystal.cone_sites.iter().copied().collect();
        let mut shears_3d = Vec::new();
        for f in &faces {
            let k = f.len();
            for m in 0..k {
                let (u, v, w) = (f[m], f[(m + 1) % k], f[(m + k - 1) % k]);
                if v == w {
                    continue;
                }
                let e_uv = edge_map[&(u.min(v), u.max(v))];
                let e_uw = edge_map[&(u.min(w), u.max(w))];
                let c_bar = if cone_set.contains(&u) {
                    develop_cone_corner_c_bar(crystal, f, m)
                        .unwrap_or_else(|| dot2(dx_local(u, v, e_uv), dx_local(u, w, e_uw)))
                } else {
                    dot2(dx_local(u, v, e_uv), dx_local(u, w, e_uw))
                };
                shears_3d.push(ShearTerm {
                    u,
                    v,
                    w,
                    c_bar,
                    alpha: 0.5 / (edge_rest[e_uv] * edge_rest[e_uw]).sqrt(),
                    edge_uv: e_uv,
                    edge_uw: e_uw,
                });
            }
        }

        let mut shears_2d = Vec::new();
        for u in 0..n {
            if cone_set.contains(&u) {
                continue;
            }
            let row = &crystal.adj[u];
            for a in 0..row.len() {
                let v = row[a].to;
                let e_uv = edge_map[&(u.min(v), u.max(v))];
                let dv = dx_local(u, v, e_uv);
                for row_b in &row[(a + 1)..] {
                    let w = row_b.to;
                    let e_uw = edge_map[&(u.min(w), u.max(w))];
                    let dw = dx_local(u, w, e_uw);
                    shears_2d.push(ShearTerm {
                        u,
                        v,
                        w,
                        c_bar: dot2(dv, dw),
                        alpha: 0.5 / (edge_rest[e_uv] * edge_rest[e_uw]).sqrt(),
                        edge_uv: e_uv,
                        edge_uw: e_uw,
                    });
                }
            }
        }
        if !cone_set.is_empty() {
            shears_2d.extend(shears_3d.iter().filter(|t| cone_set.contains(&t.u)).copied());
        }

        Ok(Self {
            n,
            edge_rest,
            stretches,
            shears_2d,
            shears_3d,
            faces,
            is_boundary,
            mean_bond_len,
            chart_2d,
            has_valid_frames,
            diagnostics,
        })
    }
}

pub(crate) fn eval_gsv_fg<const D: usize>(
    x: &[f64],
    grad: &mut [f64],
    stretches: &[StretchTerm],
    shears: &[ShearTerm],
) -> (f64, f64) {
    grad.fill(0.0);
    let mut e_str = 0.0;
    let mut e_shr = 0.0;
    for s in stretches {
        let mut dy = [0.0_f64; D];
        let mut n2 = 0.0;
        for k in 0..D {
            dy[k] = x[D * s.v + k] - x[D * s.u + k];
            n2 += dy[k] * dy[k];
        }
        let r = (n2 - s.l0 * s.l0) / (2.0 * s.l0);
        e_str += 0.5 * r * r;
        let c = r / s.l0;
        for k in 0..D {
            let g = c * dy[k];
            grad[D * s.v + k] += g;
            grad[D * s.u + k] -= g;
        }
    }
    for sh in shears {
        let mut d1 = [0.0_f64; D];
        let mut d2 = [0.0_f64; D];
        let mut dot12 = 0.0;
        for k in 0..D {
            d1[k] = x[D * sh.v + k] - x[D * sh.u + k];
            d2[k] = x[D * sh.w + k] - x[D * sh.u + k];
            dot12 += d1[k] * d2[k];
        }
        let r = sh.alpha * (dot12 - sh.c_bar);
        e_shr += 0.5 * r * r;
        let c = sh.alpha * r;
        for k in 0..D {
            let gv = c * d2[k];
            let gw = c * d1[k];
            grad[D * sh.v + k] += gv;
            grad[D * sh.w + k] += gw;
            grad[D * sh.u + k] -= gv + gw;
        }
    }
    (e_str, e_shr)
}

pub(crate) struct GsvDiagnostics {
    pub stretch_energy: f64,
    pub shear_energy: f64,
    pub bond_strain: Vec<f64>,
    pub bond_shear_strain: Vec<f64>,
    pub bond_gsv_energy: Vec<f64>,
    pub site_gsv_energy: Vec<f64>,
    pub site_shear: Vec<f64>,
}

pub(crate) fn compute_gsv_diagnostics<const D: usize>(
    x: &[f64],
    model: &GsvModel,
    shears: &[ShearTerm],
) -> GsvDiagnostics {
    let e_cnt = model.stretches.len();
    let mut bond_strain = vec![0.0_f64; e_cnt];
    let mut bond_gsv_energy = vec![0.0_f64; e_cnt];
    let mut bond_shear_sq = vec![0.0_f64; e_cnt];
    let mut bond_shear_cnt = vec![0_usize; e_cnt];
    let mut site_gsv_energy = vec![0.0_f64; model.n];
    let mut site_shear = vec![0.0_f64; model.n];
    let mut e_str = 0.0;
    let mut e_shr = 0.0;

    for s in &model.stretches {
        let mut n2 = 0.0;
        for k in 0..D {
            let d = x[D * s.v + k] - x[D * s.u + k];
            n2 += d * d;
        }
        let len = n2.sqrt();
        bond_strain[s.edge_idx] = (len - s.l0) / s.l0;
        let r = (n2 - s.l0 * s.l0) / (2.0 * s.l0);
        let es = 0.5 * r * r;
        e_str += es;
        bond_gsv_energy[s.edge_idx] += es;
        site_gsv_energy[s.u] += 0.5 * es;
        site_gsv_energy[s.v] += 0.5 * es;
    }
    for sh in shears {
        let mut dot12 = 0.0;
        for k in 0..D {
            dot12 += (x[D * sh.v + k] - x[D * sh.u + k]) * (x[D * sh.w + k] - x[D * sh.u + k]);
        }
        let r = sh.alpha * (dot12 - sh.c_bar);
        let es = 0.5 * r * r;
        e_shr += es;
        site_shear[sh.u] += es;
        site_gsv_energy[sh.u] += es;
        bond_gsv_energy[sh.edge_uv] += 0.5 * es;
        bond_gsv_energy[sh.edge_uw] += 0.5 * es;
        let eps_sh = 2.0 * sh.alpha * r;
        bond_shear_sq[sh.edge_uv] += eps_sh * eps_sh;
        bond_shear_cnt[sh.edge_uv] += 1;
        bond_shear_sq[sh.edge_uw] += eps_sh * eps_sh;
        bond_shear_cnt[sh.edge_uw] += 1;
    }
    let bond_shear_strain: Vec<f64> = bond_shear_sq
        .into_iter()
        .zip(bond_shear_cnt)
        .map(|(s2, c)| if c > 0 { (s2 / (c as f64)).sqrt() } else { 0.0 })
        .collect();
    GsvDiagnostics {
        stretch_energy: e_str,
        shear_energy: e_shr,
        bond_strain,
        bond_shear_strain,
        bond_gsv_energy,
        site_gsv_energy,
        site_shear,
    }
}
