//! Discrete stretching + orientation-safe hinge bending energy and analytic gradients for `sheet3d`.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::defect::gsv_model::{GsvModel, StretchTerm, eval_gsv_fg};
use crate::types::BuiltCrystal;

#[derive(Clone, Copy)]
pub(crate) enum HingePt {
    Vert(usize),
    FaceCentroid(usize),
}

#[derive(Clone, Copy)]
pub(crate) struct Hinge {
    pub p: HingePt,
    pub q: HingePt,
    pub opp_a: HingePt,
    pub opp_b: HingePt,
    pub weight: f64,
    pub sign: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct Spring {
    pub u: usize,
    pub v: usize,
    pub l0: f64,
    pub weight: f64,
    #[allow(dead_code)]
    pub is_primary_bond: bool,
}

type EdgeIncidence = Vec<(usize, HingePt, i32, usize, usize)>;

pub(crate) fn build_hinges(faces: &[Vec<usize>]) -> Vec<Hinge> {
    let mut edge_map: HashMap<(usize, usize), EdgeIncidence> = HashMap::new();
    let mut hinges = Vec::new();

    for (fi, f) in faces.iter().enumerate() {
        let k = f.len();
        for m in 0..k {
            let u = f[m];
            let v = f[(m + 1) % k];
            let opp = if k == 3 {
                HingePt::Vert(f[(m + 2) % 3])
            } else {
                HingePt::FaceCentroid(fi)
            };
            let dir = if u < v { 1 } else { -1 };
            edge_map
                .entry((u.min(v), u.max(v)))
                .or_default()
                .push((fi, opp, dir, u, v));
        }
        if k >= 4 {
            let c_pt = HingePt::FaceCentroid(fi);
            for m in 0..k {
                let prev = f[(m + k - 1) % k];
                let curr = f[m];
                let next = f[(m + 1) % k];
                hinges.push(Hinge {
                    p: c_pt,
                    q: HingePt::Vert(curr),
                    opp_a: HingePt::Vert(prev),
                    opp_b: HingePt::Vert(next),
                    weight: 0.5,
                    sign: 1.0,
                });
            }
        }
    }

    let mut keys: Vec<(usize, usize)> = edge_map.keys().copied().collect();
    keys.sort_unstable();
    for (u, v) in keys {
        let inc = &edge_map[&(u, v)];
        if inc.len() == 2 && inc[0].0 != inc[1].0 {
            let (_, opp_a, _, u0, v0) = inc[0];
            let (_, opp_b, _, _, _) = inc[1];
            hinges.push(Hinge {
                p: HingePt::Vert(u0),
                q: HingePt::Vert(v0),
                opp_a,
                opp_b,
                weight: 1.0,
                sign: 1.0,
            });
        }
    }
    hinges
}

#[inline]
pub(crate) fn dist3_at(x: &[f64], u: usize, v: usize) -> f64 {
    let dx = x[3 * u] - x[3 * v];
    let dy = x[3 * u + 1] - x[3 * v + 1];
    let dz = x[3 * u + 2] - x[3 * v + 2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub(crate) fn eval_hinge_bending_fg(
    pos: &[f64],
    grad: &mut [f64],
    faces: &[Vec<usize>],
    hinges: &[Hinge],
    kappa_eff: f64,
    centroids: &mut [[f64; 3]],
    centroid_grad: &mut [[f64; 3]],
) -> f64 {
    if kappa_eff <= 0.0 || hinges.is_empty() {
        return 0.0;
    }

    for (fi, f) in faces.iter().enumerate() {
        let inv = 1.0 / (f.len() as f64);
        let mut c = [0.0; 3];
        for &u in f {
            c[0] += pos[3 * u] * inv;
            c[1] += pos[3 * u + 1] * inv;
            c[2] += pos[3 * u + 2] * inv;
        }
        centroids[fi] = c;
        centroid_grad[fi] = [0.0; 3];
    }

    let get_pt = |hp: HingePt, pos: &[f64], centroids: &[[f64; 3]]| -> [f64; 3] {
        match hp {
            HingePt::Vert(u) => [pos[3 * u], pos[3 * u + 1], pos[3 * u + 2]],
            HingePt::FaceCentroid(fi) => centroids[fi],
        }
    };

    let mut add_pt_grad = |hp: HingePt, g: [f64; 3]| match hp {
        HingePt::Vert(u) => {
            grad[3 * u] += g[0];
            grad[3 * u + 1] += g[1];
            grad[3 * u + 2] += g[2];
        }
        HingePt::FaceCentroid(fi) => {
            centroid_grad[fi][0] += g[0];
            centroid_grad[fi][1] += g[1];
            centroid_grad[fi][2] += g[2];
        }
    };

    const EPS2: f64 = 1e-20;
    let mut bend_e = 0.0;
    for h in hinges {
        let p = get_pt(h.p, pos, centroids);
        let q = get_pt(h.q, pos, centroids);
        let a = get_pt(h.opp_a, pos, centroids);
        let b = get_pt(h.opp_b, pos, centroids);

        let u = sub3(q, p);
        let va = sub3(a, p);
        let vb = sub3(b, p);
        let na_raw = cross3(u, va);
        let nb_raw = cross3(vb, u);
        let la = (dot3(na_raw, na_raw) + EPS2).sqrt();
        let lb = (dot3(nb_raw, nb_raw) + EPS2).sqrt();
        let na = scale3(na_raw, 1.0 / la);
        let nb = scale3(nb_raw, 1.0 / lb);

        let w = 0.5 * kappa_eff * h.weight;
        let cos_align = dot3(na, nb);
        bend_e += w * (1.0 - h.sign * cos_align);

        let wa = scale3(nb, -w * h.sign);
        let wb = scale3(na, -w * h.sign);
        let ga = scale3(sub3(wa, scale3(na, dot3(wa, na))), 1.0 / la);
        let gb = scale3(sub3(wb, scale3(nb, dot3(wb, nb))), 1.0 / lb);

        let du_a = cross3(va, ga);
        let dva = cross3(ga, u);
        let dvb = cross3(u, gb);
        let du_b = cross3(gb, vb);

        let gq = add3(du_a, du_b);
        let ga_opp = dva;
        let gb_opp = dvb;
        let gp = scale3(add3(gq, add3(ga_opp, gb_opp)), -1.0);

        add_pt_grad(h.p, gp);
        add_pt_grad(h.q, gq);
        add_pt_grad(h.opp_a, ga_opp);
        add_pt_grad(h.opp_b, gb_opp);
    }

    for (fi, f) in faces.iter().enumerate() {
        let cg = centroid_grad[fi];
        let inv = 1.0 / (f.len() as f64);
        for &u in f {
            grad[3 * u] += cg[0] * inv;
            grad[3 * u + 1] += cg[1] * inv;
            grad[3 * u + 2] += cg[2] * inv;
        }
    }

    bend_e
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_gsv_sheet_fg(
    pos: &[f64],
    grad: &mut [f64],
    model: &GsvModel,
    hinges: &[Hinge],
    kappa_eff: f64,
    centroids: &mut [[f64; 3]],
    centroid_grad: &mut [[f64; 3]],
) -> f64 {
    let (es, eh) = eval_gsv_fg::<3>(pos, grad, &model.stretches, &model.shears_3d);
    let eb = eval_hinge_bending_fg(pos, grad, &model.faces, hinges, kappa_eff, centroids, centroid_grad);
    es + eh + eb
}

#[allow(dead_code)]
pub(crate) fn eval_2d_stretch_fg(pos2: &[f64], g2: &mut [f64], springs: &[Spring]) -> f64 {
    g2.fill(0.0);
    let mut e = 0.0;
    for s in springs {
        let dx = pos2[2 * s.u] - pos2[2 * s.v];
        let dy = pos2[2 * s.u + 1] - pos2[2 * s.v + 1];
        let len = (dx * dx + dy * dy + 1e-24).sqrt();
        let diff = len - s.l0;
        e += 0.5 * s.weight * diff * diff;
        let coeff = s.weight * diff / len;
        g2[2 * s.u] += coeff * dx;
        g2[2 * s.u + 1] += coeff * dy;
        g2[2 * s.v] -= coeff * dx;
        g2[2 * s.v + 1] -= coeff * dy;
    }
    e
}

pub(crate) fn bond_stretch_energy_3d(pos: &[f64], stretches: &[StretchTerm]) -> f64 {
    stretches
        .iter()
        .map(|s| {
            let diff = dist3_at(pos, s.u, s.v) - s.l0;
            0.5 * diff * diff
        })
        .sum()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_sheet_fg(
    pos: &[f64],
    grad: &mut [f64],
    faces: &[Vec<usize>],
    springs: &[Spring],
    hinges: &[Hinge],
    kappa_eff: f64,
    centroids: &mut [[f64; 3]],
    centroid_grad: &mut [[f64; 3]],
) -> f64 {
    grad.fill(0.0);
    let mut total_e = 0.0;

    for s in springs {
        let dx = pos[3 * s.u] - pos[3 * s.v];
        let dy = pos[3 * s.u + 1] - pos[3 * s.v + 1];
        let dz = pos[3 * s.u + 2] - pos[3 * s.v + 2];
        let len = (dx * dx + dy * dy + dz * dz + 1e-24).sqrt();
        let diff = len - s.l0;
        total_e += 0.5 * s.weight * diff * diff;
        let coeff = s.weight * diff / len;
        grad[3 * s.u] += coeff * dx;
        grad[3 * s.u + 1] += coeff * dy;
        grad[3 * s.u + 2] += coeff * dz;
        grad[3 * s.v] -= coeff * dx;
        grad[3 * s.v + 1] -= coeff * dy;
        grad[3 * s.v + 2] -= coeff * dz;
    }

    total_e + eval_hinge_bending_fg(pos, grad, faces, hinges, kappa_eff, centroids, centroid_grad)
}

#[allow(dead_code)]
pub(crate) fn eval_energies(
    pos: &[f64],
    faces: &[Vec<usize>],
    springs: &[Spring],
    hinges: &[Hinge],
    kappa_eff: f64,
) -> (f64, f64, f64) {
    let mut e_bond = 0.0;
    let mut e_stretch = 0.0;
    for s in springs {
        let diff = dist3_at(pos, s.u, s.v) - s.l0;
        let term = 0.5 * s.weight * diff * diff;
        e_stretch += term;
        if s.is_primary_bond {
            e_bond += term;
        }
    }
    let mut g_dummy = vec![0.0; pos.len()];
    let mut c_buf = vec![[0.0; 3]; faces.len()];
    let mut cg_buf = vec![[0.0; 3]; faces.len()];
    let e_tot = eval_sheet_fg(
        pos,
        &mut g_dummy,
        faces,
        springs,
        hinges,
        kappa_eff,
        &mut c_buf,
        &mut cg_buf,
    );
    let bending = e_tot - e_stretch;
    let bending = if bending.is_finite() { bending.max(0.0) } else { bending };
    (e_bond, e_stretch, bending)
}

#[inline]
pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn compute_strains(
    xyz: &[[f64; 3]],
    crystal: &BuiltCrystal,
    edge_rest: &[f64],
    faces: &[Vec<usize>],
) -> (Vec<f64>, Vec<f64>) {
    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(crystal.edges.len());
    let bond_strain: Vec<f64> = crystal
        .edges
        .iter()
        .enumerate()
        .map(|(idx, &(u, v))| {
            let l0 = edge_rest[idx];
            if l0 <= 0.0 || u >= xyz.len() || v >= xyz.len() {
                0.0
            } else {
                let eps = (norm3(sub3(xyz[u], xyz[v])) - l0) / l0;
                edge_map.insert((u.min(v), u.max(v)), eps);
                eps
            }
        })
        .collect();

    let face_strain = faces
        .iter()
        .map(|f| {
            if f.is_empty() {
                return 0.0;
            }
            let sum: f64 = (0..f.len())
                .map(|m| {
                    let (a, b) = (f[m], f[(m + 1) % f.len()]);
                    edge_map.get(&(a.min(b), a.max(b))).copied().unwrap_or(0.0)
                })
                .sum();
            sum / (f.len() as f64)
        })
        .collect();

    (bond_strain, face_strain)
}

pub(crate) fn compute_vertex_omega(
    xyz: &[[f64; 3]],
    faces: &[Vec<usize>],
    is_boundary: &[bool],
) -> Vec<f64> {
    let mut angle_sum = vec![0.0_f64; xyz.len()];
    for f in faces {
        let k = f.len();
        if k < 3 {
            continue;
        }
        if k == 3 {
            for m in 0..3 {
                let prev = f[(m + 2) % 3];
                let curr = f[m];
                let next = f[(m + 1) % 3];
                let e1 = sub3(xyz[prev], xyz[curr]);
                let e2 = sub3(xyz[next], xyz[curr]);
                let d = (norm3(e1) * norm3(e2)).max(1e-15);
                angle_sum[curr] += (dot3(e1, e2) / d).clamp(-1.0, 1.0).acos();
            }
        } else {
            let inv = 1.0 / (k as f64);
            let mut c = [0.0_f64; 3];
            for &u in f {
                c[0] += xyz[u][0] * inv;
                c[1] += xyz[u][1] * inv;
                c[2] += xyz[u][2] * inv;
            }
            for m in 0..k {
                let prev = f[(m + k - 1) % k];
                let curr = f[m];
                let next = f[(m + 1) % k];
                let e_prev = sub3(xyz[prev], xyz[curr]);
                let e_cent = sub3(c, xyz[curr]);
                let e_next = sub3(xyz[next], xyz[curr]);
                let d1 = (norm3(e_prev) * norm3(e_cent)).max(1e-15);
                let d2 = (norm3(e_cent) * norm3(e_next)).max(1e-15);
                angle_sum[curr] += (dot3(e_prev, e_cent) / d1).clamp(-1.0, 1.0).acos()
                    + (dot3(e_cent, e_next) / d2).clamp(-1.0, 1.0).acos();
            }
        }
    }
    angle_sum
        .into_iter()
        .enumerate()
        .map(|(u, s)| {
            if is_boundary[u] || s == 0.0 {
                0.0
            } else {
                2.0 * PI - s
            }
        })
        .collect()
}
