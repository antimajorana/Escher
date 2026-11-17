//! Master Affine-Centralizer defect sector classifier (`Conj_H(G_0)`),
//! finite-torus Burnside class counts `#Conj(G / N*Lambda)`, and unbased fusion channels.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::{Add, Div, Mul, Sub};

use serde::{Deserialize, Serialize};

use super::analyzer::{AbelianGroup, AbelianizationData, SpaceGroupAnalyzer, imat_to_rmat};
use super::rat::{IMat, RMat, Rat, imat_vec, rmat_inv, rmat_vec};
use super::snf::{IntMat, invert_unimodular, smith_normal_form, solve_int_linear};

/// Single `Z^n / (I - R)Z^n // Z_H(R)` conjugacy class representative inside a point sector `R`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BieberbachOrbit {
    pub rep_b_prim: Vec<i64>,
    pub t_prim: Vec<Rat>,
    pub t_conv: Vec<Rat>,
    pub x_star_prim: Option<Vec<Rat>>,
    pub x_star_conv: Option<Vec<Rat>>,
    pub t_par_conv: Option<Vec<Rat>>,
    pub orbit_size_qr: usize,
    pub self_inverse: bool,
    pub chi_g: Vec<i128>,
}

/// Complete topological sector classification for one `H`-conjugacy class `[R]` of point operations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectorClass {
    pub r_idx: usize,
    pub r_prim: IMat,
    pub r_conv: RMat,
    pub class_size_p: usize,
    pub det: i64,
    pub order: usize,
    pub z_order: usize,
    pub r_rank: usize,
    pub t_r: Vec<i128>,
    pub t_r_order: i128,
    pub num_orbits: usize,
    pub orbits: Vec<BieberbachOrbit>,
    pub alpha_r: Option<Rat>,
    pub e_par_conv: Option<Vec<Rat>>,
    pub reverses_axis: bool,
    pub num_pure_classes: usize,
    pub pure_orbits: Vec<BieberbachOrbit>,
    pub num_elem_glide_orbits: usize,
    pub elem_glide_orbits: Vec<BieberbachOrbit>,
    pub has_fixed_plane: Option<bool>,
}

/// One distinct conjugacy channel in `[g_1] x [g_2] = { [g_1 (h g_2 h^-1)] }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionChannel {
    pub rep_b_prim: Vec<i64>,
    pub r_idx: usize,
    pub order: usize,
    pub det: i64,
    pub chi_g: Vec<i128>,
    pub x_star_conv: Option<Vec<Rat>>,
}

fn cartesian_tuples(ranges: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let mut out = vec![Vec::with_capacity(ranges.len())];
    for rng in ranges {
        let mut next = Vec::with_capacity(out.len() * rng.len().max(1));
        for prefix in &out {
            for &val in rng {
                let mut item = prefix.clone();
                item.push(val);
                next.push(item);
            }
        }
        out = next;
    }
    out
}

fn diff_int_mat(r: &IMat) -> IntMat {
    let n = r.len();
    let mut m = IntMat::zeros(n, n);
    for (i, r_row) in r.iter().enumerate().take(n) {
        for (j, &rv) in r_row.iter().enumerate().take(n) {
            m.set(i, j, i128::from(if i == j { 1 } else { 0 } - rv));
        }
    }
    m
}

impl SpaceGroupAnalyzer {
    /// Master Affine-Centralizer classification: `Conj_H(G_0)` on `h_sub` acting on `target_sub`.
    pub fn classify_sectors(&self, h_sub: &[usize], target_sub: &[usize]) -> Vec<SectorClass> {
        let ab_g = self.compute_abelianization(&self.all_indices());
        self.classify_sectors_with_ab(h_sub, target_sub, &ab_g)
    }

    pub fn classify_sectors_with_ab(
        &self,
        h_sub: &[usize],
        target_sub: &[usize],
        ab_g: &AbelianizationData,
    ) -> Vec<SectorClass> {
        let p_len = self.p_ops.len();
        let n = self.dim;
        let mut visited = vec![false; p_len];
        let mut orbits_p: Vec<Vec<usize>> = Vec::new();
        for &r_idx in target_sub {
            if visited[r_idx] {
                continue;
            }
            let mut orb = BTreeSet::new();
            for &s in h_sub {
                let s_inv = self.inv_idx[s];
                let conj_r = self.mul_idx(self.mul_idx(s, r_idx), s_inv);
                visited[conj_r] = true;
                orb.insert(conj_r);
            }
            orbits_p.push(orb.into_iter().collect());
        }

        let mut results = Vec::with_capacity(orbits_p.len());
        for orb_p in orbits_p {
            let r_idx = orb_p[0];
            let r_mat = &self.p_ops[r_idx];
            let z_r: Vec<usize> = h_sub
                .iter()
                .copied()
                .filter(|&s| self.mul_idx(s, r_idx) == self.mul_idx(r_idx, s))
                .collect();

            let diff = diff_int_mat(r_mat);
            let snf = smith_normal_form(&diff, true);
            let u_inv = invert_unimodular(&snf.u);
            let full_d: Vec<i128> =
                (0..n).map(|i| if i < snf.diag.len() { snf.diag[i] } else { 0 }).collect();
            let r_rank = full_d.iter().filter(|&&d| d == 0).count();
            let t_r: Vec<i128> = full_d.iter().copied().filter(|&d| d > 1).collect();
            let t_r_order: i128 = t_r.iter().copied().product::<i128>().max(1);

            let mut delta_r: HashMap<usize, Vec<i64>> = HashMap::with_capacity(z_r.len());
            for &s in &z_r {
                let n_sr = self.nu_at(s, r_idx);
                let n_rs = self.nu_at(r_idx, s);
                let d_vec: Vec<i64> = (0..n).map(|k| n_sr[k] - n_rs[k]).collect();
                delta_r.insert(s, d_vec);
            }

            let coord_key = |b: &[i64]| -> Vec<i128> {
                let b_i128: Vec<i128> = b.iter().map(|&v| i128::from(v)).collect();
                let ub = snf.u.mul_vec(&b_i128);
                (0..n)
                    .map(|i| if full_d[i] > 0 { ub[i].rem_euclid(full_d[i]) } else { ub[i] })
                    .collect()
            };
            let coord_to_b = |ck: &[i128]| -> Vec<i64> {
                let mut vec = vec![0_i128; n];
                for i in 0..n {
                    if full_d[i] != 1 {
                        for (row, vr) in vec.iter_mut().enumerate().take(n) {
                            *vr += ck[i] * u_inv.get(row, i);
                        }
                    }
                }
                vec.into_iter().map(|v| v as i64).collect()
            };
            let act_ts = |s: usize, ck: &[i128]| -> Vec<i128> {
                let b = coord_to_b(ck);
                let sb = imat_vec(&self.p_ops[s], &b);
                let d = &delta_r[&s];
                let shifted: Vec<i64> = (0..n).map(|k| sb[k] + d[k]).collect();
                coord_key(&shifted)
            };

            let ord_r = self.mat_order(r_idx);
            let pi_par = self.projector_par(r_idx);
            let mut info = SectorClass {
                r_idx,
                r_prim: r_mat.clone(),
                r_conv: self.r_conv_rat(r_idx),
                class_size_p: orb_p.len(),
                det: self.det_sign[r_idx],
                order: ord_r,
                z_order: z_r.len(),
                r_rank,
                t_r,
                t_r_order,
                num_orbits: 0,
                orbits: Vec::new(),
                alpha_r: None,
                e_par_conv: None,
                reverses_axis: false,
                num_pure_classes: 0,
                pure_orbits: Vec::new(),
                num_elem_glide_orbits: 0,
                elem_glide_orbits: Vec::new(),
                has_fixed_plane: None,
            };

            if r_rank == 0 {
                let diff_rat: RMat = (0..n)
                    .map(|i| (0..n).map(|j| Rat::from_int(diff.get(i, j) as i64)).collect())
                    .collect();
                let diff_inv = rmat_inv(&diff_rat).expect("r_rank == 0 means I - R invertible");
                let ranges: Vec<Vec<i128>> = full_d.iter().map(|&d| (0..d).collect()).collect();
                let mut seen: BTreeSet<Vec<i128>> = BTreeSet::new();
                let mut q_orbits = Vec::new();
                for ck in cartesian_tuples(&ranges) {
                    if seen.contains(&ck) {
                        continue;
                    }
                    let orb_ck: BTreeSet<Vec<i128>> = z_r.iter().map(|&s| act_ts(s, &ck)).collect();
                    let first_ck = orb_ck.iter().next().unwrap().clone();
                    seen.extend(orb_ck.iter().cloned());
                    let b_rep = coord_to_b(&first_ck);
                    let t_prim: Vec<Rat> =
                        (0..n).map(|k| self.tau[r_idx][k].add(Rat::from_int(b_rep[k]))).collect();
                    let t_conv = self.to_conv_frac(&t_prim);
                    let x_star_prim = rmat_vec(&diff_inv, &t_prim);
                    let x_star_conv = self.to_conv_frac(&x_star_prim);
                    let (ginv_b, ginv_r) = self.seitz_inv(&b_rep, r_idx);
                    let is_self_inv = h_sub.iter().any(|&s| {
                        let (cb, cr) = self.seitz_conj(&vec![0_i64; n], s, &ginv_b, ginv_r);
                        cr == r_idx && orb_ck.contains(&coord_key(&cb))
                    });
                    q_orbits.push(BieberbachOrbit {
                        chi_g: ab_g.chi(&b_rep, r_idx),
                        rep_b_prim: b_rep,
                        t_prim,
                        t_conv,
                        x_star_prim: Some(x_star_prim),
                        x_star_conv: Some(x_star_conv),
                        t_par_conv: None,
                        orbit_size_qr: orb_ck.len(),
                        self_inverse: is_self_inv,
                    });
                }
                info.num_orbits = q_orbits.len();
                info.orbits = q_orbits;
            } else if r_rank == 1 {
                let free_idx = full_d.iter().position(|&d| d == 0).unwrap();
                let free_col: Vec<Rat> = (0..n).map(|r| Rat::from_int(u_inv.get(r, free_idx) as i64)).collect();
                let e_par_prim = rmat_vec(&pi_par, &free_col);
                let tau_par = rmat_vec(&pi_par, &self.tau[r_idx]);
                let nz_idx = e_par_prim.iter().position(|v| !v.is_zero()).unwrap();
                let alpha_raw = tau_par[nz_idx].div(e_par_prim[nz_idx]);
                debug_assert!((0..n).all(|k| tau_par[k] == alpha_raw.mul(e_par_prim[k])));
                let m_shift = alpha_raw.floor_i64();
                let alpha_r = alpha_raw.sub(Rat::from_int(m_shift));
                let reverses_axis = z_r.iter().any(|&s| {
                    let sr = imat_to_rmat(&self.p_ops[s]);
                    let img = rmat_vec(&sr, &e_par_prim);
                    (0..n).all(|k| img[k] == e_par_prim[k].mul(Rat::from_int(-1)))
                });

                let shell_orbits = |m_vals: &[i64]| -> Vec<BieberbachOrbit> {
                    let mut seen_s: BTreeSet<Vec<i128>> = BTreeSet::new();
                    let mut orbs_s = Vec::new();
                    for &mv in m_vals {
                        let ranges: Vec<Vec<i128>> = (0..n)
                            .map(|i| {
                                if i == free_idx {
                                    vec![i128::from(mv)]
                                } else {
                                    (0..full_d[i]).collect()
                                }
                            })
                            .collect();
                        for ck in cartesian_tuples(&ranges) {
                            if seen_s.contains(&ck) {
                                continue;
                            }
                            let orb_ck: BTreeSet<Vec<i128>> = z_r.iter().map(|&s| act_ts(s, &ck)).collect();
                            let first_ck = orb_ck.iter().next().unwrap().clone();
                            seen_s.extend(orb_ck.iter().cloned());
                            let b_rep = coord_to_b(&first_ck);
                            let t_prim: Vec<Rat> =
                                (0..n).map(|k| self.tau[r_idx][k].add(Rat::from_int(b_rep[k]))).collect();
                            let t_conv = self.to_conv_frac(&t_prim);
                            let t_par_conv = self.to_conv_frac(&rmat_vec(&pi_par, &t_prim));
                            orbs_s.push(BieberbachOrbit {
                                chi_g: ab_g.chi(&b_rep, r_idx),
                                rep_b_prim: b_rep,
                                t_prim,
                                t_conv,
                                x_star_prim: None,
                                x_star_conv: None,
                                t_par_conv: Some(t_par_conv),
                                orbit_size_qr: orb_ck.len(),
                                self_inverse: false,
                            });
                        }
                    }
                    orbs_s
                };

                let pure_orbs = if alpha_r.is_zero() { shell_orbits(&[-m_shift]) } else { Vec::new() };
                let mut elem_m: Vec<i64> = if alpha_r.is_zero() {
                    if reverses_axis {
                        vec![1 - m_shift, -1 - m_shift]
                    } else {
                        vec![1 - m_shift]
                    }
                } else if alpha_r == Rat::new(1, 2) {
                    if reverses_axis {
                        vec![-m_shift, -1 - m_shift]
                    } else {
                        vec![-m_shift]
                    }
                } else if 2 * alpha_r.num <= alpha_r.den {
                    vec![-m_shift]
                } else {
                    vec![-1 - m_shift]
                };
                elem_m.sort_unstable();
                elem_m.dedup();
                let elem_orbs = shell_orbits(&elem_m);
                info.alpha_r = Some(alpha_r);
                info.e_par_conv = Some(self.to_conv_frac(&e_par_prim));
                info.reverses_axis = reverses_axis;
                info.num_pure_classes = pure_orbs.len();
                info.pure_orbits = pure_orbs;
                info.num_elem_glide_orbits = elem_orbs.len();
                info.elem_glide_orbits = elem_orbs;
            } else {
                let p_tau = rmat_vec(&pi_par, &self.tau[r_idx]);
                info.has_fixed_plane = Some(p_tau.iter().all(|v| v.as_i64().is_some()));
            }
            results.push(info);
        }
        results
    }

    /// Exact finite-torus conjugacy count `#Conj(G / N*Z^n)` via Master Affine-Centralizer formula.
    pub fn count_torus_conjugacy_classes(&self, n_scale: i64) -> (usize, BTreeMap<String, usize>) {
        let p_len = self.p_ops.len();
        let n = self.dim;
        let mut visited = vec![false; p_len];
        let mut orbits_p: Vec<usize> = Vec::new();
        for r_idx in 0..p_len {
            if visited[r_idx] {
                continue;
            }
            for s in 0..p_len {
                let s_inv = self.inv_idx[s];
                visited[self.mul_idx(self.mul_idx(s, r_idx), s_inv)] = true;
            }
            orbits_p.push(r_idx);
        }

        let mut total = 0usize;
        let mut by_ord = BTreeMap::new();
        for r_idx in orbits_p {
            let z_r: Vec<usize> =
                (0..p_len).filter(|&s| self.mul_idx(s, r_idx) == self.mul_idx(r_idx, s)).collect();
            let mut delta_r: Vec<(usize, Vec<i64>)> = Vec::with_capacity(z_r.len());
            for &s in &z_r {
                let n_sr = self.nu_at(s, r_idx);
                let n_rs = self.nu_at(r_idx, s);
                delta_r.push((s, (0..n).map(|k| n_sr[k] - n_rs[k]).collect()));
            }
            let r_mat = &self.p_ops[r_idx];
            let mut m_col = IntMat::zeros(n, 2 * n);
            for (i, r_row) in r_mat.iter().enumerate().take(n) {
                for (j, &rv) in r_row.iter().enumerate().take(n) {
                    m_col.set(i, j, i128::from(if i == j { 1 } else { 0 } - rv));
                }
                m_col.set(i, n + i, i128::from(n_scale));
            }
            let snf = smith_normal_form(&m_col, true);
            let u_inv = invert_unimodular(&snf.u);
            debug_assert_eq!(snf.diag.len(), n);
            let diag = &snf.diag;
            let ranges: Vec<Vec<i128>> = diag.iter().map(|&d| (0..d).collect()).collect();
            let mut seen: BTreeSet<Vec<i128>> = BTreeSet::new();
            let mut n_orb = 0usize;
            for ck in cartesian_tuples(&ranges) {
                if seen.contains(&ck) {
                    continue;
                }
                let mut b = vec![0_i64; n];
                for i in 0..n {
                    if diag[i] != 1 {
                        for (row, br) in b.iter_mut().enumerate().take(n) {
                            *br += (ck[i] * u_inv.get(row, i)) as i64;
                        }
                    }
                }
                let mut orb_ck = BTreeSet::new();
                for (s, d_vec) in &delta_r {
                    let sb = imat_vec(&self.p_ops[*s], &b);
                    let shifted: Vec<i128> =
                        (0..n).map(|k| i128::from(sb[k] + d_vec[k])).collect();
                    let u_s = snf.u.mul_vec(&shifted);
                    let key: Vec<i128> =
                        (0..n).map(|i| u_s[i].rem_euclid(diag[i])).collect();
                    orb_ck.insert(key);
                }
                seen.extend(orb_ck);
                n_orb += 1;
            }
            total += n_orb;
            let sign = if self.det_sign[r_idx] == 1 { '+' } else { '-' };
            let lbl = format!("det={sign},ord={}", self.mat_order(r_idx));
            *by_ord.entry(lbl).or_insert(0) += n_orb;
        }
        (total, by_ord)
    }

    /// Abelianized group info `((G^+)^ab, G^ab)`.
    pub fn abelianizations(&self) -> (AbelianGroup, AbelianGroup) {
        let ab_plus = self.compute_abelianization(&self.p_plus_indices());
        let ab_full = self.compute_abelianization(&self.all_indices());
        (ab_plus.group, ab_full.group)
    }

    /// Enumerate distinct unbased fusion channels `[g_1] x [g_2] = { [g_1 (h g_2 h^-1)] }`.
    pub fn unbased_fusion_channels(
        &self,
        g1_b: &[i64],
        g1_r: usize,
        g2_b: &[i64],
        g2_r: usize,
    ) -> Vec<FusionChannel> {
        let n = self.dim;
        let p_len = self.p_ops.len();
        let ab_g = self.compute_abelianization(&self.all_indices());
        let mut conj_g2: BTreeSet<(Vec<i64>, usize)> = BTreeSet::new();
        let zero = vec![0_i64; n];
        for s in 0..p_len {
            let (cb, cr) = self.seitz_conj(&zero, s, g2_b, g2_r);
            conj_g2.insert((cb, cr));
        }

        let shift_bound: i64 = if n <= 2 { 5 } else { 2 };
        let c_ranges: Vec<Vec<i128>> =
            vec![(0..=i128::from(shift_bound)).collect(); n];
        let mut seen_keys: BTreeSet<(usize, Vec<i128>, Option<Vec<i128>>)> = BTreeSet::new();
        let mut channels = Vec::new();
        let snf_by_r: Vec<(_, _)> = (0..p_len)
            .map(|ri| {
                let d_c = diff_int_mat(&self.p_ops[ri]);
                let snf_c = smith_normal_form(&d_c, true);
                (d_c, snf_c)
            })
            .collect();

        for (b2p, r2p) in &conj_g2 {
            let r_prod = self.mul_idx(g1_r, *r2p);
            let snf_prod = &snf_by_r[r_prod].1;
            let inv_r2 = &snf_by_r[*r2p].0;
            let full_d: Vec<i128> = (0..n)
                .map(|i| if i < snf_prod.diag.len() { snf_prod.diag[i] } else { 0 })
                .collect();
            let r_rank = full_d.iter().filter(|&&d| d == 0).count();

            let diff_rat: Option<RMat> = (r_rank == 0).then(|| {
                let diff_prod = &snf_by_r[r_prod].0;
                let d_mat: RMat = (0..n)
                    .map(|i| (0..n).map(|j| Rat::from_int(diff_prod.get(i, j) as i64)).collect())
                    .collect();
                rmat_inv(&d_mat).unwrap()
            });

            for c_tuple in cartesian_tuples(&c_ranges) {
                let delta_b: Vec<i64> =
                    inv_r2.mul_vec(&c_tuple).into_iter().map(|v| v as i64).collect();
                let step = imat_vec(&self.p_ops[g1_r], &delta_b);
                let (mut b_prod, r_tot) = self.seitz_mul(g1_b, g1_r, b2p, *r2p);
                for k in 0..n {
                    b_prod[k] += step[k];
                }
                let chi = ab_g.chi(&b_prod, r_tot);
                let mut best_r = r_tot;
                let mut best_ck: Option<Vec<i128>> = None;
                for s in 0..p_len {
                    let (cb, cr) = self.seitz_conj(&zero, s, &b_prod, r_tot);
                    if cr <= best_r || best_ck.is_none() {
                        let snf_c = &snf_by_r[cr].1;
                        let cb_i: Vec<i128> = cb.iter().map(|&v| i128::from(v)).collect();
                        let u_cb = snf_c.u.mul_vec(&cb_i);
                        let ck: Vec<i128> = (0..n)
                            .map(|i| {
                                if i < snf_c.diag.len() {
                                    u_cb[i].rem_euclid(snf_c.diag[i])
                                } else {
                                    u_cb[i]
                                }
                            })
                            .collect();
                        if cr < best_r || best_ck.as_ref().is_none_or(|cur| ck < *cur) {
                            best_r = cr;
                            best_ck = Some(ck);
                        }
                    }
                }
                let dedup_key = (best_r, chi.clone(), (r_rank == 0).then_some(best_ck).flatten());
                if seen_keys.insert(dedup_key) {
                    let x_star_conv = diff_rat.as_ref().map(|dinv| {
                        let t_prim: Vec<Rat> = (0..n)
                            .map(|k| self.tau[r_tot][k].add(Rat::from_int(b_prod[k])))
                            .collect();
                        self.to_conv_frac(&rmat_vec(dinv, &t_prim))
                    });
                    channels.push(FusionChannel {
                        rep_b_prim: b_prod,
                        r_idx: r_tot,
                        order: self.mat_order(r_tot),
                        det: self.det_sign[r_tot],
                        chi_g: chi,
                        x_star_conv,
                    });
                }
            }
        }
        channels
    }
}

/// Check if integer vector `b` belongs to `im_Z(I - R)`.
pub fn is_in_image_i_minus_r(r: &IMat, b: &[i64]) -> bool {
    let d = diff_int_mat(r);
    let b_i: Vec<i128> = b.iter().map(|&v| i128::from(v)).collect();
    solve_int_linear(&d, &b_i).is_some()
}
