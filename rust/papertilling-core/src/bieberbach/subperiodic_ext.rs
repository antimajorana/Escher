//! Bieberbach cohomology (`H^2(P, Z^d)`), splitness, and abelianizations `(G^+)^ab`, `G^ab`
//! for subperiodic space groups with lattice dimension `d < ambient_dim` (1D Friezes & 2D Layers).

use std::collections::{HashMap, VecDeque};
use std::ops::{Add, Sub};

use super::analyzer::{AbelianGroup, CohomologyH2, ConvOp, imat_key, imat_to_rmat, rmat_to_imat};
use super::rat::{IMat, RMat, Rat, imat_det, imat_eye, imat_mul, rmat_inv, rmat_mul, rmat_transpose, rmat_vec};
use super::snf::{hnf_compress_rows, invariant_factors_compressed, smith_normal_form, solve_int_linear, IntMat};
use super::{BieberbachSummary, SectorBrief};

/// Exact subperiodic space-group analyzer with translational lattice dimension `lat_dim <= ambient_dim`.
pub fn analyze_subperiodic_summary(
    lat_dim: usize,
    ambient_dim: usize,
    number: i32,
    symbol: &str,
    gens_conv: &[ConvOp],
    tb_lat_rows: &[Vec<Rat>],
) -> Result<BieberbachSummary, String> {
    if lat_dim == 0 || lat_dim > ambient_dim || ambient_dim > 3 {
        return Err(format!("invalid subperiodic dims lat={lat_dim}, ambient={ambient_dim}"));
    }
    if tb_lat_rows.len() != lat_dim || tb_lat_rows.iter().any(|r| r.len() != lat_dim) {
        return Err(format!("invalid tb_lat_rows for lat_dim={lat_dim}"));
    }
    let b_lat: RMat = rmat_transpose(&tb_lat_rows.to_vec());
    let b_lat_inv = rmat_inv(&b_lat)?;

    // Build block-diagonal full rational converter B_full = diag(B_lat, I_{ambient-lat}).
    let mut b_full: RMat = (0..ambient_dim)
        .map(|i| (0..ambient_dim).map(|j| if i == j { Rat::ONE } else { Rat::ZERO }).collect())
        .collect();
    for i in 0..lat_dim {
        for j in 0..lat_dim {
            b_full[i][j] = b_lat[i][j];
        }
    }
    let b_full_inv = rmat_inv(&b_full)?;

    let mut gens_prim: Vec<(IMat, IMat, Vec<Rat>)> = Vec::with_capacity(gens_conv.len());
    for (gi, g) in gens_conv.iter().enumerate() {
        if g.r.len() != ambient_dim || g.t.len() < lat_dim {
            return Err(format!("{symbol}: generator {gi} dimension mismatch"));
        }
        let r_conv_rat = imat_to_rmat(&g.r);
        let r_full_rat = rmat_mul(&b_full_inv, &rmat_mul(&r_conv_rat, &b_full));
        let r_full = rmat_to_imat(&r_full_rat, &format!("{symbol} subperiodic gen {gi}"))?;
        if imat_det(&r_full).abs() != 1 {
            return Err(format!("{symbol} gen {gi}: |det| != 1"));
        }
        let r_par: IMat = (0..lat_dim).map(|i| r_full[i][0..lat_dim].to_vec()).collect();
        let t_par = rmat_vec(&b_lat_inv, &g.t[0..lat_dim]);
        gens_prim.push((r_full, r_par, t_par));
    }

    let id_full = imat_eye(ambient_dim);
    let id_key = imat_key(&id_full);
    let mut p_full = vec![id_full];
    let mut p_par = vec![imat_eye(lat_dim)];
    let mut tau = vec![vec![Rat::ZERO; lat_dim]];
    let mut lookup = HashMap::new();
    lookup.insert(id_key, 0_usize);
    let mut q = VecDeque::from([0_usize]);

    while let Some(i1) = q.pop_front() {
        let r1_full = p_full[i1].clone();
        let r1_par = p_par[i1].clone();
        let r1_par_rat = imat_to_rmat(&r1_par);
        let t1 = tau[i1].clone();
        for (rg_full, _, tg_par) in &gens_prim {
            let r2_full = imat_mul(&r1_full, rg_full);
            let k2 = imat_key(&r2_full);
            let mut t2 = Vec::with_capacity(lat_dim);
            let r1_tg = rmat_vec(&r1_par_rat, tg_par);
            for c in 0..lat_dim {
                t2.push(t1[c].add(r1_tg[c]).frac_unit());
            }
            if let Some(&existing) = lookup.get(&k2) {
                if tau[existing] != t2 {
                    return Err(format!("{symbol}: inconsistent non-lattice shift in point group"));
                }
            } else {
                let r2_par: IMat = (0..lat_dim).map(|i| r2_full[i][0..lat_dim].to_vec()).collect();
                let new_i = p_full.len();
                p_full.push(r2_full);
                p_par.push(r2_par);
                tau.push(t2);
                lookup.insert(k2, new_i);
                q.push_back(new_i);
            }
        }
    }

    let p_len = p_full.len();
    let mut mul_table = vec![0_usize; p_len * p_len];
    for i in 0..p_len {
        for j in 0..p_len {
            let k = imat_key(&imat_mul(&p_full[i], &p_full[j]));
            mul_table[i * p_len + j] = lookup[&k];
        }
    }
    let det_sign: Vec<i64> = p_full.iter().map(imat_det).collect();
    let mut nu = vec![vec![0_i64; lat_dim]; p_len * p_len];
    for i in 0..p_len {
        let r1_rat = imat_to_rmat(&p_par[i]);
        for j in 0..p_len {
            let ij = mul_table[i * p_len + j];
            let r1_t2 = rmat_vec(&r1_rat, &tau[j]);
            let mut entry = Vec::with_capacity(lat_dim);
            for c in 0..lat_dim {
                let val = tau[i][c].add(r1_t2[c]).sub(tau[ij][c]);
                entry.push(val.as_i64().ok_or_else(|| format!("{symbol}: non-int nu entry"))?);
            }
            nu[i * p_len + j] = entry;
        }
    }

    // Verify 2-cocycle identity (d2 nu) == 0 on Z^{lat_dim}.
    for i in 0..p_len {
        for j in 0..p_len {
            let ij = mul_table[i * p_len + j];
            for k in 0..p_len {
                let jk = mul_table[j * p_len + k];
                for r in 0..lat_dim {
                    let r1_n23: i64 = (0..lat_dim).map(|c| p_par[i][r][c] * nu[j * p_len + k][c]).sum();
                    if r1_n23 - nu[ij * p_len + k][r] + nu[i * p_len + jk][r] - nu[i * p_len + j][r] != 0 {
                        return Err(format!("{symbol}: 2-cocycle identity failed"));
                    }
                }
            }
        }
    }

    // Exact H^2(P, Z^{lat_dim}) and splitness [nu] == 0.
    let h2 = if p_len <= 1 {
        CohomologyH2 { torsion: Vec::new(), is_split: true }
    } else {
        let m1 = p_len - 1;
        let n1 = lat_dim * m1;
        let n2 = lat_dim * m1 * m1;
        let mut d1 = IntMat::zeros(n2, n1);
        let mut nu_vec = vec![0_i128; n2];
        let mut rb = 0usize;
        for i in 1..p_len {
            let i0 = i - 1;
            for j in 1..p_len {
                let j0 = j - 1;
                let ij = mul_table[i * p_len + j];
                for p in 0..lat_dim {
                    nu_vec[rb + p] = i128::from(nu[i * p_len + j][p]);
                    d1.add_assign(rb + p, i0 * lat_dim + p, 1);
                    for (q_c, &pv) in p_par[i][p].iter().enumerate().take(lat_dim) {
                        d1.add_assign(rb + p, j0 * lat_dim + q_c, i128::from(pv));
                    }
                    if ij != 0 {
                        d1.add_assign(rb + p, (ij - 1) * lat_dim + p, -1);
                    }
                }
                rb += lat_dim;
            }
        }
        let torsion = invariant_factors_compressed(&d1).into_iter().filter(|&d| d > 1).collect();
        let comp_aug = hnf_compress_rows(&d1.hstack_col(&nu_vec));
        let mut comp_a = IntMat::zeros(comp_aug.rows, n1);
        let mut comp_b = Vec::with_capacity(comp_aug.rows);
        for r in 0..comp_aug.rows {
            for c in 0..n1 {
                comp_a.set(r, c, comp_aug.get(r, c));
            }
            comp_b.push(comp_aug.get(r, n1));
        }
        let is_split = solve_int_linear(&comp_a, &comp_b).is_some();
        CohomologyH2 { torsion, is_split }
    };

    let compute_ab = |sub_indices: &[usize]| -> AbelianGroup {
        let non_id: Vec<usize> = sub_indices.iter().copied().filter(|&i| i != 0).collect();
        let pos: HashMap<usize, usize> = non_id.iter().enumerate().map(|(idx, &ri)| (ri, idx)).collect();
        let num_gens = lat_dim + non_id.len();
        let mut rows: Vec<Vec<i128>> = Vec::new();
        for &ri in sub_indices {
            for j in 0..lat_dim {
                let mut row = vec![0_i128; num_gens];
                for (i, pr) in p_par[ri].iter().enumerate().take(lat_dim) {
                    row[i] = i128::from(if i == j { 1 } else { 0 } - pr[j]);
                }
                if row.iter().any(|&v| v != 0) {
                    rows.push(row);
                }
            }
        }
        for &i1 in &non_id {
            for &i2 in &non_id {
                let i12 = mul_table[i1 * p_len + i2];
                let mut row = vec![0_i128; num_gens];
                for c in 0..lat_dim {
                    row[c] = -i128::from(nu[i1 * p_len + i2][c]);
                }
                row[lat_dim + pos[&i1]] += 1;
                row[lat_dim + pos[&i2]] += 1;
                if i12 != 0
                    && let Some(&p12) = pos.get(&i12)
                {
                    row[lat_dim + p12] -= 1;
                }
                if row.iter().any(|&v| v != 0) {
                    rows.push(row);
                }
            }
        }
        if rows.is_empty() {
            return AbelianGroup { torsion: Vec::new(), free_rank: num_gens };
        }
        let comp = hnf_compress_rows(&IntMat::from_rows(&rows));
        let snf = smith_normal_form(&comp.transpose(), false);
        let mut torsion = Vec::new();
        let mut free_rank = 0usize;
        for i in 0..num_gens {
            let d = if i < snf.diag.len() { snf.diag[i] } else { 0 };
            if d > 1 {
                torsion.push(d);
            } else if d == 0 {
                free_rank += 1;
            }
        }
        AbelianGroup { torsion, free_rank }
    };

    let p_plus: Vec<usize> = (0..p_len).filter(|&i| det_sign[i] == 1).collect();
    let all_idx: Vec<usize> = (0..p_len).collect();
    let ab_plus = compute_ab(&p_plus);
    let ab_full = compute_ab(&all_idx);

    let inv_idx: Vec<usize> = (0..p_len)
        .map(|i| (0..p_len).find(|&j| mul_table[i * p_len + j] == 0).unwrap())
        .collect();
    let mut visited = vec![false; p_len];
    visited[0] = true;
    let mut sectors_full = Vec::new();
    for r_idx in 1..p_len {
        if visited[r_idx] {
            continue;
        }
        let mut size_p = 0usize;
        for s in 0..p_len {
            let c = mul_table[mul_table[s * p_len + r_idx] * p_len + inv_idx[s]];
            if !visited[c] {
                visited[c] = true;
                size_p += 1;
            }
        }
        let mut cur = r_idx;
        let mut ord = 1usize;
        while cur != 0 {
            cur = mul_table[cur * p_len + r_idx];
            ord += 1;
        }
        let mut diff = IntMat::zeros(lat_dim, lat_dim);
        for (i, pr) in p_par[r_idx].iter().enumerate().take(lat_dim) {
            for (j, &pv) in pr.iter().enumerate().take(lat_dim) {
                diff.set(i, j, i128::from(if i == j { 1 } else { 0 } - pv));
            }
        }
        let snf_d = smith_normal_form(&diff, false);
        let r_rank = lat_dim - snf_d.diag.len();
        let t_r: Vec<i128> = snf_d.diag.into_iter().filter(|&d| d > 1).collect();
        sectors_full.push(SectorBrief {
            det: det_sign[r_idx],
            order: ord,
            class_size_p: size_p.max(1),
            r_rank,
            t_r_label: AbelianGroup { torsion: t_r, free_rank: 0 }.format(),
            num_orbits: 1,
            num_pure_classes: 0,
            num_elem_glide_orbits: 0,
            alpha_r: None,
        });
    }

    Ok(BieberbachSummary {
        dim: lat_dim,
        number,
        symbol: symbol.to_string(),
        point_order: p_len,
        h2_torsion: h2.torsion.clone(),
        h2_label: h2.format(),
        is_split: h2.is_split,
        ab_plus_torsion: ab_plus.torsion.clone(),
        ab_plus_free: ab_plus.free_rank,
        ab_plus_label: ab_plus.format(),
        ab_full_torsion: ab_full.torsion.clone(),
        ab_full_free: ab_full.free_rank,
        ab_full_label: ab_full.format(),
        torus_counts: Vec::new(),
        sectors_full,
    })
}
