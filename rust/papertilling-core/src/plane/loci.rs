//! Periodic bond-graph automorphisms and affine supergroup loci (`Family.loci` / `Locus`).

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::tuning::min_separation;
use super::tuning_eval::{
    BareBond, CrystalSystem, EvalFamily, identify_wallpaper_group, imat2_ivec, imat2_vec, mat2_mul,
};
use crate::types::Recipe;

const MAX_LOCI: usize = 60;

/// An automorphism `(site k, cell n) -> (site pi[k], cell mf * n + s[k])` of the periodic graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Automorphism {
    pub mf: [[i32; 2]; 2],
    pub pi: Vec<usize>,
    pub s: Vec<[i32; 2]>,
    pub metric: Vec<Vec<f64>>,
}

/// Classification of an affine automorphism locus in parameter space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocusKind {
    Supergroup,
    Collision,
    Generic,
}

/// Affine subspace `C * v = c` in `(metric_vars, site_parameters)` with enhanced symmetry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Locus {
    /// Reduced row echelon form `[C | c]` of shape `(codim, n_vars + 1)`.
    pub aug: Vec<Vec<f64>>,
    pub codim: usize,
    pub dim: usize,
    pub group: Option<String>,
    pub sites: Option<usize>,
    pub kind: LocusKind,
    pub equations: Vec<String>,
    pub example: Option<BTreeMap<String, f64>>,
    pub n_automorphisms: usize,
}

impl Locus {
    pub(crate) fn residual_vec(&self, v: &[f64]) -> f64 {
        self.aug.iter().map(|row| {
            let lhs: f64 = row[..v.len()].iter().zip(v).map(|(&a, &x)| a * x).sum();
            (lhs - row[v.len()]).abs()
        }).fold(0.0, f64::max)
    }

    pub(crate) fn project_vec(&self, v: &[f64]) -> Vec<f64> {
        let (r, n) = (self.codim, v.len());
        if r == 0 {
            return v.to_vec();
        }
        // C is in RREF with r distinct unit pivot columns, so G = C * C^T is SPD (lambda_min >= 1).
        let mut g = vec![vec![0.0; r + 1]; r];
        for i in 0..r {
            for j in 0..r {
                g[i][j] = (0..n).map(|k| self.aug[i][k] * self.aug[j][k]).sum();
            }
            let cv: f64 = (0..n).map(|k| self.aug[i][k] * v[k]).sum();
            g[i][r] = cv - self.aug[i][n];
        }
        for col in 0..r {
            let piv = (col..r).max_by(|&a, &b| g[a][col].abs().total_cmp(&g[b][col].abs())).unwrap();
            g.swap(col, piv);
            let inv = 1.0 / g[col][col];
            for val in &mut g[col][col..=r] {
                *val *= inv;
            }
            for row in 0..r {
                if row != col {
                    let f = g[row][col];
                    for k in col..=r {
                        g[row][k] -= f * g[col][k];
                    }
                }
            }
        }
        let mut out = v.to_vec();
        for (k, out_k) in out.iter_mut().enumerate() {
            *out_k -= (0..r).map(|i| self.aug[i][k] * g[i][r]).sum::<f64>();
        }
        out
    }

    pub(crate) fn project_with_family(&self, fam: &EvalFamily, values: Option<&BTreeMap<String, f64>>) -> BTreeMap<String, f64> {
        let v = fam.v_vec(&fam.merged_values(values));
        fam.values_from_v(&self.project_vec(&v))
    }

    /// Orthogonal projection in `(metric_vars, site_parameters)` onto this locus (`Locus.project`).
    pub fn project(&self, recipe: &Recipe, values: Option<&BTreeMap<String, f64>>) -> Result<BTreeMap<String, f64>, String> {
        Ok(self.project_with_family(&EvalFamily::from_recipe(recipe)?, values))
    }

    /// Maximum absolute equation residual `max |C * v - c|` at `values`.
    pub fn residual(&self, recipe: &Recipe, values: Option<&BTreeMap<String, f64>>) -> Result<f64, String> {
        let fam = EvalFamily::from_recipe(recipe)?;
        Ok(self.residual_vec(&fam.v_vec(&fam.merged_values(values))))
    }

    /// Whether `values` satisfies every locus equation within `tol` (default `1e-7` in Python).
    pub fn contains(&self, recipe: &Recipe, values: Option<&BTreeMap<String, f64>>, tol: f64) -> Result<bool, String> {
        Ok(self.residual(recipe, values)? < tol)
    }
}

pub(crate) fn rref(rows_in: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if rows_in.is_empty() {
        return Some(Vec::new());
    }
    let (rows, cols) = (rows_in.len(), rows_in[0].len());
    let (mut m, mut r) = (rows_in.to_vec(), 0usize);
    for c in 0..(cols - 1) {
        if r >= rows {
            break;
        }
        let p = (r..rows).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs())).unwrap();
        if m[p][c].abs() < 1e-9 {
            continue;
        }
        m.swap(r, p);
        let inv = 1.0 / m[r][c];
        for val in &mut m[r] {
            *val *= inv;
        }
        for i in 0..rows {
            if i != r && m[i][c] != 0.0 {
                let factor = m[i][c];
                for k in c..cols {
                    m[i][k] -= factor * m[r][k];
                }
            }
        }
        r += 1;
    }
    if (r..rows).any(|i| m[i][cols - 1].abs() > 1e-7) {
        return None;
    }
    m.truncate(r);
    for row in &mut m {
        for x in row {
            if x.abs() < 1e-10 {
                *x = 0.0;
            }
        }
    }
    Some(m)
}

fn nice_number(x: f64) -> String {
    if x.abs() < 1e-9 {
        return "0".to_string();
    }
    for den in 1..=24i64 {
        let num = (x * den as f64).round() as i64;
        if ((num as f64 / den as f64) - x).abs() < 1e-9 {
            return if den == 1 { format!("{num}") } else { format!("{num}/{den}") };
        }
    }
    if x > 0.0 {
        let x2 = x * x;
        for den in 1..=24i64 {
            let num = (x2 * den as f64).round() as i64;
            if num > 0 && ((num as f64 / den as f64) - x2).abs() < 1e-9 {
                return if den == 1 { format!("sqrt({num})") } else { format!("sqrt({num}/{den})") };
            }
        }
    }
    format!("{x:.6}")
}

fn format_equations(aug: &[Vec<f64>], v_names: &[String], a_scale: f64) -> Vec<String> {
    let mut out = Vec::with_capacity(aug.len());
    for row in aug {
        let (c_row, rhs_val) = (&row[..row.len() - 1], row[row.len() - 1]);
        let idx: Vec<usize> = (0..c_row.len()).filter(|&k| c_row[k].abs() > 1e-9).collect();
        if idx.is_empty() {
            continue;
        }
        let p = idx[0];
        if v_names[p] == "b^2" && idx.len() == 1 {
            let b_str = if rhs_val > 0.0 { nice_number(rhs_val.sqrt() * a_scale) } else { "?".to_string() };
            out.push(format!("b = {b_str}{}", if (rhs_val - 1.0).abs() < 1e-9 { " (b = a)" } else { "" }));
            continue;
        }
        if v_names[p] == "b cos(gamma)" && idx.len() == 1 && rhs_val.abs() < 1e-9 {
            out.push("gamma = 90".to_string());
            continue;
        }
        let mut terms: Vec<String> = idx[1..].iter().map(|&k| {
            let co = -c_row[k];
            let mag = if (co.abs() - 1.0).abs() < 1e-9 { String::new() } else { format!("{} ", nice_number(co.abs())) };
            format!("{}{}{}", if co > 0.0 { "+ " } else { "- " }, mag, v_names[k])
        }).collect();
        if rhs_val.abs() > 1e-9 || terms.is_empty() {
            if terms.is_empty() {
                terms.push(nice_number(rhs_val));
            } else {
                terms.push(format!("{}{}", if rhs_val > 0.0 { "+ " } else { "- " }, nice_number(rhs_val.abs())));
            }
        }
        let rhs = terms.join(" ");
        out.push(format!("{} = {}", v_names[p], rhs.strip_prefix("+ ").unwrap_or(&rhs)));
    }
    out
}

fn metric_system(fam: &EvalFamily, mf: [[i32; 2]; 2]) -> Option<Vec<Vec<f64>>> {
    let u = fam.u_mat;
    let ut = [[u[0][0], u[1][0]], [u[0][1], u[1][1]]];
    let mf_f = [[mf[0][0] as f64, mf[0][1] as f64], [mf[1][0] as f64, mf[1][1] as f64]];
    let mft = [[mf_f[0][0], mf_f[1][0]], [mf_f[0][1], mf_f[1][1]]];
    let res = |gm: [[f64; 2]; 2]| -> [f64; 3] {
        let gp = mat2_mul(mat2_mul(ut, gm), u);
        let m_gp_m = mat2_mul(mat2_mul(mft, gp), mf_f);
        [m_gp_m[0][0] - gp[0][0], m_gp_m[0][1] - gp[0][1], m_gp_m[1][1] - gp[1][1]]
    };
    let (g0, gs): ([[f64; 2]; 2], &[[[f64; 2]; 2]]) = match fam.g.system {
        CrystalSystem::Oblique => ([[1.0, 0.0], [0.0, 0.0]], &[[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
        CrystalSystem::Rectangular => ([[1.0, 0.0], [0.0, 0.0]], &[[[0.0, 0.0], [0.0, 1.0]]]),
        CrystalSystem::Square => ([[1.0, 0.0], [0.0, 1.0]], &[]),
        CrystalSystem::Hexagonal => ([[1.0, -0.5], [-0.5, 1.0]], &[]),
    };
    let r0 = res(g0);
    if gs.is_empty() {
        return r0.iter().all(|x| x.abs() < 1e-9).then(Vec::new);
    }
    let cols: Vec<[f64; 3]> = gs.iter().map(|&gm| res(gm)).collect();
    let rows: Vec<Vec<f64>> = (0..3).map(|eq| cols.iter().map(|c| c[eq]).chain(std::iter::once(-r0[eq])).collect()).collect();
    rref(&rows)
}

#[allow(clippy::too_many_arguments)]
fn backtrack_aut(
    mf: [[i32; 2]; 2],
    metric: &[Vec<f64>],
    adj: &[Vec<(usize, [i32; 2])>],
    adjset: &[HashSet<(usize, [i32; 2])>],
    deg: &[usize],
    order: &mut Vec<usize>,
    pi: &mut [Option<usize>],
    s: &mut [[i32; 2]],
    used: &mut [bool],
    out: &mut Vec<Automorphism>,
) {
    let pick = order.iter().find_map(|&u| adj[u].iter().find_map(|&(v, n)| pi[v].is_none().then_some((u, v, n))));
    let Some((u, v, n)) = pick else {
        out.push(Automorphism { mf, pi: pi.iter().map(|x| x.unwrap()).collect(), s: s.to_vec(), metric: metric.to_vec() });
        return;
    };
    let (mn, su, wu) = (imat2_ivec(mf, n), s[u], pi[u].unwrap());
    for &(w, n2) in &adj[wu] {
        if used[w] || deg[w] != deg[v] {
            continue;
        }
        pi[v] = Some(w);
        s[v] = [n2[0] - mn[0] + su[0], n2[1] - mn[1] + su[1]];
        used[w] = true;
        order.push(v);
        let ok = adj[v].iter().all(|&(x, nx)| {
            let Some(wx) = pi[x] else { return true };
            let mnx = imat2_ivec(mf, nx);
            adjset[w].contains(&(wx, [mnx[0] + s[x][0] - s[v][0], mnx[1] + s[x][1] - s[v][1]]))
        });
        if ok {
            backtrack_aut(mf, metric, adj, adjset, deg, order, pi, s, used, out);
        }
        order.pop();
        used[w] = false;
        pi[v] = None;
    }
}

fn periodic_connected(m: usize, bonds: &[BareBond]) -> bool {
    if m == 0 { return false; }
    let mut adj = vec![Vec::new(); m];
    for &(i, j, n1, n2) in bonds {
        adj[i].push((j, n1, n2));
        adj[j].push((i, -n1, -n2));
    }
    let mut off: Vec<Option<[i32; 2]>> = vec![None; m];
    off[0] = Some([0, 0]);
    let (mut stack, mut seen_count, mut cyc) = (vec![0usize], 1usize, Vec::new());
    while let Some(u) = stack.pop() {
        let ou = off[u].unwrap();
        for &(v, d1, d2) in &adj[u] {
            let c0 = [ou[0] + d1, ou[1] + d2];
            if let Some(ov) = off[v] {
                if c0 != ov { cyc.push([c0[0] - ov[0], c0[1] - ov[1]]); }
            } else {
                off[v] = Some(c0);
                seen_count += 1;
                stack.push(v);
            }
        }
    }
    if seen_count < m { return false; }
    let mut g = 0i64;
    for p in 0..cyc.len() {
        for q in (p + 1)..cyc.len() {
            let mut b = (cyc[p][0] as i64 * cyc[q][1] as i64 - cyc[p][1] as i64 * cyc[q][0] as i64).abs();
            while b != 0 { (g, b) = (b, g % b); }
            if g == 1 { return true; }
        }
    }
    false
}

pub(crate) fn automorphisms_for_family(fam: &EvalFamily) -> Result<Vec<Automorphism>, String> {
    let m = fam.m();
    let bonds = fam.bonds();
    if m == 0 || !periodic_connected(m, &bonds) {
        return Err("automorphisms need a connected lattice".to_string());
    }
    let mut adj = vec![Vec::new(); m];
    for (i, j, n1, n2) in bonds {
        adj[i].push((j, [n1, n2]));
        adj[j].push((i, [-n1, -n2]));
    }
    let adjset: Vec<HashSet<(usize, [i32; 2])>> = adj.iter().map(|a| a.iter().copied().collect()).collect();
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let e_bound: i32 = if fam.g.system == CrystalSystem::Oblique { 2 } else { 1 };
    let mut out = Vec::new();
    for a in -e_bound..=e_bound {
        for b in -e_bound..=e_bound {
            for c in -e_bound..=e_bound {
                for d in -e_bound..=e_bound {
                    if (a * d - b * c).abs() != 1 {
                        continue;
                    }
                    let mf = [[a, b], [c, d]];
                    let mut p = [[1, 0], [0, 1]];
                    let finite = (0..6).any(|_| {
                        p = [[p[0][0] * mf[0][0] + p[0][1] * mf[1][0], p[0][0] * mf[0][1] + p[0][1] * mf[1][1]], [p[1][0] * mf[0][0] + p[1][1] * mf[1][0], p[1][0] * mf[0][1] + p[1][1] * mf[1][1]]];
                        p == [[1, 0], [0, 1]]
                    });
                    if !finite {
                        continue;
                    }
                    let Some(metric) = metric_system(fam, mf) else { continue };
                    for j0 in 0..m {
                        if deg[j0] == deg[0] && adj[0].iter().all(|&(x, nx)| x != 0 || adjset[j0].contains(&(j0, imat2_ivec(mf, nx)))) {
                            let (mut pi, mut s, mut used, mut order) = (vec![None; m], vec![[0, 0]; m], vec![false; m], vec![0usize]);
                            pi[0] = Some(j0);
                            used[j0] = true;
                            backtrack_aut(mf, &metric, &adj, &adjset, &deg, &mut order, &mut pi, &mut s, &mut used, &mut out);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn locus_of_aut(fam: &EvalFamily, aut: &Automorphism, p_mat: &[[Vec<f64>; 2]], q_vec: &[[f64; 2]]) -> Option<Vec<Vec<f64>>> {
    let (ny, d, m, mf) = (fam.metric_vars().len(), fam.site_names.len(), fam.m(), aut.mf);
    let mut rows: Vec<Vec<f64>> = aut.metric.iter().map(|mr| {
        mr[..ny].iter().copied().chain(std::iter::repeat_n(0.0, d)).chain(std::iter::once(mr[ny])).collect()
    }).collect();
    let e_list: Vec<[Vec<f64>; 2]> = (0..m).map(|k| {
        let (pk, p_pik) = (&p_mat[k], &p_mat[aut.pi[k]]);
        [
            (0..d).map(|col| mf[0][0] as f64 * pk[0][col] + mf[0][1] as f64 * pk[1][col] - p_pik[0][col]).collect(),
            (0..d).map(|col| mf[1][0] as f64 * pk[0][col] + mf[1][1] as f64 * pk[1][col] - p_pik[1][col]).collect(),
        ]
    }).collect();
    let r_list: Vec<[f64; 2]> = (0..m).map(|k| {
        let mq = imat2_vec(mf, q_vec[k]);
        [q_vec[aut.pi[k]][0] + aut.s[k][0] as f64 - mq[0], q_vec[aut.pi[k]][1] + aut.s[k][1] as f64 - mq[1]]
    }).collect();
    for k in 1..m {
        for c in 0..2 {
            let mut row = vec![0.0; ny];
            row.extend((0..d).map(|col| e_list[k][c][col] - e_list[0][c][col]));
            row.push(r_list[k][c] - r_list[0][c]);
            rows.push(row);
        }
    }
    if rows.is_empty() { Some(Vec::new()) } else { rref(&rows) }
}

fn aug_key(aug: &[Vec<f64>]) -> Vec<i64> {
    aug.iter().flatten().map(|&x| (x * 1e7).round() as i64).chain(std::iter::once(aug.len() as i64)).collect()
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        lo + (hi - lo) * (((z ^ (z >> 31)) >> 11) as f64 * (1.0 / ((1u64 << 53) as f64)))
    }
}

fn classify_locus(fam: &EvalFamily, loc: &mut Locus, rng: &mut SplitMix64) {
    loc.kind = LocusKind::Collision;
    for trial in 0..9 {
        let vals = if trial == 0 {
            loc.project_with_family(fam, None)
        } else {
            let mut rv = BTreeMap::new();
            if fam.cellpar.contains_key("b") {
                let mut r = rng.uniform(0.62, 1.7);
                for bad in [1.0, 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()] {
                    if (r - bad).abs() < 0.07 {
                        r += 0.16;
                    }
                }
                rv.insert("b".to_string(), r * fam.a);
            }
            if fam.cellpar.contains_key("gamma") {
                rv.insert("gamma".to_string(), rng.uniform(58.0, 86.0));
            }
            for nm in &fam.site_names {
                rv.insert(nm.clone(), rng.uniform(-0.97, 0.97));
            }
            loc.project_with_family(fam, Some(&rv))
        };
        if let Some(&b) = vals.get("b") && (b <= 0.2 * fam.a || b >= 5.0 * fam.a) {
            continue;
        }
        if let Some(&g_deg) = vals.get("gamma") && (g_deg <= 20.0 || g_deg >= 160.0) {
            continue;
        }
        let cell = fam.cell_at(&vals);
        if min_separation(&cell) < if trial == 0 { 0.05 } else { 1e-3 } {
            continue;
        }
        let Ok((grp, mp)) = identify_wallpaper_group(&cell) else { continue };
        loc.kind = if grp != fam.g.symbol || mp != fam.m() { LocusKind::Supergroup } else { LocusKind::Generic };
        loc.group = Some(grp);
        loc.sites = Some(mp);
        loc.example = Some(vals);
        return;
    }
}

pub(crate) fn all_loci_for_family(fam: &EvalFamily) -> Result<Vec<Locus>, String> {
    let (p_mat, q_vec) = fam.affine_sites();
    let (v_names, n_free) = (fam.v_names(), fam.names.len());
    let (mut found, mut order_keys): (HashMap<Vec<i64>, Locus>, Vec<Vec<i64>>) = (HashMap::new(), Vec::new());
    for aut in automorphisms_for_family(fam)? {
        let Some(aug) = locus_of_aut(fam, &aut, &p_mat, &q_vec) else { continue };
        let key = aug_key(&aug);
        if let Some(existing) = found.get_mut(&key) {
            existing.n_automorphisms += 1;
        } else {
            let codim = aug.len();
            let equations = format_equations(&aug, &v_names, fam.a);
            order_keys.push(key.clone());
            found.insert(key, Locus {
                aug, codim, dim: n_free.saturating_sub(codim), group: None, sites: None,
                kind: LocusKind::Collision, equations, example: None, n_automorphisms: 1,
            });
        }
    }
    let mut loci: Vec<Locus> = order_keys.iter().filter_map(|k| found.get(k).filter(|l| l.codim > 0).cloned()).collect();
    let mut frontier = loci.clone();
    while !frontier.is_empty() && found.len() < MAX_LOCI {
        let mut new_loci = Vec::new();
        for la in &frontier {
            for lb in &loci {
                let mut stacked = la.aug.clone();
                stacked.extend(lb.aug.iter().cloned());
                let Some(aug) = rref(&stacked) else { continue };
                if aug.len() <= la.codim.max(lb.codim) {
                    continue;
                }
                let key = aug_key(&aug);
                if !found.contains_key(&key) && found.len() < MAX_LOCI {
                    let codim = aug.len();
                    let loc = Locus {
                        equations: format_equations(&aug, &v_names, fam.a), aug, codim,
                        dim: n_free.saturating_sub(codim), group: None, sites: None,
                        kind: LocusKind::Collision, example: None, n_automorphisms: 0,
                    };
                    found.insert(key, loc.clone());
                    new_loci.push(loc);
                }
            }
        }
        loci.extend(new_loci.iter().cloned());
        frontier = new_loci;
    }
    let mut rng = SplitMix64(0x5eed_cafe_1234_5678);
    for loc in &mut loci {
        classify_locus(fam, loc, &mut rng);
    }
    loci.sort_by(|a, b| b.dim.cmp(&a.dim).then_with(|| aug_key(&a.aug).cmp(&aug_key(&b.aug))));
    Ok(loci)
}

pub(crate) fn supergroup_loci_for_family(fam: &EvalFamily) -> Result<Vec<Locus>, String> {
    Ok(all_loci_for_family(fam)?.into_iter().filter(|l| l.kind == LocusKind::Supergroup).collect())
}

/// Enumerate all periodic bond-graph automorphisms `(mf, pi, s, metric)` of `recipe`.
pub fn automorphisms(recipe: &Recipe) -> Result<Vec<Automorphism>, String> {
    automorphisms_for_family(&EvalFamily::from_recipe(recipe)?)
}

/// Compute all non-trivial symmetry/collision loci of `recipe` (`Family.loci`).
pub fn all_loci(recipe: &Recipe) -> Result<Vec<Locus>, String> {
    all_loci_for_family(&EvalFamily::from_recipe(recipe)?)
}

/// Compute all supergroup loci of `recipe` where symmetry strictly increases (`Family.supergroup_loci`).
pub fn supergroup_loci(recipe: &Recipe) -> Result<Vec<Locus>, String> {
    supergroup_loci_for_family(&EvalFamily::from_recipe(recipe)?)
}
