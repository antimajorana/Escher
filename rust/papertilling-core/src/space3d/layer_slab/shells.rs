//! The shells-until-connected bond rule (plan Q1) at rank 2, and the quotient-graph state it uses.
//! Mirrors `_layer_slab.shell_bonds` / `quotient_state` / `lattice_rank_index`.
//!
//! Whole distance shells (ties at relative [`SHELL_RTOL`]) are taken in increasing distance; a shell
//! that changes none of (components, rank, index) of the quotient graph is skipped; growth of the
//! bond set stops at (1, 2, 1): one connected 2-periodic net (index > 1 would be interpenetrating
//! copies). A shell is a union of bond orbits when the site set is invariant, so the bond set keeps
//! the group. Refused when a third site lies on a bond or the sites of one orbit get two degrees.

/// A bond: site i in cell 0 to site j in cell n.
pub type Bond = (usize, usize, [i64; 2]);

pub const SHELL_RTOL: f64 = 1e-7;
pub const ON_BOND_TOL: f64 = 1e-6;
pub const MAX_SHELLS: usize = 200;

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Floor division (Python's `//`).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

/// (rank, index) of the integer lattice spanned by `v` in Z^d (index 0 unless the rank is d).
pub fn lattice_rank_index(v: &[Vec<i64>], d: usize) -> (usize, i64) {
    let mut rows: Vec<Vec<i64>> = v.iter().filter(|r| r.iter().any(|&x| x != 0)).cloned().collect();
    let mut piv_rows: Vec<Vec<i64>> = Vec::new();
    for col in 0..d {
        let (mut piv, mut rest): (Vec<Vec<i64>>, Vec<Vec<i64>>) = rows.into_iter().partition(|r| r[col] != 0);
        while piv.len() > 1 {
            piv.sort_by_key(|r| r[col].abs());
            let p = piv[0].clone();
            let mut next = vec![p.clone()];
            for r in piv.into_iter().skip(1) {
                let q = floor_div(r[col], p[col]);
                let r2: Vec<i64> = r.iter().zip(&p).map(|(a, b)| a - q * b).collect();
                if r2[col] != 0 { next.push(r2) } else { rest.push(r2) }
            }
            piv = next;
        }
        if let Some(p) = piv.pop() {
            piv_rows.push(p);
        }
        rows = rest.into_iter().filter(|r| r.iter().any(|&x| x != 0)).collect();
    }
    let rank = piv_rows.len();
    if rank < d {
        return (rank, 0);
    }
    let idx = piv_rows.iter().enumerate().fold(1_i64, |a, (k, r)| a * r[k]);
    (rank, idx.abs())
}

/// (components, rank, index) of the quotient graph with m sites and these bonds.
pub fn quotient_state(m: usize, bonds: &[Bond]) -> (usize, usize, i64) {
    let mut adj: Vec<Vec<(usize, [i64; 2])>> = vec![Vec::new(); m];
    for &(i, j, n) in bonds {
        adj[i].push((j, n));
        adj[j].push((i, [-n[0], -n[1]]));
    }
    let mut pot: Vec<Option<[i64; 2]>> = vec![None; m];
    let mut comps = 0;
    let mut cyc: Vec<Vec<i64>> = Vec::new();
    for s in 0..m {
        if pot[s].is_some() {
            continue;
        }
        comps += 1;
        pot[s] = Some([0, 0]);
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            let pu = pot[u].expect("visited");
            for &(v, n) in &adj[u] {
                let w = [pu[0] + n[0], pu[1] + n[1]];
                match pot[v] {
                    None => {
                        pot[v] = Some(w);
                        stack.push(v);
                    }
                    Some(pv) if pv != w => cyc.push(vec![w[0] - pv[0], w[1] - pv[1]]),
                    _ => {}
                }
            }
        }
    }
    let (rank, index) = lattice_rank_index(&cyc, 2);
    (comps, rank, index)
}

/// Column norms of the inverse of the in-plane 2x2 lattice block (|f_k| <= |x| g_k).
pub fn inverse_column_norms(l: &[[f64; 3]; 2]) -> [f64; 2] {
    let det = l[0][0] * l[1][1] - l[0][1] * l[1][0];
    [(l[1][1] / det).hypot(l[1][0] / det), (l[0][1] / det).hypot(l[0][0] / det)]
}

/// A pair within the cutoff: (distance, i, j, n, displacement).
pub type Pair = (f64, usize, usize, [i64; 2], [f64; 3]);

/// Pairs (i ≤ j, n) with 1e-9 < |P_j + n·L − P_i| ≤ r, once each (i == j needs n > 0), sorted by
/// (distance, i, j, n); with their displacement vectors.
pub fn pairs_within(p: &[[f64; 3]], l: &[[f64; 3]; 2], r: f64) -> Vec<Pair> {
    let g = inverse_column_norms(l);
    let lim = g.map(|gk| (r * gk).ceil() as i64 + 1);
    let mut out = Vec::new();
    for n0 in -lim[0]..=lim[0] {
        for n1 in -lim[1]..=lim[1] {
            let shift = [0, 1, 2].map(|c| n0 as f64 * l[0][c] + n1 as f64 * l[1][c]);
            for i in 0..p.len() {
                for j in i..p.len() {
                    if i == j && [n0, n1] <= [0, 0] {
                        continue;
                    }
                    let v = [p[j][0] + shift[0] - p[i][0], p[j][1] + shift[1] - p[i][1], p[j][2] + shift[2] - p[i][2]];
                    let dist = dot(&v, &v).sqrt();
                    if dist > 1e-9 && dist <= r {
                        out.push((dist, i, j, [n0, n1], v));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2, a.3).cmp(&(b.1, b.2, b.3))));
    out
}

/// The shells-until-connected bonds (sorted) and the lengths of the shells used.
pub fn shell_bonds(p: &[[f64; 3]], l: &[[f64; 3]; 2], labels: &[usize]) -> Result<(Vec<Bond>, Vec<f64>), String> {
    let m = p.len();
    let area = (l[0][0] * l[1][1] - l[0][1] * l[1][0]).abs();
    let mut r = 1.6 * (area / m as f64).sqrt();
    loop {
        let pairs = pairs_within(p, l, r);
        let (mut edges, mut lengths): (Vec<Bond>, Vec<f64>) = (Vec::new(), Vec::new());
        let (mut cur, mut used, mut k) = ((m, 0, 0), 0, 0);
        while k < pairs.len() {
            let (d0, k0) = (pairs[k].0, k);
            if d0 * (1.0 + SHELL_RTOL) > r {
                break;
            }
            while k < pairs.len() && pairs[k].0 <= d0 * (1.0 + SHELL_RTOL) {
                k += 1;
            }
            used += 1;
            if used > MAX_SHELLS {
                return Err(format!("no connected 2-periodic net within {MAX_SHELLS} shells"));
            }
            let mut trial = edges.clone();
            trial.extend(pairs[k0..k].iter().map(|x| (x.1, x.2, x.3)));
            let new = quotient_state(m, &trial);
            if new != cur {
                edges = trial;
                lengths.push(d0);
                cur = new;
                if cur == (1, 2, 1) {
                    validate(p, l, labels, &edges, r)?;
                    edges.sort_unstable();
                    return Ok((edges, lengths));
                }
            }
        }
        r *= 1.5;
    }
}

fn validate(p: &[[f64; 3]], l: &[[f64; 3]; 2], labels: &[usize], edges: &[Bond], r: f64) -> Result<(), String> {
    let mut deg = vec![0usize; p.len()];
    for &(i, j, _) in edges {
        deg[i] += 1;
        deg[j] += 1;
    }
    let nl = labels.iter().max().map_or(0, |x| x + 1);
    for o in 0..nl {
        let mut ds = (0..p.len()).filter(|&s| labels[s] == o).map(|s| deg[s]);
        if let Some(first) = ds.next()
            && ds.any(|x| x != first)
        {
            return Err("sites of one orbit have different degrees (tolerance failure)".into());
        }
    }
    let mut disp: Vec<Vec<[f64; 3]>> = vec![Vec::new(); p.len()];
    for (_, i, j, _, v) in pairs_within(p, l, r) {
        disp[i].push(v);
        disp[j].push(v.map(|x| -x));
    }
    for &(i, j, n) in edges {
        let shift = [0, 1, 2].map(|c| n[0] as f64 * l[0][c] + n[1] as f64 * l[1][c]);
        let b = sub(&[p[j][0] + shift[0], p[j][1] + shift[1], p[j][2] + shift[2]], &p[i]);
        let bb = dot(&b, &b);
        for w in &disp[i] {
            let t = dot(w, &b) / bb;
            let c = cross(w, &b);
            let perp = dot(&c, &c).sqrt() / bb.sqrt();
            if t > 1e-9 && t < 1.0 - 1e-9 && perp < ON_BOND_TOL * bb.sqrt() {
                return Err("a third site lies on a bond".into());
            }
        }
    }
    Ok(())
}
