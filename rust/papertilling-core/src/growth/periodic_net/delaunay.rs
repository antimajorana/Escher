//! Delaunay polytopes of a 2- or 3-periodic point set: the maximal sets of sites on an empty sphere,
//! one per translation class (the search and its certificate are stated in `super`'s module docs).
//! Mirrors `_periodic_net.PeriodicNet._delaunay` step for step.

use std::collections::BTreeMap;

use serde::Serialize;

/// A site of the periodic set: (site index, lattice shift; the third entry is 0 when d = 2).
pub type Member = (u32, [i64; 3]);

/// Relative tolerance: a site within `RTOL · ρ` of a sphere is on it; strictly inside below that.
pub const RTOL: f64 = 1e-7;
/// The polytopes found must fill the cell measure to this relative tolerance.
pub const MEASURE_RTOL: f64 = 1e-6;
/// Search rounds (R and N grow by 1.5 per round).
pub const MAX_ROUNDS: usize = 6;
/// Spheres pass through the site and `dim` of its `N_NEAREST` nearest neighbours (first round).
pub const N_NEAREST: usize = 20;

/// The three regimes (module docs of `super`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NetMode {
    /// d = 2, every site at one height: 2D Voronoi.
    Flat,
    /// d = 2, sites at several heights: 3D Voronoi with unbounded cells.
    Slab,
    /// d = 3.
    Space,
}

pub(super) struct Periodic<'a> {
    pub lattice: [[f64; 3]; 3],
    pub d: usize,
    pub sites: &'a [[f64; 3]],
    pub mode: NetMode,
    pub zmin: f64,
    pub zmax: f64,
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn det2(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * a[1][1] - a[0][1] * a[1][0]
}

fn det3(a: &[[f64; 3]; 3]) -> f64 {
    dot(&a[0], &cross(&a[1], &a[2]))
}

/// Cramer's rule for the `dim` x `dim` system A x = b (unused entries are 0).
fn solve(a: &[[f64; 3]; 3], b: &[f64; 3], dim: usize, det: f64) -> [f64; 3] {
    let mut x = [0.0; 3];
    for (k, xk) in x.iter_mut().enumerate().take(dim) {
        let mut ak = *a;
        for (row, br) in ak.iter_mut().zip(b.iter()).take(dim) {
            row[k] = *br;
        }
        *xk = if dim == 2 { det2(&ak) } else { det3(&ak) } / det;
    }
    x
}

/// Area of a convex polygon given by its points (any order) about an interior point `c` (xy only).
pub fn polygon_area(pts: &[[f64; 3]], c: &[f64; 3]) -> f64 {
    let mut q: Vec<(f64, [f64; 3])> = pts.iter().map(|p| ((p[1] - c[1]).atan2(p[0] - c[0]), *p)).collect();
    q.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = q.len();
    let mut s = 0.0;
    for k in 0..n {
        let (a, b) = (q[k].1, q[(k + 1) % n].1);
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s.abs()
}

/// Volume of the convex hull of a few points in general 3D position (brute-force facets).
pub fn hull_volume(pts: &[[f64; 3]]) -> f64 {
    let k = pts.len();
    let g = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / k as f64, a[1] + p[1] / k as f64, a[2] + p[2] / k as f64]);
    let scale = pts.iter().flat_map(|p| sub(p, &g)).fold(0.0_f64, |m, x| m.max(x.abs()));
    let tol = 1e-9 * scale;
    let mut seen: Vec<Vec<usize>> = Vec::new();
    let mut vol = 0.0;
    for a in 0..k {
        for b in (a + 1)..k {
            for c in (b + 1)..k {
                let n = cross(&sub(&pts[b], &pts[a]), &sub(&pts[c], &pts[a]));
                let nn = norm(&n);
                if nn <= 1e-12 * scale * scale {
                    continue;
                }
                let mut n = n.map(|x| x / nn);
                let mut s: Vec<f64> = pts.iter().map(|p| dot(&sub(p, &pts[a]), &n)).collect();
                let (lo, hi) = s.iter().fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
                if hi > tol {
                    if lo < -tol {
                        continue;
                    }
                    n = n.map(|x| -x);
                    s.iter_mut().for_each(|x| *x = -*x);
                }
                let on: Vec<usize> = (0..k).filter(|&i| s[i].abs() <= tol).collect();
                if seen.contains(&on) {
                    continue;
                }
                let f: Vec<[f64; 3]> = on.iter().map(|&i| pts[i]).collect();
                let m = f.len() as f64;
                let fc = f.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / m, a[1] + p[1] / m, a[2] + p[2] / m]);
                let u0 = sub(&f[0], &fc);
                let u = u0.map(|x| x / norm(&u0));
                let v = cross(&n, &u);
                let q: Vec<[f64; 3]> = f.iter().map(|p| { let d = sub(p, &fc); [dot(&d, &u), dot(&d, &v), 0.0] }).collect();
                vol += polygon_area(&q, &[0.0; 3]) * dot(&sub(&pts[a], &g), &n) / 3.0;
                seen.push(on);
            }
        }
    }
    vol
}

/// Translation-normalised sorted members: the smallest form over the members of least site index.
pub fn canonical(t: &[Member]) -> Vec<Member> {
    let lo = t.iter().map(|m| m.0).min().unwrap_or(0);
    let mut best: Option<Vec<Member>> = None;
    for e in t.iter().filter(|m| m.0 == lo) {
        let mut rep: Vec<Member> =
            t.iter().map(|&(s, n)| (s, [n[0] - e.1[0], n[1] - e.1[1], n[2] - e.1[2]])).collect();
        rep.sort_unstable();
        if best.as_ref().is_none_or(|b| rep < *b) {
            best = Some(rep);
        }
    }
    best.unwrap_or_default()
}

impl Periodic<'_> {
    fn target(&self) -> f64 {
        let l = &self.lattice;
        match self.mode {
            NetMode::Space => det3(l).abs(),
            NetMode::Flat => norm(&cross(&l[0], &l[1])),
            NetMode::Slab => norm(&cross(&l[0], &l[1])) * (self.zmax - self.zmin),
        }
    }

    fn work(&self, p: &[f64; 3]) -> [f64; 3] {
        if self.mode == NetMode::Flat { [p[0], p[1], 0.0] } else { *p }
    }

    /// Every (site, shift) within distance R of site i (in-plane distance for a slab), in loop order.
    fn neighbourhood(&self, i: usize, r: f64) -> (Vec<Member>, Vec<[f64; 3]>) {
        let d = self.d;
        let mut lp = [[0.0; 3]; 3];
        for (a, row) in lp.iter_mut().enumerate().take(d) {
            row[..d].copy_from_slice(&self.lattice[a][..d]);
        }
        // column norms of the inverse of the d x d in-plane block
        let g: Vec<f64> = if d == 2 {
            let det = det2(&lp);
            let inv = [[lp[1][1] / det, -lp[0][1] / det], [-lp[1][0] / det, lp[0][0] / det]];
            (0..2).map(|k| inv[0][k].hypot(inv[1][k])).collect()
        } else {
            let det = det3(&lp);
            let cols = [cross(&lp[1], &lp[2]), cross(&lp[2], &lp[0]), cross(&lp[0], &lp[1])];
            // rows r0, r1, r2: the k-th column of the inverse is (r_{k+1} x r_{k+2}) / det
            cols.iter().map(|c| norm(c) / det.abs()).collect()
        };
        let lim: Vec<i64> = g.iter().map(|gk| (r * gk).ceil() as i64 + 1).collect();
        let mut ids = Vec::new();
        let mut pos = Vec::new();
        let mut n = vec![0_i64; d];
        let mut push = |n: &[i64]| {
            let n3 = [n[0], n[1], if d == 3 { n[2] } else { 0 }];
            let shift = [0, 1, 2].map(|c| (0..3).map(|a| n3[a] as f64 * self.lattice[a][c]).sum::<f64>());
            for (j, p) in self.sites.iter().enumerate() {
                let q = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
                let dv = sub(&q, &self.sites[i]);
                let dist = if self.mode == NetMode::Slab { dv[0].hypot(dv[1]) } else { norm(&dv) };
                if dist <= r {
                    ids.push((j as u32, n3));
                    pos.push(q);
                }
            }
        };
        // itertools.product order: last index fastest
        for (nk, lk) in n.iter_mut().zip(&lim) {
            *nk = -lk;
        }
        'outer: loop {
            push(&n);
            let mut k = d - 1;
            loop {
                n[k] += 1;
                if n[k] <= lim[k] {
                    break;
                }
                if k == 0 {
                    break 'outer;
                }
                n[k] = -lim[k];
                k -= 1;
            }
        }
        (ids, pos)
    }

    fn footprint_ok(&self, c: &[f64; 3], rho: f64, pi: &[f64; 3], r: f64) -> bool {
        if self.mode != NetMode::Slab {
            return norm(&sub(c, &self.work(pi))) + rho <= r;
        }
        let dz = if self.zmin <= c[2] && c[2] <= self.zmax {
            0.0
        } else {
            (c[2] - self.zmin).abs().min((c[2] - self.zmax).abs())
        };
        let rf = (rho * rho - dz * dz).max(0.0).sqrt();
        (c[0] - pi[0]).hypot(c[1] - pi[1]) + rf <= r
    }

    /// The Delaunay polytopes, sorted, or a refusal when they never fill the cell.
    pub fn polytopes(&self) -> Result<Vec<Vec<Member>>, String> {
        let m = self.sites.len();
        let target = self.target();
        let dim = if self.mode == NetMode::Flat { 2 } else { 3 };
        let (meas, expo) = match self.mode {
            NetMode::Slab => (target / (self.zmax - self.zmin), 2.0),
            NetMode::Flat => (target, 2.0),
            NetMode::Space => (target, 3.0),
        };
        let mut r = 2.5 * (meas / m as f64).powf(1.0 / expo);
        let mut nc = N_NEAREST;
        let mut total = 0.0;
        for _ in 0..MAX_ROUNDS {
            let mut found: BTreeMap<Vec<Member>, f64> = BTreeMap::new();
            let mut uncertified = false;
            for i in 0..m {
                let (ids, pos) = self.neighbourhood(i, r);
                let me = ids.iter().position(|x| *x == (i as u32, [0, 0, 0])).expect("site in its own neighbourhood");
                let w: Vec<[f64; 3]> = pos.iter().map(|p| self.work(p)).collect();
                let mut order: Vec<usize> = (0..ids.len()).collect();
                let d3: Vec<f64> = pos.iter().map(|p| norm(&sub(p, &pos[me]))).collect();
                order.sort_by(|a, b| d3[*a].total_cmp(&d3[*b]));
                let mut others: Vec<usize> = order.into_iter().filter(|&k| k != me).take(nc).collect();
                others.sort_unstable();
                let q0 = w[me];
                let mut combo = vec![0usize; dim];
                let no = others.len();
                if no < dim {
                    continue;
                }
                for (k, c) in combo.iter_mut().enumerate() {
                    *c = k;
                }
                loop {
                    self.try_sphere(&combo.iter().map(|&k| others[k]).collect::<Vec<_>>(), me, &ids, &pos, &w, q0, dim, r,
                        &mut found, &mut uncertified);
                    // next combination (lexicographic)
                    let mut k = dim;
                    let mut advanced = false;
                    while k > 0 {
                        k -= 1;
                        if combo[k] < no - dim + k {
                            combo[k] += 1;
                            for t in (k + 1)..dim {
                                combo[t] = combo[t - 1] + 1;
                            }
                            advanced = true;
                            break;
                        }
                    }
                    if !advanced {
                        break;
                    }
                }
            }
            total = found.values().sum();
            if !uncertified && (total - target).abs() <= MEASURE_RTOL * target {
                return Ok(found.into_keys().collect());
            }
            r *= 1.5;
            nc = (nc as f64 * 1.5) as usize;
        }
        Err(format!("Dirichlet complex not certified: Delaunay polytopes fill {total:.6} of {target:.6}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_sphere(
        &self,
        combo: &[usize],
        me: usize,
        ids: &[Member],
        pos: &[[f64; 3]],
        w: &[[f64; 3]],
        q0: [f64; 3],
        dim: usize,
        r: f64,
        found: &mut BTreeMap<Vec<Member>, f64>,
        uncertified: &mut bool,
    ) {
        let mut a = [[0.0; 3]; 3];
        let mut b = [0.0; 3];
        let mut scale: f64 = 0.0;
        for (row, &k) in combo.iter().enumerate() {
            let dq = sub(&w[k], &q0);
            for (ac, dqc) in a[row].iter_mut().zip(dq.iter()).take(dim) {
                *ac = 2.0 * dqc;
                scale = scale.max(dqc.abs());
            }
            b[row] = dot(&w[k], &w[k]) - dot(&q0, &q0);
        }
        let scale = scale.max(1e-300);
        let det = if dim == 2 { det2(&a) } else { det3(&a) };
        if det.abs() <= 1e-12 * (2.0 * scale).powi(dim as i32) {
            return;
        }
        let c = solve(&a, &b, dim, det);
        let rho = norm(&sub(&q0, &c));
        let dist: Vec<f64> = w.iter().map(|p| norm(&sub(p, &c))).collect();
        if dist.iter().any(|&x| x < rho * (1.0 - RTOL)) {
            return;
        }
        if !self.footprint_ok(&c, rho, &pos[me], r) {
            *uncertified = true;
            return;
        }
        let on: Vec<usize> = (0..w.len()).filter(|&k| (dist[k] - rho).abs() <= RTOL * rho).collect();
        let key = canonical(&on.iter().map(|&k| ids[k]).collect::<Vec<_>>());
        found.entry(key).or_insert_with(|| {
            let pts: Vec<[f64; 3]> = on.iter().map(|&k| w[k]).collect();
            if self.mode == NetMode::Flat { polygon_area(&pts, &c) } else { hull_volume(&pts) }
        });
    }
}
