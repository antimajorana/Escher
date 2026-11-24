//! 2D Euclidean helper primitives and Volterra dislocation/disclination cone steps.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;

use crate::defect::cone_builder::{ConeBuilder, StepErr, ray_distance};
use crate::defect::core_graph::{SiteKey, TWO_PI};
use crate::types::Isometry2;

#[inline]
pub(crate) fn norm2(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}
#[inline]
pub(crate) fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline]
pub(crate) fn add2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline]
pub(crate) fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub(crate) fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub(crate) fn mat2_inv(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
}

#[inline]
pub(crate) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

pub(crate) fn rot_mat(angle_rad: f64) -> [[f64; 2]; 2] {
    let (s, c) = angle_rad.sin_cos();
    [[c, -s], [s, c]]
}

pub(crate) fn rotation_about(centre: [f64; 2], angle_rad: f64) -> Isometry2 {
    let m = rot_mat(angle_rad);
    Isometry2 { m, t: sub2(centre, mat2_vec(m, centre)) }
}

pub(crate) fn same_iso(a: &Isometry2, b: &Isometry2, tol: f64) -> bool {
    (a.m[0][0] - b.m[0][0]).abs() <= tol
        && (a.m[0][1] - b.m[0][1]).abs() <= tol
        && (a.m[1][0] - b.m[1][0]).abs() <= tol
        && (a.m[1][1] - b.m[1][1]).abs() <= tol
        && (a.t[0] - b.t[0]).abs() <= tol
        && (a.t[1] - b.t[1]).abs() <= tol
}

impl<'a> ConeBuilder<'a> {
    pub(crate) fn add_dislocation(&mut self, b: [f64; 2], p: [f64; 2], cut: Option<[f64; 2]>) -> Result<(), String> {
        let lb = norm2(b);
        let c = cut.unwrap_or([-b[1], b[0]]);
        let lc = norm2(c);
        if lc < 1e-9 {
            return Err("cut direction must be non-zero".to_string());
        }
        let cr = cross2(c, b);
        if cr.abs() <= 1e-6 * lc * lb {
            return Err("cut direction cannot be parallel to Burgers vector b".to_string());
        }
        if cr > 0.0 {
            return self.add_dislocation_insertion(b, p, c, lb, lc);
        }
        if self.clear(p) < lb + 2.5 * self.lmax {
            return Err("dislocation point is too close to patch rim; increase R".to_string());
        }
        let ch = [c[0] / lc, c[1] / lc];
        let nh = [-ch[1], ch[0]];
        let (bn, ba, m) = (dot2(b, nh), dot2(b, ch), 1.05 * self.lmax);
        let inside = |y: [f64; 2]| {
            let r = sub2(y, p);
            let (a, n) = (dot2(r, ch), dot2(r, nh));
            (bn - m <= n && n <= m) && a >= ba.min(0.0) - m
        };
        let base = self.base_site(p, &inside)?;
        let (d_map, y_map, _) = self.develop(base, inside)?;
        let mut n_verts: Vec<usize> = d_map.keys().copied().collect();
        n_verts.sort_unstable();
        let b_inv = mat2_inv([[b[0], c[0]], [b[1], c[1]]]);
        let ts: HashMap<usize, [f64; 2]> =
            n_verts.iter().map(|&i| (i, mat2_vec(b_inv, sub2(y_map[&i], p)))).collect();
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        for &i in &n_verts {
            let [t, s] = ts[&i];
            if s > 1e-9 && (-1e-9..1.0 - 1e-9).contains(&t) {
                gone.push(i);
            } else {
                kept.push(i);
            }
        }
        let mut rule: BTreeMap<(usize, usize), ([f64; 2], i32, usize)> = BTreeMap::new();
        for chart in 0..3 {
            let mut here: BTreeMap<SiteKey, (usize, i32)> = BTreeMap::new();
            for &i in &kept {
                let [t, s] = ts[&i];
                let sh_opt = match chart {
                    0 => (s <= 1e-9 || t < -1e-9).then_some(0),
                    1 => (t >= 1.0 - 1e-9).then_some(-1),
                    _ => (s > 1e-9).then_some(if t >= 1.0 - 1e-9 { -1 } else { 0 }),
                };
                if let Some(sh) = sh_opt {
                    let y_sh = [y_map[&i][0] + (sh as f64) * b[0], y_map[&i][1] + (sh as f64) * b[1]];
                    if let Some(kloc) = self.lat.locate(y_sh) {
                        here.insert(kloc, (i, sh));
                    }
                }
            }
            self.collect_chart_bonds(&here, &mut rule);
        }
        let (q_rot, xb, yb) = (self.local_rotation(base), self.x_pos[base], y_map[&base]);
        for &i in &gone {
            self.kill(i)?;
        }
        self.apply_rule(
            &rule,
            &kept,
            &d_map,
            |j| Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [(j as f64) * b[0], (j as f64) * b[1]] },
            &HashSet::new(),
        )?;
        let core = add2(xb, mat2_vec(q_rot, sub2([p[0] + 0.5 * b[0], p[1] + 0.5 * b[1]], yb)));
        let (bq, dq) = (mat2_vec(q_rot, b), mat2_vec(q_rot, ch));
        for i in 0..self.x_pos.len() {
            if self.alive[i] {
                let r = sub2(self.x_pos[i], core);
                let frac = 1.0 - (-cross2(dq, r)).atan2(dot2(r, dq)).rem_euclid(TWO_PI) / TWO_PI;
                self.x_pos[i] = [self.x_pos[i][0] + (0.5 - frac) * bq[0], self.x_pos[i][1] + (0.5 - frac) * bq[1]];
            }
        }
        self.records.push((gone.len(), 0));
        Ok(())
    }

    fn collect_chart_bonds(
        &self,
        here: &BTreeMap<SiteKey, (usize, i32)>,
        rule: &mut BTreeMap<(usize, usize), ([f64; 2], i32, usize)>,
    ) {
        for (&img, &(u, su)) in here {
            let p_img = self.lat.pos(img);
            for (k2, kind) in self.lat.nbrs(img) {
                if let Some(&(v, sv)) = here.get(&k2)
                    && v != u
                {
                    rule.insert((u, v), (sub2(self.lat.pos(k2), p_img), su - sv, kind));
                }
            }
        }
    }

    fn add_dislocation_insertion(
        &mut self,
        b: [f64; 2],
        p: [f64; 2],
        c: [f64; 2],
        lb: f64,
        lc: f64,
    ) -> Result<(), String> {
        if self.clear(p) < lb + 2.5 * self.lmax {
            return Err("dislocation point is too close to patch rim; increase R".to_string());
        }
        let ch = [c[0] / lc, c[1] / lc];
        let nh = [-ch[1], ch[0]];
        let (bn, ba, m) = (dot2(b, nh), dot2(b, ch), 1.05 * self.lmax);
        let inside = |y: [f64; 2]| {
            let r = sub2(y, p);
            let (a, n) = (dot2(r, ch), dot2(r, nh));
            (-m <= n && n <= bn + m) && a >= ba.min(0.0) - m
        };
        let base = self.base_site(p, &inside)?;
        let (mut d_map, y_map, _) = self.develop(base, inside)?;
        let mut n_verts: Vec<usize> = d_map.keys().copied().collect();
        n_verts.sort_unstable();
        let b_inv = mat2_inv([[b[0], c[0]], [b[1], c[1]]]);
        let ts: HashMap<usize, [f64; 2]> =
            n_verts.iter().map(|&i| (i, mat2_vec(b_inv, sub2(y_map[&i], p)))).collect();
        let mut dup: HashMap<usize, usize> = HashMap::new();
        let mut fresh: HashSet<usize> = HashSet::new();
        let mut kept: Vec<usize> = Vec::with_capacity(n_verts.len() + 16);
        for &i in &n_verts {
            kept.push(i);
            let [t, s] = ts[&i];
            if s > 1e-9 && (-1e-9..1.0 - 1e-9).contains(&t) {
                let nv = self.new_copy(i)?;
                dup.insert(i, nv);
                fresh.insert(nv);
                d_map.insert(nv, d_map[&i]);
            }
        }
        let dup_left: HashSet<usize> = dup.keys().copied().collect();
        for &i in &n_verts {
            if let Some(&nv) = dup.get(&i) {
                kept.push(nv);
            }
        }
        let mut rule: BTreeMap<(usize, usize), ([f64; 2], i32, usize)> = BTreeMap::new();
        for chart in 0..4 {
            let mut here: BTreeMap<SiteKey, (usize, i32)> = BTreeMap::new();
            for &i in &n_verts {
                let [t, s] = ts[&i];
                let y = y_map[&i];
                let mut put = |pos: [f64; 2], u: usize, sh: i32| {
                    if let Some(kloc) = self.lat.locate(pos) {
                        here.insert(kloc, (u, sh));
                    }
                };
                match chart {
                    0 if s <= 1e-9 => put(y, i, 0),
                    1 if t < -1e-9 || (s > 1e-9 && t < 1.0 - 1e-9) => put(y, i, 0),
                    2 => {
                        if s <= 1e-9 && t >= -1e-9 {
                            put(y, i, 0);
                        } else if let Some(&nv) = dup.get(&i) {
                            put(y, nv, 0);
                        } else if s > 1e-9 && t >= 1.0 - 1e-9 {
                            put(y, i, 0);
                        }
                    }
                    3 if s > 1e-9 => {
                        if t < 1.0 - 1e-9 {
                            put(y, i, 0);
                        }
                        if let Some(&nv) = dup.get(&i) {
                            put(add2(y, b), nv, 1);
                        } else if t >= 1.0 - 1e-9 {
                            put(add2(y, b), i, 1);
                        }
                    }
                    _ => {}
                }
            }
            self.collect_chart_bonds(&here, &mut rule);
        }
        let (q_rot, xb, yb) = (self.local_rotation(base), self.x_pos[base], y_map[&base]);
        self.apply_rule(
            &rule,
            &kept,
            &d_map,
            |j| Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [(j as f64) * b[0], (j as f64) * b[1]] },
            &fresh,
        )?;
        let core = add2(xb, mat2_vec(q_rot, sub2([p[0] + 0.5 * b[0], p[1] + 0.5 * b[1]], yb)));
        let (bq, dq) = (mat2_vec(q_rot, b), mat2_vec(q_rot, ch));
        for i in 0..self.x_pos.len() {
            if self.alive[i] {
                let frac = if fresh.contains(&i) {
                    0.0
                } else if dup_left.contains(&i) {
                    1.0
                } else {
                    let r = sub2(self.x_pos[i], core);
                    cross2(dq, r).atan2(dot2(r, dq)).rem_euclid(TWO_PI) / TWO_PI
                };
                self.x_pos[i] = [self.x_pos[i][0] + (0.5 - frac) * bq[0], self.x_pos[i][1] + (0.5 - frac) * bq[1]];
            }
        }
        self.records.push((0, dup.len()));
        Ok(())
    }

    pub(crate) fn add_disclination(&mut self, angle_deg: f64, p: [f64; 2], cut: [f64; 2]) -> Result<[f64; 2], String> {
        let snap = self.snapshot();
        let applied = self.disclination_nudged(angle_deg, p, cut)?;
        if self.records.last().copied().unwrap_or((0, 0)) != (0, 0) {
            return Ok(applied);
        }
        // FIX 1: face-centred disclination whose seam band misses nearest neighbours (e.g. chiral_square 4b).
        self.restore(snap);
        let near = self
            .ref_pos
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.alive[i])
            .map(|(_, &r)| norm2(sub2(r, p)))
            .fold(f64::INFINITY, f64::min);
        self.pad = Some((p, near + 1.05 * self.lmax));
        let res = self.disclination_nudged(angle_deg, p, cut);
        self.pad = None;
        let applied = res?;
        if self.records.last().copied().unwrap_or((0, 0)) == (0, 0) {
            return Err("the disclination step changed nothing; try another cut direction".to_string());
        }
        Ok(applied)
    }

    fn disclination_nudged(&mut self, angle_deg: f64, p: [f64; 2], cut: [f64; 2]) -> Result<[f64; 2], String> {
        if self.clear(p) < 3.5 * self.lmax {
            return Err("disclination point is too close to patch rim; increase R".to_string());
        }
        if norm2(cut) < 1e-9 {
            return Err("cut direction must be non-zero".to_string());
        }
        let th_cut = cut[1].atan2(cut[0]);
        for nudge in [0.0, 1e-5, -1e-5, 3e-5, -3e-5, 1e-4] {
            let snap = self.snapshot();
            let th = th_cut + nudge;
            match self.disclination_step(angle_deg, p, th) {
                Ok(()) => return Ok([th.cos(), th.sin()]),
                Err(StepErr::SeamHit) => self.restore(snap),
                Err(StepErr::Config(msg)) => {
                    self.restore(snap);
                    return Err(msg);
                }
            }
        }
        Err("lattice sites lie on the cut ray for all nearby nudges; choose another cut direction".to_string())
    }

    fn disclination_step(&mut self, angle_deg: f64, p: [f64; 2], th0: f64) -> Result<(), StepErr> {
        let alpha = angle_deg.to_radians();
        let theta_open = TWO_PI - alpha;
        let (eps, m) = (1e-7, 1.05 * self.lmax);
        let (u0, u1) = ([th0.cos(), th0.sin()], [(th0 - alpha).cos(), (th0 - alpha).sin()]);
        let inside = |y: [f64; 2]| {
            let r = sub2(y, p);
            if norm2(r) < 1e-9 {
                return true;
            }
            let th = (r[1].atan2(r[0]) - th0).rem_euclid(TWO_PI);
            if (alpha > 0.0 && th >= theta_open) || (alpha < 0.0 && th <= -alpha) {
                return true;
            }
            ray_distance(y, p, u0).min(ray_distance(y, p, u1)) <= m
        };
        let base = self.base_site(p, &inside).map_err(StepErr::Config)?;
        let (mut d_map, y_map, where_map) = self.develop(base, inside).map_err(StepErr::Config)?;
        let keyof: HashMap<usize, SiteKey> = where_map.iter().map(|(&k, &i)| (i, k)).collect();
        let mut y_sorted: Vec<(usize, [f64; 2])> = y_map.iter().map(|(&i, &y)| (i, y)).collect();
        y_sorted.sort_by_key(|&(i, _)| i);
        let mut lift: Vec<(usize, f64)> = Vec::new();
        let mut core: Option<usize> = None;
        for (i, y) in y_sorted {
            let r = sub2(y, p);
            if norm2(r) < 1e-9 {
                core = Some(i);
                continue;
            }
            let th = (r[1].atan2(r[0]) - th0).rem_euclid(TWO_PI);
            if th.min(TWO_PI - th) < 1e-6 {
                return Err(StepErr::SeamHit);
            }
            lift.push((i, th));
        }
        if core.is_some_and(|c| self.cone_sites.contains(&c)) {
            return Err(StepErr::Config("two disclinations cannot sit on the same lattice site".to_string()));
        }
        let (q_rot, xb, yb) = (self.local_rotation(base), self.x_pos[base], y_map[&base]);
        let mut rep: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut phi: HashMap<usize, f64> = HashMap::new();
        let (mut gone, mut fresh) = (Vec::new(), Vec::new());
        for &(i, th) in &lift {
            let mut k = 0usize;
            while th + TWO_PI * (k as f64) < theta_open - eps {
                let v = if k == 0 {
                    i
                } else {
                    let nv = self.new_copy(i).map_err(StepErr::Config)?;
                    fresh.push(nv);
                    d_map.insert(nv, d_map[&i]);
                    nv
                };
                rep.insert((i, k), v);
                phi.insert(v, th + TWO_PI * (k as f64));
                k += 1;
            }
            if k == 0 {
                gone.push(i);
            }
        }
        let mut rule: BTreeMap<(usize, usize), ([f64; 2], i32, usize)> = BTreeMap::new();
        for (&(i, _), &u) in &rep {
            let x = y_map[&i];
            for (k2, kind) in self.lat.nbrs(keyof[&i]) {
                let q = self.lat.pos(k2);
                if norm2(sub2(q, p)) < 1e-9 {
                    continue;
                }
                let (a, bb) = (sub2(x, p), sub2(q, p));
                if cross2(a, bb).abs() < 1e-9 && dot2(a, bb) < 0.0 {
                    continue;
                }
                let phq = phi[&u] + cross2(a, bb).atan2(dot2(a, bb));
                let j = ((phq + eps) / theta_open).floor() as i32;
                let rj = rot_mat(-(j as f64) * theta_open);
                let q2 = add2(p, mat2_vec(rj, bb));
                let k_sheet = ((phq - (j as f64) * theta_open + eps) / TWO_PI).floor() as i32;
                if k_sheet < 0 {
                    continue;
                }
                let Some(kloc) = self.lat.locate(q2) else { continue };
                let Some(&i2) = where_map.get(&kloc) else { continue };
                let Some(&v) = rep.get(&(i2, k_sheet as usize)) else { continue };
                if v != u && !rule.contains_key(&(u, v)) {
                    rule.insert((u, v), (sub2(q, x), j, kind));
                    rule.entry((v, u)).or_insert((mat2_vec(rj, sub2(x, q)), -j, kind));
                }
            }
        }
        let gone_set: HashSet<usize> = gone.iter().copied().collect();
        for &i in &gone {
            self.kill(i).map_err(StepErr::Config)?;
        }
        let mut kept_all: Vec<usize> = lift.iter().map(|&(i, _)| i).filter(|i| !gone_set.contains(i)).collect();
        kept_all.extend(fresh.iter().copied());
        let fresh_set: HashSet<usize> = fresh.iter().copied().collect();
        self.apply_rule(&rule, &kept_all, &d_map, |j| rotation_about(p, -(j as f64) * theta_open), &fresh_set)
            .map_err(StepErr::Config)?;
        if let Some(core_u) = core {
            self.cone_sites.insert(core_u);
            let core_key = self.lat.locate(p).ok_or_else(|| StepErr::Config("core off lattice".to_string()))?;
            for (k2, kind) in self.lat.nbrs(core_key) {
                let Some(&i2) = where_map.get(&k2) else { continue };
                let q = self.lat.pos(k2);
                for k in 0..3 {
                    let Some(&v) = rep.get(&(i2, k)) else { continue };
                    if k > 0 || !self.nb[core_u].contains_key(&v) {
                        self.set_bond(core_u, v, sub2(q, p), Isometry2::IDENTITY, kind, &d_map, Some(core_u), None)
                            .map_err(StepErr::Config)?;
                        self.set_bond(v, core_u, sub2(p, q), Isometry2::IDENTITY, kind, &d_map, Some(core_u), None)
                            .map_err(StepErr::Config)?;
                    }
                    if let Some(eb) = self.nb[core_u].get_mut(&v) {
                        eb.ang = Some(phi[&v]);
                    }
                }
            }
        }
        let xc = add2(xb, mat2_vec(q_rot, sub2(p, yb)));
        let thd = th0 + q_rot[1][0].atan2(q_rot[0][0]);
        for i in 0..self.x_pos.len() {
            if !self.alive[i] || Some(i) == core {
                continue;
            }
            let r = sub2(self.x_pos[i], xc);
            let rad = norm2(r);
            if rad < 1e-12 {
                continue;
            }
            let mut th = (r[1].atan2(r[0]) - thd).rem_euclid(TWO_PI);
            if let Some(&phi_i) = phi.get(&i) {
                th = phi_i + (th - phi_i + PI).rem_euclid(TWO_PI) - PI;
            }
            let half_shift = 0.5 * (TWO_PI - theta_open);
            let ang = th * TWO_PI / theta_open + thd - half_shift;
            self.x_pos[i] = [xc[0] + rad * ang.cos(), xc[1] + rad * ang.sin()];
        }
        self.records.push((gone.len(), fresh.len()));
        Ok(())
    }
}
