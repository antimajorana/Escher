//! Stateful cone-safe multi-sheeted Volterra cut-and-reglue patch engine.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use crate::defect::core_graph::{
    LatticeCoreInfo, PerfectLattice, SiteKey, cross2, dot2, mat2_vec, norm2, rot_mat, same_iso,
    sub2,
};
use crate::types::{BuiltCrystal, DirectedBondData, Isometry2};

pub(crate) fn ray_distance(y: [f64; 2], p: [f64; 2], u: [f64; 2]) -> f64 {
    let r = sub2(y, p);
    if dot2(r, u) < 0.0 { norm2(r) } else { cross2(u, r).abs() }
}

#[derive(Clone, Debug)]
pub(super) struct RawBond {
    pub(super) vec: [f64; 2],
    pub(super) kind: usize,
    pub(super) seam: Option<Isometry2>,
    pub(super) ang: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct Snapshot {
    pub(super) label: Vec<(SiteKey, usize)>,
    pub(super) ref_pos: Vec<[f64; 2]>,
    pub(super) alive: Vec<bool>,
    pub(super) nb: Vec<BTreeMap<usize, RawBond>>,
    pub(super) x_pos: Vec<[f64; 2]>,
    pub(super) layers: HashMap<SiteKey, usize>,
    pub(super) cone_sites: HashSet<usize>,
    pub(super) records: Vec<(usize, usize)>,
}

pub(super) enum StepErr {
    SeamHit,
    Config(String),
}

type DevelopChart = (HashMap<usize, Isometry2>, HashMap<usize, [f64; 2]>, HashMap<SiteKey, usize>);

pub(crate) struct ConeBuilder<'a> {
    pub(super) lat: &'a PerfectLattice,
    pub(super) centre: [f64; 2],
    pub(super) r_patch: f64,
    pub(super) lmax: f64,
    pub(super) scale: f64,
    pub(super) label: Vec<(SiteKey, usize)>,
    pub(super) ref_pos: Vec<[f64; 2]>,
    pub(super) alive: Vec<bool>,
    pub(super) nb: Vec<BTreeMap<usize, RawBond>>,
    pub(super) x_pos: Vec<[f64; 2]>,
    pub(super) layers: HashMap<SiteKey, usize>,
    pub(super) cone_sites: HashSet<usize>,
    pub(super) records: Vec<(usize, usize)>,
    pub(super) pad: Option<([f64; 2], f64)>,
}

impl<'a> ConeBuilder<'a> {
    pub(crate) fn new(lat_info: &'a LatticeCoreInfo, centre: [f64; 2], r_patch: f64) -> Self {
        let lat = &lat_info.lat;
        let keys = lat.sites_in_disk(centre, r_patch);
        let n = keys.len();
        let label: Vec<(SiteKey, usize)> = keys.iter().map(|&k| (k, 0)).collect();
        let ref_pos: Vec<[f64; 2]> = keys.iter().map(|&k| lat.pos(k)).collect();
        let index: HashMap<SiteKey, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
        let mut nb = vec![BTreeMap::new(); n];
        for (i, &k) in keys.iter().enumerate() {
            for (k2, kind) in lat.nbrs(k) {
                if let Some(&j) = index.get(&k2) {
                    nb[i].insert(j, RawBond { vec: sub2(ref_pos[j], ref_pos[i]), kind, seam: None, ang: None });
                }
            }
        }
        let layers: HashMap<SiteKey, usize> = keys.into_iter().map(|k| (k, 1)).collect();
        Self {
            lat,
            centre,
            r_patch,
            lmax: lat_info.lmax,
            scale: lat_info.scale,
            label,
            x_pos: ref_pos.clone(),
            ref_pos,
            alive: vec![true; n],
            nb,
            layers,
            cone_sites: HashSet::new(),
            records: Vec::new(),
            pad: None,
        }
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            label: self.label.clone(),
            ref_pos: self.ref_pos.clone(),
            alive: self.alive.clone(),
            nb: self.nb.clone(),
            x_pos: self.x_pos.clone(),
            layers: self.layers.clone(),
            cone_sites: self.cone_sites.clone(),
            records: self.records.clone(),
        }
    }

    pub(crate) fn restore(&mut self, snap: Snapshot) {
        self.label = snap.label;
        self.ref_pos = snap.ref_pos;
        self.alive = snap.alive;
        self.nb = snap.nb;
        self.x_pos = snap.x_pos;
        self.layers = snap.layers;
        self.cone_sites = snap.cone_sites;
        self.records = snap.records;
    }

    pub(super) fn clear(&self, p: [f64; 2]) -> f64 {
        self.r_patch - norm2(sub2(p, self.centre))
    }

    pub(super) fn kill(&mut self, i: usize) -> Result<(), String> {
        if self.cone_sites.contains(&i) {
            return Err("a later cut would remove an existing site-centred disclination core".to_string());
        }
        let nbrs: Vec<usize> = self.nb[i].keys().copied().collect();
        for v in nbrs {
            if self.cone_sites.contains(&v) {
                return Err("a later cut removes a neighbour of a site-centred disclination core".to_string());
            }
            self.nb[v].remove(&i);
        }
        self.nb[i].clear();
        self.alive[i] = false;
        Ok(())
    }

    pub(super) fn new_copy(&mut self, i: usize) -> Result<usize, String> {
        if self.cone_sites.contains(&i) {
            return Err("a later cut duplicates an existing site-centred disclination core".to_string());
        }
        let key = self.label[i].0;
        let layer = self.layers.get(&key).copied().unwrap_or(0);
        self.layers.insert(key, layer + 1);
        self.label.push((key, layer));
        self.ref_pos.push(self.ref_pos[i]);
        self.alive.push(true);
        self.nb.push(BTreeMap::new());
        self.x_pos.push(self.x_pos[i]);
        Ok(self.label.len() - 1)
    }

    pub(super) fn base_site<F: Fn([f64; 2]) -> bool>(&self, p: [f64; 2], inside: &F) -> Result<usize, String> {
        let inside_eff = |y: [f64; 2]| inside(y) || self.pad.is_some_and(|(q, rho)| norm2(sub2(y, q)) <= rho);
        let mut best: Option<(i64, usize, usize)> = None;
        for i in 0..self.label.len() {
            if self.alive[i] && !self.cone_sites.contains(&i) && inside_eff(self.ref_pos[i]) {
                let d7 = (norm2(sub2(self.ref_pos[i], p)) * 1e7).round() as i64;
                let k = (d7, self.label[i].1, i);
                if best.is_none_or(|b| k < b) {
                    best = Some(k);
                }
            }
        }
        match best {
            Some((d7, _, idx)) if (d7 as f64) * 1e-7 <= self.lmax + self.scale => Ok(idx),
            _ => Err(format!("no crystal site near {p:?}: point is outside patch or inside a removed region")),
        }
    }

    pub(super) fn develop<F: Fn([f64; 2]) -> bool>(
        &self,
        base: usize,
        inside: F,
    ) -> Result<DevelopChart, String> {
        let inside_eff = |y: [f64; 2]| inside(y) || self.pad.is_some_and(|(q, rho)| norm2(sub2(y, q)) <= rho);
        let mut d_map = HashMap::from([(base, Isometry2::IDENTITY)]);
        let mut y_map = HashMap::from([(base, self.ref_pos[base])]);
        let mut dq = VecDeque::from([base]);
        while let Some(u) = dq.pop_front() {
            let du = d_map[&u];
            for (&v, e) in &self.nb[u] {
                let dv = e.seam.map_or(du, |g| du.compose(&g.inverse()));
                if let Some(&existing) = d_map.get(&v) {
                    if !same_iso(&dv, &existing, 1e-6) {
                        return Err("developed region encloses an earlier defect core (two walks disagree)".to_string());
                    }
                    continue;
                }
                let yv = dv.apply(self.ref_pos[v]);
                if inside_eff(yv) {
                    d_map.insert(v, dv);
                    y_map.insert(v, yv);
                    if !self.cone_sites.contains(&v) {
                        dq.push_back(v);
                    }
                }
            }
        }
        let mut where_map = HashMap::with_capacity(y_map.len());
        for (&i, &y) in &y_map {
            let k = self.lat.locate(y).ok_or("developed position fell off the lattice")?;
            if where_map.insert(k, i).is_some() {
                return Err("developed region overlaps itself round an earlier disclination".to_string());
            }
        }
        Ok((d_map, y_map, where_map))
    }

    pub(super) fn local_rotation(&self, u: usize) -> [[f64; 2]; 2] {
        let (mut s, mut c) = (0.0, 0.0);
        let xu = self.x_pos[u];
        for (&v, e) in &self.nb[u] {
            let dv = sub2(self.x_pos[v], xu);
            s += cross2(e.vec, dv);
            c += dot2(e.vec, dv);
        }
        if s.abs() + c.abs() > 1e-12 { rot_mat(s.atan2(c)) } else { [[1.0, 0.0], [0.0, 1.0]] }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn set_bond(
        &mut self,
        u: usize,
        v: usize,
        vec_chart: [f64; 2],
        t_iso: Isometry2,
        kind: usize,
        d_map: &HashMap<usize, Isometry2>,
        allow_cone: Option<usize>,
        ang: Option<f64>,
    ) -> Result<(), String> {
        if (self.cone_sites.contains(&u) && Some(u) != allow_cone)
            || (self.cone_sites.contains(&v) && Some(v) != allow_cone)
        {
            return Err("a later cut would overwrite bonds of an earlier site-centred disclination".to_string());
        }
        let du = d_map[&u];
        let g = d_map[&v].inverse().compose(&t_iso.compose(&du));
        let vec = mat2_vec([[du.m[0][0], du.m[1][0]], [du.m[0][1], du.m[1][1]]], vec_chart);
        let seam = (!same_iso(&g, &Isometry2::IDENTITY, 1e-6)).then_some(g);
        self.nb[u].insert(v, RawBond { vec, kind, seam, ang });
        Ok(())
    }

    pub(super) fn apply_rule<F: Fn(i32) -> Isometry2>(
        &mut self,
        rule: &BTreeMap<(usize, usize), ([f64; 2], i32, usize)>,
        kept: &[usize],
        d_map: &HashMap<usize, Isometry2>,
        t_of: F,
        fresh: &HashSet<usize>,
    ) -> Result<(), String> {
        for (&(u, v), &(_, j, _)) in rule {
            if (self.cone_sites.contains(&u) || self.cone_sites.contains(&v)) && j != 0 {
                return Err("seam crosses a bond incident on an earlier site-centred disclination".to_string());
            }
        }
        let keptset: HashSet<usize> = kept.iter().copied().filter(|u| !self.cone_sites.contains(u)).collect();
        for &u in &keptset {
            let old_nbrs: Vec<usize> = self.nb[u].keys().copied().collect();
            for v in old_nbrs {
                if keptset.contains(&v) && !rule.contains_key(&(u, v)) {
                    self.nb[u].remove(&v);
                    self.nb[v].remove(&u);
                }
            }
        }
        for (&(u, v), &(vec, j, kind)) in rule {
            if !rule.contains_key(&(v, u)) {
                continue;
            }
            if self.cone_sites.contains(&u) || self.cone_sites.contains(&v) {
                if !self.nb[u].contains_key(&v) {
                    return Err("cut runs into a missing bond of an earlier cone site".to_string());
                }
                continue;
            }
            if j == 0 && !fresh.contains(&u) && !fresh.contains(&v) {
                let e = self.nb[u].get(&v).ok_or("cut runs into the core of an earlier defect (missing bond)")?;
                if norm2(sub2(mat2_vec(d_map[&u].m, e.vec), vec)) > 1e-5 {
                    return Err("cut runs into the core of an earlier defect (altered bond)".to_string());
                }
                continue;
            }
            self.set_bond(u, v, vec, t_of(j), kind, d_map, None, None)?;
        }
        Ok(())
    }


    fn prune_rim_spurs(&mut self) -> Result<(), String> {
        // FIX 2: remove ONLY degree-0 isolated sites (keep degree-1 core sites!), then keep main component.
        for i in 0..self.label.len() {
            if self.alive[i] && !self.cone_sites.contains(&i) && self.nb[i].is_empty() {
                self.alive[i] = false;
            }
        }
        let live: Vec<usize> = (0..self.label.len()).filter(|&i| self.alive[i]).collect();
        if live.is_empty() {
            return Err("all crystal sites were removed".to_string());
        }
        let mut comps: Vec<HashSet<usize>> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for &s in &live {
            if !seen.insert(s) {
                continue;
            }
            let mut c = HashSet::from([s]);
            let mut dq = VecDeque::from([s]);
            while let Some(u) = dq.pop_front() {
                for &v in self.nb[u].keys() {
                    if self.alive[v] && seen.insert(v) {
                        c.insert(v);
                        dq.push_back(v);
                    }
                }
            }
            comps.push(c);
        }
        let main_idx = (0..comps.len()).max_by_key(|&i| comps[i].len()).unwrap_or(0);
        for (idx, c) in comps.iter().enumerate() {
            if idx == main_idx {
                continue;
            }
            if c.iter().any(|u| self.cone_sites.contains(u)) {
                return Err("a cut disconnected a disclination core from the crystal".to_string());
            }
            for &i in c {
                let nbrs: Vec<usize> = self.nb[i].keys().copied().collect();
                for v in nbrs {
                    self.nb[v].remove(&i);
                }
                self.nb[i].clear();
                self.alive[i] = false;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<BuiltCrystal, String> {
        self.prune_rim_spurs()?;
        let live: Vec<usize> = (0..self.label.len()).filter(|&i| self.alive[i]).collect();
        let new_id: HashMap<usize, usize> = live.iter().copied().enumerate().map(|(n, i)| (i, n)).collect();
        let n = live.len();
        let pos: Vec<[f64; 2]> = live.iter().map(|&i| self.x_pos[i]).collect();
        let ref_pos: Vec<[f64; 2]> = live.iter().map(|&i| self.ref_pos[i]).collect();
        let basis_idx: Vec<usize> = live.iter().map(|&i| self.label[i].0.2).collect();
        let mut adj = vec![Vec::new(); n];
        let mut edges = Vec::new();
        for (ui, &i) in live.iter().enumerate() {
            for (&v, e) in &self.nb[i] {
                let vi = new_id[&v];
                if ui < vi {
                    edges.push((ui, vi));
                }
                adj[ui].push(DirectedBondData {
                    to: vi,
                    vec: e.vec,
                    kind: e.kind,
                    seam: e.seam,
                    ang: e.ang,
                    across: false,
                    ghost_stub: None,
                });
            }
        }
        edges.sort_unstable();
        let mut cone_sites: Vec<usize> = self.cone_sites.iter().filter_map(|i| new_id.get(i).copied()).collect();
        cone_sites.sort_unstable();
        let mut crystal = BuiltCrystal {
            pos,
            ref_pos,
            basis_idx,
            side: vec![None; n],
            cone_sites,
            edges,
            adj,
            wall_sigma: None,
            is_non_orientable: false,
        };
        mark_long_bonds(&mut crystal);
        Ok(crystal)
    }
}

/// Mark bonds stretched beyond `1.8x` their reference length with half-bond `ghost_stub` endpoints.
pub fn mark_long_bonds(crystal: &mut BuiltCrystal) -> f64 {
    let mut worst = 0.0_f64;
    for u in 0..crystal.adj.len() {
        let pu = crystal.pos[u];
        for e in &mut crystal.adj[u] {
            let pv = crystal.pos[e.to];
            let ell = norm2(e.vec);
            let now = norm2(sub2(pv, pu));
            worst = worst.max((now - ell).abs() / ell.max(1e-12));
            e.ghost_stub = (ell > 1e-12 && now > 1.8 * ell)
                .then(|| [pu[0] + 0.5 * ell * (pv[0] - pu[0]) / now, pu[1] + 0.5 * ell * (pv[1] - pu[1]) / now]);
        }
    }
    worst
}
