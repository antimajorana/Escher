//! Layer-group slabs (C1 WP-L1, plan §5 T5 with Q1, Q2, Q9, Q12): exact-group sites, shell-rule
//! bonds at rank 2, in-plane growth on the Dirichlet complex, and AA (or checked oblique) stacking.
//! Mirrored by `papertilling/_layer_slab.py`.
//!
//! - **Sites.** The union of the general-position orbits stored in `data/layer_sites_80.json` (one,
//!   two or three per group), chosen offline by `tools/build_layer_sites.py` so that spglib's
//!   layer-group check returns exactly G with G's operation count. A group whose entry is not
//!   verified reads "symmetry at least G". Fractional (x, y) in the conventional cell and a
//!   Cartesian z, with a = 1 and the metric of the crystal system ([`metric`]).
//! - **Bonds.** Shells until connected at rank 2 ([`shells`]).
//! - **Growth.** The touching rule on the Dirichlet complex of the single layer
//!   ([`crate::growth::periodic_net`]); the grown patch is then stacked.
//! - **Stacking.** m copies along the normal at spacing d, copy k shifted in-plane by k·s (s
//!   fractional; AA is s = 0). Interlayer bonds are the shortest interlayer shell: one shell already
//!   connects two connected layers, so the shell rule stops there. Refused when d ≤ the slab
//!   thickness (the layers would interpenetrate) or when s breaks the group ([`offset_keeps_group`]).
//!   [PROVED] For m ≥ 2 copies let λ_g = R₂s − e·s for g = (R | t) with in-plane part R₂ and normal
//!   sign e. (1) g maps copy L + k·c (c = (s, d)) to L + k·(e·c + λ_g), so copy 1 lands on a copy
//!   only if λ_g is an in-plane translation (centring included). (2) Then the stack's operations
//!   with linear part R are exactly (R | t + δ_g), δ_g = (m−1)·s for e = −1 (copy k goes to copy
//!   m−1−k) and 0 for e = +1, since L has exactly G. They are G about a shifted origin w iff
//!   w − R₂w ≡ δ_g for every g; with w = (m−1)s/2 + u this is u − R₂u ≡ (m−1)λ_g/2, which u = 0
//!   solves for odd m. For even m it can fail: p11m with s = (1/2, 0) and m = 2 turns the mirror
//!   into a glide. u is searched on the 1/48 grid (integral system, right sides in half
//!   translations) [COMPUTATIONAL EVIDENCE: a 1/240 grid and spglib's layer check of the stacked
//!   sites give the same verdicts on all 80 groups at 7 offsets, m = 2, 3, except 7 cases where the
//!   stack keeps G's type in another setting, e.g. p11a, s = (0, 1/2), m = 2; the refusal says so].

pub mod shells;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use serde::{Deserialize, Serialize};

use super::nets3d::Net3dPatch;
use super::{layer_group_entry, LayerGroupEntry};
use crate::growth::periodic_net::{NetCell, NetMode, PeriodicNet};
use crate::growth::{grow, GrowthCaps, LayerStats, SeedKind, Stopped};
use shells::{shell_bonds, Bond, SHELL_RTOL};

const SITES_JSON: &str = include_str!("../../../../papertilling/data/layer_sites_80.json");
const DEDUPE_TOL: f64 = 1e-6;
const SQ3_2: f64 = 0.866_025_403_784_438_6;

type SharedSlabNet = Arc<Mutex<PeriodicNet>>;
type SlabNetCache = Mutex<HashMap<u8, SharedSlabNet>>;

#[derive(Deserialize)]
struct SitesEntry {
    number: u8,
    orbits: Vec<[f64; 3]>,
    exact: bool,
}

#[derive(Deserialize)]
struct SitesFile {
    groups: Vec<SitesEntry>,
}

static SITES: LazyLock<Vec<SitesEntry>> =
    LazyLock::new(|| serde_json::from_str::<SitesFile>(SITES_JSON).expect("valid layer_sites_80.json").groups);
static SLAB_NETS: LazyLock<SlabNetCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn cached_slab_net(num: u8, lattice: &[[f64; 3]; 2], positions: &[[f64; 3]]) -> Result<SharedSlabNet, String> {
    if let Some(net) = SLAB_NETS.lock().unwrap_or_else(|p| p.into_inner()).get(&num).cloned() {
        return Ok(net);
    }
    let built = Arc::new(Mutex::new(PeriodicNet::new(lattice, positions)?));
    let mut cache = SLAB_NETS.lock().unwrap_or_else(|p| p.into_inner());
    Ok(cache.entry(num).or_insert_with(|| built).clone())
}

/// In-plane lattice vectors (rows, z = 0) of a crystal system, a = 1.
pub fn metric(system: &str) -> [[f64; 3]; 2] {
    match system {
        "oblique" => [[1.0, 0.0, 0.0], [0.22, 1.24, 0.0]],
        "rectangular" => [[1.0, 0.0, 0.0], [0.0, 1.32, 0.0]],
        "square" => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        _ => [[1.0, 0.0, 0.0], [-0.5, SQ3_2, 0.0]],
    }
}

/// Every operation (R, t) modulo the conventional lattice, centring included.
pub fn operations(e: &LayerGroupEntry) -> Vec<([[f64; 3]; 3], [f64; 2])> {
    let shifts: &[[f64; 2]] = if e.centred { &[[0.0, 0.0], [0.5, 0.5]] } else { &[[0.0, 0.0]] };
    let mut out = Vec::new();
    for op in &e.ops {
        let r = [0, 1, 2].map(|a| [0, 1, 2].map(|b| op.r[a][b] as f64));
        let t = [0, 1].map(|a| op.t2[a][0] as f64 / op.t2[a][1] as f64);
        for s in shifts {
            out.push((r, [t[0] + s[0], t[1] + s[1]]));
        }
    }
    out
}

fn same(a: &[f64; 3], b: &[f64; 3]) -> bool {
    let d = [a[0] - b[0], a[1] - b[1]].map(|x| x - x.round());
    d[0].abs() < DEDUPE_TOL && d[1].abs() < DEDUPE_TOL && (a[2] - b[2]).abs() < DEDUPE_TOL
}

/// The orbit of (x, y, z): (x mod 1, y mod 1, z), first occurrence in operation order.
pub fn orbit(e: &LayerGroupEntry, seed: [f64; 3]) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::new();
    for (r, t) in operations(e) {
        let q = [0, 1, 2].map(|a| r[a][0] * seed[0] + r[a][1] * seed[1] + r[a][2] * seed[2]);
        let q = [(q[0] + t[0]).rem_euclid(1.0), (q[1] + t[1]).rem_euclid(1.0), q[2]];
        if !out.iter().any(|o| same(&q, o)) {
            out.push(q);
        }
    }
    out
}

/// One layer of a layer group: sites, bonds and the symmetry readout.
#[derive(Clone, Debug, Serialize)]
pub struct LayerSlab {
    pub number: u8,
    pub symbol: String,
    pub system: String,
    pub lattice: [[f64; 3]; 2],
    /// Fractional x, y and Cartesian z.
    pub frac: Vec<[f64; 3]>,
    pub positions: Vec<[f64; 3]>,
    /// The orbit of each site.
    pub labels: Vec<usize>,
    pub bonds: Vec<Bond>,
    /// Lengths of the bond shells used.
    pub shells: Vec<f64>,
    /// True when the site set is verified to have exactly the group (spglib, offline).
    pub exact: bool,
    pub symmetry_note: String,
    pub thickness: f64,
}

/// Build the layer of group `num` (1..=80).
pub fn build_layer(num: u8) -> Result<LayerSlab, String> {
    let e = layer_group_entry(num)?;
    let data = SITES.iter().find(|s| s.number == num);
    let exact = data.is_some_and(|d| d.exact);
    let seeds = data.map_or_else(|| vec![[0.17, 0.29, 0.18]], |d| d.orbits.clone());
    let lattice = metric(&e.system);
    let (mut frac, mut labels) = (Vec::new(), Vec::new());
    for (k, s) in seeds.iter().enumerate() {
        for q in orbit(e, *s) {
            if frac.iter().any(|f| same(&q, f)) {
                return Err(format!("orbit {k} of layer group #{num} meets an earlier orbit"));
            }
            frac.push(q);
            labels.push(k);
        }
    }
    let positions: Vec<[f64; 3]> = frac
        .iter()
        .map(|f| [0, 1, 2].map(|c| f[0] * lattice[0][c] + f[1] * lattice[1][c] + if c == 2 { f[2] } else { 0.0 }))
        .collect();
    let (bonds, shells) = shell_bonds(&positions, &lattice, &labels)?;
    let zs = positions.iter().map(|p| p[2]);
    let thickness = zs.clone().fold(f64::MIN, f64::max) - zs.fold(f64::MAX, f64::min);
    let symmetry_note = if exact {
        format!("exact layer group {} (spglib layer check) [COMPUTATIONAL EVIDENCE]", e.symbol)
    } else {
        format!("symmetry at least {}", e.symbol)
    };
    Ok(LayerSlab {
        number: num,
        symbol: e.symbol.clone(),
        system: e.system.clone(),
        lattice,
        frac,
        positions,
        labels,
        bonds,
        shells,
        exact,
        symmetry_note,
        thickness,
    })
}

fn find_site(frac: &[[f64; 3]], q: [f64; 3]) -> Option<(usize, [i64; 2])> {
    frac.iter().position(|f| same(f, &q)).map(|i| (i, [(q[0] - frac[i][0]).round() as i64, (q[1] - frac[i][1]).round() as i64]))
}

/// Check that every operation maps the site set and the bond set onto themselves.
pub fn bond_set_invariant(lay: &LayerSlab) -> Result<(), String> {
    let e = layer_group_entry(lay.number)?;
    let set: HashSet<Bond> = lay.bonds.iter().copied().collect();
    for (r, t) in operations(e) {
        let img = |f: [f64; 3]| [0, 1, 2].map(|a| r[a][0] * f[0] + r[a][1] * f[1] + r[a][2] * f[2] + if a < 2 { t[a] } else { 0.0 });
        for &(i, j, n) in &lay.bonds {
            let fi = lay.frac[i];
            let fj = [lay.frac[j][0] + n[0] as f64, lay.frac[j][1] + n[1] as f64, lay.frac[j][2]];
            let (a, na) = find_site(&lay.frac, img(fi)).ok_or("an operation maps a site off the site set")?;
            let (b, nb) = find_site(&lay.frac, img(fj)).ok_or("an operation maps a site off the site set")?;
            let dn = [nb[0] - na[0], nb[1] - na[1]];
            if !(set.contains(&(a, b, dn)) || set.contains(&(b, a, [-dn[0], -dn[1]]))) {
                return Err(format!("an operation maps bond {:?} off the bond set", (i, j, n)));
            }
        }
    }
    Ok(())
}

fn is_translation(v: [f64; 2], centred: bool) -> bool {
    let lat: &[[f64; 2]] = if centred { &[[0.0, 0.0], [0.5, 0.5]] } else { &[[0.0, 0.0]] };
    lat.iter().any(|c| (0..2).all(|a| { let x = v[a] - c[a]; (x - x.round()).abs() < 1e-9 }))
}

/// True when `layers` copies stacked with in-plane offset `s` (fractional) keep every operation of
/// the group about a shifted origin (module docs). One layer has no offset to check.
pub fn offset_keeps_group(num: u8, s: [f64; 2], layers: u32) -> Result<bool, String> {
    if layers <= 1 {
        return Ok(true);
    }
    let e = layer_group_entry(num)?;
    let mut need: Vec<([[f64; 2]; 2], [f64; 2])> = Vec::new();
    for (r, _) in operations(e) {
        let lam = [0, 1].map(|a| r[a][0] * s[0] + r[a][1] * s[1] - r[2][2] * s[a]);
        if !is_translation(lam, e.centred) {
            return Ok(false);
        }
        let h = (layers - 1) as f64 / 2.0;
        need.push(([[r[0][0], r[0][1]], [r[1][0], r[1][1]]], lam.map(|x| h * x)));
    }
    if (layers - 1).is_multiple_of(2) {
        return Ok(true);
    }
    const DEN: u32 = 48;
    Ok((0..DEN * DEN).any(|k| {
        let u = [(k / DEN) as f64 / DEN as f64, (k % DEN) as f64 / DEN as f64];
        need.iter().all(|(r2, h)| {
            let v = [0, 1].map(|a| u[a] - r2[a][0] * u[0] - r2[a][1] * u[1] - h[a]);
            is_translation(v, e.centred)
        })
    }))
}

/// m copies of a layer along the normal: spacing d and in-plane offset s per copy (AA: s = 0).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StackSpec {
    pub layers: u32,
    pub spacing: f64,
    pub offset: [f64; 2],
}

impl Default for StackSpec {
    fn default() -> Self {
        Self { layers: 1, spacing: 1.2, offset: [0.0, 0.0] }
    }
}

/// The stacking vector and the interlayer shell (site i of copy k to site j of copy k+1 in cell n).
#[derive(Clone, Debug, Serialize)]
pub struct Stack {
    pub vector: [f64; 3],
    pub interlayer: Vec<Bond>,
}

/// Stack a layer (module docs give the rule and the refusals).
pub fn stack(lay: &LayerSlab, spec: &StackSpec) -> Result<Stack, String> {
    if spec.layers < 1 {
        return Err("a stack needs at least one layer".into());
    }
    if spec.layers > 1 && spec.spacing <= lay.thickness {
        return Err(format!(
            "[PROVED] spacing {} <= slab thickness {:.4}: layers would interpenetrate",
            spec.spacing, lay.thickness
        ));
    }
    if !offset_keeps_group(lay.number, spec.offset, spec.layers)? {
        return Err(format!(
            "[PROVED] offset {:?} with {} layers breaks {}: some operation of the group is not an operation of the stack (the type may survive in another setting)",
            spec.offset, spec.layers, lay.symbol
        ));
    }
    let l = &lay.lattice;
    let c = [0, 1, 2].map(|k| spec.offset[0] * l[0][k] + spec.offset[1] * l[1][k] + if k == 2 { spec.spacing } else { 0.0 });
    let mut interlayer = Vec::new();
    if spec.layers > 1 {
        let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt(); // a site's copy right above it is at |c|
        let g = shells::inverse_column_norms(l);
        let lim = g.map(|gk| (r * gk).ceil() as i64 + 1);
        let p = &lay.positions;
        let mut cand: Vec<(f64, Bond)> = Vec::new();
        for n0 in -lim[0]..=lim[0] {
            for n1 in -lim[1]..=lim[1] {
                for i in 0..p.len() {
                    for j in 0..p.len() {
                        let v = [0, 1, 2].map(|k| p[j][k] + c[k] + n0 as f64 * l[0][k] + n1 as f64 * l[1][k] - p[i][k]);
                        cand.push(((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(), (i, j, [n0, n1])));
                    }
                }
            }
        }
        let best = cand.iter().map(|x| x.0).fold(f64::MAX, f64::min);
        interlayer = cand.into_iter().filter(|x| x.0 <= best * (1.0 + SHELL_RTOL)).map(|x| x.1).collect();
        interlayer.sort_unstable();
    }
    Ok(Stack { vector: c, interlayer })
}

/// A grown and stacked layer: the drawing, the per-layer readout and its caveats.
#[derive(Clone, Debug, Serialize)]
pub struct SlabGrowth {
    pub patch: Net3dPatch,
    pub mode: NetMode,
    pub stats: Vec<LayerStats>,
    pub stopped: Option<Stopped>,
    /// False for a non-planar layer: its Dirichlet cells are unbounded, χ is not defined (do not show it).
    pub euler_defined: bool,
    pub exact: bool,
    pub symmetry_note: String,
    /// Sites of one copy in the grown patch.
    pub cells: usize,
}

/// Grow layer group `num` in-plane by `layers` touching-rule layers from `seed`, then stack it.
pub fn grow_layer_slab(num: u8, seed: SeedKind, layers: u32, spec: &StackSpec, max_cells: usize) -> Result<SlabGrowth, String> {
    let lay = build_layer(num)?;
    let st = stack(&lay, spec)?;
    let lattice = [lay.lattice[0], lay.lattice[1]];
    let net_arc = cached_slab_net(num, &lattice, &lay.positions)?;
    let (mode, euler_defined, g) = {
        let mut net = net_arc.lock().unwrap_or_else(|p| p.into_inner());
        let s0 = net.seed_cells(seed)?;
        let g = grow(&mut *net, s0, GrowthCaps { max_layers: layers, max_cells });
        (net.mode(), net.euler_defined(), g)
    };
    let cells: Vec<NetCell> = g.layers.iter().flatten().copied().collect();
    let index: HashMap<NetCell, usize> = cells.iter().enumerate().map(|(k, c)| (*c, k)).collect();
    let n = cells.len();
    let at = |c: &NetCell, k: u32| -> [f64; 3] {
        let p = lay.positions[c.0 as usize];
        let (a, b) = (c.1[0] as f64, c.1[1] as f64);
        [0, 1, 2].map(|x| p[x] + a * lattice[0][x] + b * lattice[1][x] + k as f64 * st.vector[x])
    };
    let mut positions = Vec::with_capacity(n * spec.layers as usize);
    for k in 0..spec.layers {
        positions.extend(cells.iter().map(|c| at(c, k)));
    }
    let m = positions.len().max(1) as f64;
    let centre = positions.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / m, a[1] + p[1] / m, a[2] + p[2] / m]);
    positions.iter_mut().for_each(|p| *p = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]]);
    let mut bonds = Vec::new();
    // each bond (i, j, n) is listed once: draw it from every grown cell of site i whose partner grew too
    let link = |list: &[Bond], kind: u8, dk: usize, bonds: &mut Vec<(usize, usize, u8)>| {
        for k in 0..(spec.layers as usize - dk) {
            for (ci, c) in cells.iter().enumerate() {
                for &(_, j, nn) in list.iter().filter(|b| b.0 == c.0 as usize) {
                    if let Some(&cj) = index.get(&(j as u32, [c.1[0] + nn[0], c.1[1] + nn[1], 0])) {
                        bonds.push((k * n + ci, (k + dk) * n + cj, kind));
                    }
                }
            }
        }
    };
    link(&lay.bonds, 0, 0, &mut bonds);
    link(&st.interlayer, 1, 1, &mut bonds);
    let site_kinds: Vec<u8> = (0..spec.layers).flat_map(|_| cells.iter().map(|c| lay.labels[c.0 as usize].min(255) as u8)).collect();
    let last = g.stats.len().saturating_sub(1);
    let info = format!(
        "Layer #{} {}: {} | {} sites, {} bonds | in-plane layer {} ({} cells, {:?} seed) x {} stacked",
        num,
        lay.symbol,
        lay.symmetry_note,
        positions.len(),
        bonds.len(),
        last,
        n,
        seed,
        spec.layers
    );
    let e = layer_group_entry(num)?;
    let patch = Net3dPatch {
        name: format!("Layer #{} {} ({})", e.number, e.symbol, e.pointgroup),
        space_group_num: u16::from(e.number),
        defect_label: format!("System: {}, |P|={} | {}", e.system, e.point_order, lay.symmetry_note),
        positions,
        bonds,
        site_kinds,
        defect_axis: None,
        info,
    };
    Ok(SlabGrowth {
        patch,
        mode,
        euler_defined,
        stats: g.stats,
        stopped: g.stopped,
        exact: lay.exact,
        symmetry_note: lay.symmetry_note,
        cells: n,
    })
}

/// The default drawing of layer group `num`: one layer grown 2 touching-rule layers from a vertex.
pub fn default_slab_patch(num: u8) -> Result<Net3dPatch, String> {
    grow_layer_slab(num, SeedKind::Vertex, 2, &StackSpec::default(), 20_000).map(|g| g.patch)
}
