//! Bond-star snapping with exact antisymmetry, surrounding loops around defective clusters, and the inverse 5-7 /
//! Burgers-circuit detector working from bare coordinates (`#036`, `#037`, `#068`).
//!
//! Snapping: each undirected edge `(min, max)` is snapped once to the nearest vector of the ideal directed bond star
//! (all `+-d` of the cell, A/B labels ignored, rotated by a frame angle `theta`), and `G(j -> i) = -G(i -> j)`, so the
//! closure of a region telescopes exactly onto its boundary. Round a defect loop `theta` is parallel-transported from
//! vertex to vertex ([`transported_closure`]), so a lattice that turns along the loop (disclination dipole) is still
//! snapped in one continuous chart; a loop whose interior carries a net Frank angle gets no Burgers vector (it would
//! depend on the frame).
//!
//! Sign convention. `burgers = sum G` round the counter-clockwise loop is the crate's `closure_vector` (finish minus
//! start of the image circuit in the reference crystal), the convention of `DefectSpec::dislocation(b)`. Hirth & Lothe
//! (1982, ch. 1) define the FS/RH Burgers vector as start minus finish with the circuit right-handed about the line
//! direction; for line direction `+z` (out of the page) that is `burgers_fs_rh = -burgers`. A Volterra field whose
//! displacement jumps by `+b` counter-clockwise (`stroh_field`, Hirth & Lothe's `u_z = b theta / 2 pi`) therefore gives
//! `burgers = -b`, `burgers_fs_rh = b`.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::topo_detect::{DefectCluster, topological_charges};
use super::{BondGraph, PlanarFaces, RefLattice, ang, norm, rot, sub};
use crate::defect::burgers_indices::{BurgersIndexInfo, burgers_index_info};
use crate::types::{Cell2D, Frame2D};

/// One detected defect: a neutral cluster (dislocation), a paired `+-` disclination dipole, or a lone disclination.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectedDefect {
    /// `"dislocation"`, `"dipole"`, `"disclination"`, or `"neutral"` (zero net Frank angle and zero Burgers vector).
    pub kind: String,
    pub clusters: Vec<usize>,
    /// Net Frank angle (degrees) enclosed by the loop.
    pub frank_deg: f64,
    pub centre: [f64; 2],
    /// Counter-clockwise closed walk `[v0, ..., v0]` around the defect.
    pub loop_path: Vec<usize>,
    /// `sum G_e` along `loop_path` (cell Cartesian frame, chart of `loop_path[0]`; frame parallel-transported along the
    /// loop), when the enclosed Frank angle is zero.
    pub burgers: Option<[f64; 2]>,
    /// Hirth-Lothe FS/RH Burgers vector for line direction `+z`: `-burgers`.
    pub burgers_fs_rh: Option<[f64; 2]>,
    pub uv: Option<[i32; 2]>,
    pub index: Option<BurgersIndexInfo>,
    /// Frame angle (degrees) at `loop_path[0]`, where the transported snapping frame starts.
    pub frame_angle_deg: f64,
    pub reason: String,
}

/// Everything the coordinate-only detector reconstructs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordDetection {
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub face_areas: Vec<f64>,
    pub rim: Vec<bool>,
    pub vertex_frank: Vec<f64>,
    pub face_frank: Vec<f64>,
    pub clusters: Vec<DefectCluster>,
    pub defects: Vec<DetectedDefect>,
    pub cutoff: f64,
}

/// Default bond cutoff: midway between the longest ideal bond and the shortest ideal non-bonded distance.
pub fn default_cutoff(cell: &Cell2D, rl: &RefLattice) -> f64 {
    let mut d2 = f64::INFINITY;
    for (i, &bi) in cell.basis.iter().enumerate() {
        for (j, &bj) in cell.basis.iter().enumerate() {
            for n1 in -3..=3 {
                for n2 in -3..=3 {
                    let (f1, f2) = (f64::from(n1), f64::from(n2));
                    let d = [bj[0] + f1 * cell.a1[0] + f2 * cell.a2[0] - bi[0], bj[1] + f1 * cell.a1[1] + f2 * cell.a2[1] - bi[1]];
                    let l = norm(d);
                    if (i != j || n1 != 0 || n2 != 0) && l > rl.lmax + 1e-6 && l < d2 {
                        d2 = l;
                    }
                }
            }
        }
    }
    if d2.is_finite() { 0.5 * (rl.lmax + d2) } else { 1.2 * rl.lmax }
}

fn wrap_pi(x: f64) -> f64 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

/// Index of the star vector nearest to `d` once the star is rotated by `theta`.
pub(crate) fn nearest_star(rl: &RefLattice, d: [f64; 2], theta: f64) -> usize {
    (0..rl.star.len()).min_by(|&a, &b| norm(sub(rot(rl.star[a], theta), d)).total_cmp(&norm(sub(rot(rl.star[b], theta), d)))).unwrap_or(0)
}

/// Frame angle fitted on `edges`: start at `theta0`, then twice shift by the mean angular residual to the nearest
/// rotated star vector.
pub fn fit_frame_angle(g: &BondGraph, rl: &RefLattice, edges: &[(usize, usize)], theta0: f64) -> f64 {
    let mut th = theta0;
    for _ in 0..3 {
        if edges.is_empty() {
            break;
        }
        let s: f64 = edges.iter().map(|&(u, v)| {
            let d = sub(g.pos[v], g.pos[u]);
            wrap_pi(ang(d) - ang(rot(rl.star[nearest_star(rl, d, th)], th)))
        }).sum();
        th += s / edges.len() as f64;
    }
    th
}

/// Snapped ideal vectors `G(u -> v)` for `u < v` (cell frame), one snap per undirected edge.
pub fn snap_edges(g: &BondGraph, rl: &RefLattice, theta: f64) -> HashMap<(usize, usize), [f64; 2]> {
    g.edges.iter().map(|&(u, v)| ((u, v), rl.star[nearest_star(rl, sub(g.pos[v], g.pos[u]), theta)])).collect()
}

/// Directed lookup with exact antisymmetry `G(v -> u) = -G(u -> v)`.
pub fn g_dir(snap: &HashMap<(usize, usize), [f64; 2]>, u: usize, v: usize) -> [f64; 2] {
    if u < v {
        snap[&(u, v)]
    } else {
        let w = snap[&(v, u)];
        [-w[0], -w[1]]
    }
}

/// Angular gap (radians) between the closest two directions of the bond star.
fn star_gap(rl: &RefLattice) -> f64 {
    let mut a: Vec<f64> = rl.star.iter().map(|&v| ang(v)).collect();
    a.sort_by(f64::total_cmp);
    (0..a.len()).map(|k| if k + 1 < a.len() { a[k + 1] - a[k] } else { a[0] + 2.0 * PI - a[k] })
        .filter(|&x| x > 1e-9).fold(2.0 * PI, f64::min)
}

/// Local frame angle at `v`: `th` plus the mean residual of `v`'s bonds to their nearest star vectors rotated by `th`.
fn local_frame(g: &BondGraph, rl: &RefLattice, v: usize, th: f64) -> f64 {
    let n = g.nbrs[v].len();
    if n == 0 {
        return th;
    }
    th + g.nbrs[v].iter().map(|&w| {
        let d = sub(g.pos[w], g.pos[v]);
        wrap_pi(ang(d) - ang(rot(rl.star[nearest_star(rl, d, th)], th)))
    }).sum::<f64>() / n as f64
}

/// Burgers closure round a closed walk `[v0, ..., v0]` with a parallel-transported frame. A single frame angle fails
/// round a disclination dipole, whose lattice orientation turns by up to half a star sector along the loop. Here the walk
/// starts at the vertex whose local frame is closest to `rl.frame_angle`, the frame is carried from vertex to vertex
/// (each step re-fitted on that vertex's bonds), and every undirected edge is snapped once in the frame of its first
/// traversal (`G(v -> u) = -G(u -> v)`). The closure `sum G` is in the chart of the new start vertex: for zero enclosed
/// Frank angle it is the crate's `closure_vector` of the rotated walk. Returns `(walk, closure, start frame angle)`.
pub fn transported_closure(g: &BondGraph, rl: &RefLattice, walk: &[usize]) -> Result<(Vec<usize>, [f64; 2], f64), String> {
    let n = walk.len().saturating_sub(1);
    if n < 3 || walk[0] != walk[n] {
        return Err("REFUSED: the loop is not a closed walk".to_string());
    }
    // a snap is ambiguous only at half the star gap; refuse well before (0.45 gap = 27 deg honeycomb, 40.5 deg square)
    let tol = 0.45 * star_gap(rl);
    let k0 = (0..n).min_by(|&a, &b| {
        let da = wrap_pi(local_frame(g, rl, walk[a], rl.frame_angle) - rl.frame_angle).abs();
        let db = wrap_pi(local_frame(g, rl, walk[b], rl.frame_angle) - rl.frame_angle).abs();
        da.total_cmp(&db)
    }).unwrap_or(0);
    let w: Vec<usize> = (0..=n).map(|k| walk[(k0 + k) % n]).collect();
    let th0 = local_frame(g, rl, w[0], rl.frame_angle);
    let (mut th, mut b) = (th0, [0.0, 0.0]);
    let mut snap: HashMap<(usize, usize), [f64; 2]> = HashMap::new();
    for k in 0..n {
        if k > 0 {
            th = local_frame(g, rl, w[k], th);
        }
        let (u, v) = (w[k], w[k + 1]);
        let key = (u.min(v), u.max(v));
        if let std::collections::hash_map::Entry::Vacant(slot) = snap.entry(key) {
            // the edge's frame: midway between the frames of its two ends
            let te = 0.5 * (th + local_frame(g, rl, v, th));
            let dv = sub(g.pos[key.1], g.pos[key.0]);
            let s = rl.star[nearest_star(rl, dv, te)];
            let res = wrap_pi(ang(dv) - ang(rot(s, te))).abs();
            if res > tol {
                return Err(format!(
                    "REFUSED: bond ({u}, {v}) is {:.1} deg from every star direction in the transported frame (limit {:.1})",
                    res.to_degrees(), tol.to_degrees()));
            }
            slot.insert(s);
        }
        let e = g_dir(&snap, u, v);
        b = [b[0] + e[0], b[1] + e[1]];
    }
    let back = wrap_pi(local_frame(g, rl, w[0], th) - th0).abs();
    if back > tol {
        return Err(format!("REFUSED: the transported frame turned by {:.1} deg round the loop (it must close)", back.to_degrees()));
    }
    Ok((w, b, th0))
}

pub(crate) fn vertex_faces(pf: &PlanarFaces, nv: usize) -> Vec<Vec<usize>> {
    let mut v2f = vec![Vec::new(); nv];
    for (fi, f) in pf.faces.iter().enumerate() {
        for &v in f {
            v2f[v].push(fi);
        }
    }
    v2f
}

fn touches_rim(pf: &PlanarFaces, fi: usize) -> bool {
    pf.on_boundary(fi) || pf.faces[fi].iter().any(|&v| pf.rim[v])
}

/// Grow `seed` by `clearance` vertex-sharing rings (never into rim faces), then fill holes.
pub(crate) fn grow_region(pf: &PlanarFaces, v2f: &[Vec<usize>], seed: &BTreeSet<usize>, clearance: usize) -> BTreeSet<usize> {
    let mut reg = seed.clone();
    for _ in 0..clearance {
        let add: Vec<usize> = reg.iter().flat_map(|&fi| pf.faces[fi].iter()).flat_map(|&v| v2f[v].iter().copied())
            .filter(|&f| !reg.contains(&f) && !touches_rim(pf, f)).collect();
        reg.extend(add);
    }
    let mut outside = vec![false; pf.faces.len()];
    let mut dq: VecDeque<usize> = (0..pf.faces.len()).filter(|&f| !reg.contains(&f) && touches_rim(pf, f)).collect();
    for &f in &dq {
        outside[f] = true;
    }
    while let Some(f) = dq.pop_front() {
        for nb in pf.face_neighbours(f) {
            if !outside[nb] && !reg.contains(&nb) {
                outside[nb] = true;
                dq.push_back(nb);
            }
        }
    }
    reg.extend((0..pf.faces.len()).filter(|&f| !outside[f]));
    reg
}

/// Oriented boundaries of the face region. Each dart keeps its selected face on the left,
/// so holes have the opposite orientation from the outside boundary.
pub(crate) fn region_boundaries(pf: &PlanarFaces, reg: &BTreeSet<usize>) -> Vec<Vec<usize>> {
    let mut next = HashMap::new();
    let mut boundary = BTreeSet::new();
    for &fi in reg {
        let face = &pf.faces[fi];
        for i in 0..face.len() {
            let (u, v, w) = (face[i], face[(i + 1) % face.len()], face[(i + 2) % face.len()]);
            next.insert((u, v), w);
            if !pf.left_face.get(&(v, u)).is_some_and(|f| reg.contains(f)) {
                boundary.insert((u, v));
            }
        }
    }
    let mut walks = Vec::new();
    while let Some(&(mut u, mut v)) = boundary.first() {
        let mut walk = vec![u];
        while boundary.remove(&(u, v)) {
            walk.push(v);
            // Follow the selected faces around v. Two components touching only at v
            // have different incident sectors, hence different boundary circuits.
            let mut w = next[&(u, v)];
            while let Some(&across) = next.get(&(w, v)) {
                w = across;
            }
            (u, v) = (v, w);
        }
        walks.push(walk);
    }
    walks
}

/// The closed circuit of a region with one boundary component. Empty for every other case.
pub fn region_boundary_walk(pf: &PlanarFaces, reg: &BTreeSet<usize>) -> Vec<usize> {
    let mut walks = region_boundaries(pf, reg);
    if walks.len() == 1 { walks.pop().unwrap() } else { Vec::new() }
}

/// Net Frank angle inside a region: its faces plus the vertices strictly inside it.
pub(crate) fn region_frank(pf: &PlanarFaces, reg: &BTreeSet<usize>, walks: &[Vec<usize>], vq: &[f64], fq: &[f64]) -> f64 {
    let on_loop: BTreeSet<usize> = walks.iter().flatten().copied().collect();
    let inner: BTreeSet<usize> = reg.iter().flat_map(|&f| pf.faces[f].iter().copied()).filter(|v| !on_loop.contains(v)).collect();
    let s = reg.iter().map(|&f| fq[f]).sum::<f64>() + inner.iter().map(|&v| vq[v]).sum::<f64>();
    (s * 1e6).round() / 1e6
}

fn dual_path(pf: &PlanarFaces, from: &BTreeSet<usize>, to: &BTreeSet<usize>) -> Vec<usize> {
    let mut prev: HashMap<usize, usize> = HashMap::new();
    let mut dq: VecDeque<usize> = from.iter().copied().collect();
    let mut seen: BTreeSet<usize> = from.clone();
    while let Some(f) = dq.pop_front() {
        if to.contains(&f) {
            let mut path = vec![f];
            let mut c = f;
            while let Some(&p) = prev.get(&c) {
                path.push(p);
                c = p;
            }
            return path;
        }
        for nb in pf.face_neighbours(f) {
            if seen.insert(nb) {
                prev.insert(nb, f);
                dq.push_back(nb);
            }
        }
    }
    Vec::new()
}

/// Reconstruct the bond graph from coordinates, measure Frank charges, cluster, pair `+-` disclinations, and run a
/// counter-clockwise surrounding loop around each neutral cluster / dipole to read its Burgers vector.
/// `edges = None` uses a distance `cutoff` (default [`default_cutoff`]). `clearance` = face rings added round a seed.
pub fn detect_defects_from_coords(
    pos: &[[f64; 2]], group: &str, cell: &Cell2D, frame: Frame2D,
    edges: Option<&[(usize, usize)]>, cutoff: Option<f64>, clearance: usize,
) -> Result<CoordDetection, String> {
    if cell.basis.is_empty() || cell.bonds.is_empty() {
        return Err("REFUSED: reference cell has no sites or no bonds".to_string());
    }
    let rl = RefLattice::new(cell, frame);
    let cut = cutoff.unwrap_or_else(|| default_cutoff(cell, &rl));
    let g = match edges {
        Some(e) => BondGraph::from_edges(pos, e),
        None => BondGraph::from_cutoff(pos, cut),
    };
    let pf = PlanarFaces::new(&g);
    let tc = topological_charges(&g, &pf, &rl);
    let v2f = vertex_faces(&pf, pos.len());
    let seeds: Vec<BTreeSet<usize>> = tc.clusters.iter()
        .map(|c| c.faces.iter().copied().chain(c.vertices.iter().flat_map(|&v| v2f[v].iter().copied())).collect()).collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let charged: Vec<usize> = (0..tc.clusters.len()).filter(|&i| tc.clusters[i].frank_deg != 0.0 && !tc.clusters[i].touches_rim).collect();
    let mut cand: Vec<(f64, usize, usize)> = Vec::new();
    for (a, &i) in charged.iter().enumerate() {
        for &j in &charged[a + 1..] {
            if (tc.clusters[i].frank_deg + tc.clusters[j].frank_deg).abs() < 1e-6 {
                cand.push((norm(sub(tc.clusters[i].centre, tc.clusters[j].centre)), i, j));
            }
        }
    }
    cand.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut used = vec![false; tc.clusters.len()];
    for &(_, i, j) in &cand {
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            groups.push(vec![i, j]);
        }
    }
    for (i, c) in tc.clusters.iter().enumerate() {
        if !used[i] && !c.touches_rim {
            groups.push(vec![i]);
        }
    }
    groups.sort();
    let mut defects = Vec::new();
    for grp in groups {
        let mut seed: BTreeSet<usize> = grp.iter().flat_map(|&i| seeds[i].iter().copied()).collect();
        if grp.len() == 2 {
            seed.extend(dual_path(&pf, &seeds[grp[0]], &seeds[grp[1]]));
        }
        defects.push(loop_defect(&g, &pf, &rl, &tc, &v2f, &seed, &grp, clearance, group, cell));
    }
    Ok(CoordDetection {
        edges: g.edges.clone(), faces: pf.faces.clone(), face_areas: pf.areas.clone(), rim: pf.rim.clone(),
        vertex_frank: tc.vertex_frank, face_frank: tc.face_frank, clusters: tc.clusters, defects, cutoff: cut,
    })
}

#[allow(clippy::too_many_arguments)]
fn loop_defect(
    g: &BondGraph, pf: &PlanarFaces, rl: &RefLattice, tc: &super::TopoCharges, v2f: &[Vec<usize>],
    seed: &BTreeSet<usize>, grp: &[usize], clearance: usize, group: &str, cell: &Cell2D,
) -> DetectedDefect {
    let foreign: BTreeSet<usize> = (0..tc.clusters.len()).filter(|i| !grp.contains(i))
        .flat_map(|i| tc.clusters[i].faces.iter().copied().chain(tc.clusters[i].vertices.iter().flat_map(|&v| v2f[v].iter().copied()))).collect();
    let mut chosen = None;
    for clr in (0..=clearance).rev() {
        let reg = grow_region(pf, v2f, seed, clr);
        let clean = reg.is_disjoint(&foreign);
        if clean || clr == 0 {
            chosen = Some((reg, clean));
            break;
        }
    }
    let (reg, clean) = chosen.unwrap_or_default();
    let mut walks = region_boundaries(pf, &reg);
    let frank = region_frank(pf, &reg, &walks, &tc.vertex_frank, &tc.face_frank);
    let multiple = walks.len() > 1;
    let walk = if walks.len() == 1 { walks.pop().unwrap() } else { Vec::new() };
    let pts: Vec<[f64; 2]> = grp.iter().map(|&i| tc.clusters[i].centre).collect();
    let centre = [pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64, pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64];
    let mut d = DetectedDefect {
        kind: "disclination".to_string(), clusters: grp.to_vec(), frank_deg: frank, centre, loop_path: walk.clone(),
        burgers: None, burgers_fs_rh: None, uv: None, index: None, frame_angle_deg: rl.frame_angle.to_degrees(),
        reason: String::new(),
    };
    if multiple {
        d.reason = "REFUSED: the region has several boundary circuits; a single transported Burgers circuit is undefined".into();
        return d;
    }
    if walk.len() < 4 {
        d.reason = "REFUSED: no closed loop could be formed round the cluster (it touches the rim)".to_string();
        return d;
    }
    if frank != 0.0 {
        d.reason = "[PROVED] net Frank angle != 0 inside the loop: its closure depends on the frame; no Burgers vector".to_string();
        return d;
    }
    let (walk, mut b, th) = match transported_closure(g, rl, &walk) {
        Ok(x) => x,
        Err(why) => {
            d.reason = why;
            return d;
        }
    };
    d.loop_path = walk;
    for c in &mut b {
        if c.abs() < 1e-11 {
            *c = 0.0;
        }
    }
    d.frame_angle_deg = th.to_degrees();
    d.burgers = Some(b);
    d.burgers_fs_rh = Some([-b[0] + 0.0, -b[1] + 0.0]);
    let fr = rl.frac(b);
    let uv = [fr[0].round() as i32, fr[1].round() as i32];
    if (fr[0] - f64::from(uv[0])).abs() < 1e-6 && (fr[1] - f64::from(uv[1])).abs() < 1e-6 {
        d.uv = Some(uv);
        d.index = Some(burgers_index_info(group, cell, uv));
    }
    let zero = norm(b) < 1e-9;
    d.kind = match (grp.len(), zero) {
        (_, true) => "neutral",
        (2, false) => "dipole",
        _ => "dislocation",
    }.to_string();
    d.reason = if clean { "[PROVED] exact closure of snapped bonds round a loop enclosing zero net Frank angle".to_string() }
        else { "[PROVED] loop also encloses other defective clusters (zero net Frank angle): the closure is their combined Burgers vector".to_string() };
    d
}
