//! Dislocation Volterra + completion core gallery, chord embedding, configuration core crops,
//! and single-defect equivalence (`cores.py` + `config.equivalent_single`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use crate::defect::charge::{frame_faces, infer_patch_disk};
use crate::defect::combine_types::{CoreGalleryEntry, DislocationScanInfo, EquivalentSingleReport};
use crate::defect::cone_geometry::{add2, mat2_vec, norm2, sub2};
use crate::defect::core_graph::{
    LatticeCoreInfo, TWO_PI, core_graph_with_rim, isomorphic_cores, layer_signature, patch_rim,
};
use crate::defect::equivalence::inspect_crystal_cores;
use crate::defect::half_wall::resolve_half_wall_spec;
use crate::defect::single_cache::{SingleDefectCache, equivalent_single_cached};
use crate::defect::surgery::{DefectSpec, patch_centre_and_radius};
use crate::defect::volterra_patch::{
    enumerate_completions, first_pos, split_ring, wl_hash_labelled,
};
use crate::defect::volterra_scan::{
    ScanBundle, ScannedCoreBuild, crystal_with_dummy_chords, scan_dislocation_cores,
};
use crate::plane::draw::{ensure_embedded, relax_picture};
use crate::types::{BuiltCrystal, Cell2D, DirectedBondData, Frame2D, Isometry2};

const TURNS_DEG: [f64; 5] = [0.0, 15.0, -15.0, 30.0, -30.0];
const WORDS: [&str; 6] = ["one", "two", "three", "four", "five", "six"];
const RING_NAMES: [&str; 10] = [
    "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
    "hendecagon", "dodecagon",
];

fn count_word(c: usize) -> String {
    WORDS.get(c.wrapping_sub(1)).map_or_else(|| c.to_string(), |&w| w.to_string())
}

fn ring_word(n: usize, count: usize) -> String {
    let name = RING_NAMES.get(n.wrapping_sub(3)).map_or_else(|| format!("{n}-ring"), |&w| w.to_string());
    format!("{} {name}{}", count_word(count), if count > 1 { "s" } else { "" })
}

/// Human-readable description of core ring sizes and under-coordinated sites (`cores.describe`).
pub fn describe_core(rings: &[usize], perfect_rings: &[usize], under: &[usize]) -> String {
    let perfect: BTreeSet<usize> = perfect_rings.iter().copied().collect();
    let (mut odd, mut rest) = (BTreeMap::<usize, usize>::new(), BTreeMap::<usize, usize>::new());
    for &r in rings {
        *if perfect.contains(&r) { &mut rest } else { &mut odd }.entry(r).or_default() += 1;
    }
    let fmt_map = |m: &BTreeMap<usize, usize>| {
        m.iter().map(|(&n, &c)| ring_word(n, c)).collect::<Vec<_>>().join(" and ")
    };
    let odd_elems: Vec<usize> = odd.iter().flat_map(|(&n, &c)| std::iter::repeat_n(n, c)).collect();
    let mut head = if odd_elems == [5, 7] {
        "5-7 pair".to_string()
    } else if !odd.is_empty() {
        fmt_map(&odd)
    } else {
        format!("{} (ring sizes of the perfect lattice)", fmt_map(&rest))
    };
    if !rest.is_empty() && !odd.is_empty() {
        head.push_str(&format!(", plus {}", fmt_map(&rest)));
    }
    if !under.is_empty() {
        let k = under.len();
        head.push_str(&format!("; {} under-coordinated site{} on the ring", count_word(k), if k > 1 { "s" } else { "" }));
    } else {
        head.push_str("; every ring site fully bonded");
    }
    head
}

fn graph_ball_dist(adj: &[Vec<usize>], sources: &BTreeSet<usize>, max_dist: usize) -> BTreeMap<usize, usize> {
    let mut dist: BTreeMap<usize, usize> = sources.iter().map(|&s| (s, 0)).collect();
    let mut dq: VecDeque<usize> = sources.iter().copied().collect();
    while let Some(u) = dq.pop_front() {
        let du = dist[&u];
        if du >= max_dist {
            continue;
        }
        for &w in &adj[u] {
            if let std::collections::btree_map::Entry::Vacant(e) = dist.entry(w) {
                e.insert(du + 1);
                dq.push_back(w);
            }
        }
    }
    dist
}

fn crop_around(
    x: &BuiltCrystal, adj_ball: &[Vec<usize>], deg_base: &[usize], old_rim: &HashSet<usize>,
    sources: &BTreeSet<usize>, d_cut: usize,
) -> (BuiltCrystal, HashMap<usize, usize>, HashSet<usize>) {
    let dist = graph_ball_dist(adj_ball, sources, d_cut);
    let keep: Vec<usize> = dist.keys().copied().collect();
    let new_idx: HashMap<usize, usize> = keep.iter().copied().enumerate().map(|(i, o)| (o, i)).collect();
    let (mut adj, mut edges) = (vec![Vec::new(); keep.len()], Vec::new());
    for (i, &o) in keep.iter().enumerate() {
        for e in &x.adj[o] {
            if let Some(&ni) = new_idx.get(&e.to) {
                if i < ni { edges.push((i, ni)); }
                adj[i].push(DirectedBondData { to: ni, ghost_stub: None, ..*e });
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let y = BuiltCrystal {
        pos: keep.iter().map(|&o| x.pos[o]).collect(), ref_pos: keep.iter().map(|&o| x.ref_pos[o]).collect(),
        basis_idx: keep.iter().map(|&o| x.basis_idx[o]).collect(),
        side: keep.iter().map(|&o| x.side.get(o).copied().unwrap_or(None)).collect(),
        cone_sites: x.cone_sites.iter().filter_map(|o| new_idx.get(o).copied()).collect(),
        edges, adj, wall_sigma: x.wall_sigma, is_non_orientable: x.is_non_orientable,
    };
    let crop_rim = dist.iter()
        .filter(|&(&o, &d)| d >= d_cut - 1 || old_rim.contains(&o) || y.adj[new_idx[&o]].len() < deg_base[o])
        .map(|(&o, _)| new_idx[&o]).collect();
    (y, new_idx, crop_rim)
}

#[inline]
fn corner_angle(e: &DirectedBondData) -> f64 {
    e.ang.unwrap_or_else(|| e.vec[1].atan2(e.vec[0]).rem_euclid(TWO_PI))
}

fn arc_vec(y: &BuiltCrystal, ring: &[usize], pos: &HashMap<usize, usize>, u: usize, v: usize) -> [f64; 2] {
    let (n, mut k, mut s) = (ring.len(), pos[&u], [0.0, 0.0]);
    while ring[k % n] != v || k == pos[&u] {
        if let Some(e) = y.adj[ring[k % n]].iter().find(|e| e.to == ring[(k + 1) % n]) { s = add2(s, e.vec); }
        k += 1;
        if k - pos[&u] > n { break; }
    }
    s
}

fn gaps(angs: &[f64]) -> Vec<f64> {
    if angs.is_empty() { return vec![0.0]; }
    let mut a: Vec<f64> = angs.iter().map(|x| x.rem_euclid(TWO_PI)).collect();
    a.sort_by(|x, y| x.total_cmp(y));
    (0..a.len()).map(|k| {
        let diff = (a[(k + 1) % a.len()] - a[k]).rem_euclid(TWO_PI);
        (a[k] + 0.5 * if diff == 0.0 { TWO_PI } else { diff }).rem_euclid(TWO_PI)
    }).collect()
}

type ChordEndOption = ([f64; 2], Option<f64>);

fn chord_options(y: &BuiltCrystal, ring: &[usize], u: usize, v: usize, lat_info: &LatticeCoreInfo) -> Vec<(ChordEndOption, ChordEndOption)> {
    let pos = first_pos(ring);
    let arc = arc_vec(y, ring, &pos, u, v);
    let mut tot = [0.0, 0.0];
    for k in 0..ring.len() {
        if let Some(e) = y.adj[ring[k]].iter().find(|e| e.to == ring[(k + 1) % ring.len()]) { tot = add2(tot, e.vec); }
    }
    let at_u: Vec<f64> = y.adj[u].iter().map(corner_angle).collect();
    let at_v: Vec<f64> = y.adj[v].iter().map(corner_angle).collect();
    let tie = |t: f64, angs: &[f64]| angs.iter().any(|&x| ((t - x) - ((t - x) / TWO_PI).round_ties_even() * TWO_PI).abs() < 1e-3);
    let (mut out, mut first) = (Vec::new(), None);
    for c in [arc, sub2(arc, tot)] {
        let len = norm2(c);
        if len < 1e-9 { continue; }
        let l2 = if len <= 1.05 * lat_info.lmax { len } else { lat_info.lmean };
        for turn in TURNS_DEG {
            let t = c[1].atan2(c[0]) + turn.to_radians();
            let vec = [l2 * t.cos(), l2 * t.sin()];
            first.get_or_insert(vec);
            if !(tie(t, &at_u) || tie(t + PI, &at_v)) { out.push(((vec, None), ([-vec[0], -vec[1]], None))); }
        }
    }
    let l2 = norm2(first.unwrap_or(arc));
    for gu in gaps(&at_u) {
        let vec = [l2 * gu.cos(), l2 * gu.sin()];
        for gv in gaps(&at_v) { out.push(((vec, Some(gu)), ([-vec[0], -vec[1]], Some(gv)))); }
    }
    out
}

fn face_from(y: &BuiltCrystal, a0: usize, b0: usize) -> Vec<usize> {
    let mut order: HashMap<usize, Vec<usize>> = HashMap::new();
    let limit = 4 * y.pos.len().max(1);
    let (mut f, mut a, mut b) = (Vec::new(), a0, b0);
    loop {
        f.push(a);
        let lst = order.entry(b).or_insert_with(|| {
            let mut pairs: Vec<(f64, usize)> = y.adj[b].iter().map(|e| (corner_angle(e), e.to)).collect();
            pairs.sort_by(|x, z| x.0.total_cmp(&z.0).then_with(|| x.1.cmp(&z.1)));
            pairs.into_iter().map(|(_, x)| x).collect()
        });
        let idx = lst.iter().position(|&x| x == a).unwrap_or(0);
        a = b;
        b = lst[if idx == 0 { lst.len() - 1 } else { idx - 1 }];
        if (a, b) == (a0, b0) || f.len() > limit {
            return f;
        }
    }
}

fn sorted_vec(v: &[usize]) -> Vec<usize> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s
}

fn face_keys(y: &BuiltCrystal) -> BTreeMap<Vec<usize>, isize> {
    let mut counts = BTreeMap::new();
    for f in frame_faces(y) {
        *counts.entry(sorted_vec(&f)).or_default() += 1;
    }
    counts
}

fn set_chord_edge(y: &mut BuiltCrystal, u: usize, v: usize, fw: ChordEndOption, bw: ChordEndOption) {
    for (s, t, opt) in [(u, v, fw), (v, u, bw)] {
        if let Some(e) = y.adj[s].iter_mut().find(|e| e.to == t) {
            e.vec = opt.0;
            e.ang = opt.1;
        } else {
            y.adj[s].push(DirectedBondData {
                to: t, vec: opt.0, kind: 0, seam: None, ang: opt.1, across: false, ghost_stub: None,
            });
        }
    }
}

pub(crate) fn add_chords(
    y: &mut BuiltCrystal,
    ring: &[usize],
    chords: &[(usize, usize)],
    lat_info: &LatticeCoreInfo,
) -> (bool, Vec<Vec<usize>>) {
    let pos = first_pos(ring);
    let sub_ord: Vec<(usize, usize)> =
        chords.iter().map(|&(a, b)| if pos[&a] <= pos[&b] { (a, b) } else { (b, a) }).collect();
    let base = face_keys(y);
    let mut ok = true;

    for (i, &(u, v)) in sub_ord.iter().enumerate() {
        let split_i = split_ring(ring, &sub_ord[..=i]);
        let mut want_two: Vec<Vec<usize>> = split_i
            .iter()
            .filter(|r| (0..r.len()).any(|k| matches!((r[k], r[(k + 1) % r.len()]), (a, b) if (a == u && b == v) || (a == v && b == u))))
            .map(|r| sorted_vec(r))
            .collect();
        want_two.sort();

        let opts = chord_options(y, ring, u, v, lat_info);
        let mut matched = false;
        for &(fw, bw) in &opts {
            set_chord_edge(y, u, v, fw, bw);
            let mut got = vec![sorted_vec(&face_from(y, u, v)), sorted_vec(&face_from(y, v, u))];
            got.sort();
            if got == want_two {
                matched = true;
                break;
            }
        }
        if !matched {
            ok = false;
            if let Some(&(fw, bw)) = opts.first() {
                set_chord_edge(y, u, v, fw, bw);
            }
        }
        y.edges.push((u.min(v), u.max(v)));
    }
    y.edges.sort_unstable();
    y.edges.dedup();

    let split = split_ring(ring, &sub_ord);
    let mut want = base;
    *want.entry(sorted_vec(ring)).or_default() -= 1;
    for r in &split {
        *want.entry(sorted_vec(r)).or_default() += 1;
    }
    want.retain(|_, &mut c| c > 0);
    (ok && face_keys(y) == want, split)
}

fn find_both_routes(
    out: &mut [CoreGalleryEntry],
    rims: &[HashSet<usize>],
    cores: &[ScannedCoreBuild],
    lat_info: &LatticeCoreInfo,
    radius: usize,
    max_candidates: usize,
) {
    let vol_idxs: Vec<usize> =
        out.iter().enumerate().filter(|(_, e)| e.kind == "Volterra").map(|(i, _)| i).collect();
    for &ei in &vol_idxs {
        if out[ei].faces.len() != 1 {
            continue;
        }
        let (ring, y, src_name) = (out[ei].faces[0].clone(), out[ei].crystal.clone(), out[ei].name.clone());
        let mut under = BTreeMap::new();
        for &v in &ring {
            let pd = lat_info.perfect_degree[y.basis_idx[v]];
            if y.adj[v].len() < pd {
                under.insert(v, pd - y.adj[v].len());
            }
        }
        let edge_set: HashSet<(usize, usize)> = y.edges.iter().copied().collect();
        let (subs, _) = enumerate_completions(|u, v| edge_set.contains(&(u.min(v), u.max(v))), &ring, &under, max_candidates);
        for sub in subs {
            let y2 = crystal_with_dummy_chords(&y, &sub);
            let Ok(h_graph) = core_graph_with_rim(&y2, &rims[ei], lat_info, radius) else { continue };
            let h_wl = wl_hash_labelled(&h_graph);
            for &fi in &vol_idxs {
                if cores[fi].wl_hash == h_wl && isomorphic_cores(&h_graph, &cores[fi].graph) {
                    let note = format!("a completion of {src_name} ({} bond(s) added)", sub.len());
                    if !out[fi].also.contains(&note) {
                        out[fi].also.push(note);
                    }
                }
            }
        }
    }
}

type BuildGraphCache = (Vec<Vec<usize>>, Vec<usize>, HashSet<usize>);

pub(crate) fn gallery_from_bundle(
    bundle: &ScanBundle,
    radius: usize,
    both_routes: bool,
    max_candidates: usize,
    draw: bool,
) -> Vec<CoreGalleryEntry> {
    let lat_info = &bundle.lat_info;
    let d_cut = 2 * lat_info.r_det + radius + 3;
    let mut memo: HashMap<usize, BuildGraphCache> = HashMap::new();
    let (mut out, mut rims) = (Vec::with_capacity(bundle.cores.len()), Vec::with_capacity(bundle.cores.len()));

    for c in &bundle.cores {
        let (ref x, ref rings_walks) = bundle.builds[c.build_idx];
        let core_pt = c.info.core_point.unwrap_or([0.0, 0.0]);
        let (adj_base, deg_base, old_rim) = memo.entry(c.build_idx).or_insert_with(|| {
            let adj_b: Vec<Vec<usize>> = x.adj.iter().map(|es| es.iter().map(|e| e.to).collect()).collect();
            let deg_b: Vec<usize> = adj_b.iter().map(Vec::len).collect();
            (adj_b, deg_b, patch_rim(x, lat_info, core_pt, c.build_r))
        });

        let ring_sites: BTreeSet<usize> = rings_walks.iter().flatten().copied().collect();
        let (mut y, new_idx, crop_rim) = if c.chords.is_empty() {
            crop_around(x, adj_base, deg_base, old_rim, &ring_sites, d_cut)
        } else {
            let mut adj_ch = adj_base.clone();
            for &(u, v) in &c.chords {
                adj_ch[u].push(v);
                adj_ch[v].push(u);
            }
            crop_around(x, &adj_ch, deg_base, old_rim, &ring_sites, d_cut)
        };

        let walks: Vec<Vec<usize>> = rings_walks.iter().map(|f| f.iter().map(|v| new_idx[v]).collect()).collect();
        let chords: Vec<(usize, usize)> = c.chords.iter().map(|&(u, v)| (new_idx[&u], new_idx[&v])).collect();
        let (faces_ok, faces) = if !chords.is_empty() && !walks.is_empty() {
            add_chords(&mut y, &walks[0], &chords, lat_info)
        } else {
            (true, walks.clone())
        };
        let iso_ok = core_graph_with_rim(&y, &crop_rim, lat_info, radius)
            .ok()
            .map(|h| wl_hash_labelled(&h) == c.wl_hash && isomorphic_cores(&h, &c.graph));
        let core_sites: Vec<usize> = walks.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect();
        let under_sites: Vec<usize> = (0..y.pos.len())
            .filter(|&v| !crop_rim.contains(&v) && y.adj[v].len() < lat_info.perfect_degree[y.basis_idx[v]])
            .collect();
        let kind = if c.info.volterra.unwrap_or(false) { "Volterra" } else { "completion" };

        out.push(CoreGalleryEntry {
            name: c.info.name.clone(), kind: kind.to_string(), crystal: y, core_sites, chords,
            under_sites, rings: c.info.rings.clone(), faces, under: c.info.under.clone(),
            added: c.info.added, description: describe_core(&c.info.rings, &bundle.perfect_rings, &c.info.under),
            also: Vec::new(), faces_ok, iso_ok, perfect_rings: bundle.perfect_rings.clone(),
        });
        rims.push(crop_rim);
    }

    for m in &bundle.merged {
        if !m.volterra && out.get(m.equals).is_some_and(|e| e.kind == "Volterra") {
            out[m.equals].also.push(format!("{} of the scan ({} bond(s) added)", m.name, m.chords_len));
        }
    }
    if both_routes { find_both_routes(&mut out, &rims, &bundle.cores, lat_info, radius, max_candidates); }
    if draw {
        for e in &mut out { relax_picture(&mut e.crystal); ensure_embedded(&mut e.crystal, true); }
    }
    out
}

/// Crop `crystal` around its irregular core sites into a `CoreGalleryEntry` (`cores.config_entry`).
pub fn config_core_entry(
    crystal: &BuiltCrystal, rim: &HashSet<usize>, lat_info: &LatticeCoreInfo, radius: usize, draw: bool,
) -> Option<CoreGalleryEntry> {
    let adj: Vec<Vec<usize>> = crystal.adj.iter().map(|es| es.iter().map(|e| e.to).collect()).collect();
    let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
    let drim = graph_ball_dist(&adj, &rim.iter().copied().collect(), usize::MAX);
    let irregular: BTreeSet<usize> = (0..adj.len()).filter(|&i| {
        drim.get(&i).copied().unwrap_or(usize::MAX) > lat_info.r_det
            && !lat_info.perfect_sigs.contains(&layer_signature(&adj, &deg, i, lat_info.r_det))
    }).collect();
    if irregular.is_empty() { return None; }
    let d_cut = 2 * lat_info.r_det + radius + 3;
    let (mut y, new_idx, crop_rim) = crop_around(crystal, &adj, &deg, rim, &irregular, d_cut);
    let cores = inspect_crystal_cores(crystal);
    let faces: Vec<Vec<usize>> = cores.iter()
        .filter(|c| c.sites.len() > 2 && c.sites.iter().all(|v| new_idx.contains_key(v)))
        .map(|c| c.sites.iter().map(|v| new_idx[v]).collect()).collect();
    let mut rings: Vec<usize> = cores.iter().filter_map(|c| c.ring).collect();
    rings.sort_unstable();
    let description = cores.iter().map(|c| c.label.as_str()).collect::<Vec<_>>().join(", ");
    let core_sites: Vec<usize> = irregular.iter().filter_map(|v| new_idx.get(v).copied()).collect();
    let under_sites: Vec<usize> = (0..y.pos.len())
        .filter(|&v| !crop_rim.contains(&v) && y.adj[v].len() < lat_info.perfect_degree[y.basis_idx[v]])
        .collect();
    let mut under: Vec<usize> = under_sites.iter().map(|&v| lat_info.perfect_degree[y.basis_idx[v]] - y.adj[v].len()).collect();
    under.sort_unstable_by(|a, b| b.cmp(a));
    if draw { relax_picture(&mut y); ensure_embedded(&mut y, true); }
    Some(CoreGalleryEntry {
        name: "this configuration".to_string(), kind: "configuration".to_string(), crystal: y,
        core_sites, chords: Vec::new(), under_sites, rings, faces, under, added: 0,
        description, also: Vec::new(), faces_ok: true, iso_ok: None, perfect_rings: Vec::new(),
    })
}

pub(crate) fn resolve_patch_rim(crystal: &BuiltCrystal, lat_info: &LatticeCoreInfo, specs: &[DefectSpec]) -> HashSet<usize> {
    if !specs.is_empty() && let Ok((ctr, r_def)) = patch_centre_and_radius(lat_info, specs, None, None) {
        let max_r = crystal.ref_pos.iter().map(|&p| norm2(sub2(p, ctr))).fold(0.0_f64, f64::max);
        let extra = specs.iter().find_map(|s| match s {
            DefectSpec::HalfWall { wall_id, at, direction, sigma } => {
                let sig_iso = sigma.map(|(m, t)| Isometry2 { m, t });
                resolve_half_wall_spec(&lat_info.lat.cell, wall_id.as_deref(), sig_iso, *at, *direction)
                    .ok()
                    .map(|hw| norm2(add2(sub2(mat2_vec(hw.holonomy.m, hw.at), hw.at), hw.holonomy.t)))
            }
            _ => None,
        }).unwrap_or(0.0);
        if (max_r - r_def).abs() <= 1.2 * lat_info.lmax + extra + 1e-5 { return patch_rim(crystal, lat_info, ctr, r_def); }
    }
    let (ctr, r_inf) = infer_patch_disk(crystal);
    patch_rim(crystal, lat_info, ctr, r_inf)
}

/// Scan Volterra and non-crossing completion cores for Burgers vector `b`.
pub fn core_gallery(
    _group: &str, cell: &Cell2D, _frame: Frame2D, b: [f64; 2], radius: usize, grid: usize, draw: bool,
) -> Result<(Vec<CoreGalleryEntry>, DislocationScanInfo), String> {
    let bundle = scan_dislocation_cores(cell, b, radius, grid, 8, None)?;
    Ok((gallery_from_bundle(&bundle, radius, true, 8, draw), bundle.info))
}

/// Compare a built configuration against the single defect(s) carrying the same total charge
/// (one-shot: `single_cache::equivalent_single_cached` with an empty cache).
#[allow(clippy::too_many_arguments)]
pub fn equivalent_single(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    crystal: &BuiltCrystal,
    radius: usize,
    grid: usize,
    draw_gallery: bool,
) -> Result<EquivalentSingleReport, String> {
    let mut cache = SingleDefectCache::new(1);
    equivalent_single_cached(group, cell, frame, specs, crystal, radius, grid, draw_gallery, &mut cache)
}
