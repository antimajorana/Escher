//! Finite Volterra + non-crossing completion dislocation core scan (`config.dislocation_core_list`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use petgraph::algo::is_isomorphic;
use petgraph::graph::UnGraph;

use crate::defect::combine_types::{CoreCandidateInfo, DislocationScanInfo};
use crate::defect::cone_geometry::{mat2_vec, norm2, sub2};
use crate::defect::core_graph::{
    LatticeCoreInfo, SiteKey, core_graph_with_rim, isomorphic_cores, patch_rim,
};
use crate::defect::equivalence::{is_bipartite, same_proper_holonomy};
use crate::defect::sym_helper::{LatticeGroup, np_allclose_vec};
use crate::defect::volterra_patch::{
    ball_nodes, ball_ungraph, build_volterra_patch, candidate_cuts_for_scan, drawn_charged_faces,
    enumerate_completions, first_pos, perfect_ring_sizes, split_ring, wl_hash_labelled,
    wl_hash_unlabelled,
};
use crate::types::{BuiltCrystal, Cell2D, DirectedBondData, Isometry2};

#[derive(Clone)]
struct LabRecord {
    b_measured: Option<[f64; 2]>,
    core_point: [f64; 2],
    cut: [f64; 2],
    rings: Vec<usize>,
    under: Vec<usize>,
    added: usize,
    lattice_bonds: usize,
    reconstruction_bonds: usize,
    volterra: bool,
    build_idx: usize,
    chords: Vec<(usize, usize)>,
    wl_hash: u64,
    lab_graph: UnGraph<(), ()>,
    lab_name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ScannedCoreBuild {
    pub info: CoreCandidateInfo,
    pub wl_hash: u64,
    pub graph: UnGraph<u16, ()>,
    pub build_idx: usize,
    pub chords: Vec<(usize, usize)>,
    pub build_r: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct MergedLabCore {
    pub name: String,
    pub volterra: bool,
    pub equals: usize,
    pub chords_len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ScanBundle {
    pub lat_info: LatticeCoreInfo,
    pub cores: Vec<ScannedCoreBuild>,
    pub builds: Vec<(BuiltCrystal, Vec<Vec<usize>>)>,
    pub merged: Vec<MergedLabCore>,
    pub perfect_rings: Vec<usize>,
    pub info: DislocationScanInfo,
}

pub(crate) fn crystal_with_dummy_chords(x: &BuiltCrystal, chords: &[(usize, usize)]) -> BuiltCrystal {
    if chords.is_empty() {
        return x.clone();
    }
    let mut y = x.clone();
    for &(u, v) in chords {
        if !y.adj[u].iter().any(|e| e.to == v) {
            y.adj[u].push(DirectedBondData {
                to: v, vec: [0.0, 0.0], kind: 0, seam: None, ang: None, across: false, ghost_stub: None,
            });
            y.adj[v].push(DirectedBondData {
                to: u, vec: [0.0, 0.0], kind: 0, seam: None, ang: None, across: false, ghost_stub: None,
            });
            y.edges.push((u.min(v), u.max(v)));
        }
    }
    y.edges.sort_unstable();
    y
}

pub(crate) fn scan_dislocation_cores(
    cell: &Cell2D,
    b: [f64; 2],
    radius: usize,
    grid: usize,
    max_candidates: usize,
    r_opt: Option<f64>,
) -> Result<ScanBundle, String> {
    let lat_info = LatticeCoreInfo::new(cell);
    let grp = LatticeGroup::new(cell);
    let lat = &lat_info.lat;
    let mut r_patch = r_opt.unwrap_or_else(|| {
        (2 * lat_info.r_det + radius + 3) as f64 * 0.5 * (lat_info.lmean + lat_info.lmax)
            + 0.5 * lat_info.scale
            + norm2(b)
    });
    let cuts = candidate_cuts_for_scan(lat.a, b);
    let perf_sizes = perfect_ring_sizes(lat, lat_info.scale, lat_info.planar);
    let mut step_vecs: HashMap<(usize, usize), Vec<[f64; 2]>> = HashMap::new();
    for s in 0..lat.rule.len() {
        let p0 = lat.pos((0, 0, s));
        for (k2, _) in lat.nbrs((0, 0, s)) {
            step_vecs.entry((s, k2.2)).or_default().push(sub2(lat.pos(k2), p0));
        }
    }
    let is_lattice_bond = |ku: SiteKey, kv: SiteKey| -> bool {
        let d = sub2(lat.pos(kv), lat.pos(ku));
        let Some(ws) = step_vecs.get(&(ku.2, kv.2)) else { return false };
        [-1.0, 0.0, 1.0].into_iter().any(|m| {
            let shifted = [d[0] + m * b[0], d[1] + m * b[1]];
            ws.iter().any(|&w| np_allclose_vec(shifted, w, 1e-5, 1e-6))
        })
    };

    let (mut final_cores, mut final_builds, mut final_merged) = (Vec::new(), Vec::new(), Vec::new());
    let (mut n_builds, mut anomalies, mut capped, mut lab_cores, mut other_class, mut skipped) =
        (0, 0, 0, 0, 0, 0);

    for _attempt in 0..2 {
        n_builds = 0;
        anomalies = 0;
        capped = 0;
        let mut raw_builds: Vec<(BuiltCrystal, Vec<Vec<usize>>)> = Vec::new();
        let mut found: Vec<LabRecord> = Vec::new();

        let record = |found: &mut Vec<LabRecord>,
                      adj_cur: &[Vec<usize>],
                      rings_now: &[Vec<usize>],
                      basis_idx: &[usize],
                      mut rec: LabRecord| {
            let mut core: BTreeSet<usize> = BTreeSet::new();
            for r in rings_now {
                let odd_len = !perf_sizes.contains(&r.len());
                for &v in r {
                    if odd_len || adj_cur[v].len() != lat_info.perfect_degree[basis_idx[v]] {
                        core.insert(v);
                    }
                }
            }
            if core.is_empty() {
                core.extend(rings_now.iter().flatten().copied());
            }
            let h_graph = ball_ungraph(adj_cur, &core, radius);
            let h = wl_hash_unlabelled(&h_graph);
            for f in found.iter_mut() {
                if f.wl_hash == h
                    && f.lab_graph.node_count() == h_graph.node_count()
                    && f.lab_graph.edge_count() == h_graph.edge_count()
                    && (h_graph.node_count() == 0 || is_isomorphic(&f.lab_graph, &h_graph))
                {
                    if rec.volterra && !f.volterra {
                        rec.wl_hash = h;
                        rec.lab_graph = h_graph;
                        *f = rec;
                    }
                    return;
                }
            }
            rec.wl_hash = h;
            rec.lab_graph = h_graph;
            found.push(rec);
        };

        for i in 0..grid {
            for j in 0..grid {
                let core_pt = mat2_vec(
                    lat.a,
                    [(i as f64 + 0.371) / (grid as f64), (j as f64 + 0.413) / (grid as f64)],
                );
                for &cut in &cuts {
                    let patch = build_volterra_patch(&lat_info, b, core_pt, Some(cut), r_patch)?;
                    n_builds += 1;
                    let x = patch.crystal;
                    let adj_x: Vec<Vec<usize>> =
                        x.adj.iter().map(|es| es.iter().map(|e| e.to).collect()).collect();
                    let near = |v: usize| norm2(sub2(x.ref_pos[v], core_pt)) < r_patch - 1.5 * lat_info.lmax;
                    let (bm, rings, core_set) = if lat_info.planar {
                        let ch = drawn_charged_faces(&x);
                        if ch.is_empty() {
                            anomalies += 1;
                            continue;
                        }
                        let mut sum_b = [0.0, 0.0];
                        let mut rw = Vec::with_capacity(ch.len());
                        let mut cs = BTreeSet::new();
                        for (f, vb) in ch {
                            sum_b = [sum_b[0] + vb[0], sum_b[1] + vb[1]];
                            cs.extend(f.iter().copied());
                            rw.push(f);
                        }
                        (Some(sum_b), rw, cs)
                    } else {
                        let cs: BTreeSet<usize> = (0..x.pos.len())
                            .filter(|&v| {
                                adj_x[v].len() != lat_info.perfect_degree[x.basis_idx[v]]
                                    && norm2(sub2(x.ref_pos[v], core_pt))
                                        < r_patch - (radius as f64 + 1.5) * lat_info.lmax
                            })
                            .collect();
                        (None, Vec::new(), cs)
                    };
                    if core_set.is_empty()
                        || ball_nodes(&adj_x, &core_set, radius).into_iter().any(|v| !near(v))
                    {
                        anomalies += 1;
                        continue;
                    }
                    let under_map: BTreeMap<usize, usize> = core_set
                        .iter()
                        .copied()
                        .filter_map(|v| {
                            let pd = lat_info.perfect_degree[x.basis_idx[v]];
                            (adj_x[v].len() < pd).then(|| (v, pd - adj_x[v].len()))
                        })
                        .collect();
                    let mut under_vec: Vec<usize> = under_map.values().copied().collect();
                    under_vec.sort_unstable_by(|a, b| b.cmp(a));
                    let mut ring_sizes: Vec<usize> = rings.iter().map(Vec::len).collect();
                    ring_sizes.sort_unstable();
                    let build_idx = raw_builds.len();
                    raw_builds.push((x, rings.clone()));
                    let (ref x_ref, _) = raw_builds[build_idx];

                    let base = LabRecord {
                        b_measured: bm,
                        core_point: core_pt,
                        cut,
                        rings: ring_sizes,
                        under: under_vec,
                        added: 0,
                        lattice_bonds: 0,
                        reconstruction_bonds: 0,
                        volterra: true,
                        build_idx,
                        chords: Vec::new(),
                        wl_hash: 0,
                        lab_graph: UnGraph::new_undirected(),
                        lab_name: String::new(),
                    };
                    let fallback_rings = [core_set.iter().copied().collect::<Vec<_>>()];
                    let r_slice = if lat_info.planar { rings.as_slice() } else { &fallback_rings };
                    record(&mut found, &adj_x, r_slice, &x_ref.basis_idx, base.clone());

                    if !(lat_info.planar && rings.len() == 1) {
                        continue;
                    }
                    let ring = &rings[0];
                    let pos_ring = first_pos(ring);
                    let edge_set: HashSet<(usize, usize)> = x_ref.edges.iter().copied().collect();
                    let (subs, was_capped) = enumerate_completions(
                        |u, v| edge_set.contains(&(u.min(v), u.max(v))),
                        ring,
                        &under_map,
                        max_candidates,
                    );
                    if was_capped {
                        capped += 1;
                    }
                    let mut adj2 = adj_x.clone();
                    for sub in subs {
                        for &(u, v) in &sub {
                            adj2[u].push(v);
                            adj2[v].push(u);
                        }
                        let nl = sub
                            .iter()
                            .filter(|&&(u, v)| is_lattice_bond(patch.keys[u], patch.keys[v]))
                            .count();
                        let mut use_cnt: HashMap<usize, usize> = HashMap::new();
                        for &(u, v) in &sub {
                            *use_cnt.entry(u).or_default() += 1;
                            *use_cnt.entry(v).or_default() += 1;
                        }
                        let mut left: Vec<usize> = under_map
                            .iter()
                            .filter_map(|(v, &u_cap)| {
                                let rem = u_cap - use_cnt.get(v).copied().unwrap_or(0);
                                (rem > 0).then_some(rem)
                            })
                            .collect();
                        left.sort_unstable_by(|a, b| b.cmp(a));
                        let sub2: Vec<(usize, usize)> = sub
                            .iter()
                            .map(|&(u, v)| if pos_ring[&u] <= pos_ring[&v] { (u, v) } else { (v, u) })
                            .collect();
                        let split = split_ring(ring, &sub2);
                        let mut s_sizes: Vec<usize> = split.iter().map(Vec::len).collect();
                        s_sizes.sort_unstable();
                        let rec = LabRecord {
                            rings: s_sizes,
                            under: left,
                            added: sub.len(),
                            lattice_bonds: nl,
                            reconstruction_bonds: sub.len() - nl,
                            volterra: false,
                            chords: sub.clone(),
                            ..base.clone()
                        };
                        record(&mut found, &adj2, &split, &x_ref.basis_idx, rec);
                        for &(u, v) in &sub {
                            adj2[u].pop();
                            adj2[v].pop();
                        }
                    }
                }
            }
        }

        found.sort_by(|a, b| {
            (!a.volterra)
                .cmp(&(!b.volterra))
                .then_with(|| a.under.iter().sum::<usize>().cmp(&b.under.iter().sum::<usize>()))
                .then_with(|| a.added.cmp(&b.added))
                .then_with(|| a.rings.cmp(&b.rings))
        });
        for (k, f) in found.iter_mut().enumerate() {
            f.lab_name = format!("core {}", k + 1);
        }

        lab_cores = found.len();
        other_class = 0;
        skipped = 0;
        let mut out_cores: Vec<ScannedCoreBuild> = Vec::new();
        let mut merged: Vec<MergedLabCore> = Vec::new();
        let h_target = Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [-b[0], -b[1]] };

        for f in found {
            let Some(bm) = f.b_measured else {
                other_class += 1;
                continue;
            };
            let h_meas = Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [-bm[0], -bm[1]] };
            if !same_proper_holonomy(h_meas, h_target, &grp.plus, grp.a, grp.ainv, 1e-4) {
                other_class += 1;
                continue;
            }
            let x_ch = crystal_with_dummy_chords(&raw_builds[f.build_idx].0, &f.chords);
            let rim = patch_rim(&x_ch, &lat_info, f.core_point, r_patch);
            let Ok(h_core) = core_graph_with_rim(&x_ch, &rim, &lat_info, radius) else {
                skipped += 1;
                continue;
            };
            let h_wl = wl_hash_labelled(&h_core);
            let full_name = format!(
                "{} {}",
                if f.volterra { "Volterra" } else { "completion" },
                f.lab_name
            );
            if let Some(eq_idx) = out_cores
                .iter()
                .position(|o| o.wl_hash == h_wl && isomorphic_cores(&h_core, &o.graph))
            {
                merged.push(MergedLabCore {
                    name: full_name,
                    volterra: f.volterra,
                    equals: eq_idx,
                    chords_len: f.chords.len(),
                });
                continue;
            }
            let bip = is_bipartite(&h_core);
            out_cores.push(ScannedCoreBuild {
                info: CoreCandidateInfo {
                    name: full_name,
                    volterra: Some(f.volterra),
                    rings: f.rings,
                    under: f.under,
                    added: f.added,
                    lattice_bonds: f.lattice_bonds,
                    reconstruction_bonds: f.reconstruction_bonds,
                    bipartite: Some(bip),
                    core_point: Some(f.core_point),
                    cut: Some(f.cut),
                },
                wl_hash: h_wl,
                graph: h_core,
                build_idx: f.build_idx,
                chords: f.chords,
                build_r: r_patch,
            });
        }

        final_cores = out_cores;
        final_builds = raw_builds;
        final_merged = merged;
        if skipped == 0 {
            break;
        }
        r_patch *= 1.4;
    }

    let distinct = final_cores.len();
    Ok(ScanBundle {
        lat_info,
        cores: final_cores,
        builds: final_builds,
        merged: final_merged,
        perfect_rings: perf_sizes.into_iter().collect(),
        info: DislocationScanInfo {
            grid,
            r_patch,
            builds: n_builds,
            anomalies,
            capped,
            max_candidates,
            lab_cores,
            other_class,
            skipped_rim: skipped,
            distinct,
        },
    })
}

/// Scan Volterra and non-crossing completion dislocation cores (`config.dislocation_core_list`).
pub fn dislocation_core_list(
    cell: &Cell2D,
    b: [f64; 2],
    radius: usize,
    grid: usize,
) -> Result<(Vec<CoreCandidateInfo>, DislocationScanInfo), String> {
    let bundle = scan_dislocation_cores(cell, b, radius, grid, 8, None)?;
    let list = bundle.cores.into_iter().map(|c| c.info).collect();
    Ok((list, bundle.info))
}
