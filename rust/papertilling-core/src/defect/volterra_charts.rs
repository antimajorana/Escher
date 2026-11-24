//! Signed strip removal and insertion charts for a Volterra dislocation.
use super::cone_geometry::{add2, cross2, dot2, mat2_inv, mat2_vec, norm2, sub2};
use super::core_graph::{LatticeCoreInfo, PerfectLattice, SiteKey, TWO_PI};
use super::sym_helper::np_allclose_vec;
use super::volterra_patch::VolterraPatch;
use crate::types::{BuiltCrystal, DirectedBondData};
use std::collections::{BTreeMap, HashMap, VecDeque};

type ChartEdges = BTreeMap<(usize, usize), ([f64; 2], usize)>;

fn collect_chart_edges(
    lat: &PerfectLattice,
    where_map: &HashMap<SiteKey, usize>,
    edge_map: &mut ChartEdges,
) -> Result<(), String> {
    for (&img, &u) in where_map {
        let p_img = lat.pos(img);
        for (k2, kind) in lat.nbrs(img) {
            let Some(&v) = where_map.get(&k2) else {
                continue;
            };
            if v == u {
                continue;
            }
            let vec_uv = sub2(lat.pos(k2), p_img);
            for (pair, vec_dir) in [((u, v), vec_uv), ((v, u), [-vec_uv[0], -vec_uv[1]])] {
                if let Some(&(old_v, _)) = edge_map.get(&pair)
                    && !np_allclose_vec(old_v, vec_dir, 1e-5, 1e-8)
                {
                    return Err(format!("two charts disagree on bond {pair:?}"));
                }
                edge_map.insert(pair, (vec_dir, kind));
            }
        }
    }
    Ok(())
}

/// A clipped inserted strip may leave disconnected fragments at the outer rim.
fn keep_main_component(mut patch: VolterraPatch) -> VolterraPatch {
    let n = patch.crystal.pos.len();
    let (mut seen, mut best, mut dq) = (vec![false; n], Vec::new(), VecDeque::new());
    for s in 0..n {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        dq.push_back(s);
        let mut comp = vec![s];
        while let Some(u) = dq.pop_front() {
            for e in &patch.crystal.adj[u] {
                if !seen[e.to] {
                    seen[e.to] = true;
                    comp.push(e.to);
                    dq.push_back(e.to);
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    if best.len() == n {
        return patch;
    }
    best.sort_unstable();
    let mut new_id = vec![usize::MAX; n];
    for (ni, &oi) in best.iter().enumerate() {
        new_id[oi] = ni;
    }
    let mut adj = vec![Vec::new(); best.len()];
    let mut edges = Vec::new();
    for (ni, &oi) in best.iter().enumerate() {
        for e in &patch.crystal.adj[oi] {
            let nj = new_id[e.to];
            if nj != usize::MAX {
                if ni < nj {
                    edges.push((ni, nj));
                }
                adj[ni].push(DirectedBondData { to: nj, ..*e });
            }
        }
    }
    edges.sort_unstable();
    patch.keys = best.iter().map(|&i| patch.keys[i]).collect();
    patch.crystal.pos = best.iter().map(|&i| patch.crystal.pos[i]).collect();
    patch.crystal.ref_pos = best.iter().map(|&i| patch.crystal.ref_pos[i]).collect();
    patch.crystal.basis_idx = best.iter().map(|&i| patch.crystal.basis_idx[i]).collect();
    patch.crystal.side = vec![None; best.len()];
    patch.crystal.edges = edges;
    patch.crystal.adj = adj;
    patch
}

fn slit_embed(p: [f64; 2], c_mid: [f64; 2], dh: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let r = sub2(p, c_mid);
    let ang = (-cross2(dh, r)).atan2(dot2(r, dh)).rem_euclid(TWO_PI);
    let f = 1.0 - ang / TWO_PI;
    [p[0] - f * b[0], p[1] - f * b[1]]
}

/// Build a single Volterra dislocation on `lat_info` using SF/RH cut-and-reglue charts:
/// 3-chart strip removal when `cross2(d, b) < 0` and 4-chart strip insertion when `cross2(d, b) > 0`.
pub(crate) fn build_volterra_patch(
    lat_info: &LatticeCoreInfo,
    b: [f64; 2],
    core_pt: [f64; 2],
    cut: Option<[f64; 2]>,
    r_patch: f64,
) -> Result<VolterraPatch, String> {
    super::surgery::validate_extent(lat_info, core_pt, r_patch)?;
    let lat = &lat_info.lat;
    let fb = mat2_vec(lat.a_inv, b);
    if fb
        .iter()
        .any(|v| !v.is_finite() || v.abs() > f64::from(i32::MAX - 8))
    {
        return Err("The Burgers vector exceeds the supported lattice indices.".into());
    }
    let m_b = [
        fb[0].round_ties_even() as i32,
        fb[1].round_ties_even() as i32,
    ];
    if norm2(b) < 1e-9
        || (fb[0] - m_b[0] as f64).abs() > 1e-6
        || (fb[1] - m_b[1] as f64).abs() > 1e-6
    {
        return Err(format!("b = {b:?} is not a non-zero lattice translation"));
    }
    let lb = norm2(b);
    let d = cut.unwrap_or([-b[1] / lb, b[0] / lb]);
    let ld = d[0].hypot(d[1]);
    if !ld.is_finite() || ld == 0.0 {
        return Err("The cut direction must be finite and nonzero.".into());
    }
    let d = [d[0] / ld, d[1] / ld];
    let cr = cross2(d, b);
    if cr.abs() < 1e-12 * lb {
        return Err("cut direction is parallel to Burgers vector b".to_string());
    }
    // Both the original and translated chart addresses must fit the lattice index type.
    let shifted_centre = if cr < 0.0 {
        sub2(core_pt, b)
    } else {
        add2(core_pt, b)
    };
    super::surgery::validate_extent(lat_info, shifted_centre, r_patch)?;
    let b_inv = mat2_inv([[b[0], d[0]], [b[1], d[1]]]);
    let all_keys = lat.sites_in_disk(core_pt, r_patch);
    let c_mid = [core_pt[0] + 0.5 * b[0], core_pt[1] + 0.5 * b[1]];

    let (keys, ref_pos, pos, edge_map) = if cr < 0.0 {
        let (mut keys, mut ts, mut ref_pos) = (
            Vec::with_capacity(all_keys.len()),
            Vec::with_capacity(all_keys.len()),
            Vec::with_capacity(all_keys.len()),
        );
        for k in all_keys {
            let p = lat.pos(k);
            let [t, s] = mat2_vec(b_inv, sub2(p, core_pt));
            if s > 1e-9 && (-1e-9..(1.0 - 1e-9)).contains(&t) {
                continue;
            }
            keys.push(k);
            ts.push((t, s));
            ref_pos.push(p);
        }
        let mut edge_map = BTreeMap::new();
        for chart in 0..3 {
            let mut where_map = HashMap::with_capacity(keys.len());
            for (u, (&k, &(t, s))) in keys.iter().zip(&ts).enumerate() {
                let (ok, sh) = match chart {
                    0 => (s <= 1e-9 || t < -1e-9, 0),
                    1 => (t >= 1.0 - 1e-9, -1),
                    _ => (s > 1e-9, if t >= 1.0 - 1e-9 { -1 } else { 0 }),
                };
                if ok {
                    let img = if sh == 0 {
                        k
                    } else {
                        (k.0 + sh * m_b[0], k.1 + sh * m_b[1], k.2)
                    };
                    where_map.insert(img, u);
                }
            }
            collect_chart_edges(lat, &where_map, &mut edge_map)?;
        }
        let pos = ref_pos
            .iter()
            .map(|&p| slit_embed(p, c_mid, d, b))
            .collect();
        (keys, ref_pos, pos, edge_map)
    } else {
        let orig_n = all_keys.len();
        let (mut keys, mut ts, mut ref_pos) = (
            Vec::with_capacity(orig_n + 16),
            Vec::with_capacity(orig_n),
            Vec::with_capacity(orig_n + 16),
        );
        let mut duplicate_sources = Vec::new();
        for k in all_keys {
            let p = lat.pos(k);
            let [t, s] = mat2_vec(b_inv, sub2(p, core_pt));
            let i = keys.len();
            keys.push(k);
            ts.push((t, s));
            ref_pos.push(p);
            if s > 1e-9 && (-1e-9..(1.0 - 1e-9)).contains(&t) {
                duplicate_sources.push(i);
            }
        }
        let mut dup = HashMap::with_capacity(duplicate_sources.len());
        for i in duplicate_sources {
            let nv = keys.len();
            keys.push(keys[i]);
            ref_pos.push(ref_pos[i]);
            dup.insert(i, nv);
        }
        let mut edge_map = BTreeMap::new();
        // The four charts cover the uncut back, the left and right banks, and
        // the joined forward strip. In the forward chart, the extra copy and
        // the right bank have lattice addresses translated by b.
        for chart in 0..4 {
            let mut where_map = HashMap::with_capacity(keys.len());
            for (i, &(t, s)) in ts.iter().enumerate() {
                let k = keys[i];
                let shifted = (k.0 + m_b[0], k.1 + m_b[1], k.2);
                match chart {
                    0 if s <= 1e-9 => {
                        where_map.insert(k, i);
                    }
                    1 if t < -1e-9 || (s > 1e-9 && t < 1.0 - 1e-9) => {
                        where_map.insert(k, i);
                    }
                    2 if s <= 1e-9 && t >= -1e-9 => {
                        where_map.insert(k, i);
                    }
                    2 => {
                        if let Some(&nv) = dup.get(&i) {
                            where_map.insert(k, nv);
                        } else if s > 1e-9 && t >= 1.0 - 1e-9 {
                            where_map.insert(k, i);
                        }
                    }
                    3 if s > 1e-9 => {
                        if t < 1.0 - 1e-9 {
                            where_map.insert(k, i);
                        }
                        if let Some(&nv) = dup.get(&i) {
                            where_map.insert(shifted, nv);
                        } else if t >= 1.0 - 1e-9 {
                            where_map.insert(shifted, i);
                        }
                    }
                    _ => {}
                }
            }
            collect_chart_edges(lat, &where_map, &mut edge_map)?;
        }
        let dh_neg = [-d[0], -d[1]];
        let pos = ref_pos
            .iter()
            .enumerate()
            .map(|(u, &p)| {
                let shift_b = u >= orig_n || {
                    let (t, s) = ts[u];
                    (s <= 1e-9 && t >= -1e-9) || (s > 1e-9 && t >= 1.0 - 1e-9)
                };
                slit_embed(if shift_b { add2(p, b) } else { p }, c_mid, dh_neg, b)
            })
            .collect();
        (keys, ref_pos, pos, edge_map)
    };

    let n = keys.len();
    let (mut adj, mut edges) = (vec![Vec::new(); n], Vec::new());
    for ((u, v), (vec, kind)) in edge_map {
        if u < v {
            edges.push((u, v));
        }
        adj[u].push(DirectedBondData {
            to: v,
            vec,
            kind,
            seam: None,
            ang: None,
            across: false,
            ghost_stub: None,
        });
    }
    let basis_idx: Vec<usize> = keys.iter().map(|&k| k.2).collect();
    let crystal = BuiltCrystal {
        pos,
        ref_pos,
        basis_idx,
        side: vec![None; n],
        cone_sites: Vec::new(),
        edges,
        adj,
        wall_sigma: None,
        is_non_orientable: false,
    };
    let patch = VolterraPatch { crystal, keys };
    Ok(if cr > 0.0 {
        keep_main_component(patch)
    } else {
        patch
    })
}
