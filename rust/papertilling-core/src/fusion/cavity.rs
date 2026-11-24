//! General cavity extractor for arbitrary 2D lattices and defect specifications.
//!
//! Cuts a simply-connected topological-disk cavity around a defect core region—either directly from
//! an already-built `BuiltCrystal` or from the far-field single-defect / perfect-lattice host crystal
//! matching the algebraic total charge in `G~+` (when strict Volterra surgery refuses overlapping
//! cores)—and returns a `PlanarMap` ready for [`enumerate_cavity_cores`].

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::defect::charge::{corners_order, frame_faces, frank_angles, outer_face_index};
use crate::defect::combinations::specs_holonomy;
use crate::defect::cone_builder::ConeBuilder;
use crate::defect::cone_geometry::{norm2, sub2};
use crate::defect::core_graph::LatticeCoreInfo;
use crate::defect::equivalence::{classify_crystal_charge, classify_holonomy_charge, same_charge};
use crate::defect::surgery::{DefectSpec, build_point_defects_with_info, validate_extent};
use crate::types::{BuiltCrystal, Cell2D, Frame2D};

use super::enumerate::{CoreEnumeration, enumerate_cavity_cores, find_cavity_face};
use super::map::{MapEdge, MapVertex, PlanarMap, VertexKind};

/// Cavity extracted from a 2D crystal patch, ready for fixed-boundary core reconnection enumeration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CavityExtraction {
    pub title: String,
    pub algebraic_charge: String,
    pub far_field_source: String,
    /// Boundary cycle (`Complete`) + 1-bond external stubs (`Boundary`), `euler() == 2`, empty interior.
    pub cavity_map: PlanarMap,
    /// CCW cyclic vertex indices of the cavity face in `cavity_map`.
    pub cavity_cycle: Vec<usize>,
    pub orig_inner_verts: usize,
    pub orig_inner_edges: usize,
    pub orig_total_edges: usize,
    /// Same boundary & stubs as `cavity_map` with the original interior vertices and edges retained.
    pub reference_filled: PlanarMap,
    pub note: String,
}

fn defect_centroid(specs: &[DefectSpec]) -> [f64; 2] {
    let n = specs.len() as f64;
    specs.iter().fold([0.0, 0.0], |[sx, sy], spec| {
        let [x, y] = match *spec {
            DefectSpec::Dislocation { at, .. } => at.unwrap_or([0.0, 0.0]),
            DefectSpec::Disclination { at, .. } | DefectSpec::HalfWall { at, .. } => at,
        };
        [sx + x / n, sy + y / n]
    })
}

fn build_far_field_host(
    cell: &Cell2D,
    lat_info: &LatticeCoreInfo,
    charge: &crate::defect::combine_types::ClassifiedCharge,
    centroid: [f64; 2],
) -> Result<(BuiltCrystal, String), String> {
    match charge.kind.as_str() {
        "vacuum" => {
            let r_disk = (5.0 * lat_info.scale).max(5.0 * lat_info.lmax);
            validate_extent(lat_info, centroid, r_disk)?;
            let crystal = ConeBuilder::new(lat_info, centroid, r_disk).finish()?;
            Ok((crystal, String::from("far-field perfect lattice (vacuum total charge)")))
        }
        "dislocation" => {
            let b = charge.b.ok_or_else(|| String::from("missing Burgers vector for dislocation charge"))?;
            let p0 = [centroid[0] - 0.5 * b[0], centroid[1] - 0.5 * b[1]];
            let p_alt = [
                centroid[0] + 0.371 * cell.a1[0] + 0.413 * cell.a2[0],
                centroid[1] + 0.371 * cell.a1[1] + 0.413 * cell.a2[1],
            ];
            let mut last_err = String::from("failed to build single-dislocation far-field host");
            for spec in [DefectSpec::dislocation(b, Some(p0)), DefectSpec::dislocation(b, Some(p_alt)), DefectSpec::dislocation(b, None)] {
                match build_point_defects_with_info(lat_info, &[spec], None, None) {
                    Ok((c, _, _)) => return Ok((c, String::from("far-field single defect (overlapping cores)"))),
                    Err(e) => last_err = e,
                }
            }
            Err(last_err)
        }
        "disclination" => {
            let Some(single) = charge.single.clone() else {
                return Err(format!(
                    "cannot construct a single-defect far-field boundary for total charge {} (|angle| > 180 deg or disallowed centre)",
                    charge.label
                ));
            };
            let (c, _, _) = build_point_defects_with_info(lat_info, &[single], None, None)?;
            Ok((c, String::from("far-field single defect (overlapping cores)")))
        }
        _ => Err(format!(
            "cannot construct a single-defect far-field boundary for total charge {} ({})",
            charge.label, charge.kind
        )),
    }
}

fn perfect_face_sizes(lat_info: &LatticeCoreInfo) -> HashSet<usize> {
    let r = (3.0 * lat_info.scale + 2.0 * lat_info.lmax).max(2.5);
    let Ok(disk) = ConeBuilder::new(lat_info, [0.0, 0.0], r).finish() else { return HashSet::new(); };
    let faces = frame_faces(&disk);
    if faces.len() <= 1 { return HashSet::new(); }
    let k_out = outer_face_index(&faces);
    faces.iter().enumerate().filter(|&(i, _)| i != k_out).map(|(_, f)| f.len()).collect()
}

fn face_directed_edges(face: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let n = face.len();
    (0..n).map(move |i| (face[i], face[(i + 1) % n]))
}

fn dual_neighbors(faces: &[Vec<usize>], edge_to_face: &HashMap<(usize, usize), usize>, fi: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for (u, v) in face_directed_edges(&faces[fi]) {
        if let Some(&fj) = edge_to_face.get(&(v, u)) && fj != fi && !out.contains(&fj) {
            out.push(fj);
        }
    }
    out
}

fn connect_dual_components(
    selected: &mut HashSet<usize>,
    faces: &[Vec<usize>],
    edge_to_face: &HashMap<(usize, usize), usize>,
    k_out: usize,
) {
    let Some(&seed) = selected.iter().min() else { return; };
    loop {
        let (mut comp, mut q) = (HashSet::from([seed]), VecDeque::from([seed]));
        while let Some(fi) = q.pop_front() {
            for fj in dual_neighbors(faces, edge_to_face, fi) {
                if selected.contains(&fj) && comp.insert(fj) { q.push_back(fj); }
            }
        }
        if comp.len() == selected.len() { return; }
        let mut parent: HashMap<usize, usize> = comp.iter().map(|&fi| (fi, fi)).collect();
        // Start equally short connections in face order so reopening the same cavity picks the same bridge.
        let mut frontier: Vec<_> = comp.iter().copied().collect();
        frontier.sort_unstable();
        let mut bfs: VecDeque<_> = frontier.into();
        let mut hit = None;
        while let Some(fi) = bfs.pop_front() {
            for fj in dual_neighbors(faces, edge_to_face, fi) {
                if fj == k_out || parent.contains_key(&fj) { continue; }
                parent.insert(fj, fi);
                if selected.contains(&fj) { hit = Some(fj); break; }
                bfs.push_back(fj);
            }
            if hit.is_some() { break; }
        }
        let Some(mut cur) = hit else { return; };
        while !comp.contains(&cur) {
            selected.insert(cur);
            cur = parent[&cur];
        }
    }
}

fn expand_one_ring(selected: &mut HashSet<usize>, faces: &[Vec<usize>], edge_to_face: &HashMap<(usize, usize), usize>, k_out: usize) {
    let current: Vec<usize> = selected.iter().copied().collect();
    for fi in current {
        for fj in dual_neighbors(faces, edge_to_face, fi) {
            if fj != k_out { selected.insert(fj); }
        }
    }
}

fn count_inner_edges(selected: &HashSet<usize>, faces: &[Vec<usize>]) -> usize {
    let mut directed: HashSet<(usize, usize)> = HashSet::new();
    for &fi in selected { directed.extend(face_directed_edges(&faces[fi])); }
    directed.iter().filter(|&&(u, v)| u < v && directed.contains(&(v, u))).count()
}

fn fill_holes_and_unpinch(
    selected: &mut HashSet<usize>,
    crystal: &BuiltCrystal,
    faces: &[Vec<usize>],
    edge_to_face: &HashMap<(usize, usize), usize>,
    k_out: usize,
) -> Result<(), String> {
    let corners = corners_order(crystal);
    for _ in 0..=faces.len() {
        let (mut ext, mut q) = (HashSet::from([k_out]), VecDeque::from([k_out]));
        while let Some(fi) = q.pop_front() {
            for fj in dual_neighbors(faces, edge_to_face, fi) {
                if !selected.contains(&fj) && ext.insert(fj) { q.push_back(fj); }
            }
        }
        let mut changed = false;
        for fi in 0..faces.len() {
            if fi != k_out && !selected.contains(&fi) && !ext.contains(&fi) {
                selected.insert(fi);
                changed = true;
            }
        }
        let mut s_edges: HashSet<(usize, usize)> = HashSet::new();
        for &fi in selected.iter() { s_edges.extend(face_directed_edges(&faces[fi])); }
        let mut out_deg: HashMap<usize, usize> = HashMap::new();
        for &(u, v) in &s_edges {
            if !s_edges.contains(&(v, u)) { *out_deg.entry(u).or_insert(0) += 1; }
        }
        if let Some(v) = out_deg.iter().filter(|&(_, &d)| d >= 2).map(|(&v, _)| v).min() {
            let ring: Vec<usize> = corners[v].iter().filter_map(|&(_, nb)| edge_to_face.get(&(v, nb)).copied()).collect();
            let k = ring.len();
            let Some(start) = (0..k).find(|&i| selected.contains(&ring[i])) else {
                return Err(String::from("failed to resolve cavity boundary pinch vertex"));
            };
            let (mut wedges, mut i) = (Vec::new(), 0);
            while i < k {
                if !selected.contains(&ring[(start + i) % k]) {
                    let mut w = Vec::new();
                    while i < k && !selected.contains(&ring[(start + i) % k]) {
                        w.push(ring[(start + i) % k]);
                        i += 1;
                    }
                    wedges.push(w);
                } else {
                    i += 1;
                }
            }
            let Some(fill) = wedges.into_iter().filter(|w| !w.contains(&k_out)).min_by_key(Vec::len) else {
                return Err(String::from("extracted cavity pinches against the crystal patch rim"));
            };
            selected.extend(fill);
            changed = true;
        }
        if !changed { return Ok(()); }
    }
    Err(String::from("cavity topological-disk closure did not converge"))
}

type EdgeSet = HashSet<(usize, usize)>;

fn extract_boundary_cycle(selected: &HashSet<usize>, faces: &[Vec<usize>]) -> Result<(Vec<usize>, EdgeSet), String> {
    let mut s_edges: HashSet<(usize, usize)> = HashSet::new();
    for &fi in selected { s_edges.extend(face_directed_edges(&faces[fi])); }
    let mut next_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut total_b = 0usize;
    for &(u, v) in &s_edges {
        if !s_edges.contains(&(v, u)) {
            next_map.entry(u).or_default().push(v);
            total_b += 1;
        }
    }
    if total_b < 4 || next_map.values().any(|ns| ns.len() != 1) {
        return Err(format!("cavity boundary is not a simple closed cycle of length >= 4 (edges = {total_b})"));
    }
    let start = *next_map.keys().min().expect("non-empty boundary");
    let (mut cycle, mut cur) = (Vec::with_capacity(total_b), start);
    for _ in 0..total_b {
        cycle.push(cur);
        let Some(next) = next_map.get(&cur) else {
            return Err(String::from("cavity boundary is open"));
        };
        cur = next[0];
        if cur == start { break; }
    }
    if cur != start || cycle.len() != total_b {
        return Err(String::from("cavity boundary has multiple disconnected cycles"));
    }
    Ok((cycle, s_edges))
}

struct SubgraphSpec<'a> {
    verts: &'a [(String, [f64; 2], VertexKind)],
    vert_orig: &'a [Option<usize>],
    edges: &'a [(u32, usize, usize)],
    edge_orig: &'a [(usize, usize)],
    outer_leg: u32,
}

fn build_combinatorial_map(crystal: &BuiltCrystal, spec: &SubgraphSpec<'_>) -> Result<PlanarMap, String> {
    let nbr_rank: Vec<HashMap<usize, usize>> = corners_order(crystal)
        .iter()
        .map(|lst| lst.iter().enumerate().map(|(k, &(_, nb))| (nb, k)).collect())
        .collect();
    let mut map = PlanarMap {
        verts: spec.verts.iter().map(|(n, p, k)| MapVertex { name: n.clone(), pos: *p, kind: *k, rot: Vec::new() }).collect(),
        edges: Vec::with_capacity(spec.edges.len()),
        outer: 0,
    };
    for &(label, a, b) in spec.edges {
        let e = map.edges.len();
        map.edges.push(MapEdge { label, ends: [a, b], alive: true, ctrl: None, loop_dir_deg: 0.0 });
        map.verts[a].rot.push(2 * e);
        map.verts[b].rot.push(2 * e + 1);
    }
    for (v, orig_opt) in spec.vert_orig.iter().enumerate() {
        let Some(orig_u) = *orig_opt else { continue; };
        let ranks = &nbr_rank[orig_u];
        let mut keyed: Vec<(usize, usize)> = map.verts[v]
            .rot
            .iter()
            .map(|&d| {
                let (ea, eb) = spec.edge_orig[d / 2];
                (*ranks.get(&(if d % 2 == 0 { eb } else { ea })).unwrap_or(&usize::MAX), d)
            })
            .collect();
        keyed.sort_unstable_by_key(|&(r, _)| r);
        map.verts[v].rot = keyed.into_iter().map(|(_, d)| d).collect();
    }
    let e = map.edge_index(spec.outer_leg)?;
    let leaf = (0..2)
        .find(|&k| { let v = &map.verts[map.edges[e].ends[k]]; v.kind == VertexKind::Boundary && v.rot.len() == 1 })
        .ok_or_else(|| format!("edge {} is not an external leg", spec.outer_leg))?;
    map.outer = 2 * e + leaf;
    if map.euler() != 2 {
        return Err(format!("combinatorial cavity map is not planar connected (V - E + F = {})", map.euler()));
    }
    Ok(map)
}

fn build_verified_map(
    crystal: &BuiltCrystal,
    spec: &SubgraphSpec<'_>,
    expected_bounded_faces: usize,
    cavity_cycle: Option<&[usize]>,
) -> Result<PlanarMap, String> {
    let borrowed: Vec<(&str, [f64; 2], VertexKind)> = spec.verts.iter().map(|(n, p, k)| (n.as_str(), *p, *k)).collect();
    let matches_shape = |m: &PlanarMap| {
        m.euler() == 2
            && m.bounded_faces().len() == expected_bounded_faces
            && cavity_cycle.is_none_or(|cyc| find_cavity_face(m, cyc).is_ok())
    };
    if let Ok(m) = PlanarMap::from_drawing(&borrowed, spec.edges, spec.outer_leg) && matches_shape(&m) {
        return Ok(m);
    }
    let m = build_combinatorial_map(crystal, spec)?;
    if !matches_shape(&m) {
        return Err(format!("extracted map has {} bounded faces (expected {expected_bounded_faces})", m.bounded_faces().len()));
    }
    Ok(m)
}

/// Cut a clean simply-connected cavity around a defect core region on any 2D lattice.
///
/// Uses `existing_crystal` if provided and non-empty; otherwise computes the algebraic total charge
/// of `specs` in `G~+` and constructs the matching far-field host crystal (perfect lattice for vacuum,
/// or single dislocation / allowed single disclination), returning `Err` if no single-defect far-field
/// boundary exists (e.g. `|angle| > 180 deg`, `whole_turn`, or `partial_dislocation`).
pub fn extract_defect_cavity(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    existing_crystal: Option<&BuiltCrystal>,
    extra_rings: usize,
) -> Result<CavityExtraction, String> {
    let (total_frank, h_total) = specs_holonomy(specs)?;
    let centroid = defect_centroid(specs);
    let charge = classify_holonomy_charge(group, cell, frame, Some(total_frank), h_total, [0.0, 0.0])?;
    let lat_info = LatticeCoreInfo::new(cell);

    let (owned_crystal, far_field_source, measured_charge);
    let crystal: &BuiltCrystal = if let Some(c) = existing_crystal.filter(|c| !c.pos.is_empty()) {
        far_field_source = String::from("built configuration");
        measured_charge = Some(classify_crystal_charge(group, cell, frame, c)?);
        c
    } else {
        let (built, src) = build_far_field_host(cell, &lat_info, &charge, centroid)?;
        owned_crystal = built;
        far_field_source = src;
        measured_charge = None;
        &owned_crystal
    };

    let faces = frame_faces(crystal);
    if faces.len() < 2 { return Err(String::from("host crystal has no bounded faces")); }
    let k_out = outer_face_index(&faces);
    let outer_verts: HashSet<usize> = faces[k_out].iter().copied().collect();
    let mut edge_to_face: HashMap<(usize, usize), usize> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for e in face_directed_edges(f) { edge_to_face.insert(e, fi); }
    }

    let perf_sizes = perfect_face_sizes(&lat_info);
    let franks = frank_angles(crystal, &faces);
    let irreg_verts: HashSet<usize> = (0..crystal.adj.len())
        .filter(|&v| {
            !outer_verts.contains(&v)
                && (crystal.adj[v].len() != lat_info.perfect_degree[crystal.basis_idx[v]] || crystal.cone_sites.contains(&v))
        })
        .collect();
    let mut selected: HashSet<usize> = (0..faces.len())
        .filter(|&fi| {
            fi != k_out
                && (!perf_sizes.contains(&faces[fi].len())
                    || franks[fi].abs() > 1e-2
                    || faces[fi].iter().any(|v| irreg_verts.contains(v)))
        })
        .collect();

    if selected.is_empty() {
        let v_near = (0..crystal.pos.len())
            .filter(|v| !outer_verts.contains(v))
            .min_by(|&a, &b| norm2(sub2(crystal.pos[a], centroid)).total_cmp(&norm2(sub2(crystal.pos[b], centroid))))
            .ok_or_else(|| String::from("host crystal has no interior vertices"))?;
        for (fi, f) in faces.iter().enumerate() {
            if fi != k_out && f.contains(&v_near) { selected.insert(fi); }
        }
    } else {
        connect_dual_components(&mut selected, &faces, &edge_to_face, k_out);
    }
    if count_inner_edges(&selected, &faces) == 0 {
        expand_one_ring(&mut selected, &faces, &edge_to_face, k_out);
    }
    for _ in 0..extra_rings {
        expand_one_ring(&mut selected, &faces, &edge_to_face, k_out);
    }
    fill_holes_and_unpinch(&mut selected, crystal, &faces, &edge_to_face, k_out)?;

    let (cycle_orig, s_edges) = extract_boundary_cycle(&selected, &faces)?;
    if cycle_orig.iter().any(|v| outer_verts.contains(v)) {
        return Err(String::from("extracted cavity reaches the crystal patch rim; build with a larger patch or smaller extra_rings"));
    }

    let m = cycle_orig.len();
    let cycle_set: HashSet<usize> = cycle_orig.iter().copied().collect();
    let (mut verts, mut vert_orig, mut orig_to_map) = (Vec::new(), Vec::new(), HashMap::new());
    for (ci, &u) in cycle_orig.iter().enumerate() {
        orig_to_map.insert(u, ci);
        verts.push((format!("v{ci}"), crystal.pos[u], VertexKind::Complete));
        vert_orig.push(Some(u));
    }
    let (mut edges, mut edge_orig, mut next_label) = (Vec::new(), Vec::new(), 1u32);
    for ci in 0..m {
        let cj = (ci + 1) % m;
        edges.push((next_label, ci, cj));
        edge_orig.push((cycle_orig[ci], cycle_orig[cj]));
        next_label += 1;
    }

    let (mut outer_leg, mut stub_idx) = (None, 0usize);
    for (ci, &u) in cycle_orig.iter().enumerate() {
        for e in &crystal.adj[u] {
            if s_edges.contains(&(u, e.to)) || s_edges.contains(&(e.to, u)) { continue; }
            let stub_pos = e.ghost_stub.unwrap_or_else(|| {
                let (pu, pz) = (crystal.pos[u], crystal.pos[e.to]);
                [pu[0] + 0.45 * (pz[0] - pu[0]), pu[1] + 0.45 * (pz[1] - pu[1])]
            });
            let sv = verts.len();
            verts.push((format!("s{stub_idx}"), stub_pos, VertexKind::Boundary));
            vert_orig.push(None);
            edges.push((next_label, ci, sv));
            edge_orig.push((u, e.to));
            if outer_leg.is_none() { outer_leg = Some(next_label); }
            next_label += 1;
            stub_idx += 1;
        }
    }
    let outer_leg = outer_leg.ok_or_else(|| String::from("extracted cavity has no external stubs"))?;
    let cavity_cycle: Vec<usize> = (0..m).collect();
    let cav_spec = SubgraphSpec { verts: &verts, vert_orig: &vert_orig, edges: &edges, edge_orig: &edge_orig, outer_leg };
    let cavity_map = build_verified_map(crystal, &cav_spec, 1, Some(&cavity_cycle))?;

    let mut inner_orig: BTreeSet<usize> = BTreeSet::new();
    for &fi in &selected {
        for &v in &faces[fi] {
            if !cycle_set.contains(&v) { inner_orig.insert(v); }
        }
    }
    let (mut ref_verts, mut ref_vert_orig) = (verts, vert_orig);
    for (k, &u) in inner_orig.iter().enumerate() {
        orig_to_map.insert(u, ref_verts.len());
        ref_verts.push((format!("i{k}"), crystal.pos[u], VertexKind::Complete));
        ref_vert_orig.push(Some(u));
    }
    let mut inner_edge_pairs: Vec<(usize, usize)> =
        s_edges.iter().filter(|&&(u, v)| u < v && s_edges.contains(&(v, u))).copied().collect();
    inner_edge_pairs.sort_unstable();
    let (mut ref_edges, mut ref_edge_orig) = (edges, edge_orig);
    for &(u, v) in &inner_edge_pairs {
        ref_edges.push((next_label, orig_to_map[&u], orig_to_map[&v]));
        ref_edge_orig.push((u, v));
        next_label += 1;
    }
    let ref_spec = SubgraphSpec { verts: &ref_verts, vert_orig: &ref_vert_orig, edges: &ref_edges, edge_orig: &ref_edge_orig, outer_leg };
    let reference_filled = build_verified_map(crystal, &ref_spec, selected.len(), None)?;

    let (orig_inner_verts, orig_inner_edges, orig_total_edges) = (inner_orig.len(), inner_edge_pairs.len(), reference_filled.edges.len());
    let displayed_charge = measured_charge.as_ref().map_or(&charge.label, |q| &q.label);
    let title = format!("{group} {displayed_charge} ({m}-gon cavity)");
    let mut note = format!(
        "Extracted {m}-gon cavity from {far_field_source} ({} bounded face(s), {orig_inner_verts} interior vert(s), {orig_inner_edges} interior edge(s), {stub_idx} external stub(s), extra_rings={extra_rings}).",
        selected.len()
    );
    if measured_charge.as_ref().is_some_and(|q| !same_charge(group, cell, q, &charge, false)) {
        note.push_str(" The input-order algebraic product and measured boundary give different charges. The cavity uses the measured boundary.");
    }
    Ok(CavityExtraction {
        title, algebraic_charge: charge.label, far_field_source, cavity_map, cavity_cycle,
        orig_inner_verts, orig_inner_edges, orig_total_edges, reference_filled, note,
    })
}

/// Extract a cavity for `specs` on `(group, cell, frame)` and enumerate valid lattice core reconnections.
#[allow(clippy::too_many_arguments)]
pub fn enumerate_defect_cavity_reconnections(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    existing_crystal: Option<&BuiltCrystal>,
    extra_rings: usize,
    inner_edges_override: Option<usize>,
    max_inner_verts: usize,
    max_cores: usize,
) -> Result<(CavityExtraction, CoreEnumeration), String> {
    let extraction = extract_defect_cavity(group, cell, frame, specs, existing_crystal, extra_rings)?;
    let inner_edges = inner_edges_override.unwrap_or_else(|| extraction.orig_inner_edges.max(1));
    let first_label = extraction.cavity_map.edges.iter().map(|e| e.label).max().unwrap_or(0) + 1;
    let mut en = enumerate_cavity_cores(
        &extraction.cavity_map, &extraction.cavity_cycle, inner_edges, max_inner_verts, false, first_label, max_cores,
    )?;
    en.surrounding_name = extraction.title.clone();
    Ok((extraction, en))
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn dual_paths_of_equal_length_choose_the_first_face() {
        // Two rows of three squares. The left and right columns can be joined via either middle square.
        let faces: Vec<_> = (0..3).flat_map(|x| [x, x + 4])
            .map(|v| vec![v, v + 1, v + 5, v + 4]).collect();
        let edge_to_face = faces.iter().enumerate()
            .flat_map(|(i, f)| face_directed_edges(f).map(move |e| (e, i))).collect();
        for _ in 0..32 {
            let mut selected = HashSet::from([0, 1, 4, 5]);
            connect_dual_components(&mut selected, &faces, &edge_to_face, faces.len());
            assert_eq!(selected, HashSet::from([0, 1, 2, 4, 5]));
        }
    }

    #[test]
    fn disjoint_boundary_cycles_are_rejected_even_when_their_lengths_match() {
        let faces = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let error = extract_boundary_cycle(&HashSet::from([0, 1]), &faces).unwrap_err();
        assert!(error.contains("multiple disconnected cycles"), "{error}");
    }
}
