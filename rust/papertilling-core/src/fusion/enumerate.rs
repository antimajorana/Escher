//! Planar lattice-core enumeration inside a fixed surrounding cavity.
//!
//! Author decisions (2026-11-22): fixed boundary & edge count, lattice cores (`Complete` valence >= 3,
//! bounded face sides >= 3, `euler() == 2`), generalized to arbitrary lattice cavities with up to
//! `max_inner_verts` interior vertices (currently supporting `0` and `1`).

use std::collections::{BTreeMap, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::map::{Census, FaceWalk, MapEdge, MapVertex, PlanarMap, VertexKind};
use super::moves::Side;
use super::wang::{A, B, C, D, E, F, G, H};

/// One valid lattice core inside a fixed surrounding cavity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreConnection {
    pub id: usize,
    pub title: String,
    pub description: String,
    /// Interior edges (chords and spokes to interior vertices) as canonical vertex pairs `(u, v)` (`u < v`).
    pub chords: Vec<(usize, usize)>,
    /// Number of interior vertices added inside the cavity (`0` for pure chords, `1` for one interior vertex).
    pub inner_verts: usize,
    /// Equivalence-class index under the cavity's automorphism group (`0, 1, ...` in presentation order).
    pub orbit_id: usize,
    /// Number of enumerated cores in this symmetry orbit (equals exact group-orbit size when `!truncated`).
    pub orbit_size: usize,
    /// True when every `Complete` vertex has valence 4 and every bounded face has 4 sides.
    pub is_regular_square: bool,
    pub map: PlanarMap,
    pub census: Census,
}

/// Valid lattice cores for one fixed surrounding cavity, edge budget, and interior-vertex bound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreEnumeration {
    pub surrounding_name: String,
    pub cavity_sides: usize,
    pub inner_edges: usize,
    pub total_edges: usize,
    pub max_inner_verts: usize,
    pub boundary_vertices_fixed: bool,
    pub cores: Vec<CoreConnection>,
    pub regular_count: usize,
    pub symmetry_orbit_count: usize,
    pub cavity_symmetry_order: usize,
    pub truncated: bool,
    pub slide_connected_to_regular: Option<bool>,
    pub provenance: String,
}

struct EnumCore {
    conn: CoreConnection,
    spokes: Vec<usize>,
    cycle_chords: Vec<(usize, usize)>,
}

struct SearchCtx<'a> {
    boundary_map: &'a PlanarMap,
    walk: &'a FaceWalk,
    first_label: u32,
    cap: usize,
    deficits: Vec<usize>,
    out: Vec<EnumCore>,
    truncated: bool,
}

#[derive(Clone, Copy)]
struct SectorChord {
    sector: usize,
    a: usize,
    b: usize,
    ci: usize,
    cj: usize,
}

fn chords_cross(p: (usize, usize), q: (usize, usize)) -> bool {
    let ((a, b), (c, d)) = (p, q);
    (a < c && c < b && b < d) || (c < a && a < d && d < b)
}

fn is_cyclic_match(a: &[usize], b: &[usize]) -> bool {
    let n = a.len();
    n == b.len()
        && n > 0
        && ((0..n).any(|s| (0..n).all(|k| a[(s + k) % n] == b[k]))
            || (0..n).any(|s| (0..n).all(|k| a[(s + n - k) % n] == b[k])))
}

pub(crate) fn find_cavity_face(map: &PlanarMap, cavity_cycle: &[usize]) -> Result<FaceWalk, String> {
    if cavity_cycle.len() < 3 {
        return Err(String::from("cavity cycle must have at least 3 vertices"));
    }
    for (i, &v) in cavity_cycle.iter().enumerate() {
        if v >= map.verts.len() || map.verts[v].kind == VertexKind::Removed {
            return Err(format!("cavity vertex {v} is not active in the boundary map"));
        }
        if cavity_cycle[..i].contains(&v) {
            return Err(format!("cavity cycle repeats vertex {}", map.verts[v].name));
        }
    }
    let matches: Vec<FaceWalk> =
        map.bounded_faces().into_iter().filter(|f| is_cyclic_match(&f.vertices, cavity_cycle)).collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => Err(String::from("cavity cycle does not match any bounded face of the boundary map")),
        _ => Err(String::from("cavity cycle matches more than one bounded face")),
    }
}

fn fmt_sorted(mut xs: Vec<usize>) -> String {
    xs.sort_unstable();
    xs.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn chord_title(map: &PlanarMap, chords: &[(usize, usize)]) -> String {
    if chords.is_empty() {
        return String::from("no interior edges");
    }
    chords.iter().map(|&(u, v)| format!("{}–{}", map.verts[u].name, map.verts[v].name)).collect::<Vec<_>>().join(", ")
}

/// Insert optional interior vertex (`!spokes.is_empty()`) and non-crossing chords inside `walk`
/// in exact planar CCW order at every cavity vertex and at the interior vertex `w`.
fn build_core_map(
    boundary_map: &PlanarMap,
    walk: &FaceWalk,
    spokes: &[usize],
    cycle_chords: &[(usize, usize)],
    first_label: u32,
) -> (PlanarMap, Vec<(usize, usize)>) {
    let (n, d) = (walk.vertices.len(), spokes.len());
    let mut map = boundary_map.clone();
    let w_opt = if d > 0 {
        let w = map.verts.len();
        let (mut cx, mut cy) = (0.0, 0.0);
        for &si in spokes {
            let p = map.verts[walk.vertices[si]].pos;
            cx += p[0];
            cy += p[1];
        }
        let name = if map.verts.iter().any(|v| v.kind != VertexKind::Removed && v.name == "w") {
            format!("w{w}")
        } else {
            String::from("w")
        };
        map.verts.push(MapVertex { name, pos: [cx / d as f64, cy / d as f64], kind: VertexKind::Complete, rot: Vec::with_capacity(d) });
        Some(w)
    } else {
        None
    };

    enum Role { Spoke(usize, usize), Chord(usize, usize) }
    let mut ordered: Vec<((usize, usize), Role)> = Vec::with_capacity(d + cycle_chords.len());
    if let Some(w) = w_opt {
        for (r, &si) in spokes.iter().enumerate() {
            ordered.push(((walk.vertices[si], w), Role::Spoke(r, si)));
        }
    }
    for &(ci, cj) in cycle_chords {
        let (u, v) = (walk.vertices[ci], walk.vertices[cj]);
        if u < v { ordered.push(((u, v), Role::Chord(ci, cj))); } else { ordered.push(((v, u), Role::Chord(cj, ci))); }
    }
    ordered.sort_unstable_by_key(|&(verts, _)| verts);

    let mut inserts: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut w_darts: Vec<(usize, usize)> = Vec::with_capacity(d);
    for (k, &((u, v), ref role)) in ordered.iter().enumerate() {
        let e = map.edges.len();
        map.edges.push(MapEdge { label: first_label + k as u32, ends: [u, v], alive: true, ctrl: None, loop_dir_deg: 0.0 });
        match *role {
            Role::Spoke(r, si) => {
                let l_prev = (si + n - spokes[(r + d - 1) % d]) % n;
                inserts[si].push((2 * l_prev + 1, 2 * e));
                w_darts.push((r, 2 * e + 1));
            }
            Role::Chord(ci, cj) => {
                inserts[ci].push((2 * ((ci + n - cj) % n), 2 * e));
                inserts[cj].push((2 * ((cj + n - ci) % n), 2 * e + 1));
            }
        }
    }
    for (ci, mut list) in inserts.into_iter().enumerate() {
        if list.is_empty() { continue; }
        list.sort_unstable_by_key(|&(key, _)| key);
        let v = walk.vertices[ci];
        let in_dart = walk.darts[(ci + n - 1) % n] ^ 1;
        let rot = &mut map.verts[v].rot;
        let pos = rot.iter().position(|&d| d == in_dart).expect("cavity incoming dart is at its vertex");
        for (offset, (_, dart)) in list.into_iter().enumerate() {
            rot.insert(pos + 1 + offset, dart);
        }
    }
    if let Some(w) = w_opt {
        // `walk` is CW around the cavity, so CCW around `w` visits spokes in descending `r`.
        w_darts.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
        map.verts[w].rot = w_darts.into_iter().map(|(_, dart)| dart).collect();
    }
    (map, ordered.into_iter().map(|(verts, _)| verts).collect())
}

fn evaluate_core(
    map: PlanarMap,
    chords: Vec<(usize, usize)>,
    spokes: Vec<usize>,
    mut cycle_chords: Vec<(usize, usize)>,
) -> Option<EnumCore> {
    if map.euler() != 2 { return None; }
    let census = map.census();
    if !census.valences.iter().all(|&(_, k)| k >= 3) || !census.face_sides.iter().all(|&s| s >= 3) {
        return None;
    }
    let is_regular_square = !census.valences.is_empty()
        && census.valences.iter().all(|&(_, k)| k == 4)
        && !census.face_sides.is_empty()
        && census.face_sides.iter().all(|&s| s == 4);
    let inner_verts = usize::from(!spokes.is_empty());
    let title = chord_title(&map, &chords);
    let kind_lbl = if inner_verts == 0 { "Chords" } else { "1-vertex core" };
    let vals: Vec<usize> = census.valences.iter().map(|&(_, k)| k).collect();
    let description = format!(
        "{kind_lbl} {title}: V={}, E={}, F={}, valences {{{}}}, face sizes {{{}}}.",
        census.vertices, census.edges, census.faces, fmt_sorted(vals), fmt_sorted(census.face_sides.clone()),
    );
    for pair in &mut cycle_chords {
        if pair.0 > pair.1 { std::mem::swap(&mut pair.0, &mut pair.1); }
    }
    cycle_chords.sort_unstable();
    Some(EnumCore {
        conn: CoreConnection {
            id: 0, title, description, chords, inner_verts, orbit_id: 0, orbit_size: 1,
            is_regular_square, map, census,
        },
        spokes,
        cycle_chords,
    })
}

fn search_chords(
    ctx: &mut SearchCtx<'_>, spokes: &[usize], candidates: &[SectorChord], target: usize,
    start: usize, total_deficit: usize, chosen: &mut Vec<SectorChord>,
) {
    if ctx.truncated { return; }
    let remaining = target - chosen.len();
    if total_deficit > 2 * remaining { return; }
    if remaining == 0 {
        let cyc: Vec<(usize, usize)> = chosen.iter().map(|c| (c.ci, c.cj)).collect();
        let (map, chords) = build_core_map(ctx.boundary_map, ctx.walk, spokes, &cyc, ctx.first_label);
        if let Some(core) = evaluate_core(map, chords, spokes.to_vec(), cyc) {
            if ctx.out.len() >= ctx.cap { ctx.truncated = true; } else { ctx.out.push(core); }
        }
        return;
    }
    if start + remaining > candidates.len() { return; }
    for idx in start..=candidates.len() - remaining {
        let c = candidates[idx];
        if chosen.iter().all(|p| p.sector != c.sector || !chords_cross((p.a, p.b), (c.a, c.b))) {
            let (d0, d1) = (usize::from(ctx.deficits[c.ci] > 0), usize::from(ctx.deficits[c.cj] > 0));
            ctx.deficits[c.ci] -= d0;
            ctx.deficits[c.cj] -= d1;
            chosen.push(c);
            search_chords(ctx, spokes, candidates, target, idx + 1, total_deficit - d0 - d1, chosen);
            chosen.pop();
            ctx.deficits[c.ci] += d0;
            ctx.deficits[c.cj] += d1;
            if ctx.truncated { return; }
        }
    }
}

fn search_v1_spokes(
    ctx: &mut SearchCtx<'_>, eligible: &[usize], complete_only: bool, counts: (usize, usize),
    start: usize, total_deficit: usize, spokes: &mut Vec<usize>,
) {
    let (d_target, k_chords) = counts;
    if ctx.truncated { return; }
    let s_rem = d_target - spokes.len();
    if total_deficit > s_rem + 2 * k_chords { return; }
    if s_rem == 0 {
        let n = ctx.walk.vertices.len();
        let mut cands = Vec::new();
        for r in 0..spokes.len() {
            let (s0, s1) = (spokes[r], spokes[(r + 1) % spokes.len()]);
            let len = (s1 + n - s0) % n;
            for a in 0..len {
                for b in (a + 2)..=len {
                    let (ci, cj) = ((s0 + a) % n, (s0 + b) % n);
                    if complete_only
                        && (ctx.boundary_map.verts[ctx.walk.vertices[ci]].kind != VertexKind::Complete
                            || ctx.boundary_map.verts[ctx.walk.vertices[cj]].kind != VertexKind::Complete)
                    {
                        continue;
                    }
                    cands.push(SectorChord { sector: r, a, b, ci, cj });
                }
            }
        }
        cands.sort_by_key(|c| (2 - usize::from(ctx.deficits[c.ci] > 0) - usize::from(ctx.deficits[c.cj] > 0), c.sector, c.a, c.b));
        let mut chosen = Vec::with_capacity(k_chords);
        search_chords(ctx, spokes, &cands, k_chords, 0, total_deficit, &mut chosen);
        return;
    }
    if start + s_rem > eligible.len() { return; }
    for idx in start..=eligible.len() - s_rem {
        let si = eligible[idx];
        let dec = usize::from(ctx.deficits[si] > 0);
        ctx.deficits[si] -= dec;
        spokes.push(si);
        search_v1_spokes(ctx, eligible, complete_only, counts, idx + 1, total_deficit - dec, spokes);
        spokes.pop();
        ctx.deficits[si] += dec;
        if ctx.truncated { return; }
    }
}

fn cavity_automorphisms(boundary_map: &PlanarMap, walk: &FaceWalk) -> Vec<Vec<usize>> {
    let m = walk.vertices.len();
    let sig: Vec<(VertexKind, usize)> = walk.vertices.iter()
        .map(|&v| (boundary_map.verts[v].kind, boundary_map.verts[v].rot.len().saturating_sub(2)))
        .collect();
    let mut auts = Vec::new();
    for s in 0..m {
        let rot: Vec<usize> = (0..m).map(|i| (i + s) % m).collect();
        if (0..m).all(|i| sig[i] == sig[rot[i]]) { auts.push(rot); }
    }
    for s in 0..m {
        let refl: Vec<usize> = (0..m).map(|i| (s + m - i) % m).collect();
        if (0..m).all(|i| sig[i] == sig[refl[i]]) { auts.push(refl); }
    }
    auts
}

type OrbitKey = (Vec<usize>, Vec<(usize, usize)>);

fn canonical_orbit_key(spokes: &[usize], chords: &[(usize, usize)], auts: &[Vec<usize>]) -> OrbitKey {
    let mut best: Option<OrbitKey> = None;
    for pi in auts {
        let mut ms: Vec<usize> = spokes.iter().map(|&i| pi[i]).collect();
        ms.sort_unstable();
        let mut mc: Vec<(usize, usize)> = chords.iter()
            .map(|&(a, b)| { let (x, y) = (pi[a], pi[b]); if x < y { (x, y) } else { (y, x) } }).collect();
        mc.sort_unstable();
        let key = (ms, mc);
        if best.as_ref().is_none_or(|b| key < *b) { best = Some(key); }
    }
    best.unwrap_or_else(|| (spokes.to_vec(), chords.to_vec()))
}

fn test_slide_connected_to_regular(
    boundary_map: &PlanarMap, cores: &[CoreConnection], first_label: u32, inner_edges: usize, regular_count: usize,
) -> Option<bool> {
    if regular_count == 0 || cores.len() > 32 { return None; }
    let label_end = first_label + inner_edges as u32;
    let is_inner = |l: u32| l >= first_label && l < label_end;
    let outer_labels = boundary_map.faces().into_iter().next()?.labels;
    let mut lookup: HashMap<(usize, Vec<(usize, usize)>), usize> = HashMap::with_capacity(cores.len());
    let mut visited = vec![false; cores.len()];
    let mut queue = VecDeque::new();
    for (i, c) in cores.iter().enumerate() {
        lookup.insert((c.inner_verts, c.chords.clone()), i);
        if c.is_regular_square { visited[i] = true; queue.push_back(i); }
    }
    while let Some(idx) = queue.pop_front() {
        let (m, iv) = (&cores[idx].map, cores[idx].inner_verts);
        let in_edges: Vec<(u32, [usize; 2])> =
            m.edges.iter().filter(|e| e.alive && is_inner(e.label)).map(|e| (e.label, e.ends)).collect();
        for &(target, ends) in &in_edges {
            for &from in &ends {
                if m.verts[from].kind != VertexKind::Complete { continue; }
                for &cd in &m.verts[from].rot {
                    let control = m.label_of(cd);
                    if control == target { continue; }
                    for side in [Some(Side::Left), Some(Side::Right)] {
                        let mut next = m.clone();
                        if next.slide(target, control, from, side).is_err()
                            || next.euler() != 2
                            || next.faces().first().is_none_or(|f| f.labels != outer_labels)
                        {
                            continue;
                        }
                        let mut pairs: Vec<(usize, usize)> = next
                            .edges
                            .iter()
                            .filter(|e| e.alive && is_inner(e.label))
                            .map(|e| (e.ends[0].min(e.ends[1]), e.ends[0].max(e.ends[1])))
                            .collect();
                        if pairs.iter().any(|&(u, v)| u == v) { continue; }
                        pairs.sort_unstable();
                        if pairs.windows(2).any(|w| w[0] == w[1]) { continue; }
                        if let Some(&nxt) = lookup.get(&(iv, pairs)) && !visited[nxt] {
                            visited[nxt] = true;
                            queue.push_back(nxt);
                        }
                    }
                }
            }
        }
    }
    Some(visited.into_iter().all(|v| v))
}

fn build_provenance(en: &CoreEnumeration, cap: usize) -> String {
    let scope = if en.boundary_vertices_fixed { "complete cavity vertices only" } else { "all cavity vertices" };
    let bound = if en.inner_edges <= 2 {
        "By the handshaking lemma, any interior vertex of valence >= 3 needs at least 3 interior edges (or a loop, creating a 1-sided face), so V_inner = 0 is forced.".to_string()
    } else {
        format!("[COMPUTATIONAL EVIDENCE] Search bounded within V_inner <= {}.", en.max_inner_verts)
    };
    let cap_note = if en.truncated { format!(" Truncated at safety cap of {cap} cores.") } else { String::new() };
    let slide_note = match en.slide_connected_to_regular {
        Some(true) => " Slide-move BFS verified every enumerated core is reachable from a regular square core.",
        Some(false) => " Slide-move BFS found at least one enumerated core not reachable from a regular square core.",
        None => "",
    };
    format!(
        "Author decisions (2026-11-22): fixed boundary & edge count, lattice cores (valence >= 3, sides >= 3), generalized to arbitrary lattice cavities. {}-sided cavity with {} interior edge(s) ({scope}), enforcing euler() == 2. {bound} Found {} labelled lattice core(s) in {} symmetry orbit(s) (cavity automorphism order {}), of which {} is/are regular square.{cap_note}{slide_note}",
        en.cavity_sides, en.inner_edges, en.cores.len(), en.symmetry_orbit_count, en.cavity_symmetry_order, en.regular_count,
    )
}

/// Enumerate valid planar lattice cores inside `cavity_cycle` of `boundary_map` with `inner_edges`
/// interior edges and up to `max_inner_verts` interior vertices (currently supports `0` or `1`).
/// Passing `max_cores == 0` disables the truncation cap.
pub fn enumerate_cavity_cores(
    boundary_map: &PlanarMap, cavity_cycle: &[usize], inner_edges: usize, max_inner_verts: usize,
    complete_only: bool, first_label: u32, max_cores: usize,
) -> Result<CoreEnumeration, String> {
    if max_inner_verts > 1 {
        return Err(format!("max_inner_verts = {max_inner_verts} is not supported yet (currently supports 0 or 1)"));
    }
    if boundary_map.euler() != 2 {
        return Err(format!("boundary map is not one connected planar piece (V - E + F = {})", boundary_map.euler()));
    }
    let walk = find_cavity_face(boundary_map, cavity_cycle)?;
    let n = walk.vertices.len();
    for k in 0..inner_edges {
        let label = first_label + k as u32;
        if boundary_map.edges.iter().any(|e| e.alive && e.label == label) {
            return Err(format!("interior edge label {label} is already used in the boundary map"));
        }
    }

    let cap = if max_cores == 0 { usize::MAX } else { max_cores };
    let base_deficits: Vec<usize> = walk.vertices.iter().map(|&v| {
        if boundary_map.verts[v].kind == VertexKind::Complete { 3usize.saturating_sub(boundary_map.verts[v].rot.len()) } else { 0 }
    }).collect();
    let total_deficit: usize = base_deficits.iter().sum();

    let mut v0_cands = Vec::new();
    for i in 0..n {
        for j in (i + 2)..n {
            if (i == 0 && j == n - 1) || (complete_only
                && (boundary_map.verts[walk.vertices[i]].kind != VertexKind::Complete
                    || boundary_map.verts[walk.vertices[j]].kind != VertexKind::Complete)) {
                continue;
            }
            v0_cands.push(SectorChord { sector: 0, a: i, b: j, ci: i, cj: j });
        }
    }
    v0_cands.sort_by_key(|c| (2 - usize::from(base_deficits[c.ci] > 0) - usize::from(base_deficits[c.cj] > 0), c.ci, c.cj));

    let mut ctx0 = SearchCtx { boundary_map, walk: &walk, first_label, cap, deficits: base_deficits.clone(), out: Vec::new(), truncated: false };
    let mut chosen = Vec::with_capacity(inner_edges);
    search_chords(&mut ctx0, &[], &v0_cands, inner_edges, 0, total_deficit, &mut chosen);

    let mut ctx1 = SearchCtx { boundary_map, walk: &walk, first_label, cap, deficits: base_deficits.clone(), out: Vec::new(), truncated: false };
    if max_inner_verts >= 1 && inner_edges >= 3 {
        let eligible: Vec<usize> = (0..n).filter(|&i| !complete_only || boundary_map.verts[walk.vertices[i]].kind == VertexKind::Complete).collect();
        let d_max = eligible.len().min(inner_edges);
        let mut spokes = Vec::with_capacity(d_max);
        for d in (3..=d_max).rev() {
            ctx1.deficits.clone_from(&base_deficits);
            search_v1_spokes(&mut ctx1, &eligible, complete_only, (d, inner_edges - d), 0, total_deficit, &mut spokes);
            if ctx1.truncated { break; }
        }
    }

    let sort_enum = |xs: &mut Vec<EnumCore>| {
        xs.sort_by(|a, b| (!a.conn.is_regular_square).cmp(&(!b.conn.is_regular_square))
            .then_with(|| a.conn.inner_verts.cmp(&b.conn.inner_verts))
            .then_with(|| b.spokes.len().cmp(&a.spokes.len()))
            .then_with(|| a.conn.chords.cmp(&b.conn.chords)));
    };
    sort_enum(&mut ctx0.out);
    sort_enum(&mut ctx1.out);

    let truncated = ctx0.truncated || ctx1.truncated || ctx0.out.len().saturating_add(ctx1.out.len()) > cap;
    let mut combined: Vec<EnumCore> = if !truncated {
        ctx0.out.into_iter().chain(ctx1.out).collect()
    } else {
        let take_v0 = ctx0.out.len().min((cap / 2).max(cap.saturating_sub(ctx1.out.len())));
        let take_v1 = ctx1.out.len().min(cap.saturating_sub(take_v0));
        ctx0.out.into_iter().take(take_v0).chain(ctx1.out.into_iter().take(take_v1)).collect()
    };
    sort_enum(&mut combined);

    let auts = cavity_automorphisms(boundary_map, &walk);
    let cavity_symmetry_order = auts.len();
    let mut orbit_keys = Vec::with_capacity(combined.len());
    let mut orbit_counts: BTreeMap<OrbitKey, usize> = BTreeMap::new();
    for ec in &combined {
        let key = canonical_orbit_key(&ec.spokes, &ec.cycle_chords, &auts);
        *orbit_counts.entry(key.clone()).or_insert(0) += 1;
        orbit_keys.push(key);
    }
    let mut orbit_ids: BTreeMap<OrbitKey, usize> = BTreeMap::new();
    let mut cores = Vec::with_capacity(combined.len());
    for (i, (mut ec, key)) in combined.into_iter().zip(orbit_keys).enumerate() {
        let next_id = orbit_ids.len();
        let oid = *orbit_ids.entry(key.clone()).or_insert(next_id);
        ec.conn.id = i;
        ec.conn.orbit_id = oid;
        ec.conn.orbit_size = orbit_counts[&key];
        cores.push(ec.conn);
    }
    let symmetry_orbit_count = orbit_ids.len();
    let regular_count = cores.iter().filter(|c| c.is_regular_square).count();
    let slide_connected_to_regular = test_slide_connected_to_regular(boundary_map, &cores, first_label, inner_edges, regular_count);
    let base_edges = boundary_map.edges.iter().filter(|e| e.alive).count();
    let mut en = CoreEnumeration {
        surrounding_name: format!("{n}-gon cavity ({base_edges} fixed edges)"),
        cavity_sides: n, inner_edges, total_edges: base_edges + inner_edges, max_inner_verts,
        boundary_vertices_fixed: complete_only, cores, regular_count, symmetry_orbit_count,
        cavity_symmetry_order, truncated, slide_connected_to_regular, provenance: String::new(),
    };
    en.provenance = build_provenance(&en, cap);
    Ok(en)
}

/// Enumerate all valid planar lattice cores obtained by placing `inner_edges` non-crossing chords
/// (`max_inner_verts = 0`) inside `cavity_cycle` of `boundary_map`.
pub fn enumerate_cavity_chord_cores(
    boundary_map: &PlanarMap, cavity_cycle: &[usize], inner_edges: usize, complete_only: bool, first_label: u32,
) -> Result<CoreEnumeration, String> {
    enumerate_cavity_cores(boundary_map, cavity_cycle, inner_edges, 0, complete_only, first_label, 0)
}

/// Fixed surrounding of Wang Fig. 8(a): the 12 outer edges `1..=10, 13, 14` around the 8-gon cavity
/// `(a, b, c, d, e, f, g, h)`. Vertex indices `0..13` match `wang::A..S` (with `S` marked `Removed`).
pub fn wang_surrounding() -> PlanarMap {
    use VertexKind::{Boundary, Complete, Removed};
    let verts = [
        ("a", [0.0, 0.0], Boundary), ("b", [1.0, 1.0], Boundary), ("c", [2.0, 1.0], Complete),
        ("d", [3.0, 2.0], Complete), ("e", [4.0, 2.0], Boundary), ("f", [4.0, 1.0], Boundary),
        ("g", [2.0, 0.0], Complete), ("h", [1.0, 0.0], Complete), ("s", [3.0, 1.0], Removed),
        ("leg 13", [2.0, 2.0], Boundary), ("leg 4", [3.0, 3.0], Boundary),
        ("leg 14", [1.0, -1.0], Boundary), ("leg 8", [2.0, -1.0], Boundary),
    ];
    let edges = [
        (1, A, B), (2, B, C), (3, C, D), (4, D, 10), (5, D, E), (6, E, F),
        (7, F, G), (8, G, 12), (9, G, H), (10, H, A), (13, C, 9), (14, H, 11),
    ];
    PlanarMap::from_drawing(&verts, &edges, 4).expect("Wang surrounding is a valid drawing")
}

/// All 5 labelled lattice-core connections (2 symmetry orbits under `D_2`, order 4) for Wang Fig. 8(a)'s
/// fixed surrounding (12 fixed edges, 8-gon cavity, 2 interior edges labelled 11 and 12, endpoints on `{c,d,g,h}`).
pub fn wang_core_connections() -> CoreEnumeration {
    let mut en = enumerate_cavity_cores(&wang_surrounding(), &[A, B, C, D, E, F, G, H], 2, 1, true, 11, 256)
        .expect("Wang 8-gon cavity enumeration succeeds");
    en.surrounding_name = String::from("Wang Fig. 8(a) 8-gon cavity (12 fixed edges)");
    en.cores.sort_by_key(|c| match c.chords.as_slice() {
        [(C, H), (D, G)] => 0, [(C, G), (D, G)] => 1, [(C, G), (C, H)] => 2,
        [(C, H), (D, H)] => 3, [(D, G), (D, H)] => 4, _ => 5,
    });
    for (id, core) in en.cores.iter_mut().enumerate() {
        core.id = id;
        match core.chords.as_slice() {
            [(C, H), (D, G)] => {
                core.title = String::from("c–h, d–g (regular squares)");
                core.description = String::from("Unique regular square core: V=4, E=14, F=3, all four inner vertices 4-valent and all three inner faces 4-sided. Matches the end of both Route 1 (edge slides) and Route 2 (merge & add).");
            }
            [(C, G), (D, G)] => {
                core.description = String::from("Chords c–g and d–g: V=4, E=14, F=3, valences {3,4,4,5}, faces {3,4,5}. Appears as Step 4 of Route 1 (one slide away from the regular square core).");
            }
            [(C, H), (D, H)] => { let _ = core.map.set_curve(12, Some([2.42, 0.65])); }
            [(D, G), (D, H)] => {
                let _ = core.map.set_curve(11, Some([2.70, 0.90]));
                let _ = core.map.set_curve(12, Some([2.25, 0.78]));
            }
            _ => {}
        }
    }
    en
}
