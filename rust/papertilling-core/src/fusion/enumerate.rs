//! Exhaustive enumeration of planar lattice-core connections inside a fixed surrounding cavity.
//!
//! Under the author's decisions:
//! * **A1** (fixed boundary + fixed edge count): the surrounding cavity and total number of edges are held
//!   fixed (finite set of configurations reachable by edge-preserving rewiring).
//! * **A2** (lattice graphs only): every `Complete` vertex has valence >= 3, every bounded face has >= 3
//!   sides, and the full map is one connected planar piece (`euler() == 2`).

use serde::{Deserialize, Serialize};

use super::map::{Census, FaceWalk, MapEdge, PlanarMap, VertexKind};
use super::wang::{A, B, C, D, E, F, G, H};

/// One valid lattice core inside a fixed surrounding cavity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreConnection {
    pub id: usize,
    pub title: String,
    pub description: String,
    /// Interior chords as canonical vertex-index pairs `(u, v)` with `u < v`.
    pub chords: Vec<(usize, usize)>,
    /// True when every `Complete` vertex has valence 4 and every bounded face has 4 sides.
    pub is_regular_square: bool,
    pub map: PlanarMap,
    pub census: Census,
}

/// Complete list of valid chord-built lattice cores for one fixed surrounding cavity and edge budget.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreEnumeration {
    pub surrounding_name: String,
    pub cavity_sides: usize,
    pub inner_edges: usize,
    pub total_edges: usize,
    pub boundary_vertices_fixed: bool,
    pub cores: Vec<CoreConnection>,
    pub regular_count: usize,
    pub provenance: String,
}

fn chords_cross(p: (usize, usize), q: (usize, usize)) -> bool {
    let (a, b) = p;
    let (c, d) = q;
    (a < c && c < b && b < d) || (c < a && a < d && d < b)
}

fn is_cyclic_match(a: &[usize], b: &[usize]) -> bool {
    let n = a.len();
    if n != b.len() || n == 0 {
        return false;
    }
    let fwd = (0..n).any(|s| (0..n).all(|k| a[(s + k) % n] == b[k]));
    let rev = (0..n).any(|s| (0..n).all(|k| a[(s + n - k) % n] == b[k]));
    fwd || rev
}

fn find_cavity_face(map: &PlanarMap, cavity_cycle: &[usize]) -> Result<FaceWalk, String> {
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
    let matches: Vec<FaceWalk> = map
        .bounded_faces()
        .into_iter()
        .filter(|f| is_cyclic_match(&f.vertices, cavity_cycle))
        .collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => Err(String::from("cavity cycle does not match any bounded face of the boundary map")),
        _ => Err(String::from("cavity cycle matches more than one bounded face")),
    }
}

fn collect_non_crossing(
    candidates: &[(usize, usize)],
    target: usize,
    start: usize,
    chosen: &mut Vec<(usize, usize)>,
    out: &mut Vec<Vec<(usize, usize)>>,
) {
    if chosen.len() == target {
        out.push(chosen.clone());
        return;
    }
    let remaining = target - chosen.len();
    if start + remaining > candidates.len() {
        return;
    }
    for idx in start..=candidates.len() - remaining {
        let cand = candidates[idx];
        if chosen.iter().all(|&prev| !chords_cross(prev, cand)) {
            chosen.push(cand);
            collect_non_crossing(candidates, target, idx + 1, chosen, out);
            chosen.pop();
        }
    }
}

/// Insert non-crossing chords (indexed into `walk.vertices`) inside the bounded cavity face `walk`
/// in exact planar counter-clockwise order at every cavity vertex.
fn build_chord_map(
    boundary_map: &PlanarMap,
    walk: &FaceWalk,
    cycle_chords: &[(usize, usize)],
    first_label: u32,
) -> (PlanarMap, Vec<(usize, usize)>) {
    let n = walk.vertices.len();
    // Order chords deterministically by canonical vertex-index pair (u < v) so edge labels are canonical.
    let mut ordered: Vec<((usize, usize), (usize, usize))> = cycle_chords
        .iter()
        .map(|&(ci, cj)| {
            let (u, v) = (walk.vertices[ci], walk.vertices[cj]);
            if u < v { ((u, v), (ci, cj)) } else { ((v, u), (cj, ci)) }
        })
        .collect();
    ordered.sort_unstable_by_key(|&(verts, _)| verts);

    let mut map = boundary_map.clone();
    // Per cavity-cycle slot `ci`, store `(ccw_order_key, dart)` to insert after `in_ci = walk.darts[ci-1] ^ 1`.
    let mut inserts: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];

    for (k, &((u, v), (ci, cj))) in ordered.iter().enumerate() {
        let e = map.edges.len();
        map.edges.push(MapEdge {
            label: first_label + k as u32,
            ends: [u, v],
            alive: true,
            ctrl: None,
            loop_dir_deg: 0.0,
        });
        // `walk` has the cavity interior on its right (CW). Going CCW from `in_ci` (toward `ci - 1`)
        // to `out_ci` (toward `ci + 1`) sweeps cavity vertices in increasing order of `(ci + n - cj) % n`.
        inserts[ci].push(((ci + n - cj) % n, 2 * e));
        inserts[cj].push(((cj + n - ci) % n, 2 * e + 1));
    }

    for (ci, mut list) in inserts.into_iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        list.sort_unstable_by_key(|&(key, _)| key);
        let v = walk.vertices[ci];
        let in_dart = walk.darts[(ci + n - 1) % n] ^ 1;
        let rot = &mut map.verts[v].rot;
        let pos = rot.iter().position(|&d| d == in_dart).expect("cavity incoming dart is at its vertex");
        for (offset, (_, dart)) in list.into_iter().enumerate() {
            rot.insert(pos + 1 + offset, dart);
        }
    }

    let chords = ordered.into_iter().map(|(verts, _)| verts).collect();
    (map, chords)
}

fn sorted_vals(census: &Census) -> Vec<usize> {
    let mut v: Vec<usize> = census.valences.iter().map(|&(_, k)| k).collect();
    v.sort_unstable();
    v
}

fn sorted_faces(census: &Census) -> Vec<usize> {
    let mut f = census.face_sides.clone();
    f.sort_unstable();
    f
}

fn fmt_list(xs: &[usize]) -> String {
    xs.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn chord_title(map: &PlanarMap, chords: &[(usize, usize)]) -> String {
    if chords.is_empty() {
        return String::from("no interior chords");
    }
    chords
        .iter()
        .map(|&(u, v)| format!("{}–{}", map.verts[u].name, map.verts[v].name))
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_provenance(cavity_sides: usize, inner_edges: usize, complete_only: bool, total_cores: usize, regular: usize) -> String {
    let scope = if complete_only {
        "complete (unfrozen) cavity vertices only"
    } else {
        "all cavity vertices (including boundary vertices)"
    };
    let handshaking = if inner_edges == 2 {
        " By the handshaking lemma, any interior vertex of valence >= 3 needs at least 3 interior edges (or a loop, creating a 1-sided face), so V_inner = 0 is forced and every lattice core with 2 interior edges consists strictly of 2 non-crossing chords."
    } else {
        ""
    };
    format!(
        "Exhaustive planar chord enumeration on a {cavity_sides}-sided cavity with {inner_edges} interior edges ({scope}), enforcing euler() == 2, complete-vertex valence >= 3, and bounded-face sides >= 3.{handshaking} Found {total_cores} lattice core(s), of which {regular} is/are regular square."
    )
}

/// Enumerate all valid planar lattice cores obtained by placing `inner_edges` non-crossing chords inside
/// `cavity_cycle` of `boundary_map`.
///
/// * `cavity_cycle` may be given in either cyclic orientation (CCW or CW); it must match a bounded face
///   of `boundary_map`.
/// * Boundary edges (`|a - b| == 1` or `n - 1`) and duplicate chords are excluded so no 2-sided face is formed.
/// * When `complete_only` is true, chord endpoints are restricted to `VertexKind::Complete` vertices
///   (matching `slide`'s rule that `Boundary` vertices are frozen).
pub fn enumerate_cavity_chord_cores(
    boundary_map: &PlanarMap,
    cavity_cycle: &[usize],
    inner_edges: usize,
    complete_only: bool,
    first_label: u32,
) -> Result<CoreEnumeration, String> {
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

    let mut candidates = Vec::new();
    for i in 0..n {
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if complete_only
                && (boundary_map.verts[walk.vertices[i]].kind != VertexKind::Complete
                    || boundary_map.verts[walk.vertices[j]].kind != VertexKind::Complete)
            {
                continue;
            }
            candidates.push((i, j));
        }
    }

    let mut chord_sets = Vec::new();
    let mut chosen = Vec::with_capacity(inner_edges);
    collect_non_crossing(&candidates, inner_edges, 0, &mut chosen, &mut chord_sets);

    let mut cores = Vec::new();
    for cycle_chords in chord_sets {
        let (map, chords) = build_chord_map(boundary_map, &walk, &cycle_chords, first_label);
        if map.euler() != 2 {
            continue;
        }
        let census = map.census();
        let valid_valences = census.valences.iter().all(|&(_, k)| k >= 3);
        let valid_faces = census.face_sides.iter().all(|&s| s >= 3);
        if !valid_valences || !valid_faces {
            continue;
        }
        let is_regular_square = !census.valences.is_empty()
            && census.valences.iter().all(|&(_, k)| k == 4)
            && !census.face_sides.is_empty()
            && census.face_sides.iter().all(|&s| s == 4);
        let title = chord_title(&map, &chords);
        let description = format!(
            "Chords {title}: V={}, E={}, F={}, valences {{{}}}, face sizes {{{}}}.",
            census.vertices,
            census.edges,
            census.faces,
            fmt_list(&sorted_vals(&census)),
            fmt_list(&sorted_faces(&census)),
        );
        cores.push(CoreConnection { id: 0, title, description, chords, is_regular_square, map, census });
    }

    cores.sort_by(|a, b| (!a.is_regular_square).cmp(&(!b.is_regular_square)).then_with(|| a.chords.cmp(&b.chords)));
    for (i, core) in cores.iter_mut().enumerate() {
        core.id = i;
    }
    let regular_count = cores.iter().filter(|c| c.is_regular_square).count();
    let base_edges = boundary_map.edges.iter().filter(|e| e.alive).count();
    let provenance = build_provenance(n, inner_edges, complete_only, cores.len(), regular_count);

    Ok(CoreEnumeration {
        surrounding_name: format!("{n}-gon cavity ({base_edges} fixed edges)"),
        cavity_sides: n,
        inner_edges,
        total_edges: base_edges + inner_edges,
        boundary_vertices_fixed: complete_only,
        cores,
        regular_count,
        provenance,
    })
}

/// Fixed surrounding of Wang Fig. 8(a): the 12 outer edges `1..=10, 13, 14` around the 8-gon cavity
/// `(a, b, c, d, e, f, g, h)`. Vertex indices `0..13` match `wang::A..S` (with `S` marked `Removed`).
pub fn wang_surrounding() -> PlanarMap {
    use VertexKind::{Boundary, Complete, Removed};
    let verts = [
        ("a", [0.0, 0.0], Boundary),
        ("b", [1.0, 1.0], Boundary),
        ("c", [2.0, 1.0], Complete),
        ("d", [3.0, 2.0], Complete),
        ("e", [4.0, 2.0], Boundary),
        ("f", [4.0, 1.0], Boundary),
        ("g", [2.0, 0.0], Complete),
        ("h", [1.0, 0.0], Complete),
        ("s", [3.0, 1.0], Removed),
        ("leg 13", [2.0, 2.0], Boundary),
        ("leg 4", [3.0, 3.0], Boundary),
        ("leg 14", [1.0, -1.0], Boundary),
        ("leg 8", [2.0, -1.0], Boundary),
    ];
    let edges = [
        (1, A, B), (2, B, C), (3, C, D), (4, D, 10), (5, D, E), (6, E, F),
        (7, F, G), (8, G, 12), (9, G, H), (10, H, A), (13, C, 9), (14, H, 11),
    ];
    PlanarMap::from_drawing(&verts, &edges, 4).expect("Wang surrounding is a valid drawing")
}

/// All 5 valid lattice-core connections for Wang Fig. 8(a)'s fixed surrounding (12 fixed edges,
/// 8-gon cavity, 2 interior edges labelled 11 and 12, endpoints on the complete vertices `{c, d, g, h}`).
pub fn wang_core_connections() -> CoreEnumeration {
    let surrounding = wang_surrounding();
    let cavity = [A, B, C, D, E, F, G, H];
    let mut en = enumerate_cavity_chord_cores(&surrounding, &cavity, 2, true, 11)
        .expect("Wang 8-gon cavity enumeration succeeds");
    en.surrounding_name = String::from("Wang Fig. 8(a) 8-gon cavity (12 fixed edges)");

    // Present regular core first, then Step 4 of Route 1 ((c,g),(d,g)), then the remaining 3 cores.
    en.cores.sort_by_key(|c| match c.chords.as_slice() {
        [(C, H), (D, G)] => 0,
        [(C, G), (D, G)] => 1,
        [(C, G), (C, H)] => 2,
        [(C, H), (D, H)] => 3,
        [(D, G), (D, H)] => 4,
        _ => 5,
    });

    for (id, core) in en.cores.iter_mut().enumerate() {
        core.id = id;
        // In the drawing, h=(1,0), c=(2,1), d=(3,2) are collinear, so curve chord d-h gently into the cavity
        // so that `drawn_order(v) == map_order(v)` holds at every vertex.
        match core.chords.as_slice() {
            [(C, H), (D, G)] => {
                core.title = String::from("c–h, d–g (regular squares)");
                core.description = String::from(
                    "Unique regular square core: V=4, E=14, F=3, all four inner vertices 4-valent and all three inner faces 4-sided. Matches the end of both Route 1 (edge slides) and Route 2 (merge & add).",
                );
            }
            [(C, G), (D, G)] => {
                core.description = String::from(
                    "Chords c–g and d–g: V=4, E=14, F=3, valences {3,4,4,5}, faces {3,4,5}. Appears as Step 4 of Route 1 (one slide away from the regular square core).",
                );
            }
            [(C, H), (D, H)] => {
                let _ = core.map.set_curve(12, Some([2.42, 0.65]));
            }
            [(D, G), (D, H)] => {
                let _ = core.map.set_curve(11, Some([2.70, 0.90]));
                let _ = core.map.set_curve(12, Some([2.25, 0.78]));
            }
            _ => {}
        }
    }
    en
}
