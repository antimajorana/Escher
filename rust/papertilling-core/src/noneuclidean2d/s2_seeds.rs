//! Seeded layer growth on the spherical `{p, q}` (Platonic solids, hosohedra `{2,q}`, dihedra `{p,2}`) under the
//! touching rule (C1 WP-G3, plan §2.1): `S_{k+1}` = every face sharing a vertex with `S_k`. On S² the growth
//! closes: once `S_k` is the whole sphere it is reported as [`Stopped::Closed`] at layer `k`, where χ = 2.
//!
//! Edges are keyed by themselves, not by their vertex pair: in a hosohedron `{2,q}` the two poles bound all `q`
//! edges, and lune `k` is bounded by meridians `k` and `k + 1 (mod q)`.

use std::collections::{BTreeMap, BTreeSet};

use crate::growth::{LayerStats, SeedKind, Stopped};

/// Layer table of one seeded growth on a finite face list.
#[derive(Clone, Debug, PartialEq)]
pub struct S2Growth {
    /// Layer at which each face joins (`usize::MAX` if never, which cannot happen on a connected sphere).
    pub face_layer: Vec<usize>,
    /// Readout of `S_0 .. S_K`.
    pub stats: Vec<LayerStats>,
    /// `Closed { layer }` when `S_layer` is the whole sphere, else `None` (stopped by `max_layers`).
    pub stopped: Option<Stopped>,
}

/// Canonical edge keys of every face: meridian indices for the hosohedron (`p == 2`), else indices of sorted vertex
/// pairs.
fn face_edges(p: usize, faces: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if p == 2 {
        let q = faces.len();
        return (0..q).map(|k| vec![k, (k + 1) % q]).collect();
    }
    let mut ids: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    faces
        .iter()
        .map(|f| {
            (0..f.len())
                .map(|k| {
                    let (u, v) = (f[k], f[(k + 1) % f.len()]);
                    let n = ids.len();
                    *ids.entry((u.min(v), u.max(v))).or_insert(n)
                })
                .collect()
        })
        .collect()
}

/// Readout of the union of the faces in `s`: χ = V − E + F over its vertices, edge keys and faces; a boundary edge
/// lies in exactly one face of `s`, a boundary vertex on a boundary edge.
fn layer_stats(faces: &[Vec<usize>], edges: &[Vec<usize>], s: &BTreeSet<usize>, new_cells: usize, p: usize) -> LayerStats {
    let verts: BTreeSet<usize> = s.iter().flat_map(|&f| faces[f].iter().copied()).collect();
    let mut on: BTreeMap<usize, usize> = BTreeMap::new();
    for &f in s {
        for &e in &edges[f] {
            *on.entry(e).or_default() += 1;
        }
    }
    // Ends of edge key `e`: the poles for a hosohedron meridian, else the pair it was keyed from.
    let ends = |e: usize| -> [usize; 2] {
        if p == 2 {
            return [0, 1];
        }
        let (f, k) = edges.iter().enumerate().find_map(|(f, es)| es.iter().position(|&x| x == e).map(|k| (f, k))).unwrap();
        [faces[f][k], faces[f][(k + 1) % faces[f].len()]]
    };
    let bnd: Vec<usize> = on.iter().filter(|&(_, &c)| c == 1).map(|(&e, _)| e).collect();
    let bverts: BTreeSet<usize> = bnd.iter().flat_map(|&e| ends(e)).collect();
    LayerStats {
        cells: s.len(),
        new_cells,
        boundary_vertices: bverts.len(),
        boundary_edges: bnd.len(),
        euler: verts.len() as i64 - on.len() as i64 + s.len() as i64,
    }
}

/// Grow `seed` on the spherical tiling with these faces (vertex index cycles; `p == 2` marks a hosohedron, whose
/// faces are the digons `[0, 1]`) for at most `max_layers` layers. Seeds: `Cell`/`Face` = face 0; `Vertex` = the
/// faces at vertex 0; `Edge` = the faces on the first edge of face 0. `Dirichlet` is refused (E² only).
pub fn grow_s2(p: usize, faces: &[Vec<usize>], seed: SeedKind, max_layers: usize) -> Result<S2Growth, String> {
    if faces.is_empty() {
        return Err("no faces to grow on".into());
    }
    let edges = face_edges(p, faces);
    let at = |pred: &dyn Fn(usize) -> bool| -> BTreeSet<usize> { (0..faces.len()).filter(|&f| pred(f)).collect() };
    let mut s = match seed {
        SeedKind::Cell | SeedKind::Face => BTreeSet::from([0]),
        SeedKind::Vertex => at(&|f| faces[f].contains(&faces[0][0])),
        SeedKind::Edge => at(&|f| edges[f].contains(&edges[0][0])),
        SeedKind::Dirichlet => {
            return Err("the Dirichlet seed is defined for E² lattices only; use a vertex, edge or face seed".into());
        }
    };
    let mut face_layer = vec![usize::MAX; faces.len()];
    for &f in &s {
        face_layer[f] = 0;
    }
    let mut stats = vec![layer_stats(faces, &edges, &s, s.len(), p)];
    let mut stopped = None;
    for k in 0..=max_layers {
        if s.len() == faces.len() {
            stopped = Some(Stopped::Closed { layer: k as u32 });
            break;
        }
        if k == max_layers {
            break;
        }
        let verts: BTreeSet<usize> = s.iter().flat_map(|&f| faces[f].iter().copied()).collect();
        let grown = at(&|f| !s.contains(&f) && faces[f].iter().any(|v| verts.contains(v)));
        for &f in &grown {
            face_layer[f] = k + 1;
        }
        s.extend(grown.iter().copied());
        stats.push(layer_stats(faces, &edges, &s, grown.len(), p));
    }
    Ok(S2Growth { face_layer, stats, stopped })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noneuclidean2d::build_unified_pq;

    fn run(p: usize, q: usize, seed: SeedKind) -> S2Growth {
        let t = build_unified_pq(p, q, 1).unwrap();
        grow_s2(p, &t.faces, seed, 10).unwrap()
    }

    fn counts(g: &S2Growth) -> Vec<usize> {
        g.stats.iter().map(|s| s.cells).collect()
    }

    #[test]
    fn s2_closure_known_answers() {
        // Cube {4,3} face seed (hand): 1 face, then the 4 faces sharing a vertex with it (the 4 side faces; the
        // opposite face shares none), then all 6: closed at layer 2 with χ = 2.
        let cube = run(4, 3, SeedKind::Cell);
        assert_eq!(counts(&cube), vec![1, 5, 6]);
        assert_eq!(cube.stopped, Some(Stopped::Closed { layer: 2 }));
        assert_eq!(cube.stats.iter().map(|s| s.euler).collect::<Vec<_>>(), vec![1, 1, 2]);
        // Icosahedron {3,5} face seed: 1, 10, 19, 20 with χ 1, 1, 1, 2 [COMPUTATIONAL EVIDENCE, cox_grow.py; the
        // first step is hand: a triangle's 3 vertices each lie in 5 faces, 3 + 3*2 = 9 new faces].
        let ico = run(3, 5, SeedKind::Cell);
        assert_eq!(counts(&ico), vec![1, 10, 19, 20]);
        assert_eq!(ico.stats.iter().map(|s| s.euler).collect::<Vec<_>>(), vec![1, 1, 1, 2]);
        assert_eq!(ico.stopped, Some(Stopped::Closed { layer: 3 }));
        // Vertex and edge seeds [COMPUTATIONAL EVIDENCE, cox_grow.py]: tetrahedron 3, 4; octahedron 4, 8 and 2, 8;
        // cube 3, 6 and 2, 6; icosahedron 5, 15, 20 and 2, 12, 20; dodecahedron 3, 9, 12 and 2, 8, 12.
        for (p, q, seed, want) in [
            (3, 3, SeedKind::Vertex, vec![3, 4]),
            (3, 4, SeedKind::Vertex, vec![4, 8]),
            (3, 4, SeedKind::Edge, vec![2, 8]),
            (4, 3, SeedKind::Vertex, vec![3, 6]),
            (4, 3, SeedKind::Edge, vec![2, 6]),
            (3, 5, SeedKind::Vertex, vec![5, 15, 20]),
            (3, 5, SeedKind::Edge, vec![2, 12, 20]),
            (5, 3, SeedKind::Vertex, vec![3, 9, 12]),
            (5, 3, SeedKind::Edge, vec![2, 8, 12]),
        ] {
            let g = run(p, q, seed);
            assert_eq!(counts(&g), want, "{{{p},{q}}} {seed:?}");
            assert_eq!(g.stopped, Some(Stopped::Closed { layer: (want.len() - 1) as u32 }));
            assert_eq!(g.stats.last().unwrap().euler, 2);
            assert!(g.stats[..g.stats.len() - 1].iter().all(|s| s.euler == 1), "{{{p},{q}}} {seed:?}");
        }
    }

    #[test]
    fn hosohedra_and_dihedra() {
        // {2,q} (hand): both poles lie on every lune, so a vertex seed is already the whole sphere (closed at 0);
        // a lune touches every other lune at the poles, so a face seed closes at layer 1; the edge (meridian 0) seed
        // is the 2 lunes on it, then everything.
        for q in [2, 3, 6] {
            let v = run(2, q, SeedKind::Vertex);
            assert_eq!((counts(&v), v.stopped.clone()), (vec![q], Some(Stopped::Closed { layer: 0 })), "q={q}");
            assert_eq!(v.stats[0].euler, 2);
            let f = run(2, q, SeedKind::Cell);
            assert_eq!(counts(&f), vec![1, q], "q={q}");
            let e = run(2, q, SeedKind::Edge);
            assert_eq!(e.stats[0].cells, 2.min(q), "q={q}");
            // One lune is a disk: V - E + F = 2 - 2 + 1.
            assert_eq!(f.stats[0].euler, 1);
        }
        // {p,2} (hand): two p-gons on the same p vertices; a vertex or edge seed holds both (closed at 0); a face
        // seed is one hemisphere (χ = 1, all p edges and vertices on its boundary), then both.
        for p in [3, 5] {
            let f = run(p, 2, SeedKind::Cell);
            assert_eq!(counts(&f), vec![1, 2]);
            assert_eq!((f.stats[0].euler, f.stats[0].boundary_edges, f.stats[0].boundary_vertices), (1, p, p));
            assert_eq!(run(p, 2, SeedKind::Vertex).stopped, Some(Stopped::Closed { layer: 0 }));
            assert_eq!(run(p, 2, SeedKind::Edge).stopped, Some(Stopped::Closed { layer: 0 }));
        }
        assert!(grow_s2(4, &build_unified_pq(4, 3, 1).unwrap().faces, SeedKind::Dirichlet, 3).is_err());
    }
}

#[cfg(test)]
mod unified_tests {
    use crate::growth::SeedKind;
    use crate::noneuclidean2d::{SnubHandedness, WythoffOp, build_unified_pq, build_unified_pq_seeded, build_wythoff};

    #[test]
    fn unified_pq_reports_layers_and_stops() {
        // S²: the whole cube is drawn; the readout is the face-seed growth 1, 5, 6 (hand), closed at layer 2.
        let cube = build_unified_pq_seeded(4, 3, 5, SeedKind::Cell).unwrap();
        assert_eq!(cube.n_faces, 6);
        assert_eq!(cube.growth.layer_counts, vec![1, 5, 6]);
        assert_eq!(cube.growth.layer_euler, vec![1, 1, 2]);
        assert!(cube.growth.stopped.as_deref().unwrap().starts_with("closed at layer 2"));
        // Asked for fewer layers than closure needs: no stop.
        assert_eq!(build_unified_pq(4, 3, 1).unwrap().growth.stopped, None);
        // H² vertex seed {5,4}: S_0 = 4, S_1 = 24 tiles (cox_grow.py, and the hand layer-1 formula in
        // hyperbolic/seed_tests.rs); every requested layer is drawn, so no stop.
        let h = build_unified_pq_seeded(5, 4, 1, SeedKind::Vertex).unwrap();
        assert_eq!((h.n_faces, h.growth.layer_counts.clone(), h.growth.stopped.clone()), (24, vec![4, 24], None));
        // Layer 0 is the seed tile; layers beyond the drawing caps are reported, not silently repeated.
        assert_eq!(build_unified_pq(5, 4, 0).unwrap().n_faces, 1);
        let deep = build_unified_pq(5, 4, 9).unwrap();
        assert!(deep.growth.stopped.as_deref().is_some_and(|s| s.contains("omitted") || s.contains("truncated")));
        // E²: one-face growth only; other seeds are refused with the place they are built.
        assert_eq!(build_unified_pq(4, 4, 0).unwrap().n_faces, 1);
        let e2 = build_unified_pq_seeded(4, 4, 2, SeedKind::Vertex).unwrap_err();
        assert!(e2.contains("plane inflation"), "{e2}");
    }

    #[test]
    fn hyperbolic_wythoff_surfaces_its_stop() {
        // {7,3} tile-seed shells 1, 7, 21, 56, 147, .. (hand counts in hyperbolic/inflation_tests.rs grow by about
        // 2.6x per layer), so 20 layers cannot fit the 600-tile budget: the stop is reported.
        let deep = build_wythoff(7, 3, WythoffOp::Truncated, 20).unwrap();
        assert!(deep.stopped.as_deref().is_some_and(|s| s.contains("face budget")), "{:?}", deep.stopped);
        let shallow = build_wythoff(7, 3, WythoffOp::Truncated, 1).unwrap();
        assert_eq!(shallow.stopped, None);
        let zero = build_wythoff(7, 3, WythoffOp::Snub { chirality: SnubHandedness::Left }, 0).unwrap();
        assert!(zero.n_faces > 0 && zero.stopped.is_none());
        // S² and E² never stop.
        assert_eq!(build_wythoff(4, 3, WythoffOp::Truncated, 1).unwrap().stopped, None);
    }
}
