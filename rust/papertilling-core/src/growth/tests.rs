//! Growth tests (WP-G1). Every pinned number says where it comes from: hand derivation
//! (boxes of whole squares or cubes, polyhedron counts, incidence counts), or the
//! independent float Voronoi grower `_indep_e3` in `tests/test_growth.py`
//! ([COMPUTATIONAL EVIDENCE]; it shares no code with `e3_complex`).

use std::collections::HashMap;
use std::time::Instant;

use super::e3_complex::{E3Complex, P3};
use super::engine::{seed_cells, SeedAt};
use super::finite::FiniteComplex;
use super::{grow, CellComplex, Growth, GrowthCaps, GrowthError, SeedKind, Stopped};
use crate::honeycomb3d::{lattice_voronoi_cell, EuclideanKind};

const KINDS: [EuclideanKind; 4] = [
    EuclideanKind::Cubic,
    EuclideanKind::BitruncatedCubic,
    EuclideanKind::RhombicDodecahedral,
    EuclideanKind::AlternatedCubic,
];
/// A closed form n -> |S_n|.
type Form = fn(usize) -> usize;
/// (V, E, F) of each cell class.
type ClassCounts<'a> = &'a [(usize, usize, usize)];

const SEEDS: [SeedKind; 4] = [SeedKind::Vertex, SeedKind::Edge, SeedKind::Face, SeedKind::Cell];

fn caps(max_layers: u32) -> GrowthCaps {
    GrowthCaps { max_layers, max_cells: 50_000 }
}

fn counts<C>(g: &Growth<C>) -> Vec<usize> {
    g.stats.iter().map(|s| s.cells).collect()
}

fn e3(kind: EuclideanKind, seed: SeedKind, n: u32) -> Growth<P3> {
    let mut x = E3Complex::new(kind);
    let at = x.default_seed(seed).unwrap();
    let cells = seed_cells(&mut x, &at).unwrap();
    grow(&mut x, cells, caps(n))
}

// ---------------------------------------------------------------- planar tables (built here)

/// A patch of a regular tiling as polygons over integer vertex labels, with the rim marked
/// as every vertex whose table star is short of `q` tiles. Returns (complex, label of point).
fn planar(polys: Vec<Vec<(i32, i32)>>, q: usize) -> (FiniteComplex, HashMap<(i32, i32), u32>) {
    let mut id: HashMap<(i32, i32), u32> = HashMap::new();
    let mut deg: HashMap<u32, usize> = HashMap::new();
    let mut out = Vec::new();
    for p in polys {
        let mut row = Vec::new();
        for v in p {
            let next = id.len() as u32;
            let i = *id.entry(v).or_insert(next);
            *deg.entry(i).or_default() += 1;
            row.push(i);
        }
        out.push(row);
    }
    let rim: Vec<u32> = deg.iter().filter(|&(_, &d)| d < q).map(|(&v, _)| v).collect();
    (FiniteComplex::from_polygons(&out, &rim).unwrap(), id)
}

fn tri(a: i32, b: i32, s: i32) -> Vec<(i32, i32)> {
    // Triangular lattice in axial coordinates: up (s = 0) and down (s = 1) triangles.
    if s == 0 { vec![(a, b), (a + 1, b), (a, b + 1)] } else { vec![(a + 1, b), (a + 1, b + 1), (a, b + 1)] }
}

/// The 6 triangles at a triangular-lattice point, in cyclic order (consecutive ones share an edge).
fn hex_around(x: i32, y: i32) -> Vec<(i32, i32, i32)> {
    let star = [(x, y, 0), (x - 1, y, 0), (x, y - 1, 0), (x - 1, y, 1), (x, y - 1, 1), (x - 1, y - 1, 1)];
    let mut cyc = vec![star[0]];
    while cyc.len() < 6 {
        let last = tri(cyc[cyc.len() - 1].0, cyc[cyc.len() - 1].1, cyc[cyc.len() - 1].2);
        let next = star.iter().find(|t| !cyc.contains(t) && tri(t.0, t.1, t.2).iter().filter(|v| last.contains(v)).count() == 2);
        cyc.push(*next.unwrap());
    }
    cyc
}

fn table(name: &str, r: i32) -> (FiniteComplex, HashMap<(i32, i32), u32>) {
    let rng = || (-r..r).flat_map(|a| (-r..r).map(move |b| (a, b)));
    match name {
        "44" => planar(rng().map(|(a, b)| vec![(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)]).collect(), 4),
        "36" => planar(rng().flat_map(|(a, b)| [tri(a, b, 0), tri(a, b, 1)]).collect(), 6),
        _ => {
            // {6,3}: hexagons at triangular-lattice points; its vertices are triangles, labelled
            // by (3a + s, b) so they fit the (i32, i32) labels.
            let polys = rng().map(|(x, y)| hex_around(x, y).into_iter().map(|(a, b, s)| (3 * a + s, b)).collect()).collect();
            planar(polys, 3)
        }
    }
}

#[test]
fn planar_regular_tilings_match_closed_forms() {
    // §2.4 table: {4,4} hand (boxes of whole squares); {6,3} face and {3,6} vertex hand
    // (hexagonal numbers, hexagon of side n+1); the other forms fitted, verified to n = 8 by
    // `seed2d_n8.py` and `e2_check.py` (independent growers).
    let forms: [(&str, Form, Form, Form); 3] = [
        ("44", |n| (2 * n + 1).pow(2), |n| (2 * n + 2).pow(2), |n| (2 * n + 1) * (2 * n + 2)),
        ("36", |n| 6 * n * n + 6 * n + 1, |n| 6 * (n + 1).pow(2), |n| 2 * (3 * n + 1) * (n + 1)),
        ("63", |n| 3 * n * n + 3 * n + 1, |n| 3 * (n + 1).pow(2), |n| (3 * n + 2) * (n + 1)),
    ];
    let t = Instant::now();
    for (name, face, vertex, edge) in forms {
        let (mut x, id) = table(name, 13);
        // Seeds near the middle: the vertex (0,0) (all vertices of a regular tiling are alike),
        // an edge at it, and a tile at it.
        let v0 = id[&(0, 0)];
        let star = x.cells_at(v0).unwrap();
        let tile = star[0];
        let nb = x.vertices_of(tile).into_iter().find(|&w| w != v0 && x.edge_between(v0, w).is_ok()).unwrap();
        let e = x.edge_between(v0, nb).unwrap();
        for (seed, form) in [(SeedAt::Cell(tile), face), (SeedAt::Vertex(v0), vertex), (SeedAt::Edge(e), edge)] {
            let cells = seed_cells(&mut x, &seed).unwrap();
            let g = grow(&mut x, cells, caps(8));
            let want: Vec<usize> = (0..=8).map(form).collect();
            assert_eq!(counts(&g), want, "{{{name}}} {seed:?}");
            assert_eq!(g.stopped, None, "{{{name}}} {seed:?}: the table must not reach its rim by n = 8");
            for (k, s) in g.stats.iter().enumerate() {
                assert_eq!(s.euler, 1, "{{{name}}} {seed:?} layer {k}: a disk");
                // The boundary of a disk is one cycle.
                assert_eq!(s.boundary_vertices, s.boundary_edges, "{{{name}}} {seed:?} layer {k}");
            }
        }
    }
    println!("# time planar n=0..8, 3 tilings x 3 seeds: {:.2?}", t.elapsed());
}

#[test]
fn square_box_boundary_is_its_perimeter() {
    // Hand: a box of a x b squares has 2(a + b) boundary vertices and edges.
    let (mut x, id) = table("44", 8);
    let v0 = id[&(0, 0)];
    let seed = x.cells_at(v0).unwrap();
    let g = grow(&mut x, seed, caps(3));
    for (n, s) in g.stats.iter().enumerate() {
        let m = 2 * n + 2;
        assert_eq!((s.boundary_vertices, s.boundary_edges), (4 * m, 4 * m));
        assert_eq!(s.new_cells, if n == 0 { 4 } else { m * m - (m - 2) * (m - 2) });
    }
}

#[test]
fn rim_stops_before_a_partial_layer() {
    // Squares [-2,2)²: the cell seed (0,0)-(1,1) grows to the 3 x 3 box [-1,2]², whose
    // vertices at x or y = 2 lie on the rim, so layer 2 cannot be built whole.
    let (mut x, id) = table("44", 2);
    let tile = x.cells_at(id[&(0, 0)]).unwrap().into_iter().find(|&c| x.vertices_of(c).contains(&id[&(1, 1)])).unwrap();
    let g = grow(&mut x, vec![tile], caps(8));
    assert_eq!(counts(&g), vec![1, 9]);
    assert_eq!(g.stopped, Some(Stopped::Rim { layer: 2 }));
}

#[test]
fn closed_surfaces_end_with_closed() {
    // Hand: the cube's surface, face seed: 1, 5 (the face and its 4 edge neighbours; the
    // vertex rule adds nothing more at layer 1), 6; χ = 1, 1, 2.
    let cube = vec![vec![0, 1, 3, 2], vec![4, 6, 7, 5], vec![0, 4, 5, 1], vec![2, 3, 7, 6], vec![0, 2, 6, 4], vec![1, 5, 7, 3]];
    let mut x = FiniteComplex::from_polygons(&cube, &[]).unwrap();
    let g = grow(&mut x, vec![0], caps(10));
    assert_eq!(counts(&g), vec![1, 5, 6]);
    assert_eq!(g.stats.iter().map(|s| s.euler).collect::<Vec<_>>(), vec![1, 1, 2]);
    assert_eq!(g.stopped, Some(Stopped::Closed { layer: 2 }));
    assert_eq!((g.stats[2].boundary_vertices, g.stats[2].boundary_edges), (0, 0));
    // Hosohedron {2,3}: 3 digons on vertices 0, 1, cell i bounded by edges i and i+1 mod 3.
    // A vertex seed is the whole sphere (χ = 2 - 3 + 3 = 2); an edge seed is 2 digons (χ = 1).
    let json = r#"{"cell_vertices": [[0,1],[0,1],[0,1]], "cell_edges": [[0,1],[1,2],[2,0]],
                   "edge_vertices": [[0,1],[0,1],[0,1]]}"#;
    let mut h = FiniteComplex::from_json(json).unwrap();
    let seed = h.cells_at(0).unwrap();
    let g = grow(&mut h, seed, caps(5));
    assert_eq!((counts(&g), g.stats[0].euler, g.stopped.clone()), (vec![3], 2, Some(Stopped::Closed { layer: 0 })));
    let seed = h.seed_from_json(&serde_json::json!({"kind": "edge", "at": 0})).unwrap();
    let g = grow(&mut h, seed, caps(5));
    assert_eq!((counts(&g), g.stats[0].euler), (vec![2, 3], 1));
    let err = h.seed_from_json(&serde_json::json!({"kind": "edge", "at": [0, 1]})).unwrap_err();
    assert!(err.contains("names 3 edges"), "{err}");
}

#[test]
fn finite_polyhedra_count_euler_in_3d() {
    // A 2 x 2 x 2 block of unit cubes as polyhedra (vertex id = x + 3y + 9z). Corner cell seed:
    // layer 1 is all 8 cubes (they all share the centre vertex), then Closed. Hand: the 1-cube
    // has χ = 8 - 12 + 6 - 1 = 1, the 8-cube block 27 - 54 + 36 - 8 = 1.
    let vid = |x: u32, y: u32, z: u32| x + 3 * y + 9 * z;
    let mut cells = Vec::new();
    for (x, y, z) in (0u32..8).map(|i| (i & 1, (i >> 1) & 1, i >> 2)) {
        let c = |dx: u32, dy: u32, dz: u32| vid(x + dx, y + dy, z + dz);
        cells.push(vec![
            vec![c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)],
            vec![c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],
            vec![c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],
            vec![c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)],
            vec![c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)],
            vec![c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],
        ]);
    }
    let mut x = FiniteComplex::from_polyhedra(&cells, &[]).unwrap();
    let g = grow(&mut x, vec![0], caps(5));
    assert_eq!(counts(&g), vec![1, 8]);
    assert_eq!(g.stats.iter().map(|s| s.euler).collect::<Vec<_>>(), vec![1, 1]);
    assert_eq!(g.stopped, Some(Stopped::Closed { layer: 1 }));
    // The face between cubes 0 and 1 (x = 1) is in exactly those two cells.
    assert_eq!(x.cells_on_face(&[vid(1, 0, 0), vid(1, 1, 0), vid(1, 1, 1), vid(1, 0, 1)]).unwrap(), vec![0, 1]);
}

// ---------------------------------------------------------------- E³

#[test]
fn e3_cubic_matches_boxes() {
    // §2.4: each seed grows a box of whole cubes, a x b x c with a, b, c in {2n+1, 2n+2}.
    // Hand: χ = 1; its surface has 2(ab+bc+ca) + 2 vertices and 4(ab+bc+ca) edges.
    let t = Instant::now();
    for (seed, dims) in [
        (SeedKind::Vertex, [2, 2, 2]),
        (SeedKind::Edge, [1, 2, 2]),
        (SeedKind::Face, [1, 1, 2]),
        (SeedKind::Cell, [1, 1, 1]),
    ] {
        let g = e3(EuclideanKind::Cubic, seed, 6);
        assert_eq!(g.stopped, None);
        for (n, s) in g.stats.iter().enumerate() {
            let [a, b, c] = dims.map(|d| 2 * n + d);
            let p = a * b + b * c + c * a;
            assert_eq!((s.cells, s.euler), (a * b * c, 1), "{seed:?} n={n}");
            assert_eq!((s.boundary_vertices, s.boundary_edges), (2 * p + 2, 4 * p), "{seed:?} n={n}");
        }
    }
    println!("# time E3 cubic, 4 seeds, n=0..6: {:.2?}", t.elapsed());
}

#[test]
fn e3_cell_tables_are_the_right_polyhedra() {
    // Hand: cube (8, 12, 6), truncated octahedron (24, 36, 14), rhombic dodecahedron
    // (14, 24, 12), octahedron (6, 12, 8), tetrahedron (4, 6, 4); V - E + F = 2.
    let want: [(EuclideanKind, ClassCounts); 4] = [
        (EuclideanKind::Cubic, &[(8, 12, 6)]),
        (EuclideanKind::BitruncatedCubic, &[(24, 36, 14)]),
        (EuclideanKind::RhombicDodecahedral, &[(14, 24, 12)]),
        (EuclideanKind::AlternatedCubic, &[(6, 12, 8), (4, 6, 4), (4, 6, 4)]),
    ];
    for (kind, classes) in want {
        let x = E3Complex::new(kind);
        for (ci, &c) in classes.iter().enumerate() {
            assert_eq!(x.class_counts(ci), Some(c), "{kind:?} class {ci}");
        }
    }
    let mut kelvin = E3Complex::new(EuclideanKind::BitruncatedCubic).class_face_sizes(0);
    kelvin.sort_unstable();
    assert_eq!(kelvin, [vec![4; 6], vec![6; 8]].concat(), "6 squares and 8 hexagons");
    // Against the float Voronoi cells of `voronoi3d_dirichlet` (a separate construction).
    for (kind, lat) in [(EuclideanKind::Cubic, "SC"), (EuclideanKind::BitruncatedCubic, "BCC"), (EuclideanKind::RhombicDodecahedral, "FCC")] {
        let x = E3Complex::new(kind);
        let mut ours: Vec<[i64; 3]> = x.vertices_of([0, 0, 0]).iter().map(|v| v.map(i64::from)).collect();
        let mut theirs: Vec<[i64; 3]> =
            lattice_voronoi_cell(lat).unwrap().vertices.iter().map(|v| v.map(|t| (t * 2.0).round() as i64)).collect();
        ours.sort_unstable();
        theirs.sort_unstable();
        assert_eq!(ours, theirs, "{lat}");
    }
}

#[test]
fn e3_incidences_are_consistent() {
    // Hand incidence counts: cells at a vertex (cubic 8; Kelvin 4; rhombic 6 at the m-3m
    // vertex, 4 at the -43m vertex; octet 6 octahedra + 8 tetrahedra = 14); cells around an
    // edge (cubic 4, Kelvin 3, rhombic 3, octet 2 + 2 = 4); every face in exactly 2 cells.
    for kind in KINDS {
        let mut x = E3Complex::new(kind);
        let ring = match kind {
            EuclideanKind::Cubic | EuclideanKind::AlternatedCubic => 4,
            _ => 3,
        };
        for c in [[0, 0, 0], [2, 2, 2], [1, 1, 1], [1, 1, -1], [-1, 1, 3], [4, 0, 0]] {
            if x.class_of(c).is_none() {
                continue;
            }
            for v in x.vertices_of(c) {
                assert!(x.is_vertex(v), "{kind:?}: {v:?} of cell {c:?}");
                let at = x.cells_at(v).unwrap();
                assert!(at.contains(&c));
                let want = match kind {
                    EuclideanKind::Cubic => 8,
                    EuclideanKind::BitruncatedCubic => 4,
                    EuclideanKind::RhombicDodecahedral => if v.iter().all(|t| t % 2 != 0) { 4 } else { 6 },
                    EuclideanKind::AlternatedCubic => 14,
                };
                assert_eq!(at.len(), want, "{kind:?} vertex {v:?}");
            }
            for e in x.edges_of(c) {
                let cs = x.cells_on_edge(e).unwrap();
                assert!(cs.contains(&c) && cs.len() == ring, "{kind:?} edge {e:?}: {cs:?}");
            }
            for f in x.faces_of(c) {
                let cs = x.cells_on_face_key(f);
                assert!(cs.contains(&c) && cs.len() == 2, "{kind:?} face {f:?}: {cs:?}");
            }
        }
    }
}

#[test]
fn e3_other_kinds_match_hand_and_independent_counts() {
    // Layer 0 and the cell-seed layer 1 are hand counts: Kelvin S1 = 1 + 14 (every cell at a
    // vertex of a truncated octahedron shares a face with it); rhombic S1 = 1 + 12 face + 6
    // vertex-only neighbours; octahedron S1 = 1 + 6 + 12 octahedra + 8 + 24 tetrahedra = 51.
    // n = 0..4 agree with `_indep_e3` in tests/test_growth.py [COMPUTATIONAL EVIDENCE].
    let want: [(EuclideanKind, [[usize; 5]; 4]); 3] = [
        (EuclideanKind::BitruncatedCubic, [
            [4, 32, 108, 256, 500], [3, 28, 99, 240, 475], [2, 24, 90, 224, 450], [1, 15, 65, 175, 369],
        ]),
        (EuclideanKind::RhombicDodecahedral, [
            [6, 44, 146, 344, 670], [3, 34, 125, 308, 615], [2, 28, 110, 280, 570], [1, 19, 85, 231, 489],
        ]),
        (EuclideanKind::AlternatedCubic, [
            [14, 124, 426, 1016, 1990], [4, 76, 318, 824, 1690], [2, 58, 273, 740, 1555], [1, 51, 245, 679, 1449],
        ]),
    ];
    for (kind, rows) in want {
        for (seed, row) in SEEDS.iter().zip(rows) {
            let g = e3(kind, *seed, 4);
            assert_eq!(counts(&g), row.to_vec(), "{kind:?} {seed:?}");
            // χ = 1 is measured here, not proved (§2.1); the test pins the measurement.
            assert!(g.stats.iter().all(|s| s.euler == 1), "{kind:?} {seed:?}");
        }
    }
    // Hand: a tetrahedron seed at (1/2,1/2,1/2): 16 octahedra + 23 tetrahedra (inclusion-exclusion
    // over the 8 tetrahedra at each of its 4 corners: 32 - 12 + 4 - 1) = 39.
    let mut x = E3Complex::new(EuclideanKind::AlternatedCubic);
    assert_eq!(counts(&grow(&mut x, vec![[1, 1, 1]], caps(1))), vec![1, 39]);
    // Hand: the rhombic vertex (1/2,1/2,1/2) is in 4 cells.
    assert_eq!(x.cells_at([1, 1, 1]).map(|c| c.len()).ok(), None, "not an octet vertex");
    let mut rd = E3Complex::new(EuclideanKind::RhombicDodecahedral);
    assert_eq!(rd.cells_at([1, 1, 1]).unwrap().len(), 4);
}

#[test]
fn e3_layers_to_ten_timing_pilot() {
    // §6 pilot for WP-G2's MAX_E3_SHELLS: layers 8 and 10 for the non-cubic kinds.
    for kind in &KINDS[1..] {
        for seed in [SeedKind::Vertex, SeedKind::Cell] {
            let t = Instant::now();
            let g = e3(*kind, seed, 10);
            assert_eq!(g.stopped, None);
            println!(
                "# time {kind:?} {seed:?} layers 0..=10: {:.2?}, S8 = {}, S10 = {}",
                t.elapsed(),
                g.stats[8].cells,
                g.stats[10].cells
            );
        }
    }
}

// ---------------------------------------------------------------- engine edge cases

/// One cell 0 on vertices 0 and 1; the cells at vertex 1 form an infinite set.
struct Ideal;

impl CellComplex for Ideal {
    type Cell = u32;
    type Vertex = u32;
    type Key = u32;
    fn vertices_of(&self, _c: u32) -> Vec<u32> {
        vec![0, 1]
    }
    fn edges_of(&self, _c: u32) -> Vec<u32> {
        Vec::new()
    }
    fn faces_of(&self, _c: u32) -> Vec<u32> {
        Vec::new()
    }
    fn cells_at(&mut self, v: u32) -> Result<Vec<u32>, GrowthError> {
        if v == 1 { Err(GrowthError::Infinite { reason: "ideal vertex" }) } else { Ok(vec![0]) }
    }
    fn cells_on_edge(&mut self, _e: u32) -> Result<Vec<u32>, GrowthError> {
        Ok(Vec::new())
    }
    fn cells_on_face(&mut self, _f: &[u32]) -> Result<Vec<u32>, GrowthError> {
        Ok(Vec::new())
    }
}

#[test]
fn errors_caps_and_empty_seeds() {
    let g = grow(&mut Ideal, vec![0, 0], caps(3));
    assert_eq!(g.layers, vec![vec![0]], "the seed is deduplicated");
    assert_eq!(g.stats[0].boundary_vertices, 1, "the ideal vertex is on the boundary");
    assert_eq!(g.stopped, Some(Stopped::Infinite { layer: 1, reason: "ideal vertex" }));
    // At max_layers = 0 nothing beyond the seed is attempted, so no stop is reported.
    assert_eq!(grow(&mut Ideal, vec![0], caps(0)).stopped, None);
    let g = grow(&mut Ideal, Vec::new(), caps(3));
    assert!(g.layers.is_empty() && g.stats.is_empty() && g.stopped.is_none());
    // Cap (hand: cubic cell seed S1 = 27): cap 27 builds S1, cap 26 stops before it.
    let mut x = E3Complex::new(EuclideanKind::Cubic);
    let g = grow(&mut x, vec![[0, 0, 0]], GrowthCaps { max_layers: 1, max_cells: 27 });
    assert_eq!((counts(&g), g.stopped), (vec![1, 27], None));
    let g = grow(&mut x, vec![[0, 0, 0]], GrowthCaps { max_layers: 5, max_cells: 26 });
    assert_eq!(counts(&g), vec![1], "no partial layer is shown");
    assert_eq!(g.stopped, Some(Stopped::CellCap { layer: 1, partial: 27 }));
    let g = grow(&mut x, vec![[0, 0, 0], [2, 0, 0]], GrowthCaps { max_layers: 5, max_cells: 1 });
    assert_eq!((g.layers.len(), g.stopped), (0, Some(Stopped::CellCap { layer: 0, partial: 2 })));
    // Seeds refused with a reason.
    assert!(x.default_seed(SeedKind::Dirichlet).unwrap_err().contains("Dirichlet"));
    let (mut f, _) = table("44", 2);
    assert!(f.seed_from_json(&serde_json::json!({"kind": "dirichlet"})).unwrap_err().contains("coordinates"));
}

#[test]
fn json_forms_of_seed_and_stop() {
    // The wire form the Python mirror reproduces (frozen with mod.rs).
    assert_eq!(serde_json::to_string(&SeedKind::Dirichlet).unwrap(), "\"dirichlet\"");
    let s = serde_json::to_string(&Stopped::CellCap { layer: 3, partial: 61_000 }).unwrap();
    assert_eq!(s, r#"{"kind":"cell_cap","layer":3,"partial":61000}"#);
    let g = grow(&mut E3Complex::new(EuclideanKind::Cubic), vec![[0, 0, 0]], caps(0));
    let s = serde_json::to_string(&g).unwrap();
    assert_eq!(
        s,
        r#"{"layers":[[[0,0,0]]],"stats":[{"cells":1,"new_cells":1,"boundary_vertices":8,"boundary_edges":12,"euler":1}],"stopped":null}"#
    );
}
