//! Seed tests for `plane::inflation` under the touching rule (C1 WP-G3, plan §2.4): face, vertex and edge seeds on
//! the regular tilings {4,4}, {3,6}, {6,3}, the most symmetric vertex (default seed, Q3), kagome χ, and the §6 timing
//! pilot.
//!
//! Where the numbers come from. Every pinned count is either derived by hand in the comment next to it, or is one of
//! the four fitted forms of plan §2.4 that were checked for n = 0..8 by two growers independent of this repository
//! (`seed2d_n8.py` and `e2_check.py`, plain set vertex completion on hand-built incidence tables); those are pinned for
//! n = 0..8 only and marked [COMPUTATIONAL EVIDENCE].

use std::time::Instant;

use super::catalogue::parse_catalogue_str;
use super::inflation::{InflationResult, InflationRule, InflationSeed, PlaneTiling, inflate_with};
use super::test_fixtures::named;
use crate::types::Cell2D;

const N: usize = 8;

fn bdf(cell: &Cell2D, seed: InflationSeed, n_max: usize) -> InflationResult {
    inflate_with(cell, seed, &InflationRule::Bdf, n_max).unwrap()
}

/// The edge seed on the cell's first bond `(i, j, n)`: vertices `(i, 0)` and `(j, n)`.
fn first_edge(cell: &Cell2D) -> InflationSeed {
    let b = &cell.bonds[0];
    InflationSeed::Edge((b.i, [0, 0]), (b.j, [b.n1, b.n2]))
}

/// `|S_n|` for n = 0..=N under the touching rule, checking that FillHoles never fired (pure vertex completion) and
/// that every S_n is a disk (χ = 1, no pinch vertex).
fn touching_counts(cell: &Cell2D, seed: InflationSeed) -> Vec<usize> {
    let res = bdf(cell, seed, N);
    for s in &res.shells {
        assert_eq!(s.filled, 0, "{seed:?} n={}: FillHoles fired, so this is not the bare touching rule", s.n);
        assert_eq!((s.topology.chi, s.topology.pinch_vertices), (1, 0), "{seed:?} n={}", s.n);
    }
    res.shells.iter().map(|s| s.topology.faces).collect()
}

fn form(f: impl Fn(usize) -> usize) -> Vec<usize> {
    (0..=N).map(f).collect()
}

#[test]
fn seeds_square_closed_forms() {
    // {4,4} [PROVED by hand]: S_n is always a box of whole squares, and the touching rule turns an a x b box into the
    // (a+2) x (b+2) box (every square touching the boundary, corners included). Face seed 1x1 -> (2n+1)^2; vertex
    // seed 2x2 -> (2n+2)^2; edge seed 1x2 -> (2n+1)(2n+2).
    let cell = named("square").cell;
    assert_eq!(touching_counts(&cell, InflationSeed::Face(0)), form(|n| (2 * n + 1).pow(2)));
    assert_eq!(touching_counts(&cell, InflationSeed::VertexStar(0)), form(|n| (2 * n + 2).pow(2)));
    assert_eq!(touching_counts(&cell, first_edge(&cell)), form(|n| (2 * n + 1) * (2 * n + 2)));
}

#[test]
fn seeds_triangular_closed_forms() {
    // {3,6} vertex seed [PROVED by hand]: S_n is every triangle with a vertex at lattice distance <= n from v, the
    // triangles of the regular hexagon of side n + 1 about v, which holds 6 (n+1)^2 unit triangles.
    // Face seed 6n^2 + 6n + 1 and edge seed 2(3n+1)(n+1): fitted forms [COMPUTATIONAL EVIDENCE, n = 0..8, plan §2.4].
    let cell = named("triangular").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert!(t.face_sizes().iter().all(|&k| k == 3));
    assert_eq!(touching_counts(&cell, InflationSeed::VertexStar(0)), form(|n| 6 * (n + 1).pow(2)));
    for f in 0..t.orbit_count() {
        assert_eq!(touching_counts(&cell, InflationSeed::Face(f)), form(|n| 6 * n * n + 6 * n + 1), "face orbit {f}");
    }
    assert_eq!(touching_counts(&cell, first_edge(&cell)), form(|n| 2 * (3 * n + 1) * (n + 1)));
}

#[test]
fn seeds_honeycomb_closed_forms() {
    // {6,3}: hexagons are the points of a triangular lattice, and two hexagons touch (share a vertex) exactly when
    // they share an edge, i.e. are lattice neighbours. So S_n is the graph ball of radius n about the seed points.
    // Face seed [PROVED by hand]: the ball about one point is the hexagon of side n, 3n^2 + 3n + 1 points.
    // Vertex seed [PROVED by hand]: the ball about a unit triangle of points is the 120-degree hexagon with sides
    // n+1, n alternating; it is the triangle of side 3n+1 ((3n+2)(3n+3)/2 points) minus three corner triangles of
    // side n-1 (n(n+1)/2 points each), giving 3(n+1)^2.
    // Edge seed (3n+2)(n+1): fitted form [COMPUTATIONAL EVIDENCE, n = 0..8, plan §2.4].
    let cell = named("honeycomb").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert_eq!(t.face_sizes(), vec![6]);
    assert_eq!(touching_counts(&cell, InflationSeed::Face(0)), form(|n| 3 * n * n + 3 * n + 1));
    for s in 0..cell.basis.len() {
        assert_eq!(touching_counts(&cell, InflationSeed::VertexStar(s)), form(|n| 3 * (n + 1).pow(2)), "site {s}");
    }
    assert_eq!(touching_counts(&cell, first_edge(&cell)), form(|n| (3 * n + 2) * (n + 1)));
}

#[test]
fn edge_seed_is_the_two_faces_on_the_bond_and_refuses_non_edges() {
    // Hand: every bond of a regular tiling has two distinct faces on it.
    for key in ["square", "triangular", "honeycomb", "kagome"] {
        let cell = named(key).cell;
        let t = PlaneTiling::new(&cell).unwrap();
        for b in &cell.bonds {
            let seed = InflationSeed::Edge((b.i, [0, 0]), (b.j, [b.n1, b.n2]));
            assert_eq!(t.seed_faces(seed).unwrap().len(), 2, "{key} bond {b:?}");
            // The reversed pair names the same edge.
            let rev = InflationSeed::Edge((b.j, [b.n1, b.n2]), (b.i, [0, 0]));
            assert_eq!(t.seed_faces(rev).unwrap(), t.seed_faces(seed).unwrap(), "{key} reversed");
        }
    }
    // A square vertex and its diagonal neighbour share two faces' worth of corner but no bond: refused, with a reason.
    let t = PlaneTiling::new(&named("square").cell).unwrap();
    let err = t.seed_faces(InflationSeed::Edge((0, [0, 0]), (0, [1, 1]))).unwrap_err();
    assert!(err.contains("not joined by a bond"), "{err}");
    let err = t.seed_faces(InflationSeed::Edge((7, [0, 0]), (0, [1, 0]))).unwrap_err();
    assert!(err.contains("out of range"), "{err}");
}

#[test]
fn most_symmetric_vertex_orders() {
    // Hand: site-symmetry orders of the vertices (point group of the site modulo translations): square 4mm = 8,
    // triangular 6mm = 12, honeycomb 3m = 6 (both sites), kagome 2mm = 4 (all three sites, so site 0 wins the tie).
    for (key, order) in [("square", 8), ("triangular", 12), ("honeycomb", 6), ("kagome", 4)] {
        let t = PlaneTiling::new(&named(key).cell).unwrap();
        assert_eq!(t.most_symmetric_vertex(), (0, order), "{key}");
        assert_eq!(t.default_seed(), InflationSeed::VertexStar(0), "{key}");
    }
}

#[test]
fn kagome_touching_rule_layers_are_disks() {
    // Plan §2.4: kagome under the touching rule is checked for χ = 1 (and no pinch) at every layer; no count is
    // pinned until one is derived by hand.
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    let b = &cell.bonds[0];
    let seeds = [
        t.default_seed(),
        InflationSeed::Face(0),
        InflationSeed::Edge((b.i, [0, 0]), (b.j, [b.n1, b.n2])),
    ];
    for seed in seeds {
        let res = bdf(&cell, seed, N);
        for s in &res.shells {
            assert_eq!((s.topology.chi, s.topology.pinch_vertices), (1, 0), "{seed:?} n={}", s.n);
        }
        let counts: Vec<usize> = res.shells.iter().map(|s| s.topology.faces).collect();
        println!("kagome touching {seed:?}: {counts:?} (filled {:?})", res.shells.iter().map(|s| s.filled).collect::<Vec<_>>());
        assert!(counts.windows(2).all(|w| w[1] > w[0]));
    }
}

const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");

/// Plan §6 E² gallery pilot: cost of n = 10, 20, 40 under the touching rule on kagome, {4,4} and the p1 points
/// drawing with the most faces per cell. Decides the slider K and the face cap (WP-U2). Timing only.
#[test]
#[ignore = "timing pilot (plan §6); run with --ignored --nocapture"]
fn pilot_e2_layers_timing() {
    let cats = parse_catalogue_str(POINTS_JSON).unwrap();
    let p1 = cats["p1"]
        .drawings
        .iter()
        .map(|e| e.family.cell())
        .filter(|c| PlaneTiling::new(c).is_ok())
        .max_by_key(|c| PlaneTiling::new(c).map(|t| t.orbit_count()).unwrap_or(0))
        .expect("a p1 drawing");
    let cases = [("kagome", named("kagome").cell), ("square", named("square").cell), ("p1 max faces", p1)];
    for (name, cell) in cases {
        let seed = PlaneTiling::new(&cell).unwrap().default_seed();
        for n in [10, 20, 40] {
            let t0 = Instant::now();
            let res = bdf(&cell, seed, n);
            println!("# time E2 {name} {seed:?} n={n}: {:?} ({} faces)", t0.elapsed(), res.faces.len());
        }
    }
}
