//! Honeycomb growth wiring (WP-G2): E³ through `grow_honeycomb`, the face-shell rule,
//! paracompact refusals and the ball truncation, and the drawn complex.

use std::time::Instant;

use super::growth_build::{grow_honeycomb, GrowthRule, HoneycombKind, DRAW_ALL_FACES_MAX, MAX_GROWTH_CELLS, MAX_GROWTH_LAYERS};
use super::honeycombs_h3::build_hyperbolic_honeycomb;
use super::paracompact_ball::{paracompact_ball, paracompact_ball_with};
use super::{CellComplex3d, EuclideanKind};
use crate::growth::{GrowthCaps, SeedKind, Stopped};

const PARACOMPACT: [(u32, u32, u32); 11] = [
    (3, 3, 6), (4, 3, 6), (5, 3, 6), (3, 4, 4), (4, 4, 3), (4, 4, 4), (3, 6, 3), (6, 3, 3), (6, 3, 4), (6, 3, 5), (6, 3, 6),
];

fn caps(n: u32) -> GrowthCaps {
    GrowthCaps { max_layers: n, max_cells: MAX_GROWTH_CELLS }
}

fn totals(kind: HoneycombKind, seed: SeedKind, rule: GrowthRule, n: u32) -> Vec<usize> {
    grow_honeycomb(kind, seed, rule, caps(n)).unwrap().stats.iter().map(|s| s.cells).collect()
}

fn drawn_euler(cx: &CellComplex3d) -> i64 {
    cx.euler()
}

#[test]
fn e3_touching_rule_boxes() {
    // [hand] boxes of whole cubes (plan §2.4): cell (2n+1)³, vertex (2n+2)³.
    let cubic = HoneycombKind::Schlafli(4, 3, 4);
    let want: Vec<usize> = (0..5).map(|n| (2 * n + 1usize).pow(3)).collect();
    assert_eq!(totals(cubic, SeedKind::Cell, GrowthRule::Touching, 4), want);
    let want: Vec<usize> = (0..5).map(|n| (2 * n + 2usize).pow(3)).collect();
    assert_eq!(totals(HoneycombKind::Euclid(EuclideanKind::Cubic), SeedKind::Vertex, GrowthRule::Touching, 4), want);
    // Ten layers fit under the cap for every kind (largest S10 is 21 274 < 120 000, WP-G1 pilot).
    let t = Instant::now();
    for k in [EuclideanKind::Cubic, EuclideanKind::BitruncatedCubic, EuclideanKind::RhombicDodecahedral, EuclideanKind::AlternatedCubic] {
        let g = grow_honeycomb(HoneycombKind::Euclid(k), SeedKind::Vertex, GrowthRule::Touching, caps(MAX_GROWTH_LAYERS)).unwrap();
        assert_eq!(g.stats.len(), 11, "{k:?}");
        assert_eq!(g.stopped, None);
        assert!(g.stats.iter().all(|s| s.euler == 1), "{k:?}: a box of cells is a ball [measured]");
    }
    eprintln!("# time E3 4 kinds, vertex seed, 10 layers with drawing: {:.2}s", t.elapsed().as_secs_f64());
}

#[test]
fn face_shells_known_counts() {
    // [hand] cubic face shells: cubes at L1 distance <= n, the centred octahedral numbers
    // (2n+1)(2n²+2n+3)/3 = 1, 7, 25, 63.
    let want: Vec<usize> = (0..4).map(|n: usize| (2 * n + 1) * (2 * n * n + 2 * n + 3) / 3).collect();
    assert_eq!(totals(HoneycombKind::Schlafli(4, 3, 4), SeedKind::Cell, GrowthRule::FaceShells, 3), want);
    // [COMPUTATIONAL EVIDENCE, face_shells.py] new cells per shell {4,3,5} 1, 6, 30, 126 and
    // {5,3,4} 1, 12, 102, 812.
    let cum = |v: &[usize]| v.iter().scan(0, |a, x| { *a += x; Some(*a) }).collect::<Vec<_>>();
    assert_eq!(totals(HoneycombKind::Schlafli(4, 3, 5), SeedKind::Cell, GrowthRule::FaceShells, 3), cum(&[1, 6, 30, 126]));
    assert_eq!(totals(HoneycombKind::Schlafli(5, 3, 4), SeedKind::Cell, GrowthRule::FaceShells, 3), cum(&[1, 12, 102, 812]));
    // The old face-adjacency builder (an independent BFS kept for the app) agrees, shells 0..=2.
    for (p, q, r) in [(4, 3, 5), (5, 3, 4), (5, 3, 5), (3, 5, 3)] {
        let new = totals(HoneycombKind::Schlafli(p, q, r), SeedKind::Cell, GrowthRule::FaceShells, 2);
        let old: Vec<usize> = (0..=2).map(|s| build_hyperbolic_honeycomb(p, q, r, s, 1).unwrap().cells.len()).collect();
        assert_eq!(new, old, "{{{p},{q},{r}}}");
    }
    // S³ closes under the face rule too: the 600-cell from one cell ends with all 600.
    let g = grow_honeycomb(HoneycombKind::Schlafli(3, 3, 5), SeedKind::Cell, GrowthRule::FaceShells, caps(40)).unwrap();
    assert_eq!(g.stats.last().unwrap().cells, 600);
    assert!(matches!(g.stopped, Some(Stopped::Closed { .. })));
}

#[test]
fn paracompact_refused_at_layer_at_most_one() {
    // [PROVED] (plan §1 Q4) for all 11, under the touching rule.
    for (p, q, r) in PARACOMPACT {
        let cls = super::classify_schlafli(p, q, r);
        for seed in [SeedKind::Vertex, SeedKind::Edge, SeedKind::Face, SeedKind::Cell] {
            let g = grow_honeycomb(HoneycombKind::Schlafli(p, q, r), seed, GrowthRule::Touching, caps(3)).unwrap();
            let Some(Stopped::Infinite { layer, .. }) = g.stopped else { panic!("{{{p},{q},{r}}} {seed:?}: {:?}", g.stopped) };
            let want = if cls.has_infinite_euclidean_cells || seed == SeedKind::Vertex { 0 } else { 1 };
            assert_eq!(layer, want, "{{{p},{q},{r}}} {seed:?}");
            assert_eq!(g.stats.len(), want as usize);
        }
    }
}

#[test]
fn paracompact_ball_counts() {
    // [hand] {6,3,3} and {4,4,3}: the seed face is centred at the origin, so a tiny ball holds
    // that face and its 2 cells. (Larger radii are checked for monotonicity and stability only.)
    for (p, q, r) in [(6, 3, 3), (4, 4, 3)] {
        let b = paracompact_ball(p, q, r, 0.01).unwrap();
        assert_eq!((b.faces, b.cells), (1, 2), "{{{p},{q},{r}}}");
    }
    // [hand] {3,3,6}: the central tetrahedron is centred at the origin; its 4 faces are at the
    // inradius ρ, so a ball of radius ρ − ε holds no face and ρ + ε holds 4 faces in 5 cells.
    let real = super::coxeter_gram::CoxeterRealisation::new(3, 3, 6).unwrap();
    let rho = real.w[2][0].acosh();
    assert_eq!(paracompact_ball(3, 3, 6, rho - 1e-6).unwrap().faces, 0);
    let b = paracompact_ball(3, 3, 6, rho + 1e-6).unwrap();
    assert_eq!((b.faces, b.cells), (4, 5));
    let t = Instant::now();
    for (p, q, r) in PARACOMPACT {
        let mut last = 0;
        for radius in [0.5, 1.0, 1.5, 2.0] {
            let b = paracompact_ball(p, q, r, radius).unwrap();
            assert!(b.faces >= last, "{{{p},{q},{r}}} monotone in R");
            last = b.faces;
            assert!(b.reason.starts_with("[PROVED]"));
            assert_eq!(b.complex.faces.len(), b.faces);
        }
        // Neighbour discovery does not depend on how far past R the search runs (the old
        // patch-ring dependence behind χ = −163 is gone).
        let (a, b) = (paracompact_ball_with(p, q, r, 1.0, 1).unwrap(), paracompact_ball_with(p, q, r, 1.0, 2).unwrap());
        assert_eq!((a.faces, a.cells), (b.faces, b.cells), "{{{p},{q},{r}}}");
    }
    eprintln!("# time balls, 11 honeycombs x 4 radii + search check: {:.2}s", t.elapsed().as_secs_f64());
    assert!(paracompact_ball(4, 3, 5, 1.0).is_err());
}

#[test]
fn chi_minus_163_regression_6_3_3() {
    // The old χ = −163 case (plan §1 Q4) is {6,3,3} with shells = 2, patch_layers = 2 of the
    // face-adjacency builder (re-found 2026-11-20 by sweeping the Python fallback): its cell
    // count depends on the drawing rings (shells = 1 gives 8 cells with one ring, 20 with two),
    // because neighbours were discovered through the drawn patch. The old builder still shows
    // the dependence; the ball does not: its counts are the same for search depths R + D,
    // R + 2D and R + 3D at every radius tested (R = 2 with R + 3D passes the 20 000-face
    // exploration cap and is refused with a reason, so it is not in the list).
    let one = build_hyperbolic_honeycomb(6, 3, 3, 1, 1).unwrap().cells.len();
    let two = build_hyperbolic_honeycomb(6, 3, 3, 1, 2).unwrap().cells.len();
    assert_ne!(one, two, "the old builder's patch-ring dependence");
    for radius in [0.5, 1.0, 1.5] {
        let counts: Vec<(usize, usize)> =
            (1..=3).map(|s| paracompact_ball_with(6, 3, 3, radius, s).unwrap()).map(|b| (b.faces, b.cells)).collect();
        assert!(counts.windows(2).all(|w| w[0] == w[1]), "R = {radius}: {counts:?}");
    }
}

#[test]
fn drawn_complex_matches_readout() {
    // Below DRAW_ALL_FACES_MAX the drawn complex is S_K itself: V − E + F − C equals the readout χ.
    let g = grow_honeycomb(HoneycombKind::Schlafli(5, 3, 4), SeedKind::Cell, GrowthRule::Touching, caps(2)).unwrap();
    assert!(!g.boundary_only && g.complex.cells.len() == 1985);
    assert_eq!(drawn_euler(&g.complex), g.stats[2].euler);
    assert_eq!(g.complex.cell_shell.iter().filter(|&&k| k == 1).count(), 62);
    assert!(g.complex.points.iter().all(|x| super::norm3(x) < 1.0));
    // Above it only the boundary faces: for a ball (χ = 1 measured) that is a sphere, V − E + F = 2.
    let t = Instant::now();
    let g = grow_honeycomb(HoneycombKind::Schlafli(4, 3, 5), SeedKind::Vertex, GrowthRule::Touching, caps(2)).unwrap();
    eprintln!("# time {{4,3,5}} vertex S2 with boundary drawing: {:.2}s", t.elapsed().as_secs_f64());
    assert!(g.stats[2].cells > DRAW_ALL_FACES_MAX && g.boundary_only);
    assert!(g.complex.cells.is_empty());
    let (v, e, f, _) = g.complex.counts();
    assert_eq!(v as i64 - e as i64 + f as i64, 2);
    // S³ drawing: the whole 600-cell, stereographic, χ = 0.
    let g = grow_honeycomb(HoneycombKind::Schlafli(3, 3, 5), SeedKind::Vertex, GrowthRule::Touching, caps(10)).unwrap();
    assert_eq!(g.complex.counts(), (120, 720, 1200, 600));
    assert!(g.complex.points.iter().all(|x| x.iter().all(|c| c.is_finite())));
    // E³ drawing: the cubic cell seed S1 is a 3×3×3 box: 64 vertices, 144 edges, 108 faces.
    let g = grow_honeycomb(HoneycombKind::Euclid(EuclideanKind::Cubic), SeedKind::Cell, GrowthRule::Touching, caps(1)).unwrap();
    assert_eq!(g.complex.counts(), (64, 144, 108, 27));
    // Refusals keep their [PROVED] reason.
    assert!(grow_honeycomb(HoneycombKind::Schlafli(7, 3, 3), SeedKind::Vertex, GrowthRule::Touching, caps(1)).unwrap_err().contains("PROVED"));
    assert_eq!(HoneycombKind::parse("{4,3,5}").unwrap(), HoneycombKind::Schlafli(4, 3, 5));
    assert_eq!(HoneycombKind::parse("rhombic_dodecahedral").unwrap(), HoneycombKind::Euclid(EuclideanKind::RhombicDodecahedral));
}

#[test]
#[ignore = "pilot: about 10 s; run with --ignored"]
fn pilot_default_cap_with_drawing() {
    // Round-7 timing rule: every build at the default cap, drawing included, within a few seconds.
    // The slider's worst cases: MAX_GROWTH_LAYERS layers, stopped by the cap.
    let m = MAX_GROWTH_LAYERS;
    let cases = [
        ((4, 3, 5), SeedKind::Cell, 3, GrowthRule::Touching),
        ((5, 3, 4), SeedKind::Cell, 3, GrowthRule::Touching),
        ((3, 5, 3), SeedKind::Cell, 3, GrowthRule::Touching),
        ((5, 3, 5), SeedKind::Vertex, 2, GrowthRule::Touching),
        ((3, 5, 3), SeedKind::Vertex, 3, GrowthRule::Touching),
        ((5, 3, 5), SeedKind::Cell, m, GrowthRule::Touching),
        ((4, 3, 5), SeedKind::Vertex, m, GrowthRule::Touching),
        ((3, 5, 3), SeedKind::Vertex, m, GrowthRule::Touching),
        ((5, 3, 4), SeedKind::Cell, m, GrowthRule::FaceShells),
        ((5, 3, 5), SeedKind::Cell, m, GrowthRule::FaceShells),
    ];
    for (pqr, seed, n, rule) in cases {
        let t = Instant::now();
        let g = grow_honeycomb(HoneycombKind::Schlafli(pqr.0, pqr.1, pqr.2), seed, rule, caps(n)).unwrap();
        eprintln!(
            "# time {pqr:?} {seed:?} {rule:?} n={n} with drawing: {:.2}s totals {:?} stopped {:?} faces drawn {} min_gap {:?}",
            t.elapsed().as_secs_f64(),
            g.stats.iter().map(|s| s.cells).collect::<Vec<_>>(),
            g.stopped,
            g.complex.faces.len(),
            g.precision.and_then(|p| p.min_gap)
        );
    }
}
