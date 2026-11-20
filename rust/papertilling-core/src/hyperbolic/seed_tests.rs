//! Seed tests for H² vertex completion (C1 WP-G3, plan §2.4 and §5 T2): tile, vertex and edge seeds, the exact
//! boundary counters started from those seeds, 0-based layers, refusals, and the §6 timing pilot.
//!
//! Sources of the pinned numbers:
//! * Hand: the seed letters (`boundary_counts::seed_letters` doc), the layer-1 formulas below, the {5,4} tile-seed
//!   shells 1, 11, 51, 201, 761 (`inflation_tests.rs`), and the Euclidean closed forms (`plane/inflation_seed_tests.rs`).
//! * [COMPUTATIONAL EVIDENCE] `cox_grow.py` (the C1 critique's independent script, kept outside the repository, 2026-11-19): an independent grower over the Coxeter
//!   group in root coordinates (tile = coset of <R0,R1>, vertex = coset of <R1,R2>, relative-rounded keys), run as
//!   `cox_grow.py p,q seed 3`; it shares no code with this crate. It also reported χ = 1 at every layer of the {5,4}
//!   vertex and edge seeds. The plan asked for a `FiniteComplex` run on a combinatorial {p,q} ball as the third
//!   method; that type belongs to WP-G1 and is an empty stub on this branch, so `cox_grow.py` stands in for it.

use std::collections::BTreeSet;
use std::time::Instant;

use super::boundary_counts::{
    boundary_word_counts_from, letter_counts_from, seed_letters, seed_word, vertex_completion_counts_seeded,
};
use super::inflation_seeds::grow_vertex_completion_seeded;
use super::{DrawCaps, HyperbolicStyle, HyperbolicTiling, build_hyperbolic_tiling_seeded, build_with};
use crate::growth::SeedKind;

const SEEDS: [SeedKind; 3] = [SeedKind::Cell, SeedKind::Vertex, SeedKind::Edge];

/// One `{p, q}` pair and its cumulative |S_0..S_3| per seed (cell, vertex, edge).
type CoxRow = ((usize, usize), [[u64; 4]; 3]);

/// Cumulative |S_0..S_3| from `cox_grow.py` [COMPUTATIONAL EVIDENCE], per seed (cell, vertex, edge).
const COX: [CoxRow; 6] = [
    ((5, 4), [[1, 11, 51, 201], [4, 24, 100, 384], [2, 16, 70, 272]]),
    ((7, 3), [[1, 8, 29, 85], [3, 15, 48, 135], [2, 12, 40, 114]]),
    ((4, 5), [[1, 13, 61, 241], [5, 30, 125, 480], [2, 18, 80, 312]]),
    ((3, 7), [[1, 16, 61, 181], [7, 35, 112, 315], [2, 20, 72, 210]]),
    ((6, 4), [[1, 13, 85, 505], [4, 32, 196, 1152], [2, 20, 126, 744]]),
    ((8, 3), [[1, 9, 41, 161], [3, 18, 75, 288], [2, 14, 60, 232]]),
];

fn cumulative(shells: &[u64]) -> Vec<u64> {
    shells.iter().scan(0, |acc, &a| {
        *acc += a;
        Some(*acc)
    }).collect()
}

fn geometric(p: usize, q: usize, seed: SeedKind, n: usize) -> Vec<u64> {
    let vc = grow_vertex_completion_seeded(p, q, seed, n, 1_000_000).unwrap();
    assert!(vc.stopped.is_none(), "{{{p},{q}}} {seed:?}: {:?}", vc.stopped);
    vc.shell_sizes.iter().map(|&a| a as u64).collect()
}

#[test]
fn geometric_equals_counters_equals_cox_grow() {
    for ((p, q), rows) in COX {
        for (seed, want) in SEEDS.into_iter().zip(rows) {
            let geo = geometric(p, q, seed, 3);
            let letters = vertex_completion_counts_seeded(p, q, seed, 3).unwrap();
            let (a0, word) = seed_word(p, q, seed).unwrap();
            let words = boundary_word_counts_from(p, q, a0, word, 3).unwrap();
            assert_eq!(geo, letters, "{{{p},{q}}} {seed:?}: geometric vs letter counter");
            assert_eq!(geo, words, "{{{p},{q}}} {seed:?}: geometric vs boundary word");
            assert_eq!(cumulative(&geo), want.to_vec(), "{{{p},{q}}} {seed:?}: vs cox_grow.py");
        }
    }
}

#[test]
fn tile_seed_keeps_its_shells() {
    // {5,4} tile seed |S_0..S_4| = 1, 11, 51, 201, 761 (hand counts in inflation_tests.rs; plan §2.4).
    assert_eq!(cumulative(&geometric(5, 4, SeedKind::Cell, 4)), vec![1, 11, 51, 201, 761]);
    assert_eq!(geometric(5, 4, SeedKind::Face, 4), geometric(5, 4, SeedKind::Cell, 4));
}

#[test]
fn layer_one_hand_formulas() {
    // Hand, from the seed letters and the step rule of boundary_counts.rs (a letter k with o = q - k outside tiles
    // contributes o - 1 new tiles when q > 3; for q = 3 every letter 1 contributes one tile):
    //   vertex seed, q > 3: a_1 = q (q - 3) + q (p - 3) (q - 2);   q = 3: a_1 = 3 (p - 3);
    //   edge seed,   q > 3: a_1 = 2 (q - 3) + 2 (p - 2) (q - 2);   q = 3: a_1 = 2 (p - 2).
    let (mut v, mut e) = (Vec::new(), Vec::new());
    for (p, q) in [(5, 4), (4, 5), (6, 4), (7, 3), (8, 3), (5, 5), (3, 7), (3, 8)] {
        let (vh, eh) = if q > 3 {
            (q * (q - 3) + q * (p - 3) * (q - 2), 2 * (q - 3) + 2 * (p - 2) * (q - 2))
        } else {
            (3 * (p - 3), 2 * (p - 2))
        };
        v.push((geometric(p, q, SeedKind::Vertex, 1)[1] as usize, vh));
        e.push((geometric(p, q, SeedKind::Edge, 1)[1] as usize, eh));
    }
    assert!(v.iter().all(|(a, b)| a == b), "vertex seed (got, hand): {v:?}");
    assert!(e.iter().all(|(a, b)| a == b), "edge seed (got, hand): {e:?}");
}

#[test]
fn counters_reproduce_euclidean_closed_forms() {
    // The letter recursion read on the Euclidean tilings, against the closed forms of plan §2.4 (n = 0..8):
    // {4,4} all three and {3,6} vertex and {6,3} tile and vertex are hand-derived (plane/inflation_seed_tests.rs);
    // the other four are fits checked n = 0..8 by independent growers [COMPUTATIONAL EVIDENCE].
    type F = fn(u64) -> u64;
    let forms: [((usize, usize), [F; 3]); 3] = [
        ((4, 4), [|n| (2 * n + 1).pow(2), |n| (2 * n + 2).pow(2), |n| (2 * n + 1) * (2 * n + 2)]),
        ((3, 6), [|n| 6 * n * n + 6 * n + 1, |n| 6 * (n + 1).pow(2), |n| 2 * (3 * n + 1) * (n + 1)]),
        ((6, 3), [|n| 3 * n * n + 3 * n + 1, |n| 3 * (n + 1).pow(2), |n| (3 * n + 2) * (n + 1)]),
    ];
    for ((p, q), fs) in forms {
        for (seed, f) in SEEDS.into_iter().zip(fs) {
            let want: Vec<u64> = (0..=8).map(f).collect();
            let (a0, cnt) = seed_letters(p, q, seed).unwrap();
            assert_eq!(cumulative(&letter_counts_from(p, q, a0, cnt, 8).unwrap()), want, "{{{p},{q}}} {seed:?} letters");
            let (a0, word) = seed_word(p, q, seed).unwrap();
            assert_eq!(cumulative(&boundary_word_counts_from(p, q, a0, word, 8).unwrap()), want, "{{{p},{q}}} {seed:?} word");
        }
    }
}

/// χ = V - E + F of a drawn patch, from its face vertex lists.
fn euler(t: &HyperbolicTiling) -> i64 {
    let used: BTreeSet<usize> = t.faces.iter().flatten().copied().collect();
    used.len() as i64 - t.edges.len() as i64 + t.faces.len() as i64
}

#[test]
fn seeded_patches_are_zero_based_disks() {
    // Layer 0 is the seed set: 1 tile, q = 4 tiles at a vertex, 2 tiles on an edge (hand). Later layers are the
    // cox_grow totals of COX[0]; every drawn S_n is a disk (χ = 1; cox_grow measured the same).
    for (seed, totals) in SEEDS.into_iter().zip(COX[0].1) {
        for (n, &want) in totals.iter().enumerate() {
            let t = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::RegularPQ, n, 0, seed).unwrap();
            assert_eq!((t.faces.len() as u64, t.complete_through_shell, t.layers), (want, n, n), "{seed:?} n={n}");
            assert_eq!(euler(&t), 1, "{seed:?} n={n}");
            assert!(t.face_shell.iter().filter(|&&s| s == 0).count() as u64 == totals[0]);
        }
    }
    // The dual path takes seeds too: {4,5} drawn as the dual of {5,4}.
    let d = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::DualQP, 0, 0, SeedKind::Vertex).unwrap();
    assert_eq!(d.faces.len(), 5);
    // Layer 0 on the kagome and wedge paths (tile seed) builds without panicking.
    for (style, delta) in [(HyperbolicStyle::RectifiedKagome, 0), (HyperbolicStyle::RegularPQ, 1)] {
        let t = build_hyperbolic_tiling_seeded(7, 3, style, 0, delta, SeedKind::Cell).unwrap();
        assert!(!t.faces.is_empty() && t.layers == 0, "{style:?} delta {delta}");
    }
}

#[test]
fn seeds_are_refused_with_reasons() {
    let kag = build_hyperbolic_tiling_seeded(7, 3, HyperbolicStyle::RectifiedKagome, 2, 0, SeedKind::Vertex);
    assert!(kag.unwrap_err().contains("kagome cluster"));
    let wedge = build_hyperbolic_tiling_seeded(7, 3, HyperbolicStyle::RegularPQ, 2, 1, SeedKind::Edge);
    assert!(wedge.unwrap_err().contains("wedge core"));
    let dir = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::RegularPQ, 1, 0, SeedKind::Dirichlet);
    assert!(dir.unwrap_err().contains("E² lattices only"));
    // A cap smaller than the seed: refused, never an empty or partial seed.
    let caps = DrawCaps { max_faces: 3, max_centre_radius: 0.998 };
    let small = build_with(5, 4, HyperbolicStyle::RegularPQ, 1, 0, caps, SeedKind::Vertex);
    assert!(small.unwrap_err().contains("seed alone (4 tiles)"));
}

#[test]
fn budget_estimate_is_exact_even_for_deep_requests() {
    // {12,12} vertex seed: a_1 = q (q - 3) + q (p - 3)(q - 2) = 108 + 1080 = 1188 (hand formula above). Asking for
    // 64 layers overflows u64 in the letter counter far beyond any budget; the estimate must still be the exact
    // 1188 (not the fallback bound 12 * 12 * 11 = 1584), and the stop must say so.
    let vc = grow_vertex_completion_seeded(12, 12, SeedKind::Vertex, 64, 1000).unwrap();
    assert_eq!(vc.stopped.as_deref(), Some("face budget 1000: shell 1 (~1188 faces) not built"));
    assert_eq!(vc.shell_sizes, vec![12]);
}

/// Plan §6 H² pilot: build and draw cost on {5,4}, {7,3}, {8,8}, {12,12} up to 20 000 faces, vertex seed. Decides the
/// face cap that replaces the 1..=4 clamp. Timing only.
#[test]
#[ignore = "timing pilot (plan §6); run with --ignored --nocapture"]
fn pilot_h2_layers_timing() {
    for (p, q) in [(5, 4), (7, 3), (8, 8), (12, 12)] {
        for budget in [1_000, 5_000, 20_000] {
            let t0 = Instant::now();
            let vc = grow_vertex_completion_seeded(p, q, SeedKind::Vertex, 64, budget).unwrap();
            let grow = t0.elapsed();
            let t1 = Instant::now();
            let caps = DrawCaps { max_faces: budget, max_centre_radius: 0.998 };
            let drawn = build_with(p, q, HyperbolicStyle::RegularPQ, 64, 0, caps, SeedKind::Vertex).unwrap();
            println!(
                "# time H2 {{{p},{q}}} vertex budget {budget}: grow {grow:?} ({} faces, S_0..S_{}, stop {:?}), draw {:?} ({} faces, S_0..S_{} complete)",
                vc.faces.len(),
                vc.shell_sizes.len() - 1,
                vc.stopped,
                t1.elapsed(),
                drawn.faces.len(),
                drawn.complete_through_shell
            );
        }
    }
}

#[test]
fn seeded_drawings_are_centred_on_the_seed() {
    // The vertex seed is drawn with its vertex at the disk centre; the edge seed with its edge's midpoint there, so
    // the two ends of that edge are opposite points z, -z (hand: the boost fixes the geodesic through the seed).
    let v = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::RegularPQ, 0, 0, SeedKind::Vertex).unwrap();
    assert!(v.poincare_pts.iter().any(|z| z[0].hypot(z[1]) < 1e-12));
    let e = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::RegularPQ, 0, 0, SeedKind::Edge).unwrap();
    let opposite = |a: [f64; 2], b: [f64; 2]| (a[0] + b[0]).hypot(a[1] + b[1]) < 1e-12 && a[0].hypot(a[1]) > 1e-3;
    assert!(e.poincare_pts.iter().any(|&a| e.poincare_pts.iter().any(|&b| opposite(a, b))));
    // The tile seed keeps the tile centre at the disk centre (no centre point is a vertex).
    let c = build_hyperbolic_tiling_seeded(5, 4, HyperbolicStyle::RegularPQ, 0, 0, SeedKind::Cell).unwrap();
    assert!(c.poincare_pts.iter().all(|z| z[0].hypot(z[1]) > 1e-3));
}

/// Layers are no longer clamped, so every path is built at the deepest request (64): each must stop at its caps with
/// a reason, never panic or exceed them. Covers all 92 pairs p, q <= 12 on the regular, kagome and dual paths with
/// wedge delta 0, -1, +1 (tile seed), and the vertex and edge seeds on the regular path.
#[test]
fn deepest_request_stops_at_the_caps_on_every_path() {
    let t0 = Instant::now();
    let (mut built, mut refused) = (0, Vec::new());
    for p in 3..=12 {
        for q in 3..=12 {
            if (p - 2) * (q - 2) <= 4 {
                continue;
            }
            let mut cases: Vec<(HyperbolicStyle, i32, SeedKind)> = Vec::new();
            for style in [HyperbolicStyle::RegularPQ, HyperbolicStyle::RectifiedKagome, HyperbolicStyle::DualQP] {
                for delta in [0, -1, 1] {
                    cases.push((style, delta, SeedKind::Cell));
                }
            }
            cases.push((HyperbolicStyle::RegularPQ, 0, SeedKind::Vertex));
            cases.push((HyperbolicStyle::RegularPQ, 0, SeedKind::Edge));
            for (style, delta, seed) in cases {
                match build_hyperbolic_tiling_seeded(p, q, style, 64, delta, seed) {
                    Ok(t) => {
                        built += 1;
                        assert!(t.faces.len() <= super::DRAW_CAPS.max_faces, "{{{p},{q}}} {style:?} {delta} {seed:?}");
                        assert!(t.complete_through_shell < 64 && t.face_shell.len() == t.faces.len());
                        assert!(t.growth_rule.contains("omitted") || t.growth_rule.contains("truncated") || t.growth_rule.contains("precision"),
                            "{{{p},{q}}} {style:?} {delta} {seed:?}: no stop reason in {}", t.growth_rule);
                    }
                    Err(e) => refused.push(format!("{{{p},{q}}} {style:?} {delta} {seed:?}: {e}")),
                }
            }
        }
    }
    println!("# time all pairs/paths at layers 64: {:?} ({built} built, {} refused)", t0.elapsed(), refused.len());
    assert!(refused.is_empty(), "{refused:#?}");
}
