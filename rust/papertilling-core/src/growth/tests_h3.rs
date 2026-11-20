//! H³ and S³ growth tests (WP-G2): the coset adapter under the touching rule.
//!
//! Provenance of the numbers (plan §2.4):
//! - [hand] cell counts of {p,q}, c_v = number of cells of the vertex figure {q,r}, and the
//!   layer-1 formula F + E(r−3) + V(c_v − 1 − q − q(r−3));
//! - [COMPUTATIONAL EVIDENCE, same coset method] deeper counts from the critique's
//!   `cox_grow.py` (root-basis Coxeter matrices, relative-rounded keys), an implementation that
//!   shares no code with this crate but uses the same mathematics.

use std::time::Instant;

use super::engine::seed_cells;
use super::h3_complex::{CoxeterComplex, IDEAL_VERTEX_REASON, INFINITE_CELL_REASON};
use super::{grow, Growth, GrowthCaps, SeedKind, Stopped};
use crate::honeycomb3d::growth_build::MAX_GROWTH_CELLS;

fn run(p: u32, q: u32, r: u32, seed: SeedKind, layers: u32, cap: usize) -> (Growth<u32>, CoxeterComplex) {
    let mut x = CoxeterComplex::new(p, q, r).unwrap();
    let at = x.default_seed(seed).unwrap();
    let s0 = seed_cells(&mut x, &at).unwrap();
    let caps = GrowthCaps { max_layers: layers, max_cells: cap };
    x.begin_growth(caps);
    let g = grow(&mut x, s0, caps);
    x.end_growth();
    (g, x)
}

#[test]
fn lookahead_lookup_keeps_the_readout() {
    // The final-step lookup and the cap guard (begin_growth) must not change any readout value;
    // the last two cases stop on the cap (5 000 < S3 = 59 647 and 2 000 < S2 = 9 240).
    let cases = [
        (5, 3, 4, SeedKind::Cell, 2, 250_000),
        (4, 3, 5, SeedKind::Vertex, 1, 250_000),
        (3, 3, 5, SeedKind::Vertex, 5, 250_000),
        (5, 3, 4, SeedKind::Cell, 3, 5_000),
        (5, 3, 4, SeedKind::Vertex, 4, 2_000),
    ];
    for (p, q, r, seed, n, cap) in cases {
        let (fast, _) = run(p, q, r, seed, n, cap);
        let mut x = CoxeterComplex::new(p, q, r).unwrap();
        let at = x.default_seed(seed).unwrap();
        let s0 = seed_cells(&mut x, &at).unwrap();
        let full = grow(&mut x, s0, GrowthCaps { max_layers: n, max_cells: cap });
        assert_eq!(fast.stats, full.stats, "{{{p},{q},{r}}} {seed:?}");
        assert_eq!(fast.stopped, full.stopped);
    }
}

#[test]
fn cap_bounds_the_interned_cells() {
    // An over-cap layer stops after interning about `max_cells` cells, not the whole layer
    // (cap guard in `begin_growth`); the readout of the layers shown is unchanged.
    // [COMPUTATIONAL EVIDENCE, cox_grow.py] {5,3,5} vertex S1 = 3 550; S2 is far over 5 000.
    for cap in [5_000, 20_000] {
        let t = Instant::now();
        let (g, x) = run(5, 3, 5, SeedKind::Vertex, 10, cap);
        assert_eq!(totals(&g), [20, 3550]);
        assert_eq!(g.stopped, Some(Stopped::CellCap { layer: 2, partial: cap + 1 }));
        let interned = x.precision().keys[0];
        // a single star query adds at most c_v = 20 cells past the cap
        assert!(interned <= cap + 20, "interned {interned} cells for cap {cap}");
        eprintln!("# time {{5,3,5}} vertex over cap {cap}: {:.2}s, {interned} cells interned", t.elapsed().as_secs_f64());
    }
}

fn totals(g: &Growth<u32>) -> Vec<usize> {
    g.stats.iter().map(|s| s.cells).collect()
}

const COMPACT: [(u32, u32, u32); 4] = [(4, 3, 5), (5, 3, 4), (5, 3, 5), (3, 5, 3)];

#[test]
fn cell_shapes_and_vertex_stars_by_hand() {
    // [hand] cube (8,12,6), dodecahedron (20,30,12), icosahedron (12,30,20); c_v = faces of
    // the vertex figure: icosahedron {3,5} 20, octahedron {3,4} 8, dodecahedron {5,3} 12.
    let want = [((8, 12, 6), 20), ((20, 30, 12), 8), ((20, 30, 12), 20), ((12, 30, 20), 12)];
    for ((p, q, r), ((v, e, f), cv)) in COMPACT.into_iter().zip(want) {
        let x = CoxeterComplex::new(p, q, r).unwrap();
        assert_eq!(x.local_counts(), (v, e, f, Some(cv)), "{{{p},{q},{r}}}");
    }
}

#[test]
fn compact_layer_one_matches_hand_formula() {
    for (p, q, r) in COMPACT {
        let x = CoxeterComplex::new(p, q, r).unwrap();
        let (v, e, f, cv) = x.local_counts();
        let cv = cv.unwrap() as i64;
        let (v, e, f, qi, ri) = (v as i64, e as i64, f as i64, i64::from(q), i64::from(r));
        // [hand] §2.4: 110, 62, 272, 92.
        let formula = f + e * (ri - 3) + v * (cv - 1 - qi - qi * (ri - 3));
        let (g, _) = run(p, q, r, SeedKind::Cell, 1, 250_000);
        assert_eq!(g.stats[1].new_cells as i64, formula, "{{{p},{q},{r}}}");
        // vertex-seed layer 0 is the vertex star: c_v cells.
        let (g, _) = run(p, q, r, SeedKind::Vertex, 0, 250_000);
        assert_eq!(g.stats[0].cells as i64, cv);
    }
    let pin: Vec<i64> = COMPACT
        .iter()
        .map(|&(p, q, r)| run(p, q, r, SeedKind::Cell, 1, 1000).0.stats[1].new_cells as i64)
        .collect();
    assert_eq!(pin, [110, 62, 272, 92]);
}

#[test]
#[ignore = "pilot: about 2 min; run with --ignored"]
fn pilot_cell_seed_depth() {
    let cases = [
        (5, 3, 4, SeedKind::Cell, 3),
        (4, 3, 5, SeedKind::Cell, 3),
        (4, 3, 5, SeedKind::Vertex, 2),
        (5, 3, 4, SeedKind::Vertex, 2),
        (3, 5, 3, SeedKind::Vertex, 2),
        (5, 3, 5, SeedKind::Vertex, 2),
        (5, 3, 5, SeedKind::Cell, 2),
        (3, 5, 3, SeedKind::Cell, 3),
    ];
    for (p, q, r, seed, n) in cases {
        let t = Instant::now();
        let (g, x) = run(p, q, r, seed, n, 250_000);
        eprintln!(
            "# time {{{p},{q},{r}}} {seed:?} seed n=0..{n}: {:.2}s totals {:?} chi {:?} stopped {:?} precision {:?}",
            t.elapsed().as_secs_f64(),
            totals(&g),
            g.stats.iter().map(|s| s.euler).collect::<Vec<_>>(),
            g.stopped,
            x.precision()
        );
    }
}

#[test]
fn ideal_vertices_and_infinite_cells_refuse() {
    // [PROVED] (plan §1 Q4): an ideal vertex lies in infinitely many cells.
    let (g, _) = run(4, 3, 6, SeedKind::Cell, 3, 1000);
    assert_eq!(g.stats.len(), 1);
    assert_eq!(g.stopped, Some(Stopped::Infinite { layer: 1, reason: IDEAL_VERTEX_REASON }));
    let mut x = CoxeterComplex::new(3, 3, 6).unwrap();
    let at = x.default_seed(SeedKind::Vertex).unwrap();
    assert!(seed_cells(&mut x, &at).is_err());
    assert_eq!(CoxeterComplex::new(6, 3, 3).err().as_deref(), Some(INFINITE_CELL_REASON));
}

fn relation_residual(rr: &[crate::honeycomb3d::coxeter_gram::Mat4; 4], pqr: (u32, u32, u32)) -> f64 {
    use crate::honeycomb3d::coxeter_gram::{identity4, mat_mul};
    let m = |i: usize, j: usize| -> u32 {
        match (i.min(j), i.max(j)) {
            (a, b) if a == b => 1,
            (0, 1) => pqr.0,
            (1, 2) => pqr.1,
            (2, 3) => pqr.2,
            _ => 2,
        }
    };
    let mut worst = 0.0f64;
    for i in 0..4 {
        for j in i..4 {
            let g = mat_mul(&rr[i], &rr[j]);
            let mut acc = identity4();
            for _ in 0..m(i, j) {
                acc = mat_mul(&acc, &g);
            }
            let id = identity4();
            for a in 0..4 {
                for b in 0..4 {
                    worst = worst.max((acc[a][b] - id[a][b]).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn ldl_realisation_satisfies_coxeter_relations() {
    use crate::honeycomb3d::coxeter_gram::{form, CoxeterRealisation};
    use crate::honeycomb3d::coxeter_ldl::ldl_realisation;
    // (R_i R_j)^{m_ij} = 1 and ⟨n_i, n_j⟩ = G_ij, checked numerically on the matrices.
    for pqr in COMPACT.into_iter().chain([(3, 3, 5), (5, 3, 3), (4, 3, 3), (3, 3, 6), (3, 4, 4)]) {
        let (p, q, r) = pqr;
        let real = ldl_realisation(p, q, r).unwrap();
        let res = relation_residual(&real.reflections, pqr);
        let old = relation_residual(&CoxeterRealisation::new(p, q, r).unwrap().reflections, pqr);
        eprintln!("{{{p},{q},{r}}} relation residual: LDL {res:.2e}, eigen {old:.2e}");
        assert!(res < 1e-13, "{res}");
        let ms = [p, q, r];
        for i in 0..4 {
            for j in 0..4 {
                let want = match (i as i64 - j as i64).abs() {
                    0 => 1.0,
                    1 => -(std::f64::consts::PI / f64::from(ms[i.min(j)])).cos(),
                    _ => 0.0,
                };
                assert!((form(&real.eta, &real.normals[i], &real.normals[j]) - want).abs() < 1e-14);
            }
        }
        assert!(real.w[3] == [1.0, 0.0, 0.0, 0.0]);
    }
}

#[test]
fn compact_layer_two_and_vertex_s2_coset_counts() {
    // [COMPUTATIONAL EVIDENCE, same coset method] cox_grow.py (plan §2.4): cell-seed layer-2
    // new cells 3458, 1922, 45 632, 4412; vertex-seed S2 totals 23 100, 9 240, 30 906, and
    // {5,3,5} vertex S1 = 3 550 with S2 over the cap. chi = 1 measured at every layer.
    // chi = 1 for the cell seeds of {4,3,5}, {5,3,4}, {3,5,3} to layer 2 and {5,3,5} vertex S0, S1
    // was also re-measured with cox_grow.py (euler mode) on 2026-11-20.
    let t = Instant::now();
    let mut cell2 = Vec::new();
    for (p, q, r) in COMPACT {
        let (g, _) = run(p, q, r, SeedKind::Cell, 2, 250_000);
        if (p, q, r) != (5, 3, 5) {
            assert!(g.stats.iter().all(|s| s.euler == 1), "{{{p},{q},{r}}} cell chi");
        }
        cell2.push(g.stats[2].new_cells);
    }
    assert_eq!(cell2, [3458, 1922, 45_632, 4412]);
    eprintln!("# time cell-seed layer 2, 4 honeycombs: {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    for ((p, q, r), want) in [((4, 3, 5), 23_100), ((5, 3, 4), 9_240), ((3, 5, 3), 30_906)] {
        let (g, _) = run(p, q, r, SeedKind::Vertex, 2, 250_000);
        assert_eq!(totals(&g)[2], want, "{{{p},{q},{r}}}");
        assert!(g.stats.iter().all(|s| s.euler == 1), "{{{p},{q},{r}}} chi");
        assert_eq!(g.stopped, None);
    }
    eprintln!("# time vertex-seed S2, 3 honeycombs: {:.2}s", t.elapsed().as_secs_f64());
    for cap in [50_000, 250_000] {
        let (g, _) = run(5, 3, 5, SeedKind::Vertex, 2, cap);
        assert_eq!(totals(&g), [20, 3550]);
        assert!(g.stats.iter().all(|s| s.euler == 1), "{{5,3,5}} vertex chi");
        assert_eq!(g.stopped, Some(Stopped::CellCap { layer: 2, partial: cap + 1 }));
    }
    // {4,3,5} edge seed: S0, S1 = 5, 280 (§2.4); r = 5 cells around an edge [hand].
    let (g, _) = run(4, 3, 5, SeedKind::Edge, 1, 250_000);
    assert_eq!(totals(&g), [5, 280]);
    assert!(g.stats.iter().all(|s| s.euler == 1));
    // Face seed: the 2 cells on a face [hand].
    for (p, q, r) in COMPACT {
        assert_eq!(run(p, q, r, SeedKind::Face, 0, 1000).0.stats[0].cells, 2);
    }
}

#[test]
fn s3_polychora_close() {
    // 600-cell vertex seed 20, 130, 330, 530, 600, chi 1,1,1,1,0, Closed at 4
    // [COMPUTATIONAL EVIDENCE, cox_grow.py; chi(S3) = 0 and the 600 cells are hand facts].
    let (g, _) = run(3, 3, 5, SeedKind::Vertex, 10, 250_000);
    assert_eq!(totals(&g), [20, 130, 330, 530, 600]);
    assert_eq!(g.stats.iter().map(|s| s.euler).collect::<Vec<_>>(), [1, 1, 1, 1, 0]);
    assert_eq!(g.stopped, Some(Stopped::Closed { layer: 4 }));
    assert_eq!((g.stats[4].boundary_vertices, g.stats[4].boundary_edges), (0, 0));
    // [hand] 5-cell: 4 cells at a vertex, then all 5. 16-cell: vertex figure octahedron, 8
    // cells, then all 16. 8-cell: the 4 facets x_i = 0 at the vertex 0000 hold every vertex
    // but 1111, and each facet x_i = 1 has such a vertex, so S1 = all 8.
    for ((p, q, r), want) in [((3, 3, 3), [4, 5]), ((3, 3, 4), [8, 16]), ((4, 3, 3), [4, 8])] {
        let (g, _) = run(p, q, r, SeedKind::Vertex, 10, 1000);
        assert_eq!(totals(&g), want, "{{{p},{q},{r}}}");
        assert_eq!(g.stopped, Some(Stopped::Closed { layer: 1 }));
        assert_eq!(g.stats[1].euler, 0);
    }
    // 120-cell and 24-cell close with all their cells (Coxeter, Regular Polytopes, Table I(ii)).
    for ((p, q, r), all) in [((5, 3, 3), 120), ((3, 4, 3), 24)] {
        let (g, _) = run(p, q, r, SeedKind::Cell, 20, 1000);
        assert_eq!(*totals(&g).last().unwrap(), all);
        assert!(matches!(g.stopped, Some(Stopped::Closed { .. })));
        assert_eq!(g.stats.last().unwrap().euler, 0);
    }
}

#[test]
fn key_gap_at_least_100_tol_and_tiny_merges() {
    // §2.3 at the deepest builds the cap allows that run in a test budget: no two distinct keys
    // within 100 × tol, and every merge far below tol.
    let t = Instant::now();
    // {4,3,5} cell S3 = 107 359 is the deepest build under the default cap MAX_GROWTH_CELLS.
    let cases = [(5, 3, 4, SeedKind::Cell, 3), (4, 3, 5, SeedKind::Cell, 3), (4, 3, 5, SeedKind::Vertex, 2), (3, 3, 5, SeedKind::Vertex, 5)];
    for (p, q, r, seed, n) in cases {
        let (g, x) = run(p, q, r, seed, n, MAX_GROWTH_CELLS);
        assert!(!matches!(g.stopped, Some(Stopped::CellCap { .. })), "{{{p},{q},{r}}} {seed:?} fits the default cap");
        let pr = x.precision();
        eprintln!("{{{p},{q},{r}}} {seed:?} n={n}: cells {:?} precision {pr:?}", totals(&g));
        if (p, q, r) == (5, 3, 4) {
            // [COMPUTATIONAL EVIDENCE, cox_grow.py, plan §1 Q13] cell-seed S3 = 59 647.
            assert_eq!(totals(&g)[3], 59_647);
        }
        if (p, q, r, seed) == (4, 3, 5, SeedKind::Cell) {
            // [COMPUTATIONAL EVIDENCE, cox_grow.py, plan §1 Q13] cell-seed S3 = 107 359.
            assert_eq!(totals(&g)[3], 107_359);
        }
        assert_eq!(pr.min_gap, None);
        assert!(pr.max_merge < pr.tol / 1e4, "{pr:?}");
    }
    eprintln!("# time gap check: {:.2}s", t.elapsed().as_secs_f64());
}
