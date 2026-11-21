//! Known-answer tests: Schläfli classification, the six polychora, H³ honeycomb patches.

use std::time::Instant;

use super::coxeter_gram::{classify_schlafli, CoxeterRealisation, HoneycombRegime as R};
use super::honeycombs_e3::{build_euclidean_honeycomb, EuclideanKind};
use super::honeycombs_h3::{build_honeycomb, build_hyperbolic_honeycomb};
use super::polychora_s3::build_polychoron;

const COMPACT: [(u32, u32, u32); 4] = [(3, 5, 3), (4, 3, 5), (5, 3, 4), (5, 3, 5)];
const PARACOMPACT: [(u32, u32, u32); 11] = [
    (3, 3, 6), (4, 3, 6), (5, 3, 6), (3, 4, 4), (4, 4, 3), (4, 4, 4), (3, 6, 3), (6, 3, 3), (6, 3, 4), (6, 3, 5), (6, 3, 6),
];
const SPHERICAL: [(u32, u32, u32); 6] = [(3, 3, 3), (4, 3, 3), (3, 3, 4), (3, 4, 3), (5, 3, 3), (3, 3, 5)];

#[test]
fn classification_table_is_exact() {
    for (p, q, r) in SPHERICAL {
        let c = classify_schlafli(p, q, r);
        assert_eq!(c.regime, R::SphericalS3, "{p},{q},{r}");
        assert!(c.det_g > 0.0);
    }
    let e = classify_schlafli(4, 3, 4);
    assert_eq!(e.regime, R::EuclideanE3);
    assert!(e.det_g.abs() < 1e-12);
    for (p, q, r) in COMPACT {
        let c = classify_schlafli(p, q, r);
        assert_eq!(c.regime, R::HyperbolicCompactH3, "{p},{q},{r}");
        assert!(c.det_g < 0.0);
    }
    // Ideal vertices: Euclidean vertex figure {3,6},{4,4},{6,3}; infinite cells: Euclidean cell.
    let ideal = [(3, 3, 6), (4, 3, 6), (5, 3, 6), (6, 3, 6), (3, 4, 4), (4, 4, 4), (3, 6, 3)];
    let infinite = [(4, 4, 3), (4, 4, 4), (6, 3, 3), (6, 3, 4), (6, 3, 5), (6, 3, 6), (3, 6, 3)];
    for (p, q, r) in PARACOMPACT {
        let c = classify_schlafli(p, q, r);
        assert_eq!(c.regime, R::HyperbolicParacompactH3, "{p},{q},{r}");
        assert!(c.det_g <= -1.0 / 16.0 + 1e-12, "{p},{q},{r}: {}", c.det_g);
        assert_eq!(c.has_ideal_vertices, ideal.contains(&(p, q, r)));
        assert_eq!(c.has_infinite_euclidean_cells, infinite.contains(&(p, q, r)));
    }
    // {3,3,6} attains det G = -1/16 exactly: 1 - 1/4 - 1/4 - 3/4 + 3/16.
    assert!((classify_schlafli(3, 3, 6).det_g + 1.0 / 16.0).abs() < 1e-12);
    // Every symbol with 3 <= p,q,r <= 8 is one of the 22 above or hyperideal.
    let mut counts = [0usize; 6];
    for p in 3..=8 {
        for q in 3..=8 {
            for r in 3..=8 {
                counts[classify_schlafli(p, q, r).regime as usize] += 1;
            }
        }
    }
    assert_eq!(&counts[..4], &[6, 1, 4, 11]);
    let h = classify_schlafli(7, 3, 3);
    assert_eq!(h.regime, R::HyperidealRefused);
    assert!(h.reason.starts_with("[PROVED") && h.reason.contains("cell {7,3} is hyperbolic"));
    let h = classify_schlafli(3, 3, 7);
    assert!(h.reason.contains("vertex figure {3,7} is hyperbolic") && h.reason.contains("ultra-ideal"));
    let d = classify_schlafli(4, 4, 2);
    assert_eq!(d.regime, R::DegenerateRefused);
    assert!(d.reason.contains("reducible") && d.det_g.abs() < 1e-12);
    assert!(build_honeycomb(4, 4, 2, 0, 1).is_err());
    assert!(build_honeycomb(7, 3, 3, 0, 1).unwrap_err().contains("PROVED"));
}

#[test]
fn realisation_reproduces_gram_matrix() {
    for (p, q, r) in SPHERICAL.iter().chain(COMPACT.iter()).chain(PARACOMPACT.iter()) {
        let real = CoxeterRealisation::new(*p, *q, *r).unwrap();
        let ms = [[1, *p, 2, 2], [*p, 1, *q, 2], [2, *q, 1, *r], [2, 2, *r, 1]];
        for (j, row) in ms.iter().enumerate() {
            for (k, &m) in row.iter().enumerate() {
                let want = if j == k { 1.0 } else { -(std::f64::consts::PI / m as f64).cos() };
                let got = super::coxeter_gram::form(&real.eta, &real.normals[j], &real.normals[k]);
                assert!((got - want).abs() < 1e-10, "{p},{q},{r} G[{j}][{k}] {got} vs {want}");
            }
        }
    }
}

#[test]
fn six_polychora_counts() {
    let want = [
        ((3, 3, 3), (5, 10, 10, 5)),
        ((4, 3, 3), (16, 32, 24, 8)),
        ((3, 3, 4), (8, 24, 32, 16)),
        ((3, 4, 3), (24, 96, 96, 24)),
        ((5, 3, 3), (600, 1200, 720, 120)),
        ((3, 3, 5), (120, 720, 1200, 600)),
    ];
    for ((p, q, r), counts) in want {
        let t = Instant::now();
        // Counts: Coxeter, Regular Polytopes (3rd ed., 1973), Table I(ii); V - E + F - C = χ(S³) = 0.
        let cx = build_polychoron(p, q, r).unwrap();
        let dt = t.elapsed().as_secs_f64();
        eprintln!("# time {{{p},{q},{r}}}: {dt:.4} s");
        assert_eq!(cx.counts(), counts, "{{{p},{q},{r}}}");
        assert_eq!(cx.euler(), 0);
        assert!(cx.faces.iter().all(|f| f.len() == p as usize));
        assert!(cx.points.iter().all(|x| x.iter().all(|c| c.is_finite())));
        assert!(cx.coords4.iter().all(|v| (v.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-9));
    }
    assert!(build_polychoron(4, 3, 4).is_err());
}

fn shell_one_formula(v: usize, e: usize, f: usize, p: usize) -> (usize, usize, usize, usize) {
    // Two cells meet in a face; for r >= 4 the face neighbours of the central cell share nothing else.
    (v + f * (v - p), e + f * (e - p), f + f * (f - 1), f + 1)
}

#[test]
fn compact_h3_shells() {
    let cube = build_hyperbolic_honeycomb(4, 3, 5, 0, 1).unwrap();
    assert_eq!(cube.counts(), (8, 12, 6, 1));
    assert_eq!(build_hyperbolic_honeycomb(4, 3, 5, 1, 1).unwrap().counts(), shell_one_formula(8, 12, 6, 4));
    assert_eq!(shell_one_formula(8, 12, 6, 4), (32, 60, 36, 7));
    let dod = build_hyperbolic_honeycomb(5, 3, 4, 0, 1).unwrap();
    assert_eq!(dod.counts(), (20, 30, 12, 1));
    assert_eq!(build_hyperbolic_honeycomb(5, 3, 4, 1, 1).unwrap().counts(), (200, 330, 144, 13));
    let sw = build_hyperbolic_honeycomb(5, 3, 5, 1, 1).unwrap();
    assert_eq!(sw.counts(), shell_one_formula(20, 30, 12, 5));
    let ico = build_hyperbolic_honeycomb(3, 5, 3, 0, 1).unwrap();
    assert_eq!(ico.counts(), (12, 30, 20, 1));
    assert_eq!(build_hyperbolic_honeycomb(3, 5, 3, 1, 1).unwrap().cells.len(), 21);
    for (p, q, r) in COMPACT {
        for s in 0..=2 {
            let t = Instant::now();
            let cx = build_hyperbolic_honeycomb(p, q, r, s, 1).unwrap();
            eprintln!("# time {{{p},{q},{r}}} shells={s}: {:.4} s, cells {}", t.elapsed().as_secs_f64(), cx.cells.len());
            assert!(cx.points.iter().all(|x| super::norm3(x) < 1.0));
            assert!(cx.is_ideal.iter().all(|b| !b));
            // Poincaré and Klein coordinates agree: k = 2p / (1 + |p|²).
            for (pp, kk) in cx.points.iter().zip(&cx.klein) {
                let s2 = super::dot3(pp, pp);
                assert!((0..3).all(|i| (2.0 * pp[i] / (1.0 + s2) - kk[i]).abs() < 1e-9));
            }
            // Shell-2 patches are balls of cells: V - E + F - C = 1.
            assert_eq!(cx.euler(), 1, "{{{p},{q},{r}}} shells={s}");
        }
    }
    // Central cell is centred at the origin of the Poincaré ball.
    let c = cube.points.iter().fold([0.0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
    assert!(super::norm3(&c) < 1e-9);
    assert!(build_hyperbolic_honeycomb(4, 3, 5, 3, 1).is_err());
}

#[test]
fn paracompact_h3_patches() {
    let t = build_hyperbolic_honeycomb(3, 3, 6, 0, 1).unwrap();
    assert_eq!(t.counts(), (4, 6, 4, 1));
    assert!(t.is_ideal.iter().all(|&b| b));
    assert!(t.points.iter().all(|x| (super::norm3(x) - 1.0).abs() < 1e-12));
    let c = build_hyperbolic_honeycomb(4, 3, 6, 1, 1).unwrap();
    assert_eq!(c.counts(), shell_one_formula(8, 12, 6, 4));
    // Infinite {6,3} cell truncated to the seed hexagon + 1 ring: 7 hexagons, 24 vertices, 30 edges.
    let h = build_hyperbolic_honeycomb(6, 3, 3, 0, 1).unwrap();
    assert_eq!(h.counts(), (24, 30, 7, 1));
    assert!(h.cell_truncated[0]);
    // {4,4} cell, 1 ring by edge adjacency: plus-shaped 5 squares, 12 vertices, 16 edges.
    assert_eq!(build_hyperbolic_honeycomb(4, 4, 3, 0, 1).unwrap().counts(), (12, 16, 5, 1));
    // 2 rings of a {6,3}: 1 + 6 + 12 = 19 hexagons; of a {4,4} by edge adjacency: 1 + 4 + 8 = 13 squares.
    assert_eq!(build_hyperbolic_honeycomb(6, 3, 3, 0, 2).unwrap().faces.len(), 19);
    assert_eq!(build_hyperbolic_honeycomb(4, 4, 3, 0, 2).unwrap().faces.len(), 13);
    for (p, q, r) in PARACOMPACT {
        let cls = classify_schlafli(p, q, r);
        for s in 0..=1 {
            let tm = Instant::now();
            let cx = build_hyperbolic_honeycomb(p, q, r, s, 1).unwrap();
            eprintln!("# time {{{p},{q},{r}}} shells={s}: {:.4} s, cells {}", tm.elapsed().as_secs_f64(), cx.cells.len());
            for (x, &ideal) in cx.points.iter().zip(&cx.is_ideal) {
                assert_eq!(ideal, cls.has_ideal_vertices);
                let n = super::norm3(x);
                if ideal {
                    assert!((n - 1.0).abs() < 1e-12);
                } else {
                    assert!(n < 1.0);
                }
            }
            assert!(cx.faces.iter().all(|f| f.len() == p as usize));
        }
    }
    assert!(build_hyperbolic_honeycomb(6, 3, 3, 0, 3).is_err());
}

#[test]
fn cubic_patch_matches_shell_formula() {
    // {4,3,4} has r = 4, so the shell-1 formula holds as for {4,3,5}.
    let cx = build_euclidean_honeycomb(EuclideanKind::Cubic, 1).unwrap();
    assert_eq!(cx.counts(), (32, 60, 36, 7));
    assert_eq!(build_honeycomb(4, 3, 4, 0, 1).unwrap().counts(), (8, 12, 6, 1));
}
