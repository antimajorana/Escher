//! Known-answer tests for BDF vertex completion on regular hyperbolic `{p, q}`.
//!
//! Sources of the expected values (none is read off the code under test):
//! - hand counts written out in the comments below;
//! - BDF, arXiv:1805.02665 Eqs. (10)-(11): `lambda_+ = gamma + sqrt(gamma^2 - 1)`,
//!   `gamma = ((p-2)(q-2) - 2)/2`;
//! - an independent Python script (hyperboloid reflections, its own code, not sharing this module),
//!   written and run on 2026-11-18 by a subagent of an AI coding workflow; the script is NOT in the
//!   repository, and its numbers are transcribed here from its output. It grew the same shells and
//!   checked chi = 1 / no pinch for each. It is cited as `GT` below. Its first run gave a_0..a_5
//!   for seven tilings; a later run of the same script (same day, by a review subagent) extended
//!   {7,3} to a_7 and {5,4}, {8,3}, {3,7}, {4,5} to a_6 (see [`GT_SHELLS`] for which values have
//!   saved output). With those transcribed, `a_{k+1} = 2 gamma a_k - a_{k-1}` is checkable against
//!   GT up to k = 6 for {7,3}, k = 5 for the next four and k = 4 for {4,6}, {6,4}.
//!
//! Asserts beyond that range compare the module's own two counters (geometric dedup vs boundary
//! combinatorics) and are labelled as consistency checks, not as independently confirmed values.

use std::collections::HashMap;
use std::time::Instant;

use super::inflation::*;
use super::{build_hyperbolic_tiling, build_hyperbolic_tiling_capped, DrawCaps, HyperbolicStyle, DRAW_CAPS};

fn shells(p: usize, q: usize, n: usize) -> Vec<usize> {
    let c = vertex_completion_counts(p, q, n).unwrap();
    assert!(c.stopped.is_none(), "{{{p},{q}}} stopped: {:?}", c.stopped);
    c.shells.iter().map(|s| s.shell).collect()
}

/// Shell 1 of one `p`-gon: the `p` tiles across its edges, plus at each of its `p` vertices the
/// `q - 3` tiles that touch the seed only at that vertex: `p + p (q - 3) = p (q - 2)`.
#[test]
fn first_shell_hand_count() {
    for (p, q, hand) in [(7, 3, 7), (5, 4, 10), (3, 7, 15), (4, 5, 12), (8, 3, 8), (4, 6, 16), (6, 4, 12)] {
        assert_eq!(hand, p * (q - 2));
        assert_eq!(shells(p, q, 1), vec![1, hand], "{{{p},{q}}}");
    }
    // Same hand formula over every hyperbolic pair the app exposes (3..=12).
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            assert_eq!(shells(p, q, 1)[1], p * (q - 2), "{{{p},{q}}}");
        }
    }
}

/// Smoke test over the app's whole range: every hyperbolic pair, style and layer count builds, all
/// points lie in the open unit disk, and the seed (face 0) is a p-gon (q-gon for the dual).
#[test]
fn patch_smoke_all_pairs_styles_layers() {
    let t = Instant::now();
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            for style in [HyperbolicStyle::RegularPQ, HyperbolicStyle::RectifiedKagome, HyperbolicStyle::DualQP] {
                for layers in 1..=4 {
                    let h = build_hyperbolic_tiling(p, q, style, layers, 0).unwrap();
                    let seed = if style == HyperbolicStyle::DualQP { q } else { p };
                    assert_eq!(h.faces[0].len(), seed);
                    assert!(h.faces.len() <= DRAW_CAPS.max_faces && h.face_shell.len() == h.faces.len());
                    assert!(h.poincare_pts.iter().all(|z| z[0] * z[0] + z[1] * z[1] < 1.0));
                }
            }
        }
    }
    println!("# time smoke over all pairs/styles/layers: {:?}", t.elapsed());
}

/// `{7,3}` by hand (q = 3, so vertex completion = edge completion).
/// Shell 1: 7 heptagons across the 7 edges. Each has 7 - 2 = 5 new vertices; neighbours share one,
/// so the boundary has 7 * 4 = 28 vertices: 21 lying in 1 patch face, 7 in 2 patch faces.
/// Shell 2: one new face per boundary edge (28), except that at each 2-face vertex the faces across
/// its two edges coincide (q - 2 = 1 face left there): 28 - 7 = 21.
/// Shell 3: of those 21, 7 touch 2 old edges (new vertices 7 - 3 = 4) and 14 touch 1 (5 new).
/// Boundary = 7 * 3 + 14 * 4 = 77 vertices, 21 of them shared (2 faces). Shell 3 = 77 - 21 = 56.
/// GT continues 147, 385.
#[test]
fn shells_73_hand_and_ground_truth() {
    let a = shells(7, 3, 5);
    assert_eq!(&a[..4], &[1, 7, 21, 56], "hand count");
    assert_eq!(&a[4..], &[147, 385], "GT");
}

/// `{5,4}` by hand. Shell 1: 5 edge faces + 5 corner faces = 10, alternating around the seed.
/// Edge faces have 5 - 2 = 3 new vertices, corner faces 5 - 1 = 4; neighbours share one, so the
/// boundary has 5 * 2 + 5 * 3 = 25 vertices: 10 shared (2 faces, 2 outside faces left) and 15
/// private (1 face, 3 outside faces: 1 corner face each). Shell 2 = 25 edge faces + 15 corner = 40.
/// GT continues 150, 560, 2090.
#[test]
fn shells_54_hand_and_ground_truth() {
    let a = shells(5, 4, 5);
    assert_eq!(&a[..3], &[1, 10, 40], "hand count");
    assert_eq!(&a[3..], &[150, 560, 2090], "GT");
}

/// Cumulative `|S_n|`, n = 0..5, from GT for seven tilings.
const GT_CUMULATIVE: [(usize, usize, [usize; 6]); 7] = [
    (7, 3, [1, 8, 29, 85, 232, 617]),
    (3, 7, [1, 16, 61, 181, 496, 1321]),
    (5, 4, [1, 11, 51, 201, 761, 2851]),
    (4, 5, [1, 13, 61, 241, 913, 3421]),
    (8, 3, [1, 9, 41, 161, 609, 2281]),
    (4, 6, [1, 17, 113, 673, 3937, 22961]),
    (6, 4, [1, 13, 85, 505, 2953, 17221]),
];

#[test]
fn cumulative_counts_match_ground_truth_and_combinatorial_counter() {
    for (p, q, gt) in GT_CUMULATIVE {
        let t = Instant::now();
        let c = vertex_completion_counts(p, q, 5).unwrap();
        let geo: Vec<usize> = c.shells.iter().map(|s| s.cumulative).collect();
        let dt = t.elapsed();
        assert_eq!(geo, gt.to_vec(), "{{{p},{q}}} geometric vs GT");
        let comb = vertex_completion_counts_combinatorial(p, q, 5).expect("convex");
        let shell_geo: Vec<u64> = c.shells.iter().map(|s| s.shell as u64).collect();
        assert_eq!(comb, shell_geo, "{{{p},{q}}} combinatorial vs geometric");
        // Dedup gap: duplicates are numerical (tiny); distinct centres are >= 2r apart (adjacent
        // faces are exactly 2r apart, r the inradius).
        let two_r = 2.0 * BaseFace::new(p, q).unwrap().inradius;
        println!(
            "{{{p},{q}}} n<=5: {:?} in {dt:?}; merged <= {:.1e}, distinct >= {:.6} (2r = {two_r:.6})",
            geo, c.max_merged_centre, c.min_distinct_centre
        );
        assert!(c.max_merged_centre < 1e-6);
        assert!(c.min_distinct_centre > two_r - 1e-6);
    }
}

#[test]
fn runtime_n5_73_and_45_well_under_a_second() {
    for (p, q) in [(7, 3), (4, 5)] {
        let t = Instant::now();
        let c = vertex_completion_counts(p, q, 5).unwrap();
        let dt = t.elapsed();
        println!("# time {{{p},{q}}} n_max=5: {dt:?} ({} faces)", c.shells.last().unwrap().cumulative);
        if !cfg!(debug_assertions) {
            assert!(dt.as_secs_f64() < 1.0);
        }
    }
}

/// BDF Eq. (10)-(11) by hand: {7,3}: gamma = 3/2, lambda_+ = (3 + sqrt 5)/2 = phi^2;
/// {5,4}: gamma = 2, lambda_+ = 2 + sqrt 3; {4,6}: gamma = 3, lambda_+ = 3 + 2 sqrt 2.
#[test]
fn lambda_plus_closed_forms() {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    assert!((lambda_plus(7, 3) - phi * phi).abs() < 1e-12);
    assert!((lambda_plus(3, 7) - (3.0 + 5f64.sqrt()) / 2.0).abs() < 1e-12);
    assert!((lambda_plus(5, 4) - (2.0 + 3f64.sqrt())).abs() < 1e-12);
    assert!((lambda_plus(4, 6) - (3.0 + 2.0 * 2f64.sqrt())).abs() < 1e-12);
    assert!((lambda_plus(4, 4) - 1.0).abs() < 1e-12, "Euclidean: gamma = 1");
    assert_eq!(lambda_plus(3, 6), 1.0);
    assert_eq!(lambda_plus(6, 3), 1.0);
    // Spherical pairs (gamma = -1/2, 0, 1/2 by Eq. 11): no real lambda_+, documented as NaN.
    for (p, q) in [(3, 3), (3, 4), (3, 5)] {
        assert!(gamma_pq(p, q) < 1.0);
        assert!(lambda_plus(p, q).is_nan(), "{{{p},{q}}}");
    }
}

/// GT shells `a_0 .. a_N`. `a_0 .. a_5` are the differences of [`GT_CUMULATIVE`] (first GT run).
/// The extension: {7,3} a_6 = 1008, a_7 = 2639 and {4,5} a_6 = 9360 are transcribed from that run's
/// saved JSON output; {5,4} a_6 = 7800, {8,3} a_6 = 6240 and {3,7} a_6 = 2160 from the values quoted
/// in the review subagent's report of that run (its raw output for them was not kept).
const GT_SHELLS: [(usize, usize, &[u64]); 7] = [
    (7, 3, &[1, 7, 21, 56, 147, 385, 1008, 2639]),
    (5, 4, &[1, 10, 40, 150, 560, 2090, 7800]),
    (8, 3, &[1, 8, 32, 120, 448, 1672, 6240]),
    (3, 7, &[1, 15, 45, 120, 315, 825, 2160]),
    (4, 5, &[1, 12, 48, 180, 672, 2508, 9360]),
    (4, 6, &[1, 16, 96, 560, 3264, 19024]),
    (6, 4, &[1, 12, 72, 420, 2448, 14268]),
];

/// Recurrence `a_{k+1} = 2 gamma a_k - a_{k-1}` from k = 2, failing at k = 1 (computational evidence
/// from GT, not stated in the paper). The combinatorial counter must equal the GT shells wherever
/// GT has them, so the recurrence is GT-confirmed for k + 1 <= N (k <= 6 for {7,3}, 5 for the next
/// four, 4 for {4,6}, {6,4}); beyond it (k up to 7, or 6 for the fast-growing pairs) it is only a
/// consistency check of this module's own counter.
#[test]
fn recurrence_from_k2_and_failure_at_k1() {
    // {7,3} at k = 1: 2 gamma a_1 - a_0 = 3 * 7 - 1 = 20, but a_2 = 21 (hand count above).
    assert_eq!(3 * 7 - 1, 20);
    assert_eq!(shells(7, 3, 2)[2], 21);
    for (p, q, gt) in GT_SHELLS {
        // Transcription check: the first six GT shells sum to GT_CUMULATIVE.
        let cum = GT_CUMULATIVE.iter().find(|c| (c.0, c.1) == (p, q)).unwrap().2;
        let sums: Vec<u64> = gt[..6].iter().scan(0, |s, &a| { *s += a; Some(*s) }).collect();
        assert_eq!(sums, cum.map(|c| c as u64).to_vec(), "{{{p},{q}}} GT table");
        let two_gamma = ((p - 2) * (q - 2) - 2) as u64;
        let n = if two_gamma >= 6 { 7 } else { 8 };
        let a = vertex_completion_counts_combinatorial(p, q, n).expect("convex");
        assert_eq!(&a[..gt.len()], gt, "{{{p},{q}}} combinatorial vs GT");
        let k_gt = gt.len() - 2;
        for k in 2..n {
            let label = if k <= k_gt { "GT-confirmed" } else { "consistency check only (beyond GT)" };
            assert_eq!(a[k + 1], two_gamma * a[k] - a[k - 1], "{{{p},{q}}} k = {k}: {label}");
        }
        assert_ne!(a[2], two_gamma * a[1] - a[0], "{{{p},{q}}} recurrence should fail at k = 1");
    }
}

/// Observed identity (not stated in the paper, no proof here): {7,3} shells `a_n = 7 F_{2n}` for n >= 1, with
/// Fibonacci F_2 = 1, F_4 = 3, F_6 = 8, F_8 = 21, F_10 = 55 (7, 21, 56, 147, 385, as in the hand
/// count and GT above). It agrees with the recurrence: F_{2n+2} = 3 F_{2n} - F_{2n-2}, and
/// 2 gamma = 3 for {7,3}. Checked on the geometric counter to n = 5 and on the combinatorial
/// counter to n = 9. Hand count to n = 3 and GT to n = 7 (7 F_12 = 1008, 7 F_14 = 2639); n = 8, 9
/// are a consistency check only.
#[test]
fn shells_73_are_seven_times_even_fibonacci_observed() {
    let mut fib = vec![0_u64, 1];
    for i in 2..=20 {
        fib.push(fib[i - 1] + fib[i - 2]);
    }
    assert_eq!([fib[2], fib[4], fib[6], fib[8], fib[10]], [1, 3, 8, 21, 55]);
    let geo = shells(7, 3, 5);
    for (n, &a) in geo.iter().enumerate().skip(1) {
        assert_eq!(a as u64, 7 * fib[2 * n], "geometric n = {n}");
    }
    let comb = vertex_completion_counts_combinatorial(7, 3, 9).expect("convex");
    for (n, &a) in comb.iter().enumerate().skip(1) {
        let source = match n {
            1..=3 => "hand count",
            4..=7 => "GT",
            _ => "consistency check only (beyond GT)",
        };
        assert_eq!(a, 7 * fib[2 * n], "combinatorial n = {n}: {source}");
    }
}

/// Cross-check of the two independent counters (geometric dedup vs boundary combinatorics) over every
/// hyperbolic pair the app exposes (3..=12), through S_2. No expected values: the methods must agree.
#[test]
fn geometric_and_combinatorial_agree_all_pairs_to_s2() {
    let t = Instant::now();
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            let geo: Vec<u64> = shells(p, q, 2).into_iter().map(|a| a as u64).collect();
            assert_eq!(vertex_completion_counts_combinatorial(p, q, 2), Some(geo), "{{{p},{q}}}");
        }
    }
    println!("# time all-pairs S_2 cross-check: {:?}", t.elapsed());
}

/// Convergence check only: `a_n / a_{n-1} -> lambda_+` (loose tolerance).
#[test]
fn ratio_approaches_lambda_plus() {
    for (p, q, _) in GT_CUMULATIVE {
        let c = vertex_completion_counts(p, q, 5).unwrap();
        let r = c.shells[5].ratio.unwrap();
        assert!((r - lambda_plus(p, q)).abs() < 0.01, "{{{p},{q}}}: ratio {r} vs {}", lambda_plus(p, q));
    }
}

/// Every `S_n` is a closed disk: chi = V - E + F = 1, each edge in <= 2 faces, and no pinch vertex
/// (each boundary vertex has exactly 2 boundary edges, i.e. its patch faces form one arc).
#[test]
fn every_shell_is_a_closed_disk() {
    for (p, q, n_max) in [(7, 3, 4), (5, 4, 4), (3, 7, 4), (4, 5, 4), (8, 3, 3), (4, 6, 3), (6, 4, 3)] {
        let patch = grow_vertex_completion(p, q, n_max, usize::MAX).unwrap();
        for n in 0..=n_max {
            assert_closed_disk(&patch, n);
        }
    }
}

/// `S_n` of `patch` has chi = 1, each edge in <= 2 faces, and no pinch vertex.
fn assert_closed_disk(patch: &VcPatch, n: usize) {
    let (p, q) = (patch.base.p, patch.base.q);
    let mut verts = HypPointSet::new(patch.base.edge_length());
    let mut edge_faces: HashMap<(usize, usize), usize> = HashMap::new();
    let nf = patch.shell.iter().take_while(|&&s| s <= n).count();
    for i in 0..nf {
        let idx: Vec<usize> = patch.face_vertices(i).into_iter().map(|x| verts.insert(x).0).collect();
        for k in 0..p {
            let (u, v) = (idx[k], idx[(k + 1) % p]);
            *edge_faces.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    let (nv, ne) = (verts.pts.len() as i64, edge_faces.len() as i64);
    assert_eq!(nv - ne + nf as i64, 1, "{{{p},{q}}} S_{n}: chi");
    assert!(edge_faces.values().all(|&c| c <= 2));
    let mut bdeg = vec![0_usize; verts.pts.len()];
    for (&(u, v), _) in edge_faces.iter().filter(|&(_, &c)| c == 1) {
        bdeg[u] += 1;
        bdeg[v] += 1;
    }
    assert!(bdeg.iter().all(|&d| d == 0 || d == 2), "{{{p},{q}}} S_{n}: pinch vertex");
}

#[test]
fn non_hyperbolic_and_budget_stop() {
    assert!(vertex_completion_counts(4, 4, 2).is_err());
    assert!(vertex_completion_counts(6, 3, 2).is_err());
    assert!(vertex_completion_counts_combinatorial(4, 4, 2).is_none());
    // {12,12}: shell 1 = 12 * 10 = 120 (hand); later shells blow up, so a small budget stops early.
    let c = vertex_completion_counts_budget(12, 12, 6, 50_000).unwrap();
    assert_eq!(c.shells[1].shell, 120);
    assert!(c.stopped.is_some());
}

// ---- drawn patch (build_hyperbolic_tiling) ----

/// {7,3}, layers 2: |S_2| = 1 + 7 + 21 = 29 (hand count above). The pre-upgrade edge-BFS also gave
/// 29 (measured on e09ee15 in a scratch copy), as it must for q = 3.
#[test]
fn patch_73_layers2_is_s2() {
    let t = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, 0).unwrap();
    assert_eq!(t.faces.len(), 29);
    assert_eq!(t.faces[0].len(), 7);
    assert_eq!(t.complete_through_shell, 2);
    let per_shell = (0..3).map(|s| t.face_shell.iter().filter(|&&x| x == s).count()).collect::<Vec<_>>();
    assert_eq!(per_shell, vec![1, 7, 21]);
    // Closed disk: chi = V - E + F = 1.
    assert_eq!(t.poincare_pts.len() as i64 - t.edges.len() as i64 + 29, 1);
}

/// {5,4}: layers 1 and 2 give 1 + 10 = 11 and 11 + 40 = 51 faces (hand counts above); the old
/// edge-BFS gave 6 and 21 (it missed the corner tiles).
#[test]
fn patch_54_includes_corner_tiles() {
    for (layers, faces) in [(1, 11), (2, 51)] {
        let t = build_hyperbolic_tiling(5, 4, HyperbolicStyle::RegularPQ, layers, 0).unwrap();
        assert_eq!(t.faces.len(), faces, "layers {layers}");
        assert_eq!(t.complete_through_shell, layers);
    }
}

/// Face-cap mechanism with the old caps (180 faces, radius 0.985): {7,3} layers 4 would be
/// |S_4| = 232 > 180, so only S_3 (85 faces, hand count) is drawn.
#[test]
fn patch_face_cap_keeps_whole_shells() {
    let old = DrawCaps { max_faces: 180, max_centre_radius: 0.985 };
    let t = build_hyperbolic_tiling_capped(7, 3, HyperbolicStyle::RegularPQ, 4, 0, old).unwrap();
    assert_eq!(t.faces.len(), 85);
    assert_eq!(t.complete_through_shell, 3);
    assert!(t.growth_rule.contains("omitted"), "{}", t.growth_rule);
}

/// Default caps make the layers slider meaningful: {5,4} layers 1..4 draw |S_1..S_4| = 11, 51, 201,
/// 761 (hand counts 1 + 10, + 40; GT shells 150, 560), {7,3} 8, 29, 85, 232 (hand / GT), all complete.
#[test]
fn default_caps_draw_four_whole_shells() {
    for (p, q, want) in [(5, 4, [11, 51, 201, 761]), (7, 3, [8, 29, 85, 232])] {
        for (l, &w) in want.iter().enumerate() {
            let t = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, l + 1, 0).unwrap();
            assert_eq!((t.faces.len(), t.complete_through_shell), (w, l + 1), "{{{p},{q}}} layers {}", l + 1);
        }
    }
}

/// Styles: dual of {7,3} is {3,7} (|S_1| = 1 + 15 = 16 triangles); kagome keeps the p-gon count.
#[test]
fn patch_styles() {
    let d = build_hyperbolic_tiling(7, 3, HyperbolicStyle::DualQP, 1, 0).unwrap();
    assert_eq!((d.p, d.q, d.faces.len(), d.faces[0].len()), (3, 7, 16, 3));
    let k = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RectifiedKagome, 2, 0).unwrap();
    assert_eq!(k.faces.len(), 29);
    // Kagome vertices are edge midpoints: E(S_2) of {7,3} = V + F - 1 (chi = 1).
    let r = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, 0).unwrap();
    assert_eq!(k.poincare_pts.len(), r.edges.len());
    let cone = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, -1).unwrap();
    assert!(cone.growth_rule.contains("edge-BFS for q = 3"), "{}", cone.growth_rule);
}

/// {12,12} layers 1 with the old radius cap 0.985: S_1 = 121 faces fits the face cap, but corner tiles
/// sit at Poincare radius tanh(R) with cosh R = cot^2(pi/12) ~ 13.9, i.e. ~0.9974 > 0.985: shell 1 is
/// truncated and flagged. (Under the default cap 0.998 they are drawn.)
#[test]
fn patch_radius_cap_flags_truncation() {
    let old = DrawCaps { max_faces: 180, max_centre_radius: 0.985 };
    let t = build_hyperbolic_tiling_capped(12, 12, HyperbolicStyle::RegularPQ, 1, 0, old).unwrap();
    assert!(t.faces.len() < 121);
    assert_eq!(t.complete_through_shell, 0);
    assert!(t.growth_rule.contains("truncated"), "{}", t.growth_rule);
}

/// Large q (regression: an absolute 1e-6 star-closure tolerance rejected these, while the pre-upgrade
/// e09ee15 drew a patch). Layers 2 must build, and the drawn patch must be a closed disk:
/// V - E + F = 1. Shell 1 (p (q - 2) faces; in the drawn patch its corner faces lie beyond the radius
/// cap or the whole shell exceeds the face cap, so it is drawn partly or not at all) is checked on the
/// counter instead: hand count p (q - 2) and S_1 a closed disk,
/// which exercises every vertex star. Very large q builds, or fails or stops with a precision message.
/// The closed-disk check is topological (on a distance dedup of the vertices); it does not check
/// that the points lie on the hyperboloid, so beyond p, q <= 12 these are empirical checks.
#[test]
fn large_q_pairs_build_and_are_disks() {
    let t = Instant::now();
    for (p, q) in [(12, 43), (8, 71), (7, 82), (4, 200), (3, 383)] {
        assert_eq!(shells(p, q, 1), vec![1, p * (q - 2)], "{{{p},{q}}} hand count");
        assert_closed_disk(&grow_vertex_completion(p, q, 1, usize::MAX).unwrap(), 1);
        let h = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, 2, 0)
            .unwrap_or_else(|e| panic!("{{{p},{q}}}: {e}"));
        let chi = h.poincare_pts.len() as i64 - h.edges.len() as i64 + h.faces.len() as i64;
        println!("{{{p},{q}}}: {} faces, chi = {chi}; {}", h.faces.len(), h.growth_rule);
        assert_eq!(chi, 1, "{{{p},{q}}}");
        assert_eq!(h.faces[0].len(), p);
        assert!(h.poincare_pts.iter().all(|z| z[0] * z[0] + z[1] * z[1] < 1.0));
    }
    for (p, q) in [(12, 43), (8, 71), (7, 82), (4, 200), (3, 383), (4, 1000), (3, 10000)] {
        // Shell 1 on the uncapped counter: hand count p (q - 2) and a closed disk, or a precision Err.
        match vertex_completion_counts(p, q, 1) {
            Ok(c) if c.stopped.is_some() => {
                let s = c.stopped.unwrap();
                println!("{{{p},{q}}} S_1: stopped: {s}");
                assert!(s.contains("precision"), "{{{p},{q}}}: {s}");
            }
            Ok(c) => {
                let r = BaseFace::new(p, q).unwrap().inradius;
                println!(
                    "{{{p},{q}}} S_1: merged <= {:.1e}, distinct >= {:.4} (2r = {:.4})",
                    c.max_merged_centre,
                    c.min_distinct_centre,
                    2.0 * r
                );
                assert_eq!(c.shells[1].shell, p * (q - 2), "{{{p},{q}}} hand count");
                assert!(c.max_merged_centre < 0.1 * r && c.min_distinct_centre > 1.9 * r);
                assert_closed_disk(&grow_vertex_completion(p, q, 1, usize::MAX).unwrap(), 1);
            }
            Err(e) => assert!(e.contains("precision"), "{{{p},{q}}}: {e}"),
        }
    }
    for (p, q) in [(4, 1000), (3, 10000)] {
        match build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, 2, 0) {
            Ok(h) => {
                let chi = h.poincare_pts.len() as i64 - h.edges.len() as i64 + h.faces.len() as i64;
                println!("{{{p},{q}}}: Ok, {} faces, chi = {chi}; {}", h.faces.len(), h.growth_rule);
                assert_eq!(chi, 1, "{{{p},{q}}}");
            }
            Err(e) => {
                println!("{{{p},{q}}}: Err {e}");
                assert!(e.contains("precision"), "{{{p},{q}}}: {e}");
            }
        }
    }
    println!("# time large-q builds: {:?}", t.elapsed());
}

/// Regression (round-2 review): for these very large q the f64 star walk drifted, and S_1 came back
/// Ok without a stop while not being a closed disk (chi = 0 or a pinch vertex). There must be no
/// such silent outcome: each pair is an Err or a precision stop, or S_1 has the hand count p (q - 2)
/// and is a closed disk. This rules out wrong counts and wrong topology only: S_1 can still pass with
/// face vertices off the hyperboloid, outside the Poincare disk (round-3 review: {6,1082}, {4,2221},
/// hand counts right), which no guard in the module detects (see its doc, "Precision").
#[test]
fn very_large_q_never_silently_wrong() {
    let t = Instant::now();
    // The nine pairs the review found; {4,2235} passes the star check and stops at the T limit.
    let pairs =
        [(3, 3803), (8, 919), (10, 681), (9, 772), (3, 3831), (3, 3838), (5, 1563), (6, 1248), (7, 1066), (4, 2235)];
    for (p, q) in pairs {
        match grow_vertex_completion(p, q, 1, 4_000_000) {
            Err(e) => assert!(e.contains("precision"), "{{{p},{q}}}: {e}"),
            Ok(patch) => match &patch.stopped {
                Some(s) => assert!(s.contains("precision"), "{{{p},{q}}}: {s}"),
                None => {
                    assert_eq!(patch.shell_sizes, vec![1, p * (q - 2)], "{{{p},{q}}} hand count");
                    assert_closed_disk(&patch, 1);
                }
            },
        }
    }
    println!("# time very large q: {:?}", t.elapsed());
}

/// Non-finite input is not silently treated as distance 0 (`f64::max(NaN, 0)` would give 0).
#[test]
fn non_finite_points_are_rejected() {
    let o = [0.0, 0.0, 1.0];
    assert!(hyp_dist([f64::NAN; 3], o).is_nan());
    assert!(hyp_dist([f64::INFINITY, 0.0, f64::INFINITY], o).is_nan());
    assert!(hyp_dist([0.0, 0.0, f64::INFINITY], o).is_nan(), "Minkowski norm -inf must not clamp to 0");
    assert_eq!(hyp_dist(o, o), 0.0);
    let mut set = HypPointSet::new(1.0);
    set.insert(o);
    let (i, new) = set.insert([f64::NAN, 0.0, 1.0]);
    assert!(new && i == 1 && set.suspect == 1, "a NaN point must be flagged, not merged");
    assert_eq!(set.insert(o), (0, false));
}

/// Band guard: with `min_sep = 1`, a point at distance 0.3 (inside [1/4, 3/4]) is flagged as suspect;
/// one at 1e-9 (a numerical duplicate) or 1.0 (a distinct neighbour) is not.
#[test]
fn ambiguous_dedup_distance_is_flagged() {
    let at = |d: f64| [d.sinh(), 0.0, d.cosh()];
    for (d, flagged) in [(1e-9, false), (1.0, false), (0.3, true), (0.7, true)] {
        let mut set = HypPointSet::new(1.0);
        set.insert(at(0.0));
        set.insert(at(d));
        assert_eq!(set.suspect > 0, flagged, "distance {d}");
    }
}

/// {12,12}: geometric and combinatorial counters agree through S_2; shell 3 reaches face centres
/// with t > 1e6 (distance ~ 3 * 2R ~ 20), so growth stops with an f64 precision note.
#[test]
fn precision_limit_stops_cleanly() {
    let t = Instant::now();
    let c = vertex_completion_counts(12, 12, 4).unwrap();
    println!("# time {{12,12}} to precision stop: {:?}; {:?}", t.elapsed(), c.stopped);
    assert_eq!(c.shells.len(), 3);
    assert!(c.stopped.as_deref().unwrap_or("").contains("precision"));
    let comb = vertex_completion_counts_combinatorial(12, 12, 2).unwrap();
    assert_eq!(comb, c.shells.iter().map(|s| s.shell as u64).collect::<Vec<_>>());
}
