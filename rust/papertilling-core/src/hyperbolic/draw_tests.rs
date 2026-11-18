//! Tests of the drawn patch (caps, wedge path) and of the letter-count shell counter.
//!
//! Expected values are hand counts (derivations in the comments), BDF Eq. (10)-(11), or, for the
//! `q = 3` wedge regression, the face set of the pre-upgrade edge-BFS (e09ee15 / 9ce9798), re-run
//! here from a verbatim copy of its algorithm ([`old_wedge_centroids`]).

use std::f64::consts::PI;
use std::time::Instant;

use super::inflation::*;
use super::wedge::{grow_polygon_vertex_completion, Polygon};
use super::{build_hyperbolic_tiling, build_hyperbolic_tiling_capped, orthogonal_circle, DrawCaps, HyperbolicStyle};

// ---- verbatim copy of the pre-upgrade wedge edge-BFS (e09ee15 mod.rs), caps as parameters ----

fn c_sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
fn c_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]] }
fn c_conj(a: [f64; 2]) -> [f64; 2] { [a[0], -a[1]] }
fn c_norm2(a: [f64; 2]) -> f64 { a[0] * a[0] + a[1] * a[1] }

fn reflect_geodesic(z: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    if let Some((c, r2)) = orthogonal_circle(a, b) {
        let dz = c_conj(c_sub(z, c));
        let denom = c_norm2(dz).max(1e-18);
        [c[0] + r2 * dz[0] / denom, c[1] - r2 * dz[1] / denom]
    } else {
        let len = c_norm2(a).sqrt().max(1e-18);
        let u = [a[0] / len, a[1] / len];
        let u2 = c_mul(u, u);
        c_mul(u2, c_conj(z))
    }
}

/// Old wedge patch: BFS over reflections across edges, Euclidean 1e-3 centroid dedup, centroid
/// radius cap, `max_faces` cap (mid-BFS). Returns the Euclidean centroids of the faces' Poincare
/// vertices and whether a cap fired. (Bookkeeping added: the largest merged and smallest kept
/// Euclidean centroid distance, via [`old_wedge_stats`].)
fn old_wedge_centroids(p: usize, q: usize, lay: usize, wedge_delta: i32, max_faces: usize, rcap: f64) -> (Vec<[f64; 2]>, bool) {
    let (c, capped, _, _) = old_wedge_stats(p, q, lay, wedge_delta, max_faces, rcap);
    (c, capped)
}

fn old_wedge_stats(p: usize, q: usize, lay: usize, wedge_delta: i32, max_faces: usize, rcap: f64) -> (Vec<[f64; 2]>, bool, f64, f64) {
    let (mut max_merged, mut min_kept) = (0.0_f64, f64::INFINITY);
    let p_core = (p as i32 + wedge_delta).clamp(3, 12) as usize;
    let half_edge_hyp = ((PI / (q as f64)).cos() / (PI / (p as f64)).sin()).max(1.0).acosh();
    let rc_hyp = (half_edge_hyp.sinh() / (PI / (p_core as f64)).sin()).asinh();
    let r_core = (0.5 * rc_hyp).tanh().min(0.96);
    let central: Vec<[f64; 2]> = (0..p_core)
        .map(|k| {
            let th = 2.0 * PI * (k as f64) / (p_core as f64);
            [r_core * th.cos(), r_core * th.sin()]
        })
        .collect();
    let mut poly_verts = vec![central];
    let mut seen_centers: Vec<[f64; 2]> = vec![[0.0, 0.0]];
    let mut capped = false;
    let mut bfs = std::collections::VecDeque::from([(0_usize, 0_usize)]);
    while let Some((pid, depth)) = bfs.pop_front() {
        if depth >= lay || poly_verts.len() >= max_faces {
            capped |= depth < lay;
            continue;
        }
        let cur_poly = poly_verts[pid].clone();
        let m = cur_poly.len();
        for e in 0..m {
            let (a, b) = (cur_poly[e], cur_poly[(e + 1) % m]);
            let next_poly: Vec<[f64; 2]> = cur_poly.iter().map(|&z| reflect_geodesic(z, a, b)).collect();
            let cen = next_poly.iter().fold([0.0, 0.0], |acc, &v| [acc[0] + v[0], acc[1] + v[1]]);
            let cen = [cen[0] / (m as f64), cen[1] / (m as f64)];
            if c_norm2(cen) > rcap * rcap {
                capped = true;
                continue;
            }
            let dmin = seen_centers.iter().map(|&c| (c[0] - cen[0]).hypot(c[1] - cen[1])).fold(f64::INFINITY, f64::min);
            if dmin < 1e-3 {
                max_merged = max_merged.max(dmin);
                continue;
            }
            min_kept = min_kept.min(dmin);
            seen_centers.push(cen);
            poly_verts.push(next_poly);
            bfs.push_back((poly_verts.len() - 1, depth + 1));
            if poly_verts.len() >= max_faces {
                capped = true;
                break;
            }
        }
    }
    (seen_centers, capped, max_merged, min_kept)
}

/// Euclidean centroids of the Poincare vertices of each drawn face (the old code's face "centre").
fn centroids(pts: &[[f64; 2]], faces: &[Vec<usize>]) -> Vec<[f64; 2]> {
    faces
        .iter()
        .map(|f| {
            let s = f.iter().fold([0.0, 0.0], |a, &i| [a[0] + pts[i][0], a[1] + pts[i][1]]);
            [s[0] / f.len() as f64, s[1] / f.len() as f64]
        })
        .collect()
}

/// Every point of `a` has a partner in `b` within `tol` and the sizes agree.
fn same_point_sets(a: &[[f64; 2]], b: &[[f64; 2]], tol: f64) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.iter().any(|y| (x[0] - y[0]).hypot(x[1] - y[1]) < tol))
}

const NO_CAPS: DrawCaps = DrawCaps { max_faces: 1_000_000, max_centre_radius: 0.985 };

/// q = 3: the three faces at a vertex are pairwise edge-adjacent, so vertex adjacency is edge
/// adjacency and the new rule (one-step star walks) is edge-BFS. The new face set must equal the old
/// one (count and centroids to 1e-9) wherever the old run hit no face cap: every q = 3 wedge case
/// p = 7..=12, delta = +-1, layers 1..=4, with the old radius cap 0.985 on both sides.
#[test]
fn wedge_q3_equals_old_edge_bfs() {
    let t = Instant::now();
    let mut compared = 0;
    for p in 7..=12 {
        for delta in [-1, 1] {
            for lay in 1..=4 {
                let (old, _) = old_wedge_centroids(p, 3, lay, delta, 1_000_000, 0.985);
                let h = build_hyperbolic_tiling_capped(p, 3, HyperbolicStyle::RegularPQ, lay, delta, NO_CAPS).unwrap();
                let new = centroids(&h.poincare_pts, &h.faces);
                assert!(same_point_sets(&old, &new, 1e-9), "{{{p},3}} delta {delta} layers {lay}: old {} vs new {}", old.len(), new.len());
                compared += 1;
            }
        }
    }
    println!("# time q=3 wedge regression ({compared} cases): {:?}", t.elapsed());
}

/// Same comparison through the public entry point with its default caps, where the old run (with
/// its 180-face cap) was not capped: this is the face set the app showed before.
#[test]
fn wedge_q3_default_caps_match_old_when_uncapped() {
    let mut compared = 0;
    for p in 7..=12 {
        for delta in [-1, 1] {
            for lay in 1..=4 {
                let (old, capped) = old_wedge_centroids(p, 3, lay, delta, 180, 0.985);
                let h = build_hyperbolic_tiling(p, 3, HyperbolicStyle::RegularPQ, lay, delta).unwrap();
                if !capped && h.complete_through_shell == lay {
                    assert!(same_point_sets(&old, &centroids(&h.poincare_pts, &h.faces), 1e-9), "{{{p},3}} {delta} {lay}");
                    compared += 1;
                }
            }
        }
    }
    println!("default-cap q = 3 wedge cases equal to the old output: {compared} of 48 (the rest were capped)");
    assert!(compared >= 12, "{compared}");
}

/// With a regular `p`-gon of the `{p, q}` circumradius the star walks close, and the wedge grower is
/// BDF vertex completion: hand counts |S_1| = 1 + p (q - 2), {5,4}: 1, 10, 40 and {7,3}: 1, 7, 21, 56
/// (derivations in inflation_tests).
#[test]
fn polygon_grower_on_regular_pq_is_bdf() {
    for (p, q, hand) in [(5, 4, vec![1, 10, 40]), (7, 3, vec![1, 7, 21, 56]), (4, 5, vec![1, 12, 48]), (3, 7, vec![1, 15, 45])] {
        let r = BaseFace::new(p, q).unwrap().circumradius;
        let g = grow_polygon_vertex_completion(&Polygon::regular(p, r), q, hand.len() - 1, 1_000_000, 0.9999, 1e-3);
        // Merges are numerical duplicates only; distinct centres are >= 2r apart (r the inradius).
        assert!(g.max_merged < 1e-9 && g.suspect == 0, "{{{p},{q}}}: merged {:.1e}", g.max_merged);
        assert!(g.min_distinct > 2.0 * BaseFace::new(p, q).unwrap().inradius - 1e-9);
        let sizes: Vec<usize> = (0..hand.len()).map(|s| g.shell.iter().filter(|&&x| x == s).count()).collect();
        assert_eq!(sizes, hand, "{{{p},{q}}}");
    }
}

/// q >= 4 hand count. {5,4} with delta = -1: the core is a square (p_core = 4) whose angle alpha is
/// not pi/2 (cot(alpha/2) = cosh R tan(pi/4) = cosh R > 1, so alpha < pi/2). At each core vertex the
/// two-step walks give the 2 edge neighbours and two "opposite" faces, the core rotated by +-2 alpha
/// about the vertex, which differ since 4 alpha != 2 pi. Edge neighbours: 4 (each found from both
/// ends); opposite faces: 2 per vertex = 8. |S_1| = 1 + 4 + 8 = 13 (old edge-BFS: 1 + 4 = 5).
/// delta = +1: hexagon core, same argument: 1 + 6 + 12 = 19 (old: 7).
#[test]
fn wedge_q4_first_shell_hand_count() {
    for (delta, n_core, hand, old_hand) in [(-1, 4, 13, 5), (1, 6, 19, 7)] {
        let inr = ((PI / 4.0).cos() / (PI / 5.0).sin()).acosh();
        let rc = (inr.sinh() / (PI / n_core as f64).sin()).asinh();
        let alpha = 2.0 * (1.0 / (rc.cosh() * (PI / n_core as f64).tan())).atan();
        assert!((4.0 * alpha - 2.0 * PI).abs() > 0.1, "alpha = {alpha}");
        let h = build_hyperbolic_tiling(5, 4, HyperbolicStyle::RegularPQ, 1, delta).unwrap();
        assert_eq!(h.faces.len(), hand, "delta {delta}: {}", h.growth_rule);
        assert_eq!(h.faces[0].len(), n_core);
        assert_eq!(old_wedge_centroids(5, 4, 1, delta, 180, 0.985).0.len(), old_hand);
    }
}

/// The wedge path builds for every pair, style, delta and layer count, within the caps, with all
/// points in the open disk, and the kagome vertices (core-copy edge midpoints) as many as the edges.
#[test]
fn wedge_smoke_all_pairs() {
    let t = Instant::now();
    let mut worst = (0.0, (0, 0, 0, 0));
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            for delta in [-1, 1] {
                for lay in 1..=4 {
                    let t1 = Instant::now();
                    let h = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, lay, delta).unwrap();
                    let dt = t1.elapsed().as_secs_f64();
                    if dt > worst.0 {
                        worst = (dt, (p, q, delta as usize, lay));
                    }
                    assert!(h.faces.len() <= super::DRAW_CAPS.max_faces && h.face_shell.len() == h.faces.len());
                    assert!(h.poincare_pts.iter().all(|z| z[0] * z[0] + z[1] * z[1] < 1.0));
                }
            }
        }
    }
    let k = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RectifiedKagome, 2, -1).unwrap();
    let r = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, -1).unwrap();
    assert_eq!(k.poincare_pts.len(), r.edges.len());
    println!("# time wedge smoke: {:?}; slowest build {:.1} ms at {:?}", t.elapsed(), 1e3 * worst.0, worst.1);
}

// ---- letter-count counter ----

/// The letter counter equals the whole-word counter for every pair the app exposes, through the
/// largest n whose boundary word stays below ~3e6 letters (n >= 2 everywhere, n = 6 for slow growth).
#[test]
fn letter_counter_equals_word_counter_all_pairs() {
    let t = Instant::now();
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            let letters = vertex_completion_counts_combinatorial(p, q, 8).expect("hyperbolic, no overflow");
            let n_word = (1..=8).take_while(|&n| letters[n] * p as u64 <= 3_000_000).last().unwrap_or(1).min(6);
            let word = vertex_completion_counts_boundary_word(p, q, n_word).expect("convex");
            assert_eq!(&letters[..=n_word], &word[..], "{{{p},{q}}} n <= {n_word}");
            assert_eq!(letters[1], (p * (q - 2)) as u64, "hand count of shell 1");
        }
    }
    println!("# time letter vs word, all pairs: {:?}", t.elapsed());
}

/// What the atlas table shows: exact counts to n = 6 for all 92 pairs, no `None` (u64 is enough:
/// the largest, {12,12} a_6, is about 120 * 98^5 ~ 1.1e12). Timed; the geometric table it replaces
/// is timed in `ui_table_timing_geometric_before` (ignored, run by hand).
#[test]
fn ui_table_combinatorial_all_92_pairs() {
    let t = Instant::now();
    let mut pairs = 0;
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            let a = vertex_completion_counts_combinatorial(p, q, 6).expect("exact counts");
            assert_eq!(a.len(), 7);
            pairs += 1;
        }
    }
    assert_eq!(pairs, 92);
    println!("# time combinatorial table, 92 pairs to n = 6: {:?}", t.elapsed());
}

#[test]
#[ignore = "timing of the old geometric table (~5 s); run by hand"]
fn ui_table_timing_geometric_before() {
    let t = Instant::now();
    let mut worst = (0.0, (0, 0));
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            let t1 = Instant::now();
            let _ = vertex_completion_counts_budget(p, q, 6, 20_000);
            let dt = t1.elapsed().as_secs_f64();
            if dt > worst.0 {
                worst = (dt, (p, q));
            }
        }
    }
    println!("# time geometric table, 92 pairs: {:?}; worst {:.1} ms at {:?}", t.elapsed(), 1e3 * worst.0, worst.1);
}

#[test]
#[ignore = "pilot of the drawing caps; run by hand"]
fn pilot_draw_caps() {
    for (max_faces, rcap) in [(180, 0.985), (1000, 0.985), (1000, 0.995), (1000, 0.998), (1500, 0.998), (2000, 0.998)] {
        let caps = DrawCaps { max_faces, max_centre_radius: rcap };
        let t = Instant::now();
        let mut worst = (0.0, (0, 0, 0));
        // Pairs whose layers 1..4 give 4 (resp. >= 2) different face counts.
        let (mut four, mut two) = (0, 0);
        for p in 3..=12 {
            for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
                let mut counts = Vec::new();
                for lay in 1..=4 {
                    let t1 = Instant::now();
                    let h = build_hyperbolic_tiling_capped(p, q, HyperbolicStyle::RegularPQ, lay, 0, caps).unwrap();
                    counts.push(h.faces.len());
                    let dt = t1.elapsed().as_secs_f64();
                    if dt > worst.0 {
                        worst = (dt, (p, q, lay));
                    }
                }
                counts.dedup();
                four += usize::from(counts.len() == 4);
                two += usize::from(counts.len() >= 2);
            }
        }
        println!("caps {max_faces}/{rcap}: of 92 pairs, {four} have 4 different patches for layers 1..4, {two} at least 2");
        let row = |p, q| {
            (1..=4)
                .map(|l| {
                    let h = build_hyperbolic_tiling_capped(p, q, HyperbolicStyle::RegularPQ, l, 0, caps).unwrap();
                    format!("{}/S{}", h.faces.len(), h.complete_through_shell)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "caps {max_faces}/{rcap}: all 92x4 in {:?}, worst {:.1} ms at {:?}; {{5,4}} {}; {{7,3}} {}; {{3,7}} {}; {{4,5}} {}; {{8,8}} {}",
            t.elapsed(),
            1e3 * worst.0,
            worst.1,
            row(5, 4),
            row(7, 3),
            row(3, 7),
            row(4, 5),
            row(8, 8)
        );
    }
}

/// Measurement only: the old q = 3 (and q = 4) wedge runs' largest merged / smallest kept Euclidean
/// centroid distance, and the new grower's hyperbolic ones, printed (to choose the merge distance).
#[test]
#[ignore = "measurement; run by hand"]
fn measure_wedge_centre_separation() {
    for (p, q) in [(7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3), (5, 4), (6, 4)] {
        for delta in [-1, 1] {
            let (c, _, mm, mk) = old_wedge_stats(p, q, 4, delta, 1_000_000, 0.985);
            let p_core = (p as i32 + delta).clamp(3, 12) as usize;
            let inr = ((PI / q as f64).cos() / (PI / p as f64).sin()).acosh();
            let rc = 2.0 * ((0.5 * (inr.sinh() / (PI / p_core as f64).sin()).asinh()).tanh().min(0.96)).atanh();
            let g = grow_polygon_vertex_completion(&Polygon::regular(p_core, rc), q, 4, 1_000_000, 0.985, 1e-3);
            println!(
                "{{{p},{q}}} {delta:+}: old {} faces, merged <= {mm:.2e}, kept >= {mk:.2e} (Eucl.); new {} faces, merged <= {:.2e}, kept >= {:.2e} (hyp.), suspect {}",
                c.len(),
                g.faces.len(),
                g.max_merged,
                g.min_distinct,
                g.suspect
            );
        }
    }
}

/// Measurement only: largest drift off the hyperboloid, `|<x, x> + 1|`, of the vertices of faces whose
/// centre lies within Poincare radius `rcap`, over all 92 pairs at layers 4 (compute budget 8000).
#[test]
#[ignore = "measurement; run by hand"]
fn measure_drawn_drift() {
    for rcap in [0.985, 0.995, 0.998, 0.999] {
        let mut worst = (0.0_f64, (0, 0));
        for p in 3..=12 {
            for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
                let vc = grow_vertex_completion(p, q, 4, 8000).unwrap();
                for i in 0..vc.faces.len() {
                    let z = hyperboloid_to_poincare(vc.centre(i));
                    if z[0] * z[0] + z[1] * z[1] > rcap * rcap {
                        continue;
                    }
                    for x in vc.face_vertices(i) {
                        let d = (x[0] * x[0] + x[1] * x[1] - x[2] * x[2] + 1.0).abs();
                        if d > worst.0 {
                            worst = (d, (p, q));
                        }
                    }
                }
            }
        }
        println!("rcap {rcap}: max |<x,x>+1| = {:.2e} at {:?}", worst.0, worst.1);
    }
}
