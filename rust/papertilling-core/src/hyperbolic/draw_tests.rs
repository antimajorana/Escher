//! Tests of the drawn patch (caps, wedge path) and of the letter-count shell counter.
//!
//! Expected values are hand counts (derivations in the comments), BDF Eq. (10)-(11), or, for the
//! `q = 3` wedge regression, the pre-upgrade edge-BFS algorithm re-run at the edge-matched core
//! size ([`old_wedge_centroids`]).

use std::f64::consts::PI;
use std::time::Instant;

use super::inflation::*;
use super::wedge::{grow_polygon_vertex_completion, Polygon};
use super::{build_hyperbolic_tiling, build_hyperbolic_tiling_capped, orthogonal_circle, DrawCaps, HyperbolicStyle};

// ---- edge-matched copy of the pre-upgrade wedge edge-BFS (e09ee15 mod.rs), caps as parameters ----

#[test]
fn almost_radial_geodesic_keeps_distinct_endpoints() {
    let (a, b) = ([0.1, 0.0], [0.6, 1e-7]);
    let arc = super::sample_geodesic_arc(a, b, 8);
    assert_eq!((arc[0], arc[8]), (a, b));
    assert!(arc[4][0] > a[0] && arc[4][0] < b[0]);
    // In the Klein disk all sampled points lie on the line joining its endpoints.
    let (u, v) = (super::to_klein(a), super::to_klein(b));
    for z in arc {
        let p = super::to_klein(z);
        assert!(((p[0] - u[0]) * (v[1] - u[1]) - (p[1] - u[1]) * (v[0] - u[0])).abs() < 1e-14);
    }
}

fn c_sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
fn c_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]] }
fn c_conj(a: [f64; 2]) -> [f64; 2] { [a[0], -a[1]] }
fn c_norm2(a: [f64; 2]) -> f64 { a[0] * a[0] + a[1] * a[1] }

fn pdist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = c_norm2(c_sub(a, b)).sqrt();
    let den = c_norm2(c_sub([1.0, 0.0], c_mul(c_conj(a), b))).sqrt();
    2.0 * (d / den).min(1.0 - 1e-15).atanh()
}

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

/// Old wedge patch: BFS over reflections across edges, centroid radius cap, `max_faces` cap (mid-BFS),
/// and either pre-upgrade Euclidean `1e-3` centroid dedup (`hyp_dedup = false`) or hyperbolic `1e-3`
/// centre dedup (`hyp_dedup = true`).
fn old_wedge_centroids(
    p: usize,
    q: usize,
    lay: usize,
    wedge_delta: i32,
    max_faces: usize,
    rcap: f64,
    hyp_dedup: bool,
) -> (Vec<[f64; 2]>, bool) {
    let (c, capped, _, _) = old_wedge_stats_with(p, q, lay, wedge_delta, max_faces, rcap, hyp_dedup);
    (c, capped)
}

fn old_wedge_stats(p: usize, q: usize, lay: usize, wedge_delta: i32, max_faces: usize, rcap: f64) -> (Vec<[f64; 2]>, bool, f64, f64) {
    old_wedge_stats_with(p, q, lay, wedge_delta, max_faces, rcap, false)
}

fn old_wedge_stats_with(
    p: usize,
    q: usize,
    lay: usize,
    wedge_delta: i32,
    max_faces: usize,
    rcap: f64,
    hyp_dedup: bool,
) -> (Vec<[f64; 2]>, bool, f64, f64) {
    let (mut max_merged, mut min_kept) = (0.0_f64, f64::INFINITY);
    let p_core = (p as i32 + wedge_delta).clamp(3, 12) as usize;
    let half_edge_hyp = ((PI / (p as f64)).cos() / (PI / (q as f64)).sin()).max(1.0).acosh();
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
    let mut seen_hyp: Vec<[f64; 2]> = vec![[0.0, 0.0]];
    let mut capped = false;
    let mut bfs = std::collections::VecDeque::from([(0_usize, 0_usize, [0.0, 0.0])]);
    while let Some((pid, depth, hc0)) = bfs.pop_front() {
        if depth >= lay || poly_verts.len() >= max_faces {
            capped |= depth < lay;
            continue;
        }
        let cur_poly = poly_verts[pid].clone();
        let m = cur_poly.len();
        for e in 0..m {
            let (a, b) = (cur_poly[e], cur_poly[(e + 1) % m]);
            let next_poly: Vec<[f64; 2]> = cur_poly.iter().map(|&z| reflect_geodesic(z, a, b)).collect();
            let hc = reflect_geodesic(hc0, a, b);
            let cen = next_poly.iter().fold([0.0, 0.0], |acc, &v| [acc[0] + v[0], acc[1] + v[1]]);
            let cen = [cen[0] / (m as f64), cen[1] / (m as f64)];
            if c_norm2(cen) > rcap * rcap {
                capped = true;
                continue;
            }
            let dmin = if hyp_dedup {
                seen_hyp.iter().map(|&c| pdist(c, hc)).fold(f64::INFINITY, f64::min)
            } else {
                seen_centers.iter().map(|&c| (c[0] - cen[0]).hypot(c[1] - cen[1])).fold(f64::INFINITY, f64::min)
            };
            if dmin < 1e-3 {
                max_merged = max_merged.max(dmin);
                continue;
            }
            min_kept = min_kept.min(dmin);
            seen_centers.push(cen);
            seen_hyp.push(hc);
            poly_verts.push(next_poly);
            bfs.push_back((poly_verts.len() - 1, depth + 1, hc));
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

/// Bidirectional point-set agreement within `tol` (equal cardinality, every point of `a` within `tol`
/// of a partner in `b` and every point of `b` within `tol` of a partner in `a`).
fn same_point_sets(a: &[[f64; 2]], b: &[[f64; 2]], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter().all(|x| b.iter().any(|y| (x[0] - y[0]).hypot(x[1] - y[1]) < tol))
        && b.iter().all(|y| a.iter().any(|x| (x[0] - y[0]).hypot(x[1] - y[1]) < tol))
}

const NO_CAPS: DrawCaps = DrawCaps { max_faces: 1_000_000, max_centre_radius: 0.985 };

/// q = 3: the three faces at a vertex are pairwise edge-adjacent, so vertex adjacency is edge
/// adjacency and the new rule (one-step star walks) is edge-BFS. Under `1e-3` hyperbolic centre dedup
/// (`hyp_dedup = true`), all 48 cases (`p = 7..=12`, `delta = +-1`, `layers = 1..=4`, radius cap 0.985)
/// match empirically in face count (even though `HypPointSet` checks only 27 ambient cells vs the
/// reference's exhaustive scan). Raw per-face centroids from `grow_polygon_vertex_completion` match
/// `old` to `< 1e-9` on all 47/48 cases except `(10, +1, 4)` (`2333` faces, where a symmetric pair
/// `6.45e-5 < 1e-3` apart swaps representative, shifting its centroid by `~7.12e-5 < 1e-4`), proving
/// that the post-vertex-dedup shifts at `(7, -1, 4)` (`2.27e-5 < 3e-5`) and `(11, +1, 4)` (`5.51e-6 < 1e-5`)
/// come strictly from `< 1e-3` vertex dedup.
#[test]
fn wedge_q3_equals_old_edge_bfs() {
    let t = Instant::now();
    let (mut compared, mut max_raw_47) = (0, 0.0_f64);
    for p in 7..=12 {
        for delta in [-1, 1] {
            let p_core = (p as i32 + delta).clamp(3, 12) as usize;
            let half_edge_pq = ((PI / (p as f64)).cos() / (PI / 3.0).sin()).max(1.0).acosh();
            let rc_hyp = (half_edge_pq.sinh() / (PI / (p_core as f64)).sin()).asinh();
            let poly = Polygon::regular(p_core, rc_hyp);
            for lay in 1..=4 {
                let (old, _) = old_wedge_centroids(p, 3, lay, delta, 1_000_000, 0.985, true);
                let g = grow_polygon_vertex_completion(&poly, 3, lay, NO_CAPS.max_faces, NO_CAPS.max_centre_radius, 1e-3);
                let raw_new: Vec<[f64; 2]> = g.faces.iter().map(|w| {
                    let n = poly.n as f64;
                    poly.vertices.iter().fold([0.0, 0.0], |acc, &v| {
                        let z = hyperboloid_to_poincare(mat_vec(w, v));
                        [acc[0] + z[0] / n, acc[1] + z[1] / n]
                    })
                }).collect();
                let raw_tol = if (p, delta, lay) == (10, 1, 4) { 1e-4 } else { 1e-9 };
                assert!(same_point_sets(&old, &raw_new, raw_tol), "raw {{{p},3}} delta {delta} layers {lay}");
                if (p, delta, lay) != (10, 1, 4) {
                    let d = old.iter().map(|x| raw_new.iter().map(|y| (x[0] - y[0]).hypot(x[1] - y[1])).fold(f64::INFINITY, f64::min)).fold(0.0_f64, f64::max);
                    max_raw_47 = max_raw_47.max(d);
                }
                let h = build_hyperbolic_tiling_capped(p, 3, HyperbolicStyle::RegularPQ, lay, delta, NO_CAPS).unwrap();
                let new = centroids(&h.poincare_pts, &h.faces);
                let tol = match (p, delta, lay) {
                    (10, 1, 4) => 1e-4,
                    (7, -1, 4) => 3e-5,
                    (11, 1, 4) => 1e-5,
                    _ => 1e-9,
                };
                assert!(same_point_sets(&old, &new, tol), "{{{p},3}} delta {delta} layers {lay}: old {} vs new {}", old.len(), new.len());
                compared += 1;
            }
        }
    }
    println!("# time q=3 wedge regression ({compared} cases, max_raw_47={max_raw_47:.2e}): {:?}", t.elapsed());
}

/// Same comparison through the public entry point with its default caps against the literal
/// pre-upgrade Euclidean `1e-3` / `180`-face algorithm, where the old run was not capped.
#[test]
fn wedge_q3_default_caps_match_old_when_uncapped() {
    let mut compared = 0;
    for p in 7..=12 {
        for delta in [-1, 1] {
            for lay in 1..=4 {
                let (old, capped) = old_wedge_centroids(p, 3, lay, delta, 180, 0.985, false);
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

/// With a regular `p`-gon of the `{p, q}` circumradius the star walks + even-`q` half-turn close,
/// and the wedge grower equals BDF vertex completion:
/// * `{5,4}`: `1, 10, 40` and `{7,3}`: `1, 7, 21, 56` (hand counts in `inflation_tests`).
/// * `{4,5}`: `S_1` has `4` edge + `8` corner squares = `12`; boundary has `12` shared (`k = 2, o = 3`)
///   and `8` private corner (`k = 1, o = 4`) vertices -> `a_2 = 20 + 12*1 + 8*2 = 48`.
/// * `{3,7}`: `S_1` has `3` edge + `12` corner triangles = `15`; the `3` edge triangles touch `S_1` at
///   both corners (`3` vertices `k = 3, o = 4`), leaving `9` shared vertices (`k = 2, o = 5`) on `12`
///   boundary edges -> `a_2 = 12 + 3*2 + 9*3 = 45`.
#[test]
fn polygon_grower_on_regular_pq_is_bdf() {
    for (p, q, hand) in [(5, 4, vec![1, 10, 40]), (7, 3, vec![1, 7, 21, 56]), (4, 5, vec![1, 12, 48]), (3, 7, vec![1, 15, 45])] {
        let r = BaseFace::new(p, q).unwrap().circumradius;
        let g = grow_polygon_vertex_completion(&Polygon::regular(p, r), q, hand.len() - 1, 1_000_000, 0.9999, 1e-3);
        // Merges are numerical duplicates only; distinct centres are >= 2r apart (r the inradius).
        assert!(
            g.max_merged < 1e-9 && g.near_coincident_merged == 0 && g.near_coincident_kept == 0,
            "{{{p},{q}}}: merged {:.1e}",
            g.max_merged
        );
        assert!(g.min_distinct > 2.0 * BaseFace::new(p, q).unwrap().inradius - 1e-9);
        let sizes: Vec<usize> = (0..hand.len()).map(|s| g.shell.iter().filter(|&&x| x == s).count()).collect();
        assert_eq!(sizes, hand, "{{{p},{q}}}");
    }
}

/// Wedge first-shell hand counts (`walk = (q - 1) / 2` each side + centred opposite `H_{v_j}` for
/// even `q`) and edge-matched core size `cosh(l/2) = cos(pi/p) / sin(pi/q)`:
/// * `{5,4}` `delta = -1` (`p_core = 4`, `alpha ~ 76.345 deg`): `4*alpha != 360 deg` and
///   `5*alpha != 360 deg`. Each corner gets `1 + 1` edge copies (shared by neighbouring corners -> `4`)
///   and `1` centred opposite copy (`4`), so `|S_1| = 1 + 4 + 4 = 9`, keeping `C_4` + radial mirrors and
///   leaving equal gaps `pi - 2*alpha ~ 27.309 deg` on either side of each opposite copy.
/// * `{5,4}` `delta = +1` (`p_core = 6`): `|S_1| = 1 + 6 + 6 = 13`, keeping `C_6` + radial mirrors.
/// * `{4,6}` `delta = +1` (`p_core = 5`, `q = 6`): `1 + 5*(6 - 2) = 21`; `{3,7}` `delta = +1` (`p_core = 4`, `q = 7`): `21`.
#[test]
fn wedge_q4_first_shell_hand_count() {
    for (delta, n_core, hand) in [(-1, 4, 9), (1, 6, 13)] {
        let half_edge = ((PI / 5.0).cos() / (PI / 4.0).sin()).acosh();
        let rc = (half_edge.sinh() / (PI / n_core as f64).sin()).asinh();
        let alpha = 2.0 * (1.0 / (rc.cosh() * (PI / n_core as f64).tan())).atan();
        assert!((4.0 * alpha - 2.0 * PI).abs() > 0.1 && (5.0 * alpha - 2.0 * PI).abs() > 0.1, "alpha = {alpha}");
        let h = build_hyperbolic_tiling(5, 4, HyperbolicStyle::RegularPQ, 1, delta).unwrap();
        assert_eq!(h.faces.len(), hand, "delta {delta}: {}", h.growth_rule);
        assert_eq!(h.faces[0].len(), n_core);
        let cens = centroids(&h.poincare_pts, &h.faces);
        let rot = |c: [f64; 2]| {
            let th = 2.0 * PI / n_core as f64;
            [th.cos() * c[0] - th.sin() * c[1], th.sin() * c[0] + th.cos() * c[1]]
        };
        assert!(cens.iter().all(|&c| cens.iter().any(|&d| (rot(c)[0] - d[0]).hypot(rot(c)[1] - d[1]) < 1e-9)));
        for k in 0..n_core {
            let th = PI * k as f64 / n_core as f64;
            let refl = |c: [f64; 2]| {
                let s = 2.0 * (c[0] * th.cos() + c[1] * th.sin());
                [s * th.cos() - c[0], s * th.sin() - c[1]]
            };
            assert!(cens.iter().all(|&c| cens.iter().any(|&d| (refl(c)[0] - d[0]).hypot(refl(c)[1] - d[1]) < 1e-9)));
        }
        if delta == -1 {
            // Boost v0 = faces[0][0] = [r0, 0] to the origin via Poincare Mobius map z -> (z - r0)/(1 - r0*z),
            // where geodesic edges incident to v0 become straight rays at exact hyperbolic angles.
            let v0_id = h.faces[0][0];
            let r0 = h.poincare_pts[v0_id][0];
            let mob = |p: [f64; 2]| c_mul(c_sub(p, [r0, 0.0]), {
                let d = c_sub([1.0, 0.0], [r0 * p[0], r0 * p[1]]);
                let n2 = c_norm2(d);
                [d[0] / n2, -d[1] / n2]
            });
            let mut spans: Vec<(f64, f64)> = h.faces
                .iter()
                .filter_map(|f| {
                    let pos = f.iter().position(|&id| id == v0_id)?;
                    let a = mob(h.poincare_pts[f[(pos + 1) % n_core]]);
                    let b = mob(h.poincare_pts[f[(pos + n_core - 1) % n_core]]);
                    let (t1, t2) = (a[1].atan2(a[0]), b[1].atan2(b[0]));
                    let mid = (t1 + 0.5 * ((t2 - t1 + PI).rem_euclid(2.0 * PI) - PI)).rem_euclid(2.0 * PI);
                    let mid = if mid > PI { mid - 2.0 * PI } else { mid };
                    Some((mid - 0.5 * alpha, mid + 0.5 * alpha))
                })
                .collect();
            assert_eq!(spans.len(), 4, "core + 2 edge faces + 1 centred opposite at v0");
            spans.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Ordered by mid-angle around v0: E- (~ -103.65 deg), Opp (0 deg), E+ (~ +103.65 deg), F0 (pi).
            let opp = spans.iter().find(|s| ((s.0 + s.1) * 0.5).abs() < 1e-6).expect("centred opposite on +x ray");
            let above = spans.iter().find(|s| (s.0 - opp.1).abs() < 1.0 && s.0 > opp.1).unwrap();
            let below = spans.iter().find(|s| (opp.0 - s.1).abs() < 1.0 && opp.0 > s.1).unwrap();
            let (gap_plus, gap_minus) = (above.0 - opp.1, opp.0 - below.1);
            let want_gap = PI - 2.0 * alpha;
            assert!((gap_plus - want_gap).abs() < 1e-9 && (gap_minus - want_gap).abs() < 1e-9, "gaps {gap_plus} / {gap_minus} vs {want_gap}");
        }
    }
    assert_eq!(build_hyperbolic_tiling(4, 6, HyperbolicStyle::RegularPQ, 1, 1).unwrap().faces.len(), 21);
    assert_eq!(build_hyperbolic_tiling(3, 7, HyperbolicStyle::RegularPQ, 1, 1).unwrap().faces.len(), 21);
}

/// The wedge path builds for every pair, style, delta and layer count, within the caps, with all
/// points in the open disk, `|S_1| = 1 + p_core (q - 2)` on all 184 pairs, and `wedge + kagome`
/// keeping `|V_kagome| = |E_regular|` for both `q = 3` and `q = 4`.
#[test]
fn wedge_smoke_all_pairs() {
    let t = Instant::now();
    let mut worst = (0.0, (0, 0, 0, 0));
    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            for delta in [-1, 1] {
                let p_core = (p as i32 + delta).clamp(3, 12) as usize;
                let s1 = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, 1, delta).unwrap();
                assert_eq!(s1.faces.len(), 1 + p_core * (q - 2), "{{{p},{q}}} d={delta}");
                assert_eq!(s1.complete_through_shell, 1);
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
    for (p, q, lay) in [(7, 3, 2), (5, 4, 1)] {
        let k = build_hyperbolic_tiling(p, q, HyperbolicStyle::RectifiedKagome, lay, -1).unwrap();
        let r = build_hyperbolic_tiling(p, q, HyperbolicStyle::RegularPQ, lay, -1).unwrap();
        assert_eq!((k.poincare_pts.len(), k.faces.len(), k.face_shell), (r.edges.len(), r.faces.len(), r.face_shell));
        assert_ne!(k.poincare_pts, r.poincare_pts);
    }
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
            let half_edge = ((PI / p as f64).cos() / (PI / q as f64).sin()).max(1.0).acosh();
            let rc = (half_edge.sinh() / (PI / p_core as f64).sin()).asinh();
            let g = grow_polygon_vertex_completion(&Polygon::regular(p_core, rc), q, 4, 1_000_000, 0.985, 1e-3);
            println!(
                "{{{p},{q}}} {delta:+}: old {} faces, merged <= {mm:.2e}, kept >= {mk:.2e} (Eucl.); new {} faces, merged <= {:.2e}, kept >= {:.2e} (hyp.), suspect {}",
                c.len(),
                g.faces.len(),
                g.max_merged,
                g.min_distinct,
                g.near_coincident_merged + g.near_coincident_kept
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

/// H^2 rectified `(p.q.p.q)` kagome grown by the cluster rule:
/// * `p > q`: `(7,3)` gives hand-derived `8, 43, 141, 400` (`4` omitted at `1000`); `(5,4)` gives hand-derived `6, 26, 81` and regression pins `226, 606`.
/// * `p < q`: `(3,7)` (`P_0` = 3 heptagons around `F_0`) gives hand-derived `19, 76` and regression pins `229, 631` (`4` omitted); `F_0` stays at `(0,0,1)` in `{p,q}` orientation.
/// * `p == q`: `(5,5)` gives hand-derived `1, 11, 51, 201, 761` (`{5,4}` rotated by `pi/5`).
///
/// At `layers = 1` across all 92 pairs (and deeper shells on the four pairs above), every uncapped shell is a disk (`chi = 1`, `bdry_deg in {0, 2}`) with rectified edge length `l_k`.
#[test]
fn kagome_cluster_known_counts_and_disks() {
    let default_caps = DrawCaps { max_faces: 1000, max_centre_radius: 0.998 };
    let seq = |p, q, lay| build_hyperbolic_tiling_capped(p, q, HyperbolicStyle::RectifiedKagome, lay, 0, default_caps).unwrap();
    for (lay, want) in [(1, 43), (2, 141), (3, 400), (4, 400)] {
        let h = seq(7, 3, lay);
        assert_eq!(h.faces.len(), want, "(7,3) L={lay}");
        assert_eq!(h.complete_through_shell, lay.min(3));
    }
    for (lay, want) in [(1, 26), (2, 81), (3, 226), (4, 606)] {
        assert_eq!((seq(5, 4, lay).faces.len(), seq(5, 4, lay).complete_through_shell), (want, lay));
    }
    for (lay, want) in [(1, 76), (2, 229), (3, 631), (4, 631)] {
        let h = seq(3, 7, lay);
        assert_eq!((h.faces.len(), h.complete_through_shell), (want, lay.min(3)));
    }
    for (lay, want) in [(1, 11), (2, 51), (3, 201), (4, 761)] {
        assert_eq!((seq(5, 5, lay).faces.len(), seq(5, 5, lay).complete_through_shell), (want, lay));
    }
    // Orientation check for p < q: F_0 vertices on (3,7) match BaseFace(3,7).edge_midpoints at origin.
    let k37 = seq(3, 7, 1);
    let b37 = BaseFace::new(3, 7).unwrap();
    let f0_pts: Vec<[f64; 2]> = k37.faces[0].iter().map(|&i| k37.poincare_pts[i]).collect();
    let exp_f0: Vec<[f64; 2]> = b37.edge_midpoints.iter().map(|&v| hyperboloid_to_poincare(v)).collect();
    assert!(same_point_sets(&f0_pts, &exp_f0, 1e-12), "F0 orientation mismatch on (3,7)");

    for p in 3..=12 {
        for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
            let h = build_hyperbolic_tiling(p, q, HyperbolicStyle::RectifiedKagome, 1, 0).unwrap();
            let chi = h.poincare_pts.len() as i64 - h.edges.len() as i64 + h.faces.len() as i64;
            assert_eq!(chi, 1, "({p}.{q}.{p}.{q}) chi = {chi}");
            let mut ecnt = std::collections::HashMap::new();
            for f in &h.faces {
                for k in 0..f.len() {
                    let (u, v) = (f[k], f[(k + 1) % f.len()]);
                    *ecnt.entry((u.min(v), u.max(v))).or_insert(0usize) += 1;
                }
            }
            let mut bdeg = vec![0usize; h.poincare_pts.len()];
            for (&(u, v), &c) in &ecnt {
                if c == 1 {
                    bdeg[u] += 1;
                    bdeg[v] += 1;
                }
            }
            assert!(bdeg.iter().all(|&d| d == 0 || d == 2), "({p}.{q}.{p}.{q}) pinch vertex");
            let lk = if p == q {
                BaseFace::new(p, 4).unwrap().edge_length()
            } else {
                let b = if p > q { BaseFace::new(p, q).unwrap() } else { BaseFace::new(q, p).unwrap() };
                hyp_dist(b.edge_midpoints[0], b.edge_midpoints[1])
            };
            assert!(h.edges.iter().all(|&(u, v)| (pdist(h.poincare_pts[u], h.poincare_pts[v]) - lk).abs() < 1e-5 * lk));
        }
    }
}

#[test]
fn hyperboloid_projection_preserves_outer_disk_sites() {
    for r in [0.96, 0.98, 0.995] {
        let [x, y, t] = super::to_hyperboloid([r, 0.0]);
        assert!((x / (1.0 + t) - r).abs() < 1e-12);
        assert_eq!(y, 0.0);
        assert!((t*t - x*x - 1.0).abs() < 1e-9);
    }
}
