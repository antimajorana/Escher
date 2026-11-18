//! Known-answer tests (ITA plane-group diagrams) and stored-lattice property tests for
//! `symmetry_elements`.

use std::collections::BTreeSet;

use crate::plane::cell::{mat2_inv, mat2_vec};
use crate::plane::family::LatticeFamily;
use crate::plane::groups::has_rotations;
use crate::plane::stored_lattices::stored_lattices;
use crate::plane::symmetry_elements::{
    MAX_REPEAT, SymLineKind, SymmetryElements2D, compute_symmetry_elements, group_isometries_in_cell,
};
use crate::types::{Cell2D, Frame2D, Isometry2};

fn cell(a1: [f64; 2], a2: [f64; 2]) -> Cell2D {
    Cell2D { a1, a2, basis: Vec::new(), bonds: Vec::new() }
}

fn square() -> Cell2D {
    cell([1.0, 0.0], [0.0, 1.0])
}

fn hex() -> Cell2D {
    cell([1.0, 0.0], [-0.5, 0.5 * 3.0_f64.sqrt()])
}

fn elements(group: &str, c: &Cell2D, repeat: [i32; 2]) -> SymmetryElements2D {
    compute_symmetry_elements(group, c, Frame2D::default(), repeat).unwrap()
}

/// Conventional fractional coordinates (basis `c1, c2`) reduced mod 1, in units of 1/12.
fn frac12(p: [f64; 2], c1: [f64; 2], c2: [f64; 2]) -> (i32, i32) {
    let inv = mat2_inv([[c1[0], c2[0]], [c1[1], c2[1]]]).unwrap();
    let f = mat2_vec(inv, p);
    let r = |x: f64| ((x - x.floor()) * 12.0).round() as i32 % 12;
    (r(f[0]), r(f[1]))
}

/// Rotation centres per conventional cell: `(12 f_x, 12 f_y, order, wyckoff)`.
fn centre_classes(el: &SymmetryElements2D, c1: [f64; 2], c2: [f64; 2]) -> BTreeSet<(i32, i32, u8, String)> {
    el.centres
        .iter()
        .map(|c| {
            let (x, y) = frac12(c.pos, c1, c2);
            (x, y, c.order, c.wyckoff.clone())
        })
        .collect()
}

fn set(items: &[(i32, i32, u8, &str)]) -> BTreeSet<(i32, i32, u8, String)> {
    items.iter().map(|&(x, y, o, w)| (x, y, o, w.to_string())).collect()
}

/// Line direction in degrees modulo 180, rounded.
fn angle(a: [f64; 2], b: [f64; 2]) -> i32 {
    let deg = (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees();
    (deg.rem_euclid(180.0).round() as i32) % 180
}

fn directions(el: &SymmetryElements2D, kind: SymLineKind) -> BTreeSet<i32> {
    el.lines.iter().filter(|l| l.kind == kind).map(|l| angle(l.a, l.b)).collect()
}

#[test]
fn p1_has_no_elements() {
    let el = elements("p1", &cell([1.0, 0.0], [0.3, 1.2]), [3, 3]);
    assert!(el.lines.is_empty() && el.centres.is_empty());
}

#[test]
fn p2_four_twofold_centres_1a_to_1d() {
    let c = cell([1.0, 0.0], [0.3, 1.1]);
    let el = elements("p2", &c, [3, 3]);
    let want = set(&[(0, 0, 2, "1a"), (0, 6, 2, "1b"), (6, 0, 2, "1c"), (6, 6, 2, "1d")]);
    assert_eq!(centre_classes(&el, c.a1, c.a2), want);
    // Hand count on the closed asymmetric window s in [-1, 1], t in [-1, 2]: 5 x 7 half-integer points.
    assert_eq!(elements("p2", &c, [2, 3]).centres.len(), 35);
    assert!(el.lines.is_empty());
}

#[test]
fn pm_and_pg_parallel_lines_two_per_cell() {
    let c = cell([1.0, 0.0], [0.0, 1.37]);
    let pm = elements("pm", &c, [1, 1]);
    assert!(pm.centres.is_empty());
    let mut xs: Vec<i64> = pm.lines.iter().map(|l| (l.a[0] * 1e6).round() as i64).collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![0, 500_000, 1_000_000], "mirrors x = 0, 1/2 (and 1 on the closed window)");
    for l in &pm.lines {
        assert_eq!(l.kind, SymLineKind::Mirror);
        assert_eq!(angle(l.a, l.b), 90);
        assert_eq!(l.glide, [0.0, 0.0]);
    }
    let pg = elements("pg", &c, [1, 1]);
    assert_eq!(pg.lines.len(), 3);
    for l in &pg.lines {
        assert_eq!(l.kind, SymLineKind::Glide);
        assert_eq!(angle(l.a, l.b), 90);
        assert!(l.glide[0].abs() < 1e-12 && (l.glide[1].abs() - 0.5 * 1.37).abs() < 1e-12, "{:?}", l.glide);
        let fx = l.a[0] * 2.0;
        assert!((fx - fx.round()).abs() < 1e-12, "glide at x = {}", l.a[0]);
    }
}

#[test]
fn cm_alternating_mirrors_and_glides() {
    let (c1, c2) = ([1.0, 0.0], [0.0, 1.6]);
    // Primitive basis B = C * U with U = [[1/2, -1/2], [1/2, 1/2]].
    let p = cell([0.5, 0.8], [-0.5, 0.8]);
    let el = elements("cm", &p, [4, 4]);
    assert!(el.centres.is_empty());
    let (mut n_m, mut n_g) = (0, 0);
    for l in &el.lines {
        assert_eq!(angle(l.a, l.b), 90);
        let q = (frac12(l.a, c1, c2).0, l.kind);
        match q {
            (0 | 6, SymLineKind::Mirror) => n_m += 1,
            (3 | 9, SymLineKind::Glide) => {
                assert!((l.glide[1].abs() - 0.8).abs() < 1e-12);
                n_g += 1;
            }
            other => panic!("unexpected line {other:?} at {:?}", l.a),
        }
    }
    assert!(n_m > 0 && n_g > 0);
}

#[test]
fn cmm_centres_on_primitive_rhombic_cell() {
    // ITA c2mm (No. 9): 2a (0,0)+(1/2,1/2), 2b (0,1/2)+(1/2,0), 4c (1/4,1/4),(3/4,1/4)+centring; all 2-fold.
    let (c1, c2) = ([1.0, 0.0], [0.0, 1.6]);
    let el = elements("cmm", &cell([0.5, 0.8], [-0.5, 0.8]), [4, 4]);
    let want = set(&[
        (0, 0, 2, "2a"),
        (6, 6, 2, "2a"),
        (0, 6, 2, "2b"),
        (6, 0, 2, "2b"),
        (3, 3, 2, "4c"),
        (9, 3, 2, "4c"),
        (3, 9, 2, "4c"),
        (9, 9, 2, "4c"),
    ]);
    assert_eq!(centre_classes(&el, c1, c2), want);
}

#[test]
fn app_maximum_window_hand_counts() {
    let t0 = std::time::Instant::now();
    // p3, window [-4, 5]^2: 10 x 10 integer points (1a) + 9 x 9 each for 1b and 1c.
    let p3 = elements("p3", &hex(), [9, 9]);
    assert_eq!(p3.centres.len(), 100 + 81 + 81);
    // pm, window x in [-4, 5]: mirrors at x = -4, -3.5, ..., 5.
    let pm = elements("pm", &cell([1.0, 0.0], [0.0, 1.37]), [9, 9]);
    assert_eq!(pm.lines.len(), 19);
    println!("# time p3 + pm at repeat [9, 9]: {:?}", t0.elapsed());
}

#[test]
fn p4_centres_1a_1b_fourfold_2c_twofold() {
    let el = elements("p4", &square(), [3, 3]);
    let want = set(&[(0, 0, 4, "1a"), (6, 6, 4, "1b"), (6, 0, 2, "2c"), (0, 6, 2, "2c")]);
    assert_eq!(centre_classes(&el, [1.0, 0.0], [0.0, 1.0]), want);
}

#[test]
fn p3_and_p6_centres_in_hexagonal_cell() {
    let h = hex();
    let p3 = elements("p3", &h, [3, 3]);
    assert_eq!(centre_classes(&p3, h.a1, h.a2), set(&[(0, 0, 3, "1a"), (4, 8, 3, "1b"), (8, 4, 3, "1c")]));
    let p6 = elements("p6", &h, [3, 3]);
    let want = set(&[
        (0, 0, 6, "1a"),
        (4, 8, 3, "2b"),
        (8, 4, 3, "2b"),
        (6, 0, 2, "3c"),
        (0, 6, 2, "3c"),
        (6, 6, 2, "3c"),
    ]);
    assert_eq!(centre_classes(&p6, h.a1, h.a2), want);
}

#[test]
fn p4m_p6m_mirror_directions_and_p4g_lines() {
    let p4m = elements("p4m", &square(), [2, 2]);
    assert_eq!(directions(&p4m, SymLineKind::Mirror), BTreeSet::from([0, 45, 90, 135]));
    let p6m = elements("p6m", &hex(), [2, 2]);
    assert_eq!(directions(&p6m, SymLineKind::Mirror), BTreeSet::from([0, 30, 60, 90, 120, 150]));
    // p4g (ITA No. 12): mirrors only along the diagonals; glide lines along both axes.
    let p4g = elements("p4g", &square(), [2, 2]);
    assert_eq!(directions(&p4g, SymLineKind::Mirror), BTreeSet::from([45, 135]));
    let g = directions(&p4g, SymLineKind::Glide);
    assert!(g.contains(&0) && g.contains(&90), "{g:?}");
}

fn on_mirror(el: &SymmetryElements2D, p: [f64; 2]) -> bool {
    el.lines.iter().filter(|l| l.kind == SymLineKind::Mirror).any(|l| {
        let d = [l.b[0] - l.a[0], l.b[1] - l.a[1]];
        let cr = d[0] * (p[1] - l.a[1]) - d[1] * (p[0] - l.a[0]);
        cr.abs() / d[0].hypot(d[1]) < 1e-9
    })
}

#[test]
fn p3m1_vs_p31m_mirror_incidence() {
    let h = hex();
    let a = elements("p3m1", &h, [3, 3]);
    let threefold: Vec<_> = a.centres.iter().filter(|c| c.order == 3).collect();
    assert!(!threefold.is_empty());
    assert!(threefold.iter().all(|c| on_mirror(&a, c.pos)), "p3m1: every 3-fold centre lies on a mirror");
    let b = elements("p31m", &h, [3, 3]);
    let (mut n_a, mut n_b) = (0, 0);
    for c in b.centres.iter().filter(|c| c.order == 3) {
        match c.wyckoff.as_str() {
            "1a" => {
                assert!(on_mirror(&b, c.pos));
                n_a += 1;
            }
            "2b" => {
                assert!(!on_mirror(&b, c.pos));
                n_b += 1;
            }
            w => panic!("p31m: unexpected 3-fold Wyckoff {w}"),
        }
    }
    assert!(n_a > 0 && n_b > 0);
}

#[test]
fn invalid_inputs_are_errors() {
    let f = Frame2D::default();
    assert!(compute_symmetry_elements("p7", &square(), f, [1, 1]).is_err());
    assert!(compute_symmetry_elements("p4", &cell([1.0, 0.0], [2.0, 0.0]), f, [1, 1]).is_err());
    assert!(compute_symmetry_elements("p4", &cell([f64::NAN, 0.0], [0.0, 1.0]), f, [1, 1]).is_err());
    assert!(compute_symmetry_elements("p4", &square(), f, [0, 1]).is_err());
}

fn approx_mat(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> bool {
    (0..2).all(|r| (0..2).all(|c| (a[r][c] - b[r][c]).abs() < 1e-9))
}

fn lattice_diff(c: &Cell2D, d: [f64; 2]) -> bool {
    let f = mat2_vec(mat2_inv(c.mat_a()).unwrap(), d);
    (f[0] - f[0].round()).abs() < 1e-8 && (f[1] - f[1].round()).abs() < 1e-8
}

fn same_mod_lattice(c: &Cell2D, x: &Isometry2, y: &Isometry2) -> bool {
    approx_mat(x.m, y.m) && lattice_diff(c, [x.t[0] - y.t[0], x.t[1] - y.t[1]])
}

fn test_frames() -> Vec<Frame2D> {
    let (c, s) = (0.37_f64.cos(), 0.37_f64.sin());
    vec![
        Frame2D::default(),
        Frame2D { q: [[c, -s], [s, c]], origin: [0.31, -0.17] },
        Frame2D { q: [[c, s], [s, -c]], origin: [-1.4, 2.25] },
    ]
}

#[test]
fn cartesian_ops_agree_with_family_ops_in_cell() {
    let mut seen = BTreeSet::new();
    let mut n = 0;
    for lat in stored_lattices() {
        if !seen.insert(lat.group.clone()) && !lat.label.starts_with("named/") {
            continue;
        }
        let mut frames = test_frames();
        frames.push(lat.frame);
        for fr in frames {
            let fam = LatticeFamily::from_recipe(&lat.recipe, Some(fr)).unwrap();
            let c = fam.cell();
            let mine = group_isometries_in_cell(&lat.group, &c, fr).unwrap();
            let theirs = fam.ops_in_cell().unwrap();
            for x in &mine {
                assert!(theirs.iter().any(|y| same_mod_lattice(&c, x, y)), "{}: extra op {x:?}", lat.label);
            }
            for y in &theirs {
                assert!(mine.iter().any(|x| same_mod_lattice(&c, x, y)), "{}: missing op {y:?}", lat.label);
            }
            n += 1;
        }
    }
    assert!(seen.len() == 17 && n > 17 * 4, "groups {} cases {n}", seen.len());
}

fn maps_sites(c: &Cell2D, iso: &Isometry2) -> bool {
    let inv = mat2_inv(c.mat_a()).unwrap();
    let fr: Vec<[f64; 2]> = c.basis.iter().map(|&p| mat2_vec(inv, p)).collect();
    c.basis.iter().all(|&p| {
        let g = mat2_vec(inv, iso.apply(p));
        fr.iter().any(|q| {
            let d = [g[0] - q[0], g[1] - q[1]];
            (d[0] - d[0].round()).abs() < 1e-6 && (d[1] - d[1].round()).abs() < 1e-6
        })
    })
}

/// Checks one lattice and returns its `(order, wyckoff)` centre classes.
fn check_lattice(label: &str, group: &str, c: &Cell2D, fr: Frame2D) -> BTreeSet<(u8, String)> {
    let el = compute_symmetry_elements(group, c, fr, [3, 3]).unwrap_or_else(|e| panic!("{label}: {e}"));
    let has_refl = group_isometries_in_cell(group, c, fr).unwrap().iter().any(|o| {
        o.m[0][0] * o.m[1][1] - o.m[0][1] * o.m[1][0] < 0.0
    });
    assert_eq!(!el.lines.is_empty(), has_refl, "{label}: line presence");
    assert_eq!(!el.centres.is_empty(), has_rotations(group).unwrap(), "{label}: centre presence");
    for l in &el.lines {
        assert!(maps_sites(c, &l.isometry()), "{label}: {:?} line {l:?} breaks the site set", l.kind);
    }
    for r in &el.centres {
        assert!(maps_sites(c, &r.isometry()), "{label}: centre {r:?} breaks the site set");
    }
    el.centres.iter().map(|r| (r.order, r.wyckoff.clone())).collect()
}

#[test]
fn every_stored_lattice_is_invariant_under_its_elements() {
    let t0 = std::time::Instant::now();
    let all = stored_lattices();
    let skew = test_frames()[2];
    let mut n = 0;
    for lat in all {
        let own = check_lattice(&lat.label, &lat.group, &lat.cell, lat.frame);
        let fam = LatticeFamily::from_recipe(&lat.recipe, Some(skew)).unwrap();
        let skewed = check_lattice(&lat.label, &lat.group, &fam.cell(), skew);
        assert_eq!(own, skewed, "{}: Wyckoff classes depend on the frame", lat.label);
        n += 1;
    }
    println!("# time symmetry elements over {n} stored lattices x 2 frames: {:?}", t0.elapsed());
    assert!(n >= 500, "only {n} lattices");
}

fn seg_len(l: &crate::plane::symmetry_elements::SymLine2D) -> f64 {
    (l.b[0] - l.a[0]).hypot(l.b[1] - l.a[1])
}

#[test]
fn p4m_unit_window_has_no_corner_slivers() {
    // ITA p4mm on [0,1]^2: mirrors x, y = 0, 1/2, 1 and the diagonals y = x, y = 1 - x; diagonal glides
    // (glide (1/2, 1/2)) on y = x +- 1/2, y = 1/2 - x, y = 3/2 - x. Lines y = x +- 1, y = -x, y = 2 - x
    // only touch corners and must not appear.
    let el = elements("p4m", &square(), [1, 1]);
    let h = 0.5_f64.sqrt();
    let mirrors: Vec<_> = el.lines.iter().filter(|l| l.kind == SymLineKind::Mirror).collect();
    let glides: Vec<_> = el.lines.iter().filter(|l| l.kind == SymLineKind::Glide).collect();
    assert_eq!((mirrors.len(), glides.len()), (8, 4), "{:?}", el.lines);
    for l in &el.lines {
        assert!(seg_len(l) > h - 1e-9, "short segment {l:?}");
    }
    for g in &glides {
        assert!((g.glide[0].abs() - 0.5).abs() < 1e-12 && (g.glide[1].abs() - 0.5).abs() < 1e-12, "{g:?}");
    }
}

#[test]
fn huge_cell_lines_are_finite() {
    let s = 1e15;
    let el = elements("pm", &cell([s, 0.0], [0.0, s]), [1, 1]);
    let mut xs: Vec<f64> = el.lines.iter().map(|l| l.a[0] / s).collect();
    xs.sort_by(f64::total_cmp);
    assert_eq!(xs.len(), 3, "{:?}", el.lines);
    for (x, want) in xs.iter().zip([0.0, 0.5, 1.0]) {
        assert!((x - want).abs() < 1e-12, "{xs:?}");
    }
    for l in &el.lines {
        assert!(l.a.iter().chain(l.b.iter()).all(|v| v.is_finite()));
        assert!((seg_len(l) / s - 1.0).abs() < 1e-12);
    }
}

#[test]
fn far_frame_origin_is_reduced_modulo_the_lattice() {
    // p2 on [0,1]^2: 2-fold centres at the 3 x 3 half-integer points; an integer origin shift changes nothing.
    let far = |o: [f64; 2]| {
        compute_symmetry_elements("p2", &square(), Frame2D { q: [[1.0, 0.0], [0.0, 1.0]], origin: o }, [1, 1]).unwrap()
    };
    let el = far([1e10, 0.0]);
    assert_eq!(el.centres.len(), 9);
    let want = set(&[(0, 0, 2, "1a"), (0, 6, 2, "1b"), (6, 0, 2, "1c"), (6, 6, 2, "1d")]);
    assert_eq!(centre_classes(&el, [1.0, 0.0], [0.0, 1.0]), want);
    // Origin at x = 1/4 (mod 1): centres at x = 1/4, 3/4 and y = 0, 1/2, 1.
    let el = far([1e10 + 0.25, 0.0]);
    assert_eq!(el.centres.len(), 6);
    let want = set(&[(3, 0, 2, "1a"), (3, 6, 2, "1b"), (9, 0, 2, "1c"), (9, 6, 2, "1d")]);
    assert_eq!(centre_classes(&el, [1.0, 0.0], [0.0, 1.0]), want);
}

#[test]
fn repeat_cap_and_large_window_hand_count() {
    assert!(compute_symmetry_elements("p6", &hex(), Frame2D::default(), [MAX_REPEAT + 1, 1]).is_err());
    let t0 = std::time::Instant::now();
    // p6 on an R x R window: (R+1)^2 six-fold (1a), 2R^2 three-fold (2b), 2R(R+1) + R^2 two-fold (3c).
    let r = i64::from(MAX_REPEAT);
    let el = elements("p6", &hex(), [MAX_REPEAT, MAX_REPEAT]);
    let count = |o: u8| el.centres.iter().filter(|c| c.order == o).count() as i64;
    assert_eq!((count(6), count(3), count(2)), ((r + 1) * (r + 1), 2 * r * r, 2 * r * (r + 1) + r * r));
    let pm = elements("p6m", &hex(), [MAX_REPEAT, MAX_REPEAT]);
    assert!(!pm.lines.is_empty());
    println!("# time p6 + p6m at repeat [{MAX_REPEAT}, {MAX_REPEAT}]: {:?}", t0.elapsed());
}
