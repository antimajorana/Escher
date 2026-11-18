//! Known-answer tests of the Dirichlet seed, the most symmetric point and the boundary statistics of
//! `plane::inflation`. Expected values come from hand arguments written next to them.

use std::collections::{BTreeMap, HashSet};

use super::inflation::{CentreKind, CentreSet, InflationRule, InflationSeed, PlaneTiling, inflate, inflate_with};
use super::test_fixtures::named;

fn centroid(poly: &[[f64; 2]]) -> [f64; 2] {
    let k = poly.len() as f64;
    [poly.iter().map(|p| p[0]).sum::<f64>() / k, poly.iter().map(|p| p[1]).sum::<f64>() / k]
}

/// Whether the point set `pts` is mapped onto itself by `op` (images matched within 1e-6).
fn invariant(pts: &[[f64; 2]], op: impl Fn([f64; 2]) -> [f64; 2]) -> bool {
    pts.iter().all(|&p| {
        let q = op(p);
        pts.iter().any(|r| (r[0] - q[0]).hypot(r[1] - q[1]) < 1e-6)
    })
}

/// `(key, site-symmetry order, rotation order k, mirrors through the point)` of the centre each named lattice
/// should pick, with the face counts of its Dirichlet seed `(faces, faces of the centre's polygon size)`.
/// * square (p4m): face centre and vertex both 4mm (order 8); the face centroid wins the tie. One face orbit, its
///   nearest translate to the centre is the face itself (the next ones are a lattice vector away): 1 square.
/// * honeycomb (p6m): hexagon centre 6mm (12) beats a site 3m (6) and a bond midpoint 2mm (4). 1 hexagon.
/// * kagome (p6m): hexagon centre 6mm (12) beats the triangle centres 3m (6) and the sites 2mm (4). The up
///   triangles nearest the centre are the 3 around the hexagon, equidistant, all kept as ties; likewise the 3 down
///   ones: hexagon + 6 triangles = 7.
/// * shastry_sutherland (p4g, square-grid realization: a1 = (2, 0), a2 = (0, 2), sites (2,0), (1,1), (2,1),
///   (1,0), diagonals in the plaquettes centred (2.5, 0.5) along (1, 1) and (1.5, 1.5) along (-1, 1)): plain
///   squares centred (1.5, 0.5) + Z(1,1) are 4-fold points without mirrors (order 4; the 90-degree rotation about
///   (1.5, 0.5) carries the (1,1) diagonal at (2.5, 0.5) to the (-1,1) one at (1.5, 1.5), a mirror through it
///   does not); dimer centres are 2mm (order 4) but bond midpoints, so the face centroid wins. Relative to c, the
///   Dirichlet cell is [-1, 1]^2: the centre square, the other plain square orbit at the 4 corners (+-1, +-1)
///   (ties), and one triangle per orbit, centroids (5/6, 1/6) and its rotations by 90 degrees (the triangle
///   (0.5,-0.5), (1.5,0.5), (0.5,0.5) of the dimer at (1, 0)): 1 + 4 + 4 = 9 faces, 5 squares.
const CASES: [(&str, usize, usize, bool, usize, usize); 4] = [
    ("square", 8, 4, true, 1, 1),
    ("honeycomb", 12, 6, true, 1, 1),
    ("kagome", 12, 6, true, 7, 1),
    ("shastry_sutherland", 4, 4, false, 9, 5),
];

#[test]
fn dirichlet_seed_known_counts() {
    for (key, order, _, _, faces, big) in CASES {
        let cell = named(key).cell;
        let t = PlaneTiling::new(&cell).unwrap();
        let sp = t.most_symmetric_point();
        assert_eq!(sp.order, order, "{key}");
        let CentreKind::Face(f) = sp.kind else { panic!("{key}: centre {sp:?} is not a face centroid") };
        let size = t.face_sizes()[f];
        assert_eq!(size, *t.face_sizes().iter().max().unwrap(), "{key}");
        let res = inflate(&cell, InflationSeed::Dirichlet, 0).unwrap();
        let s0 = &res.shells[0];
        assert_eq!((s0.topology.faces, s0.filled, s0.topology.chi), (faces, 0, 1), "{key}");
        assert_eq!(res.faces.iter().filter(|g| g.polygon.len() == size).count(), big, "{key}");
        // An explicit centre gives the same seed.
        let at = inflate(&cell, InflationSeed::DirichletAt(sp.point), 0).unwrap();
        assert_eq!(at.shells[0].topology.faces, faces, "{key}");
    }
}

#[test]
fn dirichlet_seed_symmetry_preserved_through_n2() {
    // The seed is G_c-invariant (G_c the site-symmetry group of c), vertex completion, the cluster rule and
    // FillHoles commute with G_c, so every shell is G_c-invariant. Checked with hand-written operations about c:
    // the rotation by 2 pi / k and, where the point group has mirrors, the mirror along a1 (for p4m and p6m the
    // lattice direction a1 through a 4mm / 6mm point is a mirror line).
    for (key, _, k, mirror, _, _) in CASES {
        let cell = named(key).cell;
        let c = PlaneTiling::new(&cell).unwrap().most_symmetric_point().point;
        let th = 2.0 * std::f64::consts::PI / k as f64;
        let rot = |p: [f64; 2]| {
            let d = [p[0] - c[0], p[1] - c[1]];
            [c[0] + th.cos() * d[0] - th.sin() * d[1], c[1] + th.sin() * d[0] + th.cos() * d[1]]
        };
        let u = { let n = cell.a1[0].hypot(cell.a1[1]); [cell.a1[0] / n, cell.a1[1] / n] };
        let refl = |p: [f64; 2]| {
            let d = [p[0] - c[0], p[1] - c[1]];
            let s = 2.0 * (d[0] * u[0] + d[1] * u[1]);
            [c[0] + s * u[0] - d[0], c[1] + s * u[1] - d[1]]
        };
        for rule in [InflationRule::Bdf, InflationRule::Cluster(CentreSet::MaxPolygon)] {
            let res = inflate_with(&cell, InflationSeed::Dirichlet, &rule, 2).unwrap();
            for n in 0..=2 {
                let pts: Vec<[f64; 2]> = res.faces.iter().filter(|f| f.shell <= n).map(|f| centroid(&f.polygon)).collect();
                assert!(invariant(&pts, rot), "{key} {rule:?} n={n}: not {k}-fold symmetric");
                assert!(!mirror || invariant(&pts, refl), "{key} {rule:?} n={n}: not mirror symmetric");
            }
        }
    }
}

#[test]
fn boundary_statistics_square_and_triangular() {
    // {4,4} face seed: S_n is the a x a block, a = 2n + 1. Its boundary vertices are the 4 corners (1 square each)
    // and 4(a - 1) - 4 = 8n side vertices (2 squares each); the boundary tiles are the 4(a - 1) = 8n squares of the
    // outer ring (n >= 1), 1 for the single square.
    let res = inflate(&named("square").cell, InflationSeed::Face(0), 4).unwrap();
    for s in &res.shells {
        let mut want = BTreeMap::from([(1, 4)]);
        if s.n > 0 {
            want.insert(2, 8 * s.n);
        }
        assert_eq!(s.boundary_vertex_types, want, "square n={}", s.n);
        assert_eq!(s.topology.boundary_tiles, if s.n == 0 { 1 } else { 8 * s.n }, "square n={}", s.n);
    }
    // {3,6} face seed: S_n (n >= 1) is the lattice hexagon with sides alternating (n+1, n), all corners 120
    // degrees. Perimeter 3(2n + 1) = 6n + 3 unit edges and as many boundary vertices: 6 corners (2 triangles each)
    // and 6n - 3 mid-side vertices (3 triangles each). Each boundary edge lies on its own triangle (at a 120-degree
    // corner the two boundary edges are on the 2 different corner triangles), so 6n + 3 tiles have a boundary edge.
    // A tile touching the boundary only at a vertex does not count: along a side of length L there are 2L - 1 such
    // strip triangles, so the vertex definition would give 2(6n + 3) - 6 = 12n (12 vs 9 at n = 1, where only the
    // seed triangle is interior). n = 0: one triangle, 3 vertices of type 1.
    let res = inflate(&named("triangular").cell, InflationSeed::Face(0), 4).unwrap();
    for s in &res.shells {
        let (want, tiles) = match s.n {
            0 => (BTreeMap::from([(1, 3)]), 1),
            n => (BTreeMap::from([(2, 6), (3, 6 * n - 3)]), 6 * n + 3),
        };
        assert_eq!((&s.boundary_vertex_types, s.topology.boundary_tiles), (&want, tiles), "triangular n={}", s.n);
    }
}

#[test]
fn torus_square_face_seed_wrap_step() {
    // {4,4} on the L x L torus (one square per cell), face seed: S_n is the a x a block, a = 2n + 1, while a < L.
    // Its a + 1 vertex columns (rows) are distinct mod L for a <= L - 1, so up to there the quotient block is the
    // planar block (chi = 1, no pinch; for a = L - 1 the wrap-around bonds are merely dangling). The first a >= L
    // covers every column and row: the whole torus, F = V = L^2, E = 2 L^2, chi = 0. So the wrap step is the first n
    // with 2n + 1 >= L, n = floor(L / 2), for both rules (every square is a centre).
    let t = PlaneTiling::new(&named("square").cell).unwrap();
    let seed = HashSet::from([(0, [0, 0])]);
    for l in [4i32, 5, 6] {
        for rule in [InflationRule::Bdf, InflationRule::Cluster(CentreSet::MaxPolygon)] {
            let res = t.inflate_torus(&seed, &rule, [l, l], 4).unwrap();
            let wrap = (l / 2) as usize;
            assert_eq!(res.wrap_step, Some(wrap), "L = {l} {rule:?}");
            for s in res.shells.iter().filter(|s| s.n < wrap) {
                let a = 2 * s.n + 1;
                assert_eq!((s.faces, s.vertices, s.chi), (a * a, (a + 1) * (a + 1), 1), "L = {l} n = {}", s.n);
            }
            let w = &res.shells[wrap];
            let l2 = (l * l) as usize;
            assert_eq!((w.faces, w.vertices, w.edges, w.chi), (l2, l2, 2 * l2, 0), "L = {l}");
        }
    }
}
