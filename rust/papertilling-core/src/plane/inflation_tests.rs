//! Known-answer tests for `plane::inflation` (simply-connected chamber vertex completion).
//!
//! Every expected count below comes from a hand argument written next to it, never from the code under test.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::catalogue::{NamedLattice, parse_catalogue_str, parse_named_str};
use super::inflation::{FaceId, InflationResult, InflationSeed, PatchTopology, PlaneTiling, VertexId, inflate};
use super::test_fixtures::named;
use crate::types::{Bond2D, Cell2D};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

fn tiles(res: &InflationResult) -> Vec<usize> {
    res.shells.iter().map(|s| s.topology.faces).collect()
}

/// Inflate and return the tile counts for n = 0..=n_max.
fn counts(cell: &Cell2D, seed: InflationSeed, n_max: usize) -> Vec<usize> {
    tiles(&inflate(cell, seed, n_max).unwrap())
}

#[test]
fn inflation_pilot_square_face_seed() {
    // {4,4}: the squares sharing a vertex with one square form its 3x3 block, so |S_1| = 9.
    let t0 = Instant::now();
    let res = inflate(&named("square").cell, InflationSeed::Face(0), 1).unwrap();
    println!("# time pilot square n=1: {:?}", t0.elapsed());
    assert_eq!(res.shells[1].topology.faces, 9);
}

#[test]
fn inflation_square_known_counts() {
    // {4,4}, one site and one square per primitive cell (torus Euler: V - E + F = 1 - 2 + 1 = 0).
    // Face seed: S_n is the (2n+1) x (2n+1) block; vertex completion of an a x a block adds every square
    // touching its boundary, the (a+2) x (a+2) block (corner squares touch the block's corner vertices).
    // Vertex seed: S_0 is the 2 x 2 block around a vertex, so S_n is the (2n+2) x (2n+2) block.
    let cell = named("square").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert_eq!(t.face_sizes(), vec![4]);
    let face: Vec<usize> = (0..=4).map(|n| (2 * n + 1) * (2 * n + 1)).collect();
    let vert: Vec<usize> = (0..=4).map(|n| (2 * n + 2) * (2 * n + 2)).collect();
    assert_eq!(counts(&cell, InflationSeed::Face(0), 4), face);
    assert_eq!(counts(&cell, InflationSeed::VertexStar(0), 4), vert);
    // Block boundary: an a x a block has 4a boundary edges, (a+1)^2 vertices, 2a(a+1) edges.
    let res = inflate(&cell, InflationSeed::Face(0), 4).unwrap();
    for s in &res.shells {
        let a = 2 * s.n + 1;
        assert_eq!(s.topology.boundary_edges, 4 * a);
        assert_eq!(s.topology.vertices, (a + 1) * (a + 1));
        assert_eq!(s.topology.edges, 2 * a * (a + 1));
        assert_eq!(s.filled, 0);
    }
    // Shell growth N_n / N_{n-1} with N_n = (2n+1)^2 - (2n-1)^2 = 8n: N_1/N_0 = 8, N_2/N_1 = 16/8 = 2.
    assert_eq!(res.shells[1].growth, Some(8.0));
    assert_eq!(res.shells[2].growth, Some(2.0));
    assert_eq!(res.faces.len(), 81);
    assert_eq!(res.faces.iter().filter(|f| f.shell == 4).count(), 81 - 49);
}

#[test]
fn inflation_honeycomb_known_counts() {
    // {6,3}: each vertex lies on 3 mutually edge-adjacent hexagons, so "shares a vertex" = "shares an edge" and
    // S_n is a graph ball in the dual triangular lattice of hexagon centres.
    // Face seed: the ball of radius n around a point has 1 + 6(1 + ... + n) = 3n(n+1) + 1 points.
    // Vertex seed: S_0 = a unit triangle of 3 centres; points within distance n of it form a lattice hexagon with
    // edge lengths alternating (n+1, n), i.e. a triangle of side L = 3n+1 with three corner triangles of side
    // n-1 cut off: (L+1)(L+2)/2 - 3 n(n+1)/2 = 3(n+1)^2 points.
    let cell = named("honeycomb").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert!(t.face_sizes().iter().all(|&k| k == 6), "{:?}", t.face_sizes());
    let face: Vec<usize> = (0..=4).map(|n| 3 * n * (n + 1) + 1).collect();
    let vert: Vec<usize> = (0..=4).map(|n| 3 * (n + 1) * (n + 1)).collect();
    for f in 0..t.orbit_count() {
        assert_eq!(counts(&cell, InflationSeed::Face(f), 4), face, "face orbit {f}");
    }
    for s in 0..cell.basis.len() {
        assert_eq!(counts(&cell, InflationSeed::VertexStar(s), 4), vert, "site {s}");
    }
}

#[test]
fn inflation_triangular_known_counts() {
    // {3,6}: S_n is a lattice hexagon with side lengths alternating (a, b): a triangle of side T = a + 2b with
    // three corner triangles of side b cut off, holding T^2 - 3b^2 unit triangles. The triangles outside such a
    // convex hexagon that touch its boundary vertices are the strips of 2L+1 triangles along each side of length
    // L (a mid-side vertex has 3 outer triangles, all in its strip; a 120-degree corner has 4, 2 from each strip;
    // a length-0 side is a 60-degree corner with 5 = 2 + 1 + 2). Adding them moves all six side lines out by one
    // row, (a, b) -> (a+1, b+1); check: (T+3)^2 - 3(b+1)^2 - (T^2 - 3b^2) = 6(a+b+1) = sum over sides of 2L+1.
    // Face seed: S_0 = one triangle, (a, b) = (1, 0); S_n: (n+1, n), T = 3n+1, (3n+1)^2 - 3n^2 = 6n^2 + 6n + 1.
    // Vertex seed: S_0 = the hexagon of 6 triangles, (1, 1); S_n: (n+1, n+1), 9(n+1)^2 - 3(n+1)^2 = 6(n+1)^2.
    let cell = named("triangular").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert!(t.face_sizes().iter().all(|&k| k == 3), "{:?}", t.face_sizes());
    let face: Vec<usize> = (0..=4).map(|n| 6 * n * n + 6 * n + 1).collect();
    let vert: Vec<usize> = (0..=4).map(|n| 6 * (n + 1) * (n + 1)).collect();
    for f in 0..t.orbit_count() {
        assert_eq!(counts(&cell, InflationSeed::Face(f), 4), face, "face orbit {f}");
    }
    for s in 0..cell.basis.len() {
        assert_eq!(counts(&cell, InflationSeed::VertexStar(s), 4), vert, "site {s}");
    }
}

#[test]
fn inflation_euclidean_corona_recurrence() {
    // Corona sizes N_n = |S~_n| - |S~_{n-1}| of the FACE seeds, from the closed forms pinned above:
    // {4,4}: (2n+1)^2 - (2n-1)^2 = 8n.  {6,3}: (3n(n+1) + 1) - (3(n-1)n + 1) = 6n (the spec's |Q(S_n)| = 6n).
    // {3,6}: (6n^2 + 6n + 1) - (6(n-1)^2 + 6(n-1) + 1) = 12n.
    // A sequence c*n has zero second difference: a_{k+1} = 2 a_k - a_{k-1}, the gamma = 1 case of the hyperbolic
    // corona recurrence a_{k+1} = 2 gamma a_k - a_{k-1} (double eigenvalue 1: the Jordan block of the BDF boundary
    // substitution at gamma = 1). It holds from k = 2 on (a_1..a_5 below); at k = 1 it needs a_0 = 0, but the
    // seed shell is N_0 = 1 tile, e.g. square 2*8 - 1 = 15 != 16 = N_2.
    for (key, c) in [("square", 8), ("honeycomb", 6), ("triangular", 12)] {
        let cell = named(key).cell;
        for f in 0..PlaneTiling::new(&cell).unwrap().orbit_count() {
            let a: Vec<usize> = inflate(&cell, InflationSeed::Face(f), 5).unwrap().shells.iter().map(|s| s.new_tiles).collect();
            assert_eq!(a, [1, c, 2 * c, 3 * c, 4 * c, 5 * c], "{key} face orbit {f}");
            for k in 2..=4 {
                assert_eq!(a[k + 1], 2 * a[k] - a[k - 1], "{key} face orbit {f}, k = {k}");
            }
        }
    }
}

fn square_patch(cells: &[[i32; 2]]) -> HashSet<FaceId> {
    cells.iter().map(|&o| (0, o)).collect()
}

#[test]
fn inflation_topology_controls_on_square_patches() {
    // All squares below are lattice translates (0, o) of the single square orbit.
    let t = PlaneTiling::new(&named("square").cell).unwrap();
    // NEGATIVE CONTROL: two squares touching at one vertex. V = 4 + 4 - 1 = 7, E = 8, F = 2, chi = 1, yet the
    // shared vertex sees in/out/in/out around it: one pinch vertex. chi = 1 alone does not certify a disk.
    let bowtie = t.topology(&square_patch(&[[0, 0], [1, 1]]));
    assert_eq!((bowtie.vertices, bowtie.edges, bowtie.faces, bowtie.chi), (7, 8, 2, 1));
    assert_eq!(bowtie.pinch_vertices, 1);
    assert_eq!(bowtie.dangling_bonds, 0);
    // Two squares with a one-square gap: the two bonds crossing the gap join patch vertices but lie on no patch
    // face. V = 8, E = 8, F = 2, chi = 2.
    let gap = t.topology(&square_patch(&[[0, 0], [2, 0]]));
    assert_eq!((gap.vertices, gap.edges, gap.chi, gap.dangling_bonds, gap.pinch_vertices), (8, 8, 2, 2, 0));
    // Ring of 8 squares around (1, 1): V = 16, E = 24 (all edges of the 3x3 block), F = 8, chi = 0.
    // FillHoles adds exactly the centre square, giving the 3x3 block (chi = 1, 12 boundary edges).
    let ring: Vec<[i32; 2]> = (0..3).flat_map(|i| (0..3).map(move |j| [i, j])).filter(|&o| o != [1, 1]).collect();
    let mut patch = square_patch(&ring);
    let top = t.topology(&patch);
    assert_eq!((top.vertices, top.edges, top.faces, top.chi, top.boundary_edges), (16, 24, 8, 0, 16));
    let added: Vec<FaceId> = t.fill_holes(&mut patch).into_iter().collect();
    assert_eq!(added, vec![(0, [1, 1])]);
    assert_eq!(t.topology(&patch).chi, 1);
    // Plus shape without its centre: four squares meeting the hole only at corners. V = 16 - 4 = 12, E = 16,
    // F = 4, chi = 0, and each of the 4 hole corners is a pinch. The hole is a bounded component of the open
    // complement (its only contacts with the outside are patch vertices), so shared-edge FillHoles fills it;
    // shared-vertex adjacency would have joined it to the outside. After filling: F = 5, chi = 1, no pinch.
    let mut plus = square_patch(&[[1, 0], [0, 1], [2, 1], [1, 2]]);
    let top = t.topology(&plus);
    assert_eq!((top.vertices, top.edges, top.chi, top.pinch_vertices), (12, 16, 0, 4));
    assert_eq!(t.fill_holes(&mut plus).len(), 1);
    let top = t.topology(&plus);
    assert_eq!((top.faces, top.chi, top.pinch_vertices), (5, 1, 0));
    // A C-shape open along an edge is not a hole: ring minus (0, 1) leaves (1, 1) edge-connected to the outside.
    let mut c_shape = square_patch(&ring.iter().copied().filter(|&o| o != [0, 1]).collect::<Vec<_>>());
    assert!(t.fill_holes(&mut c_shape).is_empty());
}

/// Independent from-scratch topology of `patch` (the definitions of `PatchTopology`), with dangling bonds read
/// from `cell.bonds`, to check the incremental counts that `inflate` keeps.
fn naive_topology(cell: &Cell2D, t: &PlaneTiling, patch: &HashSet<FaceId>) -> PatchTopology {
    let key = |u: VertexId, w: VertexId| if u < w { (u, w) } else { (w, u) };
    let mut verts: HashSet<VertexId> = HashSet::new();
    let mut sides: HashMap<(VertexId, VertexId), usize> = HashMap::new();
    for &face in patch {
        let vs = t.face_vertices(face);
        for k in 0..vs.len() {
            verts.insert(vs[k]);
            *sides.entry(key(vs[k], vs[(k + 1) % vs.len()])).or_insert(0) += 1;
        }
    }
    let pinch_vertices = verts
        .iter()
        .filter(|&&v| {
            let ring: Vec<bool> = t.vertex_faces(v).iter().map(|f| patch.contains(f)).collect();
            (0..ring.len()).filter(|&k| ring[k] && !ring[(k + ring.len() - 1) % ring.len()]).count() >= 2
        })
        .count();
    let mut dangling_bonds = 0;
    for b in &cell.bonds {
        for &v in verts.iter().filter(|v| v.0 == b.i) {
            let w = (b.j, [v.1[0] + b.n1, v.1[1] + b.n2]);
            dangling_bonds += usize::from(verts.contains(&w) && !sides.contains_key(&key(v, w)));
        }
    }
    let (faces, vertices, edges) = (patch.len(), verts.len(), sides.len());
    // A boundary tile has an edge whose other side is not a patch face (an edge seen once).
    let boundary_tiles = patch
        .iter()
        .filter(|&&f| {
            let vs = t.face_vertices(f);
            (0..vs.len()).any(|k| sides[&key(vs[k], vs[(k + 1) % vs.len()])] == 1)
        })
        .count();
    PatchTopology {
        faces,
        vertices,
        edges,
        boundary_edges: sides.values().filter(|&&n| n == 1).count(),
        chi: vertices as i64 - edges as i64 + faces as i64,
        pinch_vertices,
        dangling_bonds,
        boundary_tiles,
    }
}

/// Check chi = 1, no pinch and strictly growing shells for n = 0..=4, and that the frontier-based `inflate` agrees
/// shell by shell (face sets and full topology) with a from-scratch loop over the public full-patch
/// `completion`, `fill_holes` and `topology`, and with `naive_topology`. Returns failure descriptions and the
/// number of faces FillHoles added over all shells (a measure of how often vertex completion traps pockets).
fn disk_failures(label: &str, cell: &Cell2D, seed: InflationSeed) -> (Vec<String>, usize) {
    let res = match inflate(cell, seed, 4) {
        Ok(r) => r,
        Err(e) => return (vec![format!("{label} {seed:?}: inflate error {e}")], 0),
    };
    let mut bad = Vec::new();
    let ids: HashSet<FaceId> = res.faces.iter().map(|f| (f.orbit, f.offset)).collect();
    if ids.len() != res.faces.len() {
        bad.push(format!("{label} {seed:?}: {} drawn faces but {} distinct ids", res.faces.len(), ids.len()));
    }
    let t = PlaneTiling::new(cell).unwrap();
    let mut reference = t.seed_faces(seed).unwrap();
    t.fill_holes(&mut reference);
    for s in &res.shells {
        if s.n > 0 {
            let grown = t.completion(&reference);
            reference.extend(grown);
            t.fill_holes(&mut reference);
        }
        let tp = &s.topology;
        let drawn: HashSet<FaceId> = res.faces.iter().filter(|f| f.shell <= s.n).map(|f| (f.orbit, f.offset)).collect();
        if drawn != reference || *tp != t.topology(&reference) || *tp != naive_topology(cell, &t, &reference) {
            bad.push(format!("{label} {seed:?} n={}: inflate differs from the from-scratch reference shell", s.n));
        }
        // Boundary vertex types from scratch: patch vertices with a non-patch face corner, by patch corner count.
        let mut types = std::collections::BTreeMap::new();
        for v in reference.iter().flat_map(|&f| t.face_vertices(f)).collect::<HashSet<_>>() {
            let inside = t.vertex_faces(v).iter().filter(|f| reference.contains(f)).count();
            if inside < t.vertex_faces(v).len() {
                *types.entry(inside).or_insert(0) += 1;
            }
        }
        if types != s.boundary_vertex_types {
            bad.push(format!("{label} {seed:?} n={}: boundary vertex types differ from scratch", s.n));
        }
        let grew = s.n == 0 || s.new_tiles > 0;
        if tp.chi != 1 || tp.pinch_vertices != 0 || !grew {
            bad.push(format!("{label} {seed:?} n={}: chi={} pinch={} new={}", s.n, tp.chi, tp.pinch_vertices, s.new_tiles));
        }
    }
    (bad, res.shells.iter().map(|s| s.filled).sum())
}

/// The distinctive part of the `Err` that `inflate` returns when `S~_0` is not a closed disk.
const NOT_DISK: &str = "is not a closed disk";

/// Disk-check outcome over many cells: failures, Dirichlet seeds rejected as not a disk, runs where FillHoles fired.
#[derive(Default)]
struct Sweep {
    bad: Vec<String>,
    dirichlet_err: Vec<String>,
    filled: Vec<String>,
}

impl Sweep {
    /// Every Face and VertexStar seed of `cell` must give disk shells. Dirichlet must either give disk shells or be
    /// rejected with the S~_0-not-a-disk error (any other error is a failure).
    fn add(&mut self, label: &str, cell: &Cell2D) {
        // `PlaneTiling::new` checks the torus Euler characteristic, that no bonds cross or pass through a site, and
        // that the orbit polygons tile the cell: every stored drawing must pass (no false rejects).
        let t = PlaneTiling::new(cell).unwrap_or_else(|e| panic!("{label}: {e}"));
        let seeds = (0..t.orbit_count()).map(InflationSeed::Face);
        for seed in seeds.chain((0..cell.basis.len()).map(InflationSeed::VertexStar)) {
            let (b, filled) = disk_failures(label, cell, seed);
            self.bad.extend(b);
            if filled > 0 {
                self.filled.push(format!("{label} {seed:?}: {filled} faces"));
            }
        }
        match inflate(cell, InflationSeed::Dirichlet, 4) {
            Err(e) if e.contains(NOT_DISK) => self.dirichlet_err.push(format!("{label}: {e}")),
            _ => self.bad.extend(disk_failures(label, cell, InflationSeed::Dirichlet).0),
        }
    }

    fn report(&self, what: &str) {
        println!("{what} Dirichlet rejected as not a disk: {} {:#?}", self.dirichlet_err.len(), self.dirichlet_err);
        println!("{what} Face/VertexStar runs where FillHoles added faces ({}): {:#?}", self.filled.len(), self.filled);
    }
}

#[test]
fn inflation_named_lattices_shells_are_disks() {
    let t0 = Instant::now();
    let all: Vec<NamedLattice> = parse_named_str(NAMED_JSON).unwrap().named;
    assert_eq!(all.len(), 21);
    let mut sw = Sweep::default();
    for lat in &all {
        sw.add(&lat.key, &lat.cell);
    }
    println!("# time named sweep (21 lattices, n <= 4): {:?}", t0.elapsed());
    sw.report("named");
    assert!(sw.bad.is_empty(), "{:#?}", sw.bad);
}

#[test]
fn inflation_named_lattices_dirichlet() {
    // Every named lattice: the Dirichlet seed is accepted (S~_0 a closed disk) and every BDF shell n <= 4 is a
    // closed disk that agrees with the from-scratch reference.
    let mut bad = Vec::new();
    for lat in &parse_named_str(NAMED_JSON).unwrap().named {
        bad.extend(disk_failures(&lat.key, &lat.cell, InflationSeed::Dirichlet).0);
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn inflation_kagome_hexagon_seed_known_counts() {
    // Kagome (3.6.3.6) is the rectified triangular lattice T: a kagome vertex is the midpoint of a T-edge, a hexagon
    // is centred on a T-vertex, a triangle is the medial triangle of a T-triangle. The faces at the kagome vertex
    // on T-edge ab are the hexagons a, b and the two triangles on ab (hex, tri, hex, tri), so per cell: 1 hexagon,
    // 2 triangles, 3 vertices.
    // n = 1 directly: the 6 vertices of a hexagon each add its 2 edge triangles (shared by neighbouring vertices:
    // 6 triangles) and the opposite hexagon (6 hexagons), so |S~_1| = 1 + 6 + 6 = 13.
    // General n: S_n = {hexagons on the graph ball B(n) of T} U {triangles of the lattice hexagon H_n of side n}.
    // Its kagome vertices are the T-edges with an endpoint in B(n); completion adds both ends (B(n+1)) and the
    // triangles with a vertex in B(n), which are exactly those of H_{n+1} (no triangle of H_{n+1} has all three
    // corners on its boundary ring). So |S_n| = (3n(n+1) + 1) + 6n^2 = 9n^2 + 3n + 1: 1, 13, 43, 91, 157. S_n
    // is H_n plus hexagon pieces glued along its boundary, a disk, so FillHoles adds nothing.
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    let mut sizes = t.face_sizes();
    let hex = sizes.iter().position(|&k| k == 6).unwrap();
    sizes.sort_unstable();
    assert_eq!((sizes, cell.basis.len()), (vec![3, 3, 6], 3));
    let res = inflate(&cell, InflationSeed::Face(hex), 4).unwrap();
    assert_eq!(tiles(&res), (0..=4).map(|n| 9 * n * n + 3 * n + 1).collect::<Vec<_>>());
    assert!(res.shells.iter().all(|s| s.filled == 0 && s.topology.chi == 1));
}

#[test]
fn inflation_rejects_crossing_bonds() {
    // Square lattice plus both diagonals: V = 1, E = 4 per cell, and the two diagonals cross inside every square.
    // Tracing faces with the angular rotation E, NE, N, NW, W, SW, S, SE gives one 8-gon (E, NW, S, NE, W, SE, N,
    // SW: each step turns to the rotation neighbour of the reversed dart), so V - E + F = 1 - 4 + 1 = -2 != 0.
    let mut cell = named("square").cell;
    cell.bonds.push(Bond2D { i: 0, j: 0, n1: 1, n2: 1, kind: 1 });
    cell.bonds.push(Bond2D { i: 0, j: 0, n1: 1, n2: -1, kind: 1 });
    let err = PlaneTiling::new(&cell).unwrap_err();
    assert!(err.contains("V - E + F = 1 - 4 + 1 != 0"), "{err}");
    assert!(inflate(&cell, InflationSeed::Face(0), 1).unwrap_err().contains("crossing bonds"));
}

/// Honeycomb graph (2 sites, bonds A->B at offsets (-1,0), (0,0), (0,1)) drawn with a bad basis.
fn crossing_honeycomb() -> Cell2D {
    let s3 = 3f64.sqrt();
    Cell2D {
        a1: [s3, 0.0],
        a2: [-s3 / 2.0, 1.5],
        basis: vec![[0.26680, 1.23784], [2.60191, 1.35026]],
        bonds: [[-1, 0], [0, 0], [0, 1]].iter().map(|o| Bond2D { i: 0, j: 1, n1: o[0], n2: o[1], kind: 0 }).collect(),
    }
}

#[test]
fn inflation_rejects_crossing_drawing_of_a_torus_map() {
    // The honeycomb graph is a torus map whatever the drawing: V - E + F = 2 - 3 + 1 = 0 (one hexagon orbit), and a
    // single orbit's signed area is (winding) x det, so the Euler and area guards cannot see a bad drawing.
    // Hand check with a1 = (sqrt3, 0), a2 = (-sqrt3/2, 1.5): P = A(0,0) = (0.26680, 1.23784), Q = B(0,0) =
    // (2.60191, 1.35026); R = A(-1,-1) = A - a1 - a2 = (-0.59923, -0.26216), S = B(-1,0) = B - a1 = (0.86986, 1.35026)
    // (bond A->B at offset (0,1) shifted by (-1,-1)). With d = Q - P = (2.33511, 0.11242), e = S - R = (1.46909,
    // 1.61242): cross(d, R - P) = -3.4053 < 0 < 0.1947 = cross(d, S - P), and cross(e, P - R) = 0.8072 > 0 >
    // -2.7928 = cross(e, Q - R). Both pairs straddle, so PQ and RS cross properly, at t = 0.8072 / 3.6000 = 0.224
    // along PQ, near (0.790, 1.263).
    let err = PlaneTiling::new(&crossing_honeycomb()).unwrap_err();
    assert!(err.contains("two periodic bond images cross properly"), "{err}");
    assert!(inflate(&crossing_honeycomb(), InflationSeed::Face(0), 1).is_err());
}

#[test]
fn inflation_site_on_bond_rejected_for_any_representative() {
    // Kagome cell: a1 = (2, 0), a2 = (-1, sqrt3), sites s0 = (-1, sqrt3), s1 = (0, sqrt3), s2 = (1/2, sqrt3/2).
    // Bond 1->2 at offset (0, 1) runs from s1 = (0, sqrt3) to s2 + a2 = (-1/2, 3 sqrt3/2); its midpoint is
    // (-1/4, 5 sqrt3/4) = (-0.25, 2.16506). Moving s0 there (graph unchanged) puts a site inside a bond it is not
    // an end of. Fractional coordinates (c1, c2) = ((x + y/sqrt3)/2, y/sqrt3): s1 = (1/2, 1), s2 + a2 = (1/2, 3/2),
    // s0 = (1/2, 5/4) (the midpoint, as it must be).
    let s3 = 3f64.sqrt();
    let mut on_bond = named("kagome").cell;
    assert!((on_bond.a1[0] - 2.0).abs() < 1e-12 && (on_bond.a2[1] - s3).abs() < 1e-12, "{on_bond:?}");
    on_bond.basis[0] = [-0.25, 5.0 * s3 / 4.0];
    // The same drawing with s0 represented 5 cells along a1: s0 + 5 a1 = (9.75, 2.16506), fractional (11/2, 5/4),
    // and each bond at s0 compensated (i = 0: n1 + 5; j = 0: n1 - 5), so every bond vector is unchanged. The
    // on-bond image is now s0 - 5 a1, i.e. shift s1 = -5, outside clearance's window [0 - 2, 0 + 2] for this bond.
    let mut shifted = on_bond.clone();
    shifted.basis[0][0] += 10.0;
    for b in &mut shifted.bonds {
        b.n1 += 5 * (i32::from(b.i == 0) - i32::from(b.j == 0));
    }
    // NEGATIVE CONTROL: on the shifted representatives the raw window scan misses the site.
    assert!(crate::plane::tuning::clearance(&shifted) > 1e-6);
    for (what, cell) in [("on_bond", &on_bond), ("shifted by 5 a1", &shifted)] {
        let err = PlaneTiling::new(cell).unwrap_err();
        assert!(err.contains("lies on a bond it is not an end of"), "{what}: {err}");
    }
}

#[test]
fn inflation_perturbed_cells_never_accept_crossings() {
    // Jitter the basis of lattices with few orbits (deterministic LCG, up to 0.45 of the spacing per axis). Any cell
    // `PlaneTiling::new` accepts must have no proper crossing by the separate Cartesian scan `Cell2D::has_crossings`
    // over images in [-5, 5]^2 (enough here: every bond spans at most a few cells).
    let t0 = Instant::now();
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rnd = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut accepted, mut rejected, mut by_guard, mut bad) = (0usize, 0usize, 0usize, Vec::new());
    for key in ["honeycomb", "chiral_square", "brick_pg", "chiral_kagome", "brick", "square", "kagome"] {
        let base = named(key).cell;
        let h = ((base.a1[0] * base.a2[1] - base.a1[1] * base.a2[0]).abs() / base.basis.len() as f64).sqrt();
        for trial in 0..300 {
            let mut cell = base.clone();
            for p in &mut cell.basis {
                *p = [p[0] + 0.9 * h * rnd(), p[1] + 0.9 * h * rnd()];
            }
            match PlaneTiling::new(&cell) {
                Ok(_) if cell.has_crossings() => bad.push(format!("{key} trial {trial}: {:?}", cell.basis)),
                Ok(_) => accepted += 1,
                Err(e) => {
                    rejected += 1;
                    by_guard += usize::from(e.contains("cross properly") || e.contains("lies on a bond"));
                }
            }
        }
    }
    println!("# time perturbation sweep: {:?} ({accepted} accepted, {rejected} rejected, {by_guard} by the planarity guard)", t0.elapsed());
    // The planarity guard must be exercised: some crossing drawings pass the Euler check and reach it.
    assert!(accepted > 0 && by_guard > 0);
    assert!(bad.is_empty(), "{} accepted crossing cells: {bad:#?}", bad.len());
}

fn face_size_label(cell: &Cell2D) -> (String, bool) {
    let mut sizes = PlaneTiling::new(cell).map(|t| t.face_sizes()).unwrap_or_default();
    sizes.sort_unstable();
    let mixed = sizes.windows(2).any(|w| w[0] != w[1]);
    (format!("{sizes:?}"), mixed)
}

const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");
const SHEET_JSON: &str = include_str!("../../../../papertilling/data/sheet.json");
const MINIMAL_B_JSON: &str = include_str!("../../../../papertilling/data/minimal_b.json");

/// Sweep every drawing of one stored catalogue (Face and VertexStar seeds, n = 0..=4); Dirichlet rejections are
/// printed only.
fn catalogue_sweep(file: &str, text: &str) {
    let t0 = Instant::now();
    let cats = parse_catalogue_str(text).unwrap();
    let (mut sw, mut mixed, mut total) = (Sweep::default(), 0usize, 0usize);
    for (grp, cat) in &cats {
        for (idx, ent) in cat.drawings.iter().enumerate() {
            let cell = ent.family.cell();
            mixed += usize::from(face_size_label(&cell).1);
            total += 1;
            sw.add(&format!("{file}/{grp}/drawings[{idx}]"), &cell);
        }
    }
    println!("# time {file}: {:?} for {total} drawings ({mixed} with mixed face sizes)", t0.elapsed());
    sw.report(file);
    assert!(mixed > 0);
    assert!(sw.bad.is_empty(), "{:#?}", sw.bad);
}

#[test]
fn inflation_catalogue_points_shells_are_disks() {
    catalogue_sweep("points", POINTS_JSON);
}

#[test]
fn inflation_catalogue_sheet_shells_are_disks() {
    catalogue_sweep("sheet", SHEET_JSON);
}

#[test]
fn inflation_catalogue_minimal_b_shells_are_disks() {
    catalogue_sweep("minimal_b", MINIMAL_B_JSON);
}

#[test]
fn inflation_handoff_mixed_polygon_entries_are_disks() {
    // Plain examples of mixed-polygon entries (author decision 6: catalogue numbering is 1-based, the app labels
    // entry k as #k+1; "p4m #6, p6 #5, pmm #5" came from a handoff prompt and carry no special status): p4m all[5],
    // p6 all[4], pmm all[4], and the neighbours all[6], all[5], all[5]. Each is taken from sheet.json, points.json
    // and minimal_b.json when that list is long enough. For n = 0..=4 and every seed: chi = 1 and no pinch vertex (`Sweep`). "Zero dangling
    // bonds" is not asserted: the definition is the author's call. In the `PatchTopology::dangling_bonds` sense (a
    // bond between two patch vertices on no patch face) it is NOT zero for most entries (see
    // `inflation_dangling_bond_notches`); `DANGLING` lines print the nonzero counts. (The legacy `prune` sense, a
    // vertex with < 2 patch edges, holds by construction for any union of polygon faces, so it certifies nothing.)
    let mut sw = Sweep::default();
    let mut checked = Vec::new();
    for (file, text) in [("sheet", SHEET_JSON), ("points", POINTS_JSON), ("minimal_b", MINIMAL_B_JSON)] {
        let cats = parse_catalogue_str(text).unwrap();
        for (grp, idx) in [("p4m", 5), ("p6", 4), ("pmm", 4), ("p4m", 6), ("p6", 5), ("pmm", 5)] {
            let Some(ent) = cats[grp].all.get(idx) else { continue };
            let (cell, label) = (ent.family.cell(), format!("{file}/{grp}/all[{idx}]"));
            // Plain BDF vertex completion (no FillHoles) from Face(0), for comparison: chi of S_0..S_4.
            let t = PlaneTiling::new(&cell).unwrap();
            let mut plain: HashSet<FaceId> = t.seed_faces(InflationSeed::Face(0)).unwrap();
            let mut chis = vec![t.topology(&plain).chi];
            for _ in 0..4 {
                plain.extend(t.completion(&plain));
                chis.push(t.topology(&plain).chi);
            }
            println!("{label} face sizes {}; plain BDF chi {chis:?}", face_size_label(&cell).0);
            sw.add(&label, &cell);
            let seeds = (0..t.orbit_count()).map(InflationSeed::Face).chain((0..cell.basis.len()).map(InflationSeed::VertexStar));
            for seed in seeds.chain([InflationSeed::Dirichlet]) {
                // A Dirichlet seed whose S~_0 is pinched is rejected (sheet/p6/all[5]: chi = 1, 1 pinch); it is
                // counted in `Sweep::dirichlet_err` and printed, any other error fails.
                let res = match inflate(&cell, seed, 4) {
                    Err(e) if seed == InflationSeed::Dirichlet && e.contains(NOT_DISK) => continue,
                    r => r.unwrap_or_else(|e| panic!("{label} {seed:?}: {e}")),
                };
                let dangling: Vec<usize> = res.shells.iter().map(|s| s.topology.dangling_bonds).collect();
                if dangling != vec![0; 5] {
                    println!("DANGLING {label} {seed:?}: {dangling:?}");
                }
            }
            checked.push(label);
        }
    }
    sw.report("handoff");
    println!("handoff entries checked: {checked:?}");
    // The 1-based sheet entries must be among those checked.
    for want in ["sheet/p4m/all[5]", "sheet/p6/all[4]", "sheet/pmm/all[4]"] {
        assert!(checked.iter().any(|c| c == want), "{want} missing");
    }
    assert!(sw.bad.is_empty(), "{:#?}", sw.bad);
}

#[test]
fn inflation_dangling_bond_notches() {
    // sheet p6 all[4] (1-based "p6 #5"): site 0 is a hub with 6 bonds and 6 hexagons around it; sites 1, 2, 3 and
    // 4, 5, 6 form small triangles (bonds 1-2, 1-3, 2-3 and 4-5, 4-6, 5-6 at offset (0, 0)), each a degree-3 site
    // with faces (hexagon, hexagon, triangle). Each of the 6 hub neighbours is such a site: its two hexagons lie in
    // the star of site 0, and its triangle is a notch whose other two corners are corners of those hexagons, so
    // the triangle's third bond joins two patch vertices and lies on no patch face. The notch is edge-adjacent to
    // the outside across that bond, so FillHoles leaves it. S~_0 = VertexStar(0): 6 faces, 6 dangling bonds.
    let cell = parse_catalogue_str(SHEET_JSON).unwrap()["p6"].all[4].family.cell();
    let hub: Vec<_> = cell.bonds.iter().filter(|b| b.i == 0 || b.j == 0).collect();
    assert_eq!(hub.len(), 6);
    for (i, j) in [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)] {
        assert!(cell.bonds.iter().any(|b| (b.i, b.j, b.n1, b.n2) == (i, j, 0, 0)), "bond {i}-{j}");
    }
    let s0 = &inflate(&cell, InflationSeed::VertexStar(0), 0).unwrap().shells[0].topology;
    assert_eq!((s0.faces, s0.chi, s0.pinch_vertices, s0.dangling_bonds), (6, 1, 0, 6));
}

#[test]
#[ignore = "timing only: cost of inflate on a mixed-polygon sheet entry at larger n"]
fn inflation_timing_sheet_p6_all5() {
    let cats = parse_catalogue_str(SHEET_JSON).unwrap();
    let cell = cats["p6"].all[5].family.cell();
    for n in [8, 16, 32, 64] {
        let t0 = Instant::now();
        match inflate(&cell, InflationSeed::Face(0), n) {
            Ok(res) => println!("# time sheet p6 all[5] Face(0) n={n}: {:?} ({} faces)", t0.elapsed(), res.faces.len()),
            Err(e) => println!("# sheet p6 all[5] Face(0) n={n}: {e}"),
        }
    }
}


