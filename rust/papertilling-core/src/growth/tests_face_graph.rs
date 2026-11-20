//! Defected-crystal growth tests (C1 WP-D1, plan §5 T7 and the B1 check of §5 TB).
//!
//! Where the numbers come from:
//! - {4,4} disclination forms (plan §2.4) are hand-derived: a +90° (−90°) wedge leaves k = 3 (k = 5) quadrants, so the
//!   face-seed patch is 1 + (k/4)·((2n+1)² − 1) = 1 + k·n(n+1) and the vertex-seed patch is (k/4)·(2n+2)² = k(n+1)².
//!   An independent scratch grower (`cone44.py`: k quarter-planes glued cyclically, bare vertex completion, no code
//!   from this repository) reproduced all four forms for n = 0..8 [COMPUTATIONAL EVIDENCE].
//! - The Dirichlet seed sizes are derived by hand below from the `DirichletAt` definition in `plane::inflation`.
//! - The pristine-disk comparison uses the WP-G3 E² grower (`plane::inflation`) as the independent method.
//! - The dislocation's known Burgers vector is the spec's `b`.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::defect::charge::{loop_charge, total_charge};
use crate::defect::core_graph::LatticeCoreInfo;
use crate::defect::defect_growth::{CrystalGrowth, DEFECT_MAX_CELLS, grow_crystal, grow_point_defects, start_radius};
use crate::defect::grain_boundary::perfect_disk;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::growth::face_graph::boundary_loop;
use crate::growth::{CellComplex, Stopped};
use crate::plane::catalogue::{NamedLattice, parse_named_str};
use crate::plane::draw::ensure_embedded;
use crate::plane::inflation::{InflationRule, InflationSeed, PlaneTiling, inflate_with};
use crate::plane::relax::relax_picture;
use crate::types::{BuiltCrystal, Cell2D};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

fn named(key: &str) -> NamedLattice {
    parse_named_str(NAMED_JSON).expect("named.json").get(key).expect("named lattice").clone()
}

fn square() -> Cell2D {
    let c = named("square").cell;
    // The face centre 1b is (1/2, 1/2) and the site 1a is the origin only for this frame.
    assert_eq!((c.a1, c.a2, c.basis.clone()), ([1.0, 0.0], [0.0, 1.0], vec![[0.0, 0.0]]));
    c
}

fn key(p: [f64; 2]) -> (i64, i64) {
    ((p[0] * 1e6).round() as i64, (p[1] * 1e6).round() as i64)
}

/// Every complete layer is a disk (χ = 1), and no vertex of a shown layer is on the rim.
fn check_disks_and_rim(name: &str, cg: &CrystalGrowth) {
    let rim: BTreeSet<u32> = cg.complex.rim.iter().map(|&s| s as u32).collect();
    for (k, s) in cg.growth.stats.iter().enumerate() {
        assert_eq!(s.euler, 1, "{name}: layer {k} is not a disk");
        for &c in &cg.growth.layers[k] {
            assert!(cg.complex.complex.vertices_of(c).iter().all(|v| !rim.contains(v)), "{name}: layer {k} touches the rim");
        }
    }
}

fn counts(cg: &CrystalGrowth) -> Vec<usize> {
    cg.growth.stats.iter().map(|s| s.cells).collect()
}

/// Layers 0..=n of the defect-free disk equal the E² touching-rule growth with `DirichletAt(c)`, face for face.
fn pristine_case(name: &str, cell: &Cell2D, c: [f64; 2], radius: f64, n: u32) {
    let disk = perfect_disk(cell, c, radius);
    let info = LatticeCoreInfo::new(cell);
    let cg = grow_crystal(&disk, &info.lat, c, n, DEFECT_MAX_CELLS).unwrap();
    assert_eq!(cg.complete_layers(), n as usize + 1, "{name}: radius {radius} too small ({:?})", cg.growth.stopped);
    assert_eq!((cg.seed.missing, cg.seed.core_faces), (0, 0), "{name}: nothing is cut in a perfect disk");
    check_disks_and_rim(name, &cg);
    let (faces, refp) = (&cg.complex.faces, &disk.ref_pos);
    let mine: BTreeSet<(usize, Vec<(i64, i64)>)> = cg
        .growth
        .layers
        .iter()
        .enumerate()
        .flat_map(|(k, layer)| {
            layer.iter().map(move |&f| {
                let mut p: Vec<_> = faces[f as usize].iter().map(|&s| key(refp[s])).collect();
                p.sort_unstable();
                (k, p)
            })
        })
        .collect();
    let e2 = inflate_with(cell, InflationSeed::DirichletAt(c), &InflationRule::Bdf, n as usize).unwrap();
    assert!(!e2.seed_repaired, "{name}: the E² Dirichlet seed was repaired, so the seeds differ by construction");
    assert!(e2.shells.iter().all(|s| s.filled == 0), "{name}: FillHoles fired in the E² growth");
    let theirs: BTreeSet<(usize, Vec<(i64, i64)>)> = e2
        .faces
        .iter()
        .map(|f| {
            let mut p: Vec<_> = f.polygon.iter().map(|&q| key(q)).collect();
            p.sort_unstable();
            (f.shell, p)
        })
        .collect();
    assert_eq!(mine, theirs, "{name}: layers differ from the E² grower");
}

#[test]
fn pristine_disk_equals_e2_dirichlet_growth() {
    let sq = square();
    // DirichletAt(c) keeps, per face orbit, the translates whose centroid is nearest to c (ties included).
    // At the face centre 1b the nearest centroid is that face's own (distance 0, unique): 1 seed tile.
    // At the site 1a the four squares at the site all have centroids at distance √2/2: 4 seed tiles (the vertex star).
    // So the 1b Dirichlet seed is the face seed, not the 9 tiles that touch the face.
    for (name, c, n0) in [("square 1b", [0.5, 0.5], 1usize), ("square 1a", [0.0, 0.0], 4)] {
        let info = LatticeCoreInfo::new(&sq);
        let cg = grow_crystal(&perfect_disk(&sq, c, 9.0), &info.lat, c, 0, DEFECT_MAX_CELLS).unwrap();
        assert_eq!(cg.seed.cells.len(), n0, "{name}: Dirichlet seed size");
        pristine_case(name, &sq, c, 9.0, 4);
    }
    for k in ["honeycomb", "kagome", "triangular"] {
        let cell = named(k).cell;
        let c = PlaneTiling::new(&cell).unwrap().most_symmetric_point().point;
        let a = (cell.a1[0].hypot(cell.a1[1])).max(cell.a2[0].hypot(cell.a2[1]));
        pristine_case(k, &cell, c, 12.0 * a, 4);
    }
}

/// The square-lattice disclination with apex `at`, cut `cut`, grown by `n` layers from radius `r`.
fn disclination(angle: f64, at: [f64; 2], cut: [f64; 2], n: u32, r: Option<f64>) -> crate::defect::defect_growth::DefectGrowth {
    let spec = [DefectSpec::Disclination { angle_deg: angle, at, cut: Some(cut) }];
    grow_point_defects(&square(), &spec, n, r, false, DEFECT_MAX_CELLS).unwrap()
}

#[test]
fn square_disclinations_follow_the_hand_forms() {
    const N: u32 = 5;
    // Apex at the face centre 1b, cut along the mirror line y = 1/2: the cut halves of squares re-glue into whole
    // squares and the centre square becomes a triangle (+90°) or a pentagon (−90°). The 1b Dirichlet seed is that
    // one cut face, so the Dirichlet layers are the face-seed forms.
    for (angle, k, faces) in [(90.0, 3usize, 3usize), (-90.0, 5, 5)] {
        let dg = disclination(angle, [0.5, 0.5], [1.0, 0.0], N, None);
        let cg = &dg.result;
        check_disks_and_rim("1b", cg);
        assert_eq!(cg.complete_layers(), N as usize + 1, "1b {angle}: {}", dg.note);
        assert_eq!((cg.seed.perfect_faces, cg.seed.carried, cg.seed.missing, cg.seed.core_faces), (1, 0, 1, 1), "1b {angle}");
        assert_eq!(cg.complex.faces[cg.seed.cells[0] as usize].len(), faces, "1b {angle}: the core face");
        let want: Vec<usize> = (0..=N as usize).map(|n| 1 + k * n * (n + 1)).collect();
        assert_eq!(counts(cg), want, "1b {angle}");
    }
    // Apex at the site 1a, cut along a bond line: the apex has degree 3 (+90°) or 5 (−90°). The 1a Dirichlet seed is
    // the 4 squares at the site; +90° removes one of them (3 carried, 1 missing, no defect face since the remaining
    // squares are whole), −90° adds a copy of one (4 carried, in 5 crystal faces). Either way the seed is the apex
    // star, so the layers are the vertex-seed forms k(n+1)².
    for (angle, k, carried) in [(90.0, 3usize, 3usize), (-90.0, 5, 4)] {
        let dg = disclination(angle, [0.0, 0.0], [1.0, 0.0], N, None);
        let cg = &dg.result;
        check_disks_and_rim("1a", cg);
        assert_eq!(cg.complete_layers(), N as usize + 1, "1a {angle}: {}", dg.note);
        assert_eq!((cg.seed.perfect_faces, cg.seed.carried, cg.seed.core_faces), (4, carried, 0), "1a {angle}");
        let want: Vec<usize> = (0..=N as usize).map(|n| k * (n + 1) * (n + 1)).collect();
        assert_eq!(counts(cg), want, "1a {angle}");
    }
}

#[test]
fn rebuild_loop_grows_the_crystal_until_layers_are_complete() {
    // From radius 4 the +90° cone cannot hold 4 complete layers (layer 4 spans ±4.5 about the apex), so the loop
    // rebuilds at 4 × 1.5^m; the counts must not depend on the radius the crystal was built at.
    let small = disclination(90.0, [0.5, 0.5], [1.0, 0.0], 4, Some(4.0));
    assert!(small.rebuilds >= 1, "expected a rebuild: {}", small.note);
    assert!((small.radius - 4.0 * 1.5_f64.powi(small.rebuilds as i32)).abs() < 1e-9);
    let big = disclination(90.0, [0.5, 0.5], [1.0, 0.0], 4, Some(14.0));
    assert_eq!(big.rebuilds, 0);
    let (a, b) = (counts(&small.result), counts(&big.result));
    let m = a.len().min(b.len());
    assert!(m >= 2);
    assert_eq!(a[..m], b[..m]);
    // Asking more layers than 3 rebuilds can reach: only complete layers, a Rim stop, and a note that says so.
    let far = disclination(90.0, [0.5, 0.5], [1.0, 0.0], 40, Some(4.0));
    assert_eq!(far.rebuilds, 3);
    let k = far.result.complete_layers() as u32;
    assert_eq!(far.result.growth.stopped, Some(Stopped::Rim { layer: k }));
    assert!(far.note.contains(&format!("layer {k} reaches the crystal's rim")), "{}", far.note);
    check_disks_and_rim("far", &far.result);
}

#[test]
fn last_requested_layer_touching_the_rim_is_not_shown() {
    // The engine checks the rim only before building the next layer, so a request that ends exactly on the first layer
    // touching the rim must still be cut here.
    let spec = [DefectSpec::Disclination { angle_deg: 90.0, at: [0.5, 0.5], cut: Some([1.0, 0.0]) }];
    let cell = square();
    let crystal = build_point_defects(&cell, &spec, Some(5.0), Some([0.5, 0.5])).unwrap();
    let info = LatticeCoreInfo::new(&cell);
    let free = grow_crystal(&crystal, &info.lat, [0.5, 0.5], 50, DEFECT_MAX_CELLS).unwrap();
    let Some(Stopped::Rim { layer: k }) = free.growth.stopped else { panic!("{:?}", free.growth.stopped) };
    assert!(k >= 1);
    for ask in [k, k + 1] {
        let at = grow_crystal(&crystal, &info.lat, [0.5, 0.5], ask, DEFECT_MAX_CELLS).unwrap();
        assert_eq!(at.complete_layers(), k as usize, "ask {ask}");
        assert_eq!(at.growth.stopped, Some(Stopped::Rim { layer: k }), "ask {ask}");
    }
    let below = grow_crystal(&crystal, &info.lat, [0.5, 0.5], k - 1, DEFECT_MAX_CELLS).unwrap();
    assert_eq!((below.complete_layers(), below.growth.stopped.clone()), (k as usize, None));
}

/// −h.t round the boundary of each complete layer, from the first layer whose boundary has a non-zero Burgers vector.
fn burgers_per_layer(crystal: &BuiltCrystal, cg: &CrystalGrowth) -> Vec<[f64; 2]> {
    let mut acc: Vec<u32> = Vec::new();
    cg.growth
        .layers
        .iter()
        .map(|layer| {
            acc.extend(layer.iter().copied());
            let lp = boundary_loop(&cg.complex, &acc).unwrap();
            let (h, _) = loop_charge(crystal, &lp);
            assert!((h.m[0][0] - 1.0).abs() < 1e-9 && h.m[1][0].abs() < 1e-9, "a dislocation loop has no rotation");
            [-h.t[0], -h.t[1]]
        })
        .collect()
}

#[test]
fn dislocation_layers_are_disks_and_keep_their_burgers_vector() {
    let t0 = Instant::now();
    for (lat, b) in [("square", [1.0, 0.0]), ("square", [0.0, 1.0]), ("triangular", [1.0, 0.0])] {
        let cell = named(lat).cell;
        let spec = [DefectSpec::dislocation(b, Some([0.371, 0.413]))];
        let dg = grow_point_defects(&cell, &spec, 5, None, false, DEFECT_MAX_CELLS).unwrap();
        let cg = &dg.result;
        check_disks_and_rim(lat, cg);
        assert_eq!(cg.complete_layers(), 6, "{lat}: {}", dg.note);
        let whole = total_charge(&dg.crystal).unwrap();
        assert_eq!(whole.kind, "dislocation");
        let wb = whole.burgers.unwrap();
        assert!((wb[0] - b[0]).hypot(wb[1] - b[1]) < 1e-6, "{lat}: outer Burgers {wb:?} vs spec {b:?}");
        let per = burgers_per_layer(&dg.crystal, cg);
        let first = per.iter().position(|p| p[0].hypot(p[1]) > 1e-6).expect("some layer encloses the core");
        println!("{lat} b={b:?}: seed {:?}, first enclosing layer {first}, counts {:?}", cg.seed, counts(cg));
        assert!(first <= 1, "{lat}: layer {first} is the first to enclose the core");
        for (k, p) in per.iter().enumerate().skip(first) {
            assert!((p[0] - b[0]).hypot(p[1] - b[1]) < 1e-6, "{lat}: layer {k} Burgers {p:?} vs {b:?}");
        }
    }
    println!("# time dislocation growth (3 cases, 5 layers): {:.2} s", t0.elapsed().as_secs_f64());
}

#[test]
fn relaxed_growth_keeps_counts_and_says_where_positions_come_from() {
    let spec = [DefectSpec::Disclination { angle_deg: -90.0, at: [0.5, 0.5], cut: Some([1.0, 0.0]) }];
    let raw = grow_point_defects(&square(), &spec, 3, None, false, DEFECT_MAX_CELLS).unwrap();
    let rel = grow_point_defects(&square(), &spec, 3, None, true, DEFECT_MAX_CELLS).unwrap();
    assert_eq!(counts(&raw.result), counts(&rel.result));
    assert!(rel.note.contains("relaxation"), "{}", rel.note);
    assert!(rel.crystal.pos.iter().zip(&raw.crystal.pos).any(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]) > 1e-6));
}

#[test]
fn walls_and_impossible_requests_are_refused_with_a_reason() {
    let hw = [DefectSpec::half_wall(None, [0.0, 0.0], None, None)];
    let e = grow_point_defects(&square(), &hw, 2, None, false, DEFECT_MAX_CELLS).unwrap_err();
    assert!(e.contains("point defects first"), "{e}");
    // A core outside the crystal carries no seed.
    let disk = perfect_disk(&square(), [0.0, 0.0], 4.0);
    let info = LatticeCoreInfo::new(&square());
    let e = grow_crystal(&disk, &info.lat, [40.5, 0.5], 2, DEFECT_MAX_CELLS).unwrap_err();
    assert!(e.contains("no face of the crystal carries the Dirichlet seed"), "{e}");
    let mut mob = disk.clone();
    mob.is_non_orientable = true;
    assert!(grow_crystal(&mob, &info.lat, [0.5, 0.5], 2, DEFECT_MAX_CELLS).unwrap_err().contains("orientable"));
}

#[test]
fn b1_relaxed_drawing_of_a_perfect_disk_does_not_move_it() {
    // B1 (plan §5 TB): a rigid motion in `pos − ref_pos` would show as a displacement in Detect. The app's
    // RelaxedDrawing path is `relax_picture` then `ensure_embedded(false)`; on a defect-free disk both must leave every
    // site where it was.
    for k in ["square", "honeycomb", "kagome", "triangular"] {
        let cell = named(k).cell;
        let mut d = perfect_disk(&cell, [0.0, 0.0], 6.0);
        relax_picture(&mut d);
        let _ = ensure_embedded(&mut d, false);
        let u = d.pos.iter().zip(&d.ref_pos).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).fold(0.0_f64, f64::max);
        assert!(u <= 1e-9, "{k}: Detect displacement {u:e}");
    }
}

#[test]
fn pilot_radius_needed_per_layer() {
    // §6 pilot: the radius at which layers 0..=n are complete, n = 1..6, on {4,4} (+90° cone), kagome (+60° at the
    // hexagon centre) and a {4,4} dislocation; and the rule start_radius gives. Measured, not pinned.
    let t0 = Instant::now();
    let kg = named("kagome").cell;
    let hex = PlaneTiling::new(&kg).unwrap().most_symmetric_point().point;
    let cases: Vec<(&str, Cell2D, DefectSpec)> = vec![
        ("square +90 1b", square(), DefectSpec::Disclination { angle_deg: 90.0, at: [0.5, 0.5], cut: Some([1.0, 0.0]) }),
        ("kagome +60 hex", kg, DefectSpec::disclination(60.0, hex)),
        ("square disloc", square(), DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))),
    ];
    for (name, cell, spec) in cases {
        let info = LatticeCoreInfo::new(&cell);
        let core = crate::defect::defect_growth::defect_core(std::slice::from_ref(&spec), [0.0, 0.0]).unwrap();
        let tc = Instant::now();
        let crystal = build_point_defects(&cell, std::slice::from_ref(&spec), Some(16.0), None).unwrap();
        let cg = grow_crystal(&crystal, &info.lat, core, 6, DEFECT_MAX_CELLS).unwrap();
        let mut acc = BTreeSet::new();
        let mut row = Vec::new();
        for layer in &cg.growth.layers {
            acc.extend(layer.iter().copied());
            let far = acc
                .iter()
                .flat_map(|&c| cg.complex.faces[c as usize].iter())
                .map(|&s| (crystal.ref_pos[s][0] - core[0]).hypot(crystal.ref_pos[s][1] - core[1]))
                .fold(0.0_f64, f64::max);
            row.push(format!("{far:.2}"));
        }
        let est: Vec<String> = (1..=6).map(|n| format!("{:.2}", start_radius(&info.lat, core, n).unwrap())).collect();
        println!(
            "pilot {name}: complete {} at R=16 (sites {}), farest corner of S_0..: [{}]; start_radius n=1..6: [{}]; # time {:.3} s",
            cg.complete_layers(),
            crystal.pos.len(),
            row.join(", "),
            est.join(", "),
            tc.elapsed().as_secs_f64()
        );
        check_disks_and_rim(name, &cg);
    }
    println!("# time pilot total {:.2} s", t0.elapsed().as_secs_f64());
}

#[test]
fn stone_wales_core_grows_like_the_bond_star_of_the_honeycomb() {
    // Stone-Wales (Q8 point defects) on a perfect honeycomb disk, flipped at the bond nearest the disk centre, grown
    // from the Dirichlet seed about the bond midpoint (the flip rotates about it, so it stays put).
    // Hand: DirichletAt(midpoint) is the 2 hexagons sharing the bond (centroids at equal distance, ties included); the
    // flip cuts both, so the seed is the 4 defect faces 5-5-7-7 (every face with u or v; the heptagons reach the cut
    // hexagons' corners at the other neighbours of u and v).
    // [PROVED] The 4 defect faces cover the same 16 sites as the 4 hexagons around the bond, and every other face is
    // unchanged, so by induction S_n has as many faces as perfect-honeycomb growth from that 4-hexagon bond star.
    // Those counts come from the independent `sw_hex.py` (triangular-lattice grower, no repository code):
    // 4, 14, 30, 52, 80, 114 = (3n+4)(n+1).
    let cell = named("honeycomb").cell;
    let c = PlaneTiling::new(&cell).unwrap().most_symmetric_point().point;
    let a = (cell.a1[0].hypot(cell.a1[1])).max(cell.a2[0].hypot(cell.a2[1]));
    let disk = perfect_disk(&cell, c, 14.0 * a);
    let mid = |&(u, v): &(usize, usize)| [0.5 * (disk.ref_pos[u][0] + disk.ref_pos[v][0]), 0.5 * (disk.ref_pos[u][1] + disk.ref_pos[v][1])];
    let dist = |p: [f64; 2]| (p[0] - c[0]).hypot(p[1] - c[1]);
    let ei = (0..disk.edges.len()).min_by(|&i, &j| dist(mid(&disk.edges[i])).total_cmp(&dist(mid(&disk.edges[j])))).unwrap();
    let core = mid(&disk.edges[ei]);
    let sw = crate::defect::stone_wales::stone_wales_flip(&disk, ei, None).unwrap();
    assert_eq!(sw.after_rings, [5, 5, 7, 7]);
    let info = LatticeCoreInfo::new(&cell);
    let cg = grow_crystal(&sw.crystal, &info.lat, core, 5, DEFECT_MAX_CELLS).unwrap();
    check_disks_and_rim("stone-wales", &cg);
    assert_eq!((cg.seed.perfect_faces, cg.seed.carried, cg.seed.missing, cg.seed.core_faces), (2, 0, 2, 4), "{:?}", cg.seed);
    let mut rings: Vec<usize> = cg.seed.cells.iter().map(|&f| cg.complex.faces[f as usize].len()).collect();
    rings.sort_unstable();
    assert_eq!(rings, [5, 5, 7, 7]);
    assert_eq!(cg.complete_layers(), 6, "{:?}", cg.growth.stopped);
    assert_eq!(counts(&cg), (0..=5usize).map(|n| (3 * n + 4) * (n + 1)).collect::<Vec<_>>());
}

#[test]
fn develop_face_rejects_unclosed_dislocation_core_and_defect_core_matches_surgery() {
    // Hand: a 4-bond walk at ref_pos [(0,1),(0,0),(1,0),(1,1)] with bond vectors (0,-1),(1,0),(0,1),(0,1) and seam
    // translation (-1,-1) on bond 3 visits the 4 corners of [0,1]^2 before landing at p_4 = (1,2) != (0,1) (Burgers
    // vector b = (1,1)); develop_face must reject it rather than calling it an uncut unit square.
    use crate::growth::face_graph::develop_face;
    use crate::types::{DirectedBondData, Isometry2};
    let mk = |to, vec, seam: Option<Isometry2>| DirectedBondData { to, vec, kind: 0, across: seam.is_some(), seam, ang: None, ghost_stub: None };
    let seam = Some(Isometry2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [-1.0, -1.0] });
    let mut crystal = perfect_disk(&square(), [0.5, 0.5], 2.0);
    crystal.ref_pos = vec![[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
    crystal.pos = crystal.ref_pos.clone();
    crystal.adj = vec![
        vec![mk(1, [0.0, -1.0], None)],
        vec![mk(2, [1.0, 0.0], None)],
        vec![mk(3, [0.0, 1.0], None)],
        vec![mk(0, [0.0, 1.0], seam)],
    ];
    for b in 0..4 {
        assert!(develop_face(&crystal, &[0, 1, 2, 3], b).is_none(), "base {b} must fail closure");
    }
    // Multi-defect `at: None` at slot 1 on unit square sits at (0.371 + 3*1, 0.413) = (3.371, 0.413); mean with (10, 0) is (6.6855, 0.2065).
    let specs = [DefectSpec::disclination(90.0, [10.0, 0.0]), DefectSpec::dislocation([1.0, 0.0], None)];
    let core = crate::defect::defect_growth::defect_core_in_cell(&square(), &specs).unwrap();
    assert!((core[0] - 6.6855).abs() < 1e-12 && (core[1] - 0.2065).abs() < 1e-12, "{core:?}");
}

