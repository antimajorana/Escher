//! Tests of the cluster growth rule, the Dirichlet seed, boundary statistics and tile classes of
//! `plane::inflation`. Expected values come from hand arguments written next to them, never from the code under test.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use super::catalogue::{parse_catalogue_str, parse_named_str};
use super::inflation::{
    CentreSet, FaceId, InflationResult, InflationRule, InflationSeed, PlaneTiling, VertexId, inflate, inflate_with,
};
use super::test_fixtures::named;
use crate::types::Cell2D;

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");
const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");
const SHEET_JSON: &str = include_str!("../../../../papertilling/data/sheet.json");
const MINIMAL_B_JSON: &str = include_str!("../../../../papertilling/data/minimal_b.json");

const CLUSTER: InflationRule = InflationRule::Cluster(CentreSet::MaxPolygon);

/// Faces sharing an edge with `f`, from the public corner lists only (independent of the tiling's cross-edge table).
fn edge_neighbours(t: &PlaneTiling, f: FaceId) -> Vec<FaceId> {
    let vs = t.face_vertices(f);
    let consecutive = |g: FaceId, u: VertexId, w: VertexId| {
        let ws = t.face_vertices(g);
        (0..ws.len()).any(|k| {
            let (a, b) = (ws[k], ws[(k + 1) % ws.len()]);
            (a, b) == (u, w) || (a, b) == (w, u)
        })
    };
    let mut out = Vec::new();
    for k in 0..vs.len() {
        let (u, w) = (vs[k], vs[(k + 1) % vs.len()]);
        out.extend(t.vertex_faces(u).into_iter().filter(|&g| g != f && consecutive(g, u, w)));
    }
    out
}

/// From-scratch cluster shells `S_0..S_n_max` (the definition in `inflation_grow.rs`, full sets every step). With
/// `with_seed`, every shell also contains `FillHoles(seed)` (a variant measured, not implemented); without `fill`,
/// the shell is `P_n U dec(P_n)` with no FillHoles at all (the seed is still filled).
fn cluster_reference(t: &PlaneTiling, seed: InflationSeed, mask: &[bool], n_max: usize, with_seed: bool, fill: bool) -> Vec<HashSet<FaceId>> {
    let mut s0 = t.seed_faces(seed).unwrap();
    t.fill_holes(&mut s0);
    let mut ring = s0.clone();
    let mut p: HashSet<FaceId> = ring.iter().copied().filter(|g| mask[g.0]).collect();
    while p.is_empty() {
        let more = t.completion(&ring);
        ring.extend(more);
        p = ring.iter().copied().filter(|g| mask[g.0]).collect();
    }
    let mut out = Vec::new();
    for n in 0..=n_max {
        if n > 0 {
            let more: Vec<FaceId> = t.completion(&p).into_iter().filter(|g| mask[g.0]).collect();
            p.extend(more);
        }
        let mut s = p.clone();
        for &c in &p {
            s.extend(edge_neighbours(t, c).into_iter().filter(|g| !mask[g.0]));
        }
        if with_seed {
            s.extend(s0.iter().copied());
        }
        if fill {
            t.fill_holes(&mut s);
        }
        out.push(s);
    }
    out
}

/// Every stored cell the plane inflation tests load: the 21 named lattices and every catalogue drawing.
fn all_cells() -> Vec<(String, Cell2D)> {
    let mut out: Vec<(String, Cell2D)> =
        parse_named_str(NAMED_JSON).unwrap().named.into_iter().map(|l| (l.key.clone(), l.cell)).collect();
    for (file, text) in [("points", POINTS_JSON), ("sheet", SHEET_JSON), ("minimal_b", MINIMAL_B_JSON)] {
        for (grp, cat) in parse_catalogue_str(text).unwrap() {
            for (idx, ent) in cat.drawings.iter().enumerate() {
                out.push((format!("{file}/{grp}/drawings[{idx}]"), ent.family.cell()));
            }
        }
    }
    out
}

type Tally = BTreeMap<&'static str, usize>;

fn is_disk(tp: &super::inflation::PatchTopology) -> bool {
    tp.chi == 1 && tp.pinch_vertices == 0
}

/// Run the cluster rule (max-polygon centres) from every Face, VertexStar and the Dirichlet seed of `cell`, n <= 4,
/// without the S~_0 disk requirement, and tally every shell; failing shells are appended to `bad`. Also tallies
/// the BDF Dirichlet seed (S~_0 a disk or not) and the variant with the seed kept in every cluster shell.
fn tally_cell(label: &str, cell: &Cell2D, tally: &mut Tally, bad: &mut Vec<String>, cell_notes: &mut Vec<String>) {
    let t = PlaneTiling::new(cell).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mask = t.centre_mask(&CentreSet::MaxPolygon).unwrap();
    // Cell-level structure: centre faces with no other centre at a vertex (P cannot grow through them), and
    // non-centre orbits with no edge on a centre (they enter a shell only through FillHoles).
    let isolated: Vec<usize> = (0..t.orbit_count())
        .filter(|&f| mask[f] && !t.completion(&HashSet::from([(f, [0, 0])])).iter().any(|g| mask[g.0]))
        .collect();
    let undecorated: Vec<usize> = (0..t.orbit_count())
        .filter(|&f| !mask[f] && !edge_neighbours(&t, (f, [0, 0])).iter().any(|g| mask[g.0]))
        .collect();
    *tally.entry("cells").or_default() += 1;
    *tally.entry("cells with isolated centre orbits").or_default() += usize::from(!isolated.is_empty());
    *tally.entry("cells with undecorated orbits").or_default() += usize::from(!undecorated.is_empty());
    if !isolated.is_empty() || !undecorated.is_empty() {
        cell_notes.push(format!("{label}: sizes {:?}, isolated centre orbits {isolated:?}, orbits with no edge on a centre {undecorated:?}", t.face_sizes()));
    }
    let bdf = t.grow(t.seed_faces(InflationSeed::Dirichlet).unwrap(), &InflationRule::Bdf, 0).unwrap();
    let key = if is_disk(&bdf.shells[0].topology) { "BDF Dirichlet S0 disk" } else { "BDF Dirichlet S0 NOT disk" };
    *tally.entry(key).or_default() += 1;
    if !is_disk(&bdf.shells[0].topology) {
        bad.push(format!("{label} BDF Dirichlet S0: {:?}", bdf.shells[0].topology));
    }
    // The removed CellStar seed (union of the basis-site stars at offset (0, 0)), for comparison.
    let mut star: HashSet<FaceId> = (0..cell.basis.len()).flat_map(|s| t.vertex_faces((s, [0, 0]))).collect();
    t.fill_holes(&mut star);
    *tally.entry(if is_disk(&t.topology(&star)) { "old CellStar S0 disk" } else { "old CellStar S0 NOT disk" }).or_default() += 1;
    let seeds = (0..t.orbit_count()).map(InflationSeed::Face).chain((0..cell.basis.len()).map(InflationSeed::VertexStar));
    for seed in seeds.chain([InflationSeed::Dirichlet]) {
        *tally.entry("runs").or_default() += 1;
        let res = match t.grow(t.seed_faces(seed).unwrap(), &CLUSTER, 4) {
            Ok(r) => r,
            Err(e) => {
                *tally.entry("errors").or_default() += 1;
                bad.push(format!("{label} {seed:?}: error {e}"));
                continue;
            }
        };
        let reference = cluster_reference(&t, seed, &mask, 4, false, true);
        let raw = cluster_reference(&t, seed, &mask, 4, false, false);
        let variant = cluster_reference(&t, seed, &mask, 4, true, true);
        // Growth of |P_n|: zero (stalled), constant first difference (1D chains), or growing (2D).
        let pc: Vec<usize> = res.shells.iter().map(|s| s.centres.unwrap()).collect();
        let d: Vec<usize> = (1..=4).map(|n| pc[n] - pc[n - 1]).collect();
        let growth = if d[3] == 0 {
            "P growth: stalled"
        } else if d[1] == d[2] && d[2] == d[3] {
            "P growth: linear (1D)"
        } else {
            "P growth: 2D"
        };
        *tally.entry(growth).or_default() += 1;
        if growth != "P growth: 2D" {
            bad.push(format!("{label} {seed:?}: {growth}, |P_n| = {pc:?}"));
        }
        for s in &res.shells {
            *tally.entry("shells").or_default() += 1;
            let tp = &s.topology;
            let drawn: HashSet<FaceId> = res.faces.iter().filter(|f| f.shell <= s.n).map(|f| (f.orbit, f.offset)).collect();
            if drawn != reference[s.n] {
                *tally.entry("reference mismatch").or_default() += 1;
                bad.push(format!("{label} {seed:?} n={}: differs from the from-scratch reference", s.n));
            }
            // Disk without FillHoles ever firing: the unfilled P_n U dec(P_n) is already the shell.
            let key = match (is_disk(tp), raw[s.n] == reference[s.n]) {
                (true, true) => "shell disk, no FillHoles",
                (true, false) => "shell disk after FillHoles",
                (false, _) => "shell NOT disk",
            };
            *tally.entry(key).or_default() += 1;
            if !is_disk(tp) {
                bad.push(format!("{label} {seed:?} n={}: chi={} pinch={} filled={}", s.n, tp.chi, tp.pinch_vertices, s.filled));
            }
            *tally.entry("shells with dangling bonds").or_default() += usize::from(tp.dangling_bonds > 0);
            let dmax = tally.entry("max dangling bonds").or_default();
            *dmax = (*dmax).max(tp.dangling_bonds);
            let vt = t.topology(&variant[s.n]);
            *tally.entry("variant with seed: shell NOT disk").or_default() += usize::from(!is_disk(&vt));
        }
    }
}

#[test]
#[ignore = "measurement: cluster rule over every stored cell, prints the tally (run with --ignored --nocapture)"]
fn cluster_measurement_sweep() {
    let cells = all_cells();
    let t0 = Instant::now();
    let (mut bad, mut notes) = (Vec::new(), Vec::new());
    let mut by_source: BTreeMap<String, Tally> = BTreeMap::new();
    for (label, cell) in &cells {
        let src = if label.contains('/') { label.split('/').next().unwrap() } else { "named" };
        tally_cell(label, cell, by_source.entry(src.to_string()).or_default(), &mut bad, &mut notes);
    }
    println!("# time cluster sweep: {:?} over {} cells", t0.elapsed(), cells.len());
    println!("by source: {by_source:#?}");
    println!("cell notes ({}):
{}", notes.len(), notes.join("
"));
    println!("failing ({}):
{}", bad.len(), bad.join("
"));
}

fn kagome_hex(t: &PlaneTiling) -> usize {
    t.face_sizes().iter().position(|&k| k == 6).unwrap()
}

/// `(hexagons, triangles)` per shell of a kagome inflation.
fn hex_tri(res: &InflationResult) -> Vec<(usize, usize)> {
    res.shells
        .iter()
        .map(|s| {
            let h = res.faces.iter().filter(|f| f.shell <= s.n && f.polygon.len() == 6).count();
            (h, s.topology.faces - h)
        })
        .collect()
}

#[test]
fn cluster_kagome_hexagon_seed_known_counts() {
    // Kagome = rectified triangular lattice T (see inflation_kagome_hexagon_seed_known_counts): a hexagon sits on a
    // T-vertex, a triangle is the medial triangle of a T-triangle, a kagome vertex is the midpoint of a T-edge.
    // Centres = hexagons (the max polygon). Two hexagons share a kagome vertex iff their T-vertices are adjacent and
    // never share an edge, so P_n = the hexagons on the graph ball B(n) of T: 1 + 6(1 + .. + n) = 3n(n+1) + 1.
    // A triangle shares an edge with hexagon a iff a is a corner of its T-triangle, so the decoration is the
    // T-triangles with a vertex in B(n): those of the lattice hexagon of side n+1, 6(n+1)^2 of them. Total
    // 7, 31, 73, 133. Hexagons plus the triangles of a convex lattice hexagon: a disk, nothing to fill.
    // Dangling bonds: a non-patch kagome edge lies on hexagon X and triangle (X, a, b) with none of X, a, b in B(n).
    // Its ends m(Xa), m(Xb) are patch vertices only if X, a and b all lie on the ring at distance n+1 (each pair
    // adjacent to a common point of B(n)), three mutually adjacent ring points, which a hexagonal ring has not: 0.
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    let want: Vec<(usize, usize)> = (0..=3).map(|n| (3 * n * (n + 1) + 1, 6 * (n + 1) * (n + 1))).collect();
    for seed in [InflationSeed::Face(kagome_hex(&t)), InflationSeed::Dirichlet] {
        let res = inflate_with(&cell, seed, &CLUSTER, 3).unwrap();
        assert_eq!(hex_tri(&res), want, "{seed:?}");
        for s in &res.shells {
            assert_eq!(s.centres, Some(3 * s.n * (s.n + 1) + 1));
            let tp = &s.topology;
            assert_eq!((tp.chi, tp.pinch_vertices, tp.dangling_bonds, s.filled), (1, 0, 0, 0), "{seed:?} n={}", s.n);
        }
    }
    // Plain BDF from the same 7-tile seed (hexagon + its 6 edge triangles). S_1: each hexagon vertex adds the
    // hexagon opposite (6), each triangle tip adds the tip triangle (6): 19. From n = 2 on S_n equals the BDF shell
    // of the single hexagon (9n^2 + 3n + 1 = 43, 91): S_1 contains that S_1, and the tip triangles' vertices only
    // touch faces of that S_2. That shell has 6n notches: for each pair of adjacent T-vertices on the ring at
    // distance n, the outward T-triangle is not in the patch but two of its edges lie on ring hexagons, so its third
    // bond joins two patch vertices on no patch face. The tip triangles are exactly the 6 notches of n = 1, so the
    // dangling counts are 0, 0, 12, 18.
    let res = inflate_with(&cell, InflationSeed::Dirichlet, &InflationRule::Bdf, 3).unwrap();
    let got: Vec<(usize, usize)> = res.shells.iter().map(|s| (s.topology.faces, s.topology.dangling_bonds)).collect();
    assert_eq!(got, vec![(7, 0), (19, 0), (43, 12), (91, 18)]);
}

#[test]
fn cluster_seed_fallback_kagome_triangle() {
    // Seed = one kagome triangle, the medial triangle of a T-triangle (a, b, c): no centre in it, so P_0 = the
    // hexagons at its vertices, {a, b, c}. Decoration: the T-triangles with a vertex in {a, b, c}, by inclusion-
    // exclusion 3*6 - 3*2 + 1 = 13 (each edge of abc lies on 2 triangles, abc itself on all three). |S_0| = 16.
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    let tri = t.face_sizes().iter().position(|&k| k == 3).unwrap();
    let res = inflate_with(&cell, InflationSeed::Face(tri), &CLUSTER, 0).unwrap();
    assert_eq!((res.shells[0].centres, res.shells[0].topology.faces), (Some(3), 16));
}

type Form = fn(usize) -> usize;

#[test]
fn cluster_equals_bdf_on_regular_tilings() {
    // Every face of {4,4}, {6,3}, {3,6} is a centre, so S_n = FillHoles(P_n) with P_{n+1} = P_n U U(P_n): BDF, given
    // that no hole appears (the BDF shells there have filled == 0, pinned in inflation_tests). Closed forms as there.
    let forms: [(&str, Form, Form); 3] = [
        ("square", |n| (2 * n + 1) * (2 * n + 1), |n| (2 * n + 2) * (2 * n + 2)),
        ("honeycomb", |n| 3 * n * (n + 1) + 1, |n| 3 * (n + 1) * (n + 1)),
        ("triangular", |n| 6 * n * n + 6 * n + 1, |n| 6 * (n + 1) * (n + 1)),
    ];
    for (key, face, vert) in forms {
        let cell = named(key).cell;
        let t = PlaneTiling::new(&cell).unwrap();
        let faces = (0..t.orbit_count()).map(|f| (InflationSeed::Face(f), face));
        for (seed, form) in faces.chain((0..cell.basis.len()).map(|s| (InflationSeed::VertexStar(s), vert))) {
            let res = inflate_with(&cell, seed, &CLUSTER, 4).unwrap();
            let got: Vec<usize> = res.shells.iter().map(|s| s.topology.faces).collect();
            assert_eq!(got, (0..=4).map(form).collect::<Vec<_>>(), "{key} {seed:?}");
            assert!(res.shells.iter().all(|s| s.centres == Some(s.topology.faces) && s.filled == 0), "{key} {seed:?}");
        }
    }
}

#[test]
fn cluster_one_dimensional_centres_elongated_triangular() {
    // elongated_triangular (3^3.4^2): rows of unit squares between rows of unit triangles (bond geometry checked
    // below: every bond has length 1 and |dy| in {0, sqrt3/2, 1}, the dy = 0 bonds horizontal). A square's
    // corners lie on the two horizontal lines of its row, and the next row of squares lies on other lines (a
    // triangle row of height sqrt3/2 in between), so two squares share a vertex only within a row, where that
    // means sharing an edge. P_n from one square is a run of 2n + 1 squares: the max-polygon cluster rule grows in
    // one direction only (a strip, not a 2D ball).
    let cell = named("elongated_triangular").cell;
    for b in &cell.bonds {
        let d: Vec<f64> = (0..2)
            .map(|ax| cell.basis[b.j][ax] + b.n1 as f64 * cell.a1[ax] + b.n2 as f64 * cell.a2[ax] - cell.basis[b.i][ax])
            .collect();
        assert!((d[0].hypot(d[1]) - 1.0).abs() < 1e-9, "bond {b:?} = {d:?}");
        assert!([0.0, 3f64.sqrt() / 2.0, 1.0].iter().any(|&h| (d[1].abs() - h).abs() < 1e-9), "bond {b:?} = {d:?}");
    }
    let t = PlaneTiling::new(&cell).unwrap();
    let sq = t.face_sizes().iter().position(|&k| k == 4).unwrap();
    let res = inflate_with(&cell, InflationSeed::Face(sq), &CLUSTER, 4).unwrap();
    assert_eq!(res.shells.iter().map(|s| s.centres.unwrap()).collect::<Vec<_>>(), vec![1, 3, 5, 7, 9]);
}

#[test]
fn cluster_centre_set_errors_and_tile_class_selector() {
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    assert!(t.centre_mask(&CentreSet::Orbits(vec![7])).is_err());
    assert!(t.centre_mask(&CentreSet::Orbits(vec![])).is_err());
    assert!(t.centre_mask(&CentreSet::TileClasses(vec![2])).is_err());
    // p6m kagome: the hexagon is one class; the up and down triangles are swapped by the 6-fold rotation, so they
    // form a second class: 2 classes, and selecting the hexagon's class equals MaxPolygon.
    let classes = t.tile_classes();
    let hex = kagome_hex(&t);
    assert_eq!(classes.iter().max(), Some(&1));
    let max = t.centre_mask(&CentreSet::MaxPolygon).unwrap();
    assert_eq!(t.centre_mask(&CentreSet::TileClasses(vec![classes[hex]])).unwrap(), max);
    assert_eq!(t.centre_mask(&CentreSet::Orbits(vec![hex])).unwrap(), max);
}

fn root(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        x = p[x];
    }
    x
}

#[test]
fn tile_classes_match_dsymbol_tile_count() {
    // Delaney-Dress: the tiles of a periodic tiling up to its symmetry group G are the <s0, s1>-orbits of its
    // D-symbol (s0 changes the vertex, s1 the edge, keeping the tile). `Cell2D::dsymbol` is built under the same
    // group as `tile_classes`, so the class count must equal that orbit count on every named lattice, and each
    // class must have a single face size.
    for lat in &parse_named_str(NAMED_JSON).unwrap().named {
        let t = PlaneTiling::new(&lat.cell).unwrap();
        let ds = lat.cell.dsymbol().unwrap();
        let n = ds.s0.len();
        let mut parent: Vec<usize> = (0..n).collect();
        for c in 0..n {
            for d in [ds.s0[c], ds.s1[c]] {
                let (a, b) = (root(&mut parent, c), root(&mut parent, d));
                parent[a.max(b)] = a.min(b);
            }
        }
        let orbits: HashSet<usize> = (0..n).map(|c| root(&mut parent, c)).collect();
        let classes = t.tile_classes();
        assert_eq!(classes.iter().max().unwrap() + 1, orbits.len(), "{}", lat.key);
        let sizes = t.face_sizes();
        for f in 0..t.orbit_count() {
            assert!((0..t.orbit_count()).all(|g| classes[f] != classes[g] || sizes[f] == sizes[g]), "{}", lat.key);
        }
    }
}

#[test]
fn cluster_named_lattices_match_reference_and_are_disks() {
    // Every named lattice, every Face / VertexStar seed and the Dirichlet seed, n <= 4: the incremental cluster
    // shells equal the from-scratch definition, and each is a closed disk (measured: none needs FillHoles).
    let t0 = Instant::now();
    let mut bad = Vec::new();
    for lat in &parse_named_str(NAMED_JSON).unwrap().named {
        let t = PlaneTiling::new(&lat.cell).unwrap();
        let mask = t.centre_mask(&CentreSet::MaxPolygon).unwrap();
        let seeds = (0..t.orbit_count()).map(InflationSeed::Face).chain((0..lat.cell.basis.len()).map(InflationSeed::VertexStar));
        for seed in seeds.chain([InflationSeed::Dirichlet]) {
            let res = inflate_with(&lat.cell, seed, &CLUSTER, 4).unwrap();
            let reference = cluster_reference(&t, seed, &mask, 4, false, true);
            for s in &res.shells {
                let drawn: HashSet<FaceId> = res.faces.iter().filter(|f| f.shell <= s.n).map(|f| (f.orbit, f.offset)).collect();
                if drawn != reference[s.n] || !is_disk(&s.topology) || s.topology != t.topology(&reference[s.n]) {
                    bad.push(format!("{} {seed:?} n={}", lat.key, s.n));
                }
            }
        }
    }
    println!("# time named cluster reference check: {:?}", t0.elapsed());
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn cluster_pilot_kagome() {
    let t0 = Instant::now();
    let cell = named("kagome").cell;
    let t = PlaneTiling::new(&cell).unwrap();
    let hex = t.face_sizes().iter().position(|&k| k == 6).unwrap();
    let res = inflate_with(&cell, InflationSeed::Face(hex), &CLUSTER, 3).unwrap();
    println!("# time pilot kagome cluster n=3: {:?}", t0.elapsed());
    // Hexagon seed, n = 1: the hexagon and its 6 vertex-neighbour hexagons, each with 4 edge triangles: 7 + 24.
    assert_eq!(res.shells[1].topology.faces, 31);
}

#[test]
fn auto_rule_kagome_and_regular_vs_fallback_lattices() {
    let kag = named("kagome").cell;
    let kag_hex = kagome_hex(&PlaneTiling::new(&kag).unwrap());
    let kag_res = inflate(&kag, InflationSeed::Dirichlet, 3).unwrap();
    assert_eq!(kag_res.rule_used, CLUSTER);
    assert!(kag_res.readout.contains("cluster"), "{}", kag_res.readout);
    assert_eq!(hex_tri(&kag_res), (0..=3).map(|n| (3 * n * (n + 1) + 1, 6 * (n + 1) * (n + 1))).collect::<Vec<_>>());

    for key in ["square", "honeycomb", "triangular", "shastry_sutherland"] {
        let res = inflate(&named(key).cell, InflationSeed::Dirichlet, 2).unwrap();
        assert_eq!(res.rule_used, CLUSTER, "{key}: {}", res.readout);
        assert!(res.readout.contains("cluster"), "{key}: {}", res.readout);
    }

    let pts_cat = parse_catalogue_str(POINTS_JSON).unwrap();
    let fallbacks = [
        ("elongated_triangular", named("elongated_triangular").cell, "1D strips"),
        ("shastrys_ladder", named("shastrys_ladder").cell, "1D strips"),
        ("points/p6/drawings[0]", pts_cat["p6"].drawings[0].family.cell(), "connected component"),
        ("points/p31m/drawings[6]", pts_cat["p31m"].drawings[6].family.cell(), "connected component"),
        ("points/p4/drawings[7]", pts_cat["p4"].drawings[7].family.cell(), "shares no edge"),
    ];
    for (label, cell, why) in fallbacks {
        let res = inflate(&cell, InflationSeed::Dirichlet, 3).unwrap();
        assert_eq!(res.rule_used, InflationRule::Bdf, "{label}: {}", res.readout);
        assert!(res.readout.contains("plain BDF (auto fallback"), "{label}: {}", res.readout);
        assert!(res.readout.contains(why), "{label}: missing `{why}` in {}", res.readout);
        assert!(res.shells.iter().all(|s| is_disk(&s.topology)), "{label}");
    }

    // Hand overrides bypass automatic selection and state override in readout.
    let kag_bdf = inflate_with(&kag, InflationSeed::Face(kag_hex), &InflationRule::Bdf, 3).unwrap();
    assert_eq!(kag_bdf.rule_used, InflationRule::Bdf);
    assert_eq!(kag_bdf.readout, "plain BDF (override)");
    assert_eq!(kag_bdf.shells.iter().map(|s| s.topology.faces).collect::<Vec<_>>(), vec![1, 13, 43, 91]);

    let el = named("elongated_triangular").cell;
    let el_sq = PlaneTiling::new(&el).unwrap().face_sizes().iter().position(|&k| k == 4).unwrap();
    let el_cl = inflate_with(&el, InflationSeed::Face(el_sq), &CLUSTER, 4).unwrap();
    assert_eq!(el_cl.rule_used, CLUSTER);
    assert!(el_cl.readout.contains("override"), "{}", el_cl.readout);
    assert_eq!(el_cl.shells.iter().map(|s| s.centres.unwrap()).collect::<Vec<_>>(), vec![1, 3, 5, 7, 9]);
}

#[test]
fn auto_rule_all_685_cells_give_disks() {
    let cells = all_cells();
    assert_eq!(cells.len(), 685);
    let (mut cluster_n, mut bdf_n) = (0usize, 0usize);
    let mut bad = Vec::new();
    for (label, cell) in &cells {
        let res = match inflate(cell, InflationSeed::Dirichlet, 4) {
            Ok(r) => r,
            Err(e) => {
                bad.push(format!("{label}: {e}"));
                continue;
            }
        };
        match res.rule_used {
            InflationRule::Cluster(CentreSet::MaxPolygon) => cluster_n += 1,
            InflationRule::Bdf => bdf_n += 1,
            other => bad.push(format!("{label}: unexpected rule {other:?}")),
        }
        for s in &res.shells {
            if !is_disk(&s.topology) {
                bad.push(format!("{label} n={}: {:?}", s.n, s.topology));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    // Measured regression pin over all 685 stored cells (0 disk failures: 574 eligible for cluster, 111 plain-BDF fallback).
    assert_eq!((cluster_n, bdf_n), (574, 111));
}
