//! Coordinate-only detection against the crystal builders, `burgers_circuit` and `charge::frank_angles`:
//! every crystal is reduced to `pos` before detection.

use std::collections::BTreeMap;
use std::time::Instant;

use super::*;
use crate::defect::charge::{frame_faces, frank_angles};
use crate::defect::circuit::{burgers_circuit, shift_circuit_basepoint};
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::plane::draw::relax_picture;
use crate::types::{Bond2D, BuiltCrystal, Cell2D, Frame2D};

fn square_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]],
        bonds: vec![Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 }, Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 }],
    }
}

fn honeycomb_cell() -> Cell2D {
    let s3 = 3.0_f64.sqrt();
    Cell2D {
        a1: [s3, 0.0], a2: [0.5 * s3, 1.5], basis: vec![[0.0, 0.0], [0.5 * s3, 0.5]],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 0, n1: 0, n2: 1, kind: 0 },
        ],
    }
}

fn relaxed(cell: &Cell2D, specs: &[DefectSpec], r: f64) -> BuiltCrystal {
    let mut c = build_point_defects(cell, specs, Some(r), Some([0.0, 0.0])).unwrap();
    relax_picture(&mut c);
    c
}

fn detect(c: &BuiltCrystal, group: &str, cell: &Cell2D) -> CoordDetection {
    let pos: Vec<[f64; 2]> = c.pos.clone();
    detect_defects_from_coords(&pos, group, cell, Frame2D::default(), None, None, 1).unwrap()
}

/// Every detected face that is also a `frame_faces` face carries the same Frank angle as `frank_angles`.
fn assert_face_signs_match(c: &BuiltCrystal, det: &CoordDetection) {
    let ff = frame_faces(c);
    let fa = frank_angles(c, &ff);
    let key = |f: &[usize]| { let mut k = f.to_vec(); k.sort_unstable(); k };
    let reference: BTreeMap<Vec<usize>, f64> = ff.iter().zip(&fa).map(|(f, &a)| (key(f), a)).collect();
    let mut matched = 0;
    for (f, &q) in det.faces.iter().zip(&det.face_frank) {
        if let Some(&a) = reference.get(&key(f)) {
            assert!((a - q).abs() < 1e-6, "face {f:?}: frank_angles {a} vs detector {q}");
            matched += 1;
        }
    }
    assert!(matched * 10 >= det.faces.len() * 9, "only {matched} of {} faces matched", det.faces.len());
}

#[test]
fn dislocation_round_trip_square_and_honeycomb() {
    let s3 = 3f64.sqrt();
    // Cores sit inside a plaquette: (0.37, 0.41) in the unit square, (0.07, 1.09) beside the hexagon centre (0, 1) of
    // this honeycomb cell (A at the origin, B at (sqrt3/2, 1/2)). (0.37, 0.41) would be 0.17 from the A-B bond midpoint
    // (sqrt3/4, 1/4) of this cell: the cut then leaves a 1-coordinated site 0.6 (relaxed) from a non-neighbour, below the
    // bond cutoff (1 + sqrt3) / 2, which no coordinate-only bond graph can reproduce.
    for (group, cell, b, at, weber) in [
        ("p4m", square_cell(), [1.0, 0.0], [0.37, 0.41], None),
        ("p6m", honeycomb_cell(), [s3, 0.0], [0.07, 1.09], Some("1/3[2 -1 -1 0]")),
    ] {
        let t0 = Instant::now();
        let c = relaxed(&cell, &[DefectSpec::dislocation(b, Some(at))], 7.0);
        let det = detect(&c, group, &cell);
        println!("# time {group} build + detect: {:.3} s ({} sites)", t0.elapsed().as_secs_f64(), c.pos.len());
        let mut ce: Vec<(usize, usize)> = c.edges.iter().map(|&(u, v)| (u.min(v), u.max(v))).collect();
        ce.sort_unstable();
        assert_eq!(ce, det.edges, "{group}: bond graph rebuilt from coordinates differs from the crystal's");
        assert_face_signs_match(&c, &det);
        let ds: Vec<&DetectedDefect> = det.defects.iter().filter(|d| d.kind != "neutral").collect();
        assert_eq!(ds.len(), 1, "{group}: {:?}", det.defects.iter().map(|d| &d.kind).collect::<Vec<_>>());
        let d = ds[0];
        let bd = d.burgers.unwrap();
        assert_eq!(d.kind, "dislocation");
        assert!((bd[0] - b[0]).abs() < 1e-9 && (bd[1] - b[1]).abs() < 1e-9, "{group}: b {bd:?}");
        assert_eq!(d.uv, Some([1, 0]));
        assert_eq!(d.index.as_ref().and_then(|i| i.weber_str.as_deref()), weber);
        // parity: the same loop through the exact parallel-transport engine
        let bc = burgers_circuit(&c, &d.loop_path, Some((group, &cell, Frame2D::default()))).unwrap();
        assert!((bc.closure_vector[0] - bd[0]).abs() < 1e-9 && (bc.closure_vector[1] - bd[1]).abs() < 1e-9);
        assert_eq!(bc.charge.unwrap().kind, "dislocation");
        // Nye: sum_f alpha A_f telescopes to the outer closure, which is b
        let nye = nye_tensor(&c.pos, &cell, Frame2D::default(), None, None, None).unwrap();
        let t = nye.total.unwrap();
        assert!((t[0] - b[0]).abs() < 1e-9 && (t[1] - b[1]).abs() < 1e-9, "{group}: nye {t:?}");
        assert!((nye.boundary_closure[0] - t[0]).abs() < 1e-9 && (nye.boundary_closure[1] - t[1]).abs() < 1e-9);
        let outer = burgers_circuit(&c, &nye.boundary_walk, None).unwrap().closure_vector;
        assert!((outer[0] - t[0]).abs() < 1e-9 && (outer[1] - t[1]).abs() < 1e-9, "{group}: outer circuit {outer:?}");
    }
}

#[test]
fn separated_disclination_dipole_honeycomb() {
    let s3 = 3f64.sqrt();
    let cell = honeycomb_cell();
    let (c1, d) = ([0.5 * s3, -0.5], [2.0 * s3, 0.0]);
    let specs = [DefectSpec::disclination(60.0, c1), DefectSpec::disclination(-60.0, [c1[0] + d[0], c1[1] + d[1]])];
    let c = relaxed(&cell, &specs, 8.0);
    let det = detect(&c, "p6m", &cell);
    assert_face_signs_match(&c, &det);
    let mut charged: Vec<(f64, Vec<usize>)> = det.clusters.iter().filter(|k| k.frank_deg != 0.0)
        .map(|k| (k.frank_deg, k.faces.iter().map(|&f| det.faces[f].len()).collect())).collect();
    charged.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(charged, vec![(-60.0, vec![7]), (60.0, vec![5])]);
    let dip: Vec<&DetectedDefect> = det.defects.iter().filter(|x| x.kind == "dipole").collect();
    assert_eq!(dip.len(), 1);
    let b = dip[0].burgers.unwrap();
    // |b| = 2 sin(30 deg) |d| = |d| and |b.d| = |d|^2 / 2 for b = +-(I - R(60)) d
    assert!((norm(b) - norm(d)).abs() < 1e-9 && ((b[0] * d[0] + b[1] * d[1]).abs() - 6.0).abs() < 1e-9, "b {b:?}");
    // burgers_circuit expresses the closure in the first vertex's chart (rotated by 60 deg between the cuts);
    // the physical-frame value must be one of the basepoint values
    let closures: Vec<[f64; 2]> = dip[0].loop_path[..dip[0].loop_path.len() - 1].iter()
        .map(|&v| burgers_circuit(&c, &shift_circuit_basepoint(&dip[0].loop_path, v).unwrap(), None).unwrap().closure_vector).collect();
    assert!(closures.iter().any(|cv| (cv[0] - b[0]).abs() < 1e-9 && (cv[1] - b[1]).abs() < 1e-9), "{closures:?}");
    // the transported frame starts at loop_path[0], so b is exactly burgers_circuit's closure of the returned walk
    let own = burgers_circuit(&c, &dip[0].loop_path, None).unwrap().closure_vector;
    assert!((own[0] - b[0]).abs() < 1e-9 && (own[1] - b[1]).abs() < 1e-9, "own {own:?} vs b {b:?}");
}

#[test]
fn lone_disclination_gets_no_burgers_vector() {
    let s3 = 3f64.sqrt();
    let cell = honeycomb_cell();
    let c = relaxed(&cell, &[DefectSpec::disclination(60.0, [0.5 * s3, -0.5])], 6.0);
    let det = detect(&c, "p6m", &cell);
    assert_eq!(det.defects.len(), 1);
    let d = &det.defects[0];
    assert_eq!((d.kind.as_str(), d.frank_deg, d.burgers), ("disclination", 60.0, None));
    assert!(d.reason.starts_with("[PROVED]"));
    assert!(nye_tensor(&c.pos, &cell, Frame2D::default(), None, None, None).unwrap().total.is_none());
}

/// Perfect honeycomb patch (explicit bonds) with the bond nearest the origin rotated by 90 degrees.
fn stone_wales() -> (Vec<[f64; 2]>, Vec<(usize, usize)>) {
    let cell = honeycomb_cell();
    let mut pos = Vec::new();
    let mut idx = BTreeMap::new();
    for n1 in -6..=6 { for n2 in -6..=6 { for (s, bs) in cell.basis.iter().enumerate() {
        let p = [bs[0] + f64::from(n1) * cell.a1[0] + f64::from(n2) * cell.a2[0], bs[1] + f64::from(n1) * cell.a1[1] + f64::from(n2) * cell.a2[1]];
        if norm(p) < 7.0 { idx.insert((n1, n2, s), pos.len()); pos.push(p); }
    } } }
    let mut edges = Vec::new();
    for (&(n1, n2, s), &u) in &idx { for b in cell.bonds.iter().filter(|b| b.i == s) {
        if let Some(&v) = idx.get(&(n1 + b.n1, n2 + b.n2, b.j)) { edges.push((u, v)); }
    } }
    let (p, q) = edges.iter().copied().min_by(|&(a, b), &(c, d)| {
        norm([pos[a][0] + pos[b][0], pos[a][1] + pos[b][1]]).total_cmp(&norm([pos[c][0] + pos[d][0], pos[c][1] + pos[d][1]]))
    }).unwrap();
    let mid = [0.5 * (pos[p][0] + pos[q][0]), 0.5 * (pos[p][1] + pos[q][1])];
    let outer: Vec<usize> = edges.iter().filter(|&&(a, b)| (a == p || a == q || b == p || b == q) && (a, b) != (p, q))
        .map(|&(a, b)| if a == p || a == q { b } else { a }).collect();
    for v in [p, q] {
        let r = rot(sub(pos[v], mid), std::f64::consts::FRAC_PI_2);
        pos[v] = [mid[0] + r[0], mid[1] + r[1]];
    }
    edges.retain(|&(a, b)| !(a == p || a == q || b == p || b == q));
    edges.push((p, q));
    for w in outer {
        edges.push((w, if norm(sub(pos[w], pos[p])) < norm(sub(pos[w], pos[q])) { p } else { q }));
    }
    (pos, edges)
}

#[test]
fn stone_wales_is_neutral_with_zero_burgers() {
    let (pos, edges) = stone_wales();
    let det = detect_defects_from_coords(&pos, "p6m", &honeycomb_cell(), Frame2D::default(), Some(&edges), None, 1).unwrap();
    let mut sizes: Vec<(usize, f64)> = det.faces.iter().zip(&det.face_frank).filter(|(_, q)| **q != 0.0).map(|(f, &q)| (f.len(), q)).collect();
    sizes.sort_by_key(|a| a.0);
    assert_eq!(sizes, vec![(5, 60.0), (5, 60.0), (7, -60.0), (7, -60.0)]);
    assert!(det.vertex_frank.iter().all(|&q| q == 0.0));
    assert_eq!(det.clusters.len(), 1);
    assert_eq!(det.defects.len(), 1);
    assert_eq!((det.defects[0].kind.as_str(), det.defects[0].burgers), ("neutral", Some([0.0, 0.0])));
}

#[test]
fn nye_annulus_counts_its_clockwise_hole_and_both_disconnected_components() {
    let pos: Vec<_> = (0..16).map(|i| [f64::from(i % 4), f64::from(i / 4)]).collect();
    let mut edges = Vec::new();
    for i in 0..16 {
        if i % 4 < 3 { edges.push((i, i + 1)); }
        if i < 12 { edges.push((i, i + 4)); }
    }
    let faces = PlanarFaces::new(&BondGraph::from_edges(&pos, &edges));
    let by_corner = |v| faces.faces.iter().position(|f| f.iter().min() == Some(&v)).unwrap();
    let ring: Vec<_> = (0..9).filter(|&i| i != by_corner(5)).collect();
    let islands = vec![by_corner(0), by_corner(10)];
    let point_contact = vec![by_corner(0), by_corner(5)];
    for (region, expected) in [(ring, [-1.0, 9.0]), (islands, [1.0, 1.0]), (point_contact, [1.0, 1.0])] {
        let walks = nye_loops::region_boundaries(&faces, &region.iter().copied().collect());
        let mut areas: Vec<_> = walks.iter().map(|w| signed_area(&pos, w)).collect();
        areas.sort_by(f64::total_cmp);
        assert_eq!(areas, expected);
        let nye = nye_tensor(&pos, &square_cell(), Frame2D::default(), Some(&edges), None, Some(&region)).unwrap();
        assert!(nye.boundary_walk.is_empty());
        assert_eq!(nye.total, Some([0.0; 2]));
        assert_eq!(nye.boundary_closure, [0.0; 2]);
    }
}
