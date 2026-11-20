//! Points-catalogue tests (C1 WP-U4). Expected values come from hand derivations or from the independent
//! scratch script `u4_vor_touch.py` (scipy/Qhull Voronoi of a finite textbook block, touching = a shared
//! Voronoi vertex; every maximal vertex of the origin cell tried, all agree); none is read from the code
//! under test.

use std::time::Instant;

use super::*;

const CAP: usize = 50_000;

fn len(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Bonds at drawn site `i` and their lengths.
fn bonds_at(g: &PointsGrowth, i: usize) -> Vec<f64> {
    let p = &g.patch.positions;
    g.patch.bonds.iter().filter(|b| b.0 == i || b.1 == i).map(|b| len(p[b.0], p[b.1])).collect()
}

fn layer_counts(num: u16, seed: SeedKind, layers: u32) -> Vec<usize> {
    let g = grow_points3d(num, 0, seed, layers, CAP).unwrap_or_else(|e| panic!("#{num}: {e}"));
    g.stats.iter().map(|s| s.cells).collect()
}

#[test]
fn every_group_shows_seven_entries_and_the_coverage_agrees() {
    // the sweep's coverage report (a separate file) states >= 7 entries for all 230 groups
    for num in 1..=230_u16 {
        let g = points_group(num).unwrap();
        assert_eq!(g.group, num);
        assert_eq!(g.shown().len(), 7, "#{num}");
        let c = points_coverage(num).expect("coverage row");
        assert_eq!(c.entries, g.entries.len(), "#{num}: the coverage counts every entry found");
        for e in &g.entries {
            assert_eq!(e.frac.len(), e.sites, "#{num}");
            assert_eq!(e.labels.len(), e.sites, "#{num}");
            assert!(e.bonds.iter().all(|b| b.0 < e.sites && b.1 < e.sites), "#{num}");
        }
        assert!(coverage_line(num).starts_with("7 of 7 found"), "{}", coverage_line(num));
    }
    assert!(points_group(0).is_err() && points_group(231).is_err());
    assert!(grow_points3d(221, 99, SeedKind::Cell, 1, CAP).is_err());
    let note = points_note();
    assert!(note.contains("COMPUTATIONAL EVIDENCE") && note.contains("coordination sequences"), "{note}");
}

#[test]
fn known_first_entries_have_the_textbook_bonds() {
    // plan §5 T4: entry 1 of Pm-3m, Im-3m, Fm-3m, Fd-3m is sc, bcc, fcc, diamond. Conventional a = 1, so the
    // bond lengths are 1, sqrt(3)/2, 1/sqrt(2), sqrt(3)/4 (hand); the seed cell (drawn site 0) has all its
    // bonded neighbours in layer 1 because each bond crosses a shared Voronoi face.
    for (num, deg, l) in [
        (221_u16, 6, 1.0),
        (229, 8, 0.75_f64.sqrt()),
        (225, 12, 0.5_f64.sqrt()),
        (227, 4, 3.0_f64.sqrt() / 4.0),
    ] {
        let g = grow_points3d(num, 0, SeedKind::Cell, 1, CAP).unwrap();
        let b = bonds_at(&g, 0);
        assert_eq!(b.len(), deg, "#{num}");
        assert!(b.iter().all(|x| (x - l).abs() < 1e-9), "#{num}: {b:?}");
        // every drawn bond has the same length (one bond orbit)
        let p = &g.patch.positions;
        assert!(g.patch.bonds.iter().all(|b| (len(p[b.0], p[b.1]) - l).abs() < 1e-9), "#{num}");
    }
}

#[test]
fn growth_matches_the_independent_voronoi_counts() {
    // sc by hand: unit cubes, so a cell seed gives (2n+1)^3 and a vertex seed (2n+2)^3. The others from
    // u4_vor_touch.py (output pasted in the WP-U4 report).
    let cases: [(u16, [usize; 3], [usize; 3]); 4] = [
        (221, [1, 27, 125], [8, 64, 216]),
        (229, [1, 15, 65], [4, 32, 108]),
        (225, [1, 19, 85], [6, 44, 146]),
        (227, [1, 29, 147], [6, 62, 232]),
    ];
    for (num, cell, vertex) in cases {
        let t = Instant::now();
        assert_eq!(layer_counts(num, SeedKind::Cell, 2), cell, "#{num} cell seed");
        assert_eq!(layer_counts(num, SeedKind::Vertex, 2), vertex, "#{num} vertex seed");
        println!("# time #{num}: {:.3} s", t.elapsed().as_secs_f64());
    }
    // a ball of convex cells: chi = 1 for every layer (V - E + F - C of a contractible union)
    let g = grow_points3d(221, 0, SeedKind::Cell, 2, CAP).unwrap();
    assert!(g.stats.iter().all(|s| s.euler == 1), "{:?}", g.stats);
}

#[test]
fn the_cell_cap_stops_growth_visibly() {
    // sc, cell seed: layer 2 has 125 cells (hand), so a cap of 100 stops there with nothing past layer 1
    let g = grow_points3d(221, 0, SeedKind::Cell, 4, 100).unwrap();
    assert_eq!(g.cells, 27);
    assert!(matches!(g.stopped, Some(Stopped::CellCap { layer: 2, .. })), "{:?}", g.stopped);
}

#[test]
#[ignore = "timing pilot: run with --ignored --nocapture"]
fn pilot_time_largest_entries() {
    for (num, idx) in [(1_u16, 0_usize), (227, 0), (226, 4), (228, 2), (230, 0)] {
        let sites = points_group(num).unwrap().entries[idx].sites;
        for layers in [1_u32, 2] {
            let t = Instant::now();
            let r = grow_points3d(num, idx, SeedKind::Vertex, layers, CAP);
            let n = r.as_ref().map_or(0, |g| g.cells);
            println!("# time #{num} entry {idx} ({sites} sites) layers {layers}: {:.3} s, {n} cells, ok={}", t.elapsed().as_secs_f64(), r.is_ok());
        }
    }
}
