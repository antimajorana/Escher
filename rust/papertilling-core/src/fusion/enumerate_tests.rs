//! Unit tests for the generalized fixed-cavity lattice-core enumerator (`fusion::enumerate`).
//!
//! All expected counts are independent hand calculations and published combinatorial formulas
//! (Catalan triangulations, Kirkman-Cayley diagonal-dissection counts, binomial spoke counts,
//! dihedral Burnside orbit counts, and Euler's formula).

use super::enumerate::{
    enumerate_cavity_chord_cores, enumerate_cavity_cores, wang_core_connections, wang_surrounding,
};
use super::map::{PlanarMap, VertexKind};
use super::wang::{A, B, C, D, E, F, G, H, gate_route, hamiltonian_route};

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

fn cyclic_eq(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && (a.is_empty() || (0..b.len()).any(|s| b.iter().cycle().skip(s).take(b.len()).eq(a.iter())))
}

/// Regular hexagon cavity with 6 identical external stubs (exact `D_6` cavity symmetry, order 12).
fn symmetric_hexagon() -> PlanarMap {
    let mut verts = Vec::with_capacity(12);
    let mut edges = Vec::with_capacity(12);
    for k in 0..6 {
        let th = (k as f64) * std::f64::consts::FRAC_PI_3;
        verts.push((format!("v{k}"), [th.cos(), th.sin()], VertexKind::Complete));
    }
    for k in 0..6 {
        let th = (k as f64) * std::f64::consts::FRAC_PI_3;
        verts.push((format!("s{k}"), [1.8 * th.cos(), 1.8 * th.sin()], VertexKind::Boundary));
    }
    for k in 0..6 {
        edges.push((k as u32 + 1, k, (k + 1) % 6));
    }
    for k in 0..6 {
        edges.push((k as u32 + 7, k, 6 + k));
    }
    let borrowed: Vec<(&str, [f64; 2], VertexKind)> = verts.iter().map(|(n, p, k)| (n.as_str(), *p, *k)).collect();
    PlanarMap::from_drawing(&borrowed, &edges, 7).expect("symmetric hexagon is a valid drawing")
}

#[test]
fn wang_core_connections_produces_exact_five_cores_two_orbits_and_slide_bfs() {
    let en = wang_core_connections();
    assert_eq!(en.cavity_sides, 8);
    assert_eq!(en.inner_edges, 2);
    assert_eq!(en.total_edges, 14);
    assert_eq!(en.max_inner_verts, 1);
    assert!(en.boundary_vertices_fixed);
    assert!(!en.truncated);
    assert_eq!(en.cores.len(), 5);
    assert_eq!(en.regular_count, 1);
    // Wang's 8-gon has legs on {c,d,g,h} (indices 2,3,6,7): D_2 subgroup of D_8 has order 4.
    // The 5 labelled cores partition into 2 symmetry orbits: {c-h, d-g} (size 1) and the 4 {3,4,4,5} cores (size 4).
    assert_eq!(en.cavity_symmetry_order, 4);
    assert_eq!(en.symmetry_orbit_count, 2);
    assert_eq!(en.slide_connected_to_regular, Some(true));
    assert!(en.provenance.contains("handshaking lemma"));
    assert!(en.provenance.contains("Author decisions (2026-11-22)"));

    let expected_chords: [Vec<(usize, usize)>; 5] = [
        vec![(C, H), (D, G)],
        vec![(C, G), (D, G)],
        vec![(C, G), (C, H)],
        vec![(C, H), (D, H)],
        vec![(D, G), (D, H)],
    ];
    for (i, (core, exp)) in en.cores.iter().zip(&expected_chords).enumerate() {
        assert_eq!(core.id, i);
        assert_eq!(core.inner_verts, 0);
        assert_eq!(&core.chords, exp, "core {i}");
        assert_eq!(core.map.euler(), 2, "core {i} must be connected planar");
        let c = &core.census;
        assert_eq!((c.vertices, c.edges, c.faces), (4, 14, 3), "core {i}");
        assert_eq!(c.square_curvature, 0, "core {i}");
        assert!(c.valences.iter().all(|&(_, k)| k >= 3), "core {i} valences");
        assert!(c.face_sides.iter().all(|&s| s >= 3), "core {i} faces");
        let vals = sorted(c.valences.iter().map(|&(_, k)| k).collect());
        let faces = sorted(c.face_sides.clone());
        if i == 0 {
            assert!(core.is_regular_square);
            assert_eq!((core.orbit_id, core.orbit_size), (0, 1));
            assert_eq!(vals, vec![4, 4, 4, 4]);
            assert_eq!(faces, vec![4, 4, 4]);
        } else {
            assert!(!core.is_regular_square);
            assert_eq!((core.orbit_id, core.orbit_size), (1, 4));
            assert_eq!(vals, vec![3, 4, 4, 5]);
            assert_eq!(faces, vec![3, 4, 5]);
        }
    }
}

#[test]
fn wang_core_0_matches_end_of_both_routes_and_core_1_matches_step_4() {
    let en = wang_core_connections();
    let gate = gate_route().unwrap();
    let ham = hamiltonian_route().unwrap();

    let core0 = &en.cores[0].map;
    let fused_gate = &gate[5].map;
    let fused_ham = &ham[2].map;
    let relabel_ham = |l: u32| match l { 15 => 12, 16 => 11, other => other };
    for v in [A, B, C, D, E, F, G, H] {
        assert!(cyclic_eq(&core0.map_order(v), &fused_gate.map_order(v)));
        let ham_order: Vec<u32> = fused_ham.map_order(v).into_iter().map(relabel_ham).collect();
        assert!(cyclic_eq(&core0.map_order(v), &ham_order));
    }

    let core1 = &en.cores[1].map;
    let step4 = &gate[4].map;
    for v in [A, B, C, D, E, F, G, H] {
        assert!(cyclic_eq(&core1.map_order(v), &step4.map_order(v)));
    }
}

#[test]
fn every_wang_core_drawing_matches_its_planar_map() {
    let en = wang_core_connections();
    for core in &en.cores {
        for v in 0..core.map.verts.len() {
            if core.map.verts[v].kind == VertexKind::Removed {
                continue;
            }
            assert!(
                cyclic_eq(&core.map.drawn_order(v), &core.map.map_order(v)),
                "core {} ({}), vertex {}",
                core.id,
                core.title,
                core.map.verts[v].name
            );
        }
    }
}

#[test]
fn unconstrained_8gon_cavity_matches_kirkman_cayley_120_and_safety_cap() {
    let surrounding = wang_surrounding();
    let ccw_cavity = [A, H, G, F, E, D, C, B];
    let en = enumerate_cavity_chord_cores(&surrounding, &ccw_cavity, 2, false, 11).unwrap();
    // Kirkman-Cayley T(8, 2) = (1/3) * C(5, 2) * C(9, 2) = 120.
    assert_eq!(en.cores.len(), 120);
    assert_eq!(en.regular_count, 1);
    assert!(!en.truncated);
    assert_eq!(en.slide_connected_to_regular, None, "BFS skipped when cores.len() > 32");
    assert!(en.cores[0].is_regular_square);
    assert_eq!(en.cores[0].chords, vec![(C, H), (D, G)]);
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert_eq!(core.inner_verts, 0);
        assert_eq!((core.census.vertices, core.census.edges, core.census.faces), (4, 14, 3));
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }

    let capped = enumerate_cavity_cores(&surrounding, &ccw_cavity, 2, 0, false, 11, 10).unwrap();
    assert_eq!(capped.cores.len(), 10);
    assert!(capped.truncated);
    assert!(capped.provenance.contains("Truncated"));
}

#[test]
fn hexagon_three_edges_v_inner_le_1_matches_exact_hand_count_34_and_d6_orbits() {
    let hex = symmetric_hexagon();
    let en = enumerate_cavity_cores(&hex, &[0, 1, 2, 3, 4, 5], 3, 1, false, 20, 256).unwrap();
    // Exact hand count:
    //   V_inner = 0: Catalan C_4 = 14 triangulations (3 D_6 orbits: sizes 2, 6, 6).
    //   V_inner = 1: C(6, 3) = 20 3-spoke stars (3 D_6 orbits: {4,4,4} size 2, {3,3,6} size 6, {3,4,5} size 12).
    //   Total = 14 + 20 = 34 labelled cores in 6 D_6 orbits (|D_6| = 12).
    assert_eq!(en.cores.len(), 34);
    assert!(!en.truncated);
    assert_eq!(en.cavity_symmetry_order, 12);
    assert_eq!(en.symmetry_orbit_count, 6);
    assert!(en.provenance.contains("within V_inner <= 1"));
    assert!(en.provenance.contains("[COMPUTATIONAL EVIDENCE]"));

    let v0: Vec<_> = en.cores.iter().filter(|c| c.inner_verts == 0).collect();
    let v1: Vec<_> = en.cores.iter().filter(|c| c.inner_verts == 1).collect();
    assert_eq!(v0.len(), 14, "Catalan C_4 = 14");
    assert_eq!(v1.len(), 20, "C(6, 3) = 20");

    // Verify the 2 symmetric 3-square cores with a central 3-valent vertex (spokes to {0,2,4} and {1,3,5}).
    let three_squares: Vec<_> = v1
        .iter()
        .filter(|c| sorted(c.census.face_sides.clone()) == vec![4, 4, 4])
        .collect();
    assert_eq!(three_squares.len(), 2);
    for core in &three_squares {
        assert_eq!(core.orbit_size, 2);
        assert!(core.census.valences.iter().any(|(name, deg)| name == "w" && *deg == 3));
    }

    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
}

#[test]
fn hexagon_four_edges_v_inner_1_with_sector_chords_matches_exact_hand_count_105() {
    let hex = symmetric_hexagon();
    // Exact hand count for 6-gon with inner_edges = 4, max_inner_verts = 1:
    //   V_inner = 0: 0 (a 6-gon triangulation has at most 3 chords).
    //   V_inner = 1, d=4 spokes + 0 chords: C(6, 4) = 15.
    //   V_inner = 1, d=3 spokes + 1 sector chord:
    //     (2,2,2): 2 * (1+1+1) = 6; (1,2,3): 12 * (0+1+3) = 48; (1,1,4): 6 * (0+0+6) = 36 -> 90.
    //   Total = 15 + 90 = 105.
    let en = enumerate_cavity_cores(&hex, &[0, 1, 2, 3, 4, 5], 4, 1, false, 20, 256).unwrap();
    assert_eq!(en.cores.len(), 105);
    assert!(!en.truncated);
    for core in &en.cores {
        assert_eq!(core.inner_verts, 1);
        assert_eq!(core.map.euler(), 2);
        assert_eq!(core.chords.len(), 4);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }

    // Refusals: unsupported max_inner_verts > 1, label collision, non-face cycle.
    assert!(enumerate_cavity_cores(&hex, &[0, 1, 2, 3, 4, 5], 3, 2, false, 20, 256).is_err());
    assert!(enumerate_cavity_chord_cores(&hex, &[0, 1, 2, 3, 4, 5], 1, false, 1).is_err());
    assert!(enumerate_cavity_chord_cores(&hex, &[0, 2, 4], 1, false, 20).is_err());
}
