//! Unit tests for the fixed-cavity lattice-core enumerator (`fusion::enumerate`).
//!
//! All expected counts are independent hand calculations and published combinatorial formulas
//! (Kirkman-Cayley diagonal-dissection counts for convex n-gons).

use super::enumerate::{enumerate_cavity_chord_cores, wang_core_connections, wang_surrounding};
use super::map::{PlanarMap, VertexKind};
use super::wang::{A, B, C, D, E, F, G, H, gate_route, hamiltonian_route};

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

fn cyclic_eq(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && (a.is_empty() || (0..b.len()).any(|s| b.iter().cycle().skip(s).take(b.len()).eq(a.iter())))
}

#[test]
fn wang_core_connections_produces_exact_five_lattice_cores() {
    let en = wang_core_connections();
    assert_eq!(en.cavity_sides, 8);
    assert_eq!(en.inner_edges, 2);
    assert_eq!(en.total_edges, 14);
    assert!(en.boundary_vertices_fixed);
    assert_eq!(en.cores.len(), 5);
    assert_eq!(en.regular_count, 1);
    assert!(en.provenance.contains("handshaking lemma"));

    let expected_chords: [Vec<(usize, usize)>; 5] = [
        vec![(C, H), (D, G)],
        vec![(C, G), (D, G)],
        vec![(C, G), (C, H)],
        vec![(C, H), (D, H)],
        vec![(D, G), (D, H)],
    ];
    for (i, (core, exp)) in en.cores.iter().zip(&expected_chords).enumerate() {
        assert_eq!(core.id, i);
        assert_eq!(&core.chords, exp, "core {i}");
        assert_eq!(core.map.euler(), 2, "core {i} must be connected planar");
        let c = &core.census;
        assert_eq!((c.vertices, c.edges, c.faces), (4, 14, 3), "core {i}");
        assert_eq!(c.square_curvature, 0, "core {i}");
        assert!(c.valences.iter().all(|&(_, k)| k >= 3), "core {i} A2 valences");
        assert!(c.face_sides.iter().all(|&s| s >= 3), "core {i} A2 faces");
        let vals = sorted(c.valences.iter().map(|&(_, k)| k).collect());
        let faces = sorted(c.face_sides.clone());
        if i == 0 {
            assert!(core.is_regular_square);
            assert_eq!(vals, vec![4, 4, 4, 4]);
            assert_eq!(faces, vec![4, 4, 4]);
        } else {
            assert!(!core.is_regular_square);
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

    // Core 0: identical cyclic edge-label order around every vertex to gate_route()[5] and hamiltonian_route()[2].
    let core0 = &en.cores[0].map;
    let fused_gate = &gate[5].map;
    let fused_ham = &ham[2].map;
    let relabel_ham = |l: u32| match l {
        15 => 12,
        16 => 11,
        other => other,
    };
    for v in [A, B, C, D, E, F, G, H] {
        assert!(
            cyclic_eq(&core0.map_order(v), &fused_gate.map_order(v)),
            "core 0 vs gate[5] at vertex {}",
            core0.verts[v].name
        );
        let ham_order: Vec<u32> = fused_ham.map_order(v).into_iter().map(relabel_ham).collect();
        assert!(
            cyclic_eq(&core0.map_order(v), &ham_order),
            "core 0 vs ham[2] at vertex {}",
            core0.verts[v].name
        );
    }

    // Core 1 ((c, g), (d, g)): matches Step 4 of gate_route() around every vertex.
    let core1 = &en.cores[1].map;
    let step4 = &gate[4].map;
    for v in [A, B, C, D, E, F, G, H] {
        assert!(
            cyclic_eq(&core1.map_order(v), &step4.map_order(v)),
            "core 1 vs gate[4] at vertex {}",
            core1.verts[v].name
        );
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
            let drawn = core.map.drawn_order(v);
            let mapped = core.map.map_order(v);
            assert!(
                cyclic_eq(&drawn, &mapped),
                "core {} ({}), vertex {}: drawn {drawn:?} != map {mapped:?}",
                core.id,
                core.title,
                core.map.verts[v].name
            );
        }
    }
}

#[test]
fn unconstrained_8gon_cavity_matches_kirkman_cayley_120_with_unique_regular_core() {
    let surrounding = wang_surrounding();
    // Pass cavity in reversed (CCW) order as well to verify orientation invariance.
    let ccw_cavity = [A, H, G, F, E, D, C, B];
    let en = enumerate_cavity_chord_cores(&surrounding, &ccw_cavity, 2, false, 11).unwrap();
    // Kirkman-Cayley number T(8, 2) = (1/3) * C(5, 2) * C(9, 2) = (1/3) * 10 * 36 = 120.
    // Equivalently: C(20, 2) - C(8, 4) = 190 - 70 = 120 non-crossing pairs of diagonals of an 8-gon.
    assert_eq!(en.cores.len(), 120);
    assert_eq!(en.regular_count, 1, "unique regular square core even over all 120 chord pairs");
    assert!(en.cores[0].is_regular_square);
    assert_eq!(en.cores[0].chords, vec![(C, H), (D, G)]);
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert_eq!((core.census.vertices, core.census.edges, core.census.faces), (4, 14, 3));
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
}

#[test]
fn enumerator_matches_catalan_triangulations_on_hexagon_and_refuses_bad_input() {
    // Positive control: a 6-gon cavity with 3 interior chords gives Catalan C_4 = 14 triangulations.
    let b = VertexKind::Boundary;
    let verts = [
        ("v0", [1.0, 0.0], b),
        ("v1", [0.5, 0.866], b),
        ("v2", [-0.5, 0.866], b),
        ("v3", [-1.0, 0.0], b),
        ("v4", [-0.5, -0.866], b),
        ("v5", [0.5, -0.866], b),
        ("leg", [2.0, 0.0], b),
    ];
    let edges = [(1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 3, 4), (5, 4, 5), (6, 5, 0), (7, 0, 6)];
    let hex = PlanarMap::from_drawing(&verts, &edges, 7).unwrap();
    let en = enumerate_cavity_chord_cores(&hex, &[0, 1, 2, 3, 4, 5], 3, false, 10).unwrap();
    assert_eq!(en.cores.len(), 14, "Catalan C_4 = 14 triangulations of a hexagon");
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert_eq!(core.census.face_sides, vec![3, 3, 3, 3]);
    }
    // Refusals: label collision and non-face cycle.
    assert!(enumerate_cavity_chord_cores(&hex, &[0, 1, 2, 3, 4, 5], 1, false, 1).is_err());
    assert!(enumerate_cavity_chord_cores(&hex, &[0, 2, 4], 1, false, 10).is_err());
}
