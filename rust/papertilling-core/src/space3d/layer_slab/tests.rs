//! Layer-slab tests (C1 WP-L1). Sources of the expected values are stated per test; none is read
//! from the code under test.

use std::collections::{HashSet, VecDeque};

use super::shells::{lattice_rank_index, quotient_state, Bond};
use super::*;
use crate::space3d::nets3d::build_layer_slab_patch;

/// Independent connectivity check (no HNF): BFS over the derived net on cells |n| <= 6 from
/// (site 0, cell 0) must reach every site of cell 0 and site 0 of cells (1,0) and (0,1).
fn connected_and_two_periodic(m: usize, bonds: &[Bond]) -> bool {
    let lim = 6;
    let mut adj: Vec<Vec<(usize, [i64; 2])>> = vec![Vec::new(); m];
    for &(i, j, n) in bonds {
        adj[i].push((j, n));
        adj[j].push((i, [-n[0], -n[1]]));
    }
    let mut seen: HashSet<(usize, [i64; 2])> = HashSet::from([(0, [0, 0])]);
    let mut q = VecDeque::from([(0usize, [0i64, 0])]);
    while let Some((u, c)) = q.pop_front() {
        for &(v, n) in &adj[u] {
            let w = (v, [c[0] + n[0], c[1] + n[1]]);
            if w.1[0].abs() <= lim && w.1[1].abs() <= lim && seen.insert(w) {
                q.push_back(w);
            }
        }
    }
    (0..m).all(|s| seen.contains(&(s, [0, 0]))) && seen.contains(&(0, [1, 0])) && seen.contains(&(0, [0, 1]))
}

#[test]
fn all_80_groups_are_exact_connected_two_periodic_and_invariant() {
    for num in 1..=80u8 {
        let lay = build_layer(num).unwrap_or_else(|e| panic!("#{num}: {e}"));
        let e = layer_group_entry(num).unwrap();
        // general positions: |orbit| = point order x centring (from the group table, not the builder)
        let general = e.point_order * if e.centred { 2 } else { 1 };
        let orbits = lay.labels.iter().max().unwrap() + 1;
        assert_eq!(lay.positions.len(), orbits * general, "#{num} site count");
        assert!(lay.exact, "#{num}: {}", lay.symmetry_note);
        assert!(!lay.bonds.is_empty(), "#{num} has no bonds");
        assert!(connected_and_two_periodic(lay.positions.len(), &lay.bonds), "#{num} not connected / 2-periodic");
        assert_eq!(quotient_state(lay.positions.len(), &lay.bonds), (1, 2, 1), "#{num}");
        bond_set_invariant(&lay).unwrap_or_else(|err| panic!("#{num}: {err}"));
    }
}

#[test]
fn p1_gets_bonds_and_three_sites() {
    // p1: one or two sites always have an inversion centre (plan Q1 note), so the exact set has 3 sites;
    // the old 0.88 cutoff gave p1 no bonds at all.
    let lay = build_layer(1).unwrap();
    assert_eq!(lay.positions.len(), 3);
    assert!(lay.bonds.len() >= 3);
}

#[test]
fn rank_index_by_hand() {
    // (2,0),(0,3): rank 2, index 6; (1,1),(1,-1): index 2; (2,4) alone: rank 1
    assert_eq!(lattice_rank_index(&[vec![2, 0], vec![0, 3]], 2), (2, 6));
    assert_eq!(lattice_rank_index(&[vec![1, 1], vec![1, -1]], 2), (2, 2));
    assert_eq!(lattice_rank_index(&[vec![2, 4], vec![-1, -2]], 2), (1, 0));
    assert_eq!(lattice_rank_index(&[vec![3, 1], vec![5, 2], vec![0, 0]], 2), (2, 1));
}

#[test]
fn one_site_square_net_has_coordination_four() {
    // engine check independent of the group data (plan §5 T5): the nearest shell of a unit square lattice
    // is the 4 neighbours at distance 1, one shell connects it at rank 2 and index 1: 2 bonds per cell.
    let l = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let (bonds, lengths) = shells::shell_bonds(&[[0.0, 0.0, 0.0]], &l, &[0]).unwrap();
    assert_eq!(bonds.len(), 2);
    assert_eq!(lengths.len(), 1);
    assert!((lengths[0] - 1.0).abs() < 1e-12);
}

#[test]
fn default_patch_is_built_for_all_80_groups() {
    for num in 1..=80u8 {
        let p = build_layer_slab_patch(num).unwrap_or_else(|e| panic!("#{num}: {e}"));
        assert!(!p.bonds.is_empty(), "#{num}");
        assert_eq!(p.positions.len(), p.site_kinds.len());
        assert!(p.bonds.iter().all(|&(a, b, _)| a < p.positions.len() && b < p.positions.len() && a != b));
    }
    assert!(build_layer_slab_patch(81).is_err());
}

#[test]
fn offset_rule_by_hand() {
    // R2 s = e s mod the in-plane translations, derived per operation by hand:
    // p1 (#1): identity only, any s; p-1 (#2): -s = -s, any s;
    // p11m (#4, mirror z, e = -1): s = -s, so 2s in Z^2; p112 (#3, 2-fold about z): -s = s, the same;
    // p4 (#49): (-s1, s0) = (s0, s1) gives s in {0, (1/2,1/2)}; p3 (#65): s in {0, (1/3,2/3), (2/3,1/3)}.
    // m = 3 (odd): only the condition on lambda_g (module docs) can fail
    let ok = |n, s| offset_keeps_group(n, s, 3).unwrap();
    assert!(ok(1, [0.3, 0.1]) && ok(2, [0.37, 0.81]));
    assert!(ok(4, [0.5, 0.0]) && !ok(4, [0.3, 0.0]));
    assert!(ok(3, [0.5, 0.5]) && !ok(3, [0.25, 0.0]));
    assert!(ok(49, [0.5, 0.5]) && !ok(49, [0.5, 0.0]));
    assert!(ok(65, [1.0 / 3.0, 2.0 / 3.0]) && ok(65, [2.0 / 3.0, 1.0 / 3.0]) && !ok(65, [0.5, 0.0]));
    // even m, by hand: p11m, s = (1/2, 0), 2 copies at in-plane 0 and 1/2; a z-mirror between them must
    // shift by 1/2, a glide, and m_z has no translation part to absorb it (u - R2 u = 0): refused. With 3
    // copies the middle copy carries the mirror and copies 0, 2 sit at in-plane 0 and 1 = 0: kept.
    // p112, s = (1/2, 0), m = 2: copy 1's 2-fold axes s + (1/2)Z^2 include the origin, axes of copy 0: kept.
    // p-1: lambda = 0 for the inversion, kept for any s and m. One layer: nothing to check.
    let keep = |n, s, m| offset_keeps_group(n, s, m).unwrap();
    assert!(!keep(4, [0.5, 0.0], 2) && keep(4, [0.5, 0.0], 3) && !keep(4, [0.5, 0.0], 4));
    assert!(keep(3, [0.5, 0.0], 2) && keep(2, [0.37, 0.81], 2) && keep(49, [0.3, 0.0], 1));
    let lay4 = build_layer(4).unwrap();
    let even = StackSpec { layers: 2, spacing: lay4.thickness + 0.5, offset: [0.5, 0.0] };
    assert!(stack(&lay4, &even).unwrap_err().contains("breaks"));
    assert!(stack(&lay4, &StackSpec { layers: 3, ..even }).is_ok());
    let lay = build_layer(49).unwrap();
    let spec = StackSpec { layers: 2, spacing: lay.thickness + 0.5, offset: [0.5, 0.0] };
    assert!(stack(&lay, &spec).unwrap_err().contains("breaks"));
}

#[test]
fn aa_stacks_repeat_the_layer() {
    for num in [1u8, 48, 61, 80] {
        let lay = build_layer(num).unwrap();
        let spec1 = StackSpec { layers: 1, spacing: lay.thickness + 0.4, offset: [0.0, 0.0] };
        let one = grow_layer_slab(num, SeedKind::Vertex, 2, &spec1, 20_000).unwrap();
        let b0 = one.patch.bonds.len();
        let two = grow_layer_slab(num, SeedKind::Vertex, 2, &StackSpec { layers: 2, ..spec1 }, 20_000).unwrap();
        let s = two.patch.bonds.iter().filter(|b| b.2 == 1).count();
        assert!(s > 0, "#{num} no interlayer bond");
        for m in 1..=4u32 {
            let g = grow_layer_slab(num, SeedKind::Vertex, 2, &StackSpec { layers: m, ..spec1 }, 20_000).unwrap();
            let n = one.patch.positions.len();
            assert_eq!(g.patch.positions.len(), m as usize * n, "#{num} m={m}");
            // in-layer bonds m B, interlayer (m-1) S
            let inl = g.patch.bonds.iter().filter(|b| b.2 == 0).count();
            let inter = g.patch.bonds.iter().filter(|b| b.2 == 1).count();
            assert_eq!((inl, inter), (m as usize * b0, (m as usize - 1) * s), "#{num} m={m}");
            // AA: every copy projects onto copy 0
            for k in 1..m as usize {
                for i in 0..n {
                    let (a, b) = (g.patch.positions[i], g.patch.positions[k * n + i]);
                    assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
                    assert!((b[2] - a[2] - k as f64 * spec1.spacing).abs() < 1e-9);
                }
            }
        }
    }
}

#[test]
fn mirror_group_interlayer_shell_is_the_top_sites() {
    // p4/mmm (#61) has the mirror z, so each site at z_max has the copy of its mirror image (same x, y,
    // z_min) right above it at d - t; every other interlayer pair is farther. So S = #sites at z_max,
    // all of length d - t (hand derivation).
    let lay = build_layer(61).unwrap();
    let d = lay.thickness + 0.1;
    let st = stack(&lay, &StackSpec { layers: 2, spacing: d, offset: [0.0, 0.0] }).unwrap();
    let zmax = lay.positions.iter().map(|p| p[2]).fold(f64::MIN, f64::max);
    let top = lay.positions.iter().filter(|p| (p[2] - zmax).abs() < 1e-9).count();
    assert_eq!(st.interlayer.len(), top);
    assert_eq!(top, lay.positions.len() / 2);
    let refused = stack(&lay, &StackSpec { layers: 2, spacing: lay.thickness, offset: [0.0, 0.0] });
    assert!(refused.unwrap_err().contains("interpenetrate"));
}

#[test]
fn growth_readout_flags_chi_for_slabs() {
    let g = grow_layer_slab(61, SeedKind::Cell, 3, &StackSpec::default(), 20_000).unwrap();
    assert_eq!(g.mode, NetMode::Slab);
    assert!(!g.euler_defined);
    assert_eq!(g.stats.len(), 4);
    assert!(g.stats.windows(2).all(|w| w[1].cells > w[0].cells));
    assert!(g.exact && g.symmetry_note.starts_with("exact layer group"));
}

#[test]
fn layer_slab_parity_dump() {
    // Writes the Rust results for the Python parity test when PAPERTILLING_PARITY_DIR is set.
    let Ok(dir) = std::env::var("PAPERTILLING_PARITY_DIR") else { return };
    let mut groups = Vec::new();
    for num in 1..=80u8 {
        let lay = build_layer(num).unwrap();
        let mut row = serde_json::json!({"number": num, "sites": lay.positions.len(), "bonds": lay.bonds, "shells": lay.shells});
        if [1u8, 2, 4, 12, 26, 48, 49, 57, 65, 69].contains(&num) {
            let net = PeriodicNet::new(&lay.lattice, &lay.positions).unwrap();
            let g = grow_layer_slab(num, SeedKind::Vertex, 3, &StackSpec::default(), 20_000).unwrap();
            row["polytopes"] = serde_json::json!(net.polytopes());
            row["growth"] = serde_json::json!(g.stats.iter().map(|s| s.cells).collect::<Vec<_>>());
            let st = stack(&lay, &StackSpec { layers: 2, spacing: lay.thickness + 0.3, offset: [0.0, 0.0] }).unwrap();
            row["interlayer"] = serde_json::json!(st.interlayer);
        }
        groups.push(row);
    }
    let path = std::path::Path::new(&dir).join("layer_slab_parity.json");
    std::fs::write(&path, serde_json::to_string(&groups).unwrap()).unwrap();
}
