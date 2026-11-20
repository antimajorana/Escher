//! Known answers come from the paper and from hand counts written here, never from the code under test:
//! Wang, Liu, Lu, Phys. Rev. B 109, 195165 (2024), Eqs. (56)-(58) and Fig. 8, and the table of operator
//! images in the author's note "Unitary fusion of four dislocations".

use std::collections::BTreeSet;

use super::map::{PlanarMap, VertexKind};
use super::moves::Side;
use super::wang::{C, D, G, H, S, fig_8a, gate_route, hamiltonian_route};

type Ops = BTreeSet<Vec<u32>>;
/// (complete vertices, edges, bounded faces, sorted valences, sorted face sizes).
type StepCounts = (usize, usize, usize, Vec<usize>, Vec<usize>);

fn ops(list: &[&[u32]]) -> Ops {
    list.iter().map(|l| l.to_vec()).collect()
}

fn graph_ops(m: &PlanarMap) -> (Ops, Ops) {
    let c = m.census();
    (c.vertex_ops.into_iter().collect(), c.face_ops.into_iter().collect())
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

/// A Pauli product on edges 1..=16: bit `l` of `.0` is X on edge l, of `.1` is Z on edge l.
type Pauli = (u64, u64);

fn pauli_x(labels: &[u32]) -> Pauli {
    (labels.iter().map(|&l| 1u64 << l).sum(), 0)
}

fn pauli_z(labels: &[u32]) -> Pauli {
    (0, labels.iter().map(|&l| 1u64 << l).sum())
}

/// CX(c -> t): X_c -> X_c X_t, Z_t -> Z_c Z_t (the convention of the note, Sec. 2.1).
fn cx(p: Pauli, c: u32, t: u32) -> Pauli {
    let (mut x, mut z) = p;
    if x >> c & 1 == 1 {
        x ^= 1 << t;
    }
    if z >> t & 1 == 1 {
        z ^= 1 << c;
    }
    (x, z)
}

fn hadamard(p: Pauli, q: u32) -> Pauli {
    let (x, z) = p;
    let (xb, zb) = (x >> q & 1, z >> q & 1);
    ((x & !(1 << q)) | (zb << q), (z & !(1 << q)) | (xb << q))
}

/// Rank over GF(2) of Paulis written as one 128-bit row (signs play no role here).
fn rank(rows: &[Pauli]) -> usize {
    let mut basis: Vec<u128> = Vec::new();
    for &(x, z) in rows {
        let mut r = (u128::from(x) << 64) | u128::from(z);
        for &b in &basis {
            r = r.min(r ^ b);
        }
        if r != 0 {
            basis.push(r);
        }
    }
    basis.len()
}

fn same_span(a: &[Pauli], b: &[Pauli]) -> bool {
    let both: Vec<Pauli> = a.iter().chain(b).copied().collect();
    rank(a) == rank(b) && rank(a) == rank(&both)
}

fn graph_paulis(m: &PlanarMap) -> Vec<Pauli> {
    let c = m.census();
    c.vertex_ops.iter().map(|l| pauli_x(l)).chain(c.face_ops.iter().map(|l| pauli_z(l))).collect()
}

#[test]
fn fig_8a_reproduces_the_seven_terms_of_eq_56() {
    let (v, f) = graph_ops(&fig_8a());
    // Eq. (56): X3 X4 X5 X11, X11 X12, X12 X7 X8 X9, X2 X13 X3, X9 X14 X10.
    assert_eq!(v, ops(&[&[3, 4, 5, 11], &[11, 12], &[7, 8, 9, 12], &[2, 3, 13], &[9, 10, 14]]));
    // Eq. (56): Z1 Z2 Z3 Z11 Z12 Z9 Z10 and Z5 Z6 Z7 Z12 Z11.
    assert_eq!(f, ops(&[&[1, 2, 3, 9, 10, 11, 12], &[5, 6, 7, 11, 12]]));
}

#[test]
fn fig_8a_valences_and_face_sizes() {
    let m = fig_8a();
    let c = m.census();
    // Read off Fig. 8(a): s has two edges, c and h three, d and g four; one face of 7 sides, one of 5.
    let val = |name: &str| c.valences.iter().find(|(n, _)| n == name).map(|&(_, k)| k);
    assert_eq!((val("s"), val("c"), val("h"), val("d"), val("g")), (Some(2), Some(3), Some(3), Some(4), Some(4)));
    assert_eq!(sorted(c.face_sides.clone()), vec![5, 7]);
    assert_eq!((c.vertices, c.edges, c.faces), (5, 14, 2));
    // (4-2) + 2 (4-3) + (4-7) + (4-5) = 0: a patch holding only dislocations has no net curvature.
    assert_eq!(c.square_curvature, 0);
}

#[test]
fn gate_route_counts_match_hand_counts_at_every_step() {
    let steps = gate_route().unwrap();
    assert_eq!(steps.len(), 6);
    // (complete vertices, edges, bounded faces, sorted valences, sorted face sizes), counted by hand from
    // the description of each move; an edge met on both sides of a face counts twice.
    let expected: [StepCounts; 6] = [
        (5, 14, 2, vec![2, 3, 3, 4, 4], vec![5, 7]),
        (5, 14, 2, vec![1, 3, 3, 4, 5], vec![4, 8]),
        (4, 14, 3, vec![3, 3, 4, 6], vec![1, 4, 7]),
        (4, 14, 3, vec![3, 3, 5, 5], vec![2, 4, 6]),
        (4, 14, 3, vec![3, 4, 4, 5], vec![3, 4, 5]),
        (4, 14, 3, vec![4, 4, 4, 4], vec![4, 4, 4]),
    ];
    for (k, (st, exp)) in steps.iter().zip(&expected).enumerate() {
        let c = st.map.census();
        let vals = sorted(c.valences.iter().map(|&(_, n)| n).collect());
        assert_eq!((c.vertices, c.edges, c.faces, vals, sorted(c.face_sides.clone())), exp.clone(), "step {k}");
        assert_eq!(c.square_curvature, 0, "step {k}: slides and the leaf-to-loop step keep the net curvature");
    }
    // The note, Sec. 5: Delta V = -1, Delta E = 0, Delta F = +1.
    let (first, last) = (steps[0].map.census(), steps[5].map.census());
    assert_eq!(last.vertices as i64 - first.vertices as i64, -1);
    assert_eq!(last.edges, first.edges);
    assert_eq!(last.faces as i64 - first.faces as i64, 1);
}

#[test]
fn gate_route_operators_match_the_note() {
    let steps = gate_route().unwrap();
    // After CX(11->12): leaf X11, its neighbour X3 X4 X5 X11 X12, surrounding face Z1 Z2 Z3 Z9 Z10 Z12.
    let (v, f) = graph_ops(&steps[1].map);
    assert!(v.contains(&vec![11]) && v.contains(&vec![3, 4, 5, 11, 12]));
    assert!(f.contains(&vec![1, 2, 3, 9, 10, 12]));
    // After H(11): vertex X3 X4 X5 X12, face inside the loop Z11, face round the loop Z1 Z2 Z3 Z9 Z10 Z12 Z11.
    let (v, f) = graph_ops(&steps[2].map);
    assert!(v.contains(&vec![3, 4, 5, 12]));
    assert!(f.contains(&vec![11]) && f.contains(&vec![1, 2, 3, 9, 10, 11, 12]));
    // Fused lattice, the note's Eq. for S~, B, A and its transformed remainder (= Eq. (58) of the paper
    // with l15 -> 12, l16 -> 11).
    let (v, f) = graph_ops(&steps[5].map);
    assert_eq!(v, ops(&[&[3, 4, 5, 12], &[7, 8, 9, 12], &[2, 3, 11, 13], &[9, 10, 11, 14]]));
    assert_eq!(f, ops(&[&[3, 9, 11, 12], &[1, 2, 10, 11], &[5, 6, 7, 12]]));
}

#[test]
fn every_gate_step_spans_the_conjugated_stabilizers() {
    // Independent path: conjugate the seven terms of Eq. (56) by the gates as Pauli operators and compare
    // the group they generate with the vertex and face terms read off the map after the same step.
    let mut stab: Vec<Pauli> = vec![
        pauli_x(&[3, 4, 5, 11]), pauli_x(&[11, 12]), pauli_x(&[7, 8, 9, 12]), pauli_x(&[2, 3, 13]),
        pauli_x(&[9, 10, 14]), pauli_z(&[1, 2, 3, 9, 10, 11, 12]), pauli_z(&[5, 6, 7, 11, 12]),
    ];
    let gates: [&dyn Fn(Pauli) -> Pauli; 5] = [
        &|p| cx(p, 11, 12), &|p| hadamard(p, 11), &|p| cx(p, 12, 11), &|p| cx(p, 3, 11), &|p| cx(p, 9, 11),
    ];
    let steps = gate_route().unwrap();
    assert!(same_span(&stab, &graph_paulis(&steps[0].map)));
    for (k, gate) in gates.iter().enumerate() {
        stab = stab.iter().map(|&p| gate(p)).collect();
        assert_eq!(rank(&stab), 7);
        assert!(same_span(&stab, &graph_paulis(&steps[k + 1].map)), "after gate {}", k + 1);
    }
    // The note's table, last row: S -> Z3 Z9 Z11 Z12.
    assert_eq!(stab[1], pauli_z(&[3, 9, 11, 12]));
    // Negative control: a wrong last gate (the earlier draft's CX(3->12)) does not give the fused lattice.
    let mut wrong: Vec<Pauli> = graph_paulis(&steps[4].map);
    wrong = wrong.iter().map(|&p| cx(p, 3, 12)).collect();
    assert!(!same_span(&wrong, &graph_paulis(&steps[5].map)));
}

#[test]
fn hamiltonian_route_matches_eqs_57_and_58_and_the_gate_route() {
    let steps = hamiltonian_route().unwrap();
    assert_eq!(steps.len(), 3);
    // Eq. (57).
    let (v, f) = graph_ops(&steps[1].map);
    assert_eq!(v, ops(&[&[3, 4, 5, 15], &[7, 8, 9, 15], &[2, 3, 13], &[9, 10, 14]]));
    assert_eq!(f, ops(&[&[1, 2, 3, 9, 10, 15], &[5, 6, 7, 15]]));
    // Eq. (58).
    let (v, f) = graph_ops(&steps[2].map);
    assert_eq!(v, ops(&[&[3, 4, 5, 15], &[7, 8, 9, 15], &[2, 3, 13, 16], &[9, 10, 14, 16]]));
    assert_eq!(f, ops(&[&[1, 2, 10, 16], &[3, 9, 15, 16], &[5, 6, 7, 15]]));
    // Same lattice as the gate route under l15 -> 12, l16 -> 11: same cyclic order round every vertex.
    let fused = &gate_route().unwrap()[5].map;
    let relabel = |l: u32| match l {
        15 => 12,
        16 => 11,
        other => other,
    };
    for v in [C, D, G, H] {
        let a: Vec<u32> = steps[2].map.map_order(v).into_iter().map(relabel).collect();
        let b = fused.map_order(v);
        let turned = (0..b.len()).any(|k| b.iter().cycle().skip(k).take(b.len()).copied().eq(a.iter().copied()));
        assert!(turned, "vertex {v}: {a:?} vs {b:?}");
    }
    // Wang et al. add a qubit (14 -> 13 -> 14 edges); the gate route never changes the edge count.
    assert_eq!(steps.iter().map(|s| s.map.census().edges).collect::<Vec<_>>(), vec![14, 13, 14]);
}

#[test]
fn the_drawing_of_every_step_shows_the_same_map() {
    for route in [gate_route().unwrap(), hamiltonian_route().unwrap()] {
        for (k, st) in route.iter().enumerate() {
            for v in 0..st.map.verts.len() {
                if st.map.verts[v].kind == VertexKind::Removed {
                    continue;
                }
                let (a, b) = (st.map.drawn_order(v), st.map.map_order(v));
                let turned = a.len() == b.len()
                    && (a.is_empty() || (0..b.len()).any(|s| b.iter().cycle().skip(s).take(b.len()).eq(a.iter())));
                assert!(turned, "step {k} ({}), vertex {}: drawn {a:?}, map {b:?}", st.title, st.map.verts[v].name);
            }
        }
    }
}

#[test]
fn moves_refuse_with_a_reason() {
    let mut m = fig_8a();
    // Edge 5 is not next to edge 3 at d (edge 4 and edge 11 lie between): sliding would cross an edge.
    assert!(m.clone().slide(5, 11, D, None).is_ok());
    assert!(m.clone().slide(5, 3, D, None).unwrap_err().contains("cross another edge"));
    // At the two-valent vertex both sides are possible and the side must be given.
    assert!(m.clone().slide(12, 11, S, None).unwrap_err().contains("give the side"));
    // Nothing dangles in Fig. 8(a).
    assert!(m.leaf_to_loop(11).unwrap_err().contains("not dangling"));
    // Corners on different faces cannot be joined: h is not on the five-sided face.
    assert!(m.add_edge(16, H, 9, super::wang::E, 5).unwrap_err().contains("same face"));
    assert_eq!(m, fig_8a(), "a refused move leaves the map unchanged");
    // The wrong side at the first gate is caught at the fourth: edge 11 is then not next to edge 3.
    m.slide(12, 11, S, Some(Side::Left)).unwrap();
    m.leaf_to_loop(11).unwrap();
    m.slide(11, 12, D, None).unwrap();
    assert!(m.slide(11, 3, D, None).is_err());
}

#[test]
fn positive_control_square_patch() {
    // A 3 x 3 block of squares cut from the plain square lattice: the 4 inner vertices have four edges and
    // the 9 faces four sides. Vertices on a 4 x 4 grid, all rim vertices cut by the picture, one leg added.
    let mut verts: Vec<(String, [f64; 2], VertexKind)> = Vec::new();
    for j in 0..4 {
        for i in 0..4 {
            let inner = (1..3).contains(&i) && (1..3).contains(&j);
            let kind = if inner { VertexKind::Complete } else { VertexKind::Boundary };
            verts.push((format!("{i},{j}"), [f64::from(i), f64::from(j)], kind));
        }
    }
    verts.push(("leg".to_string(), [-1.0, 0.0], VertexKind::Boundary));
    let mut edges: Vec<(u32, usize, usize)> = Vec::new();
    let mut label = 0;
    for j in 0..4_usize {
        for i in 0..4_usize {
            for (di, dj) in [(1_usize, 0_usize), (0, 1)] {
                if i + di < 4 && j + dj < 4 {
                    label += 1;
                    edges.push((label, 4 * j + i, 4 * (j + dj) + i + di));
                }
            }
        }
    }
    edges.push((99, 0, 16));
    let borrowed: Vec<(&str, [f64; 2], VertexKind)> = verts.iter().map(|(n, p, k)| (n.as_str(), *p, *k)).collect();
    let c = PlanarMap::from_drawing(&borrowed, &edges, 99).unwrap().census();
    assert_eq!((c.vertices, c.faces), (4, 9));
    assert!(c.valences.iter().all(|&(_, k)| k == 4) && c.face_sides.iter().all(|&s| s == 4));
    assert_eq!(c.square_curvature, 0);
}
