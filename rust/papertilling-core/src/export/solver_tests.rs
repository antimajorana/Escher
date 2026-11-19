//! Hand-count tests for the solver export: shell coordinations and orbit splits, extended XYZ layout, torus
//! permutation groups, and an open D5 cone cluster.

use std::time::Instant;

use super::*;
use crate::defect::DefectCatalogue;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::plane::catalogue::{NamedLattice, parse_named_str};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

fn named(key: &str) -> NamedLattice {
    parse_named_str(NAMED_JSON).expect("named.json").get(key).expect("named lattice").clone()
}

fn orbit_coords(sc: &ShellClassification, s: usize) -> Vec<Vec<usize>> {
    sc.shells[s].orbits.iter().map(|o| o.coordination_by_basis.clone()).collect()
}

#[test]
fn shell_hand_counts_square_honeycomb_kagome() {
    let t0 = Instant::now();
    // square: J1 = 4 (distance 1), J2 = 4 (sqrt 2), J3 = 4 (2); one orbit each
    let sq = named("square");
    let s = classify_cell_shells(&sq.group, &sq.cell, 3).unwrap();
    let d1 = s.shells[0].distance;
    for (k, (coord, dist)) in [(4, 1.0), (4, 2f64.sqrt()), (4, 2.0)].into_iter().enumerate() {
        assert_eq!(s.shells[k].coordination_by_basis, vec![coord]);
        assert!((s.shells[k].distance / d1 - dist).abs() < 1e-9);
        assert_eq!(s.shells[k].orbits.len(), 1);
    }
    // honeycomb: J1 = 3, J2 = 6 (sqrt 3), J3 = 3 (2); one orbit each
    let hc = named("honeycomb");
    let h = classify_cell_shells(&hc.group, &hc.cell, 3).unwrap();
    let d1 = h.shells[0].distance;
    for (k, (coord, dist)) in [(3, 1.0), (6, 3f64.sqrt()), (3, 2.0)].into_iter().enumerate() {
        assert_eq!(h.shells[k].coordination_by_basis, vec![coord; 2]);
        assert!((h.shells[k].distance / d1 - dist).abs() < 1e-9);
        assert_eq!(h.shells[k].orbits.len(), 1, "honeycomb J{}", k + 1);
    }
    assert_eq!(h.n_symmetry_ops, 12);
    // kagome: J1 = 4, J2 = 4 (sqrt 3), J3 = 6 (2) split into chains (4) and hexagon diagonals (2)
    let kg = named("kagome");
    let g = classify_cell_shells(&kg.group, &kg.cell, 3).unwrap();
    let d1 = g.shells[0].distance;
    for (k, (coord, dist)) in [(4, 1.0), (4, 3f64.sqrt()), (6, 2.0)].into_iter().enumerate() {
        assert_eq!(g.shells[k].coordination_by_basis, vec![coord; 3]);
        assert!((g.shells[k].distance / d1 - dist).abs() < 1e-9);
    }
    assert_eq!(g.shells[0].orbits.len(), 1);
    assert_eq!(g.shells[1].orbits.len(), 1);
    let mut j3 = orbit_coords(&g, 2);
    j3.sort();
    assert_eq!(j3, vec![vec![2; 3], vec![4; 3]]);
    // the chain orbit continues bonds in a straight line: its bond vector is twice a J1 bond vector
    let chain = g.shells[2].orbits.iter().find(|o| o.coordination_by_basis[0] == 4).unwrap();
    let b = chain.representative;
    assert_eq!(b.i, b.j, "a straight 2-step chain joins equivalent sites of the same sublattice");
    println!("# time shells (3 lattices): {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn extxyz_layout_cell_and_crystal() {
    let hc = named("honeycomb");
    let txt = export_cell_extxyz(&hc.cell, 2, 3, None).unwrap();
    assert!(!txt.contains('\r'));
    let lines: Vec<&str> = txt.lines().collect();
    assert_eq!(lines.len(), 2 * 3 * 2 + 2);
    assert_eq!(lines[0], "12");
    assert!(lines[1].starts_with("Lattice=\""));
    assert!(lines[1].ends_with(" Properties=species:S:1:pos:R:3:sublattice:I:1:coordination:I:1 pbc=\"T T F\""));
    let lat: Vec<f64> = lines[1].split('"').nth(1).unwrap().split(' ').map(|x| x.parse().unwrap()).collect();
    assert_eq!(lat.len(), 9);
    assert!((lat[0] - 2.0 * hc.cell.a1[0]).abs() < 1e-8 && (lat[4] - 3.0 * hc.cell.a2[1]).abs() < 1e-8 && lat[8] == 20.0);
    let row: Vec<&str> = lines[3].split(' ').collect();
    assert_eq!(row, vec!["B", &fmt_f8(hc.cell.basis[1][0])[..], &fmt_f8(hc.cell.basis[1][1])[..], "0.00000000", "1", "3"]);
    assert_eq!(fmt_f8(-0.0), "0.00000000");
    assert_eq!(fmt_f8(-1e-14), "0.00000000");
    let sq = named("square");
    let crystal = crate::defect::grain_boundary::perfect_disk(&sq.cell, [0.0, 0.0], 2.0);
    let n = crystal.pos.len();
    assert_eq!(n, 13); // |n1|^2 + |n2|^2 <= 4: 1 + 4 + 4 + 4 = 13 sites
    let z: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
    let txt = export_crystal_extxyz(&crystal, Some(&z), None).unwrap();
    let lines: Vec<&str> = txt.lines().collect();
    assert_eq!(lines.len(), n + 2);
    assert!(lines[1].starts_with("Lattice=\"14.00000000 0.00000000 0.00000000 0.00000000 14.00000000 0.00000000 0.00000000 0.00000000 16.00000000\""));
    assert!(lines[1].ends_with("pbc=\"F F F\""));
    let centre = (0..n).find(|&i| crystal.pos[i] == [0.0, 0.0]).unwrap();
    assert_eq!(lines[2 + centre], format!("C 0.00000000 0.00000000 {} 0 4", fmt_f8(z[centre])));
    let bad = export_crystal_extxyz(&crystal, None, Some(&vec!["c".to_string(); n]));
    assert!(bad.is_err());
}

#[test]
fn square_4x4_torus_group_order_128() {
    let t0 = Instant::now();
    let sq = named("square");
    let ex = periodic_supercell_permutations(&sq.group, &sq.cell, 4, 4, 3).unwrap();
    let r = &ex.report;
    assert_eq!(r.n_sites, 16);
    assert_eq!(r.order, 16 * 8);
    assert!(r.group_closed);
    assert_eq!(r.permutations[0], (0..16).collect::<Vec<_>>());
    // hand counts per site on the 4x4 torus: J1 4 distinct neighbours, J2 4, J3: (+-2, 0) coincide -> 2 sites x 2
    let n = 16;
    for (s, (distinct, total)) in [(4, 4), (4, 4), (2, 4)].into_iter().enumerate() {
        let mut a = vec![vec![0u32; n]; n];
        for &(u, v, m) in &ex.shell_edges[s] {
            a[u][v] = m;
            a[v][u] = m;
        }
        for row in &a {
            assert_eq!(row.iter().filter(|&&m| m > 0).count(), distinct);
            assert_eq!(row.iter().sum::<u32>() as usize, total);
        }
        for p in &r.permutations {
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(a[p[i]][p[j]], a[i][j]);
                }
            }
        }
    }
    println!("# time 4x4 square torus permutations: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn honeycomb_3x3_torus_group() {
    let hc = named("honeycomb");
    let ex = periodic_supercell_permutations(&hc.group, &hc.cell, 3, 3, 2).unwrap();
    // p6m on a 3x3 torus: 12 point ops x 9 translations, all compatible with the (3a1, 3a2) sublattice
    assert_eq!(ex.report.order, 12 * 9);
    assert!(ex.report.group_closed);
    assert_eq!(ex.site_labels, vec![0; 18]);
}

#[test]
fn open_honeycomb_pentagon_cone_has_d5() {
    let t0 = Instant::now();
    let hc = named("honeycomb");
    let cat = DefectCatalogue::new(&hc.group, &hc.cell, hc.frame);
    let p = cat.centre_classes.iter().find(|c| c.order == 6).expect("6-fold centre").point;
    // +60 deg removes a 60 deg wedge -> 5-fold pentagon cone (D5); -60 deg inserts a wedge -> 7-fold heptagon cone (D7)
    let crystal = build_point_defects(&hc.cell, &[DefectSpec::disclination(60.0, p)], Some(4.0), None).unwrap();
    let pos3: Vec<[f64; 3]> = crystal.pos.iter().map(|q| [q[0], q[1], 0.0]).collect();
    let labels = vec![0usize; pos3.len()];
    let rep = cluster_site_permutations(&pos3, &labels, std::slice::from_ref(&crystal.edges)).unwrap();
    assert_eq!(rep.order, 10, "D5 of the pentagon-centred cone");
    assert!(rep.group_closed);
    assert_eq!(rep.permutations[0], (0..pos3.len()).collect::<Vec<_>>());
    let crystal7 = build_point_defects(&hc.cell, &[DefectSpec::disclination(-60.0, p)], Some(4.0), None).unwrap();
    let pos7: Vec<[f64; 3]> = crystal7.pos.iter().map(|q| [q[0], q[1], 0.0]).collect();
    let rep7 = cluster_site_permutations(&pos7, &vec![0usize; pos7.len()], std::slice::from_ref(&crystal7.edges)).unwrap();
    assert_eq!(rep7.order, 14, "D7 of the heptagon-centred cone");
    assert!(rep7.group_closed);
    // a 3D check: lifting the cluster off the plane with z = |x|^2 keeps the rotations and vertical mirrors
    let c: [f64; 2] = {
        let n = crystal.pos.len() as f64;
        let s = crystal.pos.iter().fold([0.0, 0.0], |a, q| [a[0] + q[0], a[1] + q[1]]);
        [s[0] / n, s[1] / n]
    };
    let lifted: Vec<[f64; 3]> = crystal.pos.iter().map(|q| [q[0], q[1], 0.05 * ((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2))]).collect();
    let rep3 = cluster_site_permutations(&lifted, &labels, std::slice::from_ref(&crystal.edges)).unwrap();
    assert_eq!(rep3.order, 10);
    println!("# time D5 cone cluster: {:.4} s", t0.elapsed().as_secs_f64());
}
