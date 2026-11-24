use super::*;

#[test]
fn regular_coxeter_relations_are_exact_for_all_supported_symbols() {
    let check = |g: ReflectionGroup, edges: &[usize]| {
        for (i, &m) in edges.iter().enumerate() {
            let near = [(i, i + 1)]
                .repeat(m)
                .into_iter()
                .flat_map(|(i, j)| [i, j])
                .collect::<Vec<_>>();
            assert!(
                g.is_identity(&ReflectionWord(near[..2 * m].to_vec()))
                    .unwrap()
            );
            assert!(
                !g.is_identity(&ReflectionWord(near[..2 * (m - 1)].to_vec()))
                    .unwrap()
            );
        }
        for i in 0..g.dimension {
            assert!(g.is_identity(&ReflectionWord(vec![i, i])).unwrap());
            for j in (i + 2)..g.dimension {
                assert!(g.is_identity(&ReflectionWord(vec![i, j, i, j])).unwrap());
            }
        }
    };
    for p in 3..=12 {
        for q in 3..=12 {
            if (p - 2) * (q - 2) > 4 {
                check(ReflectionGroup::h2(p, q).unwrap(), &[p, q]);
            }
        }
    }
    let mut count = 0;
    for p in 3..=6 {
        for q in 3..=6 {
            for r in 3..=6 {
                if let Ok(g) = ReflectionGroup::h3(p, q, r) {
                    check(g, &[p as usize, q as usize, r as usize]);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 15);
}

#[test]
fn triangle_mirror_conjugates_the_wedge_rotation_to_its_inverse() {
    let g = ReflectionGroup::h2(7, 3).unwrap();
    let mirror = g.parse("0").unwrap();
    let rotation = g.parse("01").unwrap();
    assert_eq!(g.kind(&mirror).unwrap(), "Reflection");
    let image = mirror.conjugate(&rotation);
    assert!(!g.same_isometry(&rotation, &image).unwrap());
    assert!(g.same_isometry(&rotation.inverse(), &image).unwrap());
    assert!(g.is_identity(&rotation.compose(&image)).unwrap());
    assert_eq!(
        g.kind(&g.parse("012").unwrap()).unwrap(),
        "Glide reflection"
    );
    assert!(g.parse("3").is_err());
    assert!(g.parse("").is_err());
    assert!(g.parse("0120120120120").is_err());
}

#[test]
fn dodecahedral_honeycomb_edges_close_by_five_but_adjacent_rotations_do_not_commute() {
    let g = ReflectionGroup::h3(5, 3, 5).unwrap();
    assert!(g.is_identity(&g.parse("2323232323").unwrap()).unwrap());
    assert_eq!(g.kind(&g.parse("3").unwrap()).unwrap(), "Reflection");
    assert!(
        !g.same_isometry(&g.parse("0102").unwrap(), &g.parse("0201").unwrap())
            .unwrap()
    );
    assert!(ReflectionGroup::h3(4, 3, 4).is_err()); // Euclidean
    assert!(ReflectionGroup::h3(3, 3, 7).is_err()); // hyperideal
}

#[test]
fn wedge_mismatch_vanishes_for_the_regular_heptagon_and_changes_sign_on_either_side() {
    assert!(wedge_angle_mismatch(7, 3, 7).unwrap().abs() < 1e-12);
    assert!(wedge_angle_mismatch(7, 3, 6).unwrap() > 0.0);
    assert!(wedge_angle_mismatch(7, 3, 8).unwrap() < 0.0);
    // The hyperbolic {4,5} edge length agrees with alpha=2*pi/5.
    assert!(wedge_angle_mismatch(4, 5, 4).unwrap().abs() < 1e-12);
}
