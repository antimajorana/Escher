use super::*;
use crate::space3d::{list_layer_groups, list_space_groups_3d};

const ZERO: [i32; 3] = [0; 3];

#[test]
fn every_crystallographic_group_has_one_rep_per_point_operation() {
    for entry in list_space_groups_3d() {
        let group = CrystalHolonomies::space_group(entry.number).unwrap();
        assert_eq!(
            group.representatives().len(),
            entry.point_order,
            "{}",
            entry.symbol
        );
        verify_reps(&group);
    }
    for entry in list_layer_groups() {
        let group = CrystalHolonomies::layer_group(entry.number).unwrap();
        assert_eq!(
            group.representatives().len(),
            entry.point_order,
            "{}",
            entry.symbol
        );
        assert!(group.element(0, [0, 0, 1]).is_none());
        verify_reps(&group);
    }
}

fn verify_reps(group: &CrystalHolonomies) {
    for a in group.representatives() {
        assert_eq!(a.compose(&a.inverse()), AffineHolonomy::identity());
        let (power, translation) = a.closing_power();
        assert!([1, 2, 3, 4, 6].contains(&power));
        assert_eq!(translation.linear(), AffineHolonomy::identity().linear());
        let shift_in_basis = rmat_vec(&group.basis_inverse, translation.shift());
        assert!(shift_in_basis.iter().all(|c| c.as_i64().is_some()));
        assert!(shift_in_basis[group.rank()..].iter().all(|c| c.is_zero()));
    }
}

#[test]
fn two_one_screw_and_two_glides_square_to_full_translations() {
    // P2₁(x,y,z) = (-x, y+1/2, -z), Pc(x,y,z) = (x,-y,z+1/2).
    let cases = [
        (
            CrystalHolonomies::space_group(4).unwrap(),
            "Screw rotation",
            [0, 1, 0],
        ),
        (
            CrystalHolonomies::space_group(7).unwrap(),
            "Glide reflection",
            [0, 0, 1],
        ),
        // Layer p11a sends (x,y,z) to (x+1/2, y, -z).
        (
            CrystalHolonomies::layer_group(5).unwrap(),
            "Glide reflection",
            [1, 0, 0],
        ),
    ];
    for (group, kind, periods) in cases {
        let op = group.element(1, ZERO).unwrap();
        let translation = group.element(0, periods).unwrap();
        assert_eq!(op.kind(), kind);
        assert_eq!(op.compose(&op), translation);
        assert_eq!(op.closing_power(), (2, translation));
    }
    let mirror = CrystalHolonomies::layer_group(4)
        .unwrap()
        .element(1, ZERO)
        .unwrap();
    assert_eq!(mirror.kind(), "Reflection");
    assert!(!mirror.is_proper());
    assert!(mirror.preserves_layer_orientation());
}

#[test]
fn exact_proper_conjugacy_solves_unbounded_translations() {
    // In P2, conjugating the y-axis half-turn by t_x gives [R | (2,0,0)].
    // Odd x shifts cannot be obtained by conjugating a primitive P2 operation.
    let group = CrystalHolonomies::space_group(3).unwrap();
    let a = group.element(1, ZERO).unwrap();
    let even = group.element(1, [200, 0, 0]).unwrap();
    let odd = group.element(1, [201, 0, 0]).unwrap();
    assert!(group.properly_conjugate(&a, &even));
    assert!(!group.properly_conjugate(&a, &odd));
    let tx = group.element(0, [1, 0, 0]).unwrap();
    assert_ne!(tx.compose(&a), a.compose(&tx));
    assert_eq!(a.conjugate(&tx), tx.inverse());

    // In Pm a pure translation normal to the mirror is reversed by the
    // mirror, and no proper point operation relates it to its inverse.
    let group = CrystalHolonomies::space_group(6).unwrap();
    let base = group.element(0, [0, 1, 0]).unwrap();
    let mirror = group.element(1, ZERO).unwrap();
    assert_eq!(mirror.conjugate(&base), base.inverse());
    assert!(!group.properly_conjugate(&base, &base.inverse()));
}

#[test]
fn switch_registry_between_pure_mirror_and_half_period_glide() {
    let pm = CrystalHolonomies::space_group(6).unwrap();
    let mirror = pm.element(1, ZERO).unwrap();
    let (glide, member) = pm.alternate_registry(&mirror).unwrap();
    assert!(!member);
    assert_eq!(glide.kind(), "Glide reflection");
    assert_eq!(glide.linear(), mirror.linear());
    assert_eq!(glide.compose(&glide).shift()[1], Rat::ZERO);

    let pc = CrystalHolonomies::space_group(7).unwrap();
    let (replacement, member) = pc
        .alternate_registry(&pc.element(1, ZERO).unwrap())
        .unwrap();
    assert!(!member);
    assert_eq!(replacement, mirror);

    let layer = CrystalHolonomies::layer_group(5).unwrap();
    let (replacement, member) = layer
        .alternate_registry(&layer.element(1, ZERO).unwrap())
        .unwrap();
    assert!(!member);
    assert_eq!(
        replacement,
        CrystalHolonomies::layer_group(4)
            .unwrap()
            .element(1, ZERO)
            .unwrap()
    );
}
