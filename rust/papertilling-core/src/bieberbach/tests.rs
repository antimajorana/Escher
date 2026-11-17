use std::time::Instant;

use super::{analyze_wallpaper_group, from_wallpaper_group, ConvOp, Rat, SpaceGroupAnalyzer};

#[test]
fn all_17_wallpaper_groups_match_cohomology_abelianization_and_68_torus_counts() {
    let t0 = Instant::now();
    type ExpectedRow<'a> = (&'a str, usize, &'a [i128], bool, &'a str, &'a str, [usize; 4]);
    let expected: &[ExpectedRow<'_>] = &[
        ("p1", 1, &[], true, "Z^2", "Z^2", [4, 9, 16, 36]),
        ("p2", 2, &[], true, "Z_2^3", "Z_2^3", [8, 6, 14, 24]),
        ("pm", 2, &[2], true, "Z^2", "Z_2^2 x Z", [8, 9, 20, 36]),
        ("pg", 2, &[2], false, "Z^2", "Z_2 x Z", [8, 9, 20, 36]),
        ("cm", 2, &[], true, "Z^2", "Z_2 x Z", [5, 9, 14, 27]),
        ("pmm", 4, &[2, 2], true, "Z_2^3", "Z_2^4", [16, 9, 25, 36]),
        ("pmg", 4, &[2, 2], false, "Z_2^3", "Z_2^3", [10, 9, 19, 30]),
        ("pgg", 4, &[2, 2], false, "Z_2^3", "Z_2 x Z_4", [10, 9, 19, 30]),
        ("cmm", 4, &[], true, "Z_2^3", "Z_2^3", [10, 9, 16, 24]),
        ("p4", 4, &[], true, "Z_2 x Z_4", "Z_2 x Z_4", [10, 6, 13, 18]),
        ("p4m", 8, &[2], true, "Z_2 x Z_4", "Z_2^3", [14, 9, 20, 27]),
        ("p4g", 8, &[2], false, "Z_2 x Z_4", "Z_2 x Z_4", [11, 9, 17, 24]),
        ("p3", 3, &[], true, "Z_3^2", "Z_3^2", [4, 11, 8, 20]),
        ("p3m1", 6, &[], true, "Z_3^2", "Z_2", [5, 10, 10, 19]),
        ("p31m", 6, &[], true, "Z_3^2", "Z_6", [5, 10, 10, 19]),
        ("p6", 6, &[], true, "Z_6", "Z_6", [8, 10, 10, 16]),
        ("p6m", 12, &[], true, "Z_6", "Z_2^2", [10, 11, 14, 20]),
    ];

    for &(grp, p_ord, h2, split, ab_p, ab_f, tor) in expected {
        let sum = analyze_wallpaper_group(grp).unwrap();
        assert_eq!(sum.point_order, p_ord, "{grp} |P|");
        assert_eq!(sum.h2_torsion, h2, "{grp} H^2");
        assert_eq!(sum.is_split, split, "{grp} split");
        assert_eq!(sum.ab_plus_label, ab_p, "{grp} (G+)^ab");
        assert_eq!(sum.ab_full_label, ab_f, "{grp} G^ab");
        let got_tor: Vec<usize> = sum.torus_counts.iter().map(|&(_, c)| c).collect();
        assert_eq!(got_tor, tor, "{grp} torus counts");
    }
    eprintln!("# time (17 wallpaper groups): {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
}

#[test]
fn universal_dipole_braiding_and_fusion_channels_on_wallpaper_groups() {
    let cm = from_wallpaper_group("cm").unwrap();
    let p4g = from_wallpaper_group("p4g").unwrap();
    let p6m = from_wallpaper_group("p6m").unwrap();
    let p31m = from_wallpaper_group("p31m").unwrap();

    // Check chi_G conjugacy invariance on first 4 elements
    let ab = p4g.compute_abelianization(&p4g.all_indices());
    for r1 in 0..p4g.p_ops.len().min(4) {
        for r2 in 0..p4g.p_ops.len().min(4) {
            let (cb, cr) = p4g.seitz_conj(&[1, 0], r1, &[0, 1], r2);
            assert_eq!(ab.chi(&cb, cr), ab.chi(&[0, 1], r2));
        }
    }

    // p6m 60 + 60 -> 120 fusion channels must span both 1a and 2b Wyckoff classes
    let c6_idx = p6m
        .p_ops
        .iter()
        .position(|r| p6m.det_sign[p6m.p_lookup[&super::analyzer::imat_key(r)]] == 1 && p6m.mat_order(p6m.p_lookup[&super::analyzer::imat_key(r)]) == 6)
        .unwrap();
    let channels = p6m.unbased_fusion_channels(&[0, 0], c6_idx, &[0, 0], c6_idx);
    assert!(!channels.is_empty());
    let rot120: Vec<_> = channels.iter().filter(|ch| ch.order == 3 && ch.det == 1).collect();
    assert!(channels.iter().any(|ch| ch.order == 1), "achiral p6m also contains mirror-conjugated 0-deg dislocation channel");
    assert!(rot120.len() >= 2, "60x60->120 should produce multiple Wyckoff channels, got {:?}", channels);

    // Check p31m chiral-sublattice T+ fusion into non-zero dislocation class
    let c3_idx = p31m
        .p_ops
        .iter()
        .enumerate()
        .find(|&(i, _)| p31m.det_sign[i] == 1 && p31m.mat_order(i) == 3)
        .unwrap()
        .0;
    let m2_idx = p31m
        .p_ops
        .iter()
        .enumerate()
        .find(|&(i, r)| p31m.det_sign[i] == -1 && r[0][1] == 1 && r[1][0] == 1)
        .unwrap()
        .0;
    let (cb, cr) = p31m.seitz_conj(&[0, 0], m2_idx, &[1, 0], c3_idx);
    let (fb, fr) = p31m.seitz_mul(&[1, 0], c3_idx, &cb, cr);
    assert_eq!(fr, 0);
    let ab31 = p31m.compute_abelianization(&p31m.all_indices());
    assert!(ab31.chi(&fb, fr).iter().any(|&v| v != 0), "chiral fusion in p31m should be non-vacuum");
    let _ = cm;
}

#[test]
fn tetragonal_3d_123_vs_136_screw_dispiration_and_speed() {
    let t0 = Instant::now();
    let i3 = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
    let c2z = vec![vec![-1, 0, 0], vec![0, -1, 0], vec![0, 0, 1]];
    let c4z = vec![vec![0, -1, 0], vec![1, 0, 0], vec![0, 0, 1]];
    let c2y = vec![vec![-1, 0, 0], vec![0, 1, 0], vec![0, 0, -1]];
    let inv3 = vec![vec![-1, 0, 0], vec![0, -1, 0], vec![0, 0, -1]];
    let tb3 = vec![
        vec![Rat::ONE, Rat::ZERO, Rat::ZERO],
        vec![Rat::ZERO, Rat::ONE, Rat::ZERO],
        vec![Rat::ZERO, Rat::ZERO, Rat::ONE],
    ];
    let half = vec![Rat::new(1, 2), Rat::new(1, 2), Rat::new(1, 2)];
    let z3 = vec![Rat::ZERO, Rat::ZERO, Rat::ZERO];

    let gens_123 = vec![
        ConvOp { r: c2z.clone(), t: z3.clone() },
        ConvOp { r: c4z.clone(), t: z3.clone() },
        ConvOp { r: c2y.clone(), t: z3.clone() },
        ConvOp { r: inv3.clone(), t: z3.clone() },
        ConvOp { r: i3.clone(), t: vec![Rat::ONE, Rat::ZERO, Rat::ZERO] },
        ConvOp { r: i3.clone(), t: vec![Rat::ZERO, Rat::ONE, Rat::ZERO] },
        ConvOp { r: i3.clone(), t: vec![Rat::ZERO, Rat::ZERO, Rat::ONE] },
    ];
    let gens_136 = vec![
        ConvOp { r: c2z, t: z3.clone() },
        ConvOp { r: c4z, t: half.clone() },
        ConvOp { r: c2y, t: half },
        ConvOp { r: inv3, t: z3 },
        ConvOp { r: i3.clone(), t: vec![Rat::ONE, Rat::ZERO, Rat::ZERO] },
        ConvOp { r: i3.clone(), t: vec![Rat::ZERO, Rat::ONE, Rat::ZERO] },
        ConvOp { r: i3, t: vec![Rat::ZERO, Rat::ZERO, Rat::ONE] },
    ];

    let sg123 = SpaceGroupAnalyzer::new(3, 123, "P4/mmm", &gens_123, &tb3).unwrap();
    let sg136 = SpaceGroupAnalyzer::new(3, 136, "P4_2/mnm", &gens_136, &tb3).unwrap();

    let h2_123 = sg123.compute_cohomology_h2();
    let h2_136 = sg136.compute_cohomology_h2();
    assert_eq!(h2_123.torsion, vec![2, 2, 2, 2]);
    assert!(h2_123.is_split);
    assert_eq!(h2_136.torsion, vec![2, 2, 2, 2]);
    assert!(!h2_136.is_split);

    let (ab123_p, ab123_f) = sg123.abelianizations();
    assert_eq!(ab123_p.format(), "Z_2^4");
    assert_eq!(ab123_f.format(), "Z_2^5");
    let (ab136_p, ab136_f) = sg136.abelianizations();
    assert_eq!(ab136_p.format(), "Z_2^2 x Z_4");
    assert_eq!(ab136_f.format(), "Z_2^2 x Z_4");

    let non_id_plus_136: Vec<usize> = sg136.p_plus_indices().into_iter().filter(|&i| i != 0).collect();
    let secs_136 = sg136.classify_sectors(&sg136.all_indices(), &non_id_plus_136);
    let c4_136 = secs_136.iter().find(|s| s.order == 4).unwrap();
    assert_eq!(c4_136.alpha_r, Some(Rat::new(1, 2)));
    assert_eq!(c4_136.num_pure_classes, 0, "4_2 screw axis has zero pure rotation classes");

    let non_id_plus_123: Vec<usize> = sg123.p_plus_indices().into_iter().filter(|&i| i != 0).collect();
    let secs_123 = sg123.classify_sectors(&sg123.all_indices(), &non_id_plus_123);
    let c4_123 = secs_123.iter().find(|s| s.order == 4).unwrap();
    assert!(c4_123.num_pure_classes > 0);
    eprintln!("# time (3D #123 + #136): {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
}
