//! Known-answer tests for the general defect Combinations ("Equivalence") table engine,
//! derived strictly from wallpaper-group algebra and hand formulas.

use std::sync::atomic::AtomicBool;

use super::*;
use crate::types::{Bond2D, Cell2D, Frame2D};

fn square_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![
            Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
        ],
    }
}

fn honeycomb_cell() -> Cell2D {
    let s3 = 3.0_f64.sqrt();
    // Standard 60-degree primitive honeycomb cell: a1 = (1, 0), a2 = (1/2, sqrt(3)/2),
    // 2 sites at (a1 + a2)/3 and 2*(a1 + a2)/3.
    let a1 = [1.0, 0.0];
    let a2 = [0.5, 0.5 * s3];
    let b0 = [(a1[0] + a2[0]) / 3.0, (a1[1] + a2[1]) / 3.0];
    let b1 = [2.0 * b0[0], 2.0 * b0[1]];
    Cell2D {
        a1,
        a2,
        basis: vec![b0, b1],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 1, n1: -1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 1, n1: 0, n2: -1, kind: 0 },
        ],
    }
}

/// Chiral square lattice (`p4`): tilted 4-gon around (0,0) joined by bridges across 2c.
fn chiral_p4_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.35, 0.15], [0.85, 0.35], [0.65, 0.85], [0.15, 0.65]],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: -1, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 2, n1: 0, n2: -1, kind: 0 },
            Bond2D { i: 2, j: 3, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 3, j: 0, n1: 0, n2: 1, kind: 0 },
            Bond2D { i: 0, j: 2, n1: 0, n2: -1, kind: 0 },
            Bond2D { i: 1, j: 3, n1: 1, n2: 0, kind: 0 },
        ],
    }
}

/// Non-symmorphic `p4g` tiling: oppositely tilted 4-gons around 2a(i)=(0,0) and 2a(ii)=(0.5,0.5).
fn p4g_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.35, 0.15], [0.85, 0.35], [0.65, 0.85], [0.15, 0.65]],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: -1, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 2, n1: 0, n2: -1, kind: 0 },
            Bond2D { i: 2, j: 3, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 3, j: 0, n1: 0, n2: 1, kind: 0 },
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 2, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 2, j: 3, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 3, j: 0, n1: 0, n2: 0, kind: 0 },
        ],
    }
}

fn fast_opts(max_sep: f64) -> CombinationsOptions {
    CombinationsOptions {
        max_sep,
        radius: 2,
        grid: 1,
        draw_thumbnails: false,
        include_single_gallery: false,
        bmax: 2.0,
    }
}

fn has_id_pair(entry: &CombinationEntry, a: &str, b: &str) -> bool {
    entry.defect_ids.len() == 2
        && ((entry.defect_ids[0] == a && entry.defect_ids[1] == b)
            || (entry.defect_ids[0] == b && entry.defect_ids[1] == a))
}

fn find_built<'a>(
    table: &'a CombinationsTable,
    a: &str,
    b: &str,
    sep: f64,
) -> Vec<(&'a ChargeGroup, &'a CoreGroup, &'a CombinationEntry)> {
    let mut out = Vec::new();
    for cg in &table.charge_groups {
        for core in &cg.core_groups {
            for m in &core.members {
                if has_id_pair(m, a, b) && (m.separation - sep).abs() < 1e-4 {
                    out.push((cg, core, m));
                }
            }
        }
    }
    out
}

/// Verify the exact kinematic dipole identity `|b| = 2 sin(|theta_1| / 2) * sep` for every
/// disclination dipole (`theta_1 + theta_2 = 0`) in `table`.
fn assert_all_disclination_dipoles_obey_burgers_law(table: &CombinationsTable, tag: &str) {
    let mut checked = 0usize;
    for cg in &table.charge_groups {
        for core in &cg.core_groups {
            for m in &core.members {
                if m.specs.len() != 2 {
                    continue;
                }
                let (
                    DefectSpec::Disclination { angle_deg: a1, .. },
                    DefectSpec::Disclination { angle_deg: a2, .. },
                ) = (&m.specs[0], &m.specs[1])
                else {
                    continue;
                };
                if (a1 + a2).abs() > 1e-6 {
                    continue;
                }
                assert_eq!(cg.charge.kind, "dislocation", "{tag}: dipole {} must give dislocation", m.label);
                let got_b = cg.charge.length.expect("dislocation must carry |b|");
                let want_b = 2.0 * (a1.abs().to_radians() * 0.5).sin() * m.separation;
                assert!(
                    (got_b - want_b).abs() < 1e-5,
                    "{tag}: dipole {} has |b|={got_b} != 2 sin(|{a1}|/2)*{:.6} = {want_b}",
                    m.label,
                    m.separation
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "{tag}: expected at least one disclination dipole");
}

#[test]
fn square_p4m_disclination_pairs_algebraic_laws() {
    let cell = square_cell();
    let table = build_combinations_table(
        "p4m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DisclinationPairs,
        &fast_opts(1.5),
        None,
    )
    .expect("square p4m disclination pairs");
    assert!(!table.cancelled);
    assert_eq!(table.group, "p4m");
    assert_eq!(table.built_count + table.refused_count, table.total_candidates);

    // 1. Exact dipole Burgers formula |b| = 2 sin(|theta|/2) * sep on every dipole.
    assert_all_disclination_dipoles_obey_burgers_law(&table, "square p4m");

    // 2. Even-sublattice theorem: every 1a/1a or 1b/1b +-90 pair has |b|^2 even (never |b|=1);
    //    shortest at d = (1, 0) has |b| = sqrt(2).
    for cls in ["1a", "1b"] {
        let a = format!("D({cls}, +90)");
        let b = format!("D({cls}, -90)");
        let mut saw_d10 = false;
        for cg in &table.charge_groups {
            for core in &cg.core_groups {
                for m in &core.members {
                    if has_id_pair(m, &a, &b) {
                        let b_len = cg.charge.length.expect("dislocation length");
                        let b2 = (b_len * b_len).round() as i64;
                        assert_eq!(b2 % 2, 0, "{cls} +-90 dipole {} must have even |b|^2, got {b2}", m.label);
                        assert!((b_len - 1.0).abs() > 1e-3, "{cls} +-90 dipole can never give |b|=1");
                        if (m.separation - 1.0).abs() < 1e-5 {
                            assert!((b_len - 2.0_f64.sqrt()).abs() < 1e-5);
                            saw_d10 = true;
                        }
                    }
                }
            }
        }
        assert!(saw_d10, "expected {cls}/{cls} +-90 dipole at sep=1.0 with |b|=sqrt(2)");
    }

    // 3. D(1a, +90) + D(1b, -90) at sep = sqrt(0.5) -> dislocation |b| = 1, MatchesSingle (volterra).
    let ab = find_built(&table, "D(1a, +90)", "D(1b, -90)", 0.5_f64.sqrt());
    assert_eq!(ab.len(), 1);
    assert_eq!(ab[0].0.charge.kind, "dislocation");
    assert!((ab[0].0.charge.length.unwrap() - 1.0).abs() < 1e-5);
    assert_eq!(ab[0].1.status, SingleMatchStatus::MatchesSingle);
    assert_eq!(ab[0].1.match_tier, "volterra");

    // 4. D(1a, -90) + D(1b, +90) at sep = sqrt(0.5) ALSO builds (build-order fallback) and gives |b| = 1.
    let ba = find_built(&table, "D(1a, -90)", "D(1b, +90)", 0.5_f64.sqrt());
    assert_eq!(ba.len(), 1, "D(1a, -90) + D(1b, +90) must build via order fallback");
    assert_eq!(ba[0].0.charge.kind, "dislocation");
    assert!((ba[0].0.charge.length.unwrap() - 1.0).abs() < 1e-5);

    // 5. D(1b, -180) + D(1b, +180) at sep = 1.0 -> |b| = 2 (NOT sqrt(2)).
    let bb180 = find_built(&table, "D(1b, -180)", "D(1b, +180)", 1.0);
    assert_eq!(bb180.len(), 1);
    assert!((bb180[0].0.charge.length.unwrap() - 2.0).abs() < 1e-5);

    // 6. D(1b, -180) + D(2c, +180) at sep = 0.5 -> |b| = 1.
    let bc180 = find_built(&table, "D(1b, -180)", "D(2c, +180)", 0.5);
    assert_eq!(bc180.len(), 1);
    assert!((bc180[0].0.charge.length.unwrap() - 1.0).abs() < 1e-5);

    // 7. D(1b, -180) + D(1a, +180) at sep = sqrt(0.5) -> |b| = sqrt(2).
    let ba180 = find_built(&table, "D(1b, -180)", "D(1a, +180)", 0.5_f64.sqrt());
    assert_eq!(ba180.len(), 1);
    assert!((ba180[0].0.charge.length.unwrap() - 2.0_f64.sqrt()).abs() < 1e-5);

    // 8. D(1a, -180) + D(1a, -180) at sep = 1.0 -> whole_turn (-360 deg), NoSingleDefect, core & thumbnail present.
    let wt = find_built(&table, "D(1a, -180)", "D(1a, -180)", 1.0);
    assert_eq!(wt.len(), 1);
    assert_eq!(wt[0].0.charge.kind, "whole_turn");
    assert!((wt[0].0.charge.angle + 360.0).abs() < 1e-5);
    assert!(!wt[0].0.has_single_defect);
    assert_eq!(wt[0].1.status, SingleMatchStatus::NoSingleDefect);
    assert!(wt[0].1.core_nodes > 0);
    assert!(wt[0].1.thumbnail.is_some());

    // 9. D(1a, +180) + D(1a, +180) is refused (>= 360 deg).
    let ref_360: Vec<_> = table
        .refused
        .iter()
        .filter(|r| r.defect_ids == ["D(1a, +180)", "D(1a, +180)"])
        .collect();
    assert!(!ref_360.is_empty(), "D(1a, +180) + D(1a, +180) must be refused");
    assert!(ref_360.iter().all(|r| r.reason.contains("360")));

    // 10. D(2c, +180) + D(2c, -180) keeps (-0.5, 0.5) and (+0.5, 0.5) distinct under G_+.
    let cc180 = find_built(&table, "D(2c, +180)", "D(2c, -180)", 0.5_f64.sqrt());
    assert_eq!(cc180.len(), 2, "2c +-180 must keep (-0.5,0.5) and (+0.5,0.5) distinct under G_+");
    let has_disp = |dx: f64, dy: f64| {
        cc180.iter().any(|(_, _, m)| {
            let d = m.displacements[0];
            (d[0] - dx).abs() < 1e-5 && (d[1] - dy).abs() < 1e-5
        })
    };
    assert!(has_disp(-0.5, 0.5), "missing (-0.5, 0.5) representative");
    assert!(has_disp(0.5, 0.5), "missing (+0.5, 0.5) representative");
}

#[test]
fn square_p4m_dislocation_pairs_glide_climb_and_sum() {
    let cell = square_cell();
    let table = build_combinations_table(
        "p4m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DislocationPairs,
        &fast_opts(1.5),
        None,
    )
    .expect("square p4m dislocation pairs");
    assert!(!table.cancelled);

    let vac = table
        .charge_groups
        .iter()
        .find(|cg| cg.charge.kind == "vacuum")
        .expect("expected vacuum charge group for antiparallel dislocation dipoles");

    let mut glide_pos = None;
    let mut glide_neg = None;
    let mut climb_pos = None;
    let mut climb_neg = None;
    for core in &vac.core_groups {
        for m in &core.members {
            if m.defect_ids != ["T1", "T1"] || (m.separation - 1.0).abs() > 1e-5 {
                continue;
            }
            let d = m.displacements[0];
            if (d[0] - 1.0).abs() < 1e-5 && d[1].abs() < 1e-5 {
                glide_pos = Some(m.delta_v);
            } else if (d[0] + 1.0).abs() < 1e-5 && d[1].abs() < 1e-5 {
                glide_neg = Some(m.delta_v);
            } else if d[0].abs() < 1e-5 && (d[1] - 1.0).abs() < 1e-5 {
                climb_pos = Some(m.delta_v);
            } else if d[0].abs() < 1e-5 && (d[1] + 1.0).abs() < 1e-5 {
                climb_neg = Some(m.delta_v);
            }
        }
    }
    assert_eq!(glide_pos, Some(Some(0)), "glide dipole d=(1,0) must have delta_v=0; refused={:?}", table.refused);
    assert_eq!(
        glide_neg,
        Some(Some(0)),
        "glide dipole d=(-1,0) must not be falsely coalesced and must have delta_v=0; refused={:?}",
        table.refused
    );
    let dv_pos = climb_pos.flatten().expect("climb dipole d=(0,1) must yield exact delta_v");
    let dv_neg = climb_neg.flatten().expect("climb dipole d=(0,-1) must yield exact delta_v");
    assert_eq!(dv_pos.abs(), 1, "square climb dipole |delta_v| = |b x d| / area = 1");
    assert_eq!(dv_neg, -dv_pos, "opposite climb dipoles must have opposite signs of delta_v");

    // (+1,0) + (0,1) -> dislocation |b| = sqrt(2).
    let has_sqrt2 = table.charge_groups.iter().any(|cg| {
        cg.charge.kind == "dislocation"
            && cg.charge.length.is_some_and(|l| (l - 2.0_f64.sqrt()).abs() < 1e-5)
            && cg.core_groups.iter().any(|c| c.members.iter().any(|m| m.defect_ids == ["T1", "T1"]))
    });
    assert!(has_sqrt2, "T1(+1,0) + T1(0,1) must produce dislocation |b| = sqrt(2)");
}

#[test]
fn square_p4m_mixed_pairs_disclination_plus_dislocation() {
    let cell = square_cell();
    let table = build_combinations_table(
        "p4m",
        &cell,
        Frame2D::default(),
        &CombinationMode::MixedPairs,
        &fast_opts(1.2),
        None,
    )
    .expect("square p4m mixed pairs");
    assert!(!table.cancelled);

    // D(1a, +90) + T1 shifts the +90 rotation centre from 1a to 1b: R_{90} * T_{-b} fixes b(1-i)/2 in 1b.
    let found = table.charge_groups.iter().any(|cg| {
        cg.charge.kind == "disclination"
            && (cg.charge.angle - 90.0).abs() < 1e-5
            && cg.charge.centre_class.as_deref() == Some("1b")
            && cg.core_groups.iter().any(|c| {
                c.members.iter().any(|m| m.defect_ids == ["D(1a, +90)", "T1"])
            })
    });
    assert!(found, "D(1a, +90) + T1 must produce D(1b, +90)");
}

#[test]
fn square_p4m_triple_f90_v90_v90_has_two_orbits_both_at_1b() {
    let cell = square_cell();
    let mode = CombinationMode::Triple {
        ids: ["D(1b, +90)".to_string(), "D(1a, -90)".to_string(), "D(1a, -90)".to_string()],
    };
    let table = build_combinations_table("p4m", &cell, Frame2D::default(), &mode, &fast_opts(1.5), None)
        .expect("square p4m triple");
    assert!(!table.cancelled);
    assert_eq!(table.total_candidates, 2, "adjacent vs opposite corners of the face -> exactly 2 orbits");
    assert_eq!(table.built_count, 2, "refused={:?}", table.refused);
    assert_eq!(table.refused_count, 0);
    assert_eq!(table.charge_groups.len(), 1, "both orbits share the exact same G~+ charge");

    let cg = &table.charge_groups[0];
    assert_eq!(cg.charge.kind, "disclination");
    assert!((cg.charge.angle + 90.0).abs() < 1e-5);
    assert_eq!(cg.charge.centre_class.as_deref(), Some("1b"));

    let mut seps: Vec<f64> = cg.core_groups.iter().flat_map(|c| c.members.iter().map(|m| m.separation)).collect();
    seps.sort_by(|a, b| a.total_cmp(b));
    assert!((seps[0] - 1.0).abs() < 1e-5, "adjacent corners diameter = 1.0");
    assert!((seps[1] - 2.0_f64.sqrt()).abs() < 1e-5, "opposite corners diameter = sqrt(2)");
}

#[test]
fn honeycomb_p6m_dipoles_and_two_site_delta_v() {
    let cell = honeycomb_cell();
    let discl_table = build_combinations_table(
        "p6m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DisclinationPairs,
        &fast_opts(1.1),
        None,
    )
    .expect("honeycomb p6m disclination pairs");
    assert_eq!(discl_table.group, "p6m");
    assert_all_disclination_dipoles_obey_burgers_law(&discl_table, "honeycomb p6m");

    let disloc_table = build_combinations_table(
        "p6m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DislocationPairs,
        &fast_opts(1.1),
        None,
    )
    .expect("honeycomb p6m dislocation pairs");
    let vac = disloc_table
        .charge_groups
        .iter()
        .find(|cg| cg.charge.kind == "vacuum")
        .expect("honeycomb vacuum dipole group");

    let mut glide_dvs = Vec::new();
    let mut climb_dvs = Vec::new();
    for core in &vac.core_groups {
        for m in &core.members {
            if m.defect_ids != ["T1", "T1"] || (m.separation - 1.0).abs() > 1e-5 {
                continue;
            }
            let DefectSpec::Dislocation { b: b0, .. } = m.specs[0] else { continue };
            let d = m.displacements[0];
            let area_cross = cross2(b0, d).abs();
            if area_cross < 1e-5 {
                glide_dvs.push(m.delta_v);
            } else {
                climb_dvs.push(m.delta_v);
            }
        }
    }
    assert!(!glide_dvs.is_empty(), "expected honeycomb glide dipoles");
    assert!(glide_dvs.iter().all(|&dv| dv == Some(0)), "honeycomb glide dipoles must have delta_v = 0");
    assert!(climb_dvs.len() >= 2, "expected both opposite-sign honeycomb climb dipoles");
    let mut vals: Vec<i32> = climb_dvs.into_iter().map(|dv| dv.expect("exact delta_v")).collect();
    vals.sort_unstable();
    assert!(vals.contains(&-2) && vals.contains(&2), "2-site honeycomb climb dipoles must give delta_v = +-2, got {vals:?}");
}

#[test]
fn chiral_p4_and_split_p4g_disclination_dipoles() {
    for (name, cell, expect_split) in [("p4", chiral_p4_cell(), false), ("p4g", p4g_cell(), true)] {
        let cat = catalogue_for_combinations(name, &cell, Frame2D::default(), 2.0);
        assert_eq!(cat.group, name);
        if expect_split {
            assert!(cat.centre_classes.iter().any(|c| c.split), "{name} must have split centre classes");
        }
        let table = build_combinations_table(
            name,
            &cell,
            Frame2D::default(),
            &CombinationMode::DisclinationPairs,
            &fast_opts(1.1),
            None,
        )
        .unwrap_or_else(|e| panic!("{name} disclination pairs failed: {e}"));
        assert!(!table.cancelled);
        assert_all_disclination_dipoles_obey_burgers_law(&table, name);
    }
}

#[test]
fn cancellation_stops_immediately() {
    let cell = square_cell();
    let cancel = AtomicBool::new(true);
    let table = build_combinations_table(
        "p4m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DisclinationPairs,
        &fast_opts(1.5),
        Some(&cancel),
    )
    .expect("cancelled call returns Ok");
    assert!(table.cancelled);
    assert_eq!(table.built_count, 0);
}
