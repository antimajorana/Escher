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

fn assert_all_within_max_sep(table: &CombinationsTable, tag: &str) {
    for cg in &table.charge_groups {
        for core in &cg.core_groups {
            for m in &core.members {
                assert!(
                    m.separation <= table.max_sep + 1e-6,
                    "{tag}: built {} has sep {:.6} > max_sep {:.6}",
                    m.label,
                    m.separation,
                    table.max_sep
                );
            }
        }
    }
}

fn assert_algebraic_matches_built(cat: &DefectCatalogue, table: &CombinationsTable, tag: &str) {
    for cg in &table.charge_groups {
        for core in &cg.core_groups {
            for m in &core.members {
                let alg = algebraic_specs_classified(cat, Frame2D::default(), &m.specs)
                    .unwrap_or_else(|| panic!("{tag}: algebraic charge missing for {}", m.label));
                assert_eq!(alg.kind, cg.charge.kind, "{tag}: kind mismatch on {}", m.label);
                assert!((alg.angle - cg.charge.angle).abs() < 1e-4, "{tag}: angle mismatch on {}", m.label);
                if cg.charge.kind == "dislocation" {
                    let la = alg.length.expect("alg |b|");
                    let lb = cg.charge.length.expect("built |b|");
                    assert!((la - lb).abs() < 1e-4, "{tag}: |b| mismatch on {}: {la} vs {lb}", m.label);
                } else if cg.charge.kind == "disclination" {
                    assert_eq!(alg.centre_class, cg.charge.centre_class, "{tag}: centre mismatch on {}", m.label);
                }
            }
        }
    }
}

#[test]
fn square_p4m_disclination_pairs_algebraic_laws() {
    let cell = square_cell();
    let cat = catalogue_for_combinations("p4m", &cell, Frame2D::default(), 2.0);
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
    // Exact hand count of G_+ + swap orbits on square p4m (4 species on 1a, 4 on 1b, 2 on 2c):
    // Within 1.5a: 1a-1a (10 pairs * 2 C4-orbits: sep 1, sqrt(2)) = 20; 1b-1b = 20;
    // 1a-1b (16 * 1 orbit: sep sqrt(0.5)) = 16; 1a-2c (8 * 4 orbits: 0.5, two at sqrt(1.25), 1.5) = 32;
    // 1b-2c = 32; 2c-2c: distinct (-180/+180) has 2+2+2 = 6 orbits (sqrt(0.5), 1, sqrt(2)),
    // while each identical pair merges the two sqrt(0.5) diagonals via swap+R_{-90} -> 1+2+2 = 5 -> 2*5 = 10.
    // Total = 20 + 20 + 16 + 32 + 32 + 16 = 136; Frank +360 deg (+180/+180) refused = 2+2+1+4+4+5 = 18.
    assert_eq!(table.total_candidates, 136);
    assert_eq!(table.built_count + table.refused_count, 136);
    let frank360 = table.refused.iter().filter(|r| r.reason.contains("360")).count();
    assert_eq!(frank360, 18, "exact count of Frank +360 deg orbits within 1.5a");
    assert!(table.refused.iter().all(|r| r.algebraic_charge.is_some()));
    assert!(table.refused.iter().all(|r| r.reason.contains("360") || r.reason.contains("encloses an earlier defect core")));
    assert_all_within_max_sep(&table, "square p4m discl 1.5a");
    assert_algebraic_matches_built(&cat, &table, "square p4m discl 1.5a");

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
    //    Hand derivation of the core match: removing a +90 deg quadrant at vertex 1a drops its valence
    //    from 4 to 3 (3-valent site), while inserting a -90 deg quadrant at the adjacent face 1b turns
    //    that square face into a pentagon -> the exact unit square Volterra edge-dislocation (3-5) core.
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
    assert_eq!(ref_360.len(), 2, "D(1a, +180) + D(1a, +180) has 2 refused orbits within 1.5a");
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
    let cat = catalogue_for_combinations("p4m", &cell, Frame2D::default(), 2.0);
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
    // Exact P_+ (C_4) + swap hand count at max_sep = 1.5 (8 non-zero lattice displacements):
    // T1+T1: fix b_a=(1,0); b_b=(1,0) halved (8/2=4), b_b=(-1,0) fixed by R=-I (-R d = d -> 8),
    // b_b=(0,+-1) exchanged (8) -> 4+8+8 = 20. T2+T2 = 20. T1+T2 (no swap): 4*8 = 32. Total = 72.
    assert_eq!(table.total_candidates, 72);
    assert_eq!(table.built_count + table.refused_count, 72);
    assert_all_within_max_sep(&table, "square p4m disloc 1.5a");
    assert_algebraic_matches_built(&cat, &table, "square p4m disloc 1.5a");

    let vac = table
        .charge_groups
        .iter()
        .find(|cg| cg.charge.kind == "vacuum")
        .expect("expected vacuum charge group for antiparallel dislocation dipoles");

    let (mut glide_pos, mut glide_neg, mut climb_pos, mut climb_neg) = (None, None, None, None);
    for core in &vac.core_groups {
        for m in &core.members {
            if m.defect_ids != ["T1", "T1"] || (m.separation - 1.0).abs() > 1e-5 {
                continue;
            }
            let DefectSpec::Dislocation { b: b0, .. } = m.specs[0] else { continue };
            let d = m.displacements[0];
            let cr = cross2(b0, d);
            let dt = dot2(b0, d);
            if cr.abs() < 1e-6 {
                if dt > 0.0 { glide_pos = Some(m.delta_v); } else { glide_neg = Some(m.delta_v); }
            } else if dt.abs() < 1e-6 {
                if cr > 0.0 { climb_pos = Some(m.delta_v); } else { climb_neg = Some(m.delta_v); }
            }
        }
    }
    assert_eq!(glide_pos, Some(Some(0)), "glide dipole (b.d > 0) must have delta_v=0; refused={:?}", table.refused);
    assert_eq!(
        glide_neg,
        Some(Some(0)),
        "glide dipole (b.d < 0) must not be falsely coalesced and must have delta_v=0; refused={:?}",
        table.refused
    );
    let dv_pos = climb_pos.flatten().expect("climb dipole (b x d > 0) must yield exact delta_v");
    let dv_neg = climb_neg.flatten().expect("climb dipole (b x d < 0) must yield exact delta_v");
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
        &fast_opts(1.0),
        None,
    )
    .expect("square p4m mixed pairs");
    assert!(!table.cancelled);
    // Exact hand count of Stab^+(p_a) orbits at max_sep = 1.0 (4 lattice displacements, 2 classes T1, T2 of 4):
    // Stab^+(p_a) acts freely on (b, d), giving |members| * |disps| / |Stab^+| orbits per (discl, disloc):
    // 1a (4 discl, |C4|=4): 4 * 2 * (4*4/4) = 32; 1b: 32; 2c (2 discl, |C2|=2): 2 * 2 * (4*4/2) = 32. Total = 96.
    assert_eq!(table.total_candidates, 96);
    assert_eq!(table.built_count + table.refused_count, 96);
    assert_all_within_max_sep(&table, "square p4m mixed 1.0a");

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
fn square_p4m_triple_f90_v90_v90_charge_and_strict_overlap_rule() {
    let cell = square_cell();
    let mode = CombinationMode::Triple {
        ids: ["D(1b, +90)".to_string(), "D(1a, -90)".to_string(), "D(1a, -90)".to_string()],
    };
    let table = build_combinations_table("p4m", &cell, Frame2D::default(), &mode, &fast_opts(1.5), None)
        .expect("square p4m triple");
    assert!(!table.cancelled);
    // A face centre with two of its corners: the corners are adjacent (1 apart) or opposite (sqrt 2 apart).
    assert_eq!(table.total_candidates, 2, "adjacent vs opposite corners of the face -> exactly 2 orbits");
    assert_eq!(table.built_count + table.refused_count, 2);
    assert_all_within_max_sep(&table, "square p4m triple");
    let mut seps: Vec<f64> = table
        .charge_groups
        .iter()
        .flat_map(|g| g.core_groups.iter().flat_map(|c| c.members.iter().map(|m| m.separation)))
        .chain(table.refused.iter().map(|r| r.separation))
        .collect();
    seps.sort_by(|a, b| a.total_cmp(b));
    assert!((seps[0] - 1.0).abs() < 1e-5, "adjacent corners diameter = 1.0");
    assert!((seps[1] - 2.0_f64.sqrt()).abs() < 1e-5, "opposite corners diameter = sqrt(2)");

    // Group algebra: +90 - 90 - 90 = -90, and the product of the three rotations fixes a face centre (1b).
    // Every placement that builds must read exactly that off the lattice.
    assert!(table.built_count >= 1, "refused={:?}", table.refused);
    assert_eq!(table.charge_groups.len(), 1, "built placements share one charge");
    let cg = &table.charge_groups[0];
    assert_eq!(cg.charge.kind, "disclination");
    assert!((cg.charge.angle + 90.0).abs() < 1e-5);
    assert_eq!(cg.charge.centre_class.as_deref(), Some("1b"));

    // The author's rule (2026-11-22): the validity check stays strict. Three cores this close can overlap;
    // such a placement is refused for that reason and still carries its exact algebraic total charge.
    assert!(!table.refused.is_empty(), "adjacent-corner triple is refused under strict overlap rule");
    for r in &table.refused {
        assert!(r.reason.contains("encloses an earlier defect core"), "{}: {}", r.label, r.reason);
        let alg = r.algebraic_charge.as_deref().expect("refused triple must carry algebraic_charge");
        assert!(
            alg.starts_with("disclination D(1b, -90 deg)"),
            "expected D(1b, -90 deg) algebraic charge, got {alg:?}"
        );
    }
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

#[test]
fn square_p4m_exact_orbit_counts_at_1a_and_pair_order_independence() {
    let cell = square_cell();
    // 1. Disclination pairs within 1.0a: 1a-1a (10*1=10), 1b-1b (10*1=10), 1a-1b (16*1=16),
    //    1a-2c (8*1=8), 1b-2c (8*1=8), 2c-2c (distinct: 2+2=4; 2 identical: 2*(1+2)=6 -> 10).
    //    Total = 62; Frank +360 deg (+180/+180) refused = 1+1+1+1+1+3 = 8.
    let discl = build_combinations_table(
        "p4m", &cell, Frame2D::default(), &CombinationMode::DisclinationPairs, &fast_opts(1.0), None,
    ).expect("square p4m discl 1.0a");
    assert_eq!(discl.total_candidates, 62);
    assert_eq!(discl.built_count + discl.refused_count, 62);
    assert_eq!(discl.refused.iter().filter(|r| r.reason.contains("360")).count(), 8);
    assert!(discl.refused.iter().all(|r| r.algebraic_charge.is_some()));
    assert!(discl.refused.iter().all(|r| r.reason.contains("360") || r.reason.contains("encloses an earlier defect core")));
    assert_all_within_max_sep(&discl, "square p4m discl 1.0a");

    // 2. Dislocation pairs within 1.0a under P_+ (C_4) + swap (4 unit displacements):
    //    T1+T1: b_b=(1,0) halved -> 2, b_b=(-1,0) fixed by R=-I (-R d = d) -> 4, b_b=(0,+-1) swapped -> 4 => 10.
    //    T2+T2 = 10; T1+T2 (no swap) = 4*4 = 16. Total = 10 + 16 + 10 = 36.
    let disloc = build_combinations_table(
        "p4m", &cell, Frame2D::default(), &CombinationMode::DislocationPairs, &fast_opts(1.0), None,
    ).expect("square p4m disloc 1.0a");
    assert_eq!(disloc.total_candidates, 36);
    assert_eq!(disloc.built_count + disloc.refused_count, 36);
    assert_all_within_max_sep(&disloc, "square p4m disloc 1.0a");

    // 3. Pair order-independence: D(1a, +90) at (0,0) + D(1b, -90) at (0.5,0.5) built as [0,1] vs [1,0].
    let lat_info = LatticeCoreInfo::new(&cell);
    let cat = catalogue_for_combinations("p4m", &cell, Frame2D::default(), 2.0);
    let s0 = DefectSpec::disclination(90.0, [0.0, 0.0]);
    let s1 = DefectSpec::disclination(-90.0, [0.5, 0.5]);
    let (c01, _, _) = build_point_defects_with_info(&lat_info, &[s0.clone(), s1.clone()], None, None).expect("[0,1]");
    let (c10, _, _) = build_point_defects_with_info(&lat_info, &[s1.clone(), s0.clone()], None, None).expect("[1,0]");
    let q01 = classify_crystal_charge_with_cat(&cat, &c01).expect("charge [0,1]");
    let q10 = classify_crystal_charge_with_cat(&cat, &c10).expect("charge [1,0]");
    assert!(same_charge_in_grp(&cat.grp, &q01, &q10, false), "total charge must be order-independent");
    assert_eq!(q01.kind, "dislocation");
    assert!((q01.length.unwrap() - 1.0).abs() < 1e-5);
    let h01 = core_graph_with_rim(&c01, &resolve_patch_rim(&c01, &lat_info, &[s0.clone(), s1.clone()]), &lat_info, 2).expect("core [0,1]");
    let h10 = core_graph_with_rim(&c10, &resolve_patch_rim(&c10, &lat_info, &[s1, s0]), &lat_info, 2).expect("core [1,0]");
    assert!(isomorphic_cores(&h01, &h10), "D(1a,+90)+D(1b,-90) yields isomorphic 3-5 cores in both orders");
}
