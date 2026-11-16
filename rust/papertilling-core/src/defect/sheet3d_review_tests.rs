//! Hand-counted topology and numerical safety regressions, deliberately small.

use super::*;
use crate::defect::charge::{check_frames, total_charge};

fn crystal_from_graph(pos: Vec<[f64; 2]>, edges: Vec<(usize, usize)>) -> BuiltCrystal {
    let n = pos.len();
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in &edges {
        for (a, b) in [(u, v), (v, u)] {
            adj[a].push(DirectedBondData {
                to: b,
                vec: [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]],
                kind: 0,
                seam: None,
                ang: None,
                across: false,
                ghost_stub: None,
            });
        }
    }
    BuiltCrystal {
        ref_pos: pos.clone(),
        pos,
        edges,
        adj,
        basis_idx: vec![0; n],
        side: vec![None; n],
        cone_sites: Vec::new(),
        wall_sigma: None,
        is_non_orientable: false,
    }
}

fn square_grid(side: usize) -> BuiltCrystal {
    let pos = (0..side)
        .flat_map(|y| (0..side).map(move |x| [x as f64, y as f64]))
        .collect();
    let mut edges = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let u = y * side + x;
            if x + 1 < side {
                edges.push((u, u + 1));
            }
            if y + 1 < side {
                edges.push((u, u + side));
            }
        }
    }
    crystal_from_graph(pos, edges)
}

#[test]
fn sheet3d_review_large_boundary_has_all_64_tiles_and_exact_boundary() {
    let (_, cell, _) = named_cell("square");
    let grid = square_grid(9);
    let res = relax_sheet_3d(&grid, &cell, &Sheet3DOptions::default(), None);
    assert_eq!(
        (res.xyz.len(), grid.edges.len(), res.faces.len()),
        (81, 144, 64)
    );
    assert_eq!(81_isize - 144 + res.faces.len() as isize, 1);
    assert_eq!(res.is_boundary.iter().filter(|&&b| b).count(), 32);
    for (u, &boundary) in res.is_boundary.iter().enumerate() {
        assert_eq!(
            boundary,
            u % 9 == 0 || u % 9 == 8 || u / 9 == 0 || u / 9 == 8
        );
    }
    assert!(res.faces.iter().all(|f| f.len() == 4));
    assert!(res.has_finite_geometry());
}

#[test]
fn sheet3d_review_core_ring_longer_than_exterior_is_retained() {
    // A 32-gon inside a square, joined by 32 fan triangles and four outer triangles.
    // V=36, E=72, bounded F=37, boundary=4: the largest walk is NOT the exterior.
    let mut pos: Vec<_> = (0..32)
        .map(|i| {
            let angle = PI / 4.0 + TWO_PI * i as f64 / 32.0;
            [angle.cos(), angle.sin()]
        })
        .collect();
    pos.extend([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]);
    let mut edges = std::collections::BTreeSet::new();
    for i in 0..32 {
        edges.insert((i.min((i + 1) % 32), i.max((i + 1) % 32)));
    }
    for k in 0..4 {
        let outer = 32 + k;
        edges.insert((outer.min(32 + (k + 1) % 4), outer.max(32 + (k + 1) % 4)));
        for j in 0..=8 {
            edges.insert(((8 * k + j + 28) % 32, outer));
        }
    }
    let crystal = crystal_from_graph(pos, edges.into_iter().collect());
    assert_eq!(check_frames(&crystal), 0);
    let (faces, boundary, diagnostics) = extract_interior_faces(&crystal, &crystal.pos, true);
    assert!(diagnostics.is_empty());
    assert_eq!(
        (crystal.pos.len(), crystal.edges.len(), faces.len()),
        (36, 72, 37)
    );
    assert_eq!(faces.iter().filter(|f| f.len() == 32).count(), 1);
    assert_eq!(faces.iter().filter(|f| f.len() == 3).count(), 36);
    assert!(boundary[..32].iter().all(|b| !b));
    assert!(boundary[32..].iter().all(|b| *b));
    // Face topology is intrinsic: even a completely collapsed display chart cannot remove the core.
    let collapsed_chart = vec![[0.0, 0.0]; 36];
    let (faces, boundary, diagnostics) = extract_interior_faces(&crystal, &collapsed_chart, true);
    assert!(diagnostics.is_empty());
    assert_eq!(faces.len(), 37);
    assert_eq!(faces.iter().filter(|f| f.len() == 32).count(), 1);
    assert_eq!(boundary.iter().filter(|&&b| b).count(), 4);
}

#[test]
fn sheet3d_review_disconnected_patches_remove_one_exterior_each() {
    let mut a = square_grid(3);
    let b = square_grid(3);
    let offset = a.pos.len();
    let mut positions = a.pos.clone();
    positions.extend(b.pos.iter().map(|p| [p[0] + 10.0, p[1]]));
    a.edges
        .extend(b.edges.iter().map(|&(u, v)| (u + offset, v + offset)));
    let crystal = crystal_from_graph(positions, a.edges);
    let (faces, boundary, diagnostics) = extract_interior_faces(&crystal, &crystal.pos, true);
    assert!(diagnostics.is_empty());
    assert_eq!(faces.len(), 8);
    assert_eq!(boundary.iter().filter(|&&b| b).count(), 16);
    assert_eq!(18_isize - 24 + faces.len() as isize, 2);
}

#[test]
fn sheet3d_review_display_ghosts_do_not_change_physics_or_original_crystal() {
    let (_, cell, _) = named_cell("square");
    let crystal = square_grid(4);
    let original = crystal.clone();
    let baseline = relax_sheet_3d(&crystal, &cell, &Sheet3DOptions::default(), None);
    let mut marked = crystal.clone();
    for (u, row) in marked.adj.iter_mut().enumerate() {
        for e in row {
            e.ghost_stub = Some(marked.pos[u]);
        }
    }
    let before = marked.clone();
    let with_marks = relax_sheet_3d(&marked, &cell, &Sheet3DOptions::default(), None);
    assert_eq!(baseline, with_marks);
    assert!(
        with_marks
            .bond_rest_len
            .iter()
            .all(|&l| (l - 1.0).abs() < 1e-12)
    );
    assert_eq!(crystal, original);
    assert_eq!(marked, before);

    let defect = build_point_defects(
        &cell,
        &[DefectSpec::disclination(90.0, [0.0, 0.0])],
        Some(3.5),
        None,
    )
    .unwrap();
    let before = defect.clone();
    let charge = total_charge(&defect).unwrap();
    let _ = relax_sheet_3d(
        &defect,
        &cell,
        &Sheet3DOptions {
            max_iters: 2,
            ..Default::default()
        },
        None,
    );
    assert_eq!(defect, before);
    assert_eq!(total_charge(&defect).unwrap(), charge);
}

#[test]
fn sheet3d_review_mobius_keeps_seam_constraints_and_reports_unsupported_faces() {
    let (group, cell, frame) = named_cell("square");
    let wall = DefectCatalogue::new(&group, &cell, frame)
        .build_wall("Wc1", Some(3.0))
        .unwrap();
    let original = wall.clone();
    let opts = Sheet3DOptions {
        max_iters: 4,
        ..Default::default()
    };
    let res = relax_sheet_3d(&wall, &cell, &opts, None);
    assert_eq!(res.xyz.len(), wall.pos.len());
    assert!(res.has_finite_geometry());
    let mut seams = 0;
    for (i, &(u, v)) in wall.edges.iter().enumerate() {
        let e = wall.adj[u].iter().find(|e| e.to == v).unwrap();
        if e.across {
            seams += 1;
            assert!(e.ghost_stub.is_some());
            assert!(res.bond_rest_len[i] > 0.0);
            assert!((res.bond_rest_len[i] - e.vec[0].hypot(e.vec[1])).abs() < 1e-12);
        }
    }
    assert!(seams > 0 && res.faces.is_empty());
    // Independently sum EVERY original bond, not just the ones the implementation chose to retain.
    let known_energy: f64 = wall
        .edges
        .iter()
        .enumerate()
        .map(|(i, &(u, v))| {
            let a = res.xyz[u];
            let b = res.xyz[v];
            let length = (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]);
            0.5 * (length - res.bond_rest_len[i]).powi(2)
        })
        .sum();
    assert!((res.bond_stretch_energy - known_energy).abs() < 1e-8);
    assert!(
        res.diagnostics
            .iter()
            .any(|s| s.contains("Möbius") && s.contains("not a physical"))
    );
    assert_eq!(wall, original);
}

#[test]
fn sheet3d_review_invalid_inputs_are_diagnosed_before_frame_walks() {
    let (_, cell, _) = named_cell("square");
    let crystal = square_grid(3);
    let opts = Sheet3DOptions::default();
    let assert_invalid =
        |c: &BuiltCrystal, cell: &Cell2D, o: &Sheet3DOptions, warm: Option<&[f64]>| {
            let r = relax_sheet_3d(c, cell, o, warm);
            assert!(r.xyz.is_empty() && !r.converged && !r.diagnostics.is_empty());
            assert!(
                r.has_finite_geometry(),
                "even a refusal must be safe to render"
            );
        };
    let mut warm = vec![0.01; crystal.pos.len()];
    warm[1] = f64::NAN;
    assert_invalid(&crystal, &cell, &opts, Some(&warm));
    warm[1] = f64::INFINITY;
    assert_invalid(&crystal, &cell, &opts, Some(&warm));
    assert_invalid(&crystal, &cell, &opts, Some(&[]));
    assert_invalid(
        &crystal,
        &cell,
        &Sheet3DOptions {
            kappa_b: f64::INFINITY,
            ..opts
        },
        None,
    );
    assert_invalid(
        &crystal,
        &cell,
        &Sheet3DOptions {
            grad_tol: f64::NAN,
            ..opts
        },
        None,
    );
    let mut invalid = crystal.clone();
    invalid.adj[0][0].to = usize::MAX;
    assert_invalid(&invalid, &cell, &opts, None);
    invalid = crystal.clone();
    invalid.adj.pop();
    assert_invalid(&invalid, &cell, &opts, None);
    invalid = crystal.clone();
    invalid.adj[1].retain(|e| e.to != 0);
    assert_invalid(&invalid, &cell, &opts, None);
    invalid = crystal.clone();
    invalid.edges.push(invalid.edges[0]);
    assert_invalid(&invalid, &cell, &opts, None);
    invalid = crystal.clone();
    invalid.pos[0][0] = f64::NAN;
    assert_invalid(&invalid, &cell, &opts, None);
    let mut invalid_cell = cell.clone();
    invalid_cell.basis[0][0] = f64::NAN;
    assert_invalid(&crystal, &invalid_cell, &opts, None);
    let zeros = vec![0.0; crystal.pos.len()];
    let r = relax_sheet_3d(&crystal, &cell, &opts, Some(&zeros));
    assert_eq!(r.iterations, 0);
    assert!(r.xyz.iter().all(|p| p[2] == 0.0));
}

#[test]
fn sheet3d_review_zero_budget_never_claims_convergence() {
    let (_, cell, _) = named_cell("square");
    let crystal = square_grid(3);
    let warm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let res = relax_sheet_3d(
        &crystal,
        &cell,
        &Sheet3DOptions {
            max_iters: 0,
            ..Default::default()
        },
        Some(&warm),
    );
    assert_eq!(res.iterations, 0);
    assert!(!res.converged && res.planar_2d_converged);
    assert!(res.diagnostics.iter().any(|s| s.contains("not converged")));
}

#[test]
fn sheet3d_review_raw_folded_dipole_unfolds_without_creasing() {
    let (_, cell, _) = named_cell("square");
    let raw_dipole = build_point_defects(
        &cell,
        &[
            DefectSpec::disclination(90.0, [0.0, 0.0]),
            DefectSpec::disclination(-90.0, [0.5, 0.5]),
        ],
        Some(8.0),
        None,
    )
    .unwrap();
    let res = relax_sheet_3d(&raw_dipole, &cell, &Sheet3DOptions::default(), None);
    assert!(res.has_finite_geometry());
    assert!(res.planar_2d_converged, "diagnostics: {:?}", res.diagnostics);
    assert!(
        res.stretch_energy < 0.03 && res.stretch_energy < 0.25 * res.planar_2d_stretch_energy,
        "folded surgical chart creased: E_stretch={:.5}, E_2d={:.5}",
        res.stretch_energy,
        res.planar_2d_stretch_energy
    );
    assert!(res.bend_energy < 0.05, "unexpected crease: E_bend={:.5}", res.bend_energy);
}
