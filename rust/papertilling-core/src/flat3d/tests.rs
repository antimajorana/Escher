use std::collections::BTreeSet;
use std::f64::consts::PI;

use crate::bieberbach::from_wallpaper_group;
use crate::plane::catalogue::parse_named_str;
use crate::space3d::build_analyzer_3d;
use crate::space3d::nets3d::CanonicalNet3d;
use crate::types::{Bond2D, Cell2D};

use super::lattice_symmetry::lattice_holohedry;
use super::*;

/// Published `H_1` of the ten platycosms (Conway-Rossetti 2003; Wolf, Spaces of Constant
/// Curvature, Sec. 3.5; theorem locator to verify), keyed by ITA number, in the crate's `Z_a^k x Z^r` format.
const PUBLISHED_H1: [(u16, &str); 13] = [
    (1, "Z^3"),
    (4, "Z_2^2 x Z"),
    (144, "Z_3 x Z"),
    (145, "Z_3 x Z"),
    (76, "Z_2 x Z"),
    (78, "Z_2 x Z"),
    (169, "Z"),
    (170, "Z"),
    (19, "Z_4^2"),
    (7, "Z_2 x Z^2"),
    (9, "Z^2"),
    (29, "Z_2^2 x Z"),
    (33, "Z_4 x Z"),
];

fn named_cell(key: &str) -> (Cell2D, [f64; 2]) {
    let cat = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
    let nl = cat.get(key).unwrap();
    (nl.cell.clone(), nl.frame.origin)
}

fn honeycomb_60_cell() -> Cell2D {
    let (hc120, _) = named_cell("honeycomb");
    build_supercell_2d(&hc120, [[1, 1], [0, 1]]).unwrap()
}

#[test]
fn torsion_free_2d_and_3d_exact_classification() {
    let rep2 = classify_all_2d_torsion_free().unwrap();
    let tf2: Vec<&str> = rep2.iter().filter(|r| r.is_torsion_free).map(|r| r.symbol.as_str()).collect();
    assert_eq!(tf2, vec!["p1", "pg"]);

    let rep3 = classify_all_3d_torsion_free().unwrap();
    let tf3: Vec<i32> = rep3.iter().filter(|r| r.is_torsion_free).map(|r| r.number).collect();
    assert_eq!(tf3, vec![1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170]);

    // P4_2 (#77) fails via (4_2)^2 = 2 rotation fixing [0, 0, 0]; P2_1/c (#14) is non-split yet has inversion fixed points.
    let sg77 = build_analyzer_3d(77).unwrap();
    let r77 = analyze_torsion_free(&sg77);
    assert!(!r77.is_torsion_free);
    assert!(!is_torsion_free_exact(&sg77));
    assert!(r77.obstructions.iter().any(|o| o.power_k == 2 || o.order == 2));

    // H_1 = G^ab of each Bieberbach group, recomputed from the ITA operations, matches the literature.
    for (num, h1) in PUBLISHED_H1 {
        let rep = analyze_torsion_free(&build_analyzer_3d(num).unwrap());
        assert!(rep.is_torsion_free, "#{num}");
        assert_eq!(rep.ab_full_label, h1, "H_1 of ITA #{num}");
        let p = lookup_platycosm(&num.to_string()).unwrap();
        assert_eq!(p.h1_label, h1, "table H_1 of {}", p.code);
    }

    let sg14 = build_analyzer_3d(14).unwrap();
    let r14 = analyze_torsion_free(&sg14);
    assert!(!r14.is_split && !r14.is_torsion_free);
    assert!(from_wallpaper_group("p2").is_ok_and(|sg| !is_torsion_free_exact(&sg)));
}

#[test]
fn platycosm_catalogue_quotients_and_obstructions() {
    assert_eq!(list_platycosms().len(), 10);
    let mut itas: Vec<u16> = list_platycosms().iter().flat_map(|p| p.ita_numbers.iter().copied()).collect();
    itas.sort_unstable();
    assert_eq!(itas, vec![1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170]);

    let c2 = lookup_platycosm("c2").unwrap();
    let pa2 = lookup_platycosm("+a2").unwrap();
    assert_eq!(c2.h1_label, pa2.h1_label);
    assert!(c2.orientable && !pa2.orientable);
    assert_eq!(lookup_platycosm("c22").unwrap().b1, 0);
    // Wolf's names: G5 = hexacosm (c6), G6 = Hantzsche-Wendt didicosm (c22), B4 = -a2 (Pna2_1).
    assert_eq!(lookup_platycosm("G5").unwrap().code, "c6");
    assert_eq!(lookup_platycosm("G6").unwrap().code, "c22");
    assert_eq!(lookup_platycosm("b4").unwrap().ita_numbers, &[33]);
    assert_eq!(list_platycosms().iter().filter(|p| p.orientable).count(), 6);

    let sc = Cell3D::from_canonical_net(CanonicalNet3d::SimpleCubic);
    for code in ["c1", "c2", "c4", "c22", "+a1", "-a1", "+a2", "-a2"] {
        let q = build_platycosm_quotient(&sc, code, None, false).unwrap();
        assert_eq!(q.status, "OK", "{code}: {}", q.reason);
        assert_eq!(q.euler_chi_3, 0, "{code} chi_3 != 0");
        assert_eq!(q.cw_v as i64 - q.cw_e as i64 + q.cw_f as i64 - q.cw_c as i64, 0);
    }
    let q_sc_c4 = build_platycosm_quotient(&sc, "c4", Some([2, 2, 4]), false).unwrap();
    assert_eq!(q_sc_c4.status, "OK");
    assert_eq!(q_sc_c4.vertices.len(), 4);
    assert_eq!(q_sc_c4.bonds.len(), 12);

    // Hexagonal nets on c3 & c6 succeed; on c4 fail with a [PROVED] holohedry obstruction
    // (hexagonal holohedry 6/mmm has no 4-fold axis).
    for net in [CanonicalNet3d::HCP, CanonicalNet3d::BetaQuartz] {
        let c = Cell3D::from_canonical_net(net);
        for code in ["c3", "c6"] {
            let q = build_platycosm_quotient(&c, code, None, false).unwrap();
            assert_eq!(q.status, "OK", "{net:?} on {code}: {}", q.reason);
            assert_eq!(q.euler_chi_3, 0);
        }
        let obs = check_platycosm_compatibility(&c, "c4", None, false).unwrap();
        assert_eq!(obs.status, "IMPOSSIBLE");
        assert!(obs.reason.contains("IMPOSSIBLE [PROVED]") && obs.reason.contains("4-fold"), "{}", obs.reason);
    }

    // Rutile (#136) embeds c4 at (1, 1, 2) via a shifted 4_1 axis; Diamond embeds c22 at (1, 1, 1).
    let rutile = Cell3D::from_canonical_net(CanonicalNet3d::Rutile136);
    assert_eq!(build_platycosm_quotient(&rutile, "c4", Some([1, 1, 2]), false).unwrap().status, "OK");
    let diamond = Cell3D::from_canonical_net(CanonicalNet3d::Diamond);
    assert_eq!(build_platycosm_quotient(&diamond, "c22", Some([1, 1, 1]), false).unwrap().status, "OK");
    // Uncoloured diamond (Fd-3m) has 4_1 screws with axial translation c/4 + (c/2)Z: c4 embeds with
    // c' = c (8 sites / 4 = 2 vertices, 16 bonds / 4 = 4) but not with c' = 4c (needs translation c'/4 = c).
    let q_dia = build_platycosm_quotient(&diamond, "c4", Some([1, 1, 1]), true).unwrap();
    assert_eq!((q_dia.status.as_str(), q_dia.vertices.len(), q_dia.bonds.len()), ("OK", 2, 4));
    assert_eq!(build_platycosm_quotient(&diamond, "c4", Some([1, 1, 4]), true).unwrap().status, "NOT_FOUND");

    // SimpleCubic on c6 is IMPOSSIBLE [PROVED] (cubic holohedry m-3m has no 6-fold axis). SimpleCubic on
    // c3 is NOT impossible: (R_[111] | (1,0,0)) is a 3_1 screw of Pm-3m since R x R^2 sums (1,0,0) to
    // (1,1,1); the standard-setting search misses it, so the refusal is only [COMPUTATIONAL EVIDENCE].
    let sc_c6 = check_platycosm_compatibility(&sc, "c6", None, false).unwrap();
    assert_eq!(sc_c6.status, "IMPOSSIBLE");
    assert!(sc_c6.reason.contains("IMPOSSIBLE [PROVED]"));
    let sc_c3 = check_platycosm_compatibility(&sc, "c3", None, false).unwrap();
    assert_eq!(sc_c3.status, "NOT_FOUND");
    assert!(sc_c3.reason.contains("[COMPUTATIONAL EVIDENCE]") && !sc_c3.compatible);
    let tri = Cell3D::from_fractional([[1.0, 0.0, 0.0], [0.3, 1.1, 0.0], [0.2, 0.4, 1.2]], &[([0.0, 0.0, 0.0], 0)], vec![], None);
    assert!(check_platycosm_compatibility(&tri, "c22", None, false).unwrap().reason.contains("IMPOSSIBLE [PROVED]"));
}

#[test]
fn flat_3d_slabs_and_chimneys() {
    let sc = Cell3D::from_canonical_net(CanonicalNet3d::SimpleCubic);
    let torus = build_flat_slab_3d(&sc, FlatSlabKind::TorusSlab, [3, 3, 3]).unwrap();
    assert_eq!(torus.status, "OK");
    assert_eq!(torus.positions.len(), 27);
    assert_eq!(torus.boundary_sites.len(), 18);
    assert!(torus.orientable && torus.b1 == 2);

    let klein = build_flat_slab_3d(&sc, FlatSlabKind::KleinSlab, [4, 4, 3]).unwrap();
    assert_eq!(klein.status, "OK");
    assert_eq!(klein.positions.len(), 48);
    assert_eq!(klein.boundary_sites.len(), 32);
    assert!(!klein.orientable && klein.b1 == 1 && klein.h1_label == "Z_2 x Z");
    assert!(klein.bonds.iter().any(|b| b.2));

    let mob = build_flat_slab_3d(&sc, FlatSlabKind::MoebiusTube, [4, 3, 3]).unwrap();
    assert_eq!(mob.status, "OK");
    assert_eq!(mob.positions.len(), 36);
    assert_eq!(mob.boundary_sites.len(), 24);
    assert!(!mob.orientable && mob.b1 == 2);

    let chimney = build_flat_slab_3d(&sc, FlatSlabKind::CylinderChimney, [6, 3, 2]).unwrap();
    assert_eq!((chimney.manifold_dim, chimney.compact_dims, chimney.noncompact_dims), (3, 1, 2));
    assert_eq!(chimney.positions.len(), 36);
    assert_eq!(chimney.b1, 1);

    let cyl_s1 = build_flat_slab_3d(&sc, FlatSlabKind::CylinderCrossS1, [4, 3, 4]).unwrap();
    assert_eq!(cyl_s1.status, "OK");
    assert_eq!(cyl_s1.positions.len(), 48);

    let bq = Cell3D::from_canonical_net(CanonicalNet3d::BetaQuartz);
    let bq_klein = build_flat_slab_3d(&bq, FlatSlabKind::KleinSlab, [3, 3, 2]).unwrap();
    assert_eq!(bq_klein.status, "IMPOSSIBLE");
    assert!(bq_klein.reason.contains("IMPOSSIBLE [PROVED]"));
}

#[test]
fn chiral_nanotubes_and_nanoribbons_exact_counts() {
    let (hc120, _) = named_cell("honeycomb");
    let hc60 = honeycomb_60_cell();

    // Verify exact 4(n^2+nm+m^2)/gcd(2n+m, 2m+n) formula on both 60° and converted 120° cells.
    for (n, m, expected_atoms) in [(5, 5, 20), (6, 0, 24), (4, 2, 56)] {
        let nt_conv = build_nanotube_opts(&hc120, n, m, 1, true, true).unwrap();
        assert_eq!(nt_conv.atoms_per_unit, expected_atoms, "120->60 ({n},{m})");
        assert!(nt_conv.coordination_numbers.iter().all(|&d| d == 3), "({n},{m}) 3-coord");

        let nt_direct = build_nanotube_opts(&hc60, n, m, 1, true, false).unwrap();
        assert_eq!(nt_direct.atoms_per_unit, expected_atoms, "direct 60 ({n},{m})");
        assert!(nt_direct.coordination_numbers.iter().all(|&d| d == 3));
    }

    // Open-ended 2-period (4, 2) tube has 112 atoms and seam bonds.
    let open_tube = build_nanotube(&hc60, 4, 2, 2).unwrap();
    assert_eq!(open_tube.positions_3d.len(), 112);
    assert!(open_tube.bonds.iter().any(|b| b.2));

    // Direct 120° basis uses the actual metric tensor cleanly (4, 2 -> 8 atoms; 6, 0 -> 24 atoms).
    assert_eq!(build_nanotube(&hc120, 6, 0, 1).unwrap().atoms_per_unit, 24);
    assert_eq!(build_nanotube(&hc120, 4, 2, 1).unwrap().atoms_per_unit, 8);

    // Irrational metric ratio refuses diagonal (1, 1) with [PROVED] message.
    let g76 = 76.0_f64.to_radians();
    let irr = Cell2D {
        a1: [1.0, 0.0],
        a2: [1.23 * g76.cos(), 1.23 * g76.sin()],
        basis: vec![[0.0, 0.0]],
        bonds: vec![Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 }],
    };
    let err = build_nanotube(&irr, 1, 1, 1).unwrap_err();
    assert!(err.contains("REFUSED [COMPUTATIONAL EVIDENCE]"), "{err}");

    // Saito-Dresselhaus-Dresselhaus (1998, Ch. 3): |Ch| = a sqrt(n^2+nm+m^2) with a = sqrt(3) for unit
    // C-C bonds, chiral angle cos(theta) = (2n+m) / (2 sqrt(n^2+nm+m^2)): (5,5) armchair 30 deg,
    // (6,0) zigzag 0 deg, (4,2): acos(10 / (2 sqrt 28)) = 19.1066 deg.
    for (n, m, theta) in [(5, 5, 30.0_f64), (6, 0, 0.0), (4, 2, (10.0 / (2.0 * 28.0_f64.sqrt())).acos().to_degrees())] {
        let t = build_nanotube(&hc60, n, m, 1).unwrap();
        let q = f64::from(n * n + n * m + m * m);
        assert!((t.circumference - 3.0_f64.sqrt() * q.sqrt()).abs() < 1e-9, "({n},{m}) |Ch|");
        assert!((2.0 * t.radius - t.circumference / PI).abs() < 1e-9, "({n},{m}) diameter");
        assert!((t.chiral_angle_deg - theta).abs() < 1e-4, "({n},{m}) chiral angle");
    }

    let rib = build_nanoribbon(&hc60, [1, 0], 3, 4, true).unwrap();
    assert_eq!(rib.positions_2d.len(), 24);
    assert!(!rib.edge_sites.is_empty());
    assert!(rib.bonds.iter().any(|b| b.2));
}

#[test]
fn symmetry_centred_flakes_exact_invariance() {
    let (hc, cen) = named_cell("honeycomb");
    // Hexagon center 1a = [sqrt(3)/2, 0.5]: R1=1.1 -> 6-ring, R2=2.7 -> 24-site coronene!
    for poly in [None, Some(6)] {
        let f1 = build_symmetric_flake("p6m", &hc, cen, poly, 1.1).unwrap();
        assert_eq!(f1.crystal.pos.len(), 6);
        assert_eq!(f1.crystal.edges.len(), 6);
        assert_eq!(f1.point_group_order, 12);
        assert!(f1.is_dihedral);
        assert_eq!(f1.site_orbits.len(), 1);

        let f2 = build_symmetric_flake("p6m", &hc, cen, poly, 2.7).unwrap();
        assert_eq!(f2.crystal.pos.len(), 24);
        assert_eq!(f2.crystal.edges.len(), 30);
        assert_eq!(f2.point_group_order, 12);
        assert_eq!(f2.site_orbits.len(), 3);

        let edge_set: BTreeSet<(usize, usize)> = f2.crystal.edges.iter().copied().collect();
        for perm in &f2.vertex_perms {
            for &(u, v) in &f2.crystal.edges {
                let (pu, pv) = (perm[u], perm[v]);
                assert!(edge_set.contains(&(pu.min(pv), pu.max(pv))));
            }
        }
    }

    let ws = build_wigner_seitz_flake("p6m", &hc, cen, 1.8).unwrap();
    assert_eq!(ws.point_group_order, 12);
    assert!(!ws.crystal.pos.is_empty());
}

#[test]
fn niggli_3d_proper_det_supercells_strain_and_bcs() {
    let fcc = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
    let bcc = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]];
    for cell in [fcc, bcc, [[1.2, 0.3, -0.4], [0.0, 1.5, 0.5], [0.0, 0.0, 1.8]]] {
        // Apply unimodular shear (det = +1 and det = -1 check via row combination).
        let skewed = [
            [cell[0][0] + 2.0 * cell[1][0] - cell[2][0], cell[0][1] + 2.0 * cell[1][1] - cell[2][1], cell[0][2] + 2.0 * cell[1][2] - cell[2][2]],
            [cell[1][0] - 3.0 * cell[2][0], cell[1][1] - 3.0 * cell[2][1], cell[1][2] - 3.0 * cell[2][2]],
            cell[2],
        ];
        let red = niggli_reduce_3d(skewed).unwrap();
        let det_p = red.p_mat[0][0] * (red.p_mat[1][1] * red.p_mat[2][2] - red.p_mat[1][2] * red.p_mat[2][1])
            - red.p_mat[0][1] * (red.p_mat[1][0] * red.p_mat[2][2] - red.p_mat[1][2] * red.p_mat[2][0])
            + red.p_mat[0][2] * (red.p_mat[1][0] * red.p_mat[2][1] - red.p_mat[1][1] * red.p_mat[2][0]);
        assert_eq!(det_p, 1);
        let idem = niggli_reduce_3d(red.reduced).unwrap();
        assert_eq!(idem.p_mat, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }
    // Hand-checked type-II Niggli cell with xi = eta = 0 > zeta: A = 1 <= B = 1.53 <= C = 2.25,
    // |zeta| = 0.6 <= A, A + B + xi + eta + zeta > 0; Krivy-Gruber A4 must leave it unchanged.
    let typ2 = [[1.0, 0.0, 0.0], [-0.3, 1.2, 0.0], [0.0, 0.0, 1.5]];
    let r2 = niggli_reduce_3d(typ2).unwrap();
    assert_eq!(r2.p_mat, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    assert_eq!(r2.niggli_type, 2);

    // Holohedry orders (ITA Vol. A): cubic m-3m = 48, hexagonal 6/mmm = 24, triclinic -1 = 2.
    let hc = lattice_holohedry(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    assert_eq!((hc.order, hc.proper_orders.iter().copied().collect::<Vec<_>>(), hc.has_222), (48, vec![1, 2, 3, 4], true));
    let hx = lattice_holohedry(&[[1.0, 0.0, 0.0], [-0.5, 0.75_f64.sqrt(), 0.0], [0.0, 0.0, 1.35]]).unwrap();
    assert_eq!((hx.order, hx.proper_orders.iter().copied().collect::<Vec<_>>(), hx.has_222), (24, vec![1, 2, 3, 6], true));
    let tr = lattice_holohedry(&[[1.0, 0.0, 0.0], [0.3, 1.1, 0.0], [0.2, 0.4, 1.2]]).unwrap();
    assert_eq!((tr.order, tr.proper_orders.len(), tr.has_222), (2, 1, false));

    let sc = Cell3D::from_canonical_net(CanonicalNet3d::SimpleCubic);
    let sup3 = build_supercell_3d(&sc, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]).unwrap();
    assert_eq!(sup3.basis.len(), 8);
    assert_eq!(sup3.bonds.len(), 24);

    let th = 0.4_f64;
    let rot_z = [[th.cos(), th.sin(), 0.0], [-th.sin(), th.cos(), 0.0], [0.0, 0.0, 1.0]];
    let st_rot = cell_strain_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rot_z).unwrap();
    assert!(st_rot.green_lagrange.iter().flatten().all(|&x| x.abs() < 1e-12));
    let st_ax = cell_strain_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    assert!((st_ax.infinitesimal_strain[0][0] - 0.1).abs() < 1e-12);
    assert!((st_ax.green_lagrange[0][0] - 0.105).abs() < 1e-12);
    // Von Mises sqrt(2/3 e':e') of E = diag(0.105, 0, 0): e' = 0.035 * diag(2, -1, -1) -> 0.07.
    assert!((st_ax.von_mises_equivalent - 0.07).abs() < 1e-12);

    let bc_3d = apply_boundary_conditions_3d(&sc, [2, 2, 2], [BoundaryCondition::Periodic, BoundaryCondition::Antiperiodic, BoundaryCondition::Open]).unwrap();
    assert_eq!(bc_3d.positions.len(), 8);
    assert!(bc_3d.bonds.iter().any(|b| (b.phase_rad.abs() - PI).abs() < 1e-9 && (b.weight_real + 1.0).abs() < 1e-9));
    // Open along z leaves 4 z-bonds instead of 8!
    assert_eq!(bc_3d.bonds.len(), 20);
}
