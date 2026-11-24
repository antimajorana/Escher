//! Stage 5 tests. Expected values are published facts or hand counts (sources in comments), never
//! values copied from the code under test.

use std::time::Instant;

use crate::dsymbol::GeometryKind;

use super::*;
use crate::hyperbolic::inflation::hyp_dist;

fn chord3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mink_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    hyp_dist(a, b)
}

/// `max - min` of the edge lengths under `dist`.
fn edge_length_spread(pts: &[[f64; 3]], edges: &[(usize, usize)], dist: fn([f64; 3], [f64; 3]) -> f64) -> f64 {
    let lens: Vec<f64> = edges.iter().map(|&(u, v)| dist(pts[u], pts[v])).collect();
    let lo = lens.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = lens.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

#[test]
fn unified_pq_spherical_platonic_hosohedra_and_dihedra() {
    assert!(delta_pq(1, 4).is_err());
    assert!(delta_pq(4, 1).is_err());

    // Coxeter, Regular Polytopes (3rd ed. 1973), Table I(i): (p, q, V, E, F).
    let platonic = [
        (3, 3, 4, 6, 4),
        (4, 3, 8, 12, 6),
        (3, 4, 6, 12, 8),
        (5, 3, 20, 30, 12),
        (3, 5, 12, 30, 20),
    ];
    for (p, q, ev, ee, ef) in platonic {
        let t = build_unified_pq(p, q, 1).unwrap();
        assert_eq!(t.geometry_kind, GeometryKind::Spherical);
        assert_eq!((t.n_vertices, t.n_edges, t.n_faces, t.euler_chi), (ev, ee, ef, 2));
        assert!(t.coords_3d.iter().all(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0).abs() < 1e-11));
        assert!(t.coords_2d.iter().all(|z| z[0].is_finite() && z[1].is_finite()));
        assert!(t.faces.iter().all(|f| f.len() == p));
    }

    for q in 2..=6 {
        let h = build_unified_pq(2, q, 1).unwrap();
        assert_eq!(h.geometry_kind, GeometryKind::Spherical);
        assert_eq!((h.n_vertices, h.n_edges, h.n_faces, h.euler_chi), (2, q, q, 2));
        assert_eq!(h.edge_arcs_3d.len(), q);
        // Verify distinct meridians are sampled explicitly rather than collapsed onto a chord.
        assert!(h.edge_arcs_3d[0].len() > 2);
        let mid0 = h.edge_arcs_3d[0][h.edge_arcs_3d[0].len() / 2];
        let mid1 = h.edge_arcs_3d[1][h.edge_arcs_3d[1].len() / 2];
        let sep = ((mid0[0] - mid1[0]).powi(2) + (mid0[1] - mid1[1]).powi(2) + (mid0[2] - mid1[2]).powi(2)).sqrt();
        assert!(sep > 0.5);
    }

    for p in 3..=6 {
        let d = build_unified_pq(p, 2, 1).unwrap();
        assert_eq!(d.geometry_kind, GeometryKind::Spherical);
        assert_eq!((d.n_vertices, d.n_edges, d.n_faces, d.euler_chi), (p, p, 2, 2));
        assert!(d.faces.iter().all(|f| f.len() == p));
    }
}

#[test]
fn unified_pq_euclidean_unit_bonds_and_hyperbolic() {
    // Hand counts of vertex-completion shells: {4,4} (2n+1)^2 squares; {6,3} centred hexagonal numbers
    // 3n(n+1)+1; {3,6}: 13 = 3 vertex stars of 6 minus overlaps, then 50 - 13 = 37 by Euler on a disk
    // (12 vertices, each in 6 triangles).
    for (p, q, f1, f2) in [(4, 4, 9, 25), (6, 3, 7, 19), (3, 6, 13, 37)] {
        let t1 = build_unified_pq(p, q, 1).unwrap();
        let t2 = build_unified_pq(p, q, 2).unwrap();
        assert_eq!(t1.geometry_kind, GeometryKind::Euclidean);
        assert_eq!(t1.n_faces, f1);
        assert_eq!(t2.n_faces, f2);
        for &(u, v) in &t2.edges {
            let len = (t2.coords_2d[u][0] - t2.coords_2d[v][0]).hypot(t2.coords_2d[u][1] - t2.coords_2d[v][1]);
            assert!((len - 1.0).abs() < 1e-11);
        }
    }
    // {7,3} vertex-completion shells 1, 7, 21 (cumulative 29): 7 edge-neighbours of the heptagon,
    // then 21 = 7 * 3 faces around the second ring (BDF vertex completion, hand count).
    let h73 = build_unified_pq(7, 3, 2).unwrap();
    assert_eq!(h73.geometry_kind, GeometryKind::Hyperbolic);
    assert_eq!(h73.n_faces, 29);
    assert!(h73.curvature_kappa < 0.0);
}

#[test]
fn conway_orbifolds_17_wallpaper_and_bad_rejections() {
    // chi = 2 - cost (Conway et al., The Symmetries of Things, 2008): *532 -> 1/60 (|G| = 120 = 2/chi),
    // 532 -> 1/30, *432 -> 1/24, *332 -> 1/12, *2.2.10 -> 1/20, 2.2.10 -> 1/10, *237 -> -1/84.
    assert_eq!(wallpaper_orbifold_table().len(), 17);
    for &(wg, orb) in wallpaper_orbifold_table() {
        assert_eq!(wallpaper_to_orbifold(wg).unwrap(), orb);
        assert_eq!(orbifold_to_wallpaper(orb).unwrap(), wg);
        let parsed = parse_orbifold(orb).unwrap();
        assert_eq!(parsed.chi_num, 0);
        assert_eq!(parsed.geometry_kind, GeometryKind::Euclidean);
    }

    assert_eq!((parse_orbifold("*532").unwrap().chi_num, parse_orbifold("*532").unwrap().chi_den), (1, 60));
    assert_eq!((parse_orbifold("532").unwrap().chi_num, parse_orbifold("532").unwrap().chi_den), (1, 30));
    assert_eq!((parse_orbifold("*432").unwrap().chi_num, parse_orbifold("*432").unwrap().chi_den), (1, 24));
    assert_eq!((parse_orbifold("*332").unwrap().chi_num, parse_orbifold("*332").unwrap().chi_den), (1, 12));
    assert_eq!((parse_orbifold("*2.2.10").unwrap().chi_num, parse_orbifold("*2.2.10").unwrap().chi_den), (1, 20));
    assert_eq!((parse_orbifold("2 2 10").unwrap().chi_num, parse_orbifold("2 2 10").unwrap().chi_den), (1, 10));
    assert_eq!((parse_orbifold("*(12)(12)").unwrap().chi_num, parse_orbifold("*(12)(12)").unwrap().chi_den), (1, 12));
    assert_eq!((parse_orbifold("(12)(12)").unwrap().chi_num, parse_orbifold("(12)(12)").unwrap().chi_den), (1, 6));
    assert_eq!((parse_orbifold("*237").unwrap().chi_num, parse_orbifold("*237").unwrap().chi_den), (-1, 84));
    assert_eq!((parse_orbifold("*(2,3,7)").unwrap().chi_num, parse_orbifold("*(2,3,7)").unwrap().chi_den), (-1, 84));
    assert_eq!((parse_orbifold("237").unwrap().chi_num, parse_orbifold("237").unwrap().chi_den), (-1, 42));

    for bad in ["3", "*3", "35", "*35", "(10)", "*(10)", "2 7", "*2.7"] {
        let err = parse_orbifold(bad).unwrap_err();
        assert!(err.contains("[PROVED: bad orbifold — no manifold cover]"), "{bad}: {err}");
    }
}

#[test]
fn conway_orbifold_boundary_cycles_orientation_and_i128_bounds() {
    assert_eq!(parse_orbifold("o*234 x").unwrap().canonical, "*432xxx");
    assert_eq!(parse_orbifold("*( 12 )(12)").unwrap().canonical, "*(12,12)");
    // Two decorated mirror boundaries vs one decorated + one empty:
    // chi(*23*45) = 2 - 2 - (1/4 + 1/3 + 3/8 + 2/5) = -163/120 (hand calculation).
    for sym in ["*23*45", "*45*23", "*32*54"] {
        let p = parse_orbifold(sym).unwrap();
        assert_eq!(p.canonical, "*54*32", "{sym}");
        assert_eq!((p.boundary_components, p.chi_num, p.chi_den), (2, -163, 120), "{sym}");
        assert_eq!((p.geometry_kind, p.wallpaper_group.as_deref()), (GeometryKind::Hyperbolic, None));
    }
    for sym in ["*5432*", "**2345"] {
        let p = parse_orbifold(sym).unwrap();
        assert_eq!(p.canonical, "*5432*", "{sym}");
        assert_eq!((p.boundary_components, p.chi_num, p.chi_den), (2, -163, 120), "{sym}");
    }
    // Two 1-corner boundaries must not match wallpaper pm ("**"); chi(*2*2) = 2 - 2 - 2*(1/4) = -1/2.
    let two_corners = parse_orbifold("*2*2").unwrap();
    assert_eq!((two_corners.canonical.as_str(), two_corners.wallpaper_group.as_deref()), ("*2*2", None));
    assert_eq!((two_corners.chi_num, two_corners.chi_den), (-1, 2));

    // Orientable base (crosscaps == 0): global reflection reverses all boundary cycles simultaneously,
    // distinguishing concordant *234*234 (~ *432*432) from mirror-paired *234*432 (~ *432*234);
    // chi = 2 - 2 - 2*(1/4 + 1/3 + 3/8) = -23/12.
    let conc_a = parse_orbifold("*234*234").unwrap();
    let conc_b = parse_orbifold("*432*432").unwrap();
    let opp_a = parse_orbifold("*234*432").unwrap();
    let opp_b = parse_orbifold("*432*234").unwrap();
    assert_eq!((conc_a.canonical.as_str(), conc_b.canonical.as_str()), ("*432*432", "*432*432"));
    assert_eq!((opp_a.canonical.as_str(), opp_b.canonical.as_str()), ("*432*423", "*432*423"));
    assert_ne!(conc_a.canonical, opp_a.canonical);
    assert_eq!((conc_a.chi_num, conc_a.chi_den), (-23, 12));
    assert_eq!(parse_orbifold("o*234*234").unwrap().canonical, "o*432*432");
    assert_eq!(parse_orbifold("o*234*432").unwrap().canonical, "o*432*423");

    // Nonorientable base (crosscaps > 0): sliding one boundary around a crosscap reverses it alone;
    // chi(*234*234x) = -1 - 23/12 = -35/12.
    let nonor_a = parse_orbifold("*234*234x").unwrap();
    let nonor_b = parse_orbifold("*234*432x").unwrap();
    assert_eq!((nonor_a.canonical.as_str(), nonor_b.canonical.as_str()), ("*432*432x", "*432*432x"));
    assert_eq!((nonor_a.chi_num, nonor_a.chi_den), (-35, 12));

    // Large valid u32 orders whose unreduced denominator product (> 3.40e38) overflows i128,
    // while the exact reduced fraction has den = 4294967291 * 4294967279 * 4294967231 <= i128::MAX.
    let fit = parse_orbifold("4294967291,4294967279,4294967231,4294967291").unwrap();
    assert_eq!(
        (fit.chi_num, fit.chi_den, fit.geometry_kind),
        (-158_456_321_745_008_244_180_050_234_010_i128, 79_228_160_909_397_609_687_688_407_659_i128, GeometryKind::Hyperbolic)
    );

    // Four distinct primes near 2^32: reduced denominator ≈ 3.4028e38 > i128::MAX -> representability error.
    let over = parse_orbifold("4294967291,4294967279,4294967231,4294967197").unwrap_err();
    assert!(over.contains("i128"), "{over}");
}

#[test]
fn wythoff_archimedean_and_vertex_config_obstructions() {
    // Archimedean counts: Coxeter, Longuet-Higgins & Miller (1954); Cromwell, Polyhedra (1997).
    let trunc_ico = build_wythoff(3, 5, WythoffOp::Truncated, 1).unwrap();
    assert_eq!(trunc_ico.vertex_symbol, "5.6.6");
    assert_eq!((trunc_ico.n_vertices, trunc_ico.n_edges, trunc_ico.n_faces), (60, 90, 32));
    assert_eq!(trunc_ico.face_type_counts[&5], 12);
    assert_eq!(trunc_ico.face_type_counts[&6], 20);

    let bitrunc_dod = build_wythoff(5, 3, WythoffOp::Bitruncated, 1).unwrap();
    assert_eq!(bitrunc_dod.vertex_symbol, "5.6.6");
    assert_eq!((bitrunc_dod.n_vertices, bitrunc_dod.n_edges, bitrunc_dod.n_faces), (60, 90, 32));

    let cuboct = build_wythoff(4, 3, WythoffOp::Rectified, 1).unwrap();
    assert_eq!(cuboct.vertex_symbol, "3.4.3.4");
    assert_eq!((cuboct.n_vertices, cuboct.n_edges, cuboct.n_faces), (12, 24, 14));

    let rhombicub = build_wythoff(4, 3, WythoffOp::Cantellated, 1).unwrap();
    assert_eq!(rhombicub.vertex_symbol, "3.4.4.4");
    assert_eq!((rhombicub.n_vertices, rhombicub.n_edges, rhombicub.n_faces), (24, 48, 26));

    // Rhombicosidodecahedron 3.4.5.4: V 60, E 120, F 62 = 20 triangles + 30 squares + 12 pentagons.
    let rid = build_wythoff(5, 3, WythoffOp::Cantellated, 1).unwrap();
    assert_eq!(rid.vertex_symbol, "3.4.5.4");
    assert_eq!((rid.n_vertices, rid.n_edges, rid.n_faces), (60, 120, 62));
    assert_eq!((rid.face_type_counts[&3], rid.face_type_counts[&4], rid.face_type_counts[&5]), (20, 30, 12));
    // Truncated icosidodecahedron 4.6.10: V 120, E 180, F 62.
    let grid = build_wythoff(3, 5, WythoffOp::Omnitruncated, 1).unwrap();
    assert_eq!((grid.n_vertices, grid.n_edges, grid.n_faces), (120, 180, 62));
    // Snub dodecahedron 3^4.5: V 60, E 150, F 92 = 80 triangles + 12 pentagons.
    let snub_d = build_wythoff(5, 3, WythoffOp::Snub { chirality: SnubHandedness::Left }, 1).unwrap();
    assert_eq!((snub_d.n_vertices, snub_d.n_edges, snub_d.n_faces), (60, 150, 92));
    assert!(snub_d.provenance.contains("[COMPUTATIONAL EVIDENCE]"));

    let snub_l = build_wythoff(4, 3, WythoffOp::Snub { chirality: SnubHandedness::Left }, 1).unwrap();
    let snub_r = build_wythoff(4, 3, WythoffOp::Snub { chirality: SnubHandedness::Right }, 1).unwrap();
    assert_eq!(snub_l.vertex_symbol, "3.3.3.3.4");
    assert_eq!((snub_l.n_vertices, snub_l.n_edges, snub_l.n_faces), (24, 60, 38));
    assert_eq!((snub_r.n_vertices, snub_r.n_edges, snub_r.n_faces), (24, 60, 38));
    assert_ne!(snub_l.coords_3d, snub_r.coords_3d);
    // Uniform polyhedra have all edges equal (definition); chords on the unit sphere.
    for t in [&snub_l, &snub_d, &rid] {
        assert!(edge_length_spread(&t.coords_3d, &t.edges, chord3) < 1e-6, "{}", t.vertex_symbol);
    }

    let prism5 = build_wythoff(2, 5, WythoffOp::Truncated, 1).unwrap();
    assert_eq!((prism5.n_vertices, prism5.n_edges, prism5.n_faces), (10, 15, 7));
    let antiprism5 = build_wythoff(2, 5, WythoffOp::Snub { chirality: SnubHandedness::Left }, 1).unwrap();
    assert_eq!((antiprism5.n_vertices, antiprism5.n_edges, antiprism5.n_faces), (10, 20, 12));
    assert!(edge_length_spread(&prism5.coords_3d, &prism5.edges, chord3) < 1e-9);
    for hand in [SnubHandedness::Left, SnubHandedness::Right] {
        let ap = build_wythoff(2, 5, WythoffOp::Snub { chirality: hand }, 1).unwrap();
        assert!(edge_length_spread(&ap.coords_3d, &ap.edges, chord3) < 1e-9, "{hand:?}");
    }
    // k = 3 antiprism is the octahedron: rings at z = +-1/sqrt(3).
    let oct = build_wythoff(2, 3, WythoffOp::Snub { chirality: SnubHandedness::Left }, 1).unwrap();
    assert!(oct.coords_3d.iter().all(|v| (v[2].abs() - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12));

    // H^2 snub sr{7,3} = 3^4.7: all edges of the patch have one hyperbolic length.
    let h_snub = build_wythoff(7, 3, WythoffOp::Snub { chirality: SnubHandedness::Left }, 1).unwrap();
    assert!(h_snub.faces.len() > 10);
    assert!(edge_length_spread(&h_snub.coords_3d, &h_snub.edges, mink_dist) < 1e-6);
    let h_trunc = build_wythoff(7, 3, WythoffOp::Truncated, 1).unwrap();
    assert!(edge_length_spread(&h_trunc.coords_3d, &h_trunc.edges, mink_dist) < 1e-6);

    let vc_e2 = parse_vertex_config("6.4.3.4").unwrap();
    assert_eq!(vc_e2.canonical, "3.4.6.4");
    assert_eq!(vc_e2.geometry_kind, GeometryKind::Euclidean);
    assert!(vc_e2.angle_defect_rad.abs() < 1e-12);
    assert!(!vc_e2.is_chiral);

    let vc_chiral = parse_vertex_config("3.3.4.3.5").unwrap();
    assert!(vc_chiral.is_chiral);
    let vc_snub = parse_vertex_config("3.3.3.3.4").unwrap();
    assert!(vc_snub.is_snub_chiral);
    // Snub square tiling 3^2.4.3.4 is achiral (wallpaper group p4g, which contains mirrors).
    let snub_sq = parse_vertex_config("3.3.4.3.4").unwrap();
    assert!(!snub_sq.is_snub_chiral);
    assert_eq!(snub_sq.geometry_kind, GeometryKind::Euclidean);
    // Rhombicosidodecahedron vertex 3.4.5.4: Delta = pi/15, V = 4 pi / Delta = 60.
    let vc_rid = parse_vertex_config("4.5.4.3").unwrap();
    assert_eq!((vc_rid.canonical.as_str(), vc_rid.angle_defect_pi_num, vc_rid.angle_defect_pi_den), ("3.4.5.4", 1, 15));
    assert_eq!(vc_rid.geometry_kind, GeometryKind::Spherical);
    // 3.7.42: angle sum 2 pi but no tiling (odd 7-gon between 3 and 42).
    // The 21 Euclidean vertex types minus parity failures leave the 11 Archimedean tilings
    // (Grunbaum & Shephard 1987, section 2.1).
    let euclid_21 = [
        "3.7.42", "3.8.24", "3.9.18", "3.10.15", "3.12.12", "4.5.20", "4.6.12", "4.8.8", "5.5.10", "6.6.6",
        "3.3.4.12", "3.4.3.12", "3.3.6.6", "3.6.3.6", "3.4.4.6", "3.4.6.4", "4.4.4.4", "3.3.3.4.4",
        "3.3.4.3.4", "3.3.3.3.6", "3.3.3.3.3.3",
    ];
    let mut ok = 0;
    for s in euclid_21 {
        if let Ok(info) = parse_vertex_config(s) {
            assert_eq!(info.geometry_kind, GeometryKind::Euclidean, "{s}");
            ok += 1;
        }
    }
    assert_eq!(ok, 11);
    // Seven triangles at a vertex: angle sum 7 pi / 3 > 2 pi, hence H^2 ({3,7}).
    assert_eq!(parse_vertex_config("3.3.3.3.3.3.3").unwrap().geometry_kind, GeometryKind::Hyperbolic);

    for bad in ["5.5.10", "3.4.5", "3.4.4.6", "3.3.4.12"] {
        let err = parse_vertex_config(bad).unwrap_err();
        assert!(err.contains("[PROVED: odd-gon parity obstruction]"), "{bad}: {err}");
    }
}

#[test]
fn uhp_geodesics_3d_surfaces_and_papercraft_net() {
    let z1 = [0.25, -0.35];
    let z2 = [-0.45, 0.30];
    let w1 = poincare_to_uhp(z1).unwrap();
    let w2 = poincare_to_uhp(z2).unwrap();
    let rt1 = uhp_to_poincare(w1).unwrap();
    assert!((rt1[0] - z1[0]).hypot(rt1[1] - z1[1]) < 1e-13);

    let d_p = hyp_distance_poincare(z1, z2).unwrap();
    let d_u = hyp_distance_uhp(w1, w2).unwrap();
    assert!((d_p - d_u).abs() < 1e-12);

    let arc = geodesic_arc_poincare(z1, z2, 8).unwrap();
    assert_eq!(arc.len(), 9);
    let mid_d = hyp_distance_poincare(z1, arc[4]).unwrap();
    assert!((mid_d - 0.5 * d_p).abs() < 1e-11);

    // Numerical Gaussian curvature check K == -1 on Pseudosphere, Dini, Kuen.
    for kind in [H2SurfaceKind::Pseudosphere, H2SurfaceKind::Dini { twist: 0.4 }, H2SurfaceKind::Kuen] {
        let (u, v) = (0.7_f64, 1.1_f64);
        let h = 1e-5;
        let sub3 = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot3 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross3 = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let p0 = surface_point(kind, u, v);
        let ru = sub3(surface_point(kind, u + h, v), surface_point(kind, u - h, v)).map(|c| c / (2.0 * h));
        let rv = sub3(surface_point(kind, u, v + h), surface_point(kind, u, v - h)).map(|c| c / (2.0 * h));
        let ruu = [
            (surface_point(kind, u + h, v)[0] - 2.0 * p0[0] + surface_point(kind, u - h, v)[0]) / (h * h),
            (surface_point(kind, u + h, v)[1] - 2.0 * p0[1] + surface_point(kind, u - h, v)[1]) / (h * h),
            (surface_point(kind, u + h, v)[2] - 2.0 * p0[2] + surface_point(kind, u - h, v)[2]) / (h * h),
        ];
        let rvv = [
            (surface_point(kind, u, v + h)[0] - 2.0 * p0[0] + surface_point(kind, u, v - h)[0]) / (h * h),
            (surface_point(kind, u, v + h)[1] - 2.0 * p0[1] + surface_point(kind, u, v - h)[1]) / (h * h),
            (surface_point(kind, u, v + h)[2] - 2.0 * p0[2] + surface_point(kind, u, v - h)[2]) / (h * h),
        ];
        let ruv = [
            (surface_point(kind, u + h, v + h)[0] - surface_point(kind, u + h, v - h)[0]
                - surface_point(kind, u - h, v + h)[0]
                + surface_point(kind, u - h, v - h)[0])
                / (4.0 * h * h),
            (surface_point(kind, u + h, v + h)[1] - surface_point(kind, u + h, v - h)[1]
                - surface_point(kind, u - h, v + h)[1]
                + surface_point(kind, u - h, v - h)[1])
                / (4.0 * h * h),
            (surface_point(kind, u + h, v + h)[2] - surface_point(kind, u + h, v - h)[2]
                - surface_point(kind, u - h, v + h)[2]
                + surface_point(kind, u - h, v - h)[2])
                / (4.0 * h * h),
        ];
        let n_raw = cross3(ru, rv);
        let n_len = dot3(n_raw, n_raw).sqrt();
        let n_unit = n_raw.map(|c| c / n_len);
        let (e, f, g) = (dot3(ru, ru), dot3(ru, rv), dot3(rv, rv));
        let (l, m, nn) = (dot3(ruu, n_unit), dot3(ruv, n_unit), dot3(rvv, n_unit));
        let k_gauss = (l * nn - m * m) / (e * g - f * f);
        assert!((k_gauss + 1.0).abs() < 1e-3, "{kind:?}: K={k_gauss}");
    }

    let cube = build_unified_pq(4, 3, 1).unwrap();
    let net = unfold_papercraft_net(&cube.coords_3d, &cube.faces).unwrap();
    assert_eq!(net.pieces.len(), 6);
    assert_eq!(net.tree_edges.len(), 5);
    assert!(net.max_edge_length_error < 1e-11);
}

#[test]
fn fast_h2_kernels_100k_scaling_queries_and_mutations() {
    let mut small_counts = Vec::new();
    for kernel in [H2Kernel::GraphCombinatorial, H2Kernel::DunhamHyperboloid, H2Kernel::SpatialHashPoincare] {
        // 29 = 1 + 7 + 21 faces: shells 0..2 of {7,3} vertex completion (a disk, so V - E + F = 1).
        let small = build_hyperbolic_large(7, 3, 29, kernel).unwrap();
        assert_eq!(small.n_active_faces, 29);
        assert_eq!(small.n_vertices as i64 - small.n_edges as i64 + small.n_active_faces as i64, 1);
        // Every vertex of the 8 shell-0/1 faces is interior to the patch: exactly q = 3 faces.
        for f in &small.faces[..8] {
            for &v in f {
                assert_eq!(small.vertex_faces[v].len(), 3, "{kernel:?}");
            }
        }
        small_counts.push((small.n_vertices, small.n_edges));

        let t0 = Instant::now();
        let large = build_hyperbolic_large(7, 3, 100_000, kernel).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        assert!(large.n_active_faces >= 100_000, "{kernel:?}: faces={}", large.n_active_faces);
        assert!(large.n_vertices >= 100_000, "{kernel:?}: verts={}", large.n_vertices);
        println!("# time {kernel:?} {{7,3}} 100k faces: {dt:.3}s");
        // Measured 0.15-0.22 s (graph), 0.8-0.9 s (spatial hash), 1.2-1.6 s (Dunham) single-threaded in
        // release; 15 s leaves ~10x headroom for a loaded machine yet still catches an O(N^2) regression.
        #[cfg(not(debug_assertions))]
        assert!(dt < 15.0, "{kernel:?} took {dt:.3}s (>= 15s)");
        let _ = dt;
    }
    // The three kernels build the same combinatorial patch.
    assert!(small_counts.windows(2).all(|w| w[0] == w[1]), "{small_counts:?}");

    let mut patch = build_hyperbolic_large(7, 3, 29, H2Kernel::SpatialHashPoincare).unwrap();
    assert_eq!(patch.edge_neighbors(0).len(), 7);
    assert_eq!(patch.vertex_neighbors(0).len(), 7);
    let ball0 = patch.hyperbolic_ball_query([0.0, 0.0], 1.5).unwrap();
    assert!(ball0.contains(&0) && ball0.len() >= 8);

    assert!(patch.remove_face(0));
    assert_eq!(patch.n_active_faces, 28);
    assert!(!patch.hyperbolic_ball_query([0.0, 0.0], 0.2).unwrap().contains(&0));
    assert!(patch.restore_face(0));
    assert_eq!(patch.n_active_faces, 29);

    // Shell-1 faces are surrounded once shell 2 is complete, so grow from the last (outermost) face.
    let rim_face = patch.faces.len() - 1;
    let empty_side = patch.face_neighbors[rim_face].iter().position(Option::is_none).unwrap();
    let (u, v) = (patch.faces[rim_face][empty_side], patch.faces[rim_face][(empty_side + 1) % 7]);
    let new_fid = patch.add_face_across_edge(rim_face, empty_side).unwrap();
    assert_eq!(patch.n_active_faces, 30);
    assert!(patch.edge_neighbors(rim_face).contains(&new_fid));
    // The new heptagon is regular: all 7 sides have the {7,3} edge length, cosh(l/2) = cos(pi/7)/sin(pi/3).
    let edge_len = 2.0 * ((std::f64::consts::PI / 7.0).cos() / (std::f64::consts::PI / 3.0).sin()).acosh();
    let cyc = patch.faces[new_fid].clone();
    assert_eq!((cyc[0], cyc[1]), (v, u));
    for k in 0..7 {
        let d = hyp_distance_poincare(patch.poincare_pts[cyc[k]], patch.poincare_pts[cyc[(k + 1) % 7]]).unwrap();
        assert!((d - edge_len).abs() < 1e-9, "side {k}: {d} vs {edge_len}");
    }

    let kept = patch.filter_by_region(RegionFilter::ShellRange(0, 1));
    assert_eq!(kept, 8);
}

/// Shell (face-layer) sizes by hand [PROVED]: a shell boundary is a cycle of vertices lying in 1 or 2 patch
/// faces (3 for p = 3). {p,3}: a degree-2 vertex takes one face over its two boundary edges, every other
/// edge one face; such faces leave p - 4 resp. p - 3 new boundary vertices (one of degree 2 each), so
/// (a, b) = (#deg-1, #deg-2) maps to shell size a and (a', b') = ((p-5) b + (p-4)(a-b), a). This gives
/// {7,3}: 1, 7, 21, 56 (= 7 F_2n, Fibonacci) and {8,3}: 1, 8, 32, 120. {p,q>=4}: a degree-d vertex gets
/// q - d - 2 corner faces (p - 2 new boundary vertices) and every edge one face (p - 3 new vertices), giving
/// {5,4}: 1, 10, 40, 150 and {4,5}: 1, 12, 48, 180. {3,7}: corners 5 - d, edge triangles add no vertex:
/// 1, 15, 45, 120.
#[test]
fn h2_kernels_agree_with_hand_shell_counts() {
    let cases: [(usize, usize, [usize; 4]); 5] = [
        (7, 3, [1, 7, 21, 56]),
        (8, 3, [1, 8, 32, 120]),
        (5, 4, [1, 10, 40, 150]),
        (4, 5, [1, 12, 48, 180]),
        (3, 7, [1, 15, 45, 120]),
    ];
    for (p, q, sizes) in cases {
        let target: usize = sizes.iter().sum();
        let pi = std::f64::consts::PI;
        let edge_len = 2.0 * ((pi / p as f64).cos() / (pi / q as f64).sin()).acosh();
        let mut vef = Vec::new();
        let mut centres: Vec<Vec<[f64; 2]>> = Vec::new();
        for kernel in [H2Kernel::GraphCombinatorial, H2Kernel::DunhamHyperboloid, H2Kernel::SpatialHashPoincare] {
            let pt = build_hyperbolic_large(p, q, target, kernel).unwrap();
            let tag = format!("{{{p},{q}}} {kernel:?}");
            let mut per_shell = [0usize; 4];
            for &s in &pt.face_shell {
                per_shell[s] += 1;
            }
            assert_eq!(per_shell, sizes, "{tag}");
            assert_eq!(pt.n_vertices as i64 - pt.n_edges as i64 + pt.faces.len() as i64, 1, "{tag}");
            // Neighbour table equals the one read off shared vertex pairs.
            let mut by_edge = std::collections::HashMap::new();
            for (fi, f) in pt.faces.iter().enumerate() {
                for s in 0..p {
                    let (u, v) = (f[s], f[(s + 1) % p]);
                    by_edge.entry((u.min(v), u.max(v))).or_insert_with(Vec::new).push((fi, s));
                }
            }
            assert_eq!(by_edge.len(), pt.n_edges, "{tag}");
            let mut want = vec![vec![None; p]; pt.faces.len()];
            for occ in by_edge.values() {
                assert!(occ.len() <= 2, "{tag}: edge in {} faces", occ.len());
                if let [(fa, sa), (fb, sb)] = occ[..] {
                    want[fa][sa] = Some(fb);
                    want[fb][sb] = Some(fa);
                }
            }
            assert_eq!(pt.face_neighbors, want, "{tag}");
            for (fi, f) in pt.faces.iter().enumerate() {
                for k in 0..p {
                    let d = hyp_distance_poincare(pt.poincare_pts[f[k]], pt.poincare_pts[f[(k + 1) % p]]).unwrap();
                    assert!((d - edge_len).abs() < 1e-8, "{tag}: face {fi} side {k}: {d} vs {edge_len}");
                    let nv = pt.vertex_faces[f[k]].len();
                    assert!(nv <= q, "{tag}: vertex in {nv} > q faces");
                    if pt.face_shell[fi] <= 2 {
                        assert_eq!(nv, q, "{tag}: face {fi} (shell {}) vertex {k}", pt.face_shell[fi]);
                    }
                }
            }
            vef.push((pt.n_vertices, pt.n_edges, pt.faces.len()));
            centres.push(pt.face_centres_poincare.clone());
        }
        assert!(vef.windows(2).all(|w| w[0] == w[1]), "{{{p},{q}}}: {vef:?}");
        // Same faces in the disk: every centre has a partner in each other kernel.
        for other in &centres[1..] {
            for c in &centres[0] {
                let best = other.iter().map(|o| hyp_distance_poincare(*c, *o).unwrap()).fold(f64::INFINITY, f64::min);
                assert!(best < 1e-7, "{{{p},{q}}}: centre {c:?} unmatched ({best})");
            }
        }
    }
}

#[test]
fn tgcell_quotients_exact_invariants() {
    // (name, p, q, genus, chi, V, E, F) with E = |G|/2, V = |G|/q, F = |G|/p for the orientation-
    // preserving group G: Z_8 (8), GL(2,3) (48, Bolza), PSL(2,7) (168, Klein quartic: 24 heptagons,
    // 84 edges, 56 vertices), A_5 (60, Bring {5,5}), order 24, Z_10, Z_2 x Z_6.
    let expected = [
        ("bolza_8_8", 8, 8, 2, -2, 1, 4, 1),
        ("bolza_8_3", 8, 3, 2, -2, 16, 24, 6),
        ("klein_7_3", 7, 3, 3, -4, 56, 84, 24),
        ("bring_5_5", 5, 5, 4, -6, 12, 30, 12),
        ("genus2_6_4", 6, 4, 2, -2, 6, 12, 4),
        ("genus2_10_5", 10, 5, 2, -2, 2, 5, 1),
        ("genus2_6_6", 6, 6, 2, -2, 2, 6, 2),
    ];
    for (name, ep, eq, eg, echi, ev, ee, ef) in expected {
        let qc = tgcell_quotient(name).unwrap();
        verify_tgcell_invariants(&qc).unwrap();
        assert_eq!((qc.p, qc.q, qc.genus, qc.chi), (ep, eq, eg, echi), "{name}");
        assert_eq!((qc.n_vertices, qc.n_edges, qc.n_faces), (ev, ee, ef), "{name}");
        assert_eq!(2 - 2 * (qc.genus as i64), qc.n_vertices as i64 - qc.n_edges as i64 + qc.n_faces as i64);
        assert_eq!(qc.side_pairings.len() * 2, qc.fundamental_polygon_poincare.len());
        // A spanning tree of the F faces uses F - 1 edges; the other E - F + 1 edges are side pairings.
        assert_eq!(qc.side_pairings.len(), ee - ef + 1, "{name}");
    }
    assert!(tgcell_quotient("no_such_surface").is_err());
}
