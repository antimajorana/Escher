//! Known-answer tests: Voronoi cells, Euclidean honeycombs, Dirichlet domains, P / D / G.

use std::time::Instant;

use super::dirichlet_quotient::dirichlet_manifold;
use super::honeycombs_e3::{build_euclidean_honeycomb, EuclideanKind};
use super::tpms_nets::{tpms_net, TpmsKind};
use super::tpms_surfaces::{nodal_surface_mesh, petrie_coxeter_surface};
use super::voronoi3d_dirichlet::{lattice_voronoi_cell, voronoi_cell_3d};
use super::{dot3, norm3, sub3};

#[test]
fn lattice_voronoi_cells() {
    let sc = lattice_voronoi_cell("SC").unwrap();
    assert_eq!(sc.counts(), (8, 12, 6));
    assert_eq!(sc.face_sizes(), vec![4; 6]);
    let fcc = lattice_voronoi_cell("FCC").unwrap();
    assert_eq!(fcc.counts(), (14, 24, 12));
    assert_eq!(fcc.face_sizes(), vec![4; 12]);
    let bcc = lattice_voronoi_cell("BCC").unwrap();
    assert_eq!(bcc.counts(), (24, 36, 14));
    assert_eq!(bcc.face_sizes(), [vec![4; 6], vec![6; 8]].concat());
    // With the 8 nearest BCC neighbours only, the cell is the octahedron |x|+|y|+|z| <= 3/2.
    let mut nn = Vec::new();
    for s in 0..8 {
        nn.push([0, 1, 2].map(|k| if (s >> k) & 1 == 1 { -1.0 } else { 1.0 }));
    }
    assert_eq!(voronoi_cell_3d([0.0; 3], &nn).unwrap().counts(), (6, 12, 8));
    assert!(voronoi_cell_3d([0.0; 3], &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).is_err());
    // Volume check by the divergence theorem: Voronoi cell volume = lattice covolume (SC 1, FCC 2, BCC 4).
    for (cell, vol) in [(&sc, 1.0), (&fcc, 2.0), (&bcc, 4.0)] {
        let mut v = 0.0;
        for f in &cell.faces {
            for k in 1..f.len() - 1 {
                let (a, b, c) = (cell.vertices[f[0]], cell.vertices[f[k]], cell.vertices[f[k + 1]]);
                v += dot3(&a, &super::cross3(&b, &c)).abs() / 6.0;
            }
        }
        assert!((v - vol).abs() < 1e-9, "{v} vs {vol}");
    }
}

#[test]
fn euclidean_honeycomb_patches() {
    let k = build_euclidean_honeycomb(EuclideanKind::BitruncatedCubic, 1).unwrap();
    assert_eq!(k.cells.len(), 15);
    assert_eq!(k.euler(), 1);
    // Each vertex of the central truncated octahedron has 4 edges (4 cells meet, pairwise in faces).
    let central: std::collections::BTreeSet<usize> = k.cells[0].iter().flat_map(|&f| k.faces[f].clone()).collect();
    assert_eq!(central.len(), 24);
    for v in central {
        assert_eq!(k.edges.iter().filter(|e| e.contains(&v)).count(), 4);
    }
    let rd = build_euclidean_honeycomb(EuclideanKind::RhombicDodecahedral, 1).unwrap();
    assert_eq!(rd.cells.len(), 13);
    assert_eq!(rd.euler(), 1);
    assert!(rd.faces.iter().all(|f| f.len() == 4));
    let oct = build_euclidean_honeycomb(EuclideanKind::AlternatedCubic, 1).unwrap();
    assert_eq!(oct.counts(), (14, 36, 32, 9));
    assert!(oct.faces.iter().all(|f| f.len() == 3));
    for kind in [EuclideanKind::Cubic, EuclideanKind::BitruncatedCubic, EuclideanKind::RhombicDodecahedral, EuclideanKind::AlternatedCubic] {
        let t = Instant::now();
        let cx = build_euclidean_honeycomb(kind, 2).unwrap();
        eprintln!("# time {kind:?} shells=2: {:.4} s, cells {}", t.elapsed().as_secs_f64(), cx.cells.len());
        assert_eq!(cx.euler(), 1, "{kind:?}");
    }
    assert_eq!(build_euclidean_honeycomb(EuclideanKind::Cubic, 2).unwrap().cells.len(), 25);
}

#[test]
fn dirichlet_domains() {
    let sw = dirichlet_manifold("seifert_weber").unwrap().quotient().unwrap();
    assert_eq!((sw.cells, sw.faces, sw.edges, sw.vertices), (1, 6, 6, 1));
    assert_eq!(sw.edge_class_sizes, vec![5; 6]);
    assert_eq!(sw.vertex_class_sizes, vec![20]);
    assert_eq!(sw.euler, 0);
    assert_eq!((sw.h1_torsion.clone(), sw.h1_free_rank), (vec![5, 5, 5], 0));
    assert_eq!(sw.h1_label, "Z_5^3");
    assert!(sw.evidence.starts_with("[PROVED"));
    let phs = dirichlet_manifold("poincare_homology_sphere").unwrap();
    assert_eq!(phs.geometry, "S3");
    let q = phs.quotient().unwrap();
    assert_eq!((q.cells, q.faces, q.edges, q.vertices), (1, 6, 10, 5));
    assert_eq!(q.edge_class_sizes, vec![3; 10]);
    assert_eq!(q.vertex_class_sizes, vec![4; 5]);
    assert_eq!((q.euler, q.h1_label.as_str()), (0, "0"));
    for (name, counts) in [("cubic_3_torus", (3, 3, 1)), ("bcc_3_torus", (7, 12, 6)), ("fcc_3_torus", (6, 8, 3))] {
        let q = dirichlet_manifold(name).unwrap().quotient().unwrap();
        assert_eq!((q.faces, q.edges, q.vertices), counts, "{name}");
        assert_eq!((q.euler, q.h1_free_rank, q.h1_label.as_str()), (0, 3, "Z^3"), "{name}");
    }
    assert!(dirichlet_manifold("hyperbolic_3_torus").is_err());
}

fn angle_cosines(net: &super::tpms_nets::TriplyPeriodicNet, site: usize) -> Vec<f64> {
    let mut vecs = Vec::new();
    for &(i, j, off) in &net.edges {
        let d = sub3(&net.positions[j], &net.positions[i]);
        let d = [d[0] + off[0] as f64, d[1] + off[1] as f64, d[2] + off[2] as f64];
        if i == site {
            vecs.push(d);
        }
        if j == site {
            vecs.push(d.map(|c| -c));
        }
    }
    let mut out = Vec::new();
    for (a, va) in vecs.iter().enumerate() {
        for vb in &vecs[a + 1..] {
            out.push(dot3(va, vb) / (norm3(va) * norm3(vb)));
        }
    }
    out
}

#[test]
fn labyrinth_nets() {
    let p = tpms_net(TpmsKind::P, 1).unwrap();
    assert_eq!((p.positions.len(), p.space_group_number), (1, 221));
    assert!(p.coordination.iter().all(|&c| c == 6) && (p.bond_length - 1.0).abs() < 1e-12);
    let d = tpms_net(TpmsKind::D, 1).unwrap();
    assert_eq!((d.positions.len(), d.space_group_number), (8, 227));
    assert!(d.coordination.iter().all(|&c| c == 4));
    assert!((d.bond_length - 3f64.sqrt() / 4.0).abs() < 1e-12);
    assert!(angle_cosines(&d, 0).iter().all(|c| (c + 1.0 / 3.0).abs() < 1e-12));
    let g = tpms_net(TpmsKind::G, 1).unwrap();
    assert_eq!((g.positions.len(), g.space_group_number), (8, 214));
    assert!(g.coordination.iter().all(|&c| c == 3));
    assert!((g.bond_length - 2f64.sqrt() / 4.0).abs() < 1e-12);
    for s in 0..8 {
        assert!(angle_cosines(&g, s).iter().all(|c| (c + 0.5).abs() < 1e-12), "120° at site {s}");
    }
    for (kind, num) in [(TpmsKind::P, 229), (TpmsKind::D, 224), (TpmsKind::G, 230)] {
        let two = tpms_net(kind, 2).unwrap();
        assert_eq!(two.space_group_number, num);
        // The labyrinths lie on opposite sides of the classical nodal surface.
        for (x, &lab) in two.positions.iter().zip(&two.labyrinth_of) {
            let f = kind.nodal(x);
            assert!(if lab == 0 { f > 0.5 } else { f < -0.5 }, "{kind:?} {x:?} {f}");
        }
        assert!(two.edges.iter().all(|&(i, j, _)| two.labyrinth_of[i] == two.labyrinth_of[j]));
    }
}

#[test]
fn periodic_surfaces_have_genus_three() {
    let p = petrie_coxeter_surface(TpmsKind::P).unwrap();
    assert_eq!((p.vertices.len(), p.edges.len(), p.faces.len()), (12, 24, 8));
    assert_eq!((p.euler_conventional, p.genus_primitive), (-4, 3));
    assert!(p.closed_manifold && p.faces.iter().all(|f| f.len() == 6));
    let d = petrie_coxeter_surface(TpmsKind::D).unwrap();
    assert_eq!((d.vertices.len(), d.edges.len(), d.faces.len()), (16, 48, 16));
    assert_eq!((d.euler_conventional, d.genus_primitive), (-16, 3));
    assert!(d.closed_manifold && p.evidence.starts_with("[PROVED") && d.evidence.starts_with("[PROVED"));
    for (s, per_vertex) in [(&p, 4), (&d, 6)] {
        for v in 0..s.vertices.len() {
            assert_eq!(s.faces.iter().filter(|f| f.contains(&v)).count(), per_vertex);
        }
    }
    for (kind, chi) in [(TpmsKind::P, -4), (TpmsKind::D, -16), (TpmsKind::G, -8)] {
        for n in [12, 24] {
            let t = Instant::now();
            let m = nodal_surface_mesh(kind, n).unwrap();
            eprintln!("# time nodal {kind:?} n={n}: {:.4} s, {} triangles", t.elapsed().as_secs_f64(), m.faces.len());
            assert!(m.closed_manifold, "{kind:?} n={n}");
            assert!(m.evidence.starts_with("[COMPUTATIONAL EVIDENCE"));
            assert_eq!(m.connected_components, 1, "{kind:?} n={n}");
            assert_eq!((m.euler_conventional, m.genus_primitive), (chi, 3), "{kind:?} n={n}");
            let worst = m.vertices.iter().map(|x| kind.nodal(x).abs()).fold(0.0, f64::max);
            assert!(worst < 0.2, "{kind:?} n={n} level-set error {worst}");
        }
    }
}
