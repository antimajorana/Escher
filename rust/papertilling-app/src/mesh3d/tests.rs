use std::collections::{HashMap, HashSet};

use papertilling_core::surface::{projective_plane, sphere};
use papertilling_core::types::SurfaceMesh;

use super::*;
use crate::backend::{AppBackend, CatalogueItemUi, SurfaceBuildParamsUi, SurfaceTargetUi};

fn cell_of(item: &CatalogueItemUi) -> papertilling_core::types::Cell2D {
    papertilling_core::plane::family::LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).unwrap().cell()
}

fn translucent_tris(gpu: &GpuMeshData) -> impl Iterator<Item = &[Vertex3d; 3]> {
    gpu.faces[gpu.opaque_vertex_count..].as_chunks::<3>().0.iter()
}

fn assert_sane(label: &str, gpu: &GpuMeshData) {
    assert!(gpu.opaque_vertex_count <= gpu.faces.len() && gpu.opaque_vertex_count.is_multiple_of(3), "{label}: opaque split");
    assert_eq!(gpu.faces.len() % 3, 0, "{label}: triangle list");
    assert_eq!(gpu.wires.len() % 2, 0, "{label}: line list");
    for v in gpu.faces.iter().chain(&gpu.wires) {
        assert!(v.position.iter().chain(&v.normal).chain(&v.color).all(|x| x.is_finite()), "{label}: non-finite vertex");
        let r = v.position.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(r <= 1.15, "{label}: vertex outside the normalised ball ({r})");
        let nl = v.normal.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nl - 1.0).abs() < 1e-3, "{label}: normal not unit ({nl})");
    }
    for v in &gpu.faces[..gpu.opaque_vertex_count] {
        assert_eq!(v.color[3], 1.0, "{label}: markers are opaque");
    }
    for t in translucent_tris(gpu) {
        assert!(t.iter().all(|v| v.color[3] < 1.0), "{label}: faces are translucent");
    }
}

#[test]
fn face_palette_distinguishes_rings_and_matches_the_2d_ring_palette() {
    let rings = [3, 4, 5, 6, 7, 8];
    for i in 0..rings.len() {
        for j in (i + 1)..rings.len() {
            assert_ne!(face_size_rgba(rings[i], true), face_size_rgba(rings[j], true));
        }
    }
    for k in 3..10 {
        let c2 = crate::scene2d::ring_tint_rgba(k);
        for orientable in [true, false] {
            let c3 = face_size_rgba(k, orientable);
            for ch in 0..3 {
                assert!((c3[ch] * 255.0 - f32::from(c2[ch])).abs() < 0.5, "ring {k}: 3D hue differs from 2D");
            }
        }
    }
}

#[test]
fn mesh_from_surface_torus_rp2_cover_and_genus2() {
    let backend = AppBackend::new();
    let kagome = backend.find_named("kagome").unwrap();
    let params = SurfaceBuildParamsUi::default();

    let (torus, rep_t) = backend.build_surface(kagome, SurfaceTargetUi::Torus, &params).unwrap();
    assert!(rep_t.ok && torus.chi == 0);
    let gpu_torus = mesh_from_surface(&torus, false);
    assert!(gpu_torus.faces.len() > torus.faces.len() * 3 && gpu_torus.wires.len() > torus.bonds.len() * 2);
    assert_sane("kagome torus", &gpu_torus);

    let (rp2, rep_p) =
        crate::live_ops::build_surface_mesh_auto(kagome, SurfaceTargetUi::ProjectivePlane, params.centre_mode, 2).unwrap();
    assert!(rep_p.ok && rp2.chi == 1 && rp2.cover.is_some());
    let (gpu_quot, gpu_cov) = (mesh_from_surface(&rp2, false), mesh_from_surface(&rp2, true));
    assert_eq!((gpu_quot.vertex_count, gpu_quot.edge_count, gpu_quot.face_count), (rp2.sites.len(), rp2.bonds.len(), rp2.faces.len()));
    assert_eq!(gpu_cov.vertex_count, 2 * gpu_quot.vertex_count);
    assert_eq!(gpu_cov.face_count, 2 * gpu_quot.face_count);
    assert!(gpu_cov.faces.len() > gpu_quot.faces.len());
    assert_sane("kagome RP2", &gpu_quot);
    assert_sane("kagome RP2 cover", &gpu_cov);

    let (g2, rep_g2) = backend.build_surface(kagome, SurfaceTargetUi::Genus2, &params).unwrap();
    assert!(rep_g2.ok && g2.chi == -2);
    let gpu_g2 = mesh_from_surface(&g2, false);
    assert_eq!(gpu_g2.vertex_count, g2.xyz.len());
    assert!(gpu_g2.faces.len() > 100 && gpu_g2.wires.len() >= g2.bonds.len() * 2);
    assert_sane("kagome genus 2", &gpu_g2);
}

/// The double cover of a projective plane is a sphere: every lifted face closes along cover bonds, every cover bond
/// borders exactly two lifted faces, and V - E + F = 2 (Euler characteristic of S^2) with V, E, F twice the quotient's.
fn assert_cover_is_a_sphere(label: &str, m: &SurfaceMesh) {
    let cov = m.cover.as_ref().expect("RP2 has a double cover");
    let bonds: HashSet<(usize, usize)> = cov.bonds.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    assert_eq!(bonds.len(), cov.bonds.len(), "{label}: double cover bond");
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for f in &cov.faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            assert!(bonds.contains(&(a.min(b), a.max(b))), "{label}: lifted face edge {a}-{b} is no cover bond");
            *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    assert!(uses.values().all(|&c| c == 2) && uses.len() == bonds.len(), "{label}: cover bonds not in 2 faces");
    let (v, e, f) = (cov.xyz.len() as i64, cov.bonds.len() as i64, cov.faces.len() as i64);
    assert_eq!(v - e + f, 2, "{label}: cover is not a sphere");
    assert_eq!((v, e, f), (2 * m.sites.len() as i64, 2 * m.bonds.len() as i64, 2 * m.faces.len() as i64), "{label}: 2:1");
}

/// RP2 drawn once: every quotient face is one spherical patch (no triangle cuts through the ball), and every
/// quotient face is drawn exactly once (the hemisphere model).
fn assert_rp2_quotient_patches(label: &str, m: &SurfaceMesh) {
    let gpu = mesh_from_surface(m, false);
    assert_sane(label, &gpu);
    let longest = translucent_tris(&gpu)
        .map(|t| (0..3).map(|k| geom::dist3(t[k].position, t[(k + 1) % 3].position)).fold(0.0_f32, f32::max))
        .fold(0.0_f32, f32::max);
    assert!(longest < 1.2, "{label}: a face triangle spans {longest} (cuts through the sphere)");
    let lifts = rp2::upper_lifts(m, m.cover.as_ref().unwrap()).unwrap();
    assert_eq!(lifts.len(), m.faces.len());
    assert!(lifts.iter().enumerate().all(|(f, &l)| l / 2 == f), "{label}: one lift per face");
}

#[test]
fn rp2_double_cover_is_a_sphere_and_the_quotient_view_has_no_torn_faces() {
    let backend = AppBackend::new();
    let get = |k: &str| backend.find_named(k).unwrap();
    // Known answers: the hemi-dodecahedron (honeycomb on RP2, size (1, 0)) is the Petersen graph, V 10, E 15, F 6;
    // the hemi-cube (square, corners at sites) has V 4, E 6, F 3.
    let hc = get("honeycomb");
    let pet = projective_plane(&hc.group, &cell_of(hc), Some((1, 0)), None).unwrap();
    assert_eq!((pet.sites.len(), pet.bonds.len(), pet.faces.len()), (10, 15, 6));
    let sq = get("square");
    let hemi_cube = projective_plane(&sq.group, &cell_of(sq), Some((1, 0)), Some("site")).unwrap();
    assert_eq!((hemi_cube.sites.len(), hemi_cube.bonds.len(), hemi_cube.faces.len()), (4, 6, 3));
    let mut cases = vec![("honeycomb (1,0)", pet), ("square site (1,0)", hemi_cube)];
    // V of these projective planes computed by Python (dump_surface_table.py fixture), pasted
    for (key, size, centre, v_py) in [
        ("honeycomb", (1, 1), None, 30),
        ("kagome", (1, 1), None, 45),
        ("triangular", (2, 0), None, 21),
        ("square", (2, 0), Some("face centre"), 12),
    ] {
        let it = get(key);
        let m = projective_plane(&it.group, &cell_of(it), Some(size), centre).unwrap();
        assert_eq!(m.sites.len(), v_py, "{key}");
        cases.push((key, m));
    }
    let p3m1 = &backend.catalogue_for_group("p3m1", crate::backend::CatalogueRuleUi::Points)[0];
    let hemi_oct = projective_plane(&p3m1.group, &cell_of(p3m1), Some((1, 1)), None).unwrap();
    assert_eq!(hemi_oct.sites.len(), 18, "p3m1 #1 (Python: 18)");
    cases.push(("p3m1 #1 hemi-octahedron", hemi_oct));
    let rect = get("rectangular");
    cases.push(("rectangular hemi-pillow", projective_plane(&rect.group, &cell_of(rect), Some((6, 6)), Some("site")).unwrap()));
    let mut torn_before = 0;
    for (label, m) in &cases {
        assert_cover_is_a_sphere(label, m);
        assert_rp2_quotient_patches(label, m);
        // the former quotient drawing (faces following the bond curves of one lift per site) tore faces whose
        // corners sit on both sides of the antipodal seam: fan triangles through the ball
        let old = quotient_mesh(m);
        let longest = translucent_tris(&old)
            .map(|t| (0..3).map(|k| geom::dist3(t[k].position, t[(k + 1) % 3].position)).fold(0.0_f32, f32::max))
            .fold(0.0_f32, f32::max);
        torn_before += usize::from(longest > 1.2);
    }
    assert!(torn_before > 0, "negative control: the old drawing should tear some RP2 faces");
}

#[test]
fn sphere_meshes_have_unit_normals_and_no_holes_at_the_poles() {
    let backend = AppBackend::new();
    let hc = backend.find_named("honeycomb").unwrap();
    // C60: the honeycomb on the icosahedron at size (1, 1) is the truncated icosahedron, V 60, E 90, F 32.
    let c60 = sphere(&hc.group, &cell_of(hc), Some((1, 1)), None).unwrap();
    assert_eq!((c60.sites.len(), c60.bonds.len(), c60.faces.len()), (60, 90, 32));
    let gpu = mesh_from_surface(&c60, false);
    assert_sane("C60", &gpu);
    assert_eq!((gpu.vertex_count, gpu.edge_count, gpu.face_count), (60, 90, 32));
    // 12 pentagons + 20 hexagons: two fan triangles per bond side, i.e. 2 E fan triangles of 16 arc pieces at most.
    let n_tri = translucent_tris(&gpu).count();
    assert!(n_tri >= 2 * 90, "{n_tri} face triangles for C60");
    // the 12 pentagons carry the +60 cones at their centres (tinted towards the cone colour), the 20 hexagons do not
    let pent = geom::tint_towards(face_size_rgba(5, true), geom::cone_rgba(60.0), 0.22);
    let hex = face_size_rgba(6, true);
    let n_pent = translucent_tris(&gpu).filter(|t| t[0].color == pent).count();
    let n_hex = translucent_tris(&gpu).filter(|t| t[0].color == hex).count();
    assert!(n_pent > 0 && n_hex > 0 && n_pent + n_hex == n_tri, "pentagons {n_pent}, hexagons {n_hex}");
}

#[test]
fn back_to_front_order_is_a_permutation_sorted_by_view_depth() {
    let backend = AppBackend::new();
    let kagome = backend.find_named("kagome").unwrap();
    let (mesh, _) = backend.build_surface(kagome, SurfaceTargetUi::KleinBottle, &SurfaceBuildParamsUi::default()).unwrap();
    let gpu = mesh_from_surface(&mesh, false);
    for (yaw, pitch) in [(0.0, 0.0), (0.55, 0.35), (2.0, -1.2), (-3.0, 1.5)] {
        let rot = orbit_rotation(yaw, pitch);
        let idx = back_to_front_indices(&gpu.faces, gpu.opaque_vertex_count, &rot);
        let n_tri = (gpu.faces.len() - gpu.opaque_vertex_count) / 3;
        assert_eq!(idx.len(), 3 * n_tri);
        let mut seen = vec![false; gpu.faces.len()];
        for &i in &idx {
            assert!((i as usize) >= gpu.opaque_vertex_count && !seen[i as usize]);
            seen[i as usize] = true;
        }
        let depth: Vec<f32> = idx
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| t.iter().map(|&i| sort::view_depth(&rot, gpu.faces[i as usize].position)).sum::<f32>() / 3.0)
            .collect();
        assert!(depth.windows(2).all(|w| w[0] <= w[1] + 1e-6), "not back to front");
        // the farthest triangle is behind the nearest one: camera looks down -z
        assert!(depth.first() <= depth.last());
    }
    assert!(back_to_front_indices(&[], 0, &orbit_rotation(0.0, 0.0)).is_empty());
}

#[test]
fn perspective_maps_origin_inside_wgpu_unit_depth() {
    let rot = orbit_rotation(0.4, -0.25);
    let mvp = build_mvp(1.5, 3.4, &rot);
    let ndc_z = mvp[3][2] / mvp[3][3];
    assert!((0.0..=1.0).contains(&ndc_z));
    // At yaw = pitch = 0 the camera looks down -z: the point z = +1 is nearer than z = -1 and must get the smaller
    // depth (the LessEqual depth test keeps the nearest), and the whole normalised ball (closest zoom 1.45) must fit
    // between the near and far planes.
    for dist in [1.45, 3.1, 9.0] {
        let view = build_mvp(1.0, dist, &orbit_rotation(0.0, 0.0));
        let ndc = |p: [f32; 3]| {
            let c: [f32; 4] = core::array::from_fn(|r| (0..3).map(|k| view[k][r] * p[k]).sum::<f32>() + view[3][r]);
            c[2] / c[3]
        };
        let (near_z, far_z) = (ndc([0.0, 0.0, 1.15]), ndc([0.0, 0.0, -1.15]));
        assert!(near_z < far_z, "distance {dist}");
        assert!((0.0..=1.0).contains(&near_z) && (0.0..=1.0).contains(&far_z), "distance {dist}");
        assert!(sort::view_depth(&orbit_rotation(0.0, 0.0), [0.0, 0.0, 1.0]) > sort::view_depth(&orbit_rotation(0.0, 0.0), [0.0, 0.0, -1.0]));
    }
}
