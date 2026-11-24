use std::time::Instant;

use super::*;
use crate::live_ops::{build_surface_mesh, build_surface_mesh_auto};
use crate::mesh3d::mesh_from_surface;

/// Python's `surface_table` (pure Python, `PAPERTILLING_NO_RUST=1`) on the first `points` lattice of each group and on
/// some named lattices, pasted from `rust/papertilling-core/tests/fixtures/surface_table_py.json` (written by
/// `dump_surface_table.py` there). Columns: torus, cylinder, Moebius band, Klein bottle, sphere, projective plane,
/// genus 2, genus 3; a number is OK with that many sites, IMP is IMPOSSIBLE, NF is NOT FOUND.
const PYTHON_TABLE: [(&str, &str); 25] = [
    ("p1#1", "198 216 IMP IMP IMP IMP 216 216"),
    ("p2#1", "207 208 IMP IMP 4 IMP 72 72"),
    ("pm#1", "240 204 228 243 IMP IMP 216 216"),
    ("pg#1", "224 240 232 240 IMP IMP 288 288"),
    ("cm#1", "234 234 247 216 IMP IMP 216 216"),
    ("pmm#1", "216 208 259 238 4 7 72 72"),
    ("pmg#1", "264 220 247 264 IMP IMP 144 144"),
    ("pgg#1", "256 192 232 234 IMP IMP 288 288"),
    ("cmm#1", "210 220 247 216 4 13 72 72"),
    ("p4#1", "220 240 IMP IMP 24 IMP 288 288"),
    ("p4m#1", "220 216 259 221 6 6 72 72"),
    ("p4g#1", "220 240 328 204 24 6 288 288"),
    ("p3#1", "234 234 IMP IMP IMP IMP 216 216"),
    ("p3m1#1", "234 234 325 210 12 18 216 216"),
    ("p31m#1", "234 234 247 216 IMP IMP 216 216"),
    ("p6#1", "216 216 IMP IMP 60 IMP 432 432"),
    ("p6m#1", "220 220 247 216 12 16 72 72"),
    ("square", "220 216 259 221 6 6 72 72"),
    ("honeycomb", "210 208 270 224 20 30 144 144"),
    ("kagome", "234 208 325 210 30 45 216 216"),
    ("triangular", "220 220 247 216 12 16 72 72"),
    ("chiral_square", "220 240 IMP IMP 24 IMP 288 288"),
    ("shastry_sutherland", "220 240 369 204 24 6 288 288"),
    ("zigzag", "210 204 261 234 IMP IMP 144 144"),
    ("brick_pg", "144 192 336 216 IMP IMP 576 576"),
];

struct PyRow {
    status: &'static str,
    v: Option<usize>,
}

fn python_row(label: &str) -> Vec<PyRow> {
    let (_, cells) = PYTHON_TABLE.iter().find(|(l, _)| *l == label).expect("pasted Python row");
    cells
        .split(' ')
        .map(|c| match c {
            "IMP" => PyRow { status: "IMPOSSIBLE", v: None },
            "NF" => PyRow { status: "NOT FOUND", v: None },
            n => PyRow { status: "OK", v: Some(n.parse().unwrap()) },
        })
        .collect()
}

/// No NaN anywhere and unit normals (degenerate triangles near poles, cones or handle joins fall back to a unit
/// normal), every vertex inside the normalised ball the camera is set up for.
fn assert_finite_unit_normals(what: &str, gpu: &GpuMeshData) {
    for v in gpu.faces.iter().chain(&gpu.wires) {
        assert!(v.position.iter().chain(&v.normal).all(|x| x.is_finite()), "{what}: non-finite vertex");
        let nl = v.normal.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((nl - 1.0).abs() < 1e-3, "{what}: normal length {nl}");
        assert!(v.position.iter().map(|x| x * x).sum::<f32>().sqrt() <= 1.15, "{what}: vertex outside the ball");
    }
}

fn blank_canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::SurfaceNet { mesh: empty_net("blank") })
}

fn net_counts(m: &SurfaceMesh) -> (usize, usize) {
    (m.uv.len(), m.bonds.len())
}

#[test]
fn verdict_matrix_equals_python_surface_table_on_the_first_lattice_of_each_group() {
    let backend = AppBackend::new();
    let t0 = Instant::now();
    let matrix = backend.surface_verdict_matrix();
    println!("# verdict matrix (17 surface tables): {:.2} s", t0.elapsed().as_secs_f64());
    assert_eq!(matrix.len(), 17);
    for row in matrix {
        let py = python_row(&format!("{}#1", row.group));
        for (c_idx, t) in SurfaceTargetUi::ALL.iter().enumerate() {
            let (exp, got) = (&py[c_idx], &row.cells[c_idx]);
            assert_eq!(got.verdict.badge(), exp.status, "{} {}: {}", row.group, t.label(), got.reason);
            assert_eq!(got.sites, exp.v, "{} {}: sites", row.group, t.label());
        }
    }
    // the row of the lattice on screen agrees with the same Python table
    let kagome = backend.find_named("kagome").unwrap();
    let cells = live_ops::build_verdict_cells(kagome).unwrap();
    for (got, exp) in cells.iter().zip(python_row("kagome")) {
        assert_eq!((got.verdict.badge(), got.sites), (exp.status, exp.v));
    }
}

#[test]
fn auto_size_builds_the_surface_of_the_table_cell_and_refusals_carry_their_verdict() {
    let backend = AppBackend::new();
    for key in ["square", "honeycomb", "kagome", "triangular", "chiral_square", "shastry_sutherland", "zigzag", "brick_pg"] {
        let item = backend.find_named(key).unwrap();
        for (t, exp) in SurfaceTargetUi::ALL.into_iter().zip(python_row(key)) {
            match build_surface_mesh_auto(item, t, CentreModeUi::Auto, 2) {
                Ok((mesh, rep)) => {
                    assert_eq!(exp.status, "OK", "{key} {}", t.label());
                    assert!(rep.ok, "{key} {}", t.label());
                    assert_eq!(Some(mesh.sites.len()), exp.v, "{key} {}: V", t.label());
                }
                Err(msg) => assert!(msg.starts_with(exp.status), "{key} {}: '{msg}' vs python {}", t.label(), exp.status),
            }
        }
    }
}

#[test]
fn centre_modes_follow_the_frozen_order_face_centre_before_site() {
    let backend = AppBackend::new();
    let sq = backend.find_named("square").unwrap();
    // square lattice on the sphere: corners at face centres give the octahedron (V 6, E 12, F 8), at sites the cube
    // (V 8, E 12, F 6); auto takes the face-centre class first (frozen decision, RUST_PLAN section 9)
    let build = |cm| build_surface_mesh_auto(sq, SurfaceTargetUi::Sphere, cm, 2).unwrap().0;
    let counts = |m: &SurfaceMesh| (m.sites.len(), m.bonds.len(), m.faces.len());
    assert_eq!(counts(&build(CentreModeUi::Auto)), (6, 12, 8));
    assert_eq!(counts(&build(CentreModeUi::Face)), (6, 12, 8));
    assert_eq!(counts(&build(CentreModeUi::Site)), (8, 12, 6));
    // p4g (Shastry-Sutherland): alternating-centre (2a(i)+2a(ii)) hemi-cube RP2 succeeds (V=6, E=15, F=10); sphere is found.
    let ss = backend.find_named("shastry_sutherland").unwrap();
    let (rp2, rp2_rep) = build_surface_mesh_auto(ss, SurfaceTargetUi::ProjectivePlane, CentreModeUi::Auto, 2).unwrap();
    assert!(rp2_rep.ok && rp2.sites.len() == 6 && rp2.bonds.len() == 15 && rp2.faces.len() == 10);
    assert!(build_surface_mesh_auto(ss, SurfaceTargetUi::Sphere, CentreModeUi::Auto, 2).is_ok());
    // Proof (c): zigzag (pmg) sphere & RP2 are IMPOSSIBLE (lattice) even when global pmg symmetry has order-2 rotations.
    let zz = backend.find_named("zigzag").unwrap();
    let zz_err = build_surface_mesh_auto(zz, SurfaceTargetUi::Sphere, CentreModeUi::Auto, 2).unwrap_err();
    assert!(zz_err.starts_with("IMPOSSIBLE (lattice)"), "{zz_err}");
    // a proof from the symmetry is IMPOSSIBLE whatever the size: p4 has no mirror or glide
    let ch = backend.find_named("chiral_square").unwrap();
    let p = SurfaceBuildParamsUi { h: 3, k: 2, ..Default::default() };
    for t in [SurfaceTargetUi::MobiusBand, SurfaceTargetUi::KleinBottle, SurfaceTargetUi::ProjectivePlane] {
        assert!(build_surface_mesh(ch, t, &p).unwrap_err().starts_with("IMPOSSIBLE"), "{}", t.label());
    }
}

#[test]
fn custom_sizes_are_used_as_shown_with_hand_counted_tori() {
    let backend = AppBackend::new();
    // a torus (h, 0) x (0, k) holds h k cells: square (1 site, 2 bonds, 1 face per cell), honeycomb (2, 3, 1),
    // kagome (3, 6, 3: two triangles and a hexagon)
    for (key, per_cell) in [("square", (1, 2, 1)), ("honeycomb", (2, 3, 1)), ("kagome", (3, 6, 3))] {
        let item = backend.find_named(key).unwrap();
        for (h, k) in [(3, 4), (5, 3), (24, 24)] {
            let p = SurfaceBuildParamsUi { h, k, ..Default::default() };
            let (m, rep) = build_surface_mesh(item, SurfaceTargetUi::Torus, &p).unwrap();
            let n = (h * k) as usize;
            assert_eq!((m.sites.len(), m.bonds.len(), m.faces.len()), (n * per_cell.0, n * per_cell.1, n * per_cell.2), "{key} {h}x{k}");
            assert!(rep.ok);
        }
    }
    // a sphere of size (h, k) on the honeycomb has 20 (h^2 + h k + k^2) sites (Goldberg polyhedra: C20, C60, C80, ...)
    let hc = backend.find_named("honeycomb").unwrap();
    for (h, k) in [(1, 0), (1, 1), (2, 0), (2, 1), (3, 3)] {
        let p = SurfaceBuildParamsUi { h, k, ..Default::default() };
        let (m, _) = build_surface_mesh(hc, SurfaceTargetUi::Sphere, &p).unwrap();
        assert_eq!(m.sites.len() as i32, 20 * (h * h + h * k + k * k), "honeycomb sphere ({h}, {k})");
        assert_eq!(m.cones.len(), 12);
    }
}

#[test]
fn every_offered_size_builds_or_refuses_without_panic_and_the_largest_are_fast() {
    let backend = AppBackend::new();
    let mut items: Vec<CatalogueItemUi> = ["kagome", "honeycomb", "square", "rectangular", "shastry_sutherland", "brick_pg"]
        .iter()
        .map(|k| backend.find_named(k).unwrap().clone())
        .collect();
    for g in ["p1", "pgg", "p3", "p31m", "p6"] {
        items.push(backend.catalogue_for_group(g, crate::backend::CatalogueRuleUi::Points)[0].clone());
    }
    let mut worst = (0.0_f64, String::new());
    for item in &items {
        let max_rot = symmetry_facts(&item.group).max_rotation_order();
        for t in SurfaceTargetUi::ALL {
            let sc = size_controls(t, max_rot);
            let (h0, h1, k0, k1) = (*sc.h_range.start(), *sc.h_range.end(), *sc.k_range.start(), *sc.k_range.end());
            for (h, k) in [(h0, k0), (h1, k0), (h0, k1), (h1, k1), (h1 / 2, k1 / 2)] {
                for cm in [CentreModeUi::Auto, CentreModeUi::Site] {
                    let p = SurfaceBuildParamsUi { h, k, genus_g: 2, centre_mode: cm };
                    let t0 = Instant::now();
                    let res = build_surface_mesh(item, t, &p);
                    if let Ok((m, _)) = &res {
                        let what = format!("{} {} ({h}, {k})", item.label, t.label());
                        let gpu = mesh_from_surface(m, false);
                        assert_finite_unit_normals(&what, &gpu);
                        let order = crate::mesh3d::back_to_front_indices(&gpu.faces, gpu.opaque_vertex_count, &crate::mesh3d::orbit_rotation(0.3, 0.2));
                        assert_eq!(order.len(), gpu.faces.len() - gpu.opaque_vertex_count, "{what}");
                        if m.cover.is_some() {
                            assert_finite_unit_normals(&what, &mesh_from_surface(m, true));
                        }
                    }
                    let secs = t0.elapsed().as_secs_f64();
                    if secs > worst.0 {
                        worst = (secs, format!("{} {} ({h}, {k}) {:?}", item.label, t.label(), cm));
                    }
                    if let Err(msg) = &res {
                        let ok_prefix = ["IMPOSSIBLE", "NOT FOUND", "invalid size"].iter().any(|p| msg.starts_with(p));
                        assert!(ok_prefix, "{} {} ({h}, {k}): unclassified refusal '{msg}'", item.label, t.label());
                    }
                }
            }
        }
    }
    println!("# slowest offered size: {:.3} s for {}", worst.0, worst.1);
    assert!(worst.0 < 3.0, "slowest offered size takes {:.2} s: {}", worst.0, worst.1);
}

#[test]
fn net_and_3d_follow_every_control_change_and_refusals_clear_both() {
    let backend = AppBackend::new();
    let mut canvas = blank_canvas();
    let mut view = View3dState::default();
    let mut st = SurfacesTabState::new();
    let kagome = backend.find_named("kagome").unwrap().clone();
    let chiral = backend.find_named("chiral_square").unwrap().clone();
    let check = |st: &SurfacesTabState, canvas: &Canvas2D, view: &View3dState, what: &str| match st.current_mesh.as_ref() {
        Some(m) => {
            assert_eq!(canvas.counts(), net_counts(m), "{what}: 2D net is not the built surface");
            let gpu = view.mesh_data();
            let mult = if st.show_cover && m.cover.is_some() { 2 } else { 1 };
            assert_eq!((gpu.vertex_count, gpu.face_count), (mult * m.sites.len(), mult * m.faces.len()), "{what}: 3D");
            assert!(view.caption().contains(&format!("V={}", mult * m.sites.len())), "{what}: caption");
            if m.name.ends_with(" (sphere)") {
                assert!(view.caption().contains("sphere · ") && view.caption().contains("gluing template"), "{what}: {}", view.caption());
            }
        }
        None => {
            assert_eq!(canvas.counts(), (0, 0), "{what}: a refused surface left a stale net");
            assert!(view.mesh_data().faces.is_empty(), "{what}: a refused surface left a stale 3D mesh");
            assert!(st.refusal_msg.as_deref().is_some());
        }
    };
    let mut gen_before = view.mesh_generation();
    let mut step = |st: &mut SurfacesTabState, canvas: &mut Canvas2D, view: &mut View3dState, item: &CatalogueItemUi, what: &str| {
        st.activate_and_sync(item, &backend, canvas, view);
        assert!(view.mesh_generation() > gen_before, "{what}: 3D mesh not re-uploaded");
        gen_before = view.mesh_generation();
        check(st, canvas, view, what);
    };
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome sphere auto");
    st.auto_size = false;
    st.params.h = 2;
    st.params.k = 1;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome sphere (2, 1)");
    st.target = SurfaceTargetUi::ProjectivePlane;
    st.show_cover = true;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome RP2 cover");
    st.show_cover = false;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome RP2");
    st.params.centre_mode = CentreModeUi::Site;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome RP2 site");
    st.target = SurfaceTargetUi::MobiusBand;
    step(&mut st, &mut canvas, &mut view, &chiral, "chiral square Moebius (IMPOSSIBLE)");
    assert!(st.refusal_msg.as_deref().unwrap().starts_with("IMPOSSIBLE"));
    st.target = SurfaceTargetUi::Torus;
    step(&mut st, &mut canvas, &mut view, &chiral, "chiral square torus");
    st.target = SurfaceTargetUi::Genus3;
    st.params.h = 1;
    st.params.k = 1;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome genus 3 on a 1 x 1 torus (NOT FOUND)");
    assert!(st.refusal_msg.as_deref().unwrap().starts_with("NOT FOUND"));
    st.auto_size = true;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome genus 3 auto");
    assert_eq!(st.current_mesh.as_ref().unwrap().chi, -4);
    // out-of-range sizes (e.g. left over from another target) are clamped to what the controls offer
    st.target = SurfaceTargetUi::Sphere;
    st.auto_size = false;
    st.params.h = -5;
    st.params.k = 99;
    step(&mut st, &mut canvas, &mut view, &kagome, "kagome sphere clamped");
    assert_eq!((st.params.h, st.params.k), (1, 8));
}

#[test]
fn special_locus_cell_enables_non_orientable_and_rp2_surfaces_on_chiral_recipe() {
    let backend = AppBackend::new();
    let mut canvas = blank_canvas();
    let oblique = backend.find_named("oblique").unwrap();

    // Off-locus oblique (p2) has no mirrors/glides -> IMPOSSIBLE for Mobius & RP2.
    assert!(
        build_surface_mesh_auto(oblique, SurfaceTargetUi::MobiusBand, CentreModeUi::Auto, 2)
            .unwrap_err()
            .starts_with("IMPOSSIBLE")
    );
    assert!(
        build_surface_mesh_auto(oblique, SurfaceTargetUi::ProjectivePlane, CentreModeUi::Auto, 2)
            .unwrap_err()
            .starts_with("IMPOSSIBLE")
    );

    // Move oblique onto b=1, gamma=90 (p4m square special locus): cell-aware symmetry_facts_of_cell allows both.
    let mut gallery = crate::gallery_tab::GalleryTabState::new(&backend, &mut canvas);
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("oblique")).unwrap();
    gallery.select_named(idx, &backend, &mut canvas);
    gallery.set_param("b", 1.0, &mut canvas).unwrap();
    gallery.set_param("gamma", 90.0, &mut canvas).unwrap();
    let tuned = gallery.active_item().clone();

    let (mob, mob_rep) = build_surface_mesh_auto(&tuned, SurfaceTargetUi::MobiusBand, CentreModeUi::Auto, 2).unwrap();
    assert!(mob_rep.ok && !mob.orientable);
    let (rp2, rp2_rep) = build_surface_mesh_auto(&tuned, SurfaceTargetUi::ProjectivePlane, CentreModeUi::Auto, 2).unwrap();
    assert!(rp2_rep.ok && !rp2.orientable);
}

#[test]
fn background_custom_size_build_matches_the_synchronous_build_and_is_debounced() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    let kagome = backend.find_named("kagome").unwrap().clone();
    let (mut cv_a, mut cv_s) = (blank_canvas(), blank_canvas());
    let (mut v_a, mut v_s) = (View3dState::default(), View3dState::default());
    let mut asy = SurfacesTabState::new();
    let mut syn = SurfacesTabState::new();
    for st in [&mut asy, &mut syn] {
        st.target = SurfaceTargetUi::Torus;
        st.auto_size = false;
    }
    // a 4 x 3 torus of kagome: 3 sites x 12 cells = 36 vertices, 6 bonds x 12 = 72 edges
    asy.params.h = 2;
    asy.request_build(&kagome, &ctx);
    asy.params.h = 4;
    asy.params.k = 3;
    asy.request_build(&kagome, &ctx);
    assert!(asy.is_building());
    let g0 = v_a.mesh_generation();
    assert!(asy.wait_build(&ctx, &mut cv_a, &mut v_a));
    assert!(!asy.is_building());
    syn.params.h = 4;
    syn.params.k = 3;
    let gs = v_s.mesh_generation();
    syn.activate_and_sync(&kagome, &backend, &mut cv_s, &mut v_s);
    let (ma, ms) = (asy.current_mesh.as_ref().unwrap(), syn.current_mesh.as_ref().unwrap());
    assert_eq!((ma.sites.len(), ma.faces.len()), (ms.sites.len(), ms.faces.len()));
    assert_eq!(ma.sites.len(), 36, "kagome 4 x 3 torus: 3 sites per cell x 12 cells");
    assert_eq!(cv_a.counts(), cv_s.counts());
    assert_eq!(v_a.mesh_generation() - g0, v_s.mesh_generation() - gs, "the superseded request built nothing");

    // a synchronous rebuild cancels a queued background request
    asy.params.h = 6;
    asy.request_build(&kagome, &ctx);
    asy.activate_and_sync(&kagome, &backend, &mut cv_a, &mut v_a);
    assert!(!asy.is_building());
    assert!(!asy.wait_build(&ctx, &mut cv_a, &mut v_a));
}

#[test]
fn a_build_finishing_inside_a_newer_requests_debounce_is_never_shown() {
    let backend = AppBackend::new();
    let ctx = egui::Context::default();
    let kagome = backend.find_named("kagome").unwrap().clone();
    let mut cv = blank_canvas();
    let mut view = View3dState::default();
    let mut st = SurfacesTabState::new();
    st.target = SurfaceTargetUi::Torus;
    st.auto_size = false;
    st.params.h = 2;
    st.params.k = 2;
    st.request_build(&kagome, &ctx);
    std::thread::sleep(super::build_async::DEBOUNCE + std::time::Duration::from_millis(10));
    assert!(!st.pump_build(&ctx, &mut cv, &mut view), "submitted, not finished yet");
    st.params.h = 3;
    st.request_build(&kagome, &ctx); // queued behind the running 2 x 2 build
    std::thread::sleep(std::time::Duration::from_millis(60)); // 2 x 2 torus is done, 3 x 2 still debouncing
    assert!(!st.pump_build(&ctx, &mut cv, &mut view));
    assert!(st.current_mesh.is_none(), "the superseded 2 x 2 torus is never applied");
    assert!(st.wait_build(&ctx, &mut cv, &mut view));
    // kagome 3 x 2 simple torus: 3 sites per cell x 6 cells = 18 sites (k=1 has multi-edges, refused by C1)
    assert_eq!(st.current_mesh.as_ref().unwrap().sites.len(), 18);
}
