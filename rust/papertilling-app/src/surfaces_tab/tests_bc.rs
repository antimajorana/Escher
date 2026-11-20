//! WP-U3 tests: the boundary options of the Surfaces tab. Every expected number is derived by hand or by a theorem,
//! never copied from the code under test: the honeycomb patch of the independent `seed2d.py` pilot (3(n+1)^2 hexagons
//! after n growth steps), the torus counts V = L^2, E = 2L^2, F = L^2 of {4,4} and V = 2n^2, E = 3n^2, F = n^2 of
//! the honeycomb n x n torus (Euler characteristic 0), and Gauss-Bonnet (disclinations 360 deg x chi).

use std::time::Instant;

use super::bc::{BC_TARGETS, BcControls, build_bc, cone_note, summarize};
use super::*;
use crate::backend::VerdictKindUi;
use crate::live_ops::build_surface_mesh_auto;

fn blank_canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::SurfaceNet { mesh: empty_net("blank") })
}

fn lattice(backend: &AppBackend, name: &str) -> CatalogueItemUi {
    backend.find_named(name).unwrap_or_else(|| panic!("named lattice {name}")).clone()
}

fn euler(m: &SurfaceMesh) -> i64 {
    m.sites.len() as i64 - m.bonds.len() as i64 + m.faces.len() as i64
}

#[test]
fn default_is_an_open_patch_with_four_layers_of_hexagons() {
    // seed2d.py (independent): honeycomb vertex seed, n = 4 growth steps -> 3(n+1)^2 = 75 hexagons
    let backend = AppBackend::new();
    let hc = lattice(&backend, "honeycomb");
    let mut st = SurfacesTabState::new();
    assert_eq!((st.target, st.bc.layers), (SurfaceTargetUi::Open, 4));
    let (mut canvas, mut view) = (blank_canvas(), View3dState::default());
    st.activate_and_sync(&hc, &backend, &mut canvas, &mut view);
    let m = st.current_mesh.as_ref().expect("open patch built");
    assert_eq!((m.faces.len(), m.chi, m.closed), (75, 1, false));
    assert_eq!(euler(m), 1, "a disk");
    assert!(st.notes.picture_note.is_none(), "flat patch: no 3D caption needed");
    // layers = 1 -> 3 * 2^2 = 12
    st.bc.layers = 1;
    st.activate_and_sync(&hc, &backend, &mut canvas, &mut view);
    assert_eq!(st.current_mesh.as_ref().unwrap().faces.len(), 12);
}

#[test]
fn chosen_options_set_the_size_of_the_torus_the_cylinder_and_the_hexagonal_torus() {
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    let hc = lattice(&backend, "honeycomb");
    let ctl = |f: &dyn Fn(&mut BcControls)| {
        let mut c = BcControls::default();
        f(&mut c);
        c
    };
    // {4,4} torus on the supercell (4,0),(2,4): det 16 cells -> V = 16, E = 32, F = 16
    let (m, rep, _) = build_bc(&sq, SurfaceTargetUi::Torus, &ctl(&|c| c.t2 = [2, 4])).unwrap();
    assert_eq!((m.sites.len(), m.bonds.len(), m.faces.len()), (16, 32, 16));
    assert!(rep.ok && euler(&m) == 0);
    // a cylinder is a torus cut open: chi = 0 and an open rim
    let (m, rep, n) = build_bc(&sq, SurfaceTargetUi::Cylinder, &ctl(&|c| c.wrap = [4, 0])).unwrap();
    assert!(rep.ok && !m.closed && m.chi == 0, "cylinder chi = 0");
    assert!(n.picture_note.is_none(), "the rolled cylinder is isometric: no caption");
    // honeycomb 3 x 3 torus: 2 sites, 3 bonds, 1 face per cell -> V = 18, E = 27, F = 9; the hexagonal domain is the same quotient
    let flat = ctl(&|c| {
        c.t1 = [3, 0];
        c.t2 = [0, 3];
    });
    let (a, _, _) = build_bc(&hc, SurfaceTargetUi::Torus, &flat).unwrap();
    let (b, rep, nb) = build_bc(&hc, SurfaceTargetUi::TorusHex, &flat).unwrap();
    for m in [&a, &b] {
        assert_eq!((m.sites.len(), m.bonds.len(), m.faces.len()), (18, 27, 9));
    }
    assert!(rep.ok);
    assert!(!nb.readout.iter().any(|l| l.contains("reduce to 2")), "a 60 degree rhombus has a hexagonal Wigner-Seitz cell");
    // a rectangular supercell of a square lattice: the 3 pairs reduce to 2, said in the read-out
    let (_, _, nr) = build_bc(&sq, SurfaceTargetUi::TorusHex, &BcControls::default()).unwrap();
    assert!(nr.readout.iter().any(|l| l.contains("3 pairs reduce to 2")), "{:?}", nr.readout);
}

#[test]
fn twisted_gluings_follow_the_lattice_and_refuse_with_a_proof_when_there_is_no_mirror() {
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    let (m, rep, note) = build_bc(&sq, SurfaceTargetUi::MobiusBand, &BcControls::default()).unwrap();
    assert!(rep.ok && !m.orientable && !m.closed && m.chi == 0, "Moebius band: one-sided, chi = 0, open rim");
    assert!(note.picture_note.is_some_and(|n| n.contains("not isometric")));
    let (k, rep, note) = build_bc(&sq, SurfaceTargetUi::KleinBottle, &BcControls::default()).unwrap();
    assert!(rep.ok && !k.orientable && k.closed && k.chi == 0, "Klein bottle: non-orientable, closed, chi = 0");
    assert_eq!(euler(&k), 0);
    assert!(note.picture_note.is_some());
    // p4 (chiral square) has no mirror or glide: the twist is refused, and the reason says proved
    let chiral = lattice(&backend, "chiral_square");
    for t in [SurfaceTargetUi::MobiusBand, SurfaceTargetUi::KleinBottle] {
        let e = build_bc(&chiral, t, &BcControls::default()).unwrap_err();
        assert!(e.starts_with("IMPOSSIBLE"), "{e}");
    }
    let s = summarize(&chiral).unwrap();
    assert!(s.glides.is_empty());
    assert_eq!(s.verdict(SurfaceTargetUi::KleinBottle).unwrap().kind, VerdictKindUi::Impossible);
    assert_eq!(s.verdict(SurfaceTargetUi::Torus).unwrap().kind, VerdictKindUi::Ok);
    assert_eq!(s.verdict(SurfaceTargetUi::Open).unwrap().kind, VerdictKindUi::Ok);
}

#[test]
fn sphere_and_projective_plane_show_gauss_bonnet_disclinations() {
    // Gauss-Bonnet: total curvature 2 pi chi, i.e. disclination angles add up to 360 deg x chi
    let backend = AppBackend::new();
    let kagome = lattice(&backend, "kagome");
    for (t, expected, word) in [(SurfaceTargetUi::Sphere, 720.0, "+720"), (SurfaceTargetUi::ProjectivePlane, 360.0, "+360")] {
        let (m, _) = build_surface_mesh_auto(&kagome, t, CentreModeUi::Auto, 2).unwrap();
        let sum: f64 = m.cones.iter().map(|c| c.angle_deg).sum();
        assert!((sum - expected).abs() < 1e-6, "{t:?}: disclinations add up to {sum}");
        let note = cone_note(&m, t).unwrap();
        assert!(note.contains(word) && note.contains("Cone metric, not flat"), "{note}");
    }
    // and the tab shows the line next to the checks
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    st.target = SurfaceTargetUi::Sphere;
    st.activate_and_sync(&kagome, &backend, &mut canvas, &mut view);
    assert!(st.notes.readout.iter().any(|l| l.contains("Gauss-Bonnet")));
}

#[test]
fn the_3d_caption_says_not_isometric_except_for_flat_and_cylinder_pictures() {
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    for (t, labelled) in [
        (SurfaceTargetUi::Open, false),
        (SurfaceTargetUi::Cylinder, false),
        (SurfaceTargetUi::Torus, true),
        (SurfaceTargetUi::TorusHex, true),
        (SurfaceTargetUi::MobiusBand, true),
        (SurfaceTargetUi::KleinBottle, true),
    ] {
        st.target = t;
        st.auto_size = false;
        st.bc.explicit = true;
        st.activate_and_sync(&sq, &backend, &mut canvas, &mut view);
        assert!(st.current_mesh.is_some(), "{t:?}: {:?}", st.refusal_msg);
        assert_eq!(view.caption().contains("topological picture, not isometric"), labelled, "{t:?}");
    }
}

#[test]
fn legacy_sizes_still_map_to_the_same_options_and_a_refusal_clears_the_net() {
    // before the user edits the options, presets and sessions set (h, k): torus (h,0),(0,k) -> 12 kagome cells x 3 sites
    let backend = AppBackend::new();
    let kagome = lattice(&backend, "kagome");
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    st.target = SurfaceTargetUi::Torus;
    st.auto_size = false;
    (st.params.h, st.params.k) = (4, 3);
    st.activate_and_sync(&kagome, &backend, &mut canvas, &mut view);
    assert_eq!(st.current_mesh.as_ref().unwrap().sites.len(), 36);
    // edited options win over (h, k)
    st.bc = BcControls { explicit: true, t1: [6, 0], t2: [0, 2], ..BcControls::default() };
    st.activate_and_sync(&kagome, &backend, &mut canvas, &mut view);
    assert_eq!(st.current_mesh.as_ref().unwrap().sites.len(), 36, "6 x 2 cells x 3 sites");
    let chiral = lattice(&backend, "chiral_square");
    st.target = SurfaceTargetUi::KleinBottle;
    st.activate_and_sync(&chiral, &backend, &mut canvas, &mut view);
    assert!(st.refusal_msg.as_deref().unwrap().starts_with("IMPOSSIBLE"));
    assert_eq!(canvas.counts(), (0, 0));
    assert!(st.notes.picture_note.is_none());
}

#[test]
fn boundary_verdicts_are_fast_on_every_named_lattice() {
    // pilot (timing-discipline): the six default builds plus the glide search run on the UI thread once per lattice
    let backend = AppBackend::new();
    let mut worst = (0.0_f64, String::new());
    for name in ["square", "honeycomb", "kagome", "triangular", "chiral_square", "shastry_sutherland", "brick_pg"] {
        let item = lattice(&backend, name);
        let t0 = Instant::now();
        let s = summarize(&item).expect("summary");
        let dt = t0.elapsed().as_secs_f64();
        println!("# time summarize({name}) = {dt:.3} s");
        assert_eq!(s.verdicts.len(), BC_TARGETS.len());
        if dt > worst.0 {
            worst = (dt, name.to_string());
        }
    }
    assert!(worst.0 < 3.0, "slowest summary {} s on {}", worst.0, worst.1);
}

#[test]
fn open_patches_of_the_square_and_triangular_tilings_match_the_independent_counts() {
    // seed2d.py (independent): vertex seed after n steps: {4,4} 4(n+1)^2 squares, {3,6} 6(n+1)^2 triangles
    let backend = AppBackend::new();
    for (name, per) in [("square", 4usize), ("triangular", 6)] {
        let item = lattice(&backend, name);
        for n in [1, 4] {
            let ctl = BcControls { layers: n, ..BcControls::default() };
            let (m, rep, _) = build_bc(&item, SurfaceTargetUi::Open, &ctl).unwrap();
            assert_eq!(m.faces.len(), per * (n as usize + 1).pow(2), "{name} n = {n}");
            assert!(rep.ok && euler(&m) == 1, "{name}: a disk");
        }
    }
}

#[test]
fn lattices_with_a_mirror_or_glide_get_a_twist_at_the_default_options() {
    // the groups of these lattices (p4m, p6m, p6m, pg) contain a mirror or a glide, so Moebius and Klein exist
    // (plan §2.5); the verdict shown before any option is touched must not be a refusal
    let backend = AppBackend::new();
    for name in ["square", "honeycomb", "kagome", "brick_pg"] {
        let s = summarize(&lattice(&backend, name)).unwrap();
        assert!(!s.glides.is_empty(), "{name}: glide classes");
        for t in [SurfaceTargetUi::MobiusBand, SurfaceTargetUi::KleinBottle] {
            let v = s.verdict(t).unwrap();
            assert_eq!(v.kind, VerdictKindUi::Ok, "{name} {t:?}: {}", v.text);
        }
    }
}

#[test]
fn options_map_from_h_k_again_once_explicit_is_cleared() {
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    st.choose_target(SurfaceTargetUi::Torus);
    assert!(st.bc.explicit && !st.auto_size);
    // what `apply_surfaces` / a manual-size preset do: new (h, k) and the options map from them again
    (st.params.h, st.params.k) = (2, 3);
    st.bc.explicit = false;
    st.activate_and_sync(&sq, &backend, &mut canvas, &mut view);
    assert_eq!(st.current_mesh.as_ref().unwrap().sites.len(), 6, "{{4,4}} torus (2,0),(0,3): 6 cells, 1 site each");
    // the sphere goes back to the table sizes
    st.choose_target(SurfaceTargetUi::Sphere);
    assert!(st.auto_size);
}
