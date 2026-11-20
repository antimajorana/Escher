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
    // ({4,4} needs L >= 3 in each direction to avoid double edges; 3 x 4 = 12 cells, 1 site each)
    (st.params.h, st.params.k) = (3, 4);
    st.bc.explicit = false;
    st.activate_and_sync(&sq, &backend, &mut canvas, &mut view);
    assert_eq!(st.current_mesh.as_ref().unwrap().sites.len(), 12, "{{4,4}} torus (3,0),(0,4): 12 cells, 1 site each");
    // the sphere goes back to the table sizes
    st.choose_target(SurfaceTargetUi::Sphere);
    assert!(st.auto_size);
}

#[test]
fn auto_size_still_sets_not_isometric_picture_note_and_exact_view_flags() {
    use papertilling_core::surface::report_bc::NOT_ISOMETRIC_NOTE;
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    for (t, want_note, want_exact) in [
        (SurfaceTargetUi::Open, None, true),
        (SurfaceTargetUi::Cylinder, None, true),
        (SurfaceTargetUi::Torus, Some(NOT_ISOMETRIC_NOTE), true),
        (SurfaceTargetUi::TorusHex, Some(NOT_ISOMETRIC_NOTE), true),
        (SurfaceTargetUi::MobiusBand, Some(NOT_ISOMETRIC_NOTE), true),
        (SurfaceTargetUi::KleinBottle, Some(NOT_ISOMETRIC_NOTE), true),
        (SurfaceTargetUi::Sphere, None, false),
        (SurfaceTargetUi::ProjectivePlane, None, false),
        (SurfaceTargetUi::Genus2, Some(NOT_ISOMETRIC_NOTE), false),
        (SurfaceTargetUi::Genus3, Some(NOT_ISOMETRIC_NOTE), false),
    ] {
        st.target = t;
        st.auto_size = true; // verdict-cell click, 17x8 matrix click, preset, smoke
        st.activate_and_sync(&sq, &backend, &mut canvas, &mut view);
        assert!(st.current_mesh.is_some(), "{t:?}: {:?}", st.refusal_msg);
        assert_eq!(st.notes.picture_note, want_note, "{t:?} picture_note");
        assert_eq!(st.notes.exact_view, want_exact, "{t:?} exact_view");
        assert_eq!(view.caption().contains(NOT_ISOMETRIC_NOTE), want_note.is_some(), "{t:?} 3D caption");
    }
    // Gauss-Bonnet on genus >= 2 branched double covers: chi = 2 - 2g gives -720 deg (g=2) and -1440 deg (g=3)
    for (t, expected, word) in [(SurfaceTargetUi::Genus2, -720.0, "-720"), (SurfaceTargetUi::Genus3, -1440.0, "-1440")] {
        assert!(bc::plain_label(t).contains("(cone metric)"), "{t:?}: {}", bc::plain_label(t));
        let (m, _) = build_surface_mesh_auto(&sq, t, CentreModeUi::Auto, 3).unwrap();
        let sum: f64 = m.cones.iter().map(|c| c.angle_deg).sum();
        assert!((sum - expected).abs() < 1e-6, "{t:?}: branch cones add up to {sum}");
        let note = cone_note(&m, t).unwrap();
        assert!(note.contains(word) && note.contains("Cone metric, not flat"), "{note}");
    }
}

#[test]
fn oblique_tuned_to_p4m_locus_allows_k_zero_on_sphere_via_size_controls_for() {
    let backend = AppBackend::new();
    let mut canvas = blank_canvas();
    let oblique = lattice(&backend, "oblique");
    // Off-locus p2 has max rotation order 2 -> pillow requires k >= 1.
    assert_eq!(*size_controls_for(SurfaceTargetUi::Sphere, &oblique).k_range.start(), 1);

    // Tuned to b = 1, gamma = 90 deg, the realized cell is p4m (order 4) while item.group stays "p2".
    let mut gallery = crate::gallery_tab::GalleryTabState::new(&backend, &mut canvas);
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("oblique")).unwrap();
    gallery.select_named(idx, &backend, &mut canvas);
    gallery.set_param("b", 1.0, &mut canvas).unwrap();
    gallery.set_param("gamma", 90.0, &mut canvas).unwrap();
    let tuned = gallery.active_item().clone();
    assert_eq!(tuned.group, "p2");
    assert_eq!(*size_controls_for(SurfaceTargetUi::Sphere, &tuned).k_range.start(), 0);

    // Cube sphere (h, k) = (2, 0) with face-centre corners: 6 faces x 4 sites = 24 V, 4*24/2 = 48 E, 2+48-24 = 26 F.
    let (mut st, mut view) = (SurfacesTabState::new(), View3dState::default());
    st.target = SurfaceTargetUi::Sphere;
    st.auto_size = false;
    (st.params.h, st.params.k) = (2, 0);
    st.activate_and_sync(&tuned, &backend, &mut canvas, &mut view);
    assert_eq!((st.params.h, st.params.k), (2, 0), "(2, 0) must not clamp to (2, 1)");
    let m = st.current_mesh.as_ref().expect("sphere (2, 0) on tuned p4m locus");
    assert_eq!((m.sites.len(), m.bonds.len(), m.faces.len(), m.cones.len()), (24, 48, 26, 8));
}

#[test]
fn open_patch_at_layer_zero_gives_three_hexagon_vertex_star_on_honeycomb() {
    // Hand count of S_0 (vertex star of {6,3}): 3 hexagons sharing 1 central degree-3 vertex and 3 spokes ->
    // V = 1 + 3 + 9 = 13, E = 3 + 12 = 15, F = 3, chi = 1.
    let backend = AppBackend::new();
    let hc = lattice(&backend, "honeycomb");
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    st.bc.layers = 0;
    st.bc.explicit = true;
    st.clamp_params(&hc);
    assert_eq!(st.bc.layers, 0, "layer 0 (S_0) must not clamp up to 1");
    st.activate_and_sync(&hc, &backend, &mut canvas, &mut view);
    let m = st.current_mesh.as_ref().expect("honeycomb S_0 built");
    assert_eq!((m.sites.len(), m.bonds.len(), m.faces.len()), (13, 15, 3));
    assert!(st.current_report.as_ref().unwrap().ok && euler(m) == 1 && !m.closed);
}

#[test]
fn non_simple_quotient_returns_not_found_too_small_never_impossible() {
    // {4,4} cylinder: wrap [1, 0] closes a1 into a self-loop; wrap [2, 0] maps +a1 and -a1 to a double bond.
    let backend = AppBackend::new();
    let sq = lattice(&backend, "square");
    for wrap in [[1, 0], [2, 0]] {
        let ctl = BcControls { explicit: true, wrap, ..BcControls::default() };
        let err = build_bc(&sq, SurfaceTargetUi::Cylinder, &ctl).unwrap_err();
        assert!(err.starts_with("NOT FOUND (too small)"), "{wrap:?}: {err}");
        assert!(!err.contains("IMPOSSIBLE"), "{wrap:?}: {err}");
        assert_eq!(bc::kind_of(&err), VerdictKindUi::NotFound, "{wrap:?}: {err}");
    }
    let (mut st, mut canvas, mut view) = (SurfacesTabState::new(), blank_canvas(), View3dState::default());
    st.choose_target(SurfaceTargetUi::Cylinder);
    st.bc.wrap = [1, 0];
    st.activate_and_sync(&sq, &backend, &mut canvas, &mut view);
    let msg = st.refusal_msg.as_deref().expect("refusal message");
    assert!(msg.starts_with("NOT FOUND (too small)") && !msg.contains("IMPOSSIBLE"), "{msg}");
    assert_eq!(bc::kind_of(msg), VerdictKindUi::NotFound);
    assert!(st.current_mesh.is_none());
}

#[test]
fn deck_open_and_torus_hex_stat_cards_and_viewport_headers_reflect_bc_verdicts_and_exact_view() {
    use egui_kittest::kittest::Queryable;
    use crate::analysis_mode::tests::harness;
    use crate::game_menu::test_util::{equip_state, open_deck};
    use crate::game_menu::{CardId, Realm};

    let mut h = harness();
    h.run_steps(2);
    let cards = h.state_mut().deck_cards(Realm::Manifolds, 0, "");
    open_deck(&mut h);
    h.state_mut().deck_select_realm(Realm::Manifolds);
    for t in [SurfaceTargetUi::Open, SurfaceTargetUi::TorusHex] {
        let idx = cards.iter().position(|c| c.id == CardId::Surface(t)).unwrap_or_else(|| panic!("missing {t:?} card"));
        assert_eq!(cards[idx].badge.map(|b| b.label()), Some("[PROVED OK]"), "{t:?}");
        assert!(cards[idx].subtitle.starts_with("OK"), "{t:?}: {}", cards[idx].subtitle);
        h.state_mut().deck.cursor[Realm::Manifolds.number() - 1] = idx;
        h.run_steps(2);
        assert!(h.query_all_by_label_contains("Status: [PROVED OK] explicit cell built").next().is_some(), "{t:?} stat card");
        assert!(h.query_all_by_label_contains("No verdict for this lattice.").next().is_none(), "{t:?} stat card");
    }

    // Equip Open: exact 2D domain header + inspector line, no non-isometric 3D suffix.
    equip_state(&mut h, CardId::Surface(SurfaceTargetUi::Open));
    assert!(h.query_all_by_label_contains("(exact isometric domain)").next().is_some());
    assert!(h.query_all_by_label_contains("Exact view: flat 2D fundamental domain with edge identifications").next().is_some());
    assert!(h.query_all_by_label_contains("— topological picture, not isometric").next().is_none());

    // Equip Torus: exact 2D domain header + non-isometric 3D suffix; Sphere: neither flat-domain nor non-isometric header.
    equip_state(&mut h, CardId::Surface(SurfaceTargetUi::Torus));
    assert!(h.query_all_by_label_contains("(exact isometric domain)").next().is_some());
    assert!(h.query_all_by_label_contains("— topological picture, not isometric").next().is_some());
    equip_state(&mut h, CardId::Surface(SurfaceTargetUi::Sphere));
    assert!(h.query_all_by_label_contains("(exact isometric domain)").next().is_none());
    assert!(h.query_all_by_label_contains("Exact view: flat 2D fundamental domain with edge identifications").next().is_none());
    assert!(h.query_all_by_label_contains("— topological picture, not isometric").next().is_none());
}
