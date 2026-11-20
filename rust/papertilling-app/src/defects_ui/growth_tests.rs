//! C1 WP-U7 tests: defected-lattice growth in the Defects catalogue and the Stone–Wales panel.
//!
//! Known answers, never read from the code under test:
//! - {4,4} with a ±90° disclination whose apex is the face centre 1b, grown from the face seed: |S_n| = 1 + k·n(n+1),
//!   k = 3 for +90° (a quarter removed) and k = 5 for −90° (a quarter inserted). By hand: the perfect square lattice
//!   gives (2n+1)² = 1 + 4n(n+1) = 1 + 4·(the n(n+1) faces per quarter); a cone with k quarters gives 1 + k·n(n+1).
//!   Independently checked by the scratch script `cone44.py` (quarter-planes glued round the apex, no repository
//!   code): [1, 7, 19, 37, 61] and [1, 11, 31, 61, 101].
//!   The Dirichlet seed at 1b is that one face (the Wigner–Seitz cell of a face centre is the face itself, and the
//!   neighbouring face centroids lie at distance 1 > 1/2), so the face seed and the Q8 seed agree.
//! - Stone–Wales on the honeycomb, grown from the 4 faces about the rotated bond: (3n+4)(n+1) = 4, 14, 30, 52, 80
//!   (independent `sw_hex.py`, a triangular-lattice grower).
//! - χ = V − E + F = 1 for every complete layer (a disk).

use papertilling_core::defect::defect_growth::{DEFECT_MAX_CELLS, grow_point_defects};
use papertilling_core::defect::grain_boundary::perfect_disk;
use papertilling_core::defect::stone_wales::stone_wales_flip;
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::plane::inflation::PlaneTiling;

use super::{GrowthOutcome, GrowthRequest, GrowthUi, SeedChoice, WALLS_REFUSED, build_grown, complete_label, grow_stone_wales};
use crate::backend::{AppBackend, CatalogueItemUi, DefectSelectionUi};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::defects_tab::{DefectSubTab, DefectsTabState};
use crate::live_ops::defect_context;
use crate::live_ops_ao::Defect2dMode;

fn named(backend: &AppBackend, k: &str) -> CatalogueItemUi {
    backend.find_named(k).unwrap_or_else(|| panic!("{k} is a named lattice")).clone()
}

/// Index of the square lattice's disclination class of `angle` at the face centre 1b = (1/2, 1/2) (fractional).
fn square_1b(item: &CatalogueItemUi, angle: f64) -> usize {
    let ctx = defect_context(item).unwrap();
    let (a1, a2) = (ctx.cell.a1, ctx.cell.a2);
    let det = a1[0] * a2[1] - a1[1] * a2[0];
    ctx.cat
        .disclinations
        .iter()
        .position(|e| {
            let p = e.point;
            let f = [(p[0] * a2[1] - p[1] * a2[0]) / det, (a1[0] * p[1] - a1[1] * p[0]) / det];
            let half = |x: f64| (x - x.floor() - 0.5).abs() < 1e-6;
            (e.angle - angle).abs() < 1e-9 && half(f[0]) && half(f[1])
        })
        .unwrap_or_else(|| panic!("square has a {angle:+} deg class at 1b"))
}

#[test]
fn square_disclination_at_a_face_centre_grows_as_the_cone_formula() {
    let backend = AppBackend::new();
    let sq = named(&backend, "square");
    for (angle, k) in [(90.0, 3usize), (-90.0, 5usize)] {
        let i = square_1b(&sq, angle);
        let req = GrowthRequest { layers: 4, seed: SeedChoice::DirichletCore };
        let (crystal, readout, v) = build_grown(&sq, &DefectSelectionUi::Disclination(i), req, Defect2dMode::RelaxedDrawing).unwrap();
        let want: Vec<usize> = (0..=4usize).map(|n| 1 + k * n * (n + 1)).collect();
        let got: Vec<usize> = v.stats.iter().map(|s| s.cells).collect();
        assert_eq!(got, want, "{angle:+}: |S_n| (cone with {k} quarters)");
        assert!(v.stats.iter().all(|s| s.euler == 1), "{angle:+}: χ = 1 on every complete layer");
        // the layer count shown is the core's own count of complete layers (independent call, same inputs)
        let ctx = defect_context(&sq).unwrap();
        let e = &ctx.cat.disclinations[i];
        let spec = [DefectSpec::Disclination { angle_deg: e.angle, at: e.point, cut: None }];
        let core = grow_point_defects(&ctx.cell, &spec, 4, None, true, DEFECT_MAX_CELLS).unwrap();
        assert_eq!(v.complete, core.result.complete_layers());
        assert_eq!(v.complete, 5);
        assert!(complete_label(&v).contains("0..4") && readout.title.contains(&complete_label(&v)), "{}", readout.title);
        // colouring: the seed face's corners are layer 0; sites outside S_4 are uncoloured (grey)
        assert_eq!(v.site_layer.len(), crystal.pos.len());
        assert_eq!(v.site_layer.iter().filter(|l| **l == Some(0)).count(), if angle > 0.0 { 3 } else { 5 }, "the core face is a {}-gon", 4 - 1 + if angle > 0.0 { 0 } else { 2 });
        assert!(v.site_layer.iter().any(Option::is_none), "the crystal is larger than S_4");
        let ov = v.overlay();
        assert_eq!(ov.bond_rgba.as_ref().map(Vec::len), Some(crystal.edges.len()));
        assert_eq!(ov.legend.as_ref().map(|l| l.max_val), Some(4.0));
    }
}

#[test]
fn nearest_site_seed_is_offered_under_more() {
    // A −90° apex at the face centre 1b: the sites nearest the core are the 5 corners of the core pentagon, and the
    // faces containing them are the pentagon and the 5 + 5 faces sharing an edge or a corner with it: 11 = |S_1| of
    // the face seed. So the nearest-site seed grows S'_n = S_{n+1}: 11, 31, 61 (cone formula above, k = 5).
    let backend = AppBackend::new();
    let sq = named(&backend, "square");
    let i = square_1b(&sq, -90.0);
    let req = GrowthRequest { layers: 2, seed: SeedChoice::NearestSites };
    let (_, _, v) = build_grown(&sq, &DefectSelectionUi::Disclination(i), req, Defect2dMode::Raw).unwrap();
    let got: Vec<usize> = v.stats.iter().map(|s| s.cells).collect();
    assert_eq!(got, [11, 31, 61][..v.complete].to_vec());
    assert!(v.complete >= 2, "{got:?} {}", v.note);
}

#[test]
fn walls_are_refused_with_the_reason() {
    let backend = AppBackend::new();
    let kag = named(&backend, "honeycomb");
    let req = GrowthRequest { layers: 2, seed: SeedChoice::DirichletCore };
    let err = build_grown(&kag, &DefectSelectionUi::Wall(0), req, Defect2dMode::Raw).unwrap_err();
    assert_eq!(err, WALLS_REFUSED);
    let ui = GrowthUi { on: true, ..Default::default() };
    assert_eq!(ui.request(DefectSubTab::Walls), None);
    assert_eq!(ui.request(DefectSubTab::Dislocations).map(|r| r.layers), Some(4), "default 4 layers");
}

#[test]
fn stone_wales_on_the_honeycomb_grows_like_the_bond_star() {
    let backend = AppBackend::new();
    let cell = defect_context(&named(&backend, "honeycomb")).unwrap().cell.clone();
    let c = PlaneTiling::new(&cell).unwrap().most_symmetric_point().point;
    let a = cell.a1[0].hypot(cell.a1[1]);
    let disk = perfect_disk(&cell, c, 12.0 * a);
    let mid = |k: usize| {
        let (u, v) = disk.edges[k];
        [0.5 * (disk.ref_pos[u][0] + disk.ref_pos[v][0]) - c[0], 0.5 * (disk.ref_pos[u][1] + disk.ref_pos[v][1]) - c[1]]
    };
    let k = (0..disk.edges.len()).min_by(|&i, &j| mid(i)[0].hypot(mid(i)[1]).total_cmp(&mid(j)[0].hypot(mid(j)[1]))).unwrap();
    let sw = stone_wales_flip(&disk, k, None).unwrap();
    let v = grow_stone_wales(&disk, &sw.crystal, k, &cell, 4).unwrap();
    let got: Vec<usize> = v.stats.iter().map(|s| s.cells).collect();
    assert_eq!(got, (0..=4usize).map(|n| (3 * n + 4) * (n + 1)).collect::<Vec<_>>());
    assert!(v.stats.iter().all(|s| s.euler == 1));
}

#[test]
fn defects_tab_shows_the_core_count_and_colours_by_layer() {
    // End to end through the tab's worker: ticking Layers on a dislocation builds the grown crystal, the readout
    // count equals the core's own count, and the canvas carries the layer colouring.
    let backend = AppBackend::new();
    let hc = named(&backend, "honeycomb");
    let mut st = DefectsTabState::new(&hc, &backend);
    let mut canvas = Canvas2D::new(Scene2D::CrystalPatch { crystal: st.last_crystal.clone(), title: String::new(), note: String::new() });
    st.growth.on = true;
    st.growth.layers = 3;
    st.rebuild_crystal(&hc, &backend, &mut canvas, true);
    st.wait_idle(&mut canvas);
    let GrowthOutcome::Shown(v) = st.growth.outcome.clone() else { panic!("not grown: {:?}", st.growth.outcome) };
    let ctx = defect_context(&hc).unwrap();
    let e = &ctx.cat.dislocations[0];
    let spec = [DefectSpec::Dislocation { b: e.b, at: Some(e.at), cut: None }];
    let core = grow_point_defects(&ctx.cell, &spec, 3, None, true, DEFECT_MAX_CELLS).unwrap();
    assert_eq!(v.complete, core.result.complete_layers());
    assert!(v.stats.iter().all(|s| s.euler == 1), "χ = 1");
    assert_eq!(st.last_crystal.pos.len(), core.crystal.pos.len(), "the picture is the grown crystal");
    let ov = canvas.gauge_overlay().expect("layer colouring on the canvas");
    assert!(ov.note_suffix.as_deref().is_some_and(|s| s.contains(&v.complete.to_string())));
    // Walls: the request is dropped and the plain wall is built
    st.sub_tab = DefectSubTab::Walls;
    st.rebuild_crystal(&hc, &backend, &mut canvas, true);
    st.wait_idle(&mut canvas);
    assert_eq!(st.growth.outcome, GrowthOutcome::Off);
}
