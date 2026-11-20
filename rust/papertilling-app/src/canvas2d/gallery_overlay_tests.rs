//! Gallery overlays (ITA symmetry elements, Voronoi cells): geometry counts by hand, layering on the canvas and in
//! the export, theming, and the view-only contract.

use super::Canvas2D;
use crate::backend::AppBackend;
use crate::gallery_tab::GalleryTabState;
use crate::scene2d::overlays::{GalleryOverlayToggles, build_gallery_overlay, centre_near, rotation_symbol};
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;
use crate::theme::tokens::{SCENE_ITA_GLIDE, SCENE_ITA_MIRROR, SCENE_ITA_ROTATION, SCENE_VORONOI_FILL};
use papertilling_core::types::Cell2D;

const BOTH: GalleryOverlayToggles = GalleryOverlayToggles { symmetry: true, voronoi: true };

fn named_cell(backend: &AppBackend, key: &str) -> (Cell2D, String, papertilling_core::types::Frame2D) {
    let it = backend.find_named(key).unwrap().clone();
    let (cell, _) = backend.build_cell(&it, &[]);
    (cell, it.group, it.frame)
}

#[test]
fn kagome_overlay_counts_match_the_p6mm_diagram_and_the_sites() {
    let backend = AppBackend::new();
    let (cell, group, frame) = named_cell(&backend, "kagome");
    assert_eq!(group, "p6m");
    let ov = build_gallery_overlay(&cell, &group, frame, 1, BOTH);
    assert!(ov.note.is_none(), "{:?}", ov.note);
    // The kagome cell has a site (a 2-fold 3c point of p6mm) at its origin modulo the lattice, and its frame puts
    // the 6-fold point at fractional (0, 1/2). On that closed cell: 6-fold at (0, 1/2), (1, 1/2) -> 2; the two
    // 3-fold 2b points inside; 2-fold at the 4 corners and (1/2, 0), (1/2, 1), (1/2, 1/2) -> 7. Total 11.
    let det = cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0];
    let frac = |p: [f64; 2]| [(p[0] * cell.a2[1] - p[1] * cell.a2[0]) / det, (cell.a1[0] * p[1] - cell.a1[1] * p[0]) / det];
    let on_lattice = |f: [f64; 2]| (f[0] - f[0].round()).abs() < 1e-9 && (f[1] - f[1].round()).abs() < 1e-9;
    assert!(cell.basis.iter().any(|&b| on_lattice(frac(b))), "precondition: a site at the cell origin mod the lattice");
    let o = frac(frame.origin);
    assert!((o[0] - o[0].round()).abs() < 1e-9 && (o[1] - o[1].round() - 0.5).abs() % 1.0 < 1e-9, "precondition: 6-fold at (0, 1/2), got {o:?}");
    let by_order = |n: u8| ov.centres.iter().filter(|c| c.order == n).count();
    assert_eq!((by_order(6), by_order(3), by_order(2)), (2, 2, 7));
    assert_eq!(ov.over.polygons.len(), 11, "one symbol per centre");
    assert!(ov.over.polygons.iter().all(|p| p.fill == SCENE_ITA_ROTATION));
    assert!(ov.over.lines.iter().any(|l| l.color == SCENE_ITA_MIRROR), "p6mm has mirrors");
    // kagome: 3 sites per cell, one Voronoi polygon each, per drawn cell
    assert_eq!(ov.under.polygons.len(), 3);
    let ov3 = build_gallery_overlay(&cell, &group, frame, 3, BOTH);
    assert_eq!(ov3.under.polygons.len(), 27, "3 sites x 9 cells");
    assert!(ov3.under.polygons.iter().all(|p| p.fill == SCENE_VORONOI_FILL));
}

#[test]
fn p6mm_with_origin_on_the_six_fold_point_matches_the_ita_diagram() {
    // triangular lattice in the ITA hexagonal setting (gamma = 120 deg), identity frame: origin on the 6-fold point.
    // Closed unit cell: 6-fold at the 4 corners, 3-fold at 2b (1/3,2/3), (2/3,1/3), 2-fold at 3c (1/2,0), (0,1/2),
    // (1/2,1/2) plus the far-edge copies (1/2,1), (1,1/2): 4 + 2 + 5 = 11.
    let h = 3f64.sqrt() / 2.0;
    let cell = Cell2D { a1: [1.0, 0.0], a2: [-0.5, h], basis: vec![[0.0, 0.0]], bonds: Vec::new() };
    let ov = build_gallery_overlay(&cell, "p6m", papertilling_core::types::Frame2D::default(), 1, GalleryOverlayToggles { symmetry: true, voronoi: false });
    let by_order = |n: u8| ov.centres.iter().filter(|c| c.order == n).count();
    assert_eq!((by_order(6), by_order(3), by_order(2)), (4, 2, 5));
    let six: Vec<&str> = ov.centres.iter().filter(|c| c.order == 6).map(|c| c.wyckoff.as_str()).collect();
    assert!(six.iter().all(|w| *w == "1a"), "{six:?}");
}

#[test]
fn glide_only_and_mirror_free_groups_draw_the_right_line_kinds() {
    let backend = AppBackend::new();
    // brick_pg is in pg: glides, no mirrors; chiral_square is in p4: no lines at all, only centres
    let (cell, group, frame) = named_cell(&backend, "brick_pg");
    assert_eq!(group, "pg");
    let ov = build_gallery_overlay(&cell, &group, frame, 2, GalleryOverlayToggles { symmetry: true, voronoi: false });
    assert!(!ov.over.lines.is_empty() && ov.over.lines.iter().all(|l| l.color == SCENE_ITA_GLIDE && l.dashed && l.arrow_mid));
    assert!(ov.centres.is_empty() && ov.under.polygons.is_empty());
    let (cell, group, frame) = named_cell(&backend, "chiral_square");
    let ov = build_gallery_overlay(&cell, &group, frame, 2, GalleryOverlayToggles { symmetry: true, voronoi: false });
    assert!(ov.over.lines.is_empty(), "p4 has no mirror or glide");
    assert!(ov.centres.iter().any(|c| c.order == 4) && ov.centres.iter().any(|c| c.order == 2));
}

#[test]
fn rotation_symbols_have_the_ita_shapes() {
    let s = 1.0;
    assert_eq!(rotation_symbol([0.0, 0.0], 3, s).len(), 3);
    assert_eq!(rotation_symbol([0.0, 0.0], 4, s).len(), 4);
    assert_eq!(rotation_symbol([0.0, 0.0], 6, s).len(), 6);
    assert!(rotation_symbol([0.0, 0.0], 5, s).is_empty());
    // lens: tips at (+-size, 0), as wide as 2 size, half as tall (arcs of half-angle 60 deg: height size/sqrt3 each side)
    let lens = rotation_symbol([0.0, 0.0], 2, s);
    let max_x = lens.iter().map(|p| p[0].abs()).fold(0.0, f32::max);
    let max_y = lens.iter().map(|p| p[1].abs()).fold(0.0, f32::max);
    assert!((max_x - 1.0).abs() < 1e-5, "{max_x}");
    assert!((max_y - 1.0 / 3f32.sqrt()).abs() < 1e-5, "{max_y}");
}

#[test]
fn hover_picks_the_nearest_centre_within_the_radius() {
    let backend = AppBackend::new();
    let (cell, group, frame) = named_cell(&backend, "square");
    let ov = build_gallery_overlay(&cell, &group, frame, 1, GalleryOverlayToggles { symmetry: true, voronoi: false });
    let c = centre_near(&ov.centres, [0.49, 0.52], 0.1).expect("the 4-fold centre at the cell centre");
    assert_eq!(c.order, 4);
    assert!((c.pos[0] - 0.5).abs() < 1e-9 && (c.pos[1] - 0.5).abs() < 1e-9);
    assert!(centre_near(&ov.centres, [0.25, 0.25], 0.1).is_none());
}

#[test]
fn canvas_draws_voronoi_under_and_symmetry_over_and_export_keeps_that_order() {
    let backend = AppBackend::new();
    let (cell, group, frame) = named_cell(&backend, "kagome");
    let mut canvas = Canvas2D::new(Scene2D::PeriodicCell { cell: cell.clone(), repeat: 1, show_unit_cell: false });
    let base_polys = canvas.compiled.polygons.len();
    let ov = build_gallery_overlay(&cell, &group, frame, 1, BOTH);
    let n_under = ov.under.polygons.len();
    canvas.set_gallery_overlay(Some(ov));
    assert_eq!(canvas.compiled.polygons.len(), base_polys + n_under);
    assert!(canvas.compiled.polygons[..n_under].iter().all(|p| p.fill == SCENE_VORONOI_FILL), "Voronoi first = underneath");
    let dir = std::env::temp_dir().join(format!("pt_overlay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kagome.svg");
    canvas.export_to_path(&path, 1).unwrap();
    let svg = std::fs::read_to_string(&path).unwrap();
    let last_site = svg.rfind("<circle").expect("sites");
    let first_symbol = svg.find("fill=\"#2e7d32\"").expect("rotation symbols (46,125,50)");
    assert!(first_symbol > last_site, "symbols are drawn over the sites in the export");
    // a new scene drops the overlay
    canvas.set_scene(Scene2D::PeriodicCell { cell, repeat: 1, show_unit_cell: false }, false);
    assert!(canvas.gallery_overlay.is_none());
    assert_eq!(canvas.compiled.polygons.len(), base_polys);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn studio_maps_overlay_colours_and_paper_keeps_them() {
    let (paper, studio) = (ThemeTokens::paper(), ThemeTokens::studio());
    let raw = |c: [u8; 4]| egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
    assert_eq!(paper.remap_scene_ink(SCENE_ITA_MIRROR), raw(SCENE_ITA_MIRROR));
    assert_eq!(paper.scene_fill(SCENE_VORONOI_FILL), raw(SCENE_VORONOI_FILL));
    // egui stores premultiplied colours: compare the unmultiplied channels
    let m = studio.remap_scene_ink(SCENE_ITA_MIRROR).to_srgba_unmultiplied();
    let want = studio.ita_mirror.to_srgba_unmultiplied();
    assert_eq!(m[..3], want[..3]);
    assert_eq!(m[3], SCENE_ITA_MIRROR[3].min(want[3]));
    assert_eq!(studio.scene_fill(SCENE_VORONOI_FILL), studio.voronoi_fill);
    assert_eq!(studio.scene_fill(SCENE_ITA_ROTATION), studio.ita_rotation);
}

#[test]
fn toggling_overlays_is_view_only() {
    let backend = AppBackend::new();
    let mut canvas = Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 1,
        show_unit_cell: false,
    });
    let mut st = GalleryTabState::new(&backend, &mut canvas);
    let (rev, item) = (st.revision(), st.active_item().clone());
    st.set_overlays(BOTH, &mut canvas);
    assert!(canvas.gallery_overlay.is_some());
    assert_eq!(st.revision(), rev, "an overlay never changes the lattice");
    assert_eq!(st.active_item().recipe, item.recipe);
    // patch size follows: the default honeycomb (C1) has 2 sites per cell, so a 3 x 3 patch -> 18 Voronoi cells
    assert_eq!(canvas.gallery_overlay.as_ref().unwrap().under.polygons.len(), 2 * st.repeat as usize * st.repeat as usize);
    st.set_overlays(GalleryOverlayToggles::default(), &mut canvas);
    assert!(canvas.gallery_overlay.is_none());
}

#[test]
fn overlay_extent_joins_the_fit_so_nothing_is_cut_off() {
    // one-site square cell whose canonical bonds point to negative cells: the patch bbox and the overlay window differ
    let cell = Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![
            papertilling_core::types::Bond2D { i: 0, j: 0, n1: -1, n2: 0, kind: 0 },
            papertilling_core::types::Bond2D { i: 0, j: 0, n1: 0, n2: -1, kind: 0 },
        ],
    };
    let mut canvas = Canvas2D::new(Scene2D::PeriodicCell { cell: cell.clone(), repeat: 3, show_unit_cell: false });
    let ov = build_gallery_overlay(&cell, "p4m", papertilling_core::types::Frame2D::default(), 3, BOTH);
    let (ou, oo) = (ov.under.bbox, ov.over.bbox);
    canvas.set_gallery_overlay(Some(ov));
    let (lo, hi) = canvas.compiled.bbox;
    for (a, b) in [ou, oo] {
        assert!(lo[0] <= a[0] && lo[1] <= a[1] && hi[0] >= b[0] && hi[1] >= b[1], "{a:?}..{b:?} outside fit {lo:?}..{hi:?}");
    }
}

#[test]
fn export_draws_rotation_symbols_over_the_mirror_lines() {
    let backend = AppBackend::new();
    let (cell, group, frame) = named_cell(&backend, "square");
    let mut canvas = Canvas2D::new(Scene2D::PeriodicCell { cell: cell.clone(), repeat: 1, show_unit_cell: false });
    canvas.set_gallery_overlay(Some(build_gallery_overlay(&cell, &group, frame, 1, BOTH)));
    let dir = std::env::temp_dir().join(format!("pt_overlay_order_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("square.svg");
    canvas.export_to_path(&path, 1).unwrap();
    let svg = std::fs::read_to_string(&path).unwrap();
    let last_mirror = svg.rfind("stroke=\"#c62828\"").expect("p4mm mirrors (198,40,40)");
    let first_symbol = svg.find("fill=\"#2e7d32\"").expect("rotation symbols");
    assert!(first_symbol > last_mirror, "symbols after the mirror lines, as on the canvas");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn defects_under_coordination_ring_keeps_its_studio_colour() {
    // [198, 40, 40, 235] is the Defects ring stroke (scene_crystal.rs); it must not be taken for the ITA mirror role
    let studio = ThemeTokens::studio();
    let ring = studio.remap_scene_ink([198, 40, 40, 235]).to_srgba_unmultiplied();
    let mirror = studio.ita_mirror.to_srgba_unmultiplied();
    assert_ne!(ring[..3], mirror[..3]);
    assert_ne!(SCENE_ITA_MIRROR, [198, 40, 40, 235]);
}

#[test]
fn legend_is_lifted_above_the_circuit_inset_when_they_meet() {
    use super::circuit::reference_inset_rect;
    use super::gauge_overlay::legend_clear_of;
    use super::overlay_legend::categorical_legend_rect;
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(300.0, 420.0));
    let inset = reference_inset_rect(rect).unwrap();
    let legend = categorical_legend_rect(rect, 2).unwrap();
    assert!(legend.intersects(inset), "precondition: they meet on a 300 px canvas");
    let moved = legend_clear_of(legend, inset, rect).unwrap();
    assert!(!moved.intersects(inset));
    assert!((moved.bottom() - (inset.top() - 8.0)).abs() < 1e-3);
    // wide canvas: untouched
    let wide = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(900.0, 600.0));
    let l = categorical_legend_rect(wide, 2).unwrap();
    assert_eq!(legend_clear_of(l, reference_inset_rect(wide).unwrap(), wide), Some(l));
}

