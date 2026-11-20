#![allow(clippy::field_reassign_with_default)]
//! Atlas points-catalogue view tests (C1 WP-U4). Expected values are hand-derived (stated per test).

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d, build_net_3d};

use super::*;
use crate::app::{LatticesView, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::atlas_tab::space3d_view::species_label;
use crate::theme::{StudioTheme, typography};

fn sc_choice(layers: u32) -> CatalogueChoice {
    CatalogueChoice { sg: 221, idx: 0, seed: SeedKind::Cell, layers }
}

fn sp221() -> Space3dPanelState {
    let mut sp = Space3dPanelState::default();
    sp.selected_sg = 221;
    sp
}

#[test]
fn without_a_choice_the_canonical_net_is_drawn_unchanged() {
    let mut sp = Space3dPanelState::default();
    let want = build_net_3d(sp.selected_net, LineDefect3d::None);
    assert!(sp.view.catalogue.is_none());
    assert_eq!(sp.ensure_net().positions, want.positions);
    assert_eq!(sp.ensure_net().bonds, want.bonds);
}

#[test]
fn a_chosen_entry_is_grown_drawn_and_framed() {
    let mut sp = sp221();
    sp.view.picked = Some(3);
    assert!(sp.choose_catalogue(sc_choice(1)));
    assert_eq!(sp.view.picked, None, "a new drawing clears the pick");
    // Pm-3m entry 1 is sc: one Dirichlet cube and its 26 touching cubes, 3^3 = 27 sites (hand)
    let n = sp.ensure_net().positions.len();
    assert_eq!(n, 27);
    // the frame is the entry's lattice: every sc site has integer fractional coordinates (hand: site 1a at 0)
    let fr = sp.frame();
    for p in sp.ensure_net().positions.clone() {
        let f = fr.fractional(p);
        assert!(f.iter().all(|x| (x - x.round()).abs() < 1e-9), "{f:?}");
    }
    // the cube has 3 * 3 * 2 * 3 = 54 unit bonds (hand), the centre site 6 of them
    let net = sp.ensure_net();
    assert_eq!(net.bonds.len(), 54);
    assert_eq!(sp.view.catalogue.as_ref().unwrap().orbit_label(0), "orbit 1 (Wyckoff 1a)");
    // layers 2: 5^3 = 125 (hand)
    assert!(sp.choose_catalogue(sc_choice(2)));
    assert_eq!(sp.ensure_net().positions.len(), 125);
}

#[test]
fn a_refused_choice_keeps_the_drawing_and_shows_the_reason() {
    let mut sp = sp221();
    assert!(sp.choose_catalogue(sc_choice(1)));
    assert!(!sp.choose_catalogue(CatalogueChoice { idx: 99, ..sc_choice(1) }));
    assert!(sp.view.catalogue_error.as_deref().is_some_and(|e| e.starts_with("Refused")));
    assert_eq!(sp.ensure_net().positions.len(), 27);
    assert!(sp.choose_catalogue(sc_choice(0)));
    assert!(sp.view.catalogue_error.is_none());
}

#[test]
fn another_group_drops_the_entry() {
    let mut sp = sp221();
    assert!(sp.choose_catalogue(sc_choice(1)));
    sp.selected_sg = 136;
    let want = build_net_3d(sp.selected_net, LineDefect3d::None);
    assert_eq!(sp.ensure_net().positions, want.positions);
    assert!(sp.view.catalogue.is_none());
}

#[test]
fn perovskite_labels_follow_the_coordination() {
    // ABX3 with B at the cube corner: B has 6 X at a/2 (octahedral), A has 12 X at a/sqrt(2) (hand). The kind
    // whose interior site has 6 anions at 1/2 must be captioned the B site.
    let p = build_net_3d(CanonicalNet3d::Perovskite, LineDefect3d::None);
    let d = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
    let centre = p.positions.iter().position(|q| d(*q, [0.0, 0.0, 0.0]) < 1e-9).expect("site at the block centre");
    let anions_at_half = p
        .positions
        .iter()
        .zip(&p.site_kinds)
        .filter(|(q, k)| **k == 2 && (d(**q, p.positions[centre]) - 0.5).abs() < 1e-9)
        .count();
    assert_eq!(anions_at_half, 6);
    assert_eq!(species_label(CanonicalNet3d::Perovskite, p.site_kinds[centre]), "B site (octahedral)");
    let a_kind = 1 - p.site_kinds[centre];
    assert_eq!(species_label(CanonicalNet3d::Perovskite, a_kind), "A site");
}

#[test]
fn entry_labels_and_stop_text() {
    let g = points_group(227).unwrap();
    // Fd-3m entry 1 is diamond: Wyckoff 8a, 2 sites per primitive cell, degree 4 (textbook)
    assert_eq!(entry_label(0, &g.entries[0]), "1. 8a | 2 sites/cell | degree 4");
    let t = stop_text_cap(Some(&Stopped::CellCap { layer: 3, partial: 777 }), 9);
    assert_eq!(t.as_deref(), Some("Stopped: layer 3 would pass 9 cells and was not built."));
    assert!(!t.unwrap().contains("777"), "CellCap partial lower bound must never print as a count");
    assert!(stop_text_cap(Some(&Stopped::Closed { layer: 2 }), 9).unwrap().contains("Closed at layer 2"));
    let inf = stop_text_cap(Some(&Stopped::Infinite { layer: 1, reason: "ideal vertex" }), 9).unwrap();
    assert!(inf.contains("ideal vertex") && !inf.contains("Infinite {"));
    let prec = stop_text_cap(Some(&Stopped::Precision { layer: 4, gap: 1e-8 }), 9).unwrap();
    assert!(prec.contains("precision") && !prec.contains("Precision {"));
    let rim = stop_text_cap(Some(&Stopped::Rim { layer: 2 }), 9).unwrap();
    assert!(rim.contains("outer boundary") && !rim.contains("Rim {"));
    assert!(stop_text_cap(None, 9).is_none());
}

#[test]
fn kittest_points3d_catalogue_no_plan_leaks_and_reselect_canonical_net() {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    app.atlas.sub_tab = AtlasSubTab::SpaceGroups3D;
    app.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
    let mut h = Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, app: &mut PapertillingApp| {
                let _ = app.frame_ui(ui);
            },
            app,
        );
    h.run_steps(4);

    assert!(h.state().atlas.pristine_only);
    assert_eq!(h.state().atlas.sub_tab, AtlasSubTab::SpaceGroups3D);
    assert_eq!(h.state().atlas.space3d.selected_sg, 136);
    assert_eq!(h.state().atlas.space3d.selected_net, CanonicalNet3d::Rutile136);
    assert!(h.state().atlas.space3d.view.catalogue.is_none());

    for leaked in ["plan T4", "plan Q1", "plan Q10"] {
        assert!(
            h.query_all_by_label_contains(leaked).next().is_none(),
            "internal plan code '{leaked}' leaked into SpaceGroups3D UI"
        );
    }
    assert!(!points_note().contains("plan T4") && !points_note().contains("plan Q10"));

    let g136 = points_group(136).unwrap();
    assert_eq!(g136.shown().len(), 7);
    let first_lbl = entry_label(0, &g136.entries[0]);
    for (k, e) in g136.shown().iter().enumerate() {
        let lbl = entry_label(k, e);
        assert!(h.query_by_label(&lbl).is_some(), "missing shown entry #{k}: {lbl}");
    }
    h.get_by_label(&first_lbl).click();
    h.run_steps(3);
    assert!(h.state().atlas.space3d.view.catalogue.is_some());
    assert_eq!(h.state().atlas.space3d.selected_net, CanonicalNet3d::Rutile136);

    // Re-selecting the already-stored canonical net from "3D Crystal Net" drops the catalogue entry.
    h.get_all_by_role(egui::accesskit::Role::ComboBox)
        .find(|n| n.value().as_deref() == Some("Catalogue entry (pick canonical…)"))
        .expect("3D Crystal Net ComboBox shows catalogue placeholder")
        .click();
    h.run_steps(2);
    h.get_by_label(CanonicalNet3d::Rutile136.label()).click();
    h.run_steps(3);
    assert!(h.state().atlas.space3d.view.catalogue.is_none());
    assert_eq!(h.state().atlas.space3d.selected_net, CanonicalNet3d::Rutile136);
    assert!(!h.state().atlas.dirty);

    // Clicking the #136 quick button while a #136 catalogue entry is active also returns to canonical Rutile.
    h.get_by_label(&first_lbl).click();
    h.run_steps(3);
    assert!(h.state().atlas.space3d.view.catalogue.is_some());
    h.get_by_label("#136 P4_2/mnm (Rutile)").click();
    h.run_steps(3);
    assert!(h.state().atlas.space3d.view.catalogue.is_none());
}
