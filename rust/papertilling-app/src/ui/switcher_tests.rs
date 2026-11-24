//! Quick Lattice Switcher: shortcut, palette selection on non-Gallery tabs, focus guard and the Recent LRU.

use egui::{Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::lattice_switcher::{LatticeSwitcher, RECENT_MAX};
use crate::app::{NavTab, PapertillingApp};
use crate::backend::AppBackend;
use crate::theme::{StudioTheme, typography};

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(
            |ui, app| {
                let _ = app.frame_ui(ui);
            },
            app,
        )
}

fn ctrl_k(h: &mut Harness<'static, PapertillingApp>) {
    h.key_press_modifiers(Modifiers::COMMAND, Key::K);
    h.run_steps(2);
}

fn wait_defects(h: &mut Harness<'static, PapertillingApp>) {
    for _ in 0..400 {
        h.step();
        let d = &h.state().defects;
        if !d.is_busy() && !d.initializing {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn ctrl_k_opens_the_palette_and_escape_closes_it() {
    let mut h = harness();
    h.run_steps(2);
    assert!(!h.state().switcher.open);
    ctrl_k(&mut h);
    assert!(h.state().switcher.open, "Ctrl+K opens the switcher");
    assert!(h.query_by_label("Named 21").is_some(), "palette lists the named section");
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(!h.state().switcher.open, "Esc closes the switcher");
}

#[test]
fn palette_pick_on_defects_tab_reloads_defects_without_a_tab_round_trip() {
    let mut h = harness();
    h.run_steps(2);
    h.state_mut().switch_tab(NavTab::Defects);
    wait_defects(&mut h);
    assert_eq!(h.state().gallery.active_item().named_key.as_deref(), Some("honeycomb"), "C1: the app opens on the p6m honeycomb");

    ctrl_k(&mut h);
    h.get_by_role(egui::accesskit::Role::TextInput).type_text("kagome p6m");
    h.run_steps(2);
    h.key_press(Key::Enter);
    h.run_steps(2);
    wait_defects(&mut h);

    let app = h.state();
    assert!(!app.switcher.open, "a pick closes the palette");
    assert_eq!(app.active_tab, NavTab::Defects, "the tab is kept");
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("kagome"));
    assert_eq!(app.defects.last_item.named_key.as_deref(), Some("kagome"), "Defects reloaded at once");
    assert!(app.defects.error.is_none(), "{:?}", app.defects.error);
    assert_eq!(app.switcher.recent.front().and_then(|r| r.named_key.as_deref()), Some("kagome"));
}

#[test]
fn palette_pick_by_ita_number_on_surfaces_tab_rebuilds_the_surface() {
    let mut h = harness();
    h.run_steps(2);
    h.state_mut().switch_tab(NavTab::Surfaces);
    h.run_steps(2);
    ctrl_k(&mut h);
    // "square 11": the named square lattice of p4m, ITA 11 (Named 21 section comes first after Recent)
    h.get_by_role(egui::accesskit::Role::TextInput).type_text("square 11");
    h.run_steps(2);
    h.key_press(Key::Enter);
    h.run_steps(3);
    let app = h.state();
    assert_eq!(app.gallery.active_item().group, "p4m");
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("square"));
    assert!(app.surfaces.active_disp.starts_with(&app.gallery.active_item().label), "{}", app.surfaces.active_disp);
}

#[test]
fn ctrl_k_is_ignored_while_a_text_field_has_focus() {
    let mut h = harness();
    h.run_steps(2);
    h.get_by_role(egui::accesskit::Role::TextInput).focus();
    h.run_steps(2);
    ctrl_k(&mut h);
    assert!(!h.state().switcher.open, "typing Ctrl+K in the Gallery search box must not open the palette");
}

#[test]
fn repicking_the_current_lattice_keeps_tuned_values_and_tab_state() {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    let backend = AppBackend::new();
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("chiral_kagome")).unwrap();
    app.select_lattice_and_sync(backend.named_lattices()[idx].clone());
    let s = app.gallery.sliders()[0].clone();
    let _ = app.gallery.set_param(&s.name, s.value + 4.0 * s.step, &mut app.canvas2d);
    let (vals, rev) = (app.gallery.param_values(), app.gallery.revision());
    // Ctrl+K, Enter on Recent #1 = the lattice already open
    app.select_lattice_and_sync(app.gallery.base_item().clone());
    assert_eq!(app.gallery.param_values(), vals, "tuned values survive");
    assert_eq!(app.gallery.revision(), rev, "nothing changed");
}

#[test]
fn leaving_the_gallery_applies_a_pending_background_move_first() {
    let mut app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    let backend = AppBackend::new();
    let idx = backend.named_lattices().iter().position(|it| it.named_key.as_deref() == Some("straight_p1")).unwrap();
    app.select_lattice_and_sync(backend.named_lattices()[idx].clone());
    let s = app.gallery.sliders()[0].clone();
    let ctx = egui::Context::default();
    app.gallery.drag_param_async(&s.name, s.value + 2.0 * s.step, &ctx);
    app.switch_tab(NavTab::Defects);
    assert!(!app.gallery.retune_pending(), "nothing left to land on the Defects tab later");
    let moved = app.gallery.param_values().into_iter().find(|(n, _)| *n == s.name).unwrap().1;
    assert!((moved - s.value).abs() > 0.5 * s.step, "the move was applied, not dropped");
    assert_eq!(app.defects.last_item.recipe, app.gallery.active_item().recipe, "Defects built the moved lattice");
}

#[test]
fn custom_builder_lattices_are_not_recorded_in_recent() {
    let backend = AppBackend::new();
    let mut custom = backend.named_lattices()[0].clone();
    custom.alias_key.rule = String::from("custom");
    let mut sw = LatticeSwitcher::default();
    sw.push_recent(&custom);
    assert!(sw.recent.is_empty());
}

#[test]
fn recent_is_an_lru_of_ten_unique_lattices() {
    let backend = AppBackend::new();
    let named = backend.named_lattices();
    let mut sw = LatticeSwitcher::default();
    for it in named.iter().take(12) {
        sw.push_recent(it);
    }
    assert_eq!(sw.recent.len(), RECENT_MAX);
    assert_eq!(sw.recent.front().unwrap().alias_key, named[11].alias_key, "newest first");
    assert_eq!(sw.recent.back().unwrap().alias_key, named[2].alias_key, "two oldest dropped");
    sw.push_recent(&named[5]);
    assert_eq!(sw.recent.len(), RECENT_MAX, "revisiting does not duplicate");
    assert_eq!(sw.recent.front().unwrap().alias_key, named[5].alias_key, "revisited moves to the front");
}

#[test]
fn palette_pick_on_non_lattice_driven_view_routes_to_plane2d() {
    use crate::app::{AppMode, LatticesView};

    let mut h = harness();
    h.run_steps(2);
    h.state_mut().set_advanced(true);
    h.state_mut().select_lattices_view(LatticesView::ThreeDAndSubperiodic);
    h.run_steps(2);
    assert_eq!(h.state().active_tab, NavTab::Atlas);
    assert!(!h.state().view_is_lattice_driven());

    ctrl_k(&mut h);
    h.query_all_by_role(egui::accesskit::Role::TextInput)
        .last()
        .expect("switcher search input")
        .type_text("kagome p6m");
    h.run_steps(2);
    h.key_press(Key::Enter);
    h.run_steps(2);

    {
        let app = h.state();
        assert!(!app.switcher.open, "picking a lattice closes the palette");
        assert_eq!(
            (app.active_mode, app.lattices_view, app.active_tab),
            (AppMode::Lattices, LatticesView::Plane2D, NavTab::Gallery)
        );
        assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("kagome"));
        assert!(app.canvas2d.counts().0 > 0 && app.canvas2d.counts().1 > 0, "Gallery scene drawn");
    }

    // Also verify re-picking the already-selected tuned lattice from NavTab::Atlas routes to Plane2D
    // while keeping its tuned parameter values.
    let (vals, rev, tuned_recipe) = {
        let app = h.state_mut();
        let chiral = app.backend.find_named("chiral_kagome").expect("chiral_kagome").clone();
        app.select_lattice_and_sync(chiral);
        let s = app.gallery.sliders()[0].clone();
        let _ = app.gallery.set_param(&s.name, s.value + 4.0 * s.step, &mut app.canvas2d);
        (app.gallery.param_values(), app.gallery.revision(), app.gallery.active_item().recipe.clone())
    };
    h.state_mut().select_lattices_view(LatticesView::ThreeDAndSubperiodic);
    h.run_steps(2);
    assert_eq!(h.state().active_tab, NavTab::Atlas);
    assert!(!h.state().view_is_lattice_driven());

    // Ctrl+K highlights Recent #0 (`chiral_kagome`); pressing Enter re-picks it.
    ctrl_k(&mut h);
    h.key_press(Key::Enter);
    h.run_steps(2);

    let app = h.state();
    assert!(!app.switcher.open);
    assert_eq!(
        (app.active_mode, app.lattices_view, app.active_tab),
        (AppMode::Lattices, LatticesView::Plane2D, NavTab::Gallery)
    );
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("chiral_kagome"));
    assert_eq!(app.gallery.param_values(), vals, "tuned slider values preserved");
    assert_eq!(app.gallery.revision(), rev, "revision preserved");
    assert_eq!(app.gallery.active_item().recipe, tuned_recipe, "tuned recipe preserved");
    assert!(app.canvas2d.counts().0 > 0 && app.canvas2d.counts().1 > 0, "Gallery scene drawn");
}
