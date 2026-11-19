//! The Top Bar "Presets" menu lists every preset on every tab, and clicking one applies it.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use crate::app::{AppMode, NavTab, PapertillingApp};
use crate::presets::PRESETS;
use crate::theme::{StudioTheme, typography};

fn harness() -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1600.0, 1000.0))
        .with_step_dt(0.05)
        .with_max_steps(64)
        .with_accessibility_check(false)
        .with_context_setup(|ctx| {
            typography::register_fonts(ctx);
            StudioTheme::Paper.apply(ctx);
        })
        .build_ui_state(|ui, app| { let _ = app.frame_ui(ui); }, app)
}

/// Close any open menu (Escape), check it is closed, then open the Presets menu by clicking its Top Bar button.
fn open_menu(h: &mut Harness<'static, PapertillingApp>, tab: NavTab) {
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(h.query_by_label(PRESETS[0].name).is_none(), "{tab:?}: the menu is closed before the click");
    h.get_by_label("Presets").click();
    h.run_steps(3);
}

#[test]
fn presets_menu_lists_every_preset_on_every_tab_and_a_click_applies_it() {
    let mut h = harness();
    h.run_steps(2);
    for tab in NavTab::ALL {
        h.state_mut().switch_tab(tab);
        h.run_steps(2);
        open_menu(&mut h, tab);
        for mode in AppMode::SHOWN {
            assert!(
                h.get_all_by_label(mode.label()).count() >= 1,
                "{tab:?}: menu lists mode heading '{}'",
                mode.label()
            );
        }
        // C1: the Analysis heading and the hidden presets (Analysis, 3D line defect) are not listed
        assert!(h.query_by_label(AppMode::Analysis.label()).is_none(), "{tab:?}: no Analysis heading");
        for p in PRESETS {
            assert_eq!(h.query_by_label(p.name).is_some(), p.shown(), "{tab:?}: menu lists '{}' iff shown", p.name);
        }
    }

    h.state_mut().switch_tab(NavTab::Gallery);
    h.run_steps(2);
    open_menu(&mut h, NavTab::Gallery);
    h.get_by_label("Square-lattice torus 6x6").click();
    h.run_steps(3);
    let app = h.state();
    assert_eq!(app.active_tab, NavTab::Surfaces, "the click switched to the preset's tab");
    assert_eq!(app.gallery.active_item().named_key.as_deref(), Some("square"));
    assert_eq!((app.surfaces.auto_size, app.surfaces.params.h, app.surfaces.params.k), (false, 6, 6));
    assert!(app.export_status.contains("Square-lattice torus 6x6"), "{}", app.export_status);
    assert!(h.query_by_label("Chiral kagome (p6, no mirrors)").is_none(), "the menu closed after the pick");
}
