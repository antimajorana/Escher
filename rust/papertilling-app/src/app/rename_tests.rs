use super::*;
use papertilling_core::defect::combine_types::NotebookConfigJson;

fn app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn select_named(app: &mut PapertillingApp, key: &str) {
    let index = app.backend.named_lattices().iter().position(|item| item.named_key.as_deref() == Some(key)).unwrap();
    app.gallery.select_named(index, &app.backend, &mut app.canvas2d);
}

fn rename(app: &mut PapertillingApp, name: &str) {
    let revision = app.gallery.revision();
    app.gallery.rename_active_lattice(name, &mut app.backend, &mut app.canvas2d);
    if revision != app.gallery.revision() {
        app.sync_active_tab_after_rename();
    }
}

#[test]
fn review_rename_preserves_hidden_combine_pins_and_defects_selection() {
    let mut app = app();
    select_named(&mut app, "square");
    let item = app.gallery.active_item().clone();
    app.switch_tab(NavTab::Combine);
    app.combine.wait_idle(&item, Some(&mut app.canvas2d));
    app.combine.apply_preset_dislocation(&item, &app.backend, &mut app.canvas2d);
    app.combine.wait_idle(&item, Some(&mut app.canvas2d));
    let pins = app.combine.specs.clone();
    assert_eq!(pins.len(), 1);

    app.switch_tab(NavTab::Defects);
    app.defects.wait_idle(&mut app.canvas2d);
    app.defects.sub_tab = crate::defects_tab::DefectSubTab::Walls;
    app.defects.selected_wall = 1;
    app.defects.rebuild_crystal(&item, &app.backend, &mut app.canvas2d, false);
    app.defects.wait_idle(&mut app.canvas2d);
    app.switch_tab(NavTab::Gallery);
    rename(&mut app, "Renamed Square");
    let renamed = app.gallery.active_item().clone();
    app.switch_tab(NavTab::Combine);
    app.combine.wait_idle(&renamed, Some(&mut app.canvas2d));
    assert_eq!(app.combine.specs, pins, "a label is not a lattice change");
    app.switch_tab(NavTab::Defects);
    app.defects.wait_idle(&mut app.canvas2d);
    assert_eq!(app.defects.selected_wall, 1);
}

#[test]
fn review_saving_an_unchanged_name_never_changes_another_tabs_scene() {
    let mut app = app();
    let item = app.gallery.active_item().clone();
    for tab in [NavTab::Surfaces, NavTab::Defects, NavTab::Combine] {
        app.switch_tab(tab);
        match tab {
            NavTab::Defects => app.defects.wait_idle(&mut app.canvas2d),
            NavTab::Combine => app.combine.wait_idle(&item, Some(&mut app.canvas2d)),
            _ => {}
        }
        let before = crate::export::render_scene_svg(app.canvas2d.scene(), 600, 600);
        let label = app.gallery.active_item().label.clone();
        rename(&mut app, &label);
        let after = crate::export::render_scene_svg(app.canvas2d.scene(), 600, 600);
        assert!(after == before, "unchanged name corrupted {tab:?} scene");
    }
}

#[test]
fn review_import_identity_uses_geometry_not_a_colliding_alias() {
    let backend = AppBackend::new();
    let cfg = NotebookConfigJson::from_json_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/config_py_honeycomb_cut.json"
    ))).unwrap();
    let mut cfg = cfg;
    cfg.lattice_label = String::from("Triangular");
    let imported = backend.item_from_notebook_json(&cfg).unwrap();
    let honeycomb = backend.find_named("honeycomb").unwrap();
    assert_eq!(imported.named_key.as_deref(), Some("honeycomb"));
    assert_eq!(imported.alias_key, honeycomb.alias_key);
    assert_eq!(imported.label, "Triangular");
}

#[test]
fn review_renaming_an_unrelated_lattice_keeps_an_imported_experiment_name() {
    let mut app = app();
    app.switch_tab(NavTab::Combine);
    let mut cfg = NotebookConfigJson::from_json_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/config_py_square_pair.json"
    ))).unwrap();
    cfg.lattice_label = String::from("Experiment");
    app.combine.load_notebook_json(&cfg, &app.backend, &mut app.canvas2d).unwrap();
    rename(&mut app, "Renamed Kagome");
    assert_eq!(app.combine.to_notebook_json(app.gallery.active_item()).lattice_label, "Experiment");
}
