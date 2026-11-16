use super::*;
use crate::backend::AppBackend;
use crate::live_combine::combine_lattice_info;
use crate::live_ops::build_pristine_disk;
use papertilling_core::defect::combine_api::equivalent_single;
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::family::LatticeFamily;

fn small_disk() -> papertilling_core::types::BuiltCrystal {
    static DISK: std::sync::OnceLock<papertilling_core::types::BuiltCrystal> = std::sync::OnceLock::new();
    DISK.get_or_init(|| {
        let backend = AppBackend::new();
        let sq = backend.find_named("square").unwrap();
        let cell = LatticeFamily::from_recipe(&sq.recipe, Some(sq.frame)).unwrap().cell();
        build_pristine_disk(&LatticeCoreInfo::new(&cell), 2.6)
    })
    .clone()
}

fn fake_entry(name: &str, kind: &str) -> CoreGalleryEntry {
    CoreGalleryEntry {
        name: name.to_string(),
        kind: kind.to_string(),
        crystal: small_disk(),
        core_sites: vec![0, 1, 2],
        chords: vec![(0, 1)],
        under_sites: vec![2],
        rings: vec![5, 7],
        faces: vec![vec![0, 1, 2]],
        under: vec![3],
        added: 1,
        description: String::from("test completion"),
        also: vec![String::from("V1")],
        faces_ok: true,
        iso_ok: Some(true),
        perfect_rings: vec![4],
    }
}

/// The real square-lattice report of +90 site / -90 face (C01: matches Volterra core 1), grid 1.
fn square_pair_report() -> EquivalentSingleReport {
    let backend = AppBackend::new();
    let sq = backend.find_named("square").unwrap();
    let info = combine_lattice_info(sq);
    let specs = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
    let crystal = build_point_defects(&info.cell, &specs, None, None).unwrap();
    equivalent_single(&sq.group, &info.cell, sq.frame, &specs, &crystal, 2, 1, true).unwrap()
}

#[test]
fn compile_core_card_renders_faces_chords_and_under_sites() {
    let entry = fake_entry("C1", "completion");
    let prims = compile_core_card(&entry);
    assert_eq!(prims.polygons.len(), 1);
    assert_eq!(prims.polygons[0].fill, ODD_FACE_RGBA);
    // the chord (0, 1) is drawn once, dashed and magenta, and not also as a plain bond
    let chord_lines: Vec<_> = prims.lines.iter().filter(|l| l.color == CHORD_RGBA).collect();
    assert_eq!(chord_lines.len(), 1);
    assert!(chord_lines[0].dashed);
    let (p0, p1) = (entry.crystal.pos[0], entry.crystal.pos[1]);
    let chord_as_plain = prims.lines.iter().any(|l| !l.dashed && {
        let (a, b) = ([p0[0] as f32, p0[1] as f32], [p1[0] as f32, p1[1] as f32]);
        (l.a == a && l.b == b) || (l.a == b && l.b == a)
    });
    assert!(!chord_as_plain);
    assert_eq!(prims.circles.iter().filter(|c| c.stroke.0 == UNDER_SITE_RGBA).count(), 1);
    let mut cfg = fake_entry("this configuration", "configuration");
    cfg.chords.clear();
    assert_eq!(compile_core_card(&cfg).polygons[0].fill, CONFIG_FACE_RGBA);
}

#[test]
fn card_selection_follows_python_show_gallery() {
    // 3 Volterra + 12 completion cores
    let mut report = square_pair_report();
    report.gallery = (1..=3).map(|k| fake_entry(&format!("Volterra core {k}"), "Volterra")).collect();
    report.gallery.extend((4..=15).map(|k| fake_entry(&format!("completion core {k}"), "completion")));
    let names = |s: &CardSelection| s.completions.iter().map(|e| e.name.clone()).collect::<Vec<_>>();

    // 7 matching completions: all 7 shown first (cap = max(7, 5)); the old fixed cap of 6 hid one MATCH
    report.matches = (9..=15).map(|k| format!("completion core {k}")).collect();
    let sel = select_cards(&report, false);
    assert_eq!((sel.volterra.len(), sel.completions.len(), sel.hidden), (3, 7, 5));
    assert_eq!(names(&sel), report.matches);

    // one match: it comes first, filled up to 5 completion cards
    report.matches = vec![String::from("completion core 12")];
    let sel = select_cards(&report, false);
    assert_eq!(names(&sel)[0], "completion core 12");
    assert_eq!((sel.completions.len(), sel.hidden), (5, 7));

    // a Volterra match: Volterra cards all shown, completions capped at 5
    report.matches = vec![String::from("Volterra core 2")];
    assert_eq!(select_cards(&report, false).completions.len(), 5);

    // nothing matches: every completion core (30 cards at most in all)
    report.matches.clear();
    let sel = select_cards(&report, false);
    assert_eq!((sel.completions.len(), sel.hidden), (12, 0));
    report.gallery.extend((16..=40).map(|k| fake_entry(&format!("completion core {k}"), "completion")));
    let sel = select_cards(&report, false);
    assert_eq!((sel.volterra.len() + sel.completions.len(), sel.hidden), (MAX_PANELS, 37 - 27));
    assert_eq!(select_cards(&report, true).hidden, 0);
}

#[test]
fn panel_draws_every_state_headless() {
    let real = square_pair_report();
    assert_eq!(real.matches, vec![String::from("Volterra core 1")]);
    let mut many = real.clone();
    many.gallery.extend((3..=60).map(|k| fake_entry(&format!("completion core {k}"), "completion")));
    many.matches.clear();
    let mut empty = real.clone();
    (empty.config_core, empty.gallery) = (None, Vec::new());
    let cases: Vec<Option<Result<EquivalentSingleReport, String>>> =
        vec![None, Some(Err(String::from("surgery refused"))), Some(Ok(empty)), Some(Ok(real)), Some(Ok(many.clone()))];

    let ctx = egui::Context::default();
    let mut state = CorePanelState::default();
    for case in &cases {
        for _ in 0..2 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 260.0))),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| show_core_panel(&mut state, ui, case.as_ref()));
        }
    }
    // the 61-card gallery: only cards scrolled into view (about 900 / 190 of them) were compiled
    let compiled = state.cards.cards.len();
    assert!(compiled > 0 && compiled < 10, "{compiled} cards compiled");
    // a new report invalidates the cache
    state.cards.sync(&square_pair_report());
    assert!(state.cards.cards.is_empty());
    state.collapsed = true;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| show_core_panel(&mut state, ui, Some(&Ok(many.clone()))));
}

#[test]
fn subtitle_is_cut_with_an_ellipsis() {
    let mut e = fake_entry("V", "Volterra");
    e.rings = vec![5];
    e.description = String::from("one pentagon; one under-coordinated site on the ring");
    let s = card_subtitle(&e, "Volterra");
    assert_eq!(s.chars().count(), 30);
    assert!(s.starts_with("rings [5] · one pentagon") && s.ends_with('…'));
    e.rings.clear();
    assert_eq!(card_subtitle(&e, "Volterra"), "Volterra");
}

#[test]
fn changing_any_card_position_invalidates_the_cached_picture() {
    let mut report = square_pair_report();
    let mut cache = CardCache::default();
    cache.sync(&report);
    let entry = report.config_core.as_ref().unwrap();
    cache.get(entry);
    assert!(!cache.cards.is_empty());
    // Same charge, verdict, graph sizes and FIRST position: the old sampled key missed this move.
    report.config_core.as_mut().unwrap().crystal.pos[1][0] += 0.25;
    cache.sync(&report);
    assert!(cache.cards.is_empty());
}

#[test]
fn core_cards_draw_absolute_ghost_endpoints_and_never_the_full_ghost_bond() {
    let mut entry = fake_entry("ghost regression", "configuration");
    entry.chords.clear();
    for p in &mut entry.crystal.pos { p[0] += 10.0; p[1] += 20.0; }
    let (u, v) = entry.crystal.edges[0];
    let (pu, pv) = (entry.crystal.pos[u], entry.crystal.pos[v]);
    let a = [pu[0] + 0.125, pu[1] + 0.25];
    let b = [pv[0] - 0.125, pv[1] - 0.25];
    entry.crystal.adj[u].iter_mut().find(|e| e.to == v).unwrap().ghost_stub = Some(a);
    entry.crystal.adj[v].iter_mut().find(|e| e.to == u).unwrap().ghost_stub = Some(b);
    let prims = compile_core_card(&entry);
    let point = |p: [f64; 2]| [p[0] as f32, p[1] as f32];
    assert!(prims.lines.iter().any(|l| l.dashed && l.a == point(pu) && l.b == point(a)));
    assert!(prims.lines.iter().any(|l| l.dashed && l.a == point(pv) && l.b == point(b)));
    assert!(!prims.lines.iter().any(|l| !l.dashed &&
        ((l.a == point(pu) && l.b == point(pv)) || (l.a == point(pv) && l.b == point(pu)))));
}
