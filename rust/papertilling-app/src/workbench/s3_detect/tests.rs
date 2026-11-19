//! S3 panel tests. Expected values are hand counts or published formulas, never values read back from the panel:
//! - A Volterra dislocation built with b = m a1 + n a2 (the catalogue entry of the Defects tab) is detected with exactly
//!   that closure b (builder convention, `DefectSpec::dislocation`); on a hexagonal cell b = a1 is 1/3<2 -1 -1 0>.
//! - The discrete Nye total over all faces telescopes onto the outer boundary closure (antisymmetric snapping).
//! - Stone-Wales on a honeycomb: two pentagons (+60°) and two heptagons (-60°), net Frank 0, Burgers vector 0.
//! - GSV moduli, hand sums: square C1122 = 0, C1111 = 2 C1212; honeycomb C1122 = 0, C1111 = 2 C1212; both isotropic,
//!   so B = 2μ(λ+μ)/(λ+2μ) I (Hirth & Lothe 1982 ch. 3).
//! - The Stroh field's displacement circulates by its b round any loop; the panel feeds it the FS/RH vector -b.
//! - A GIF file starts with "GIF89a" and stores its width little-endian at bytes 6..8 (GIF89a specification).
//! - Plane check: a unit square with both diagonals, a 3 x 3 grid at cutoff 1.5 (> sqrt 2), a collinear overlap and a
//!   T junction are refused; CNA of a hexagon + centre patch is six (2,0,0) spokes and six (1,0,0) rim bonds.
//! - Green-Lagrange strain is zero under a rigid 30° rotation; F = diag(1.1, 1) gives trace 0.105, max shear 0.0525.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{FieldKind, S3Job, S3Out, S3Sub, Settings, Source, run_job};
use crate::analysis_mode::tests::{assert_no_forbidden_tokens, harness, wait_until};
use crate::app::{DefectsView, NavTab, PapertillingApp};
use crate::workbench::{WbCtx, WbPanel, WorkbenchView};
use egui_kittest::kittest::Queryable;
use papertilling_core::defect::{perfect_disk, stone_wales_flip};
use papertilling_core::defect_fields::stroh_dislocation_field;

fn new_app() -> PapertillingApp {
    PapertillingApp::new(false, String::from("test"), String::from("test"))
}

fn frame(app: &mut PapertillingApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.frame_ui(ui);
    });
}

fn settle(app: &mut PapertillingApp, ctx: &egui::Context, what: &str) {
    let t0 = Instant::now();
    for _ in 0..4 {
        frame(app, ctx);
    }
    while app.workbench_busy() || app.workbench.dirty {
        assert!(t0.elapsed() < Duration::from_secs(60), "{what}: still busy after 60 s");
        std::thread::sleep(Duration::from_millis(10));
        frame(app, ctx);
    }
}

/// App on the S3 view of the named lattice, settled (the Defects crystal built and detected).
fn s3_app(lattice: &str, ctx: &egui::Context) -> PapertillingApp {
    let mut app = new_app();
    let item = app.backend.find_named(lattice).cloned().unwrap_or_else(|| panic!("named lattice {lattice}"));
    app.select_lattice_and_sync(item);
    app.select_defects_view(DefectsView::DetectFields);
    settle(&mut app, ctx, "enter S3");
    assert_eq!((app.active_tab, app.workbench.view), (NavTab::Workbench, WorkbenchView::S3Detect));
    app
}

fn enter_sub(app: &mut PapertillingApp, ctx: &egui::Context, s: S3Sub) {
    app.workbench.s3.sub = s;
    app.workbench_call(|wb, cx, c, v| wb.on_enter(cx, c, v));
    settle(app, ctx, s.label());
    assert!(app.workbench.s3.error().is_none(), "{s:?}: {:?}", app.workbench.s3.error());
}

fn source_of(app: &PapertillingApp) -> Source {
    let c = &app.defects.last_crystal;
    Source::new(c.pos.clone(), c.ref_pos.clone(), c.edges.clone(), &app.gallery.active_item().group, app.gallery.cell())
}

fn run(src: &Source, sub: S3Sub, set: &Settings) -> Result<S3Out, String> {
    run_job(&S3Job { sub, src: Arc::new(src.clone()), set: set.clone() })
}

fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
}

/// Builder's b of the Defects tab's default dislocation: catalogue entry 0, `m a1 + n a2` in `gallery.cell()`.
fn catalogue_b(app: &PapertillingApp) -> ([f64; 2], [i32; 2]) {
    let cat = app.backend.defect_catalogue(app.gallery.active_item());
    let (m, n) = cat.dislocations[0].burgers_mn;
    let c = app.gallery.cell();
    let (fm, fn_) = (f64::from(m), f64::from(n));
    ([fm * c.a1[0] + fn_ * c.a2[0], fm * c.a1[1] + fn_ * c.a2[1]], [m, n])
}

#[test]
fn default_dislocation_is_detected_with_its_burgers_vector() {
    let ctx = egui::Context::default();
    for (lat, weber) in [("square", None), ("honeycomb", Some("1/3["))] {
        let mut app = s3_app(lat, &ctx);
        let (b, uv) = catalogue_b(&app);
        let o = app.workbench.s3.detect.as_ref().expect("detection stored");
        eprintln!("{lat}: graph {} matches builder bonds: {:?}", o.graph_source, o.graph_matches_crystal);
        let src = source_of(&app);
        let own = Settings { own_bonds: true, ..Default::default() };
        let mut variants = vec![("own bonds", run(&src, S3Sub::Detect, &own).unwrap())];
        if o.graph_matches_crystal == Some(true) {
            variants.push(("cutoff", run(&src, S3Sub::Detect, &Settings::default()).unwrap()));
        }
        let mut own_sel = None;
        for (what, out) in variants {
            let S3Out::Detect(d) = out else { unreachable!() };
            if what == "own bonds" {
                own_sel = d.det.defects.iter().position(|x| x.kind == "dislocation");
            }
            let ds: Vec<_> = d.det.defects.iter().filter(|x| x.kind != "neutral").collect();
            assert_eq!(ds.len(), 1, "{lat} {what}: {:?}", d.det.defects.iter().map(|x| (&x.kind, &x.reason)).collect::<Vec<_>>());
            assert_eq!(ds[0].kind, "dislocation");
            let got = ds[0].burgers.unwrap();
            assert!(close(got, b, 1e-9), "{lat} {what}: detected b {got:?}, built with {b:?}");
            assert!(close(ds[0].burgers_fs_rh.unwrap(), [-b[0], -b[1]], 1e-12), "FS/RH = -b");
            assert_eq!(ds[0].uv, Some(uv));
            if let Some(w) = weber {
                assert!(ds[0].index.as_ref().and_then(|i| i.weber_str.as_deref()).is_some_and(|s| s.starts_with(w)));
            }
        }
        // Nye: Σ α A_f over all faces equals the outer closure; Vitek components wrap into (-|b|/2, |b|/2]
        app.workbench.s3.set.own_bonds = true;
        app.workbench.s3.set.selected = Some(own_sel.unwrap());
        enter_sub(&mut app, &ctx, S3Sub::Fields);
        let f = app.workbench.s3.fields.as_ref().unwrap();
        let nye = f.nye.as_ref().unwrap();
        let t = nye.total.expect("zero net Frank angle: a total");
        assert!(close(t, nye.boundary_closure, 1e-9), "{lat}: Nye {t:?} vs closure {:?}", nye.boundary_closure);
        assert!(close(t, b, 1e-9), "{lat}: Nye total {t:?} vs b {b:?}");
        let (arrows, vb) = f.vitek.as_ref().unwrap();
        let half = 0.5 * vb[0].hypot(vb[1]);
        assert!(arrows.iter().all(|a| a.wrapped > -half - 1e-12 && a.wrapped <= half + 1e-12));
        assert!(f.strain.is_ok() && f.max_u > 0.0, "{lat}: {:?}", f.strain.as_ref().err());
    }
}

#[test]
fn stone_wales_on_a_honeycomb_disk_is_neutral() {
    let ctx = egui::Context::default();
    let app = s3_app("honeycomb", &ctx);
    let cell = app.gallery.cell().clone();
    let a = cell.a1[0].hypot(cell.a1[1]);
    let disk = perfect_disk(&cell, [0.0, 0.0], 4.0 * a);
    let mid = |&(u, v): &(usize, usize)| {
        let (p, q) = (disk.pos[u], disk.pos[v]);
        (0.5 * (p[0] + q[0])).hypot(0.5 * (p[1] + q[1]))
    };
    let k = (0..disk.edges.len()).min_by(|&x, &y| mid(&disk.edges[x]).total_cmp(&mid(&disk.edges[y]))).unwrap();
    let sw = stone_wales_flip(&disk, k, None).unwrap();
    let c = sw.crystal;
    let src = Source::new(c.pos.clone(), c.ref_pos.clone(), c.edges.clone(), &app.gallery.active_item().group, &cell);
    let S3Out::Detect(d) = run(&src, S3Sub::Detect, &Settings { own_bonds: true, ..Default::default() }).unwrap() else {
        unreachable!()
    };
    let mut charged: Vec<(usize, i64)> = d
        .det
        .faces
        .iter()
        .zip(&d.det.face_frank)
        .filter(|(_, q)| **q != 0.0)
        .map(|(f, q)| (f.len(), q.round() as i64))
        .collect();
    charged.sort_unstable();
    assert_eq!(charged, vec![(5, 60), (5, 60), (7, -60), (7, -60)]);
    assert_eq!(d.det.defects.len(), 1);
    assert_eq!((d.det.defects[0].kind.as_str(), d.det.defects[0].burgers), ("neutral", Some([0.0, 0.0])));
    assert!(d.det.defects[0].reason.starts_with("[PROVED]"));
}

#[test]
fn elastic_moduli_match_hand_sums_and_the_isotropic_formula() {
    let ctx = egui::Context::default();
    for lat in ["square", "honeycomb"] {
        let mut app = s3_app(lat, &ctx);
        enter_sub(&mut app, &ctx, S3Sub::Elastic);
        let e = app.workbench.s3.elastic.as_ref().unwrap();
        let c = &e.c;
        let (c11, c22, c12, c66) = (c[0][0][0][0], c[1][1][1][1], c[0][0][1][1], c[0][1][0][1]);
        // measured: the app's honeycomb cell gives relative deviations ~2e-12 (Born-Huang pseudoinverse round-off)
        let tol = 1e-10 * c11;
        assert!(c12.abs() < tol && (c11 - 2.0 * c66).abs() < tol && (c11 - c22).abs() < tol, "{lat}: {c:?}");
        assert_eq!(e.class, papertilling_core::defect_fields::ElasticClass::Isotropic, "{lat}");
        let want = 2.0 * c66 * (c12 + c66) / (c12 + 2.0 * c66);
        assert!((e.bl[0][0] - want).abs() < 1e-10 && (e.bl[1][1] - want).abs() < 1e-10 && e.bl[0][1].abs() < 1e-10, "{lat}");

        // the grid field is fed the FS/RH vector of the selected dislocation; its circulation is that vector
        let det = &e.det.det;
        let sel = e.selected;
        let g = e.field.as_ref().unwrap_or_else(|why| panic!("{lat}: {why}"));
        let fs = det.defects[sel].burgers_fs_rh.unwrap();
        assert!(close(g.b, fs, 1e-12), "{lat}: field b {:?} vs FS/RH {fs:?}", g.b);
        assert_eq!(g.sigma.len(), g.n * g.n);
        assert!(g.sigma.iter().all(|s| s.is_finite() && *s > 0.0));
        let n = 2048;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let t = -std::f64::consts::PI + std::f64::consts::TAU * (k as f64 + 0.5) / n as f64;
                [2.0 * t.cos(), 2.0 * t.sin()]
            })
            .collect();
        let f = stroh_dislocation_field(c, g.b, &pts).unwrap();
        let mut circ = [0.0; 2];
        for (k, p) in pts.iter().enumerate() {
            let tang = [-p[1], p[0]];
            for (j, cj) in circ.iter_mut().enumerate() {
                *cj += (f.grad[k][j][0] * tang[0] + f.grad[k][j][1] * tang[1]) * std::f64::consts::TAU / n as f64;
            }
        }
        assert!(close(circ, fs, 1e-9), "{lat}: circulation {circ:?} vs {fs:?}");
    }
}

#[test]
fn motion_gif_has_the_gif89a_header_and_exports() {
    let ctx = egui::Context::default();
    let mut app = s3_app("honeycomb", &ctx);
    app.workbench.s3.set.gif_px = 96;
    app.workbench.s3.set.gif_frames = 4;
    enter_sub(&mut app, &ctx, S3Sub::Gif);
    let o = app.workbench.s3.gif.as_ref().unwrap();
    let bytes = o.bytes.as_ref().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(&bytes[..6], b"GIF89a");
    assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 96);
    assert_eq!(*bytes.last().unwrap(), 0x3B, "GIF trailer");
    let path = std::env::temp_dir().join(format!("papertilling_s3_{}.gif", std::process::id()));
    super::gif::export_to(&path, bytes).unwrap();
    assert_eq!(std::fs::read(&path).unwrap().len(), bytes.len());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn refusals_carry_reasons() {
    let ctx = egui::Context::default();
    let app = s3_app("honeycomb", &ctx);
    let src = source_of(&app);
    let tiny = Source::new(vec![[0.0, 0.0], [1.0, 0.0]], Vec::new(), vec![(0, 1)], "p6m", app.gallery.cell());
    assert!(run(&tiny, S3Sub::Detect, &Settings::default()).is_err_and(|e| e.contains("REFUSED")));
    let bad = [
        (S3Sub::Detect, Settings { auto_cutoff: false, cutoff_rel: 9.0, ..Default::default() }),
        (S3Sub::Detect, Settings { clearance: 9, ..Default::default() }),
        (S3Sub::Elastic, Settings { grid: 41, ..Default::default() }),
        (S3Sub::Elastic, Settings { manual_b: true, manual_uv: [0, 0], ..Default::default() }),
        (S3Sub::Gif, Settings { gif_frames: 1, ..Default::default() }),
    ];
    for (sub, set) in bad {
        let e = run(&src, sub, &set).err().unwrap_or_else(|| panic!("{sub:?} {set:?} accepted"));
        assert!(!e.is_empty());
    }
    // a missing defect is a per-result refusal, not a failed job
    let S3Out::Elastic(e) = run(&src, S3Sub::Elastic, &Settings { selected: Some(99), ..Default::default() }).unwrap() else { unreachable!() };
    assert!(e.field.as_ref().is_err_and(|w| w.contains("does not exist")));
    // a manual b on the same crystal is accepted
    let S3Out::Elastic(e) = run(&src, S3Sub::Elastic, &Settings { manual_b: true, ..Default::default() }).unwrap() else { unreachable!() };
    assert!(e.field.is_ok());
    // no source crystal: the panel waits (with a note), it does not report an error
    let mut p = super::Panel::default();
    let cx = WbCtx { item: app.gallery.active_item(), cell: app.gallery.cell(), smoke: false, backend: &app.backend, defect_crystal: None };
    p.submit(&cx, None);
    assert!(p.waiting.is_some() && p.error().is_none() && !p.is_busy());
}

#[test]
fn s3_panel_state_survives_json_roundtrip() {
    let mut p = super::Panel { sub: S3Sub::Elastic, ..Default::default() };
    p.set = Settings { own_bonds: true, clearance: 2, field: FieldKind::Vitek, grid: 20, manual_b: true, manual_uv: [1, -1], gif_px: 200, ..Default::default() };
    let v = p.save();
    let back: serde_json::Value = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    let mut q = super::Panel::default();
    let mut notes = Vec::new();
    q.restore(&back, &mut notes);
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!((q.sub, &q.set), (S3Sub::Elastic, &p.set));
    q.restore(&serde_json::json!({"sub": "Nope", "settings": {"grid": 99, "clearance": 50, "cutoff_rel": "x", "gif_px": 5}}), &mut notes);
    assert_eq!(notes.len(), 1);
    assert_eq!((q.sub, q.set.grid, q.set.clearance, q.set.gif_px), (S3Sub::Elastic, 80, 4, 64), "junk ignored, sizes clamped");
}

#[test]
fn s3_subrows_quarantine_clean() {
    let mut h = harness();
    let hc = h.state().backend.find_named("honeycomb").cloned().unwrap();
    h.state_mut().select_lattice_and_sync(hc);
    h.run_steps(3);
    h.get_by_label("2. Defects").click();
    h.run_steps(2);
    h.get_by_label("Detect & Fields").click();
    wait_until(&mut h, "S3", |a| !a.workbench_busy() && !a.workbench.dirty);
    for s in S3Sub::ALL.into_iter().chain([S3Sub::Detect]) {
        h.get_by_label(s.label()).click();
        wait_until(&mut h, s.label(), |a| !a.workbench_busy() && !a.workbench.dirty);
        assert_eq!(h.state().workbench.s3.sub, s);
        assert!(h.state().workbench.s3.error().is_none(), "{s:?}: {:?}", h.state().workbench.s3.error());
        assert_no_forbidden_tokens(&h, s.label());
        if s == S3Sub::Fields {
            for k in FieldKind::ALL {
                h.state_mut().workbench.s3.set.field = k;
                h.state_mut().workbench.dirty = true;
                h.run_steps(2);
                assert_no_forbidden_tokens(&h, k.label());
            }
        }
    }
}

/// Pilot timing of every S3 job on the default honeycomb dislocation (printed with `# time`; no time is asserted).
#[test]
fn pilot_timing_s3_jobs() {
    let ctx = egui::Context::default();
    let app = s3_app("honeycomb", &ctx);
    let src = source_of(&app);
    eprintln!("# source: {} sites, {} bonds", src.pos.len(), src.edges.len());
    for sub in S3Sub::ALL {
        let t = Instant::now();
        let r = run(&src, sub, &Settings::default());
        eprintln!("# time S3 {sub:?} default: {:.3}s ({})", t.elapsed().as_secs_f64(), if r.is_ok() { "ok" } else { "refused" });
    }
    let t = Instant::now();
    let big = Settings { grid: 80, extent: 20.0, gif_frames: 60, gif_px: 800, ..Default::default() };
    let _ = run(&src, S3Sub::Elastic, &big);
    eprintln!("# time S3 Elastic 80x80 grid: {:.3}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let _ = run(&src, S3Sub::Gif, &big);
    eprintln!("# time S3 GIF 60 frames 800 px: {:.3}s", t.elapsed().as_secs_f64());
}

/// Triangular patch: a centre and its six nearest neighbours at unit distance.
fn hex_patch() -> Vec<[f64; 2]> {
    let mut p = vec![[0.0, 0.0]];
    p.extend((0..6_i32).map(|k| {
        let t = f64::from(k) * std::f64::consts::FRAC_PI_3;
        [t.cos(), t.sin()]
    }));
    p
}

#[test]
fn plane_check_refuses_crossing_touching_and_overlapping_bonds() {
    use super::graph_check::check_plane;
    use papertilling_core::defect_fields::pairs_within;
    let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    assert!(check_plane(&sq, &[(0, 1), (1, 2), (2, 3), (0, 3)]).is_ok());
    let both_diagonals = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)];
    assert!(check_plane(&sq, &both_diagonals).is_err_and(|e| e.contains("not plane")));
    // 3 x 3 square grid: 1.3 keeps the 12 nearest bonds; 1.5 > sqrt 2 adds the 8 diagonals, which cross in pairs
    let grid: Vec<[f64; 2]> = (0..9_i32).map(|k| [f64::from(k % 3), f64::from(k / 3)]).collect();
    assert_eq!(pairs_within(&grid, 1.3).len(), 12);
    assert!(check_plane(&grid, &pairs_within(&grid, 1.3)).is_ok());
    assert_eq!(pairs_within(&grid, 1.5).len(), 20);
    assert!(check_plane(&grid, &pairs_within(&grid, 1.5)).is_err());
    // triangular patch at 1.3: 6 spokes + 6 rim bonds, plane
    let tri = hex_patch();
    assert_eq!(pairs_within(&tri, 1.3).len(), 12);
    assert!(check_plane(&tri, &pairs_within(&tri, 1.3)).is_ok());
    // collinear overlap: 0-2 runs along 0-1 (shared site, same direction); 0-1 and 1-2 only touch at site 1
    let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    assert!(check_plane(&line, &[(0, 1), (1, 2)]).is_ok());
    assert!(check_plane(&line, &[(0, 1), (1, 2), (0, 2)]).is_err());
    // a site on another bond's interior (T junction), and coincident sites
    let t = [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
    assert!(check_plane(&t, &[(0, 1), (2, 3)]).is_err());
    assert!(check_plane(&[[0.0, 0.0], [0.0, 0.0]], &[(0, 1)]).is_err_and(|e| e.contains("zero")));
}

#[test]
fn cna_runs_on_plane_graphs_and_is_skipped_on_dense_ones() {
    use super::graph_check::cna_histogram;
    use papertilling_core::defect_fields::BondGraph;
    // hand count: a spoke has the two flanking rim sites in common (not bonded: sqrt 3 apart) -> (2,0,0);
    // a rim bond has only the centre in common -> (1,0,0); six of each
    let mut got = cna_histogram(&BondGraph::from_cutoff(&hex_patch(), 1.3)).unwrap();
    got.sort_unstable();
    assert_eq!(got, vec![((1, 0, 0), 6), ((2, 0, 0), 6)]);
    // 8 sites in convex position, all 28 pairs bonded: each bond's 6 common neighbours are bonded to the other 5
    let oct: Vec<[f64; 2]> = (0..8_i32)
        .map(|k| {
            let t = f64::from(k) * std::f64::consts::FRAC_PI_4;
            [t.cos(), t.sin()]
        })
        .collect();
    let all: Vec<(usize, usize)> = (0..8).flat_map(|i| (i + 1..8).map(move |j| (i, j))).collect();
    assert!(cna_histogram(&BondGraph::from_edges(&oct, &all)).is_err_and(|e| e.contains("not computed")));
}

#[test]
fn cutoff_ends_are_refused_with_reasons() {
    let at_one = Settings { auto_cutoff: false, cutoff_rel: 1.0, ..Default::default() };
    assert!(at_one.validate(S3Sub::Detect).is_err_and(|e| e.contains("dropped by rounding")));
    assert!(Settings { cutoff_rel: 1.001, ..at_one }.validate(S3Sub::Detect).is_ok());
    let mut r = Settings::default();
    r.restore(&serde_json::json!({ "cutoff_rel": 1.0 }));
    assert!((r.cutoff_rel - 1.001).abs() < 1e-15, "restore clamps into the range");
    // square lattice: 1.5 longest bonds > sqrt 2 brings in the crossing diagonals
    let ctx = egui::Context::default();
    let app = s3_app("square", &ctx);
    let wide = Settings { auto_cutoff: false, cutoff_rel: 1.5, ..Default::default() };
    assert!(run(&source_of(&app), S3Sub::Detect, &wide).is_err_and(|e| e.contains("not plane")));
}

#[test]
fn stale_results_are_dropped_and_waiting_hides_old_state() {
    let ctx = egui::Context::default();
    let app = s3_app("square", &ctx);
    let src = source_of(&app);
    let (item, cell) = (app.gallery.active_item(), app.gallery.cell());
    let cx = WbCtx { item, cell, smoke: false, backend: &app.backend, defect_crystal: Some(&app.defects.last_crystal) };
    let mut p = super::Panel::default();
    let key_a = p.job_key(&src);
    // A built, job B (another cutoff) still in flight, the inputs return to A: B is forgotten and nothing is rebuilt
    p.built[0] = Some(key_a.clone());
    p.pending = Some((S3Sub::Detect, String::from("B")));
    p.submit(&cx, None);
    assert!(p.pending.is_none() && !p.is_busy() && p.worker.is_none());
    assert_eq!(p.wanted[0].as_deref(), Some(key_a.as_str()));
    // a late result whose key is not the wanted one changes nothing, failure or success
    let own = Settings { own_bonds: true, ..Default::default() };
    p.pending = Some((S3Sub::Detect, String::from("B")));
    p.accept(Ok(Err(String::from("late failure"))));
    p.pending = Some((S3Sub::Detect, String::from("B")));
    p.accept(Ok(run(&src, S3Sub::Detect, &own)));
    assert!(p.detect.is_none() && p.error().is_none());
    assert_eq!(p.built[0].as_deref(), Some(key_a.as_str()));
    // the wanted key is stored
    p.pending = Some((S3Sub::Detect, key_a.clone()));
    p.accept(Ok(run(&src, S3Sub::Detect, &own)));
    assert!(p.detect.is_some());
    // the source goes away: the old error is hidden, the job in flight forgotten and its late result dropped
    p.errors[0] = Some(String::from("old error"));
    p.pending = Some((S3Sub::Detect, key_a.clone()));
    let gone = WbCtx { item, cell, smoke: false, backend: &app.backend, defect_crystal: None };
    p.submit(&gone, None);
    assert!(p.waiting.is_some() && p.error().is_none() && p.pending.is_none() && p.wanted[0].is_none());
    p.pending = Some((S3Sub::Detect, key_a));
    p.accept(Ok(Err(String::from("late failure for the old crystal"))));
    assert_eq!(p.errors[0].as_deref(), Some("old error"), "dropped, not stored");
    assert!(p.detect.is_some());
}

#[test]
fn site_strain_is_zero_under_a_rigid_rotation() {
    use super::fields::{max_shear, site_strain, trace};
    use papertilling_core::defect_fields::{local_deformation_and_strain, pairs_within};
    let r = hex_patch();
    let edges = pairs_within(&r, 1.3);
    let (s, c) = std::f64::consts::FRAC_PI_6.sin_cos();
    let rot: Vec<[f64; 2]> = r.iter().map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1]]).collect();
    for l in &local_deformation_and_strain(&r, &rot, &edges).unwrap() {
        let e = site_strain(l).expect("every site has two independent bonds");
        assert!(trace(e).abs() < 1e-12 && max_shear(e) < 1e-12, "{e:?}");
        // the linear strain would paint 2 (cos 30° - 1) = -0.268 here
        assert!((trace(l.eps.as_ref().unwrap()) - 2.0 * (c - 1.0)).abs() < 1e-12);
    }
    // uniform stretch F = diag(1.1, 1): E11 = (1.21 - 1) / 2 = 0.105, E22 = E12 = 0, so max shear 0.0525
    let st: Vec<[f64; 2]> = r.iter().map(|p| [1.1 * p[0], p[1]]).collect();
    for l in &local_deformation_and_strain(&r, &st, &edges).unwrap() {
        let e = site_strain(l).unwrap();
        assert!((trace(e) - 0.105).abs() < 1e-12 && (max_shear(e) - 0.0525).abs() < 1e-12, "{e:?}");
    }
}
