//! Timing harness for the Combine tab hot paths (run by hand, prints `# time` lines):
//! `cargo test --release -p papertilling-app combine_bench -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use papertilling_core::defect::combine_api::{
    classify_crystal_charge, equivalent_single, inspect_crystal_cores,
};
use papertilling_core::defect::core_graph::core_graph;
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::draw::relax_picture;
use papertilling_core::plane::planarity;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::live_combine::{
    CombineBuildOptions, CombineDrawMode, CombineLatticeInfo, build_combined_crystal,
    combine_lattice_info, preset_neighbour_pair, preset_shortest_dislocation,
};
use crate::live_ops::summarize_rings_and_core;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn opts(radius: Option<f64>) -> CombineBuildOptions {
    CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: radius.is_none(),
        manual_radius: radius.unwrap_or(6.0),
        core_radius: 2,
    }
}

/// Move the last disclination of a preset pair `steps` cells further along `a1`.
fn spread(info: &CombineLatticeInfo, mut specs: Vec<DefectSpec>, steps: f64) -> Vec<DefectSpec> {
    if let Some(DefectSpec::Disclination { at, .. }) = specs.last_mut() {
        at[0] += steps * info.cell.a1[0];
        at[1] += steps * info.cell.a1[1];
    }
    specs
}

fn stage_breakdown(name: &str, item: &CatalogueItemUi, info: &CombineLatticeInfo, specs: &[DefectSpec], radius: Option<f64>) {
    let o = opts(radius);
    let t = Instant::now();
    let preview = build_combined_crystal(item, info, specs, &o, true);
    println!("# time preview {name}: {:.1} ms (N={})", ms(t), preview.crystal.pos.len());
    let t = Instant::now();
    let out = build_combined_crystal(item, info, specs, &o, false);
    let total = ms(t);
    let r = out.applied_radius;
    let t = Instant::now();
    let Ok(mut c) = build_point_defects(&info.cell, specs, Some(r), None) else {
        println!("# time {name} surgery refused: {}", out.note);
        return;
    };
    let t_surg = ms(t);
    let t = Instant::now();
    relax_picture(&mut c);
    let t_relax = ms(t);
    let t = Instant::now();
    let chk = planarity::check(&c);
    let t_plan = ms(t);
    let t = Instant::now();
    let _ = classify_crystal_charge(&item.group, &info.cell, item.frame, &c);
    let t_charge = ms(t);
    let t = Instant::now();
    let _ = core_graph(&c, &info.lat_info, 2);
    let t_core = ms(t);
    let t = Instant::now();
    let _ = summarize_rings_and_core(&c, &info.lat_info);
    let t_sum = ms(t);
    let t = Instant::now();
    let _ = inspect_crystal_cores(&c);
    let t_insp = ms(t);
    println!(
        "# time build {name}: total={total:.1} ms (N={}, R={r:.1}) | surgery={t_surg:.1} relax={t_relax:.1} planarity={t_plan:.1} (crossings={}) charge={t_charge:.1} core_graph={t_core:.1} summarize={t_sum:.1} inspect={t_insp:.1}",
        c.pos.len(),
        chk.crossings
    );
}

fn compare_timing(name: &str, item: &CatalogueItemUi, info: &CombineLatticeInfo, specs: &[DefectSpec]) {
    let out = build_combined_crystal(item, info, specs, &opts(None), false);
    let mut cache = papertilling_core::defect::combine_api::SingleDefectCache::default();
    for pass in ["cold", "warm"] {
        let t = Instant::now();
        let rep = papertilling_core::defect::combine_api::equivalent_single_cached(
            &item.group, &info.cell, item.frame, specs, &out.crystal, 2, 3, true, &mut cache,
        );
        println!("# time compare_cached {name} grid=3 draw=true {pass}: {:.1} ms (candidate_match={:.4} ms, eq={:?})", ms(t), cache.last_match_ms(), rep.map(|r| r.equivalent));
    }
    for grid in 1..=3 {
        for draw in [false, true] {
            let t = Instant::now();
            let rep = equivalent_single(&item.group, &info.cell, item.frame, specs, &out.crystal, 2, grid, draw);
            let verdict = rep.as_ref().map_or_else(|e| format!("ERR {e}"), |r| format!("{} cards={} eq={}", r.kind, r.gallery.len(), r.equivalent));
            println!("# time compare {name} grid={grid} draw={draw}: {:.1} ms ({verdict})", ms(t));
        }
    }
}

#[test]
#[ignore = "timing harness, run by hand"]
fn combine_bench_build_stages() {
    let backend = AppBackend::new();
    for key in ["square", "honeycomb", "kagome"] {
        let item = backend.find_named(key).expect("named lattice");
        let t = Instant::now();
        let info = combine_lattice_info(item);
        println!("# time lattice_info {key}: {:.1} ms (r_auto={:.2})", ms(t), info.r_auto_base);
        let pair = preset_neighbour_pair(&info);
        stage_breakdown(&format!("{key} pair d=1"), item, &info, &pair, None);
        stage_breakdown(&format!("{key} pair d=+6a1"), item, &info, &spread(&info, pair.clone(), 6.0), None);
        stage_breakdown(&format!("{key} pair R=2x"), item, &info, &pair, Some(2.0 * info.r_auto_base));
        stage_breakdown(&format!("{key} disloc"), item, &info, &preset_shortest_dislocation(&info), None);
    }
}

#[test]
#[ignore = "timing harness, run by hand"]
fn combine_bench_snap_per_frame() {
    let backend = AppBackend::new();
    for key in ["square", "kagome", "shastry_sutherland", "chiral_triangular"] {
        let item = backend.find_named(key).expect("named lattice");
        let info = combine_lattice_info(item);
        let Some(&ang) = info.allowed_angles.first() else { continue };
        let n = 50;
        let t = Instant::now();
        for k in 0..n {
            let _ = crate::live_combine::snap_disclination_centre(item, &info, ang as f64, None, [0.37 * k as f64, 0.11 * k as f64]);
        }
        println!("# time snap {key} (no class filter, as on hover/drag): {:.2} ms per call", ms(t) / n as f64);
        let cid = info.centres.first().map(|c| c.id.clone());
        let t = Instant::now();
        for k in 0..n {
            let _ = crate::live_combine::snap_disclination_centre(item, &info, ang as f64, cid.as_deref(), [0.37 * k as f64, 0.11 * k as f64]);
        }
        println!("# time snap {key} (class filter): {:.3} ms per call", ms(t) / n as f64);
    }
}

#[test]
#[ignore = "timing harness, run by hand"]
fn combine_bench_equivalent_single() {
    let backend = AppBackend::new();
    for key in ["square", "honeycomb"] {
        let item = backend.find_named(key).expect("named lattice");
        let info = combine_lattice_info(item);
        compare_timing(&format!("{key} pair"), item, &info, &preset_neighbour_pair(&info));
        compare_timing(&format!("{key} disloc"), item, &info, &preset_shortest_dislocation(&info));
    }
}

#[test]
#[ignore = "timing harness, run by hand"]
fn combine_bench_ui_submission_and_drag_frames() {
    use crate::canvas2d::{Canvas2D, Scene2D};
    use crate::combine_tab::{CombineTabState, ToolMode};
    let backend = AppBackend::new();
    for key in ["square", "honeycomb", "shastry_sutherland", "chiral_triangular"] {
        let ctx = egui::Context::default();
        let item = backend.find_named(key).unwrap();
        let mut st = CombineTabState::new(item, &backend);
        let mut canvas = Canvas2D::new(Scene2D::CrystalPatch {
            crystal: st.last_build.crystal.clone(), title: String::new(), note: String::new(),
        });
        let mut view3d = crate::view3d::View3dState::default();
        st.wait_idle(item, Some(&mut canvas));
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                st.show_side_panel(ui, item, &backend, &mut canvas);
                st.show_central(ui, item, &backend, &mut canvas, &mut view3d);
            });
        }
        st.tool_mode = ToolMode::PlaceDislocation;
        let t = Instant::now();
        st.place_at_world([0.371, 0.413], item, &backend, &mut canvas);
        println!("# time UI place {key}: {:.3} ms", ms(t));
        st.wait_idle(item, Some(&mut canvas));
        st.tool_mode = ToolMode::PanAndDrag;
        let idx = st.specs.len() - 1;
        let mut times = Vec::new();
        for k in 0..40 {
            let t = Instant::now();
            st.move_pin_to(idx, [0.45 + 0.07 * k as f64, 0.413], item, &backend, &mut canvas);
            let input = egui::RawInput {
                events: if k == 0 { vec![egui::Event::PointerButton {
                    pos: egui::pos2(700.0, 350.0), button: egui::PointerButton::Primary,
                    pressed: true, modifiers: egui::Modifiers::NONE,
                }] } else { Vec::new() },
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                st.show_side_panel(ui, item, &backend, &mut canvas);
                st.show_central(ui, item, &backend, &mut canvas, &mut view3d);
            });
            times.push(ms(t));
        }
        assert!(st.dragging, "synthetic drag must stay held until release");
        times.sort_by(f64::total_cmp);
        println!("# time UI drag frames {key}: median={:.3} p95={:.3} max={:.3} ms", times[20], times[38], times[39]);
        let t = Instant::now();
        st.finish_drag(item);
        println!("# time UI release {key}: {:.3} ms", ms(t));
        st.wait_idle(item, Some(&mut canvas));
        for grid in 1..=3 {
            st.grid = grid;
            let t = Instant::now();
            st.run_compare(item);
            let submit = ms(t);
            st.wait_idle(item, Some(&mut canvas));
            println!("# time UI compare {key} grid={grid}: submit={submit:.3} worker={:.3} ms", st.compare_ms);
        }
        let t = Instant::now();
        st.reload_for_item(item, &backend, &mut canvas, true);
        println!("# time UI lattice reload {key}: {:.3} ms", ms(t));
        st.wait_idle(item, Some(&mut canvas));
    }
}

#[test]
#[ignore = "3D worker timing harness; run manually after the small sheet pilot"]
fn combine_bench_sheet_worker() {
    use std::sync::Arc;
    use std::time::Duration;
    use crate::combine_worker::{BuildJob, spawn_build_worker};
    for key in ["square", "honeycomb"] {
        let backend = AppBackend::new();
        let item = backend.find_named(key).unwrap();
        let info = Arc::new(combine_lattice_info(item));
        let mut worker = spawn_build_worker();
        for radius in [4.5, info.r_auto_base] {
            for preview in [false, true] {
                let t = Instant::now();
                worker.submit(BuildJob {
                    item: item.clone(), info: Arc::clone(&info),
                    specs: preset_neighbour_pair(&info), opts: opts(Some(radius)),
                    preview, kappa_b: 0.02, reuse_2d: None,
                });
                let (out, sheet) = worker.wait(Duration::from_secs(30)).expect("3D worker completed").unwrap();
                println!("# time 2D+3D worker {key} R={radius:.2} preview={preview}: {:.3} ms; N={} iterations={} converged={}",
                    ms(t), out.crystal.pos.len(), sheet.sheet.iterations, sheet.sheet.converged);
                assert_eq!(sheet.sheet.xyz.len(), out.crystal.pos.len());
                assert!(sheet.sheet.xyz.iter().flatten().all(|x| x.is_finite()));
            }
        }
    }
}
