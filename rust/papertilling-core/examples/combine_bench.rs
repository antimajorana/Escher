//! Stage timings of the Combine-tab live update and of `equivalent_single`.
//!
//! Run: `cargo run --release --manifest-path rust/Cargo.toml -p papertilling-core --example combine_bench`
//! Optional arguments: `live` (live-update stages only), `equiv` (`equivalent_single` only),
//! `big` (also the ~2000-site radius), `reps=N` (repetitions, the minimum is printed).
//! Every timing line starts with `# time <stage> <case> <ms>`.

use std::collections::BTreeMap;
use std::time::Instant;

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::charge::{check_frames, frame_faces, infer_patch_disk, total_charge};
use papertilling_core::defect::combine_api::{classify_crystal_charge, equivalent_single, inspect_crystal_cores};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph, core_graph_with_rim, patch_rim};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::catalogue::{NamedLattice, parse_named_str};
use papertilling_core::plane::draw::{PreviewReport, ensure_embedded, relax_picture, relax_preview};
use papertilling_core::plane::planarity;
use papertilling_core::plane::relax::RelaxSetup;
use papertilling_core::types::{BuiltCrystal, Cell2D};

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

struct Case {
    name: String,
    lattice: String,
    specs: Vec<DefectSpec>,
    radius: f64,
}

/// Radius rule of the app's Combine tab (`live_combine.rs`): a ~500-site disk grown with the defect spread.
fn auto_radius(cell: &Cell2D, info: &LatticeCoreInfo, specs: &[DefectSpec], scale_sites: f64) -> f64 {
    let area = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs().max(1e-6);
    let density = (cell.basis.len().max(1) as f64) / area;
    let r_n = (scale_sites / (std::f64::consts::PI * density)).sqrt();
    let base = r_n.max((6.2 * info.lmax).max(3.6 * info.scale)).clamp(3.0, if scale_sites > 600.0 { 40.0 } else { 14.0 });
    let pts: Vec<[f64; 2]> = specs
        .iter()
        .map(|s| match *s {
            DefectSpec::Disclination { at, .. } | DefectSpec::HalfWall { at, .. } => at,
            DefectSpec::Dislocation { at, .. } => at.unwrap_or([0.0, 0.0]),
        })
        .collect();
    let n = pts.len() as f64;
    let c = [pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n];
    let max_d = pts.iter().map(|p| (p[0] - c[0]).hypot(p[1] - c[1])).fold(0.0_f64, f64::max);
    let max_b = specs
        .iter()
        .filter_map(|s| match *s {
            DefectSpec::Dislocation { b, .. } => Some(b[0].hypot(b[1])),
            DefectSpec::Disclination { .. } | DefectSpec::HalfWall { .. } => None,
        })
        .fold(0.0_f64, f64::max);
    base.max(max_d + (5.8 * info.lmax + max_b).max(3.6 * info.scale))
}

fn pair_specs(lat: &NamedLattice, sep: i32) -> Vec<DefectSpec> {
    let cat = DefectCatalogue::new(&lat.group, &lat.cell, lat.frame);
    let top = cat.centre_classes.iter().map(|c| c.order).max().expect("rotation centres");
    let alpha = 360.0 / (top as f64);
    let tops: Vec<_> = cat.centre_classes.iter().filter(|c| c.order == top).collect();
    let p_plus = tops[0].point;
    let base_minus = if tops.len() >= 2 { tops[tops.len() - 1].point } else { p_plus };
    let k = if tops.len() >= 2 { sep - 1 } else { sep };
    let a1 = lat.cell.a1;
    let p_minus = [base_minus[0] + (k as f64) * a1[0], base_minus[1] + (k as f64) * a1[1]];
    vec![DefectSpec::disclination(alpha, p_plus), DefectSpec::disclination(-alpha, p_minus)]
}

fn disloc_spec(lat: &NamedLattice) -> Vec<DefectSpec> {
    let cat = DefectCatalogue::new(&lat.group, &lat.cell, lat.frame);
    let d = cat.dislocations.first().expect("dislocation class");
    vec![DefectSpec::dislocation(d.b, Some(d.at))]
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn summarize(crystal: &BuiltCrystal, info: &LatticeCoreInfo) -> usize {
    let faces = frame_faces(crystal);
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for f in &faces {
        *hist.entry(f.len()).or_default() += 1;
    }
    for r in [2_usize, 1] {
        if let Ok(g) = core_graph(crystal, info, r) {
            return g.node_count() + hist.len();
        }
    }
    hist.len()
}

fn min_time(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(ms(t));
    }
    best
}

fn live_case(case: &Case, lat: &NamedLattice, reps: usize) {
    let cell = &lat.cell;
    let info = LatticeCoreInfo::new(cell);
    let mut built = None;
    let t_build = min_time(reps, || built = Some(build_point_defects(cell, &case.specs, Some(case.radius), None)));
    let raw = match built.expect("ran") {
        Ok(c) => c,
        Err(e) => {
            println!("# skip {} (build refused: {e})", case.name);
            return;
        }
    };
    let n = raw.pos.len();
    let mut relaxed = raw.clone();
    let t_relax = min_time(reps, || {
        relaxed = raw.clone();
        relax_picture(&mut relaxed);
    });
    let mut preview = raw.clone();
    let mut prev_rep = PreviewReport::default();
    let t_preview = min_time(reps, || {
        preview = raw.clone();
        prev_rep = relax_preview(&mut preview, f64::INFINITY);
    });
    let mut chk = None;
    let t_check = min_time(reps, || chk = Some(planarity::check(&relaxed)));
    let chk = chk.expect("ran");
    let prev_chk = planarity::check(&preview);
    let setup = RelaxSetup::from_crystal(&raw, &raw.pos);
    let flat = |c: &BuiltCrystal| -> Vec<f64> { c.pos.iter().flat_map(|p| *p).collect() };
    let full_cost = setup.cost(&flat(&relaxed));
    let max_dev = preview.pos.iter().zip(&relaxed.pos).map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1])).fold(0.0_f64, f64::max);
    println!(
        "# preview {}: steps={} cg={} cost raw={:.4e} preview={:.4e} full={full_cost:.4e} max|preview-full|={max_dev:.4}",
        case.name, prev_rep.steps, prev_rep.cg_iters, prev_rep.cost_start, prev_rep.cost_end
    );
    let mut emb_note = String::new();
    let t_embed = min_time(reps, || {
        let mut c = relaxed.clone();
        emb_note = ensure_embedded(&mut c, false).stage;
    });
    let t_charge = min_time(reps, || {
        let _ = classify_crystal_charge(&lat.group, cell, lat.frame, &relaxed);
    });
    let mut core_n = 0;
    let t_core = min_time(reps, || {
        core_n = core_graph(&relaxed, &info, 2).map_or(0, |g| g.node_count());
    });
    let t_inspect = min_time(reps, || {
        let _ = inspect_crystal_cores(&relaxed);
    });
    let t_summary = min_time(reps, || {
        let _ = summarize(&relaxed, &info);
    });
    let c = &case.name;
    if std::env::args().any(|a| a == "detail") {
        let sub = [
            ("sub_lattice_core_info", min_time(reps, || drop(LatticeCoreInfo::new(cell)))),
            ("sub_frame_faces", min_time(reps, || drop(frame_faces(&relaxed)))),
            ("sub_split_frame_faces", min_time(reps, || drop(planarity::split_frame_faces(&relaxed)))),
            ("sub_infer_patch_disk", min_time(reps, || {
                let _ = std::hint::black_box(infer_patch_disk(&relaxed));
            })),
            ("sub_check_frames", min_time(reps, || {
                let _ = std::hint::black_box(check_frames(&relaxed));
            })),
            ("sub_total_charge", min_time(reps, || drop(total_charge(&relaxed)))),
            ("sub_relax_setup", min_time(reps, || drop(RelaxSetup::from_crystal(&raw, &raw.pos)))),
            ("sub_core_graph_with_rim", {
                let (ctr, r) = infer_patch_disk(&relaxed);
                let rim = patch_rim(&relaxed, &info, ctr, r);
                min_time(reps, || drop(core_graph_with_rim(&relaxed, &rim, &info, 2)))
            }),
        ];
        for (stage, t) in sub {
            println!("# time {stage} {c} {t:.3}");
        }
    }
    println!(
        "# case {c}: lattice={} R={:.2} sites={n} bonds={} crossings={} folds={:?} crowded={} core_nodes={core_n} embed_stage={emb_note} preview_crossings={} preview_folds={:?}",
        case.lattice,
        case.radius,
        relaxed.edges.len(),
        chk.crossings,
        chk.folded,
        chk.crowded,
        prev_chk.crossings,
        prev_chk.folded,
    );
    for (stage, t) in [
        ("build_point_defects", t_build),
        ("relax_picture", t_relax),
        ("relax_preview", t_preview),
        ("planarity_check", t_check),
        ("ensure_embedded", t_embed),
        ("classify_charge", t_charge),
        ("core_graph2", t_core),
        ("inspect_cores", t_inspect),
        ("summarize_rings", t_summary),
    ] {
        println!("# time {stage} {c} {t:.3}");
    }
    let live = t_build + t_relax + t_check + t_charge + t_core + t_inspect + t_summary;
    println!("# time live_total {c} {live:.3}");
    println!("# time live_total_preview {c} {:.3}", live - t_relax + t_preview);
}

fn equiv_case(lat: &NamedLattice, grid: usize, draw: bool, reps: usize) {
    let specs = disloc_spec(lat);
    let crystal = build_point_defects(&lat.cell, &specs, None, None).expect("dislocation builds");
    let mut out = None;
    let t = min_time(reps, || {
        out = Some(equivalent_single(&lat.group, &lat.cell, lat.frame, &specs, &crystal, 2, grid, draw));
    });
    let rep = out.expect("ran").expect("equivalent_single ok");
    let name = format!("{}_disloc_grid{grid}_draw{}", lat.key, if draw { 1 } else { 0 });
    println!(
        "# case {name}: equivalent={} matches={:?} gallery={} candidates={}",
        rep.equivalent,
        rep.matches,
        rep.gallery.len(),
        rep.candidates.len()
    );
    println!("# time equivalent_single {name} {t:.3}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let only_live = args.iter().any(|a| a == "live");
    let only_equiv = args.iter().any(|a| a == "equiv");
    let big = args.iter().any(|a| a == "big");
    let reps: usize = args.iter().find_map(|a| a.strip_prefix("reps=")).and_then(|s| s.parse().ok()).unwrap_or(3);
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    let get = |k: &str| named.get(k).unwrap_or_else(|| panic!("named lattice {k}"));

    if !only_equiv {
        let mut cases = Vec::new();
        for (key, seps) in [("square", [1, 6]), ("honeycomb", [1, 4]), ("kagome", [1, 4])] {
            let lat = get(key);
            let info = LatticeCoreInfo::new(&lat.cell);
            for sep in seps {
                let specs = pair_specs(lat, sep);
                for (tag, sites) in [("500", 500.0), ("2000", 2000.0)] {
                    if sites > 600.0 && !big {
                        continue;
                    }
                    let radius = auto_radius(&lat.cell, &info, &specs, sites);
                    cases.push(Case { name: format!("{key}_pair_sep{sep}_n{tag}"), lattice: key.to_string(), specs: specs.clone(), radius });
                }
            }
            if key == "square" || key == "honeycomb" {
                let specs = disloc_spec(lat);
                for (tag, sites) in [("500", 500.0), ("2000", 2000.0)] {
                    if sites > 600.0 && !big {
                        continue;
                    }
                    let radius = auto_radius(&lat.cell, &info, &specs, sites);
                    cases.push(Case { name: format!("{key}_disloc_n{tag}"), lattice: key.to_string(), specs: specs.clone(), radius });
                }
            }
        }
        let filter: Option<&str> = args.iter().find_map(|a| a.strip_prefix("only="));
        for case in cases.iter().filter(|c| filter.is_none_or(|f| c.name.contains(f))) {
            live_case(case, get(&case.lattice), reps);
        }
    }
    if !only_live {
        for key in ["square", "honeycomb"] {
            for grid in [1, 2, 3] {
                for draw in [false, true] {
                    equiv_case(get(key), grid, draw, reps.min(2));
                }
            }
        }
    }
}
