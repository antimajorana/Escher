//! Combine-tab snapping and Burgers-vector helpers against the pure-Python package.
//! Expected values: tests/fixtures/combine_parity_py.json (`snap`), written by
//! tools/checks/combine_parity_dump.py from `papertilling.defects.build.LatticeData` (centres, nearest_centre).

use serde_json::Value;

use super::*;
use crate::backend::{AppBackend, CatalogueRuleUi};

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/combine_parity_py.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read fixture")).expect("parse fixture")
}

fn v2(v: &Value) -> [f64; 2] {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
}

fn dist(p: [f64; 2], q: [f64; 2]) -> f64 {
    (p[0] - q[0]).hypot(p[1] - q[1])
}

/// The app's catalogue item for a fixture lattice; its recipe and frame must be the ones Python used.
fn item_for(backend: &AppBackend, lat: &Value) -> CatalogueItemUi {
    let item = match lat["source"].as_str().unwrap() {
        "named" => backend.find_named(lat["name"].as_str().unwrap()).expect("named lattice").clone(),
        rule => {
            assert_eq!(rule, "points");
            let idx = lat["index"].as_u64().unwrap() as usize;
            backend.catalogue_for_group(lat["group"].as_str().unwrap(), CatalogueRuleUi::Points)[idx].clone()
        }
    };
    let recipe: papertilling_core::types::Recipe = serde_json::from_value(lat["recipe"].clone()).unwrap();
    let frame: papertilling_core::types::Frame2D = serde_json::from_value(lat["frame"].clone()).unwrap();
    assert_eq!(item.group, recipe.group);
    let cell_rs = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).unwrap().cell();
    let cell_py = LatticeFamily::from_recipe(&recipe, Some(frame)).unwrap().cell();
    assert!(cell_rs.basis.iter().all(|p| cell_py.basis.iter().any(|q| dist(*p, *q) < 1e-6)), "{}: app lattice differs from Python's", item.label);
    assert!(dist(cell_rs.a1, cell_py.a1) < 1e-6 && dist(cell_rs.a2, cell_py.a2) < 1e-6);
    item
}

struct PyClass {
    label: String,
    order: usize,
    members: Vec<[f64; 2]>,
    angles: Vec<f64>,
}

fn py_classes(snap: &Value) -> Vec<PyClass> {
    snap["centres"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| PyClass {
            label: c["label"].as_str().unwrap().to_string(),
            order: c["order"].as_u64().unwrap() as usize,
            members: c["members"].as_array().unwrap().iter().map(v2).collect(),
            angles: c["angles"].as_array().unwrap().iter().map(|a| a.as_f64().unwrap()).collect(),
        })
        .collect()
}

/// Nearest point of a Python class (members + lattice translations) to `near`, by brute force.
fn nearest_in_class(info: &CombineLatticeInfo, class: &PyClass, near: [f64; 2]) -> f64 {
    let a = info.lat_info.lat.a;
    let mut best = f64::INFINITY;
    for m in &class.members {
        for i in -6..=6 {
            for j in -6..=6 {
                let p = [m[0] + a[0][0] * i as f64 + a[0][1] * j as f64, m[1] + a[1][0] * i as f64 + a[1][1] * j as f64];
                best = best.min(dist(p, near));
            }
        }
    }
    best
}

fn in_class(info: &CombineLatticeInfo, class: &PyClass, p: [f64; 2]) -> bool {
    class.members.iter().any(|&m| same_mod_cell(&info.lat_info, p, m))
}

#[test]
fn centre_classes_match_python_labels_orders_and_members() {
    let (fx, backend) = (fixture(), AppBackend::new());
    for snap in fx["snap"].as_array().unwrap() {
        let item = item_for(&backend, &snap["lattice"]);
        let info = combine_lattice_info(&item);
        let py = py_classes(snap);
        assert_eq!(info.centres.len(), py.len(), "{}: number of G+ centre classes", item.label);
        for c in &info.centres {
            let pc = py.iter().find(|p| p.label == c.label).unwrap_or_else(|| panic!("{}: Rust class {} not in Python", item.label, c.label));
            assert_eq!(c.order, pc.order, "{} {}", item.label, c.label);
            assert_eq!(c.angles, pc.angles, "{} {}", item.label, c.label);
            assert_eq!(c.unit_seeds.len(), pc.members.len(), "{} {}: class size", item.label, c.label);
            for &s in &c.unit_seeds {
                assert!(in_class(&info, pc, s), "{} {}: seed {s:?} not a member of the Python class", item.label, c.label);
            }
        }
    }
}

#[test]
fn class_filter_snaps_onto_that_class_only() {
    let (fx, backend) = (fixture(), AppBackend::new());
    let mut checked = 0;
    for snap in fx["snap"].as_array().unwrap() {
        let item = item_for(&backend, &snap["lattice"]);
        let info = combine_lattice_info(&item);
        let py = py_classes(snap);
        let queries: Vec<[f64; 2]> = snap["nearest"].as_array().unwrap().iter().map(|q| v2(&q["near"])).collect();
        for c in &info.centres {
            let pc = py.iter().find(|p| p.label == c.label).unwrap();
            for &ang in &c.angles {
                for &near in &queries {
                    let (p, lbl) = snap_disclination_centre(&item, &info, ang, Some(&c.id), near);
                    assert_eq!(lbl, c.label);
                    assert!(in_class(&info, pc, p), "{} {} {ang}: {p:?} is not in class {}", item.label, c.id, c.label);
                    let want = nearest_in_class(&info, pc, near);
                    assert!((dist(p, near) - want).abs() < 1e-6, "{} {}: not the nearest point of the class", item.label, c.label);
                    assert!(info.lat_info.lat.is_symmetry(&iso_rotation_about(p, ang), 2.2), "{ang} not allowed at {p:?}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "only {checked} filtered snaps checked");
}

#[test]
fn unfiltered_snap_matches_python_nearest_centre() {
    let (fx, backend) = (fixture(), AppBackend::new());
    for snap in fx["snap"].as_array().unwrap() {
        let item = item_for(&backend, &snap["lattice"]);
        let info = combine_lattice_info(&item);
        let py = py_classes(snap);
        for q in snap["nearest"].as_array().unwrap() {
            let (ang, near, py_pt) = (q["angle"].as_f64().unwrap(), v2(&q["near"]), v2(&q["point"]));
            let (p, lbl) = snap_disclination_centre(&item, &info, ang, None, near);
            assert!((dist(p, near) - dist(py_pt, near)).abs() < 1e-6, "{} {ang} near {near:?}: {p:?} vs Python {py_pt:?}", item.label);
            let pc = py.iter().find(|c| in_class(&info, c, p)).expect("snapped onto a Python centre");
            assert_eq!(lbl, pc.label, "{}: label of the snapped centre", item.label);
            if dist(p, py_pt) < 1e-6 {
                assert_eq!(lbl, q["label"].as_str().unwrap());
            }
        }
    }
}

#[test]
fn cached_snapper_equals_core_nearest_allowed_centre_and_is_fast() {
    use papertilling_core::defect::combine_api::nearest_allowed_centre;
    let backend = AppBackend::new();
    let mut items: Vec<CatalogueItemUi> = ["square", "kagome", "shastry_sutherland", "chiral_triangular", "honeycomb"]
        .iter()
        .map(|k| backend.find_named(k).unwrap_or_else(|| panic!("named {k}")).clone())
        .collect();
    assert_ne!(items[2].frame.origin, [0.0, 0.0], "shastry_sutherland is stored with a shifted frame origin");
    for g in ["p3", "p31m", "p4g", "p6", "cmm"] {
        items.push(backend.catalogue_for_group(g, CatalogueRuleUi::Points)[0].clone());
    }
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for item in &items {
        let info = combine_lattice_info(item);
        let a = info.lat_info.lat.a;
        let mut angles: Vec<f64> = info.allowed_angles.iter().map(|&d| d as f64).collect();
        angles.push(37.0); // allowed nowhere: both must refuse
        let (mut t_core, mut t_cached, mut n) = (0.0, 0.0, 0usize);
        for &ang in &angles {
            for _ in 0..30 {
                let (u, v) = (6.0 * rnd() - 3.0, 6.0 * rnd() - 3.0);
                let near = [a[0][0] * u + a[0][1] * v, a[1][0] * u + a[1][1] * v];
                let t0 = std::time::Instant::now();
                let want = nearest_allowed_centre(&item.group, &info.cell, item.frame, ang, near);
                t_core += t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let got = info.snapper.nearest(ang, near);
                t_cached += t1.elapsed().as_secs_f64();
                n += 1;
                assert_eq!(got, want, "{} angle {ang} near {near:?}", item.label);
                let app = snap_disclination_centre(item, &info, ang, None, near);
                match want {
                    Ok(w) => assert_eq!(app, w),
                    Err(_) => assert_eq!(app, (near, String::from("unconstrained"))),
                }
            }
        }
        println!(
            "# time unfiltered snap {:24}: {:.3} ms/call rebuilding the catalogue, {:.4} ms/call cached ({n} calls)",
            item.label,
            1e3 * t_core / n as f64,
            1e3 * t_cached / n as f64
        );
    }
}

#[test]
fn filter_that_forbids_the_angle_falls_back_to_nearest_allowed_centre() {
    // Square lattice, hand: 2-fold class 2c does not allow 90 deg. Nearest 4-fold centre to (0.1, 0.45):
    // (0, 0) at 0.461, (1/2, 1/2) at 0.403, (0, 1) at 0.559 -> the face centre (1/2, 1/2), class 1b.
    let backend = AppBackend::new();
    let sq = backend.find_named("square").unwrap();
    let info = combine_lattice_info(sq);
    let c2 = info.centres.iter().find(|c| c.label == "2c").unwrap();
    let (p, lbl) = snap_disclination_centre(sq, &info, 90.0, Some(&c2.id), [0.1, 0.45]);
    assert!(dist(p, [0.5, 0.5]) < 1e-9 && lbl == "1b", "{p:?} {lbl}");
}

#[test]
fn dragging_without_filter_may_change_class_like_python_snap() {
    // p3 #1: classes 1b (1/3, 2/3) and 1c (2/3, 1/3) both allow +-120. The drag path snaps without the class
    // filter (Python build(snap=True) semantics): a 1b pin dragged next to a 1c point lands on 1c.
    let backend = AppBackend::new();
    let p3 = &backend.catalogue_for_group("p3", CatalogueRuleUi::Points)[0];
    let info = combine_lattice_info(p3);
    let c1c = info.centres.iter().find(|c| c.label == "1c").unwrap().point;
    let near = [c1c[0] + 0.02, c1c[1] - 0.01];
    let (p, lbl) = snap_disclination_centre(p3, &info, 120.0, None, near);
    assert!(dist(p, c1c) < 1e-9 && lbl == "1c");
    let b_id = &info.centres.iter().find(|c| c.label == "1b").unwrap().id;
    let (pb, lblb) = snap_disclination_centre(p3, &info, 120.0, Some(b_id), near);
    assert_eq!(lblb, "1b");
    assert!(dist(pb, c1c) > 0.3);
    // the pin label names the class, so such a class change is visible (catalogue naming D(label, angle))
    assert_eq!(spec_short_label(&info, &DefectSpec::disclination(120.0, pb)), "D(1b, +120°)");
    assert_eq!(spec_short_label(&info, &DefectSpec::disclination(-120.0, p)), "D(1c, -120°)");
    let shifted = [p[0] + info.cell.a1[0] - info.cell.a2[0], p[1] + info.cell.a1[1] - info.cell.a2[1]];
    assert_eq!(centre_class_at(&info, shifted).map(|c| c.label.as_str()), Some("1c"));
    assert_eq!(spec_short_label(&info, &DefectSpec::disclination(120.0, near)), "D(+120°)");
}

#[test]
fn default_dislocation_position_follows_python_rule() {
    // Python validate_specs: at = 0.371 a1 + 0.413 a2 + 3 idx a1 with (a1, a2) the reduced primitive cell of
    // LatticeData (dumped as snap[].a1, a2).
    let (fx, backend) = (fixture(), AppBackend::new());
    for snap in fx["snap"].as_array().unwrap() {
        let item = item_for(&backend, &snap["lattice"]);
        let info = combine_lattice_info(&item);
        let (a1, a2) = (v2(&snap["a1"]), v2(&snap["a2"]));
        let b = info.dislocations.first().map_or(info.cell.a1, |d| d.b);
        for idx in 0..3 {
            let k = 0.371 + 3.0 * idx as f64;
            let want = [k * a1[0] + 0.413 * a2[0], k * a1[1] + 0.413 * a2[1]];
            let got = spec_world_pos(&info, idx, &DefectSpec::dislocation(b, None));
            assert!(dist(got, want) < 1e-9, "{} idx {idx}: marker at {got:?}, Python puts the cut end at {want:?}", item.label);
        }
    }
}

#[test]
fn burgers_mn_round_trips_with_catalogue_classes() {
    let (fx, backend) = (fixture(), AppBackend::new());
    for snap in fx["snap"].as_array().unwrap() {
        let item = item_for(&backend, &snap["lattice"]);
        let info = combine_lattice_info(&item);
        assert!(!info.dislocations.is_empty());
        for d in &info.dislocations {
            let (m, n) = d.burgers_mn;
            assert!(dist(burgers_from_mn(&info, m, n), d.b) < 1e-9, "{} {}: (m, n) = {:?} gives another b", item.label, d.id, d.burgers_mn);
            assert_eq!(mn_from_burgers(&info, d.b), Some(d.burgers_mn), "{} {}", item.label, d.id);
            assert_eq!(spec_short_label(&info, &DefectSpec::dislocation(d.b, None)), format!("T({m},{n})"));
        }
        assert_eq!(mn_from_burgers(&info, [0.5 * info.burgers_basis[0][0], 0.5 * info.burgers_basis[0][1]]), None);
    }
}
