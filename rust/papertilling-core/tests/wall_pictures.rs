//! Wall pictures (`DefectCatalogue::wall_picture`, the Defects tab's "Relaxed / Embedded" wall drawing) on the 12
//! lattices of `tools/checks/drawing_check.py` (Stage 4 gate), under the rules of `tests/test_drawing.py`: no
//! crossing, overlap or crowded site, except a glide-breaking wall repaired locally with at most one crowded site
//! noted, and a wall that cannot be drawn cleanly returned as its rigid picture with a note. Measured here by an
//! independent brute-force copy of `drawing_check.measure` (not by `plane::planarity`).
//!
//! Reference: `tests/fixtures/wall_pictures.json`, written by the Python package
//! (`PAPERTILLING_NO_RUST=1 python rust/papertilling-core/tests/fixtures/dump_wall_pictures.py`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use serde::Deserialize;

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::plane::catalogue::{parse_catalogue_str, parse_named_str};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{BuiltCrystal, Frame2D};

const FIXTURE: &str = include_str!("fixtures/wall_pictures.json");
const POINTS_JSON: &str = include_str!("../../../papertilling/data/points.json");
const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

#[derive(Deserialize)]
struct Fixture {
    lattices: BTreeMap<String, RefLattice>,
}

#[derive(Deserialize)]
struct RefLattice {
    #[serde(rename = "R0")]
    r0: f64,
    walls: BTreeMap<String, RefWall>,
}

#[derive(Deserialize)]
struct RefWall {
    stage: String,
    sites: usize,
    bonds: usize,
    crossings: usize,
}

/// The 12 lattices (tag, catalogue), built the way the app builds them.
fn gate_lattices() -> Vec<(String, DefectCatalogue)> {
    let points = parse_catalogue_str(POINTS_JSON).expect("points.json");
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    let mut out = Vec::new();
    for (tag, g, k) in [("p4g#1", "p4g", 0), ("p6m#4", "p6m", 3)] {
        let e = &points[g].all[k];
        let cell = LatticeFamily::from_recipe(&e.recipe, Some(Frame2D::default())).unwrap().cell();
        out.push((tag.to_string(), DefectCatalogue::new(&e.recipe.group, &cell, Frame2D::default())));
    }
    for key in ["square", "kagome", "honeycomb", "shastry_sutherland", "chiral_square", "chiral_kagome", "zigzag", "sutherlands_brick", "brick_pg", "straight_p1"] {
        let nl = named.named.iter().find(|n| n.key == key).unwrap();
        let cell = LatticeFamily::from_recipe(&nl.recipe, Some(nl.frame)).unwrap().cell();
        out.push((key.to_string(), DefectCatalogue::new(&nl.group, &cell, nl.frame)));
    }
    out
}

#[derive(Debug, Default, PartialEq)]
struct Measure {
    crossings: usize,
    overlaps: usize,
    crowded: usize,
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { 1.0 } else if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) }
}

/// `drawing_check.measure` (plain_crossings, overlaps, crowded) by brute force over all pairs.
fn measure(c: &BuiltCrystal) -> Measure {
    let x = &c.pos;
    let ghost = |u: usize, v: usize| c.adj[u].iter().any(|d| d.to == v && d.ghost_stub.is_some());
    let plain: Vec<(usize, usize, usize)> = c
        .edges
        .iter()
        .filter(|&&(u, v)| !ghost(u, v) && !ghost(v, u))
        .map(|&(u, v)| (u, v, c.adj[u].iter().find(|d| d.to == v).map_or(0, |d| d.kind)))
        .collect();
    let lens: Vec<f64> = plain.iter().map(|&(u, v, _)| sub(x[v], x[u]).iter().map(|t| t * t).sum::<f64>().sqrt()).collect();
    let tol = 0.01 * median(lens.clone());
    let mut by_kind: HashMap<usize, Vec<f64>> = HashMap::new();
    for (k, &(_, _, t)) in plain.iter().enumerate() {
        by_kind.entry(t).or_default().push(lens[k]);
    }
    let unit = by_kind.into_values().map(median).fold(f64::INFINITY, f64::min);
    let ctol = 0.35 * if unit.is_finite() { unit } else { 1.0 };
    let mut m = Measure::default();
    for k in 0..plain.len() {
        let (a, b, _) = plain[k];
        let (pa, r) = (x[a], sub(x[b], x[a]));
        let lk = lens[k].max(1e-300);
        let dk = [r[0] / lk, r[1] / lk];
        for j in (k + 1)..plain.len() {
            let (c2, d2, _) = plain[j];
            let q = sub(x[d2], x[c2]);
            let shared = a == c2 || a == d2 || b == c2 || b == d2;
            let den = cross(r, q);
            if !shared && den.abs() > 1e-14 * lk * lens[j] {
                let w = sub(x[c2], pa);
                let (t, u) = (cross(w, q) / den, cross(w, r) / den);
                if t > 1e-9 && t < 1.0 - 1e-9 && u > 1e-9 && u < 1.0 - 1e-9 {
                    m.crossings += 1;
                }
            }
            let (wa, wb) = (sub(x[c2], pa), sub(x[d2], pa));
            if cross(dk, wa).abs() <= tol && cross(dk, wb).abs() <= tol {
                let (ea, eb) = (wa[0] * dk[0] + wa[1] * dk[1], wb[0] * dk[0] + wb[1] * dk[1]);
                if lk.min(ea.max(eb)) - 0.0_f64.max(ea.min(eb)) > tol {
                    m.overlaps += 1;
                }
            }
        }
        for (i, &p) in x.iter().enumerate() {
            if i == a || i == b {
                continue;
            }
            let w = sub(p, pa);
            let (along, off) = (w[0] * dk[0] + w[1] * dk[1], cross(dk, w).abs());
            if off <= tol && along > tol && along < lk - tol {
                m.overlaps += 1;
            }
            let (da, db) = (w[0].hypot(w[1]), sub(p, x[b])[0].hypot(sub(p, x[b])[1]));
            if off < ctol && (0.0..=lk).contains(&along) && da >= ctol && db >= ctol {
                m.crowded += 1;
            }
        }
    }
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            if sub(x[i], x[j])[0].hypot(sub(x[i], x[j])[1]) < ctol {
                m.crowded += 1;
            }
        }
    }
    m
}

fn edge_set(c: &BuiltCrystal) -> HashSet<(usize, usize)> {
    c.edges.iter().copied().collect()
}

#[test]
fn wall_pictures_follow_the_drawing_rules_on_the_12_gate_lattices() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture");
    let mut errs = Vec::new();
    let mut python_clean_rust_rigid = Vec::new();
    let t0 = Instant::now();
    for (tag, cat) in gate_lattices() {
        let refl = &fixture.lattices[&tag];
        assert!((cat.r0 - refl.r0).abs() < 1e-9, "{tag}: R0");
        let ids: Vec<String> = cat.domain_walls.iter().map(|w| w.id.clone())
            .chain(cat.twin_walls.iter().map(|w| w.id.clone()))
            .chain(cat.glide_breaking.iter().map(|w| w.id.clone()))
            .collect();
        assert_eq!(ids.len(), refl.walls.len(), "{tag}: number of Wa+Wb+Wd walls");
        for id in &ids {
            let ctx = format!("{tag} {id}");
            let rw = &refl.walls[id];
            let rigid = cat.build_wall(id, None).expect("build");
            if (rigid.pos.len(), rigid.edges.len()) != (rw.sites, rw.bonds) {
                errs.push(format!("{ctx}: rigid build {} sites {} bonds, python {} {}", rigid.pos.len(), rigid.edges.len(), rw.sites, rw.bonds));
            }
            let (pic, info) = cat.wall_picture(id, None).expect("picture");
            let info = info.expect("orientable wall has drawing info");
            if edge_set(&pic) != edge_set(&rigid) || pic.pos.len() != rigid.pos.len() {
                errs.push(format!("{ctx}: the picture changed the graph"));
            }
            let m = measure(&pic);
            if info.stage == "rigid" {
                if pic.pos != rigid.pos || !info.note.starts_with("rigid picture") {
                    errs.push(format!("{ctx}: rigid fallback without the rigid positions or its note ({:?})", info.note));
                }
                if m.crossings > 0 && !info.note.contains(&format!("{} crossing", m.crossings)) {
                    errs.push(format!("{ctx}: note {:?} does not give the {} crossings", info.note, m.crossings));
                }
                if rw.stage != "rigid" {
                    python_clean_rust_rigid.push(format!("{ctx} (python {})", rw.stage));
                }
            } else {
                let local_ok = id.starts_with("Wd") && m.crowded <= 1 && info.note.contains("crowded");
                if m.crossings > 0 || m.overlaps > 0 || (m.crowded > 0 && !local_ok) {
                    errs.push(format!("{ctx}: stage {} keeps {m:?}, note {:?}", info.stage, info.note));
                }
            }
            if rw.stage == "rigid" && rw.crossings == 0 && info.stage == "rigid" && m.crossings > 0 {
                errs.push(format!("{ctx}: rigid picture crosses, python's does not"));
            }
        }
        println!("# time {tag}: {} walls, {:.1} s so far", ids.len(), t0.elapsed().as_secs_f64());
    }
    println!("# walls drawn clean by Python but rigid in Rust: {python_clean_rust_rigid:?}");
    assert!(errs.is_empty(), "{} problems:\n{}", errs.len(), errs.join("\n"));
}

#[test]
fn wall_picture_known_answers_from_test_drawing() {
    let cats: BTreeMap<String, DefectCatalogue> = gate_lattices().into_iter().collect();
    // test_p4g_wd1_not_crowded: rigid picture has crowded sites and overlaps; the picture is opened, clean, and
    // keeps every in-domain bond length.
    let p4g = &cats["p4g#1"];
    let rigid = p4g.build_wall("Wd1", None).unwrap();
    let m0 = measure(&rigid);
    assert!(m0.crowded > 0 && m0.overlaps > 0, "negative control {m0:?}");
    let (pic, info) = p4g.wall_picture("Wd1", None).unwrap();
    let info = info.unwrap();
    assert_eq!((info.stage.as_str(), info.note.as_str()), ("opened", ""));
    assert_eq!(measure(&pic), Measure::default());
    for &(u, v) in &rigid.edges {
        if !rigid.adj[u].iter().any(|d| d.to == v && d.across) {
            let l = |c: &BuiltCrystal| sub(c.pos[v], c.pos[u])[0].hypot(sub(c.pos[v], c.pos[u])[1]);
            assert!((l(&rigid) - l(&pic)).abs() < 1e-9, "in-domain bond {u}-{v} changed length");
        }
    }
    // test_wall_entry_embedded: p6m lattices[3] Wd2 crosses when rigid; the picture has no crossing or overlap,
    // at most one crowded site (then noted), and the domains keep their bond lengths (median change < 0.05).
    let p6m = &cats["p6m#4"];
    let rigid = p6m.build_wall("Wd2", None).unwrap();
    assert!(measure(&rigid).crossings > 0);
    let (pic, info) = p6m.wall_picture("Wd2", None).unwrap();
    let (m, note) = (measure(&pic), info.unwrap().note);
    assert!(m.crossings == 0 && m.overlaps == 0 && (m.crowded == 0 || (m.crowded <= 1 && note.contains("crowded"))), "{m:?} {note}");
    let mut rel: Vec<f64> = rigid
        .edges
        .iter()
        .filter(|&&(u, v)| !rigid.adj[u].iter().any(|d| d.to == v && d.across))
        .map(|&(u, v)| {
            let l = |c: &BuiltCrystal| sub(c.pos[v], c.pos[u])[0].hypot(sub(c.pos[v], c.pos[u])[1]);
            (l(&pic) / l(&rigid) - 1.0).abs()
        })
        .collect();
    rel.sort_by(f64::total_cmp);
    assert!(rel[rel.len() / 2] < 0.05, "median in-domain change {}", rel[rel.len() / 2]);
    // test_mobius_keeps_stubs_and_plot_smoke: square Wd1 overlaps when rigid, not in its picture; a Moebius seam
    // keeps its stub picture (no drawing info, same positions).
    let sq = &cats["square"];
    assert!(measure(&sq.build_wall("Wd1", None).unwrap()).overlaps > 0);
    assert_eq!(measure(&sq.wall_picture("Wd1", None).unwrap().0).overlaps, 0);
    let (seam, none) = sq.wall_picture("Wc1", None).unwrap();
    assert!(none.is_none() && seam.pos == sq.build_wall("Wc1", None).unwrap().pos);
    assert!(seam.adj.iter().flatten().any(|d| d.ghost_stub.is_some()));
}
