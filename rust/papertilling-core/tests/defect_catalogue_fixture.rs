//! Defect catalogue against Python's `DefectCatalogue` on the lattices the app's Defects tab offers: the first
//! `points.json` lattice of each of the 17 groups and the 21 named lattices (`named.json`, with their frames).
//!
//! Reference: `tests/fixtures/defect_catalogues.json`, written by the Python package (not by this code) with
//! `PAPERTILLING_NO_RUST=1 python rust/papertilling-core/tests/fixtures/dump_defect_catalogues.py`.
//! Every mismatch is collected and reported at once.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Deserialize;

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::plane::catalogue::{parse_catalogue_str, parse_named_str};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Frame2D, Recipe};

const FIXTURE: &str = include_str!("fixtures/defect_catalogues.json");
const POINTS_JSON: &str = include_str!("../../../papertilling/data/points.json");
const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

#[derive(Deserialize)]
struct Fixture {
    lattices: BTreeMap<String, RefCat>,
}

#[derive(Deserialize)]
struct RefCat {
    group: String,
    chiral: bool,
    #[serde(rename = "R0")]
    r0: f64,
    b_min: f64,
    dislocations: Vec<RefDisloc>,
    centre_classes: Vec<RefCentre>,
    disclinations: Vec<RefDiscl>,
    mirrors: Vec<RefMirror>,
    normaliser: RefNormaliser,
    domain_walls: Vec<RefDomain>,
    twin_walls: Vec<RefTwin>,
    mobius_seams: Vec<RefMobius>,
    glide_breaking: Vec<RefGlide>,
}

#[derive(Deserialize)]
struct RefDisloc {
    id: String,
    b_frac: [i32; 2],
    b: [f64; 2],
    length: f64,
    angle: f64,
    size: usize,
    self_inverse: bool,
    inverse_id: String,
    mirror_id: Option<String>,
}

#[derive(Deserialize)]
struct RefCentre {
    id: String,
    order: usize,
    gclass: String,
    label: String,
    location: String,
    mirror_of: Option<String>,
    point: [f64; 2],
    size: usize,
    split: bool,
    angles: Vec<f64>,
}

#[derive(Deserialize)]
struct RefDiscl {
    id: String,
    centre_class: String,
    wyckoff: String,
    gclass: String,
    order: usize,
    angle: f64,
    location: String,
    point: [f64; 2],
    inverse_id: String,
    mirror_id: Option<String>,
}

#[derive(Deserialize)]
struct RefMirror {
    id: String,
    kind: String,
    glide: f64,
    period: f64,
    offset: f64,
    spacing: f64,
    through_site: bool,
    size: usize,
    direction: [f64; 2],
}

#[derive(Deserialize)]
struct RefCoset {
    id: String,
    kind: String,
    centre_map: String,
    sites_to: String,
    check_group: bool,
    check_geometry: bool,
}

#[derive(Deserialize)]
struct RefNormaliser {
    index: usize,
    continuous: usize,
    complete: bool,
    ita_generic_index: Option<usize>,
    note: String,
    holohedry_order: usize,
    point_group_order: usize,
    cosets: Vec<RefCoset>,
}

#[derive(Deserialize)]
struct RefDomain {
    id: String,
    coset: String,
    #[serde(rename = "type")]
    kind: String,
    centre_map: String,
    sites_to: String,
    complete: bool,
}

#[derive(Deserialize)]
struct RefTwin {
    id: String,
    direction_frac: [i32; 2],
    through_site: bool,
    #[serde(rename = "sigma_in_G")]
    sigma_in_g: bool,
    network_invisible: bool,
    n_irregular: usize,
    visible: bool,
    domain_coset: Option<String>,
    class_size: usize,
    offset: f64,
}

#[derive(Deserialize)]
struct RefMobius {
    id: String,
    line: String,
    line_kind: String,
}

#[derive(Deserialize)]
struct RefGlide {
    id: String,
    line: String,
    line_kind: String,
    slip: f64,
    slip_over_period: f64,
    same_domains_as: Option<String>,
    same_as_twin: Option<String>,
}

/// (tag, group passed by the app, recipe, frame), in the app's order.
fn app_lattices() -> Vec<(String, String, Recipe, Frame2D)> {
    let points = parse_catalogue_str(POINTS_JSON).expect("points.json");
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    let mut out = Vec::new();
    for (g, cat) in &points {
        let e = cat.all.first().expect("first points lattice");
        out.push((format!("{g}#1"), e.recipe.group.clone(), e.recipe.clone(), Frame2D::default()));
    }
    for nl in &named.named {
        out.push((nl.key.clone(), nl.group.clone(), nl.recipe.clone(), nl.frame));
    }
    out
}

struct Diff<'a> {
    tag: &'a str,
    errs: &'a mut Vec<String>,
}

impl Diff<'_> {
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, what: &str, got: T, want: T) {
        if got != want {
            self.errs.push(format!("{} {what}: rust {got:?} != python {want:?}", self.tag));
        }
    }
    fn close(&mut self, what: &str, got: f64, want: f64, tol: f64) {
        if (got - want).abs() > tol {
            self.errs.push(format!("{} {what}: rust {got} != python {want}", self.tag));
        }
    }
    fn close2(&mut self, what: &str, got: [f64; 2], want: [f64; 2], tol: f64) {
        self.close(&format!("{what}[0]"), got[0], want[0], tol);
        self.close(&format!("{what}[1]"), got[1], want[1], tol);
    }
    fn len(&mut self, what: &str, got: usize, want: usize) -> bool {
        self.eq(&format!("{what} count"), got, want);
        got == want
    }
}

fn compare(d: &mut Diff, cat: &DefectCatalogue, r: &RefCat) {
    d.eq("group", cat.group.as_str(), r.group.as_str());
    d.eq("chiral", cat.chiral, r.chiral);
    d.close("R0", cat.r0, r.r0, 1e-7);
    d.close("b_min", cat.b_min, r.b_min, 1e-7);
    if d.len("dislocations", cat.dislocations.len(), r.dislocations.len()) {
        for (g, e) in cat.dislocations.iter().zip(&r.dislocations) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} b_frac"), g.b_frac, e.b_frac);
            d.close2(&format!("{k} b"), g.b, e.b, 1e-7);
            d.close(&format!("{k} length"), g.length, e.length, 1e-7);
            d.close(&format!("{k} angle"), g.angle, e.angle, 1e-5);
            d.eq(&format!("{k} size"), g.size, e.size);
            d.eq(&format!("{k} self_inverse"), g.self_inverse, e.self_inverse);
            d.eq(&format!("{k} inverse_id"), &g.inverse_id, &e.inverse_id);
            d.eq(&format!("{k} mirror_id"), &g.mirror_id, &e.mirror_id);
        }
    }
    if d.len("centre_classes", cat.centre_classes.len(), r.centre_classes.len()) {
        for (g, e) in cat.centre_classes.iter().zip(&r.centre_classes) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} order"), g.order, e.order);
            d.eq(&format!("{k} gclass"), &g.gclass, &e.gclass);
            d.eq(&format!("{k} label"), &g.label, &e.label);
            d.eq(&format!("{k} location"), &g.location, &e.location);
            d.eq(&format!("{k} mirror_of"), &g.mirror_of, &e.mirror_of);
            d.close2(&format!("{k} point"), g.point, e.point, 1e-6);
            d.eq(&format!("{k} size"), g.size, e.size);
            d.eq(&format!("{k} split"), g.split, e.split);
            d.eq(&format!("{k} angles"), &g.angles, &e.angles);
        }
    }
    if d.len("disclinations", cat.disclinations.len(), r.disclinations.len()) {
        for (g, e) in cat.disclinations.iter().zip(&r.disclinations) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} centre_class"), &g.centre_class, &e.centre_class);
            d.eq(&format!("{k} wyckoff"), &g.wyckoff, &e.wyckoff);
            d.eq(&format!("{k} gclass"), &g.gclass, &e.gclass);
            d.eq(&format!("{k} order"), g.order, e.order);
            d.eq(&format!("{k} angle"), g.angle, e.angle);
            d.eq(&format!("{k} location"), &g.location, &e.location);
            d.close2(&format!("{k} point"), g.point, e.point, 1e-6);
            d.eq(&format!("{k} inverse_id"), &g.inverse_id, &e.inverse_id);
            d.eq(&format!("{k} mirror_id"), &g.mirror_id, &e.mirror_id);
        }
    }
    if d.len("mirrors", cat.mirrors.len(), r.mirrors.len()) {
        for (g, e) in cat.mirrors.iter().zip(&r.mirrors) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} kind"), &g.kind, &e.kind);
            d.close(&format!("{k} glide"), g.glide, e.glide, 1e-6);
            d.close(&format!("{k} period"), g.period, e.period, 1e-6);
            d.close(&format!("{k} offset"), g.offset, e.offset, 1e-6);
            d.close(&format!("{k} spacing"), g.spacing, e.spacing, 1e-6);
            d.eq(&format!("{k} through_site"), g.through_site, e.through_site);
            d.eq(&format!("{k} size"), g.size, e.size);
            d.close2(&format!("{k} direction"), g.direction, e.direction, 1e-6);
        }
    }
    let (n, rn) = (&cat.normaliser, &r.normaliser);
    d.eq("normaliser index", n.index, rn.index);
    d.eq("normaliser continuous", n.continuous, rn.continuous);
    d.eq("normaliser complete", n.complete, rn.complete);
    d.eq("normaliser ita_generic_index", n.ita_generic_index, rn.ita_generic_index);
    d.eq("normaliser note", &n.note, &rn.note);
    d.eq("normaliser holohedry_order", n.holohedry_order, rn.holohedry_order);
    d.eq("normaliser point_group_order", n.point_group_order, rn.point_group_order);
    if d.len("cosets", n.cosets.len(), rn.cosets.len()) {
        for (g, e) in n.cosets.iter().zip(&rn.cosets) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} kind"), &g.kind, &e.kind);
            d.eq(&format!("{k} centre_map"), &g.centre_map, &e.centre_map);
            d.eq(&format!("{k} sites_to"), &g.sites_to, &e.sites_to);
            d.eq(&format!("{k} check_group"), g.check_group, e.check_group);
            d.eq(&format!("{k} check_geometry"), g.check_geometry, e.check_geometry);
        }
    }
    if d.len("domain_walls", cat.domain_walls.len(), r.domain_walls.len()) {
        for (g, e) in cat.domain_walls.iter().zip(&r.domain_walls) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} coset"), &g.coset, &e.coset);
            d.eq(&format!("{k} type"), &g.kind, &e.kind);
            d.eq(&format!("{k} centre_map"), &g.centre_map, &e.centre_map);
            d.eq(&format!("{k} sites_to"), &g.sites_to, &e.sites_to);
            d.eq(&format!("{k} complete"), g.complete, e.complete);
        }
    }
    if d.len("twin_walls", cat.twin_walls.len(), r.twin_walls.len()) {
        for (g, e) in cat.twin_walls.iter().zip(&r.twin_walls) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} direction_frac"), g.direction_frac, e.direction_frac);
            d.eq(&format!("{k} through_site"), g.through_site, e.through_site);
            d.eq(&format!("{k} sigma_in_G"), g.sigma_in_g, e.sigma_in_g);
            d.eq(&format!("{k} network_invisible"), g.network_invisible, e.network_invisible);
            d.eq(&format!("{k} n_irregular"), g.n_irregular, e.n_irregular);
            d.eq(&format!("{k} visible"), g.visible, e.visible);
            d.eq(&format!("{k} domain_coset"), &g.domain_coset, &e.domain_coset);
            d.eq(&format!("{k} class_size"), g.class_size, e.class_size);
            d.close(&format!("{k} offset"), g.offset, e.offset, 1e-6);
        }
    }
    if d.len("mobius_seams", cat.mobius_seams.len(), r.mobius_seams.len()) {
        for (g, e) in cat.mobius_seams.iter().zip(&r.mobius_seams) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} line"), &g.line, &e.line);
            d.eq(&format!("{k} line_kind"), &g.line_kind, &e.line_kind);
        }
    }
    if d.len("glide_breaking", cat.glide_breaking.len(), r.glide_breaking.len()) {
        for (g, e) in cat.glide_breaking.iter().zip(&r.glide_breaking) {
            let k = &e.id;
            d.eq(&format!("{k} id"), &g.id, &e.id);
            d.eq(&format!("{k} line"), &g.line, &e.line);
            d.eq(&format!("{k} line_kind"), &g.line_kind, &e.line_kind);
            d.close(&format!("{k} slip"), g.slip, e.slip, 1e-6);
            d.close(&format!("{k} slip_over_period"), g.slip_over_period, e.slip_over_period, 1e-6);
            d.eq(&format!("{k} same_domains_as"), &g.same_domains_as, &e.same_domains_as);
            d.eq(&format!("{k} same_as_twin"), &g.same_as_twin, &e.same_as_twin);
        }
    }
}

/// Regression (Rust had p2 16, p3 6, p3m1 2, p31m/p4m/cmm 1). Hand derivation of [N_E(G) : G] for a generic
/// metric, |P_N| / |P| x [T_N : T]: p2 -> p2 (a/2, b/2): 1 x 4; p4m -> p4m ((a-b)/2, (a+b)/2): 1 x 2;
/// p3 -> p6m ((2a+b)/3, (-a+b)/3): 4 x 3; p3m1 -> the same p6m: 2 x 3; p31m -> p6m: 2 x 1; p6m: 1;
/// cmm -> pmm (a/2, b/2): 1 x 2 (the centred lattice has index 2 in it).
#[test]
fn ita_generic_normaliser_index_known_answers() {
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    for (key, want) in [
        ("oblique", 4),
        ("square", 2),
        ("chiral_triangular", 12),
        ("breathing_kagome", 6),
        ("fractal_triangular", 2),
        ("triangular", 1),
        ("elongated_triangular", 2),
    ] {
        let nl = named.named.iter().find(|n| n.key == key).expect(key);
        let cell = LatticeFamily::from_recipe(&nl.recipe, Some(nl.frame)).expect("recipe").cell();
        let cat = DefectCatalogue::new(&nl.group, &cell, nl.frame);
        assert_eq!(cat.normaliser.ita_generic_index, Some(want), "{key} ({})", cat.group);
        assert_eq!(cat.normaliser.index, want, "{key}: generic metric, index equals ITA's");
        assert_eq!(cat.normaliser.note, "index equals ITA's generic value", "{key}");
    }
}

#[test]
fn defect_catalogues_match_python_on_17_first_and_21_named_lattices() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("parse fixture");
    let lats = app_lattices();
    assert_eq!(lats.len(), 38, "17 first lattices + 21 named lattices");
    assert_eq!(fixture.lattices.len(), 38);
    let mut errs = Vec::new();
    let t0 = Instant::now();
    for (tag, group, recipe, frame) in &lats {
        let refc = fixture.lattices.get(tag).unwrap_or_else(|| panic!("fixture has no {tag}"));
        let cell = LatticeFamily::from_recipe(recipe, Some(*frame)).expect("recipe").cell();
        let t = Instant::now();
        let cat = DefectCatalogue::new(group, &cell, *frame);
        println!("# time {tag}: {:.3} s", t.elapsed().as_secs_f64());
        compare(&mut Diff { tag, errs: &mut errs }, &cat, refc);
    }
    println!("# time 38 catalogues: {:.2} s", t0.elapsed().as_secs_f64());
    assert!(errs.is_empty(), "{} mismatches with Python:\n{}", errs.len(), errs.join("\n"));
}
