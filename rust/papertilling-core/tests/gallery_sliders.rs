//! Gallery sliders against the Python reference (`tests/fixtures/gallery_sliders.json`, written by
//! `tests/fixtures/gallery_sliders_gen.py` from the pure-Python papertilling: tuning.allowed_interval,
//! tuning.special_points and the slider logic of widgets.explore). 174 cases: entries 0, 1 and the last of every
//! group under the three rules, and the 21 named lattices; plus the Wyckoff letters and symmetry index of all
//! 533 catalogue entries.

use std::collections::BTreeMap;

use serde::Deserialize;

use papertilling_core::plane::catalogue::parse_catalogue_str;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::label_point;
use papertilling_core::plane::tuning::{SliderState, allowed_interval, slider_value};
use papertilling_core::types::Recipe;

const FIXTURE: &str = include_str!("fixtures/gallery_sliders.json");
const DATA: [(&str, &str); 3] = [
    ("points", include_str!("../../../papertilling/data/points.json")),
    ("sheet", include_str!("../../../papertilling/data/sheet.json")),
    ("bonds", include_str!("../../../papertilling/data/minimal_b.json")),
];
/// Interval / edge / segment tolerance (the scan resolution is ~1e-6; same as the Stage 2 gate).
const TOL: f64 = 1e-4;

type Sp = (f64, String, usize);

#[derive(Deserialize)]
struct Param {
    range: [f64; 3],
    limits: [f64; 2],
    interval: [f64; 2],
    edges: [Option<f64>; 2],
    current_ok: bool,
    segments: Vec<[f64; 2]>,
    special: Vec<Sp>,
}

#[derive(Deserialize)]
struct Move {
    name: String,
    value: f64,
    limits: BTreeMap<String, [f64; 2]>,
    special: BTreeMap<String, Vec<Sp>>,
}

#[derive(Deserialize)]
struct Case {
    rule: String,
    group: String,
    index: Option<usize>,
    named: Option<String>,
    recipe: Recipe,
    names: Vec<String>,
    values: BTreeMap<String, f64>,
    params: BTreeMap<String, Param>,
    #[serde(rename = "move", default)]
    moved: Option<Move>,
}

#[derive(Deserialize)]
struct Fixture {
    n_scan: usize,
    cases: Vec<Case>,
    entries: Vec<(String, String, usize, String, usize, Vec<String>)>,
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("valid gallery_sliders.json")
}

fn tag(c: &Case) -> String {
    format!("{} {} {:?} {:?}", c.rule, c.group, c.index, c.named)
}

fn check_specials(ctx: &str, got: &[papertilling_core::plane::tuning::SpecialPoint], want: &[Sp]) {
    let g: Vec<(f64, &str)> = got.iter().map(|s| (s.value, s.group.as_str())).collect();
    assert_eq!(got.len(), want.len(), "{ctx}: special values {g:?} vs Python {want:?}");
    for (s, w) in got.iter().zip(want) {
        assert!((s.value - w.0).abs() < 1e-6 && s.group == w.1, "{ctx}: special {g:?} vs Python {want:?}");
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn python_slider_limits_specials_and_coupled_moves() {
    let fx = fixture();
    assert_eq!(fx.n_scan, 401);
    assert_eq!(fx.cases.len(), 174);
    let mut n_params = 0;
    for c in &fx.cases {
        let t = tag(c);
        let mut st = SliderState::new(c.recipe.clone()).unwrap_or_else(|e| panic!("{t}: {e}"));
        let names: Vec<&str> = st.sliders.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, c.names.iter().map(String::as_str).collect::<Vec<_>>(), "{t}: names");
        for (k, &v) in &c.values {
            assert!(close(st.values[k], v, 1e-8), "{t}: value {k}");
        }
        for s in &st.sliders {
            let p = &c.params[&s.name];
            let ctx = format!("{t} {}", s.name);
            n_params += 1;
            assert!(close(s.nominal_range.0, p.range[0], 1e-9) && close(s.nominal_range.1, p.range[1], 1e-9), "{ctx}: range");
            assert!(close(s.step, p.range[2], 1e-15), "{ctx}: step");
            assert_eq!(s.allowed.current_ok, p.current_ok, "{ctx}: current_ok");
            let (ia, ib) = s.allowed.interval;
            assert!(close(ia, p.interval[0], TOL) && close(ib, p.interval[1], TOL), "{ctx}: interval {:?} vs {:?}", s.allowed.interval, p.interval);
            for (got, want) in [(s.allowed.edges.0, p.edges[0]), (s.allowed.edges.1, p.edges[1])] {
                assert_eq!(got.is_some(), want.is_some(), "{ctx}: edge presence");
                if let (Some(g), Some(w)) = (got, want) {
                    assert!(close(g, w, TOL), "{ctx}: edge {g} vs {w}");
                }
            }
            assert_eq!(s.allowed.segments.len(), p.segments.len(), "{ctx}: segments");
            for (g, w) in s.allowed.segments.iter().zip(&p.segments) {
                assert!(close(g.lo, w[0], TOL) && close(g.hi, w[1], TOL), "{ctx}: segment");
            }
            assert!(close(s.limits.0, p.limits[0], 1e-9) && close(s.limits.1, p.limits[1], 1e-9), "{ctx}: limits {:?} vs Python {:?}", s.limits, p.limits);
            check_specials(&ctx, &s.special, &p.special);
        }
        if let Some(m) = &c.moved {
            let v = st.set_value(&m.name, m.value).unwrap();
            assert!(close(v, m.value, 1e-12), "{t}: move value {v} vs {}", m.value);
            for s in &st.sliders {
                let want = m.limits[&s.name];
                let ctx = format!("{t} after {} = {}: {}", m.name, m.value, s.name);
                assert!(close(s.limits.0, want[0], 1e-9) && close(s.limits.1, want[1], 1e-9), "{ctx}: limits {:?} vs Python {want:?}", s.limits);
                check_specials(&ctx, &s.special, &m.special[&s.name]);
            }
        }
    }
    assert!(n_params > 400, "{n_params}");
}

/// Independent route (cell.rs, not the tuning scan): no two sites coincide, no crossing, no bond through a site, and
/// the D-symbol key of the family under its group is the reference key.
fn valid_by_cell_route(fam: &LatticeFamily, vals: &BTreeMap<String, f64>, key: &[(u32, u32, u32, u32, u32)]) -> bool {
    let f = fam.with_values(vals.iter().map(|(k, &v)| (k.as_str(), v))).unwrap();
    let cell = f.cell();
    if cell.min_separation() < 1e-6 || cell.has_crossings() || cell.clearance() < 1e-6 {
        return false;
    }
    f.dsymbol().and_then(|d| d.canonical_key().map_err(|e| e.to_string())).is_ok_and(|k| k == key)
}

#[test]
fn slider_drags_stay_inside_limits_and_out_of_forbidden_spans() {
    let fx = fixture();
    let mut n_checked = 0;
    for c in &fx.cases {
        let t = tag(c);
        let base = SliderState::new(c.recipe.clone()).unwrap();
        let fam = LatticeFamily::from_recipe(&c.recipe, None).unwrap();
        let key = fam.dsymbol().unwrap().canonical_key().unwrap();
        for s in &base.sliders {
            let p = &c.params[&s.name];
            let (lo, hi) = (p.range[0] - 1.0, p.range[1] + 1.0);
            let mut raws: Vec<f64> = (0..=60).map(|i| lo + (hi - lo) * f64::from(i) / 60.0).collect();
            raws.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, s.limits.0, s.limits.1]);
            raws.extend(p.segments.iter().map(|g| 0.5 * (g[0] + g[1])));
            raws.extend(p.special.iter().flat_map(|sp| [sp.0, sp.0 - 0.4 * s.step, sp.0 + 0.4 * s.step]));
            for &raw in &raws {
                let v = slider_value(raw, s.step, &s.special, s.limits);
                assert!(s.limits.0 <= v && v <= s.limits.1, "{t} {}: raw {raw} -> {v} outside {:?}", s.name, s.limits);
                // Python's forbidden segments (strict interior; the slider ends were tested valid)
                assert!(!p.segments.iter().any(|g| g[0] < v && v < g[1]), "{t} {}: raw {raw} -> {v} in a forbidden span", s.name);
            }
            // beyond the limits: clamped (test_widget_limits_bar_and_snap); within half a step of a special: snapped
            assert_eq!(slider_value(s.limits.1 + 0.3, s.step, &s.special, s.limits), s.limits.1);
            assert_eq!(slider_value(s.limits.0 - 0.3, s.step, &s.special, s.limits), s.limits.0);
            // a drag puts the value on the grid first, so the grid point nearest a special value snaps onto it
            // (an assignment off the grid snaps from within half a step: snap_to_special, as in the widget test)
            for sp in s.special.iter().filter(|sp| s.limits.0 <= sp.value && sp.value <= s.limits.1) {
                assert_eq!(slider_value(sp.value, s.step, &s.special, s.limits), sp.value, "{t}");
                let near = papertilling_core::plane::tuning::snap_to_special(sp.value - 0.4 * s.step, s.step, &s.special, s.limits);
                assert_eq!(near, sp.value, "{t}");
            }
            // the slider ends are lattices, checked by another route (cell.rs crossing / clearance / D-symbol)
            if c.rule != "named" && n_checked < 400 {
                for end in [s.limits.0, s.limits.1] {
                    let mut vals = base.values.clone();
                    vals.insert(s.name.clone(), end);
                    let on_special = s.special.iter().any(|sp| close(sp.value, end, 1e-9));
                    assert!(on_special || valid_by_cell_route(&fam, &vals, &key), "{t} {} end {end}", s.name);
                    n_checked += 1;
                }
            }
        }
    }
    assert!(n_checked > 200, "{n_checked}");
}

#[test]
fn coupled_moves_keep_every_other_value_inside_its_new_limits() {
    let fx = fixture();
    for c in fx.cases.iter().filter(|c| c.moved.is_some()) {
        let m = c.moved.as_ref().unwrap();
        let mut st = SliderState::new(c.recipe.clone()).unwrap();
        // drag across the whole nominal range and back: every stored value stays inside its own live limits
        let (lo, hi) = st.spec(&m.name).unwrap().nominal_range;
        for raw in [hi + 1.0, lo - 1.0, m.value, 0.5 * (lo + hi)] {
            let v = st.drag_to(&m.name, raw).unwrap();
            for s in &st.sliders {
                let x = st.values[&s.name];
                assert!(s.limits.0 <= x && x <= s.limits.1, "{}: {} = {x} outside {:?} after {} -> {v}", tag(c), s.name, s.limits, m.name);
                assert!(s.allowed.current_ok, "{}: {} not valid after {} -> {v}", tag(c), s.name, m.name);
            }
        }
        st.reset().unwrap();
        assert_eq!(st.values, SliderState::new(c.recipe.clone()).unwrap().values);
    }
}

#[test]
fn cell_parameters_are_never_forbidden_even_where_the_sites_are() {
    // tuning.allowed_interval (Python): for b and gamma, interval = (lo, hi), no segment, current_ok = True whatever
    // the site parameters are (proved: cell parameters never break planarity; tuning.py docstring).
    let fx = fixture();
    let mut n = 0;
    for c in fx.cases.iter().filter(|c| c.names.iter().any(|n| n == "gamma")) {
        let Some((site, p)) = c.params.iter().find(|(k, p)| k.as_str() != "b" && k.as_str() != "gamma" && !p.segments.is_empty()) else {
            continue;
        };
        let mut ov = c.values.clone();
        ov.insert(site.clone(), 0.5 * (p.segments[0][0] + p.segments[0][1])); // inside a Python forbidden segment
        let r_site = allowed_interval(&c.recipe, site, p.range[0], p.range[1], Some(&ov), None).unwrap();
        assert!(!r_site.current_ok, "{}", tag(c));
        for (name, lo, hi) in [("b", 0.4, 3.0), ("gamma", 45.0, 135.0)] {
            let r = allowed_interval(&c.recipe, name, lo, hi, Some(&ov), None).unwrap();
            let (lo, hi) = (lo.min(ov[name]), hi.max(ov[name]));
            assert!(r.current_ok && r.interval == (lo, hi) && r.segments.is_empty() && r.edges == (None, None), "{} {name}: {r:?}", tag(c));
        }
        n += 1;
    }
    assert!(n >= 5, "{n}");
}

#[test]
fn degenerate_and_non_finite_values_are_rejected_or_harmless() {
    let fx = fixture();
    let p1 = fx.cases.iter().find(|c| c.rule == "bonds" && c.group == "p1" && c.index == Some(0)).unwrap();
    let mut ov = BTreeMap::new();
    ov.insert("x1".to_string(), f64::NAN);
    assert!(SliderState::with_values(p1.recipe.clone(), Some(&ov)).is_err());
    let mut st = SliderState::new(p1.recipe.clone()).unwrap();
    assert!(st.set_value("x1", f64::NAN).is_err());
    assert!(st.set_value("no_such", 0.0).is_err());
    let before = st.values.clone();
    assert!(st.drag_to("x1", f64::NAN).is_err());
    assert_eq!(st.values, before);
    // flat cell (gamma = 0, b tiny): no panic, finite limits everywhere
    let mut deg = BTreeMap::new();
    deg.insert("gamma".to_string(), 0.0);
    deg.insert("b".to_string(), 1e-12);
    let st = SliderState::with_values(p1.recipe.clone(), Some(&deg)).unwrap();
    assert!(st.sliders.iter().all(|s| s.limits.0.is_finite() && s.limits.1.is_finite() && s.limits.0 <= s.limits.1));
}

#[test]
fn wyckoff_letters_and_symmetry_index_match_python_for_every_entry() {
    let fx = fixture();
    assert_eq!(fx.entries.len(), 533);
    let mut seen = 0;
    for (rule, text) in DATA {
        let cats = parse_catalogue_str(text).unwrap();
        for (group, cat) in &cats {
            for (k, e) in cat.all.iter().enumerate() {
                let row = fx.entries.iter().find(|r| r.0 == rule && &r.1 == group && r.2 == k).unwrap_or_else(|| panic!("{rule} {group} {k}"));
                let mut letters: Vec<char> = e.recipe.sites.iter().map(|&p| label_point(group, p).unwrap().letter).collect();
                letters.sort_unstable();
                letters.dedup();
                assert_eq!(letters.iter().collect::<String>(), row.3, "{rule} {group} #{k} Wyckoff");
                assert_eq!(e.symmetry_index, row.4, "{rule} {group} #{k} symmetry index");
                let names: Vec<String> = e.family.names();
                assert_eq!(names, row.5, "{rule} {group} #{k} parameter names");
                seen += 1;
            }
        }
    }
    assert_eq!(seen, 533);
}
