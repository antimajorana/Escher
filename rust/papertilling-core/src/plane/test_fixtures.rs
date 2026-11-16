//! Test-only spread of crystals (point defects, multi-defect configurations, walls, a Moebius seam and a
//! perfect disk on several named lattices) for the old-versus-new parity tests of the drawing layer.

use crate::defect::catalogue::DefectCatalogue;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::plane::catalogue::{NamedLattice, parse_named_str};
use crate::types::BuiltCrystal;

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

pub(crate) fn named(key: &str) -> NamedLattice {
    parse_named_str(NAMED_JSON).expect("named.json").get(key).expect("named lattice").clone()
}

fn top_pair(lat: &NamedLattice, order: usize, sep: f64) -> Vec<DefectSpec> {
    let cat = DefectCatalogue::new(&lat.group, &lat.cell, lat.frame);
    let p = cat.centre_classes.iter().find(|c| c.order == order).expect("rotation centre").point;
    let alpha = 360.0 / (order as f64);
    let q = [p[0] + sep * lat.cell.a1[0], p[1] + sep * lat.cell.a1[1]];
    vec![DefectSpec::disclination(alpha, p), DefectSpec::disclination(-alpha, q)]
}

/// `(name, crystal)` pairs: raw surgery output (unrelaxed) plus wall and seam crystals.
pub(crate) fn crystals() -> Vec<(String, BuiltCrystal)> {
    let sq = named("square");
    let hc = named("honeycomb");
    let kg = named("kagome");
    let cs = named("chiral_square");
    let zz = named("zigzag");
    let mut cases: Vec<(&str, &NamedLattice, Vec<DefectSpec>, Option<f64>)> = vec![
        ("square_pair", &sq, vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])], None),
        ("square_pair_far", &sq, vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [4.5, 0.5])], Some(9.0)),
        ("square_disloc", &sq, vec![DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))], Some(8.0)),
        ("square_m90_face", &sq, vec![DefectSpec::disclination(-90.0, [0.5, 0.5])], None),
        ("square_p90_site", &sq, vec![DefectSpec::disclination(90.0, [0.0, 0.0])], None),
        (
            "square_m180_x3",
            &sq,
            vec![
                DefectSpec::disclination(-180.0, [0.0, 0.5]),
                DefectSpec::disclination(-180.0, [3.0, 0.5]),
                DefectSpec::disclination(-180.0, [0.0, 3.5]),
            ],
            None,
        ),
        ("honeycomb_pair", &hc, top_pair(&hc, 6, 1.0), None),
        ("honeycomb_p180_m180", &hc, top_pair(&hc, 2, 1.0), None),
        ("kagome_pair", &kg, top_pair(&kg, 6, 1.0), Some(8.0)),
        ("chiral_square_p90_p90", &cs, vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(90.0, [1.0, 0.0])], None),
        (
            "zigzag_p180_disloc",
            &zz,
            vec![DefectSpec::disclination(180.0, [0.0, 0.5]), DefectSpec::dislocation([2.0, 0.0], Some([-0.826, 3.64]))],
            None,
        ),
    ];
    let hc_disloc = {
        let cat = DefectCatalogue::new(&hc.group, &hc.cell, hc.frame);
        let d = cat.dislocations.first().expect("dislocation class");
        vec![DefectSpec::dislocation(d.b, Some(d.at))]
    };
    cases.push(("honeycomb_disloc", &hc, hc_disloc, Some(8.0)));
    let mut out = Vec::new();
    for (name, lat, specs, r) in cases {
        if let Ok(c) = build_point_defects(&lat.cell, &specs, r, None) {
            out.push((name.to_string(), c));
        }
    }
    let sq_cat = DefectCatalogue::new(&sq.group, &sq.cell, sq.frame);
    for id in ["Wd1", "Wb1", "Wc1"] {
        if let Some(c) = sq_cat.build_wall(id, Some(5.0)) {
            out.push((format!("square_wall_{id}"), c));
        }
    }
    let hc_cat = DefectCatalogue::new(&hc.group, &hc.cell, hc.frame);
    if let Some(c) = hc_cat.build_wall("Wb1", Some(5.0)) {
        out.push(("honeycomb_wall_Wb1".to_string(), c));
    }
    out
}

#[test]
fn fixture_spread_builds() {
    let all = crystals();
    let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
    assert!(all.len() >= 14, "only {} fixtures built: {names:?}", all.len());
    assert!(all.iter().any(|(_, c)| c.wall_sigma.is_some()), "{names:?}");
    assert!(all.iter().any(|(_, c)| c.is_non_orientable), "{names:?}");
}
