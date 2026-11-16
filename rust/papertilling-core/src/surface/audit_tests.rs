//! Audit of the surface engine against the pure-Python reference (`tests/fixtures/surface_table_py.json`,
//! written by `tests/fixtures/dump_surface_table.py` with `PAPERTILLING_NO_RUST=1`): every value compared here was
//! computed by the Python package, never copied from this crate.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::polyhedra::build_polyhedron;
use super::{
    RefusalKind, check_surface, cylinder, genus_surface, klein_bottle, mobius_band, surface_table, torus,
};
use crate::types::{Bond2D, Cell2D, SurfaceMesh};

const FIXTURE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/surface_table_py.json"));

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct CellJson {
    a1: [f64; 2],
    a2: [f64; 2],
    basis: Vec<[f64; 2]>,
    bonds: Vec<[i64; 5]>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(non_snake_case)]
pub(crate) struct Row {
    pub(crate) status: String,
    #[serde(default)]
    V: Option<usize>,
    #[serde(default)]
    E: Option<usize>,
    #[serde(default)]
    F: Option<usize>,
    #[serde(default)]
    chi: Option<i32>,
    #[serde(default)]
    orientable: Option<bool>,
    #[serde(default)]
    closed: Option<bool>,
    #[serde(default)]
    cone_angles: Option<Vec<f64>>,
    #[serde(default)]
    face_hist: Option<BTreeMap<String, usize>>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Deserialize)]
struct PolyRow {
    projective: bool,
    size: [i32; 2],
    centre: Option<String>,
    #[serde(flatten)]
    row: Row,
}

#[derive(Deserialize)]
struct FlatRow {
    kind: String,
    h: i32,
    k: i32,
    #[serde(flatten)]
    row: Row,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    group: String,
    cell: CellJson,
    table: BTreeMap<String, Row>,
    poly: Vec<PolyRow>,
    flat: Vec<FlatRow>,
}

fn to_cell(c: &CellJson) -> Cell2D {
    Cell2D {
        a1: c.a1,
        a2: c.a2,
        basis: c.basis.clone(),
        bonds: c
            .bonds
            .iter()
            .map(|b| Bond2D { i: b[0] as usize, j: b[1] as usize, n1: b[2] as i32, n2: b[3] as i32, kind: b[4] as usize })
            .collect(),
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(FIXTURE).expect("parse surface_table_py.json")
}

/// Compare a built mesh with a Python `OK` row; returns the list of mismatches (empty when equal).
pub(crate) fn mesh_mismatches(mesh: &SurfaceMesh, exp: &Row) -> Vec<String> {
    let mut bad = Vec::new();
    let mut cmp = |what: &str, got: String, want: String| {
        if got != want {
            bad.push(format!("{what}: rust {got} vs python {want}"));
        }
    };
    cmp("V", mesh.sites.len().to_string(), format!("{}", exp.V.unwrap_or(usize::MAX)));
    cmp("E", mesh.bonds.len().to_string(), format!("{}", exp.E.unwrap_or(usize::MAX)));
    cmp("F", mesh.faces.len().to_string(), format!("{}", exp.F.unwrap_or(usize::MAX)));
    cmp("chi", mesh.chi.to_string(), format!("{}", exp.chi.unwrap_or(i32::MIN)));
    cmp("orientable", mesh.orientable.to_string(), format!("{:?}", exp.orientable.unwrap_or(!mesh.orientable)));
    cmp("closed", mesh.closed.to_string(), format!("{:?}", exp.closed.unwrap_or(!mesh.closed)));
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for f in &mesh.faces {
        *hist.entry(f.len().to_string()).or_insert(0) += 1;
    }
    cmp("face_hist", format!("{hist:?}"), format!("{:?}", exp.face_hist.clone().unwrap_or_default()));
    let mut got: Vec<i64> = mesh.cones.iter().map(|c| (c.angle_deg * 1e4).round() as i64).collect();
    got.sort_unstable();
    let mut want: Vec<i64> = exp.cone_angles.iter().flatten().map(|a| (a * 1e4).round() as i64).collect();
    want.sort_unstable();
    cmp("cones", format!("{got:?}"), format!("{want:?}"));
    bad
}

fn status_of_string_err(msg: &str) -> &'static str {
    if msg.starts_with("IMPOSSIBLE") {
        "IMPOSSIBLE"
    } else if msg.starts_with("NOT FOUND") {
        "NOT FOUND"
    } else {
        "INVALID"
    }
}

fn status_of_built(res: &Result<SurfaceMesh, String>) -> String {
    match res {
        Ok(m) if check_surface(m).ok => "OK".into(),
        Ok(_) => "CHECKS FAILED".into(),
        Err(e) => status_of_string_err(e).into(),
    }
}

#[test]
fn surface_table_matches_python_on_17_first_lattices_and_named() {
    let fx = fixture();
    assert!(fx.cases.len() >= 17 + 4, "fixture lists {} cases", fx.cases.len());
    let mut failures = Vec::new();
    for case in &fx.cases {
        let cell = to_cell(&case.cell);
        let t0 = std::time::Instant::now();
        let rows = surface_table(&case.group, &cell);
        let secs = t0.elapsed().as_secs_f64();
        println!("# {:24} {:5} surface_table {secs:.2}s", case.label, case.group);
        for row in rows {
            let key = if row.surface == "Mobius band" { "Moebius band".to_string() } else { row.surface.clone() };
            let exp = &case.table[&key];
            if row.status != exp.status {
                failures.push(format!(
                    "{} {}: status rust {} ({}) vs python {} ({})",
                    case.label, key, row.status, row.reason, exp.status, exp.reason.clone().unwrap_or_default()
                ));
                continue;
            }
            if let Some(mesh) = &row.mesh {
                for m in mesh_mismatches(mesh, exp) {
                    failures.push(format!("{} {key}: {m}", case.label));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn sphere_and_rp2_sizes_and_centre_modes_match_python() {
    let fx = fixture();
    let mut failures = Vec::new();
    let mut n_rows = 0;
    for case in &fx.cases {
        let cell = to_cell(&case.cell);
        for pr in &case.poly {
            n_rows += 1;
            let size = Some((pr.size[0], pr.size[1]));
            let res = build_polyhedron(&case.group, &cell, size, pr.centre.as_deref(), pr.projective);
            let what = format!(
                "{} {} size {:?} centre {:?}",
                case.label,
                if pr.projective { "RP2" } else { "sphere" },
                pr.size,
                pr.centre
            );
            let got = match &res {
                Ok(s) if s.checks().ok => "OK",
                Ok(_) => "CHECKS FAILED",
                Err(e) => match e.kind {
                    RefusalKind::Impossible => "IMPOSSIBLE",
                    RefusalKind::NotFound { .. } => "NOT FOUND",
                    RefusalKind::InvalidArg => "INVALID",
                },
            };
            if got != pr.row.status {
                let why = res.as_ref().err().map(ToString::to_string).unwrap_or_default();
                failures.push(format!("{what}: rust {got} ({why}) vs python {}", pr.row.status));
                continue;
            }
            if let Ok(s) = &res {
                for m in mesh_mismatches(&s.mesh, &pr.row) {
                    failures.push(format!("{what}: {m}"));
                }
            }
        }
    }
    assert!(n_rows > 0);
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn flat_and_genus_at_tab_sizes_match_python() {
    let fx = fixture();
    let mut failures = Vec::new();
    for case in &fx.cases {
        let cell = to_cell(&case.cell);
        for fr in &case.flat {
            let (h, k) = (fr.h, fr.k);
            let res = match fr.kind.as_str() {
                "torus" => torus(&cell, [h, 0], [0, k.max(1)]),
                "cylinder" => cylinder(&cell, [0, k.max(3)], [h, 0]),
                "Klein bottle" => klein_bottle(&cell, 0, 2 * k.max(2) + 1, h.max(2)),
                "Moebius band" => mobius_band(&cell, 0, 2 * k.max(2) + 1, h),
                "genus 2" => genus_surface(&cell, 2),
                "genus 3" => genus_surface(&cell, 3),
                other => panic!("unknown flat kind {other}"),
            };
            let what = format!("{} {} (h={h}, k={k})", case.label, fr.kind);
            let got = status_of_built(&res);
            if got != fr.row.status {
                let why = res.as_ref().err().cloned().unwrap_or_default();
                failures.push(format!("{what}: rust {got} ({why}) vs python {}", fr.row.status));
                continue;
            }
            if let Ok(mesh) = &res
                && fr.row.V.is_some()
            {
                for m in mesh_mismatches(mesh, &fr.row) {
                    failures.push(format!("{what}: {m}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn negative_control_the_comparison_sees_a_missing_face_or_a_moved_cone() {
    let fx = fixture();
    let case = fx.cases.iter().find(|c| c.label == "honeycomb").unwrap();
    let cell = to_cell(&case.cell);
    let row = case.poly.iter().find(|p| !p.projective && p.size == [1, 1] && p.centre.is_none()).unwrap();
    let mut mesh = build_polyhedron(&case.group, &cell, Some((1, 1)), None, false).unwrap().mesh;
    assert!(mesh_mismatches(&mesh, &row.row).is_empty());
    assert_eq!((mesh.sites.len(), mesh.bonds.len(), mesh.faces.len()), (60, 90, 32), "C60");
    mesh.cones[0].angle_deg = 90.0;
    assert!(!mesh_mismatches(&mesh, &row.row).is_empty());
    mesh.faces.pop();
    assert!(mesh_mismatches(&mesh, &row.row).len() >= 2);
}
