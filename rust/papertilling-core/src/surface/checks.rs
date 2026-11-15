//! Combinatorial and topological invariant checks for `SurfaceMesh` (`Surface.checks()`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::types::{SurfaceCheckReport, SurfaceMesh};

/// Check all five topological and local-coordination invariants of `Surface.checks()`.
pub fn check_surface(mesh: &SurfaceMesh) -> SurfaceCheckReport {
    let v_cnt = mesh.sites.len();
    let e_cnt = mesh.bonds.len();
    let f_cnt = mesh.faces.len();

    let no_loops = mesh
        .bonds
        .iter()
        .all(|&(u, v, _)| u != v && u < v_cnt && v < v_cnt);

    let mut pairs = HashSet::with_capacity(e_cnt);
    let no_double_bonds = mesh
        .bonds
        .iter()
        .all(|&(u, v, _)| pairs.insert(if u <= v { (u, v) } else { (v, u) }));

    let mut deg = vec![0usize; v_cnt];
    for &(u, v, _) in &mesh.bonds {
        if u < v_cnt {
            deg[u] += 1;
        }
        if v < v_cnt {
            deg[v] += 1;
        }
    }

    let edge_set: HashSet<usize> = mesh.edge_sites.iter().copied().collect();
    let mut site_cones: HashMap<usize, f64> = HashMap::new();
    for c in &mesh.cones {
        if let Some(u) = c.site {
            *site_cones.entry(u).or_insert(0.0) += c.angle_deg;
        }
    }

    let plane_coordination = (0..v_cnt).all(|u| {
        if edge_set.contains(&u) {
            return true;
        }
        let pdeg = if mesh.plane_degree.len() == v_cnt {
            Some(mesh.plane_degree[u])
        } else {
            mesh.plane_degree.get(mesh.sites[u].0).copied()
        };
        let Some(pdeg) = pdeg else {
            return false;
        };
        if let Some(&frank) = site_cones.get(&u) {
            let expected = (pdeg as f64) * (360.0 - frank) / 360.0;
            expected >= 1.0 - 1e-6
                && (expected - expected.round()).abs() < 1e-4
                && deg[u] == expected.round() as usize
        } else {
            deg[u] == pdeg
        }
    });

    let euler = (v_cnt as i32) - (e_cnt as i32) + (f_cnt as i32) == mesh.chi;
    let faces_at_least_3 = mesh
        .faces
        .iter()
        .all(|f| f.len() >= 3 && f.iter().all(|&u| u < v_cnt));

    let ok = no_loops && no_double_bonds && plane_coordination && euler && faces_at_least_3;
    SurfaceCheckReport {
        ok,
        no_loops,
        no_double_bonds,
        plane_coordination,
        euler,
        faces_at_least_3,
    }
}

/// Histogram mapping face polygon side count `k` to the number of `k`-gon faces on `mesh`.
pub fn face_histogram(mesh: &SurfaceMesh) -> BTreeMap<usize, usize> {
    let mut hist = BTreeMap::new();
    for f in &mesh.faces {
        *hist.entry(f.len()).or_insert(0) += 1;
    }
    hist
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use super::{check_surface, face_histogram};
    use crate::surface::flat::{
        default_cylinder, default_klein_bottle, default_mobius_band, default_torus, klein_bottle,
        torus,
    };
    use crate::surface::genus::{default_genus_surface, genus_surface};
    use crate::types::{Bond2D, Cell2D, ConeMark, SurfaceMesh};

    fn b(i: usize, j: usize, n1: i32, n2: i32) -> Bond2D {
        Bond2D { i, j, n1, n2, kind: 0 }
    }

    fn named_cells() -> (Cell2D, Cell2D, Cell2D, Cell2D) {
        let s3 = 3.0_f64.sqrt();
        let sq = Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![b(0, 0, -1, 0), b(0, 0, 0, -1)],
        };
        let hc = Cell2D {
            a1: [s3, 0.0],
            a2: [-0.5 * s3, 1.5],
            basis: vec![[0.5 * s3, 1.5], [s3, 1.0]],
            bonds: vec![b(0, 1, -1, 0), b(0, 1, 0, 0), b(0, 1, 0, 1)],
        };
        let kg = Cell2D {
            a1: [2.0, 0.0],
            a2: [-1.0, s3],
            basis: vec![[-1.0, s3], [0.0, s3], [0.5, 0.5 * s3]],
            bonds: vec![
                b(0, 1, -1, 0), b(0, 1, 0, 0), b(0, 2, -1, 0),
                b(0, 2, 0, 1), b(1, 2, 0, 0), b(1, 2, 0, 1),
            ],
        };
        let tr = Cell2D {
            a1: [1.0, 0.0],
            a2: [-0.5, 0.5 * s3],
            basis: vec![[0.0, 0.0]],
            bonds: vec![b(0, 0, -1, -1), b(0, 0, -1, 0), b(0, 0, 0, -1)],
        };
        (sq, hc, kg, tr)
    }

    #[test]
    fn check_surface_site_cone_cube_and_negative_controls() {
        let cube = SurfaceMesh {
            name: "cube".to_string(),
            sites: (0..8).map(|k| (0, k, 0, 0)).collect(),
            bonds: vec![
                (0, 1, [1.0, 0.0]), (1, 2, [0.0, 1.0]), (2, 3, [-1.0, 0.0]), (3, 0, [0.0, -1.0]),
                (4, 5, [1.0, 0.0]), (5, 6, [0.0, 1.0]), (6, 7, [-1.0, 0.0]), (7, 4, [0.0, -1.0]),
                (0, 4, [0.0, 1.0]), (1, 5, [0.0, 1.0]), (2, 6, [0.0, 1.0]), (3, 7, [0.0, 1.0]),
            ],
            faces: vec![
                vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![0, 1, 5, 4],
                vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7],
            ],
            plane_degree: vec![4; 8],
            edge_sites: vec![],
            chi: 2,
            orientable: true,
            closed: true,
            cones: (0..8)
                .map(|u| ConeMark {
                    angle_deg: 90.0,
                    site: Some(u),
                    face: None,
                    class_label: None,
                    location: "site".to_string(),
                })
                .collect(),
            uv: vec![[0.0, 0.0]; 8],
            domain_polygons: vec![],
            domain_edges: vec![],
            slits: vec![],
            xyz: vec![[0.0, 0.0, 0.0]; 8],
            bond_curves_3d: vec![],
            cover: None,
        };
        assert!(check_surface(&cube).ok);
        assert_eq!(face_histogram(&cube), BTreeMap::from([(4, 6)]));

        let (sq, _, _, _) = named_cells();
        let bad_kb = klein_bottle(&sq, 0, 1, 1).unwrap();
        assert!(!check_surface(&bad_kb).ok);
    }

    #[derive(Deserialize)]
    struct GateFile {
        known: BTreeMap<String, KnownExpect>,
        inputs_17: BTreeMap<String, InputEntry>,
        table_17x8: BTreeMap<String, BTreeMap<String, TableExpect>>,
    }

    #[derive(Deserialize)]
    struct RawCell {
        a1: [f64; 2],
        a2: [f64; 2],
        basis: Vec<[f64; 2]>,
        bonds: Vec<[i64; 5]>,
    }

    #[derive(Deserialize)]
    struct InputEntry {
        cell: RawCell,
    }

    #[derive(Deserialize)]
    struct KnownExpect {
        #[serde(rename = "V")]
        v: usize,
        #[serde(rename = "E")]
        e: usize,
        #[serde(rename = "F")]
        f: usize,
        chi: i32,
        orientable: bool,
        closed: bool,
        face_hist: BTreeMap<String, usize>,
        cone_count: usize,
        cone_angles: Vec<f64>,
        checks_ok: bool,
    }

    #[derive(Deserialize)]
    struct TableExpect {
        status: String,
        #[serde(rename = "V")]
        v: Option<usize>,
        #[serde(rename = "E")]
        e: Option<usize>,
        #[serde(rename = "F")]
        f: Option<usize>,
        chi: Option<i32>,
        orientable: Option<bool>,
        closed: Option<bool>,
        face_hist: Option<BTreeMap<String, usize>>,
        cone_angles: Option<Vec<f64>>,
        checks_ok: Option<bool>,
    }

    #[derive(Deserialize)]
    struct NamedFile {
        named: Vec<NamedEntry>,
    }

    #[derive(Deserialize)]
    struct NamedEntry {
        key: String,
        cell: RawCell,
    }

    fn raw_to_cell(rc: &RawCell) -> Cell2D {
        Cell2D {
            a1: rc.a1,
            a2: rc.a2,
            basis: rc.basis.clone(),
            bonds: rc
                .bonds
                .iter()
                .map(|x| b(x[0] as usize, x[1] as usize, x[2] as i32, x[3] as i32))
                .collect(),
        }
    }

    fn assert_mesh_matches(label: &str, mesh: &SurfaceMesh, exp: &KnownExpect, check_3d: bool) {
        assert_eq!(check_surface(mesh).ok, exp.checks_ok, "{label}: checks_ok");
        assert_eq!(mesh.sites.len(), exp.v, "{label}: V");
        assert_eq!(mesh.bonds.len(), exp.e, "{label}: E");
        assert_eq!(mesh.faces.len(), exp.f, "{label}: F");
        assert_eq!(mesh.chi, exp.chi, "{label}: chi");
        assert_eq!(mesh.orientable, exp.orientable, "{label}: orientable");
        assert_eq!(mesh.closed, exp.closed, "{label}: closed");
        let hist: BTreeMap<String, usize> = face_histogram(mesh)
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        assert_eq!(hist, exp.face_hist, "{label}: face_hist");
        assert_eq!(mesh.cones.len(), exp.cone_count, "{label}: cone_count");
        let mut got: Vec<f64> = mesh.cones.iter().map(|c| c.angle_deg).collect();
        got.sort_by(|a, b| a.total_cmp(b));
        let mut want = exp.cone_angles.clone();
        want.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(got.len(), want.len(), "{label}: cone_angles len");
        for (ga, ea) in got.iter().zip(&want) {
            assert!((ga - ea).abs() < 1e-6, "{label}: cone angle {ga} vs {ea}");
        }
        if check_3d {
            assert_eq!(mesh.xyz.len(), exp.v, "{label}: xyz len");
            assert_eq!(mesh.bond_curves_3d.len(), exp.e, "{label}: bond_curves_3d len");
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for &p in &mesh.xyz {
                for ax in 0..3 {
                    assert!(p[ax].is_finite(), "{label}: non-finite xyz");
                    lo[ax] = lo[ax].min(p[ax]);
                    hi[ax] = hi[ax].max(p[ax]);
                }
            }
            let (mut clo, mut chi_b) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for (idx, curve) in mesh.bond_curves_3d.iter().enumerate() {
                let (a, b_idx, _) = mesh.bonds[idx];
                assert!(curve.len() >= 2, "{label}: curve {idx} len < 2");
                if mesh.chi < 0 && exp.e <= 1500 {
                    assert_eq!(curve.len(), 7, "{label}: smooth handlebody curve {idx} len");
                }
                assert_eq!(curve[0], mesh.xyz[a], "{label}: curve {idx} start");
                assert_eq!(*curve.last().unwrap(), mesh.xyz[b_idx], "{label}: curve {idx} end");
                for &pt in curve {
                    for ax in 0..3 {
                        assert!(pt[ax].is_finite(), "{label}: non-finite bond_curves_3d");
                        clo[ax] = clo[ax].min(pt[ax]);
                        chi_b[ax] = chi_b[ax].max(pt[ax]);
                    }
                }
            }
            for ax in 0..3 {
                assert!(hi[ax] - lo[ax] > 0.1, "{label}: xyz 3D extent axis {ax} <= 0.1");
                assert!(chi_b[ax] - clo[ax] > 0.1, "{label}: bond_curves_3d 3D extent axis {ax} <= 0.1");
            }
        }
    }

    fn run_known_suite(tag: &str, cells: &(Cell2D, Cell2D, Cell2D, Cell2D), gate: &GateFile) {
        let (sq, hc, kg, tr) = cells;
        let known_cases: [(&str, SurfaceMesh); 10] = [
            ("square_torus_4_3", torus(sq, [4, 0], [0, 3]).unwrap()),
            ("honeycomb_torus_4_3", torus(hc, [4, 0], [0, 3]).unwrap()),
            ("kagome_torus_4_3", torus(kg, [4, 0], [0, 3]).unwrap()),
            ("triangular_torus_skew", torus(tr, [3, 1], [-1, 2]).unwrap()),
            ("square_genus_2", genus_surface(sq, 2).unwrap()),
            ("honeycomb_genus_2", genus_surface(hc, 2).unwrap()),
            ("kagome_genus_2", genus_surface(kg, 2).unwrap()),
            ("square_genus_3", genus_surface(sq, 3).unwrap()),
            ("honeycomb_genus_3", genus_surface(hc, 3).unwrap()),
            ("kagome_genus_3", genus_surface(kg, 3).unwrap()),
        ];
        let mut expected_s3a_keys: Vec<&str> = gate
            .known
            .keys()
            .map(String::as_str)
            .filter(|k| k.contains("torus") || k.contains("genus"))
            .collect();
        expected_s3a_keys.sort_unstable();
        let mut got_keys: Vec<&str> = known_cases.iter().map(|(k, _)| *k).collect();
        got_keys.sort_unstable();
        assert_eq!(got_keys, expected_s3a_keys, "must cover 100% of torus & genus known keys");
        for (key, mesh) in &known_cases {
            let exp = gate.known.get(*key).unwrap_or_else(|| panic!("missing {key}"));
            assert_mesh_matches(&format!("{tag}:{key}"), mesh, exp, true);
        }
    }

    #[test]
    fn gate_stage3_s3a_known_and_table_17x6() {
        let Ok(gate_path) = std::env::var("PT_GATE_STAGE3") else {
            return;
        };
        let t0 = std::time::Instant::now();
        let raw = std::fs::read_to_string(&gate_path).expect("failed to read PT_GATE_STAGE3");
        let gate: GateFile = serde_json::from_str(&raw).expect("failed to parse PT_GATE_STAGE3");

        run_known_suite("analytic", &named_cells(), &gate);
        if let Ok(named_path) = std::env::var("PT_NAMED_JSON") {
            let nraw = std::fs::read_to_string(&named_path).expect("failed to read PT_NAMED_JSON");
            let nf: NamedFile = serde_json::from_str(&nraw).expect("failed to parse PT_NAMED_JSON");
            let by_key: BTreeMap<String, Cell2D> =
                nf.named.iter().map(|e| (e.key.clone(), raw_to_cell(&e.cell))).collect();
            let cells = (
                by_key["square"].clone(),
                by_key["honeycomb"].clone(),
                by_key["kagome"].clone(),
                by_key["triangular"].clone(),
            );
            run_known_suite("named_json", &cells, &gate);
        }

        assert_eq!(gate.inputs_17.len(), 17, "expected all 17 wallpaper groups");
        assert_eq!(gate.table_17x8.len(), 17, "expected all 17 table_17x8 rows");
        let cols = ["torus", "cylinder", "Moebius band", "Klein bottle", "genus 2", "genus 3"];
        let (mut ok_cnt, mut imp_cnt) = (0usize, 0usize);
        for (grp, inp) in &gate.inputs_17 {
            let cell = raw_to_cell(&inp.cell);
            for col in cols {
                let exp = &gate.table_17x8[grp][col];
                let res = match col {
                    "torus" => default_torus(&cell),
                    "cylinder" => default_cylinder(&cell),
                    "Moebius band" => default_mobius_band(&cell),
                    "Klein bottle" => default_klein_bottle(&cell),
                    "genus 2" => default_genus_surface(&cell, 2),
                    "genus 3" => default_genus_surface(&cell, 3),
                    _ => unreachable!(),
                };
                match exp.status.as_str() {
                    "OK" => {
                        ok_cnt += 1;
                        let mesh = res.unwrap_or_else(|e| panic!("{grp} {col} failed: {e}"));
                        let cone_angles = exp.cone_angles.clone().unwrap();
                        let ke = KnownExpect {
                            v: exp.v.unwrap(), e: exp.e.unwrap(), f: exp.f.unwrap(),
                            chi: exp.chi.unwrap(), orientable: exp.orientable.unwrap(),
                            closed: exp.closed.unwrap(), face_hist: exp.face_hist.clone().unwrap(),
                            cone_count: cone_angles.len(), cone_angles,
                            checks_ok: exp.checks_ok.unwrap(),
                        };
                        assert_mesh_matches(&format!("{grp}/{col}"), &mesh, &ke, true);
                    }
                    "IMPOSSIBLE" | "NOT FOUND" => {
                        imp_cnt += 1;
                        let err = res.expect_err(&format!("{grp} {col} expected {}", exp.status));
                        assert!(err.starts_with(&format!("{}:", exp.status)), "{grp} {col} got {err}");
                    }
                    other => panic!("unexpected status {other}"),
                }
            }
        }
        assert_eq!(ok_cnt + imp_cnt, 102);
        println!(
            "# time gate_stage3_s3a: {:.3}s (known=10/10, table_17x6={}/102 OK + {} IMPOSSIBLE/NOT_FOUND)",
            t0.elapsed().as_secs_f64(),
            ok_cnt,
            imp_cnt
        );
    }
}
