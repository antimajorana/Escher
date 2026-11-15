//! Stage 5 PyO3 bindings: degree-labelled core isomorphism, irregular-site layer
//! signatures, nearest allowed rotation centre, core gallery, and single-defect equivalence.

use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

use papertilling_core::defect::combine_api;
use papertilling_core::defect::combine_types::{CoreGalleryEntry, DislocationScanInfo};
use papertilling_core::defect::core_graph::{
    LayerSignature, LatticeCoreInfo, core_graph_with_rim, isomorphic_cores as core_isomorphic_cores,
    layer_signature,
};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::types::{BuiltCrystal, Cell2D, DirectedBondData, Frame2D};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

static EMPTY_LAT: OnceLock<LatticeCoreInfo> = OnceLock::new();

fn parse_frame(frame_json: Option<&str>) -> Result<Frame2D, String> {
    match frame_json {
        Some(s) => serde_json::from_str(s).map_err(|e| e.to_string()),
        None => Ok(Frame2D::default()),
    }
}

pub(crate) fn pure_isomorphic_cores(
    n1: usize,
    deg1: &[u16],
    edges1: &[(usize, usize)],
    n2: usize,
    deg2: &[u16],
    edges2: &[(usize, usize)],
) -> Result<bool, String> {
    let validate = |n: usize, deg: &[u16], edges: &[(usize, usize)], tag: &str| -> Result<(), String> {
        if deg.len() != n {
            return Err(format!("{tag}: deg length {} != n ({n})", deg.len()));
        }
        if let Some(&(u, v)) = edges.iter().find(|&&(u, v)| u >= n || v >= n) {
            return Err(format!("{tag}: edge ({u}, {v}) out of bounds for n={n}"));
        }
        Ok(())
    };
    validate(n1, deg1, edges1, "graph 1")?;
    validate(n2, deg2, edges2, "graph 2")?;
    if n1 != n2 {
        return Ok(false);
    }
    if n1 == 0 {
        return Ok(true);
    }
    let lat_info = EMPTY_LAT.get_or_init(|| {
        LatticeCoreInfo::new(&Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: Vec::new(),
            bonds: Vec::new(),
        })
    });
    let empty_rim = HashSet::new();
    let make_graph = |n: usize, deg: &[u16], edges: &[(usize, usize)]| -> Result<_, String> {
        let adj: Vec<Vec<DirectedBondData>> = (0..n)
            .map(|u| {
                let stub = DirectedBondData {
                    to: u,
                    vec: [0.0, 0.0],
                    kind: 0,
                    seam: None,
                    ang: None,
                    across: false,
                    ghost_stub: None,
                };
                vec![stub; usize::from(deg[u])]
            })
            .collect();
        let mut norm_edges: Vec<(usize, usize)> = edges
            .iter()
            .filter_map(|&(u, v)| (u != v).then_some((u.min(v), u.max(v))))
            .collect();
        norm_edges.sort_unstable();
        norm_edges.dedup();
        let crystal = BuiltCrystal {
            pos: vec![[0.0, 0.0]; n],
            ref_pos: vec![[0.0, 0.0]; n],
            basis_idx: vec![0; n],
            side: vec![None; n],
            cone_sites: Vec::new(),
            edges: norm_edges,
            adj,
            wall_sigma: None,
            is_non_orientable: false,
        };
        core_graph_with_rim(&crystal, &empty_rim, lat_info, 0)
    };
    let g1 = make_graph(n1, deg1, edges1)?;
    let g2 = make_graph(n2, deg2, edges2)?;
    Ok(core_isomorphic_cores(&g1, &g2))
}

pub(crate) fn pure_layer_signatures_irregular(
    adj: &[Vec<usize>],
    deg: &[usize],
    cand: &[usize],
    r_det: usize,
    perfect_sigs: Vec<LayerSignature>,
) -> Result<Vec<usize>, String> {
    let n = adj.len();
    if deg.len() != n {
        return Err(format!("deg length {} != adj length {n}", deg.len()));
    }
    for (u, nbrs) in adj.iter().enumerate() {
        if let Some(&w) = nbrs.iter().find(|&&w| w >= n) {
            return Err(format!("adj[{u}] contains out-of-bounds vertex {w} (n={n})"));
        }
    }
    if let Some(&v) = cand.iter().find(|&&v| v >= n) {
        return Err(format!("cand contains out-of-bounds vertex {v} (n={n})"));
    }
    let pset: BTreeSet<LayerSignature> = perfect_sigs
        .into_iter()
        .map(|mut s| {
            s.sort_unstable();
            s
        })
        .collect();
    Ok(cand
        .iter()
        .copied()
        .filter(|&v| !pset.contains(&layer_signature(adj, deg, v, r_det)))
        .collect())
}

pub(crate) fn pure_equivalent_single(
    group: &str,
    cell_json: &str,
    specs_json: &str,
    radius: usize,
    grid: usize,
    frame_json: Option<&str>,
    draw_gallery: bool,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let specs: Vec<DefectSpec> = serde_json::from_str(specs_json).map_err(|e| e.to_string())?;
    let frame = parse_frame(frame_json)?;
    let crystal = build_point_defects(&cell, &specs, None, None)?;
    let rep = combine_api::equivalent_single(
        group,
        &cell,
        frame,
        &specs,
        &crystal,
        radius,
        grid,
        draw_gallery,
    )?;
    serde_json::to_string(&rep).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct CoreGalleryJson {
    gallery: Vec<CoreGalleryEntry>,
    scan: DislocationScanInfo,
}

pub(crate) fn pure_core_gallery(
    group: &str,
    cell_json: &str,
    b: [f64; 2],
    radius: usize,
    grid: usize,
    frame_json: Option<&str>,
    draw: bool,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let frame = parse_frame(frame_json)?;
    let (gallery, scan) = combine_api::core_gallery(group, &cell, frame, b, radius, grid, draw)?;
    serde_json::to_string(&CoreGalleryJson { gallery, scan }).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct NearestCentreJson {
    point: [f64; 2],
    label: String,
}

pub(crate) fn pure_nearest_allowed_centre(
    group: &str,
    cell_json: &str,
    angle_deg: f64,
    near: [f64; 2],
    frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let frame = parse_frame(frame_json)?;
    let (point, label) = combine_api::nearest_allowed_centre(group, &cell, frame, angle_deg, near)?;
    serde_json::to_string(&NearestCentreJson { point, label }).map_err(|e| e.to_string())
}

/// Exact VF2 degree-labelled core-graph isomorphism check.
#[pyfunction]
pub fn isomorphic_cores(
    n1: usize,
    deg1: Vec<u16>,
    edges1: Vec<(usize, usize)>,
    n2: usize,
    deg2: Vec<u16>,
    edges2: Vec<(usize, usize)>,
) -> PyResult<bool> {
    pure_isomorphic_cores(n1, &deg1, &edges1, n2, &deg2, &edges2).map_err(PyValueError::new_err)
}

/// Return candidate vertices whose layered BFS ball signature of radius `r_det` is not in `perfect_sigs`.
#[pyfunction]
pub fn layer_signatures_irregular(
    adj: Vec<Vec<usize>>,
    deg: Vec<usize>,
    cand: Vec<usize>,
    r_det: usize,
    perfect_sigs: Vec<LayerSignature>,
) -> PyResult<Vec<usize>> {
    pure_layer_signatures_irregular(&adj, &deg, &cand, r_det, perfect_sigs)
        .map_err(PyValueError::new_err)
}

/// Build point defects and compare against single-defect cores (`combine_api::equivalent_single`).
#[pyfunction]
#[pyo3(signature = (group, cell_json, specs_json, radius=2, grid=3, frame_json=None, draw_gallery=false))]
pub fn equivalent_single_json(
    group: &str,
    cell_json: &str,
    specs_json: &str,
    radius: usize,
    grid: usize,
    frame_json: Option<&str>,
    draw_gallery: bool,
) -> PyResult<String> {
    pure_equivalent_single(group, cell_json, specs_json, radius, grid, frame_json, draw_gallery)
        .map_err(PyValueError::new_err)
}

/// Scan Volterra and non-crossing completion cores for Burgers vector `b` (`combine_api::core_gallery`).
#[pyfunction]
#[pyo3(signature = (group, cell_json, b, radius=2, grid=3, frame_json=None, draw=false))]
pub fn core_gallery_json(
    group: &str,
    cell_json: &str,
    b: [f64; 2],
    radius: usize,
    grid: usize,
    frame_json: Option<&str>,
    draw: bool,
) -> PyResult<String> {
    pure_core_gallery(group, cell_json, b, radius, grid, frame_json, draw)
        .map_err(PyValueError::new_err)
}

/// Snap `near` to the nearest allowed Wyckoff rotation centre permitting `angle_deg`.
#[pyfunction]
#[pyo3(signature = (group, cell_json, angle_deg, near, frame_json=None))]
pub fn nearest_allowed_centre_json(
    group: &str,
    cell_json: &str,
    angle_deg: f64,
    near: [f64; 2],
    frame_json: Option<&str>,
) -> PyResult<String> {
    pure_nearest_allowed_centre(group, cell_json, angle_deg, near, frame_json)
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use papertilling_core::defect::core_graph::layer_signature;

    use super::{
        pure_core_gallery, pure_equivalent_single, pure_isomorphic_cores,
        pure_layer_signatures_irregular, pure_nearest_allowed_centre,
    };

    #[test]
    fn isomorphic_cores_positive_negative_and_bounds() {
        assert!(pure_isomorphic_cores(0, &[], &[], 0, &[], &[]).unwrap());
        assert!(!pure_isomorphic_cores(0, &[], &[], 1, &[4], &[]).unwrap());

        // Relabelled 4-cycles with adjacent [3, 3] and adjacent [4, 4] degrees -> isomorphic.
        let e_cyc = [(0, 1), (1, 2), (2, 3), (3, 0)];
        assert!(pure_isomorphic_cores(4, &[3, 3, 4, 4], &e_cyc, 4, &[4, 3, 3, 4], &e_cyc).unwrap());
        // Opposite [3, 4, 3, 4] on the same 4-cycle -> NOT isomorphic.
        assert!(!pure_isomorphic_cores(4, &[3, 3, 4, 4], &e_cyc, 4, &[3, 4, 3, 4], &e_cyc).unwrap());

        // Bounds checks.
        assert!(pure_isomorphic_cores(2, &[3], &[], 2, &[3, 3], &[]).is_err());
        assert!(pure_isomorphic_cores(2, &[3, 3], &[(0, 2)], 2, &[3, 3], &[]).is_err());
    }

    #[test]
    fn layer_signatures_irregular_detects_defect_vertices() {
        // 3x3 periodic torus of 4-regular square sites (9 vertices): every vertex has the same r=1 signature.
        let mut adj = vec![Vec::new(); 9];
        for r in 0..3 {
            for c in 0..3 {
                let u = 3 * r + c;
                adj[u] = vec![
                    3 * r + (c + 1) % 3,
                    3 * r + (c + 2) % 3,
                    3 * ((r + 1) % 3) + c,
                    3 * ((r + 2) % 3) + c,
                ];
            }
        }
        let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
        let perf = vec![layer_signature(&adj, &deg, 0, 1)];
        let cand: Vec<usize> = (0..9).collect();
        assert_eq!(pure_layer_signatures_irregular(&adj, &deg, &cand, 1, perf.clone()).unwrap(), Vec::<usize>::new());

        // Break bond (0, 1): vertices 0 and 1 drop to degree 3, and their distance-1 neighbours see them.
        adj[0].retain(|&w| w != 1);
        adj[1].retain(|&w| w != 0);
        let deg2: Vec<usize> = adj.iter().map(Vec::len).collect();
        let irr = pure_layer_signatures_irregular(&adj, &deg2, &cand, 1, perf).unwrap();
        assert!(irr.contains(&0) && irr.contains(&1));
        assert!(!irr.contains(&8)); // vertex (2,2) is at graph distance 2 from both 0 and 1
    }

    #[test]
    fn stage5_json_wrappers_validate_and_serialize() {
        let sq_cell = r#"{"a1":[1.0,0.0],"a2":[0.0,1.0],"basis":[[0.0,0.0]],"bonds":[{"i":0,"j":0,"n1":1,"n2":0,"kind":0},{"i":0,"j":0,"n1":0,"n2":1,"kind":0}]}"#;
        let pair_specs = r#"[{"kind":"disclination","angle":90.0,"at":[0.0,0.0]},{"kind":"disclination","angle":-90.0,"at":[0.5,0.5]}]"#;

        assert!(pure_equivalent_single("p4m", "bad_json", pair_specs, 2, 1, None, false).is_err());
        assert!(pure_core_gallery("p4m", "bad_json", [1.0, 0.0], 2, 1, None, false).is_err());
        assert!(pure_nearest_allowed_centre("p4m", "bad_json", 90.0, [0.1, 0.1], None).is_err());

        let eq_s = pure_equivalent_single("p4m", sq_cell, pair_specs, 2, 1, None, false).unwrap();
        let eq_v: serde_json::Value = serde_json::from_str(&eq_s).unwrap();
        assert_eq!(eq_v["equivalent"], serde_json::Value::Bool(true));
        assert_eq!(eq_v["matches"], serde_json::json!(["Volterra core 1"]));

        let gal_s = pure_core_gallery("p4m", sq_cell, [1.0, 0.0], 2, 1, None, false).unwrap();
        let gal_v: serde_json::Value = serde_json::from_str(&gal_s).unwrap();
        assert_eq!(gal_v["gallery"].as_array().unwrap().len(), 2);

        let nc_s = pure_nearest_allowed_centre("p4m", sq_cell, 90.0, [0.48, 0.52], None).unwrap();
        let nc_v: serde_json::Value = serde_json::from_str(&nc_s).unwrap();
        assert_eq!(nc_v["label"], serde_json::json!("1b"));
        assert_eq!(nc_v["point"], serde_json::json!([0.5, 0.5]));
    }
}
