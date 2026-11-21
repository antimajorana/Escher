//! Stage 2 PyO3 bindings: solver exports (extended XYZ, neighbour shells/orbits,
//! site-permutation groups) and defect builders (CSL grain boundaries, Stone-Wales, point defects).

use papertilling_core::defect::{
    PointDefectOp, apply_point_defects, list_csl_grain_boundaries, stone_wales_flip,
};
use papertilling_core::export::{
    classify_cell_shells, cluster_site_permutations, export_cell_extxyz, export_crystal_extxyz,
    periodic_supercell_permutations,
};
use papertilling_core::types::{BuiltCrystal, Cell2D};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

fn parse_opt_json<T: serde::de::DeserializeOwned>(raw: Option<&str>) -> Result<Option<T>, String> {
    let Some(s) = raw else { return Ok(None) };
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "null" {
        return Ok(None);
    }
    serde_json::from_str(trimmed).map(Some).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PosArg {
    P3([f64; 3]),
    P2([f64; 2]),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EdgeArg {
    Pair([usize; 2]),
    Weighted(usize, usize, u32),
}

pub(crate) fn pure_export_cell_extxyz(
    cell_json: &str,
    nx: usize,
    ny: usize,
    species_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let species: Option<Vec<String>> = parse_opt_json(species_json)?;
    export_cell_extxyz(&cell, nx, ny, species.as_deref())
}

pub(crate) fn pure_export_crystal_extxyz(
    crystal_json: &str,
    species_json: Option<&str>,
    z_json: Option<&str>,
) -> Result<String, String> {
    let crystal: BuiltCrystal = serde_json::from_str(crystal_json).map_err(|e| e.to_string())?;
    let species: Option<Vec<String>> = parse_opt_json(species_json)?;
    let z: Option<Vec<f64>> = parse_opt_json(z_json)?;
    export_crystal_extxyz(&crystal, z.as_deref(), species.as_deref())
}

pub(crate) fn pure_classify_cell_shells(
    group: &str,
    cell_json: &str,
    max_shells: usize,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let res = classify_cell_shells(group, &cell, max_shells)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

pub(crate) fn pure_cluster_site_permutations(
    positions_json: &str,
    labels_json: &str,
    shells_json: &str,
) -> Result<String, String> {
    let raw_pos: Vec<PosArg> = serde_json::from_str(positions_json).map_err(|e| e.to_string())?;
    let positions: Vec<[f64; 3]> = raw_pos
        .into_iter()
        .map(|p| match p {
            PosArg::P3(v) => v,
            PosArg::P2([x, y]) => [x, y, 0.0],
        })
        .collect();
    let labels: Vec<usize> = serde_json::from_str(labels_json).map_err(|e| e.to_string())?;
    let raw_shells: Vec<Vec<EdgeArg>> = serde_json::from_str(shells_json).map_err(|e| e.to_string())?;
    let shells: Vec<Vec<(usize, usize)>> = raw_shells
        .into_iter()
        .map(|sh| {
            sh.into_iter()
                .filter_map(|e| match e {
                    EdgeArg::Pair([a, b]) => Some((a, b)),
                    EdgeArg::Weighted(a, b, w) => (w > 0).then_some((a, b)),
                })
                .collect()
        })
        .collect();
    let rep = cluster_site_permutations(&positions, &labels, &shells)?;
    serde_json::to_string(&rep).map_err(|e| e.to_string())
}

pub(crate) fn pure_periodic_supercell_solver(
    group: &str,
    cell_json: &str,
    nx: usize,
    ny: usize,
    max_shells: usize,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let res = periodic_supercell_permutations(group, &cell, nx, ny, max_shells)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

pub(crate) fn pure_list_csl_grain_boundaries(
    group: &str,
    cell_json: &str,
    max_sigma: u32,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let res = list_csl_grain_boundaries(group, &cell, max_sigma);
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

pub(crate) fn pure_stone_wales_flip(
    crystal_json: &str,
    edge_idx: usize,
    cell_json: Option<&str>,
) -> Result<String, String> {
    let crystal: BuiltCrystal = serde_json::from_str(crystal_json).map_err(|e| e.to_string())?;
    let cell: Option<Cell2D> = parse_opt_json(cell_json)?;
    let res = stone_wales_flip(&crystal, edge_idx, cell.as_ref())?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

pub(crate) fn pure_apply_point_defects(crystal_json: &str, ops_json: &str) -> Result<String, String> {
    let crystal: BuiltCrystal = serde_json::from_str(crystal_json).map_err(|e| e.to_string())?;
    let ops: Vec<PointDefectOp> = serde_json::from_str(ops_json).map_err(|e| e.to_string())?;
    let res = apply_point_defects(&crystal, &ops)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[pyfunction]
#[pyo3(signature = (cell_json, nx=1, ny=1, species_json=None))]
pub fn export_cell_extxyz_rs(
    cell_json: &str,
    nx: usize,
    ny: usize,
    species_json: Option<&str>,
) -> PyResult<String> {
    pure_export_cell_extxyz(cell_json, nx, ny, species_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (crystal_json, species_json=None, z_json=None))]
pub fn export_crystal_extxyz_rs(
    crystal_json: &str,
    species_json: Option<&str>,
    z_json: Option<&str>,
) -> PyResult<String> {
    pure_export_crystal_extxyz(crystal_json, species_json, z_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, max_shells=3))]
pub fn classify_cell_shells_json(
    group: &str,
    cell_json: &str,
    max_shells: usize,
) -> PyResult<String> {
    pure_classify_cell_shells(group, cell_json, max_shells).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn cluster_site_permutations_json(
    positions_json: &str,
    labels_json: &str,
    shells_json: &str,
) -> PyResult<String> {
    pure_cluster_site_permutations(positions_json, labels_json, shells_json)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, nx, ny, max_shells=3))]
pub fn periodic_supercell_solver_json(
    group: &str,
    cell_json: &str,
    nx: usize,
    ny: usize,
    max_shells: usize,
) -> PyResult<String> {
    pure_periodic_supercell_solver(group, cell_json, nx, ny, max_shells)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, max_sigma=31))]
pub fn list_csl_grain_boundaries_json(
    group: &str,
    cell_json: &str,
    max_sigma: u32,
) -> PyResult<String> {
    pure_list_csl_grain_boundaries(group, cell_json, max_sigma).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (crystal_json, edge_idx, cell_json=None))]
pub fn stone_wales_flip_json(
    crystal_json: &str,
    edge_idx: usize,
    cell_json: Option<&str>,
) -> PyResult<String> {
    pure_stone_wales_flip(crystal_json, edge_idx, cell_json).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn apply_point_defects_json(crystal_json: &str, ops_json: &str) -> PyResult<String> {
    pure_apply_point_defects(crystal_json, ops_json).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(export_cell_extxyz_rs, m)?)?;
    m.add_function(wrap_pyfunction!(export_crystal_extxyz_rs, m)?)?;
    m.add_function(wrap_pyfunction!(classify_cell_shells_json, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_site_permutations_json, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_supercell_solver_json, m)?)?;
    m.add_function(wrap_pyfunction!(list_csl_grain_boundaries_json, m)?)?;
    m.add_function(wrap_pyfunction!(stone_wales_flip_json, m)?)?;
    m.add_function(wrap_pyfunction!(apply_point_defects_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use papertilling_core::defect::perfect_disk;
    use papertilling_core::types::Cell2D;

    use super::*;

    #[test]
    fn solver_py_wrappers_roundtrip() {
        let sq = r#"{"a1":[1.0,0.0],"a2":[0.0,1.0],"basis":[[0.0,0.0]],"bonds":[{"i":0,"j":0,"n1":1,"n2":0,"kind":0},{"i":0,"j":0,"n1":0,"n2":1,"kind":0}]}"#;
        let xyz = pure_export_cell_extxyz(sq, 2, 2, Some("null")).unwrap();
        assert!(xyz.starts_with("4\n") && xyz.contains("pbc=\"T T F\""));

        let sh: serde_json::Value =
            serde_json::from_str(&pure_classify_cell_shells("p4m", sq, 3).unwrap()).unwrap();
        assert_eq!(sh["shells"][0]["coordination_by_basis"], serde_json::json!([4]));

        let sup: serde_json::Value =
            serde_json::from_str(&pure_periodic_supercell_solver("p4m", sq, 4, 4, 3).unwrap())
                .unwrap();
        assert_eq!(sup["report"]["order"], 128);
        assert_eq!(sup["report"]["group_closed"], true);

        let cl: serde_json::Value = serde_json::from_str(
            &pure_cluster_site_permutations(
                "[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]",
                "[0,0,0,0]",
                "[[[0,1],[1,2],[2,3],[3,0]]]",
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(cl["order"], 8);
        assert_eq!(cl["group_closed"], true);

        let csl: serde_json::Value =
            serde_json::from_str(&pure_list_csl_grain_boundaries("p4m", sq, 13).unwrap()).unwrap();
        assert_eq!(csl.as_array().unwrap()[0]["sigma"], 5);

        let cell: Cell2D = serde_json::from_str(sq).unwrap();
        let disk = perfect_disk(&cell, [0.0, 0.0], 2.0);
        let cjson = serde_json::to_string(&disk).unwrap();
        let cxyz = pure_export_crystal_extxyz(&cjson, None, None).unwrap();
        assert!(cxyz.contains("pbc=\"F F F\""));

        let centre = disk.pos.iter().position(|p| *p == [0.0, 0.0]).unwrap();
        let ops = format!(r#"[{{"Vacancy":{{"site":{centre},"reconstruct_ring":false}}}}]"#);
        let pd: serde_json::Value =
            serde_json::from_str(&pure_apply_point_defects(&cjson, &ops).unwrap()).unwrap();
        assert_eq!(pd["crystal"]["pos"].as_array().unwrap().len(), disk.pos.len() - 1);

        let hc = r#"{"a1":[1.7320508075688772,0.0],"a2":[0.8660254037844386,1.5],"basis":[[0.0,0.0],[0.0,1.0]],"bonds":[{"i":0,"j":1,"n1":0,"n2":0,"kind":0},{"i":0,"j":1,"n1":0,"n2":-1,"kind":0},{"i":0,"j":1,"n1":1,"n2":-1,"kind":0}]}"#;
        let hcell: Cell2D = serde_json::from_str(hc).unwrap();
        let hdisk = perfect_disk(&hcell, [0.0, 0.0], 4.0);
        let eidx = (0..hdisk.edges.len())
            .min_by(|&a, &b| {
                let d = |k: usize| {
                    let (u, v) = hdisk.edges[k];
                    (0.5 * (hdisk.pos[u][0] + hdisk.pos[v][0]))
                        .hypot(0.5 * (hdisk.pos[u][1] + hdisk.pos[v][1]))
                };
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        let sw: serde_json::Value = serde_json::from_str(
            &pure_stone_wales_flip(&serde_json::to_string(&hdisk).unwrap(), eidx, None).unwrap(),
        )
        .unwrap();
        assert_eq!(sw["before_rings"], serde_json::json!([6, 6, 6, 6]));
    }
}
