//! Stage 2–5 Python JSON bridges: Burgers circuits (`burgers_circuit_json`,
//! `auto_burgers_circuit_json`), 3D elastic relaxation (`relax_surface_3d_json`,
//! `relax_defect_sheet_3d_json`), and tight-binding spectrum (`spectrum_json`).

use std::collections::BTreeMap;

use papertilling_core::defect::gauge::{
    FusionChannel, PlaquetteHolonomy, SpaceGroupExtension, WyckoffRotocentre, fusion_channels,
    plaquette_holonomies, ribbon_faces, wyckoff_rotocentres_with_frame,
};
use papertilling_core::defect::gsv_elastic::relax_elastic_2d;
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::defect::{auto_burgers_circuit, burgers_circuit};
use papertilling_core::spectrum::{Eigenvectors, SpectrumOptions, TightBindingInput, solve_spectrum};
use papertilling_core::surface::elastic3d::{
    estimate_planar_frank_deg, relax_indexed_elastic, relax_surface_3d,
};
use papertilling_core::types::{BuiltCrystal, Cell2D, ConeMark, Frame2D, SurfaceCover3D, SurfaceMesh};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

fn default_true() -> bool { true }
fn default_kappa() -> f64 { 0.02 }
fn default_max_iter() -> usize { 400 }
fn default_t0() -> f64 { 1.0 }
fn default_seam_sign() -> i32 { 1 }

fn parse_frame(frame_json: Option<&str>) -> Result<Frame2D, String> {
    match frame_json {
        Some(s) => serde_json::from_str(s).map_err(|e| e.to_string()),
        None => Ok(Frame2D::default()),
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PointListArg { One([f64; 2]), Many(Vec<[f64; 2]>) }

fn parse_point_list(raw: Option<&str>) -> Result<Vec<[f64; 2]>, String> {
    let Some(s) = raw else { return Ok(Vec::new()); };
    let val: serde_json::Value = serde_json::from_str(s).map_err(|e| e.to_string())?;
    if val.is_null() { return Ok(Vec::new()); }
    match serde_json::from_value::<PointListArg>(val).map_err(|e| e.to_string())? {
        PointListArg::One(p) => Ok(vec![p]),
        PointListArg::Many(ps) => Ok(ps),
    }
}

pub(crate) fn pure_burgers_circuit(
    group: &str, cell_json: &str, specs_json: &str, path_json: &str,
    r_patch: Option<f64>, frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let specs: Vec<DefectSpec> = serde_json::from_str(specs_json).map_err(|e| e.to_string())?;
    let path: Vec<usize> = serde_json::from_str(path_json).map_err(|e| e.to_string())?;
    let crystal = build_point_defects(&cell, &specs, r_patch, None)?;
    let res = burgers_circuit(&crystal, &path, Some((group, &cell, parse_frame(frame_json)?)))?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

pub(crate) fn pure_auto_burgers_circuit(
    group: &str, cell_json: &str, specs_json: &str, target_json: Option<&str>,
    exclude_json: Option<&str>, r_patch: Option<f64>, frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let specs: Vec<DefectSpec> = serde_json::from_str(specs_json).map_err(|e| e.to_string())?;
    let (target, exclude) = (parse_point_list(target_json)?, parse_point_list(exclude_json)?);
    let crystal = build_point_defects(&cell, &specs, r_patch, None)?;
    let info = (group, &cell, parse_frame(frame_json)?);
    let res = auto_burgers_circuit(&crystal, Some(info), &target, &exclude, 2, &specs)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
struct ConeArg {
    #[serde(default)] site: Option<usize>,
    #[serde(default)] sites: Vec<usize>,
    #[serde(default)] face: Option<usize>,
    #[serde(default)] faces: Vec<usize>,
    #[serde(default)] angle: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum CoverBondArg { WithVec(usize, usize, [f64; 2]), Plain(usize, usize) }

#[derive(Deserialize)]
struct CoverArg {
    xyz: Vec<[f64; 3]>,
    bonds: Vec<CoverBondArg>,
    faces: Vec<Vec<usize>>,
    #[serde(default)] cones: Vec<ConeArg>,
    #[serde(default)] cone_faces: Vec<usize>,
}

#[derive(Deserialize)]
struct SurfacePayload {
    #[serde(default)] kind: String,
    #[serde(default)] uv: Vec<[f64; 2]>,
    #[serde(default)] xyz: Option<Vec<[f64; 3]>>,
    bonds: Vec<(usize, usize, [f64; 2])>,
    faces: Vec<Vec<usize>>,
    #[serde(default)] cones: Vec<ConeArg>,
    #[serde(default)] chi: i32,
    #[serde(default = "default_true")] orientable: bool,
    #[serde(default)] closed: bool,
    #[serde(default)] cover: Option<CoverArg>,
    #[serde(default = "default_kappa")] kappa_b: f64,
    #[serde(default = "default_max_iter", alias = "max_iters")] max_iter: usize,
}

pub(crate) fn pure_relax_surface_3d(payload_json: &str) -> Result<String, String> {
    let req: SurfacePayload = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
    let v_count = if !req.uv.is_empty() { req.uv.len() }
        else if let Some(ref x) = req.xyz { x.len() }
        else { req.bonds.iter().flat_map(|&(a, b, _)| [a, b]).max().map_or(0, |m| m + 1) };
    let cones = req.cones.into_iter().map(|c| ConeMark {
        angle_deg: c.angle, site: c.site.or_else(|| c.sites.first().copied()),
        face: c.face.or_else(|| c.faces.first().copied()), class_label: None, location: String::new(),
    }).collect();
    let cover = req.cover.map(|cov| {
        let mut cone_faces = cov.cone_faces;
        if cone_faces.is_empty() {
            for c in &cov.cones {
                if let Some(f) = c.face { cone_faces.push(f); }
                cone_faces.extend_from_slice(&c.faces);
            }
        }
        let mut cone_sites = Vec::new();
        for c in &cov.cones {
            if let Some(s) = c.site { cone_sites.push(s); }
            cone_sites.extend_from_slice(&c.sites);
        }
        Box::new(SurfaceCover3D {
            xyz: cov.xyz,
            bonds: cov.bonds.into_iter().map(|b| match b {
                CoverBondArg::WithVec(u, v, _) | CoverBondArg::Plain(u, v) => (u, v),
            }).collect(),
            faces: cov.faces, site_rgba: Vec::new(), cone_sites, cone_faces,
        })
    });
    let mesh = SurfaceMesh {
        name: req.kind, sites: vec![(0, 0, 0, 0); v_count], bonds: req.bonds, faces: req.faces,
        plane_degree: Vec::new(), edge_sites: Vec::new(), chi: req.chi, orientable: req.orientable,
        closed: req.closed, cones, uv: req.uv, domain_polygons: Vec::new(), domain_edges: Vec::new(),
        slits: Vec::new(), xyz: req.xyz.unwrap_or_default(), bond_curves_3d: Vec::new(), cover,
    };
    serde_json::to_string(&relax_surface_3d(&mesh, req.kappa_b, req.max_iter)?).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
struct DefectSheetPayload {
    positions: Vec<[f64; 2]>,
    edges: Vec<(usize, usize)>,
    #[serde(default)] faces: Vec<Vec<usize>>,
    #[serde(default)] edge_l0: Vec<f64>,
    #[serde(default)] spoke_lengths: BTreeMap<usize, Vec<f64>>,
    #[serde(default)] lmean: Option<f64>,
    #[serde(default = "default_kappa")] kappa_b: f64,
    #[serde(default = "default_max_iter", alias = "max_iters")] max_iter: usize,
}

#[allow(clippy::approx_constant)]
pub(crate) fn pure_relax_defect_sheet_3d(payload_json: &str) -> Result<String, String> {
    let req: DefectSheetPayload = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
    let v_count = req.positions.len();
    if req.edges.iter().any(|&(u, v)| u >= v_count || v >= v_count) {
        return Err(format!("edge index out of bounds for V={v_count}"));
    }
    let edge_l0: Vec<f64> = if req.edge_l0.is_empty() {
        req.edges.iter().map(|&(u, v)| {
            let (p, q) = (req.positions[u], req.positions[v]);
            (p[0] - q[0]).hypot(p[1] - q[1]).max(1e-6)
        }).collect()
    } else if req.edge_l0.len() == req.edges.len() { req.edge_l0 }
    else { return Err(format!("edge_l0 len {} != edges len {}", req.edge_l0.len(), req.edges.len())); };

    let bmap: BTreeMap<(usize, usize), f64> = req.edges.iter().zip(&edge_l0)
        .map(|(&(u, v), &l)| ((u.min(v), u.max(v)), l)).collect();
    let a0 = if !edge_l0.is_empty() { edge_l0.iter().sum::<f64>() / (edge_l0.len() as f64) }
        else { req.lmean.unwrap_or(1.0) };
    let lmean = req.lmean.unwrap_or(a0);
    let frank = estimate_planar_frank_deg(&req.faces, &bmap, &req.spoke_lengths);

    let mut x0 = Vec::with_capacity(v_count);
    if v_count > 0 {
        let (mx, my) = (
            req.positions.iter().map(|p| p[0]).sum::<f64>() / (v_count as f64),
            req.positions.iter().map(|p| p[1]).sum::<f64>() / (v_count as f64),
        );
        let polar: Vec<(f64, f64)> = req.positions.iter().map(|p| ((p[0] - mx).hypot(p[1] - my), (p[1] - my).atan2(p[0] - mx))).collect();
        let rmax = polar.iter().map(|x| x.0).fold(0.0_f64, f64::max).max(lmean).max(1.0);
        let denom = (12.0 * a0 * a0).max(0.25 * rmax * rmax);
        for (i, (&p, &(r, th))) in req.positions.iter().zip(&polar).enumerate() {
            let mut z = if frank > 3.0 {
                0.55 * (2.0 * (frank / 360.0).clamp(0.02, 0.45)).sqrt().clamp(0.25, 0.95) * (rmax - r) + 0.02 * a0 * (2.0 * th).cos()
            } else if frank < -3.0 {
                0.5 * ((-frank / 360.0).clamp(0.02, 0.65)).sqrt().clamp(0.25, 0.75) * r * (2.0 * th).cos() + 0.02 * a0 * (3.0 * th).sin()
            } else {
                0.35 * a0 * (-(r * r) / denom).exp() * (1.0 + 0.35 * th.cos()) + 0.1 * a0 * (r / rmax) * (2.0 * th).cos()
            };
            z += 1e-3 * a0 * ((i as f64) * 1.61803398875).sin();
            x0.push([p[0], p[1], z]);
        }
    }
    let bond_edge_idx: Vec<usize> = (0..req.edges.len()).collect();
    let res = relax_indexed_elastic(
        v_count, &x0, &req.edges, &edge_l0, &bond_edge_idx, &req.faces,
        &req.spoke_lengths, req.kappa_b, req.max_iter, false, true,
    )?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
struct SpectrumPayload {
    n: usize,
    #[serde(default)] closed: bool,
    #[serde(default)] edges: Vec<(usize, usize)>,
    #[serde(default)] r_ij: Vec<f64>,
    #[serde(default)] p_i: Vec<[f64; 2]>,
    #[serde(default)] p_j: Vec<[f64; 2]>,
    #[serde(default)] seam: Vec<bool>,
    #[serde(default)] seeds: Vec<Vec<usize>>,
    #[serde(default)] beta: f64,
    #[serde(default)] flux: f64,
    #[serde(default)] mu: f64,
    #[serde(default)] filling: Option<f64>,
    #[serde(default = "default_seam_sign")] seam_sign: i32,
    #[serde(default = "default_true")] multiplet: bool,
    #[serde(default = "default_t0")] t0: f64,
    #[serde(default)] core_radius: Option<f64>,
    #[serde(default)] level_idx: Option<isize>,
}

#[derive(Serialize)]
struct SpectrumResultJson<'a> {
    eigenvalues: &'a [f64], eigenvectors: &'a Eigenvectors, zero_mode_count: usize,
    core_weight: &'a [f64], multiplets: Vec<(f64, usize, usize, usize)>, ldos: &'a [f64],
    mu: f64, delta_rho: &'a [f64], cone_charges: &'a [f64], #[serde(rename = "delta_Q")] delta_q: &'a [f64],
}

pub(crate) fn pure_spectrum(payload_json: &str) -> Result<String, String> {
    let req: SpectrumPayload = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
    let input = TightBindingInput {
        n: req.n, closed: req.closed, edges: req.edges, r_ij: req.r_ij,
        p_i: req.p_i, p_j: req.p_j, seam: req.seam, seeds: req.seeds,
    };
    let opts = SpectrumOptions {
        beta: req.beta, flux: req.flux, mu: req.mu, filling: req.filling,
        seam_sign: req.seam_sign, multiplet: req.multiplet, t0: req.t0,
        core_radius: req.core_radius, level_idx: req.level_idx,
    };
    let res = solve_spectrum(&input, &opts)?;
    let multiplets = res.multiplets.iter().map(|m| (m.energy, m.start, m.end, m.dim)).collect();
    serde_json::to_string(&SpectrumResultJson {
        eigenvalues: &res.eigenvalues, eigenvectors: &res.eigenvectors,
        zero_mode_count: res.zero_mode_count, core_weight: &res.core_weight,
        multiplets, ldos: &res.ldos, mu: res.mu, delta_rho: &res.delta_rho,
        cone_charges: &res.cone_charges, delta_q: &res.delta_q,
    }).map_err(|e| e.to_string())
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, specs_json, path_json, r_patch=None, frame_json=None))]
pub fn burgers_circuit_json(
    group: &str, cell_json: &str, specs_json: &str, path_json: &str, r_patch: Option<f64>, frame_json: Option<&str>,
) -> PyResult<String> {
    pure_burgers_circuit(group, cell_json, specs_json, path_json, r_patch, frame_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, specs_json, target_json=None, exclude_json=None, r_patch=None, frame_json=None))]
pub fn auto_burgers_circuit_json(
    group: &str, cell_json: &str, specs_json: &str,
    target_json: Option<&str>, exclude_json: Option<&str>, r_patch: Option<f64>, frame_json: Option<&str>,
) -> PyResult<String> {
    pure_auto_burgers_circuit(group, cell_json, specs_json, target_json, exclude_json, r_patch, frame_json)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn relax_surface_3d_json(payload_json: &str) -> PyResult<String> {
    pure_relax_surface_3d(payload_json).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn relax_defect_sheet_3d_json(payload_json: &str) -> PyResult<String> {
    pure_relax_defect_sheet_3d(payload_json).map_err(PyValueError::new_err)
}

#[derive(Serialize)]
struct GaugeObservablesJson {
    extension: SpaceGroupExtension,
    wyckoff_rotocentres: Vec<WyckoffRotocentre>,
    fusion_channels: Option<Vec<FusionChannel>>,
    ribbon_faces: Vec<Vec<usize>>,
    plaquettes: Vec<PlaquetteHolonomy>,
}

pub(crate) fn pure_gauge_observables(
    group: &str, cell_json: &str, specs_json: &str, r_patch: Option<f64>, frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let specs: Vec<DefectSpec> = serde_json::from_str(specs_json).map_err(|e| e.to_string())?;
    let frame = parse_frame(frame_json)?;
    let extension = SpaceGroupExtension::from_cell(group, &cell)?;
    let wyckoff_rotocentres = wyckoff_rotocentres_with_frame(group, &cell, frame)?;
    let fusion_channels = (specs.len() == 2)
        .then(|| fusion_channels(group, &cell, &specs))
        .transpose()?;
    let crystal = build_point_defects(&cell, &specs, r_patch, None)?;
    let ribbon_faces = ribbon_faces(&crystal);
    let plaquettes = plaquette_holonomies(&crystal, Some(group), Some(&cell));
    serde_json::to_string(&GaugeObservablesJson {
        extension, wyckoff_rotocentres, fusion_channels, ribbon_faces, plaquettes,
    }).map_err(|e| e.to_string())
}

pub(crate) fn pure_relax_elastic_2d(
    group: &str, cell_json: &str, specs_json: &str, r_patch: Option<f64>, frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let specs: Vec<DefectSpec> = serde_json::from_str(specs_json).map_err(|e| e.to_string())?;
    let _ = parse_frame(frame_json)?;
    if !group.is_empty() {
        SpaceGroupExtension::from_cell(group, &cell)?;
    }
    let crystal = build_point_defects(&cell, &specs, r_patch, None)?;
    let res = relax_elastic_2d(&crystal)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, specs_json, r_patch=None, frame_json=None))]
pub fn gauge_observables_json(
    group: &str, cell_json: &str, specs_json: &str, r_patch: Option<f64>, frame_json: Option<&str>,
) -> PyResult<String> {
    pure_gauge_observables(group, cell_json, specs_json, r_patch, frame_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, specs_json, r_patch=None, frame_json=None))]
pub fn relax_elastic_2d_json(
    group: &str, cell_json: &str, specs_json: &str, r_patch: Option<f64>, frame_json: Option<&str>,
) -> PyResult<String> {
    pure_relax_elastic_2d(group, cell_json, specs_json, r_patch, frame_json).map_err(PyValueError::new_err)
}

pub(crate) fn pure_relax_elastic_2d_patch(payload_json: &str) -> Result<String, String> {
    let crystal: BuiltCrystal = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
    let res = relax_elastic_2d(&crystal)?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[pyfunction]
pub fn relax_elastic_2d_patch_json(payload_json: &str) -> PyResult<String> {
    pure_relax_elastic_2d_patch(payload_json).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn spectrum_json(payload_json: &str) -> PyResult<String> {
    pure_spectrum(payload_json).map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insights_json_wrappers_roundtrip() {
        let sq = r#"{"a1":[1.0,0.0],"a2":[0.0,1.0],"basis":[[0.0,0.0]],"bonds":[{"i":0,"j":0,"n1":1,"n2":0,"kind":0},{"i":0,"j":0,"n1":0,"n2":1,"kind":0}]}"#;
        let specs = r#"[{"kind":"disclination","angle":90.0,"at":[0.0,0.0]},{"kind":"disclination","angle":-90.0,"at":[0.5,0.5]}]"#;
        let auto_v: serde_json::Value = serde_json::from_str(&pure_auto_burgers_circuit("p4m", sq, specs, None, None, None, None).unwrap()).unwrap();
        assert_eq!(auto_v["charge"]["kind"], "dislocation");
        let path_s = serde_json::to_string(&auto_v["path"]).unwrap();
        let circ_v: serde_json::Value = serde_json::from_str(&pure_burgers_circuit("p4m", sq, specs, &path_s, None, None).unwrap()).unwrap();
        assert!((circ_v["burgers_norm"].as_f64().unwrap() - 1.0).abs() < 1e-6);

        let surf_p = r#"{"kind":"patch","uv":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],"bonds":[[0,1,[1.0,0.0]],[1,2,[0.0,1.0]],[2,3,[-1.0,0.0]],[3,0,[0.0,-1.0]]],"faces":[[0,1,2,3]],"chi":1,"orientable":true,"closed":false,"kappa_b":0.05,"max_iter":100}"#;
        assert!(serde_json::from_str::<serde_json::Value>(&pure_relax_surface_3d(surf_p).unwrap()).unwrap()["converged"].as_bool().unwrap());

        let sh_p = r#"{"positions":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],"edges":[[0,1],[1,2],[2,3],[3,0]],"faces":[[0,1,2,3]],"edge_l0":[1.0,1.0,1.0,1.0],"spoke_lengths":{"0":[0.7071067811865476,0.7071067811865476,0.7071067811865476,0.7071067811865476]},"kappa_b":0.05,"max_iter":100}"#;
        assert!(serde_json::from_str::<serde_json::Value>(&pure_relax_defect_sheet_3d(sh_p).unwrap()).unwrap()["stretch_energy"].as_f64().unwrap() < 1e-8);

        let tb_p = r#"{"n":4,"closed":true,"edges":[[0,1],[1,2],[2,3],[3,0]],"r_ij":[1.0,1.0,1.0,1.0],"p_i":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],"p_j":[[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]],"seam":[false,false,false,false],"seeds":[[0]],"flux":0.25}"#;
        let tb_v: serde_json::Value = serde_json::from_str(&pure_spectrum(tb_p).unwrap()).unwrap();
        assert!(tb_v["multiplets"][0].is_array() && tb_v["eigenvectors"][0][0].is_array() && tb_v.get("delta_Q").is_some());

        let g_v: serde_json::Value = serde_json::from_str(&pure_gauge_observables("p4m", sq, specs, Some(5.0), None).unwrap()).unwrap();
        assert!(g_v["wyckoff_rotocentres"].is_array() && !g_v["ribbon_faces"].as_array().unwrap().is_empty() && g_v["fusion_channels"].is_array());
        let el_v: serde_json::Value = serde_json::from_str(&pure_relax_elastic_2d("p4m", sq, specs, Some(5.0), None).unwrap()).unwrap();
        assert!(el_v["converged"].as_bool().unwrap() && el_v["total_energy"].as_f64().unwrap() > 0.0);
    }
}
