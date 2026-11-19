//! The 4 compact and 11 paracompact regular honeycombs of H³, built in the hyperboloid model of
//! R^{3,1} and drawn in the Poincaré ball (Klein coordinates are kept too).
//!
//! With a Euclidean vertex figure (`(q-2)(r-2) = 4`) the seed vertex is light-like and every
//! vertex lies on the sphere at infinity (`|p| = 1`, `is_ideal`). With a Euclidean cell
//! (`(p-2)(q-2) = 4`) each cell is an infinite horospherical `{4,4}` / `{6,3}` tiling and is cut to
//! the faces within `patch_layers` edge-adjacency rings of its seed face.

use super::coxeter_gram::{classify_schlafli, CoxeterRealisation, HoneycombRegime};
use super::coxeter_orbit::{build_orbit_complex, OrbitOptions};
use super::honeycombs_e3::{build_euclidean_honeycomb, EuclideanKind};
use super::polychora_s3::build_polychoron;
use super::CellComplex3d;

pub const MAX_H3_SHELLS: u32 = 2;
pub const MAX_PATCH_LAYERS: u32 = 2;

/// Build a patch of the regular H³ honeycomb `{p,q,r}`: `shells` face-adjacency shells of cells
/// (`0..=2`) around the central cell; infinite cells truncated to `patch_layers` (`1..=2`) rings.
pub fn build_hyperbolic_honeycomb(p: u32, q: u32, r: u32, shells: u32, patch_layers: u32) -> Result<CellComplex3d, String> {
    let cls = classify_schlafli(p, q, r);
    let regime = match cls.regime {
        HoneycombRegime::HyperbolicCompactH3 => "HyperbolicCompactH3",
        HoneycombRegime::HyperbolicParacompactH3 => "HyperbolicParacompactH3",
        _ => return Err(format!("{{{p},{q},{r}}} is not a regular H³ honeycomb: {}", cls.reason)),
    };
    if shells > MAX_H3_SHELLS {
        return Err(format!("shells = {shells} exceeds the maximum {MAX_H3_SHELLS}"));
    }
    if cls.has_infinite_euclidean_cells && !(1..=MAX_PATCH_LAYERS).contains(&patch_layers) {
        return Err(format!("patch_layers = {patch_layers} must be in 1..={MAX_PATCH_LAYERS} for infinite cells"));
    }
    let real = CoxeterRealisation::new(p, q, r)?;
    let opts = OrbitOptions {
        shells,
        patch_layers,
        infinite_cells: cls.has_infinite_euclidean_cells,
        ideal_vertices: cls.has_ideal_vertices,
    };
    build_orbit_complex(&real, p, &opts, &format!("{{{p},{q},{r}}}"), regime)
}

/// Any buildable regular `{p,q,r}`: the whole polychoron (S³), a cubic patch (E³, `shells` face
/// shells), or an H³ patch. Refuses hyperideal and degenerate symbols with the `[PROVED]` reason.
pub fn build_honeycomb(p: u32, q: u32, r: u32, shells: u32, patch_layers: u32) -> Result<CellComplex3d, String> {
    let cls = classify_schlafli(p, q, r);
    match cls.regime {
        HoneycombRegime::SphericalS3 => build_polychoron(p, q, r),
        HoneycombRegime::EuclideanE3 => build_euclidean_honeycomb(EuclideanKind::Cubic, shells),
        HoneycombRegime::HyperbolicCompactH3 | HoneycombRegime::HyperbolicParacompactH3 => {
            build_hyperbolic_honeycomb(p, q, r, shells, patch_layers)
        }
        _ => Err(cls.reason),
    }
}
