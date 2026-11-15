//! Geometry-agnostic "pieces + pairings" surface types, flat quotients, genus-`g` covers,
//! polyhedral spheres/RP2, and `surface_table`.

pub mod checks;
pub mod flat;
pub mod genus;
pub mod pieces;
pub mod pieces_glue;
pub mod polyhedra;
pub(crate) mod quotient;
pub mod report;
pub mod solids;

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::types::Cell2D;

pub use checks::{check_surface, face_histogram};
pub use flat::{
    GlideAxis, cylinder, default_cylinder, default_klein_bottle, default_mobius_band,
    default_torus, glides, klein_bottle, mobius_band, torus,
};
pub use genus::{default_genus_surface, genus_surface, genus_surface_with_options};
pub use pieces::{
    ConeSurface, LatticeData, Net2D, PairingSpec, PairingTestRow, Piece2D, check_surface_mesh,
    glue, pairing_test,
};
pub use polyhedra::{CentreClass, candidate_centres, projective_plane, sphere};
pub use report::{
    FlatAttempt, SURFACES, SurfaceTableRow, SymmetryFacts, surface_table_with, symmetry_facts,
    try_rp2_for_table, try_sphere_for_table,
};

/// Evaluate all 8 canonical `SURFACES` rows (`torus`, `cylinder`, `Moebius band`, `Klein bottle`,
/// `sphere`, `projective plane`, `genus 2`, `genus 3`) on `(group, cell)`.
pub fn surface_table(group: &str, cell: &Cell2D) -> Vec<SurfaceTableRow> {
    let gl = glides(cell);
    let to_err = |msg: String| {
        if let Some(rest) = msg.strip_prefix("IMPOSSIBLE: ") {
            SurfaceError::impossible(rest)
        } else {
            SurfaceError::not_found(msg, String::new(), false)
        }
    };
    surface_table_with(
        group,
        cell,
        gl.len(),
        |name, _att| match name {
            "torus" => default_torus(cell).map_err(to_err),
            "cylinder" => default_cylinder(cell).map_err(to_err),
            "Moebius band" | "Mobius band" => default_mobius_band(cell).map_err(to_err),
            "Klein bottle" => default_klein_bottle(cell).map_err(to_err),
            _ => Err(SurfaceError::invalid(format!("unknown flat surface {name}"))),
        },
        |g, (sz_u, sz_v)| {
            genus_surface_with_options(cell, g as usize, [sz_u, 0], [0, sz_v], None)
                .map_err(|msg| SurfaceError::not_found(msg, String::new(), true))
        },
    )
}

/// Classification of surface construction refusals (`IMPOSSIBLE` vs `NOT FOUND`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    Impossible,
    NotFound { retry: bool },
    InvalidArg,
}

/// Detailed fields of a [`SurfaceError`], boxed to keep `Result<_, SurfaceError>` pointer-sized.
#[derive(Debug, Clone)]
pub struct SurfaceErrorData {
    pub kind: RefusalKind,
    pub reason: String,
    pub detail: String,
    pub tried: Vec<String>,
    pub hint: Option<String>,
    pub centres: Vec<String>,
    pub causes: Vec<String>,
}

/// Structured refusal returned when a surface construction is impossible or refused at given settings.
#[derive(Debug, Clone)]
pub struct SurfaceError(Box<SurfaceErrorData>);

impl std::ops::Deref for SurfaceError {
    type Target = SurfaceErrorData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SurfaceError {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl SurfaceError {
    fn base(kind: RefusalKind, reason: impl Into<String>, detail: impl Into<String>) -> Self {
        Self(Box::new(SurfaceErrorData {
            kind,
            reason: reason.into(),
            detail: detail.into(),
            tried: Vec::new(),
            hint: None,
            centres: Vec::new(),
            causes: Vec::new(),
        }))
    }
    pub fn impossible(reason: impl Into<String>) -> Self {
        Self::base(RefusalKind::Impossible, reason, String::new())
    }
    pub fn not_found(reason: impl Into<String>, detail: impl Into<String>, retry: bool) -> Self {
        Self::base(RefusalKind::NotFound { retry }, reason, detail)
    }
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::base(RefusalKind::InvalidArg, reason, String::new())
    }
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)?;
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        if !self.tried.is_empty() {
            write!(f, " [tried: {}]", self.tried.join("; "))?;
        }
        if let Some(h) = &self.hint {
            write!(f, " ({h})")?;
        }
        Ok(())
    }
}

impl std::error::Error for SurfaceError {}

#[inline]
pub(super) fn add2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] + b[0], a[1] + b[1]] }
#[inline]
pub(super) fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] { [a[0] - b[0], a[1] - b[1]] }
#[inline]
pub(super) fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[0] + a[1] * b[1] }
#[inline]
pub(super) fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 { a[0] * b[1] - a[1] * b[0] }
#[inline]
pub(super) fn norm2(a: [f64; 2]) -> f64 { dot2(a, a).sqrt() }
#[inline]
pub(super) fn mat2_vec(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}
#[inline]
pub(super) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}
pub(super) fn mat2_inv(m: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], String> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-14 { Err("singular 2x2 matrix".into()) } else { Ok([[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]) }
}
#[inline]
pub(super) fn rot2(angle: f64) -> [[f64; 2]; 2] {
    let (s, c) = angle.sin_cos();
    [[c, -s], [s, c]]
}
#[inline]
pub(super) fn frac_mod(x: f64) -> f64 {
    let mut g = x - (x + 1e-6).floor();
    if (g - 1.0).abs() < 1e-6 || g.abs() < 1e-12 { g = 0.0; }
    g
}

#[derive(Debug, Clone)]
pub(super) struct IndexUf {
    parent: Vec<usize>,
}

impl IndexUf {
    pub(super) fn new(n: usize) -> Self { Self { parent: (0..n).collect() } }
    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    pub(super) fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb { self.parent[ra.max(rb)] = ra.min(rb); }
    }
    pub(super) fn roots(&mut self) -> Vec<usize> {
        (0..self.parent.len()).filter(|&i| self.find(i) == i).collect()
    }
}

pub type SiteLabel = (usize, i32, i32);

pub(super) fn apply_to_site(mf: [[i32; 2]; 2], tf: [f64; 2], f_basis: &[[f64; 2]], k: usize) -> Option<(usize, [i32; 2])> {
    let fk = f_basis[k];
    let g = [mf[0][0] as f64 * fk[0] + mf[0][1] as f64 * fk[1] + tf[0], mf[1][0] as f64 * fk[0] + mf[1][1] as f64 * fk[1] + tf[1]];
    for (j, fj) in f_basis.iter().enumerate() {
        let (d0, d1) = (g[0] - fj[0], g[1] - fj[1]);
        let (r0, r1) = (d0.round(), d1.round());
        if (d0 - r0).abs() < 1e-6 && (d1 - r1).abs() < 1e-6 {
            return Some((j, [r0 as i32, r1 as i32]));
        }
    }
    None
}

pub(super) fn symmetry_ops_on_cell(
    a: [[f64; 2]; 2],
    a_inv: [[f64; 2]; 2],
    f_basis: &[[f64; 2]],
    bonds: &[(usize, usize, i32, i32)],
    darts: &HashSet<(usize, usize, i32, i32)>,
    only_identity: bool,
) -> Vec<([[i32; 2]; 2], [f64; 2])> {
    let mut cart_mats = Vec::with_capacity(64);
    if only_identity {
        cart_mats.push([[1.0, 0.0], [0.0, 1.0]]);
    } else {
        for k in 0..12 { cart_mats.push(rot2(k as f64 * std::f64::consts::PI / 6.0)); }
        for n1 in -3..=3 {
            for n2 in -3..=3 {
                let v = mat2_vec(a, [n1 as f64, n2 as f64]);
                if (n1 != 0 || n2 != 0) && norm2(v) > 1e-6 {
                    let ang = v[1].atan2(v[0]);
                    for x in [ang, ang + std::f64::consts::FRAC_PI_2] {
                        let (s, c) = (2.0 * x).sin_cos();
                        cart_mats.push([[c, s], [s, -c]]);
                    }
                }
            }
        }
    }
    let (mut seen_mf, mut ops) = (HashSet::new(), Vec::<([[i32; 2]; 2], [f64; 2])>::new());
    for m_cart in cart_mats {
        let mf_f = mat2_mul(mat2_mul(a_inv, m_cart), a);
        let r = [[mf_f[0][0].round(), mf_f[0][1].round()], [mf_f[1][0].round(), mf_f[1][1].round()]];
        if (mf_f[0][0] - r[0][0]).abs() > 1e-6 || (mf_f[0][1] - r[0][1]).abs() > 1e-6
            || (mf_f[1][0] - r[1][0]).abs() > 1e-6 || (mf_f[1][1] - r[1][1]).abs() > 1e-6
        {
            continue;
        }
        let mf = [[r[0][0] as i32, r[0][1] as i32], [r[1][0] as i32, r[1][1] as i32]];
        if (mf[0][0] * mf[1][1] - mf[0][1] * mf[1][0]).abs() != 1 || !seen_mf.insert(mf) || f_basis.is_empty() {
            continue;
        }
        let mf0 = [mf[0][0] as f64 * f_basis[0][0] + mf[0][1] as f64 * f_basis[0][1], mf[1][0] as f64 * f_basis[0][0] + mf[1][1] as f64 * f_basis[0][1]];
        for fj in f_basis {
            let t = [frac_mod(fj[0] - mf0[0]), frac_mod(fj[1] - mf0[1])];
            let Some(imgs): Option<Vec<_>> = (0..f_basis.len()).map(|k| apply_to_site(mf, t, f_basis, k)).collect() else { continue };
            let bonds_ok = bonds.iter().all(|&(i, j, n1, n2)| {
                let ((pi, ci), (pj, cj)) = (imgs[i], imgs[j]);
                darts.contains(&(pi, pj, mf[0][0] * n1 + mf[0][1] * n2 + cj[0] - ci[0], mf[1][0] * n1 + mf[1][1] * n2 + cj[1] - ci[1]))
            });
            if bonds_ok && !ops.iter().any(|&(m2, t2)| m2 == mf && (t[0] - t2[0] - (t[0] - t2[0]).round()).abs() < 1e-5 && (t[1] - t2[1] - (t[1] - t2[1]).round()).abs() < 1e-5) {
                ops.push((mf, t));
            }
        }
    }
    ops
}

/// Target surface topologies supported by the surface engine (Stage 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SurfaceKind {
    Torus,
    Cylinder,
    MobiusBand,
    KleinBottle,
    Sphere,
    ProjectivePlane,
    Genus(u32),
}

/// A boundary half-edge endpoint on a polygonal chart piece `(piece_id, boundary_step)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoundarySide {
    pub piece: usize,
    pub edge: usize,
}

/// Identification between two boundary sides, optionally reversing boundary orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgePairing {
    pub left: BoundarySide,
    pub right: BoundarySide,
    pub orientation_reversing: bool,
}

/// One combinatorial chart piece in a pieces-and-pairings decomposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfacePiece {
    pub id: usize,
    pub num_vertices: usize,
    pub faces: Vec<Vec<usize>>,
}

/// Elementary disclination cone carried by a curved or branched surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConeDefect {
    pub frank_angle_deg: i32,
    pub centre_class: String,
    pub multiplicity: usize,
}

/// Outcome of attempting to realise a lattice on a target surface (`surface_table` verdicts).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceVerdict {
    Ok {
        vertices: usize,
        edges: usize,
        faces: usize,
        orientable: bool,
        cones: Vec<ConeDefect>,
    },
    Impossible { reason: String },
    NotFound { reason: String, hint: Option<String> },
}

/// Geometry-free topological interface shared by flat quotients, polyhedra, and higher-genus covers.
pub trait SurfaceTopology {
    fn kind(&self) -> SurfaceKind;
    fn vertices(&self) -> usize;
    fn edges(&self) -> usize;
    fn faces(&self) -> usize;
    fn is_orientable(&self) -> bool;
    fn cones(&self) -> &[ConeDefect];

    fn euler_characteristic(&self) -> isize {
        self.vertices() as isize - self.edges() as isize + self.faces() as isize
    }
}
