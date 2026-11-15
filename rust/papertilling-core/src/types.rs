//! Shared data structures across plane, surface, defect, and draw modules.
//! Every parallel track uses these exact definitions so modules merge cleanly.

use serde::{Deserialize, Serialize};

fn default_a() -> f64 {
    1.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bond2D {
    pub i: usize,
    pub j: usize,
    pub n1: i32,
    pub n2: i32,
    pub kind: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cell2D {
    pub a1: [f64; 2],
    pub a2: [f64; 2],
    pub basis: Vec<[f64; 2]>,
    pub bonds: Vec<Bond2D>,
}

/// Cartesian 2D Euclidean isometry `x -> m * x + t` (`det(m) = +-1`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Isometry2 {
    pub m: [[f64; 2]; 2],
    pub t: [f64; 2],
}

impl Isometry2 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0], [0.0, 1.0]],
        t: [0.0, 0.0],
    };

    pub fn apply(&self, x: [f64; 2]) -> [f64; 2] {
        [
            self.m[0][0] * x[0] + self.m[0][1] * x[1] + self.t[0],
            self.m[1][0] * x[0] + self.m[1][1] * x[1] + self.t[1],
        ]
    }

    pub fn apply_linear(&self, v: [f64; 2]) -> [f64; 2] {
        [
            self.m[0][0] * v[0] + self.m[0][1] * v[1],
            self.m[1][0] * v[0] + self.m[1][1] * v[1],
        ]
    }

    pub fn det(&self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    /// Compose `self` after `other`: `(self * other)(x) = self(other(x))`.
    pub fn compose(&self, other: &Self) -> Self {
        let m = [
            [
                self.m[0][0] * other.m[0][0] + self.m[0][1] * other.m[1][0],
                self.m[0][0] * other.m[0][1] + self.m[0][1] * other.m[1][1],
            ],
            [
                self.m[1][0] * other.m[0][0] + self.m[1][1] * other.m[1][0],
                self.m[1][0] * other.m[0][1] + self.m[1][1] * other.m[1][1],
            ],
        ];
        let t = self.apply(other.t);
        Self { m, t }
    }

    /// Inverse of an orthogonal 2D isometry (`m^{-1} = m^T`).
    pub fn inverse(&self) -> Self {
        let mt = [
            [self.m[0][0], self.m[1][0]],
            [self.m[0][1], self.m[1][1]],
        ];
        let t = [
            -(mt[0][0] * self.t[0] + mt[0][1] * self.t[1]),
            -(mt[1][0] * self.t[0] + mt[1][1] * self.t[1]),
        ];
        Self { m: mt, t }
    }
}

/// Fractional 2D group operation `x_frac -> mf * x_frac + t` (`mf in GL_2(Z)`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FracOp2D {
    pub mf: [[i32; 2]; 2],
    pub t: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame2D {
    pub q: [[f64; 2]; 2],
    pub origin: [f64; 2],
}

impl Default for Frame2D {
    fn default() -> Self {
        Self {
            q: [[1.0, 0.0], [0.0, 1.0]],
            origin: [0.0, 0.0],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    pub group: String,
    pub sites: Vec<[f64; 2]>,
    pub bonds: Vec<[[f64; 2]; 2]>,
    #[serde(default = "default_a")]
    pub a: f64,
    #[serde(default)]
    pub b: Option<f64>,
    #[serde(default)]
    pub gamma: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConeMark {
    pub angle_deg: f64,
    pub site: Option<usize>,
    pub face: Option<usize>,
    pub class_label: Option<String>,
    pub location: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainEdge2D {
    pub p: [f64; 2],
    pub q: [f64; 2],
    pub label: String,
    pub color_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCheckReport {
    pub ok: bool,
    pub no_loops: bool,
    pub no_double_bonds: bool,
    pub plane_coordination: bool,
    pub euler: bool,
    pub faces_at_least_3: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub name: String,
    pub sites: Vec<(usize, i32, i32, usize)>,
    pub bonds: Vec<(usize, usize, [f64; 2])>,
    pub faces: Vec<Vec<usize>>,
    pub plane_degree: Vec<usize>,
    pub edge_sites: Vec<usize>,
    pub chi: i32,
    pub orientable: bool,
    pub closed: bool,
    pub cones: Vec<ConeMark>,
    pub uv: Vec<[f64; 2]>,
    pub domain_polygons: Vec<Vec<[f64; 2]>>,
    pub domain_edges: Vec<DomainEdge2D>,
    pub slits: Vec<([f64; 2], [f64; 2], String)>,
    pub xyz: Vec<[f64; 3]>,
    pub bond_curves_3d: Vec<Vec<[f64; 3]>>,
    pub cover: Option<Box<SurfaceCover3D>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCover3D {
    pub xyz: Vec<[f64; 3]>,
    pub bonds: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub site_rgba: Vec<[f32; 4]>,
    pub cone_sites: Vec<usize>,
    pub cone_faces: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectedBondData {
    pub to: usize,
    pub vec: [f64; 2],
    pub kind: usize,
    pub seam: Option<Isometry2>,
    pub ang: Option<f64>,
    pub across: bool,
    pub ghost_stub: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuiltCrystal {
    pub pos: Vec<[f64; 2]>,
    pub ref_pos: Vec<[f64; 2]>,
    pub basis_idx: Vec<usize>,
    pub side: Vec<Option<char>>,
    pub cone_sites: Vec<usize>,
    pub edges: Vec<(usize, usize)>,
    pub adj: Vec<Vec<DirectedBondData>>,
    pub wall_sigma: Option<Isometry2>,
    pub is_non_orientable: bool,
}
