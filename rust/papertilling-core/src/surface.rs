//! Geometry-agnostic "pieces + pairings" surface types and topological invariants.

use serde::{Deserialize, Serialize};

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
    Impossible {
        reason: String,
    },
    NotFound {
        reason: String,
        hint: Option<String>,
    },
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
