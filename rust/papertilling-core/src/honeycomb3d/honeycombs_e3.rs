//! Euclidean honeycombs: cubic `{4,3,4}` (SC Voronoi), bitruncated cubic (Kelvin, BCC Voronoi),
//! rhombic dodecahedral (FCC Voronoi), and the alternated cubic (octet) tetrahedra + octahedra.
//! Patches are `shells` face-adjacency shells of cells around the central cell.

use std::collections::{HashMap, VecDeque};

use serde::Serialize;

use super::voronoi3d_dirichlet::{lattice_basis, lattice_voronoi_cell};
use super::{CellComplex3d, ComplexBuilder, PointIndex};

pub const MAX_E3_SHELLS: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum EuclideanKind {
    Cubic,
    BitruncatedCubic,
    RhombicDodecahedral,
    AlternatedCubic,
}

impl EuclideanKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cubic" => Ok(Self::Cubic),
            "bitruncated_cubic" => Ok(Self::BitruncatedCubic),
            "rhombic_dodecahedral" => Ok(Self::RhombicDodecahedral),
            "alternated_cubic" => Ok(Self::AlternatedCubic),
            _ => Err(format!("unknown Euclidean honeycomb {s:?} (cubic, bitruncated_cubic, rhombic_dodecahedral, alternated_cubic)")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cubic => "cubic {4,3,4} (SC Voronoi cells)",
            Self::BitruncatedCubic => "bitruncated cubic (Kelvin; BCC Voronoi truncated octahedra)",
            Self::RhombicDodecahedral => "rhombic dodecahedral (FCC Voronoi cells)",
            Self::AlternatedCubic => "alternated cubic (octet: tetrahedra + octahedra)",
        }
    }
}

type RawCell = Vec<Vec<[f64; 3]>>;

fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn voronoi_cells(lattice: &str, k: i32) -> Result<(Vec<RawCell>, usize), String> {
    let cell = lattice_voronoi_cell(lattice)?;
    let basis = lattice_basis(lattice)?;
    let mut out = Vec::new();
    let mut centre = 0;
    for a in -k..=k {
        for b in -k..=k {
            for c in -k..=k {
                if (a, b, c) == (0, 0, 0) {
                    centre = out.len();
                }
                let (fa, fb, fc) = (a as f64, b as f64, c as f64);
                let t = [0, 1, 2].map(|i| fa * basis[0][i] + fb * basis[1][i] + fc * basis[2][i]);
                out.push(cell.faces.iter().map(|f| f.iter().map(|&v| add3(&cell.vertices[v], &t)).collect()).collect());
            }
        }
    }
    Ok((out, centre))
}

/// Octet honeycomb with vertices at odd-sum integer points: octahedra centred at even-sum points,
/// tetrahedra centred at half-integer points (the odd-sum corners of the unit cube around them).
fn octet_cells(k: i32) -> (Vec<RawCell>, usize) {
    let mut out = Vec::new();
    let mut centre = 0;
    let signs = [-1.0, 1.0];
    for a in -k..=k {
        for b in -k..=k {
            for c in -k..=k {
                let o = [a as f64, b as f64, c as f64];
                if (a + b + c).rem_euclid(2) == 0 {
                    if (a, b, c) == (0, 0, 0) {
                        centre = out.len();
                    }
                    let mut faces = Vec::new();
                    for sx in signs {
                        for sy in signs {
                            for sz in signs {
                                faces.push(vec![add3(&o, &[sx, 0.0, 0.0]), add3(&o, &[0.0, sy, 0.0]), add3(&o, &[0.0, 0.0, sz])]);
                            }
                        }
                    }
                    out.push(faces);
                }
                let h = [o[0] + 0.5, o[1] + 0.5, o[2] + 0.5];
                let mut corners = Vec::new();
                for sx in signs {
                    for sy in signs {
                        for sz in signs {
                            let x = add3(&h, &[0.5 * sx, 0.5 * sy, 0.5 * sz]);
                            if (x[0] + x[1] + x[2]).round().rem_euclid(2.0) == 1.0 {
                                corners.push(x);
                            }
                        }
                    }
                }
                let faces = (0..4).map(|skip| (0..4).filter(|&i| i != skip).map(|i| corners[i]).collect()).collect();
                out.push(faces);
            }
        }
    }
    (out, centre)
}

/// Build a patch of a Euclidean honeycomb (`shells` in `0..=3`).
pub fn build_euclidean_honeycomb(kind: EuclideanKind, shells: u32) -> Result<CellComplex3d, String> {
    if shells > MAX_E3_SHELLS {
        return Err(format!("shells = {shells} exceeds the maximum {MAX_E3_SHELLS}"));
    }
    let k = shells as i32 + 1;
    let (raw, centre) = match kind {
        EuclideanKind::Cubic => voronoi_cells("SC", k)?,
        EuclideanKind::BitruncatedCubic => voronoi_cells("BCC", k)?,
        EuclideanKind::RhombicDodecahedral => voronoi_cells("FCC", k)?,
        EuclideanKind::AlternatedCubic => octet_cells(k),
    };
    // Global vertex ids and face keys, then face adjacency between cells.
    let mut index = PointIndex::<3>::new(1e-7);
    let cells_ids: Vec<Vec<Vec<usize>>> =
        raw.iter().map(|c| c.iter().map(|f| f.iter().map(|x| index.insert(*x).0).collect()).collect()).collect();
    let mut face_cells: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (ci, c) in cells_ids.iter().enumerate() {
        for f in c {
            let mut key = f.clone();
            key.sort_unstable();
            face_cells.entry(key).or_default().push(ci);
        }
    }
    let mut shell_of: HashMap<usize, u32> = HashMap::from([(centre, 0)]);
    let mut order = vec![centre];
    let mut queue = VecDeque::from([centre]);
    while let Some(ci) = queue.pop_front() {
        let s = shell_of[&ci];
        if s >= shells {
            continue;
        }
        for f in &cells_ids[ci] {
            let mut key = f.clone();
            key.sort_unstable();
            for &cj in &face_cells[&key] {
                if let std::collections::hash_map::Entry::Vacant(e) = shell_of.entry(cj) {
                    e.insert(s + 1);
                    order.push(cj);
                    queue.push_back(cj);
                }
            }
        }
    }
    let mut b = ComplexBuilder::new(kind.label(), "EuclideanE3");
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let all = index.points().to_vec();
    for &ci in &order {
        let mut faces = Vec::new();
        for f in &cells_ids[ci] {
            let cyc = f
                .iter()
                .map(|&g| {
                    *remap.entry(g).or_insert_with(|| {
                        let x = all[g];
                        b.out.points.push(x);
                        b.out.coords4.push([x[0], x[1], x[2], 1.0]);
                        b.out.is_ideal.push(false);
                        b.out.points.len() - 1
                    })
                })
                .collect();
            faces.push(b.add_face(cyc));
        }
        b.add_cell(faces, shell_of[&ci], false);
    }
    Ok(b.finish())
}
