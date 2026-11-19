//! Labyrinth (skeletal) nets of the Schwarz P, Schwarz D and Schoen gyroid G surfaces, built
//! clean-room from classical crystallography (no external net database):
//!
//! * P: primitive cubic net, vertices at `(0,0,0)` (Pm-3m #221); the second labyrinth is the
//!   same net shifted by `(1/2,1/2,1/2)` (pair: Im-3m #229).
//! * D: diamond net, FCC + FCC+(1/4,1/4,1/4) (Fd-3m #227); second labyrinth shifted by
//!   `(1/2,1/2,1/2)` (pair: Pn-3m #224).
//! * G: the chiral 3-connected net on the 8a sites of I4_132 (#214), `(1/8,1/8,1/8)`,
//!   `(3/8,7/8,5/8)`, `(7/8,5/8,3/8)`, `(5/8,3/8,7/8)` + body centring; the second labyrinth is its
//!   mirror image `x ↦ -x` (pair: Ia-3d #230).
//!
//! Provenance: site coordinates are International Tables for Crystallography Vol. A Wyckoff
//! positions (221 1a; 227 8a; 214 8a), i.e. published crystallographic facts; the net names
//! (pcu, dia, srs) are used only as labels. Nodal approximations: H. G. von Schnering, R. Nesper,
//! "Nodal surfaces of Fourier series", Z. Phys. B 83 (1991) 407-412. No EPINET / RCSR data is used.
//!
//! Coordinates are fractional in the conventional cubic cell; edges join nearest neighbours of
//! the same labyrinth and carry the lattice offset of their second end.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TpmsKind {
    P,
    D,
    G,
}

impl TpmsKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "P" | "p" => Ok(Self::P),
            "D" | "d" => Ok(Self::D),
            "G" | "g" => Ok(Self::G),
            _ => Err(format!("unknown triply periodic surface {s:?} (P, D, G)")),
        }
    }

    /// Classical nodal approximation (level set 0), with `X = 2πx` etc.
    pub fn nodal(self, x: &[f64; 3]) -> f64 {
        let t = x.map(|c| 2.0 * std::f64::consts::PI * c);
        match self {
            Self::P => t[0].cos() + t[1].cos() + t[2].cos(),
            Self::D => t[0].cos() * t[1].cos() * t[2].cos() + t[0].sin() * t[1].sin() * t[2].sin(),
            Self::G => t[0].sin() * t[1].cos() + t[1].sin() * t[2].cos() + t[2].sin() * t[0].cos(),
        }
    }

    /// Number of conventional-cell sublattices that preserve the two sides (P: 1, D: 4 (FCC), G: 2 (BCC)).
    pub fn side_preserving_cells(self) -> u32 {
        match self {
            Self::P => 1,
            Self::D => 4,
            Self::G => 2,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TriplyPeriodicNet {
    pub name: String,
    pub kind: TpmsKind,
    pub space_group: String,
    pub space_group_number: u32,
    pub labyrinths: u32,
    pub positions: Vec<[f64; 3]>,
    pub labyrinth_of: Vec<u32>,
    /// `(i, j, offset)`: site `i` joined to site `j` translated by `offset` (conventional cells).
    pub edges: Vec<(usize, usize, [i32; 3])>,
    pub coordination: Vec<usize>,
    pub bond_length: f64,
}

fn wrap(x: [f64; 3]) -> [f64; 3] {
    x.map(|c| {
        let w = c.rem_euclid(1.0);
        if (w - 1.0).abs() < 1e-12 { 0.0 } else { w }
    })
}

fn single_labyrinth(kind: TpmsKind) -> Vec<[f64; 3]> {
    let fcc = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];
    match kind {
        TpmsKind::P => vec![[0.0; 3]],
        TpmsKind::D => fcc.iter().copied().chain(fcc.iter().map(|f| f.map(|c| c + 0.25))).map(wrap).collect(),
        TpmsKind::G => {
            let base = [[1.0, 1.0, 1.0], [3.0, 7.0, 5.0], [7.0, 5.0, 3.0], [5.0, 3.0, 7.0]].map(|v| v.map(|c| c / 8.0));
            base.iter().copied().chain(base.iter().map(|v| v.map(|c| c + 0.5))).map(wrap).collect()
        }
    }
}

/// Build the labyrinth net of `kind` (`labyrinths` = 1 for one labyrinth, 2 for both).
pub fn tpms_net(kind: TpmsKind, labyrinths: u32) -> Result<TriplyPeriodicNet, String> {
    if !(1..=2).contains(&labyrinths) {
        return Err("labyrinths must be 1 or 2".into());
    }
    let a = single_labyrinth(kind);
    let mut positions = a.clone();
    let mut labyrinth_of = vec![0u32; a.len()];
    if labyrinths == 2 {
        let second: Vec<[f64; 3]> = match kind {
            TpmsKind::G => a.iter().map(|v| wrap(v.map(|c| -c))).collect(),
            _ => a.iter().map(|v| wrap(v.map(|c| c + 0.5))).collect(),
        };
        positions.extend(second);
        labyrinth_of.resize(2 * a.len(), 1);
    }
    let (name, sg, num) = match (kind, labyrinths) {
        (TpmsKind::P, 1) => ("pcu (Schwarz P labyrinth)", "Pm-3m", 221),
        (TpmsKind::P, _) => ("pcu pair (both Schwarz P labyrinths)", "Im-3m", 229),
        (TpmsKind::D, 1) => ("dia (Schwarz D labyrinth)", "Fd-3m", 227),
        (TpmsKind::D, _) => ("double diamond (both Schwarz D labyrinths)", "Pn-3m", 224),
        (TpmsKind::G, 1) => ("srs (gyroid labyrinth, chiral)", "I4_132", 214),
        (TpmsKind::G, _) => ("double gyroid (both gyroid labyrinths)", "Ia-3d", 230),
    };
    let n = positions.len();
    let mut cands: Vec<(usize, usize, [i32; 3], f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if labyrinth_of[i] != labyrinth_of[j] {
                continue;
            }
            for ox in -1..=1 {
                for oy in -1..=1 {
                    for oz in -1..=1 {
                        let off = [ox, oy, oz];
                        if i == j && off == [0, 0, 0] {
                            continue;
                        }
                        let d: f64 = (0..3).map(|k| (positions[j][k] + off[k] as f64 - positions[i][k]).powi(2)).sum::<f64>().sqrt();
                        cands.push((i, j, off, d));
                    }
                }
            }
        }
    }
    let dmin = cands.iter().map(|c| c.3).fold(f64::INFINITY, f64::min);
    let mut edges = Vec::new();
    let mut coordination = vec![0usize; n];
    for (i, j, off, d) in cands {
        if d > dmin * (1.0 + 1e-6) {
            continue;
        }
        coordination[i] += 1;
        if i < j || (i == j && off > [0, 0, 0]) {
            edges.push((i, j, off));
        }
    }
    Ok(TriplyPeriodicNet {
        name: name.into(),
        kind,
        space_group: sg.into(),
        space_group_number: num,
        labyrinths,
        positions,
        labyrinth_of,
        edges,
        coordination,
        bond_length: dmin,
    })
}
