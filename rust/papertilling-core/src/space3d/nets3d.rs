//! Canonical 3D crystal nets, 3D line defects (screw dislocation, pure 90 deg #123
//! disclination, forced 4_2 #136 screw-dispiration), and 80-layer slab net visualizer.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::layer_group_entry;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalNet3d {
    SimpleCubic,
    BCC,
    FCC,
    Diamond,
    HCP,
    Tetragonal123,
    Rutile136,
    Perovskite,
    Pyrochlore,
    BetaQuartz,
}

impl CanonicalNet3d {
    pub const ALL: [Self; 10] = [
        Self::SimpleCubic,
        Self::BCC,
        Self::FCC,
        Self::Diamond,
        Self::HCP,
        Self::Tetragonal123,
        Self::Rutile136,
        Self::Perovskite,
        Self::Pyrochlore,
        Self::BetaQuartz,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::SimpleCubic => "Simple Cubic (#221)",
            Self::BCC => "BCC (#229 Im-3m)",
            Self::FCC => "FCC (#225 Fm-3m)",
            Self::Diamond => "Diamond (#227 Fd-3m)",
            Self::HCP => "HCP (#194 P6_3/mmc)",
            Self::Tetragonal123 => "Tetragonal (#123 P4/mmm)",
            Self::Rutile136 => "Rutile (#136 P4_2/mnm)",
            Self::Perovskite => "Perovskite (#221 Pm-3m)",
            Self::Pyrochlore => "Pyrochlore (#227 Fd-3m)",
            Self::BetaQuartz => "beta-Quartz (#180 P6_222)",
        }
    }

    pub fn space_group_num(self) -> u16 {
        match self {
            Self::SimpleCubic | Self::Perovskite => 221,
            Self::BCC => 229,
            Self::FCC => 225,
            Self::Diamond | Self::Pyrochlore => 227,
            Self::HCP => 194,
            Self::Tetragonal123 => 123,
            Self::Rutile136 => 136,
            Self::BetaQuartz => 180,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineDefect3d {
    None,
    ScrewDislocationZ,
    PureDisclination90_123,
    ScrewDispiration42_136,
}

impl LineDefect3d {
    pub const ALL: [Self; 4] = [
        Self::None,
        Self::ScrewDislocationZ,
        Self::PureDisclination90_123,
        Self::ScrewDispiration42_136,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "Pristine 3D Net",
            Self::ScrewDislocationZ => "Screw Dislocation (b = c e_z)",
            Self::PureDisclination90_123 => "Pure 90° Wedge (#123, Δz = 0)",
            Self::ScrewDispiration42_136 => "Forced 4_2 Screw-Dispiration (#136, Δz = c/2)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Net3dPatch {
    pub name: String,
    pub space_group_num: u16,
    pub defect_label: String,
    pub positions: Vec<[f64; 3]>,
    pub bonds: Vec<(usize, usize, u8)>,
    pub site_kinds: Vec<u8>,
    pub defect_axis: Option<([f64; 3], [f64; 3])>,
    pub info: String,
}

type NetMotif = ([[f64; 3]; 3], Vec<([f64; 3], u8)>, f64);

fn net_motif(net: CanonicalNet3d) -> NetMotif {
    let sc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let hex = [[1.0, 0.0, 0.0], [-0.5, 0.8660254, 0.0], [0.0, 0.0, 1.35]];
    match net {
        CanonicalNet3d::SimpleCubic => (sc, vec![([0.0, 0.0, 0.0], 0)], 1.05),
        CanonicalNet3d::BCC => (sc, vec![([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)], 0.92),
        CanonicalNet3d::FCC => (
            sc,
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.0], 0),
                ([0.5, 0.0, 0.5], 0),
                ([0.0, 0.5, 0.5], 0),
            ],
            0.76,
        ),
        CanonicalNet3d::Diamond => (
            sc,
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.0], 0),
                ([0.5, 0.0, 0.5], 0),
                ([0.0, 0.5, 0.5], 0),
                ([0.25, 0.25, 0.25], 1),
                ([0.75, 0.75, 0.25], 1),
                ([0.75, 0.25, 0.75], 1),
                ([0.25, 0.75, 0.75], 1),
            ],
            0.48,
        ),
        CanonicalNet3d::HCP => (
            hex,
            vec![([1.0 / 3.0, 2.0 / 3.0, 0.25], 0), ([2.0 / 3.0, 1.0 / 3.0, 0.75], 0)],
            1.02,
        ),
        CanonicalNet3d::Tetragonal123 => (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.15]],
            vec![([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)],
            0.96,
        ),
        CanonicalNet3d::Rutile136 => (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.72]],
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.5], 0),
                ([0.3, 0.3, 0.0], 1),
                ([0.7, 0.7, 0.0], 1),
                ([0.8, 0.2, 0.5], 1),
                ([0.2, 0.8, 0.5], 1),
            ],
            0.48,
        ),
        CanonicalNet3d::Perovskite => (
            sc,
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.5], 1),
                ([0.5, 0.0, 0.0], 2),
                ([0.0, 0.5, 0.0], 2),
                ([0.0, 0.0, 0.5], 2),
            ],
            0.55,
        ),
        CanonicalNet3d::Pyrochlore => (
            sc,
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.25, 0.25, 0.0], 0),
                ([0.25, 0.0, 0.25], 0),
                ([0.0, 0.25, 0.25], 0),
                ([0.5, 0.5, 0.5], 1),
            ],
            0.42,
        ),
        CanonicalNet3d::BetaQuartz => (
            hex,
            vec![
                ([0.5, 0.0, 0.0], 0),
                ([0.0, 0.5, 1.0 / 3.0], 0),
                ([0.5, 0.5, 2.0 / 3.0], 0),
            ],
            0.72,
        ),
    }
}

/// Build a 3D ball-and-stick patch for `net` with optional 3D topological line defect along `z`.
pub fn build_net_3d(net: CanonicalNet3d, defect: LineDefect3d) -> Net3dPatch {
    let (lat, motif, bond_cutoff) = net_motif(net);
    let nx = if defect == LineDefect3d::None { 2 } else { 3 };
    let ny = nx;
    let nz = 2;

    let mut raw_pos = Vec::new();
    let mut raw_kinds = Vec::new();
    let mid = 0.5 * (nx as f64);
    let mid_z = 0.5 * (nz as f64) * lat[2][2];

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for &(f, k) in &motif {
                    let fx = ix as f64 + f[0];
                    let fy = iy as f64 + f[1];
                    let fz = iz as f64 + f[2];
                    let p = [
                        fx * lat[0][0] + fy * lat[1][0] + fz * lat[2][0] - mid * lat[0][0],
                        fx * lat[0][1] + fy * lat[1][1] + fz * lat[2][1] - mid * lat[1][1],
                        fx * lat[0][2] + fy * lat[1][2] + fz * lat[2][2] - mid_z,
                    ];
                    raw_pos.push(p);
                    raw_kinds.push(k);
                }
            }
        }
    }

    let mut raw_bonds = Vec::new();
    let c2 = bond_cutoff * bond_cutoff;
    for i in 0..raw_pos.len() {
        for j in (i + 1)..raw_pos.len() {
            let dx = raw_pos[j][0] - raw_pos[i][0];
            let dy = raw_pos[j][1] - raw_pos[i][1];
            let dz = raw_pos[j][2] - raw_pos[i][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > 1e-4 && d2 <= c2 {
                raw_bonds.push((i, j));
            }
        }
    }

    let c_len = lat[2][2];
    let mut positions = Vec::new();
    let mut site_kinds = Vec::new();
    let mut map_idx = vec![None; raw_pos.len()];
    let cut_axis = matches!(defect, LineDefect3d::PureDisclination90_123 | LineDefect3d::ScrewDispiration42_136);

    for (idx, &p) in raw_pos.iter().enumerate() {
        let th = p[1].atan2(p[0]);
        if cut_axis && th.abs() < 0.24 * PI {
            continue;
        }
        let next_p = match defect {
            LineDefect3d::None => p,
            LineDefect3d::ScrewDislocationZ => [p[0], p[1], p[2] + c_len * th / (2.0 * PI)],
            LineDefect3d::PureDisclination90_123 | LineDefect3d::ScrewDispiration42_136 => {
                let r = p[0].hypot(p[1]);
                let alpha = if th >= 0.0 { th - 0.25 * PI } else { th + 2.0 * PI - 0.25 * PI };
                let u = (alpha / (1.5 * PI)).clamp(0.0, 1.0);
                let th_new = 2.0 * PI * u;
                let dz = if defect == LineDefect3d::ScrewDispiration42_136 {
                    0.5 * c_len * u
                } else {
                    0.0
                };
                [r * th_new.cos(), r * th_new.sin(), p[2] + dz]
            }
        };
        map_idx[idx] = Some(positions.len());
        positions.push(next_p);
        site_kinds.push(raw_kinds[idx]);
    }

    let mut bonds = Vec::new();
    for (i, j) in raw_bonds {
        if let (Some(u), Some(v)) = (map_idx[i], map_idx[j]) {
            let len = ((positions[v][0] - positions[u][0]).powi(2)
                + (positions[v][1] - positions[u][1]).powi(2)
                + (positions[v][2] - positions[u][2]).powi(2))
            .sqrt();
            if len < 1.85 * bond_cutoff {
                let seam_b = (defect != LineDefect3d::None) && len > 1.28 * bond_cutoff;
                bonds.push((u, v, if seam_b { 1_u8 } else { 0_u8 }));
            }
        }
    }
    let defect_axis = (defect != LineDefect3d::None).then_some(([0.0, 0.0, -mid_z - 0.4], [0.0, 0.0, mid_z + 0.6]));
    let info = format!("{} | {} | {} sites, {} bonds", net.label(), defect.label(), positions.len(), bonds.len());
    Net3dPatch {
        name: net.label().to_string(),
        space_group_num: net.space_group_num(),
        defect_label: defect.label().to_string(),
        positions,
        bonds,
        site_kinds,
        defect_axis,
        info,
    }
}

/// Build a 3D slab patch for any of the 80 layer groups (`num = 1..=80`).
pub fn build_layer_slab_patch(num: u8) -> Result<Net3dPatch, String> {
    let e = layer_group_entry(num)?;
    let (a1, a2): ([f64; 2], [f64; 2]) = match e.system.as_str() {
        "oblique" => ([1.0, 0.0], [0.22, 1.24]),
        "rectangular" => ([1.0, 0.0], [0.0, 1.32]),
        "square" => ([1.0, 0.0], [0.0, 1.0]),
        _ => ([1.0, 0.0], [-0.5, 0.8660254]),
    };
    let seed = [0.17_f64, 0.29, 0.18];
    let mut cell_pts: Vec<[f64; 3]> = Vec::new();
    for op in &e.ops {
        let fx = (op.r[0][0] as f64 * seed[0] + op.r[0][1] as f64 * seed[1] + op.r[0][2] as f64 * seed[2]
            + (op.t2[0][0] as f64) / (op.t2[0][1] as f64))
            .rem_euclid(1.0);
        let fy = (op.r[1][0] as f64 * seed[0] + op.r[1][1] as f64 * seed[1] + op.r[1][2] as f64 * seed[2]
            + (op.t2[1][0] as f64) / (op.t2[1][1] as f64))
            .rem_euclid(1.0);
        let fz = op.r[2][0] as f64 * seed[0] + op.r[2][1] as f64 * seed[1] + op.r[2][2] as f64 * seed[2];
        let shifts: &[[f64; 2]] = if e.centred { &[[0.0, 0.0], [0.5, 0.5]] } else { &[[0.0, 0.0]] };
        for &[sx, sy] in shifts {
            let p_c = [(fx + sx).rem_euclid(1.0), (fy + sy).rem_euclid(1.0), fz];
            if !cell_pts.iter().any(|&p| (p[0] - p_c[0]).hypot(p[1] - p_c[1]) < 1e-3 && (p[2] - p_c[2]).abs() < 1e-3) {
                cell_pts.push(p_c);
            }
        }
    }
    let mut positions = Vec::new();
    let mut site_kinds = Vec::new();
    for ix in -1_i32..=0 {
        for iy in -1_i32..=0 {
            for &p in &cell_pts {
                let fx = p[0] + ix as f64;
                let fy = p[1] + iy as f64;
                positions.push([fx * a1[0] + fy * a2[0], fx * a1[1] + fy * a2[1], p[2]]);
                site_kinds.push(u8::from(p[2] < 0.0));
            }
        }
    }
    let mut bonds = Vec::new();
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let d = ((positions[j][0] - positions[i][0]).powi(2)
                + (positions[j][1] - positions[i][1]).powi(2)
                + (positions[j][2] - positions[i][2]).powi(2))
            .sqrt();
            if d > 0.05 && d < 0.88 {
                bonds.push((i, j, 0_u8));
            }
        }
    }
    Ok(Net3dPatch {
        name: format!("Layer #{} {} ({})", e.number, e.symbol, e.pointgroup),
        space_group_num: u16::from(e.number),
        defect_label: format!("System: {}, |P|={}", e.system, e.point_order),
        positions,
        bonds,
        site_kinds,
        defect_axis: None,
        info: format!("Layer #{} {}: {} atoms in 2x2 slab", e.number, e.symbol, cell_pts.len() * 4),
    })
}
