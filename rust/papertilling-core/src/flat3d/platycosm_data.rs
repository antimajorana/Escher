//! Metadata table, ITA operation expansion, and exact CW 3-complex
//! Euler-characteristic (`chi_3 = V - E + F - C = 0`) verification for the
//! 10 closed flat 3D Conway-Rossetti platycosms.

use std::collections::{BTreeSet, HashSet};

use serde::Serialize;

use crate::space3d::space_group_3d_entry;

/// Metadata for one of the 10 Conway-Rossetti closed flat 3D platycosms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PlatycosmInfo {
    pub code: &'static str,
    pub name: &'static str,
    pub orientable: bool,
    pub ita_numbers: &'static [u16],
    pub ita_symbols: &'static [&'static str],
    pub point_order: usize,
    pub crystal_system: &'static str,
    pub h1_label: &'static str,
    pub b1: usize,
    pub h1_torsion: &'static [u32],
}

macro_rules! p_info {
    ($c:expr, $nm:expr, $or:expr, $itas:expr, $syms:expr, $po:expr, $sys:expr, $h1:expr, $b1:expr, $tor:expr) => {
        PlatycosmInfo {
            code: $c,
            name: $nm,
            orientable: $or,
            ita_numbers: $itas,
            ita_symbols: $syms,
            point_order: $po,
            crystal_system: $sys,
            h1_label: $h1,
            b1: $b1,
            h1_torsion: $tor,
        }
    };
}

/// Complete table of all 10 closed flat 3D platycosms covering all 13 torsion-free 3D ITA groups.
///
/// Published facts (provenance): codes, names and `H_1` from J. H. Conway & J. P. Rossetti,
/// "Describing the platycosms", arXiv:math/0311476 (2003), table of the ten platycosms; the same
/// `H_1` list appears in J. A. Wolf, *Spaces of Constant Curvature* (Sec. 3.5; theorem locator to verify, classes
/// `G1..G6`, `B1..B4`). ITA numbers: Hahn (ed.), International Tables for Crystallography Vol. A:
/// P1 #1, P2_1 #4, P3_1/P3_2 #144/#145, P4_1/P4_3 #76/#78, P6_1/P6_5 #169/#170, P2_12_12_1 #19,
/// Pc #7, Cc #9, Pca2_1 #29, Pna2_1 #33 (13 ITA types = 10 affine classes + 3 enantiomorphic pairs).
/// `H_1` is also re-derived independently as `G^ab` by the Bieberbach analyzer in the tests.
pub const PLATYCOSMS: [PlatycosmInfo; 10] = [
    p_info!("c1", "torocosm", true, &[1], &["P1"], 1, "triclinic", "Z^3", 3, &[]),
    p_info!("c2", "dicosm", true, &[4], &["P2_1"], 2, "monoclinic", "Z_2^2 x Z", 1, &[2, 2]),
    p_info!("c3", "tricosm", true, &[144, 145], &["P3_1", "P3_2"], 3, "trigonal", "Z_3 x Z", 1, &[3]),
    p_info!("c4", "tetracosm", true, &[76, 78], &["P4_1", "P4_3"], 4, "tetragonal", "Z_2 x Z", 1, &[2]),
    p_info!("c6", "hexacosm", true, &[169, 170], &["P6_1", "P6_5"], 6, "hexagonal", "Z", 1, &[]),
    p_info!("c22", "didicosm", true, &[19], &["P2_12_12_1"], 4, "orthorhombic", "Z_4^2", 0, &[4, 4]),
    p_info!("+a1", "first amphicosm", false, &[7], &["Pc"], 2, "monoclinic", "Z_2 x Z^2", 2, &[2]),
    p_info!("-a1", "second amphicosm", false, &[9], &["Cc"], 2, "monoclinic", "Z^2", 2, &[]),
    p_info!("+a2", "first amphidicosm", false, &[29], &["Pca2_1"], 4, "orthorhombic", "Z_2^2 x Z", 1, &[2, 2]),
    p_info!("-a2", "second amphidicosm", false, &[33], &["Pna2_1"], 4, "orthorhombic", "Z_4 x Z", 1, &[4]),
];

/// Return all 10 Conway-Rossetti platycosms.
#[must_use]
pub fn list_platycosms() -> &'static [PlatycosmInfo] {
    &PLATYCOSMS
}

/// Wolf's names (`G1..G6` orientable, `B1..B4` non-orientable) for the Conway-Rossetti codes.
pub const WOLF_ALIASES: [(&str, &str); 10] = [
    ("g1", "c1"),
    ("g2", "c2"),
    ("g3", "c3"),
    ("g4", "c4"),
    ("g5", "c6"),
    ("g6", "c22"),
    ("b1", "+a1"),
    ("b2", "-a1"),
    ("b3", "+a2"),
    ("b4", "-a2"),
];

/// Look up a platycosm by Conway code (`"c1".."c22"`, `"+a1".."-a2"`), Wolf name (`"G1".."B4"`),
/// name, or ITA symbol/number.
///
/// # Errors
/// Returns an error if the query matches no platycosm.
pub fn lookup_platycosm(query: &str) -> Result<&'static PlatycosmInfo, String> {
    let q = query.trim();
    let lower = q.to_lowercase();
    let ql = WOLF_ALIASES.iter().find(|(w, _)| *w == lower).map_or(lower.as_str(), |(_, c)| *c);
    if let Some(p) = PLATYCOSMS.iter().find(|p| p.code == ql || p.name == ql) {
        return Ok(p);
    }
    if let Ok(num) = q.parse::<u16>()
        && let Some(p) = PLATYCOSMS.iter().find(|p| p.ita_numbers.contains(&num))
    {
        return Ok(p);
    }
    let sym_clean: String = ql.chars().filter(|c| *c != '_').collect();
    PLATYCOSMS
        .iter()
        .find(|p| p.ita_symbols.iter().any(|s| s.to_lowercase().replace('_', "") == sym_clean))
        .ok_or_else(|| format!("unknown platycosm {query:?}"))
}

#[derive(Clone, Copy)]
pub(crate) struct PlatyOp {
    pub r: [[i32; 3]; 3],
    pub t: [f64; 3],
    pub t_num: [i32; 3],
    pub t_den: i32,
}

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

pub(crate) fn ops_for_ita(ita_num: u16, axis_mode: usize) -> Result<Vec<PlatyOp>, String> {
    if ita_num == 4 && axis_mode > 0 {
        let id = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let (r2, t_num) = if axis_mode == 1 {
            ([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, 1])
        } else {
            ([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0])
        };
        let t = [0.5 * f64::from(t_num[0]), 0.5 * f64::from(t_num[1]), 0.5 * f64::from(t_num[2])];
        return Ok(vec![
            PlatyOp { r: id, t: [0.0; 3], t_num: [0; 3], t_den: 1 },
            PlatyOp { r: r2, t, t_num, t_den: 2 },
        ]);
    }
    let e = space_group_3d_entry(ita_num)?;
    let mut ops = Vec::with_capacity(e.ops.len() * 2);
    for op in &e.ops {
        let r = [
            [op.r[0][0] as i32, op.r[0][1] as i32, op.r[0][2] as i32],
            [op.r[1][0] as i32, op.r[1][1] as i32, op.r[1][2] as i32],
            [op.r[2][0] as i32, op.r[2][1] as i32, op.r[2][2] as i32],
        ];
        let mut den = 1_i32;
        for &[_, d] in &op.t {
            den = (den * (d as i32)) / gcd_i32(den, d as i32);
        }
        let t_num = [
            (op.t[0][0] as i32) * (den / op.t[0][1] as i32),
            (op.t[1][0] as i32) * (den / op.t[1][1] as i32),
            (op.t[2][0] as i32) * (den / op.t[2][1] as i32),
        ];
        let t = [
            (op.t[0][0] as f64) / (op.t[0][1] as f64),
            (op.t[1][0] as f64) / (op.t[1][1] as f64),
            (op.t[2][0] as f64) / (op.t[2][1] as f64),
        ];
        ops.push(PlatyOp { r, t, t_num, t_den: den });
    }
    if e.centring == "C" {
        let base = ops.clone();
        for op in base {
            let lcm = (op.t_den * 2) / gcd_i32(op.t_den, 2);
            let (s, h) = (lcm / op.t_den, lcm / 2);
            ops.push(PlatyOp {
                r: op.r,
                t: [(op.t[0] + 0.5).rem_euclid(1.0), (op.t[1] + 0.5).rem_euclid(1.0), op.t[2]],
                t_num: [
                    (op.t_num[0] * s + h).rem_euclid(lcm),
                    (op.t_num[1] * s + h).rem_euclid(lcm),
                    (op.t_num[2] * s).rem_euclid(lcm),
                ],
                t_den: lcm,
            });
        }
    }
    Ok(ops)
}

/// Count the cells of the quotient CW 3-complex `R^3 / G` induced by a `G`-invariant cellulation of
/// the conventional torus `R^3 / Z^3` (cubical grid of fineness `k`; for `c3`/`c6` the triangular-prism
/// grid), checking that `G / Z^3` acts freely on cells and that `chi = V - E + F - C = 0` (every closed
/// odd-dimensional manifold has `chi = 0`, Poincare duality).
pub(crate) fn verify_cw_euler_zero(
    code: &str,
    ops: &[PlatyOp],
    l_sup: [usize; 3],
) -> Result<(usize, usize, usize, usize), String> {
    let hex = matches!(code, "c3" | "c6");
    let mult = if hex { 6_i32 } else { 2_i32 };
    let (k1, k2, k3) = if hex {
        let d = if code == "c6" { 6 } else { 3 };
        (l_sup[0].max(1) as i32, l_sup[1].max(1) as i32, (l_sup[2].max(d).div_ceil(d) * d) as i32)
    } else {
        let d = match code { "c1" => 1, "c4" => 4, _ => 2 };
        (
            (l_sup[0].max(d).div_ceil(d) * d) as i32,
            (l_sup[1].max(d).div_ceil(d) * d) as i32,
            (l_sup[2].max(d).div_ceil(d) * d) as i32,
        )
    };
    let (m1, m2, m3) = (k1 * mult, k2 * mult, k3 * mult);
    let mut dims_cells: [BTreeSet<[i32; 3]>; 4] = Default::default();
    for i in 0..k1 {
        for j in 0..k2 {
            for k in 0..k3 {
                let (bx, by, bz) = (i * mult, j * mult, k * mult);
                if hex {
                    dims_cells[0].insert([bx, by, bz]);
                    for o in [[bx + 3, by, bz], [bx, by + 3, bz], [bx + 3, by + 3, bz], [bx, by, bz + 3]] {
                        dims_cells[1].insert(o);
                    }
                    for o in [[bx + 4, by + 2, bz], [bx + 2, by + 4, bz], [bx + 3, by, bz + 3], [bx, by + 3, bz + 3], [bx + 3, by + 3, bz + 3]] {
                        dims_cells[2].insert(o);
                    }
                    dims_cells[3].insert([bx + 4, by + 2, bz + 3]);
                    dims_cells[3].insert([bx + 2, by + 4, bz + 3]);
                } else {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            for dz in 0..2 {
                                dims_cells[(dx + dy + dz) as usize].insert([bx + dx, by + dy, bz + dz]);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut counts = [0usize; 4];
    for (dim_idx, cells) in dims_cells.iter().enumerate() {
        let mut seen = HashSet::new();
        for &c in cells {
            let mut orbit = BTreeSet::new();
            for op in ops {
                let (tx, ty, tz) = ((op.t_num[0] * m1) / op.t_den, (op.t_num[1] * m2) / op.t_den, (op.t_num[2] * m3) / op.t_den);
                orbit.insert([
                    (op.r[0][0] * c[0] + op.r[0][1] * c[1] + op.r[0][2] * c[2] + tx).rem_euclid(m1),
                    (op.r[1][0] * c[0] + op.r[1][1] * c[1] + op.r[1][2] * c[2] + ty).rem_euclid(m2),
                    (op.r[2][0] * c[0] + op.r[2][1] * c[1] + op.r[2][2] * c[2] + tz).rem_euclid(m3),
                ]);
            }
            if orbit.len() != ops.len() {
                return Err(format!("platycosm CW cellularization failed freeness at dim={dim_idx}"));
            }
            if let Some(&rep) = orbit.iter().next()
                && seen.insert(rep)
            {
                counts[dim_idx] += 1;
            }
        }
    }
    let chi = counts[0] as i64 - counts[1] as i64 + counts[2] as i64 - counts[3] as i64;
    if chi != 0 {
        return Err(format!("non-zero 3-complex Euler characteristic {chi} != 0"));
    }
    Ok((counts[0], counts[1], counts[2], counts[3]))
}
