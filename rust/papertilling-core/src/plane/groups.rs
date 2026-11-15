//! The 17 Euclidean plane (wallpaper) groups, conventional/primitive settings,
//! 72-row International Tables (ITA) Wyckoff positions, and `G+` proper rotation-centre classes.

use serde::{Deserialize, Serialize};

pub use crate::plane::groups_data::CrystalSystem;
use crate::plane::groups_data::{GROUP_ALIASES, GROUP_SPECS, RawGroupSpec};
use crate::types::FracOp2D;

/// Canonical ITA short Hermann-Mauguin symbols for the 17 wallpaper groups.
pub const WALLPAPER_GROUPS: [&str; 17] = [
    "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1",
    "p31m", "p6", "p6m",
];

/// Centred-to-primitive basis matrix `U` (`B_prim = A_conv * U`) for `cm` and `cmm`.
pub const CENTRED_U: [[f64; 2]; 2] = [[0.5, -0.5], [0.5, 0.5]];
/// Inverse `U^{-1}` mapping conventional fractional coordinates to primitive fractional coordinates.
pub const CENTRED_U_INV: [[f64; 2]; 2] = [[1.0, 1.0], [-1.0, 1.0]];
/// Identity `2x2` matrix for primitive groups.
pub const IDENTITY_2: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

const TOL: f64 = 1e-6;

/// Identifier for one of the 17 plane groups (`1..=17`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlaneGroupId(pub u8);

/// Exact 2D fractional affine symmetry operation `x -> m * x + (t_num / t_den)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RationalOp2D {
    pub m: [[i32; 2]; 2],
    pub t_num: [i32; 2],
    pub t_den: i32,
}

/// One row of the ITA Wyckoff table of a wallpaper group (`base + D * u` in conventional coords).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WyckoffRow {
    pub letter: char,
    pub multiplicity: u8,
    pub site_symmetry: String,
    pub kind: String,
    pub coords: Vec<String>,
    pub base: [f64; 2],
    #[serde(rename = "D")]
    pub d_mat: [Vec<f64>; 2],
    pub free: Vec<char>,
}

/// ITA Wyckoff classification of a point in conventional fractional coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WyckoffLabel {
    pub letter: char,
    pub multiplicity: u8,
    pub site_symmetry: String,
    pub kind: String,
    pub name: String,
}

/// Class of proper rotation centres under `G+` (rotations and all lattice translations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProperCentreClass {
    pub name: String,
    pub letter: char,
    pub multiplicity: u8,
    pub order: usize,
    pub site_symmetry: String,
    pub rep: [f64; 2],
    pub points: Vec<[f64; 2]>,
    pub g_orbit_splits: bool,
    pub mirror_partner: Option<String>,
}

/// Resolve a wallpaper group name or ITA full symbol alias to its static group specification.
pub fn group_spec(group: &str) -> Result<&'static RawGroupSpec, String> {
    let cleaned: String = group
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect();
    let key = GROUP_ALIASES
        .iter()
        .find(|&&(alias, _)| alias == cleaned)
        .map_or(cleaned.as_str(), |&(_, canon)| canon);
    GROUP_SPECS
        .iter()
        .find(|g| g.symbol == key)
        .ok_or_else(|| format!("unknown wallpaper group {group:?}; choose from {WALLPAPER_GROUPS:?}"))
}

pub fn canonical_symbol(group: &str) -> Result<&'static str, String> {
    Ok(group_spec(group)?.symbol)
}

pub fn full_symbol(group: &str) -> Result<&'static str, String> {
    Ok(group_spec(group)?.full_symbol)
}

pub fn ita_number(group: &str) -> Result<u8, String> {
    Ok(group_spec(group)?.ita_number)
}

pub fn crystal_system(group: &str) -> Result<CrystalSystem, String> {
    Ok(group_spec(group)?.system)
}

pub fn is_centred(group: &str) -> Result<bool, String> {
    Ok(group_spec(group)?.centred)
}

pub fn is_chiral(group: &str) -> Result<bool, String> {
    Ok(group_spec(group)?.chiral)
}

pub fn has_rotations(group: &str) -> Result<bool, String> {
    Ok(group_spec(group)?.has_rotations)
}

pub fn prim_u(group: &str) -> Result<[[f64; 2]; 2], String> {
    Ok(if is_centred(group)? { CENTRED_U } else { IDENTITY_2 })
}

pub fn prim_u_inv(group: &str) -> Result<[[f64; 2]; 2], String> {
    Ok(if is_centred(group)? { CENTRED_U_INV } else { IDENTITY_2 })
}

/// Conventional cell vectors `(a1, a2)` with Python defaults (`WallpaperGroup.lattice`).
pub fn conventional_cell(
    group: &str,
    a: f64,
    b: Option<f64>,
    gamma: Option<f64>,
) -> Result<([f64; 2], [f64; 2]), String> {
    let spec = group_spec(group)?;
    let cell = match spec.system {
        CrystalSystem::Oblique => {
            let b_val = b.unwrap_or(1.23 * a);
            let g_rad = gamma.unwrap_or(76.0).to_radians();
            ([a, 0.0], [b_val * g_rad.cos(), b_val * g_rad.sin()])
        }
        CrystalSystem::Rectangular => {
            let default_ratio = if spec.centred { 1.6 } else { 1.37 };
            let b_val = b.unwrap_or(default_ratio * a);
            ([a, 0.0], [0.0, b_val])
        }
        CrystalSystem::Square => ([a, 0.0], [0.0, a]),
        CrystalSystem::Hexagonal => ([a, 0.0], [-0.5 * a, 0.5 * 3.0_f64.sqrt() * a]),
    };
    Ok(cell)
}

#[derive(Clone, Copy)]
struct ParsedComponent {
    cx: i32,
    cy: i32,
    num: i32,
    den: i32,
}

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn add_rat(n1: i32, d1: i32, n2: i32, d2: i32) -> (i32, i32) {
    let num = n1 * d2 + n2 * d1;
    let den = d1 * d2;
    let g = gcd_i32(num, den);
    (num / g, den / g)
}

fn parse_component(comp: &str) -> Result<ParsedComponent, String> {
    let s: String = comp.chars().filter(|c| !c.is_whitespace()).collect();
    if s.is_empty() {
        return Err("empty coordinate component".to_string());
    }
    let mut out = ParsedComponent { cx: 0, cy: 0, num: 0, den: 1 };
    let bytes = s.as_bytes();
    let mut start = 0;
    for idx in 1..=bytes.len() {
        if idx == bytes.len() || bytes[idx] == b'+' || bytes[idx] == b'-' {
            let term = &s[start..idx];
            start = idx;
            let (sign, rest) = if let Some(r) = term.strip_prefix('-') {
                (-1, r)
            } else if let Some(r) = term.strip_prefix('+') {
                (1, r)
            } else {
                (1, term)
            };
            if let Some(coef_str) = rest.strip_suffix('x') {
                let c = if coef_str.is_empty() {
                    1
                } else {
                    coef_str.parse::<i32>().map_err(|e| e.to_string())?
                };
                out.cx += sign * c;
            } else if let Some(coef_str) = rest.strip_suffix('y') {
                let c = if coef_str.is_empty() {
                    1
                } else {
                    coef_str.parse::<i32>().map_err(|e| e.to_string())?
                };
                out.cy += sign * c;
            } else if let Some((ns, ds)) = rest.split_once('/') {
                let n = sign * ns.parse::<i32>().map_err(|e| e.to_string())?;
                let d = ds.parse::<i32>().map_err(|e| e.to_string())?;
                (out.num, out.den) = add_rat(out.num, out.den, n, d);
            } else {
                let n = sign * rest.parse::<i32>().map_err(|e| e.to_string())?;
                (out.num, out.den) = add_rat(out.num, out.den, n, 1);
            }
        }
    }
    Ok(out)
}

fn parse_coord_pair(expr: &str) -> Result<[ParsedComponent; 2], String> {
    let (c0, c1) = expr
        .split_once(',')
        .ok_or_else(|| format!("invalid coordinate expression {expr:?}"))?;
    Ok([parse_component(c0)?, parse_component(c1)?])
}

pub(crate) fn frac_mod_scalar(x: f64) -> f64 {
    let g = x - (x + TOL).floor();
    if (g - 1.0).abs() < TOL { 0.0 } else { g }
}

pub(crate) fn frac_mod(v: [f64; 2]) -> [f64; 2] {
    [frac_mod_scalar(v[0]), frac_mod_scalar(v[1])]
}

pub(crate) fn eq_mod1(p: [f64; 2], q: [f64; 2], tol: f64) -> bool {
    let d0 = p[0] - q[0];
    let d1 = p[1] - q[1];
    (d0 - d0.round_ties_even()).abs() < tol && (d1 - d1.round_ties_even()).abs() < tol
}

fn apply_frac_op(op: &FracOp2D, p: [f64; 2]) -> [f64; 2] {
    [
        f64::from(op.mf[0][0]) * p[0] + f64::from(op.mf[0][1]) * p[1] + op.t[0],
        f64::from(op.mf[1][0]) * p[0] + f64::from(op.mf[1][1]) * p[1] + op.t[1],
    ]
}

/// Exact rational symmetry operations of `group` in conventional fractional coordinates.
pub fn rational_ops(group: &str) -> Result<Vec<RationalOp2D>, String> {
    let spec = group_spec(group)?;
    let mut base_ops = Vec::with_capacity(spec.general_ops.len() * if spec.centred { 2 } else { 1 });
    for &s in spec.general_ops {
        let [c0, c1] = parse_coord_pair(s)?;
        let lcm = (c0.den * c1.den) / gcd_i32(c0.den, c1.den);
        let n0 = (c0.num * (lcm / c0.den)).rem_euclid(lcm);
        let n1 = (c1.num * (lcm / c1.den)).rem_euclid(lcm);
        base_ops.push(RationalOp2D {
            m: [[c0.cx, c0.cy], [c1.cx, c1.cy]],
            t_num: [n0, n1],
            t_den: lcm,
        });
    }
    if spec.centred {
        let half = base_ops.clone();
        for op in half {
            let (n0, d0) = add_rat(op.t_num[0], op.t_den, 1, 2);
            let (n1, d1) = add_rat(op.t_num[1], op.t_den, 1, 2);
            let lcm = (d0 * d1) / gcd_i32(d0, d1);
            base_ops.push(RationalOp2D {
                m: op.m,
                t_num: [
                    (n0 * (lcm / d0)).rem_euclid(lcm),
                    (n1 * (lcm / d1)).rem_euclid(lcm),
                ],
                t_den: lcm,
            });
        }
    }
    Ok(base_ops)
}

/// Symmetry operations of `group` in conventional fractional coordinates (including centring).
pub fn group_ops(group: &str) -> Result<Vec<FracOp2D>, String> {
    let spec = group_spec(group)?;
    let mut ops = Vec::with_capacity(spec.general_ops.len() * if spec.centred { 2 } else { 1 });
    for &s in spec.general_ops {
        let [c0, c1] = parse_coord_pair(s)?;
        let t = frac_mod([
            f64::from(c0.num) / f64::from(c0.den),
            f64::from(c1.num) / f64::from(c1.den),
        ]);
        ops.push(FracOp2D { mf: [[c0.cx, c0.cy], [c1.cx, c1.cy]], t });
    }
    if spec.centred {
        let base_len = ops.len();
        for idx in 0..base_len {
            let op = ops[idx];
            ops.push(FracOp2D {
                mf: op.mf,
                t: frac_mod([op.t[0] + 0.5, op.t[1] + 0.5]),
            });
        }
    }
    Ok(ops)
}

/// Operations of `G / T_prim` in the primitive fractional basis, in exact table order (`_prim_ops`).
pub fn prim_ops(group: &str) -> Result<Vec<FracOp2D>, String> {
    let spec = group_spec(group)?;
    let ops = group_ops(group)?;
    let u = if spec.centred { CENTRED_U } else { IDENTITY_2 };
    let ui = if spec.centred { CENTRED_U_INV } else { IDENTITY_2 };
    let reps = if spec.centred { &ops[..ops.len() / 2] } else { &ops[..] };
    let mut out = Vec::with_capacity(reps.len());
    for op in reps {
        let mf_f = [
            [f64::from(op.mf[0][0]), f64::from(op.mf[0][1])],
            [f64::from(op.mf[1][0]), f64::from(op.mf[1][1])],
        ];
        // Ui * Mf * U
        let mut mu = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                mu[r][c] = mf_f[r][0] * u[0][c] + mf_f[r][1] * u[1][c];
            }
        }
        let mut uimu = [[0i32; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                let v = ui[r][0] * mu[0][c] + ui[r][1] * mu[1][c];
                uimu[r][c] = v.round_ties_even() as i32;
            }
        }
        let t = [
            ui[0][0] * op.t[0] + ui[0][1] * op.t[1],
            ui[1][0] * op.t[0] + ui[1][1] * op.t[1],
        ];
        out.push(FracOp2D { mf: uimu, t });
    }
    Ok(out)
}

/// ITA Wyckoff table of `group` matching `papertilling.groups.wyckoff_table`.
pub fn wyckoff_table(group: &str) -> Result<Vec<WyckoffRow>, String> {
    let spec = group_spec(group)?;
    let mut out = Vec::with_capacity(spec.wyckoff.len());
    for raw in spec.wyckoff {
        let s0 = raw.coords[0];
        let [c0, c1] = parse_coord_pair(s0)?;
        let mut free = Vec::new();
        if s0.contains('x') {
            free.push('x');
        }
        if s0.contains('y') {
            free.push('y');
        }
        let base = [
            f64::from(c0.num) / f64::from(c0.den),
            f64::from(c1.num) / f64::from(c1.den),
        ];
        let mut d0 = Vec::with_capacity(free.len());
        let mut d1 = Vec::with_capacity(free.len());
        for &v in &free {
            if v == 'x' {
                d0.push(f64::from(c0.cx));
                d1.push(f64::from(c1.cx));
            } else {
                d0.push(f64::from(c0.cy));
                d1.push(f64::from(c1.cy));
            }
        }
        let kind = match free.len() {
            0 => "point",
            1 => "line",
            _ => "general",
        }
        .to_string();
        out.push(WyckoffRow {
            letter: raw.letter,
            multiplicity: raw.multiplicity,
            site_symmetry: raw.site_symmetry.to_string(),
            kind,
            coords: raw.coords.iter().map(|&c| c.to_string()).collect(),
            base,
            d_mat: [d0, d1],
            free,
        });
    }
    Ok(out)
}

fn orbit_conventional(ops: &[FracOp2D], p: [f64; 2], tol: f64) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for op in ops {
        let raw = apply_frac_op(op, p);
        let q = [raw[0] - (raw[0] + tol).floor(), raw[1] - (raw[1] + tol).floor()];
        if !pts.iter().any(|&r| eq_mod1(q, r, tol)) {
            pts.push(q);
        }
    }
    pts
}

fn in_wyckoff_row(q: [f64; 2], row: &WyckoffRow, tol: f64) -> bool {
    if row.kind == "general" {
        return true;
    }
    let v = [q[0] - row.base[0], q[1] - row.base[1]];
    if row.kind == "point" {
        return eq_mod1(v, [0.0, 0.0], tol);
    }
    let (dx, dy) = (row.d_mat[0][0], row.d_mat[1][0]);
    let cross = v[0] * dy - v[1] * dx;
    (cross - cross.round_ties_even()).abs() < tol
}

/// Classify a conventional fractional point `pt` into its ITA Wyckoff position (`label_point`).
pub fn label_point(group: &str, pt: [f64; 2]) -> Result<WyckoffLabel, String> {
    let sym = canonical_symbol(group)?;
    let ops = group_ops(sym)?;
    let rows = wyckoff_table(sym)?;
    let orb = orbit_conventional(&ops, pt, TOL);
    let hits: Vec<&WyckoffRow> = rows
        .iter()
        .filter(|r| usize::from(r.multiplicity) == orb.len() && orb.iter().any(|&q| in_wyckoff_row(q, r, TOL)))
        .collect();
    if hits.len() != 1 {
        return Err(format!(
            "{sym}: point ({}, {}) matches {} rows",
            pt[0],
            pt[1],
            hits.len()
        ));
    }
    let r = hits[0];
    Ok(WyckoffLabel {
        letter: r.letter,
        multiplicity: r.multiplicity,
        site_symmetry: r.site_symmetry.clone(),
        kind: r.kind.clone(),
        name: format!("{}{}", r.multiplicity, r.letter),
    })
}

/// Rotation centres of one conventional cell partitioned into `G+` classes (`proper_centre_classes`).
/// When a `G`-orbit splits into two `G+`-classes, `(i)` contains the FIRST coordinate in `ITA_TABLES`.
pub fn proper_centre_classes(group: &str) -> Result<Vec<ProperCentreClass>, String> {
    let sym = canonical_symbol(group)?;
    let ops = group_ops(sym)?;
    let rows = wyckoff_table(sym)?;
    let proper: Vec<FracOp2D> = ops
        .iter()
        .copied()
        .filter(|op| op.mf[0][0] * op.mf[1][1] - op.mf[0][1] * op.mf[1][0] == 1)
        .collect();
    let improper: Vec<FracOp2D> = ops
        .iter()
        .copied()
        .filter(|op| op.mf[0][0] * op.mf[1][1] - op.mf[0][1] * op.mf[1][0] == -1)
        .collect();

    let mut out = Vec::new();
    for r in rows {
        if r.kind != "point" {
            continue;
        }
        let pts = orbit_conventional(&ops, r.base, TOL);
        let mut classes: Vec<Vec<[f64; 2]>> = Vec::new();
        for q in pts {
            if !classes.iter().any(|cl| cl.iter().any(|&c| eq_mod1(q, c, TOL))) {
                let mut raw_cl = Vec::new();
                for op in &proper {
                    let mq = apply_frac_op(op, q);
                    let red = [mq[0] - (mq[0] + TOL).floor(), mq[1] - (mq[1] + TOL).floor()];
                    if !raw_cl.iter().any(|&d| eq_mod1(red, d, TOL)) {
                        raw_cl.push(red);
                    }
                }
                classes.push(raw_cl);
            }
        }
        let nm = format!("{}{}", r.multiplicity, r.letter);
        let names = if classes.len() == 1 {
            vec![nm]
        } else {
            vec![format!("{nm}(i)"), format!("{nm}(ii)")]
        };
        let order = proper
            .iter()
            .filter(|op| eq_mod1(apply_frac_op(op, r.base), r.base, TOL))
            .count();
        let splits = classes.len() == 2;
        let mut recs: Vec<ProperCentreClass> = classes
            .into_iter()
            .zip(names)
            .map(|(cl, name)| ProperCentreClass {
                name,
                letter: r.letter,
                multiplicity: r.multiplicity,
                order,
                site_symmetry: r.site_symmetry.clone(),
                rep: cl[0],
                points: cl,
                g_orbit_splits: splits,
                mirror_partner: None,
            })
            .collect();
        if let Some(imp0) = improper.first() {
            let snapshot = recs.clone();
            for rec in &mut recs {
                let img = apply_frac_op(imp0, rec.rep);
                let partner = snapshot
                    .iter()
                    .find(|o| o.points.iter().any(|&c| eq_mod1(img, c, TOL)))
                    .map(|o| o.name.clone())
                    .ok_or_else(|| format!("{sym}: mirror partner not found for {}", rec.name))?;
                rec.mirror_partner = Some(partner);
            }
        }
        out.extend(recs);
    }
    Ok(out)
}
