//! Conway orbifold notation, 17-wallpaper bijection, exact rational Euler characteristic,
//! and Thurston's four bad-orbifold rejections.
//!
//! Sources ([PROVED], published): Conway, "The orbifold notation for surface groups", in *Groups,
//! Combinatorics and Geometry*, LMS Lecture Notes 165 (1992) 438-447; Conway, Burgiel & Goodman-Strauss,
//! *The Symmetries of Things* (2008), Part I, Magic Theorem (cost: `o` = 2, `*` and `x` = 1, cone `n` =
//! (n-1)/n, corner `n` = (n-1)/(2n); `chi = 2 - cost`) and its list of the 17 plane groups (exactly the
//! orbifolds of cost 2, table below). Bad orbifolds `n`, `mn` (m != n), `*n`, `*mn` (m != n):
//! Thurston, *The Geometry and Topology of Three-Manifolds*, lecture notes 1980, ch. 13.

use serde::{Deserialize, Serialize};

use crate::dsymbol::GeometryKind;
use crate::plane::groups::canonical_symbol as canonical_wallpaper_symbol;

pub const WALLPAPER_ORBIFOLD_TABLE: [(&str, &str); 17] = [
    ("p6m", "*632"),
    ("p6", "632"),
    ("p4m", "*442"),
    ("p4", "442"),
    ("p4g", "4*2"),
    ("p3m1", "*333"),
    ("p31m", "3*3"),
    ("p3", "333"),
    ("pmm", "*2222"),
    ("pmg", "22*"),
    ("pgg", "22x"),
    ("cmm", "2*22"),
    ("pm", "**"),
    ("pg", "xx"),
    ("cm", "*x"),
    ("p2", "2222"),
    ("p1", "o"),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConwayOrbifold {
    pub input: String,
    pub canonical: String,
    pub handles: u32,
    pub crosscaps: u32,
    pub boundary_components: u32,
    pub cones: Vec<u32>,
    pub corners: Vec<u32>,
    pub chi_num: i128,
    pub chi_den: i128,
    pub chi_orb: f64,
    pub geometry_kind: GeometryKind,
    pub wallpaper_group: Option<String>,
    pub provenance: String,
}

pub fn wallpaper_orbifold_table() -> &'static [(&'static str, &'static str)] {
    &WALLPAPER_ORBIFOLD_TABLE
}

pub fn wallpaper_to_orbifold(group: &str) -> Result<&'static str, String> {
    let canon = canonical_wallpaper_symbol(group)?;
    WALLPAPER_ORBIFOLD_TABLE
        .iter()
        .find(|&&(g, _)| g == canon)
        .map(|&(_, orb)| orb)
        .ok_or_else(|| format!("missing orbifold mapping for wallpaper group {canon:?}"))
}

pub fn orbifold_to_wallpaper(symbol: &str) -> Result<&'static str, String> {
    let parsed = parse_orbifold(symbol)?;
    parsed
        .wallpaper_group
        .as_deref()
        .and_then(|g| {
            WALLPAPER_ORBIFOLD_TABLE
                .iter()
                .find(|&&(wg, _)| wg == g)
                .map(|&(wg, _)| wg)
        })
        .ok_or_else(|| format!("orbifold {symbol:?} (chi = {}/{}) is not a Euclidean wallpaper group", parsed.chi_num, parsed.chi_den))
}

/// Orbifold cost is additive. Reduce with arbitrary precision before converting to the public fraction.
fn exact_orbifold_chi(
    symbol: &str, handles: u32, crosscaps: u32, boundaries: u32, cones: &[u32], corners: &[u32],
) -> Result<(i128, i128), String> {
    use num_rational::BigRational;
    let topological_chi = 2 - 2 * i64::from(handles) - i64::from(crosscaps) - i64::from(boundaries);
    let mut chi = BigRational::from_integer(topological_chi.into());
    for (orders, factor) in [(cones, 1_u64), (corners, 2)] {
        for &order in orders {
            chi -= BigRational::new((order - 1).into(), (factor * u64::from(order)).into());
        }
    }
    let out_of_range = |_| format!("orbifold Euler characteristic for {symbol:?} is not representable as an exact i128 fraction");
    Ok((
        i128::try_from(chi.numer()).map_err(out_of_range)?,
        i128::try_from(chi.denom()).map_err(out_of_range)?,
    ))
}

fn parse_digit_chunk(chunk: &str, out: &mut Vec<u32>) -> Result<(), String> {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let has_delims = trimmed.contains('.') || trimmed.contains(',') || trimmed.chars().any(char::is_whitespace);
    if has_delims {
        for part in trimmed.split(|c: char| c == '.' || c == ',' || c.is_whitespace()) {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            let val: u32 = p.parse().map_err(|_| format!("invalid orbifold order {p:?}"))?;
            if val < 2 {
                return Err(format!("orbifold orders must be >= 2, got {val}"));
            }
            out.push(val);
        }
    } else {
        for ch in trimmed.chars() {
            let d = ch
                .to_digit(10)
                .ok_or_else(|| format!("unexpected character {ch:?} in orbifold symbol"))?;
            if d < 2 {
                return Err(format!("compact orbifold cone/corner order must be >= 2, got {d}"));
            }
            out.push(d);
        }
    }
    Ok(())
}

fn format_order_seq(orders: &[u32]) -> String {
    if orders.iter().all(|&n| n < 10) {
        orders.iter().map(|n| n.to_string()).collect()
    } else {
        format!("({})", orders.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","))
    }
}

fn canonical_rot_desc(vals: &[u32]) -> Vec<u32> {
    if vals.len() <= 1 {
        return vals.to_vec();
    }
    (0..vals.len())
        .map(|rot| (0..vals.len()).map(|i| vals[(rot + i) % vals.len()]).collect::<Vec<_>>())
        .max()
        .unwrap()
}

fn canonical_cycle_desc(vals: &[u32]) -> Vec<u32> {
    if vals.len() <= 1 {
        return vals.to_vec();
    }
    let rev: Vec<u32> = vals.iter().copied().rev().collect();
    canonical_rot_desc(vals).max(canonical_rot_desc(&rev))
}

/// Canonicalize per-boundary corner cycles (Conway 1992; Conway, Burgiel & Goodman-Strauss 2008).
///
/// On an orientable base (`crosscaps == 0`), an orientation-preserving homeomorphism rotates each
/// boundary cycle independently and permutes boundary components, whereas an orientation-reversing
/// homeomorphism reverses *all* boundary cycles simultaneously. When `crosscaps > 0`, sliding a
/// single boundary around a one-sided crosscap loop reverses that boundary cycle independently.
fn canonical_boundary_cycles(boundaries: &[Vec<u32>], crosscaps: u32) -> Vec<Vec<u32>> {
    if crosscaps > 0 {
        let mut out: Vec<Vec<u32>> = boundaries.iter().map(|b| canonical_cycle_desc(b)).collect();
        out.sort_unstable_by(|a, b| b.cmp(a));
        return out;
    }
    let mut fwd: Vec<Vec<u32>> = boundaries.iter().map(|b| canonical_rot_desc(b)).collect();
    fwd.sort_unstable_by(|a, b| b.cmp(a));
    let mut rev: Vec<Vec<u32>> = boundaries
        .iter()
        .map(|b| {
            let r: Vec<u32> = b.iter().copied().rev().collect();
            canonical_rot_desc(&r)
        })
        .collect();
    rev.sort_unstable_by(|a, b| b.cmp(a));
    fwd.max(rev)
}

fn match_wallpaper_group(
    handles: u32,
    crosscaps: u32,
    cones_desc: &[u32],
    boundary_cycles: &[Vec<u32>],
) -> Option<&'static str> {
    match (handles, crosscaps, cones_desc, boundary_cycles) {
        (0, 0, [], [b]) if b.as_slice() == [6, 3, 2] => Some("p6m"),
        (0, 0, [6, 3, 2], []) => Some("p6"),
        (0, 0, [], [b]) if b.as_slice() == [4, 4, 2] => Some("p4m"),
        (0, 0, [4, 4, 2], []) => Some("p4"),
        (0, 0, [4], [b]) if b.as_slice() == [2] => Some("p4g"),
        (0, 0, [], [b]) if b.as_slice() == [3, 3, 3] => Some("p3m1"),
        (0, 0, [3], [b]) if b.as_slice() == [3] => Some("p31m"),
        (0, 0, [3, 3, 3], []) => Some("p3"),
        (0, 0, [], [b]) if b.as_slice() == [2, 2, 2, 2] => Some("pmm"),
        (0, 0, [2, 2], [b]) if b.is_empty() => Some("pmg"),
        (0, 1, [2, 2], []) => Some("pgg"),
        (0, 0, [2], [b]) if b.as_slice() == [2, 2] => Some("cmm"),
        (0, 0, [], [b0, b1]) if b0.is_empty() && b1.is_empty() => Some("pm"),
        (0, 2, [], []) => Some("pg"),
        (0, 1, [], [b]) if b.is_empty() => Some("cm"),
        (0, 0, [2, 2, 2, 2], []) => Some("p2"),
        (1, 0, [], []) => Some("p1"),
        _ => None,
    }
}

/// Parse a Conway orbifold signature, verify it is not one of Thurston's 4 bad orbifolds,
/// and compute its exact rational orbifold Euler characteristic `chi_orb`.
pub fn parse_orbifold(symbol: &str) -> Result<ConwayOrbifold, String> {
    let trimmed = symbol.trim();
    if trimmed.is_empty() {
        return Err("empty orbifold symbol".to_string());
    }
    if trimmed == "1" {
        return Ok(ConwayOrbifold {
            input: symbol.to_string(),
            canonical: "1".to_string(),
            handles: 0,
            crosscaps: 0,
            boundary_components: 0,
            cones: Vec::new(),
            corners: Vec::new(),
            chi_num: 2,
            chi_den: 1,
            chi_orb: 2.0,
            geometry_kind: GeometryKind::Spherical,
            wallpaper_group: None,
            provenance: "[PROVED]".to_string(),
        });
    }

    let normalized: String = trimmed.chars().map(|c| if c == '×' { 'x' } else { c }).collect();
    let chars: Vec<char> = normalized.chars().collect();
    let (mut handles, mut crosscaps, mut boundaries) = (0u32, 0u32, 0u32);
    let mut cones = Vec::new();
    let mut corners = Vec::new();
    let mut boundary_cycles: Vec<Vec<u32>> = Vec::new();
    let mut in_mirror = false;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        match ch {
            '*' => {
                boundaries = boundaries
                    .checked_add(1)
                    .ok_or_else(|| format!("too many mirror boundaries in orbifold symbol {symbol:?}"))?;
                boundary_cycles.push(Vec::new());
                in_mirror = true;
                i += 1;
            }
            'o' | 'O' => {
                handles = handles
                    .checked_add(1)
                    .ok_or_else(|| format!("too many handles in orbifold symbol {symbol:?}"))?;
                in_mirror = false;
                i += 1;
            }
            'x' | 'X' => {
                crosscaps = crosscaps
                    .checked_add(1)
                    .ok_or_else(|| format!("too many crosscaps in orbifold symbol {symbol:?}"))?;
                in_mirror = false;
                i += 1;
            }
            '(' => {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && chars[end] != ')' {
                    end += 1;
                }
                if end >= chars.len() {
                    return Err(format!("unclosed '(' in orbifold symbol {symbol:?}"));
                }
                let inner: String = chars[start..end].iter().collect();
                let mut parsed = Vec::new();
                let inner = inner.trim();
                let inner_has_sep = inner.contains(',') || inner.contains('.') || inner.chars().any(char::is_whitespace);
                if inner_has_sep {
                    parse_digit_chunk(inner, &mut parsed)?;
                } else {
                    let val: u32 = inner
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid parenthesized orbifold order ({inner})"))?;
                    if val < 2 {
                        return Err(format!("orbifold orders must be >= 2, got {val}"));
                    }
                    parsed.push(val);
                }
                if in_mirror {
                    boundary_cycles.last_mut().expect("in_mirror implies a boundary").extend(&parsed);
                    corners.extend(parsed);
                } else {
                    cones.extend(parsed);
                }
                i = end + 1;
            }
            c if c.is_ascii_digit() || c == '.' || c == ',' || c.is_whitespace() => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',' || chars[i].is_whitespace())
                {
                    i += 1;
                }
                let chunk: String = chars[start..i].iter().collect();
                let mut parsed = Vec::new();
                parse_digit_chunk(&chunk, &mut parsed)?;
                if in_mirror {
                    boundary_cycles.last_mut().expect("in_mirror implies a boundary").extend(&parsed);
                    corners.extend(parsed);
                } else {
                    cones.extend(parsed);
                }
            }
            _ => return Err(format!("unexpected character {ch:?} in orbifold symbol {symbol:?}")),
        }
    }

    if handles == 0 && crosscaps == 0 {
        let is_bad = (boundaries == 0 && cones.len() == 1)
            || (boundaries == 1 && cones.is_empty() && corners.len() == 1)
            || (boundaries == 0 && cones.len() == 2 && cones[0] != cones[1])
            || (boundaries == 1 && cones.is_empty() && corners.len() == 2 && corners[0] != corners[1]);
        if is_bad {
            return Err(format!(
                "[PROVED: bad orbifold — no manifold cover]: {symbol:?} is one of Thurston's four bad orbifolds"
            ));
        }
    }

    let (chi_num, chi_den) = exact_orbifold_chi(symbol, handles, crosscaps, boundaries, &cones, &corners)?;
    let geometry_kind = match chi_num.cmp(&0) {
        std::cmp::Ordering::Greater => GeometryKind::Spherical,
        std::cmp::Ordering::Equal => GeometryKind::Euclidean,
        std::cmp::Ordering::Less => GeometryKind::Hyperbolic,
    };

    let mut c_desc = cones.clone();
    c_desc.sort_unstable_by(|a, b| b.cmp(a));
    let canon_bnds = canonical_boundary_cycles(&boundary_cycles, crosscaps);
    let wallpaper_group = match_wallpaper_group(handles, crosscaps, &c_desc, &canon_bnds).map(str::to_string);
    let wallpaper_orb = wallpaper_group
        .as_deref()
        .and_then(|wg| WALLPAPER_ORBIFOLD_TABLE.iter().find(|&&(g, _)| g == wg).map(|&(_, orb)| orb));
    let canonical = if let Some(orb) = wallpaper_orb {
        orb.to_string()
    } else {
        // A handle on a nonorientable surface is equivalent to two additional crosscaps.
        let (h, x) = if crosscaps > 0 {
            (0, crosscaps as usize + 2 * handles as usize)
        } else {
            (handles as usize, 0)
        };
        let mut s = "o".repeat(h);
        s.push_str(&format_order_seq(&c_desc));
        for cyc in &canon_bnds {
            s.push('*');
            s.push_str(&format_order_seq(cyc));
        }
        s.push_str(&"x".repeat(x));
        s
    };

    Ok(ConwayOrbifold {
        input: symbol.to_string(),
        canonical,
        handles,
        crosscaps,
        boundary_components: boundaries,
        cones,
        corners,
        chi_num,
        chi_den,
        chi_orb: (chi_num as f64) / (chi_den as f64),
        geometry_kind,
        wallpaper_group,
        provenance: "[PROVED]".to_string(),
    })
}
