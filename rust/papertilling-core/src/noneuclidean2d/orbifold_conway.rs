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

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.1 + b.0 * a.1;
    let den = a.1 * b.1;
    let g = gcd_u128(num.unsigned_abs(), den.unsigned_abs()) as i128;
    let (rn, rd) = (num / g, den / g);
    if rd < 0 { (-rn, -rd) } else { (rn, rd) }
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

fn canonical_cycle_desc(vals: &[u32]) -> Vec<u32> {
    if vals.len() <= 1 {
        return vals.to_vec();
    }
    let mut best: Option<Vec<u32>> = None;
    let mut seq = vals.to_vec();
    for _ in 0..2 {
        for rot in 0..seq.len() {
            let cand: Vec<u32> = (0..seq.len()).map(|i| seq[(rot + i) % seq.len()]).collect();
            if best.as_ref().is_none_or(|b| cand > *b) {
                best = Some(cand);
            }
        }
        seq.reverse();
    }
    best.unwrap()
}

fn match_wallpaper_group(
    handles: u32,
    crosscaps: u32,
    boundaries: u32,
    cones: &[u32],
    corners: &[u32],
) -> Option<&'static str> {
    let mut c_sorted = cones.to_vec();
    c_sorted.sort_unstable_by(|a, b| b.cmp(a));
    let cor = canonical_cycle_desc(corners);
    match (handles, crosscaps, boundaries, c_sorted.as_slice(), cor.as_slice()) {
        (0, 0, 1, [], [6, 3, 2]) => Some("p6m"),
        (0, 0, 0, [6, 3, 2], []) => Some("p6"),
        (0, 0, 1, [], [4, 4, 2]) => Some("p4m"),
        (0, 0, 0, [4, 4, 2], []) => Some("p4"),
        (0, 0, 1, [4], [2]) => Some("p4g"),
        (0, 0, 1, [], [3, 3, 3]) => Some("p3m1"),
        (0, 0, 1, [3], [3]) => Some("p31m"),
        (0, 0, 0, [3, 3, 3], []) => Some("p3"),
        (0, 0, 1, [], [2, 2, 2, 2]) => Some("pmm"),
        (0, 0, 1, [2, 2], []) => Some("pmg"),
        (0, 1, 0, [2, 2], []) => Some("pgg"),
        (0, 0, 1, [2], [2, 2]) => Some("cmm"),
        (0, 0, 2, [], []) => Some("pm"),
        (0, 2, 0, [], []) => Some("pg"),
        (0, 1, 1, [], []) => Some("cm"),
        (0, 0, 0, [2, 2, 2, 2], []) => Some("p2"),
        (1, 0, 0, [], []) => Some("p1"),
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
    let (mut cones, mut corners) = (Vec::new(), Vec::new());
    let mut in_mirror = false;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        match ch {
            '*' => {
                boundaries += 1;
                in_mirror = true;
                i += 1;
            }
            'o' | 'O' => {
                handles += 1;
                in_mirror = false;
                i += 1;
            }
            'x' | 'X' => {
                crosscaps += 1;
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
                let target = if in_mirror { &mut corners } else { &mut cones };
                let inner_has_sep = inner.contains(',') || inner.contains('.') || inner.chars().any(char::is_whitespace);
                if inner_has_sep {
                    parse_digit_chunk(&inner, target)?;
                } else {
                    let val: u32 = inner
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid parenthesized orbifold order ({inner})"))?;
                    if val < 2 {
                        return Err(format!("orbifold orders must be >= 2, got {val}"));
                    }
                    target.push(val);
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
                let target = if in_mirror { &mut corners } else { &mut cones };
                parse_digit_chunk(&chunk, target)?;
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

    let mut chi = (2_i128 - 2 * i128::from(handles) - i128::from(crosscaps) - i128::from(boundaries), 1_i128);
    for &n in &cones {
        chi = rat_add(chi, (-(i128::from(n) - 1), i128::from(n)));
    }
    for &m in &corners {
        chi = rat_add(chi, (-(i128::from(m) - 1), 2 * i128::from(m)));
    }
    let geometry_kind = match chi.0.cmp(&0) {
        std::cmp::Ordering::Greater => GeometryKind::Spherical,
        std::cmp::Ordering::Equal => GeometryKind::Euclidean,
        std::cmp::Ordering::Less => GeometryKind::Hyperbolic,
    };

    let wallpaper_group = match_wallpaper_group(handles, crosscaps, boundaries, &cones, &corners).map(str::to_string);
    let wallpaper_orb = wallpaper_group
        .as_deref()
        .and_then(|wg| WALLPAPER_ORBIFOLD_TABLE.iter().find(|&&(g, _)| g == wg).map(|&(_, orb)| orb));
    let canonical = if let Some(orb) = wallpaper_orb {
        orb.to_string()
    } else {
        let mut s = "o".repeat(handles as usize);
        let mut c_desc = cones.clone();
        c_desc.sort_unstable_by(|a, b| b.cmp(a));
        s.push_str(&format_order_seq(&c_desc));
        if boundaries > 0 {
            s.push('*');
            let cor_desc = canonical_cycle_desc(&corners);
            s.push_str(&format_order_seq(&cor_desc));
            for _ in 1..boundaries {
                s.push('*');
            }
        }
        s.push_str(&"x".repeat(crosscaps as usize));
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
        chi_num: chi.0,
        chi_den: chi.1,
        chi_orb: (chi.0 as f64) / (chi.1 as f64),
        geometry_kind,
        wallpaper_group,
        provenance: "[PROVED]".to_string(),
    })
}
