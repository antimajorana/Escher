//! Burgers-vector index notation: `[u, v]` in the cell basis (`b = u a1 + v a2`), its family `<u, v>` (orbit under the
//! cell's point group), and the exact 4-index Miller-Bravais (Weber) direction on hexagonal cells.
//!
//! Weber: write `b = U e1 + V e2` in 120-degree axes (`e1 = a1`, `e2 = a2` for a 120-degree cell; `e2 = a2 - a1` for a
//! 60-degree cell, so `(U, V) = (u + v, v)`), then `[u v t w] = (1/3)[2U - V, 2V - U, -(U + V), 0]`, written without the
//! `1/3` when all three numerators are divisible by 3.

use serde::{Deserialize, Serialize};

use crate::defect::cone_geometry::norm2;
use crate::types::Cell2D;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weber4Index {
    pub indices: [i32; 4],
    pub third_scale: bool,
}

impl Weber4Index {
    pub fn format(&self) -> String {
        let [a, b, c, d] = self.indices;
        if self.third_scale { format!("1/3[{a} {b} {c} {d}]") } else { format!("[{a} {b} {c} {d}]") }
    }
}

/// Transparent alias for `&'static str` so `serde_derive` generates `Deserialize<'de>` without `'de: 'static`.
pub type StaticStr = &'static str;

pub(crate) fn deser_static_str<'de, D: serde::Deserializer<'de>>(de: D) -> Result<StaticStr, D::Error> {
    let s = <String as serde::Deserialize>::deserialize(de)?;
    match s.as_str() {
        "[PROVED]" => Ok("[PROVED]"),
        "[COMPUTATIONAL EVIDENCE]" => Ok("[COMPUTATIONAL EVIDENCE]"),
        "[NOT RELAXED]" => Ok("[NOT RELAXED]"),
        "hexagonal" => Ok("hexagonal"),
        "square" => Ok("square"),
        "[PROVED: exact CSL \u{3a3}]" => Ok("[PROVED: exact CSL \u{3a3}]"),
        "[COMPUTATIONAL EVIDENCE: finite symmetric-tilt patch]" => Ok("[COMPUTATIONAL EVIDENCE: finite symmetric-tilt patch]"),
        "[PROVED: exact integer orbit closure under the bond-preserving cell symmetry]" => {
            Ok("[PROVED: exact integer orbit closure under the bond-preserving cell symmetry]")
        }
        "[COMPUTATIONAL EVIDENCE: geometric search, tol 1e-4; closure checked exactly]" => {
            Ok("[COMPUTATIONAL EVIDENCE: geometric search, tol 1e-4; closure checked exactly]")
        }
        "[PROVED: integer space-group action, adjacency invariance and closure checked exactly]" => {
            Ok("[PROVED: integer space-group action, adjacency invariance and closure checked exactly]")
        }
        "" => Ok(""),
        _ => Err(serde::de::Error::custom(format!("unknown static label: {s:?}"))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurgersIndexInfo {
    pub uv: [i32; 2],
    pub uv_str: String,
    pub family_uv: [i32; 2],
    pub family_str: String,
    pub weber_hex: Option<Weber4Index>,
    pub weber_str: Option<String>,
    #[serde(deserialize_with = "deser_static_str")]
    pub provenance: StaticStr,
}

/// Weber index from 120-degree hexagonal components `(U, V)`.
pub fn weber_from_hex_uv(uu: i32, vv: i32) -> Weber4Index {
    let (u3, v3, t3) = (2 * uu - vv, 2 * vv - uu, -(uu + vv));
    if u3 % 3 == 0 && v3 % 3 == 0 {
        Weber4Index { indices: [u3 / 3, v3 / 3, t3 / 3, 0], third_scale: false }
    } else {
        Weber4Index { indices: [u3, v3, t3, 0], third_scale: true }
    }
}

/// `(U, V)` in 120-degree axes when `group` is trigonal/hexagonal and the cell metric is hexagonal, else `None`.
pub fn hex_components(group: &str, cell: &Cell2D, uv: [i32; 2]) -> Option<(i32, i32)> {
    if !matches!(group, "p3" | "p3m1" | "p31m" | "p6" | "p6m") {
        return None;
    }
    let (l1, l2) = (norm2(cell.a1), norm2(cell.a2));
    if l1 < 1e-12 || (l1 - l2).abs() > 1e-5 * l1 {
        return None;
    }
    let cos_g = (cell.a1[0] * cell.a2[0] + cell.a1[1] * cell.a2[1]) / (l1 * l2);
    let (uu, vv, e2) = if (cos_g + 0.5).abs() <= 1e-4 {
        (uv[0], uv[1], cell.a2)
    } else if (cos_g - 0.5).abs() <= 1e-4 {
        (uv[0] + uv[1], uv[1], [cell.a2[0] - cell.a1[0], cell.a2[1] - cell.a1[1]])
    } else {
        return None;
    };
    let b = [f64::from(uv[0]) * cell.a1[0] + f64::from(uv[1]) * cell.a2[0], f64::from(uv[0]) * cell.a1[1] + f64::from(uv[1]) * cell.a2[1]];
    let r = [f64::from(uu) * cell.a1[0] + f64::from(vv) * e2[0], f64::from(uu) * cell.a1[1] + f64::from(vv) * e2[1]];
    ((r[0] - b[0]).hypot(r[1] - b[1]) <= 1e-5 * l1).then_some((uu, vv))
}

/// Family representative of `uv` under the cell's bond-preserving point group: prefer both components `>= 0`, then the
/// largest `u`, then the largest `v`.
pub fn family_representative(cell: &Cell2D, uv: [i32; 2]) -> [i32; 2] {
    let mut mats: Vec<[[i32; 2]; 2]> = cell.symmetry_frac_ops(true).into_iter().map(|o| o.mf).collect();
    if mats.is_empty() {
        mats.push([[1, 0], [0, 1]]);
    }
    mats.iter()
        .map(|m| [m[0][0] * uv[0] + m[0][1] * uv[1], m[1][0] * uv[0] + m[1][1] * uv[1]])
        .max_by_key(|w| (w[0] >= 0 && w[1] >= 0, w[0], w[1]))
        .unwrap_or(uv)
}

/// Index notation of the Burgers vector `b = u a1 + v a2` (e.g. `DislocationEntry::b_frac`).
pub fn burgers_index_info(group: &str, cell: &Cell2D, uv: [i32; 2]) -> BurgersIndexInfo {
    let family_uv = family_representative(cell, uv);
    let weber_hex = hex_components(group, cell, uv).map(|(uu, vv)| weber_from_hex_uv(uu, vv));
    BurgersIndexInfo {
        uv,
        uv_str: format!("[{}, {}]", uv[0], uv[1]),
        family_uv,
        family_str: format!("<{}, {}>", family_uv[0], family_uv[1]),
        weber_str: weber_hex.as_ref().map(Weber4Index::format),
        weber_hex,
        provenance: "[PROVED]",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bond2D;

    fn hex_cell(gamma_deg: f64) -> Cell2D {
        let g = gamma_deg.to_radians();
        let bonds = vec![
            Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
            Bond2D { i: 0, j: 0, n1: 1, n2: if gamma_deg > 90.0 { 1 } else { -1 }, kind: 0 },
        ];
        Cell2D { a1: [1.0, 0.0], a2: [g.cos(), g.sin()], basis: vec![[0.0, 0.0]], bonds }
    }

    #[test]
    fn weber_on_120_and_60_degree_cells() {
        // hand values: (U,V)=(1,0) -> 1/3[2 -1 -1 0]; (1,1) -> 1/3[1 1 -2 0]; (2,1) -> [1 0 -1 0]
        assert_eq!(weber_from_hex_uv(1, 0).format(), "1/3[2 -1 -1 0]");
        assert_eq!(weber_from_hex_uv(1, 1).format(), "1/3[1 1 -2 0]");
        assert_eq!(weber_from_hex_uv(2, 1).format(), "[1 0 -1 0]");
        let c120 = hex_cell(120.0);
        for (uv, s) in [([1, 0], "1/3[2 -1 -1 0]"), ([1, 1], "1/3[1 1 -2 0]"), ([2, 1], "[1 0 -1 0]")] {
            assert_eq!(burgers_index_info("p6m", &c120, uv).weber_str.as_deref(), Some(s));
        }
        // 60-degree axes: b = u a1 + v a2 = (u + v) e1 + v e2 with e2 = a2 - a1
        let c60 = hex_cell(60.0);
        assert_eq!(hex_components("p6m", &c60, [1, 0]), Some((1, 0)));
        assert_eq!(hex_components("p6m", &c60, [0, 1]), Some((1, 1)));
        assert_eq!(hex_components("p6m", &c60, [1, 1]), Some((2, 1)));
        assert_eq!(burgers_index_info("p6", &c60, [0, 1]).weber_str.as_deref(), Some("1/3[1 1 -2 0]"));
        assert_eq!(burgers_index_info("p3", &c60, [1, 1]).weber_str.as_deref(), Some("[1 0 -1 0]"));
        // every Weber index reconstructs (U, V): (1/3)(u3 e1 + v3 e2 + t3 (-e1 - e2)) = U e1 + V e2
        for uu in -3..=3 {
            for vv in -3..=3 {
                let w = weber_from_hex_uv(uu, vv);
                let k = if w.third_scale { 1 } else { 3 };
                let [a, b, c, _] = w.indices;
                assert_eq!(a + b + c, 0);
                assert_eq!((k * (a - c), k * (b - c)), (3 * uu, 3 * vv));
            }
        }
    }

    #[test]
    fn square_has_no_weber_and_uses_families() {
        let sq = Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 }, Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 }],
        };
        let info = burgers_index_info("p4m", &sq, [0, -1]);
        assert_eq!(info.weber_hex, None);
        assert_eq!(info.uv_str, "[0, -1]");
        assert_eq!(info.family_str, "<1, 0>");
        assert_eq!(burgers_index_info("p4m", &sq, [-1, 1]).family_uv, [1, 1]);
        assert_eq!(info.provenance, "[PROVED]");
    }
}
