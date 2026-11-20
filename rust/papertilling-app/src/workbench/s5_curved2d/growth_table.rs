//! Shared 2D layer-growth readout helpers for S5 (`["Layer", "Cells", "New", "∂V", "∂E", "χ"]`), seed selector UI,
//! non-disk χ flags, and exact `PlaneTiling` growth for regular Euclidean `{4,4}`, `{3,6}`, `{6,3}` across all seeds.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::growth::{LayerStats, SeedKind};
use papertilling_core::noneuclidean2d::{PqGrowth, UnifiedPqTiling};
use papertilling_core::plane::inflation::{InflationRule, InflationSeed, PlaneTiling, VertexId, inflate_with};
use papertilling_core::types::{Bond2D, Cell2D};

pub(crate) const GROWTH_HEADERS: [&str; 6] = ["Layer", "Cells", "New", "∂V", "∂E", "χ"];

pub(crate) const SEEDS_2D: [(SeedKind, &str); 3] = [
    (SeedKind::Vertex, "Vertex"),
    (SeedKind::Edge, "Edge"),
    (SeedKind::Cell, "Cell"),
];

pub(crate) fn seed_label_2d(seed: SeedKind) -> &'static str {
    match seed {
        SeedKind::Vertex => "Vertex",
        SeedKind::Edge => "Edge",
        SeedKind::Cell | SeedKind::Face => "Cell",
        SeedKind::Dirichlet => "Dirichlet",
    }
}

pub(crate) fn parse_seed_2d(s: &str) -> Option<SeedKind> {
    match s {
        "Vertex" | "vertex" => Some(SeedKind::Vertex),
        "Edge" | "edge" => Some(SeedKind::Edge),
        "Cell" | "cell" | "Face" | "face" => Some(SeedKind::Cell),
        _ => None,
    }
}

pub(crate) fn seed_controls(ui: &mut egui::Ui, seed: &mut SeedKind) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("Seed:");
        for (s, lbl) in SEEDS_2D {
            changed |= ui.selectable_value(seed, s, lbl).changed();
        }
    });
    changed
}

pub(crate) fn stats_to_rows(stats: &[LayerStats]) -> Vec<Vec<String>> {
    stats
        .iter()
        .enumerate()
        .map(|(k, s)| {
            vec![
                k.to_string(),
                s.cells.to_string(),
                s.new_cells.to_string(),
                s.boundary_vertices.to_string(),
                s.boundary_edges.to_string(),
                s.euler.to_string(),
            ]
        })
        .collect()
}

/// Flag any layer whose Euler characteristic differs from 1 before S² closure (or from 2 at S² closure).
pub(crate) fn non_disk_flags(stats: &[LayerStats], closed_s2_layer: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    for (k, s) in stats.iter().enumerate() {
        if Some(k) == closed_s2_layer {
            if s.euler != 2 {
                out.push(format!("Layer {k} closed with χ = {} ≠ 2 [COMPUTATIONAL EVIDENCE].", s.euler));
            }
        } else if s.euler != 1 {
            out.push(format!("Layer {k} is not a disk (χ = {}) [COMPUTATIONAL EVIDENCE].", s.euler));
        }
    }
    out
}

/// Incremental O(N) per-layer topological readout (`V`, `E`, `F`, `∂V`, `∂E`, `χ = V − E + F`) for polygon faces
/// with `p >= 3` (where distinct edges have distinct endpoint pairs).
pub(crate) fn patch_layer_stats(faces: &[Vec<usize>], face_shell: &[usize], max_layer: usize) -> Vec<LayerStats> {
    if faces.is_empty() {
        return Vec::new();
    }
    let max_vid = faces.iter().flatten().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_layer + 1];
    for (fi, &s) in face_shell.iter().enumerate().take(faces.len()) {
        if s <= max_layer {
            buckets[s].push(fi);
        }
    }
    let mut seen_v = vec![false; max_vid + 1];
    let mut v_bnd_deg = vec![0u32; max_vid + 1];
    let mut edge_deg: HashMap<u64, u8> = HashMap::new();
    let (mut n_verts, mut n_faces, mut bnd_edges, mut bnd_verts) = (0usize, 0usize, 0usize, 0usize);
    let mut out = Vec::with_capacity(max_layer + 1);
    for bucket in &buckets {
        if bucket.is_empty() && n_faces == 0 {
            continue;
        }
        for &fi in bucket {
            let f = &faces[fi];
            if f.len() < 2 {
                continue;
            }
            n_faces += 1;
            for &v in f {
                if !seen_v[v] {
                    seen_v[v] = true;
                    n_verts += 1;
                }
            }
            for m in 0..f.len() {
                let (u, w) = (f[m], f[(m + 1) % f.len()]);
                if u == w {
                    continue;
                }
                let (a, b) = (u.min(w), u.max(w));
                let key = ((a as u64) << 32) | (b as u64);
                let deg = edge_deg.entry(key).or_insert(0);
                *deg = deg.saturating_add(1);
                if *deg == 1 {
                    bnd_edges += 1;
                    for x in [a, b] {
                        if v_bnd_deg[x] == 0 {
                            bnd_verts += 1;
                        }
                        v_bnd_deg[x] += 1;
                    }
                } else if *deg == 2 {
                    bnd_edges -= 1;
                    for x in [a, b] {
                        v_bnd_deg[x] -= 1;
                        if v_bnd_deg[x] == 0 {
                            bnd_verts -= 1;
                        }
                    }
                }
            }
        }
        let n_edges = edge_deg.len();
        out.push(LayerStats {
            cells: n_faces,
            new_cells: bucket.len(),
            boundary_vertices: bnd_verts,
            boundary_edges: bnd_edges,
            euler: n_verts as i64 - n_edges as i64 + n_faces as i64,
        });
    }
    out
}

/// Unit-edge `Cell2D` of the three regular Euclidean tilings `{4,4}`, `{3,6}`, `{6,3}`.
fn regular_e2_cell(p: usize, q: usize) -> Result<Cell2D, String> {
    match (p, q) {
        (4, 4) => Ok(Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
            ],
        }),
        (3, 6) => {
            let h = 3.0_f64.sqrt() * 0.5;
            Ok(Cell2D {
                a1: [1.0, 0.0],
                a2: [0.5, h],
                basis: vec![[0.0, 0.0]],
                bonds: vec![
                    Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
                    Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
                    Bond2D { i: 0, j: 0, n1: -1, n2: 1, kind: 0 },
                ],
            })
        }
        (6, 3) => {
            let s3 = 3.0_f64.sqrt();
            Ok(Cell2D {
                a1: [s3, 0.0],
                a2: [0.5 * s3, 1.5],
                basis: vec![[0.0, 0.0], [0.0, 1.0]],
                bonds: vec![
                    Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
                    Bond2D { i: 0, j: 1, n1: 0, n2: -1, kind: 0 },
                    Bond2D { i: 0, j: 1, n1: 1, n2: -1, kind: 0 },
                ],
            })
        }
        _ => Err(format!("{{{p},{q}}} is not a regular Euclidean tiling")),
    }
}

/// Build `{4,4}`, `{3,6}`, or `{6,3}` from `Vertex`, `Edge`, or `Cell` seed via `PlaneTiling` touching-rule growth.
pub(crate) fn build_e2_regular_seeded(
    p: usize,
    q: usize,
    layers: usize,
    seed: SeedKind,
) -> Result<(UnifiedPqTiling, Vec<LayerStats>), String> {
    let cell = regular_e2_cell(p, q)?;
    let tiling = PlaneTiling::new(&cell)?;
    let infl_seed = match seed {
        SeedKind::Vertex => InflationSeed::VertexStar(0),
        SeedKind::Edge => {
            let b = &cell.bonds[0];
            InflationSeed::Edge((b.i, [0, 0]), (b.j, [b.n1, b.n2]))
        }
        SeedKind::Cell | SeedKind::Face => InflationSeed::Face(0),
        SeedKind::Dirichlet => InflationSeed::Dirichlet,
    };
    let res = inflate_with(&cell, infl_seed, &InflationRule::Bdf, layers)?;

    let mut vert_ids: BTreeMap<VertexId, usize> = BTreeMap::new();
    let mut coords_2d: Vec<[f64; 2]> = Vec::new();
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(res.faces.len());
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();

    for f in &res.faces {
        let v_ids = tiling.face_vertices((f.orbit, f.offset));
        let mut cyc = Vec::with_capacity(v_ids.len());
        for (vid, &pt) in v_ids.into_iter().zip(&f.polygon) {
            let idx = *vert_ids.entry(vid).or_insert_with(|| {
                let id = coords_2d.len();
                coords_2d.push(pt);
                id
            });
            cyc.push(idx);
        }
        for m in 0..cyc.len() {
            let (u, v) = (cyc[m], cyc[(m + 1) % cyc.len()]);
            if u != v {
                edge_set.insert((u.min(v), u.max(v)));
            }
        }
        faces.push(cyc);
    }

    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let coords_3d: Vec<[f64; 3]> = coords_2d.iter().map(|&[x, y]| [x, y, 0.0]).collect();
    let edge_arcs_2d: Vec<Vec<[f64; 2]>> = edges.iter().map(|&(u, v)| vec![coords_2d[u], coords_2d[v]]).collect();
    let edge_arcs_3d: Vec<Vec<[f64; 3]>> = edges.iter().map(|&(u, v)| vec![coords_3d[u], coords_3d[v]]).collect();
    let (n_v, n_e, n_f) = (coords_2d.len(), edges.len(), faces.len());

    let stats: Vec<LayerStats> = res
        .shells
        .iter()
        .map(|s| LayerStats {
            cells: s.topology.faces,
            new_cells: s.new_tiles,
            boundary_vertices: s.boundary_vertex_types.values().sum(),
            boundary_edges: s.topology.boundary_edges,
            euler: s.topology.chi,
        })
        .collect();

    let growth = PqGrowth {
        seed: Some(seed),
        face_layer: res.faces.iter().map(|f| Some(f.shell)).collect(),
        layer_counts: stats.iter().map(|s| s.cells).collect(),
        layer_euler: stats.iter().map(|s| s.euler).collect(),
        stopped: None,
    };

    let tiling_out = UnifiedPqTiling {
        p,
        q,
        layers,
        delta: 0,
        geometry_kind: GeometryKind::Euclidean,
        curvature_kappa: 0.0,
        euler_chi: n_v as i64 - n_e as i64 + n_f as i64,
        n_vertices: n_v,
        n_edges: n_e,
        n_faces: n_f,
        coords_3d,
        coords_2d,
        edges,
        edge_arcs_2d,
        edge_arcs_3d,
        faces,
        provenance: "[PROVED]".to_string(),
        growth,
    };
    Ok((tiling_out, stats))
}
