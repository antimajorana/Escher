use std::collections::{HashMap, HashSet};

use papertilling_core::types::BuiltCrystal;

use crate::scene2d::{
    CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, bond_palette, compute_bbox,
    ring_tint_rgba, site_palette,
};

/// Bonds drawn as two dotted stubs instead of in full (as `Defect.plot` and `drawing_check.drawn`): a bond is a
/// ghost bond when either of its directions carries a `ghost_stub` (stretched bonds, Möbius seam bonds).
pub(crate) fn ghost_bonds(crystal: &BuiltCrystal) -> HashSet<(usize, usize)> {
    let mut out = HashSet::new();
    for (u, nbrs) in crystal.adj.iter().enumerate() {
        for d in nbrs.iter().filter(|d| d.ghost_stub.is_some()) {
            out.insert((u.min(d.to), u.max(d.to)));
        }
    }
    out
}

fn push_stub(lines: &mut Vec<LinePrim>, circles: &mut Vec<CirclePrim>, a: [f64; 2], b: [f64; 2]) {
    let (a, b) = ([a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]);
    lines.push(LinePrim { a, b, color: [184, 38, 134, 235], width: 2.0, dashed: true, arrow_mid: false });
    circles.push(CirclePrim {
        center: b,
        radius_world: 0.045,
        min_px: 2.0,
        max_px: 4.2,
        fill: [255, 255, 255, 220],
        stroke: ([184, 38, 134, 235], 1.2),
    });
}

pub(crate) fn compile_crystal_patch(crystal: &BuiltCrystal, title: &str, note: &str) -> ScenePrimitives {
    let mut polygons = Vec::new();
    let mut lines = Vec::new();
    let mut circles = Vec::new();
    let mut labels = Vec::new();
    let ghosts = ghost_bonds(crystal);

    for (ring_pts, is_core) in extract_crystal_faces(crystal, &ghosts) {
        if is_core && !ring_pts.is_empty() {
            let k = ring_pts.len() as f32;
            let cx = ring_pts.iter().map(|p| p[0]).sum::<f32>() / k;
            let cy = ring_pts.iter().map(|p| p[1]).sum::<f32>() / k;
            labels.push(TextPrim {
                pos: [cx, cy],
                text: ring_pts.len().to_string(),
                color: [42, 48, 58, 225],
                size_px: 10.5,
            });
        }
        let fill = if is_core {
            ring_tint_rgba(ring_pts.len())
        } else {
            [235, 240, 246, 85]
        };
        let stroke = is_core.then_some(([120, 128, 142, 110], 0.8));
        polygons.push(PolyPrim {
            points: ring_pts,
            fill,
            stroke,
        });
    }

    let mut edge_meta: HashMap<(usize, usize), (usize, bool)> = HashMap::new();
    for (u, nbrs) in crystal.adj.iter().enumerate() {
        for d in nbrs {
            let key = if u < d.to { (u, d.to) } else { (d.to, u) };
            let entry = edge_meta.entry(key).or_insert((d.kind, false));
            entry.1 |= d.across;
        }
    }

    let n_pos = crystal.pos.len();
    for &(u, v) in &crystal.edges {
        if u >= n_pos || v >= n_pos {
            continue;
        }
        let key = if u < v { (u, v) } else { (v, u) };
        if ghosts.contains(&key) {
            let (pu, pv) = (crystal.pos[u], crystal.pos[v]);
            let mid = [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1])];
            let stub = |a: usize, b: usize| crystal.adj[a].iter().find(|d| d.to == b).and_then(|d| d.ghost_stub).unwrap_or(mid);
            push_stub(&mut lines, &mut circles, pu, stub(u, v));
            push_stub(&mut lines, &mut circles, pv, stub(v, u));
            continue;
        }
        let (kind, across) = edge_meta.get(&key).copied().unwrap_or((0, false));
        let (color, width) = if across {
            ([222, 86, 24, 255], 2.35)
        } else {
            (bond_palette(kind), 1.7)
        };
        lines.push(LinePrim {
            a: [crystal.pos[u][0] as f32, crystal.pos[u][1] as f32],
            b: [crystal.pos[v][0] as f32, crystal.pos[v][1] as f32],
            color,
            width,
            dashed: false,
            arrow_mid: false,
        });
    }

    let cone_set: HashSet<usize> = crystal.cone_sites.iter().copied().collect();
    let anom_deg = interior_anomalous_degrees(crystal);
    for (u, p) in crystal.pos.iter().enumerate() {
        let pos = [p[0] as f32, p[1] as f32];
        let side = crystal.side.get(u).copied().flatten();
        let basis = crystal.basis_idx.get(u).copied().unwrap_or(0);
        let fill = match side {
            Some('A') => [34, 108, 196, 255],
            Some('B') => [26, 158, 114, 255],
            _ => site_palette(basis),
        };
        if cone_set.contains(&u) {
            circles.push(CirclePrim {
                center: pos,
                radius_world: 0.145,
                min_px: 5.5,
                max_px: 11.5,
                fill: [235, 52, 38, 75],
                stroke: ([224, 36, 28, 255], 2.1),
            });
        } else if let Some(&is_under) = anom_deg.get(&u) {
            let scol = if is_under {
                [198, 40, 40, 235]
            } else {
                [106, 27, 154, 235]
            };
            circles.push(CirclePrim {
                center: pos,
                radius_world: 0.135,
                min_px: 5.0,
                max_px: 10.0,
                fill: [0, 0, 0, 0],
                stroke: (scol, 1.8),
            });
        }
        circles.push(CirclePrim {
            center: pos,
            radius_world: 0.072,
            min_px: 2.5,
            max_px: 6.6,
            fill: if cone_set.contains(&u) { [228, 38, 30, 255] } else { fill },
            stroke: ([18, 32, 48, 255], 1.1),
        });
    }

    let bbox = compute_bbox(
        circles
            .iter()
            .map(|c| c.center)
            .chain(lines.iter().flat_map(|l| [l.a, l.b])),
    );
    ScenePrimitives {
        site_count: crystal.pos.len(),
        bond_count: crystal.edges.len(),
        polygons,
        lines,
        circles,
        labels,
        title: title.to_owned(),
        note: note.to_owned(),
        bbox,
    }
}

fn interior_anomalous_degrees(crystal: &BuiltCrystal) -> HashMap<usize, bool> {
    let n = crystal.pos.len();
    if n < 12 || crystal.wall_sigma.is_some() || crystal.is_non_orientable {
        return HashMap::new();
    }
    let nf = n as f64;
    let cx = crystal.pos.iter().map(|p| p[0]).sum::<f64>() / nf;
    let cy = crystal.pos.iter().map(|p| p[1]).sum::<f64>() / nf;
    let max_r = crystal
        .pos
        .iter()
        .map(|p| (p[0] - cx).hypot(p[1] - cy))
        .fold(0.0_f64, f64::max);
    let inner_cut = 0.70 * max_r;

    let mut deg = vec![0_usize; n];
    for &(u, v) in &crystal.edges {
        if u < n && v < n {
            deg[u] += 1;
            deg[v] += 1;
        }
    }
    let mut hist: HashMap<(usize, usize), usize> = HashMap::new();
    for u in 0..n {
        if (crystal.pos[u][0] - cx).hypot(crystal.pos[u][1] - cy) <= inner_cut {
            let b = crystal.basis_idx.get(u).copied().unwrap_or(0);
            *hist.entry((b, deg[u])).or_insert(0) += 1;
        }
    }
    let mut modal: HashMap<usize, (usize, usize)> = HashMap::new();
    for ((b, d), cnt) in hist {
        let entry = modal.entry(b).or_insert((d, 0));
        if cnt > entry.1 {
            *entry = (d, cnt);
        }
    }
    let mut out = HashMap::new();
    for u in 0..n {
        if (crystal.pos[u][0] - cx).hypot(crystal.pos[u][1] - cy) <= inner_cut {
            let b = crystal.basis_idx.get(u).copied().unwrap_or(0);
            if let Some(&(z0, _)) = modal.get(&b)
                && deg[u] != z0
            {
                out.insert(u, deg[u] < z0);
            }
        }
    }
    out
}

/// Faces of the drawn picture (ghost bonds are not drawn, so they bound no face), each flagged core or bulk
/// by its distance from the centroid of the sites.
fn extract_crystal_faces(crystal: &BuiltCrystal, ghosts: &HashSet<(usize, usize)>) -> Vec<(Vec<[f32; 2]>, bool)> {
    let n = crystal.pos.len();
    if n < 3 || crystal.edges.is_empty() {
        return Vec::new();
    }
    let centroid = [
        (crystal.pos.iter().map(|p| p[0]).sum::<f64>() / n as f64) as f32,
        (crystal.pos.iter().map(|p| p[1]).sum::<f64>() / n as f64) as f32,
    ];
    let mut lengths = Vec::with_capacity(crystal.edges.len());
    let mut nbrs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(u, v) in &crystal.edges {
        if u < n && v < n && u != v && !ghosts.contains(&(u.min(v), u.max(v))) {
            let dx = crystal.pos[v][0] - crystal.pos[u][0];
            let dy = crystal.pos[v][1] - crystal.pos[u][1];
            let len = dx.hypot(dy);
            if len > 1e-6 {
                lengths.push(len);
                nbrs[u].push((v, dy.atan2(dx)));
                nbrs[v].push((u, (-dy).atan2(-dx)));
            }
        }
    }
    if lengths.is_empty() {
        return Vec::new();
    }
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let med_len = lengths[lengths.len() / 2];

    for list in &mut nbrs {
        list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        list.dedup_by_key(|x| x.0);
    }

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut raw_faces: Vec<(Vec<[f32; 2]>, f32)> = Vec::new();
    let mut max_face_r = 0.0_f32;

    for u0 in 0..n {
        for &(v0, _) in &nbrs[u0] {
            if visited.contains(&(u0, v0)) {
                continue;
            }
            let mut cycle = Vec::new();
            let (mut a, mut b) = (u0, v0);
            let mut valid = true;
            for _ in 0..14 {
                visited.insert((a, b));
                cycle.push(a);
                let list = &nbrs[b];
                if list.len() < 2 {
                    valid = false;
                    break;
                }
                let Some(pos_in_b) = list.iter().position(|&(w, _)| w == a) else {
                    valid = false;
                    break;
                };
                // Next clockwise incoming = previous angle in CCW order -> minimal interior left face
                let next_idx = if pos_in_b == 0 { list.len() - 1 } else { pos_in_b - 1 };
                let c = list[next_idx].0;
                a = b;
                b = c;
                if (a, b) == (u0, v0) {
                    break;
                }
            }
            if !valid || (a, b) != (u0, v0) || cycle.len() < 3 || cycle.len() > 12 {
                continue;
            }
            let mut area2 = 0.0_f64;
            let mut long_edge = false;
            let mut cx = 0.0_f32;
            let mut cy = 0.0_f32;
            let mut pts = Vec::with_capacity(cycle.len());
            for m in 0..cycle.len() {
                let p = crystal.pos[cycle[m]];
                let q = crystal.pos[cycle[(m + 1) % cycle.len()]];
                area2 += p[0] * q[1] - q[0] * p[1];
                if (q[0] - p[0]).hypot(q[1] - p[1]) > med_len * 2.6 {
                    long_edge = true;
                }
                pts.push([p[0] as f32, p[1] as f32]);
                cx += p[0] as f32 / (cycle.len() as f32);
                cy += p[1] as f32 / (cycle.len() as f32);
            }
            if area2 > 1e-5 && !long_edge {
                let r = (cx - centroid[0]).hypot(cy - centroid[1]);
                max_face_r = max_face_r.max(r);
                raw_faces.push((pts, r));
            }
        }
    }

    let mut outer_counts: HashMap<usize, usize> = HashMap::new();
    let mut total_outer = 0_usize;
    let mut all_counts: HashMap<usize, usize> = HashMap::new();
    for (pts, r) in &raw_faces {
        *all_counts.entry(pts.len()).or_insert(0) += 1;
        if *r >= 0.52 * max_face_r {
            *outer_counts.entry(pts.len()).or_insert(0) += 1;
            total_outer += 1;
        }
    }
    let modal_len = all_counts
        .iter()
        .max_by_key(|&(_, &c)| c)
        .map_or(6, |(&k, _)| k);

    raw_faces
        .into_iter()
        .map(|(pts, _)| {
            let k = pts.len();
            let is_bulk = if total_outer >= 4 {
                outer_counts.get(&k).copied().unwrap_or(0) * 8 >= total_outer
            } else {
                k == modal_len
            };
            (pts, !is_bulk)
        })
        .collect()
}
