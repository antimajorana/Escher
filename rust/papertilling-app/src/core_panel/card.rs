//! One core card: drawing primitives of a `CoreGalleryEntry` (faces, bonds, dashed chords, under-coordinated
//! sites, zoomed on the core) and their painting into a card rectangle.

use std::collections::{HashMap, HashSet};

use papertilling_core::defect::combine_types::CoreGalleryEntry;

use super::{CHORD_RGBA, CONFIG_FACE_RGBA, ODD_FACE_RGBA, SPLIT_FACE_RGBA, UNDER_SITE_RGBA};
use crate::canvas2d::{draw_dashed_egui, to_color32};
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, bond_palette, compute_bbox};

pub(crate) fn compile_core_card(entry: &CoreGalleryEntry) -> ScenePrimitives {
    let crystal = &entry.crystal;
    let n = crystal.pos.len();
    let mut polygons = Vec::new();
    let mut lines = Vec::new();
    let mut circles = Vec::new();
    let mut labels = Vec::new();
    let ghosts = crate::scene_crystal::ghost_bonds(crystal);

    let perfect_set: HashSet<usize> = entry.perfect_rings.iter().copied().collect();
    for face in &entry.faces {
        if face.len() < 3 || face.iter().any(|&u| u >= n)
            || (0..face.len()).any(|i| {
                let (a, b) = (face[i], face[(i + 1) % face.len()]);
                ghosts.contains(&(a.min(b), a.max(b)))
            }) {
            continue;
        }
        let pts: Vec<[f32; 2]> = face.iter().map(|&u| [crystal.pos[u][0] as f32, crystal.pos[u][1] as f32]).collect();
        let fill = if entry.kind == "configuration" {
            CONFIG_FACE_RGBA
        } else if perfect_set.contains(&face.len()) {
            SPLIT_FACE_RGBA
        } else {
            ODD_FACE_RGBA
        };
        polygons.push(PolyPrim { points: pts.clone(), fill, stroke: Some(([110, 118, 130, 95], 0.7)) });
        let kf = pts.len() as f32;
        let cx = pts.iter().map(|p| p[0]).sum::<f32>() / kf;
        let cy = pts.iter().map(|p| p[1]).sum::<f32>() / kf;
        labels.push(TextPrim { pos: [cx, cy], text: face.len().to_string(), color: [45, 50, 58, 235], size_px: 9.5 });
    }

    let chord_set: HashSet<(usize, usize)> = entry.chords.iter().map(|&(u, v)| if u < v { (u, v) } else { (v, u) }).collect();

    let mut edge_kind: HashMap<(usize, usize), usize> = HashMap::new();
    for (u, nbrs) in crystal.adj.iter().enumerate() {
        for d in nbrs {
            if let Some(stub) = d.ghost_stub
                && let Some(&pu) = crystal.pos.get(u)
            {
                lines.push(LinePrim {
                    a: [pu[0] as f32, pu[1] as f32],
                    b: [stub[0] as f32, stub[1] as f32],
                    color: [136, 136, 136, 210],
                    width: 1.1,
                    dashed: true,
                    arrow_mid: false,
                });
                continue;
            }
            let key = if u < d.to { (u, d.to) } else { (d.to, u) };
            edge_kind.entry(key).or_insert(d.kind);
        }
    }

    let mut plain_lengths: Vec<f64> = Vec::new();
    for &(u, v) in &crystal.edges {
        if u >= n || v >= n {
            continue;
        }
        let key = if u < v { (u, v) } else { (v, u) };
        if chord_set.contains(&key) || ghosts.contains(&key) {
            continue;
        }
        let kind = edge_kind.get(&key).copied().unwrap_or(0);
        let (p0, p1) = (crystal.pos[u], crystal.pos[v]);
        let len = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
        if len > 1e-6 {
            plain_lengths.push(len);
        }
        lines.push(LinePrim {
            a: [p0[0] as f32, p0[1] as f32],
            b: [p1[0] as f32, p1[1] as f32],
            color: bond_palette(kind),
            width: 1.35,
            dashed: false,
            arrow_mid: false,
        });
    }

    let mut ordered_chords: Vec<(usize, usize)> = chord_set.into_iter().collect();
    ordered_chords.sort_unstable();
    for (u, v) in ordered_chords {
        if u < n && v < n {
            lines.push(LinePrim {
                a: [crystal.pos[u][0] as f32, crystal.pos[u][1] as f32],
                b: [crystal.pos[v][0] as f32, crystal.pos[v][1] as f32],
                color: CHORD_RGBA,
                width: 2.5,
                dashed: true,
                arrow_mid: false,
            });
        }
    }

    for p in &crystal.pos {
        circles.push(CirclePrim {
            center: [p[0] as f32, p[1] as f32],
            radius_world: 0.055,
            min_px: 1.9,
            max_px: 3.8,
            fill: [28, 34, 44, 255],
            stroke: ([0, 0, 0, 0], 0.0),
        });
    }

    for &u in &entry.under_sites {
        if let Some(p) = crystal.pos.get(u) {
            circles.push(CirclePrim {
                center: [p[0] as f32, p[1] as f32],
                radius_world: 0.145,
                min_px: 5.2,
                max_px: 8.4,
                fill: [0, 0, 0, 0],
                stroke: (UNDER_SITE_RGBA, 1.8),
            });
        }
    }

    let bbox = compute_zoomed_bbox(crystal, &entry.core_sites, &mut plain_lengths, &circles, &lines);
    ScenePrimitives {
        site_count: n,
        bond_count: crystal.edges.len(),
        polygons,
        lines,
        circles,
        labels,
        title: entry.name.clone(),
        note: entry.description.clone(),
        bbox,
    }
}

/// Square box on the core sites, padded by 2.35 median bond lengths (Python `cores._draw_panel`, zoom_bonds).
fn compute_zoomed_bbox(
    crystal: &papertilling_core::types::BuiltCrystal,
    core_sites: &[usize],
    plain_lengths: &mut [f64],
    circles: &[CirclePrim],
    lines: &[LinePrim],
) -> ([f32; 2], [f32; 2]) {
    let valid_core: Vec<[f64; 2]> = core_sites.iter().filter_map(|&u| crystal.pos.get(u).copied()).collect();
    if valid_core.is_empty() {
        return compute_bbox(circles.iter().map(|c| c.center).chain(lines.iter().flat_map(|l| [l.a, l.b])));
    }
    plain_lengths.sort_by(f64::total_cmp);
    let unit = if plain_lengths.is_empty() { 1.0 } else { plain_lengths[plain_lengths.len() / 2] };
    let k = valid_core.len() as f64;
    let cx = valid_core.iter().map(|p| p[0]).sum::<f64>() / k;
    let cy = valid_core.iter().map(|p| p[1]).sum::<f64>() / k;
    let max_d = valid_core.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).fold(0.0_f64, f64::max);
    let half = (max_d + 2.35 * unit).max(1.2 * unit) as f32;
    ([cx as f32 - half, cy as f32 - half], [cx as f32 + half, cy as f32 + half])
}

pub(crate) fn paint_card_primitives(painter: &egui::Painter, rect: egui::Rect, prims: &ScenePrimitives) {
    painter.rect_filled(rect, 3.0, egui::Color32::WHITE);
    let (bmin, bmax) = prims.bbox;
    let wc = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
    let sx = (bmax[0] - bmin[0]).max(0.5);
    let sy = (bmax[1] - bmin[1]).max(0.5);
    let zoom = ((rect.width() - 10.0) / sx).min((rect.height() - 10.0) / sy).max(4.0);
    let c = rect.center();
    let to_screen = |p: [f32; 2]| -> egui::Pos2 { egui::pos2(c.x + (p[0] - wc[0]) * zoom, c.y - (p[1] - wc[1]) * zoom) };

    for poly in &prims.polygons {
        if poly.points.len() >= 3 {
            let pts: Vec<egui::Pos2> = poly.points.iter().copied().map(to_screen).collect();
            let fill = to_color32(poly.fill);
            let mut cp = egui::Pos2::ZERO;
            for &pt in &pts {
                cp.x += pt.x / (pts.len() as f32);
                cp.y += pt.y / (pts.len() as f32);
            }
            for m in 0..pts.len() {
                painter.add(egui::Shape::convex_polygon(vec![cp, pts[m], pts[(m + 1) % pts.len()]], fill, egui::Stroke::NONE));
            }
        }
    }
    for ln in &prims.lines {
        let (p0, p1) = (to_screen(ln.a), to_screen(ln.b));
        let stroke = egui::Stroke::new(ln.width, to_color32(ln.color));
        if ln.dashed {
            draw_dashed_egui(painter, p0, p1, stroke, 5.2, 3.4);
        } else {
            painter.line_segment([p0, p1], stroke);
        }
    }
    for circ in &prims.circles {
        let pos = to_screen(circ.center);
        let r = (circ.radius_world * zoom).clamp(circ.min_px, circ.max_px);
        if circ.fill[3] > 0 {
            painter.circle_filled(pos, r, to_color32(circ.fill));
        }
        if circ.stroke.1 > 0.0 && circ.stroke.0[3] > 0 {
            painter.circle_stroke(pos, r, egui::Stroke::new(circ.stroke.1, to_color32(circ.stroke.0)));
        }
    }
    for lbl in &prims.labels {
        painter.text(to_screen(lbl.pos), egui::Align2::CENTER_CENTER, &lbl.text, egui::FontId::proportional(lbl.size_px), to_color32(lbl.color));
    }
    painter.rect_stroke(rect, 3.0, egui::Stroke::new(0.8, egui::Color32::from_gray(215)), egui::StrokeKind::Inside);
}
