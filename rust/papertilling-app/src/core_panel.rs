use std::collections::{HashMap, HashSet};

use papertilling_core::defect::combine_types::{CoreGalleryEntry, EquivalentSingleReport};

use crate::canvas2d::{draw_dashed_egui, to_color32};
use crate::scene2d::{
    CirclePrim, LinePrim, PolyPrim, ScenePrimitives, TextPrim, bond_palette, compute_bbox,
};

pub const CHORD_RGBA: [u8; 4] = [0xd8, 0x1b, 0x60, 255]; // #d81b60 magenta
pub const UNDER_SITE_RGBA: [u8; 4] = [0xc6, 0x28, 0x28, 255]; // #c62828 red
pub const ODD_FACE_RGBA: [u8; 4] = [0xf2, 0xa3, 0x3a, 195]; // #f2a33a warm orange
pub const SPLIT_FACE_RGBA: [u8; 4] = [0xf7, 0xd9, 0xa8, 195]; // #f7d9a8 light amber
pub const CONFIG_FACE_RGBA: [u8; 4] = [0x8f, 0xb8, 0xe8, 195]; // #8fb8e8 blue
pub const MATCH_BORDER_RGB: egui::Color32 = egui::Color32::from_rgb(0x2e, 0x7d, 0x32); // #2e7d32

#[derive(Default)]
pub struct CorePanelState {
    pub collapsed: bool,
    pub show_all_completions: bool,
}

pub(crate) fn compile_core_card(entry: &CoreGalleryEntry) -> ScenePrimitives {
    let crystal = &entry.crystal;
    let n = crystal.pos.len();
    let mut polygons = Vec::new();
    let mut lines = Vec::new();
    let mut circles = Vec::new();
    let mut labels = Vec::new();

    let perfect_set: HashSet<usize> = entry.perfect_rings.iter().copied().collect();
    for face in &entry.faces {
        if face.len() < 3 || face.iter().any(|&u| u >= n) {
            continue;
        }
        let pts: Vec<[f32; 2]> = face
            .iter()
            .map(|&u| [crystal.pos[u][0] as f32, crystal.pos[u][1] as f32])
            .collect();
        let fill = if entry.kind == "configuration" {
            CONFIG_FACE_RGBA
        } else if perfect_set.contains(&face.len()) {
            SPLIT_FACE_RGBA
        } else {
            ODD_FACE_RGBA
        };
        polygons.push(PolyPrim {
            points: pts.clone(),
            fill,
            stroke: Some(([110, 118, 130, 95], 0.7)),
        });
        let kf = pts.len() as f32;
        let cx = pts.iter().map(|p| p[0]).sum::<f32>() / kf;
        let cy = pts.iter().map(|p| p[1]).sum::<f32>() / kf;
        labels.push(TextPrim {
            pos: [cx, cy],
            text: face.len().to_string(),
            color: [45, 50, 58, 235],
            size_px: 9.5,
        });
    }

    let chord_set: HashSet<(usize, usize)> = entry
        .chords
        .iter()
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();

    let mut edge_kind: HashMap<(usize, usize), usize> = HashMap::new();
    for (u, nbrs) in crystal.adj.iter().enumerate() {
        for d in nbrs {
            if let Some(stub) = d.ghost_stub
                && let Some(&pu) = crystal.pos.get(u)
            {
                lines.push(LinePrim {
                    a: [pu[0] as f32, pu[1] as f32],
                    b: [(pu[0] + stub[0]) as f32, (pu[1] + stub[1]) as f32],
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
        if chord_set.contains(&key) {
            continue;
        }
        let kind = edge_kind.get(&key).copied().unwrap_or(0);
        let p0 = crystal.pos[u];
        let p1 = crystal.pos[v];
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

fn compute_zoomed_bbox(
    crystal: &papertilling_core::types::BuiltCrystal,
    core_sites: &[usize],
    plain_lengths: &mut [f64],
    circles: &[CirclePrim],
    lines: &[LinePrim],
) -> ([f32; 2], [f32; 2]) {
    let valid_core: Vec<[f64; 2]> = core_sites
        .iter()
        .filter_map(|&u| crystal.pos.get(u).copied())
        .collect();
    if valid_core.is_empty() {
        return compute_bbox(
            circles
                .iter()
                .map(|c| c.center)
                .chain(lines.iter().flat_map(|l| [l.a, l.b])),
        );
    }
    plain_lengths.sort_by(f64::total_cmp);
    let unit = if plain_lengths.is_empty() {
        1.0
    } else {
        plain_lengths[plain_lengths.len() / 2]
    };
    let k = valid_core.len() as f64;
    let cx = valid_core.iter().map(|p| p[0]).sum::<f64>() / k;
    let cy = valid_core.iter().map(|p| p[1]).sum::<f64>() / k;
    let max_d = valid_core
        .iter()
        .map(|p| (p[0] - cx).hypot(p[1] - cy))
        .fold(0.0_f64, f64::max);
    let half = (max_d + 2.35 * unit).max(1.2 * unit) as f32;
    ([cx as f32 - half, cy as f32 - half], [cx as f32 + half, cy as f32 + half])
}

pub fn show_core_panel(
    state: &mut CorePanelState,
    ui: &mut egui::Ui,
    compare_res: Option<&Result<EquivalentSingleReport, String>>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Side-by-Side Core Gallery");
        if ui
            .small_button(if state.collapsed { "Expand ▲" } else { "Collapse ▼" })
            .clicked()
        {
            state.collapsed = !state.collapsed;
        }
        if !state.collapsed {
            ui.separator();
            legend_chip(ui, to_color32(CONFIG_FACE_RGBA), "Config core");
            legend_chip(ui, to_color32(ODD_FACE_RGBA), "Odd ring");
            legend_chip(ui, to_color32(SPLIT_FACE_RGBA), "Perfect ring");
            legend_chip(ui, to_color32(CHORD_RGBA), "Chord (-- #d81b60)");
            legend_chip(ui, to_color32(UNDER_SITE_RGBA), "Under-coord (○)");
            legend_chip(ui, MATCH_BORDER_RGB, "MATCH");
        }
    });

    if state.collapsed {
        return;
    }
    ui.add_space(3.0);

    let Some(res) = compare_res else {
        ui.small(
            "Click 'Compare with Single Defect' (or enable Auto-compare) to view Configuration, Volterra, and Completion cores side by side.",
        );
        return;
    };
    let report = match res {
        Ok(rep) => rep,
        Err(err) => {
            ui.colored_label(
                egui::Color32::from_rgb(176, 86, 18),
                format!("Core gallery status: {err}"),
            );
            return;
        }
    };

    let match_set: HashSet<&str> = report.matches.iter().map(String::as_str).collect();
    let volterra: Vec<&CoreGalleryEntry> = report
        .gallery
        .iter()
        .filter(|e| e.kind == "Volterra")
        .collect();
    let mut completions: Vec<&CoreGalleryEntry> = report
        .gallery
        .iter()
        .filter(|e| e.kind != "Volterra")
        .collect();
    completions.sort_by_key(|e| !match_set.contains(e.name.as_str()));

    let comp_limit = if state.show_all_completions {
        completions.len()
    } else {
        completions.len().min(6)
    };

    ui.horizontal(|ui| {
        let v_col = if report.equivalent {
            MATCH_BORDER_RGB
        } else {
            egui::Color32::from_rgb(196, 108, 12)
        };
        ui.colored_label(v_col, egui::RichText::new(&report.verdict).strong());
        if completions.len() > 6 {
            ui.checkbox(
                &mut state.show_all_completions,
                format!("Show all {} completion cores", completions.len()),
            );
        }
    });

    egui::ScrollArea::horizontal()
        .id_salt("core_gallery_scroll")
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(cfg_entry) = &report.config_core {
                    draw_core_card(ui, cfg_entry, "Configuration Core", false);
                    if !volterra.is_empty() || !completions.is_empty() {
                        ui.separator();
                    }
                }
                for entry in &volterra {
                    let is_match = match_set.contains(entry.name.as_str());
                    draw_core_card(ui, entry, "Volterra", is_match);
                }
                if !volterra.is_empty() && comp_limit > 0 {
                    ui.separator();
                }
                for entry in completions.iter().take(comp_limit) {
                    let is_match = match_set.contains(entry.name.as_str());
                    draw_core_card(ui, entry, "Completion", is_match);
                }
                if report.config_core.is_none() && volterra.is_empty() && completions.is_empty() {
                    ui.label(format!(
                        "Charge kind '{}': {} candidate(s) checked ({})",
                        report.kind,
                        report.candidates.len(),
                        if report.matches.is_empty() {
                            "no match"
                        } else {
                            "matched"
                        }
                    ));
                }
            });
        });
}

fn legend_chip(ui: &mut egui::Ui, color: egui::Color32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.small(text);
}

fn draw_core_card(
    ui: &mut egui::Ui,
    entry: &CoreGalleryEntry,
    group_tag: &str,
    is_match: bool,
) {
    let card_w = 184.0_f32;
    let card_h = 204.0_f32;
    let (card_rect, resp) =
        ui.allocate_exact_size(egui::vec2(card_w, card_h), egui::Sense::hover());

    let mut tip = format!("{} ({})\n{}", entry.name, group_tag, entry.description);
    if !entry.also.is_empty() {
        tip.push_str(&format!("\nAlso reached by: {}", entry.also.join(", ")));
    }
    resp.on_hover_text(tip);

    let bg = if is_match {
        egui::Color32::from_rgb(244, 251, 245)
    } else {
        egui::Color32::from_rgb(250, 251, 253)
    };
    let painter = ui.painter_at(card_rect);
    painter.rect_filled(card_rect, 5.0, bg);

    painter.text(
        card_rect.left_top() + egui::vec2(7.0, 5.0),
        egui::Align2::LEFT_TOP,
        &entry.name,
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(30, 38, 48),
    );
    if is_match {
        let badge = egui::Rect::from_min_size(
            egui::pos2(card_rect.right() - 52.0, card_rect.top() + 4.0),
            egui::vec2(46.0, 15.0),
        );
        painter.rect_filled(badge, 3.0, MATCH_BORDER_RGB);
        painter.text(
            badge.center(),
            egui::Align2::CENTER_CENTER,
            "MATCH",
            egui::FontId::proportional(9.5),
            egui::Color32::WHITE,
        );
    }

    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(card_rect.left() + 6.0, card_rect.top() + 22.0),
        egui::pos2(card_rect.right() - 6.0, card_rect.bottom() - 28.0),
    );
    paint_card_primitives(&ui.painter_at(plot_rect), plot_rect, &compile_core_card(entry));

    let rings_txt = if entry.rings.is_empty() {
        group_tag.to_string()
    } else {
        let r_str: Vec<String> = entry.rings.iter().map( usize::to_string).collect();
        format!("rings [{}] · {}", r_str.join(","), entry.description)
    };
    let short_sub = if rings_txt.chars().count() > 30 {
        let s: String = rings_txt.chars().take(29).collect();
        format!("{s}…")
    } else {
        rings_txt
    };
    painter.text(
        egui::pos2(card_rect.left() + 6.0, card_rect.bottom() - 15.0),
        egui::Align2::LEFT_CENTER,
        short_sub,
        egui::FontId::proportional(9.5),
        egui::Color32::from_rgb(76, 84, 96),
    );

    let border = if is_match {
        egui::Stroke::new(3.0, MATCH_BORDER_RGB)
    } else {
        egui::Stroke::new(1.0, egui::Color32::from_gray(192))
    };
    painter.rect_stroke(card_rect, 5.0, border, egui::StrokeKind::Inside);
}

fn paint_card_primitives(painter: &egui::Painter, rect: egui::Rect, prims: &ScenePrimitives) {
    painter.rect_filled(rect, 3.0, egui::Color32::WHITE);
    let (bmin, bmax) = prims.bbox;
    let wc = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
    let sx = (bmax[0] - bmin[0]).max(0.5);
    let sy = (bmax[1] - bmin[1]).max(0.5);
    let zoom = ((rect.width() - 10.0) / sx)
        .min((rect.height() - 10.0) / sy)
        .max(4.0);
    let c = rect.center();
    let to_screen = |p: [f32; 2]| -> egui::Pos2 {
        egui::pos2(c.x + (p[0] - wc[0]) * zoom, c.y - (p[1] - wc[1]) * zoom)
    };

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
                painter.add(egui::Shape::convex_polygon(
                    vec![cp, pts[m], pts[(m + 1) % pts.len()]],
                    fill,
                    egui::Stroke::NONE,
                ));
            }
        }
    }
    for ln in &prims.lines {
        let p0 = to_screen(ln.a);
        let p1 = to_screen(ln.b);
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
            painter.circle_stroke(
                pos,
                r,
                egui::Stroke::new(circ.stroke.1, to_color32(circ.stroke.0)),
            );
        }
    }
    for lbl in &prims.labels {
        painter.text(
            to_screen(lbl.pos),
            egui::Align2::CENTER_CENTER,
            &lbl.text,
            egui::FontId::proportional(lbl.size_px),
            to_color32(lbl.color),
        );
    }
    painter.rect_stroke(
        rect,
        3.0,
        egui::Stroke::new(0.8, egui::Color32::from_gray(215)),
        egui::StrokeKind::Inside,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AppBackend;
    use crate::live_ops::build_pristine_disk;
    use papertilling_core::defect::core_graph::LatticeCoreInfo;
    use papertilling_core::plane::family::LatticeFamily;

    #[test]
    fn compile_core_card_renders_faces_chords_and_under_sites() {
        let backend = AppBackend::new();
        let sq = backend.find_named("square").unwrap();
        let cell = LatticeFamily::from_recipe(&sq.recipe, Some(sq.frame)).unwrap().cell();
        let crystal = build_pristine_disk(&LatticeCoreInfo::new(&cell), 2.6);
        let entry = CoreGalleryEntry {
            name: String::from("C1"),
            kind: String::from("completion"),
            crystal,
            core_sites: vec![0, 1, 2],
            chords: vec![(0, 1)],
            under_sites: vec![2],
            rings: vec![5, 7],
            faces: vec![vec![0, 1, 2]],
            under: vec![3],
            added: 1,
            description: String::from("test completion"),
            also: vec![String::from("V1")],
            faces_ok: true,
            iso_ok: Some(true),
            perfect_rings: vec![4],
        };
        let prims = compile_core_card(&entry);
        assert_eq!(prims.polygons.len(), 1);
        assert_eq!(prims.polygons[0].fill, ODD_FACE_RGBA);
        assert!(prims.lines.iter().any(|l| l.dashed && l.color == CHORD_RGBA));
        assert!(prims.circles.iter().any(|c| c.stroke.0 == UNDER_SITE_RGBA));
    }
}
