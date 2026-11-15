use std::io::Write as _;
use std::path::Path;

use crate::canvas2d::Scene2D;
use crate::scene2d::{ScenePrimitives, compile_scene};

pub fn export_scene_to_path(scene: &Scene2D, path: &Path) -> Result<String, String> {
    let is_svg = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"));
    if is_svg {
        let svg = render_scene_svg(scene, 960, 960);
        std::fs::write(path, svg).map_err(|e| e.to_string())?;
        Ok(format!("Exported SVG ({})", path.file_name().and_then(|s| s.to_str()).unwrap_or("scene.svg")))
    } else {
        let png_bytes = encode_scene_png(scene, 1024, 1024)?;
        std::fs::write(path, png_bytes).map_err(|e| e.to_string())?;
        Ok(format!("Exported high-DPI PNG ({})", path.file_name().and_then(|s| s.to_str()).unwrap_or("scene.png")))
    }
}

fn compute_transform(prims: &ScenePrimitives, width: u32, height: u32) -> ([f32; 2], [f32; 2], f32) {
    let (bmin, bmax) = prims.bbox;
    let wc = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
    let span_x = (bmax[0] - bmin[0]).max(0.5);
    let span_y = (bmax[1] - bmin[1]).max(0.5);
    let top_pad = if !prims.note.is_empty() { 76.0 } else if !prims.title.is_empty() { 48.0 } else { 36.0 };
    let avail_w = (width as f32 - 64.0).max(32.0);
    let avail_h = (height as f32 - top_pad - 36.0).max(32.0);
    let scale = (avail_w / span_x).min(avail_h / span_y);
    let sc = [width as f32 * 0.5, top_pad + avail_h * 0.5];
    (wc, sc, scale)
}

fn hex_rgb(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn render_scene_svg(scene: &Scene2D, width: u32, height: u32) -> String {
    let prims = compile_scene(scene);
    let (wc, sc, scale) = compute_transform(&prims, width, height);
    let map_pt = |p: [f32; 2]| -> (f32, f32) {
        (sc[0] + (p[0] - wc[0]) * scale, sc[1] - (p[1] - wc[1]) * scale)
    };

    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#f8f9fc\"/>\n"
    );

    for poly in &prims.polygons {
        if poly.points.len() < 3 {
            continue;
        }
        let mut pts_str = String::new();
        for (idx, &p) in poly.points.iter().enumerate() {
            let (x, y) = map_pt(p);
            if idx > 0 {
                pts_str.push(' ');
            }
            pts_str.push_str(&format!("{x:.2},{y:.2}"));
        }
        let fill_op = poly.fill[3] as f32 / 255.0;
        let stroke_attr = if let Some((scol, sw)) = poly.stroke {
            format!(
                " stroke=\"{}\" stroke-opacity=\"{:.2}\" stroke-width=\"{sw:.2}\"",
                hex_rgb(scol),
                scol[3] as f32 / 255.0
            )
        } else {
            String::from(" stroke=\"none\"")
        };
        out.push_str(&format!(
            "<polygon points=\"{pts_str}\" fill=\"{}\" fill-opacity=\"{fill_op:.2}\"{stroke_attr}/>\n",
            hex_rgb(poly.fill)
        ));
    }

    for ln in &prims.lines {
        let (x1, y1) = map_pt(ln.a);
        let (x2, y2) = map_pt(ln.b);
        let dash = if ln.dashed { " stroke-dasharray=\"7,5\"" } else { "" };
        let op = ln.color[3] as f32 / 255.0;
        out.push_str(&format!(
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"{}\" stroke-opacity=\"{op:.2}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\"{dash}/>\n",
            hex_rgb(ln.color),
            ln.width
        ));
        if ln.arrow_mid {
            let (dx, dy) = (x2 - x1, y2 - y1);
            let len = dx.hypot(dy);
            if len >= 10.0 {
                let (ux, uy) = (dx / len, dy / len);
                let (nx, ny) = (-uy, ux);
                let (tx, ty) = (x1 + dx * 0.56, y1 + dy * 0.56);
                let w = 6.5_f32;
                let (ax, ay) = (tx - ux * w + nx * (w * 0.62), ty - uy * w + ny * (w * 0.62));
                let (bx, by) = (tx - ux * w - nx * (w * 0.62), ty - uy * w - ny * (w * 0.62));
                out.push_str(&format!(
                    "<polyline points=\"{ax:.2},{ay:.2} {tx:.2},{ty:.2} {bx:.2},{by:.2}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.2}\" stroke-linecap=\"round\"/>\n",
                    hex_rgb(ln.color),
                    ln.width
                ));
            }
        }
    }

    for circ in &prims.circles {
        let (cx, cy) = map_pt(circ.center);
        let r = (circ.radius_world * scale).clamp(circ.min_px, circ.max_px * 1.35);
        let fill_op = circ.fill[3] as f32 / 255.0;
        let (scol, sw) = circ.stroke;
        out.push_str(&format!(
            "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r:.2}\" fill=\"{}\" fill-opacity=\"{fill_op:.2}\" stroke=\"{}\" stroke-width=\"{sw:.2}\"/>\n",
            hex_rgb(circ.fill),
            hex_rgb(scol)
        ));
    }

    for lbl in &prims.labels {
        let (x, y) = map_pt(lbl.pos);
        out.push_str(&format!(
            "<text x=\"{x:.2}\" y=\"{y:.2}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{:.1}\" fill=\"{}\">{}</text>\n",
            lbl.size_px,
            hex_rgb(lbl.color),
            xml_escape(&lbl.text)
        ));
    }

    if !prims.title.is_empty() {
        out.push_str(&format!(
            "<text x=\"16\" y=\"22\" font-family=\"sans-serif\" font-size=\"14\" font-weight=\"bold\" fill=\"#263040\">{}</text>\n",
            xml_escape(&prims.title)
        ));
    }
    if !prims.note.is_empty() {
        let y_box = if prims.title.is_empty() { 10 } else { 32 };
        let bw = width.saturating_sub(32);
        out.push_str(&format!(
            "<rect x=\"16\" y=\"{y_box}\" width=\"{bw}\" height=\"26\" rx=\"4\" fill=\"#fff4d6\" stroke=\"#daa030\" stroke-width=\"1\"/>\n\
             <text x=\"26\" y=\"{}\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#7c440a\">{}</text>\n",
            y_box + 17,
            xml_escape(&prims.note)
        ));
    }

    out.push_str("</svg>\n");
    out
}

pub fn encode_scene_png(scene: &Scene2D, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let prims = compile_scene(scene);
    let (wc, sc, scale) = compute_transform(&prims, width, height);
    let map_pt = |p: [f32; 2]| -> [f32; 2] {
        [sc[0] + (p[0] - wc[0]) * scale, sc[1] - (p[1] - wc[1]) * scale]
    };

    let mut rgba = [248_u8, 249, 252, 255].repeat((width * height) as usize);

    for poly in &prims.polygons {
        if poly.points.len() >= 3 {
            let pts: Vec<[f32; 2]> = poly.points.iter().copied().map(map_pt).collect();
            let mut c = [0.0_f32; 2];
            for p in &pts {
                c[0] += p[0] / (pts.len() as f32);
                c[1] += p[1] / (pts.len() as f32);
            }
            for m in 0..pts.len() {
                fill_triangle(&mut rgba, width, height, c, pts[m], pts[(m + 1) % pts.len()], poly.fill);
            }
            if let Some((scol, sw)) = poly.stroke {
                for m in 0..pts.len() {
                    draw_aa_segment(&mut rgba, width, height, pts[m], pts[(m + 1) % pts.len()], scol, sw);
                }
            }
        }
    }

    for ln in &prims.lines {
        let p0 = map_pt(ln.a);
        let p1 = map_pt(ln.b);
        if ln.dashed {
            draw_dashed_aa(&mut rgba, (width, height), (p0, p1), ln.color, ln.width, (8.0, 5.5));
        } else {
            draw_aa_segment(&mut rgba, width, height, p0, p1, ln.color, ln.width * 1.15);
        }
        if ln.arrow_mid {
            let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
            let len = dx.hypot(dy);
            if len >= 10.0 {
                let (ux, uy) = (dx / len, dy / len);
                let (nx, ny) = (-uy, ux);
                let tip = [p0[0] + dx * 0.56, p0[1] + dy * 0.56];
                let w = 7.0_f32;
                let a = [tip[0] - ux * w + nx * (w * 0.62), tip[1] - uy * w + ny * (w * 0.62)];
                let b = [tip[0] - ux * w - nx * (w * 0.62), tip[1] - uy * w - ny * (w * 0.62)];
                draw_aa_segment(&mut rgba, width, height, tip, a, ln.color, ln.width * 1.15);
                draw_aa_segment(&mut rgba, width, height, tip, b, ln.color, ln.width * 1.15);
            }
        }
    }

    for circ in &prims.circles {
        let c = map_pt(circ.center);
        let r = (circ.radius_world * scale).clamp(circ.min_px, circ.max_px * 1.4);
        draw_aa_circle(&mut rgba, width, height, c, r, circ.fill, circ.stroke);
    }

    if !prims.note.is_empty() && width > 40 && height > 48 {
        let y0 = if prims.title.is_empty() { 10_i32 } else { 28_i32 };
        let x1 = width as i32 - 16;
        for y in y0..(y0 + 22).min(height as i32) {
            for x in 16..x1 {
                let border = x == 16 || x == x1 - 1 || y == y0 || y == y0 + 21;
                let col = if border { [218, 160, 48, 255] } else { [255, 244, 214, 235] };
                blend_pixel(&mut rgba, width, height, x, y, col);
            }
        }
    }

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&rgba).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    buf.flush().map_err(|e| e.to_string())?;
    Ok(buf)
}

fn blend_pixel(rgba: &mut [u8], width: u32, height: u32, x: i32, y: i32, src: [u8; 4]) {
    if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height || src[3] == 0 {
        return;
    }
    let idx = ((y as u32 * width + x as u32) * 4) as usize;
    let sa = src[3] as f32 / 255.0;
    let inv = 1.0 - sa;
    rgba[idx] = (src[0] as f32 * sa + rgba[idx] as f32 * inv).round() as u8;
    rgba[idx + 1] = (src[1] as f32 * sa + rgba[idx + 1] as f32 * inv).round() as u8;
    rgba[idx + 2] = (src[2] as f32 * sa + rgba[idx + 2] as f32 * inv).round() as u8;
    rgba[idx + 3] = 255;
}

fn fill_triangle(rgba: &mut [u8], width: u32, height: u32, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [u8; 4]) {
    let min_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i32;
    let max_x = a[0].max(b[0]).max(c[0]).ceil().min(width as f32 - 1.0) as i32;
    let min_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as i32;
    let max_y = a[1].max(b[1]).max(c[1]).ceil().min(height as f32 - 1.0) as i32;
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if area.abs() < 1e-4 {
        return;
    }
    let sign = area.signum();
    for y in min_y..=max_y {
        let py = y as f32 + 0.5;
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let w0 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) * sign;
            let w1 = ((c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])) * sign;
            let w2 = ((a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])) * sign;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                blend_pixel(rgba, width, height, x, y, color);
            }
        }
    }
}

fn draw_dashed_aa(
    rgba: &mut [u8],
    (width, height): (u32, u32),
    (p0, p1): ([f32; 2], [f32; 2]),
    color: [u8; 4],
    stroke_w: f32,
    (dash, gap): (f32, f32),
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let total = dx.hypot(dy);
    if total < 1e-3 {
        return;
    }
    let (ux, uy) = (dx / total, dy / total);
    let mut s = 0.0;
    while s < total {
        let e = (s + dash).min(total);
        draw_aa_segment(
            rgba,
            width,
            height,
            [p0[0] + ux * s, p0[1] + uy * s],
            [p0[0] + ux * e, p0[1] + uy * e],
            color,
            stroke_w,
        );
        s += dash + gap;
    }
}

fn draw_aa_segment(rgba: &mut [u8], width: u32, height: u32, p0: [f32; 2], p1: [f32; 2], color: [u8; 4], stroke_w: f32) {
    let half = (stroke_w * 0.5).max(0.55);
    let pad = half + 1.2;
    let min_x = (p0[0].min(p1[0]) - pad).floor().max(0.0) as i32;
    let max_x = (p0[0].max(p1[0]) + pad).ceil().min(width as f32 - 1.0) as i32;
    let min_y = (p0[1].min(p1[1]) - pad).floor().max(0.0) as i32;
    let max_y = (p0[1].max(p1[1]) + pad).ceil().min(height as f32 - 1.0) as i32;
    let vx = p1[0] - p0[0];
    let vy = p1[1] - p0[1];
    let len2 = vx * vx + vy * vy;
    for y in min_y..=max_y {
        let py = y as f32 + 0.5;
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let t = if len2 > 1e-6 {
                (((px - p0[0]) * vx + (py - p0[1]) * vy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dist = (px - (p0[0] + t * vx)).hypot(py - (p0[1] + t * vy));
            let cov = (half + 0.5 - dist).clamp(0.0, 1.0);
            if cov > 0.0 {
                let a = (color[3] as f32 * cov).round() as u8;
                blend_pixel(rgba, width, height, x, y, [color[0], color[1], color[2], a]);
            }
        }
    }
}

fn draw_aa_circle(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    center: [f32; 2],
    r: f32,
    fill: [u8; 4],
    stroke: ([u8; 4], f32),
) {
    let (scol, sw) = stroke;
    let outer = r + sw * 0.5 + 1.2;
    let min_x = (center[0] - outer).floor().max(0.0) as i32;
    let max_x = (center[0] + outer).ceil().min(width as f32 - 1.0) as i32;
    let min_y = (center[1] - outer).floor().max(0.0) as i32;
    let max_y = (center[1] + outer).ceil().min(height as f32 - 1.0) as i32;
    for y in min_y..=max_y {
        let py = y as f32 + 0.5;
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let d = (px - center[0]).hypot(py - center[1]);
            let fill_cov = (r + 0.5 - d).clamp(0.0, 1.0);
            if fill_cov > 0.0 && fill[3] > 0 {
                let a = (fill[3] as f32 * fill_cov).round() as u8;
                blend_pixel(rgba, width, height, x, y, [fill[0], fill[1], fill[2], a]);
            }
            if sw > 0.0 && scol[3] > 0 {
                let ring_cov = (sw * 0.5 + 0.5 - (d - r).abs()).clamp(0.0, 1.0);
                if ring_cov > 0.0 {
                    let a = (scol[3] as f32 * ring_cov).round() as u8;
                    blend_pixel(rgba, width, height, x, y, [scol[0], scol[1], scol[2], a]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AppBackend, DefectSelectionUi, SurfaceBuildParamsUi, SurfaceTargetUi};

    #[test]
    fn svg_and_png_exports_succeed_for_all_three_scene2d_variants() {
        let backend = AppBackend::new();
        let kagome = backend.find_named("kagome").unwrap();

        // 1. PeriodicCell
        let (cell, _) = backend.build_cell(kagome, &[]);
        let s_cell = Scene2D::PeriodicCell {
            cell,
            repeat: 3,
            show_unit_cell: true,
        };
        let svg_cell = render_scene_svg(&s_cell, 320, 320);
        assert!(svg_cell.starts_with("<svg ") && svg_cell.contains("<polygon ") && svg_cell.contains("a1"));
        let png_cell = encode_scene_png(&s_cell, 160, 160).unwrap();
        assert_eq!(&png_cell[..8], b"\x89PNG\r\n\x1a\n");

        // 2. SurfaceNet (genus-2 with slits and cones)
        let (g2_mesh, _) = backend
            .build_surface(kagome, SurfaceTargetUi::Genus2, &SurfaceBuildParamsUi::default())
            .unwrap();
        let s_net = Scene2D::SurfaceNet { mesh: g2_mesh };
        let svg_net = render_scene_svg(&s_net, 320, 320);
        assert!(svg_net.contains("stroke-dasharray") && svg_net.contains("<polyline "));
        let png_net = encode_scene_png(&s_net, 160, 160).unwrap();
        assert_eq!(&png_net[..8], b"\x89PNG\r\n\x1a\n");

        // 3. CrystalPatch (wall with note banner & domain tinting)
        let cat = backend.defect_catalogue(kagome);
        let sel = DefectSelectionUi::Wall(cat.walls.len().saturating_sub(1));
        let (crystal, readout) = backend.build_defect(kagome, &sel, 3.5, true);
        let s_patch = Scene2D::CrystalPatch {
            crystal,
            title: readout.title,
            note: String::from("Locally repaired glide-breaking wall (crowded = 1)"),
        };
        let svg_patch = render_scene_svg(&s_patch, 320, 320);
        assert!(svg_patch.contains("crowded = 1") && svg_patch.contains("<circle "));
        let png_patch = encode_scene_png(&s_patch, 160, 160).unwrap();
        assert_eq!(&png_patch[..8], b"\x89PNG\r\n\x1a\n");
    }
}
