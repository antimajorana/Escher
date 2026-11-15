use std::io::Write as _;
use std::path::Path;

pub fn export_patch_to_path(
    sites: &[[f32; 2]],
    bonds: &[(usize, usize)],
    path: &Path,
) -> Result<String, String> {
    let is_svg = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"));
    if is_svg {
        let svg = render_svg(sites, bonds, 640, 640);
        std::fs::write(path, svg).map_err(|e| e.to_string())?;
        Ok("Exported SVG lattice preview".to_owned())
    } else {
        let png_bytes = encode_png(sites, bonds, 640, 640)?;
        std::fs::write(path, png_bytes).map_err(|e| e.to_string())?;
        Ok("Exported PNG lattice preview".to_owned())
    }
}

pub fn render_svg(sites: &[[f32; 2]], bonds: &[(usize, usize)], width: u32, height: u32) -> String {
    let (cx, cy, scale) = (width as f32 * 0.5, height as f32 * 0.5, 72.0_f32);
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#f8f9fc\"/>\n"
    );
    for &(i, j) in bonds {
        let (a, b) = (sites[i], sites[j]);
        let (x1, y1) = (cx + a[0] * scale, cy - a[1] * scale);
        let (x2, y2) = (cx + b[0] * scale, cy - b[1] * scale);
        out.push_str(&format!(
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"#34495e\" stroke-width=\"2\"/>\n"
        ));
    }
    for &p in sites {
        let (x, y) = (cx + p[0] * scale, cy - p[1] * scale);
        out.push_str(&format!(
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"5\" fill=\"#1f77b4\" stroke=\"#122a42\" stroke-width=\"1.2\"/>\n"
        ));
    }
    out.push_str("</svg>\n");
    out
}

pub fn encode_png(
    sites: &[[f32; 2]],
    bonds: &[(usize, usize)],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let mut rgba = [248_u8, 249, 252, 255].repeat((width * height) as usize);
    let (cx, cy, scale) = (width as f32 * 0.5, height as f32 * 0.5, 72.0_f32);
    for &(i, j) in bonds {
        let (a, b) = (sites[i], sites[j]);
        draw_line(
            &mut rgba,
            width,
            height,
            (cx + a[0] * scale, cy - a[1] * scale),
            (cx + b[0] * scale, cy - b[1] * scale),
            [52, 73, 94, 255],
        );
    }
    for &p in sites {
        draw_disc(
            &mut rgba,
            width,
            height,
            (cx + p[0] * scale, cy - p[1] * scale),
            4.5,
            [31, 119, 180, 255],
        );
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

fn draw_line(rgba: &mut [u8], width: u32, height: u32, p0: (f32, f32), p1: (f32, f32), color: [u8; 4]) {
    let steps = ((p1.0 - p0.0).hypot(p1.1 - p0.1).ceil() as i32).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (p0.0 + (p1.0 - p0.0) * t).round() as i32;
        let y = (p0.1 + (p1.1 - p0.1) * t).round() as i32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                put_pixel(rgba, width, height, x + ox, y + oy, color);
            }
        }
    }
}

fn draw_disc(rgba: &mut [u8], width: u32, height: u32, center: (f32, f32), r: f32, color: [u8; 4]) {
    let ir = r.ceil() as i32;
    let (cx, cy) = (center.0.round() as i32, center.1.round() as i32);
    for dy in -ir..=ir {
        for dx in -ir..=ir {
            if (dx * dx + dy * dy) as f32 <= r * r {
                put_pixel(rgba, width, height, cx + dx, cy + dy, color);
            }
        }
    }
}

fn put_pixel(rgba: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x >= 0 && y >= 0 && (x as u32) < width && (y as u32) < height {
        let idx = ((y as u32 * width + x as u32) * 4) as usize;
        rgba[idx..idx + 4].copy_from_slice(&color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_and_svg_exports_have_valid_headers() {
        let sites = [[0.0, 0.0], [0.5, 0.0], [0.25, 0.433]];
        let bonds = [(0, 1), (1, 2), (2, 0)];
        let png_bytes = encode_png(&sites, &bonds, 96, 96).expect("PNG encoding succeeds");
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");
        let svg = render_svg(&sites, &bonds, 96, 96);
        assert!(svg.starts_with("<svg ") && svg.contains("<circle "));
    }
}
