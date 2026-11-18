use super::export_png::{Export3dInfo, clamp_export_size, compose_overlays, export_plan, write_atomic};
use super::{ColorLegend3d, Legend3dMap};
use crate::export::raster::Raster;
use crate::export::text_ttf::measure_text;
use crate::theme::ThemeTokens;

#[test]
fn size_is_the_viewport_times_the_scale_scaled_down_uniformly_to_the_limits() {
    let big = u64::MAX;
    // 1200 x 800 at 4x = 4800 x 3200, inside 8192
    assert_eq!(clamp_export_size([1200.0, 800.0], 4, 8192, big), Ok(Export3dInfo { w: 4800, h: 3200, clamped: false }));
    // limit 4096: factor 4096 / 4800, height 3200 * 4096 / 4800 = 2730.67 -> 2731 (aspect within 1 px)
    let c = clamp_export_size([1200.0, 800.0], 4, 4096, big).unwrap();
    assert_eq!(c, Export3dInfo { w: 4096, h: 2731, clamped: true });
    assert!((f64::from(c.h) - f64::from(c.w) * 800.0 / 1200.0).abs() <= 1.0);
    // pixel budget 40 000 000 on a 2048 x 2048 view at 4x (8192^2 = 67 108 864 px): side 8192 sqrt(40e6 / 8192^2)
    // = 6324.56 -> 6325, and 6325^2 = 40 005 625 > 40e6, so one column goes: 6324 x 6325 = 39 999 300
    assert_eq!(clamp_export_size([2048.0, 2048.0], 4, 8192, 40_000_000), Ok(Export3dInfo { w: 6324, h: 6325, clamped: true }));
    // 1x of a fractional viewport rounds; no scale or an empty view is an error
    assert_eq!(clamp_export_size([640.4, 480.6], 1, 8192, big), Ok(Export3dInfo { w: 640, h: 481, clamped: false }));
    assert!(clamp_export_size([1200.0, 800.0], 0, 8192, big).is_err());
    assert!(clamp_export_size([0.0, 800.0], 2, 8192, big).is_err());
    assert!(clamp_export_size([f32::NAN, 800.0], 2, 8192, big).is_err());
}

#[test]
fn plan_is_the_on_screen_pixels_times_the_scale_within_the_brief_limits() {
    let mib256 = 256_u64 << 20;
    // 800 x 600 pt at 150 % = 1200 x 900 screen px: 1x = 1200 x 900 with the screen's 1 px wires (None); 2x =
    // 2400 x 1800 with 2 px wires (2400 / 1200)
    assert_eq!(export_plan(egui::vec2(800.0, 600.0), 1.5, 1, 8192, mib256), Ok((Export3dInfo { w: 1200, h: 900, clamped: false }, None)));
    assert_eq!(
        export_plan(egui::vec2(800.0, 600.0), 1.5, 2, 8192, mib256),
        Ok((Export3dInfo { w: 2400, h: 1800, clamped: false }, Some(2.0)))
    );
    // 1800 x 1700 pt at 100 %, 4x = 7200 x 6800: both sides <= 8192 and 4 x 7200 x 6800 = 195 840 000 B < 256 MiB, so
    // nothing is scaled down (no extra pixel cap); 4 px wires
    assert_eq!(
        export_plan(egui::vec2(1800.0, 1700.0), 1.0, 4, 8192, mib256),
        Ok((Export3dInfo { w: 7200, h: 6800, clamped: false }, Some(4.0)))
    );
    // 1200 x 800 at 4x on a 4096 device: 4096 x 2731 as in the size test, wires 4096 / 1200 px
    let (info, wire) = export_plan(egui::vec2(1200.0, 800.0), 1.0, 4, 4096, mib256).unwrap();
    assert_eq!(info, Export3dInfo { w: 4096, h: 2731, clamped: true });
    assert!((wire.unwrap() - 4096.0 / 1200.0).abs() < 1e-4);
}

#[test]
fn atomic_write_leaves_the_whole_file_and_no_temp_or_nothing_on_error() {
    let dir = std::env::temp_dir().join(format!("papertilling_t3_atomic_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.png");
    std::fs::write(&path, b"old contents, longer than the new").unwrap();
    // replaces an existing file with exactly the new bytes, and the .part is gone
    write_atomic(&path, b"new").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"new");
    assert!(!dir.join("a.png.part").exists());
    // a missing directory: an error, nothing at the destination, no temp
    let bad = dir.join("missing_dir").join("b.png");
    assert!(write_atomic(&bad, b"x").unwrap_err().contains("cannot write"));
    assert!(!bad.exists() && !dir.join("missing_dir").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

fn bg(t: &ThemeTokens) -> [u8; 4] {
    t.bg_viewport_3d.to_srgba_unmultiplied()
}

/// Ink of the image against the background: (bounding box `[x0, y0, x1, y1]` of changed pixels, coverage sum in
/// pixels from the red channel, `(bg_r - r) / (bg_r - ink_r)`).
fn ink(img: &Raster, bg: [u8; 4], ink_r: u8, region: [u32; 4]) -> Option<([u32; 4], f64)> {
    let mut bb: Option<[u32; 4]> = None;
    let mut sum = 0.0;
    for y in region[1]..region[3] {
        for x in region[0]..region[2] {
            let i = 4 * (y * img.w + x) as usize;
            if img.rgba[i..i + 4] != bg {
                bb = Some(bb.map_or([x, y, x, y], |b| [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]));
                sum += (f64::from(bg[0]) - f64::from(img.rgba[i])) / (f64::from(bg[0]) - f64::from(ink_r));
            }
        }
    }
    bb.map(|b| (b, sum))
}

fn canvas(view: egui::Vec2, k: f32, t: &ThemeTokens) -> Raster {
    Raster::new((view.x * k) as u32, (view.y * k) as u32, bg(t)).unwrap()
}

#[test]
fn caption_ink_lies_top_left_and_scales_with_the_png_scale() {
    let t = ThemeTokens::paper();
    let view = egui::vec2(400.0, 300.0);
    let cap = "Torus T2 — 48 faces";
    let ink_r = t.text_primary.r();
    let mut found = Vec::new();
    for k in [1.0_f32, 2.0] {
        let mut img = canvas(view, k, &t);
        compose_overlays(&mut img, view, k, cap, None, &t);
        let all = [0, 0, img.w, img.h];
        let (bb, sum) = ink(&img, bg(&t), ink_r, all).expect("caption drawn");
        // inset: the text box starts at (8, 6) k and is measure_text(cap, 12 k) large (1 px antialiasing margin)
        let [tw, th] = measure_text(cap, 12.0 * k);
        assert!(bb[0] as f32 >= 8.0 * k - 1.0 && bb[1] as f32 >= 6.0 * k - 1.0, "k={k}: {bb:?}");
        assert!((bb[2] as f32) <= 8.0 * k + tw + 1.0 && (bb[3] as f32) <= 6.0 * k + th + 1.0, "k={k}: {bb:?} vs {tw}x{th}");
        assert!(bb[2] < img.w / 2 && bb[3] < img.h / 4, "caption stays in the top-left inset");
        found.push((bb, sum));
    }
    let ((b1, s1), (b2, s2)) = (found[0], found[1]);
    // box x 2: edges at 2x the 1x position within 2 px (each edge is a pixel boundary of an antialiased outline)
    let size1 = [f64::from(b1[2] - b1[0] + 1), f64::from(b1[3] - b1[1] + 1)];
    let size2 = [f64::from(b2[2] - b2[0] + 1), f64::from(b2[3] - b2[1] + 1)];
    for a in 0..2 {
        assert!((size2[a] - 2.0 * size1[a]).abs() <= 2.0, "box {size1:?} -> {size2:?}");
    }
    // ink x 4: the rasterizer's coverage is the exact outline area (text_ttf), which scales by k^2. Two effects
    // shrink the 1x ink slightly: coverage is clamped to 1 where neighbouring glyphs' antialiased edges share a
    // pixel (relatively more such pixels at 1x), and the alpha byte rounds +-0.5/255 per pixel, i.e. +-0.002 px
    // each. Both are edge effects of order (edge pixels / area) ~ 1 / stroke width in px (~1.5 px at 12 px), so a
    // few percent at most; tolerance 2 % (measured 4.039, i.e. +1.0 %).
    let ratio = s2 / s1;
    eprintln!("caption ink {s1:.2} -> {s2:.2} px, ratio {ratio:.4}; box {size1:?} -> {size2:?}");
    assert!((ratio - 4.0).abs() <= 0.08, "ink {s1:.1} -> {s2:.1}, ratio {ratio:.4}");
}

fn legend() -> ColorLegend3d {
    ColorLegend3d { title: String::from("σ_xx"), min_val: -0.5, max_val: 0.5, map: Legend3dMap::Diverging }
}

#[test]
fn colorbar_box_present_iff_legend_with_end_colours_of_the_map() {
    let t = ThemeTokens::paper();
    let view = egui::vec2(400.0, 300.0);
    let leg = legend();
    // live layout (legend_rect_3d): w = min(174, 400 - 16) = 174, box (400 - 6 - 174, 300 - 84) = (220, 216) to
    // (394, 264); bar (box + (8, 19)) to (box right - 8, box top + 27) = x 228..386, y 235..243
    for (k, box_, bar) in [(1_u32, [220, 216, 394, 264], [228, 235, 386, 243]), (2, [440, 432, 788, 528], [456, 470, 772, 486])] {
        let kf = k as f32;
        let mut without = canvas(view, kf, &t);
        compose_overlays(&mut without, view, kf, "", None, &t);
        assert!(ink(&without, bg(&t), 0, [0, 0, without.w, without.h]).is_none(), "no legend, no caption: blank");

        let mut img = canvas(view, kf, &t);
        compose_overlays(&mut img, view, kf, "", Some(&leg), &t);
        let (bb, _) = ink(&img, bg(&t), 0, [0, 0, img.w, img.h]).expect("legend drawn");
        assert_eq!(bb, [box_[0], box_[1], box_[2] - 1, box_[3] - 1], "k={k}: box is the live legend rect x k");
        let mid = (bar[1] + bar[3]) / 2;
        let at = |x: u32| {
            let i = 4 * (mid * img.w + x) as usize;
            [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2], img.rgba[i + 3]]
        };
        assert_eq!(at(bar[0]), leg.rgba_of_value(-0.5), "k={k}: first column = colour at min");
        assert_eq!(at(bar[2] - 1), leg.rgba_of_value(0.5), "k={k}: last column = colour at max");
        assert_ne!(at(bar[0]), at(bar[2] - 1));
    }
    // a viewport below the live HUD's 200 x 130 minimum shows no legend box
    let small = egui::vec2(180.0, 120.0);
    let mut img = canvas(small, 1.0, &t);
    compose_overlays(&mut img, small, 1.0, "", Some(&leg), &t);
    assert!(ink(&img, bg(&t), 0, [0, 0, img.w, img.h]).is_none());
}
