//! PNG pixel scale: a 2x / 4x export is the 1x composition at 2 / 4 times the resolution (exact sides, the
//! box-downsampled 2x picture matches the 1x one), sides beyond the raster limits are errors, and the SVG does
//! not depend on the scale.

use std::time::Instant;

use super::*;
use crate::backend::AppBackend;
use crate::canvas2d::overlay_legend::spectral_overlay_data;
use crate::canvas2d::{Canvas2D, CanvasMarker, MarkerKind, Scene2D};
use crate::scene2d::compile_scene;

fn decode(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes.to_vec())).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

/// Kagome 3x3 patch with its unit cell, a Unicode title, two labelled pins and a diverging colorbar legend:
/// every primitive kind of the PNG export (unit-cell polygon, solid and dashed lines, discs, text, rects).
fn kagome_scene() -> (Scene2D, Vec<CanvasMarker>) {
    let backend = AppBackend::new();
    let kagome = backend.find_named("kagome").unwrap();
    let (cell, _) = backend.build_cell(kagome, &[]);
    let markers = vec![
        CanvasMarker { index: 0, pos: [0.0, 0.0], kind: MarkerKind::DisclinationPlus, label: String::from("+60° Ω") },
        CanvasMarker { index: 1, pos: [1.0, 0.5], kind: MarkerKind::Dislocation, label: String::from("b = a₁") },
    ];
    (Scene2D::PeriodicCell { cell, repeat: 3, show_unit_cell: true }, markers)
}

fn encode(scene: &Scene2D, markers: &[CanvasMarker], scale: u32) -> Result<Vec<u8>, String> {
    let ov = spectral_overlay_data(&[-0.3, 0.1], true);
    let legend = ExportLegend::from_overlay(&ov);
    encode_layers_png(&compile_scene(scene), &[], markers, Some("kagome — δρ(μ), ψ²"), legend.as_ref(), PngSize::export(scale)?)
}

/// Mean over all pixels and the R, G, B channels of |box2(big) - small|, `big` 2x the side of `small`.
fn mean_abs_diff_after_2x_box(big: &[u8], small: &[u8], side: usize) -> f64 {
    let mut total = 0.0_f64;
    for y in 0..side {
        for x in 0..side {
            for c in 0..3 {
                let at = |xx: usize, yy: usize| f64::from(big[(yy * 2 * side + xx) * 4 + c]);
                let avg = 0.25 * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
                total += (avg - f64::from(small[(y * side + x) * 4 + c])).abs();
            }
        }
    }
    total / (side * side * 3) as f64
}

/// Fraction of 1x pixels whose 3x3 neighbourhood (clipped to the picture) is not constant, and the largest
/// channel difference |box2(big) - small| over the other pixels.
fn edge_fraction_and_interior_max(big: &[u8], small: &[u8], side: usize) -> (f64, f64) {
    let px = |x: usize, y: usize| &small[(y * side + x) * 4..(y * side + x) * 4 + 4];
    let (mut edges, mut worst) = (0_usize, 0.0_f64);
    for y in 0..side {
        for x in 0..side {
            let near = (y.saturating_sub(1)..=(y + 1).min(side - 1)).flat_map(|yy| (x.saturating_sub(1)..=(x + 1).min(side - 1)).map(move |xx| (xx, yy)));
            if near.into_iter().any(|(xx, yy)| px(xx, yy) != px(x, y)) {
                edges += 1;
                continue;
            }
            for c in 0..3 {
                let at = |xx: usize, yy: usize| f64::from(big[(yy * 2 * side + xx) * 4 + c]);
                let avg = 0.25 * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
                worst = worst.max((avg - f64::from(px(x, y)[c])).abs());
            }
        }
    }
    (edges as f64 / (side * side) as f64, worst)
}

#[test]
fn png_2x_is_2048_square_and_box_downsamples_to_the_1x_picture() {
    let (scene, markers) = kagome_scene();
    let t = Instant::now();
    let (w1, h1, one) = decode(&encode(&scene, &markers, 1).unwrap());
    let (w2, h2, two) = decode(&encode(&scene, &markers, 2).unwrap());
    println!("# time 1x + 2x encode {:.2} s", t.elapsed().as_secs_f64());
    assert_eq!((w1, h1), (1024, 1024));
    assert_eq!((w2, h2), (2048, 2048));
    let mad = mean_abs_diff_after_2x_box(&two, &one, 1024);
    let (edge, interior_max) = edge_fraction_and_interior_max(&two, &one, 1024);
    println!("# mean |box2(2x) - 1x| = {mad:.3} levels; edge fraction {edge:.4}; max diff off edges {interior_max:.2}");

    // Bound. Every length is doubled, so box2(2x) equals 1x up to 8-bit rounding (<= 1 level: one rounding per
    // blend at each scale) wherever the 1x picture is locally constant. An "edge" pixel is a 1x pixel whose 3x3
    // neighbourhood is not constant (fraction `edge`); every other pixel is >= 1 px (1x) from any element
    // boundary, while at 2x each element's support is its 1x support dilated by < 0.6 px (1x): a ramp of half
    // width h reaches 2h + 1/2 (2x px) = h + 1/4 (1x px), plus a 2x sample's offset <= 0.354 from its 1x pixel
    // centre. At edge pixels: glyphs use exact area coverage (additive, so box2 of the 2x glyph is the 1x glyph);
    // a 1-px linear ramp clamp(1/2 - d) averaged over the four 2x samples equals the 1x value for an
    // axis-aligned edge and differs by <= 0.073 of the contrast at 45 degrees; the pixel-centre polygon fill
    // differs by <= 1/2 its contrast, (95/255) x (<= 30 levels) / 2 < 6. Stroke widths keep their ratio at both
    // scales (the 1.1 px floor is scaled, see `stroke_px`). So: off-edge pixels differ by <= 1 (asserted), and
    // charging 1 + 255/8 (> 1 + 0.073 x 255) per edge pixel, mean <= edge (1 + 255/8) + (1 - edge) interior_max.
    // This is a whole-composition check (a 1 px misplacement exceeds it, below); that each element's lengths
    // are multiplied by s is tested per element, with tight known answers, in element_scale_tests.rs.
    assert!(interior_max <= 1.0, "off-edge pixels differ only by rounding: {interior_max}");
    let bound = edge * (1.0 + 255.0 / 8.0) + (1.0 - edge) * interior_max;
    assert!(mad < bound, "mean abs diff {mad} >= bound {bound}");

    // A composition misplaced by 1 px (1x) exceeds the bound.
    let mut shifted = vec![0_u8; two.len()];
    let row = 2048 * 4;
    for y in 0..2048 {
        let src = (y + 2).min(2047);
        shifted[y * row..(y + 1) * row].copy_from_slice(&two[src * row..(src + 1) * row]);
    }
    let mad_shifted = mean_abs_diff_after_2x_box(&shifted, &one, 1024);
    println!("# with the 2x picture moved 1 px (1x) up: {mad_shifted:.3} vs bound {bound:.3}");
    assert!(mad_shifted > bound, "the bound detects a misplaced composition: {mad_shifted} vs {bound}");
}

#[test]
fn png_4x_is_4096_square_and_oversized_sides_are_errors() {
    let (scene, markers) = kagome_scene();
    let t = Instant::now();
    let (w, h, img) = decode(&encode(&scene, &markers, 4).unwrap());
    println!("# time 4x encode + decode {:.2} s", t.elapsed().as_secs_f64());
    assert_eq!((w, h), (4096, 4096));
    assert_eq!(img[..4], BACKGROUND, "corner is background");
    // 1024 x 9 = 9216 > MAX_SIDE 8192; 1024 x 8 = 8192 a side is 67 108 864 > MAX_PIXELS 40 000 000 pixels.
    for scale in [9, 8, 0, u32::MAX] {
        assert!(encode(&scene, &markers, scale).is_err(), "scale {scale} must be refused");
    }
}

#[test]
fn canvas_png_follows_the_chosen_scale_and_the_svg_ignores_it() {
    let (scene, _) = kagome_scene();
    let cv = Canvas2D::new(scene);
    let dir = std::env::temp_dir().join(format!("papertilling-scale-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (scale, side) in [(1, 1024), (2, 2048)] {
        let path = dir.join(format!("k{scale}.png"));
        let msg = cv.export_to_path(&path, scale).unwrap();
        assert!(msg.contains(&format!("{side}x{side} PNG at {scale}x")), "{msg}");
        let (w, h, _) = decode(&std::fs::read(&path).unwrap());
        assert_eq!((w, h), (side, side));
    }
    cv.export_to_path(&dir.join("a.svg"), 1).unwrap();
    cv.export_to_path(&dir.join("b.svg"), 4).unwrap();
    assert_eq!(std::fs::read(dir.join("a.svg")).unwrap(), std::fs::read(dir.join("b.svg")).unwrap());
    assert!(cv.export_to_path(&dir.join("big.png"), 9).unwrap_err().contains("9216x9216"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn diagonal_segment_coverage_matches_the_hand_computed_distance_ramp() {
    // Width 2 (half 1) from (10, 10) to (90, 90), black on white: coverage clamp(1.5 - d) at pixel centres.
    // (50, 50): d = 0 -> 255 ink. (52, 50): centre (52.5, 50.5), d = 2 / sqrt 2 = 1.414 -> coverage 0.086,
    // value 255 (1 - 0.086) = 233. (54, 50): d = 2.83 -> untouched. Same for the steep mirror (50, 10)-(52, 90).
    let mut img = super::super::raster::Raster::new(100, 100, [255, 255, 255, 255]).unwrap();
    img.segment([10.0, 10.0], [90.0, 90.0], [0, 0, 0, 255], 2.0);
    let at = |img: &super::super::raster::Raster, x: usize, y: usize| img.rgba[(y * 100 + x) * 4];
    assert_eq!((at(&img, 50, 50), at(&img, 52, 50), at(&img, 54, 50)), (0, 233, 255));
    assert_eq!((at(&img, 48, 50), at(&img, 50, 48)), (233, 233), "symmetric about the line");
    let mut steep = super::super::raster::Raster::new(100, 100, [255, 255, 255, 255]).unwrap();
    // direction (2, 80): at row 50 (centre y 50.5) the line is at x = 50 + 40.5 / 40 = 51.0125; pixel 51 has
    // centre 51.5, d = 0.4875 * 80 / sqrt(6404) = 0.487 -> full ink; pixel 53, d = 2.487 * 0.9997 = 2.486 -> none
    steep.segment([50.0, 10.0], [52.0, 90.0], [0, 0, 0, 255], 2.0);
    assert_eq!((at(&steep, 51, 50), at(&steep, 53, 50), at(&steep, 47, 50)), (0, 255, 255));
}
