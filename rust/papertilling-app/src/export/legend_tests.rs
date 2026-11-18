//! Exported legends: SVG group with title and exactly 32 stepped rects (or one rect per chip), the picture fitted
//! above the box, legend-free SVG / PNG pinned to hand-derived goldens (the pre-legend fit with a 36 px bottom pad),
//! PNG bar ends with the hue of the colour map's ends, Unicode PNG labels with no tofu, and the Canvas2D path.

use super::super::text_ttf::has_glyph;
use super::super::{encode_prims_png, encode_prims_png_with_legend, render_prims_svg, render_prims_svg_with_legend};
use super::*;
use crate::canvas2d::overlay_legend::{LegendCategory, spectral_overlay_data};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::scene2d::{ScenePrimitives, compile_scene};

fn crystal_scene() -> Scene2D {
    use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
    use papertilling_core::plane::catalogue::parse_named_str;
    let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
    let sq = named.named.into_iter().find(|n| n.key == "square").unwrap();
    let crystal = build_point_defects(&sq.cell, &[DefectSpec::disclination(90.0, [0.5, 0.5])], None, None).unwrap();
    Scene2D::CrystalPatch { crystal, title: String::from("legend export"), note: String::new() }
}

fn prims() -> ScenePrimitives {
    compile_scene(&crystal_scene())
}

/// `(cy, r)` of every `<circle ...>` in an SVG fragment.
fn svg_circles(svg: &str) -> Vec<(f32, f32)> {
    let attr = |line: &str, k: &str| -> f32 {
        let i = line.find(&format!(" {k}=\"")).unwrap() + k.len() + 3;
        line[i..i + line[i..].find('"').unwrap()].parse().unwrap()
    };
    svg.lines().filter(|l| l.starts_with("<circle ")).map(|l| (attr(l, "cy"), attr(l, "r"))).collect()
}

fn decode(bytes: &[u8]) -> (u32, Vec<u8>) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes.to_vec())).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    (info.width, buf)
}

#[test]
fn svg_colorbar_legend_has_title_32_steps_range_and_the_picture_fits_above_it() {
    let t0 = std::time::Instant::now();
    let p = prims();
    let ov = spectral_overlay_data(&[0.0, 0.25, 1.5], false);
    let leg = ExportLegend::from_overlay(&ov).expect("colorbar legend");
    let plain = render_prims_svg(&p, &[], None, 960, 960);
    let with = render_prims_svg_with_legend(&p, &[], None, &leg, 960, 960);
    assert!(!plain.contains("id=\"legend\"") && !plain.contains("legend-step") && !plain.contains("LDOS"));
    assert_eq!(with.matches("<g id=\"legend\">").count(), 1);
    assert_eq!(with.matches("class=\"legend-step\"").count(), 32);
    assert!(with.contains(">LDOS |ψ|²</text>"), "title text");
    assert!(with.contains(">0.000</text>") && with.contains(">1.500</text>"), "min / max labels");
    // The legend box (y 886..948 at 960 px) sits below every exported site disc: the picture is fitted above it.
    let body = &with[..with.find("<g id=\"legend\">").unwrap()];
    let lowest = svg_circles(body).into_iter().map(|(cy, r)| cy + r).fold(f32::MIN, f32::max);
    assert_eq!(legend_box(&leg, 960).1, 886);
    assert!(lowest < 886.0, "content reaches y = {lowest}, under the legend box");
    assert!(svg_circles(&plain).into_iter().any(|(cy, r)| cy + r > 886.0), "without a legend the 36 px pad is kept");
    assert_eq!(with.matches("<svg").count(), 1);
    println!("# time svg_colorbar_legend {:?}", t0.elapsed());
}

#[test]
fn svg_categorical_legend_has_one_chip_rect_per_category() {
    let p = prims();
    let chips = vec![LegendChip::new(LegendCategory::FluxImproper), LegendChip::new(LegendCategory::FluxRotation)];
    let with = render_prims_svg_with_legend(&p, &[], None, &ExportLegend::Categorical(&chips), 960, 960);
    assert_eq!(with.matches("class=\"legend-chip\"").count(), 2);
    assert_eq!(with.matches("class=\"legend-step\"").count(), 0);
    // Chip fills are the export colours of the flux badges: #8e24aa (142, 36, 170) and #c62828 (198, 40, 40).
    assert!(with.contains("fill=\"#8e24aa\"") && with.contains("fill=\"#c62828\""));
    assert!(with.contains("Improper flux (det=-1)") && with.contains("Rotation flux Ω (curvature)"));
}

#[test]
fn png_legend_bar_runs_blue_to_red_for_charge() {
    let p = prims();
    let ov = spectral_overlay_data(&[-0.3, 0.1], true);
    let leg = ExportLegend::from_overlay(&ov).unwrap();
    let plain = encode_prims_png(&p, &[], None, 512, 512).unwrap();
    let with = encode_prims_png_with_legend(&p, &[], None, Some(&leg), 512, 512).unwrap();
    assert_ne!(plain, with);
    let (w, img) = decode(&with);
    let (x0, y0, x1, _) = legend_box(&leg, 512);
    let y = (y0 + 31) as u32; // middle of the bar rows [y0 + 24, y0 + 38)
    let at = |x: i32| {
        let i = ((y * w + x as u32) * 4) as usize;
        [img[i], img[i + 1], img[i + 2]]
    };
    let (left, right) = (at(x0 + 12), at(x1 - 12));
    assert!(left[2] > left[0] + 100, "min end of a diverging bar is blue: {left:?}");
    assert!(right[0] > right[2] + 100, "max end is red: {right:?}");
}

#[test]
fn png_labels_keep_their_unicode_and_every_character_has_a_ttf_glyph() {
    // The PNG legend now draws the labels as the SVG does (no ASCII spelling): every character of the overlay
    // titles must be in the Inter -> DejaVu Sans chain, so none of them draws as a tofu box.
    let titles = [
        "LDOS |ψ|²",
        "Spectral charge δρ(μ)",
        "Bond stretch ε=(l-l₀)/l₀",
        "GSV E (stretch+shear, √-scale)",
        "Rotation flux Ω (curvature)",
        "Torsion α / τ-shift",
    ];
    for t in titles {
        assert!(t.chars().all(has_glyph), "{t:?}: {:?}", t.chars().filter(|c| !has_glyph(*c)).collect::<String>());
    }
    assert!(!has_glyph('中'), "a CJK ideograph is in neither bundled face: the tofu case exists");
}


#[test]
fn canvas_export_carries_the_overlay_legend_only_when_one_is_active() {
    let dir = std::env::temp_dir().join(format!("papertilling-legend-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut cv = Canvas2D::new(crystal_scene());
    cv.export_to_path(&dir.join("plain.svg"), 1).unwrap();
    let plain = std::fs::read_to_string(dir.join("plain.svg")).unwrap();
    assert!(!plain.contains("id=\"legend\""));
    cv.set_gauge_overlay(Some(spectral_overlay_data(&[-0.3, 0.1], true)));
    cv.export_to_path(&dir.join("charge.svg"), 1).unwrap();
    let svg = std::fs::read_to_string(dir.join("charge.svg")).unwrap();
    assert!(svg.contains("Spectral charge δρ(μ)") && svg.contains(">-0.300</text>") && svg.contains(">0.300</text>"));
    assert_eq!(svg.matches("class=\"legend-step\"").count(), 32);
    cv.export_to_path(&dir.join("charge.png"), 1).unwrap();
    assert!(std::fs::read(dir.join("charge.png")).unwrap().starts_with(b"\x89PNG"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Unit square bbox, no title or note: one line (0,0)-(1,1) and one disc at (0.5, 0.5) of world radius 0.05.
fn unit_prims() -> ScenePrimitives {
    use crate::scene2d::{CirclePrim, LinePrim};
    ScenePrimitives {
        polygons: Vec::new(),
        lines: vec![LinePrim { a: [0.0, 0.0], b: [1.0, 1.0], color: [10, 20, 30, 255], width: 1.5, dashed: false, arrow_mid: false }],
        circles: vec![CirclePrim {
            center: [0.5, 0.5],
            radius_world: 0.05,
            min_px: 2.0,
            max_px: 50.0,
            fill: [255, 0, 0, 255],
            stroke: ([0, 0, 0, 255], 1.0),
        }],
        labels: Vec::new(),
        site_circle_indices: Vec::new(),
        edge_line_indices: Vec::new(),
        title: String::new(),
        note: String::new(),
        bbox: ([0.0, 0.0], [1.0, 1.0]),
        site_count: 1,
        bond_count: 1,
    }
}

#[test]
fn legend_free_svg_is_the_hand_derived_pre_legend_layout() {
    // 960 x 960, no title / note: pads 36 top and bottom, 32 left and right -> available 896 x 888, scale 888,
    // centre (480, 36 + 444). World (0,0) -> (36, 924), (1,1) -> (924, 36), disc r = 0.05 * 888 = 44.4.
    let golden = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"960\" viewBox=\"0 0 960 960\">
<rect width=\"100%\" height=\"100%\" fill=\"#f8f9fc\"/>
<line x1=\"36.00\" y1=\"924.00\" x2=\"924.00\" y2=\"36.00\" stroke=\"#0a141e\" stroke-opacity=\"1.00\" stroke-width=\"1.50\" stroke-linecap=\"round\"/>
<circle cx=\"480.00\" cy=\"480.00\" r=\"44.40\" fill=\"#ff0000\" fill-opacity=\"1.00\" stroke=\"#000000\" stroke-opacity=\"1.00\" stroke-width=\"1.00\"/>
</svg>
";
    assert_eq!(render_prims_svg(&unit_prims(), &[], None, 960, 960), golden);
    // With a colorbar (box y 886..948) the bottom pad is 960 - 886 + 8 = 82: available 896 x 842, scale 842,
    // centre (480, 36 + 421). (0,0) -> (59, 878), (1,1) -> (901, 36), disc r = 0.05 * 842 = 42.1.
    let ov = spectral_overlay_data(&[0.0, 1.0], false);
    let with = render_prims_svg_with_legend(&unit_prims(), &[], None, &ExportLegend::from_overlay(&ov).unwrap(), 960, 960);
    assert!(with.contains("<line x1=\"59.00\" y1=\"878.00\" x2=\"901.00\" y2=\"36.00\" "), "{with}");
    assert!(with.contains("<circle cx=\"480.00\" cy=\"457.00\" r=\"42.10\" "), "{with}");
}

#[test]
fn legend_free_png_of_an_empty_picture_is_the_plain_background() {
    let mut empty = unit_prims();
    empty.lines.clear();
    empty.circles.clear();
    // Hand golden: a 64 x 64 RGBA buffer of the export background (248, 249, 252, 255), PNG-encoded with the
    // default encoder settings; drawing anything at all (e.g. an empty legend box) changes the bytes.
    let rgba: Vec<u8> = [248_u8, 249, 252, 255].repeat(64 * 64);
    let mut golden = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut golden, 64, 64);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        w.write_image_data(&rgba).unwrap();
        w.finish().unwrap();
    }
    assert_eq!(encode_prims_png(&empty, &[], None, 64, 64).unwrap(), golden);
    assert_eq!(encode_prims_png_with_legend(&empty, &[], None, None, 64, 64).unwrap(), golden);
}
