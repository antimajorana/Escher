//! Per-element known answers for the PNG pixel scale `s`: every length of every element is multiplied by `s`,
//! so each element alone has (a) its ink bounding box at `s` times the 1x box and (b) its ink (sum over pixels
//! and RGB of |pixel - background|) at `s^2` times the 1x ink, areas scaling as lengths squared. A missing
//! `* s` on a font size, radius or box length leaves the ink ratio near 1 and the box unscaled; on a stroke
//! width, the ratio near `s`. Plus the stroke floor: a 0.8 px stroke has cross-section 1.1 px at 1x, 2.2 px at
//! 2x and 4.4 px at 4x (the same width relative to the picture).

use std::time::Instant;

use super::*;
use crate::canvas2d::overlay_legend::spectral_overlay_data;
use crate::canvas2d::{CanvasMarker, MarkerKind};
use crate::scene2d::{CirclePrim, LinePrim, PolyPrim, TextPrim};

const INK: [u8; 4] = [20, 30, 60, 255];

fn decode(bytes: &[u8]) -> Vec<u8> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes.to_vec())).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    buf
}

/// Empty primitives over the world box [0, 10]^2 (no title or note: top and bottom pads 36 px a side at 1x).
fn base() -> ScenePrimitives {
    ScenePrimitives { bbox: ([0.0, 0.0], [10.0, 10.0]), ..Default::default() }
}

/// One element of the composition, drawn alone.
struct Case {
    name: &'static str,
    prims: ScenePrimitives,
    markers: Vec<CanvasMarker>,
    legend: bool,
}

fn render(case: &Case, scale: u32) -> Vec<u8> {
    let ov = spectral_overlay_data(&[-0.3, 0.1], true);
    let legend = case.legend.then(|| ExportLegend::from_overlay(&ov)).flatten();
    let png = encode_layers_png(&case.prims, &[], &case.markers, None, legend.as_ref(), PngSize::export(scale).unwrap()).unwrap();
    decode(&png)
}

/// Ink sum and the half-open ink box (x0, y0, x1, y1) of a square `side` x `side` picture.
fn ink(img: &[u8], side: usize) -> (f64, [usize; 4]) {
    let (mut sum, mut bx) = (0.0, [usize::MAX, usize::MAX, 0, 0]);
    for (i, px) in img.as_chunks::<4>().0.iter().enumerate() {
        let d: f64 = (0..3).map(|c| f64::from(px[c].abs_diff(BACKGROUND[c]))).sum();
        if d > 0.0 {
            let (x, y) = (i % side, i / side);
            sum += d;
            bx = [bx[0].min(x), bx[1].min(y), bx[2].max(x + 1), bx[3].max(y + 1)];
        }
    }
    (sum, bx)
}

fn cases() -> Vec<Case> {
    let with = |name, f: &dyn Fn(&mut ScenePrimitives)| {
        let mut prims = base();
        f(&mut prims);
        Case { name, prims, markers: Vec::new(), legend: false }
    };
    let line = |a, b, width, dashed| LinePrim { a, b, color: INK, width, dashed, arrow_mid: !dashed };
    let marker = CanvasMarker { index: 0, pos: [5.0, 5.0], kind: MarkerKind::DisclinationPlus, label: String::from("+60° Ω") };
    vec![
        with("label", &|p| p.labels.push(TextPrim { pos: [5.0, 5.0], text: String::from("Ωψ± Δφ₀²"), color: INK, size_px: 14.0 })),
        with("solid line + arrow", &|p| p.lines.push(line([1.0, 1.0], [9.0, 6.0], 2.0, false))),
        with("dashed line", &|p| p.lines.push(line([1.0, 8.0], [9.0, 2.0], 1.5, true))),
        with("circle at min_px", &|p| {
            let stroke = ([200, 40, 40, 255], 1.5);
            p.circles.push(CirclePrim { center: [5.0, 5.0], radius_world: 0.0, min_px: 6.0, max_px: 40.0, fill: INK, stroke });
        }),
        with("thin-stroke square", &|p| {
            let points = vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]];
            p.polygons.push(PolyPrim { points, fill: [0, 0, 0, 0], stroke: Some((INK, 0.8)) });
        }),
        with("filled triangle", &|p| {
            p.polygons.push(PolyPrim { points: vec![[1.0, 1.0], [9.0, 2.0], [4.0, 9.0]], fill: [90, 140, 200, 120], stroke: None });
        }),
        with("title", &|p| p.title = String::from("kagome — δρ(μ), ψ²")),
        with("note", &|p| p.note = String::from("periodic patch, 3 x 3 cells")),
        Case { name: "marker", prims: base(), markers: vec![marker], legend: false },
        Case { name: "colorbar legend", prims: base(), markers: Vec::new(), legend: true },
    ]
}

#[test]
fn every_element_scales_its_box_by_s_and_its_ink_by_s_squared() {
    // Tolerances. Box: the 1x box includes the 1-px antialiasing ramp tail (reach half + 0.5 px), which at
    // scale s reaches s * half + 0.5 rather than s * (half + 0.5), and faint tails round to alpha 0 differently:
    // at most s - 1/2 picture pixels per side, so |box_s - s * box_1| <= s. Ink: glyph coverage is the exact
    // pixel area, and a linear 1-px ramp sampled at unit spacing sums to the stroke width exactly, so these
    // scale as s^2 up to 8-bit rounding (<= 1/2 level a channel a blend); discs and the pixel-centre polygon
    // fill differ by a boundary term of order (perimeter / area) per pixel, a few % for the smallest element
    // (the 6 px disc). 6 % separates the answer 1 from the mutation answers 1/s^2 (length not scaled) and 1/s
    // (stroke width not scaled), which are <= 1/2.
    let t = Instant::now();
    for case in cases() {
        let (ink1, box1) = ink(&render(&case, 1), 1024);
        assert!(ink1 > 0.0, "{} draws something", case.name);
        let scales: &[u32] = if matches!(case.name, "label" | "thin-stroke square") { &[2, 4] } else { &[2] };
        for &s in scales {
            let (ink_s, box_s) = ink(&render(&case, s), 1024 * s as usize);
            let ratio = ink_s / (ink1 * f64::from(s * s));
            println!("# {:<20} {s}x: ink ratio {ratio:.4}, box {box_s:?} vs {s} x {box1:?}", case.name);
            assert!((ratio - 1.0).abs() < 0.06, "{} at {s}x: ink / (s^2 ink_1x) = {ratio}", case.name);
            for k in 0..4 {
                let want = box1[k] * s as usize;
                assert!(box_s[k].abs_diff(want) <= s as usize, "{} at {s}x: box {box_s:?} vs {s} x {box1:?}", case.name);
            }
        }
    }
    println!("# time all elements {:.2} s", t.elapsed().as_secs_f64());
}

#[test]
fn a_thin_stroke_keeps_its_relative_width_at_every_scale() {
    // Square [2, 8]^2 stroked 0.8 with a transparent fill, opaque INK on BACKGROUND. On the picture's middle
    // row the left half crosses only the left edge (vertical). Each pixel's coverage is clamp(half + 1/2 - d),
    // a trapezoid of plateau 2 half - 1 and unit ramps; sampled at unit spacing it sums to 2 half = stroke
    // width at any sub-pixel phase. Floor 1.1 px (half 0.55) applied before scaling: 1.1, 2.2, 4.4 at 1x, 2x,
    // 4x (an unscaled floor gives max(0.8 s, 1.1) = 1.1, 1.6, 3.2). Coverage read back from the red channel:
    // (BG - pixel) / (BG - INK), 8-bit rounding <= 1/(2 * 228) + 1/255 a pixel, < 0.03 over <= 6 pixels.
    let case = &cases()[4];
    assert_eq!(case.name, "thin-stroke square");
    for (s, want) in [(1_u32, 1.1), (2, 2.2), (4, 4.4)] {
        let side = 1024 * s as usize;
        let img = render(case, s);
        let row = side / 2;
        let bg = f64::from(BACKGROUND[0]);
        let width: f64 = (0..side / 2).map(|x| (bg - f64::from(img[(row * side + x) * 4])) / (bg - f64::from(INK[0]))).sum();
        println!("# {s}x: stroke cross-section {width:.4} px");
        assert!((width - want).abs() < 0.03, "{s}x: cross-section {width} px, want {want}");
    }
}
