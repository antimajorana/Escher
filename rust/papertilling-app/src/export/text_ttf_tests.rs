//! Known answers for the TTF text of the PNG export. Font metrics quoted below were read from the bundled files
//! with fontTools (units per em 2048 for both faces): Inter-Regular hhea ascent 1984, descent -494; 'H' advance
//! 1522 with ink x 180..1342, y 0..1490; DejaVu Sans 'ℏ' (U+210F, absent from Inter) advance 1298.

use super::*;

const BG: [u8; 4] = [255, 255, 255, 255];
const INK: [u8; 4] = [0, 0, 0, 255];

fn render(w: u32, h: u32, s: &str, size: f32, pos: [f32; 2], anchor: (f32, f32)) -> Raster {
    let mut img = Raster::new(w, h, BG).unwrap();
    draw_text(&mut img, pos, s, size, INK, anchor);
    img
}

/// `(x, y)` of every pixel that is not the background.
fn inked(img: &Raster) -> Vec<(u32, u32)> {
    let w = img.w;
    img.rgba.as_chunks::<4>().0.iter().enumerate().filter(|(_, c)| **c != BG).map(|(i, _)| (i as u32 % w, i as u32 / w)).collect()
}

/// Inked bounding box `[x0, y0, x1, y1]` (inclusive pixel indices).
fn ink_bbox(img: &Raster) -> [u32; 4] {
    inked(img).iter().fold([u32::MAX, u32::MAX, 0, 0], |b, &(x, y)| [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)])
}

#[test]
fn greek_math_and_sub_superscripts_draw_ink_inside_their_box_and_differ_from_tofu() {
    let (size, pos) = (32.0, [8.0, 8.0]);
    let tofu = render(64, 64, "中", size, pos, (0.0, 0.0));
    assert!(!has_glyph('中'));
    assert!(inked(&tofu).len() > 40, "tofu is a visible box");
    for c in ['Ω', 'ψ', '±', '₀', '²', 'Δ', 'φ', 'ℏ', '∇'] {
        assert!(has_glyph(c), "{c} in the Inter -> DejaVu chain");
        let s = c.to_string();
        let [w, h] = measure_text(&s, size);
        // box height (1984 + 494) / 2048 em = 38.72 px at 32 px; every advance is between 0.4 and 0.8 em
        assert!((h - 38.719).abs() < 0.01 && (12.8..25.6).contains(&w), "{c}: box {w} x {h}");
        let img = render(64, 64, &s, size, pos, (0.0, 0.0));
        let ink = inked(&img);
        let inside = |&&(x, y): &&(u32, u32)| (x as f32) >= pos[0] - 1.0 && (x as f32) <= pos[0] + w + 1.0 && (y as f32) >= pos[1] - 1.0 && (y as f32) <= pos[1] + h + 1.0;
        assert!(ink.iter().filter(inside).count() > 20, "{c}: {} inked pixels inside its box", ink.iter().filter(inside).count());
        assert_eq!(ink.iter().filter(|p| !inside(p)).count(), 0, "{c}: ink outside its box");
        assert_ne!(img.rgba, tofu.rgba, "{c} must not draw as tofu");
    }
}

#[test]
fn tofu_is_a_hollow_box_and_bad_input_never_panics() {
    // '中' at 40 px, left-top at (0, 0): baseline 1984/2048 * 40 = 38.75; box x 4..20, y 38.75 - 28 = 10.75..38.75,
    // stroke 2.4 px. The left edge (x = 5, y = 25) is ink, the centre (12, 25) and the outside (2, 25) are not.
    let img = render(48, 48, "中", 40.0, [0.0, 0.0], (0.0, 0.0));
    let at = |x: u32, y: u32| img.rgba[((y * 48 + x) * 4) as usize];
    assert_eq!(at(5, 25), 0, "left stroke");
    assert_eq!(at(18, 25), 0, "right stroke (17.6..20)");
    assert_eq!((at(12, 25), at(2, 25)), (255, 255), "hollow inside, nothing left of the box");
    assert!((measure_text("中", 40.0)[0] - 24.0).abs() < 1e-4, "tofu advance 0.6 em");


    let mut img = Raster::new(32, 32, BG).unwrap();
    for (pos, s, size) in [
        ([f32::NAN, 1.0], "x", 12.0),
        ([1.0, 1.0], "x", f32::INFINITY),
        ([1.0, 1.0], "x", -3.0),
        ([1.0, 1.0], "", 12.0),
        ([-1e9, 1e9], "far away", 12.0),
        ([0.0, 0.0], "a\u{0}\n\t😀\u{10FFFF}b", 12.0),
        ([16.0, 16.0], "huge", 1.0e5),
    ] {
        draw_text(&mut img, pos, s, size, INK, (0.5, 0.5));
    }
    assert_eq!(measure_text("x", f32::NAN), [0.0, 0.0]);
}

#[test]
fn measured_widths_follow_the_advances_kerning_and_scale_linearly() {
    // 'H' alone: advance 1522 / 2048 em; at 20.48 px that is 15.22 px.
    assert!((measure_text("H", 20.48)[0] - 15.22).abs() < 1e-3);
    // 'ℏ' is not in Inter: the DejaVu advance 1298 / 2048 em -> 12.98 px at 20.48 px.
    assert!((measure_text("ℏ", 20.48)[0] - 12.98).abs() < 1e-3);
    assert!(measure_text("WW", 14.0)[0] > measure_text("ii", 14.0)[0]);
    // Inter kerns "AV" (GPOS pair adjustment): the pair is narrower than its two letters set apart.
    let av = measure_text("AV", 100.0)[0];
    assert!(av < measure_text("A", 100.0)[0] + measure_text("V", 100.0)[0] - 1.0, "AV kerned: {av}");
    for s in ["WW", "ii", "Spectral charge δρ(μ)", "ℏ∇ψ²₀"] {
        let (a, b) = (measure_text(s, 11.0), measure_text(s, 22.0));
        assert!((b[0] - 2.0 * a[0]).abs() < 1.0 && (b[1] - 2.0 * a[1]).abs() < 1.0, "{s}: {a:?} -> {b:?}");
    }
}

#[test]
fn centre_anchor_centres_the_inked_box_on_pos() {
    // 'H' ink is symmetric in its advance ((180 + 1342) / 2 = 761 = 1522 / 2) and its vertical centre 745 units
    // above the baseline equals the text box centre (1984 - 494) / 2 = 745: anchor (0.5, 0.5) puts both on pos.
    for (pos, size) in [([100.0, 80.0], 64.0), ([60.5, 40.25], 23.0)] {
        let img = render(200, 160, "H", size, pos, (0.5, 0.5));
        let b = ink_bbox(&img);
        let c = [0.5 * (b[0] + b[2] + 1) as f32, 0.5 * (b[1] + b[3] + 1) as f32];
        assert!((c[0] - pos[0]).abs() <= 1.0 && (c[1] - pos[1]).abs() <= 1.0, "ink centre {c:?} vs {pos:?} (bbox {b:?})");
    }
    // anchor (0, 0): ink starts 180/2048 em right of pos and (1984 - 1490)/2048 em below it: (5.6, 15.4) at 64 px
    let b = ink_bbox(&render(200, 160, "H", 64.0, [10.0, 10.0], (0.0, 0.0)));
    assert!((b[0] as f32 - 15.6).abs() <= 1.0 && (b[1] as f32 - 25.4).abs() <= 1.0, "{b:?}");
}
