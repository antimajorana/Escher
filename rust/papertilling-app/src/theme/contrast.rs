//! WCAG 2.x relative luminance and contrast ratio, and the choice of label ink over translucent fills.

use egui::Color32;

/// Near-black and white label inks: over every ring tint composited on the Studio canvas one of the two
/// reaches at least 4.5:1 (WCAG AA for small text).
pub const INK_DARK: Color32 = Color32::from_rgb(8, 10, 14);
pub const INK_LIGHT: Color32 = Color32::WHITE;

fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Relative luminance of an opaque sRGB colour.
pub fn luminance(c: [u8; 3]) -> f32 {
    0.2126 * srgb_to_linear(c[0]) + 0.7152 * srgb_to_linear(c[1]) + 0.0722 * srgb_to_linear(c[2])
}

/// WCAG contrast ratio `(L_hi + 0.05) / (L_lo + 0.05)` of two opaque colours, in `[1, 21]`.
pub fn contrast_ratio(a: [u8; 3], b: [u8; 3]) -> f32 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

/// `fill` (straight, unmultiplied alpha) composited over the opaque colour `bg`.
pub fn over(fill: [u8; 4], bg: Color32) -> [u8; 3] {
    let a = f32::from(fill[3]) / 255.0;
    let mix = |f: u8, b: u8| (a * f32::from(f) + (1.0 - a) * f32::from(b)).round().clamp(0.0, 255.0) as u8;
    [mix(fill[0], bg.r()), mix(fill[1], bg.g()), mix(fill[2], bg.b())]
}

fn rgb(c: Color32) -> [u8; 3] {
    [c.r(), c.g(), c.b()]
}

/// The ink of [`INK_DARK`] / [`INK_LIGHT`] with the higher contrast against `bg`.
pub fn best_ink(bg: [u8; 3]) -> Color32 {
    if contrast_ratio(rgb(INK_DARK), bg) >= contrast_ratio(rgb(INK_LIGHT), bg) { INK_DARK } else { INK_LIGHT }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wcag_reference_values() {
        // black on white is 21:1 and a colour against itself 1:1 (WCAG 2.x definition).
        assert!((contrast_ratio([0, 0, 0], [255, 255, 255]) - 21.0).abs() < 1e-3);
        assert!((contrast_ratio([119, 119, 119], [119, 119, 119]) - 1.0).abs() < 1e-6);
        // #777777 on white is the textbook borderline case, 4.48:1.
        assert!((contrast_ratio([119, 119, 119], [255, 255, 255]) - 4.48).abs() < 0.01);
    }

    #[test]
    fn compositing_is_linear_in_alpha() {
        let bg = Color32::from_rgb(0, 0, 0);
        assert_eq!(over([255, 255, 255, 255], bg), [255, 255, 255]);
        assert_eq!(over([255, 255, 255, 0], bg), [0, 0, 0]);
        assert_eq!(over([200, 100, 50, 128], bg), [100, 50, 25]);
    }
}
