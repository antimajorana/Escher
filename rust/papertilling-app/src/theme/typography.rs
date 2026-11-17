use std::sync::Arc;

use egui::{FontData, FontDefinitions, FontFamily, FontId, TextStyle};

const INTER_REGULAR: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.ttf");
const INTER_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/Inter-SemiBold.ttf");
const DEJAVU_SANS: &[u8] = include_bytes!("../../assets/fonts/DejaVuSans.ttf");
const JETBRAINS_MONO: &[u8] = include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf");

pub fn register_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    fonts.font_data.insert(
        String::from("inter_regular"),
        Arc::new(FontData::from_static(INTER_REGULAR)),
    );
    fonts.font_data.insert(
        String::from("inter_semibold"),
        Arc::new(FontData::from_static(INTER_SEMIBOLD)),
    );
    fonts.font_data.insert(
        String::from("dejavu_sans"),
        Arc::new(FontData::from_static(DEJAVU_SANS)),
    );
    fonts.font_data.insert(
        String::from("jetbrains_mono"),
        Arc::new(FontData::from_static(JETBRAINS_MONO)),
    );

    if let Some(prop) = fonts.families.get_mut(&FontFamily::Proportional) {
        prop.insert(0, String::from("dejavu_sans"));
        prop.insert(0, String::from("inter_regular"));
    }
    if let Some(mono) = fonts.families.get_mut(&FontFamily::Monospace) {
        mono.insert(0, String::from("dejavu_sans"));
        mono.insert(0, String::from("jetbrains_mono"));
    }
    fonts.families.insert(
        FontFamily::Name(Arc::from("semibold")),
        vec![String::from("inter_semibold"), String::from("dejavu_sans")],
    );

    ctx.set_fonts(fonts);
    ctx.all_styles_mut(|style| {
        style.text_styles.insert(TextStyle::Heading, FontId::new(16.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Body, FontId::new(13.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Button, FontId::new(13.0, FontFamily::Proportional));
        style.text_styles.insert(TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace));
        style.text_styles.insert(TextStyle::Small, FontId::new(11.5, FontFamily::Proportional));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_fonts_register_and_cover_scientific_glyphs() {
        let ctx = egui::Context::default();
        register_fonts(&ctx);
        let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
        let id = FontId::new(13.0, FontFamily::Proportional);
        ctx.fonts_mut(|f| {
            for ch in ['Ω', 'ψ', 'δ', 'ρ', 'κ', 'χ', 'Δ', 'π', 'β', 'μ', 'φ', '±', '·', '₀', '₁', '²', '³'] {
                assert!(
                    f.has_glyph(&id, ch),
                    "missing glyph '{ch}' in proportional fallback chain"
                );
            }
        });
    }
}
