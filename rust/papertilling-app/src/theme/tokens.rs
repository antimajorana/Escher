use egui::Color32;

use super::StudioTheme;

#[derive(Clone, Copy, Debug)]
pub struct ThemeTokens {
    pub is_light: bool,
    // Chrome & surfaces
    pub bg_app: Color32,
    pub bg_panel: Color32,
    pub bg_card: Color32,
    pub bg_card_hover: Color32,
    pub bg_card_selected: Color32,
    pub bg_canvas_2d: Color32,
    pub bg_viewport_3d: Color32,
    pub bg_hud: Color32,
    pub bg_popover: Color32,
    pub border_subtle: Color32,
    pub border_strong: Color32,
    pub border_focus: Color32,
    // Typography & status
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_muted: Color32,
    pub text_on_accent: Color32,
    pub accent_primary: Color32,
    pub accent_soft: Color32,
    pub status_ok: Color32,
    pub status_warn: Color32,
    pub status_err: Color32,
    pub band_valid: Color32,
    pub band_forbidden: Color32,
    // Geometry & overlays (Stages 2-6 future fields carry targeted allow(dead_code))
    #[allow(dead_code)] // Stage 2-3 overlay token
    pub bond_primary: Color32,
    #[allow(dead_code)] // Stage 2-3 overlay token
    pub bond_ghost: Color32,
    #[allow(dead_code)] // Stage 2-3 overlay token
    pub site_a: Color32,
    #[allow(dead_code)] // Stage 2-3 overlay token
    pub site_b: Color32,
    #[allow(dead_code)] // Stage 3 overlay token
    pub unit_cell_vec: Color32,
    #[allow(dead_code)] // Stage 3 overlay token
    pub unit_cell_fill: Color32,
    #[allow(dead_code)] // Stage 3 ITA overlay token
    pub ita_mirror: Color32,
    #[allow(dead_code)] // Stage 3 ITA overlay token
    pub ita_glide: Color32,
    #[allow(dead_code)] // Stage 3 ITA overlay token
    pub ita_rotation: Color32,
    #[allow(dead_code)] // Stage 3 Voronoi overlay token
    pub voronoi_edge: Color32,
    #[allow(dead_code)] // Stage 3 Voronoi overlay token
    pub voronoi_fill: Color32,
    #[allow(dead_code)] // Stage 6 selection overlay token
    pub selection_ring: Color32,
    pub cut_ray_line: Color32,
    #[allow(dead_code)] // Stage 6 cut-ray handle token
    pub cut_ray_handle: Color32,
    pub cut_half_wall: Color32,
    pub cut_insertion: Color32,
    pub glide_rail: Color32,
    pub locked_pin_ring: Color32,
}

impl ThemeTokens {
    pub fn for_theme(theme: StudioTheme) -> Self {
        match theme {
            StudioTheme::Paper => Self::paper(),
            StudioTheme::Studio => Self::studio(),
        }
    }

    pub fn paper() -> Self {
        Self {
            is_light: true,
            bg_app: Color32::from_rgb(243, 245, 248),
            bg_panel: Color32::from_rgb(250, 251, 253),
            bg_card: Color32::from_rgb(255, 255, 255),
            bg_card_hover: Color32::from_rgb(241, 245, 250),
            bg_card_selected: Color32::from_rgb(232, 240, 252),
            bg_canvas_2d: Color32::from_rgb(248, 249, 252),
            bg_viewport_3d: Color32::from_rgb(244, 246, 250),
            bg_hud: Color32::from_rgba_unmultiplied(245, 247, 251, 236),
            bg_popover: Color32::from_rgb(255, 255, 255),
            border_subtle: Color32::from_rgb(218, 224, 234),
            border_strong: Color32::from_rgb(176, 186, 202),
            border_focus: Color32::from_rgb(43, 108, 176),

            text_primary: Color32::from_rgb(24, 32, 44),
            text_secondary: Color32::from_rgb(71, 85, 105),
            text_muted: Color32::from_rgb(84, 98, 120),
            text_on_accent: Color32::WHITE,
            accent_primary: Color32::from_rgb(29, 95, 214),
            accent_soft: Color32::from_rgb(228, 239, 253),
            status_ok: Color32::from_rgb(18, 112, 56),
            status_warn: Color32::from_rgb(154, 78, 8),
            status_err: Color32::from_rgb(186, 38, 32),
            band_valid: Color32::from_rgb(216, 238, 198),
            band_forbidden: Color32::from_rgb(238, 168, 168),

            bond_primary: Color32::from_rgb(44, 62, 80),
            bond_ghost: Color32::from_rgba_unmultiplied(110, 148, 206, 185),
            site_a: Color32::from_rgb(34, 108, 196),
            site_b: Color32::from_rgb(26, 158, 114),
            unit_cell_vec: Color32::from_rgb(32, 94, 176),
            unit_cell_fill: Color32::from_rgba_unmultiplied(218, 234, 252, 95),
            ita_mirror: Color32::from_rgb(198, 40, 40),
            ita_glide: Color32::from_rgb(21, 101, 192),
            ita_rotation: Color32::from_rgb(46, 125, 50),
            voronoi_edge: Color32::from_rgba_unmultiplied(100, 116, 139, 170),
            voronoi_fill: Color32::from_rgba_unmultiplied(226, 232, 240, 64),
            selection_ring: Color32::from_rgb(29, 95, 214),
            cut_ray_line: Color32::from_rgba_unmultiplied(123, 31, 162, 215),
            cut_ray_handle: Color32::from_rgb(123, 31, 162),
            cut_half_wall: Color32::from_rgba_unmultiplied(214, 128, 0, 235),
            cut_insertion: Color32::from_rgba_unmultiplied(0, 121, 107, 215),
            glide_rail: Color32::from_rgba_unmultiplied(230, 145, 0, 220),
            locked_pin_ring: Color32::from_rgba_unmultiplied(55, 71, 79, 200),
        }
    }

    pub fn studio() -> Self {
        Self {
            is_light: false,
            bg_app: Color32::from_rgb(16, 19, 25),
            bg_panel: Color32::from_rgb(23, 27, 35),
            bg_card: Color32::from_rgb(30, 36, 48),
            bg_card_hover: Color32::from_rgb(37, 44, 59),
            bg_card_selected: Color32::from_rgb(36, 52, 77),
            bg_canvas_2d: Color32::from_rgb(19, 22, 28),
            bg_viewport_3d: Color32::from_rgb(16, 19, 25),
            bg_hud: Color32::from_rgba_unmultiplied(26, 31, 41, 232),
            bg_popover: Color32::from_rgb(30, 36, 48),
            border_subtle: Color32::from_rgb(45, 53, 68),
            border_strong: Color32::from_rgb(72, 84, 106),
            border_focus: Color32::from_rgb(91, 155, 255),

            text_primary: Color32::from_rgb(230, 236, 245),
            text_secondary: Color32::from_rgb(172, 183, 200),
            text_muted: Color32::from_rgb(144, 156, 174),
            text_on_accent: Color32::from_rgb(14, 19, 26),
            accent_primary: Color32::from_rgb(102, 163, 255),
            accent_soft: Color32::from_rgb(32, 50, 78),
            status_ok: Color32::from_rgb(76, 208, 138),
            status_warn: Color32::from_rgb(242, 169, 73),
            status_err: Color32::from_rgb(248, 122, 122),
            band_valid: Color32::from_rgb(44, 84, 54),
            band_forbidden: Color32::from_rgb(108, 44, 48),

            bond_primary: Color32::from_rgb(184, 199, 217),
            bond_ghost: Color32::from_rgba_unmultiplied(128, 174, 240, 195),
            site_a: Color32::from_rgb(96, 165, 250),
            site_b: Color32::from_rgb(52, 211, 153),
            unit_cell_vec: Color32::from_rgb(102, 163, 255),
            unit_cell_fill: Color32::from_rgba_unmultiplied(59, 130, 246, 56),
            ita_mirror: Color32::from_rgb(248, 113, 113),
            ita_glide: Color32::from_rgb(96, 165, 250),
            ita_rotation: Color32::from_rgb(74, 222, 128),
            voronoi_edge: Color32::from_rgba_unmultiplied(148, 163, 184, 180),
            voronoi_fill: Color32::from_rgba_unmultiplied(51, 65, 85, 72),
            selection_ring: Color32::from_rgb(102, 163, 255),
            cut_ray_line: Color32::from_rgba_unmultiplied(192, 132, 252, 225),
            cut_ray_handle: Color32::from_rgb(192, 132, 252),
            cut_half_wall: Color32::from_rgba_unmultiplied(251, 191, 36, 235),
            cut_insertion: Color32::from_rgba_unmultiplied(45, 212, 191, 220),
            glide_rail: Color32::from_rgba_unmultiplied(251, 191, 36, 220),
            locked_pin_ring: Color32::from_rgba_unmultiplied(160, 174, 192, 210),
        }
    }

    /// Keep chromatic colors intact while lifting dark ink strokes in `Studio` mode so
    /// precompiled dark bonds, sites, and chart outlines stay crisp on a dark canvas.
    pub fn remap_scene_ink(&self, rgba: [u8; 4]) -> Color32 {
        if self.is_light {
            return Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
        }
        let [r, g, b, a] = rgba;
        let rf = f32::from(r) / 255.0;
        let gf = f32::from(g) / 255.0;
        let bf = f32::from(b) / 255.0;
        let max_c = rf.max(gf).max(bf);
        let min_c = rf.min(gf).min(bf);
        let chroma = max_c - min_c;
        let luma = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;

        if chroma < 0.18 && luma < 0.45 {
            // Neutral or slate dark ink: lift cleanly into cool light slate.
            let lift = (1.0 - luma).clamp(0.55, 0.90);
            let nr = ((lift * 214.0) + rf * 32.0).round().clamp(0.0, 255.0) as u8;
            let ng = ((lift * 222.0) + gf * 32.0).round().clamp(0.0, 255.0) as u8;
            let nb = ((lift * 234.0) + bf * 32.0).round().clamp(0.0, 255.0) as u8;
            return Color32::from_rgba_unmultiplied(nr, ng, nb, a);
        }
        if luma < 0.32 && max_c > 1e-4 {
            let boost = (0.85 / max_c).clamp(1.0, 2.8);
            let floor = 0.16_f32;
            let nr = ((rf * boost).max(floor) * 255.0).round().clamp(0.0, 255.0) as u8;
            let ng = ((gf * boost).max(floor) * 255.0).round().clamp(0.0, 255.0) as u8;
            let nb = ((bf * boost).max(floor) * 255.0).round().clamp(0.0, 255.0) as u8;
            return Color32::from_rgba_unmultiplied(nr, ng, nb, a);
        }
        Color32::from_rgba_unmultiplied(r, g, b, a)
    }

    /// WCAG 2.1 contrast ratio between two opaque colors (returns `1.0..=21.0`).
    #[cfg(test)]
    pub fn contrast_wcag_ratio(fg: Color32, bg: Color32) -> f32 {
        let l1 = relative_luminance(fg);
        let l2 = relative_luminance(bg);
        let (hi, lo) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };
        (hi + 0.05) / (lo + 0.05)
    }
}

#[cfg(test)]
fn srgb_to_linear(channel: u8) -> f32 {
    let c = f32::from(channel) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
fn relative_luminance(c: Color32) -> f32 {
    0.2126 * srgb_to_linear(c.r()) + 0.7152 * srgb_to_linear(c.g()) + 0.0722 * srgb_to_linear(c.b())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_status_err_and_badge_contrast_meet_wcag_aa() {
        let studio = ThemeTokens::studio();
        for (bg_name, bg) in [
            ("bg_card_selected", studio.bg_card_selected),
            ("accent_soft", studio.accent_soft),
            ("bg_card_hover", studio.bg_card_hover),
            ("bg_card", studio.bg_card),
        ] {
            let r = ThemeTokens::contrast_wcag_ratio(studio.status_err, bg);
            assert!(r >= 4.5, "Studio status_err on {bg_name}: {r:.2} < 4.5");
        }
        for (fg_name, fill) in [
            ("status_err", studio.status_err),
            ("status_warn", studio.status_warn),
            ("accent_primary", studio.accent_primary),
        ] {
            let r = ThemeTokens::contrast_wcag_ratio(studio.text_on_accent, fill);
            assert!(r >= 6.0, "Studio text_on_accent on {fg_name}: {r:.2} < 6.0");
        }
    }
}
