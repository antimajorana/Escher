pub mod tokens;
pub mod typography;

pub use tokens::ThemeTokens;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StudioTheme {
    #[default]
    Paper,
    Studio,
}

impl StudioTheme {
    #[cfg(test)]
    pub const ALL: [Self; 2] = [Self::Paper, Self::Studio];

    pub fn label(self) -> &'static str {
        match self {
            Self::Paper => "Paper",
            Self::Studio => "Studio",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Paper => Self::Studio,
            Self::Studio => Self::Paper,
        }
    }

    pub fn tokens(self) -> ThemeTokens {
        ThemeTokens::for_theme(self)
    }

    pub fn apply(&self, ctx: &egui::Context) {
        let t = self.tokens();
        let mut visuals = if t.is_light {
            egui::Visuals::light()
        } else {
            egui::Visuals::dark()
        };

        visuals.panel_fill = t.bg_panel;
        visuals.window_fill = t.bg_popover;
        visuals.faint_bg_color = t.bg_card;
        visuals.extreme_bg_color = t.bg_app;
        visuals.code_bg_color = t.bg_card;

        visuals.window_stroke = egui::Stroke::new(1.0, t.border_strong);
        visuals.override_text_color = Some(t.text_primary);
        visuals.hyperlink_color = t.accent_primary;
        visuals.warn_fg_color = t.status_warn;
        visuals.error_fg_color = t.status_err;

        visuals.selection.bg_fill = t.accent_soft;
        visuals.selection.stroke = egui::Stroke::new(1.0, t.accent_primary);

        visuals.widgets.noninteractive.bg_fill = t.bg_panel;
        visuals.widgets.noninteractive.weak_bg_fill = t.bg_card;
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, t.border_subtle);
        visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, t.text_secondary);

        visuals.widgets.inactive.bg_fill = t.bg_card;
        visuals.widgets.inactive.weak_bg_fill = t.bg_card;
        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, t.border_subtle);
        visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, t.text_primary);

        visuals.widgets.hovered.bg_fill = t.bg_card_hover;
        visuals.widgets.hovered.weak_bg_fill = t.bg_card_hover;
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, t.border_strong);
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.2, t.text_primary);

        visuals.widgets.active.bg_fill = t.bg_card_selected;
        visuals.widgets.active.weak_bg_fill = t.bg_card_selected;
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.5, t.border_focus);
        visuals.widgets.active.fg_stroke = egui::Stroke::new(1.5, t.text_primary);

        visuals.widgets.open.bg_fill = t.bg_card_selected;
        visuals.widgets.open.weak_bg_fill = t.bg_card_selected;
        visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, t.border_focus);
        visuals.widgets.open.fg_stroke = egui::Stroke::new(1.2, t.text_primary);

        let egui_theme = if t.is_light {
            egui::Theme::Light
        } else {
            egui::Theme::Dark
        };
        ctx.set_theme(egui_theme);
        ctx.set_visuals_of(egui_theme, visuals.clone());
        ctx.set_visuals(visuals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_and_studio_tokens_meet_wcag_aa_contrast_for_text() {
        for theme in StudioTheme::ALL {
            let t = theme.tokens();
            let c_primary = ThemeTokens::contrast_wcag_ratio(t.text_primary, t.bg_panel);
            let c_secondary = ThemeTokens::contrast_wcag_ratio(t.text_secondary, t.bg_panel);
            let c_muted = ThemeTokens::contrast_wcag_ratio(t.text_muted, t.bg_panel);
            assert!(c_primary >= 7.0, "{theme:?} primary contrast {c_primary}");
            assert!(c_secondary >= 4.5, "{theme:?} secondary contrast {c_secondary}");
            assert!(c_muted >= 4.5, "{theme:?} muted contrast {c_muted}");
        }
    }

    #[test]
    fn remap_scene_ink_lifts_dark_neutral_strokes_in_studio_mode() {
        let paper = ThemeTokens::paper();
        let studio = ThemeTokens::studio();
        let dark_bond = [44, 62, 80, 255];
        assert_eq!(paper.remap_scene_ink(dark_bond), egui::Color32::from_rgb(44, 62, 80));
        let lifted = studio.remap_scene_ink(dark_bond);
        let ratio = ThemeTokens::contrast_wcag_ratio(lifted, studio.bg_canvas_2d);
        assert!(ratio >= 4.5, "lifted bond contrast {ratio} too low on dark canvas");
    }
}
