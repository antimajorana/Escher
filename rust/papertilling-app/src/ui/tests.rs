use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use super::viewport_pane::show_split_2d_3d;
use crate::app::{NavTab, PapertillingApp};
use crate::theme::{StudioTheme, ThemeTokens, typography};

fn app_harness(theme: StudioTheme) -> Harness<'static, PapertillingApp> {
    let app = PapertillingApp::new(false, String::from("kittest"), String::from("kittest"));
    Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_theme(if theme == StudioTheme::Paper {
            egui::Theme::Light
        } else {
            egui::Theme::Dark
        })
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .with_context_setup(move |ctx| {
            typography::register_fonts(ctx);
            theme.apply(ctx);
        })
        .build_ui_state(
            |ui, app| {
                let _ = app.frame_ui(ui);
            },
            app,
        )
}

/// The Inspector toggle lives in the HUD "View" menu: open it, click the checkbox, close the menu with Esc.
fn toggle_inspector_via_view_menu(harness: &mut Harness<'static, PapertillingApp>) {
    harness.get_by_label("View").click();
    harness.run_steps(2);
    harness.get_by_label("Right panel (Ctrl+I)").click();
    harness.run_steps(2);
    if egui::Popup::is_any_open(&harness.ctx) {
        harness.key_press(egui::Key::Escape);
    }
}

#[test]
fn studio_shell_three_column_layout_bounds_are_valid_and_non_overlapping() {
    let mut harness = app_harness(StudioTheme::Paper);
    harness.run();

    let r = &harness.state().test_rects;
    let (top, status, left, right, center) = (r.top, r.status, r.left, r.right, r.center);

    assert!(top.width() > 1200.0 && top.height() >= 20.0, "top={top:?}");
    assert!(status.width() > 1200.0 && status.height() >= 16.0, "status={status:?}");
    assert!((260.0..=420.0).contains(&left.width()), "left width={}", left.width());
    assert!((260.0..=440.0).contains(&right.width()), "right width={}", right.width());
    assert!(center.width() > 300.0 && center.height() > 300.0, "center={center:?}");

    assert!(top.bottom() <= left.top() + 1.0);
    assert!(left.right() <= center.left() + 1.0);
    assert!(center.right() <= right.left() + 1.0);
    assert!(center.bottom() <= status.top() + 1.0);
    assert!(!left.intersects(right));
}

#[test]
fn right_inspector_collapses_widens_center_and_still_polls_background_tabs() {
    let mut harness = app_harness(StudioTheme::Paper);
    harness.run();

    let center_before = harness.state().test_rects.center.width();
    let right_w = harness.state().test_rects.right.width();
    assert!(harness.state().right_inspector_open);

    toggle_inspector_via_view_menu(&mut harness);
    harness.run();
    assert!(!harness.state().right_inspector_open);
    assert_eq!(harness.state().test_rects.right, egui::Rect::NOTHING);
    let center_after = harness.state().test_rects.center.width();
    assert!(
        center_after >= center_before + right_w - 8.0,
        "center should widen when inspector collapses: {center_before} -> {center_after} (right={right_w})"
    );

    // Background polling on Combine/Defects must continue when Right Inspector is collapsed.
    harness.state_mut().switch_tab(NavTab::Combine);
    for _ in 0..40 {
        harness.step();
        if !harness.state().combine.is_building() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        !harness.state().combine.last_build.crystal.pos.is_empty()
            && harness.state().canvas2d.counts().0 > 0,
        "2D crystal must still sync on Combine when Right Inspector is collapsed"
    );

    toggle_inspector_via_view_menu(&mut harness);
    harness.run_steps(2);
    assert!(harness.state().right_inspector_open);
    assert!(harness.state().test_rects.right.width() >= 260.0);
}

#[test]
fn split_2d_3d_drag_resizes_panes_clamps_and_double_click_resets_to_half() {
    struct SplitState {
        ratio: f32,
        left_rect: egui::Rect,
        right_rect: egui::Rect,
    }
    let mut harness = Harness::builder()
        .with_size(egui::vec2(800.0, 400.0))
        .with_step_dt(0.05)
        .with_accessibility_check(false)
        .build_ui_state(
            |ui, st: &mut SplitState| {
                let tokens = ThemeTokens::paper();
                show_split_2d_3d(
                    ui,
                    &mut st.ratio,
                    "kittest_split",
                    &tokens,
                    |ui| {
                        st.left_rect = ui.available_rect_before_wrap();
                        ui.label("left_2d");
                    },
                    |ui| {
                        st.right_rect = ui.available_rect_before_wrap();
                        ui.label("right_3d");
                    },
                );
            },
            SplitState {
                ratio: 0.5,
                left_rect: egui::Rect::NOTHING,
                right_rect: egui::Rect::NOTHING,
            },
        );
    harness.run();

    let w0_left = harness.state().left_rect.width();
    let w0_right = harness.state().right_rect.width();
    assert!((w0_left - w0_right).abs() < 2.0);

    // Drag divider right by +140 px.
    let div_x = harness.state().left_rect.right() + 3.0;
    let div_y = harness.state().left_rect.center().y;
    let start = egui::pos2(div_x, div_y);
    let mid_drag = egui::pos2(div_x + 140.0, div_y);
    harness.hover_at(start);
    harness.step();
    harness.drag_at(start);
    harness.step();
    harness.hover_at(mid_drag);
    harness.step();
    harness.drop_at(mid_drag);
    harness.run();

    let r_after = harness.state().ratio;
    assert!(r_after > 0.60 && r_after <= 0.80, "expected ratio > 0.60, got {r_after}");
    assert!(harness.state().left_rect.width() > w0_left + 100.0);
    assert!(harness.state().right_rect.width() < w0_right - 100.0);

    // Extreme right drag clamps cleanly to 0.8.
    let cur_div = egui::pos2(harness.state().left_rect.right() + 3.0, div_y);
    harness.hover_at(cur_div);
    harness.step();
    harness.drag_at(cur_div);
    harness.step();
    harness.hover_at(egui::pos2(790.0, div_y));
    harness.step();
    harness.drop_at(egui::pos2(790.0, div_y));
    harness.run();
    assert!((harness.state().ratio - 0.8).abs() < 1e-4, "ratio={}", harness.state().ratio);

    // Double-click divider resets ratio to 0.5.
    let clamped_div = egui::pos2(harness.state().left_rect.right() + 3.0, div_y);
    harness.hover_at(clamped_div);
    harness.step();
    for _ in 0..2 {
        harness.event(egui::Event::PointerButton {
            pos: clamped_div,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
        harness.event(egui::Event::PointerButton {
            pos: clamped_div,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        harness.step();
    }
    harness.run();
    assert!(
        (harness.state().ratio - 0.5).abs() < 1e-4,
        "double-click should reset ratio to 0.5, got {}",
        harness.state().ratio
    );
}

#[test]
fn split_2d_3d_does_not_overflow_at_ultra_narrow_width() {
    struct NarrowState {
        ratio: f32,
        total_rect: egui::Rect,
        left_rect: egui::Rect,
        right_rect: egui::Rect,
    }
    for width in [140.0_f32, 100.0] {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(width, 300.0))
            .with_step_dt(0.05)
            .with_accessibility_check(false)
            .build_ui_state(
                |ui, st: &mut NarrowState| {
                    st.total_rect = ui.available_rect_before_wrap();
                    let tokens = ThemeTokens::paper();
                    show_split_2d_3d(
                        ui,
                        &mut st.ratio,
                        "kittest_narrow_split",
                        &tokens,
                        |ui| {
                            st.left_rect = ui.max_rect();
                        },
                        |ui| {
                            st.right_rect = ui.max_rect();
                        },
                    );
                },
                NarrowState {
                    ratio: 0.5,
                    total_rect: egui::Rect::NOTHING,
                    left_rect: egui::Rect::NOTHING,
                    right_rect: egui::Rect::NOTHING,
                },
            );
        harness.run();
        let st = harness.state();
        assert!(
            st.left_rect.left() >= st.total_rect.left() - 0.5,
            "width={width}: left_rect {:?} < total {:?}",
            st.left_rect,
            st.total_rect
        );
        assert!(
            st.right_rect.right() <= st.total_rect.right() + 0.5,
            "width={width}: right_rect {:?} overflowed total {:?}",
            st.right_rect,
            st.total_rect
        );
        assert!(
            !st.left_rect.intersects(st.right_rect),
            "width={width}: left {:?} intersected right {:?}",
            st.left_rect,
            st.right_rect
        );
    }
}

#[test]
fn theme_toggle_updates_visuals_and_tokens_meet_wcag_contrast_across_all_painters() {
    let mut harness = app_harness(StudioTheme::Paper);
    harness.run();
    assert_eq!(harness.state().theme, StudioTheme::Paper);
    assert!(!harness.ctx.style_of(harness.ctx.theme()).visuals.dark_mode);

    harness.get_by_label("Theme: Light").click();
    harness.run();
    assert_eq!(harness.state().theme, StudioTheme::Studio);
    assert!(harness.ctx.style_of(harness.ctx.theme()).visuals.dark_mode);

    harness.get_by_label("Theme: Dark").click();
    harness.run();
    assert_eq!(harness.state().theme, StudioTheme::Paper);
    assert!(!harness.ctx.style_of(harness.ctx.theme()).visuals.dark_mode);

    for theme in StudioTheme::ALL {
        let t = theme.tokens();
        for (bg_name, bg) in [
            ("bg_panel", t.bg_panel),
            ("bg_card", t.bg_card),
            ("bg_canvas_2d", t.bg_canvas_2d),
            ("bg_viewport_3d", t.bg_viewport_3d),
            ("bg_hud", egui::Color32::from_rgb(t.bg_hud.r(), t.bg_hud.g(), t.bg_hud.b())),
        ] {
            for (fg_name, fg) in [
                ("text_primary", t.text_primary),
                ("text_secondary", t.text_secondary),
                ("text_muted", t.text_muted),
                ("status_ok", t.status_ok),
                ("status_warn", t.status_warn),
                ("status_err", t.status_err),
                ("accent_primary", t.accent_primary),
            ] {
                let ratio = ThemeTokens::contrast_wcag_ratio(fg, bg);
                assert!(
                    ratio >= 4.5,
                    "{theme:?} {fg_name} on {bg_name} contrast {ratio:.2} < 4.5"
                );
            }
        }
        assert!(ThemeTokens::contrast_wcag_ratio(t.text_on_accent, t.accent_primary) >= 4.5);
        assert!(ThemeTokens::contrast_wcag_ratio(t.text_on_accent, t.status_ok) >= 4.5);
    }

    // Verify scene ink remapping on Studio dark canvas for all dark bond/site colors.
    let studio = ThemeTokens::studio();
    for dark_rgba in [
        [44, 62, 80, 255],
        [32, 48, 76, 230],
        [44, 62, 88, 200],
        [36, 56, 88, 220],
        [28, 34, 44, 255],
    ] {
        let remapped = studio.remap_scene_ink(dark_rgba);
        let ratio = ThemeTokens::contrast_wcag_ratio(remapped, studio.bg_canvas_2d);
        assert!(
            ratio >= 4.0,
            "remapped {dark_rgba:?} -> {remapped:?} contrast {ratio:.2} too low on Studio canvas"
        );
    }
}

#[test]
fn export_menu_offers_the_png_scale_and_choosing_4x_sets_it() {
    let mut harness = app_harness(StudioTheme::Paper);
    harness.run();
    assert_eq!(harness.state().png_scale, 2, "default 2x (2048 px)");
    harness.get_by_label("Save figure").click();
    harness.run();
    let _ = harness.get_by_label("PNG scale:");
    for label in ["1x", "2x", "4x"] {
        assert_eq!(harness.get_all_by_label(label).count(), 1, "one {label} choice in the open menu");
    }
    harness.get_by_label("4x").click();
    harness.run();
    assert_eq!(harness.state().png_scale, 4);
    if harness.query_by_label("1x").is_none() {
        harness.get_by_label("Save figure").click();
        harness.run();
    }
    harness.get_by_label("1x").click();
    harness.run();
    assert_eq!(harness.state().png_scale, 1);
}
