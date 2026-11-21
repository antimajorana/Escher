//! Tight-Binding Spectrum & Bound States panel, strictly quarantined to `3. Analysis`.

use super::spectrum_worker::{SpectrumColorOverlay, SpectrumPanelState};
use crate::theme::ThemeTokens;

/// Outcome flags returned from `show_spectrum_panel`.
pub(super) struct SpectrumUiEvent {
    pub recompute: bool,
    pub redraw_colors: bool,
}

/// Render the **Tight-Binding Spectrum** inspector panel (always expanded in `3. Analysis`).
pub(super) fn show_spectrum_panel(
    ui: &mut egui::Ui,
    st: &mut SpectrumPanelState,
    allow_flux: bool,
    allow_seam_sign: bool,
    tokens: &ThemeTokens,
) -> SpectrumUiEvent {
    let mut ev = SpectrumUiEvent { recompute: false, redraw_colors: false };
    st.expanded = true;

    let header_txt = if let Some(res) = &st.result {
        format!(
            "Tight-Binding Spectrum (N={}, zero modes={})",
            res.eigenvalues.len(),
            res.zero_mode_count
        )
    } else {
        String::from("Tight-Binding Spectrum")
    };
    ui.strong(header_txt);
    ui.add_space(2.0);

    ui.horizontal_wrapped(|ui| {
        if ui.button("Compute Spectrum").clicked() {
            ev.recompute = true;
        }
        if st.is_busy() {
            ui.spinner();
            ui.small(if st.stale { "previous result — stale" } else { "diagonalizing…" });
        } else if st.stale {
            ui.small("previous result — stale");
        }
    });

    if let Some(err) = &st.last_error {
        ui.colored_label(tokens.status_warn, err);
    }

    ui.horizontal(|ui| {
        ui.small("Strain β:");
        if ui.add(egui::Slider::new(&mut st.opts.beta, 0.0..=5.0).step_by(0.1)).changed() {
            st.stale = true;
            ev.recompute = true;
        }
    });
    if allow_flux {
        ui.horizontal(|ui| {
            ui.small("Peierls φ:");
            if ui.add(egui::Slider::new(&mut st.opts.flux, -0.5..=0.5).step_by(0.025)).changed() {
                st.stale = true;
                ev.recompute = true;
            }
        });
    }
    if allow_seam_sign {
        ui.horizontal(|ui| {
            ui.small("Wc seam Z₂:");
            for sgn in [1_i32, -1_i32] {
                let lbl = if sgn == 1 { "+1" } else { "-1 (π flux)" };
                if ui.selectable_label(st.opts.seam_sign == sgn, lbl).clicked() && st.opts.seam_sign != sgn {
                    st.opts.seam_sign = sgn;
                    st.stale = true;
                    ev.recompute = true;
                }
            }
        });
    }

    let mut e_lo = -4.0_f64;
    let mut e_hi = 4.0_f64;
    if let Some(res) = &st.result && !res.eigenvalues.is_empty() {
        e_lo = res.eigenvalues[0];
        e_hi = *res.eigenvalues.last().unwrap();
        if (e_hi - e_lo).abs() < 1e-6 {
            e_lo -= 1.0;
            e_hi += 1.0;
        }
    }
    ui.horizontal(|ui| {
        ui.small("μ:");
        if ui.add(egui::Slider::new(&mut st.opts.mu, e_lo..=e_hi).step_by(0.02)).changed() {
            st.stale = true;
            ev.recompute = true;
        }
    });

    ui.horizontal_wrapped(|ui| {
        ui.small("Site color:");
        for mode in [
            SpectrumColorOverlay::None,
            SpectrumColorOverlay::LdosMultiplet,
            SpectrumColorOverlay::SpectralCharge,
        ] {
            if ui.selectable_label(st.color_overlay == mode, mode.label()).clicked() && st.color_overlay != mode {
                st.color_overlay = mode;
                if mode != SpectrumColorOverlay::None && st.result.is_none() {
                    ev.recompute = true;
                }
                ev.redraw_colors = true;
            }
        }
    });

    if let Some(res) = st.result.clone() && !res.eigenvalues.is_empty() {
        draw_eigenvalue_strip(ui, st, &res.eigenvalues, &res.core_weight, &mut ev, tokens);
        let sel_idx = st.selected_level.unwrap_or_else(|| {
            res.eigenvalues
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (a.abs()).total_cmp(&(b.abs())))
                .map_or(0, |(i, _)| i)
        });
        let e_sel = res.eigenvalues.get(sel_idx).copied().unwrap_or(0.0);
        let w_sel = res.core_weight.get(sel_idx).copied().unwrap_or(0.0);
        let mult_dim = res
            .multiplets
            .iter()
            .find(|m| sel_idx >= m.start && sel_idx < m.end)
            .map_or(1, |m| m.dim);
        ui.small(format!(
            "Zero modes (|E|<1e-6): {} | Level #{}: E={e_sel:+.4} (deg={mult_dim}, W_core={w_sel:.2})",
            res.zero_mode_count, sel_idx
        ));
        if !res.delta_q.is_empty() {
            let qs: Vec<String> = res.delta_q.iter().take(6).map(|q| format!("{q:+.4}")).collect();
            let extra = if res.delta_q.len() > 6 { "…" } else { "" };
            ui.small(format!("Core/cone δQ: [{}{}]", qs.join(", "), extra));
        }
    }

    ev
}

fn draw_eigenvalue_strip(
    ui: &mut egui::Ui,
    st: &mut SpectrumPanelState,
    evals: &[f64],
    core_weight: &[f64],
    ev: &mut SpectrumUiEvent,
    tokens: &ThemeTokens,
) {
    let desired = egui::vec2(ui.available_width().max(180.0), 42.0);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, tokens.bg_card);
    painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, tokens.border_subtle), egui::StrokeKind::Inside);

    let e_min = evals[0];
    let e_max = *evals.last().unwrap();
    let span = (e_max - e_min).max(1e-6);
    let to_x = |e: f64| rect.left() + 6.0 + (((e - e_min) / span) as f32) * (rect.width() - 12.0);

    // Fermi level mu marker
    let mx = to_x(st.opts.mu.clamp(e_min, e_max));
    painter.line_segment([egui::pos2(mx, rect.top() + 2.0), egui::pos2(mx, rect.bottom() - 2.0)], egui::Stroke::new(1.8, tokens.status_ok));

    let bulk_col = egui::Color32::from_rgba_unmultiplied(
        tokens.text_muted.r(),
        tokens.text_muted.g(),
        tokens.text_muted.b(),
        145,
    );
    for (i, &e) in evals.iter().enumerate() {
        let x = to_x(e);
        let w = core_weight.get(i).copied().unwrap_or(0.0);
        let sel = st.selected_level == Some(i);
        let col = if sel {
            tokens.accent_primary
        } else if w > 0.35 {
            tokens.status_err
        } else {
            bulk_col
        };
        let y0 = if w > 0.35 || sel { rect.top() + 5.0 } else { rect.top() + 12.0 };
        painter.line_segment([egui::pos2(x, y0), egui::pos2(x, rect.bottom() - 5.0)], egui::Stroke::new(if sel { 2.4 } else { 1.2 }, col));
    }

    if (resp.clicked() || resp.dragged()) && let Some(pos) = resp.interact_pointer_pos() {
        let frac = ((pos.x - rect.left() - 6.0) / (rect.width() - 12.0).max(1.0)).clamp(0.0, 1.0) as f64;
        let target_e = e_min + frac * span;
        let best = evals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((*a - target_e).abs()).total_cmp(&(*b - target_e).abs()))
            .map(|(i, _)| i);
        if best != st.selected_level {
            st.selected_level = best;
            if st.color_overlay == SpectrumColorOverlay::None {
                st.color_overlay = SpectrumColorOverlay::LdosMultiplet;
            }
            ev.redraw_colors = true;
        }
    }
    resp.on_hover_text("Click or drag to select an eigenstate multiplet (red = core-localized, green line = Fermi level μ)");
}
