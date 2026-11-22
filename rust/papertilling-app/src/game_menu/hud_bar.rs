//! The one-row top bar (`Panel::top("top_bar")`).
//! - Default (`Advanced` off): `Lattice | Defects` mode switch + inline sub-view switch (`Tiling | Surface` or
//!   `Allowed defects | Place & fuse`) + active lattice/group chip (`honeycomb · p6m (*632)`) + `Rename` on the left;
//!   `Advanced` toggle at the rightmost end of the row, followed by `Theme`, `Find lattice`, `View`, `Save figure`,
//!   `Session`, `Examples`, and `Undo` / `Redo`.
//! - `Advanced` on: reveals `All views (Tab)` and clickable breadcrumb segments on the left while keeping `Advanced`
//!   at the rightmost end.

use std::cell::RefCell;

use super::breadcrumb::Segment;
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};

const UNDO_SCOPE: &str = "Undo and Redo apply to Place & fuse";
/// Narrowest a breadcrumb segment gets before it is hidden (an ellipsis plus a glyph or two).
const SEG_MIN_WIDTH: f32 = 18.0;
/// Side of the lattice thumbnail before the lattice segment.
const THUMB_PX: f32 = 20.0;
/// Window widths at which the right cluster folds: below the first, View / Save figure / Session / Examples go into
/// one `Menu`; below the second, Undo / Redo go there too.
const FULL_MIN_WIDTH: f32 = 1240.0;
const COMPACT_MIN_WIDTH: f32 = 960.0;
/// Smallest window width (the viewport's `min_inner_size`).
pub(crate) const HUD_MIN_WINDOW_WIDTH: f32 = 720.0;

/// How much of the right cluster stays on the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HudDensity {
    /// Every control on the row.
    Full,
    /// View / Save figure / Session / Examples in `Menu`; Find lattice shortened.
    Compact,
    /// Also Undo / Redo in `Menu`.
    Narrow,
}

impl HudDensity {
    pub(crate) fn for_width(w: f32) -> Self {
        if w >= FULL_MIN_WIDTH {
            Self::Full
        } else if w >= COMPACT_MIN_WIDTH {
            Self::Compact
        } else {
            Self::Narrow
        }
    }
}

/// One breadcrumb slot: the segment, whether it is the lattice segment, and its hide priority (0 is kept longest).
struct Slot {
    seg: Segment,
    lattice: bool,
    prio: u8,
}

impl PapertillingApp {
    pub(crate) fn show_hud_bar(&mut self, ui: &mut egui::Ui) {
        let resp = egui::Panel::top("top_bar").exact_size(34.0).show(ui, |ui| {
            let height = ui.available_height();
            let density = HudDensity::for_width(ui.ctx().content_rect().width());
            self.deck.hud_density = density;
            let this = RefCell::new(&mut *self);
            egui::Sides::new().height(height).shrink_left().truncate().show(
                ui,
                |ui| this.borrow_mut().hud_left(ui, density),
                |ui| this.borrow_mut().hud_right(ui, density),
            );
        });
        let rect = resp.response.rect;
        let content = ui.ctx().content_rect();
        self.deck.host_rect.min = egui::pos2(content.left(), rect.bottom());
        self.deck.host_rect.max.x = content.right();
        #[cfg(test)]
        {
            self.test_rects.top = rect;
        }
    }

    fn hud_left(&mut self, ui: &mut egui::Ui, density: HudDensity) {
        let mut clip = ui.clip_rect();
        clip.max.x = clip.max.x.min(ui.max_rect().right());
        ui.set_clip_rect(clip);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        if self.advanced {
            self.hud_left_advanced(ui, density);
        } else {
            self.hud_left_default(ui, density);
        }
    }

    /// Default two-mode left bar (`Advanced` off): `Lattice | Defects`, inline sub-view switch, active lattice chip,
    /// and inline `Rename`.
    fn hud_left_default(&mut self, ui: &mut egui::Ui, density: HudDensity) {
        let narrow_renaming = self.gallery.renaming && density == HudDensity::Narrow;
        if !narrow_renaming {
            self.hud_mode_switch(ui);
            ui.separator();
            match self.active_mode {
                AppMode::Lattices => {
                    for (v, label) in [(LatticesView::Plane2D, "Tiling"), (LatticesView::Manifolds, "Surface")] {
                        if ui.selectable_label(self.lattices_view == v, label).clicked() && self.lattices_view != v {
                            self.select_lattices_view(v);
                        }
                    }
                }
                AppMode::Defects => {
                    for (v, label) in [
                        (DefectsView::Catalogue2D, "Allowed defects"),
                        (DefectsView::CombineAndFuse, "Place & fuse"),
                        (DefectsView::FuseAndCompare, "Fuse & compare"),
                    ] {
                        if ui.selectable_label(self.defects_view == v, label).clicked() && self.defects_view != v {
                            self.select_defects_view(v);
                        }
                    }
                }
                AppMode::Analysis => {}
            }
        }
        if !self.view_is_lattice_driven() {
            // The view shows no Gallery lattice (Fusion steps): no lattice chip and no Rename.
            return;
        }
        let sp = ui.spacing().item_spacing.x;
        let reserve = self.rename_reserve(sp);
        let room = ui.available_width() - reserve;
        let thumb_overhead = THUMB_PX + 3.0 * sp + 8.0;
        if room >= thumb_overhead + SEG_MIN_WIDTH {
            ui.separator();
            self.paint_lattice_thumb(ui, THUMB_PX);
            let it = self.gallery.active_item();
            let grp = crate::gallery_tab::filter::group_with_orbifold(&it.group);
            let ro = &self.gallery.readout;
            let raised = (!ro.status_ok && !ro.active_group.is_empty() && ro.active_group != it.group)
                .then_some(ro.active_group.as_str());
            let chip = match raised {
                Some(big) => format!("{} · {grp} → {big}", it.label),
                None => format!("{} · {grp}", it.label),
            };
            let hover = format!(
                "{} · group {grp} · {} sites/cell",
                it.display_with_orig(),
                it.sites_per_cell
            );
            let max_w = (ui.available_width() - reserve).max(SEG_MIN_WIDTH);
            ui.scope(|ui| {
                ui.set_max_width(max_w);
                ui.add(egui::Label::new(chip).truncate()).on_hover_text(hover);
            });
        }
        self.hud_rename_bar(ui);
    }

    /// `Advanced` left bar: `All views (Tab)`, clickable breadcrumb segments, and `Rename`.
    fn hud_left_advanced(&mut self, ui: &mut egui::Ui, _density: HudDensity) {
        self.hud_all_views_button(ui);
        let bc = self.breadcrumb();
        let has_lattice = bc.lattice.is_some();
        if !has_lattice && self.gallery.renaming {
            self.gallery.cancel_rename();
        }
        let slots: Vec<Slot> = [
            Some(Slot { seg: bc.geometry, lattice: false, prio: 3 }),
            bc.group.map(|seg| Slot { seg, lattice: false, prio: 1 }),
            bc.lattice.map(|seg| Slot { seg, lattice: true, prio: 0 }),
            Some(Slot { seg: bc.mode, lattice: false, prio: 2 }),
        ]
        .into_iter()
        .flatten()
        .collect();
        let sp = ui.spacing().item_spacing.x;
        let arrow = egui::WidgetText::from("▸").into_galley(ui, None, f32::INFINITY, egui::TextStyle::Body).size().x;
        let overhead = |s: &Slot| arrow + 2.0 * sp + if s.lattice { THUMB_PX + sp } else { 0.0 };
        let mut reserve = if has_lattice { self.rename_reserve(sp) } else { 0.0 };
        let mut room = ui.available_width() - reserve;
        let mut keep = vec![false; slots.len()];
        let mut order: Vec<usize> = (0..slots.len()).collect();
        order.sort_by_key(|&i| slots[i].prio);
        for i in order {
            let need = overhead(&slots[i]) + SEG_MIN_WIDTH;
            if room >= need {
                keep[i] = true;
                room -= need;
            }
        }
        let mut clicked = None;
        for (i, slot) in slots.iter().enumerate() {
            if keep[i] {
                let rest: f32 = (i..slots.len()).filter(|&j| keep[j]).map(|j| overhead(&slots[j])).sum();
                let count = (i..slots.len()).filter(|&j| keep[j]).count() as f32;
                let share = ((ui.available_width() - reserve - rest) / count).max(SEG_MIN_WIDTH);
                ui.weak("▸");
                if slot.lattice {
                    self.paint_lattice_thumb(ui, THUMB_PX);
                }
                let resp = ui
                    .scope(|ui| {
                        ui.set_max_width(share);
                        ui.add(egui::Button::new(slot.seg.text.as_str()).frame(false).truncate())
                            .on_hover_text(slot.seg.hover.as_str())
                    })
                    .inner;
                if resp.clicked() {
                    clicked = Some(slot.seg.realm);
                }
            }
            if slot.lattice {
                self.hud_rename_bar(ui);
                reserve = 0.0;
            }
        }
        if let Some(realm) = clicked {
            if self.deck.open {
                self.deck_select_realm(realm);
            } else {
                self.deck_open(realm, ui.ctx());
            }
        }
    }

    fn hud_mode_switch(&mut self, ui: &mut egui::Ui) {
        for (mode, label) in [(AppMode::Lattices, "Lattice"), (AppMode::Defects, "Defects")] {
            if ui.selectable_label(self.active_mode == mode, label).clicked() && self.active_mode != mode {
                if self.deck.open {
                    self.deck_close();
                }
                self.switch_mode(mode);
            }
        }
    }

    /// Width kept free for the rename controls: what they took last frame in the same state, else an estimate.
    fn rename_reserve(&self, sp: f32) -> f32 {
        let renaming = self.gallery.renaming;
        match self.deck.hud_rename_w {
            Some((was_renaming, w)) if was_renaming == renaming => w + sp,
            _ if renaming => 300.0,
            _ => 110.0,
        }
    }

    fn hud_rename_bar(&mut self, ui: &mut egui::Ui) {
        let rev0 = self.gallery.revision();
        let renaming = self.gallery.renaming;
        let w = ui
            .scope(|ui| self.gallery.show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "top_rename"))
            .response
            .rect
            .width();
        self.deck.hud_rename_w = Some((renaming, w));
        if self.gallery.revision() != rev0 {
            self.sync_active_tab_after_rename();
        }
    }

    fn hud_all_views_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("All views (Tab)")
            .on_hover_text("All geometries, symmetry groups, tilings, surfaces, and defect views")
            .clicked()
        {
            if self.deck.open {
                self.deck_close();
            } else {
                let realm = self.deck_entry_realm();
                self.deck_open(realm, ui.ctx());
            }
        }
    }

    /// Right cluster, drawn right to left: `Advanced` sits at the rightmost end, followed by Theme, Find lattice,
    /// View, Save figure, Session, Examples (or `Menu` when narrower), and Redo / Undo.
    fn hud_right(&mut self, ui: &mut egui::Ui, density: HudDensity) {
        let mut adv = self.advanced;
        if ui
            .toggle_value(&mut adv, "Advanced")
            .on_hover_text("Show non-Euclidean tilings, 3D space/layer groups, defect detection, and extra exports")
            .changed()
        {
            self.set_advanced(adv);
        }
        if ui
            .button(format!("Theme: {}", self.theme.label()))
            .on_hover_text("Toggle between Light and Dark theme")
            .clicked()
        {
            self.theme = self.theme.toggle();
            self.theme.apply(ui.ctx());
        }
        let search = if density == HudDensity::Full { "Find lattice (Ctrl+K)" } else { "Find" };
        if ui.button(search).on_hover_text("Find any stored 2D lattice (Ctrl+K)").clicked() {
            if self.deck.open {
                self.deck.focus_filter = true;
            } else {
                self.toggle_switcher();
            }
        }
        if density == HudDensity::Full {
            ui.menu_button("View", |ui| self.hud_view_menu(ui));
            self.show_export_menu(ui);
            self.session_menu(ui);
            self.show_presets_menu(ui);
        } else {
            ui.menu_button("Menu", |ui| {
                ui.menu_button("View", |ui| self.hud_view_menu(ui));
                self.show_export_menu(ui);
                self.session_menu(ui);
                self.show_presets_menu(ui);
                if density == HudDensity::Narrow {
                    ui.separator();
                    self.hud_undo_redo(ui, true);
                }
            });
        }
        if density != HudDensity::Narrow {
            self.hud_undo_redo(ui, false);
        }
    }

    fn hud_view_menu(&mut self, ui: &mut egui::Ui) {
        if ui.button("Reset camera").clicked() {
            self.canvas2d.reset();
            self.view3d.reset();
        }
        ui.checkbox(&mut self.left_rail_open, "Left panel (Ctrl+B)");
        ui.checkbox(&mut self.right_inspector_open, "Right panel (Ctrl+I)");
    }

    /// Undo / Redo for Place & fuse (`in_menu`: Undo first, as a menu reads top to bottom).
    fn hud_undo_redo(&mut self, ui: &mut egui::Ui, in_menu: bool) {
        let in_combine = self.active_mode == AppMode::Defects && self.active_tab == NavTab::Combine;
        let free = in_combine && !self.combine.dragging && !self.combine.cut_dragging;
        let can_redo = free && !self.combine.redo_stack.is_empty();
        let can_undo = free && !self.combine.undo_stack.is_empty();
        let order = if in_menu { [false, true] } else { [true, false] };
        for redo in order {
            let (can, label) = if redo { (can_redo, "Redo") } else { (can_undo, "Undo") };
            if ui.add_enabled(can, egui::Button::new(label)).on_disabled_hover_text(UNDO_SCOPE).clicked() {
                if redo {
                    self.combine_redo();
                } else {
                    self.combine_undo();
                }
            }
        }
    }
}
