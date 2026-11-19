//! The one-row HUD over the viewport (`Panel::top("top_bar")`, §4.3). Left (shrinks): the Codex button, the
//! breadcrumb geometry ▸ group ▸ lattice ▸ mode (each segment opens the deck at its realm) and the Gallery rename
//! controls after the lattice segment. Right (fixed, drawn first): Theme, Search, View, Export, Session, Presets,
//! Redo, Undo and the 2D/3D/Split toggles; narrower windows fold them into one `Menu` ([`HudDensity`]).
//!
//! Width rule on the left: the Codex button and the rename controls always keep their full width; breadcrumb
//! segments share what is left, truncate to an ellipsis, and are hidden entirely (geometry first, then mode,
//! then group, the lattice last) when even an ellipsis no longer fits.

use std::cell::RefCell;

use super::breadcrumb::Segment;
use crate::app::{AppMode, NavTab, PapertillingApp};
use crate::mesh3d::DefectViewMode;

const UNDO_SCOPE: &str = "Undo and Redo apply to Combine, Fuse & Move";
/// Narrowest a breadcrumb segment gets before it is hidden (an ellipsis plus a glyph or two).
const SEG_MIN_WIDTH: f32 = 18.0;
/// Side of the lattice thumbnail before the lattice segment.
const THUMB_PX: f32 = 20.0;
/// Window widths at which the right cluster folds: below the first, View / Export / Session / Presets go into one
/// `Menu`; below the second, Undo / Redo and the 2D / 3D / Split toggles go there too.
const FULL_MIN_WIDTH: f32 = 1240.0;
const COMPACT_MIN_WIDTH: f32 = 960.0;
/// Smallest window width (the viewport's `min_inner_size`): the narrow HUD still fits the Codex button and the
/// rename controls in full there (checked by `tests_review::hud_keeps_codex_and_rename_whole`).
pub(crate) const HUD_MIN_WINDOW_WIDTH: f32 = 720.0;

/// How much of the right cluster stays on the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HudDensity {
    /// Every control on the row.
    Full,
    /// View / Export / Session / Presets in `Menu`; Search shortened.
    Compact,
    /// Also Undo / Redo and the view toggles in `Menu`.
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
            // `Sides::show` takes both closures at once; they run one after the other, so a RefCell lends `self`
            // to each in turn (never both at the same time).
            let this = RefCell::new(&mut *self);
            egui::Sides::new().height(height).shrink_left().truncate().show(
                ui,
                |ui| this.borrow_mut().hud_left(ui),
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

    fn hud_left(&mut self, ui: &mut egui::Ui) {
        // Sides lets the shrinking side overflow its max rect; clip it so it never paints over the right cluster.
        // Only the right edge is clamped, so focus / hover outlines on the left are not cut.
        let mut clip = ui.clip_rect();
        clip.max.x = clip.max.x.min(ui.max_rect().right());
        ui.set_clip_rect(clip);
        // `Sides::truncate` would shorten every label; only the breadcrumb segments may truncate (they ask for it).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        self.hud_codex_button(ui);
        let bc = self.breadcrumb();
        let has_lattice = bc.lattice.is_some();
        if !has_lattice && self.gallery.renaming {
            // the view no longer shows the lattice, so its rename controls are gone: end the rename rather than
            // leave it pending with no field (it would keep holding the deck keys)
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
        // the rename controls come first; then keep segments in priority order while an ellipsis still fits
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
                // drawn even when the lattice segment itself is hidden: renaming is never cut off
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

    /// Width kept free for the rename controls: what they took last frame in the same state, else an estimate.
    fn rename_reserve(&self, sp: f32) -> f32 {
        let renaming = self.gallery.renaming;
        match self.deck.hud_rename_w {
            Some((was_renaming, w)) if was_renaming == renaming => w + sp,
            _ if renaming => 300.0,
            _ => 120.0,
        }
    }

    /// The Gallery rename bar (measured for the next frame's reserve), with the revision check that keeps the
    /// active tab in sync (from the old top bar).
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

    fn hud_codex_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("❖ Codex (Tab)")
            .on_hover_text("Command Deck: every geometry, group, tiling, manifold, defect and preset")
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

    /// Right cluster, drawn right to left: Theme, Search, then View, Export, Session, Presets (or `Menu`), Redo,
    /// Undo and the view toggles (in `Menu` too when narrow).
    fn hud_right(&mut self, ui: &mut egui::Ui, density: HudDensity) {
        if ui
            .button(format!("Theme: {}", self.theme.label()))
            .on_hover_text("Toggle between Light (Paper) and Dark (Studio) theme")
            .clicked()
        {
            self.theme = self.theme.toggle();
            self.theme.apply(ui.ctx());
        }
        let search = if density == HudDensity::Full { "Search (Ctrl+K)" } else { "Search" };
        if ui.button(search).on_hover_text("Search all stored lattices (Ctrl+K)").clicked() {
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
                    ui.menu_button("Layout", |ui| self.hud_view_toggles(ui, true));
                }
            });
        }
        if density != HudDensity::Narrow {
            self.hud_undo_redo(ui, false);
            self.hud_view_toggles(ui, false);
        }
    }

    fn hud_view_menu(&mut self, ui: &mut egui::Ui) {
        if ui.button("Reset views").clicked() {
            self.canvas2d.reset();
            self.view3d.reset();
        }
        ui.checkbox(&mut self.left_rail_open, "Sidebar (Ctrl+B)");
        ui.checkbox(&mut self.right_inspector_open, "Inspector (Ctrl+I)");
    }

    /// Undo / Redo for Combine (`in_menu`: Undo first, as a menu reads top to bottom).
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

    /// `2D` / `3D` / `Split` for the tabs with a view mode (Defects, Combine, Analysis); disabled elsewhere.
    /// `in_menu`: 2D first (top to bottom); on the row they are drawn right to left.
    fn hud_view_toggles(&mut self, ui: &mut egui::Ui, in_menu: bool) {
        let target = match self.active_tab {
            NavTab::Defects => Some((&mut self.defects.view_mode, &mut self.defects.view3d_dirty)),
            NavTab::Combine => Some((&mut self.combine.view_mode, &mut self.combine.view3d_dirty)),
            NavTab::Analysis => Some((&mut self.analysis.view_mode, &mut self.analysis.view3d_dirty)),
            _ => None,
        };
        let mut modes = [
            (DefectViewMode::Split2D3D, "Split"),
            (DefectViewMode::Sheet3D, "3D"),
            (DefectViewMode::Planar2D, "2D"),
        ];
        if in_menu {
            modes.reverse();
        }
        match target {
            Some((mode, dirty)) => {
                for (vm, label) in modes {
                    if ui.selectable_label(*mode == vm, label).clicked() && *mode != vm {
                        *mode = vm;
                        *dirty = true;
                    }
                }
            }
            None => {
                for (_, label) in modes {
                    ui.add_enabled(false, egui::Button::selectable(false, label))
                        .on_disabled_hover_text("This view has a fixed layout");
                }
            }
        }
    }
}
