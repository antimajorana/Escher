//! The one-row HUD over the viewport (`Panel::top("top_bar")`, §4.3). Left (shrinks, truncates): the Codex
//! button and the breadcrumb geometry ▸ group ▸ lattice ▸ mode, each segment opening the deck at its realm.
//! Right (fixed, drawn first): Theme, Search, View, Export, Session, Presets, Redo, Undo and the 2D/3D/Split
//! toggles. `egui::Sides::shrink_left` keeps it one row at any width.

use std::cell::RefCell;

use super::breadcrumb::Segment;
use crate::app::{AppMode, NavTab, PapertillingApp};
use crate::mesh3d::DefectViewMode;

const UNDO_SCOPE: &str = "Undo and Redo apply to Combine, Fuse & Move";
/// Narrowest a breadcrumb segment gets before the row runs out (an ellipsis plus a glyph or two).
const SEG_MIN_WIDTH: f32 = 18.0;

impl PapertillingApp {
    pub(crate) fn show_hud_bar(&mut self, ui: &mut egui::Ui) {
        let resp = egui::Panel::top("top_bar").exact_size(34.0).show(ui, |ui| {
            let height = ui.available_height();
            // `Sides::show` takes both closures at once; they run one after the other, so a RefCell lends `self`
            // to each in turn (never both at the same time).
            let this = RefCell::new(&mut *self);
            egui::Sides::new().height(height).shrink_left().truncate().show(
                ui,
                |ui| this.borrow_mut().hud_left(ui),
                |ui| this.borrow_mut().hud_right(ui),
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

    /// Left side (shrinks): Codex button, then geometry ▸ group ▸ lattice ▸ mode, each a frameless button that
    /// truncates to an ellipsis; the rename bar follows the lattice segment (or the mode segment without one).
    fn hud_left(&mut self, ui: &mut egui::Ui) {
        // Sides lets the shrinking side overflow its max rect; clip it so it never paints over the right cluster.
        // Only the right edge is clamped, so focus / hover outlines on the left are not cut.
        let mut clip = ui.clip_rect();
        clip.max.x = clip.max.x.min(ui.max_rect().right());
        ui.set_clip_rect(clip);
        self.hud_codex_button(ui);
        let bc = self.breadcrumb();
        let has_lattice = bc.lattice.is_some();
        let segs: Vec<(Segment, bool)> = [
            Some((bc.geometry, false)),
            bc.group.map(|s| (s, false)),
            bc.lattice.map(|s| (s, true)),
            Some((bc.mode, false)),
        ]
        .into_iter()
        .flatten()
        .collect();
        let n = segs.len();
        let mut clicked = None;
        for (k, (seg, is_lattice)) in segs.into_iter().enumerate() {
            let Segment { text, hover, realm } = seg;
            let rename_here = is_lattice || (!has_lattice && k + 1 == n);
            ui.weak("▸");
            if is_lattice {
                self.paint_lattice_thumb(ui, 20.0);
            }
            let reserve = if rename_here { self.rename_reserve() } else { 0.0 };
            let share = ((ui.available_width() - reserve) / (n - k) as f32).max(SEG_MIN_WIDTH);
            let resp = ui
                .scope(|ui| {
                    ui.set_max_width(share);
                    ui.add(egui::Button::new(text).frame(false).truncate()).on_hover_text(hover)
                })
                .inner;
            if resp.clicked() {
                clicked = Some(realm);
            }
            if rename_here {
                self.hud_rename_bar(ui);
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

    /// Width kept free for the rename bar while the segments share the rest.
    fn rename_reserve(&self) -> f32 {
        if self.gallery.renaming { 300.0 } else { 90.0 }
    }

    /// The Gallery rename bar, with the revision check that keeps the active tab in sync (from the old top bar).
    fn hud_rename_bar(&mut self, ui: &mut egui::Ui) {
        let rev0 = self.gallery.revision();
        self.gallery.show_rename_bar(ui, &mut self.backend, &mut self.canvas2d, "top_rename");
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
                self.deck_open(self.current_realm(), ui.ctx());
            }
        }
    }

    /// Right cluster, drawn right to left: Theme, Search, View, Export, Session, Presets, Redo, Undo, view toggles.
    fn hud_right(&mut self, ui: &mut egui::Ui) {
        if ui
            .button(format!("Theme: {}", self.theme.label()))
            .on_hover_text("Toggle between Light (Paper) and Dark (Studio) theme")
            .clicked()
        {
            self.theme = self.theme.toggle();
            self.theme.apply(ui.ctx());
        }
        if ui.button("Search (Ctrl+K)").on_hover_text("Search all stored lattices").clicked() {
            if self.deck.open {
                self.deck.focus_filter = true;
            } else {
                self.toggle_switcher();
            }
        }
        ui.menu_button("View", |ui| {
            if ui.button("Reset views").clicked() {
                self.canvas2d.reset();
                self.view3d.reset();
            }
            ui.checkbox(&mut self.left_rail_open, "Sidebar (Ctrl+B)");
            ui.checkbox(&mut self.right_inspector_open, "Inspector (Ctrl+I)");
        });
        self.show_export_menu(ui);
        self.session_menu(ui);
        self.show_presets_menu(ui);
        let in_combine = self.active_mode == AppMode::Defects && self.active_tab == NavTab::Combine;
        let free = in_combine && !self.combine.dragging && !self.combine.cut_dragging;
        let can_redo = free && !self.combine.redo_stack.is_empty();
        if ui.add_enabled(can_redo, egui::Button::new("Redo")).on_disabled_hover_text(UNDO_SCOPE).clicked() {
            self.combine_redo();
        }
        let can_undo = free && !self.combine.undo_stack.is_empty();
        if ui.add_enabled(can_undo, egui::Button::new("Undo")).on_disabled_hover_text(UNDO_SCOPE).clicked() {
            self.combine_undo();
        }
        self.hud_view_toggles(ui);
    }

    /// `2D` / `3D` / `Split` for the tabs with a view mode (Defects, Combine, Analysis); disabled elsewhere.
    fn hud_view_toggles(&mut self, ui: &mut egui::Ui) {
        let target = match self.active_tab {
            NavTab::Defects => Some((&mut self.defects.view_mode, &mut self.defects.view3d_dirty)),
            NavTab::Combine => Some((&mut self.combine.view_mode, &mut self.combine.view3d_dirty)),
            NavTab::Analysis => Some((&mut self.analysis.view_mode, &mut self.analysis.view3d_dirty)),
            _ => None,
        };
        // right to left: Split, 3D, 2D
        let modes = [
            (DefectViewMode::Split2D3D, "Split"),
            (DefectViewMode::Sheet3D, "3D"),
            (DefectViewMode::Planar2D, "2D"),
        ];
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
