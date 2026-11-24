//! Applying a curated preset (`crate::presets`): lattice, tab and the tab's settings, then the tab's normal builds.
//!
//! Sequence (the same for every preset, so the result never depends on what was on screen before):
//! 1. go to the Gallery, select the preset lattice (`select_lattice_and_sync`) and reset its free parameters to
//!    their defaults (a preset names the stored lattice, not a tuned one);
//! 2. set every field the target tab owns (Gallery view toggles are set here, while the Gallery is drawn);
//! 3. `switch_tab(target)`: the tab's own (re)load path runs, i.e. the off-thread `LatestWorker` builds of Defects
//!    and Combine, the synchronous surface build of a tab switch, the Atlas scene build;
//! 4. Defects: the catalogue indices are set after the switch (a lattice change resets them) and one more build is
//!    submitted (the worker keeps only the newest job). Combine: the pins need the lattice's Combine catalogue, which
//!    loads off-thread after a lattice change; they are parked in `preset_pending` and placed by
//!    `pump_preset_pending` (every frame) once it is loaded, after ONE undo checkpoint of the pins they replace when
//!    those are the user's pins on the same lattice (a lattice reload drops pins and undo history, as any lattice
//!    change does, and is reported in the status line).

use papertilling_core::defect::surgery::DefectSpec;

use super::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp, item_key};
use crate::aliases::LatticeAliasKey;
use crate::analysis_mode::AnalysisTarget;
use crate::atlas_tab::AtlasSubTab;
use crate::backend::{CatalogueItemUi, CentreModeUi};
use crate::canvas2d::gauge_overlay::CrystalOverlayMode;
use crate::defects_inspectors::DefectInspectors;
use crate::defects_tab::DefectSubTab;
use crate::live_combine::{CombineLatticeInfo, preset_neighbour_pair};
use crate::mesh3d::{DefectViewMode, Sheet3dColorMode, Sheet3dViewOpts, SurfaceColor3d};
use crate::presets::{AtlasScene, CombinePins, PRESETS, PresetLattice, PresetSettings, SurfaceSize};
use crate::scene2d::overlays::GalleryOverlayToggles;

/// Async follow-up work for a preset waiting on a tab's background worker.
pub(crate) enum PendingPreset {
    Combine {
        name: &'static str,
        alias_key: LatticeAliasKey,
        pins: CombinePins,
        /// Push one undo checkpoint of the replaced pins (they are the user's: Combine kept its lattice).
        checkpoint: bool,
    },
    CombineSpecs {
        item_key: super::ItemKey,
        specs: Vec<DefectSpec>,
        checkpoint: bool,
    },
    AnalysisSpectral {
        alias_key: LatticeAliasKey,
    },
}

/// The specs `pins` stands for on a lattice with Combine catalogue `info` (empty: not possible on this lattice).
pub(crate) fn preset_specs(pins: CombinePins, info: &CombineLatticeInfo) -> Vec<DefectSpec> {
    match pins {
        CombinePins::NeighbourDisclinationPair => {
            if info.centres.is_empty() { Vec::new() } else { preset_neighbour_pair(info) }
        }
        CombinePins::DislocationDipole { cells } => {
            let Some(d0) = info.dislocations.first() else { return Vec::new() };
            let (b, at) = (d0.b, d0.at);
            let cross = |v: [f64; 2]| (b[0] * v[1] - b[1] * v[0]).abs() / v[0].hypot(v[1]);
            let (a1, a2) = (info.cell.a1, info.cell.a2);
            let step = if cross(a1) >= cross(a2) { a1 } else { a2 };
            let c = f64::from(cells);
            let at2 = [at[0] + c * step[0], at[1] + c * step[1]];
            vec![DefectSpec::dislocation(b, Some(at)), DefectSpec::dislocation([-b[0], -b[1]], Some(at2))]
        }
    }
}

impl PapertillingApp {
    /// The stored lattice a preset names (`None`: unknown key or index).
    pub(crate) fn preset_lattice_item(&self, lattice: PresetLattice) -> Option<CatalogueItemUi> {
        match lattice {
            PresetLattice::Named(key) => self.backend.find_named(key).cloned(),
            PresetLattice::Catalogue { group, rule, index } => {
                self.backend.catalogue_for_group(group, rule).get(index).cloned()
            }
        }
    }

    /// Apply preset `idx` of [`PRESETS`] (see the module documentation). Never waits for a build.
    pub(crate) fn apply_preset(&mut self, idx: usize) -> Result<(), String> {
        let preset = *PRESETS.get(idx).ok_or_else(|| format!("no preset #{idx}"))?;
        let item = match preset.lattice {
            Some(l) => Some(self.preset_lattice_item(l).ok_or_else(|| format!("Preset '{}': lattice {l:?} not found", preset.name))?),
            None => None,
        };
        self.preset_pending = None;
        self.session.pending_combine = None;
        self.canvas2d.reset();
        self.view3d.reset();
        self.switch_tab(NavTab::Gallery);
        self.gallery.cancel_rename();
        if let Some(item) = item {
            self.select_lattice_and_sync(item);
            self.gallery.finish_pending_retune(&mut self.canvas2d);
            self.gallery.reset_params(&mut self.canvas2d);
        }
        match preset.settings {
            PresetSettings::Gallery { repeat, symmetry, voronoi, unit_cell } => {
                self.gallery.repeat = repeat.clamp(1, 6);
                self.gallery.show_unit_cell = unit_cell;
                self.gallery.overlays = GalleryOverlayToggles { symmetry, voronoi };
                self.gallery.set_inflation_enabled(false, &mut self.canvas2d);
            }
            PresetSettings::Surfaces { target, size, shape } => {
                let s = &mut self.surfaces;
                s.target = target;
                s.show_cover = false;
                s.elastic.shape_mode = shape;
                s.elastic.kappa_b = 0.02;
                // the auto build's corner classes: Auto (face centre before site), as every preset count assumes
                s.params.centre_mode = CentreModeUi::Auto;
                s.elastic.color_mode = SurfaceColor3d::FaceRingSize;
                match size {
                    SurfaceSize::Auto => self.surfaces.auto_size = true,
                    SurfaceSize::Manual { h, k } => {
                        self.surfaces.auto_size = false;
                        self.surfaces.bc.explicit = false; // the preset's (h, k) size the boundary options
                        (self.surfaces.params.h, self.surfaces.params.k) = (h, k);
                    }
                }
            }
            PresetSettings::Defects { sub_tab, radius, draw, view, colour, .. } => {
                let d = &mut self.defects;
                d.sub_tab = sub_tab;
                d.wall_half = false;
                d.radius = radius;
                d.draw_mode = draw;
                d.view_mode = view;
                reset_sheet_and_inspectors(&mut d.inspectors, &mut d.sheet_opts, colour);
                d.view3d_dirty = true;
            }
            PresetSettings::Combine { pins, view } => {
                let key = self.gallery.active_item().alias_key.clone();
                let had_override = self.combine.override_item.is_some();
                if had_override {
                    // an imported .json lattice is shown: the preset is for the Gallery lattice, so reload for it
                    let it = self.gallery.active_item().clone();
                    self.combine_seen = item_key(&it);
                    self.combine.reload_for_item(&it, &self.backend, &mut self.canvas2d, false);
                }
                // The pins being replaced are the user's only when Combine keeps its lattice (no reload on the switch
                // below, no catalogue load running): only then is there something for the one undo checkpoint to
                // restore. A reload (other lattice, retuned lattice, imported .json) drops pins and undo history,
                // as any lattice change does; the seed pair it places meanwhile is not checkpointed.
                let checkpoint = !had_override
                    && item_key(self.gallery.active_item()) == self.combine_seen
                    && !self.combine.is_initializing();
                let c = &mut self.combine;
                c.view_mode = view;
                c.opts = crate::combine::default_build_opts(c.info.r_auto_base);
                c.grid = 1;
                reset_sheet_and_inspectors(&mut c.inspectors, &mut c.sheet_opts, Sheet3dColorMode::FaceRingSize);
                c.show_branch_cuts = false;
                c.show_glide_lines = true;
                c.motion_mode = crate::combine::MotionMode::Conserved;
                c.tool_mode = crate::combine::ToolMode::PanAndDrag;
                c.dragging = false;
                c.clear_kinematic_drag_state(false);
                c.clear_selection();
                c.json_status.clear();
                c.view3d_dirty = true;
                self.preset_pending = Some(PendingPreset::Combine {
                    name: preset.name,
                    alias_key: key,
                    pins,
                    checkpoint,
                });
            }
            PresetSettings::Fuse { reaction } => {
                let it = self.gallery.active_item().clone();
                self.fuse_seen = item_key(&it);
                self.fuse.sync_views = true;
                self.fuse.show_burgers_loop = true;
                self.fuse.show_target_sites = true;
                self.fuse.relaxed = true;
                self.fuse.auto_radius = true;
                self.fuse.reload_for_item(&it, &mut self.canvas2d);
                self.fuse.apply_preset(reaction, &it, true);
            }
            PresetSettings::Analysis {
                target,
                defect_sub_tab,
                defect_index,
                radius,
                draw,
                view,
                overlay,
            } => {
                let it = self.gallery.active_item().clone();
                self.defects_seen = item_key(&it);
                self.defects.reload_for_item(&it, &self.backend, &mut self.canvas2d, false);
                let d = &mut self.defects;
                d.sub_tab = defect_sub_tab;
                d.wall_half = false;
                d.radius = radius;
                d.draw_mode = draw;
                d.view_mode = DefectViewMode::Split2D3D;
                reset_sheet_and_inspectors(&mut d.inspectors, &mut d.sheet_opts, Sheet3dColorMode::FaceRingSize);
                match defect_sub_tab {
                    DefectSubTab::Dislocations => d.selected_disloc = defect_index,
                    DefectSubTab::Disclinations => d.selected_disclin = defect_index,
                    DefectSubTab::Walls => d.selected_wall = defect_index,
                    DefectSubTab::Combinations => {}
                }
                d.submit_rebuild(&it, true);
                let a = &mut self.analysis;
                a.target = target;
                a.view_mode = view;
                a.reset_transient_for_session();
                a.spectrum.opts = Default::default();
                a.spectrum.color_overlay = overlay;
                a.view3d_dirty = true;
                self.preset_pending = Some(PendingPreset::AnalysisSpectral { alias_key: it.alias_key });
            }
            PresetSettings::Atlas(scene) => match scene {
                AtlasScene::SpaceGroup { number, net, defect } => {
                    self.atlas.set_pristine_only(false);
                    self.atlas.sub_tab = AtlasSubTab::SpaceGroups3D;
                    self.atlas.last_defect_subtab = AtlasSubTab::SpaceGroups3D;
                    let s = &mut self.atlas.space3d;
                    (s.selected_sg, s.selected_net, s.selected_defect) = (number, net, defect);
                    s.filter.clear();
                    s.view = Default::default();
                    s.ensure_cached();
                }
                AtlasScene::Hyperbolic { p, q, style, layers } => {
                    self.atlas.set_pristine_only(true);
                    self.atlas.sub_tab = AtlasSubTab::HyperbolicH2;
                    let h = &mut self.atlas.hyperbolic;
                    (h.p, h.q, h.style, h.layers, h.wedge_delta) = (p, q, style, layers, 0);
                    h.disk_proj = crate::atlas_tab::hyperbolic_ui::DiskProjection::Poincare;
                    h.pan = Default::default();
                    h.disk_zoom = 1.0;
                    h.picked = None;
                    h.hovered = None;
                    h.view_changed = false;
                }
            },
        }
        match preset.settings {
            PresetSettings::Gallery { .. } => self.select_lattices_view(LatticesView::Plane2D),
            PresetSettings::Surfaces { .. } => self.select_lattices_view(LatticesView::Manifolds),
            PresetSettings::Defects { .. } => self.select_defects_view(DefectsView::Catalogue2D),
            PresetSettings::Combine { .. } => self.select_defects_view(DefectsView::CombineAndFuse),
            PresetSettings::Fuse { .. } => self.select_defects_view(DefectsView::FuseAndCompare),
            PresetSettings::Analysis { .. } => self.switch_mode(AppMode::Analysis),
            PresetSettings::Atlas(AtlasScene::SpaceGroup { .. }) => {
                self.select_defects_view(DefectsView::AtlasDefects);
                self.atlas.sub_tab = AtlasSubTab::SpaceGroups3D;
                self.atlas.last_defect_subtab = AtlasSubTab::SpaceGroups3D;
                self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d);
            }
            PresetSettings::Atlas(AtlasScene::Hyperbolic { .. }) => {
                self.select_lattices_view(LatticesView::NonEuclidean);
                self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d);
            }
        }
        if let PresetSettings::Defects { sub_tab, index, .. } = preset.settings {
            let d = &mut self.defects;
            match sub_tab {
                DefectSubTab::Dislocations => d.selected_disloc = index,
                DefectSubTab::Disclinations => d.selected_disclin = index,
                DefectSubTab::Walls => d.selected_wall = index,
                DefectSubTab::Combinations => {}
            }
            let it = self.gallery.active_item().clone();
            d.rebuild_crystal(&it, &self.backend, &mut self.canvas2d, true);
        }
        self.export_status = format!("Preset: {}", preset.name);
        self.pump_preset_pending(None);
        Ok(())
    }

    /// Open explicit defect specs (from Combinations) in `Place & fuse` (`Combine`).
    pub(crate) fn open_specs_in_combine(&mut self, specs: Vec<DefectSpec>) {
        self.session.pending_combine = None;
        let item = self.gallery.active_item().clone();
        let key = item_key(&item);
        let had_override = self.combine.override_item.is_some();
        if had_override {
            self.combine_seen = key.clone();
            self.combine.reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
        }
        let checkpoint = !had_override && key == self.combine_seen && !self.combine.is_initializing();
        self.preset_pending = Some(PendingPreset::CombineSpecs {
            item_key: key,
            specs,
            checkpoint,
        });
        self.select_defects_view(DefectsView::CombineAndFuse);
        self.pump_preset_pending(None);
    }

    /// Finish parked preset follow-ups once the active tab's background build/catalogue load completes (every frame).
    /// Dropped if the user switches tabs or changes the lattice meanwhile.
    pub(crate) fn pump_preset_pending(&mut self, ctx: Option<&egui::Context>) {
        let Some(pending) = &self.preset_pending else { return };
        let item = self.gallery.active_item().clone();
        match pending {
            PendingPreset::Combine { alias_key, .. } => {
                if self.active_tab != NavTab::Combine || item.alias_key != *alias_key || self.combine.override_item.is_some() {
                    self.preset_pending = None;
                    return;
                }
                if self.combine.is_initializing() {
                    return;
                }
                let Some(PendingPreset::Combine { name, pins, checkpoint, .. }) = self.preset_pending.take() else { return };
                if let Some(err) = &self.combine.init_error {
                    self.export_status = format!("Preset '{name}': lattice load failed ({err})");
                    return;
                }
                let specs = preset_specs(pins, &self.combine.info);
                if specs.is_empty() {
                    self.export_status = format!("Preset '{name}': these pins do not exist on this lattice");
                    return;
                }
                let c = &mut self.combine;
                c.clear_kinematic_drag_state(false);
                c.clear_selection();
                if c.specs != specs {
                    if checkpoint {
                        c.push_undo();
                    }
                    c.specs = specs;
                }
                // always rebuilt: the preset also reset the build options and colours, which the current build may predate
                let auto = c.auto_compare;
                c.request_build(&item, false, auto, true);
                if !checkpoint {
                    self.export_status = format!("Preset: {name} (Combine reloaded its lattice: earlier pins have no undo)");
                }
            }
            PendingPreset::CombineSpecs { item_key: want_key, .. } => {
                if self.active_tab != NavTab::Combine || item_key(&item) != *want_key || self.combine.override_item.is_some() {
                    self.preset_pending = None;
                    return;
                }
                if self.combine.is_initializing() {
                    return;
                }
                let Some(PendingPreset::CombineSpecs { specs, checkpoint, .. }) = self.preset_pending.take() else { return };
                if let Some(err) = &self.combine.init_error {
                    self.export_status = format!("Open in Place & fuse: lattice load failed ({err})");
                    return;
                }
                let c = &mut self.combine;
                c.dragging = false;
                c.clear_kinematic_drag_state(false);
                c.clear_selection();
                if c.specs != specs {
                    if checkpoint {
                        c.push_undo();
                    }
                    c.specs = specs;
                }
                c.request_build(&item, false, true, true);
            }
            PendingPreset::AnalysisSpectral { alias_key } => {
                if self.active_mode != AppMode::Analysis
                    || self.active_tab != NavTab::Analysis
                    || self.analysis.target != AnalysisTarget::SingleDefect
                    || item.alias_key != *alias_key
                {
                    self.preset_pending = None;
                    return;
                }
                if self.defects.is_busy() || self.defects.initializing {
                    return;
                }
                self.preset_pending = None;
                if self.defects.error.is_none()
                    && !self.defects.last_crystal.pos.is_empty()
                    && self.analysis.spectrum.result.is_none()
                    && !self.analysis.spectrum.is_busy()
                {
                    let (surf, def, comb) = (&self.surfaces, &self.defects, &self.combine);
                    self.analysis.submit_active_source(&item, surf, def, comb, ctx);
                }
            }
        }
    }
}

/// Reset 3D sheet options and non-spectral inspectors so a Defects or Combine preset never depends on earlier state.
fn reset_sheet_and_inspectors(
    insp: &mut DefectInspectors,
    sheet_opts: &mut Sheet3dViewOpts,
    colour: Sheet3dColorMode,
) {
    let safe_colour = if matches!(colour, Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge) {
        Sheet3dColorMode::FaceRingSize
    } else {
        colour
    };
    *sheet_opts = Sheet3dViewOpts { color_mode: safe_colour, ..Default::default() };
    insp.clear_for_switch();
    insp.circuit = Default::default();
    insp.overlay_mode = CrystalOverlayMode::Standard;
}
