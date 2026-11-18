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

use super::{NavTab, PapertillingApp, item_key};
use crate::aliases::LatticeAliasKey;
use crate::backend::{CatalogueItemUi, CentreModeUi};
use crate::canvas2d::gauge_overlay::CrystalOverlayMode;
use crate::defects_tab::DefectSubTab;
use crate::live_combine::{CombineLatticeInfo, preset_neighbour_pair};
use crate::defects_inspectors::DefectInspectors;
use crate::mesh3d::{Sheet3dColorMode, Sheet3dViewOpts, SurfaceColor3d};
use crate::presets::{AtlasScene, CombinePins, PRESETS, PresetLattice, PresetSettings, SurfaceSize};
use crate::spectrum_worker::SpectrumColorOverlay;
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
    DefectsSpectral {
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
                self.gallery.set_overlays(GalleryOverlayToggles { symmetry, voronoi }, &mut self.canvas2d);
            }
            PresetSettings::Surfaces { target, size, shape } => {
                let s = &mut self.surfaces;
                s.target = target;
                s.show_cover = false;
                s.elastic.shape_mode = shape;
                s.elastic.kappa_b = 0.02;
                // the auto build's corner classes: Auto (face centre before site), as every preset count assumes
                s.params.centre_mode = CentreModeUi::Auto;
                // plain ring-size colours: no spectrum panel / overlay left over from the previous surface
                s.spectrum.clear_for_switch();
                s.spectrum.expanded = false;
                s.spectrum.color_overlay = SpectrumColorOverlay::None;
                s.spectrum.opts = Default::default();
                s.elastic.color_mode = SurfaceColor3d::FaceRingSize;
                match size {
                    SurfaceSize::Auto => self.surfaces.auto_size = true,
                    SurfaceSize::Manual { h, k } => {
                        self.surfaces.auto_size = false;
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
                c.json_status.clear();
                c.view3d_dirty = true;
                self.preset_pending = Some(PendingPreset::Combine {
                    name: preset.name,
                    alias_key: key,
                    pins,
                    checkpoint,
                });
            }
            PresetSettings::Atlas(scene) => match scene {
                AtlasScene::SpaceGroup { number, net, defect } => {
                    self.atlas.sub_tab = crate::atlas_tab::AtlasSubTab::SpaceGroups3D;
                    let s = &mut self.atlas.space3d;
                    (s.selected_sg, s.selected_net, s.selected_defect) = (number, net, defect);
                    s.filter.clear();
                    s.view = Default::default();
                    s.ensure_cached();
                }
                AtlasScene::Hyperbolic { p, q, style, layers } => {
                    self.atlas.sub_tab = crate::atlas_tab::AtlasSubTab::HyperbolicH2;
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
        self.switch_tab(preset.tab());
        if let PresetSettings::Defects { sub_tab, index, colour, .. } = preset.settings {
            let d = &mut self.defects;
            match sub_tab {
                DefectSubTab::Dislocations => d.selected_disloc = index,
                DefectSubTab::Disclinations => d.selected_disclin = index,
                DefectSubTab::Walls => d.selected_wall = index,
            }
            let it = self.gallery.active_item().clone();
            d.rebuild_crystal(&it, &self.backend, &mut self.canvas2d, true);
            if matches!(colour, Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge) {
                self.preset_pending = Some(PendingPreset::DefectsSpectral { alias_key: it.alias_key });
            }
        }
        self.export_status = format!("Preset: {}", preset.name);
        self.pump_preset_pending(None);
        Ok(())
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
            PendingPreset::DefectsSpectral { alias_key } => {
                if self.active_tab != NavTab::Defects || item.alias_key != *alias_key {
                    self.preset_pending = None;
                    return;
                }
                if self.defects.is_busy() || self.defects.initializing {
                    return;
                }
                self.preset_pending = None;
                let d = &mut self.defects;
                if d.error.is_none()
                    && !d.last_crystal.pos.is_empty()
                    && d.inspectors.spectrum.result.is_none()
                    && !d.inspectors.spectrum.is_busy()
                {
                    d.inspectors.spectrum.submit_crystal(&d.last_crystal, Some(&d.last_sheet), ctx);
                }
            }
        }
    }
}

/// Reset 3D sheet options and inspectors (Burgers circuit, 2D overlay, tight-binding spectrum) so a Defects or
/// Combine preset never depends on earlier state. LDOS / charge configures the spectral color mode and 2D overlay
/// (while `CollapsingHeader::default_open(false)` in `spectrum_ui.rs` keeps the visual fold state); any other
/// colouring clears both.
fn reset_sheet_and_inspectors(
    insp: &mut DefectInspectors,
    sheet_opts: &mut Sheet3dViewOpts,
    colour: Sheet3dColorMode,
) {
    *sheet_opts = Sheet3dViewOpts { color_mode: colour, ..Default::default() };
    insp.clear_for_switch();
    insp.circuit = Default::default();
    insp.spectrum.opts = Default::default();
    let spectral = matches!(colour, Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge);
    if spectral {
        insp.on_sheet_color_changed(colour);
    } else {
        insp.spectrum.color_overlay = SpectrumColorOverlay::None;
        insp.overlay_mode = CrystalOverlayMode::Standard;
    }
    insp.spectrum.expanded = spectral;
}
