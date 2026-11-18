//! Session capture / restore for [`PapertillingApp`] (see `crate::session` for the file format and what is not
//! persisted), the load-at-start and save-on-exit hooks, and the Top Bar "Session" menu.
//!
//! Restore order: Gallery tab, lattice (re-resolved from the catalogues by alias key / named key, never from a stored
//! recipe), slider values through `SliderState` (clamped and snapped), Gallery view state, per-tab settings, then the
//! stored tab through `switch_tab`, so only that tab runs its normal build (synchronous catalogue / lattice / surface
//! / Atlas prep, followed by background builds on Defects and Combine). The camera comes last because tab switches
//! may set meshes. Out-of-range numeric values (sliders, radius, layout, h/k, genus, camera) clamp silently and
//! `null` parameters are skipped; unknown variants, missing lattices, out-of-range defect indices, unsupported PNG
//! scale, Combine mismatches and spectral colour resets add status-bar notes.

use std::path::{Path, PathBuf};

use papertilling_core::defect::combine_types::{CONFIG_FORMAT, NotebookConfigJson};
use papertilling_core::defect::surgery::DefectSpec;

use super::{NavTab, PapertillingApp, item_key};
use crate::atlas_tab::AtlasSubTab;
use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi, CentreModeUi, SurfaceTargetUi};
use crate::canvas2d::gauge_overlay::CrystalOverlayMode;
use crate::defects_tab::DefectSubTab;
use crate::gallery_tab::browser::BrowserView;
use crate::gallery_tab::filter::{ChiralFacet, SystemFacet};
use crate::live_ops_ao::Defect2dMode;
use crate::mesh3d::sheet_modes::{DefectViewMode, Sheet3dColorMode};
use crate::scene2d::overlays::GalleryOverlayToggles;
use crate::session::{self, LatticeSession, SessionError, SessionState, name_of, parse_named};
use crate::spectrum_worker::SpectrumColorOverlay;
use crate::theme::StudioTheme;
use crate::ui::lattice_switcher::RECENT_MAX;

const THEMES: [StudioTheme; 2] = [StudioTheme::Paper, StudioTheme::Studio];
const BROWSER_VIEWS: [BrowserView; 2] = [BrowserView::Cards, BrowserView::Table];

/// Where the session lives for this run: `path` is `None` in `--smoke-test` mode and in tests (nothing is read or
/// written, as for the in-memory alias store).
#[derive(Debug)]
pub(crate) struct SessionHost {
    pub(crate) path: Option<PathBuf>,
    /// False after the file at `path` failed to load with a reason that forbids replacing it.
    pub(crate) exit_save_allowed: bool,
    /// Combine configuration restored for the current lattice, applied when the Combine tab is next opened.
    pub(crate) pending_combine: Option<NotebookConfigJson>,
    /// Non-empty when a deferred Combine restore failed on entering the Combine tab during `apply_session`.
    pub(crate) combine_note: Option<String>,
}

impl SessionHost {
    pub(crate) fn for_mode(smoke_test: bool) -> Self {
        let path = if smoke_test || cfg!(test) { None } else { session::default_session_path() };
        Self { path, exit_save_allowed: true, pending_combine: None, combine_note: None }
    }
}

/// The stored catalogue / named lattice `key` (with `named_key` as a fallback for a named lattice), or `None`.
pub(crate) fn resolve_lattice(backend: &AppBackend, key: &crate::aliases::LatticeAliasKey, named_key: Option<&str>) -> Option<CatalogueItemUi> {
    if key.rule == "named" {
        let by_key = backend.named_lattices().iter().find(|n| n.alias_key == *key);
        let by_name = || named_key.and_then(|k| backend.find_named(k)).filter(|n| n.group == key.group);
        return by_key.or_else(by_name).cloned();
    }
    let rule = CatalogueRuleUi::ALL.into_iter().find(|r| r.slug() == key.rule)?;
    if !backend.groups().iter().any(|g| g.symbol == key.group) {
        return None;
    }
    backend.catalogue_for_group(&key.group, rule).into_iter().find(|it| it.alias_key == *key)
}

fn finite32(v: Option<f32>) -> Option<f32> {
    v.filter(|x| x.is_finite())
}

/// `true` when `cfg` describes `item` (same group, frame and recipe): the only Combine set-up that is restored.
fn combine_cfg_matches(cfg: &NotebookConfigJson, item: &CatalogueItemUi) -> bool {
    cfg.group == item.group && cfg.frame == item.frame && cfg.recipe == item.recipe
}

fn defect_spec_finite(spec: &DefectSpec) -> bool {
    match spec {
        DefectSpec::Dislocation { b, at, cut } => {
            b.iter().all(|x| x.is_finite())
                && at.is_none_or(|p| p.iter().all(|x| x.is_finite()))
                && cut.is_none_or(|c| c.iter().all(|x| x.is_finite()))
        }
        DefectSpec::Disclination { angle_deg, at, cut } => {
            angle_deg.is_finite()
                && at.iter().all(|x| x.is_finite())
                && cut.is_none_or(|c| c.iter().all(|x| x.is_finite()))
        }
        DefectSpec::HalfWall { at, direction, sigma, .. } => {
            at.iter().all(|x| x.is_finite())
                && direction.is_none_or(|d| d.iter().all(|x| x.is_finite()))
                && sigma.is_none_or(|(m, t)| m.iter().flatten().chain(&t).all(|x| x.is_finite()))
        }
    }
}

fn combine_cfg_finite(cfg: &NotebookConfigJson) -> bool {
    cfg.options.r_patch.is_none_or(f64::is_finite)
        && cfg.defects.iter().all(defect_spec_finite)
        && cfg.recipe.a.is_finite()
        && cfg.recipe.b.is_none_or(f64::is_finite)
        && cfg.recipe.gamma.is_none_or(f64::is_finite)
        && cfg.recipe.sites.iter().flatten().all(|x| x.is_finite())
        && cfg.recipe.bonds.iter().flatten().flatten().all(|x| x.is_finite())
        && cfg.frame.q.iter().flatten().chain(&cfg.frame.origin).all(|x| x.is_finite())
}

impl PapertillingApp {
    /// Everything that is persisted, as it is now.
    pub(crate) fn capture_session(&self) -> SessionState {
        let base = self.gallery.base_item();
        let active = self.gallery.active_item();
        let g = &self.gallery;
        let d = &self.defects;
        let s = &self.surfaces;
        let combine = self
            .session
            .pending_combine
            .as_ref()
            .filter(|cfg| combine_cfg_matches(cfg, active))
            .cloned()
            .or_else(|| {
                (item_key(active) == self.combine_seen
                    && !self.combine.specs.is_empty()
                    && !self.combine.is_initializing())
                .then(|| {
                    let mut cfg = self.combine.to_notebook_json(active);
                    cfg.summary = None;
                    cfg
                })
            });
        SessionState {
            schema_version: session::SESSION_SCHEMA_VERSION,
            active_tab: Some(name_of(self.active_tab)),
            theme: Some(name_of(self.theme)),
            layout: session::LayoutSession {
                left_rail_width: Some(self.left_rail_width),
                right_inspector_open: Some(self.right_inspector_open),
                right_inspector_width: Some(self.right_inspector_width),
                split_ratio: Some(self.viewport_split_ratio),
            },
            png_scale: Some(self.png_scale),
            lattice: Some(LatticeSession { alias_key: base.alias_key.clone(), group: base.group.clone(), named_key: base.named_key.clone() }),
            gallery: session::GallerySession {
                params: g.param_values().into_iter().map(|(n, v)| (n, Some(v))).collect(),
                repeat: Some(g.repeat),
                show_unit_cell: Some(g.show_unit_cell),
                overlay_symmetry: Some(g.overlays.symmetry),
                overlay_voronoi: Some(g.overlays.voronoi),
                browser_view: Some(name_of(g.view)),
                filter_query: Some(g.filter.query.clone()),
                filter_system: g.filter.system.map(name_of),
                filter_chirality: Some(name_of(g.filter.chirality)),
            },
            recent: self.switcher.recent.iter().map(|r| r.alias_key.clone()).collect(),
            surfaces: session::SurfacesSession {
                target: Some(name_of(s.target)),
                h: Some(s.params.h),
                k: Some(s.params.k),
                genus_g: Some(s.params.genus_g),
                centre_mode: Some(name_of(s.params.centre_mode)),
                auto_size: Some(s.auto_size),
                show_cover: Some(s.show_cover),
            },
            defects: session::DefectsSession {
                kind: Some(name_of(d.sub_tab)),
                dislocation_index: Some(d.selected_disloc),
                disclination_index: Some(d.selected_disclin),
                wall_index: Some(d.selected_wall),
                wall_half: Some(d.wall_half),
                radius: Some(d.radius),
                relaxed: Some(d.draw_mode != Defect2dMode::Raw),
                draw_mode: Some(name_of(d.draw_mode)),
                view_mode: Some(name_of(d.view_mode)),
                color_mode: Some(name_of(d.sheet_opts.color_mode)),
            },
            combine,
            camera: session::CameraSession {
                yaw: Some(self.view3d.yaw),
                pitch: Some(self.view3d.pitch),
                distance: Some(self.view3d.distance),
                offset: Some([self.view3d.camera_offset.x, self.view3d.camera_offset.y]),
                z_up: Some(self.view3d.z_up),
            },
            atlas_sub_view: Some(name_of(self.atlas.sub_tab)),
        }
    }

    /// Apply `st` (see the module documentation for the order); returns the notes on values that fell back.
    pub(crate) fn apply_session(&mut self, st: &SessionState, ctx: Option<&egui::Context>) -> Vec<String> {
        let mut notes = Vec::new();
        self.preset_pending = None;
        self.session.combine_note = None;
        self.switch_tab(NavTab::Gallery);
        match parse_named(&THEMES, st.theme.as_deref()) {
            Some(t) => {
                self.theme = t;
                if let Some(ctx) = ctx {
                    t.apply(ctx);
                }
            }
            None if st.theme.is_some() => notes.push(String::from("unknown theme (kept current)")),
            None => {}
        }
        self.apply_layout(st, &mut notes);
        self.apply_recent(&st.recent, &mut notes);
        self.apply_lattice(st, &mut notes);
        self.apply_gallery_view(&st.gallery, &mut notes);
        self.apply_surfaces(&st.surfaces, &mut notes);
        self.apply_defects(&st.defects, &mut notes);
        self.session.pending_combine = None;
        if let Some(cfg) = &st.combine {
            if cfg.format != CONFIG_FORMAT {
                notes.push(String::from("Combine set-up has unsupported format (not restored)"));
            } else if !combine_cfg_finite(cfg) {
                notes.push(String::from("Combine set-up had non-finite values (not restored)"));
            } else if combine_cfg_matches(cfg, self.gallery.active_item()) {
                self.session.pending_combine = Some(cfg.clone());
            } else {
                notes.push(String::from("Combine set-up was for another lattice (not restored)"));
            }
        }
        match parse_named(&AtlasSubTab::ALL, st.atlas_sub_view.as_deref()) {
            Some(sub) => {
                self.atlas.sub_tab = sub;
                self.atlas.dirty = true;
            }
            None if st.atlas_sub_view.is_some() => notes.push(String::from("unknown Atlas sub-view (kept current)")),
            None => {}
        }
        match parse_named(&NavTab::ALL, st.active_tab.as_deref()) {
            Some(tab) => self.switch_tab(tab),
            None if st.active_tab.is_some() => notes.push(String::from("unknown tab (Gallery shown)")),
            None => {}
        }
        if let Some(msg) = self.session.combine_note.take() {
            notes.push(msg);
        }
        let c = &st.camera;
        let offset = c.offset.filter(|o| o.iter().all(|v| v.is_finite()));
        self.view3d.restore_camera(finite32(c.yaw), finite32(c.pitch), finite32(c.distance), offset, c.z_up);
        notes
    }

    fn apply_layout(&mut self, st: &SessionState, notes: &mut Vec<String>) {
        let l = &st.layout;
        if let Some(w) = finite32(l.left_rail_width) {
            self.left_rail_width = w.clamp(260.0, 420.0);
        }
        if let Some(open) = l.right_inspector_open {
            self.right_inspector_open = open;
        }
        if let Some(w) = finite32(l.right_inspector_width) {
            self.right_inspector_width = w.clamp(260.0, 440.0);
        }
        if let Some(r) = finite32(l.split_ratio) {
            self.viewport_split_ratio = r.clamp(0.2, 0.8);
        }
        match st.png_scale {
            Some(s @ (1 | 2 | 4)) => self.png_scale = s,
            Some(s) => notes.push(format!("PNG scale {s} unsupported (kept {}x)", self.png_scale)),
            None => {}
        }
    }

    fn apply_recent(&mut self, keys: &[crate::aliases::LatticeAliasKey], notes: &mut Vec<String>) {
        let items: Vec<CatalogueItemUi> = keys.iter().filter_map(|k| resolve_lattice(&self.backend, k, None)).collect();
        if items.len() < keys.len() {
            notes.push(format!("{} recent lattice(s) no longer found", keys.len() - items.len()));
        }
        self.switcher.recent.clear();
        for it in items.iter().take(RECENT_MAX).rev() {
            self.switcher.push_recent(it);
        }
    }

    fn apply_lattice(&mut self, st: &SessionState, notes: &mut Vec<String>) {
        let Some(l) = &st.lattice else { return };
        let item = resolve_lattice(&self.backend, &l.alias_key, l.named_key.as_deref()).filter(|it| l.group.is_empty() || it.group == l.group);
        let Some(item) = item else {
            let what = if l.alias_key.rule == "custom" { "custom-built lattice cannot be restored" } else { "lattice not found" };
            notes.push(format!("{what} ({} {} {}): default lattice kept", l.alias_key.group, l.alias_key.rule, l.alias_key.dsymbol_key_or_index));
            return;
        };
        let same_base = item.alias_key == self.gallery.base_item().alias_key;
        self.select_lattice_and_sync(item);
        if same_base && self.gallery.sliders().iter().any(|s| s.value != s.default_value) {
            // a lattice that was already selected keeps its values: start from its defaults, as the file describes
            self.gallery.reset_params(&mut self.canvas2d);
        }
        let sliders: Vec<(String, f64)> = self.gallery.sliders().iter().map(|s| (s.name.clone(), s.value)).collect();
        for (name, cur) in sliders {
            if let Some(Some(v)) = st.gallery.params.get(&name)
                && (!v.is_finite() || (*v != cur && self.gallery.set_param(&name, *v, &mut self.canvas2d).is_none()))
            {
                notes.push(format!("parameter {name} = {v} not applied"));
            }
        }
        let unknown = st.gallery.params.keys().filter(|n| !self.gallery.sliders().iter().any(|s| &s.name == *n)).count();
        if unknown > 0 {
            notes.push(format!("{unknown} unknown parameter(s) ignored"));
        }
    }

    fn apply_gallery_view(&mut self, g: &session::GallerySession, notes: &mut Vec<String>) {
        if let Some(r) = g.repeat {
            self.gallery.set_repeat(r, &mut self.canvas2d);
        }
        if let Some(show) = g.show_unit_cell {
            self.gallery.set_show_unit_cell(show, &mut self.canvas2d);
        }
        let ov = GalleryOverlayToggles {
            symmetry: g.overlay_symmetry.unwrap_or(self.gallery.overlays.symmetry),
            voronoi: g.overlay_voronoi.unwrap_or(self.gallery.overlays.voronoi),
        };
        if ov != self.gallery.overlays {
            self.gallery.set_overlays(ov, &mut self.canvas2d);
        }
        match parse_named(&BROWSER_VIEWS, g.browser_view.as_deref()) {
            Some(v) => self.gallery.view = v,
            None if g.browser_view.is_some() => notes.push(String::from("unknown browser view (kept current)")),
            None => {}
        }
        if let Some(q) = &g.filter_query {
            self.gallery.filter.query = q.chars().take(200).collect();
        }
        self.gallery.filter.system = parse_named(&SystemFacet::ALL, g.filter_system.as_deref());
        if g.filter_system.is_some() && self.gallery.filter.system.is_none() {
            notes.push(String::from("unknown crystal-system filter (cleared)"));
        }
        match parse_named(&ChiralFacet::ALL, g.filter_chirality.as_deref()) {
            Some(c) => self.gallery.filter.chirality = c,
            None if g.filter_chirality.is_some() => notes.push(String::from("unknown chirality filter (kept current)")),
            None => {}
        }
    }

    fn apply_surfaces(&mut self, s: &session::SurfacesSession, notes: &mut Vec<String>) {
        let st = &mut self.surfaces;
        match parse_named(&SurfaceTargetUi::ALL, s.target.as_deref()) {
            Some(t) => st.target = t,
            None if s.target.is_some() => notes.push(String::from("unknown surface target (kept current)")),
            None => {}
        }
        if let Some(h) = s.h {
            st.params.h = h;
        }
        if let Some(k) = s.k {
            st.params.k = k;
        }
        if let Some(g) = s.genus_g {
            st.params.genus_g = g.clamp(2, 3);
        }
        match parse_named(&CentreModeUi::ALL, s.centre_mode.as_deref()) {
            Some(c) => st.params.centre_mode = c,
            None if s.centre_mode.is_some() => notes.push(String::from("unknown surface centre mode (kept current)")),
            None => {}
        }
        if let Some(a) = s.auto_size {
            st.auto_size = a;
        }
        if let Some(c) = s.show_cover {
            st.show_cover = c;
        }
        // (h, k) into the ranges the controls offer for this target and lattice
        st.clamp_params(self.gallery.active_item());
    }

    fn apply_defects(&mut self, s: &session::DefectsSession, notes: &mut Vec<String>) {
        let item = self.gallery.active_item().clone();
        if item_key(&item) != self.defects_seen {
            // as `switch_tab` would on opening Defects, but now, so the restored selection is not reset there
            self.defects_seen = item_key(&item);
            self.defects.reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
        }
        let d = &mut self.defects;
        match parse_named(&DefectSubTab::ALL, s.kind.as_deref()) {
            Some(k) => d.sub_tab = k,
            None if s.kind.is_some() => notes.push(String::from("unknown defect kind (kept current)")),
            None => {}
        }
        let wanted = [s.dislocation_index, s.disclination_index, s.wall_index];
        let cat_empty = d.catalogue.dislocations.is_empty() && d.catalogue.disclinations.is_empty() && d.catalogue.walls.is_empty();
        if cat_empty && wanted.iter().any(|i| i.is_some_and(|i| i > 0)) {
            // only the lengths are needed: the build worker computes the same catalogue again
            d.catalogue = self.backend.defect_catalogue(&item);
        }
        let lens = [d.catalogue.dislocations.len(), d.catalogue.disclinations.len(), d.catalogue.walls.len()];
        let slots = [&mut d.selected_disloc, &mut d.selected_disclin, &mut d.selected_wall];
        for ((want, len), slot) in wanted.into_iter().zip(lens).zip(slots) {
            match want {
                Some(i) if i == 0 || i < len => *slot = i,
                Some(i) => {
                    *slot = 0;
                    notes.push(format!("defect #{i} out of range (first one selected)"));
                }
                None => {}
            }
        }
        if let Some(h) = s.wall_half {
            d.wall_half = h;
        }
        if let Some(r) = s.radius.filter(|r| r.is_finite()) {
            d.radius = r.clamp(2.4, 8.0);
        }
        match parse_named(&Defect2dMode::ALL, s.draw_mode.as_deref()) {
            Some(m) => d.draw_mode = m,
            None if s.draw_mode.is_some() => notes.push(String::from("unknown 2D drawing mode (kept current)")),
            None => {
                if let Some(relaxed) = s.relaxed {
                    d.draw_mode = if relaxed { Defect2dMode::RelaxedDrawing } else { Defect2dMode::Raw };
                }
            }
        }
        d.relaxed = d.draw_mode != Defect2dMode::Raw;
        match parse_named(&DefectViewMode::ALL, s.view_mode.as_deref()) {
            Some(v) => d.view_mode = v,
            None if s.view_mode.is_some() => notes.push(String::from("unknown 3D view mode (kept current)")),
            None => {}
        }
        match parse_named(&Sheet3dColorMode::ALL, s.color_mode.as_deref()) {
            Some(c) => {
                let spectral = matches!(c, Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge);
                d.sheet_opts.color_mode = if spectral { Sheet3dColorMode::FaceRingSize } else { c };
                d.inspectors.spectrum.color_overlay = SpectrumColorOverlay::None;
                if spectral
                    || matches!(
                        d.inspectors.overlay_mode,
                        CrystalOverlayMode::SpectralLdos | CrystalOverlayMode::SpectralCharge
                    )
                {
                    d.inspectors.overlay_mode = CrystalOverlayMode::Standard;
                }
                if spectral {
                    notes.push(String::from("3D spectral colouring reset to ring size (recompute in Spectrum panel)"));
                }
            }
            None if s.color_mode.is_some() => notes.push(String::from("unknown 3D colour mode (kept current)")),
            None => {}
        }
        d.view3d_dirty = true;
    }

    /// Called when the Combine tab opens: load the restored Combine set-up if it still fits the current lattice.
    /// Returns `true` when Combine was restored and synced (so callers skip the default reload).
    pub(crate) fn apply_pending_combine(&mut self) -> bool {
        let Some(cfg) = self.session.pending_combine.take() else { return false };
        let item = self.gallery.active_item().clone();
        if !combine_cfg_matches(&cfg, &item) {
            return false; // the lattice changed since the restore: the set-up no longer applies
        }
        match self.combine.load_notebook_json(&cfg, &self.backend, &mut self.canvas2d) {
            Ok(_) => {
                self.combine.override_item = None;
                self.combine.json_status.clear();
                self.combine_seen = item_key(&item);
                self.combine.sync_to_canvas(&item, &mut self.canvas2d, true);
                self.combine.sync_view3d(&item, &mut self.view3d);
                true
            }
            Err(e) => {
                let msg = format!("Combine set-up not restored ({e})");
                self.export_status = format!("Session: {msg}");
                self.session.combine_note = Some(msg);
                false
            }
        }
    }

    /// Apply a session read from `path` and report it in the status bar (`Ok(None)`: no file there).
    pub(crate) fn open_session_from(&mut self, path: &Path, ctx: Option<&egui::Context>) -> Result<Option<Vec<String>>, SessionError> {
        let Some((st, dropped)) = session::load_session_file(path)? else { return Ok(None) };
        let mut notes = self.apply_session(&st, ctx);
        if dropped > 0 {
            notes.insert(0, format!("{dropped} invalid field(s) reset to defaults"));
        }
        self.export_status = if notes.is_empty() {
            String::from("Session restored")
        } else {
            format!("Session restored ({})", notes.join("; "))
        };
        Ok(Some(notes))
    }

    /// Restore the OS-config session at start-up (no-op in smoke-test mode and tests).
    pub fn load_session_at_start(&mut self, ctx: &egui::Context) {
        let Some(path) = self.session.path.clone() else { return };
        if let Err(e) = self.open_session_from(&path, Some(ctx)) {
            self.session.exit_save_allowed = e.may_overwrite();
            let kept = if e.may_overwrite() { "" } else { "; file left untouched" };
            self.export_status = format!("Session not restored: {e}; started with defaults{kept}");
        }
    }

    pub(crate) fn save_session_to(&self, path: &Path, lock: bool) -> Result<(), String> {
        if lock {
            session::save_session_file(path, &self.capture_session())
        } else {
            session::save_session_file_unlocked(path, &self.capture_session())
        }
    }

    /// Save the OS-config session (on exit). Returns whether a file was written.
    pub(crate) fn save_session_on_exit(&mut self) -> bool {
        let Some(path) = self.session.path.clone().filter(|_| self.session.exit_save_allowed) else { return false };
        self.finish_gallery_moves();
        let dir = crate::aliases::parent_dir(&path);
        if std::fs::create_dir_all(dir).is_err() {
            return false;
        }
        self.save_session_to(&path, true).is_ok()
    }

    /// The Top Bar "Session" menu: save / open a session file anywhere (dialogs are skipped in smoke-test mode).
    pub(crate) fn session_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Session", |ui| {
            if ui.button("Save Session As...").on_hover_text("Save tab, lattice, parameters, views and camera (.json)").clicked() && !self.smoke_test {
                ui.close();
                self.finish_gallery_moves();
                let dlg = rfd::FileDialog::new().set_title("Save session").set_file_name(session::SESSION_FILENAME).add_filter("Session", &["json"]);
                if let Some(path) = dlg.save_file() {
                    let lock = self.session.path.as_deref() == Some(path.as_path());
                    self.export_status = match self.save_session_to(&path, lock) {
                        Ok(()) => format!("Session saved to {}", path.display()),
                        Err(e) => format!("Session save error: {e}"),
                    };
                }
            }
            if ui.button("Open Session...").on_hover_text("Restore a saved session (.json)").clicked() && !self.smoke_test {
                ui.close();
                let dlg = rfd::FileDialog::new().set_title("Open session").add_filter("Session", &["json"]);
                if let Some(path) = dlg.pick_file() {
                    let ctx = ui.ctx().clone();
                    match self.open_session_from(&path, Some(&ctx)) {
                        Ok(Some(_)) => {}
                        Ok(None) => self.export_status = String::from("Session file not found"),
                        Err(e) => self.export_status = format!("Session not opened: {e}"),
                    }
                }
            }
        });
    }
}
