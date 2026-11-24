//! Side-by-side Defect Fusion & Single-Defect Equivalence Explorer (`Defects > Fuse & compare`).
//!
//! Lets the user pick Defect A and Defect B from the active lattice's catalogue, step or drag B
//! across allowed lattice positions relative to A, and inspect `[Combined crystal (A + B)]` next to
//! `[Equivalent single defect C]` (Volterra and healed/completion cores) with synced zoom, pan, and
//! enclosing Burgers / Frank holonomy loops.

mod choices;
mod panel_left;
mod panel_right;
mod viewport;
#[cfg(test)]
mod tests;

pub(crate) use choices::{choice_label, choice_short_label};
use choices::{compute_slots_for_b, enumerate_choices, run_fuse_job, sort_slots, spec_from_choice};
use papertilling_core::defect::single_cache::SingleDefectCache;
use papertilling_core::defect::single_crystal::{SingleEquivalenceBundle, SingleEquivalentCrystal};
use papertilling_core::defect::surgery::DefectSpec;

use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasMarker, MarkerKind, Scene2D};
use crate::combine_worker::{LatestWorker, empty_crystal};
use crate::defects_inspectors::DefectInspectors;
use crate::live_combine::{
    CombineBuildOutput, CombineLatticeInfo, combine_lattice_info, map_ref_to_drawn, spec_short_label,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FuseDefectChoice {
    Disclination {
        class_idx: usize,
        angle_deg: i32,
    },
    Dislocation {
        disloc_idx: usize,
        member_idx: usize,
        sign: i8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusePreset {
    DipoleToDislocation,
    OppositeAnnihilate,
    DisclinationPlusDislocation,
    DislocationSum,
}

impl FusePreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::DipoleToDislocation => "+α + −α → Dislocation",
            Self::OppositeAnnihilate => "+α + −α → Vacuum",
            Self::DisclinationPlusDislocation => "Disclination + Dislocation",
            Self::DislocationSum => "Dislocation + Dislocation",
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::DipoleToDislocation => "Compare an adjacent pair of opposite disclinations with single dislocation cores",
            Self::OppositeAnnihilate => "Opposite disclinations on the same Wyckoff class annihilate when brought together",
            Self::DisclinationPlusDislocation => "Explore how a dislocation changes the combined rotation centre",
            Self::DislocationSum => "Two dislocations combine by adding their Burgers vectors",
        }
    }
}

pub(super) struct FuseJob {
    pub item: CatalogueItemUi,
    pub info: CombineLatticeInfo,
    pub specs: [DefectSpec; 2],
    pub relaxed: bool,
    pub auto_radius: bool,
    pub manual_radius: f64,
}

pub(crate) struct FuseBuildResult {
    pub combined: CombineBuildOutput,
    pub bundle: Result<SingleEquivalenceBundle, String>,
}

pub struct FuseTabState {
    pub(crate) info: CombineLatticeInfo,
    pub(crate) choices: Vec<FuseDefectChoice>,
    pub(crate) choice_a: usize,
    pub(crate) choice_b: usize,
    pub(crate) pos_a: [f64; 2],
    pub(crate) pos_b: [f64; 2],
    pub(crate) slots_b: Vec<[f64; 2]>,
    pub(crate) selected_candidate: usize,
    pub sync_views: bool,
    pub show_burgers_loop: bool,
    pub show_target_sites: bool,
    pub relaxed: bool,
    pub auto_radius: bool,
    pub manual_radius: f64,
    pub(crate) dragging_pin: Option<usize>,
    pub(crate) single_canvas: Canvas2D,
    pub(crate) combined_inspectors: DefectInspectors,
    pub(crate) single_inspectors: DefectInspectors,
    pub(crate) last_combined: CombineBuildOutput,
    pub(crate) last_bundle: Option<Result<SingleEquivalenceBundle, String>>,
    pub(crate) open_in_combine: Option<Vec<DefectSpec>>,
    pub(crate) needs_fit_both: bool,
    worker: LatestWorker<FuseJob, FuseBuildResult>,
}

fn empty_combined(item: &CatalogueItemUi, info: &CombineLatticeInfo) -> CombineBuildOutput {
    CombineBuildOutput {
        lattice_key: (item.group.clone(), info.cell.clone()),
        crystal: empty_crystal(),
        elapsed_ms: 0.0,
        applied_radius: info.r_auto_base,
        charge_label: String::new(),
        charge_detail: String::new(),
        core_summary: String::new(),
        local_cores: Vec::new(),
        patch_centre: [0.0, 0.0],
        applied_cuts: Vec::new(),
        coalesced_count: 0,
        note: String::new(),
        preview: false,
        refused: None,
    }
}

impl FuseTabState {
    pub fn new(item: &CatalogueItemUi, _backend: &AppBackend) -> Self {
        let info = combine_lattice_info(item);
        let choices = enumerate_choices(&info);
        let placeholder = Scene2D::CrystalPatch {
            crystal: empty_crystal(),
            title: String::from("Computing equivalence…"),
            note: String::new(),
        };
        let mut combined_inspectors = DefectInspectors::new();
        let mut single_inspectors = DefectInspectors::new();
        combined_inspectors.circuit.enabled = true;
        single_inspectors.circuit.enabled = true;

        let mut state = Self {
            manual_radius: info.r_auto_base.max(5.5),
            last_combined: empty_combined(item, &info),
            info,
            choices,
            choice_a: 0,
            choice_b: 0,
            pos_a: [0.0, 0.0],
            pos_b: [1.0, 0.0],
            slots_b: Vec::new(),
            selected_candidate: 0,
            sync_views: true,
            show_burgers_loop: true,
            show_target_sites: true,
            relaxed: true,
            auto_radius: true,
            dragging_pin: None,
            single_canvas: Canvas2D::new(placeholder),
            combined_inspectors,
            single_inspectors,
            last_bundle: None,
            open_in_combine: None,
            needs_fit_both: true,
            worker: LatestWorker::spawn("fuse-compare", SingleDefectCache::default(), run_fuse_job),
        };
        state.select_default_preset_quiet();
        state
    }

    pub fn is_busy(&self) -> bool {
        self.worker.is_busy()
    }

    pub(crate) fn current_specs(&self) -> [DefectSpec; 2] {
        if self.choices.is_empty() {
            return [DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0])), DefectSpec::dislocation([-1.0, 0.0], Some([1.0, 0.0]))];
        }
        let ca = &self.choices[self.choice_a.min(self.choices.len() - 1)];
        let cb = &self.choices[self.choice_b.min(self.choices.len() - 1)];
        [spec_from_choice(&self.info, ca, self.pos_a), spec_from_choice(&self.info, cb, self.pos_b)]
    }

    pub(crate) fn take_open_in_combine(&mut self) -> Option<Vec<DefectSpec>> {
        self.open_in_combine.take()
    }

    pub fn reload_for_item(&mut self, item: &CatalogueItemUi, left_canvas: &mut Canvas2D) {
        self.worker.cancel();
        self.info = combine_lattice_info(item);
        self.choices = enumerate_choices(&self.info);
        self.manual_radius = self.info.r_auto_base.max(5.5);
        self.selected_candidate = 0;
        self.dragging_pin = None;
        self.last_combined = empty_combined(item, &self.info);
        self.combined_inspectors.clear_for_switch();
        self.single_inspectors.clear_for_switch();
        self.select_default_preset_quiet();
        self.submit_build(item, true);
        self.sync_canvases(item, left_canvas, true);
    }

    pub fn activate_and_sync(&mut self, item: &CatalogueItemUi, left_canvas: &mut Canvas2D) {
        if self.last_bundle.is_none() && !self.is_busy() {
            self.submit_build(item, true);
        }
        self.sync_canvases(item, left_canvas, false);
    }

    pub(crate) fn available_presets(&self) -> Vec<FusePreset> {
        let has_discl = self.choices.iter().any(|c| matches!(c, FuseDefectChoice::Disclination { .. }));
        let has_disloc = self.choices.iter().any(|c| matches!(c, FuseDefectChoice::Dislocation { .. }));
        let mut out = Vec::new();
        if has_discl {
            out.push(FusePreset::DipoleToDislocation);
            out.push(FusePreset::OppositeAnnihilate);
        }
        if has_discl && has_disloc {
            out.push(FusePreset::DisclinationPlusDislocation);
        }
        if has_disloc {
            out.push(FusePreset::DislocationSum);
        }
        out
    }

    pub(crate) fn apply_preset(&mut self, preset: FusePreset, item: &CatalogueItemUi, fit: bool) {
        self.set_preset_choices(preset);
        self.submit_build(item, fit);
    }

    fn select_default_preset_quiet(&mut self) {
        let presets = self.available_presets();
        if let Some(&first) = presets.first() {
            self.set_preset_choices(first);
        }
    }

    pub(crate) fn default_pos_for_choice(&self, idx: usize) -> [f64; 2] {
        match self.choices.get(idx) {
            Some(FuseDefectChoice::Disclination { class_idx, .. }) => self.info.centres[*class_idx].point,
            Some(FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign }) => {
                let option = &self.info.dislocations[*disloc_idx];
                let (m, n) = option.members[*member_idx];
                let sign = i32::from(*sign);
                let [bx, by] = crate::live_combine::burgers_from_mn(&self.info, m * sign, n * sign);
                [option.at[0] + 0.5 * bx, option.at[1] + 0.5 * by]
            }
            None => [0.0, 0.0],
        }
    }

    fn swap_defects(&mut self) {
        std::mem::swap(&mut self.choice_a, &mut self.choice_b);
        std::mem::swap(&mut self.pos_a, &mut self.pos_b);
        let old_anchor = self.pos_b;
        self.refresh_slots_b();
        if matches!(self.choices[self.choice_b], FuseDefectChoice::Dislocation { .. })
            && (self.pos_b[0] - old_anchor[0]).hypot(self.pos_b[1] - old_anchor[1]) > 1e-4
        {
            // Dislocation positions are continuous, so preserve the swapped position even
            // when it is between the preset half-cell steps.
            self.pos_b = old_anchor;
            self.slots_b.push(old_anchor);
            sort_slots(&mut self.slots_b, self.pos_a);
        }
    }

    pub(crate) fn refresh_slots_b(&mut self) {
        if self.choices.is_empty() {
            self.slots_b.clear();
            return;
        }
        let cb = self.choices[self.choice_b.min(self.choices.len() - 1)].clone();
        self.slots_b = compute_slots_for_b(&self.info, &cb, self.pos_a);
        if let Some(&nearest) = self.slots_b.iter().min_by(|a, b| {
            let da = (a[0] - self.pos_b[0]).hypot(a[1] - self.pos_b[1]);
            let db = (b[0] - self.pos_b[0]).hypot(b[1] - self.pos_b[1]);
            da.total_cmp(&db)
        }) {
            self.pos_b = nearest;
        }
    }

    pub(crate) fn current_slot_index(&self) -> Option<usize> {
        self.slots_b
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (a[0] - self.pos_b[0]).hypot(a[1] - self.pos_b[1]);
                let db = (b[0] - self.pos_b[0]).hypot(b[1] - self.pos_b[1]);
                da.total_cmp(&db)
            })
            .filter(|(_, p)| (p[0] - self.pos_b[0]).hypot(p[1] - self.pos_b[1]) < 1e-3)
            .map(|(i, _)| i)
    }

    pub(crate) fn submit_build(&mut self, item: &CatalogueItemUi, reset_fit: bool) {
        if reset_fit {
            self.needs_fit_both = true;
        }
        if self.last_bundle.take().is_some() {
            self.sync_single_canvas_only(item, false);
        }
        self.worker.submit(FuseJob {
            item: item.clone(),
            info: self.info.clone(),
            specs: self.current_specs(),
            relaxed: self.relaxed,
            auto_radius: self.auto_radius,
            manual_radius: self.manual_radius,
        });
    }

    pub(crate) fn poll(&mut self, ctx: &egui::Context, item: &CatalogueItemUi, left_canvas: &mut Canvas2D) {
        self.worker.set_repaint_ctx(ctx);
        match self.worker.poll() {
            Some(Ok(res)) => {
                self.last_combined = res.combined;
                if let Ok(ref b) = res.bundle {
                    self.selected_candidate = b.default_index;
                } else {
                    self.selected_candidate = 0;
                }
                self.last_bundle = Some(res.bundle);
                let fit = self.needs_fit_both;
                self.needs_fit_both = false;
                self.sync_canvases(item, left_canvas, fit);
            }
            Some(Err(err)) => {
                self.last_combined = empty_combined(item, &self.info);
                self.last_combined.charge_label = String::from("Charge unavailable");
                self.last_bundle = Some(Err(err));
                self.sync_canvases(item, left_canvas, false);
            }
            None => {}
        }
    }

    pub(crate) fn active_single_candidate(&self) -> Option<&SingleEquivalentCrystal> {
        let Some(Ok(bundle)) = &self.last_bundle else { return None };
        bundle.candidates.get(self.selected_candidate).or_else(|| bundle.candidates.first())
    }

    pub(crate) fn build_markers(&self) -> Vec<CanvasMarker> {
        let specs = self.current_specs();
        specs
            .iter()
            .enumerate()
            .map(|(i, sp)| {
                let ref_pt = if i == 0 { self.pos_a } else { self.pos_b };
                let drawn = if self.last_combined.crystal.pos.is_empty() {
                    ref_pt
                } else {
                    map_ref_to_drawn(&self.last_combined.crystal, ref_pt)
                };
                let kind = match *sp {
                    DefectSpec::Disclination { angle_deg, .. } if angle_deg >= 0.0 => MarkerKind::DisclinationPlus,
                    DefectSpec::Disclination { .. } => MarkerKind::DisclinationMinus,
                    _ => MarkerKind::Dislocation,
                };
                let tag = if i == 0 { "A" } else { "B" };
                CanvasMarker {
                    index: i,
                    pos: drawn,
                    kind,
                    label: format!("{tag}: {}", spec_short_label(&self.info, sp)),
                }
            })
            .collect()
    }

    pub(crate) fn sync_canvases(&mut self, item: &CatalogueItemUi, left_canvas: &mut Canvas2D, reset_view: bool) {
        self.combined_inspectors.circuit.enabled = !self.is_busy() && self.last_combined.refused.is_none();

        let a_short = self
            .choices
            .get(self.choice_a)
            .map_or_else(|| String::from("A"), |c| choice_short_label(&self.info, c));
        let b_short = self
            .choices
            .get(self.choice_b)
            .map_or_else(|| String::from("B"), |c| choice_short_label(&self.info, c));
        let left_title = if self.is_busy() {
            String::from("Building crystal…")
        } else if self.last_combined.refused.is_some() {
            format!("{} ({}) — pristine reference", item.label, item.group)
        } else if self.last_combined.crystal.pos.is_empty() {
            String::from("Combined crystal unavailable")
        } else {
            format!("{} ({}) — {a_short} + {b_short}", item.label, item.group)
        };
        left_canvas.set_scene_with_title(
            Scene2D::CrystalPatch {
                crystal: self.last_combined.crystal.clone(),
                title: left_title.clone(),
                note: self.last_combined.note.clone(),
            },
            Some(left_title),
            reset_view,
        );
        let markers = self.build_markers();
        left_canvas.set_export_markers(&markers);
        self.combined_inspectors.refresh_circuit(&item.group, &self.info.cell, item.frame, &self.last_combined.crystal, &[]);
        self.combined_inspectors.sync_to_canvas(left_canvas, &self.last_combined.crystal, None);
        if !self.show_burgers_loop {
            left_canvas.set_circuit_overlay(None);
        }

        self.sync_single_canvas_only(item, reset_view);
    }

    pub(crate) fn sync_single_canvas_only(&mut self, item: &CatalogueItemUi, reset_view: bool) {
        // Retain the circuit measurements for the inspector when the drawing is hidden.
        self.single_inspectors.circuit.enabled = true;
        if let Some(cand) = self.active_single_candidate().cloned() {
            let match_tag = if cand.is_match {
                " (★ matches combined core)"
            } else if cand.kind == "vacuum" {
                " (reference)"
            } else {
                " (core differs)"
            };
            let right_title = format!("{}{match_tag}", cand.name);
            self.single_canvas.set_scene_with_title(
                Scene2D::CrystalPatch {
                    crystal: cand.crystal.clone(),
                    title: right_title.clone(),
                    note: String::new(),
                },
                Some(right_title),
                reset_view,
            );
            self.single_canvas.center_on_scene();
            self.single_inspectors.refresh_circuit(&item.group, &self.info.cell, item.frame, &cand.crystal, &[]);
            self.single_inspectors.sync_to_canvas(&mut self.single_canvas, &cand.crystal, None);
            if !self.show_burgers_loop {
                self.single_canvas.set_circuit_overlay(None);
            }
        } else {
            let (title, note) = match &self.last_bundle {
                Some(Ok(b)) if b.report.single.is_none() && b.report.kind != "vacuum" => {
                    (format!("No single elementary defect for {}", b.report.charge.label), b.report.verdict.clone())
                }
                Some(Ok(b)) => (String::from("Candidate drawing unavailable"), b.report.verdict.clone()),
                Some(Err(e)) => (String::from("Single-defect comparison unavailable"), e.clone()),
                None => (String::from("Computing equivalence…"), String::new()),
            };
            self.single_canvas.set_scene(Scene2D::CrystalPatch { crystal: empty_crystal(), title, note }, reset_view);
            self.single_inspectors.clear_for_switch();
        }
    }
}
