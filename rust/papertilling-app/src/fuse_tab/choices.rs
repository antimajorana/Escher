//! Catalogue choice enumeration, reaction presets, target-site shells, and background job runner for `FuseTabState`.

use papertilling_core::defect::single_cache::SingleDefectCache;
use papertilling_core::defect::single_crystal::build_single_equivalence_bundle;
use papertilling_core::defect::surgery::DefectSpec;

use crate::live_combine::{
    CombineBuildOptions, CombineDrawMode, CombineLatticeInfo, align_crystal_orientation,
    build_combined_crystal, burgers_from_mn,
};

use super::{FuseBuildResult, FuseDefectChoice, FuseJob, FusePreset, FuseTabState};

pub(super) fn enumerate_choices(info: &CombineLatticeInfo) -> Vec<FuseDefectChoice> {
    let mut out = Vec::new();
    for (class_idx, c) in info.centres.iter().enumerate() {
        let mut angs: Vec<i32> = c
            .angles
            .iter()
            .map(|&a| a.round() as i32)
            .filter(|&a| a != 0 && a.abs() <= 180)
            .collect();
        angs.sort_by_key(|&a| (a.abs(), -a));
        angs.dedup();
        for angle_deg in angs {
            out.push(FuseDefectChoice::Disclination { class_idx, angle_deg });
        }
    }
    for (disloc_idx, d) in info.dislocations.iter().enumerate() {
        for (member_idx, _) in d.members.iter().enumerate() {
            out.push(FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign: 1 });
            if d.members.len() == 1 {
                out.push(FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign: -1 });
            }
        }
    }
    out
}

pub(crate) fn choice_label(info: &CombineLatticeInfo, ch: &FuseDefectChoice) -> String {
    match *ch {
        FuseDefectChoice::Disclination { class_idx, angle_deg } => {
            let c = &info.centres[class_idx];
            format!("D({}, {:+}°) · {}", c.label, angle_deg, c.location)
        }
        FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign } => {
            let d = &info.dislocations[disloc_idx];
            let (m0, n0) = d.members[member_idx];
            let (m, n) = (m0 * i32::from(sign), n0 * i32::from(sign));
            let b = burgers_from_mn(info, m, n);
            let prefix = if sign < 0 { "−" } else { "" };
            format!("{prefix}{} · b=[{m},{n}] (|b|={:.2})", d.id, b[0].hypot(b[1]))
        }
    }
}

pub(crate) fn choice_short_label(info: &CombineLatticeInfo, ch: &FuseDefectChoice) -> String {
    match *ch {
        FuseDefectChoice::Disclination { class_idx, angle_deg } => {
            format!("D({}, {:+}°)", info.centres[class_idx].label, angle_deg)
        }
        FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign } => {
            let d = &info.dislocations[disloc_idx];
            let (m0, n0) = d.members[member_idx];
            format!("b=[{},{}]", m0 * i32::from(sign), n0 * i32::from(sign))
        }
    }
}

pub(super) fn spec_from_choice(info: &CombineLatticeInfo, ch: &FuseDefectChoice, pos: [f64; 2]) -> DefectSpec {
    match *ch {
        FuseDefectChoice::Disclination { angle_deg, .. } => DefectSpec::disclination(f64::from(angle_deg), pos),
        FuseDefectChoice::Dislocation { disloc_idx, member_idx, sign } => {
            let (m0, n0) = info.dislocations[disloc_idx].members[member_idx];
            let b = burgers_from_mn(info, m0 * i32::from(sign), n0 * i32::from(sign));
            DefectSpec::dislocation(b, Some([pos[0] - 0.5 * b[0], pos[1] - 0.5 * b[1]]))
        }
    }
}

pub(super) fn compute_slots_for_b(info: &CombineLatticeInfo, ch_b: &FuseDefectChoice, pos_a: [f64; 2]) -> Vec<[f64; 2]> {
    let a1 = info.cell.a1;
    let a2 = info.cell.a2;
    let scale = a1[0].hypot(a1[1]).max(a2[0].hypot(a2[1])).max(1e-6);
    let max_d = 2.8 * scale;
    let mut pts: Vec<[f64; 2]> = Vec::new();

    match *ch_b {
        FuseDefectChoice::Disclination { class_idx, .. } => {
            let c = &info.centres[class_idx];
            for n1 in -4..=4 {
                for n2 in -4..=4 {
                    let shift = [f64::from(n1) * a1[0] + f64::from(n2) * a2[0], f64::from(n1) * a1[1] + f64::from(n2) * a2[1]];
                    for &seed in &c.unit_seeds {
                        let p = [seed[0] + shift[0], seed[1] + shift[1]];
                        if (p[0] - pos_a[0]).hypot(p[1] - pos_a[1]) <= max_d {
                            pts.push(p);
                        }
                    }
                }
            }
        }
        FuseDefectChoice::Dislocation { .. } => {
            for h1 in -5..=5 {
                for h2 in -5..=5 {
                    let (f1, f2) = (0.5 * f64::from(h1), 0.5 * f64::from(h2));
                    let p = [pos_a[0] + f1 * a1[0] + f2 * a2[0], pos_a[1] + f1 * a1[1] + f2 * a2[1]];
                    if (p[0] - pos_a[0]).hypot(p[1] - pos_a[1]) <= max_d {
                        pts.push(p);
                    }
                }
            }
        }
    }

    let q = |x: f64| (x * 1e4).round() as i64;
    pts.sort_by(|u, v| {
        let du = (u[0] - pos_a[0]).hypot(u[1] - pos_a[1]);
        let dv = (v[0] - pos_a[0]).hypot(v[1] - pos_a[1]);
        q(du)
            .cmp(&q(dv))
            .then_with(|| q(u[0] - pos_a[0]).cmp(&q(v[0] - pos_a[0])))
            .then_with(|| q(u[1] - pos_a[1]).cmp(&q(v[1] - pos_a[1])))
    });
    let mut dedup: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for p in pts {
        if dedup.iter().all(|q_pt| (p[0] - q_pt[0]).hypot(p[1] - q_pt[1]) > 1e-4) {
            dedup.push(p);
        }
    }
    dedup
}

pub(super) fn run_fuse_job(cache: &mut SingleDefectCache, job: FuseJob) -> FuseBuildResult {
    let opts = CombineBuildOptions {
        draw_mode: if job.relaxed { CombineDrawMode::FastRelaxed } else { CombineDrawMode::Raw },
        full_planar_repair: true,
        auto_radius: job.auto_radius,
        manual_radius: job.manual_radius,
        core_radius: 2,
    };
    let combined = build_combined_crystal(&job.item, &job.info, &job.specs, &opts, false);
    let bundle = if let Some(ref err) = combined.refused {
        Err(err.clone())
    } else {
        build_single_equivalence_bundle(
            &job.item.group,
            &job.info.cell,
            job.item.frame,
            &job.specs,
            &combined.crystal,
            combined.applied_radius,
            2,
            1,
            job.relaxed,
            cache,
        )
        .map(|mut b| {
            for cand in &mut b.candidates {
                if job.relaxed {
                    align_crystal_orientation(&mut cand.crystal);
                }
            }
            b
        })
    };
    FuseBuildResult { combined, bundle }
}

impl FuseTabState {
    pub(super) fn set_preset_choices(&mut self, preset: FusePreset) {
        if self.choices.is_empty() {
            return;
        }
        match preset {
            FusePreset::DipoleToDislocation => {
                if let Some(ia) = self.choices.iter().position(|c| matches!(c, FuseDefectChoice::Disclination { angle_deg, .. } if *angle_deg > 0)) {
                    let (ca_cls, ang) = match self.choices[ia] {
                        FuseDefectChoice::Disclination { class_idx, angle_deg } => (class_idx, angle_deg),
                        _ => (0, 60),
                    };
                    let ib = self
                        .choices
                        .iter()
                        .position(|c| matches!(c, FuseDefectChoice::Disclination { class_idx, angle_deg } if *class_idx != ca_cls && *angle_deg == -ang))
                        .or_else(|| self.choices.iter().position(|c| matches!(c, FuseDefectChoice::Disclination { angle_deg, .. } if *angle_deg == -ang)))
                        .unwrap_or(ia);
                    self.choice_a = ia;
                    self.choice_b = ib;
                    self.pos_a = self.default_pos_for_choice(self.choice_a);
                    self.refresh_slots_b();
                    if let Some(&p) = self.slots_b.iter().find(|p| (p[0] - self.pos_a[0]).hypot(p[1] - self.pos_a[1]) > 1e-3) {
                        self.pos_b = p;
                    }
                }
            }
            FusePreset::OppositeAnnihilate => {
                if let Some(ia) = self.choices.iter().position(|c| matches!(c, FuseDefectChoice::Disclination { angle_deg, .. } if *angle_deg > 0)) {
                    let (ca_cls, ang) = match self.choices[ia] {
                        FuseDefectChoice::Disclination { class_idx, angle_deg } => (class_idx, angle_deg),
                        _ => (0, 60),
                    };
                    let ib = self
                        .choices
                        .iter()
                        .position(|c| matches!(c, FuseDefectChoice::Disclination { class_idx, angle_deg } if *class_idx == ca_cls && *angle_deg == -ang))
                        .unwrap_or(ia);
                    self.choice_a = ia;
                    self.choice_b = ib;
                    self.pos_a = self.default_pos_for_choice(self.choice_a);
                    self.refresh_slots_b();
                    if let Some(&p) = self.slots_b.iter().find(|p| (p[0] - self.pos_a[0]).hypot(p[1] - self.pos_a[1]) > 1e-3) {
                        self.pos_b = p;
                    }
                }
            }
            FusePreset::DisclinationPlusDislocation => {
                let ia = self.choices.iter().position(|c| matches!(c, FuseDefectChoice::Disclination { angle_deg, .. } if *angle_deg > 0)).unwrap_or(0);
                let ib = self.choices.iter().position(|c| matches!(c, FuseDefectChoice::Dislocation { sign: 1, .. })).unwrap_or(0);
                self.choice_a = ia;
                self.choice_b = ib;
                self.pos_a = self.default_pos_for_choice(self.choice_a);
                self.refresh_slots_b();
                if let Some(&p) = self.slots_b.iter().find(|p| (p[0] - self.pos_a[0]).hypot(p[1] - self.pos_a[1]) > 0.4) {
                    self.pos_b = p;
                }
            }
            FusePreset::DislocationSum => {
                let dislocs: Vec<usize> = self
                    .choices
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| matches!(c, FuseDefectChoice::Dislocation { sign: 1, .. }).then_some(i))
                    .collect();
                let ia = dislocs.first().copied().unwrap_or(0);
                let ib = dislocs.get(1).copied().unwrap_or(ia);
                self.choice_a = ia;
                self.choice_b = ib;
                self.pos_a = self.default_pos_for_choice(self.choice_a);
                self.refresh_slots_b();
                if let Some(&p) = self.slots_b.iter().find(|p| (p[0] - self.pos_a[0]).hypot(p[1] - self.pos_a[1]) > 0.8) {
                    self.pos_b = p;
                }
            }
        }
    }
}
