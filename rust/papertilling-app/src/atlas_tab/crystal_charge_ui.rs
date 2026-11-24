//! Based affine charge and proper affine conjugacy for space and layer groups.

use papertilling_core::space3d::holonomy::{AffineHolonomy, CrystalHolonomies};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CrystalGroup {
    Space(u16),
    Layer(u8),
}

#[derive(Clone, Copy, Default)]
struct Choice {
    rep: usize,
    periods: [i32; 3],
}

struct Controls {
    group: CrystalHolonomies,
    choices: [Choice; 4],
    reverse_loop: bool,
    labels: Vec<String>,
}

impl Controls {
    fn new(group: CrystalHolonomies) -> Self {
        let reps = group.representatives();
        let glide = reps.iter().position(|op| op.kind() == "Glide reflection");
        let a_rep = glide.or_else(|| (reps.len() > 1).then_some(1)).unwrap_or(0);
        let a = Choice {
            rep: a_rep,
            periods: [i32::from(a_rep == 0), 0, 0],
        };
        let probe = reps
            .iter()
            .position(|op| op.kind() == "Reflection")
            .or(glide)
            .unwrap_or(0);
        let labels = reps
            .iter()
            .enumerate()
            .map(|(i, op)| format!("{i} · {} {}", op.kind(), shift_text(op)))
            .collect();
        Self {
            group,
            choices: [
                a,
                a,
                Choice {
                    rep: 0,
                    periods: [1, 0, 0],
                },
                Choice {
                    rep: probe,
                    ..Choice::default()
                },
            ],
            reverse_loop: false,
            labels,
        }
    }

    fn show(&mut self, ui: &mut egui::Ui, layers: bool) {
        let rank = self.group.rank();
        let [a, b, target, _] = &mut self.choices;
        pick_element(ui, "A", a, &self.labels, rank);
        pick_element(ui, "B", b, &self.labels, rank);
        pick_element(ui, "Candidate", target, &self.labels, rank);
        let [a, b, target, probe_op] = self.choices.map(|c| {
            self.group
                .element(c.rep, c.periods)
                .expect("picker selects a valid group element")
        });
        let fused = b.compose(&a);
        let backwards = a.compose(&b);
        ui.strong("A then B");
        describe(ui, &fused, layers);
        ui.small(if fused == backwards {
            "Reversing A and B gives the same charge."
        } else {
            "Reversing A and B changes the based charge."
        });
        let comparison = if target == fused {
            "Exact based match"
        } else if self.group.properly_conjugate(&fused, &target) {
            "Same charge up to proper affine conjugacy"
        } else {
            "Different proper affine charge"
        };
        ui.strong(comparison);
        egui::CollapsingHeader::new("Input maps").show(ui, |ui| {
            for (name, op) in [("A", &a), ("B", &b), ("Candidate", &target)] {
                ui.strong(name);
                describe(ui, op, layers);
            }
        });
        ui.separator();
        let probe = &mut self.choices[3];
        pick_element(ui, "Symmetry probe", probe, &self.labels, rank);
        ui.checkbox(&mut self.reverse_loop, "Reverse circuit traversal")
            .on_hover_text(
                "Invert the conjugated charge when comparing circuits with opposite traversal.",
            );
        let image = probe_op.conjugate(&a);
        let image = if self.reverse_loop {
            image.inverse()
        } else {
            image
        };
        ui.strong(if image == a {
            "Probe preserves the chosen charge A"
        } else {
            "Probe changes the chosen charge A"
        });
        let (m, power) = probe_op.closing_power();
        if m > 1 {
            ui.small(format!("Probe^{m} translates by {}", shift_text(&power)));
        }
        if image != a {
            ui.small("Image of A");
            describe(ui, &image, layers);
        }
        if let Some((alternate, member)) = self.group.alternate_registry(&probe_op) {
            ui.separator();
            ui.strong("Alternate wall registry");
            ui.small(format!(
                "{} · shift {}",
                alternate.kind(),
                shift_text(&alternate)
            ));
            ui.small(if member {
                "This registry is also a bulk symmetry."
            } else {
                "This registry changes the bulk symmetry action across a join."
            });
            ui.small("Registry test for a prospective join. Check the sites and bonds on a constructed wall to determine its residual symmetry.");
        }
    }
}

#[derive(Default)]
pub(super) struct CrystalChargeUi {
    cached: Option<(CrystalGroup, Result<Controls, String>)>,
}

impl CrystalChargeUi {
    pub(super) fn show(&mut self, ui: &mut egui::Ui, key: CrystalGroup) {
        if self
            .cached
            .as_ref()
            .is_none_or(|(cached, _)| *cached != key)
        {
            let group = match key {
                CrystalGroup::Space(n) => CrystalHolonomies::space_group(n),
                CrystalGroup::Layer(n) => CrystalHolonomies::layer_group(n),
            };
            self.cached = Some((key, group.map(Controls::new)));
        }
        ui.heading("Affine charge lab");
        ui.small("Group operations for circuit charges and seam tests. Rotational lifts and defect cores are separate.");
        egui::CollapsingHeader::new("Coordinates and comparison").show(ui, |ui| {
            ui.small("Use a point-operation representative followed by whole primitive periods. Readouts use the conventional fractional cell coordinates.");
            ui.small("The conjugacy test uses all primitive translations and proper ambient point operations. The probe checks the based charge of A, so the geometry and full symmetry of a built wall need separate checks.");
        });
        match &mut self.cached.as_mut().unwrap().1 {
            Ok(controls) => controls.show(ui, matches!(key, CrystalGroup::Layer(_))),
            Err(err) => {
                ui.label(err.as_str());
            }
        }
    }
}

fn pick_element(
    ui: &mut egui::Ui,
    name: &str,
    choice: &mut Choice,
    labels: &[String],
    rank: usize,
) {
    egui::ComboBox::new(name, name)
        .width(144.0)
        .selected_text(&labels[choice.rep])
        .show_ui(ui, |ui| {
            for (i, label) in labels.iter().enumerate() {
                ui.selectable_value(&mut choice.rep, i, label);
            }
        });
    ui.horizontal_wrapped(|ui| {
        ui.small("Periods");
        for (i, t) in choice.periods[..rank].iter_mut().enumerate() {
            ui.add(egui::DragValue::new(t).range(-1000..=1000).speed(0.1))
                .on_hover_text(format!("{name}: primitive period {}", i + 1));
        }
    });
}

fn shift_text(op: &AffineHolonomy) -> String {
    let t = op.shift();
    format!("({}, {}, {})", t[0], t[1], t[2])
}

fn describe(ui: &mut egui::Ui, op: &AffineHolonomy, layers: bool) {
    ui.small(format!("{} · shift {}", op.kind(), shift_text(op)));
    let r = op.linear();
    let text = r
        .iter()
        .map(|r| format!("[{:>2} {:>2} {:>2}]", r[0], r[1], r[2]))
        .collect::<Vec<_>>()
        .join("\n");
    ui.monospace(text);
    if layers {
        let orientation = if op.preserves_layer_orientation() {
            "preserves"
        } else {
            "reverses"
        };
        ui.small(format!("In-plane action {orientation} orientation."));
    }
}
