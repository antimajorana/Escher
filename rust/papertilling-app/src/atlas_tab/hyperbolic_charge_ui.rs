//! Reflection words for the regular hyperbolic tiling and honeycomb panels.

use papertilling_core::hyperbolic::holonomy::ReflectionGroup;

pub(super) struct HyperbolicChargeUi {
    a: String,
    b: String,
    candidate: String,
    probe: String,
    reverse_loop: bool,
}

impl Default for HyperbolicChargeUi {
    fn default() -> Self {
        Self {
            a: "01".into(),
            b: "10".into(),
            candidate: "e".into(),
            probe: "0".into(),
            reverse_loop: false,
        }
    }
}

impl HyperbolicChargeUi {
    pub(super) fn show(
        &mut self,
        ui: &mut egui::Ui,
        group: Result<ReflectionGroup, String>,
        h3: bool,
    ) {
        ui.heading("Hyperbolic charge lab");
        ui.small(if h3 {
            "Mirrors 0–3 in the honeycomb chamber, e for identity. Rightmost mirror acts first."
        } else {
            "Mirrors 0–2 in the regular reference chamber, e for identity. Rightmost mirror acts first."
        });
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Inverse pair").clicked() {
                self.a = "01".into();
                self.b = "10".into();
                self.candidate = "e".into();
                self.probe = "0".into();
                self.reverse_loop = false;
            }
            if ui
                .small_button(if h3 {
                    "Adjacent rotations"
                } else {
                    "Glide probe"
                })
                .clicked()
            {
                self.a = "12".into();
                self.b = "01".into();
                self.candidate = "02".into();
                self.probe = if h3 { "3".into() } else { "012".into() };
                self.reverse_loop = false;
            }
        });
        ui.push_id("hyperbolic_charge", |ui| {
            word_input(ui, "A", &mut self.a);
            word_input(ui, "B", &mut self.b);
            word_input(ui, "Candidate", &mut self.candidate);
            word_input(ui, "Symmetry probe", &mut self.probe);
        });
        ui.checkbox(&mut self.reverse_loop, "Reverse circuit traversal");
        let result = group.and_then(|g| self.show_comparison(ui, &g));
        if let Err(err) = result {
            ui.small(err);
        }
        egui::CollapsingHeader::new("What the comparison measures").show(ui, |ui| {
            ui.small("Exact equality of chamber isometries. The probe conjugates A, with optional reversal of the circuit. The result compares based charges. It leaves lifted rotations, other conjugators, glued cores, and elastic stability open.");
        });
    }

    fn show_comparison(&self, ui: &mut egui::Ui, group: &ReflectionGroup) -> Result<(), String> {
        let a = group.parse(&self.a).map_err(|e| format!("A: {e}"))?;
        let b = group.parse(&self.b).map_err(|e| format!("B: {e}"))?;
        let candidate = group
            .parse(&self.candidate)
            .map_err(|e| format!("Candidate: {e}"))?;
        let probe = group
            .parse(&self.probe)
            .map_err(|e| format!("Symmetry probe: {e}"))?;
        let fused = b.compose(&a);
        ui.strong(if group.same_isometry(&fused, &candidate)? {
            "Exact based match"
        } else {
            "Different based charge"
        });
        ui.small(format!("A then B = {fused} · {}", group.kind(&fused)?));
        if !group.same_isometry(&fused, &a.compose(&b))? {
            ui.small("Reversing A and B changes the based charge.");
        }
        ui.separator();
        ui.small(format!("Probe · {}", group.kind(&probe)?));
        let image = probe.conjugate(&a);
        let image = if self.reverse_loop {
            image.inverse()
        } else {
            image
        };
        ui.strong(if group.same_isometry(&image, &a)? {
            "Probe preserves the chosen charge A"
        } else {
            "Probe changes the chosen charge A"
        });
        ui.small(format!("Image of A = {image}"));
        Ok(())
    }
}

fn word_input(ui: &mut egui::Ui, name: &str, word: &mut String) {
    ui.horizontal(|ui| {
        let label = ui.label(name);
        ui.add(
            egui::TextEdit::singleline(word)
                .desired_width(100.0)
                .char_limit(36),
        )
        .labelled_by(label.id);
    });
}
