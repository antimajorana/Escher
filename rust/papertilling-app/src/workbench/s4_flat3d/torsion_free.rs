//! Exact torsion-free classification of the 230 3D space groups (computed once on the worker, then cached), shown as
//! a filterable table. A group is torsion-free iff no coset `(R | τ_R + Z³)` with `R ≠ 1` has a rational fixed point;
//! the core decides this in exact rational arithmetic, so each row's verdict is `[PROVED]`.

use papertilling_core::flat3d::{TorsionFreeReport, classify_all_3d_torsion_free};
use serde_json::{Value, json};

pub(super) const SCENE_TITLE: &str = "S4 torsion-free space groups";
/// The 13 torsion-free (Bieberbach) ITA numbers (Conway-Rossetti 2003; Hahn, ITA Vol. A).
pub(super) const BIEBERBACH_ITA: [i32; 13] = [1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170];

pub(super) fn build() -> Result<Vec<TorsionFreeReport>, String> {
    let r = classify_all_3d_torsion_free()?;
    if r.len() != 230 {
        return Err(format!("expected 230 space groups, the core returned {}", r.len()));
    }
    Ok(r)
}

pub(super) struct TorsionState {
    pub(super) only_free: bool,
    pub(super) query: String,
    pub(super) out: Option<Vec<TorsionFreeReport>>,
}

impl Default for TorsionState {
    fn default() -> Self {
        Self { only_free: true, query: String::new(), out: None }
    }
}

impl TorsionState {
    /// Indices of the rows that pass the filter (torsion-free only, and the ITA number or symbol containing the query).
    pub(super) fn visible(&self) -> Vec<usize> {
        let Some(rows) = &self.out else { return Vec::new() };
        let q = self.query.trim().to_lowercase();
        (0..rows.len())
            .filter(|&i| {
                let r = &rows[i];
                (!self.only_free || r.is_torsion_free)
                    && (q.is_empty() || r.number.to_string() == q || r.symbol.to_lowercase().contains(&q))
            })
            .collect()
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.only_free, "Show torsion-free rows only");
        ui.horizontal(|ui| {
            ui.label("Filter (ITA number or symbol):");
            ui.add(egui::TextEdit::singleline(&mut self.query).desired_width(90.0));
        });
        ui.small("The table is in the inspector; it is computed once per session.");
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(rows) = &self.out else {
            ui.label("Classifying the 230 space groups...");
            return;
        };
        let free: Vec<i32> = rows.iter().filter(|r| r.is_torsion_free).map(|r| r.number).collect();
        ui.label(format!(
            "[PROVED] {} of {} groups are torsion-free (exact rational fixed-point test): ITA {:?}",
            free.len(),
            rows.len(),
            free
        ));
        let matches = free == BIEBERBACH_ITA;
        ui.label(format!(
            "[PROVED] published list of the 13 Bieberbach groups (Conway-Rossetti 2003): {}",
            if matches { "matches" } else { "DOES NOT MATCH (please report)" }
        ));
        ui.separator();
        let vis = self.visible();
        ui.label(format!("{} rows shown", vis.len()));
        egui::ScrollArea::vertical().id_salt("wb_s4_tf_scroll").max_height(360.0).show(ui, |ui| {
            egui::Grid::new("wb_s4_tf_grid").striped(true).show(ui, |ui| {
                for h in ["ITA", "symbol", "free", "|P|", "split", "G^ab", "H²(P; L)"] {
                    ui.strong(h);
                }
                ui.end_row();
                for &i in &vis {
                    let r = &rows[i];
                    ui.label(format!("#{}", r.number));
                    ui.label(&r.symbol);
                    let resp = ui.label(if r.is_torsion_free { "yes" } else { "no" });
                    if let Some(o) = r.obstructions.first() {
                        resp.on_hover_text(&o.description);
                    }
                    ui.label(r.point_order.to_string());
                    ui.label(if r.is_split { "yes" } else { "no" });
                    ui.label(&r.ab_full_label);
                    ui.label(&r.h2_label);
                    ui.end_row();
                }
            });
        });
    }

    pub(super) fn save(&self) -> Value {
        json!({"only_free": self.only_free, "query": self.query})
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(b) = v.get("only_free").and_then(Value::as_bool) {
            self.only_free = b;
        }
        if let Some(q) = v.get("query").and_then(Value::as_str) {
            self.query = q.chars().take(32).collect();
        }
    }
}
