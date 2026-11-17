//! Notebook 3 import/export. Kept separate from live worker state and interaction handling.
use papertilling_core::defect::combine_types::{ConfigOptionsJson, ConfigSummaryJson, NotebookConfigJson};

use super::{CombineTabState, MANUAL_R_MAX, MANUAL_R_MIN};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::live_combine::CombineDrawMode;

impl CombineTabState {
    pub fn to_notebook_json(&self, gallery_item: &CatalogueItemUi) -> NotebookConfigJson {
        let item = self.effective_item(gallery_item);
        let summary = self.current_report().map(|rep| ConfigSummaryJson {
            charge: rep.charge.label.clone(),
            kind: rep.kind.clone(),
            equivalent: rep.equivalent,
            matches: rep.matches.clone(),
            volterra_matches: rep.volterra_matches.clone(),
            completion_matches: rep.completion_matches.clone(),
            match_tier: Some(rep.match_tier.clone()),
            verdict: rep.verdict.clone(),
        });
        NotebookConfigJson {
            format: String::from("papertilling.config.v1"),
            lattice_label: item.label.clone(),
            group: item.group.clone(),
            recipe: item.recipe.clone(),
            frame: item.frame,
            defects: self.specs.clone(),
            options: ConfigOptionsJson {
                r_patch: (!self.opts.auto_radius).then_some(self.opts.manual_radius),
                picture: if self.opts.draw_mode == CombineDrawMode::Raw { "raw" } else { "relaxed" }.into(),
                grid: self.grid,
                radius: self.opts.core_radius,
            },
            summary,
        }
    }

    pub fn load_notebook_json(&mut self, cfg: &NotebookConfigJson, backend: &AppBackend, canvas: &mut Canvas2D) -> Result<String, String> {
        if cfg.format != papertilling_core::defect::combine_types::CONFIG_FORMAT {
            return Err(format!("unsupported configuration format {:?}", cfg.format));
        }
        let imported = backend.item_from_notebook_json(cfg)?;
        self.initializer.cancel();
        self.reset_lattice_defaults(&imported, backend.combine_lattice_info(&imported));
        let o = &cfg.options;
        self.opts.auto_radius = o.r_patch.is_none();
        self.opts.manual_radius = o.r_patch.unwrap_or(self.info.r_auto_base).clamp(MANUAL_R_MIN, MANUAL_R_MAX);
        self.opts.draw_mode = if o.picture == "raw" { CombineDrawMode::Raw } else { CombineDrawMode::FastRelaxed };
        self.grid = o.grid.clamp(1, 3);
        self.opts.core_radius = o.radius.clamp(1, 4);
        self.specs = cfg.defects.clone();
        let mut clamped = Vec::new();
        if o.r_patch.is_some_and(|r| (r - self.opts.manual_radius).abs() > 1e-9) {
            clamped.push(format!("r_patch {:?} -> {}", o.r_patch.unwrap_or_default(), self.opts.manual_radius));
        }
        if o.grid != self.grid {
            clamped.push(format!("grid {} -> {}", o.grid, self.grid));
        }
        if o.radius != self.opts.core_radius {
            clamped.push(format!("radius {} -> {}", o.radius, self.opts.core_radius));
        }
        let mut msg = format!("Imported {} ({}) with {} defect(s)", imported.label, imported.group, self.specs.len());
        if !clamped.is_empty() {
            msg.push_str(&format!(" (outside the app's range: {})", clamped.join(", ")));
        }
        self.override_item = Some(imported.clone());
        self.request_build(&imported, false, self.auto_compare || cfg.summary.is_some(), true);
        self.sync_to_canvas(&imported, canvas, true);
        self.json_status = msg.clone();
        Ok(msg)
    }

    pub fn export_json_dialog(&mut self, gallery_item: &CatalogueItemUi) {
        if self.specs.is_empty() || self.initializer.is_busy() {
            self.json_status = String::from("Place at least one defect before exporting a Notebook 3 configuration.");
            return;
        }
        let dlg = rfd::FileDialog::new().set_title("Export Notebook 3 defect configuration (.json)").set_file_name("defect_config.json").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.save_file() {
            let cfg = self.to_notebook_json(gallery_item);
            let partial = if cfg.summary.is_none() && !self.specs.is_empty() { " (no summary: comparison not finished)" } else { "" };
            let res = cfg.to_json_pretty().and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()));
            self.json_status = res.map_or_else(|e| format!("JSON export error: {e}"), |()| format!("Exported {}{partial}", path.file_name().and_then(|s| s.to_str()).unwrap_or("config.json")));
        }
    }

    pub fn import_json_dialog(&mut self, backend: &AppBackend, canvas: &mut Canvas2D) {
        let dlg = rfd::FileDialog::new().set_title("Import Notebook 3 defect configuration (.json)").add_filter("JSON configuration", &["json"]);
        if let Some(path) = dlg.pick_file()
            && let Err(err) = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|raw| NotebookConfigJson::from_json_str(&raw)).and_then(|cfg| self.load_notebook_json(&cfg, backend, canvas))
        {
            self.json_status = format!("JSON import error: {err}");
        }
    }
}
