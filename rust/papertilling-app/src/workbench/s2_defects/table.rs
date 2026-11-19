//! All-defects table of the current lattice: dislocations (Burgers [uv], Weber [uvtw]), disclinations (Frank angle),
//! walls, CSL tilt boundaries (Σ, θ) and the point defects. A header click sorts ascending, a second click descending;
//! empty cells always sort last and ties keep the build order (stable sort).

use std::cmp::Ordering;

use papertilling_core::defect::{burgers_index_info, list_csl_grain_boundaries};
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use crate::backend::CatalogueItemUi;
use crate::live_ops::catalogue_for_lattice;
use crate::workbench::WbCtx;

pub(super) const SCENE_TITLE: &str = "S2 defect table";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowKind {
    Dislocation,
    Disclination,
    Wall,
    Boundary,
    Point,
}

impl RowKind {
    pub(crate) const ALL: [Self; 5] = [Self::Dislocation, Self::Disclination, Self::Wall, Self::Boundary, Self::Point];

    fn cell(self) -> &'static str {
        match self {
            Self::Dislocation => "disloc.",
            Self::Disclination => "disclin.",
            Self::Wall => "wall",
            Self::Boundary => "tilt GB",
            Self::Point => "point",
        }
    }

    fn chip(self) -> &'static str {
        match self {
            Self::Dislocation => "Dislocations",
            Self::Disclination => "Disclinations",
            Self::Wall => "Walls",
            Self::Boundary => "Tilt boundaries",
            Self::Point => "Point kinds",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DefectRow {
    pub(crate) kind: RowKind,
    pub(crate) id: String,
    pub(crate) uv: Option<String>,
    /// |b| / |a1|.
    pub(crate) b_norm: Option<f64>,
    pub(crate) weber: Option<String>,
    pub(crate) frank_deg: Option<i32>,
    pub(crate) sigma: Option<u32>,
    pub(crate) theta: Option<f64>,
    pub(crate) orbit: Option<usize>,
    pub(crate) badge: String,
}

impl DefectRow {
    fn new(kind: RowKind, id: impl Into<String>, badge: impl Into<String>) -> Self {
        Self {
            kind,
            id: id.into(),
            uv: None,
            b_norm: None,
            weber: None,
            frank_deg: None,
            sigma: None,
            theta: None,
            orbit: None,
            badge: badge.into(),
        }
    }
}

/// Tests: a row with every optional cell empty.
#[cfg(test)]
pub(crate) fn build_row_for_tests(kind: RowKind, id: &str) -> DefectRow {
    DefectRow::new(kind, id, "")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SortCol {
    Kind,
    Id,
    Uv,
    Norm,
    Weber,
    Frank,
    Sigma,
    Theta,
    Orbit,
    Badge,
}

impl SortCol {
    pub(crate) const ALL: [Self; 10] =
        [Self::Kind, Self::Id, Self::Sigma, Self::Theta, Self::Frank, Self::Uv, Self::Weber, Self::Norm, Self::Orbit, Self::Badge];

    fn header(self) -> &'static str {
        match self {
            Self::Kind => "Kind",
            Self::Id => "Id",
            Self::Uv => "[uv]",
            Self::Norm => "|b|/|a1|",
            Self::Weber => "[uvtw]",
            Self::Frank => "Frank °",
            Self::Sigma => "Σ",
            Self::Theta => "θ °",
            Self::Orbit => "Orbit",
            Self::Badge => "Badge",
        }
    }
}

enum Key<'a> {
    Num(f64),
    Text(&'a str),
}

fn key(r: &DefectRow, col: SortCol) -> Option<Key<'_>> {
    match col {
        SortCol::Kind => Some(Key::Num(r.kind as usize as f64)),
        SortCol::Id => Some(Key::Text(&r.id)),
        SortCol::Uv => r.uv.as_deref().map(Key::Text),
        SortCol::Norm => r.b_norm.map(Key::Num),
        SortCol::Weber => r.weber.as_deref().map(Key::Text),
        SortCol::Frank => r.frank_deg.map(|f| Key::Num(f64::from(f))),
        SortCol::Sigma => r.sigma.map(|s| Key::Num(f64::from(s))),
        SortCol::Theta => r.theta.map(Key::Num),
        SortCol::Orbit => r.orbit.map(|o| Key::Num(o as f64)),
        SortCol::Badge => Some(Key::Text(&r.badge)),
    }
}

/// Row order for the table: rows of the shown kinds, stably sorted by `col`; empty cells last in both directions.
pub(crate) fn sort_order(rows: &[DefectRow], col: SortCol, asc: bool, shown: &[bool; 5]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..rows.len()).filter(|&i| shown[rows[i].kind as usize]).collect();
    idx.sort_by(|&a, &b| match (key(&rows[a], col), key(&rows[b], col)) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => {
            let o = match (x, y) {
                (Key::Num(p), Key::Num(q)) => p.total_cmp(&q),
                (Key::Text(p), Key::Text(q)) => p.cmp(q),
                (Key::Num(_), Key::Text(_)) => Ordering::Less,
                (Key::Text(_), Key::Num(_)) => Ordering::Greater,
            };
            if asc { o } else { o.reverse() }
        }
    });
    idx
}

const CELL_CHARS: usize = 14;

/// Cell text of `r` in column `c`: a short form, plus the full text when it was shortened (shown on hover).
fn cell_text(r: &DefectRow, c: SortCol) -> (String, Option<String>) {
    let dash = || String::from("-");
    let full = match c {
        SortCol::Kind => r.kind.cell().to_string(),
        SortCol::Id => r.id.clone(),
        SortCol::Uv => r.uv.clone().unwrap_or_else(dash),
        SortCol::Norm => r.b_norm.map_or_else(dash, |b| format!("{b:.3}")),
        SortCol::Weber => r.weber.clone().unwrap_or_else(dash),
        SortCol::Frank => r.frank_deg.map_or_else(dash, |f| f.to_string()),
        SortCol::Sigma => r.sigma.map_or_else(dash, |s| s.to_string()),
        SortCol::Theta => r.theta.map_or_else(dash, |t| format!("{t:.2}")),
        SortCol::Orbit => r.orbit.map_or_else(dash, |o| o.to_string()),
        SortCol::Badge if r.badge.is_empty() => dash(),
        SortCol::Badge if r.badge.starts_with("[PROVED") => String::from("[PROVED]"),
        SortCol::Badge if r.badge.starts_with("[COMPUTATIONAL") => String::from("[COMP. EV.]"),
        SortCol::Badge => r.badge.clone(),
    };
    let long = if c == SortCol::Badge && full != r.badge && !r.badge.is_empty() { Some(r.badge.clone()) } else { None };
    if full.chars().count() > CELL_CHARS {
        let short: String = full.chars().take(CELL_CHARS - 1).collect::<String>() + "…";
        return (short, Some(full));
    }
    (full, long)
}

pub(crate) struct TableOut {
    pub(crate) rows: Vec<DefectRow>,
}

/// Table rows of `item` (catalogue) on `cell` (Burgers indices, CSL list), plus the point-defect kinds.
pub(crate) fn build(item: &CatalogueItemUi, cell: &Cell2D, max_sigma: u32) -> TableOut {
    let group = item.group.as_str();
    let cat = catalogue_for_lattice(item);
    let mut rows = Vec::new();
    for d in &cat.dislocations {
        let info = burgers_index_info(group, cell, [d.burgers_mn.0, d.burgers_mn.1]);
        let mut r = DefectRow::new(RowKind::Dislocation, d.id.clone(), info.provenance);
        r.uv = Some(info.uv_str);
        r.b_norm = Some(d.norm);
        r.weber = info.weber_str;
        r.frank_deg = Some(0);
        r.orbit = Some(d.orbit_size);
        rows.push(r);
    }
    for d in &cat.disclinations {
        let mut r = DefectRow::new(RowKind::Disclination, format!("{} at {}", d.id, d.wyckoff), "");
        r.frank_deg = Some(d.frank_angle_deg);
        rows.push(r);
    }
    for w in &cat.walls {
        rows.push(DefectRow::new(RowKind::Wall, format!("{} {}", w.kind.label(), w.id), ""));
    }
    for e in list_csl_grain_boundaries(group, cell, max_sigma) {
        let mut r = DefectRow::new(RowKind::Boundary, e.id.clone(), e.provenance);
        r.sigma = Some(e.sigma);
        r.theta = Some(e.theta_deg);
        rows.push(r);
    }
    for id in ["vacancy", "substitution", "interstitial", "adatom", "SW bond rotation"] {
        let mut r = DefectRow::new(RowKind::Point, id, "[PROVED]");
        r.uv = Some(String::from("[0 0]"));
        r.b_norm = Some(0.0);
        r.frank_deg = Some(0);
        rows.push(r);
    }
    TableOut { rows }
}

pub(crate) struct TableState {
    pub(crate) col: SortCol,
    pub(crate) asc: bool,
    pub(crate) shown: [bool; 5],
    pub(crate) out: Option<TableOut>,
    pub(crate) order: Vec<usize>,
    /// Burgers index calculator input `b = u a1 + v a2`.
    calc: [i32; 2],
}

impl Default for TableState {
    fn default() -> Self {
        Self { col: SortCol::Kind, asc: true, shown: [true; 5], out: None, order: Vec::new(), calc: [1, 0] }
    }
}

impl TableState {
    pub(crate) fn set_out(&mut self, out: TableOut) {
        self.out = Some(out);
        self.resort();
    }

    fn resort(&mut self) {
        self.order = self.out.as_ref().map_or_else(Vec::new, |o| sort_order(&o.rows, self.col, self.asc, &self.shown));
    }

    /// Header click: sort by `col` ascending, or flip the direction when it is already the sort column.
    pub(crate) fn header_clicked(&mut self, col: SortCol) {
        if self.col == col {
            self.asc = !self.asc;
        } else {
            (self.col, self.asc) = (col, true);
        }
        self.resort();
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui, cx: &WbCtx) -> bool {
        ui.label("Show:");
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for k in RowKind::ALL {
                changed |= ui.toggle_value(&mut self.shown[k as usize], k.chip()).changed();
            }
        });
        if changed {
            self.resort();
        }
        ui.separator();
        ui.label("Burgers index calculator, b = u a1 + v a2:");
        ui.horizontal(|ui| {
            ui.label("u");
            ui.add(egui::DragValue::new(&mut self.calc[0]).range(-6..=6));
            ui.label("v");
            ui.add(egui::DragValue::new(&mut self.calc[1]).range(-6..=6));
        });
        let info = burgers_index_info(&cx.item.group, cx.cell, self.calc);
        ui.label(format!("[uv] = {}, family {}", info.uv_str, info.family_str));
        match &info.weber_str {
            Some(w) => ui.label(format!("Weber [uvtw] = {w}")),
            None => ui.label("Weber [uvtw]: only for hexagonal cells"),
        };
        ui.small(info.provenance);
        false
    }

    pub(super) fn show_inspector(&mut self, ui: &mut egui::Ui) {
        let Some(out) = &self.out else {
            ui.label("Building the defect table ...");
            return;
        };
        ui.label(format!("{} of {} rows shown; click a header to sort.", self.order.len(), out.rows.len()));
        let mut clicked = None;
        egui::ScrollArea::both().id_salt("wb_s2_table").max_height(480.0).show(ui, |ui| {
            egui::Grid::new("wb_s2_table_grid").striped(true).num_columns(SortCol::ALL.len()).show(ui, |ui| {
                for c in SortCol::ALL {
                    let arrow = if c != self.col { "" } else if self.asc { " (asc)" } else { " (desc)" };
                    if ui.small_button(format!("{}{arrow}", c.header())).clicked() {
                        clicked = Some(c);
                    }
                }
                ui.end_row();
                for &i in &self.order {
                    let r = &out.rows[i];
                    for c in SortCol::ALL {
                        let (short, full) = cell_text(r, c);
                        let resp = ui.label(short);
                        if let Some(f) = full {
                            resp.on_hover_text(f);
                        }
                    }
                    ui.end_row();
                }
            });
        });
        if let Some(c) = clicked {
            self.header_clicked(c);
        }
    }

    pub(super) fn save(&self) -> Value {
        json!({ "col": format!("{:?}", self.col), "asc": self.asc })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(c) = v.get("col").and_then(Value::as_str).and_then(|s| SortCol::ALL.into_iter().find(|c| format!("{c:?}") == s)) {
            self.col = c;
        }
        if let Some(a) = v.get("asc").and_then(Value::as_bool) {
            self.asc = a;
        }
        self.resort();
    }
}
