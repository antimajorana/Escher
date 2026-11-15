use egui_extras::{Column, Size, StripBuilder, TableBuilder};

use crate::canvas2d::KagomeCanvas;
use crate::view3d::View3dState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NavTab {
    Gallery,
    Defects,
    CombineLive,
    Surfaces,
}

impl NavTab {
    const ALL: [Self; 4] = [
        Self::Gallery,
        Self::Defects,
        Self::CombineLive,
        Self::Surfaces,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Gallery => "Gallery",
            Self::Defects => "Defects",
            Self::CombineLive => "Combine Live",
            Self::Surfaces => "Surfaces",
        }
    }
}

const SAMPLE_GROUPS: [(&str, &str, u8, &str); 7] = [
    ("p1", "oblique", 1, "oblique net"),
    ("p2", "oblique", 2, "centred parallelogram"),
    ("pm", "rectangular", 2, "striped ladder"),
    ("p4m", "square", 8, "square / Lieb"),
    ("p4g", "square", 8, "snub square (Cairo dual)"),
    ("p6", "hexagonal", 6, "snub hexagonal"),
    ("p6m", "hexagonal", 12, "kagome / honeycomb"),
];

pub struct PapertillingApp {
    smoke_test: bool,
    smoke_reported: bool,
    adapter_name: String,
    backend_name: String,
    active_tab: NavTab,
    selected_group: usize,
    export_status: String,
    canvas2d: KagomeCanvas,
    view3d: View3dState,
}

impl PapertillingApp {
    pub fn new(smoke_test: bool, adapter_name: String, backend_name: String) -> Self {
        Self {
            smoke_test,
            smoke_reported: false,
            adapter_name,
            backend_name,
            active_tab: NavTab::Gallery,
            selected_group: 6,
            export_status: String::from("Ready"),
            canvas2d: KagomeCanvas::default(),
            view3d: View3dState::default(),
        }
    }

    fn export_dialog(&mut self) {
        let dialog = rfd::FileDialog::new()
            .set_title("Export lattice preview")
            .set_file_name("kagome_preview.png")
            .add_filter("PNG image", &["png"])
            .add_filter("SVG vector", &["svg"]);
        if let Some(path) = dialog.save_file() {
            self.export_status = match self.canvas2d.export_to_path(&path) {
                Ok(msg) => msg,
                Err(err) => format!("Export error: {err}"),
            };
        }
    }

    fn show_top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(format!("papertilling {}", papertilling_core::VERSION));
                ui.separator();
                for tab in NavTab::ALL {
                    if ui
                        .selectable_label(self.active_tab == tab, tab.label())
                        .clicked()
                    {
                        self.active_tab = tab;
                    }
                }
                ui.separator();
                if ui.button("Reset views").clicked() {
                    self.canvas2d.reset();
                    self.view3d.reset();
                }
            });
        });
    }

    fn show_side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("side_panel")
            .default_size(310.0)
            .min_size(250.0)
            .show(ui, |ui| {
                ui.heading(format!("{} (Stage 0)", self.active_tab.label()));
                ui.label("Plane-group preview table:");
                ui.add_space(4.0);

                ui.push_id("groups_table", |ui| {
                    TableBuilder::new(ui)
                        .striped(true)
                        .vscroll(false)
                        .column(Column::exact(42.0))
                        .column(Column::exact(76.0))
                        .column(Column::exact(36.0))
                        .column(Column::remainder())
                        .header(20.0, |mut header| {
                            for title in ["Grp", "System", "|P|", "Sample"] {
                                header.col(|ui| {
                                    ui.strong(title);
                                });
                            }
                        })
                        .body(|mut body| {
                            for (idx, &(symbol, sys, order, sample)) in
                                SAMPLE_GROUPS.iter().enumerate()
                            {
                                body.row(20.0, |mut row| {
                                    row.col(|ui| {
                                        if ui
                                            .selectable_label(self.selected_group == idx, symbol)
                                            .clicked()
                                        {
                                            self.selected_group = idx;
                                        }
                                    });
                                    row.col(|ui| {
                                        ui.label(sys);
                                    });
                                    row.col(|ui| {
                                        ui.label(order.to_string());
                                    });
                                    row.col(|ui| {
                                        ui.label(sample);
                                    });
                                });
                            }
                        });
                });

                ui.separator();
                ui.strong("GPU / wgpu 30 status");
                ui.label(format!("Adapter: {}", self.adapter_name));
                ui.label(format!("Backend: {}", self.backend_name));
                let (prep, paint) = self.view3d.counters.counts();
                ui.label(format!("3D passes: prepare={prep}, paint={paint}"));
                let (sites, bonds) = self.canvas2d.counts();
                ui.label(format!("2D kagome patch: {sites} sites, {bonds} bonds"));

                ui.separator();
                if ui.button("Export PNG / SVG...").clicked() && !self.smoke_test {
                    self.export_dialog();
                }
                ui.small(&self.export_status);
            });
    }

    fn show_central_split(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            StripBuilder::new(ui)
                .clip(true)
                .sizes(Size::remainder().at_least(160.0), 2)
                .horizontal(|mut strip| {
                    strip.cell(|ui| {
                        ui.strong("2D Lattice Patch (p6m kagome — drag pan, scroll zoom)");
                        ui.add_space(2.0);
                        self.canvas2d.show(ui);
                    });
                    strip.cell(|ui| {
                        ui.strong(
                            "3D Surface Viewport (12x60 icosahedron — drag orbit, scroll zoom)",
                        );
                        ui.add_space(2.0);
                        self.view3d.show(ui);
                    });
                });
        });
    }
}

impl eframe::App for PapertillingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.smoke_test {
            let (prepared, painted) = self.view3d.counters.counts();
            if prepared >= 5 && painted >= 5 {
                if !self.smoke_reported {
                    self.smoke_reported = true;
                    println!(
                        "smoke-test ok: adapter={} backend={} frames={painted}",
                        self.adapter_name, self.backend_name
                    );
                }
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            self.view3d.yaw += 0.08;
            ui.ctx().request_repaint();
        }

        self.show_top_bar(ui);
        self.show_side_panel(ui);
        self.show_central_split(ui);
    }
}
