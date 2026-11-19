use crate::app::PapertillingApp;

impl PapertillingApp {
    pub(crate) fn show_status_bar(&mut self, ui: &mut egui::Ui) {
        let resp = egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(&self.export_status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (sites, bonds) = self.canvas2d.counts();
                    let (prep, paint) = self.view3d.counters.counts();
                    ui.small(format!(
                        "2D: {sites} sites, {bonds} lines | GPU ({}) prep={prep} paint={paint} gen={}",
                        self.backend_name,
                        self.view3d.mesh_generation()
                    ));
                });
            });
        });
        self.deck.host_rect.max.y = resp.response.rect.top();
        #[cfg(test)]
        {
            self.test_rects.status = resp.response.rect;
        }
    }
}
