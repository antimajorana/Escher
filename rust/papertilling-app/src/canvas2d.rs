use std::path::Path;

use crate::export::export_patch_to_path;

pub struct KagomeCanvas {
    pub pan: egui::Vec2,
    pub zoom: f32,
    sites: Vec<[f32; 2]>,
    bonds: Vec<(usize, usize)>,
}

impl Default for KagomeCanvas {
    fn default() -> Self {
        let (sites, bonds) = build_kagome_patch(3);
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 52.0,
            sites,
            bonds,
        }
    }
}

impl KagomeCanvas {
    pub fn reset(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 52.0;
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.sites.len(), self.bonds.len())
    }

    pub fn export_to_path(&self, path: &Path) -> Result<String, String> {
        export_patch_to_path(&self.sites, &self.bonds, path)
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());
        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(18.0, 180.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(248, 249, 252));

        let center = rect.center() + self.pan;
        let to_screen =
            |p: [f32; 2]| egui::pos2(center.x + p[0] * self.zoom, center.y - p[1] * self.zoom);

        let bond_stroke = egui::Stroke::new(1.8, egui::Color32::from_rgb(52, 73, 94));
        for &(i, j) in &self.bonds {
            painter.line_segment(
                [to_screen(self.sites[i]), to_screen(self.sites[j])],
                bond_stroke,
            );
        }

        let radius = (self.zoom * 0.095).clamp(2.5, 7.0);
        let site_fill = egui::Color32::from_rgb(31, 119, 180);
        let site_rim = egui::Stroke::new(1.2, egui::Color32::from_rgb(18, 42, 66));
        for &p in &self.sites {
            let pos = to_screen(p);
            painter.circle_filled(pos, radius, site_fill);
            painter.circle_stroke(pos, radius, site_rim);
        }

        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(180)),
            egui::StrokeKind::Inside,
        );
    }
}

fn build_kagome_patch(radius: i32) -> (Vec<[f32; 2]>, Vec<(usize, usize)>) {
    let s3 = 3.0_f32.sqrt() * 0.5;
    let basis = [[0.5_f32, 0.0_f32], [0.0, 0.5], [0.5, 0.5]];
    let mut sites = Vec::new();
    for u in -radius..=radius {
        for v in -radius..=radius {
            if (u + v).abs() > radius + 1 {
                continue;
            }
            for b in basis {
                let fu = u as f32 + b[0];
                let fv = v as f32 + b[1];
                let x = fu + 0.5 * fv;
                let y = s3 * fv;
                if x * x + y * y <= (radius as f32 + 0.35).powi(2) {
                    sites.push([x, y]);
                }
            }
        }
    }
    let mut bonds = Vec::new();
    for i in 0..sites.len() {
        for j in (i + 1)..sites.len() {
            let dx = sites[i][0] - sites[j][0];
            let dy = sites[i][1] - sites[j][1];
            if (dx * dx + dy * dy - 0.25).abs() < 1e-3 {
                bonds.push((i, j));
            }
        }
    }
    (sites, bonds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kagome_origin_triangle_has_coordination_four() {
        let (sites, bonds) = build_kagome_patch(3);
        assert!(sites.len() > 20 && bonds.len() > 30);
        let mut degree = vec![0_usize; sites.len()];
        for &(i, j) in &bonds {
            degree[i] += 1;
            degree[j] += 1;
        }
        let interior_four = sites
            .iter()
            .zip(&degree)
            .filter(|(p, _)| p[0] * p[0] + p[1] * p[1] < 1.0)
            .all(|(_, &d)| d == 4);
        assert!(interior_four);
    }
}
