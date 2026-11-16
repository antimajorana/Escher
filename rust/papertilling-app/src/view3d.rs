use core::num::NonZeroU64;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu::{self, util::DeviceExt as _};
use egui::PaintCallbackInfo;

use crate::canvas2d::{ZOOM_STEP_FACTOR, ZoomKeyAction, input_zoom_factor, take_keyboard_zoom_action};
use crate::mesh3d::{GpuMeshData, Vertex3d, back_to_front_indices, build_mvp, orbit_rotation};

pub const DEPTH_BITS: u8 = 24;
pub const DIST_MIN: f32 = 0.45;
pub const DIST_MAX: f32 = 24.0;
pub const DIST_DEFAULT: f32 = 3.1;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    rot: [[f32; 4]; 4],
}

#[derive(Default)]
pub struct GpuFrameCounters {
    prepared: AtomicU32,
    painted: AtomicU32,
    uploaded: AtomicU32,
    painted_generation: AtomicU64,
    painted_primitives: AtomicU32,
}

impl GpuFrameCounters {
    pub fn counts(&self) -> (u32, u32) {
        (
            self.prepared.load(Ordering::Relaxed),
            self.painted.load(Ordering::Relaxed),
        )
    }

    pub fn uploads(&self) -> u32 {
        self.uploaded.load(Ordering::Relaxed)
    }

    pub fn painted_generation(&self) -> u64 {
        self.painted_generation.load(Ordering::Relaxed)
    }

    pub fn painted_primitives(&self) -> u32 {
        self.painted_primitives.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn record_mock_paint(&self, generation: u64, primitives: u32) {
        self.prepared.fetch_add(1, Ordering::Relaxed);
        self.painted_generation.store(generation, Ordering::Relaxed);
        self.painted_primitives.store(primitives, Ordering::Relaxed);
        self.painted.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct View3dState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Translation in camera x/y coordinates; rendering only, never part of the mesh or its identity.
    camera_offset: egui::Vec2,
    pub counters: Arc<GpuFrameCounters>,
    mesh_generation: u64,
    mesh_data: Arc<GpuMeshData>,
    overlay_caption: String,
}

impl Default for View3dState {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: 0.35,
            distance: 3.1,
            camera_offset: egui::Vec2::ZERO,
            counters: Arc::new(GpuFrameCounters::default()),
            mesh_generation: 1,
            mesh_data: Arc::new(GpuMeshData::default()),
            overlay_caption: String::new(),
        }
    }
}

impl View3dState {
    pub fn reset(&mut self) {
        (self.yaw, self.pitch, self.distance) = (0.55, 0.35, DIST_DEFAULT);
        self.camera_offset = egui::Vec2::ZERO;
    }

    pub fn zoom_by(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = (self.distance / factor).clamp(DIST_MIN, DIST_MAX);
        }
    }

    /// Dolly along the cursor ray, preserving every visible point on that ray (at any depth).
    /// No picking, geometry changes, or camera FOV convention changes are needed.
    pub fn zoom_at(&mut self, rect: egui::Rect, factor: f32, anchor: Option<egui::Pos2>) {
        let old_distance = self.distance;
        self.zoom_by(factor);
        if let Some(anchor) = anchor.filter(|p| rect.contains(*p))
            && rect.width() > 0.0 && rect.height() > 0.0
        {
            let aspect = (rect.width() / rect.height()).max(0.1);
            let projection = build_mvp(aspect, old_distance, &orbit_rotation(0.0, 0.0));
            let ndc = egui::vec2(
                2.0 * (anchor.x - rect.center().x) / rect.width(),
                2.0 * (rect.center().y - anchor.y) / rect.height(),
            );
            let delta = self.distance - old_distance; // Use the actual clamped dolly distance.
            self.camera_offset += egui::vec2(ndc.x / projection[0][0], ndc.y / projection[1][1]) * delta;
        }
    }

    fn view_mvp(&self, aspect: f32, rotation: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut translated = *rotation;
        translated[3][0] = self.camera_offset.x;
        translated[3][1] = self.camera_offset.y;
        build_mvp(aspect.max(0.1), self.distance, &translated)
    }

    pub fn set_mesh(&mut self, mesh: GpuMeshData, caption: impl Into<String>) {
        self.mesh_generation = self.mesh_generation.wrapping_add(1);
        self.mesh_data = Arc::new(mesh);
        self.overlay_caption = caption.into();
    }

    pub fn mesh_generation(&self) -> u64 {
        self.mesh_generation
    }

    pub fn has_geometry(&self) -> bool {
        !self.mesh_data.faces.is_empty() || !self.mesh_data.wires.is_empty()
    }

    pub fn is_current_mesh_painted(&self) -> bool {
        self.counters.painted_generation() == self.mesh_generation
            && self.counters.painted_primitives() > 0
    }

    #[cfg(test)]
    pub fn mesh_data(&self) -> &GpuMeshData {
        &self.mesh_data
    }

    pub fn caption(&self) -> &str {
        &self.overlay_caption
    }

    /// Orbit by `(dyaw, dpitch)` radians: yaw wraps to (-pi, pi], pitch stops short of the poles.
    pub fn orbit_by(&mut self, dyaw: f32, dpitch: f32) {
        let yaw = (self.yaw + dyaw).rem_euclid(core::f32::consts::TAU);
        self.yaw = if yaw > core::f32::consts::PI { yaw - core::f32::consts::TAU } else { yaw };
        self.pitch = (self.pitch + dpitch).clamp(-1.50, 1.50);
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let has_hud = rect.width() >= 190.0 && rect.height() >= 80.0;
        let hud_rect = egui::Rect::from_min_max(
            rect.right_bottom() - egui::vec2(164.0, 30.0),
            rect.right_bottom() - egui::vec2(6.0, 6.0),
        );
        let ptr = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
        let press_orig = ui.input(|i| i.pointer.press_origin());
        let over_hud = has_hud && ptr.is_some_and(|p| hud_rect.contains(p));
        let drag_from_hud = has_hud && press_orig.is_some_and(|p| hud_rect.contains(p));

        if !drag_from_hud && response.dragged() {
            let delta = response.drag_delta();
            self.orbit_by(delta.x * 0.01, delta.y * 0.01);
        }
        if !over_hud && !drag_from_hud && response.double_clicked() {
            self.reset();
        }
        if ui.rect_contains_pointer(rect) {
            let anchor = ptr.filter(|p| rect.contains(*p));
            let factor = ui.input(input_zoom_factor);
            if (factor - 1.0).abs() > 1e-5 {
                self.zoom_at(rect, factor, anchor);
            }
            if let Some(act) = take_keyboard_zoom_action(ui, &response) {
                match act {
                    ZoomKeyAction::In => self.zoom_at(rect, ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Out => self.zoom_at(rect, 1.0 / ZOOM_STEP_FACTOR, anchor),
                    ZoomKeyAction::Fit => self.reset(),
                }
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(18, 22, 30));
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
        let rot = orbit_rotation(self.yaw, self.pitch);
        let mvp = self.view_mvp(aspect, &rot);
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            View3dCallback {
                mvp,
                rot,
                mesh_generation: self.mesh_generation,
                mesh_data: Arc::clone(&self.mesh_data),
                counters: Arc::clone(&self.counters),
            },
        ));
        if !self.overlay_caption.is_empty() {
            painter.text(
                rect.left_top() + egui::vec2(8.0, 6.0),
                egui::Align2::LEFT_TOP,
                &self.overlay_caption,
                egui::FontId::proportional(12.0),
                egui::Color32::from_rgb(218, 226, 238),
            );
        }
        painter.text(
            rect.left_bottom() + egui::vec2(8.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            "Drag: orbit | Scroll/pinch/+/-: zoom | 0/dbl-click: fit",
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(148, 160, 178),
        );
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(64)),
            egui::StrokeKind::Inside,
        );
        if has_hud {
            ui.scope_builder(egui::UiBuilder::new().max_rect(hud_rect), |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(28, 34, 46, 224))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(86)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if ui.small_button("−").on_hover_text("Zoom out (-)").clicked() {
                                self.zoom_by(1.0 / ZOOM_STEP_FACTOR);
                            }
                            let pct = ((DIST_DEFAULT / self.distance.max(1e-3)) * 100.0).round() as i32;
                            ui.colored_label(egui::Color32::from_rgb(214, 224, 238), egui::RichText::new(format!("{pct}%")).size(11.0));
                            if ui.small_button("+").on_hover_text("Zoom in (+)").clicked() {
                                self.zoom_by(ZOOM_STEP_FACTOR);
                            }
                            if ui.small_button("Fit").on_hover_text("Reset 3D view (0 / double-click)").clicked() {
                                self.reset();
                            }
                        });
                    });
            });
        }
    }
}

struct View3dGpuResources {
    face_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uploaded_generation: u64,
    face_vb: wgpu::Buffer,
    face_count: u32,
    opaque_count: u32,
    sorted_ib: wgpu::Buffer,
    sorted_count: u32,
    sorted_rot: Option<[[f32; 4]; 4]>,
    wire_vb: wgpu::Buffer,
    wire_count: u32,
}

fn make_index_buffer(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("face_sorted_ib"),
        size: (4 * count.max(1)) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_vertex_buffer(device: &wgpu::Device, label: &str, data: &[Vertex3d]) -> (wgpu::Buffer, u32) {
    let slice = if data.is_empty() {
        &[Vertex3d::ZERO][..]
    } else {
        data
    };
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(slice),
        usage: wgpu::BufferUsages::VERTEX,
    });
    (buf, data.len() as u32)
}

pub fn init_view3d_resources(render_state: &egui_wgpu::RenderState) {
    let device = &render_state.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view3d_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    let zero_uniforms = Uniforms {
        mvp: [[0.0; 4]; 4],
        rot: [[0.0; 4]; 4],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("view3d_uniforms"),
        contents: bytemuck::bytes_of(&zero_uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("view3d_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(core::mem::size_of::<Uniforms>() as u64),
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view3d_bg"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view3d_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let depth_format = egui_wgpu::depth_format_from_bits(DEPTH_BITS, 0)
        .expect("24-bit depth format is always supported");
    let vb_layout = wgpu::VertexBufferLayout {
        array_stride: core::mem::size_of::<Vertex3d>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
    };
    let make_pipeline = |label, topology, fs_entry, depth_write, bias| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vb_layout.clone())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias,
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    let face_bias = wgpu::DepthBiasState {
        constant: 2,
        slope_scale: 2.0,
        clamp: 0.0,
    };
    let face_pipeline = make_pipeline(
        "faces",
        wgpu::PrimitiveTopology::TriangleList,
        "fs_face",
        true,
        face_bias,
    );
    let wire_pipeline = make_pipeline(
        "wires",
        wgpu::PrimitiveTopology::LineList,
        "fs_wire",
        false,
        Default::default(),
    );

    let (face_vb, face_count) = make_vertex_buffer(device, "face_vb", &[]);
    let (wire_vb, wire_count) = make_vertex_buffer(device, "wire_vb", &[]);
    let sorted_ib = make_index_buffer(device, 0);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(View3dGpuResources {
            face_pipeline,
            wire_pipeline,
            uniform_buffer,
            bind_group,
            uploaded_generation: 0,
            face_vb,
            face_count,
            opaque_count: 0,
            sorted_ib,
            sorted_count: 0,
            sorted_rot: None,
            wire_vb,
            wire_count,
        });
}

struct View3dCallback {
    mvp: [[f32; 4]; 4],
    rot: [[f32; 4]; 4],
    mesh_generation: u64,
    mesh_data: Arc<GpuMeshData>,
    counters: Arc<GpuFrameCounters>,
}

impl CallbackTrait for View3dCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<View3dGpuResources>() {
            let mesh = &self.mesh_data;
            if res.uploaded_generation != self.mesh_generation {
                let (fvb, fc) = make_vertex_buffer(device, "face_vb_dyn", &mesh.faces);
                let (wvb, wc) = make_vertex_buffer(device, "wire_vb_dyn", &mesh.wires);
                let opaque = mesh.opaque_vertex_count.min(mesh.faces.len());
                res.face_vb = fvb;
                res.face_count = fc;
                res.opaque_count = opaque as u32;
                res.sorted_ib = make_index_buffer(device, mesh.faces.len() - opaque);
                res.sorted_count = 0;
                res.sorted_rot = None;
                res.wire_vb = wvb;
                res.wire_count = wc;
                res.uploaded_generation = self.mesh_generation;
                self.counters.uploaded.fetch_add(1, Ordering::Relaxed);
            }
            if res.sorted_rot != Some(self.rot) {
                let order = back_to_front_indices(&mesh.faces, res.opaque_count as usize, &self.rot);
                if !order.is_empty() {
                    queue.write_buffer(&res.sorted_ib, 0, bytemuck::cast_slice(&order));
                }
                res.sorted_count = order.len() as u32;
                res.sorted_rot = Some(self.rot);
            }
            let uniforms = Uniforms {
                mvp: self.mvp,
                rot: self.rot,
            };
            queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            self.counters.prepared.fetch_add(1, Ordering::Relaxed);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(res) = resources.get::<View3dGpuResources>() else {
            return;
        };
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        if res.face_count > 0 {
            render_pass.set_pipeline(&res.face_pipeline);
            render_pass.set_vertex_buffer(0, res.face_vb.slice(..));
            // opaque markers first, then the translucent faces back to front (both write depth, so the wires
            // drawn next are hidden behind the nearest face)
            if res.opaque_count > 0 {
                render_pass.draw(0..res.opaque_count, 0..1);
            }
            if res.sorted_count > 0 {
                render_pass.set_index_buffer(res.sorted_ib.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..res.sorted_count, 0, 0..1);
            }
        }
        if res.wire_count > 0 {
            render_pass.set_pipeline(&res.wire_pipeline);
            render_pass.set_vertex_buffer(0, res.wire_vb.slice(..));
            render_pass.draw(0..res.wire_count, 0..1);
        }
        if res.uploaded_generation == self.mesh_generation {
            self.counters
                .painted_generation
                .store(self.mesh_generation, Ordering::Relaxed);
            self.counters
                .painted_primitives
                .store(res.face_count.saturating_add(res.wire_count), Ordering::Relaxed);
        }
        self.counters.painted.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::View3dState;
    use crate::mesh3d::GpuMeshData;

    #[test]
    fn orbit_wraps_yaw_clamps_pitch_and_reset_restores_the_default_view() {
        let mut v = View3dState::default();
        let start = (v.yaw, v.pitch, v.distance);
        for _ in 0..10_000 {
            v.orbit_by(0.37, 0.05);
        }
        assert!(v.yaw.abs() <= core::f32::consts::PI + 1e-6, "yaw {} not wrapped", v.yaw);
        assert_eq!(v.pitch, 1.50);
        v.orbit_by(0.0, -10.0);
        assert_eq!(v.pitch, -1.50);
        // wrapping keeps the angle: 10 000 x 0.37 rad
        let want = (start.0 + 10_000.0 * 0.37_f32).rem_euclid(core::f32::consts::TAU);
        let got = v.yaw.rem_euclid(core::f32::consts::TAU);
        assert!((want - got).abs() < 1e-2 || (want - got).abs() > core::f32::consts::TAU - 1e-2);
        v.distance = 7.0;
        v.reset();
        assert_eq!((v.yaw, v.pitch, v.distance), start);
    }

    #[test]
    fn set_mesh_bumps_the_generation_so_the_gpu_buffers_are_re_uploaded() {
        let mut v = View3dState::default();
        let g0 = v.mesh_generation();
        v.set_mesh(GpuMeshData::default(), "a");
        v.set_mesh(GpuMeshData::default(), "b");
        assert_eq!(v.mesh_generation(), g0 + 2);
        assert_eq!(v.caption(), "b");
    }
}

#[cfg(test)]
mod interaction_tests;
