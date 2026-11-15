use core::num::NonZeroU64;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu::{self, util::DeviceExt as _};
use egui::PaintCallbackInfo;

use crate::mesh3d::{Vertex3d, build_icosahedron_mesh, build_mvp, orbit_rotation};

pub const DEPTH_BITS: u8 = 24;

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
}

impl GpuFrameCounters {
    pub fn counts(&self) -> (u32, u32) {
        (
            self.prepared.load(Ordering::Relaxed),
            self.painted.load(Ordering::Relaxed),
        )
    }
}

pub struct View3dState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub counters: Arc<GpuFrameCounters>,
}

impl Default for View3dState {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: 0.35,
            distance: 3.4,
            counters: Arc::new(GpuFrameCounters::default()),
        }
    }
}

impl View3dState {
    pub fn reset(&mut self) {
        (self.yaw, self.pitch, self.distance) = (0.55, 0.35, 3.4);
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());
        if response.dragged() {
            let delta = response.drag_delta();
            self.yaw += delta.x * 0.01;
            self.pitch = (self.pitch + delta.y * 0.01).clamp(-1.45, 1.45);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.distance = (self.distance * (-scroll * 0.002).exp()).clamp(1.8, 8.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 24, 31));
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
        let rot = orbit_rotation(self.yaw, self.pitch);
        let mvp = build_mvp(aspect, self.distance, &rot);
        let counters = Arc::clone(&self.counters);
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            View3dCallback { mvp, rot, counters },
        ));
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(64)),
            egui::StrokeKind::Inside,
        );
    }
}

struct View3dGpuResources {
    face_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    face_vb: wgpu::Buffer,
    face_count: u32,
    wire_vb: wgpu::Buffer,
    wire_count: u32,
}

pub fn init_view3d_resources(render_state: &egui_wgpu::RenderState) {
    let device = &render_state.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view3d_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    let zero_uniforms = Uniforms { mvp: [[0.0; 4]; 4], rot: [[0.0; 4]; 4] };
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
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
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
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3],
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
            primitive: wgpu::PrimitiveState { topology, cull_mode: None, ..Default::default() },
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    let face_bias = wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 };
    let face_pipeline = make_pipeline("faces", wgpu::PrimitiveTopology::TriangleList, "fs_face", true, face_bias);
    let wire_pipeline = make_pipeline("wires", wgpu::PrimitiveTopology::LineList, "fs_wire", false, Default::default());

    let (faces, wires) = build_icosahedron_mesh();
    let mk_vb = |label, data: &[Vertex3d]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    };
    let (face_vb, wire_vb) = (mk_vb("face_vb", &faces), mk_vb("wire_vb", &wires));
    render_state.renderer.write().callback_resources.insert(View3dGpuResources {
        face_pipeline,
        wire_pipeline,
        uniform_buffer,
        bind_group,
        face_vb,
        face_count: faces.len() as u32,
        wire_vb,
        wire_count: wires.len() as u32,
    });
}

struct View3dCallback {
    mvp: [[f32; 4]; 4],
    rot: [[f32; 4]; 4],
    counters: Arc<GpuFrameCounters>,
}

impl CallbackTrait for View3dCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get::<View3dGpuResources>() {
            let uniforms = Uniforms { mvp: self.mvp, rot: self.rot };
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
        let Some(res) = resources.get::<View3dGpuResources>() else { return };
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_pipeline(&res.face_pipeline);
        render_pass.set_vertex_buffer(0, res.face_vb.slice(..));
        render_pass.draw(0..res.face_count, 0..1);
        render_pass.set_pipeline(&res.wire_pipeline);
        render_pass.set_vertex_buffer(0, res.wire_vb.slice(..));
        render_pass.draw(0..res.wire_count, 0..1);
        self.counters.painted.fetch_add(1, Ordering::Relaxed);
    }
}
