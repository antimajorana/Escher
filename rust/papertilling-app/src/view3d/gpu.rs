use core::num::NonZeroU64;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu::{self, util::DeviceExt as _};
use egui::PaintCallbackInfo;

use crate::mesh3d::{GpuMeshData, Vertex3d, back_to_front_indices};

pub const DEPTH_BITS: u8 = 24;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    rot: [[f32; 4]; 4],
    theme: [f32; 4],
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
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shader.wgsl"))),
    });
    let zero_uniforms = Uniforms {
        mvp: [[0.0; 4]; 4],
        rot: [[0.0; 4]; 4],
        theme: [1.0, 0.0, 0.0, 0.0],
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

pub(super) struct View3dCallback {
    pub mvp: [[f32; 4]; 4],
    pub rot: [[f32; 4]; 4],
    pub is_light: bool,
    pub mesh_generation: u64,
    pub mesh_data: Arc<GpuMeshData>,
    pub counters: Arc<GpuFrameCounters>,
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
                theme: [if self.is_light { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
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
