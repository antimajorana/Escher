use core::num::NonZeroU64;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu::{self, util::DeviceExt as _};
use egui::PaintCallbackInfo;

use crate::mesh3d::{GpuMeshData, Vertex3d, back_to_front_indices};

pub const DEPTH_BITS: u8 = 24;
/// Depth format of `DEPTH_BITS` without stencil, as `egui_wgpu::depth_format_from_bits(24, 0)` gives eframe's
/// depth buffer (checked by a test).
pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Uniforms {
    mvp: [[f32; 4]; 4],
    rot: [[f32; 4]; 4],
    theme: [f32; 4],
}

impl Uniforms {
    /// Uniforms of `shader.wgsl`: clip transform, view rotation (normals, lighting) and the theme flag of `fs_wire`.
    pub(super) fn new(mvp: [[f32; 4]; 4], rot: [[f32; 4]; 4], is_light: bool) -> Self {
        Self { mvp, rot, theme: [if is_light { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0] }
    }
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

pub(super) fn make_index_buffer(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("face_sorted_ib"),
        size: (4 * count.max(1)) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(super) fn make_vertex_buffer(device: &wgpu::Device, label: &str, data: &[Vertex3d]) -> (wgpu::Buffer, u32) {
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

/// Face and wire pipelines of `shader.wgsl` writing to `color_format` (24-bit depth, alpha blending, no MSAA), and
/// their uniform bind group layout. The live view builds them for the swapchain format, the offscreen PNG export for
/// its own texture format, so neither ever draws through the other's pipelines.
pub(super) struct View3dPipelines {
    pub face: wgpu::RenderPipeline,
    pub wire: wgpu::RenderPipeline,
    /// The wires as screen-space quads (`TriangleList` through `fs_wire`, `wire_quads`), built for the export only.
    pub wire_quads: Option<wgpu::RenderPipeline>,
    pub bind_layout: wgpu::BindGroupLayout,
}

/// `wire_quads`: the export's wire width in pixels (`Some`: also build the quad pipeline and scale the faces' slope
/// bias with it); `None` for the live view and 1 px exports.
pub(super) fn build_pipelines(device: &wgpu::Device, color_format: wgpu::TextureFormat, wire_quads: Option<f32>) -> View3dPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view3d_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shader.wgsl"))),
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view3d_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
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
                format: DEPTH_FORMAT,
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
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    // faces sit 2 px of depth slope behind the wires; with `w` px wide wire quads every pixel length is w times the
    // screen's, so the slope bias is too (a quad carries its centre line's depth up to w / 2 px across)
    let face_bias = wgpu::DepthBiasState {
        constant: 2,
        slope_scale: 2.0 * wire_quads.unwrap_or(1.0).max(1.0),
        clamp: 0.0,
    };
    let face = make_pipeline("faces", wgpu::PrimitiveTopology::TriangleList, "fs_face", true, face_bias);
    let wire = make_pipeline("wires", wgpu::PrimitiveTopology::LineList, "fs_wire", false, Default::default());
    let wire_quads =
        wire_quads.map(|_| make_pipeline("wire_quads", wgpu::PrimitiveTopology::TriangleList, "fs_wire", false, Default::default()));
    View3dPipelines { face, wire, wire_quads, bind_layout }
}

/// Uniform buffer (initialised to `uniforms`) and its bind group for `layout`.
pub(super) fn make_uniforms(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &Uniforms,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("view3d_uniforms"),
        contents: bytemuck::bytes_of(uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view3d_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

/// Uploaded mesh buffers of one draw: `face_vb[..opaque_count]` opaque markers, `sorted_ib` the translucent faces
/// back to front, then the wires.
pub(super) struct MeshDraw<'a> {
    pub face_vb: &'a wgpu::Buffer,
    pub face_count: u32,
    pub opaque_count: u32,
    pub sorted_ib: &'a wgpu::Buffer,
    pub sorted_count: u32,
    pub wire_vb: &'a wgpu::Buffer,
    pub wire_count: u32,
}

/// Draw sequence of the 3D view, shared by the live view and the offscreen export: opaque markers first, then the
/// translucent faces back to front (both write depth, so the wires drawn next are hidden behind the nearest face),
/// then the wires.
pub(super) fn draw_mesh(
    pass: &mut wgpu::RenderPass<'_>,
    pipes: (&wgpu::RenderPipeline, &wgpu::RenderPipeline),
    bind_group: &wgpu::BindGroup,
    m: &MeshDraw<'_>,
) {
    pass.set_bind_group(0, bind_group, &[]);
    if m.face_count > 0 {
        pass.set_pipeline(pipes.0);
        pass.set_vertex_buffer(0, m.face_vb.slice(..));
        if m.opaque_count > 0 {
            pass.draw(0..m.opaque_count, 0..1);
        }
        if m.sorted_count > 0 {
            pass.set_index_buffer(m.sorted_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.sorted_count, 0, 0..1);
        }
    }
    if m.wire_count > 0 {
        pass.set_pipeline(pipes.1);
        pass.set_vertex_buffer(0, m.wire_vb.slice(..));
        pass.draw(0..m.wire_count, 0..1);
    }
}

pub fn init_view3d_resources(render_state: &egui_wgpu::RenderState) {
    let device = &render_state.device;
    let pipes = build_pipelines(device, render_state.target_format, None);
    let (uniform_buffer, bind_group) =
        make_uniforms(device, &pipes.bind_layout, &Uniforms::new([[0.0; 4]; 4], [[0.0; 4]; 4], true));
    let (face_pipeline, wire_pipeline) = (pipes.face, pipes.wire);

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
            let uniforms = Uniforms::new(self.mvp, self.rot, self.is_light);
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
        let draw = MeshDraw {
            face_vb: &res.face_vb,
            face_count: res.face_count,
            opaque_count: res.opaque_count,
            sorted_ib: &res.sorted_ib,
            sorted_count: res.sorted_count,
            wire_vb: &res.wire_vb,
            wire_count: res.wire_count,
        };
        draw_mesh(render_pass, (&res.face_pipeline, &res.wire_pipeline), &res.bind_group, &draw);
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
