//! Offscreen GPU render of the 3D view for the PNG export: the live view's shader, uniforms and draw sequence
//! (`gpu::build_pipelines`, `gpu::draw_mesh`) into a dedicated colour + depth texture, read back through a staging
//! buffer whose rows are padded to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`. Wires wider than 1 px (exports above the
//! screen's pixel density) are drawn as screen-space quads (`wire_quads`) through the same `fs_wire`.
//!
//! Colour format: the live view draws into egui's swapchain format, which `egui_wgpu::preferred_framebuffer_format`
//! picks non-sRGB (`Bgra8Unorm` / `Rgba8Unorm`), so the shader's output values are stored unencoded. The export
//! therefore renders into `Rgba8Unorm` (or `Rgba8UnormSrgb` if the swapchain is sRGB), the RGBA twin of the
//! swapchain format: same encoding and same blending as on screen, with its own pipelines. Every GPU call runs
//! inside error scopes and a bounded wait, so a validation error, out-of-memory or a lost device returns `Err`.

use std::future::Future;
use std::sync::mpsc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use eframe::egui_wgpu;
use eframe::wgpu;

use super::gpu::{DEPTH_FORMAT, MeshDraw, Uniforms, build_pipelines, draw_mesh, make_index_buffer, make_uniforms, make_vertex_buffer};
use super::wire_quads::wire_quads;
use crate::mesh3d::{GpuMeshData, back_to_front_indices};

const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// Longest wait for the GPU (render, copy and map) before the export gives up with an error.
const GPU_WAIT: Duration = Duration::from_secs(20);

/// GPU handles of the running app, cloned from eframe's `wgpu` render state, for offscreen exports.
#[derive(Clone)]
pub struct OffscreenGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// `max_texture_dimension_2d` of the device (its granted limits, which validation enforces).
    pub max_texture_side: u32,
    /// `max_buffer_size` of the device: the staging buffer (padded rows x height) must fit in it.
    pub max_buffer_size: u64,
    /// Swapchain format of the live view (decides the sRGB-ness of the export texture).
    pub target_format: wgpu::TextureFormat,
}

impl OffscreenGpuContext {
    pub fn from_render_state(rs: &egui_wgpu::RenderState) -> Self {
        Self::new(rs.device.clone(), rs.queue.clone(), rs.target_format)
    }

    pub fn new(device: wgpu::Device, queue: wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let limits = device.limits();
        let (max_texture_side, max_buffer_size) = (limits.max_texture_dimension_2d, limits.max_buffer_size);
        Self { device, queue, max_texture_side, max_buffer_size, target_format }
    }

    /// Colour format of the export texture, [`export_color_format`] of the swapchain format.
    pub fn color_format(&self) -> wgpu::TextureFormat {
        export_color_format(self.target_format)
    }
}

/// RGBA8 format with the encoding of the swapchain format `target`: `Rgba8UnormSrgb` for an sRGB swapchain, else
/// `Rgba8Unorm` (egui's choice on every platform, so the shader output is stored unencoded as on screen).
pub(crate) fn export_color_format(target: wgpu::TextureFormat) -> wgpu::TextureFormat {
    if target.is_srgb() { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm }
}

/// What to draw: the mesh, the live view's matrices and theme flag, and the viewport background (sRGB bytes).
pub(crate) struct OffscreenScene<'a> {
    pub mesh: &'a GpuMeshData,
    pub mvp: [[f32; 4]; 4],
    pub rot: [[f32; 4]; 4],
    pub is_light: bool,
    pub background: [u8; 4],
    /// Wire width in pixels when above 1 px (screen-space quads, [`wire_quads`]); `None`: the live view's 1 px lines.
    pub wire_width: Option<f32>,
}

/// Bytes per row of a `w`-pixel RGBA8 row in a texture-to-buffer copy: `4 w` rounded up to a multiple of 256.
pub(crate) fn padded_bytes_per_row(w: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (4 * w).div_ceil(align) * align
}

/// Tightly packed `w x h` RGBA from rows of `padded_row` bytes (`None` if the buffer is too short).
pub(crate) fn unpad_rows(padded: &[u8], w: u32, h: u32, padded_row: u32) -> Option<Vec<u8>> {
    let (row, stride) = (4 * w as usize, padded_row as usize);
    if stride < row || padded.len() < stride * (h as usize).saturating_sub(1) + row {
        return None;
    }
    let mut out = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&padded[y * stride..y * stride + row]);
    }
    Some(out)
}

/// Poll `fut` to completion with a no-op waker (wgpu's native futures are ready at once); `None` after `GPU_WAIT`.
pub(crate) fn block_on<F: Future>(fut: F) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    let start = Instant::now();
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return Some(v);
        }
        if start.elapsed() > GPU_WAIT {
            return None;
        }
        std::thread::yield_now();
    }
}

/// Clear value of a background byte: `b / 255` for a Unorm target (stored as is, like egui's swapchain), the
/// sRGB-decoded value for an sRGB target (which re-encodes on write).
fn clear_component(b: u8, srgb: bool) -> f64 {
    let c = f64::from(b) / 255.0;
    if !srgb {
        c
    } else if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Render `scene` at `w x h` pixels and return tightly packed RGBA8 (alpha forced opaque).
pub(crate) fn render_rgba(gpu: &OffscreenGpuContext, scene: &OffscreenScene<'_>, w: u32, h: u32) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 || w > gpu.max_texture_side || h > gpu.max_texture_side {
        return Err(format!("3D PNG size {w}x{h} outside the GPU texture limit {}", gpu.max_texture_side));
    }
    if u64::from(padded_bytes_per_row(w)) * u64::from(h) > gpu.max_buffer_size {
        return Err(format!("3D PNG size {w}x{h} exceeds the GPU buffer limit of {} bytes", gpu.max_buffer_size));
    }
    let device = &gpu.device;
    let scopes = [
        device.push_error_scope(wgpu::ErrorFilter::Internal),
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        device.push_error_scope(wgpu::ErrorFilter::Validation),
    ];
    let staged = encode_and_submit(gpu, scene, w, h);
    // Pop every scope, innermost first (Validation, OutOfMemory, Internal), before looking at any result: wgpu
    // panics on an out-of-order pop, which an early return dropping the outer guards first would cause.
    let popped: Vec<Option<Option<wgpu::Error>>> = scopes.into_iter().rev().map(|scope| block_on(scope.pop())).collect();
    for res in popped {
        match res {
            Some(None) => {}
            Some(Some(err)) => return Err(format!("GPU error during the 3D PNG render: {err}")),
            None => return Err(String::from("GPU error scope did not resolve (device lost?)")),
        }
    }
    let (buffer, padded_row) = staged?;
    read_back(gpu, &buffer, w, h, padded_row)
}

/// Record and submit the render pass and the texture-to-buffer copy; returns the staging buffer and its row pitch.
fn encode_and_submit(gpu: &OffscreenGpuContext, scene: &OffscreenScene<'_>, w: u32, h: u32) -> Result<(wgpu::Buffer, u32), String> {
    let (device, mesh) = (&gpu.device, scene.mesh);
    let format = gpu.color_format();
    let pipes = build_pipelines(device, format, scene.wire_width);
    let (_ubuf, bind_group) = make_uniforms(device, &pipes.bind_layout, &Uniforms::new(scene.mvp, scene.rot, scene.is_light));
    let (face_vb, face_count) = make_vertex_buffer(device, "export_face_vb", &mesh.faces);
    // wires wider than 1 px: screen-space quads already in NDC, drawn with an identity MVP (same depth, same fs_wire)
    let quads = scene.wire_width.map(|width| wire_quads(&mesh.wires, &scene.mvp, [w, h], width));
    let (wire_vb, wire_count) = make_vertex_buffer(device, "export_wire_vb", quads.as_deref().unwrap_or(&mesh.wires));
    let (_ubuf_ndc, ndc_group) = make_uniforms(device, &pipes.bind_layout, &Uniforms::new(IDENTITY, scene.rot, scene.is_light));
    let opaque = mesh.opaque_vertex_count.min(mesh.faces.len());
    let order = back_to_front_indices(&mesh.faces, opaque, &scene.rot);
    let sorted_ib = make_index_buffer(device, order.len());
    if !order.is_empty() {
        gpu.queue.write_buffer(&sorted_ib, 0, bytemuck::cast_slice(&order));
    }
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let texture = |label, format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture("export3d_color", format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = texture("export3d_depth", DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let (color_view, depth_view) = (color.create_view(&Default::default()), depth.create_view(&Default::default()));
    let padded_row = padded_bytes_per_row(w);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export3d_staging"),
        size: u64::from(padded_row) * u64::from(h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let srgb = format.is_srgb();
    let bg = scene.background;
    let clear = wgpu::Color {
        r: clear_component(bg[0], srgb),
        g: clear_component(bg[1], srgb),
        b: clear_component(bg[2], srgb),
        a: 1.0,
    };
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("export3d") });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("export3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let draw = MeshDraw {
            face_vb: &face_vb,
            face_count,
            opaque_count: opaque as u32,
            sorted_ib: &sorted_ib,
            sorted_count: order.len() as u32,
            wire_vb: &wire_vb,
            wire_count: if quads.is_some() { 0 } else { wire_count },
        };
        draw_mesh(&mut pass, (&pipes.face, &pipes.wire), &bind_group, &draw);
        if let Some(quad_pipe) = pipes.wire_quads.as_ref().filter(|_| quads.is_some() && wire_count > 0) {
            pass.set_pipeline(quad_pipe);
            pass.set_bind_group(0, &ndc_group, &[]);
            pass.set_vertex_buffer(0, wire_vb.slice(..));
            pass.draw(0..wire_count, 0..1);
        }
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(h) },
        },
        size,
    );
    gpu.queue.submit([encoder.finish()]);
    Ok((buffer, padded_row))
}

/// Map the staging buffer (bounded wait), unpad its rows and force alpha opaque.
fn read_back(gpu: &OffscreenGpuContext, buffer: &wgpu::Buffer, w: u32, h: u32, padded_row: u32) -> Result<Vec<u8>, String> {
    let (tx, rx) = mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(GPU_WAIT) })
        .map_err(|e| format!("GPU wait failed during the 3D PNG readback: {e}"))?;
    match rx.recv_timeout(Duration::from_millis(100)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("3D PNG readback map failed: {e}")),
        Err(_) => return Err(String::from("3D PNG readback did not complete (device lost?)")),
    }
    let rgba = match buffer.slice(..).get_mapped_range() {
        Ok(view) => unpad_rows(&view, w, h, padded_row),
        Err(e) => return Err(format!("3D PNG readback range unavailable: {e}")),
    };
    buffer.unmap();
    let mut rgba = rgba.ok_or_else(|| String::from("3D PNG readback buffer too short"))?;
    for px in rgba.as_chunks_mut::<4>().0 {
        px[3] = 255;
    }
    Ok(rgba)
}

#[cfg(test)]
#[path = "offscreen_tests.rs"]
mod tests;
