use eframe::egui_wgpu;
use eframe::wgpu;

use super::super::View3dState;
use super::super::camera::CameraPreset;
use super::super::gpu::DEPTH_FORMAT;
use super::{OffscreenGpuContext, OffscreenScene, block_on, export_color_format, padded_bytes_per_row, render_rgba, unpad_rows};
use crate::mesh3d::{GpuMeshData, Vertex3d};
use crate::theme::ThemeTokens;

#[test]
fn padded_row_is_four_bytes_a_pixel_rounded_up_to_256() {
    // 64 px = 256 B exactly; 65 px = 260 B -> next multiple of 256; 37 px = 148 B -> 256
    assert_eq!(padded_bytes_per_row(64), 256);
    assert_eq!(padded_bytes_per_row(65), 512);
    assert_eq!(padded_bytes_per_row(37), 256);
    assert_eq!(padded_bytes_per_row(1), 256);
}

#[test]
fn unpad_recovers_the_exact_bytes_of_a_37x3_image() {
    let (w, h, stride) = (37_u32, 3_u32, 256_usize);
    // pixel byte i of row y is (y * 148 + i) mod 251; padding is 0xEE, which never reaches the output
    let mut padded = vec![0xEE_u8; stride * h as usize];
    let mut want = Vec::new();
    for y in 0..h as usize {
        for i in 0..148 {
            let b = ((y * 148 + i) % 251) as u8;
            padded[y * stride + i] = b;
            want.push(b);
        }
    }
    assert_eq!(unpad_rows(&padded, w, h, 256), Some(want));
    // the last row needs only its 148 pixel bytes, not its padding
    assert!(unpad_rows(&padded[..2 * stride + 148], w, h, 256).is_some());
    assert_eq!(unpad_rows(&padded[..2 * stride + 147], w, h, 256), None);
    assert_eq!(unpad_rows(&padded, w, h, 100), None, "a pitch below 4 w is refused");
}

#[test]
fn export_formats_follow_the_swapchain_encoding_and_eframe_depth() {
    assert_eq!(export_color_format(wgpu::TextureFormat::Bgra8Unorm), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(export_color_format(wgpu::TextureFormat::Rgba8Unorm), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(export_color_format(wgpu::TextureFormat::Bgra8UnormSrgb), wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(egui_wgpu::depth_format_from_bits(super::super::DEPTH_BITS, 0), Some(DEPTH_FORMAT));
}

/// A low-power adapter of this process, or `None` (with a note) when no GPU / driver is available.
fn test_gpu() -> Option<OffscreenGpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let opts = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::LowPower, ..Default::default() };
    let Some(Ok(adapter)) = block_on(instance.request_adapter(&opts)) else {
        eprintln!("note: no wgpu adapter in the test process; GPU offscreen test skipped");
        return None;
    };
    let Some(Ok((device, queue))) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) else {
        eprintln!("note: wgpu device request failed; GPU offscreen test skipped");
        return None;
    };
    // egui's swapchain choice (non-sRGB), so the export texture is Rgba8Unorm as in the app
    Some(OffscreenGpuContext::new(device, queue, wgpu::TextureFormat::Bgra8Unorm))
}

fn vertex(x: f32, y: f32) -> Vertex3d {
    Vertex3d { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0], color: [0.8, 0.4, 0.2, 1.0] }
}

fn px(rgba: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = 4 * (y * w + x) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

#[test]
fn gpu_renders_a_flat_triangle_with_the_shader_lighting_over_the_background() {
    let t0 = std::time::Instant::now();
    let Some(gpu) = test_gpu() else { return };
    // Top preset: view rotation = identity, camera on +z at distance 3.1 looking at the origin, so the triangle
    // (z = 0, normal +z, opaque) faces the camera. Visible half-height at the origin depth: 3.1 tan(22.5 deg) = 1.284,
    // so the triangle (|x| <= 0.6, -0.5 <= y <= 0.7) covers the centre and stays clear of every corner.
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Top);
    let mesh = GpuMeshData {
        faces: vec![vertex(-0.6, -0.5), vertex(0.6, -0.5), vertex(0.0, 0.7)],
        opaque_vertex_count: 3,
        ..Default::default()
    };
    // 50 x 40: 200 bytes a row, padded to 256 in the readback
    let (w, h) = (50_u32, 40_u32);
    let rot = view.rotation();
    let bg = [244, 246, 250, 255];
    let scene = OffscreenScene { mesh: &mesh, mvp: view.view_mvp(w as f32 / h as f32, &rot), rot, is_light: true, background: bg, wire_width: None };
    let rgba = render_rgba(&gpu, &scene, w, h).expect("offscreen render");
    assert_eq!(rgba.len(), (4 * w * h) as usize);
    // fs_face at n = (0, 0, 1) (view space, R = I): key = normalize(0.36, 0.56, 0.74) -> d_key = 0.74 / 0.995389
    // = 0.743428; fill = normalize(-0.52, -0.28, 0.48) -> d_fill = 0.48 / 0.761052 = 0.630706;
    // shade = 0.24 + 0.62 d_key + 0.14 d_fill = 0.789224; half = normalize(key + z) has z = 1.743428 / 1.867312
    // = 0.933656, spec = 0.12 * 0.933656^20 = 0.030403. rgb = c shade + spec for c = (0.8, 0.4, 0.2):
    // (0.661782, 0.346093, 0.188248) -> x 255 = (168.75, 88.25, 48.00) -> bytes (169, 88, 48) in Rgba8Unorm.
    // Tolerance 2: the unorm conversion may round either way (+-0.6 ulp) and pow/normalize differ by ~1e-6.
    let want = [169_i32, 88, 48];
    let c = px(&rgba, w, w / 2, h / 2);
    eprintln!("centre pixel {c:?}, adapter limit {}", gpu.max_texture_side);
    for k in 0..3 {
        assert!((i32::from(c[k]) - want[k]).abs() <= 2, "centre {c:?} vs shaded {want:?}");
    }
    assert_eq!(c[3], 255);
    // the clear colour bg / 255 is stored unencoded and converts back to the same byte
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
        assert_eq!(px(&rgba, w, x, y), bg, "corner ({x}, {y})");
    }
    // orientation (row 0 = top): pixel centre (31.5, 26.5) is NDC (0.26, -0.325) -> view (x, y) = (0.26 / (f / 1.25),
    // -0.325 / f) 3.1 = (0.417, -0.417) with f = 1 / tan(22.5 deg) = 2.41421: inside the triangle (half-width
    // 0.6 (0.7 - y) / 1.2 = 0.541 there). Its mirror row 13 is y = +0.417, half-width 0.141 < 0.417: background.
    let on = px(&rgba, w, 31, 26);
    // flat triangle: same shade up to 1 unit of interpolation rounding
    assert!((0..3).all(|k| on[k].abs_diff(c[k]) <= 1), "lower right of the centre is on the triangle: {on:?}");
    assert_eq!(px(&rgba, w, 31, 13), bg, "upper right of the centre is off the triangle");
    eprintln!("# time gpu triangle {:?}", t0.elapsed());
}

#[test]
fn gpu_export_writes_a_png_of_the_viewport_size_times_the_scale() {
    let t0 = std::time::Instant::now();
    let Some(gpu) = test_gpu() else { return };
    let mut view = View3dState::default();
    view.set_mesh(
        GpuMeshData { faces: vec![vertex(-0.6, -0.5), vertex(0.6, -0.5), vertex(0.0, 0.7)], opaque_vertex_count: 3, ..Default::default() },
        "Triangle",
    );
    view.apply_preset(CameraPreset::Top);
    view.last_view = Some((egui::vec2(120.0, 80.0), 1.0, 7));
    let path = std::env::temp_dir().join(format!("papertilling_t3_test_{}.png", std::process::id()));
    let tokens = ThemeTokens::paper();
    let info = view.export_png(&gpu, &tokens, 2, &path).expect("export");
    let bytes = std::fs::read(&path).expect("png written");
    let _ = std::fs::remove_file(&path);
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().expect("decodes");
    // 120 x 80 points at 2x = 240 x 160 pixels, no clamp
    assert_eq!((info.w, info.h, info.clamped), (240, 160, false));
    assert_eq!((reader.info().width, reader.info().height), (240, 160));
    let mut rgba = vec![0; reader.output_buffer_size().expect("size")];
    reader.next_frame(&mut rgba).expect("frame");
    // same camera (Top) and triangle as the render test: centre shaded (169, 88, 48) within 2; the corner (0, 0)
    // lies outside the caption inset (8, 6) x 2, so it is the Paper 3D background (244, 246, 250)
    let c = px(&rgba, 240, 120, 80);
    assert!([169_u8, 88, 48].iter().zip(c).all(|(w, g)| w.abs_diff(g) <= 2), "centre {c:?}");
    assert_eq!(px(&rgba, 240, 0, 0), [244, 246, 250, 255]);
    assert_eq!(tokens.bg_viewport_3d.to_srgba_unmultiplied(), [244, 246, 250, 255]);
    eprintln!("# time gpu export {:?}", t0.elapsed());
}

#[test]
fn gpu_render_over_the_device_texture_limit_is_an_error_not_a_panic() {
    let Some(mut gpu) = test_gpu() else { return };
    let limit = gpu.device.limits().max_texture_dimension_2d;
    // pretend the limit is higher, so the request reaches the device and fails its validation
    gpu.max_texture_side = u32::MAX;
    let mesh = GpuMeshData::default();
    let scene = OffscreenScene { mesh: &mesh, mvp: [[0.0; 4]; 4], rot: [[0.0; 4]; 4], is_light: true, background: [0, 0, 0, 255], wire_width: None };
    let err = render_rgba(&gpu, &scene, limit + 1, 1).expect_err("texture wider than the device limit");
    assert!(err.contains("GPU error"), "{err}");
    // the device keeps working afterwards
    gpu.max_texture_side = limit;
    assert_eq!(render_rgba(&gpu, &scene, 4, 4).map(|v| v.len()), Ok(64));
}

#[test]
fn gpu_wires_are_as_many_pixels_wide_as_the_export_scale() {
    let t0 = std::time::Instant::now();
    let Some(gpu) = test_gpu() else { return };
    // Top preset (rotation = identity): the wire y = 0, z = 0 from x = -0.6 to 0.6 projects to NDC y = 0, the boundary
    // between rows 23 and 24 of a 48 px high target, and spans the centre column 32 (NDC |x| <= 0.6 f / (4/3) / 3.1
    // = 0.35). A 4 px wide quad covers the rows whose centres lie within 2 px of it: 22, 23, 24, 25. The 1 px line
    // (None, the live view's LineList) covers one row. Colour (0.2, 0.6, 0.4): chroma 0.4, so fs_wire keeps it on
    // Paper -> bytes (51, 153, 102).
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Top);
    let wire = |x: f32| Vertex3d { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [0.2, 0.6, 0.4, 1.0] };
    let mesh = GpuMeshData { wires: vec![wire(-0.6), wire(0.6)], ..Default::default() };
    let (w, h) = (64_u32, 48_u32);
    let rot = view.rotation();
    let bg = [244, 246, 250, 255];
    let mut rows = Vec::new();
    for width in [None, Some(4.0)] {
        let scene = OffscreenScene { mesh: &mesh, mvp: view.view_mvp(w as f32 / h as f32, &rot), rot, is_light: true, background: bg, wire_width: width };
        let rgba = render_rgba(&gpu, &scene, w, h).expect("offscreen render");
        let on: Vec<u32> = (0..h).filter(|&y| px(&rgba, w, 32, y)[..3].iter().zip([51_u8, 153, 102]).all(|(a, b)| a.abs_diff(b) <= 1)).collect();
        let off = (0..h).filter(|&y| px(&rgba, w, 32, y) == bg).count();
        assert_eq!(on.len() + off, h as usize, "{width:?}: every pixel of the column is wire or background");
        rows.push(on);
    }
    assert_eq!(rows[0].len(), 1, "1 px line: {:?}", rows[0]);
    assert!(rows[0][0] == 23 || rows[0][0] == 24, "{:?}", rows[0]);
    assert_eq!(rows[1], vec![22, 23, 24, 25], "4 px quad");
    eprintln!("# time gpu wire width {:?}", t0.elapsed());
}
#[test]
fn gpu_wide_wires_on_a_tilted_face_are_not_cut_by_its_depth() {
    let Some(gpu) = test_gpu() else { return };
    // The 64 x 48 wire of the width test lying on an opaque face z = 0.8 y (tilted about x, towards the camera for
    // y > 0). An 8 px quad (rows 20..=27: centres within 4 px of the row 23/24 boundary) carries the line's depth over
    // its width, while the face nearer to the camera above the line is 0.8 y closer: at 3.5 px that is 3.5 px of
    // depth slope, beyond the live view's face bias of 2 px of slope, so the export scales the slope bias with the
    // wire width (2 x 8 = 16 px of slope) and rows 20, 21 stay wire-coloured.
    let mut view = View3dState::default();
    view.apply_preset(CameraPreset::Top);
    let wire = |x: f32| Vertex3d { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], color: [0.2, 0.6, 0.4, 1.0] };
    let face = |x: f32, y: f32| Vertex3d { position: [x, y, 0.8 * y], normal: [0.0, -0.8, 1.0], color: [0.8, 0.4, 0.2, 1.0] };
    let mesh = GpuMeshData {
        faces: vec![face(-0.9, -0.6), face(0.9, -0.6), face(0.0, 0.8)],
        opaque_vertex_count: 3,
        wires: vec![wire(-0.6), wire(0.6)],
        ..Default::default()
    };
    let (w, h) = (64_u32, 48_u32);
    let rot = view.rotation();
    let bg = [244, 246, 250, 255];
    let scene = OffscreenScene { mesh: &mesh, mvp: view.view_mvp(w as f32 / h as f32, &rot), rot, is_light: true, background: bg, wire_width: Some(8.0) };
    let rgba = render_rgba(&gpu, &scene, w, h).expect("offscreen render");
    let on: Vec<u32> = (0..h).filter(|&y| px(&rgba, w, 32, y)[..3].iter().zip([51_u8, 153, 102]).all(|(a, b)| a.abs_diff(b) <= 1)).collect();
    assert_eq!(on, (20..28).collect::<Vec<u32>>(), "8 px wire over the tilted face");
    // the face itself is drawn just outside the wire (row 18 above, row 29 below)
    assert!(px(&rgba, w, 32, 18) != bg && px(&rgba, w, 32, 29) != bg);
}