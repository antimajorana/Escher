use super::{clip_near, export_wire_width, wire_quads};
use crate::mesh3d::Vertex3d;

const I: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn v(x: f32, y: f32, z: f32) -> Vertex3d {
    Vertex3d { position: [x, y, z], normal: [0.0, 0.0, 1.0], color: [0.2, 0.6, 0.4, 1.0] }
}

fn close(p: [f32; 3], want: [f32; 3]) -> bool {
    p.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6)
}

fn positions(q: &[Vertex3d]) -> Vec<[f32; 3]> {
    q.iter().map(|v| v.position).collect()
}

#[test]
fn a_horizontal_segment_becomes_a_quad_of_the_width_with_square_caps() {
    // NDC (-0.5, 0) -> (0.5, 0) on 100 x 50 px, 4 px wide: half-width 2 px = 2 * 2 / 50 = 0.08 NDC in y; caps
    // 2 px = 2 * 2 / 100 = 0.04 NDC in x past each end; depth kept (0.5)
    let q = wire_quads(&[v(-0.5, 0.0, 0.5), v(0.5, 0.0, 0.5)], &I, [100, 50], 4.0);
    let (a0, a1, b0, b1) = ([-0.54, -0.08, 0.5], [-0.54, 0.08, 0.5], [0.54, -0.08, 0.5], [0.54, 0.08, 0.5]);
    let want = [a0, a1, b0, a1, b1, b0];
    assert_eq!(q.len(), 6);
    assert!(positions(&q).iter().zip(want).all(|(p, w)| close(*p, w)), "{:?}", positions(&q));
    assert!(q.iter().all(|x| x.color == [0.2, 0.6, 0.4, 1.0]));
}

#[test]
fn a_vertical_segment_is_widened_in_x_by_the_same_pixel_count() {
    // (0, -0.5) -> (0, 0.5): half-width 2 px = 0.04 NDC in x (100 px wide), caps 2 px = 0.08 NDC in y (50 px high)
    let q = wire_quads(&[v(0.0, -0.5, 0.2), v(0.0, 0.5, 0.2)], &I, [100, 50], 4.0);
    let xs: Vec<f32> = q.iter().map(|v| v.position[0]).collect();
    let ys: Vec<f32> = q.iter().map(|v| v.position[1]).collect();
    let span = |s: &[f32]| (s.iter().copied().fold(f32::MAX, f32::min), s.iter().copied().fold(f32::MIN, f32::max));
    let ((x0, x1), (y0, y1)) = (span(&xs), span(&ys));
    assert!((x0 + 0.04).abs() < 1e-6 && (x1 - 0.04).abs() < 1e-6, "{x0} {x1}");
    assert!((y0 + 0.58).abs() < 1e-6 && (y1 - 0.58).abs() < 1e-6, "{y0} {y1}");
}

#[test]
fn the_perspective_divide_is_applied_before_widening() {
    // w = 2 for every point: NDC = position / 2, so (-1, 0, 0.5) -> (1, 0, 0.5) is the first test's segment at depth 0.25
    let mut m = I;
    m[3][3] = 2.0;
    let q = wire_quads(&[v(-1.0, 0.0, 0.5), v(1.0, 0.0, 0.5)], &m, [100, 50], 4.0);
    assert!(close(q[0].position, [-0.54, -0.08, 0.25]) && close(q[4].position, [0.54, 0.08, 0.25]), "{:?}", positions(&q));
}

#[test]
fn segments_behind_the_near_plane_are_dropped_or_cut() {
    // wholly behind (clip z < 0): nothing drawn; zero screen length: nothing drawn
    assert!(wire_quads(&[v(-0.5, 0.0, -0.2), v(0.5, 0.0, -0.1)], &I, [100, 50], 4.0).is_empty());
    assert!(wire_quads(&[v(0.3, 0.3, 0.5), v(0.3, 0.3, 0.9)], &I, [100, 50], 4.0).is_empty());
    // z from -1 to 1 along x from -0.5 to 0.5 crosses z = 0 at the midpoint x = 0: the quad starts at 0 - 0.04 (cap)
    let q = wire_quads(&[v(-0.5, 0.0, -1.0), v(0.5, 0.0, 1.0)], &I, [100, 50], 4.0);
    assert!(close(q[0].position, [-0.04, -0.08, 0.0]) && close(q[4].position, [0.54, 0.08, 1.0]), "{:?}", positions(&q));
    // eye plane: w from -1 to 1 (z kept >= 0) is cut at w = 1e-6, i.e. t = 0.5 + 5e-7
    let (a, b) = clip_near([0.0, 0.0, 0.5, -1.0], [0.0, 0.0, 0.5, 1.0]).expect("half in front");
    // (f32: t carries ~3e-8 of rounding, so w = -1 + 2 t within 2e-7)
    assert!((a[3] - 1e-6).abs() < 2e-7 && b[3] == 1.0, "{a:?} {b:?}");
    assert_eq!(clip_near([0.0, 0.0, 0.5, -1.0], [0.0, 0.0, 0.5, -0.5]), None);
}

#[test]
fn wire_width_is_export_pixels_per_screen_pixel_and_absent_at_1x() {
    // 1200 px on screen: 4x export 4800 px -> 4 px wires; 1x -> the 1 px LineList (None); a 4096 px clamp -> 4096/1200
    assert_eq!(export_wire_width(4800, 1200.0), Some(4.0));
    assert_eq!(export_wire_width(1200, 1200.0), None);
    assert_eq!(export_wire_width(2400, 1200.0), Some(2.0));
    assert!((export_wire_width(4096, 1200.0).unwrap() - 3.413_333).abs() < 1e-4);
}
