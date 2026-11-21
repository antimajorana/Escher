//! 2D and 3D drawing of S5 results: stereographic pictures of S² tilings, Poincaré / Klein / upper-half-plane
//! pictures of H² patches, plain E² patches, and polygon meshes with geodesic wires.
//!
//! The canvas fills polygons as convex shapes, so faces are filled through their vertices (the vertices of a regular
//! polygon lie on a circle in every model used here, hence the filled polygon is convex) and edges are drawn as
//! sampled geodesics on top. Anything outside the clip radius is dropped and counted, never silently squashed.

use papertilling_core::noneuclidean2d::{geodesic_arc_poincare, poincare_to_klein, poincare_to_uhp};

use crate::canvas2d::Scene2D;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::wb_mesh::{mesh_from_polygons, wb_face_tint};
use crate::scene2d::{PolyPrim, ScenePrimitives, compute_bbox};
use crate::workbench::scene_helpers::{EDGE_RGBA, line_prim, ring_size_tint, unit_circle_lines};

/// Points farther than this from the origin of a 2D picture are dropped (stereographic and half-plane views).
pub(crate) const CLIP_R: f64 = 12.0;
const RIM_RGBA: [u8; 4] = [120, 120, 120, 255];

/// Planar model of H² used for the 2D picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum DiskModel {
    #[default]
    Poincare,
    Klein,
    UpperHalfPlane,
}

impl DiskModel {
    pub(crate) const ALL: [Self; 3] = [Self::Poincare, Self::Klein, Self::UpperHalfPlane];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Poincare => "Poincaré disk",
            Self::Klein => "Klein disk",
            Self::UpperHalfPlane => "Upper half-plane",
        }
    }

    /// Image of a Poincaré-disk point (None outside the model or the clip window).
    pub(crate) fn map(self, z: [f64; 2]) -> Option<[f64; 2]> {
        let w = match self {
            Self::Poincare => z,
            Self::Klein => poincare_to_klein(z),
            Self::UpperHalfPlane => poincare_to_uhp(z).ok()?,
        };
        (w[0].is_finite() && w[1].is_finite() && w[0].hypot(w[1]) <= CLIP_R).then_some(w)
    }

    /// Whether straight chords are geodesics in this model (true only for Klein).
    pub(crate) fn chords_are_geodesics(self) -> bool {
        self == Self::Klein
    }
}

/// Accumulates primitives; counts dropped faces.
pub(crate) struct Picture {
    p: ScenePrimitives,
    pub(crate) dropped: usize,
}

impl Picture {
    pub(crate) fn new(title: &str, note: &str) -> Self {
        let p = ScenePrimitives { title: title.to_string(), note: note.to_string(), ..Default::default() };
        Self { p, dropped: 0 }
    }

    fn ok(q: [f64; 2]) -> bool {
        q[0].is_finite() && q[1].is_finite() && q[0].hypot(q[1]) <= CLIP_R
    }

    /// A filled face (dropped and counted when a vertex is missing or outside the clip radius).
    pub(crate) fn face(&mut self, ring: Option<Vec<[f64; 2]>>, fill: [u8; 4], stroke: bool) {
        match ring {
            Some(r) if r.len() >= 3 && r.iter().all(|q| Self::ok(*q)) => self.p.polygons.push(PolyPrim {
                points: r.iter().map(|q| [q[0] as f32, q[1] as f32]).collect(),
                fill,
                stroke: stroke.then_some((EDGE_RGBA, 0.8)),
            }),
            _ => self.dropped += 1,
        }
    }

    /// A polyline; segments with an endpoint outside the clip radius are skipped.
    pub(crate) fn polyline(&mut self, pts: &[[f64; 2]], color: [u8; 4], width: f32) {
        for w in pts.windows(2) {
            if Self::ok(w[0]) && Self::ok(w[1]) {
                self.p.lines.push(line_prim(w[0], w[1], color, width));
                self.p.bond_count += 1;
            }
        }
    }

    pub(crate) fn add_unit_circle(&mut self) {
        self.p.lines.extend(unit_circle_lines(160, RIM_RGBA));
    }

    pub(crate) fn add_real_axis(&mut self) {
        self.p.lines.push(line_prim([-CLIP_R, 0.0], [CLIP_R, 0.0], RIM_RGBA, 1.0));
    }

    pub(crate) fn finish(mut self, site_count: usize) -> (Scene2D, usize) {
        let pts = self.p.lines.iter().flat_map(|l| [l.a, l.b]).chain(self.p.polygons.iter().flat_map(|g| g.points.clone()));
        self.p.bbox = compute_bbox(pts);
        self.p.site_count = site_count;
        if self.p.polygons.is_empty() && self.p.lines.is_empty() {
            self.p.bbox = ([-1.0, -1.0], [1.0, 1.0]);
        }
        (Scene2D::VectorScene { primitives: self.p }, self.dropped)
    }
}

/// Undirected edges of a face list (sorted, unique).
pub(crate) fn face_edges(faces: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut e: Vec<(usize, usize)> = faces
        .iter()
        .flat_map(|f| (0..f.len()).map(move |k| (f[k].min(f[(k + 1) % f.len()]), f[k].max(f[(k + 1) % f.len()]))))
        .filter(|(u, v)| u != v)
        .collect();
    e.sort_unstable();
    e.dedup();
    e
}

/// H² picture of a Poincaré-disk patch. `arcs`: draw sampled geodesic edges (small patches); otherwise the faces
/// are stroked along their chords (large patches). Returns the scene and the number of faces dropped.
pub(crate) fn h2_picture(
    pts: &[[f64; 2]],
    faces: &[Vec<usize>],
    tint: impl Fn(usize) -> [u8; 4],
    model: DiskModel,
    arcs: bool,
    title: &str,
    note: &str,
) -> (Scene2D, usize) {
    let mut pic = Picture::new(title, note);
    let at = |i: usize| pts.get(i).and_then(|z| model.map(*z));
    for (fi, f) in faces.iter().enumerate() {
        pic.face(f.iter().map(|&i| at(i)).collect(), tint(fi), !arcs);
    }
    if arcs {
        for (u, v) in face_edges(faces) {
            let (Some(a), Some(b)) = (pts.get(u), pts.get(v)) else { continue };
            if let Ok(arc) = geodesic_arc_poincare(*a, *b, 8) {
                let mapped: Vec<[f64; 2]> = arc.into_iter().filter_map(|z| model.map(z)).collect();
                pic.polyline(&mapped, EDGE_RGBA, 1.2);
            }
        }
    }
    match model {
        DiskModel::UpperHalfPlane => pic.add_real_axis(),
        _ => pic.add_unit_circle(),
    }
    pic.finish(pts.len())
}

/// E² picture (straight edges are geodesics).
pub(crate) fn e2_picture(pts: &[[f64; 2]], faces: &[Vec<usize>], title: &str, note: &str) -> (Scene2D, usize) {
    let mut pic = Picture::new(title, note);
    for f in faces {
        pic.face(f.iter().map(|&i| pts.get(i).copied()).collect(), ring_size_tint(f.len()), false);
    }
    for (u, v) in face_edges(faces) {
        if let (Some(a), Some(b)) = (pts.get(u), pts.get(v)) {
            pic.polyline(&[*a, *b], EDGE_RGBA, 1.2);
        }
    }
    pic.finish(pts.len())
}

fn norm3(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (n > 1e-9 && n.is_finite()).then(|| v.map(|c| c / n))
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Great-circle arc from `a` to `b` on the unit sphere (`segs` + 1 samples); None for antipodal points.
pub(crate) fn slerp(a: [f64; 3], b: [f64; 3], segs: usize) -> Option<Vec<[f64; 3]>> {
    let om = dot3(a, b).clamp(-1.0, 1.0).acos();
    if (std::f64::consts::PI - om).abs() < 1e-9 {
        return None;
    }
    if om < 1e-12 {
        return Some(vec![a, b]);
    }
    let s = om.sin();
    Some(
        (0..=segs)
            .map(|k| {
                let t = k as f64 / segs as f64;
                let (w0, w1) = (((1.0 - t) * om).sin() / s, (t * om).sin() / s);
                [w0 * a[0] + w1 * b[0], w0 * a[1] + w1 * b[1], w0 * a[2] + w1 * b[2]]
            })
            .collect(),
    )
}

/// Stereographic projection of the unit sphere from `pole` onto the plane through the origin orthogonal to it.
pub(crate) struct Stereo {
    pole: [f64; 3],
    e1: [f64; 3],
    e2: [f64; 3],
}

impl Stereo {
    /// Projection from the centre of `face` (that face becomes the unbounded outside, as in a Schlegel diagram).
    pub(crate) fn from_face(pts: &[[f64; 3]], face: &[usize]) -> Option<Self> {
        let mut c = [0.0; 3];
        for &i in face {
            let p = pts.get(i)?;
            (0..3).for_each(|k| c[k] += p[k]);
        }
        let pole = norm3(c)?;
        let r = if pole[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let e1 = norm3([pole[1] * r[2] - pole[2] * r[1], pole[2] * r[0] - pole[0] * r[2], pole[0] * r[1] - pole[1] * r[0]])?;
        let e2 = [pole[1] * e1[2] - pole[2] * e1[1], pole[2] * e1[0] - pole[0] * e1[2], pole[0] * e1[1] - pole[1] * e1[0]];
        Some(Self { pole, e1, e2 })
    }

    pub(crate) fn map(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let d = 1.0 - dot3(p, self.pole);
        (d > 1e-6).then(|| [dot3(p, self.e1) / d, dot3(p, self.e2) / d])
    }
}

/// S² picture of a polyhedral tiling with vertices on the unit sphere, projected from the centre of face 0 (which
/// is not filled: it is the outside). None when the vertices are not on the unit sphere or an edge is antipodal.
pub(crate) fn s2_picture(pts: &[[f64; 3]], faces: &[Vec<usize>], title: &str, note: &str) -> Option<(Scene2D, usize)> {
    if faces.is_empty() || pts.iter().any(|p| (dot3(*p, *p) - 1.0).abs() > 1e-6) {
        return None;
    }
    let st = Stereo::from_face(pts, &faces[0])?;
    let mut pic = Picture::new(title, note);
    for f in &faces[1..] {
        pic.face(f.iter().map(|&i| pts.get(i).and_then(|p| st.map(*p))).collect(), ring_size_tint(f.len()), false);
    }
    for (u, v) in face_edges(faces) {
        let arc = slerp(pts[u], pts[v], 12)?;
        let mapped: Vec<[f64; 2]> = arc.into_iter().filter_map(|p| st.map(p)).collect();
        pic.polyline(&mapped, EDGE_RGBA, 1.2);
    }
    Some(pic.finish(pts.len()))
}

/// Polyline picture (used for {2,q} / {p,2} from the core's own sampled arcs).
pub(crate) fn arcs_picture(arcs: &[Vec<[f64; 2]>], title: &str, note: &str) -> (Scene2D, usize) {
    let mut pic = Picture::new(title, note);
    for a in arcs {
        pic.polyline(a, EDGE_RGBA, 1.4);
    }
    pic.finish(0)
}

/// 3D polygon mesh tinted by ring size, with wires along the given sampled arcs (or the straight face edges).
pub(crate) fn mesh_with_arcs(pts: &[[f64; 3]], faces: &[Vec<usize>], arcs: Option<&[Vec<[f64; 3]>]>) -> GpuMeshData {
    let mut all = pts.to_vec();
    let edges = match arcs {
        Some(arcs) => {
            let mut e = Vec::new();
            for a in arcs {
                let base = all.len();
                all.extend_from_slice(a);
                e.extend((1..a.len()).map(|k| (base + k - 1, base + k)));
            }
            e
        }
        None => face_edges(faces),
    };
    let mut m = mesh_from_polygons(&all, faces, &edges, wb_face_tint);
    m.vertex_count = pts.len();
    m
}
