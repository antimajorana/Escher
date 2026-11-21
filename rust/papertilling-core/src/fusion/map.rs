//! A small planar map (graph + cyclic order of edge-ends round every vertex) that can hold the lattices
//! a Volterra cut cannot: a two-valent vertex, a dangling edge, a loop on one vertex, a double edge.
//!
//! The map is a patch cut out of a larger lattice. Vertices whose edges are all in the patch are
//! `Complete`; vertices cut by the edge of the picture (and the far ends of external legs) are
//! `Boundary`. Only complete vertices and bounded faces are counted and given operators.

use serde::{Deserialize, Serialize};

/// An edge-end: `2 * edge index + end` (`end` is 0 or 1). `d ^ 1` is the other end of the same edge.
pub type Dart = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VertexKind {
    /// Every edge of this vertex is in the patch.
    Complete,
    /// Cut by the edge of the picture: it has edges that are not drawn.
    Boundary,
    /// Deleted by a move (kept so that vertex numbers never change between steps).
    Removed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapVertex {
    pub name: String,
    /// Drawing position only; the map itself is `rot`.
    pub pos: [f64; 2],
    pub kind: VertexKind,
    /// Edge-ends at this vertex in counter-clockwise order.
    pub rot: Vec<Dart>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapEdge {
    pub label: u32,
    /// Vertex of end 0 and of end 1.
    pub ends: [usize; 2],
    pub alive: bool,
    /// Drawing only: control point of the drawn curve (`None` = straight).
    pub ctrl: Option<[f64; 2]>,
    /// Drawing only, for a loop: direction (degrees) in which the loop leaves its vertex.
    pub loop_dir_deg: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanarMap {
    pub verts: Vec<MapVertex>,
    pub edges: Vec<MapEdge>,
    /// One edge-end known to lie on the unbounded outer face (the far end of an external leg).
    pub outer: Dart,
}

/// One face: its boundary walk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceWalk {
    pub darts: Vec<Dart>,
    /// Vertex each step of the walk leaves from.
    pub vertices: Vec<usize>,
    /// Edge label of each step. An edge the face meets on both sides appears twice.
    pub labels: Vec<u32>,
}

impl FaceWalk {
    /// Number of sides, counting an edge met on both sides twice.
    pub fn sides(&self) -> usize {
        self.darts.len()
    }
}

/// Counts and operators of the complete vertices and bounded faces of a patch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Census {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    /// (vertex name, valence); a loop counts twice.
    pub valences: Vec<(String, usize)>,
    pub face_sides: Vec<usize>,
    /// Toric-code vertex term of each complete vertex: sorted labels of the edges met an odd number of times.
    pub vertex_ops: Vec<Vec<u32>>,
    /// Toric-code face term of each bounded face, same convention.
    pub face_ops: Vec<Vec<u32>>,
    /// Sum of (4 - valence) over complete vertices plus sum of (4 - sides) over bounded faces.
    pub square_curvature: i64,
}

fn odd_labels(mut labels: Vec<u32>) -> Vec<u32> {
    labels.sort_unstable();
    let mut out: Vec<u32> = Vec::new();
    for l in labels {
        if out.last() == Some(&l) {
            out.pop();
        } else {
            out.push(l);
        }
    }
    out
}

impl PlanarMap {
    /// Build the map from a straight-line drawing: the cyclic order at each vertex is read off the angles.
    /// `verts`: (name, position, kind); `edges`: (label, vertex, vertex); `outer_leg`: label of an external leg
    /// whose `Boundary` end of valence 1 lies on the outer face.
    pub fn from_drawing(
        verts: &[(&str, [f64; 2], VertexKind)],
        edges: &[(u32, usize, usize)],
        outer_leg: u32,
    ) -> Result<Self, String> {
        let mut map = PlanarMap {
            verts: verts
                .iter()
                .map(|&(name, pos, kind)| MapVertex { name: name.to_string(), pos, kind, rot: Vec::new() })
                .collect(),
            edges: Vec::with_capacity(edges.len()),
            outer: 0,
        };
        for &(label, a, b) in edges {
            if a >= verts.len() || b >= verts.len() || a == b {
                return Err(format!("edge {label}: needs two different vertices of the drawing"));
            }
            if map.edges.iter().any(|e| e.label == label) {
                return Err(format!("edge label {label} is used twice"));
            }
            let e = map.edges.len();
            map.edges.push(MapEdge { label, ends: [a, b], alive: true, ctrl: None, loop_dir_deg: 0.0 });
            map.verts[a].rot.push(2 * e);
            map.verts[b].rot.push(2 * e + 1);
        }
        for v in 0..map.verts.len() {
            let mut keyed: Vec<(f64, Dart)> = map.verts[v].rot.iter().map(|&d| (map.dart_angle(d), d)).collect();
            keyed.sort_by(|p, q| p.0.total_cmp(&q.0));
            if keyed.windows(2).any(|w| (w[1].0 - w[0].0).abs() < 1e-9) {
                return Err(format!("vertex {}: two edges leave in the same direction", map.verts[v].name));
            }
            map.verts[v].rot = keyed.into_iter().map(|(_, d)| d).collect();
        }
        let e = map.edge_index(outer_leg)?;
        let leaf = (0..2).find(|&k| {
            let v = &map.verts[map.edges[e].ends[k]];
            v.kind == VertexKind::Boundary && v.rot.len() == 1
        });
        map.outer = 2 * e + leaf.ok_or_else(|| format!("edge {outer_leg} is not an external leg"))?;
        if map.euler() != 2 {
            return Err(format!("the drawing is not one connected planar piece (V - E + F = {})", map.euler()));
        }
        Ok(map)
    }

    pub fn edge_index(&self, label: u32) -> Result<usize, String> {
        self.edges
            .iter()
            .position(|e| e.alive && e.label == label)
            .ok_or_else(|| format!("there is no edge {label}"))
    }

    pub fn vertex_of(&self, d: Dart) -> usize {
        self.edges[d / 2].ends[d % 2]
    }

    pub fn label_of(&self, d: Dart) -> u32 {
        self.edges[d / 2].label
    }

    pub fn is_loop(&self, edge: usize) -> bool {
        self.edges[edge].ends[0] == self.edges[edge].ends[1]
    }

    /// The edge-end that follows `d` counter-clockwise at its vertex.
    pub fn next_ccw(&self, d: Dart) -> Dart {
        let rot = &self.verts[self.vertex_of(d)].rot;
        let i = rot.iter().position(|&x| x == d).expect("dart is listed at its vertex");
        rot[(i + 1) % rot.len()]
    }

    /// Control point of the drawn curve of a non-loop edge (the midpoint when the edge is straight).
    pub fn control_point(&self, edge: usize) -> [f64; 2] {
        let e = &self.edges[edge];
        let (p, q) = (self.verts[e.ends[0]].pos, self.verts[e.ends[1]].pos);
        e.ctrl.unwrap_or([0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])])
    }

    /// Direction (radians) in which edge-end `d` leaves its vertex in the drawing.
    pub fn dart_angle(&self, d: Dart) -> f64 {
        let edge = d / 2;
        if self.is_loop(edge) {
            return self.edges[edge].loop_dir_deg.to_radians();
        }
        let p = self.verts[self.vertex_of(d)].pos;
        let c = self.control_point(edge);
        (c[1] - p[1]).atan2(c[0] - p[0])
    }

    /// Draw `label` as a curve with this control point, or straight again with `None` (drawing only).
    pub fn set_curve(&mut self, label: u32, ctrl: Option<[f64; 2]>) -> Result<(), String> {
        let e = self.edge_index(label)?;
        self.edges[e].ctrl = ctrl;
        Ok(())
    }

    /// All faces, each walked with the face on its right; the first one returned is the outer face.
    pub fn faces(&self) -> Vec<FaceWalk> {
        let n = 2 * self.edges.len();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        let starts = std::iter::once(self.outer).chain((0..n).filter(|&d| self.edges[d / 2].alive));
        for start in starts {
            if seen[start] {
                continue;
            }
            let mut walk = FaceWalk { darts: Vec::new(), vertices: Vec::new(), labels: Vec::new() };
            let mut d = start;
            while !seen[d] {
                seen[d] = true;
                walk.darts.push(d);
                walk.vertices.push(self.vertex_of(d));
                walk.labels.push(self.label_of(d));
                d = self.next_ccw(d ^ 1);
            }
            out.push(walk);
        }
        out
    }

    /// V - E + F over every vertex that has an edge, every edge and every face including the outer one.
    /// It is 2 exactly when the map is connected and planar.
    pub fn euler(&self) -> i64 {
        let v = self.verts.iter().filter(|v| v.kind != VertexKind::Removed && !v.rot.is_empty()).count();
        let e = self.edges.iter().filter(|e| e.alive).count();
        v as i64 - e as i64 + self.faces().len() as i64
    }

    /// The bounded faces (every face except the outer one).
    pub fn bounded_faces(&self) -> Vec<FaceWalk> {
        self.faces().into_iter().skip(1).collect()
    }

    pub fn complete_vertices(&self) -> Vec<usize> {
        (0..self.verts.len()).filter(|&v| self.verts[v].kind == VertexKind::Complete).collect()
    }

    /// Toric-code vertex term: labels of the edges with an odd number of ends at `v` (a loop cancels).
    pub fn vertex_op(&self, v: usize) -> Vec<u32> {
        odd_labels(self.verts[v].rot.iter().map(|&d| self.label_of(d)).collect())
    }

    /// Toric-code face term: labels met an odd number of times on the walk (a dangling edge cancels).
    pub fn face_op(face: &FaceWalk) -> Vec<u32> {
        odd_labels(face.labels.clone())
    }

    pub fn census(&self) -> Census {
        let cv = self.complete_vertices();
        let faces = self.bounded_faces();
        let valences: Vec<(String, usize)> =
            cv.iter().map(|&v| (self.verts[v].name.clone(), self.verts[v].rot.len())).collect();
        let face_sides: Vec<usize> = faces.iter().map(FaceWalk::sides).collect();
        let curv = valences.iter().map(|(_, k)| 4 - *k as i64).sum::<i64>()
            + face_sides.iter().map(|&s| 4 - s as i64).sum::<i64>();
        Census {
            vertices: cv.len(),
            edges: self.edges.iter().filter(|e| e.alive).count(),
            faces: faces.len(),
            vertex_ops: cv.iter().map(|&v| self.vertex_op(v)).collect(),
            face_ops: faces.iter().map(PlanarMap::face_op).collect(),
            valences,
            face_sides,
            square_curvature: curv,
        }
    }

    /// Cyclic order of edges round `v` as the DRAWING shows it (a loop is one entry). Used to check that
    /// the picture of a step shows the same map as the one the moves produced.
    pub fn drawn_order(&self, v: usize) -> Vec<u32> {
        let mut keyed: Vec<(f64, u32)> = Vec::new();
        for &d in &self.verts[v].rot {
            let l = self.label_of(d);
            if !keyed.iter().any(|&(_, x)| x == l) || !self.is_loop(d / 2) {
                keyed.push((self.dart_angle(d).rem_euclid(std::f64::consts::TAU), l));
            }
        }
        keyed.sort_by(|p, q| p.0.total_cmp(&q.0));
        keyed.into_iter().map(|(_, l)| l).collect()
    }

    /// Cyclic order of edges round `v` in the map itself (the two ends of a loop merged into one entry).
    pub fn map_order(&self, v: usize) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &d in &self.verts[v].rot {
            let l = self.label_of(d);
            if out.last() != Some(&l) {
                out.push(l);
            }
        }
        if out.len() > 1 && out.first() == out.last() {
            out.pop();
        }
        out
    }
}
