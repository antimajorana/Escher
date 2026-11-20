//! Local moves on a [`PlanarMap`]. Each move checks its own precondition and refuses with a reason.
//!
//! * `slide`: move one end of an edge along a neighbouring edge to that edge's other end. On the toric
//!   code this is the rewiring gate CX(control -> target). Vertices, edges and faces keep their numbers.
//! * `leaf_to_loop`: a dangling edge becomes a loop on its other vertex. This is the Hadamard on that
//!   edge: one vertex fewer, one face more.
//! * `merge_edges`, `add_edge`: the two changes of Hamiltonian used by Wang, Liu and Lu.

use serde::{Deserialize, Serialize};

use super::map::{Dart, PlanarMap, VertexKind};

/// Side of the control edge on which the sliding end travels, seen when moving along the control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

impl PlanarMap {
    /// The single end of edge `label` at vertex `v`.
    pub fn dart_at(&self, label: u32, v: usize) -> Result<Dart, String> {
        let e = self.edge_index(label)?;
        let here: Vec<Dart> = (0..2).map(|k| 2 * e + k).filter(|&d| self.vertex_of(d) == v).collect();
        match here.len() {
            1 => Ok(here[0]),
            0 => Err(format!("edge {label} does not end at vertex {}", self.verts[v].name)),
            _ => Err(format!("edge {label} is a loop at vertex {}", self.verts[v].name)),
        }
    }

    /// Slide the end of `target` that sits at `from` along `control` to the other end of `control`.
    /// The sliding end must be next to the control in the cyclic order at `from`, otherwise it would have
    /// to cross another edge. `side` is needed only when both sides are possible. Returns the new vertex.
    pub fn slide(&mut self, target: u32, control: u32, from: usize, side: Option<Side>) -> Result<usize, String> {
        if target == control {
            return Err(format!("edge {target} cannot slide along itself"));
        }
        let c = self.dart_at(control, from)?;
        let to = self.vertex_of(c ^ 1);
        if self.verts[to].kind == VertexKind::Removed {
            return Err(format!("edge {control} ends at a removed vertex"));
        }
        let te = self.edge_index(target)?;
        let mut options: Vec<(Dart, Side)> = Vec::new();
        for t in [2 * te, 2 * te + 1] {
            if self.vertex_of(t) != from {
                continue;
            }
            // `t` clockwise-next to `c`: it travels on the right of the control. Counter-clockwise-next: on the left.
            if self.next_ccw(t) == c {
                options.push((t, Side::Right));
            }
            if self.next_ccw(c) == t {
                options.push((t, Side::Left));
            }
        }
        if let Some(s) = side {
            options.retain(|&(_, os)| os == s);
        }
        let name = &self.verts[from].name;
        let (t, s) = match options.len() {
            1 => options[0],
            0 => {
                return Err(format!(
                    "edge {target} is not next to edge {control} at vertex {name}: the slide would cross another edge"
                ));
            }
            _ => return Err(format!("edge {target} can slide on either side of edge {control} at vertex {name}: give the side")),
        };
        self.verts[from].rot.retain(|&d| d != t);
        let back = c ^ 1;
        let rot = &mut self.verts[to].rot;
        let i = rot.iter().position(|&d| d == back).expect("control end is listed at its vertex");
        // Travelling from -> to, the right-hand side is counter-clockwise of the control's far end.
        rot.insert(if s == Side::Right { i + 1 } else { i }, t);
        self.edges[te].ends[t % 2] = to;
        self.edges[te].ctrl = None;
        Ok(to)
    }

    /// Turn the dangling edge `label` (one end on a complete vertex of valence 1) into a loop on its other
    /// vertex. The leaf vertex is removed; the loop encloses one new face with one side.
    pub fn leaf_to_loop(&mut self, label: u32) -> Result<(), String> {
        let e = self.edge_index(label)?;
        if self.is_loop(e) {
            return Err(format!("edge {label} is already a loop"));
        }
        let leaf_end = (0..2).find(|&k| {
            let v = &self.verts[self.edges[e].ends[k]];
            v.kind == VertexKind::Complete && v.rot.len() == 1
        });
        let Some(k) = leaf_end else {
            return Err(format!("edge {label} is not dangling: neither end is a complete vertex of valence 1"));
        };
        let (leaf_dart, base_dart) = (2 * e + k, 2 * e + (1 - k));
        let (leaf, base) = (self.edges[e].ends[k], self.edges[e].ends[1 - k]);
        let (p, q) = (self.verts[base].pos, self.verts[leaf].pos);
        self.edges[e].loop_dir_deg = (q[1] - p[1]).atan2(q[0] - p[0]).to_degrees();
        self.edges[e].ctrl = None;
        self.verts[leaf].rot.clear();
        self.verts[leaf].kind = VertexKind::Removed;
        let rot = &mut self.verts[base].rot;
        let i = rot.iter().position(|&d| d == base_dart).expect("base end is listed at its vertex");
        rot.insert(i + 1, leaf_dart);
        self.edges[e].ends[k] = base;
        Ok(())
    }

    /// Merge edges `first` and `second`, which meet at a complete vertex of valence 2, into one edge
    /// `new_label`; that vertex is removed. Returns the removed vertex.
    pub fn merge_edges(&mut self, first: u32, second: u32, new_label: u32) -> Result<usize, String> {
        let (e1, e2) = (self.edge_index(first)?, self.edge_index(second)?);
        if e1 == e2 || self.is_loop(e1) || self.is_loop(e2) {
            return Err("merging needs two different edges, neither of them a loop".to_string());
        }
        if new_label != first && new_label != second && self.edges.iter().any(|e| e.alive && e.label == new_label) {
            return Err(format!("edge label {new_label} is already in use"));
        }
        let shared = self.edges[e1].ends.iter().copied().find(|&v| {
            self.verts[v].kind == VertexKind::Complete
                && self.verts[v].rot.len() == 2
                && self.edges[e2].ends.contains(&v)
        });
        let Some(s) = shared else {
            return Err(format!("edges {first} and {second} do not meet at a complete vertex of valence 2"));
        };
        let d1 = self.dart_at(first, s)?;
        let far2 = self.dart_at(second, s)? ^ 1;
        let w = self.vertex_of(far2);
        for d in &mut self.verts[w].rot {
            if *d == far2 {
                *d = d1;
            }
        }
        self.edges[e1].ends[d1 % 2] = w;
        self.edges[e1].label = new_label;
        self.edges[e1].ctrl = None;
        self.edges[e2].alive = false;
        self.verts[s].rot.clear();
        self.verts[s].kind = VertexKind::Removed;
        Ok(s)
    }

    /// Add edge `label` across a face, from the corner just counter-clockwise of edge `after_v` at vertex
    /// `v` to the corner just counter-clockwise of edge `after_w` at vertex `w`. Refused unless the two
    /// corners lie on the same face (the new edge must split one face into two).
    pub fn add_edge(&mut self, label: u32, v: usize, after_v: u32, w: usize, after_w: u32) -> Result<(), String> {
        if v == w {
            return Err("add_edge joins two different vertices".to_string());
        }
        if self.edges.iter().any(|e| e.alive && e.label == label) {
            return Err(format!("edge label {label} is already in use"));
        }
        let (dv, dw) = (self.dart_at(after_v, v)?, self.dart_at(after_w, w)?);
        let before = self.clone();
        let e = self.edges.len();
        self.edges.push(super::map::MapEdge { label, ends: [v, w], alive: true, ctrl: None, loop_dir_deg: 0.0 });
        for (vertex, anchor, dart) in [(v, dv, 2 * e), (w, dw, 2 * e + 1)] {
            let rot = &mut self.verts[vertex].rot;
            let i = rot.iter().position(|&d| d == anchor).expect("anchor end is listed at its vertex");
            rot.insert(i + 1, dart);
        }
        if self.faces().len() != before.faces().len() + 1 {
            *self = before;
            return Err(format!("edge {label}: the two corners are not on the same face"));
        }
        Ok(())
    }
}
