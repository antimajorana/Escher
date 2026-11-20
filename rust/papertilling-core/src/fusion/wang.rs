//! Worked example: the fusion of four dislocations in the square-lattice toric code.
//!
//! Start lattice: Fig. 8(a) of Y.-Q. Wang, C. Liu, Y.-M. Lu, Phys. Rev. B 109, 195165 (2024), Sec. V D 4,
//! edges labelled 1..14 as there; its seven terms are their Eq. (56). Two routes to the fused lattice:
//! * `gate_route`: five unitary steps (four rewiring gates and one Hadamard);
//! * `hamiltonian_route`: Wang, Liu and Lu's two changes of Hamiltonian, their Eqs. (57) and (58).
//!
//! The positions are a drawing of the figure. What is computed is the map: which edge ends at which
//! vertex, in which cyclic order.

use serde::{Deserialize, Serialize};

use super::map::{PlanarMap, VertexKind};
use super::moves::Side;

/// One lattice of a fusion sequence, with the operation that produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FusionStep {
    pub title: String,
    /// The operation in symbols, empty for the start lattice.
    pub operation: String,
    /// What the operation does to the lattice, in plain words.
    pub explanation: String,
    /// Edges to highlight in the drawing.
    pub highlight: Vec<u32>,
    pub map: PlanarMap,
}

// Vertex numbers of the start lattice.
pub const A: usize = 0;
pub const B: usize = 1;
pub const C: usize = 2;
pub const D: usize = 3;
pub const E: usize = 4;
pub const F: usize = 5;
pub const G: usize = 6;
pub const H: usize = 7;
pub const S: usize = 8;

/// Fig. 8(a): a two-valent vertex `s`, a seven-sided face and a five-sided face. Edges 4, 8, 13, 14 are
/// external legs; vertices a, b, e, f are cut by the edge of the figure.
pub fn fig_8a() -> PlanarMap {
    use VertexKind::{Boundary, Complete};
    let verts = [
        ("a", [0.0, 0.0], Boundary),
        ("b", [1.0, 1.0], Boundary),
        ("c", [2.0, 1.0], Complete),
        ("d", [3.0, 2.0], Complete),
        ("e", [4.0, 2.0], Boundary),
        ("f", [4.0, 1.0], Boundary),
        ("g", [2.0, 0.0], Complete),
        ("h", [1.0, 0.0], Complete),
        ("s", [3.0, 1.0], Complete),
        ("leg 13", [2.0, 2.0], Boundary),
        ("leg 4", [3.0, 3.0], Boundary),
        ("leg 14", [1.0, -1.0], Boundary),
        ("leg 8", [2.0, -1.0], Boundary),
    ];
    let edges = [
        (1, A, B), (2, B, C), (3, C, D), (4, D, 10), (5, D, E), (6, E, F), (7, F, G),
        (8, G, 12), (9, G, H), (10, H, A), (11, D, S), (12, S, G), (13, C, 9), (14, H, 11),
    ];
    PlanarMap::from_drawing(&verts, &edges, 4).expect("Fig. 8(a) is a valid drawing")
}

fn step(title: &str, operation: &str, explanation: &str, highlight: &[u32], map: &PlanarMap) -> FusionStep {
    FusionStep {
        title: title.to_string(),
        operation: operation.to_string(),
        explanation: explanation.to_string(),
        highlight: highlight.to_vec(),
        map: map.clone(),
    }
}

fn start_step(map: &PlanarMap) -> FusionStep {
    step(
        "Four dislocations inside the loop",
        "",
        "The lattice after the four inner dislocations have been moved together. Vertex s has only two edges \
         (11 and 12), one face has seven sides and one has five.",
        &[11, 12],
        map,
    )
}

/// The unitary route: CX(11->12), H(11), CX(12->11), CX(3->11), CX(9->11), applied in this order.
pub fn gate_route() -> Result<Vec<FusionStep>, String> {
    let mut m = fig_8a();
    let mut out = vec![start_step(&m)];

    // Edge 12 leaves s on the side of the five-sided face, so that edge 11 is left dangling inside the
    // seven-sided face. The other side fails later: edge 11 would not be next to edge 3 at vertex d.
    m.slide(12, 11, S, Some(Side::Right))?;
    // Drawing: the leaf is shortened and edge 12 curves round it on the side of the five-sided face.
    m.verts[S].pos = [3.0, 1.45];
    m.set_curve(12, Some([4.0, 1.6]))?;
    out.push(step(
        "Edge 12 slides off the two-valent vertex",
        "CX(11 -> 12)",
        "The end of edge 12 at s slides along edge 11 to vertex d. Edge 11 is left dangling: s is now a leaf.",
        &[11, 12],
        &m,
    ));

    m.leaf_to_loop(11)?;
    out.push(step(
        "The dangling edge becomes a loop",
        "H(11)",
        "The leaf vertex s disappears and edge 11 becomes a loop on vertex d. The loop encloses a new face \
         with one side: one vertex fewer, one face more.",
        &[11],
        &m,
    ));

    m.slide(11, 12, D, None)?;
    m.set_curve(11, Some([2.32, 1.09]))?;
    m.set_curve(12, Some([2.72, 0.89]))?;
    out.push(step(
        "The loop opens along edge 12",
        "CX(12 -> 11)",
        "One end of the loop slides along edge 12 to vertex g. Edges 11 and 12 now run side by side and \
         enclose a face with two sides.",
        &[11, 12],
        &m,
    ));

    m.slide(11, 3, D, None)?;
    m.set_curve(12, None)?;
    out.push(step(
        "The face grows to a triangle",
        "CX(3 -> 11)",
        "The end of edge 11 at d slides along edge 3 to vertex c. The new face has three sides: 3, 12, 11.",
        &[11, 3],
        &m,
    ));

    m.slide(11, 9, G, None)?;
    out.push(step(
        "The face grows to a square",
        "CX(9 -> 11)",
        "The end of edge 11 at g slides along edge 9 to vertex h. Every face has four sides and every \
         vertex four edges again.",
        &[11, 9],
        &m,
    ));
    Ok(out)
}

/// Wang, Liu and Lu's route: merge edges 11 and 12 into edge 15 (Eq. 57), then add edge 16 (Eq. 58).
pub fn hamiltonian_route() -> Result<Vec<FusionStep>, String> {
    let mut m = fig_8a();
    let mut out = vec![start_step(&m)];
    m.merge_edges(11, 12, 15)?;
    out.push(step(
        "Edges 11 and 12 are merged",
        "merge 11, 12 -> 15",
        "The two edges at the two-valent vertex s become one edge, 15, from d to g, and s is removed. \
         The seven-sided face now has six sides.",
        &[15],
        &m,
    ));
    m.add_edge(16, H, 9, C, 2)?;
    out.push(step(
        "Edge 16 is added",
        "add 16",
        "A new edge from h to c splits the six-sided face into two squares. This step adds a qubit.",
        &[16],
        &m,
    ));
    Ok(out)
}
