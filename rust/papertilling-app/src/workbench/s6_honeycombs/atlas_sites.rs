//! A finite patch of the sparsest single orbit under the chosen H³ reflection group.
//! Bonds join a site to its images in the chamber mirrors which move it.

use std::collections::HashMap;

use papertilling_core::honeycomb3d::{CellComplex3d, Classification, coxeter_gram::SiteOrbit};

use super::present::Built;
use crate::scene2d::{CirclePrim, Scene2D};
use crate::workbench::scene_helpers::line_prim;

fn centre_bonds(cx: &CellComplex3d, site: SiteOrbit) -> Vec<[usize; 2]> {
    let mut bonds = Vec::new();
    match site {
        SiteOrbit::Cell => {
            let mut owner = vec![None; cx.faces.len()];
            for (cell, faces) in cx.cells.iter().enumerate() {
                for &face in faces {
                    if let Some(other) = owner[face].replace(cell) {
                        bonds.push([other, cell]);
                    }
                }
            }
        }
        SiteOrbit::Face | SiteOrbit::FaceCell => {
            let mut across_face = vec![None; cx.faces.len()];
            let mut incidence = 0;
            // Two consecutive faces about an edge bound the same cell.
            for cell in &cx.cells {
                let mut owner = HashMap::new();
                for &face in cell {
                    let index = if site == SiteOrbit::FaceCell {
                        let index = incidence;
                        incidence += 1;
                        if let Some(other) = across_face[face].replace(index) {
                            bonds.push([other, index]);
                        }
                        index
                    } else {
                        face
                    };
                    let cycle = &cx.faces[face];
                    for (&a, &b) in cycle.iter().zip(cycle.iter().cycle().skip(1)) {
                        if let Some(other) = owner.insert([a.min(b), a.max(b)], index) {
                            bonds.push([other, index]);
                        }
                    }
                }
            }
        }
        // The supported H³ symbols always have an endpoint or a face centre at
        // least as sparse as the edge centre. Vertices already have the cell wires.
        SiteOrbit::Edge | SiteOrbit::Vertex => {}
    }
    bonds
}

pub(super) fn add_orbit(built: &mut Built, classification: &Classification) {
    let Some((site, order)) = classification.finite_site_orbit() else {
        return;
    };
    let Some(cx) = built.core.complex() else {
        return;
    };
    let sites = if site == SiteOrbit::Vertex {
        &cx.klein
    } else {
        &cx.orbit_centres
    };
    let bonds = centre_bonds(cx, site);
    crate::mesh3d::wb_mesh::add_site_markers(&mut built.mesh, sites);
    crate::mesh3d::wb_mesh::add_site_bonds(&mut built.mesh, sites, &bonds);
    if let Scene2D::VectorScene { primitives } = &mut built.scene2d {
        for &[x, y, _] in sites {
            primitives
                .site_circle_indices
                .push(Some(primitives.circles.len()));
            primitives.circles.push(CirclePrim {
                center: [x as f32, y as f32],
                radius_world: 0.007,
                min_px: 1.0,
                max_px: 3.0,
                fill: [27, 82, 150, 240],
                stroke: ([255, 255, 255, 210], 0.4),
            });
        }
        for &[u, v] in &bonds {
            let [a, b] = [u, v].map(|i| [sites[i][0], sites[i][1]]);
            primitives
                .lines
                .push(line_prim(a, b, [27, 82, 150, 220], 1.4));
        }
        primitives.site_count = sites.len();
        primitives.bond_count = if site == SiteOrbit::Vertex {
            cx.edges.len()
        } else {
            bonds.len()
        };
    }
    built.rows.insert(
        0,
        (
            "Blue site orbit".into(),
            format!("{} · 1 site per {order} chambers", site.label()),
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workbench::s6_honeycombs::build_atlas_h3;
    use papertilling_core::honeycomb3d::classify_schlafli;

    #[test]
    fn square_faces_with_extra_symmetry_are_resolved_into_opposite_side_pairs() {
        let patch = papertilling_core::honeycomb3d::paracompact_ball(4, 4, 4, 0.01).unwrap();
        let cx = &patch.complex;
        assert_eq!(cx.faces.len(), 1);
        assert_eq!(centre_bonds(cx, SiteOrbit::FaceCell), [[0, 1]]);
        assert_eq!(cx.orbit_centres.len(), 2);
        let (a, b) = (cx.orbit_centres[0], cx.orbit_centres[1]);
        assert!(a.iter().zip(b).all(|(x, y)| (x + y).abs() < 1e-12));
        let radius = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((radius - 0.1_f64.tanh()).abs() < 1e-12);

        // R > acosh(2) includes every square adjacent to the central square.
        // Across the square the separation is 2t. In a common ideal cell the
        // hyperbolic height grows by exp(t), scaling cosh(d) - 1 by exp(-2t).
        let patch = papertilling_core::honeycomb3d::paracompact_ball(4, 4, 4, 1.4).unwrap();
        let cx = &patch.complex;
        let bonds = centre_bonds(cx, SiteOrbit::FaceCell);
        let sites = &cx.orbit_centres;
        let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let central: Vec<_> = sites.iter().copied().enumerate()
            .filter(|(_, a)| (dot(*a, *a).sqrt() - radius).abs() < 1e-10).collect();
        assert_eq!(central.len(), 2);
        for (i, a) in central {
            let mut distances: Vec<_> = bonds.iter().filter(|pair| pair.contains(&i)).map(|&[u, v]| {
                let b = sites[if u == i { v } else { u }];
                (1.0 - dot(a, b)) / ((1.0 - dot(a, a)) * (1.0 - dot(b, b))).sqrt()
            }).collect();
            distances.sort_by(f64::total_cmp);
            assert_eq!(distances.len(), 5);
            assert!((distances[0] - 0.2_f64.cosh()).abs() < 1e-10);
            assert!(distances[1..].iter().all(|d| (d - (1.0 + (-0.2_f64).exp())).abs() < 1e-10));
        }
    }

    #[test]
    fn known_site_stabilizers_and_dual_bonds() {
        for ((p, q, r), orbit) in [
            ((4, 3, 5), (SiteOrbit::Vertex, 120)),
            ((5, 3, 4), (SiteOrbit::Cell, 120)),
            ((3, 3, 6), (SiteOrbit::Cell, 24)),
            ((4, 4, 4), (SiteOrbit::FaceCell, 8)),
        ] {
            assert_eq!(classify_schlafli(p, q, r).finite_site_orbit(), Some(orbit));
        }
        // A dodecahedral seed has twelve faces, hence twelve adjacent centre sites.
        let h = build_atlas_h3(5, 3, 4, 1, false, 140).unwrap();
        let cx = h.0.core.complex().unwrap();
        let bonds = centre_bonds(cx, SiteOrbit::Cell);
        assert_eq!(bonds.iter().filter(|pair| pair.contains(&0)).count(), 12);
        assert_eq!(cx.orbit_centres.len(), 63);
        assert!(cx.orbit_centres[0].iter().all(|x| x.abs() < 1e-10));
    }
}
