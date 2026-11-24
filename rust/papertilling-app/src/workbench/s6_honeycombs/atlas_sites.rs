//! A finite patch of the sparsest single orbit under the chosen H³ reflection group.
//! The generating bond joins a site to its image in the opposite chamber mirror.
//! All other mirrors fix the site, so this one bond orbit connects the infinite net.

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
        SiteOrbit::Face => {
            // Two consecutive faces about an edge bound the same cell.
            for cell in &cx.cells {
                let mut owner = HashMap::new();
                for &face in cell {
                    let cycle = &cx.faces[face];
                    for (&a, &b) in cycle.iter().zip(cycle.iter().cycle().skip(1)) {
                        if let Some(other) = owner.insert([a.min(b), a.max(b)], face) {
                            bonds.push([other, face]);
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
    fn known_site_stabilizers_and_dual_bonds() {
        for ((p, q, r), orbit) in [
            ((4, 3, 5), (SiteOrbit::Vertex, 120)),
            ((5, 3, 4), (SiteOrbit::Cell, 120)),
            ((3, 3, 6), (SiteOrbit::Cell, 24)),
            ((4, 4, 4), (SiteOrbit::Face, 16)),
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
