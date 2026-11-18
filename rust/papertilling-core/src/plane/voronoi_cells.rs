//! Periodic 2D Voronoi cells (Wigner-Seitz cells of a decorated lattice), built from the same
//! per-site half-plane clipping as `voronoi_bonds`.

use crate::plane::cell::{mat2_det, mat2_inv, mat2_vec};
use crate::plane::voronoi::PeriodicVoronoi;
use crate::types::Cell2D;

/// Voronoi cell of site `site`: Cartesian vertices in counter-clockwise order around `center`,
/// the site position in the original (unwrapped) cell.
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell2D {
    pub site: usize,
    pub center: [f64; 2],
    pub vertices: Vec<[f64; 2]>,
}

fn cross3(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Drop cyclically consecutive near-duplicate vertices and vertices on a straight edge.
fn clean_polygon(mut pts: Vec<[f64; 2]>, l0: f64) -> Vec<[f64; 2]> {
    let dup = 1e-9 * l0;
    let flat = 1e-12 * l0 * l0;
    loop {
        let m = pts.len();
        if m < 3 {
            return pts;
        }
        let drop = (0..m).find(|&k| {
            let prev = pts[(k + m - 1) % m];
            let cur = pts[k];
            let next = pts[(k + 1) % m];
            (cur[0] - next[0]).hypot(cur[1] - next[1]) <= dup || cross3(prev, cur, next).abs() <= flat
        });
        match drop {
            Some(k) => {
                pts.remove(k);
            }
            None => return pts,
        }
    }
}

/// Voronoi cells of the periodic point set with fractional coordinates `frac_pts` in basis `(a1, a2)`.
/// Returns an empty list for an empty or singular input.
pub fn voronoi_cells(frac_pts: &[[f64; 2]], a1: [f64; 2], a2: [f64; 2]) -> Vec<VoronoiCell2D> {
    let Some(pc) = PeriodicVoronoi::build(frac_pts, a1, a2) else {
        return Vec::new();
    };
    frac_pts
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let center = [f[0] * a1[0] + f[1] * a2[0], f[0] * a1[1] + f[1] * a2[1]];
            let w = pc.cart_pts[i];
            let off = [center[0] - w[0], center[1] - w[1]];
            let raw: Vec<[f64; 2]> = pc.polys[i].iter().map(|v| [v.pos[0] + off[0], v.pos[1] + off[1]]).collect();
            VoronoiCell2D { site: i, center, vertices: clean_polygon(raw, pc.l0) }
        })
        .collect()
}

fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    let m = poly.len();
    0.5 * (0..m).map(|k| poly[k][0] * poly[(k + 1) % m][1] - poly[(k + 1) % m][0] * poly[k][1]).sum::<f64>()
}

/// Voronoi cells of the sites of `cell` (Cartesian basis positions). Errors on non-finite or singular
/// input and when the cells do not tile the unit cell (coincident sites, modulo the lattice, each get a
/// whole cell from the shared clipping, so their areas over-count).
pub fn voronoi_cells_of_cell(cell: &Cell2D) -> Result<Vec<VoronoiCell2D>, String> {
    let finite = [cell.a1, cell.a2].iter().chain(cell.basis.iter()).all(|v| v[0].is_finite() && v[1].is_finite());
    if !finite {
        return Err("non-finite cell".to_string());
    }
    let a_inv = mat2_inv(cell.mat_a())?;
    let frac: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
    let cells = voronoi_cells(&frac, cell.a1, cell.a2);
    if cells.len() != cell.basis.len() {
        return Err("Voronoi construction failed".to_string());
    }
    let det = mat2_det(cell.mat_a()).abs();
    let total: f64 = cells.iter().map(|c| polygon_area(&c.vertices)).sum();
    if cells.iter().any(|c| c.vertices.len() < 3) || (total - det).abs() > 1e-7 * det {
        return Err(format!("Voronoi cells do not tile the cell (area {total} vs {det}): coincident sites?"));
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plane::stored_lattices::stored_lattices;

    fn area(poly: &[[f64; 2]]) -> f64 {
        polygon_area(poly)
    }

    fn unit_square(basis: Vec<[f64; 2]>) -> Cell2D {
        Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis, bonds: Vec::new() }
    }

    #[test]
    fn coincident_sites_are_errors() {
        // Same point twice, and the same point modulo the lattice: no tiling exists.
        assert!(voronoi_cells_of_cell(&unit_square(vec![[0.0, 0.0], [0.0, 0.0]])).is_err());
        assert!(voronoi_cells_of_cell(&unit_square(vec![[0.0, 0.0], [1.0, 0.0]])).is_err());
        // Checkerboard (centred square lattice of spacing 1/sqrt2): two squares of area 1/2.
        let cells = voronoi_cells_of_cell(&unit_square(vec![[0.0, 0.0], [0.5, 0.5]])).unwrap();
        assert_eq!(cells.len(), 2);
        for c in &cells {
            assert_eq!(c.vertices.len(), 4, "{:?}", c.vertices);
            assert!((area(&c.vertices) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn square_lattice_cell_is_unit_square() {
        let cells = voronoi_cells(&[[0.0, 0.0]], [1.0, 0.0], [0.0, 1.0]);
        assert_eq!(cells.len(), 1);
        let v = &cells[0].vertices;
        assert_eq!(v.len(), 4, "{v:?}");
        assert!((area(v) - 1.0).abs() < 1e-12);
        for p in v {
            assert!((p[0].abs() - 0.5).abs() < 1e-12 && (p[1].abs() - 0.5).abs() < 1e-12, "{p:?}");
        }
    }

    #[test]
    fn triangular_lattice_cell_is_regular_hexagon() {
        let s3 = 3.0_f64.sqrt();
        // Unwrapped site (4.3, -2.7): the cell must surround the original Cartesian position.
        let f = [4.3, -2.7];
        let (a1, a2) = ([1.0, 0.0], [0.5, 0.5 * s3]);
        let cells = voronoi_cells(&[f], a1, a2);
        let c = &cells[0];
        let want = [f[0] + 0.5 * f[1], 0.5 * s3 * f[1]];
        assert!((c.center[0] - want[0]).abs() < 1e-12 && (c.center[1] - want[1]).abs() < 1e-12);
        assert_eq!(c.vertices.len(), 6);
        assert!((area(&c.vertices) - 0.5 * s3).abs() < 1e-12);
        for p in &c.vertices {
            let r = (p[0] - c.center[0]).hypot(p[1] - c.center[1]);
            assert!((r - 1.0 / s3).abs() < 1e-12, "circumradius {r}");
        }
    }

    #[test]
    fn honeycomb_cells_are_two_half_area_triangles() {
        let s3 = 3.0_f64.sqrt();
        let cells = voronoi_cells(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]], [1.0, 0.0], [0.5, 0.5 * s3]);
        assert_eq!(cells.len(), 2);
        for c in &cells {
            assert_eq!(c.vertices.len(), 3, "{:?}", c.vertices);
            assert!((area(&c.vertices) - 0.25 * s3).abs() < 1e-12);
        }
    }

    #[test]
    fn stored_lattices_voronoi_cells_tile_the_cell() {
        let t0 = std::time::Instant::now();
        let all = stored_lattices();
        let mut checked = 0usize;
        for lat in all {
            let cell = &lat.cell;
            let det = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs();
            let cells = voronoi_cells_of_cell(cell).unwrap_or_else(|e| panic!("{}: {e}", lat.label));
            let l0 = (det / cell.basis.len() as f64).sqrt();
            let tol = 1e-6 * l0;
            let total: f64 = cells.iter().map(|c| area(&c.vertices)).sum();
            assert!((total - det).abs() < 1e-9 * det, "{}: area {total} vs {det}", lat.label);
            let mut images = Vec::new();
            for n0 in -4..=4 {
                for n1 in -4..=4 {
                    for b in &cell.basis {
                        let x = f64::from(n0);
                        let y = f64::from(n1);
                        images.push([b[0] + x * cell.a1[0] + y * cell.a2[0], b[1] + x * cell.a1[1] + y * cell.a2[1]]);
                    }
                }
            }
            for c in &cells {
                let v = &c.vertices;
                let m = v.len();
                assert!(m >= 3, "{}: degenerate cell", lat.label);
                for k in 0..m {
                    let turn = cross3(v[k], v[(k + 1) % m], v[(k + 2) % m]);
                    assert!(turn > 0.0, "{}: not strictly convex CCW", lat.label);
                    assert!(cross3(v[k], v[(k + 1) % m], c.center) > 0.0, "{}: site outside", lat.label);
                }
                for p in v {
                    let d0 = (p[0] - c.center[0]).hypot(p[1] - c.center[1]);
                    let mut ties = 0usize;
                    for q in &images {
                        let d = (p[0] - q[0]).hypot(p[1] - q[1]);
                        assert!(d > d0 - tol, "{}: a site is closer than the owner", lat.label);
                        if (d - d0).abs() < tol {
                            ties += 1;
                        }
                    }
                    // `ties` counts the owner itself plus at least two other sites.
                    assert!(ties >= 3, "{}: vertex has only {ties} equidistant sites", lat.label);
                }
            }
            checked += 1;
        }
        println!("# time voronoi cells over {checked} stored lattices: {:?}", t0.elapsed());
        assert!(checked >= 500, "only {checked} lattices");
    }
}
