//! Geometry helpers of the Atlas 3D space-group view: the lattice frame of the drawn block (fractional coordinates),
//! the z-slice slab, coordination numbers, and the unit-cell box / coordinate-axes wires.
//!
//! `papertilling_core::space3d::nets3d::build_net_3d` does not export its lattice, so [`net_frame`] restates the
//! conventional lattice rows of each `CanonicalNet3d` and the block placement of `build_net_3d` (`nx = ny = 2` for a
//! pristine net, 3 with a line defect; `nz = 2`; Cartesian `p = f . L - (mid a1x, mid a2y, c)` with `mid = nx / 2`
//! and `c = a3z`). The unit test checks it against the core positions (pristine motifs land on their textbook
//! fractional coordinates). Under a line defect the positions are warped, so fractional coordinates are nominal.

use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};

/// Lattice rows `a1, a2, a3` (Cartesian) and the Cartesian origin of fractional `(0, 0, 0)` (first cell of the block).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetFrame {
    pub lat: [[f64; 3]; 3],
    pub origin: [f64; 3],
}

const SQRT3_2: f64 = 0.8660254;

pub fn net_lattice(net: CanonicalNet3d) -> [[f64; 3]; 3] {
    let cubic = |c: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, c]];
    let hex = [[1.0, 0.0, 0.0], [-0.5, SQRT3_2, 0.0], [0.0, 0.0, 1.35]];
    match net {
        CanonicalNet3d::HCP | CanonicalNet3d::BetaQuartz => hex,
        CanonicalNet3d::Tetragonal123 => cubic(1.15),
        CanonicalNet3d::Rutile136 => cubic(0.72),
        _ => cubic(1.0),
    }
}

pub fn net_frame(net: CanonicalNet3d, defect: LineDefect3d) -> NetFrame {
    let lat = net_lattice(net);
    let mid = if defect == LineDefect3d::None { 1.0 } else { 1.5 };
    NetFrame { lat, origin: [-mid * lat[0][0], -mid * lat[1][1], -lat[2][2]] }
}

impl NetFrame {
    /// Fractional coordinates `f` with `p = origin + f . lat` (Cramer's rule on the 3x3 lattice).
    pub fn fractional(&self, p: [f64; 3]) -> [f64; 3] {
        let q: [f64; 3] = core::array::from_fn(|k| p[k] - self.origin[k]);
        let l = &self.lat;
        // q = f0 a1 + f1 a2 + f2 a3: columns of M are a1, a2, a3.
        let det3 = |c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]| {
            c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
        };
        let d = det3(l[0], l[1], l[2]);
        if d.abs() < 1e-12 {
            return [0.0; 3];
        }
        [det3(q, l[1], l[2]) / d, det3(l[0], q, l[2]) / d, det3(l[0], l[1], q) / d]
    }

    /// The 12 edges of the unit cell at fractional origin `(0, 0, 0)`.
    pub fn cell_edges(&self) -> Vec<([f64; 3], [f64; 3])> {
        let corner = |i: usize, j: usize, k: usize| -> [f64; 3] {
            core::array::from_fn(|c| {
                self.origin[c] + i as f64 * self.lat[0][c] + j as f64 * self.lat[1][c] + k as f64 * self.lat[2][c]
            })
        };
        let mut out = Vec::with_capacity(12);
        for a in 0..2 {
            for b in 0..2 {
                out.push((corner(0, a, b), corner(1, a, b)));
                out.push((corner(a, 0, b), corner(a, 1, b)));
                out.push((corner(a, b, 0), corner(a, b, 1)));
            }
        }
        out
    }

    /// Cartesian x, y, z axes (3 segments of length `len`) from the cell origin.
    pub fn axes(&self, len: f64) -> [([f64; 3], [f64; 3]); 3] {
        core::array::from_fn(|k| {
            let mut tip = self.origin;
            tip[k] += len;
            (self.origin, tip)
        })
    }
}

/// Fractional z of each site in the drawn block: `(z - z_min) / (z_max - z_min)` over the block's sites (all 0
/// when the block is flat), so the full range `[0, 1]` shows every site.
pub fn z_fractions(positions: &[[f64; 3]]) -> Vec<f64> {
    let (lo, hi) = positions.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p[2]), hi.max(p[2])));
    let span = hi - lo;
    positions.iter().map(|p| if span > 1e-12 { (p[2] - lo) / span } else { 0.0 }).collect()
}

/// Sites whose fractional z lies in `range` (inclusive, 1e-6 tolerance; the ends may come in either order).
pub fn slab_mask(positions: &[[f64; 3]], range: [f32; 2]) -> Vec<bool> {
    let (lo, hi) = (range[0].min(range[1]) as f64 - 1e-6, range[0].max(range[1]) as f64 + 1e-6);
    z_fractions(positions).into_iter().map(|t| (lo..=hi).contains(&t)).collect()
}

/// Number of bonds at each site (bonds of the whole drawn block, seam bonds included).
pub fn coordination(n_sites: usize, bonds: &[(usize, usize, u8)]) -> Vec<usize> {
    let mut c = vec![0; n_sites];
    for &(u, v, _) in bonds {
        if u < n_sites && v < n_sites {
            c[u] += 1;
            c[v] += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d, build_net_3d};

    use super::*;

    fn frac_part(x: f64) -> f64 {
        x - x.floor()
    }

    #[test]
    fn lattice_frame_puts_textbook_motifs_on_their_fractional_coordinates() {
        // Simple cubic: one site per cell at the cell corner, so every fractional coordinate is an integer.
        let sc = build_net_3d(CanonicalNet3d::SimpleCubic, LineDefect3d::None);
        let fr = net_frame(CanonicalNet3d::SimpleCubic, LineDefect3d::None);
        for p in &sc.positions {
            let f = fr.fractional(*p);
            assert!(f.iter().all(|x| (x - x.round()).abs() < 1e-9), "{f:?}");
        }
        // Diamond: fcc + (1/4, 1/4, 1/4) basis, every coordinate a multiple of 1/4.
        let di = build_net_3d(CanonicalNet3d::Diamond, LineDefect3d::None);
        let fr = net_frame(CanonicalNet3d::Diamond, LineDefect3d::None);
        for p in &di.positions {
            let f = fr.fractional(*p);
            assert!(f.iter().all(|x| (4.0 * x - (4.0 * x).round()).abs() < 1e-9), "{f:?}");
        }
        // HCP: (1/3, 2/3, 1/4) and (2/3, 1/3, 3/4).
        let h = build_net_3d(CanonicalNet3d::HCP, LineDefect3d::None);
        let fr = net_frame(CanonicalNet3d::HCP, LineDefect3d::None);
        for p in &h.positions {
            let f = fr.fractional(*p).map(frac_part);
            let a = [1.0 / 3.0, 2.0 / 3.0, 0.25];
            let b = [2.0 / 3.0, 1.0 / 3.0, 0.75];
            let near = |m: [f64; 3]| (0..3).all(|k| (f[k] - m[k]).abs() < 1e-6);
            assert!(near(a) || near(b), "{f:?}");
        }
    }

    #[test]
    fn slab_keeps_exactly_the_simple_cubic_layers_in_range() {
        // 2x2x2 simple-cubic block: 8 sites in two z layers (fractional z 0 and 1), 4 per layer.
        let sc = build_net_3d(CanonicalNet3d::SimpleCubic, LineDefect3d::None);
        assert_eq!(sc.positions.len(), 8);
        let count = |r: [f32; 2]| slab_mask(&sc.positions, r).iter().filter(|&&k| k).count();
        assert_eq!(count([0.0, 1.0]), 8);
        assert_eq!(count([0.0, 0.5]), 4);
        assert_eq!(count([0.5, 1.0]), 4);
        assert_eq!(count([0.25, 0.75]), 0);
        // the kept sites are exactly those of the lower layer
        let lo = sc.positions.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        for (p, keep) in sc.positions.iter().zip(slab_mask(&sc.positions, [0.0, 0.5])) {
            assert_eq!(keep, (p[2] - lo).abs() < 1e-9);
        }
    }

    #[test]
    fn coordination_numbers_of_textbook_nets() {
        // Hand-built 3x3x3 simple cubic block with unit bonds: the centre site has 6 neighbours (the Atlas SC
        // block is 2x2x2 and has no interior site: every site there has 3).
        let idx = |i: usize, j: usize, k: usize| 9 * i + 3 * j + k;
        let mut bonds = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    if i + 1 < 3 { bonds.push((idx(i, j, k), idx(i + 1, j, k), 0)); }
                    if j + 1 < 3 { bonds.push((idx(i, j, k), idx(i, j + 1, k), 0)); }
                    if k + 1 < 3 { bonds.push((idx(i, j, k), idx(i, j, k + 1), 0)); }
                }
            }
        }
        assert_eq!(bonds.len(), 54); // 3 directions x 3 x 3 x 2
        assert_eq!(coordination(27, &bonds)[idx(1, 1, 1)], 6);
        let sc = build_net_3d(CanonicalNet3d::SimpleCubic, LineDefect3d::None);
        assert!(coordination(sc.positions.len(), &sc.bonds).iter().all(|&c| c == 3));

        // Atlas nets: the site at fractional (1, 1, 1) of the fcc block has its 12 nearest neighbours inside the
        // 2x2x2 block; the diamond site at fractional (1/4, 1/4, 1/4) has its 4 (the fcc corners (0,0,0),
        // (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)).
        for (net, f, want) in [
            (CanonicalNet3d::FCC, [1.0, 1.0, 1.0], 12),
            (CanonicalNet3d::Diamond, [0.25, 0.25, 0.25], 4),
            (CanonicalNet3d::BCC, [0.5, 0.5, 0.5], 8),
        ] {
            let patch = build_net_3d(net, LineDefect3d::None);
            let fr = net_frame(net, LineDefect3d::None);
            let site = patch
                .positions
                .iter()
                .position(|p| fr.fractional(*p).iter().zip(f).all(|(a, b)| (a - b).abs() < 1e-6))
                .expect("site present");
            let c = coordination(patch.positions.len(), &patch.bonds);
            assert_eq!(c[site], want, "{net:?}");
            assert_eq!(*c.iter().max().unwrap(), want, "{net:?}: no site exceeds the bulk coordination");
        }
    }

    #[test]
    fn unit_cell_has_twelve_edges_of_the_lattice_lengths() {
        let fr = net_frame(CanonicalNet3d::Tetragonal123, LineDefect3d::None);
        let e = fr.cell_edges();
        assert_eq!(e.len(), 12);
        let len = |s: &([f64; 3], [f64; 3])| (0..3).map(|k| (s.1[k] - s.0[k]).powi(2)).sum::<f64>().sqrt();
        // tetragonal a = b = 1, c = 1.15: 8 edges of length 1, 4 of length 1.15
        assert_eq!(e.iter().filter(|s| (len(s) - 1.0).abs() < 1e-9).count(), 8);
        assert_eq!(e.iter().filter(|s| (len(s) - 1.15).abs() < 1e-9).count(), 4);
        assert_eq!(fr.axes(1.0).len(), 3);
    }
}
