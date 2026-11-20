//! The one table of the 10 canonical 3D net motifs (C1 WP-N1).
//!
//! Every user of a canonical net (`nets3d::build_net_3d`, `flat3d::Cell3D::from_canonical_net`) reads this table,
//! and the pure-Python mirror is `papertilling/_net_motifs.py` (same sites in the same order).
//!
//! Site-kind convention: a label names the *coloured* structure, and the kinds are chosen so that the exact
//! space group of the coloured motif equals the label's group (checked with spglib in `tests/test_net_motifs.py`):
//! - BCC, Diamond, Pyrochlore, FCC, SC, HCP and beta-quartz carry one kind (two kinds would give CsCl `Pm-3m` or
//!   zincblende `F-43m`);
//! - Tetragonal123 keeps two kinds (one kind at `0` and `1/2` would be body-centred `I4/mmm`, #139);
//! - Rutile: Ti (kind 0) at 2a, O (kind 1) at 4f with `x = 0.3`;
//! - Perovskite: kind 0 at `(0,0,0)` (octahedral B site), kind 1 at `(1/2,1/2,1/2)` (A site), kind 2 = O at 3d.
//!
//! Pyrochlore is the 16c orbit of `Fd-3m` (origin choice 2 at `(0,0,0)`): the midpoints of the 16 nearest-neighbour
//! bonds of the diamond motif below, i.e. `(1/8,1/8,1/8)`, `(1/8,7/8,7/8)`, `(7/8,1/8,7/8)`, `(7/8,7/8,1/8)` plus the
//! F-centring. beta-quartz (Si only) is the 3c orbit of `P6_222` (#180): `(1/2,0,0)`, `(0,1/2,2/3)`, `(1/2,1/2,1/3)`.
//! (The pre-N1 set with `z = 1/3, 2/3` swapped is the 3c orbit of `P6_422`, #181.)

use super::nets3d::CanonicalNet3d;

/// Lattice rows (Cartesian), sites `(fractional position, kind)` and the bond cutoff of one canonical net.
#[derive(Clone, Debug, PartialEq)]
pub struct NetMotif {
    pub lattice: [[f64; 3]; 3],
    pub sites: Vec<([f64; 3], u8)>,
    pub bond_cutoff: f64,
}

/// Hexagonal `c` of HCP and beta-quartz (`a = 1`).
pub const HEX_C: f64 = 1.35;

/// F-centring translations of a cubic conventional cell.
const F_CENTRING: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];

fn cubic(c: f64) -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, c]]
}

/// Hexagonal rows `a1 = (1,0,0)`, `a2 = (-1/2, sqrt(3)/2, 0)`, `a3 = (0,0,c)` (gamma = 120 degrees).
#[must_use]
pub fn hex_lattice() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [-0.5, 0.75_f64.sqrt(), 0.0], [0.0, 0.0, HEX_C]]
}

/// F-centred copies of `base` (centring-major order), reduced into `[0, 1)`.
fn f_centred(base: &[[f64; 3]], kind: u8) -> Vec<([f64; 3], u8)> {
    F_CENTRING
        .iter()
        .flat_map(|t| base.iter().map(move |b| (core::array::from_fn(|k| (b[k] + t[k]).rem_euclid(1.0)), kind)))
        .collect()
}

/// The motif of `net`. Sites are listed in a fixed order shared with the Python mirror.
#[must_use]
pub fn net_motif(net: CanonicalNet3d) -> NetMotif {
    let (lattice, sites, bond_cutoff) = match net {
        CanonicalNet3d::SimpleCubic => (cubic(1.0), vec![([0.0, 0.0, 0.0], 0)], 1.05),
        CanonicalNet3d::BCC => (cubic(1.0), vec![([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 0)], 0.92),
        CanonicalNet3d::FCC => (cubic(1.0), f_centred(&[[0.0, 0.0, 0.0]], 0), 0.76),
        CanonicalNet3d::Diamond => {
            let mut s = f_centred(&[[0.0, 0.0, 0.0]], 0);
            s.extend([[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]].map(|p| (p, 0)));
            (cubic(1.0), s, 0.48)
        }
        CanonicalNet3d::HCP => (
            hex_lattice(),
            vec![([1.0 / 3.0, 2.0 / 3.0, 0.25], 0), ([2.0 / 3.0, 1.0 / 3.0, 0.75], 0)],
            1.02,
        ),
        CanonicalNet3d::Tetragonal123 => (cubic(1.15), vec![([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)], 0.96),
        CanonicalNet3d::Rutile136 => (
            cubic(0.72),
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.5], 0),
                ([0.3, 0.3, 0.0], 1),
                ([0.7, 0.7, 0.0], 1),
                ([0.8, 0.2, 0.5], 1),
                ([0.2, 0.8, 0.5], 1),
            ],
            0.48,
        ),
        CanonicalNet3d::Perovskite => (
            cubic(1.0),
            vec![
                ([0.0, 0.0, 0.0], 0),
                ([0.5, 0.5, 0.5], 1),
                ([0.5, 0.0, 0.0], 2),
                ([0.0, 0.5, 0.0], 2),
                ([0.0, 0.0, 0.5], 2),
            ],
            0.55,
        ),
        CanonicalNet3d::Pyrochlore => {
            let base = [[0.125, 0.125, 0.125], [0.125, 0.875, 0.875], [0.875, 0.125, 0.875], [0.875, 0.875, 0.125]];
            (cubic(1.0), f_centred(&base, 0), 0.42)
        }
        CanonicalNet3d::BetaQuartz => (
            hex_lattice(),
            vec![([0.5, 0.0, 0.0], 0), ([0.0, 0.5, 2.0 / 3.0], 0), ([0.5, 0.5, 1.0 / 3.0], 0)],
            0.72,
        ),
    };
    NetMotif { lattice, sites, bond_cutoff }
}

#[cfg(test)]
#[path = "motifs_tests.rs"]
mod tests;
