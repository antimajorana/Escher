//! Affine Seitz holonomies in crystallographic coordinates, before lifting the rotations.

use crate::bieberbach::rat::{IMat, RMat, imat_det, imat_eye, imat_mul, rmat_inv, rmat_vec};
use crate::bieberbach::{IntMat, Rat, smith_normal_form, solve_int_linear};

use super::{layer_group_entry, layer_translation_basis, space_group_3d_entry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineHolonomy {
    linear: IMat,
    shift: Vec<Rat>,
}

impl AffineHolonomy {
    pub fn identity() -> Self {
        Self {
            linear: imat_eye(3),
            shift: vec![Rat::ZERO; 3],
        }
    }

    fn translation(shift: Vec<Rat>) -> Self {
        Self {
            shift,
            ..Self::identity()
        }
    }

    pub fn linear(&self) -> &[Vec<i64>] {
        &self.linear
    }

    pub fn shift(&self) -> &[Rat] {
        &self.shift
    }

    pub fn is_proper(&self) -> bool {
        imat_det(&self.linear) == 1
    }

    /// The in-plane action of a layer operation can have the opposite orientation to its 3D action.
    pub fn preserves_layer_orientation(&self) -> bool {
        self.linear[0][0] * self.linear[1][1] - self.linear[0][1] * self.linear[1][0] == 1
    }

    /// `a.compose(b)` acts as `a(b(x))`.
    pub fn compose(&self, rhs: &Self) -> Self {
        let shift = self
            .linear
            .iter()
            .zip(&self.shift)
            .map(|(row, &lhs_t)| {
                row.iter()
                    .zip(&rhs.shift)
                    .fold(lhs_t, |sum, (&c, &t)| sum + Rat::from_int(c) * t)
            })
            .collect();
        Self {
            linear: imat_mul(&self.linear, &rhs.linear),
            shift,
        }
    }

    pub fn inverse(&self) -> Self {
        // The point part of every constructed holonomy has finite order.
        let linear = rmat_inv(&rational_matrix(&self.linear)).expect("invertible point operation");
        let shift = rmat_vec(&linear, &self.shift)
            .into_iter()
            .map(|t| Rat::ZERO - t)
            .collect();
        let linear = linear
            .iter()
            .map(|row| {
                row.iter()
                    .map(|n| n.as_i64().expect("unimodular"))
                    .collect()
            })
            .collect();
        Self { linear, shift }
    }

    pub fn conjugate(&self, op: &Self) -> Self {
        self.compose(op).compose(&self.inverse())
    }

    /// Power that removes the point part, and the exact conventional translation it leaves.
    pub fn closing_power(&self) -> (usize, Self) {
        let identity = imat_eye(3);
        let mut power = 1;
        let mut op = self.clone();
        while op.linear != identity {
            op = op.compose(self);
            power += 1;
        }
        (power, op)
    }

    pub fn kind(&self) -> &'static str {
        let (n, power) = self.closing_power();
        let drifts = power.shift.iter().any(|t| !t.is_zero());
        let trace = (0..3).map(|i| self.linear[i][i]).sum::<i64>();
        match (n, self.is_proper(), trace, drifts) {
            (1, _, _, false) => "Identity",
            (1, _, _, true) => "Translation",
            (_, true, _, false) => "Rotation",
            (_, true, _, true) => "Screw rotation",
            (2, false, 1, false) => "Reflection",
            (2, false, 1, true) => "Glide reflection",
            (2, false, -3, _) => "Inversion",
            _ => "Rotoreflection",
        }
    }
}

fn rational_matrix(matrix: &IMat) -> RMat {
    matrix
        .iter()
        .map(|row| row.iter().copied().map(Rat::from_int).collect())
        .collect()
}

#[derive(Clone, Debug)]
pub struct CrystalHolonomies {
    /// Conventional-coordinate affine representatives, one per point operation modulo primitive translations.
    reps: Vec<AffineHolonomy>,
    basis: RMat,
    basis_inverse: RMat,
    rank: usize,
}

impl CrystalHolonomies {
    pub fn space_group(number: u16) -> Result<Self, String> {
        let entry = space_group_3d_entry(number)?;
        let basis = entry
            .tb_rows
            .iter()
            .map(|v| v.iter().map(|&[n, d]| Rat::new(n, d)).collect())
            .collect();
        let ops = entry.ops.iter().map(|op| AffineHolonomy {
            linear: op.r.clone(),
            shift: op.t.iter().map(|&[n, d]| Rat::new(n, d)).collect(),
        });
        Self::new(basis, ops)
    }

    pub fn layer_group(number: u8) -> Result<Self, String> {
        let entry = layer_group_entry(number)?;
        let basis = layer_translation_basis(entry.centred);
        let ops = entry.ops.iter().map(|op| AffineHolonomy {
            linear: op.r.clone(),
            shift: op
                .t2
                .iter()
                .map(|&[n, d]| Rat::new(n, d))
                .chain([Rat::ZERO])
                .collect(),
        });
        Self::new(basis, ops)
    }

    fn new(rows: RMat, ops: impl IntoIterator<Item = AffineHolonomy>) -> Result<Self, String> {
        let rank = rows.len();
        let basis: RMat = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        if i < rank && j < rank {
                            rows[j][i]
                        } else {
                            Rat::from_int(i64::from(i == j))
                        }
                    })
                    .collect()
            })
            .collect();
        let basis_inverse = rmat_inv(&basis)?;
        let mut reps = vec![AffineHolonomy::identity()];
        for op in ops {
            if reps.iter().all(|r| r.linear != op.linear) {
                reps.push(op);
            }
        }
        Ok(Self {
            reps,
            basis,
            basis_inverse,
            rank,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn representatives(&self) -> &[AffineHolonomy] {
        &self.reps
    }

    pub fn element(&self, index: usize, periods: [i32; 3]) -> Option<AffineHolonomy> {
        // Rank-two groups have no periodic translation in the slab normal.
        if periods[self.rank..].iter().any(|&n| n != 0) {
            return None;
        }
        let t = periods.map(|n| Rat::from_int(i64::from(n)));
        let translation = AffineHolonomy::translation(rmat_vec(&self.basis, &t));
        Some(translation.compose(self.reps.get(index)?))
    }

    fn fixed_translation_matrix(&self, op: &AffineHolonomy) -> IntMat {
        let linear = rational_matrix(&op.linear);
        let mut matrix = IntMat::zeros(3, self.rank);
        for j in 0..self.rank {
            let column: Vec<_> = self.basis.iter().map(|row| row[j]).collect();
            let moved = rmat_vec(&linear, &column);
            let difference: Vec<_> = column.into_iter().zip(moved).map(|(a, b)| a - b).collect();
            for (i, v) in rmat_vec(&self.basis_inverse, &difference)
                .iter()
                .enumerate()
            {
                matrix.set(
                    i,
                    j,
                    i128::from(v.as_i64().expect("group operation preserves its lattice")),
                );
            }
        }
        matrix
    }

    fn contains(&self, op: &AffineHolonomy) -> bool {
        self.reps
            .iter()
            .find(|r| r.linear == op.linear)
            .is_some_and(|r| {
                let delta: Vec<_> = op
                    .shift
                    .iter()
                    .zip(&r.shift)
                    .map(|(&a, &b)| a - b)
                    .collect();
                let t = rmat_vec(&self.basis_inverse, &delta);
                t[..self.rank].iter().all(|t| t.as_i64().is_some())
                    && t[self.rank..].iter().all(|t| t.is_zero())
            })
    }

    /// Change the registry on a mirror plane, keeping its linear action and normal offset.
    /// Remove the tangential drift from a glide, or give a reflection half of a primitive period
    /// in the fixed sublattice. The Boolean reports whether that alternate join is a bulk symmetry.
    pub fn alternate_registry(&self, op: &AffineHolonomy) -> Option<(AffineHolonomy, bool)> {
        if !self.contains(op) || !matches!(op.kind(), "Reflection" | "Glide reflection") {
            return None;
        }
        let half = Rat::new(1, 2);
        let square = op.compose(op);
        let delta = if square.shift.iter().all(|v| v.is_zero()) {
            let fixed = self.fixed_translation_matrix(op);
            let smith = smith_normal_form(&fixed, true);
            let first_kernel_col = smith.diag.len();
            if first_kernel_col == self.rank {
                return None;
            }
            let vector: Vec<_> = (0..3)
                .map(|i| {
                    if i < self.rank {
                        Rat::from_int(smith.v.get(i, first_kernel_col) as i64) * half
                    } else {
                        Rat::ZERO
                    }
                })
                .collect();
            rmat_vec(&self.basis, &vector)
        } else {
            square.shift.iter().map(|&v| Rat::ZERO - half * v).collect()
        };
        let alternate = AffineHolonomy::translation(delta).compose(op);
        let member = self.contains(&alternate);
        Some((alternate, member))
    }

    /// Exact conjugacy by a proper group element in the ambient 3D action.
    /// For each point coset h, test h A h⁻¹ first and then solve (I-R_B) B n = t_B-t_{h A h⁻¹}
    /// over primitive lattice translations n. The lattice rank is two for layer groups.
    pub fn properly_conjugate(&self, a: &AffineHolonomy, b: &AffineHolonomy) -> bool {
        if !self.contains(a) || !self.contains(b) {
            return false;
        }
        if a == b {
            return true;
        }
        self.reps.iter().filter(|r| r.is_proper()).any(|r| {
            let c = r.conjugate(a);
            if c.linear != b.linear {
                return false;
            }
            let delta: Vec<_> = b.shift.iter().zip(c.shift).map(|(&v, u)| v - u).collect();
            // In primitive coordinates (I-R_B) is integral. A translation coset
            // mismatch cannot be made up by an integer change of basepoint.
            let rhs = rmat_vec(&self.basis_inverse, &delta);
            let Some(rhs): Option<Vec<i128>> = rhs
                .into_iter()
                .map(|v| v.as_i64().map(i128::from))
                .collect()
            else {
                return false;
            };
            let matrix = self.fixed_translation_matrix(b);
            solve_int_linear(&matrix, &rhs).is_some()
        })
    }
}

#[cfg(test)]
mod tests;
