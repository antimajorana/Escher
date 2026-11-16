//! Gauss-Newton normal equations `J^T J + mu D` with the sparsity pattern, the value order and the
//! symbolic Cholesky analysis computed once per relaxation instead of once per factorisation.
//!
//! The factor is bitwise identical to `SparseColMat::try_new_from_triplets` + `sp_cholesky` on the same
//! entry sequence: `try_new_from_triplets` is `try_new_from_indices` + `new_from_argsort` (so duplicate
//! entries are summed in the same order), and `sp_cholesky` is `SymbolicLlt::try_new` on the (identical)
//! pattern + `Llt::try_new_with_symbolic`.

use faer::Side;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};

struct Structure {
    sym: SymbolicSparseColMat<usize>,
    order: Argsort<usize>,
    llt: Option<SymbolicLlt<usize>>,
}

impl Structure {
    fn new(ndof: usize, idx: &[Pair<usize, usize>]) -> Option<Self> {
        let (sym, order) = SymbolicSparseColMat::try_new_from_indices(ndof, ndof, idx).ok()?;
        Some(Self { sym, order, llt: None })
    }

    fn factor(&mut self, vals: &[f64]) -> Option<Llt<usize, f64>> {
        let mat = SparseColMat::new_from_argsort(self.sym.clone(), &self.order, vals).ok()?;
        if self.llt.is_none() {
            self.llt = Some(SymbolicLlt::try_new(mat.symbolic(), Side::Lower).ok()?);
        }
        Llt::try_new_with_symbolic(self.llt.clone()?, mat.as_ref(), Side::Lower).ok()
    }
}

/// Cached factorisation of `J^T J` (entry sequence `idx`, values supplied per call in the same order)
/// and of its Levenberg-Marquardt shift `J^T J + mu diag(max(D, 1e-12))` (the shift entries appended).
pub(crate) struct NormalFactor {
    ndof: usize,
    idx: Vec<Pair<usize, usize>>,
    base: Option<Structure>,
    aug: Option<Structure>,
    aug_vals: Vec<f64>,
}

impl NormalFactor {
    pub(crate) fn new(ndof: usize, idx: Vec<Pair<usize, usize>>) -> Self {
        Self { ndof, idx, base: None, aug: None, aug_vals: Vec::new() }
    }

    /// Number of `J^T J` entries (with repetitions) expected in `vals`.
    pub(crate) fn len(&self) -> usize {
        self.idx.len()
    }

    /// Cholesky factor of `J^T J + mu D`, or `None` if it is not positive definite.
    pub(crate) fn factor(&mut self, vals: &[f64], diag: &[f64], mu: f64) -> Option<Llt<usize, f64>> {
        debug_assert_eq!(vals.len(), self.idx.len());
        if mu == 0.0 {
            if self.base.is_none() {
                self.base = Some(Structure::new(self.ndof, &self.idx)?);
            }
            return self.base.as_mut()?.factor(vals);
        }
        if self.aug.is_none() {
            let mut idx = self.idx.clone();
            idx.extend((0..self.ndof).map(|k| Pair::new(k, k)));
            self.aug = Some(Structure::new(self.ndof, &idx)?);
        }
        self.aug_vals.clear();
        self.aug_vals.extend_from_slice(vals);
        self.aug_vals.extend(diag.iter().map(|&dk| mu * dk.max(1e-12)));
        self.aug.as_mut()?.factor(&self.aug_vals)
    }
}
