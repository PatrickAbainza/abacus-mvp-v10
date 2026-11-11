//! Sparse `S x R` pair matrix — derived projection of the canonical 3D CSF
//! tensor (`your_task_phase_7.md` section 4; Phase 6 report section 1).
//!
//! ```text
//! A_w(s, r) := sum_{t in w} T(t, s, r)   (mode-0 slice + contraction)
//! ```
//!
//! Ported from `abacus_mvp_v7/.../graphblas/matrix.rs` (itself ported from
//! v5 `abacus_canonical_execution::matrix`) with one mechanical adaptation:
//! the v5 unit `Overflow` is `crate::TensorError::Range` (`E_rng`; P2/R5
//! fail-closed). Checked semantics identical: represent exactly or reject,
//! never wrap/saturate; zero sums dropped (P3). Storage: canonical `(s, r)`
//! ordered unique pairs (COO order, restriction of `prec_3`), never CSC.
//!
//! Golden Source: `golden_source_abacus_tensor_atomic_primitive_derivation.md`
//! Addendum O, O.1/O.2 rows 30/39/40.

use crate::prims::add;
use crate::TensorError;

/// Sparse pair matrix in canonical `(s, r)` order (COO, no CSC structure).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PairMatrix {
    indices: Vec<(u32, u32)>,
    vals: Vec<i64>,
}

/// Build a pair matrix from `(s, r, value)` occurrences: canonical order,
/// unique coordinates, **checked** value sums (O.2 rows 30/39/40).
///
/// # Errors
/// `TensorError::Range` (`E_rng`) when any coordinate sum is unrepresentable.
pub fn from_occurrences(occ: &[(u32, u32, i64)]) -> Result<PairMatrix, TensorError> {
    let mut v = occ.to_vec();
    v.sort_unstable();
    let mut out = PairMatrix::default();
    let mut cur: Option<((u32, u32), i64)> = None;
    for (k0, k1, val) in v {
        cur = match cur.take() {
            None => Some(((k0, k1), val)),
            Some((ck, cv)) if ck == (k0, k1) => Some((ck, checked_add(cv, val)?)),
            Some((ck, cv)) => {
                if cv != 0 {
                    out.indices.push(ck);
                    out.vals.push(cv);
                }
                Some(((k0, k1), val))
            }
        };
    }
    if let Some((ck, cv)) = cur {
        if cv != 0 {
            out.indices.push(ck);
            out.vals.push(cv);
        }
    }
    Ok(out)
}

/// Checked accumulation of one more value (`a_{02}`; O.2 rows 33/34/42).
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (never wraps).
pub(crate) fn checked_add(acc: i64, v: i64) -> Result<i64, TensorError> {
    add(acc, v, "graphblas checked sum (O.2/P2)")
}

/// Backwards-compatible alias: v5 callers named the overflow error
/// `Overflow`; here it is `TensorError::Range` (`E_rng`).
pub type Overflow = TensorError;

impl PairMatrix {
    /// Zero matrix.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Number of nonzero coordinates.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// True iff no nonzeros.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Canonical nonzeros (debug-checked order).
    pub fn nonzeros(&self) -> impl Iterator<Item = (u32, u32, i64)> + '_ {
        debug_assert!(
            self.indices.windows(2).all(|w| w[0] < w[1]),
            "PairMatrix order invariant violated"
        );
        self.indices
            .iter()
            .copied()
            .zip(self.vals.iter().copied())
            .map(|((s, r), v)| (s, r, v))
    }

    /// `extract` (#1, row 32): `A(s, r)` or `None` when absent (`a_{11}`
    /// binary search; P3 absent reported absent).
    #[must_use]
    pub fn get(&self, s: u32, r: u32) -> Option<i64> {
        self.indices
            .binary_search(&(s, r))
            .ok()
            .map(|i| self.vals[i])
    }

    /// Insert/overwrite one nonzero (zero drops the entry; P3 support).
    pub fn insert(&mut self, s: u32, r: u32, v: i64) {
        match self.indices.binary_search(&(s, r)) {
            Ok(i) => {
                if v == 0 {
                    self.indices.remove(i);
                    self.vals.remove(i);
                } else {
                    self.vals[i] = v;
                }
            }
            Err(pos) => {
                if v != 0 {
                    self.indices.insert(pos, (s, r));
                    self.vals.insert(pos, v);
                }
            }
        }
    }

    /// Linear-time existence map of `self`: every nonzero becomes `1`,
    /// order preserved (O.2 row 38; same result as inserting each pair
    /// one-by-one, without the quadratic `Vec::insert` memmoves).
    #[must_use]
    pub fn support_pattern(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            vals: vec![1; self.vals.len()],
        }
    }

    /// Wrap already-canonical sorted-unique pairs (restriction of `prec_3`
    /// to `(s, r)`; P3 order, P2-checked by the caller, zero-drop by the
    /// caller). Debug-checked order/uniqueness; no re-sort, no re-dedup.
    #[must_use]
    pub fn from_sorted_unique(indices: Vec<(u32, u32)>, vals: Vec<i64>) -> Self {
        debug_assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "from_sorted_unique: indices must be sorted unique"
        );
        debug_assert_eq!(
            indices.len(),
            vals.len(),
            "from_sorted_unique: indices/vals length mismatch"
        );
        Self { indices, vals }
    }

    /// `senders_to(v)`: pattern of the sparse column `A[:, v]` (canonical
    /// order). Scan over the row-major array with a receiver mask — a sparse
    /// column projection, not a CSC column fiber.
    #[must_use]
    pub fn senders_to(&self, v: u32) -> Vec<u32> {
        self.indices
            .iter()
            .filter_map(|&(s, r)| (r == v).then_some(s))
            .collect()
    }

    /// `receivers_from(u)`: pattern of the sparse row `A[u, :]` (canonical
    /// order). The row block is contiguous because storage is sender-major.
    #[must_use]
    pub fn receivers_from(&self, u: u32) -> Vec<u32> {
        let lo = self.indices.partition_point(|&(s, _)| s < u);
        let hi = self.indices.partition_point(|&(s, _)| s <= u);
        self.indices[lo..hi].iter().map(|&(_, r)| r).collect()
    }
}
