//! Row 11 (§B.2): canonicalization and the `d_inv` structural checks.
//!
//! `canonicalize` **is** `d_settle` (P4) — it is not a second implementation.
//! The checks are `d_inv`: `R_3` adjacent-window scans over `a_{01}`.

use crate::composites::{settle, strictly_increasing};
use crate::coo::entry::TxEntry;
use crate::TensorError;

/// Row 11 — canonicalize a raw entry sequence: `d_settle` (sort by `prec_3`,
/// sum equal coordinates, drop zero sums, P4 → P3 normal form).
///
/// # Errors
/// [`TensorError::Range`] if any coordinate sum is not representable.
pub fn canonicalize(v: Vec<TxEntry>) -> Result<Vec<TxEntry>, TensorError> {
    settle(v)
}

/// `d_inv` — sortedness under `prec_3`.
#[must_use]
pub fn is_sorted(v: &[TxEntry]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}

/// `d_inv` — canonical form: strictly increasing coordinates (hence unique,
/// P3) and no zero value.
#[must_use]
pub fn is_canonical(v: &[TxEntry]) -> bool {
    !v.iter().any(|e| e.value == 0) && strictly_increasing(v)
}
