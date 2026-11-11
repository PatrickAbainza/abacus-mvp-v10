//! Row 10 (§B.2) + rows 4/5: direct-sum evaluation of a raw entry sequence.
//!
//! `⟦C⟧(x) = Σ { v : (x, v) ∈ C }` (P4 semantics for a raw sequence). The full
//! decode is [`crate::tensor::decode_sum`]; this module adds the single-
//! coordinate evaluation used by the frontier's diagnostic reads.

use crate::coo::entry::TxEntry;
use crate::prims::add;
use crate::tensor::Tensor;
use crate::TensorError;

/// Row 10 — full direct-sum decode of a raw sequence (settling, P4).
///
/// # Errors
/// [`TensorError::Range`] if any coordinate sum is not representable.
pub fn decode_to_tensor(entries: &[TxEntry]) -> Result<Tensor, TensorError> {
    crate::tensor::tensor::decode_sum(entries)
}

/// Row 10 — `⟦C⟧(x)` at one coordinate: `R_3` scan + `a_{01}` equality +
/// `a_{02}` accumulation, all checked (P2).
///
/// # Errors
/// [`TensorError::Range`] if the coordinate sum is not representable.
pub fn try_eval_at(entries: &[TxEntry], t: u64, s: u32, r: u32) -> Result<i64, TensorError> {
    let mut acc = 0i64;
    for e in entries {
        if e.t == t && e.s == s && e.r == r {
            acc = add(acc, e.value, "eval_at sum (row 10)")?;
        }
    }
    Ok(acc)
}

/// Row 10 — panicking convenience wrapper over [`try_eval_at`].
///
/// Excluded from the total-function claim of P12 exactly as recorded in `K4`;
/// production paths MUST use [`try_eval_at`].
///
/// # Panics
/// Panics if the coordinate sum is not representable in `i64`.
#[must_use]
pub fn eval_at(entries: &[TxEntry], t: u64, s: u32, r: u32) -> i64 {
    try_eval_at(entries, t, s, r).expect("eval_at: sum overflow (K4 convenience wrapper)")
}
