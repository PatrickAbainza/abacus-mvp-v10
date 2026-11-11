//! `extract` + directional reductions (rows 32–34; v5 #1–#3).
//!
//! Ported from v7 `graphblas/primitives/extract.rs` (itself from v5).
//! Orientation: `A[i,j] = 1` iff sender `i -> `receiver `j`. Incoming
//! support at `v` is the **column**, outgoing at `u` is the **row**.
//! Absent goes to `None` (extract) / `Ok(0)` (reductions); overflow goes to
//! `TensorError::Range` (`E_rng`).

use super::super::matrix::{checked_add, PairMatrix};
use crate::TensorError;

/// `extract` (#1, row 32): value of `A(s, r)`, or `None` when absent.
#[must_use]
pub fn extract(a: &PairMatrix, s: u32, r: u32) -> Option<i64> {
    a.get(s, r)
}

/// `reduce_col_plus` (#2, row 33): checked sum of column `v` (all senders
/// into the receiver). Absent column goes to `Ok(0)`.
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (never wraps).
pub fn reduce_col_plus(a: &PairMatrix, v: u32) -> Result<i64, TensorError> {
    let mut acc = 0i64;
    for (_s, r, val) in a.nonzeros() {
        if r == v {
            acc = checked_add(acc, val)?;
        }
    }
    Ok(acc)
}

/// `reduce_row_plus` (#3, row 34): checked sum of row `u` (sender into all
/// receivers).
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (never wraps).
pub fn reduce_row_plus(a: &PairMatrix, u: u32) -> Result<i64, TensorError> {
    let mut acc = 0i64;
    for (s, _r, val) in a.nonzeros() {
        if s == u {
            acc = checked_add(acc, val)?;
        }
    }
    Ok(acc)
}
