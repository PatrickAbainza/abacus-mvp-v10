//! Reductions + support conversion (#7–#13, rows 38–42).
//!
//! Ported from v7 `graphblas/primitives/reduce.rs` (itself from v5 with
//! `Overflow` mapped to `TensorError::Range`):
//!
//! * `support` (#7): `A_one = [A_cnt != 0]` (existence map).
//! * `multiplicity` (#8): `A_cnt` from pair occurrences.
//! * `volume` (#9): `A_vol` from `(pair, amount)` occurrences, checked.
//! * `reduce_count` (#10) / `reduce_count_set` (#11): pattern cardinalities.
//! * `reduce_sum` (#12): checked PLUS-monoid sum (empty goes to `0`).
//! * `reduce_max` (#13): MAX reduction (empty goes to `0`, v5 identity).
//!
//! Governing: P2 (`a_{02}` checked, overflow gives `E_rng`), P3 (`R_6`
//! zero-drop in constructors).

use super::super::matrix::{checked_add, from_occurrences, PairMatrix};
use super::super::vector::Vector;
use crate::TensorError;

/// `support` (#7, row 38): every nonzero of `a_cnt` becomes `1` in `A_one`.
#[must_use]
pub fn support(a_cnt: &PairMatrix) -> PairMatrix {
    let mut out = PairMatrix::zero();
    for (s, r, _v) in a_cnt.nonzeros() {
        out.insert(s, r, 1);
    }
    out
}

/// `multiplicity` (#8, row 39): build `A_cnt` from raw pair occurrences.
/// Counts are bounded by the occurrence list length, so construction is
/// infallible by the same argument as v5.
#[must_use]
pub fn multiplicity(pairs: &[(u32, u32)]) -> PairMatrix {
    let occ: Vec<(u32, u32, i64)> = pairs.iter().map(|&(s, r)| (s, r, 1)).collect();
    from_occurrences(&occ)
        .unwrap_or_else(|_| unreachable!("multiplicity: count bounded by occurrence list length"))
}

/// `volume` (#9, row 40): build `A_vol` from `(pair, amount)` occurrences
/// with checked micro-unit sums.
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (P2).
pub fn volume(occurrences: &[((u32, u32), i64)]) -> Result<PairMatrix, TensorError> {
    let occ: Vec<(u32, u32, i64)> = occurrences
        .iter()
        .map(|(pair, v)| (pair.0, pair.1, *v))
        .collect();
    from_occurrences(&occ)
}

/// `reduce_count` (#10, row 41): cardinality of a support pattern.
#[must_use]
pub fn reduce_count(members: &[u32]) -> u64 {
    members.len() as u64
}

/// `reduce_count_set` (#11, row 41): cardinality of a vertex pattern vector.
#[must_use]
pub fn reduce_count_set(s: &Vector) -> u64 {
    s.nnz() as u64
}

/// `reduce_sum` (#12, row 42): canonical summation over micro-unit values
/// (checked; empty goes to `0`).
///
/// # Errors
/// `TensorError::Range` on exact value-domain overflow (P2).
pub fn reduce_sum(values: &[i64]) -> Result<i64, TensorError> {
    let mut acc = 0i64;
    for &v in values {
        acc = checked_add(acc, v)?;
    }
    Ok(acc)
}

/// `reduce_max` (#13, row 42): canonical maximum; empty goes to `0` (v5 identity).
#[must_use]
pub fn reduce_max(values: &[i64]) -> i64 {
    values.iter().copied().fold(0i64, i64::max)
}
