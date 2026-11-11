//! Projection layer (golden Addendum O, O.1/O.2 rows 30–32).
//!
//! Ported from v7 `graphblas/projection.rs` (itself from v5
//! `projection`), reusing the v10 row-27 read boundary (`crate::api`):
//!
//! * `prune_time_slice`: row-16 restriction `T|_w` via row-27 `time_slice`
//!   — the only tensor read.
//! * `extract_3d`: exact point evaluation via row-27 `lookup` (P3: absent
//!   reported absent).
//! * `contract_mode0` (row 30): `A_w(s,r) = sum_{t in w} T(t,s,r)` — checked
//!   `d_settle`-shape contraction (P5/`a_{02}`, `R_5`, `R_6`).
//! * `temporal_multiplicity` (row 31): `A_cnt(s,r)` — one unit per canonical
//!   `(t,s,r)` entry (P3 uniqueness gives the exact count).

use super::matrix::{from_occurrences, PairMatrix};
use crate::api::{lookup, time_slice, CanonicalTensor, TensorEntry};
use crate::TensorError;

/// Temporal slice of the canonical tensor: all canonical entries with
/// `t in [start, end)` (`prec_3` order preserved; P1 half-open).
#[must_use]
pub fn prune_time_slice(csf: &CanonicalTensor, start: u64, end: u64) -> Vec<TensorEntry> {
    time_slice(csf, start, end)
}

/// 3D extract: `T(t, s, r)` at an exact canonical coordinate; `None` when
/// the coordinate is outside `supp(T)` (P3).
#[must_use]
pub fn extract_3d(csf: &CanonicalTensor, t: u64, s: u32, r: u32) -> Option<i64> {
    lookup(csf, t, s, r)
}

/// Mode-0 contraction of the windowed tensor onto the `(s, r)` plane
/// (row 30).
///
/// # Errors
/// `TensorError::Range` (`E_rng`) when an exact sum is unrepresentable.
pub fn contract_mode0(windowed: &[TensorEntry]) -> Result<PairMatrix, TensorError> {
    let occ: Vec<(u32, u32, i64)> = windowed.iter().map(|e| (e.s, e.r, e.value)).collect();
    from_occurrences(&occ)
}

/// Temporal-support multiplicity (row 31): `A_cnt(s,r)` equals the count of
/// `t in w` with `(t,s,r)` in `supp(T)`.
#[must_use]
pub fn temporal_multiplicity(windowed: &[TensorEntry]) -> PairMatrix {
    let mut occ: Vec<(u32, u32, i64)> = Vec::with_capacity(windowed.len());
    let mut prev: Option<TensorEntry> = None;
    for e in windowed {
        debug_assert!(e.value != 0, "canonical decode never yields zero values");
        if let Some(p) = prev {
            debug_assert!(
                (p.t, p.s, p.r) < (e.t, e.s, e.r),
                "windowed entries must be prec_3-sorted"
            );
        }
        occ.push((e.s, e.r, 1));
        prev = Some(*e);
    }
    from_occurrences(&occ).unwrap_or_else(|_| {
        unreachable!("temporal_multiplicity: count overflow impossible (P3 finite support)")
    })
}
