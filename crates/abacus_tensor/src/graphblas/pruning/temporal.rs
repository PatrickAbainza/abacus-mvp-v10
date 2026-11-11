//! Temporal pruning (#14, row 43): mode-0 slice over `t in [start, end)`
//! via row-27 `time_slice` (P1 half-open; `prec_3` order preserved).
//! Ported from v7 `graphblas/pruning/temporal.rs` (itself from v5).

use super::super::projection::prune_time_slice;
use super::super::types::{TimeWindow, TransactionRecord};
use crate::api::CanonicalTensor;

/// Deterministic, order-preserving temporal slice `T|_w`.
#[must_use]
pub fn prune_time(csf: &CanonicalTensor, window: TimeWindow) -> Vec<TransactionRecord> {
    prune_time_slice(csf, window.start, window.end)
}
