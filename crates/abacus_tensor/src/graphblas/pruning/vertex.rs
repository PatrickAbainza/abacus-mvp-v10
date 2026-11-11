//! Vertex pruning (#15, row 44): retain entries whose endpoints both lie
//! in the allowed vertex pattern (masked extract, `a_{11}` + `R_3`).
//! Ported from v7 `graphblas/pruning/vertex.rs` (itself from v5).

use super::super::types::TransactionRecord;
use super::super::vector::Vector;

/// Retain only entries with sender and receiver both in `allowed`
/// (`prec_3` order preserved).
#[must_use]
pub fn prune_vertices(
    records: &[TransactionRecord],
    allowed: &Vector,
) -> Vec<TransactionRecord> {
    records
        .iter()
        .copied()
        .filter(|r| allowed.contains(r.s) && allowed.contains(r.r))
        .collect()
}
