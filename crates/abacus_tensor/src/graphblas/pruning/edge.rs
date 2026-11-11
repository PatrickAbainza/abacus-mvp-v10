//! Edge pruning (#16, row 45): retain entries whose `(sender, receiver)`
//! pair lies in the explicitly admissible pair mask (binary-searched
//! pattern vector; `a_{11}` + `R_3`).
//! Ported from v7 `graphblas/pruning/edge.rs` (itself from v5).

use super::super::types::{EdgeRef, EligiblePairs, TransactionRecord};

/// Retain only entries whose pair is admitted (`prec_3` order preserved).
#[must_use]
pub fn prune_edges(
    records: &[TransactionRecord],
    allowed: &EligiblePairs,
) -> Vec<TransactionRecord> {
    records
        .iter()
        .copied()
        .filter(|r| allowed.admits(EdgeRef::new(r.s, r.r)))
        .collect()
}
