//! `vxm` under `LOR_LAND` (#5, row 36): `v' = v x A` — receivers of frontier
//! senders (forward step).
//!
//! `v'[j] = OR_i ( v[i] AND A[i,j] )`. Ported from v7
//! `graphblas/primitives/vxm.rs` (itself from v5: row-block projections;
//! rows are sender-major contiguous, union via sort + dedup).

use super::super::matrix::PairMatrix;
use super::super::vector::Vector;

/// Boolean vector-matrix multiply: all receivers `j` with `A[i,j]=1` for
/// some active frontier sender `i`.
#[must_use]
pub fn vxm(a: &PairMatrix, frontier: &Vector) -> Vector {
    let mut members: Vec<u32> = Vec::new();
    for i in frontier.indices() {
        members.extend_from_slice(&a.receivers_from(*i));
    }
    members.sort_unstable();
    members.dedup();
    Vector::from_sorted(members)
}
