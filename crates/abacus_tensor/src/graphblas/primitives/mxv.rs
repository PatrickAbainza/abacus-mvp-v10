//! `mxv` under `LOR_LAND` (#4, row 35): `v' = A x v` — senders of frontier
//! receivers (backward step).
//!
//! `v'[i] = OR_j ( A[i,j] AND v[j] )`. Ported from v7
//! `graphblas/primitives/mxv.rs` (itself from v5: single canonical-order
//! scan with a frontier membership mask).

use super::super::matrix::PairMatrix;
use super::super::vector::Vector;

/// Boolean matrix-vector multiply: all senders `i` with `A[i,j]=1` for some
/// active frontier receiver `j`.
#[must_use]
pub fn mxv(a: &PairMatrix, frontier: &Vector) -> Vector {
    let mut out = Vector::zero();
    let mut members: Vec<u32> = Vec::new();
    for (s, r, _val) in a.nonzeros() {
        if frontier.contains(r) && members.last() != Some(&s) {
            members.push(s);
        }
    }
    debug_assert!(members.windows(2).all(|w| w[0] < w[1]));
    out.extend_unsorted(members);
    out
}
