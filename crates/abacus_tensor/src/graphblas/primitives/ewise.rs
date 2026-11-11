//! `ewise_add_lor` (#6, row 37): Boolean element-wise OR (`C = A OR B`).
//! Ported from v7 `graphblas/primitives/ewise.rs` (itself from v5:
//! delegates to the ordered pattern-union on `Vector`).

use super::super::vector::Vector;

/// Union of two frontier patterns over the vertex domain.
#[must_use]
pub fn ewise_add_lor(a: &Vector, b: &Vector) -> Vector {
    a.ewise_add_lor(b)
}
