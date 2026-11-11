//! GraphBLAS primitives #1–#13 (golden Addendum O, O.2 rows 32–42).
//!
//! Ported from v7 `graphblas/primitives` (itself from v5
//! `abacus_canonical_execution::primitives::{extract, mxv, vxm, ewise,
//! reduce}`) with one mechanical adaptation: the v5 unit `Overflow` is
//! `crate::TensorError::Range` (`E_rng`; P2/R5). Orientation and failure
//! contracts preserved: `A[i,j] = 1` iff sender `i -> `receiver `j`;
//! absent goes to `None` / `0`; overflow rejected, never wrapped.

pub mod extract;
pub mod ewise;
pub mod mxv;
pub mod reduce;
pub mod vxm;

pub use extract::{extract, reduce_col_plus, reduce_row_plus};
pub use ewise::ewise_add_lor;
pub use mxv::mxv;
pub use reduce::{multiplicity, reduce_count, reduce_count_set, reduce_max, reduce_sum, support, volume};
pub use vxm::vxm;
