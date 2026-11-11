//! Pipeline-side atomic kernels over the CSR/CSC row indexes.
//!
//! Generic terms only. Event-exists presence (committed rows).

pub mod evaluate;
pub mod row;

pub use evaluate::{evaluate_flagged_share, flag_above_baseline, require_baseline, FlaggedShare};
pub use row::{fiber_window, pair_seek, PairSeek, RowKernel};
