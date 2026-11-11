//! Physical kernels: one per (operator, structure) behind one trait.
//!
//! A library backend can be added later behind the same interface; none
//! is added now. Kernels adapt to existing storage formats and mode
//! orders; nothing here changes public behavior.

pub mod csf;
pub mod traits;

pub use traits::{KernelStats, OperatorKernel};
