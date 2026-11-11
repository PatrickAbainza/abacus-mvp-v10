//! Tensor core: rows 1–9 (§B.2) over P1/P2/P3/P5.
//!
//! A1 (totally ordered coordinate sets `T, S, R`), A2 (value set `V`, exact
//! `i64` micro-units at scale `10^6`), A3 (finite-support tensor `T : T×S×R → V`),
//! pointwise addition and negation, settling, CSF conversion.
//!
//! The submodule name is fixed by the injection structure of §7: the tensor
//! module `m_tensor` is a distinct semantic component from its siblings, and
//! the file keeps the tensor object separate from the coordinate and value
//! disciplines. Renaming for lint aesthetics would collapse that boundary.

// Allow `module_inception`: the injection `ι_tensor : m_tensor ↪ 𝒰` requires a
// distinct module named exactly as in §B.2, independent of the crate name.
#![allow(clippy::module_inception)]

pub mod coordinate;
pub mod tensor;
pub mod value;

pub use coordinate::{Coord, ReceiverId, SenderId, TimeSec};
pub use tensor::Tensor;
pub use value::{MinorUnits, MINOR_SCALE, MINOR_SCALE_EXP};
