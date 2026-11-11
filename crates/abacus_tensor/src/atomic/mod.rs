//! Atomic tensor operators over named modes (Phase 13, §5).
//!
//! `restrict` / `map` / `merge` / `reduce` are the canonical atomic
//! operators. Generic terms only (modes, coordinates, entries, keys,
//! windows). Integer-only arithmetic (i64). See
//! `docs/OPERATOR_CONTRACTS.md` for the authoritative spec.

pub mod backends;
pub mod kernels;
pub mod outcome;
pub mod reference;
pub mod types;

pub use outcome::{Indeterminate, OpReason, Outcome};
pub use types::{Coord3, Entry, LocSet, Mode, Monoid, Support};
