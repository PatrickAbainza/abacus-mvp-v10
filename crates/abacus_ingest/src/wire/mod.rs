//! The ABACSEG1 wire envelope.
//!
//! Grounded in `docs/Abacus Ingestion Wire Format v1.0.md`:
//! - `header` — the exact 176-byte layout and its §6 validation order.
//! - `segment` — header + payload, with the checked length invariants, the
//!   payload checksum, and per-record validation.
//! - `builder` — the single construction path used by tests and adapters.
//!
//! This format is deliberately **separate** from SAMCORE02 v3. Wire §7 requires
//! that the existing `SAMLDBIN` decoder and fixture behavior be preserved, and
//! Canonical Contract §12 states that the ingestion wire format "must not
//! overwrite, reinterpret, or silently change the existing fixture format".
//! Nothing in this module touches `abacus_tensor::samld`.

pub mod builder;
pub mod header;
pub mod segment;

pub use builder::{build_segment, encode_segment};
pub use header::SegmentHeader;
pub use segment::Segment;
