//! Row 27 (§B.2): the public API boundary — read-only, algebra-free slice.
//!
//! A single sanctioned **read boundary** for computation consumers. Every
//! operation here is an evaluation, restriction, decode, or load of the
//! canonical tensor and re-exposes rows 16, 15 and 17 without adding algebra:
//!
//! ```text
//! time_slice  row 16 mode-0 restriction, half-open window, prec_3 order
//! lookup      row 16 point read; None iff x ∉ supp(T) (absent, never fabricated)
//! decode      row 15 CSF(T)⟧ (regroup only)
//! deserialize row 17 artifact load
//! checksum    d_fnv over serialized bytes (provenance infrastructure)
//! ```
//!
//! Invariants: read-only (nothing here can mutate the canonical object),
//! deterministic (total iteration order only), and algebra-free.

use crate::coo::entry::TxEntry;
use crate::csf::compress::CsfTensor;
use crate::csf::serialize::Image;

/// The canonical tensor crossing this boundary: `CSF(T)`.
pub type CanonicalTensor = CsfTensor;

/// One canonical record crossing this boundary: `(t, s, r, value)` in `prec_3`
/// order with an exact micro-unit value. Never repeats a coordinate and never
/// carries a zero value (P3).
pub type TensorEntry = TxEntry;

/// `T|_w` — restriction to `t ∈ [start, end)` (P1 half-open window).
///
/// # Panics
/// Never: the window is total (an inverted or empty window yields no entries).
#[must_use]
pub fn time_slice(csf: &CanonicalTensor, start: u64, end: u64) -> Vec<TensorEntry> {
    crate::csf::query::time_slice(csf, start, end).entries
}

/// `T(t, s, r)` — exact evaluation; `None` iff the coordinate is not in the
/// support (reported absent, never fabricated as zero, P3).
#[must_use]
pub fn lookup(csf: &CanonicalTensor, t: u64, s: u32, r: u32) -> Option<i64> {
    let v = crate::csf::query::lookup(csf, t, s, r);
    (v != 0).then_some(v)
}

/// `⟦CSF(T)⟧` — decode to canonical COO (regroup only, drop nothing; P6).
#[must_use]
pub fn decode(csf: &CanonicalTensor) -> Vec<TensorEntry> {
    crate::csf::decode_csf(csf)
}

/// Load a serialized canonical tensor artifact (row 17).
///
/// # Errors
/// [`crate::TensorError::Decode`] on malformation or checksum mismatch.
pub fn deserialize(data: &[u8]) -> Result<Image, crate::TensorError> {
    crate::csf::serialize::deserialize(data)
}

/// `d_fnv` over serialized bytes (provenance infrastructure; not tensor
/// algebra).
#[must_use]
pub fn checksum(data: &[u8]) -> u64 {
    crate::csf::serialize::checksum(data)
}